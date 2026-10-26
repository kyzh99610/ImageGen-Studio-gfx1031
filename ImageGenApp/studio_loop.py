"""
studio_loop.py — 🔁 Studio Loop: render your saved settings round after round, unattended, and sort the pictures.

    studio_loop.bat <plan>            plan = settings\\loop\\<plan>.json (or a path); see backend/loop_plan.py for the format
    studio_loop.bat <plan> --attach http://127.0.0.1:7860   use an app that is already open instead of starting one

Each round renders every saved setting of the plan through the app's own API (exactly what 💾 Load + ✨ Generate do, so
Polish, the detail passes, colour strength, pose control … all apply), then scores each picture on the CPU (faces, hands,
the card's colours and identity, WD14 tags for exposure and artifacts) and files it under
outputs\\loop\\<run>\\{best, ok, review, reject}. Between rounds settings that keep producing keepers keep their share, the
ones you ⭐ in the History tab get more, and the ones that kept nothing for a while are parked. It stops at the plan's
stop time / hour budget / round count, or when a file named STOP appears in the run folder. report.md and report.html
in the run folder say what to keep and what to fix.

The app's own heat protection (Settings → cool mode / pause while hot) applies to every picture; the loop also waits
while hot before scoring. Your Generate tab's last session is put back when the loop ends.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
os.environ.setdefault("IMAGEGEN_LOOP", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # the loop only scores on the CPU: never open a second GPU context next to the app

from backend import loop_plan as LP  # noqa: E402  (pure, no torch)

MAX_FAILS = 3   # generate calls in a row that raise (the app closed or crashed): the loop stops instead of spinning until morning


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- the app
def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def running_app(ports=range(7860, 7881)):
    for p in ports:
        try:
            if b"ImageGen" in _get(f"http://127.0.0.1:{p}/", 2):
                return f"http://127.0.0.1:{p}"
        except Exception:
            pass
    return None


def free_port(start=7861):
    for p in range(start, start + 40):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


def start_app(subdir: str, port: int, logf: Path):
    env = dict(os.environ, IMAGEGEN_OUTPUT_SUBDIR=subdir)
    fh = open(logf, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(["cmd", "/c", str(APP / "launch.bat"), "--port", str(port), "--no-browser", "--no-pause"],
                            cwd=str(APP), env=env, stdout=fh, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    url = f"http://127.0.0.1:{port}"
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f"the app stopped while starting (see {logf})")
        try:
            _get(url + "/config", 3)
            return proc, url
        except Exception:
            time.sleep(1)
    raise RuntimeError("the app did not answer within 10 minutes")


def stop_app(proc):
    if proc and proc.poll() is None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)


class Api:
    """The app's named endpoints by component id, from its /config (no positional guessing)."""

    def __init__(self, url):
        from gradio_client import Client
        self.c = Client(url, verbose=False)
        cfg = json.loads(_get(url + "/config", 30))
        self.comp = {c["id"]: c for c in cfg["components"]}
        self.dep = {d["api_name"]: d for d in cfg["dependencies"] if d.get("api_name")}
        for need in ("do_load_settings", "do_generate_ui"):
            if need not in self.dep:
                raise RuntimeError(f"this app has no /{need} endpoint")

    def label(self, cid):
        return str(self.comp[cid].get("props", {}).get("label") or "")

    def default(self, cid):
        return self.comp[cid].get("props", {}).get("value")

    def load(self, name: str) -> tuple[dict, str]:
        out = self.c.predict(name, api_name="/do_load_settings")
        ids = self.dep["do_load_settings"]["outputs"]
        vals = {}
        for cid, v in zip(ids, out):
            if isinstance(v, dict) and v.get("__type__") == "update":
                if "value" in v:
                    vals[cid] = v["value"]
            else:
                vals[cid] = v
        status = str(out[-1]) if out else ""
        return vals, status

    def generate(self, vals: dict, overrides: dict, timeout=3 * 3600):
        args = []
        for cid in self.dep["do_generate_ui"]["inputs"]:
            lab = self.label(cid)
            v = vals.get(cid, self.default(cid))
            for key, ov in overrides.items():   # a key ending in '*' is a prefix ('Seed*'), else the exact label
                if lab == key or (key.endswith("*") and lab.startswith(key[:-1])):
                    v = ov(v) if callable(ov) else ov
            args.append(v)
        return self.c.submit(*args, api_name="/do_generate_ui").result(timeout=timeout)


# ---------------------------------------------------------------- scoring
class Scorer:
    def __init__(self, card_name: str):
        import config  # noqa: F401  (paths, HF_HOME)
        from backend import image_score, detail_tools, wd_tagger, thermal
        self.IS, self.DT, self.WD, self.TH = image_score, detail_tools, wd_tagger, thermal
        self.card, self.identity = None, None
        if card_name:
            from backend import character_cards as CC
            self.card = CC.load_card(card_name)
            if not self.card:
                log(f"⚠ card {card_name!r} not found: scoring without identity")
            else:
                self.identity = self.card.get("identity")

    def wait_cool(self):
        try:
            prefs = json.loads((APP / "settings" / "_prefs.json").read_text(encoding="utf-8"))
            self.TH.set_limit(prefs.get("thermal_limit", 0.0))
            waited = self.TH.wait_cool()
            if waited >= 5:
                log(f"🌡 waited {waited:.0f} s to cool before scoring")
        except Exception:
            pass

    def score(self, img, prompt: str) -> dict:
        tags = (self.card or {}).get("tags") or prompt
        sc = self.IS.score_images([img], tags, identity=self.identity)[0]
        faces = len(self.DT.detect_faces(img, "anime") or [])
        off, judged = self.DT.hands_look_off(img)
        wd = self.WD.tag_image(img, general_threshold=0.3)
        probs = dict(wd["general"])
        probs["explicit"] = max(probs.get("explicit", 0.0), wd["rating"].get("explicit", 0.0))
        return dict(stars=sc.get("stars"), flags=sc.get("flags") or [], faces=faces, hands_off=off, hands=judged, tags=probs)


# ---------------------------------------------------------------- the run
def favourites() -> set:
    try:
        return set(json.loads((APP / "settings" / "_favourites.json").read_text(encoding="utf-8")))
    except Exception:
        return set()


def write_reports(run: Path, plan: dict, records: list, stopped: str, minutes: float):
    (run / "report.md").write_text(LP.report_md(plan, records, stopped, minutes), encoding="utf-8")
    rows = []
    for name in dict.fromkeys(r["settings"] for r in records):
        recs = sorted([r for r in records if r["settings"] == name], key=LP.rank_key)
        tiles = "".join(
            f'<figure class="{r["verdict"]}"><a href="{html.escape(os.path.relpath(r["file"], run))}">'
            f'<img loading="lazy" src="{html.escape(os.path.relpath(r["file"], run))}"></a>'
            f'<figcaption>{r["verdict"]} · {r.get("stars")}★ · seed {r.get("seed")}<br>{html.escape("; ".join(r.get("why") or []))}</figcaption></figure>'
            for r in recs)
        rows.append(f"<h2>{html.escape(name)}</h2><div class=grid>{tiles}</div>")
    css = ("body{font:14px system-ui;margin:16px;background:#111;color:#ddd}.grid{display:flex;flex-wrap:wrap;gap:8px}"
           "figure{margin:0;width:200px}img{width:200px;border-radius:6px}figcaption{font-size:12px}"
           ".best img{outline:3px solid #4c4}.ok img{outline:2px solid #888}.review img{outline:2px solid #c90}.reject img{opacity:.45}")
    (run / "report.html").write_text(f"<!doctype html><meta charset=utf-8><title>Studio Loop</title><style>{css}</style>"
                                     f"<pre>{html.escape(LP.report_md(plan, records, stopped, minutes))}</pre>{''.join(rows)}",
                                     encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Studio Loop: render saved settings round after round and sort the pictures.")
    ap.add_argument("plan")
    ap.add_argument("--attach", default="", help="URL of an app that is already running")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--raw", default="", help="with --attach: the folder the app saves to (its outputs folder)")
    a = ap.parse_args(argv)

    pp = Path(a.plan)
    if not pp.exists():
        pp = APP / "settings" / "loop" / (a.plan if a.plan.endswith(".json") else a.plan + ".json")
    plan = LP.load_plan(pp)
    missing = [p["settings"] for p in plan["presets"] if not (APP / "settings" / f"{p['settings']}.json").exists()]
    if missing:
        raise SystemExit(f"saved settings not found: {', '.join(missing)}")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    # attached to an app (the 🔁 tab does it, with --raw = that app's outputs folder): the run folder sits in that folder, where the tab looks for it
    loop_root = (Path(a.raw) if (a.attach and a.raw) else APP / "outputs") / "loop"
    run = loop_root / f"{pp.stem}_{stamp}"
    for v in LP.VERDICTS:
        (run / v).mkdir(parents=True, exist_ok=True)
    (run / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"run folder {run}")

    sess = APP / "settings" / "_last_session.json"
    sess_bak = run / "_last_session.before.json"
    if sess.exists():
        shutil.copy2(sess, sess_bak)

    proc = None
    if a.attach:
        url, raw = a.attach.rstrip("/"), (Path(a.raw) if a.raw else APP / "outputs")
    else:
        other = running_app()
        if other:
            raise SystemExit(f"ImageGen Studio is already open at {other}: close it (two apps on one GPU spill VRAM) "
                             f"or pass --attach {other}")
        port = a.port or free_port()
        sub = f"loop/{run.name}/raw"
        raw = APP / "outputs" / sub
        log(f"starting the app on port {port} …")
        proc, url = start_app(sub, port, run / "app.log")
    records, started, stopped = [], dt.datetime.now(), ""
    try:
        api = Api(url)
        scorer = Scorer(plan["card"])
        counts = {p["settings"]: p["count"] for p in plan["presets"]}
        fails = 0
        rnd = 0
        while True:
            rnd += 1
            stopped = LP.should_stop(plan, started, dt.datetime.now(), rnd, counts)
            if stopped:
                break
            log(f"— round {rnd}: {counts}")
            for p in plan["presets"]:
                n = counts.get(p["settings"], 0)
                if not n:
                    continue
                if (run / "STOP").exists():
                    stopped = "STOP file"
                    break
                if LP.should_stop(plan, started, dt.datetime.now(), rnd, {"x": 1}):
                    break
                vals, status = api.load(p["settings"])
                ov = {"Batch Size": n}
                if plan["seed_base"] >= 0:
                    ov["Seed*"] = plan["seed_base"] + rnd * 1000 + len(records)
                else:
                    ov["Seed*"] = -1
                if p["prompt_suffix"]:
                    ov["Prompt"] = (lambda suf: (lambda v: (str(v or "").rstrip(", ") + ", " + suf)))(p["prompt_suffix"])
                if plan["polish"]:
                    ov["✨ Polish*"] = plan["polish"]
                # explore around the keepers (plan "explore"): from round 2 a share of the pictures are close relatives of this setting's best
                # picture so far — its seed (so the same wildcard picks) + a random variation seed at EXPLORE_STRENGTH, one call each
                n_new, n_exp = LP.explore_split(n, plan["explore"]) if rnd > 1 else (n, 0)
                seed_b = LP.best_seed(records, p["settings"]) if n_exp else None
                if seed_b is None:
                    n_new, n_exp = n, 0
                jobs = ([(dict(ov, **{"Batch Size": n_new}), False)] if n_new else []) + [
                    (dict(ov, **{"Batch Size": 1, "Seed*": seed_b, "Variation seed*": -1, "Variation strength": LP.EXPLORE_STRENGTH}), True)
                    for _ in range(n_exp)]
                for ov_j, exploring in jobs:
                    t0 = time.time()
                    try:
                        res = api.generate(vals, ov_j)
                        fails = 0
                    except Exception as e:
                        log(f"⚠ {p['settings']}: generate failed: {e}")
                        fails += 1
                        if fails >= MAX_FAILS:            # the app is gone or broken: stop instead of spinning until morning
                            stopped = f"{fails} generates in a row failed (last: {str(e)[:120]})"
                            break
                        continue
                    new = LP.own_files(raw, res, t0, bool(a.attach))
                    secs = time.time() - t0
                    log(f"{p['settings']}: {len(new)} picture(s) in {secs:.0f} s" + (f" (variation of seed {seed_b})" if exploring else ""))
                    if a.attach and not new:
                        log(f"⚠ {p['settings']}: the app's result names no picture in {raw} (is --raw its outputs folder?)")
                    from PIL import Image
                    favs = favourites()
                    for f in new:
                        scorer.wait_cool()
                        with Image.open(f) as im:
                            im.load()
                            info = dict(im.info)
                        rec = json.loads(info.get("imagegen") or "{}")
                        try:
                            s = scorer.score(im.convert("RGB"), rec.get("prompt", ""))
                            v, why = LP.verdict(s, s["tags"], s["faces"], plan["gate"], s["hands_off"])
                        except Exception as e:
                            s, v, why = {}, "review", [f"scoring failed: {e}"]
                        dest = run / v / f.name
                        shutil.move(str(f), str(dest))
                        r = dict(file=str(dest), settings=p["settings"], round=rnd, seed=rec.get("seed"), verdict=v, why=why,
                                 stars=s.get("stars"), faces=s.get("faces"), hands_off=s.get("hands_off"),
                                 favourite=f.name in favs, secs=round(secs / max(1, len(new)), 1),
                                 prompt=str(rec.get("prompt") or "")[:3000], **({"explore": True} if exploring else {}))
                        records.append(r)
                        with open(run / "records.jsonl", "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                if stopped:
                    break
                write_reports(run, plan, records, "", (dt.datetime.now() - started).total_seconds() / 60)
            if stopped:
                break
            for r in records:                      # stars given in the History tab during the run count too
                r["favourite"] = r["favourite"] or Path(r["file"]).name in favourites()
            counts = LP.next_counts(plan, LP.preset_stats(records))
    except KeyboardInterrupt:
        stopped = "interrupted"
    except Exception:
        stopped = "error: " + traceback.format_exc().splitlines()[-1]
        log(traceback.format_exc())
    finally:
        stop_app(proc)
        if sess_bak.exists():
            shutil.copy2(sess_bak, sess)
        write_reports(run, plan, records, stopped, (dt.datetime.now() - started).total_seconds() / 60)
        log(f"stopped: {stopped}. {len(records)} pictures. Report: {run / 'report.html'}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
