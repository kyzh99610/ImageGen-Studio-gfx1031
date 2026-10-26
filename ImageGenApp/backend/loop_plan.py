"""
loop_plan.py — 🔁 Studio Loop: the plan, the per-picture verdict and the next round's allocation (pure: no UI, no GPU).

The loop (tools/studio_loop.py) renders the user's own saved settings round after round, unattended, scores every picture
on the CPU, sorts it into best / ok / review / reject and, between rounds, gives more pictures to the settings that keep
producing keepers and parks the ones that keep failing. No language model runs while it renders; the run folder's report
(report.md / report.html) is what a person — or a Claude session afterwards — reads to change prompts for the next run.

A plan is a JSON file in settings/loop/:
    {"presets": ["my setting", {"settings": "other", "count": 3, "prompt_suffix": "night"}],
     "per_round": 2, "rounds": 0, "stop_at": "07:00", "max_hours": 8, "card": "",
     "gate": {"block_tags": [...], "artifact_tags": [...], "faces": 1, "min_stars_ok": 4}}
Everything but "presets" is optional.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

# WD14 tags that send a picture to reject (probability >= gate["tag_threshold"]). The defaults block real exposure and the
# usual artifacts; a plan can replace either list.
BLOCK_TAGS = ("explicit",)   # WD14's rating; add your own tags under the plan's gate.block_tags
ARTIFACT_TAGS = ("text", "english text", "watermark", "signature", "artist name", "border", "multiple views", "comic",
                 "2girls", "multiple girls", "split screen", "inset")
VERDICTS = ("best", "ok", "review", "reject")


def saved_names(result) -> list[str]:
    """The picture file names the app's Generate result says it saved (its info text has a 'Saved: a.png, b.png' line per call), in order."""
    texts: list[str] = []

    def walk(x):
        if isinstance(x, str):
            texts.append(x)
        elif isinstance(x, (list, tuple)):
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            for i in x.values():
                walk(i)
    walk(result)
    names: list[str] = []
    for t in texts:
        for line in re.findall(r"Saved:\s*([^<\n]+)", t):
            for n in line.split(","):
                n = n.strip()
                if n.lower().endswith(".png") and n not in names:
                    names.append(n)
    return names


def own_files(raw, result, t0: float, attach: bool) -> list:
    """The pictures THIS Generate call made, in the folder `raw` the app saves to. Named by the app's own result, so a picture the user makes by hand
    in an attached app while the loop runs is never swept into the run folder; only when the loop started the app itself (nothing else saves there)
    and the result names nothing does it fall back to 'new files since the call began'."""
    raw = Path(raw)
    got = [raw / n for n in saved_names(result) if (raw / n).is_file()]
    if got or attach:
        return got
    return sorted(f for f in raw.glob("*.png") if f.stat().st_mtime >= t0 - 1)

DEFAULTS = dict(per_round=2, rounds=0, stop_at="", max_hours=8.0, card="", polish="", seed_base=-1,
                gate=dict(block_tags=list(BLOCK_TAGS), artifact_tags=list(ARTIFACT_TAGS), tag_threshold=0.5,
                          faces=1, min_stars_ok=4, park_after=2))


def _num(v, lo, hi, default, cast=float):
    try:
        v = cast(v)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v))


def clean_plan(d: dict) -> dict:
    """A plan made safe: known keys only, numbers clamped, presets as [{settings, count, prompt_suffix}]. Raises ValueError
    when there is nothing to render."""
    if not isinstance(d, dict):
        raise ValueError("a plan is a JSON object")
    out = {k: v for k, v in DEFAULTS.items() if k != "gate"}
    out["per_round"] = _num(d.get("per_round"), 1, 8, DEFAULTS["per_round"], int)
    out["rounds"] = _num(d.get("rounds"), 0, 1000, 0, int)
    out["max_hours"] = _num(d.get("max_hours"), 0.05, 72, DEFAULTS["max_hours"])
    out["seed_base"] = _num(d.get("seed_base"), -1, 2**32 - 1, -1, int)
    stop = str(d.get("stop_at") or "").strip()
    if stop:
        try:
            _dt.datetime.strptime(stop, "%H:%M")
        except ValueError:
            raise ValueError(f"stop_at must look like 07:00 (got {stop!r})")
    out["stop_at"] = stop
    out["card"] = str(d.get("card") or "")
    out["polish"] = str(d.get("polish") or "")
    g = dict(DEFAULTS["gate"])
    gin = d.get("gate") if isinstance(d.get("gate"), dict) else {}
    for k in ("block_tags", "artifact_tags"):
        if isinstance(gin.get(k), list):
            g[k] = [str(t).strip().lower() for t in gin[k] if str(t).strip()]
    g["tag_threshold"] = _num(gin.get("tag_threshold"), 0.05, 0.99, g["tag_threshold"])
    g["faces"] = _num(gin.get("faces"), 0, 8, g["faces"], int)
    g["min_stars_ok"] = _num(gin.get("min_stars_ok"), 1, 5, g["min_stars_ok"], int)
    g["park_after"] = _num(gin.get("park_after"), 1, 100, g["park_after"], int)
    out["gate"] = g
    presets = []
    for p in d.get("presets") or []:
        if isinstance(p, str) and p.strip():
            presets.append(dict(settings=p.strip(), count=out["per_round"], prompt_suffix=""))
        elif isinstance(p, dict) and str(p.get("settings") or "").strip():
            presets.append(dict(settings=str(p["settings"]).strip(), count=_num(p.get("count"), 1, 8, out["per_round"], int),
                                prompt_suffix=str(p.get("prompt_suffix") or "").strip()))
    if not presets:
        raise ValueError("the plan names no saved settings under \"presets\"")
    out["presets"] = presets
    return out


def load_plan(path) -> dict:
    return clean_plan(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def verdict(score: dict, tags: dict, faces: int | None, gate: dict, hands_off: int = 0) -> tuple[str, list[str]]:
    """best / ok / review / reject for one picture, with the reasons.
    score = image_score's {'stars', 'flags'}; tags = WD14 {tag: probability} of the whole picture; faces = faces found;
    hands_off = hands the hand judge says look off."""
    why = []
    thr = gate.get("tag_threshold", 0.5)
    t = {str(k).lower().replace("_", " "): float(v) for k, v in (tags or {}).items()}
    hit = sorted({k for k in gate.get("block_tags", ()) if t.get(k, 0.0) >= thr})
    if hit:
        return "reject", [f"blocked: {', '.join(hit)}"]
    art = sorted({k for k in gate.get("artifact_tags", ()) if t.get(k, 0.0) >= thr})
    if art:
        return "reject", [f"artifact: {', '.join(art)}"]
    want = gate.get("faces", 1)
    if want and faces is not None and faces != want:
        return "reject", [f"{faces} faces"]
    stars = int((score or {}).get("stars", 0) or 0)
    flags = list((score or {}).get("flags") or [])
    if hands_off:
        why.append(f"{hands_off} hand(s) look off")
    why += flags
    if stars < gate.get("min_stars_ok", 4):
        return "review", why or [f"{stars} stars"]
    if hands_off:
        return "review", why
    if stars >= 5 and not flags:
        return "best", why
    return "ok", why


def rank_key(rec: dict):
    """Sort key within a preset: verdict first, then stars, then fewer flags."""
    return (VERDICTS.index(rec.get("verdict", "reject")), -int(rec.get("stars", 0) or 0), len(rec.get("why") or []))


def reason_key(w: str) -> str:
    """A verdict reason without its numbers, so the report counts one problem once: '1 hand(s) look off' and '2 hand(s) look off' are
    'hand(s) look off', 'grey hair 0.12' is 'grey hair', 'artifact: text' is 'artifact'."""
    k = w.split(":")[0] if ":" in w else w
    k = re.sub(r"\([^)]*\d[^)]*\)", "", k)          # '(her pictures 0.96)'
    k = re.sub(r"\b\d+(?:\.\d+)?\b", "", k)
    return re.sub(r"\s+", " ", k).strip() or w


def preset_stats(records: list[dict]) -> dict:
    """{settings: {n, best, ok, review, reject, keep_rate, last_rounds_kept: [..], reasons: {reason: n}}}."""
    st = {}
    for r in records:
        s = st.setdefault(r["settings"], dict(n=0, best=0, ok=0, review=0, reject=0, rounds={}, reasons={}, favs=0))
        s["n"] += 1
        s[r["verdict"]] += 1
        kept = r["verdict"] in ("best", "ok")
        s["rounds"][r["round"]] = s["rounds"].get(r["round"], 0) + int(kept)
        if not kept:
            for w in r.get("why") or []:
                key = reason_key(w)
                s["reasons"][key] = s["reasons"].get(key, 0) + 1
        s["favs"] += int(bool(r.get("favourite")))
    for s in st.values():
        s["keep_rate"] = round((s["best"] + s["ok"]) / s["n"], 3) if s["n"] else 0.0
        s["last_rounds_kept"] = [s["rounds"][k] for k in sorted(s["rounds"])]
    return st


def next_counts(plan: dict, stats: dict) -> dict:
    """Pictures per preset for the next round. A preset whose last `park_after` rounds kept nothing is parked (0) — its
    prompt needs a fix, see the report; a preset the user starred in History gets one more picture per star (at most
    twice its count); everything else keeps its count."""
    out = {}
    park = plan["gate"]["park_after"]
    for p in plan["presets"]:
        s = stats.get(p["settings"])
        n = p["count"]
        if s:
            last = s["last_rounds_kept"][-park:]
            if len(last) >= park and sum(last) == 0:
                out[p["settings"]] = 0
                continue
            n = min(2 * p["count"], n + s.get("favs", 0), 8)
        out[p["settings"]] = n
    return out


def should_stop(plan: dict, started: _dt.datetime, now: _dt.datetime, round_no: int, counts: dict) -> str:
    """Why the loop stops before round `round_no` ('' = go on)."""
    if plan["rounds"] and round_no > plan["rounds"]:
        return f"{plan['rounds']} round(s) done"
    if (now - started).total_seconds() >= plan["max_hours"] * 3600:
        return f"{plan['max_hours']:g} h budget used"
    if plan["stop_at"]:
        hh, mm = map(int, plan["stop_at"].split(":"))
        stop = started.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if stop <= started:
            stop += _dt.timedelta(days=1)
        if now >= stop:
            return f"stop time {plan['stop_at']} reached"
    if not any(counts.values()):
        return "every preset is parked"
    return ""


def report_md(plan: dict, records: list[dict], stopped: str, minutes: float) -> str:
    """The short text report of a run: what to keep, what to fix. Written for a person or a language model to read."""
    st = preset_stats(records)
    tot = {v: sum(1 for r in records if r["verdict"] == v) for v in VERDICTS}
    lines = [f"# Studio Loop report", "",
             f"{len(records)} pictures in {minutes:.0f} min ({stopped or 'running'}): "
             f"best {tot['best']}, ok {tot['ok']}, review {tot['review']}, reject {tot['reject']}.", "",
             "| saved setting | pictures | best | ok | review | reject | kept per round | top problems |", "|---|---|---|---|---|---|---|---|"]
    for name, s in sorted(st.items(), key=lambda kv: -kv[1]["keep_rate"]):
        probs = ", ".join(f"{k} ×{n}" for k, n in sorted(s["reasons"].items(), key=lambda kv: -kv[1])[:3]) or "-"
        lines.append(f"| {name} | {s['n']} | {s['best']} | {s['ok']} | {s['review']} | {s['reject']} | {s['last_rounds_kept']} | {probs} |")
    parked = [n for n, c in next_counts(plan, st).items() if c == 0]
    lines += ["", "## Needs a prompt fix", ""]
    lines += [f"- {n}: kept nothing in its last {plan['gate']['park_after']} rounds" for n in parked] or ["- nothing parked"]
    lines += ["", "## Best pictures", ""]
    best = sorted([r for r in records if r["verdict"] == "best"], key=rank_key)[:40]
    lines += [f"- {Path(r['file']).name} ({r['settings']}, seed {r.get('seed')})" for r in best] or ["- none yet"]
    return "\n".join(lines) + "\n"
