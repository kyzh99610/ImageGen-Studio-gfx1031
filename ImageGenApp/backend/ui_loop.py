"""
ui_loop.py — the 🔁 Studio Loop tab: make / pick a plan, start the loop on this app, stop it, see how it is doing and the best
pictures of the latest run. The loop itself is studio_loop.py, started as a child process with --attach (it renders through
this app's own API, so the GPU-job queue keeps it and anything you do by hand from running at once). `build_tab(ns)` is
called from app.build_app; `ns` is the app module (SETTINGS_DIR, OUTPUTS_DIR, _list_saved_settings, _ARGS are read at call time).
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr

from backend import loop_plan as LP

APP = Path(__file__).resolve().parent.parent
STATE = {"proc": None, "plan": "", "log": None, "started": 0.0}
_POLISH = ["", "Auto", "Cowboy shot — hires", "Cowboy shot", "Full body", "Portrait"]


def _loop_dir(ns) -> Path:
    return Path(ns.SETTINGS_DIR) / "loop"


def list_plans(ns) -> list[str]:
    d = _loop_dir(ns)
    return sorted(p.stem for p in d.glob("*.json") if not p.stem.startswith("_")) if d.is_dir() else []   # _test_* = test runs


def _safe(name: str) -> str:
    return re.sub(r"[^\w\- ]+", "", str(name or "")).strip()[:60]


def running() -> bool:
    p = STATE["proc"]
    return p is not None and p.poll() is None


def _url(ns) -> str:
    port = getattr(getattr(ns, "_ARGS", None), "port", None) or 7860
    return f"http://127.0.0.1:{port}"


def latest_run(ns, plan: str = "") -> Path | None:
    root = Path(ns.OUTPUTS_DIR) / "loop"
    if not root.is_dir():
        return None
    runs = [p for p in root.iterdir() if p.is_dir() and (not plan or p.name.startswith(plan + "_"))]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def plan_from_controls(settings, per_round, stop_at, max_hours, card, polish) -> dict:
    return LP.clean_plan({"presets": list(settings or []), "per_round": per_round, "stop_at": (stop_at or "").strip(),
                          "max_hours": max_hours, "card": card or "", "polish": polish or "", "rounds": 0})


def merged_plan(ns, name, plan: dict) -> dict:
    """The plan the controls show, merged with the saved plan of that name: a setting's own count / prompt suffix (when it is still
    in the plan), the gate, the round limit and the seed base come from the file (the tab has no controls for them). JSON-ready."""
    try:
        old = json.loads((_loop_dir(ns) / f"{_safe(name)}.json").read_text(encoding="utf-8-sig")) if _safe(name) else {}
        old = old if isinstance(old, dict) else {}
    except Exception:
        old = {}
    own = {}
    for q in old.get("presets") or []:
        if isinstance(q, dict) and q.get("settings") and (q.get("count") is not None or q.get("prompt_suffix")):
            own[str(q["settings"]).strip()] = {k: q[k] for k in ("settings", "count", "prompt_suffix") if q.get(k) not in (None, "")}
    out = {k: old[k] for k in ("gate", "rounds", "seed_base") if k in old}
    out.update({k: v for k, v in plan.items() if k not in ("gate", "rounds", "seed_base")})
    out["presets"] = [own.get(p["settings"], p["settings"]) for p in plan["presets"]]
    return out


def save_plan(ns, name, settings, per_round, stop_at, max_hours, card, polish) -> str:
    """Write the plan the controls show. Saving over an existing plan keeps what the tab has no control for: a setting's own
    count / prompt suffix (when the setting is still in the plan), the gate, the round limit and the seed base."""
    n = _safe(name)
    if not n:
        raise ValueError("Give the plan a name.")
    out = merged_plan(ns, n, plan_from_controls(settings, per_round, stop_at, max_hours, card, polish))
    d = _loop_dir(ns)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{n}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


def read_plan(ns, name) -> dict | None:
    try:
        return LP.load_plan(_loop_dir(ns) / f"{_safe(name)}.json")
    except Exception:
        return None


def groups(ns) -> list[str]:
    """The folders of saved settings ('portraits', 'scenes', …), for 'add a whole folder'."""
    return sorted({n.split("/", 1)[0] for n in ns._list_saved_settings() if "/" in n})


def add_group(ns, current, group) -> list[str]:
    """The current selection plus every saved setting of `group`, in order, no repeats."""
    cur = list(current or [])
    if group:
        cur += [n for n in ns._list_saved_settings() if n.startswith(group + "/")]
    return list(dict.fromkeys(cur))


def _hm(minutes: float) -> str:
    return f"{minutes:.0f} min" if minutes < 90 else f"{minutes / 60:.1f} h"


HEAT_FACTOR = 1.25   # the app's pause while hot adds ~25 % per picture on the RX 6800M laptop at full speed (rounds 41-54)


def check_plan(ns, plan: dict, now=None) -> str:
    """What the plan will do before it runs: each setting (found? model / LoRAs installed? wildcards there? what Polish picks),
    the time per picture (measured by an earlier run, else settings_info's rough estimate), a round's length and how many
    rounds fit before the stop time / hour budget. The problems that would waste a night come first."""
    import datetime as _dt
    from backend import character_cards as CC
    from backend import settings_info as SI
    e = html.escape
    try:
        ck = [n for n, _ in ns.list_checkpoints()]
        lo = [n for n, _ in ns.list_loras()]
    except Exception:
        ck, lo = [], []
    meas = SI.measured_seconds([_loop_dir(ns), Path(ns.OUTPUTS_DIR) / "loop"])
    rows, warn, round_s, pics = [], [], 0.0, 0
    for q in plan["presets"]:
        name = q["settings"]
        d = SI.read(ns.SETTINGS_DIR, name)
        if not d:
            warn.append(f"<b>{e(name)}</b>: saved setting not found — the loop will refuse to start")
            rows.append(f"<tr><td>{e(name)}</td><td colspan=4 style='color:#f38ba8'>not found</td></tr>")
            continue
        if plan.get("polish"):
            d = dict(d, extra=dict(SI.extra_of(d), polish=plan["polish"]))
        s = SI.summary(d, ck, lo)
        if not s["checkpoint_ok"]:
            warn.append(f"<b>{e(name)}</b>: checkpoint {e(s['checkpoint'])} is not installed (the loaded model would be used)")
        if s["loras_missing"]:
            warn.append(f"<b>{e(name)}</b>: LoRA not installed: {e(', '.join(s['loras_missing']))}")
        if s["wildcards_missing"]:
            warn.append(f"<b>{e(name)}</b>: wildcard file missing: {e(', '.join(s['wildcards_missing']))}")
        secs = meas.get(name) or s["seconds"] * HEAT_FACTOR
        n = q["count"]
        round_s += n * secs
        pics += n
        pol = s["polish"] if s["polish"] != "Auto" else "Auto → " + ", ".join(k for k, _ in s["picks"].most_common(2))
        rows.append(f"<tr><td>{e(name)}</td><td>{n}</td><td>{e(pol)}</td>"
                    f"<td>{SI.fmt_secs(secs)}{'' if meas.get(name) else ' ~'}</td><td>{e(s['checkpoint'])}</td></tr>")
    now = now or _dt.datetime.now()
    left = plan["max_hours"] * 60
    if plan["stop_at"]:
        hh, mm = map(int, plan["stop_at"].split(":"))
        stop = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if stop <= now:
            stop += _dt.timedelta(days=1)
        left = min(left, (stop - now).total_seconds() / 60)
    round_m = round_s / 60
    fit = left / round_m if round_m else 0
    if plan.get("card") and plan["card"] not in CC.list_cards():
        warn.append(f"card <b>{e(plan['card'])}</b> not found: the face check runs without her identity")
    if round_m and fit < 1:
        warn.append(f"one round (~{_hm(round_m)}) is longer than the time left ({_hm(left)}): only part of it will run")
    head = (f"<p>One round ≈ <b>{_hm(round_m)}</b> ({pics} pictures); time left {_hm(left)} → about <b>{fit:.1f} round(s)</b>, "
            f"~{int(fit * pics)} pictures. ‘~’ = a rough estimate with the heat pause (no loop run has timed that setting yet); "
            "settings that keep failing get parked and starred ones get more, so later rounds differ.</p>")
    w = "".join(f"<p style='color:#f9e2af;margin:2px 0'>⚠ {x}</p>" for x in warn)
    table = ("<table style='font-size:13px;border-collapse:collapse'><tr style='color:#a6adc8'><td>saved setting</td><td>per round</td>"
             "<td>✨ Polish</td><td>per picture</td><td>checkpoint</td></tr>" + "".join(rows) + "</table>")
    return w + head + table


def view(ns, verdict: str = "best", run: Path | None = None) -> list[str]:
    """The pictures of one verdict folder of the latest run (newest first)."""
    run = run or latest_run(ns, STATE["plan"])
    if run is None or verdict not in LP.VERDICTS:
        return []
    return [str(p) for p in sorted((run / verdict).glob("*.png"), key=lambda p: -p.stat().st_mtime)[:48]]


def star(path) -> str:
    """⭐ a loop picture: it goes into the History favourites (settings/_favourites.json, by file name) and the next round gives its
    saved setting one more picture."""
    from backend import history as H
    if not path:
        return "Click a picture first."
    name = Path(str(path)).name
    on = H.toggle_favourite(name)
    return ("⭐ Starred " if on else "☆ Unstarred ") + html.escape(name) + (
        " — its saved setting gets one more picture next round." if on else ".")


def test_plan(ns, name) -> str:
    """Write settings/loop/_test_<name>.json: the plan with ONE picture of each setting and one round (a check before a night run:
    every setting loads, renders and is scored; the run folder is _test_<name>_<time>). Returns its name, '' when there is no plan."""
    plan = read_plan(ns, name)
    if not plan:
        return ""
    t = dict(plan, rounds=1, per_round=1, stop_at="", max_hours=3,
             presets=[{"settings": q["settings"], "count": 1, "prompt_suffix": q["prompt_suffix"]} for q in plan["presets"]])
    n = "_test_" + _safe(name)
    (_loop_dir(ns) / f"{n}.json").write_text(json.dumps(t, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


def start(ns, name, popen=subprocess.Popen, test: bool = False) -> str:
    if running():
        return f"⚠ A loop is already running ({html.escape(STATE['plan'])}). Stop it first."
    if test:
        name = test_plan(ns, name)
    plan = read_plan(ns, name) if name else None
    if not plan:
        return "⚠ Pick or save a plan first."
    missing = [p["settings"] for p in plan["presets"] if not (Path(ns.SETTINGS_DIR) / f"{p['settings']}.json").exists()]
    if missing:
        return f"⚠ Saved settings not found: {html.escape(', '.join(missing))}"
    log = _loop_dir(ns) / f"{_safe(name)}.log"
    fh = open(log, "w", encoding="utf-8", errors="replace")
    STATE.update(proc=popen([sys.executable, "-u", str(APP / "studio_loop.py"), str(_loop_dir(ns) / f"{_safe(name)}.json"),
                             "--attach", _url(ns), "--raw", str(ns.OUTPUTS_DIR)],
                            cwd=str(APP), stdout=fh, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)),
                 plan=_safe(name), log=log, started=time.time())
    if test:
        return (f"🧪 Test run of “{html.escape(_safe(name)[6:])}”: one picture of each of its {len(plan['presets'])} saved setting(s), one round. "
                "🔄 Status shows how it went; then ▶ Start the real plan.")
    return (f"▶ Started “{html.escape(_safe(name))}”: {len(plan['presets'])} saved setting(s), "
            f"{plan['per_round']} per round, stops {('at ' + plan['stop_at']) if plan['stop_at'] else 'after ' + format(plan['max_hours'], 'g') + ' h'}. "
            "It renders through this app, one picture job at a time (a Generate of yours waits its turn).")


def stop(ns) -> str:
    run = latest_run(ns, STATE["plan"]) if STATE["plan"] else None
    if not running() or run is None:
        return "Nothing is running."
    (run / "STOP").write_text("stop", encoding="utf-8")
    return f"⏹ Stopping after the current saved setting (STOP written to {html.escape(str(run))})."


def status(ns) -> tuple[str, list]:
    lines = []
    if running():
        mins = (time.time() - STATE["started"]) / 60
        lines.append(f"🟢 Running “{html.escape(STATE['plan'])}” for {mins:.0f} min.")
    elif STATE["proc"] is not None:
        lines.append(f"⚪ The last loop (“{html.escape(STATE['plan'])}”) has ended.")
    log = STATE["log"]
    if log and Path(log).exists():
        tail = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
        lines.append("<pre style='white-space:pre-wrap;font-size:12px'>" + html.escape("\n".join(tail)) + "</pre>")
    run = latest_run(ns, STATE["plan"])
    gallery = []
    if run is not None:
        rep = run / "report.md"
        if rep.exists():
            head = rep.read_text(encoding="utf-8").split("## Needs a prompt fix")[0]
            lines.append(f"<p>Latest run: <code>{html.escape(str(run))}</code> — open <code>report.html</code> there for every picture.</p>"
                         "<pre style='white-space:pre-wrap;font-size:12px'>" + html.escape(head.strip()) + "</pre>")
        gallery = [str(p) for p in sorted((run / "best").glob("*.png"), key=lambda p: -p.stat().st_mtime)[:24]]
    return ("".join(lines) or "No loop has run yet."), gallery


def build_tab(ns) -> None:
    from backend import character_cards as CC
    from backend import polish as PO
    with gr.Tab("🔁 Studio Loop", id="loop"):
        gr.Markdown("Render your **saved settings** round after round without you at the computer (overnight), score every picture and "
                    "sort it into best / ok / review / reject under `outputs/loop/<run>/`. Settings that keep failing are parked, settings "
                    "whose pictures you ⭐ (here or in History) get more. Stops at the stop time, the hour budget, or ⏹ Stop.\n\n"
                    "**Evening:** pick a plan (or pick saved settings / ➕ a whole folder and 💾 save it) → **📋 Check plan** (what is missing, "
                    "time per round, how many rounds fit) → **▶ Start**. **Morning:** 🔄 Status → browse best / ok / review / reject → "
                    "⭐ the pictures you like → `report.md` in the run folder lists what each setting got wrong.")
        with gr.Row():
            plans = list_plans(ns)
            plan_dd = gr.Dropdown(plans, value=plans[0] if plans else None, label="Plan", scale=3)
            first = plans[0] if plans else ""
            cards = [""] + CC.list_cards()
            pol_choices = [c for c in _POLISH if c == "" or c in PO.CHOICES]

            def _init(key, default):
                """The control's value from the plan the dropdown starts on, read again at every page load (a .change does not fire
                for a starting value, so the controls used to open empty next to a selected plan)."""
                def f():
                    p = read_plan(ns, first) if first else None
                    if not p:
                        return default
                    v = {"settings": [q["settings"] for q in p["presets"]], "per_round": p["per_round"], "stop_at": p["stop_at"],
                         "max_hours": p["max_hours"], "card": p["card"] if p["card"] in cards else "",
                         "polish": p["polish"] if p["polish"] in pol_choices else "", "name": first,
                         "check": None}[key]
                    return check_plan(ns, p) if key == "check" else v
                return f
            refresh_btn = gr.Button("🔄", scale=0, min_width=60)
        set_ms = gr.Dropdown(ns._list_saved_settings(), value=_init("settings", []), multiselect=True, label="Saved settings to render")
        with gr.Row():
            grp_dd = gr.Dropdown(groups(ns), value=None, label="…or add every setting of a folder", scale=3)
            grp_btn = gr.Button("➕ Add folder", scale=1)
        with gr.Row():
            per_sl = gr.Slider(1, 8, value=_init("per_round", 2), step=1, label="Pictures per setting per round")
            stop_tb = gr.Textbox(value=_init("stop_at", "07:00"), label="Stop at (HH:MM, empty = only the hour budget)")
            hours_nb = gr.Number(value=_init("max_hours", 8), label="Hour budget", precision=1)
        with gr.Row():
            card_dd = gr.Dropdown(cards, value=_init("card", ""), label="Character card for the face check (optional)")
            polish_dd = gr.Dropdown(pol_choices, value=_init("polish", ""), label="✨ Polish to force (empty = each setting's own)")
        with gr.Row():
            name_tb = gr.Textbox(value=_init("name", ""), label="Plan name", scale=3)
            save_btn = gr.Button("💾 Save plan", scale=1)
        with gr.Row():
            start_btn = gr.Button("▶ Start", variant="primary")
            test_btn = gr.Button("🧪 Test run (1 of each)")
            stop_btn = gr.Button("⏹ Stop after this setting")
            stat_btn = gr.Button("🔄 Status")
            check_btn = gr.Button("📋 Check plan")
        msg = gr.HTML("")
        check_html = gr.HTML(_init("check", ""))
        stat_html = gr.HTML("")
        with gr.Row():
            verdict_rd = gr.Radio(list(LP.VERDICTS), value="best", label="Show the latest run's", scale=3)
            star_btn = gr.Button("⭐ Star the clicked picture", scale=1)
        best = gr.Gallery(label="Pictures of the latest run (click one, then ⭐: its setting gets more next round)", columns=6, height=420)
        picked = gr.State("")

        def on_loop_pick(name):
            p = read_plan(ns, name) if name else None
            if not p:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            return ([q["settings"] for q in p["presets"]], p["per_round"], p["stop_at"], p["max_hours"], p["card"],
                    p["polish"] if p["polish"] in _POLISH else "", name)

        def on_loop_save(name, settings, per_round, stop_at, max_hours, card, polish):
            try:
                n = save_plan(ns, name, settings, per_round, stop_at, max_hours, card, polish)
            except ValueError as e:
                return f'<p style="color:#f38ba8;">⚠ {html.escape(str(e))}</p>', gr.update()
            return (f'<p style="color:#a6e3a1;">✅ Saved plan “{html.escape(n)}”</p>',
                    gr.update(choices=list_plans(ns), value=n))

        def on_loop_refresh():
            return gr.update(choices=list_plans(ns)), gr.update(choices=ns._list_saved_settings()), gr.update(choices=[""] + CC.list_cards())

        def on_loop_start(name):
            return f"<p>{start(ns, name)}</p>"

        def on_loop_stop():
            return f"<p>{stop(ns)}</p>"

        def on_loop_status():
            return status(ns)

        # the helpers below are UI-only (api_name=False): the tab's named endpoints stay as they were
        def on_loop_check(name, settings, per_round, stop_at, max_hours, card, polish):
            try:
                return check_plan(ns, LP.clean_plan(merged_plan(ns, name, plan_from_controls(settings, per_round, stop_at, max_hours, card, polish))))
            except ValueError as ex:
                return f'<p style="color:#f38ba8;">⚠ {html.escape(str(ex))}</p>'

        def on_loop_check_saved(name):
            p = read_plan(ns, name) if name else None
            return check_plan(ns, p) if p else ""

        def on_loop_group(current, group):
            return gr.update(value=add_group(ns, current, group))

        def on_loop_view(verdict):
            return view(ns, verdict), ""

        def on_loop_select(verdict, evt: gr.SelectData):
            files = view(ns, verdict)
            i = evt.index if isinstance(evt.index, int) else (list(evt.index or [0]) or [0])[0]
            return files[i] if isinstance(i, int) and 0 <= i < len(files) else ""

        def on_loop_test(name):
            return f"<p>{start(ns, name, test=True)}</p>"

        def on_loop_star(path):
            return f"<p>{star(path)}</p>"

        plan_dd.change(on_loop_pick, [plan_dd], [set_ms, per_sl, stop_tb, hours_nb, card_dd, polish_dd, name_tb])
        save_btn.click(on_loop_save, [name_tb, set_ms, per_sl, stop_tb, hours_nb, card_dd, polish_dd], [msg, plan_dd])
        plan_dd.change(on_loop_check_saved, [plan_dd], [check_html], api_name=False)
        check_btn.click(on_loop_check, [name_tb, set_ms, per_sl, stop_tb, hours_nb, card_dd, polish_dd], [check_html], api_name=False)
        grp_btn.click(on_loop_group, [set_ms, grp_dd], [set_ms], api_name=False)
        verdict_rd.change(on_loop_view, [verdict_rd], [best, picked], api_name=False)
        best.select(on_loop_select, [verdict_rd], [picked], api_name=False)
        star_btn.click(on_loop_star, [picked], [msg], api_name=False)
        test_btn.click(on_loop_test, [plan_dd], [msg], api_name=False).then(on_loop_status, None, [stat_html, best], api_name=False)
        refresh_btn.click(on_loop_refresh, None, [plan_dd, set_ms, card_dd])
        refresh_btn.click(lambda: gr.update(choices=groups(ns)), None, [grp_dd], api_name=False)
        start_btn.click(on_loop_start, [plan_dd], [msg]).then(on_loop_status, None, [stat_html, best], api_name=False)
        stop_btn.click(on_loop_stop, None, [msg])
        stat_btn.click(on_loop_status, None, [stat_html, best]).then(lambda: ("best", ""), None, [verdict_rd, picked], api_name=False)
