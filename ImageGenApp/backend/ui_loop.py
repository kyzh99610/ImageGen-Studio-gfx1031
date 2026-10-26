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
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


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


def save_plan(ns, name, settings, per_round, stop_at, max_hours, card, polish) -> str:
    n = _safe(name)
    if not n:
        raise ValueError("Give the plan a name.")
    plan = plan_from_controls(settings, per_round, stop_at, max_hours, card, polish)
    d = _loop_dir(ns)
    d.mkdir(parents=True, exist_ok=True)
    out = dict(plan)
    out["presets"] = [p["settings"] for p in plan["presets"]]
    (d / f"{n}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


def read_plan(ns, name) -> dict | None:
    try:
        return LP.load_plan(_loop_dir(ns) / f"{_safe(name)}.json")
    except Exception:
        return None


def start(ns, name, popen=subprocess.Popen) -> str:
    if running():
        return f"⚠ A loop is already running ({html.escape(STATE['plan'])}). Stop it first."
    plan = read_plan(ns, name)
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
                    "whose pictures you ⭐ in History get more. Stops at the stop time, the hour budget, or ⏹ Stop.")
        with gr.Row():
            plans = list_plans(ns)
            plan_dd = gr.Dropdown(plans, value=plans[0] if plans else None, label="Plan", scale=3)
            refresh_btn = gr.Button("🔄", scale=0, min_width=60)
        set_ms = gr.Dropdown(ns._list_saved_settings(), value=[], multiselect=True, label="Saved settings to render")
        with gr.Row():
            per_sl = gr.Slider(1, 8, value=2, step=1, label="Pictures per setting per round")
            stop_tb = gr.Textbox(value="07:00", label="Stop at (HH:MM, empty = only the hour budget)")
            hours_nb = gr.Number(value=8, label="Hour budget", precision=1)
        with gr.Row():
            cards = [""] + CC.list_cards()
            card_dd = gr.Dropdown(cards, value="", label="Character card for the face check (optional)")
            polish_dd = gr.Dropdown([c for c in _POLISH if c == "" or c in PO.CHOICES], value="", label="✨ Polish to force (empty = each setting's own)")
        with gr.Row():
            name_tb = gr.Textbox(value="", label="Plan name", scale=3)
            save_btn = gr.Button("💾 Save plan", scale=1)
        with gr.Row():
            start_btn = gr.Button("▶ Start", variant="primary")
            stop_btn = gr.Button("⏹ Stop after this setting")
            stat_btn = gr.Button("🔄 Status")
        msg = gr.HTML("")
        stat_html = gr.HTML("")
        best = gr.Gallery(label="Best pictures of the latest run", columns=6, height=420)

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

        plan_dd.change(on_loop_pick, [plan_dd], [set_ms, per_sl, stop_tb, hours_nb, card_dd, polish_dd, name_tb])
        save_btn.click(on_loop_save, [name_tb, set_ms, per_sl, stop_tb, hours_nb, card_dd, polish_dd], [msg, plan_dd])
        refresh_btn.click(on_loop_refresh, None, [plan_dd, set_ms, card_dd])
        start_btn.click(on_loop_start, [plan_dd], [msg])
        stop_btn.click(on_loop_stop, None, [msg])
        stat_btn.click(on_loop_status, None, [stat_html, best])
