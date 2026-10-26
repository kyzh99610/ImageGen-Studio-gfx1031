"""
settings_info.py — what a saved setting will do, in plain words (pure: no UI, no GPU).

Used by the Generate tab's ⚙️ Save / Load box (a card under the dropdown when a setting is picked) and by the 🔁 Studio Loop
tab's plan check. It reads the setting's JSON and says: the checkpoint and LoRAs (and whether they are installed here), the
sampler / steps / CFG / size, which passes run (hires, face, hands, eyes, max detail, colour, pose / composition), what
✨ Polish = Auto will choose (for a prompt with wildcards: counted over sample draws, since each picture's own framing tag
decides), the wildcards it uses (missing ones marked) and a rough time per picture.

The time is a model, not a measurement: seconds per sampling step ≈ 1.6 × megapixels (× 1.5 with PAG), long prompts cost
~4 % more per extra 75-token chunk, Real-ESRGAN's 2x stage ~45 s, a face / eye pass ~12-15 s, a hand ~15 s. Calibrated on
the RX 6800M at full speed (a cowboy shot of AYS 12 + PAG 2 ≈ 35 s, the same with hires 1.5x Real-ESRGAN ≈ 140-170 s,
a 30-step long-prompt preset with every pass ≈ 330-410 s). The app's pause while hot comes on top. When a Studio Loop run
already timed a setting, the measured number is used instead (`measured_seconds`).
"""
from __future__ import annotations

import html
import json
import math
import re
from collections import Counter
from pathlib import Path

from backend import polish as PO
from backend import wildcards as WC

_WILD = re.compile(r"__([\w/\-]+)__")


RESERVED_DIRS = {"characters", "loop", "poses", "composition", "cache", "temp", "wildcards"}   # = app._RESERVED_SETTING_DIRS


def list_saved(settings_dir) -> list[str]:
    """The saved settings as the app lists them ('group/name' first, then top-level names), for programs that cannot import app.py
    (the ComfyUI node)."""
    s = Path(settings_dir)
    if not s.is_dir():
        return []
    top = sorted(p.stem for p in s.glob("*.json") if p.is_file() and not p.stem.startswith("_") and p.stem.casefold() != "api_keys")
    ok = lambda g: g and not g.startswith((".", "_")) and g.casefold() not in RESERVED_DIRS
    grouped = sorted(f"{d.name}/{p.stem}" for d in s.iterdir() if d.is_dir() and ok(d.name)
                     for p in d.glob("*.json") if p.is_file() and not p.stem.startswith("_"))
    return grouped + top


def read(settings_dir, name: str) -> dict | None:
    """The saved setting `name` ('group/name' or 'name'), or None."""
    if not name:
        return None
    try:
        d = json.loads((Path(settings_dir) / f"{name}.json").read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def extra_of(d: dict) -> dict:
    ex = d.get("extra") if isinstance(d.get("extra"), dict) else {}
    return dict(ex)


def wildcards_used(text: str) -> list[tuple[str, int | None]]:
    """(name, number of lines or None when the file is missing) for every __name__ in the text, in order."""
    out = []
    for n in dict.fromkeys(_WILD.findall(text or "")):
        lines = WC._lines(n)
        out.append((n, len(lines) if lines else None))
    return out


def polish_picks(d: dict, draws: int = 40) -> Counter:
    """What ✨ Polish chooses for this setting's pictures. A fixed entry: that entry for every picture. Auto: the recipe of each
    picture's own resolved prompt (wildcards drawn with seeds 0..draws-1), so a set that mixes framings shows the mix. '(off)': empty."""
    ex = extra_of(d)
    choice = PO.clean_choice(ex.get("polish"))
    if choice == "(off)":
        return Counter()
    if choice != "Auto":
        return Counter({choice: 1})
    pose = bool(ex.get("pc_on")) and str(ex.get("pc_type") or "pose").lower().startswith("pose")
    prompt = str(d.get("prompt") or "")
    if not WC.is_dynamic(prompt):
        return Counter({PO.shot_type(prompt, pose=pose): 1})
    return Counter(PO.shot_type(WC.resolve(prompt, s), pose=pose) for s in range(draws))


def passes(ex: dict) -> dict:
    """The passes the controls switch on: {hires, face, hands, eyes, max_detail} -> settings."""
    return dict(hires=(ex.get("hires_scale", 1.5), ex.get("hires_upscaler", "Lanczos"), ex.get("hires_denoise", 0.45), ex.get("hires_steps", 12))
                if ex.get("hires_on") else None,
                face=ex.get("fd_denoise", 0.35) if ex.get("fd_on") else None,
                hands=ex.get("hd_denoise", 0.35) if ex.get("hd_on") else None,
                eyes=ex.get("ed_denoise", 0.4) if ex.get("ed_on") else None,
                max_detail=bool(ex.get("md_on")))


def _chunks(text: str) -> int:
    """75-token chunks of a prompt, roughly (≈ 290 characters a chunk)."""
    return max(1, math.ceil(len(text or "") / 290))


def _resolved(text: str) -> str:
    return WC.resolve(text or "", 0) if WC.is_dynamic(text or "") else (text or "")


def estimate_seconds(d: dict, ex: dict | None = None) -> float:
    """Rough seconds for ONE picture at full speed, without the heat pause. `ex` = the extra settings that run (a Polish recipe
    applied on top of the saved ones), default the saved ones."""
    ex = dict(extra_of(d) if ex is None else ex)
    w, h = _f(d.get("width"), 832), _f(d.get("height"), 1216)
    mp = max(0.1, w * h / 1e6)
    steps = max(1.0, _f(d.get("steps"), 20))
    chunk = 1 + 0.04 * (max(_chunks(_resolved(d.get("prompt"))), _chunks(_resolved(d.get("negative_prompt")))) - 1)
    pag = 1.5 if _f(ex.get("pag_scale")) > 0 and not ex.get("pc_on") else 1.0
    pc = 1.15 if ex.get("pc_on") else 1.0          # a ControlNet step costs ~15 % more (and PAG is forced off)
    t = 6 + steps * 1.6 * mp * pag * pc * chunk
    p = passes(ex)
    if p["hires"]:
        sc, up, _den, hst = p["hires"]
        t += _f(hst, 12) * 1.6 * mp * _f(sc, 1.5) ** 2 * chunk + (45 if "ESRGAN" in str(up) else 5)
        mp *= _f(sc, 1.5) ** 2
    if p["face"] is not None:
        t += 15
    if p["hands"] is not None:
        t += 22                                     # ~1.5 confirmed hands a picture
    if p["max_detail"]:
        t += 100
        mp *= 4
    if p["eyes"] is not None:
        t += 12 if mp < 6 else 20
    return t


def estimate_with_polish(d: dict, picks: Counter | None = None) -> float:
    """Seconds per picture with what Polish will choose (weighted over the picks)."""
    picks = polish_picks(d) if picks is None else picks
    base = extra_of(d)
    if not picks:
        return estimate_seconds(d, base)
    tot = sum(picks.values())
    return sum(n * estimate_seconds(d, dict(base, **PO.recipe(name))) for name, n in picks.items()) / tot


def measured_seconds(loop_roots) -> dict:
    """{saved setting: median seconds per picture} from Studio Loop runs (records.jsonl 'secs' per picture; older runs' logs
    have 'name: n picture(s) in T s' lines). The newest data wins per setting."""
    per: dict[str, list[float]] = {}
    files = []
    for root in loop_roots:
        root = Path(root)
        if root.is_dir():
            files += [(p.stat().st_mtime, p) for p in root.glob("*/records.jsonl")]
            files += [(p.stat().st_mtime, p) for p in root.glob("*.log")]
    for _, p in sorted(files):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        got: dict[str, list[float]] = {}
        if p.name == "records.jsonl":
            for line in text.splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("secs"):
                    got.setdefault(str(r.get("settings")), []).append(float(r["secs"]))
        else:
            for name, n, secs in re.findall(r"^\S+ (.+?): (\d+) picture\(s\) in (\d+) s$", text, re.M):
                if int(n) > 0:
                    got.setdefault(name, []).append(int(secs) / int(n))
        for k, v in got.items():
            per[k] = v[-8:]                       # a newer file replaces an older one's numbers
    return {k: sorted(v)[len(v) // 2] for k, v in per.items() if v}


def fmt_secs(s: float) -> str:
    s = float(s)
    return f"{s:.0f} s" if s < 90 else (f"{s / 60:.1f} min" if s < 5400 else f"{s / 3600:.1f} h")


def _stem(p) -> str:
    return Path(str(p).replace("\\", "/")).stem


def summary(d: dict, local_checkpoints=(), local_loras=()) -> dict:
    """The facts behind the card, as a dict (also used by the Loop tab's plan check)."""
    ex = extra_of(d)
    ck = _stem(d.get("model") or "")
    ck_ok = (not ck) or any(_stem(n).lower() == ck.lower() for n in local_checkpoints)
    loras, missing = [], []
    for item in d.get("lora_slots") or [[n, 0.8] for n in d.get("loras") or []]:
        if not isinstance(item, (list, tuple)) or not item or not item[0] or item[0] == "none":
            continue
        n, w = _stem(item[0]), (_f(item[1], 0.8) if len(item) > 1 else 0.8)
        loras.append((n, w))
        if local_loras and not any(_stem(x).lower() == n.lower() for x in local_loras):
            missing.append(n)
    picks = polish_picks(d)
    wild = wildcards_used(f"{d.get('prompt') or ''} {d.get('negative_prompt') or ''}")
    return dict(checkpoint=ck, checkpoint_ok=ck_ok, loras=loras, loras_missing=missing,
                sampler=str(d.get("scheduler") or ""), steps=d.get("steps"), cfg=d.get("cfg_scale"),
                size=f"{int(_f(d.get('width'), 0))}×{int(_f(d.get('height'), 0))}", batch=int(_f(d.get("batch_size"), 1)),
                polish=PO.clean_choice(ex.get("polish")), picks=picks, passes=passes(ex),
                pag=_f(ex.get("pag_scale")), colour=_f(ex.get("cs"), 1.0),
                control=(str(ex.get("pc_type") or "pose") if ex.get("pc_on") else ""),
                wildcards=wild, wildcards_missing=[n for n, c in wild if c is None],
                seconds=estimate_with_polish(d, picks), note=str(d.get("note") or ""))


def _passes_text(p: dict) -> str:
    out = []
    if p["hires"]:
        sc, up, den, st = p["hires"]
        out.append(f"hires {_f(sc, 1.5):g}× {'Real-ESRGAN' if 'ESRGAN' in str(up) else up} {_f(den, 0.45):g} / {int(_f(st, 12))} steps")
    if p["face"] is not None:
        out.append(f"face {p['face']:g}")
    if p["hands"] is not None:
        out.append(f"hands {p['hands']:g}")
    if p["max_detail"]:
        out.append("max detail 2×")
    if p["eyes"] is not None:
        out.append(f"eyes {p['eyes']:g}")
    return ", ".join(out) or "none"


def card_html(name: str, d: dict | None, local_checkpoints=(), local_loras=(), measured: float | None = None) -> str:
    """The card shown under the Saved Settings dropdown."""
    if not d:
        return ""
    s = summary(d, local_checkpoints, local_loras)
    e = html.escape
    warn = []
    if not s["checkpoint_ok"]:
        warn.append(f"checkpoint <b>{e(s['checkpoint'])}</b> is not installed")
    if s["loras_missing"]:
        warn.append("LoRA not installed: " + e(", ".join(s["loras_missing"])))
    if s["wildcards_missing"]:
        warn.append("wildcard file missing: " + e(", ".join(f"__{n}__" for n in s["wildcards_missing"])) + " (it stays as text in the prompt)")
    rows = [("Model", e(s["checkpoint"] or "(the loaded one)") + ("" if not s["loras"] else " + " + e(", ".join(f"{n} {w:g}" for n, w in s["loras"])))),
            ("Recipe", e(f"{s['sampler']}, {s['steps']} steps, CFG {s['cfg']}, {s['size']}, batch {s['batch']}")
             + (f", PAG {s['pag']:g}" if s["pag"] else "") + (f", colour {s['colour']:g}" if s["colour"] < 1 else "")
             + (f", {e(s['control'])} control on" if s["control"] else ""))]
    if s["polish"] == "Auto":
        tot = sum(s["picks"].values()) or 1
        mix = ", ".join(f"{e(k)} {round(100 * v / tot)} %" for k, v in s["picks"].most_common())
        rows.append(("✨ Polish", "Auto — each picture's framing tag picks the recipe" + (f" (here: {mix})" if mix else "") + "."))
    elif s["polish"] != "(off)":
        rows.append(("✨ Polish", e(s["polish"]) + " — fills the passes once when loaded"))
    else:
        rows.append(("✨ Polish", "off — the passes run as saved"))
    if s["picks"]:
        top = s["picks"].most_common(1)[0][0]
        rows.append(("Passes", e(_passes_text(passes(dict(extra_of(d), **PO.recipe(top))))) + f" <span style='color:#a6adc8'>({e(top)})</span>"))
    else:
        rows.append(("Passes", e(_passes_text(s["passes"]))))
    if s["wildcards"]:
        rows.append(("Wildcards", ", ".join(f"__{e(n)}__ ({c} lines)" if c else f"<b>__{e(n)}__ missing</b>" for n, c in s["wildcards"])))
    t = (f"{fmt_secs(measured)} per picture (measured by the Studio Loop, the pause while hot included)" if measured
         else f"about {fmt_secs(s['seconds'])} per picture at full speed (a rough estimate; the pause while hot comes on top)")
    rows.append(("Time", t))
    if s["note"]:
        rows.append(("Note", e(s["note"][:300])))
    body = "".join(f"<tr><td style='color:#a6adc8;padding:1px 8px 1px 0;vertical-align:top;white-space:nowrap'>{k}</td><td>{v}</td></tr>" for k, v in rows)
    w = "".join(f"<p style='color:#f9e2af;margin:2px 0'>⚠ {x}</p>" for x in warn)
    return (f"<div style='font-size:13px;border:1px solid #45475a;border-radius:8px;padding:6px 10px;margin:4px 0'>"
            f"<b>{e(name)}</b>{w}<table style='border-collapse:collapse;margin-top:4px'>{body}</table></div>")
