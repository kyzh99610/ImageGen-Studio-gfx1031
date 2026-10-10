"""
polish.py — ✨ Polish: one click sets the detail passes (hires fix, face, eye) to the recipe for the kind of picture.

Where the recipes come from (AGENTS.md "Fifth-session round"): round 4 measured what each pass does at each face size (face detail does
nothing once the face is ≥ 25 % of the width, equals hires 1.5× at 14–20 %, and hires 1.5× + face detail is best at ≤ 8 %; hires alone rescues
a 14 % face as well as the face pass, so they are not stacked there); round 5 re-measured the face denoise per face width and the eye pass
(eye-only prompt, denoise 0.4). The hand pass is left out: it cleans texture but never fixed a finger count. The planned per-recipe chain
study (hires vs 2× + tiled detail, Real-ESRGAN vs Lanczos) was cut: the laptop powers off under sustained GPU load. What was measured instead (an Illustrious checkpoint + a character LoRA, 2 seeds per
shot, CCIP identity / pin / sharpness per chain): at cowboy framing (faces 24–28 %) nothing in the chain moves identity — it is insurance; at full-body framing (14 %) the face pass
lifts the plain picture (CCIP 0.937 → 0.953, pin .42 → .79), hires 1.5× + eyes lifts it more and sharpens (0.967, pin .90, face ×1.3) and the face pass on top of hires adds nothing —
the Full body recipe, 14 % faster than the chain with it. The time factors below are round 4's unthrottled pieces added up (the throttled chains measured 1.9× / 3.0× / 3.5×).
A recipe is a dict of the Generate tab's extra settings (`app._clean_extra` keys).
"""
from __future__ import annotations

import re

SHOTS = ("Portrait", "Cowboy shot", "Full body", "Wide shot", "Full body — max detail", "Wide shot — max detail")

# extra-settings keys of app._clean_extra that a recipe may set; anything not listed is left as the user has it
_KEYS = ("hires_on", "hires_scale", "hires_denoise", "hires_steps", "hires_upscaler", "fd_on", "fd_denoise", "hd_on", "hd_denoise",
         "ed_on", "ed_denoise", "md_on")

RECIPES: dict[str, dict] = {
    # face >= 25 % of the width: the face pass changes nothing, the eye pass tidies the eyes and adds iris / pupil detail (+13 s)
    "Portrait":    dict(hires_on=False, fd_on=False, hd_on=False, ed_on=True, ed_denoise=0.4),
    # a cowboy shot at 832x1216 puts the face at 24-28 % of the width: the face pass measured no gain there (+0.006 CCIP, -9 % face
    # sharpness), so eyes only, like Portrait (the user's call after round 5; a smaller face still gets the face pass under Full body / Wide)
    "Cowboy shot": dict(hires_on=False, fd_on=False, hd_on=False, ed_on=True, ed_denoise=0.4),
    # face ~11-14 %: hires 1.5x alone rescues it (+1.2 identity, stacking the face pass adds nothing); more pixels for the eyes. 8 hires steps, not 14 (round 8, 4 pictures: CCIP -0.001, pin +-0.00,
    # face sharpness x0.98, eye energy x0.98, nothing by eye; hires denoise 0.40 instead of 0.45 was slightly worse at both step counts: CCIP -0.004, sharpness -6 to -8 %)
    "Full body":   dict(hires_on=True, hires_scale=1.5, hires_denoise=0.45, hires_steps=8, hires_upscaler="Lanczos",
                        fd_on=False, hd_on=False, ed_on=True, ed_denoise=0.4),
    # face <= 8 %: hires 1.5x + face detail is best (+3.4 identity, the pin comes back); the face pass raises its own denoise for small faces. 14 hires steps, not 8 (round 9, 3 seeds, hassakuXL,
    # a 2 x 2 of hires steps 14 | 8 x eye steps 10 | 8: eye energy x1.00 / 0.98 / 0.74 / 0.75 - it is the hires step count that decides, the eye pass's 8 steps cost nothing; identity and face sharpness were equal,
    # the eyes ~25 % softer and visibly so in 2 of 3 pictures; 8 steps saved 23 % of the chain)
    "Wide shot":   dict(hires_on=True, hires_scale=1.5, hires_denoise=0.45, hires_steps=14, hires_upscaler="Lanczos",
                        fd_on=True, fd_denoise=0.35, hd_on=False, ed_on=True, ed_denoise=0.4),
    # round 14 (hassakuXL + waiIllustrious, 3 seeds each, 832x1216): the Full body chain + Real-ESRGAN 2x right before the eye pass (the picture comes out 2496x3648): the eyes get twice the pixels,
    # eye detail at the same display size x7 (Laplacian energy of the eye boxes at 144 px), face sharpness x1.7, CCIP +0.004, red eyes +0.025, pin -0.06 / +0.02 (hassaku / wai), whole-picture
    # high-frequency energy x1.9 (crisper linework); +49 % time. Lanczos 2x + the eye pass is free (x0.99 time) but only x2 eye energy; 2x + tiled SD detail 0.3 was slower (x2.7) and softer.
    "Full body — max detail": dict(hires_on=True, hires_scale=1.5, hires_denoise=0.45, hires_steps=8, hires_upscaler="Lanczos",
                                   fd_on=False, hd_on=False, ed_on=True, ed_denoise=0.4, md_on=True),
    # round 15 (hassakuXL, 3 seeds, 1216x832, faces 107-129 px): the Wide shot chain + Real-ESRGAN 2x before the eye pass (3648x2496): eye detail at the same display size x8.5 (Laplacian energy
    # at 144 px: 3.9 -> 33.7), face sharpness x3.0, CCIP +0.020 (z +1.0), red eyes +0.032, WD14 pin -0.05 (the clip is intact by eye), whole-picture line energy x1.7; x1.29 time (USB-C regime:
    # the 2x stage 63-94 s, the eye pass 18-31 s). The plain Wide shot's eyes are ~20 px each: the cleanup pass has nothing to draw on; by eye A is blurred, W crisp.
    "Wide shot — max detail": dict(hires_on=True, hires_scale=1.5, hires_denoise=0.45, hires_steps=14, hires_upscaler="Lanczos",
                                   fd_on=True, fd_denoise=0.35, hd_on=False, ed_on=True, ed_denoise=0.4, md_on=True),
}

# time of the whole recipe relative to the plain picture (RX 6800M, SDXL 832x1216, AYS 12 + PAG 2 ~30 s): the measured pieces added up —
# hires 1.5x ~80 s at 14 steps (round 8: 8 steps give the same picture, ~46 s), face pass ~20 s, eye pass ~13 s at 10 steps (8 steps ~10 s) (laptop heat-soaked: a cool one is faster).
# Round 8, stage seconds under cool 1.5 (Turbo, 1 plain base = 60 s): the Full body chain 222 s with 14 + 10 steps, 168 s with hires 8 (0.76x), 165 s with hires 10 + eyes 8.
# Round 9 (Manual / AC, cool 1.5), whole chain / the plain base of the same seed: Full body 8 + 8 steps 2.7x (hassakuXL's round-8 chain 2.8x), 14 + 10 steps 3.8x; Wide shot 8 + 8 3.25-3.57x, 14 hires + 8 eye steps 4.1-4.6x (mean 4.35).
TIME_FACTOR = {"Portrait": 1.3, "Cowboy shot": 1.3, "Full body": 3.0, "Wide shot": 4.5, "Full body — max detail": 4.5, "Wide shot — max detail": 5.8}

# framing tags, widest first (a prompt with several takes the widest); no tag at all → the middle recipe
_FRAMES = (
    ("Wide shot", re.compile(r"\b(very wide shot|wide shot|long shot|from far away|distant)\b", re.I)),
    ("Full body", re.compile(r"\bfull[ -]body\b", re.I)),
    ("Cowboy shot", re.compile(r"\b(cowboy shot|thigh up|waist up|three-quarter)\b", re.I)),
    ("Portrait", re.compile(r"\b(portrait|close[ -]up|face focus|upper body|bust|head ?shot|selfie)\b", re.I)),
)


def shot_type(prompt: str) -> str:
    """Which recipe fits a prompt: the framing tags it names (weights and brackets ignored), else the middle one."""
    p = re.sub(r"[()\[\]\\]", " ", prompt or "")
    for name, rx in _FRAMES:
        if rx.search(p):
            return name
    return "Cowboy shot"


def recipe(name: str) -> dict:
    """A copy of the named recipe's settings ({} for an unknown name)."""
    return {k: v for k, v in RECIPES.get(name, {}).items() if k in _KEYS}


def summary(name: str) -> str:
    r = recipe(name)
    if not r:
        return ""
    parts = []
    if r.get("hires_on"):
        parts.append(f"hires {r.get('hires_scale', 1.5):g}× ({r.get('hires_upscaler', 'Lanczos')}, denoise {r.get('hires_denoise', 0.45):g})")
    if r.get("fd_on"):
        parts.append(f"face {r.get('fd_denoise', 0.35):g}")
    if r.get("hd_on"):
        parts.append(f"hands {r.get('hd_denoise', 0.35):g}")
    if r.get("md_on"):
        parts.append("Real-ESRGAN 2× (the picture comes out twice as big)")
    if r.get("ed_on"):
        parts.append(f"eyes {r.get('ed_denoise', 0.4):g}")
    return " + ".join(parts)


def time_factor(name: str) -> float:
    return TIME_FACTOR.get(name, 1.0)
