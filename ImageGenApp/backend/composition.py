"""
composition.py — 🧍 Composition control (round 37): a depth ControlNet that copies a picture's figure, framing and hand shapes while the prompt dresses it again.

Round 36 measured it (AGENTS.md): on four of her pictures a depth map matched the source's composition in 14-15 of 16 pictures (strength 0.5 / 0.8) against 11 of 16 for an OpenPose skeleton
(a double V is a double V), kept her face (CCIP 0.93, the skeleton 0.90) and broke fewer hands (32 % against 60 %, no hand pass) — but the source's clothes and surroundings leak through the
silhouette in about 1 picture in 4 (the grey jacket of a sitting figure, a skirt's flare, a cape), so the default strength is 0.5 and the range 0.4-0.8. Same cost as the pose control
(≈ 1.8 s per step, 12 GB card full), PAG is off while it runs, the 2.5 GB model is released after every picture.

* Estimator: Depth-Anything-V2-Small (Apache-2.0, 99 MB ONNX, models/depth/), CPU, ~2 s per picture; downloaded on request (SHA-256 pinned), nothing downloads by itself.
* The current depth map lives in settings/composition/_current.png (+ .json): it is NOT part of a saved pose, the starters stay skeleton-only, and the record keeps only {model, strength, source}
  — restoring a picture means uploading the source picture again.
* ControlNet models: folders of models/controlnet/ whose name says depth (or a registered depth model); the public default is the Apache-2.0 xinsir depth model listed in pose_control.MODELS.

Pure module (no Gradio, no torch at import); backend/sampling.py runs the ControlNet pipeline exactly as for the pose control.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from backend import pose_control as pc
from config import MODELS_DIR, SETTINGS_DIR

DEPTH_DIR = MODELS_DIR / "depth"
COMP_DIR = SETTINGS_DIR / "composition"
CURRENT = "_current"
STRENGTH_MIN, STRENGTH_MAX, STRENGTH_DEFAULT = 0.4, 0.8, 0.5
# Round 38: the depth map is flattened to this many layers before it is used. Fine garment relief (collars, cuffs, the folds of a jacket, shoe shapes) is what the source's clothes leak through;
# flat layers keep the figure, the framing and (mostly) the hands. Measured on 3 sources x 4 outfits (her card, strength 0.5): the leak 0.83 -> 0.12 (0 = none, 1 = yes), composition about the same.
# 0 / 1 = off (the round-37 behaviour). control_guidance_end < 1 (0.4 / 0.6 / 0.8) did nothing: the garments are set in the first steps.
FLATTEN_LEVELS = 8
DEFAULT_MODEL = "xinsir_depth_sdxl_1.0"
ESTIMATOR_FILE = "depth_anything_v2_small.onnx"
ESTIMATORS = {
    ESTIMATOR_FILE: dict(repo="onnx-community/depth-anything-v2-small", file="onnx/model.onnx", mb=99, license="Apache-2.0",
                         sha256="afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"),
}
_SESS: dict = {}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def clamp_strength(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return STRENGTH_DEFAULT
    if x != x:
        return STRENGTH_DEFAULT
    return min(STRENGTH_MAX, max(STRENGTH_MIN, x))


# ── files ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def estimator_path() -> Path:
    return DEPTH_DIR / ESTIMATOR_FILE


def estimator_available() -> bool:
    return estimator_path().is_file()


def download_estimator(say=lambda t: None) -> str:
    if estimator_available():
        return "The depth estimator is already there."
    d = ESTIMATORS[ESTIMATOR_FILE]
    say(f"Downloading {ESTIMATOR_FILE} ({d['mb']} MB)…")
    pc._fetch(d["repo"], d["file"], estimator_path(), d["sha256"])
    _SESS.clear()
    return f"Depth estimator ready ({d['license']})."


def missing_downloads() -> list[str]:
    """What the composition download buttons would fetch, as short labels."""
    out = [] if estimator_available() else [f"depth estimator {ESTIMATORS[ESTIMATOR_FILE]['mb']} MB"]
    if pc.model_path(DEFAULT_MODEL) is None and DEFAULT_MODEL in pc.MODELS:
        out.append(f"ControlNet {pc.MODELS[DEFAULT_MODEL]['mb'] / 1000:.1f} GB")
    return out


# ── depth estimation (CPU) ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _session():
    path = estimator_path()
    key = (str(path), path.stat().st_mtime_ns if path.exists() else 0)
    if key not in _SESS:
        if pc._sha256(path) != ESTIMATORS[ESTIMATOR_FILE]["sha256"]:
            raise RuntimeError(f"{path.name} does not match its expected SHA-256 — delete it and use the download button.")
        _SESS.clear()
        _SESS[key] = pc._make_session(path)
    return _SESS[key]


def depth_map(img: Image.Image, size: int = 518) -> Image.Image:
    """The depth map of a picture as an 'L' image the size of the picture, near = white (the convention of the depth ControlNets).
    Depth-Anything predicts relative inverse depth (bigger = nearer): min-max scaled it already is that. Raises FileNotFoundError without the estimator file."""
    if not estimator_available():
        raise FileNotFoundError("The depth estimator file is missing: " + ESTIMATOR_FILE)
    S = _session()
    rgb = img.convert("RGB")
    w, h = rgb.size
    s = size / max(1, min(w, h))
    tw, th = max(14, int(round(w * s / 14)) * 14), max(14, int(round(h * s / 14)) * 14)       # the model wants multiples of 14
    x = (np.asarray(rgb.resize((tw, th), Image.BICUBIC), np.float32) / 255.0 - _MEAN) / _STD
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
    d = np.squeeze(np.asarray(S.run(None, {S.get_inputs()[0].name: x})[0], dtype=np.float32))
    if d.ndim != 2 or not np.isfinite(d).all():
        raise RuntimeError("The depth estimator returned an unusable map.")
    lo, hi = float(d.min()), float(d.max())
    d = (d - lo) / max(1e-6, hi - lo)
    return Image.fromarray((d * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC)


def flatten(depth: Image.Image, levels: int | None = None) -> Image.Image:
    """The depth map as `levels` flat layers (an 'L' image; the same size). levels < 2 returns it unchanged."""
    n = FLATTEN_LEVELS if levels is None else int(levels)
    d = depth.convert("L")
    if n < 2:
        return d
    a = np.asarray(d, np.float32) / 255.0
    return Image.fromarray((np.round(a * (n - 1)) / (n - 1) * 255).astype(np.uint8), "L")


def depth_hash(depth: Image.Image) -> str:
    """10 hex digits that identify a depth map (the record's `source`)."""
    return hashlib.sha256(depth.convert("L").tobytes()).hexdigest()[:10]


# ── the current depth map ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
def save_current(depth: Image.Image) -> str:
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    d = depth.convert("L")
    d.save(COMP_DIR / f"{CURRENT}.png")
    meta = {"v": 1, "size": [d.width, d.height], "hash": depth_hash(d)}
    tmp = COMP_DIR / f"{CURRENT}.json.part"
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    os.replace(tmp, COMP_DIR / f"{CURRENT}.json")
    return meta["hash"]


def load_current() -> Image.Image | None:
    try:
        return Image.open(COMP_DIR / f"{CURRENT}.png").convert("L")
    except OSError:
        return None


def has_current() -> bool:
    return (COMP_DIR / f"{CURRENT}.png").is_file()


def render(depth: Image.Image, W: int, H: int) -> Image.Image:
    """The control picture (RGB, W×H, black background): the same aspect ratio (±3 %) fills the canvas, another one is fitted and centred, never stretched."""
    W, H = max(64, int(W)), max(64, int(H))
    sx, sy, ox, oy = pc._layout(float(depth.width), float(depth.height), W, H)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    fit = depth.convert("RGB").resize((max(1, round(depth.width * sx)), max(1, round(depth.height * sy))), Image.LANCZOS)
    canvas.paste(fit, (round(ox), round(oy)))
    return canvas


# ── what a Generate run needs ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def build_control(ex: dict, width: int, height: int) -> tuple[dict | None, dict | None, str]:
    """(control dict for sd.control, the record's `composition_control` entry, error text) for an `ex` with pc_on and the Composition type."""
    depth = load_current()
    if depth is None:
        return None, None, "Composition control is on but no picture is loaded — upload a picture in the 🧍 box (its depth map is made on the CPU)."
    name = (ex.get("pc_model") or "").strip() or DEFAULT_MODEL
    if pc.kind_of(name) == "pose":
        return None, None, f"Composition control: '{name}' is an OpenPose model — pick a depth ControlNet (🧍 box → ControlNet model)."
    mdl = pc.model_path(name)
    if mdl is None:
        local = [m for m in pc.list_models("depth")]
        return None, None, (f"Composition control: no depth ControlNet '{name}' in models/controlnet/. "
                            + ("Pick one of: " + ", ".join(local) + "." if local else "Use the ⬇ Download button in the 🧍 box (xinsir depth, Apache-2.0)."))
    strength = clamp_strength(ex.get("pc_strength"))
    rec = {"model": mdl.name, "strength": strength, "source": depth_hash(depth)}
    if FLATTEN_LEVELS >= 2:
        rec["levels"] = FLATTEN_LEVELS                                    # what ran (round 38); a round-37 record has none = the raw depth map
    # flattened AFTER the fit: resizing a flat map would blend the layers again (the lab flattened at the generated size)
    ctl = {"image": flatten(render(depth, width, height).convert("L")).convert("RGB"), "scale": strength, "model": str(mdl), "name": "composition"}
    return ctl, rec, ""
