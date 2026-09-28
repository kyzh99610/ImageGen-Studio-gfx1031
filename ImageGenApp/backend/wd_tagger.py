"""
WD14 anime tagger (SmilingWolf/wd-vit-tagger-v3, Apache-2.0, ONNX, ~380 MB, downloaded once):
image → Danbooru tags with confidences. Used for "🏷 Interrogate" (img2img / PNG Info) and for
LoRA dataset captions. Runs with ONNX Runtime on the CPU (~0.8 s per image warm) so it never
competes with the diffusion model for VRAM.
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path

import numpy as np
from PIL import Image

REPO = "SmilingWolf/wd-vit-tagger-v3"
# Danbooru "emoticon" tags whose underscore is part of the tag
_KAOMOJI = {"0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9",
            ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"}

_lock = threading.Lock()
_session = None
_labels: list[tuple[str, int]] = []      # (name, category): 9 rating, 0 general, 4 character


def _say(progress, frac, desc):
    """Report progress; a progress callback that fails must not stop the tagging."""
    if progress is not None:
        try:
            progress(frac, desc=desc)
        except Exception:
            pass


def _load(progress=None):
    global _session, _labels
    with _lock:
        if _session is not None:
            return _session
        from huggingface_hub import hf_hub_download
        _say(progress, 0.05, "Loading the WD14 tagger (~380 MB download the first time)…")
        model = hf_hub_download(REPO, "model.onnx")
        csv_path = hf_hub_download(REPO, "selected_tags.csv")
        with open(csv_path, newline="", encoding="utf-8") as f:
            _labels = [(r["name"], int(r["category"])) for r in csv.DictReader(f)]
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # CPU: the dGPU is busy with the diffusion model, and this ViT takes ~1 s on the CPU anyway
        _session = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])
        return _session


def unload():
    global _session
    with _lock:
        _session = None


def _prepare(img: Image.Image, size: int) -> np.ndarray:
    """Model input: RGB on white, padded to a square, resized, BGR float32 0–255, NHWC."""
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    img = bg.convert("RGB")
    side = max(img.size)
    sq = Image.new("RGB", (side, side), (255, 255, 255))
    sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    if side != size:
        sq = sq.resize((size, size), Image.BICUBIC)
    arr = np.asarray(sq, dtype=np.float32)[:, :, ::-1]
    return np.ascontiguousarray(arr[None])


def _pretty(name: str) -> str:
    """Danbooru name → prompt tag: underscores → spaces (not in emoticons), brackets escaped."""
    t = name if name in _KAOMOJI else name.replace("_", " ")
    return t.replace("(", "\\(").replace(")", "\\)")


def tag_image(img: Image.Image, general_threshold: float = 0.35, character_threshold: float = 0.85,
              progress=None) -> dict:
    """{'rating': {name: p}, 'character': [(tag, p)], 'general': [(tag, p)]}, best first."""
    if img is None:
        raise ValueError("No image.")
    sess = _load(progress)
    inp = sess.get_inputs()[0]
    size = int(inp.shape[1]) if isinstance(inp.shape[1], int) else 448
    probs = sess.run(None, {inp.name: _prepare(img, size)})[0][0]
    rating, character, general = {}, [], []
    for (name, cat), p in zip(_labels, probs):
        p = float(p)
        if cat == 9:
            rating[name] = p
        elif cat == 4 and p >= character_threshold:
            character.append((_pretty(name), p))
        elif cat == 0 and p >= general_threshold:
            general.append((_pretty(name), p))
    character.sort(key=lambda x: -x[1])
    general.sort(key=lambda x: -x[1])
    return {"rating": rating, "character": character, "general": general}


def tags_text(result: dict, exclude: str = "", with_character: bool = True) -> str:
    """Comma-separated prompt: character tags first, then general tags by confidence."""
    skip = {t.strip().lower().replace("_", " ") for t in (exclude or "").split(",") if t.strip()}
    tags = ([t for t, _ in result.get("character", [])] if with_character else []) + \
           [t for t, _ in result.get("general", [])]
    out = []
    for t in tags:
        k = t.lower().replace("\\(", "(").replace("\\)", ")")
        if k not in skip and t not in out:
            out.append(t)
    return ", ".join(out)


def caption_folder(folder: str | Path, prefix: str = "", general_threshold: float = 0.35,
                   overwrite: bool = False, progress=None) -> tuple[int, int]:
    """Write a WD14 .txt caption next to every image in `folder` (prefix = trigger word first).
    Returns (written, skipped)."""
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    written = skipped = 0
    for i, p in enumerate(files):
        txt = p.with_suffix(".txt")
        if txt.exists() and not overwrite:
            skipped += 1
            continue
        _say(progress, (i + 1) / max(len(files), 1), f"Tagging {p.name}")
        with Image.open(p) as im:
            res = tag_image(im, general_threshold)
        tags = tags_text(res)
        if prefix.strip():
            tags = prefix.strip().rstrip(",") + (", " + tags if tags else "")
        txt.write_text(tags, encoding="utf-8")
        written += 1
    return written, skipped
