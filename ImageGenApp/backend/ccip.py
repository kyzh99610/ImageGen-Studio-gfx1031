"""
ccip.py — CCIP (deepghs "Character Contrastive Image Pretraining"): anime character re-identification, ONNX on the CPU.

One 768-d feature per character picture; two pictures show the same character when their cosine is high (the model's own "difference" is exactly
(1 − cos) / 2, "same character" below 0.1785). Unlike WD14's tagger feature it ignores outfit, pose and scene: her pictures in night / neon / sunset / lying
poses score 0.040–0.045 difference to a handful of references, a girl with other hair and eyes 0.39 (round-5 calibration, AGENTS.md).

deepghs/ccip_onnx, ccip-caformer-24-randaug-pruned/model_feat.onnx — 150 MB, OpenRAIL, downloaded once on request (🧬 Learn her look) and checked against the
pinned SHA-256. Nothing here downloads by itself.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO, FILE = "deepghs/ccip_onnx", "ccip-caformer-24-randaug-pruned/model_feat.onnx"
SHA256 = "4ea118d16496274f4f6e08d3afc768cc592389e8f7f32f8732ce2215c228ac5f"
SIZE_MB = 150
MODEL_TAG = "ccip-caformer-24-randaug-pruned"
INPUT = 384
_session = None


def cached_path() -> str | None:
    """The model file when it is already in the Hugging Face cache (never downloads)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(REPO, FILE)
        return p if isinstance(p, str) and Path(p).is_file() else None
    except Exception:
        return None


def available() -> bool:
    return cached_path() is not None


def download() -> str:
    """Fetch the model once; a file that isn't the pinned one is deleted and refused."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, FILE)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != SHA256:
        try:
            os.remove(p)
        except OSError:
            pass
        raise RuntimeError("The downloaded CCIP model does not match the expected SHA-256 — it was deleted.")
    return p


def preprocess(img: Image.Image) -> np.ndarray:
    """(1, 3, 384, 384) float32: the picture squashed to 384×384 (bilinear), scaled to [-1, 1]."""
    a = np.asarray(img.convert("RGB").resize((INPUT, INPUT), Image.BILINEAR), np.float32) / 255.0
    return ((a - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)


def _load():
    global _session
    if _session is None:
        path = cached_path()
        if path is None:
            return None
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = int(os.environ.get("IMAGEGEN_CCIP_THREADS", 0)) or min(4, os.cpu_count() or 2)
        _session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    return _session


def features(img: Image.Image) -> np.ndarray | None:
    """The 768-d feature of a character picture (a head-and-shoulders crop works), or None when the model isn't downloaded."""
    sess = _load()
    if sess is None:
        return None
    return np.asarray(sess.run(None, {sess.get_inputs()[0].name: preprocess(img)})[0][0], np.float32)
