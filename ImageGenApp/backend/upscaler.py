"""
backend/upscaler.py
Image upscaling with multiple backends:
  1. Lanczos (always available — PIL)
  2. Real-ESRGAN via ONNX Runtime (auto-downloads weights, no PyTorch needed)
  3. Real-ESRGAN via PyTorch basicsr (if installed)
"""

from __future__ import annotations

import gc
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from config import UPSCALERS_DIR, REALESRGAN_MODEL_URL, REALESRGAN_MODEL_PATH

UpscaleMethod = Literal["Lanczos", "Real-ESRGAN (ONNX)", "Real-ESRGAN (PyTorch)"]

# ── ONNX Real-ESRGAN model (4× upscale, ~67 MB) ───────────────────────────────
# Converted from the original Xinntao weights; MIT licensed.
_ONNX_URL  = (
    "https://github.com/facefusion/facefusion-assets/releases/download/"
    "models-3.0.0/real_esrgan_x4.onnx"
)
_ONNX_PATH = UPSCALERS_DIR / "real_esrgan_x4.onnx"


class Upscaler:
    """
    Wraps multiple upscaling methods and exposes a unified `upscale()` API.
    Falls back gracefully if a method is unavailable.
    """

    def __init__(self):
        self._onnx_session = None

    def release(self) -> None:
        """Drop the ONNX session: on DirectML it keeps its memory arena on the dGPU, next to the SD
        pipeline (the hires pass that follows peaks at ~10 of 12 GB; audit F-28)."""
        if self._onnx_session is not None:
            self._onnx_session = None
            import gc
            gc.collect()
        self._torch_upsampler = None

    # ── Public API ─────────────────────────────────────────────────────────────
    def upscale(
        self,
        image: Image.Image,
        scale: int = 4,
        method: UpscaleMethod = "Real-ESRGAN (ONNX)",
        progress_callback=None,
    ) -> tuple[Image.Image, str]:
        """
        Upscale *image* by *scale*× using the selected *method*.
        Returns (upscaled_image, info_string).
        """
        if method == "Lanczos":
            return self._lanczos(image, scale)
        elif method == "Real-ESRGAN (ONNX)":
            return self._realesrgan_onnx(image, scale, progress_callback)
        elif method == "Real-ESRGAN (PyTorch)":
            return self._realesrgan_torch(image, scale)
        else:
            return self._lanczos(image, scale)

    def available_methods(self) -> list[str]:
        methods = ["Lanczos"]
        try:
            import onnxruntime  # noqa
            methods.append("Real-ESRGAN (ONNX)")
        except ImportError:
            pass
        try:
            import basicsr  # noqa
            methods.append("Real-ESRGAN (PyTorch)")
        except ImportError:
            pass
        return methods

    # ── Lanczos (PIL) ──────────────────────────────────────────────────────────
    def _lanczos(self, image: Image.Image, scale: int) -> tuple[Image.Image, str]:
        w, h = image.size
        out = image.resize((w * scale, h * scale), Image.LANCZOS)
        return out, f"Lanczos {scale}× — {w}×{h} → {w*scale}×{h*scale}"

    # ── Real-ESRGAN via ONNX Runtime ───────────────────────────────────────────
    def _realesrgan_onnx(
        self,
        image: Image.Image,
        scale: int,
        progress_callback=None,
    ) -> tuple[Image.Image, str]:
        import onnxruntime as ort

        # Download model if missing
        if not _ONNX_PATH.exists():
            if progress_callback:
                progress_callback(0, "Downloading Real-ESRGAN ONNX model (~67 MB)…")
            _download_file(_ONNX_URL, _ONNX_PATH)

        # Build session (lazy). On the GPU use an fp16 copy of the model: ~1.5× faster
        # (832×1216 2×: 54 s → 35 s) and visually identical (PSNR ≈ 57 dB vs fp32).
        if self._onnx_session is None:
            providers = _ort_providers()
            model = _ONNX_PATH
            if providers and providers[0][0] == "DmlExecutionProvider":
                model = _fp16_model(progress_callback) or _ONNX_PATH
            self._onnx_session = ort.InferenceSession(str(model), providers=providers)
            self._onnx_fp16 = model != _ONNX_PATH

        # Pre-process: RGB float32
        img_rgb = image.convert("RGB")
        img_np  = np.array(img_rgb, dtype=np.float32) / 255.0  # H,W,3

        if progress_callback:
            progress_callback(0.1, "Running Real-ESRGAN inference…")

        out_np = _run_tiled(self._onnx_session, img_np, progress_callback)
        fallback_note = ""
        if out_np.max() < 1e-3 and img_np.max() > 0.02:
            # GPU returned nothing (driver reset / unsupported op) — redo on CPU
            print("[Upscale] GPU (DirectML) returned a blank image — retrying on CPU")
            self._onnx_session = ort.InferenceSession(
                str(_ONNX_PATH), providers=["CPUExecutionProvider"])
            self._onnx_fp16 = False
            if progress_callback:
                progress_callback(0.1, "GPU returned a blank image — retrying on CPU…")
            out_np = _run_tiled(self._onnx_session, img_np, progress_callback)
            fallback_note = " — ⚠ GPU failed, ran on CPU"

        # Post-process
        out_np  = np.clip(out_np * 255, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(out_np)

        if progress_callback:
            progress_callback(1.0, "Upscale complete.")

        # If model scale (4×) != requested scale, resize to exact target
        target_w = image.width  * scale
        target_h = image.height * scale
        if out_img.size != (target_w, target_h):
            out_img = out_img.resize((target_w, target_h), Image.LANCZOS)

        w, h = image.size
        info = (
            f"Real-ESRGAN (ONNX) {scale}× — "
            f"{w}×{h} → {out_img.width}×{out_img.height} "
            f"on {('GPU (DirectML, fp16)' if getattr(self, '_onnx_fp16', False) else 'GPU (DirectML)') if self._onnx_session.get_providers()[0] == 'DmlExecutionProvider' else 'CPU'}"
            f"{fallback_note}"
        )
        return out_img, info

    # ── Real-ESRGAN via basicsr / realesrgan package ───────────────────────────
    def _realesrgan_torch(
        self, image: Image.Image, scale: int
    ) -> tuple[Image.Image, str]:
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import torch

            if not REALESRGAN_MODEL_PATH.exists():
                _download_file(REALESRGAN_MODEL_URL, REALESRGAN_MODEL_PATH)

            if self._torch_upsampler is None:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3,
                    num_feat=64, num_block=23, num_grow_ch=32, scale=4,
                )
                from config import DEVICE
                device = torch.device("cpu" if ("privateuseone" in DEVICE or DEVICE == "directml") else DEVICE)
                self._torch_upsampler = RealESRGANer(
                    scale=4, model_path=str(REALESRGAN_MODEL_PATH),
                    model=model, half=("cuda" in DEVICE and "privateuseone" not in DEVICE), device=device,
                )

            img_np = np.array(image.convert("RGB"))
            out_np, _ = self._torch_upsampler.enhance(img_np, outscale=scale)
            out_img = Image.fromarray(out_np)

            w, h = image.size
            return out_img, (
                f"Real-ESRGAN (PyTorch) {scale}× — "
                f"{w}×{h} → {out_img.width}×{out_img.height}"
            )
        except ImportError:
            return self._realesrgan_onnx(image, scale)


# ── Utility ────────────────────────────────────────────────────────────────────
def _dml_dgpu_id() -> int:
    """DML index of the discrete GPU. DML order differs per machine: 0 is the
    780M iGPU on the 8700G workstation but the RX 6800M on the laptop."""
    try:
        import torch_directml
        from backend.hardware_detector import _classify_gpu
        for i in range(torch_directml.device_count()):
            if not _classify_gpu(torch_directml.device_name(i))[1]:
                return i
    except Exception:
        pass
    return 0


def _ort_providers() -> list:
    """Pick the best ONNX Runtime execution provider available, CPU as fallback."""
    import onnxruntime as ort
    available = ort.get_available_providers()
    if "DmlExecutionProvider" in available:
        return [("DmlExecutionProvider", {"device_id": _dml_dgpu_id()}), "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _vram_mb() -> int:
    """Return best-GPU VRAM in MB, 0 if unknown."""
    try:
        from backend.hardware_detector import get_profile
        p = get_profile()
        if p.best_gpu:
            return p.best_gpu.vram_mb
    except Exception:
        pass
    return 0


_FP16_PATH = UPSCALERS_DIR / "real_esrgan_x4_fp16.onnx"


def _fp16_model(progress_callback=None):
    """Path to an fp16 copy of the Real-ESRGAN model, converting it once (a few seconds)
    if needed. None when the converter isn't installed or conversion fails — the caller
    then uses the fp32 model."""
    try:
        if _FP16_PATH.exists() and _FP16_PATH.stat().st_mtime >= _ONNX_PATH.stat().st_mtime:
            return _FP16_PATH
        import onnx
        from onnxconverter_common import float16
        if progress_callback:
            progress_callback(0.02, "Preparing a faster fp16 copy of the upscaler (one time)…")
        m = onnx.load(str(_ONNX_PATH))
        # Resize takes its scales from Identity nodes; both must stay fp32 or loading fails.
        # (Hundreds of "number will be truncated to 1e-07" warnings are expected — hidden.)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            m16 = float16.convert_float_to_float16(m, keep_io_types=True,
                                                   op_block_list=["Resize", "Identity"])
        tmp = _FP16_PATH.with_suffix(".tmp")
        onnx.save(m16, str(tmp))
        tmp.replace(_FP16_PATH)
        print(f"[Upscale] Created fp16 model {_FP16_PATH.name}")
        return _FP16_PATH
    except ImportError:
        return None
    except Exception as e:
        print(f"[Upscale] fp16 conversion failed ({e}) — using the fp32 model")
        return None


_DML_MAX_TILE = 256
_CPU_MAX_TILE = 384


def _tile_size() -> int:
    """
    Choose a safe ONNX tile size based on available VRAM.
    Larger tiles = fewer seams + faster; smaller = less VRAM.
    """
    vram = _vram_mb()
    if vram >= 8192:
        return 768
    if vram >= 4096:
        return 512
    if vram >= 2048:
        return 384
    return 256  # safe for ≤2 GB


def _run_tiled(session, img_np: np.ndarray, progress_callback=None) -> np.ndarray:
    """
    Run the ONNX Real-ESRGAN session on *img_np* (H,W,3 float32 [0,1]) using
    overlapping tiled inference to avoid seam artifacts.
    Returns H'×W'×3 float32 [0,1].  Model output scale is assumed to be 4×.

    Each tile overlaps its neighbour by OVERLAP pixels.  The overlap regions are
    blended using a 2-D tent (triangular) weight function so tile boundaries are
    invisible in the output.
    """
    MODEL_SCALE = 4
    # Tile size is bounded by time/RAM, not VRAM: the model is 67 MB but its last
    # layers hold 64-channel maps at 4× the tile size (768 px tile → ~2.4 GB each).
    #  • DirectML: Windows resets the GPU (TDR, DXGI_ERROR_DEVICE_HUNG) if one
    #    dispatch runs > ~2 s — a 512 px tile does on an RX 6800M → all-black output.
    #  • CPU: 768 px tiles exhaust system RAM for no speed gain.
    dml = session.get_providers()[0] == "DmlExecutionProvider"
    tile = min(_tile_size(), _DML_MAX_TILE if dml else _CPU_MAX_TILE)
    overlap = min(32, tile // 8)   # pixels of overlap on each side
    step    = tile - overlap        # stride between tile starts

    h, w = img_np.shape[:2]
    out_h, out_w = h * MODEL_SCALE, w * MODEL_SCALE
    out_np = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

    # Build list of (x1,y1,x2,y2) tiles using step-based stride
    coords = [
        (x, y, min(x + tile, w), min(y + tile, h))
        for y in range(0, h, step)
        for x in range(0, w, step)
    ]
    n = len(coords)

    def _tent_weights(th: int, tw: int) -> np.ndarray:
        """2-D tent function (1 at borders, higher toward centre). Shape: (th,tw,1)."""
        wy = np.minimum(np.arange(1, th + 1), np.arange(th, 0, -1), dtype=np.float32)
        wx = np.minimum(np.arange(1, tw + 1), np.arange(tw, 0, -1), dtype=np.float32)
        return np.minimum(wy[:, np.newaxis], wx[np.newaxis, :])[:, :, np.newaxis]

    for idx, (x1, y1, x2, y2) in enumerate(coords):
        tile_np = img_np[y1:y2, x1:x2]                      # H,W,3
        tile_t  = tile_np.transpose(2, 0, 1)[np.newaxis]     # 1,3,H,W
        out_t   = session.run(None, {"input": tile_t})[0]    # 1,3,H',W'
        out_tile = out_t[0].transpose(1, 2, 0)               # H',W',3

        oy1, oy2 = y1 * MODEL_SCALE, y2 * MODEL_SCALE
        ox1, ox2 = x1 * MODEL_SCALE, x2 * MODEL_SCALE
        oh,  ow  = oy2 - oy1, ox2 - ox1

        w_tile = _tent_weights(oh, ow)
        out_np[oy1:oy2, ox1:ox2] += out_tile * w_tile
        weight[oy1:oy2, ox1:ox2] += w_tile

        if progress_callback:
            progress_callback(
                0.1 + 0.80 * (idx + 1) / n,
                f"Upscaling tile {idx + 1}/{n}…",
            )

    # Normalise by accumulated weights (handles border tiles correctly)
    out_np /= np.maximum(weight, 1e-6)
    return out_np


def _download_file(url: str, dest: Path, chunk: int = 1 << 20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} → {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
        while True:
            data = resp.read(chunk)
            if not data:
                break
            fh.write(data)
    print("Download complete.")
