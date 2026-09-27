"""
vram_estimator.py — Heuristic VRAM budget estimator for ImageGen Studio.

All numbers are conservative estimates in megabytes (MB).
Actual usage varies by driver, OS overhead, and model internals,
but these figures are calibrated to give a realistic "high-water mark"
for AMD RX 6800M / RX 6800 XT class hardware.
"""

from __future__ import annotations
import re


# ── Model weight size estimates (fp16, loaded into VRAM) ──────────────────────
# UNet + Text Encoder(s) + VAE — not double-counted with the VAE selection.
_SD15_WEIGHTS_MB   = 2_000   # SD 1.x  ~1.7 GB → round up with OS overhead
_SD20_WEIGHTS_MB   = 2_600   # SD 2.x / 2.1
_SDXL_WEIGHTS_MB   = 6_800   # SDXL base
_UNKNOWN_WEIGHTS_MB = 2_200  # conservative default for unknown checkpoints

# Additional optional VAE override (external file, replaces built-in VAE)
_EXTERNAL_VAE_MB = 400       # float16 VAE-only file

# LoRA weights are small but live in VRAM while loaded
_LORA_MB = 120

# Activation / KV-cache memory during a single denoising step.
# Attention maps scale quadratically with H×W; these are per-batch multipliers.
#   SD 1.x native resolution: 512×512 = 262144 px²
#   SDXL native resolution: 1024×1024 = 1048576 px²
_SD15_ACT_BASE_MB   = 1_300   # activations at 512×512, batch=1
_SD15_ACT_REF_PX2   = 512 * 512
_SDXL_ACT_BASE_MB   = 3_200   # activations at 1024×1024, batch=1
_SDXL_ACT_REF_PX2   = 1024 * 1024

# OS / CUDA context + fragmentation headroom
_OVERHEAD_MB = 300

# ── Upscaler estimates ─────────────────────────────────────────────────────────
_REALESRGAN_MODEL_MB  = 120    # ONNX model weights resident in VRAM
_REALESRGAN_TILE_MB   = 600    # activations for a single 512px tile (float32)
_REALESRGAN_768_MB    = 1_400  # activations for a 768px tile
_ESRGAN_PYTORCH_MB    = 380    # PyTorch ESRGAN model

# ── Inpainting estimates ───────────────────────────────────────────────────────
_LAMA_MB        = 500   # LaMa ONNX model + activations at 512×512
_OPENCV_MB      = 0     # CPU-only, no GPU VRAM


def _detect_model_class(model_name: str) -> str:
    """Classify a model string as 'sdxl', 'sd20', 'sd15', or 'unknown'."""
    if not model_name:
        return "unknown"
    # The UI passes the checkpoint's path: its header says what it really is (Pony /
    # Illustrious / NoobAI files are SDXL without "xl" in the name).
    try:
        from backend.model_manager import checkpoint_arch
        arch = checkpoint_arch(model_name)
        if arch:
            return {"sdxl": "sdxl", "sd2": "sd20", "sd1": "sd15"}[arch]
    except Exception:
        pass
    from pathlib import Path
    n = Path(model_name).name.lower() if ("\\" in model_name or model_name.count("/") > 1) else model_name.lower()
    # "xl" as its own token ("juggernautXL_v9", "sdxl", "xl-base") — not inside "pixelart"
    if (re.search(r"sdxl|xl(?![a-z])", n) or
            any(k in n for k in ("pony", "illustrious", "noobai"))):
        return "sdxl"
    if any(k in n for k in ("2.0", "2.1", "sd2", "v2-", "stabilityai/stable-diffusion-2")):
        return "sd20"
    if any(k in n for k in ("1.5", "1.4", "sd1", "v1-", "runwayml", "dreamshaper", "deliberate",
                             "realistic", "anything", "chillout", "majicmix", "rev", "epicrealism")):
        return "sd15"
    return "unknown"


def _weights_mb(model_class: str) -> int:
    return {
        "sdxl": _SDXL_WEIGHTS_MB,
        "sd20": _SD20_WEIGHTS_MB,
        "sd15": _SD15_WEIGHTS_MB,
    }.get(model_class, _UNKNOWN_WEIGHTS_MB)


def _act_mb(model_class: str, width: int, height: int, batch: int) -> int:
    """Estimate activation/KV-cache VRAM for a single step, scaled by resolution."""
    px2 = width * height
    if model_class == "sdxl":
        base = _SDXL_ACT_BASE_MB
        ref  = _SDXL_ACT_REF_PX2
    else:
        base = _SD15_ACT_BASE_MB
        ref  = _SD15_ACT_REF_PX2
    # Attention maps scale roughly quadratically with resolution but UNet
    # feature maps scale linearly; empirically ~linear for practical ranges.
    scale = px2 / ref
    return int(base * scale * batch)


def estimate_generate(
    model_name: str,
    width: int,
    height: int,
    batch: int,
    use_lora: bool = False,
    use_external_vae: bool = False,
    use_img2img: bool = False,
) -> int:
    """
    Estimate peak VRAM in MB for a txt2img / img2img generation pass.
    """
    cls      = _detect_model_class(model_name)
    weights  = _weights_mb(cls)
    act      = _act_mb(cls, width, height, batch)
    lora     = _LORA_MB if use_lora else 0
    vae_extra = _EXTERNAL_VAE_MB if use_external_vae else 0
    # img2img adds the encoded input image tensor
    i2i_extra = int(width * height * 3 * 4 / (1024 * 1024)) if use_img2img else 0
    return weights + act + lora + vae_extra + i2i_extra + _OVERHEAD_MB


def estimate_upscale(
    input_width: int,
    input_height: int,
    scale: int,
    method: str,
) -> int:
    """
    Estimate peak VRAM in MB for an upscaling operation.
    """
    method_l = method.lower()
    if "lanczos" in method_l or method_l == "":
        return 0  # pure CPU

    if "pytorch" in method_l or "esrgan" in method_l:
        # PyTorch ESRGAN: model + input tile + output tile
        tile_px2  = input_width * input_height  # full image at once (no tiling for PyTorch path)
        img_mb    = int(tile_px2 * 3 * 2 / (1024 * 1024))  # fp16
        return _ESRGAN_PYTORCH_MB + img_mb + _OVERHEAD_MB

    # Real-ESRGAN ONNX (tiled) — tile size chosen by VRAM (but we're estimating here)
    # The tiler uses at most 2 tile-sized buffers at once.
    # Assume 512px tiles (conservative, may use 768px on ≥8 GB)
    tile_mb = _REALESRGAN_TILE_MB  # 512px tile
    # Output image buffer: output dimensions in fp32
    out_w, out_h = input_width * scale, input_height * scale
    output_buf_mb = int(out_w * out_h * 3 * 4 / (1024 * 1024))

    return _REALESRGAN_MODEL_MB + tile_mb * 2 + output_buf_mb + _OVERHEAD_MB


def estimate_watermark(
    method: str,
    image_width: int = 1024,
    image_height: int = 1024,
    model_name: str = "",
) -> int:
    """
    Estimate peak VRAM in MB for watermark removal.
    """
    method_l = method.lower()
    if "opencv" in method_l:
        return _OPENCV_MB
    if "lama" in method_l:
        return _LAMA_MB + _OVERHEAD_MB
    if "sd" in method_l or "inpaint" in method_l:
        # SD inpainting path: uses the currently loaded SD model
        return estimate_generate(model_name, image_width, image_height, 1)
    return 0


def get_total_vram_mb() -> int:
    """Return total available VRAM for the selected GPU in MB (0 = unknown)."""
    try:
        from backend.hardware_detector import get_profile
        p = get_profile()
        if p.gpus:
            gpu = p.best_gpu or p.gpus[0]
            return gpu.vram_mb or 0
    except Exception:
        pass
    return 0


def vram_bar_html(
    used_mb: int,
    total_mb: int,
    label: str = "Est. VRAM",
) -> str:
    """
    Render a game-style VRAM budget bar as an HTML string.

    Colour thresholds (fraction of total):
      < 0.65  → green  (#a6e3a1)
      < 0.85  → yellow (#f9e2af)
      ≥ 0.85  → red    (#f38ba8)
    """
    if total_mb <= 0:
        return (
            '<div style="font-size:13px;color:#9399b2;padding:4px 0;">'
            f'<b>{label}:</b> {used_mb / 1024:.1f} GB estimated '
            f'(total VRAM unknown)</div>'
        )

    pct    = min(used_mb / total_mb, 1.05)   # allow slight over to show red
    pct_display = min(pct, 1.0)
    bar_w  = f"{pct_display * 100:.1f}%"
    used_g = used_mb  / 1024
    total_g = total_mb / 1024

    if pct < 0.65:
        bar_color = "#a6e3a1"   # green
        text_color = "#a6e3a1"
        icon = "🟢"
    elif pct < 0.85:
        bar_color = "#f9e2af"   # yellow
        text_color = "#f9e2af"
        icon = "🟡"
    else:
        bar_color = "#f38ba8"   # red
        text_color = "#f38ba8"
        icon = "🔴"

    # Overflow warning
    overflow = ""
    if pct > 1.0:
        overflow = (
            ' <span style="color:#f38ba8;font-weight:bold;">'
            '⚠ May exceed VRAM — try smaller resolution or batch size</span>'
        )

    return (
        f'<div style="margin:4px 0 8px 0;">'
        # Label row
        f'<div style="display:flex;justify-content:space-between;font-size:13px;'
        f'margin-bottom:3px;">'
        f'<span style="color:#cdd6f4;">{icon} <b>{label}</b></span>'
        f'<span style="color:{text_color};font-weight:bold;">'
        f'{used_g:.1f} / {total_g:.1f} GB &nbsp;({pct_display*100:.0f}%)'
        f'</span>'
        f'</div>'
        # Track
        f'<div style="background:#313244;border-radius:4px;height:10px;overflow:hidden;">'
        f'<div style="width:{bar_w};height:100%;background:{bar_color};'
        f'border-radius:4px;transition:width 0.3s ease,background 0.3s ease;">'
        f'</div>'
        f'</div>'
        f'{overflow}'
        f'</div>'
    )
