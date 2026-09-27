"""
config.py — Shared paths, device detection, and constants for ImageGen Studio.
Device selection is driven by hardware_detector which ranks all GPUs by VRAM,
RDNA generation, and dGPU-vs-iGPU preference.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The app prints ✅ → ⚠ and emoji. A real console handles that, but when output is
# redirected (log file, pipe, pythonw) Windows uses cp1252 and print() raises
# UnicodeEncodeError — e.g. after a LoRA finished training. Never let logging crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Directory layout ───────────────────────────────────────────────────────────
APP_DIR    = Path(__file__).parent.resolve()
REPO_DIR   = APP_DIR.parent
ZLUDA_DIR  = REPO_DIR / "ZLUDA_v6" / "zluda"
PYTHON_DIR = REPO_DIR / "python-3.10"

MODELS_DIR      = APP_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LORAS_DIR       = MODELS_DIR / "loras"
UPSCALERS_DIR   = MODELS_DIR / "upscalers"
VAE_DIR         = MODELS_DIR / "vae"
EMBEDDINGS_DIR       = MODELS_DIR / "embeddings"
EMBEDDINGS_SD15_DIR  = EMBEDDINGS_DIR / "sd15"
EMBEDDINGS_SDXL_DIR  = EMBEDDINGS_DIR / "sdxl"
OUTPUTS_DIR     = APP_DIR / "outputs"
PRESETS_DIR     = APP_DIR / "presets"
PRESETS_SD15    = PRESETS_DIR / "sd15"
PRESETS_XL      = PRESETS_DIR / "xl"
SETTINGS_DIR    = APP_DIR / "settings"

for _d in (CHECKPOINTS_DIR, LORAS_DIR, UPSCALERS_DIR, VAE_DIR, EMBEDDINGS_DIR,
           EMBEDDINGS_SD15_DIR, EMBEDDINGS_SDXL_DIR,
           OUTPUTS_DIR, PRESETS_SD15, PRESETS_XL, SETTINGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Device selection ───────────────────────────────────────────────────────────
# SELECTED_GPU_INDEX is set by the UI (Settings tab) or via env var IMAGEGEN_GPU.
# It maps to the torch.cuda device index (ZLUDA v6 lists HIP devices in HIP order).
def _env_gpu_index() -> int:
    """IMAGEGEN_GPU as an int; 0 for unset / empty / junk (a typo must not stop start-up)."""
    try:
        return max(0, int(str(os.environ.get("IMAGEGEN_GPU", "0")).strip() or 0))
    except ValueError:
        print(f"[Config] Ignoring IMAGEGEN_GPU={os.environ.get('IMAGEGEN_GPU')!r} (not a number) — using GPU 0")
        return 0


SELECTED_GPU_INDEX: int = _env_gpu_index()

# Hardware & compatibility defaults for AMD ROCm / ZLUDA / DirectML.
# (HSA_OVERRIDE_GFX_VERSION used to be forced to 10.3.0 here. It is a Linux ROCr variable —
#  Windows HIP ignores it: the GPU self-test passes even with a bogus value. Removed.)
if not os.environ.get("MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK"):
    os.environ["MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK"] = "0"
# Required: with cublasLt enabled every Linear-with-bias fails under ZLUDA
# (CUBLAS_STATUS_NOT_SUPPORTED — hipBLASLt has no gfx103x kernels). selftest checks it.
if not os.environ.get("DISABLE_ADDMM_CUDA_LT"):
    os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"
# AMD_SERIALIZE_KERNEL=3 used to be forced here to hide NaN latents from MIOpen
# convolutions. Disabling cuDNN (see get_device) fixes those at the source, and
# serialization costs ~1.4x per step. Set it manually only for debugging.
if not os.environ.get("ZLUDA_NO_TELEMETRY"):
    os.environ["ZLUDA_NO_TELEMETRY"] = "1"

# gfx1031 rocBLAS kernels: the launchers set ROCBLAS_TENSILE_LIBPATH (and ROCM_ARCH) from
# backend/rocm_env.py. When started some other way under ZLUDA, decide here — but only for
# a gfx1031 GPU: the library holds nothing else, so forcing it breaks every other GPU.
if not os.environ.get("ROCBLAS_TENSILE_LIBPATH") and not os.environ.get("ROCM_ARCH"):
    try:
        sys.path.insert(0, str(APP_DIR))
        from backend.rocm_env import choose as _choose_rocm
        _r = _choose_rocm(SELECTED_GPU_INDEX)
        os.environ["ROCM_ARCH"] = _r["arch"]
        if _r["tensile"]:
            os.environ["ROCBLAS_TENSILE_LIBPATH"] = _r["tensile"]
    except Exception:
        pass

if not os.environ.get("HF_HOME") and (REPO_DIR / ".hf_cache").exists():
    hf_dir = str(REPO_DIR / ".hf_cache")
    os.environ["HF_HOME"] = hf_dir
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf_dir) / "hub"))


def get_device(gpu_index: int | None = None, backend: str | None = None) -> str:
    """
    Return the torch device string for a given GPU index.
    Falls back: cuda → directml → cpu (or directml if explicitly requested).
    """
    if os.environ.get("FORCE_CPU") or backend == "cpu":
        return "cpu"
    target_backend = (backend or os.environ.get("IMAGEGEN_BACKEND", "")).lower()
    idx = gpu_index if gpu_index is not None else SELECTED_GPU_INDEX

    # If DirectML is explicitly requested
    if target_backend in ("directml", "dml") or os.environ.get("FORCE_DML"):
        try:
            import torch_directml
            dml_count = torch_directml.device_count()
            dml_idx = idx if 0 <= idx < dml_count else 0
            return str(torch_directml.device(dml_idx))
        except ImportError:
            pass

    # Try CUDA / ZLUDA (preferred default when available)
    try:
        import torch
        if torch.cuda.is_available() and target_backend not in ("directml", "dml", "cpu"):
            count = torch.cuda.device_count()
            dev_idx = idx if 0 <= idx < count else 0
            # ZLUDA on AMD: disable SDPA flash/mem-efficient (needs cutlass, unavailable)
            name = torch.cuda.get_device_name(dev_idx)
            if "ZLUDA" in name or "AMD" in name or "Radeon" in name:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                # MIOpen convs via ZLUDA on RDNA2 (gfx1031) return NaN/garbage
                # unless every kernel is serialized. PyTorch's native
                # im2col + rocBLAS convs are correct, faster than serialized
                # MIOpen, and make GPU VAE decode usable. IMAGEGEN_CUDNN=1 reverts.
                if os.environ.get("IMAGEGEN_CUDNN") != "1":
                    torch.backends.cudnn.enabled = False
            return f"cuda:{dev_idx}"
    except ImportError:
        pass

    # DirectML fallback
    try:
        import torch_directml  # type: ignore
        dml_count = torch_directml.device_count()
        dml_idx = idx if 0 <= idx < dml_count else 0
        return str(torch_directml.device(dml_idx))
    except ImportError:
        pass
    return "cpu"


# Resolved at startup — can be overridden at runtime via set_active_gpu()
DEVICE: str = get_device(SELECTED_GPU_INDEX)


def set_active_gpu(index: int, backend: str | None = None):
    """Called from the Settings UI when user picks a different GPU or backend."""
    global SELECTED_GPU_INDEX, DEVICE
    if backend is not None:
        os.environ["IMAGEGEN_BACKEND"] = backend
    SELECTED_GPU_INDEX = index
    DEVICE = get_device(index, backend=backend)
    os.environ["IMAGEGEN_GPU"] = str(index)
    # No HIP_VISIBLE_DEVICES here: ZLUDA v6 breaks with it (multi-GPU), and HIP has already
    # started by now, so changing it could only affect child processes.


def get_torch_dtype(device: str | None = None):
    """Return appropriate dtype: float16 for GPU, float32 for CPU."""
    import torch
    dev = device or DEVICE
    if "cuda" in dev or "directml" in dev or "privateuseone" in dev:
        return torch.float16
    return torch.float32

# ── Civitai API ───────────────────────────────────────────────────────────────
CIVITAI_API_BASE = "https://civitai.com/api/v1"

# ── Persistent API key storage ────────────────────────────────────────────────
_API_KEYS_FILE = SETTINGS_DIR / "api_keys.json"

def _load_api_keys() -> dict:
    """Load saved API keys from disk."""
    if _API_KEYS_FILE.exists():
        try:
            import json
            return json.loads(_API_KEYS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_api_key(name: str, value: str):
    """Persist an API key to settings/api_keys.json."""
    import json
    keys = _load_api_keys()
    keys[name] = value.strip()
    _API_KEYS_FILE.write_text(json.dumps(keys, indent=2), encoding="utf-8")

def load_api_key(name: str) -> str | None:
    """Read a single API key from the persisted file."""
    return _load_api_keys().get(name) or None

# Set CIVITAI_API_KEY: env var > saved key > None
CIVITAI_API_KEY: str | None = (
    os.environ.get("CIVITAI_API_KEY")
    or load_api_key("civitai")
)

# ── HuggingFace token ──────────────────────────────────────────────────────────
# Stored in ~/.huggingface/token by `huggingface_hub.login()`.
# Also read from HF_TOKEN env var so it's available for all diffusers downloads.
def _load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        token = load_api_key("huggingface")
    if not token:
        _hf_token_file = Path.home() / ".huggingface" / "token"
        if _hf_token_file.exists():
            token = _hf_token_file.read_text().strip() or None
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return token

HF_TOKEN: str | None = _load_hf_token()

# ── Generation defaults ────────────────────────────────────────────────────────
DEFAULT_STEPS    = 20
DEFAULT_CFG      = 7.5
DEFAULT_WIDTH    = 512
DEFAULT_HEIGHT   = 512
DEFAULT_SAMPLER  = "DPMSolverMultistepScheduler"
DEFAULT_NEGATIVE = (
    "(worst quality:1.4), (low quality:1.4), normal quality, "
    "bad anatomy, bad hands, extra fingers, missing fingers, "
    "blurry, watermark, text, signature, jpeg artifacts, "
    "deformed, disfigured, extra limbs, mutation"
)

# ── Real-ESRGAN model download URL ────────────────────────────────────────────
REALESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
REALESRGAN_MODEL_PATH = UPSCALERS_DIR / "RealESRGAN_x4plus.pth"

# ── ONNX Real-ESRGAN (lighter, no PyTorch required for upscaling) ─────────────
REALESRGAN_ONNX_URL = (
    "https://github.com/cszn/BSRGAN/releases/download/v1.0/"
    "BSRGAN.onnx"
)
