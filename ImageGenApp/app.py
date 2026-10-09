"""
app.py — ImageGen Studio
Tabbed Gradio web UI for local Stable Diffusion image generation, upscaling,
and Civitai model management, with ZLUDA / ROCm / DirectML / CPU support.

Start via:  launch.bat  (Windows, ZLUDA)
       or:  python app.py [--port 7860] [--share]
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any


# ── Command line + "already running?" check ──────────────────────────────────
# Runs before torch/diffusers are imported, so a second double-click on
# launch.bat reopens the running app in about a second instead of after a
# full start-up (and instead of crashing on the busy port).
def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageGen Studio")
    parser.add_argument("--port",   type=int,  default=7860, help="Server port")
    parser.add_argument("--host",   type=str,  default="127.0.0.1")
    parser.add_argument("--share",  action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    return parser.parse_args(argv)


def _claim_port(args: argparse.Namespace) -> bool:
    """False if ImageGen Studio already runs on args.port (it gets opened instead).
    If another program owns the port, args.port moves to the next free one."""
    import socket, urllib.request, webbrowser
    host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host

    def busy(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0

    if not busy(args.port):
        return True
    url = f"http://{host}:{args.port}/"
    try:
        page = urllib.request.urlopen(url, timeout=3).read(4096).decode("utf-8", "replace")
    except Exception:
        page = ""
    if "ImageGen Studio" in page:
        print(f"[Launch] ImageGen Studio is already running at {url} — opening it.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return False
    free = next((p for p in range(args.port + 1, args.port + 20) if not busy(p)), None)
    if free is None:
        print(f"[Launch] Port {args.port} and the next 19 are busy — pass --port N.", flush=True)
        return False
    print(f"[Launch] Port {args.port} is used by another program — using {free} instead.", flush=True)
    args.port = free
    return True


_ARGS = None
if __name__ == "__main__":
    _ARGS = _parse_args()
    if not _claim_port(_ARGS):
        sys.exit(0)

# Suppress noisy-but-harmless warnings from diffusers / DML JIT tracing
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", message=".*TracerWarning.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")   # FutureWarning in 4.44
warnings.filterwarnings("ignore", category=UserWarning, message=".*device_type of 'cuda'.*")

# Suppress the CLIP tokenizer "sequence longer than 77 tokens" warning.
# When Compel is active (truncate_long_prompts=False), long prompts are handled
# by chunk-encoding — the warning is a false alarm from the underlying HF tokenizer.
import logging as _logging
_logging.getLogger("transformers.tokenization_utils_base").setLevel(_logging.ERROR)
# Suppress Windows asyncio connection-reset noise (harmless browser disconnect events)
_logging.getLogger("asyncio").setLevel(_logging.CRITICAL)
# Suppress Gradio/uvicorn WebSocket chatter (harmless reconnect events on slow callbacks)
_logging.getLogger("uvicorn.error").setLevel(_logging.CRITICAL)
_logging.getLogger("gradio").setLevel(_logging.WARNING)

# ── Ensure app directory is on sys.path regardless of cwd ─────────────────────
# This lets `import config` and `from backend.x import y` work when Python is
# invoked with an absolute path (e.g. via zluda.exe -- python C:\...\app.py).
_APP_DIR = Path(__file__).parent.resolve()
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
os.chdir(_APP_DIR)   # also set cwd so relative paths in config.py resolve correctly

# config first: it sets HF_HOME (and the ZLUDA env), and huggingface_hub — imported by gradio —
# reads its cache location only once, at import time.
import config  # noqa: F401
# Native crashes (access violations inside torch / safetensors / ZLUDA) otherwise end the process
# with no Python traceback at all — this prints where Python was when it happened.
import faulthandler
faulthandler.enable(all_threads=True)
import gradio as gr
from PIL import Image, ImageFile
# Off (it was on): with it a truncated upload decoded silently into a half-black picture that img2img / inpaint /
# upscale then used (round 4: an inpaint source read as its first 19 rows). Readers that browse files (History,
# PNG Info, metadata) catch the error themselves.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ── Local imports ──────────────────────────────────────────────────────────────
from config import (
    APP_DIR, OUTPUTS_DIR, DEVICE, CIVITAI_API_KEY, HF_TOKEN,
    DEFAULT_STEPS, DEFAULT_CFG, DEFAULT_WIDTH, DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE, set_active_gpu,
    PRESETS_SD15, PRESETS_XL, SETTINGS_DIR,
    save_api_key as _persist_key, load_api_key as _load_key,
)
from backend.hardware_detector import get_profile, GPUInfo, NPUInfo
from backend.sd_pipeline    import SDPipeline, SCHEDULER_MAP
from backend import records as _records
from backend.upscaler       import Upscaler
from backend.civitai_client import CivitaiClient, scrub as _civ_scrub
from backend.model_manager  import (
    list_checkpoints, list_loras, list_vaes, list_upscalers,
    model_dir_summary, short_name, get_model_info,
)
from backend.trigger_reader import render_triggers_html, batch_fetch_sidecars
from backend.tag_fetcher import I2I_ENHANCER_TAGS, fetch_danbooru_tags_html
from backend.vram_estimator import (
    estimate_generate, estimate_upscale, estimate_watermark,
    get_total_vram_mb, vram_bar_html,
)

from backend.smartsplit_pipeline import (
    SmartSplitPipeline, SmartSplitConfig, SmartSplitCapability,
    detect_smartsplit_capability,
)
from backend.help_content import build_help_tab
from backend.sdxl_pipeline import SDXLPipeline
from backend.png_info import read_png_info, format_png_info_html
from backend.prompt_tools import (merge_prompts, tidy_prompt, token_report_html, insert_after_quality,
                                  preload_tokenizer, tokenizer_ready)
# torch must be imported on this thread first: the preload thread imports transformers → torch, and
# two first imports of torch at once deadlock (_DeadlockError). On the GPU paths config already did
# it; with FORCE_CPU (--cpu) nothing had, and the app died at start-up.
import torch  # noqa: E402,F401
preload_tokenizer()
from backend.lora_keywords import chips_for, lora_keywords

try:                                   # Settings → Prompt weights (saved choice)
    import json as _j0
    from backend.prompt_syntax import set_emphasis as _set_emphasis
    _set_emphasis(_j0.loads((SETTINGS_DIR / "_prefs.json").read_text(encoding="utf-8")).get("emphasis", "a1111"))
except Exception:
    pass
try:                                   # Settings → Eye style (saved choice)
    from backend.detail_tools import set_eye_style as _set_eye_style
    _set_eye_style(_j0.loads((SETTINGS_DIR / "_prefs.json").read_text(encoding="utf-8")).get("eye_style", "round"))
except Exception:
    pass
try:                                   # Settings → Cool mode / pause while hot (saved choices)
    from backend import sampling as _smp0, thermal as _thm0
    _thm0.apply_saved(_j0.loads((SETTINGS_DIR / "_prefs.json").read_text(encoding="utf-8")), _smp0)
except Exception:
    pass

# ── Singleton service objects (created once at startup) ────────────────────────
sd       = SDPipeline()       # active pipeline — swapped to SDXLPipeline when SDXL model loaded
_sd15    = sd                 # keep reference for quick switch-back
_sdxl    = SDXLPipeline()     # dedicated SDXL / Pony / Illustrious pipeline
upscaler = Upscaler()
civitai  = CivitaiClient()

# ── Bridge: SD 1.5 LoRAs to restore after the model was freed for the SDXL stage ──
_bridge_sd15_loras: dict[str, dict] = {}

# ── SmartSplit state ───────────────────────────────────────────────────────────
_smartsplit_cap:    SmartSplitCapability | None = None
_smartsplit_cfg:    SmartSplitConfig            = SmartSplitConfig(enabled=False)
_smartsplit_pipe:   SmartSplitPipeline | None   = None

# ── Generation abort flag ──────────────────────────────────────────────────────
_generation_abort = threading.Event()
try:                                   # Stop ends a cool-mode pause early
    from backend import sampling as _smp1
    _smp1.should_stop = _generation_abort.is_set
except Exception:
    pass
_autoloop_active  = threading.Event()   # set = looping, clear = stopped
# One GPU job at a time. Gradio's shared queue slot alone isn't enough: a Stop with `cancels=`
# cancelled the asyncio task and handed the slot to the next job while the stopped job's thread
# was still loading / sampling (audit F-02b). This lock is held for the whole job.
# A plain Lock, not an RLock: Gradio steps generator jobs (Auto-Loop, outfit batch) on pool threads, so
# the thread that releases it may not be the one that took it — an RLock then raises "cannot release
# un-acquired lock" and stays held, hanging every later GPU job. No GPU job calls another one.
_GPU_LOCK = threading.Lock()


def gpu_job(fn):
    """Mark `fn` as a GPU job; `_serialize_gpu_events` puts it in the shared queue slot and runs it
    under `_GPU_LOCK` (generators hold the lock until they finish)."""
    fn._gpu_job = True
    return fn


def _locked(fn):
    import functools, inspect
    if inspect.isgeneratorfunction(fn):
        @functools.wraps(fn)
        def gen(*a, **k):
            with _GPU_LOCK:
                yield from fn(*a, **k)
        return gen

    @functools.wraps(fn)
    def run(*a, **k):
        with _GPU_LOCK:
            return fn(*a, **k)
    return run

class _GenerationAborted(Exception):
    """Raised inside generation callbacks to cleanly stop an in-progress run."""
    pass

# ── Civitai search state (stored between calls) ────────────────────────────────
_search_results: list[dict[str, Any]] = []
_search_all_results: list[dict[str, Any]] = []  # full result set (all pages)
_search_page: int = 0  # current page index (0-based)
_current_model:  dict[str, Any] | None = None
_RESULTS_PER_PAGE = 20


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds into human-readable H:MM:SS or M:SS."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


# ═════════════════════════════════════════════════════════════════════════════
# Helper: device status banner
# ═════════════════════════════════════════════════════════════════════════════
def _device_badge(device: str | None = None) -> str:
    import config as _cfg
    dev = device or _cfg.DEVICE
    color_map = {"cuda": "#89b4fa", "directml": "#a6e3a1", "privateuseone": "#a6e3a1", "cpu": "#f38ba8"}
    label_map = {"cuda": "🟦 CUDA / ZLUDA (ROCm)", "directml": "🟩 DirectML", "privateuseone": "🟩 DirectML", "cpu": "🟥 CPU"}
    base = dev.split(":")[0]
    color = color_map.get(base, "#cdd6f4")
    label = label_map.get(base, dev)
    return (
        f'<span style="background:{color};color:#1e1e2e;padding:2px 8px;'
        f'border-radius:4px;font-weight:bold;font-size:13px;">{label}</span>'
    )


def _gpu_choice_label(g: GPUInfo) -> str:
    vram = f"{round(g.vram_mb / 1024)} GB" if g.vram_mb >= 1024 else f"{g.vram_mb} MB"
    tag  = "[iGPU]" if g.is_igpu else "[dGPU]"
    rdna = f"RDNA{g.rdna_gen}" if g.rdna_gen else "GPU"
    star = "★ " if g == get_profile().best_gpu else ""
    return f"{star}GPU {g.index}: {g.name}  {tag}  {vram}  {rdna}"


def _npu_badge(npu: NPUInfo) -> str:
    if npu.available:
        return (
            f'<span style="background:#cba6f7;color:#1e1e2e;padding:2px 8px;'
            f'border-radius:4px;font-weight:bold;font-size:13px;">⚡ NPU: {npu.name}</span>'
        )
    return (
        f'<span style="background:#45475a;color:#cdd6f4;padding:2px 8px;'
        f'border-radius:4px;font-size:13px;">NPU: {npu.name}</span>'
    )


# ═════════════════════════════════════════════════════════════════════════════
# Prompt presets & compatibility helpers
# ═════════════════════════════════════════════════════════════════════════════
# Universal negative prompt building blocks
_NEG_ANATOMY = (
    "bad anatomy, bad hands, bad feet, bad face, bad proportions, "
    "extra fingers, missing fingers, fused fingers, too many fingers, (mutated hands:1.3), "
    "poorly drawn hands, poorly drawn face, extra limbs, missing limbs, "
    "floating limbs, disconnected limbs, extra arms, extra legs, "
    "missing arms, missing legs, extra head, multiple heads, cloned face, duplicate"
)
_NEG_QUALITY = (
    "(worst quality:1.4), (low quality:1.4), normal quality, "
    "jpeg artifacts, compression artifacts, blurry, out of focus, noisy, grainy, "
    "oversaturated, washed out, watermark, signature, text, logo, username, "
    "speech bubble, border, cropped, out of frame, cut off"
)
_NEG_ARTIFACTS = (
    "(deformed:1.3), disfigured, malformed limbs, mutation, morbid, "
    "grotesque, ugly, draft, tiling, clipping"
)
_NEG_SFW  = f"{_NEG_QUALITY}, {_NEG_ANATOMY}, {_NEG_ARTIFACTS}"
_NEG_NSFW = f"censored, mosaic, censor bar, pixel censor, black bar, {_NEG_SFW}"
_NEG_PONY = (
    f"score_1, score_2, score_3, score_4, censored, mosaic, "
    f"{_NEG_QUALITY}, {_NEG_ANATOMY}, {_NEG_ARTIFACTS}"
)

_PROMPT_PRESETS: dict[str, dict[str, str]] = {
    # ── SFW quality boosters ─────────────────────────────────────────────────
    "SD 1.5 · Anime quality": {
        "pos": "masterpiece, best quality, highly detailed, (8k, best quality:1.2), 1girl, solo, anime, beautiful detailed eyes, cinematic lighting",
        "neg": f"((monochrome)), ((grayscale)), lowres, {_NEG_SFW}",
    },
    "SD 1.5 · Photorealistic": {
        "pos": "RAW photo, 8k uhd, dslr, masterpiece, best quality, (photorealistic:1.4), highly detailed, sharp focus, film grain, bokeh",
        "neg": f"drawing, painting, sketch, cartoon, anime, {_NEG_SFW}",
    },
    "SDXL / Illustrious / NoobAI · Anime": {
        "pos": "masterpiece, best quality, ultra-detailed, 8k resolution, anime, vibrant colors, sharp lines, professional illustration, 1girl",
        "neg": f"{_NEG_SFW}",
    },
    "SDXL · Photorealistic": {
        "pos": "professional RAW photo, DSLR quality, bokeh, sharp focus, masterpiece, best quality, 8k uhd, film grain, Fujifilm XT3",
        "neg": f"illustration, painting, drawing, art, sketch, anime, cartoon, {_NEG_SFW}",
    },
    "Pony / NoobAI · Quality tokens": {
        "pos": "score_9, score_8_up, score_7_up, masterpiece, best quality, ultra-detailed, 1girl, solo",
        "neg": _NEG_PONY,
    },
    "General · Quality Boost": {
        "pos": "masterpiece, best quality, ultra-detailed, sharp focus, professional, 8k",
        "neg": _NEG_SFW,
    },
    # ── Character-focused presets ─────────────────────────────────────────
    "🎯 Character Portrait · Pony/IL": {
        "pos": "score_9, score_8_up, score_7_up, masterpiece, best quality, ultra-detailed, 1girl, solo, upper body, looking at viewer, beautiful detailed eyes, sharp focus, cinematic lighting, professional illustration",
        "neg": _NEG_PONY,
        "steps": 35, "cfg": 6.5, "width": 832, "height": 1216,
    },
    "🎯 Character Full Body · Pony/IL": {
        "pos": "score_9, score_8_up, score_7_up, masterpiece, best quality, ultra-detailed, 1girl, solo, full body, standing, beautiful detailed eyes, sharp focus, cinematic lighting, professional illustration, detailed background",
        "neg": _NEG_PONY,
        "steps": 35, "cfg": 6.5, "width": 832, "height": 1216,
    },
}

# Quick-insert tag chips grouped by category (used by the 🏷️ Quick Tags bar)
_TAG_CATEGORIES: dict[str, list[str]] = {
    "✨ Quality": [
        "masterpiece", "best quality", "ultra-detailed", "8k uhd",
        "sharp focus", "HDR", "film grain", "bokeh", "high resolution",
        "RAW photo", "professional", "cinematic lighting",
    ],
    "🎨 Style": [
        "anime", "photorealistic", "illustration", "3D render",
        "oil painting", "watercolor", "digital art", "cel shading",
        "painterly", "concept art", "pixel art",
    ],
    "👤 Subject": [
        "1girl", "1boy", "solo", "2girls", "multiple girls",
        "couple", "group", "female focus", "male focus", "animal",
    ],
    "🌄 Scene": [
        "outdoors", "indoors", "city", "forest", "beach", "night",
        "sunset", "rain", "snow", "starry sky", "cafe", "classroom",
    ],
    "🎭 Expression / Pose": [
        "smile", "blush", "looking at viewer", "closed eyes", "surprised",
        "standing", "sitting", "walking", "arms up", "hand on hip",
    ],
    "👗 Outfit": [
        "dress", "school uniform", "suit", "hoodie", "armor", "kimono",
        "casual clothes", "hat", "glasses", "scarf",
    ],
    "🚫 Negative tags": [
        "worst quality", "low quality", "bad anatomy", "extra limbs",
        "missing fingers", "blurry", "watermark", "text",
        "ugly", "deformed", "signature",
    ],
}


def _sdxl_lineage(path: str, is_lora: bool) -> str | None:
    """'pony' / 'illustrious' for an SDXL file when its metadata or name says so."""
    import re as _re
    if not is_lora:
        try:
            from backend.sd_pipeline import _model_family
            fam = _model_family(path)
            return fam if fam in ("pony", "illustrious") else None
        except Exception:
            return None
    try:
        import json as _j, struct as _st
        with open(path, "rb") as f:
            n = _st.unpack("<Q", f.read(8))[0]
            meta = (_j.loads(f.read(n)) if 2 < n < 100 << 20 else {}).get("__metadata__") or {}
    except Exception:
        meta = {}
    base = " ".join(str(meta.get(k, "")) for k in ("ss_sd_model_name", "modelspec.title", "ss_base_model")).lower()
    # "290640" is Pony Diffusion V6 XL's Civitai version id (trainers often keep the download name)
    if "pony" in base or _re.search(r"(^|\D)290640(\D|$)", base):
        return "pony"
    if any(x in base for x in ("illustrious", "noob", "wai")):
        return "illustrious"
    return None


def _infer_base(name: str) -> str:
    """Infer SD base model type from Civitai sidecar JSON or filename heuristics."""
    import json, re
    from config import LORAS_DIR, CHECKPOINTS_DIR
    # 1) Try Civitai sidecar JSON first (most reliable)
    for search_dir in (LORAS_DIR, CHECKPOINTS_DIR):
        sidecar = search_dir / (name + ".civitai.json") if search_dir else None
        if sidecar and sidecar.exists():
            try:
                bm = json.loads(sidecar.read_text(encoding="utf-8")).get("baseModel", "").lower()
                if "pony" in bm:
                    return "pony"
                if "illustrious" in bm or "noob" in bm:
                    return "illustrious"
                if "sdxl" in bm or "xl" in bm:
                    return "sdxl"
                if "sd 1" in bm or "sd1" in bm:
                    return "sd15"
                if "sd 2" in bm or "sd2" in bm:
                    return "sd2"
                if "flux" in bm:
                    return "flux"
            except Exception:
                pass
    # 2) The safetensors header knows the architecture for sure — names like
    #    "perfectdeliberate_v8" (SDXL) or "foo_v2" (SD 1.5) fool the heuristics below.
    from backend.model_manager import checkpoint_arch, lora_arch
    try:
        in_loras = Path(str(name)).resolve().is_relative_to(LORAS_DIR.resolve())
    except Exception:
        in_loras = False
    arch = (lora_arch if in_loras else checkpoint_arch)(str(name))
    if arch in ("sd1", "sd2"):
        return {"sd1": "sd15", "sd2": "sd2"}[arch]
    # 2b) SDXL lineage (Pony vs Illustrious/NoobAI): a LoRA's training metadata names the base
    #     it was trained on; checkpoints go through the pipeline's own family detection
    if arch == "sdxl":
        lineage = _sdxl_lineage(str(name), in_loras)
        if lineage:
            return lineage
    # 3) Filename heuristics (for sdxl: tells Pony / Illustrious apart)
    n = Path(name).stem.lower()
    if "flux" in n:
        return "flux"
    if "pony" in n:
        return "pony"
    if any(x in n for x in ("illustrious", "noobai")):
        return "illustrious"
    # "IL" / "ill" as standalone token (e.g. IL_mymodel, style-IL, _ill)
    if re.search(r'(?:^|[-_ ])il(?:l)?(?:[-_ .]|$)', n):
        return "illustrious"
    if arch == "sdxl" or "sdxl" in n or re.search(r'(?:^|[-_ .])xl(?:[-_ .]|$)', n):
        return "sdxl"
    if any(x in n for x in ("v2", "sd2", "2-1", "2.1", "768-v")):
        return "sd2"
    if any(x in n for x in ("1-5", "1.5", "sd1", "sd15", "dreamshaper", "deliberate", "realistic")):
        return "sd15"
    return "unknown"


# ═════════════════════════════════════════════════════════════════════════════
# User prompt presets & generation settings persistence
# ═════════════════════════════════════════════════════════════════════════════
import json as _json

def _ocr_langs(langs_str) -> list[str]:
    """EasyOCR language list: English plus at most one other script (it refuses ch_sim with ja/ko)."""
    langs = [l.strip() for l in str(langs_str or "en").replace("+", " ").split() if l.strip()] or ["en"]
    other = [l for l in langs if l != "en"][:1]
    return (["en"] if "en" in langs or not other else []) + other


def _safe_name(name) -> str:
    """A preset / settings name usable as a file name: no folders ("../x"), no characters
    Windows forbids, no reserved device names (CON, NUL…). '' if nothing is left."""
    import re
    # leading "_" / "." are stripped too: "_name" files are the app's own (_last_session…)
    n = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name or "")).strip().lstrip("._ ").rstrip(". ")
    if n.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                                   *(f"LPT{i}" for i in range(1, 10))}:
        n = "_" + n
    return n[:120]


def _user_preset_dir(model_family: str | None = None) -> Path:
    """Return the correct presets subfolder based on current model family."""
    fam = model_family or getattr(sd, "model_family", "sd15")
    if fam in ("sdxl", "pony", "illustrious"):
        return PRESETS_XL
    return PRESETS_SD15


def _list_user_presets(model_family: str | None = None) -> list[str]:
    """List saved user prompt presets for the active model family."""
    d = _user_preset_dir(model_family)
    return sorted(p.stem for p in d.glob("*.txt") if p.is_file())


def _load_user_preset(name: str, model_family: str | None = None) -> str:
    """Read a saved prompt preset by name. Returns prompt text or empty."""
    name = _safe_name(name)
    f = _user_preset_dir(model_family) / f"{name}.txt"
    if name and f.is_file():
        return f.read_text(encoding="utf-8", errors="replace").strip()
    return ""


def _save_user_preset(name: str, prompt: str, model_family: str | None = None) -> str:
    """Save current positive prompt as a named preset. Returns status message."""
    name = _safe_name(name)
    if not name:
        return "❌ Preset name cannot be empty."
    if not (prompt or "").strip():
        return "❌ Prompt is empty — nothing to save."
    f = _user_preset_dir(model_family) / f"{name}.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(prompt.strip(), encoding="utf-8")
    fam = "XL/Pony/IL" if f.parent == PRESETS_XL else "SD 1.5"
    return f"✅ Saved preset '{name}' ({fam})"


def _delete_user_preset(name: str, model_family: str | None = None) -> str:
    """Delete a saved user preset."""
    name = _safe_name(name)
    if not name:
        return ""
    f = _user_preset_dir(model_family) / f"{name}.txt"
    if f.is_file():
        f.unlink()
        return f"✅ Deleted preset '{name}'"
    return f"❌ Preset '{name}' not found."


def _save_generation_settings(prompt, neg_prompt, scheduler, steps, cfg,
                              width, height, batch, seed, name="", model=None, vae=None,
                              lora_slots=None, auto_quality=None, extra=None, example=None) -> str:
    """Save all current generation settings to a JSON file (format 2: + checkpoint / VAE /
    LoRA slots with weights / auto-quality / hires, face detail, boosters… as set in the UI)."""
    sname = _safe_name(name) or "last_settings"
    # casefold: Windows file names aren't case-sensitive, so "API_KEYS" is api_keys.json
    if sname.casefold() in _RESERVED_SETTINGS or sname.startswith("_"):
        return f"❌ '{sname}' is a reserved name — choose another."
    steps, cfg, width, height, batch, seed, _, _ = _clean_gen_args(steps, cfg, width, height, batch, seed)
    slots = [[Path(str(l)).name if l and l != "none" else "none", round(float(_num(w, 0.8)), 3)]
             for l, w in (lora_slots or [])][:3]
    data = {
        "format": 2,
        "prompt": prompt or "", "negative_prompt": neg_prompt or "",
        "scheduler": scheduler, "steps": steps, "cfg_scale": cfg,
        "width": width, "height": height,
        "batch_size": batch, "seed": seed,
        "model": str(model or getattr(sd, "current_model", "") or ""),
        "model_family": getattr(sd, "model_family", ""),
        # older readers: the names of the LoRAs in use
        "loras": [Path(n).stem for n, _ in slots if n != "none"] if lora_slots is not None
        else getattr(sd, "loaded_loras", []),
    }
    if lora_slots is not None:
        data["lora_slots"] = slots
    if vae is not None:
        data["vae"] = Path(str(vae)).name if vae and vae != "none" else "none"
    if auto_quality is not None:
        data["auto_quality"] = bool(auto_quality)
    if extra is not None:
        data["extra"] = _clean_extra(extra)
    if example:
        data["example_image"] = Path(str(example)).name
    f = SETTINGS_DIR / f"{sname}.json"
    f.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"✅ Settings saved as '{sname}'"


# settings/ also holds api_keys.json and the auto-saved session — never list or overwrite those
_RESERVED_SETTINGS = {"api_keys"}


def _list_saved_settings() -> list[str]:
    return sorted(p.stem for p in SETTINGS_DIR.glob("*.json")
                  if p.is_file() and p.stem.casefold() not in _RESERVED_SETTINGS
                  and not p.stem.startswith("_"))


# ── Last session (restored on startup) ──────────────────────────────────────
_LAST_SESSION_FILE = SETTINGS_DIR / "_last_session.json"


def _read_last_session() -> dict:
    try:
        return _json.loads(_LAST_SESSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


_LAST_BRIDGE_FILE = SETTINGS_DIR / "_last_bridge.json"   # separate: Generate rewrites its own file


def _read_last_bridge() -> dict:
    try:
        return _json.loads(_LAST_BRIDGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_last_session(data: dict) -> None:
    try:
        _LAST_SESSION_FILE.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[Session] Could not save last session: {e}")


# Gradio 4.19 ignores server-side Accordion(open=...) updates, so open it client-side
_OPEN_I2I_JS = """
() => {
  const h = [...document.querySelectorAll('.label-wrap')].find(e => e.innerText.includes('Image-to-Image'));
  if (h && !h.classList.contains('open')) h.click();
  if (h) h.scrollIntoView({behavior: 'smooth', block: 'center'});
}
"""


# Native resolutions per model family ("label → W×H" parsed by _parse_size_preset)
_SIZE_PRESETS = [
    "SD 1.5 · 512×512 square",
    "SD 1.5 · 512×768 portrait",
    "SD 1.5 · 768×512 landscape",
    "SDXL · 1024×1024 square",
    "SDXL · 832×1216 portrait",
    "SDXL · 1216×832 landscape",
    "SDXL · 896×1152 portrait",
    "SDXL · 768×1344 tall",
    "SDXL · 1344×768 wide",
]


def _parse_size_preset(label: str | None):
    import re
    m = re.search(r"(\d+)×(\d+)", label or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


_num = _records.num                                  # moved to backend/records.py (round 9); same name, same behaviour


def _clean_gen_args(steps, cfg, width, height, batch, seed, strength=0.75, img2img=False):
    """Make generation inputs safe. Gradio 4.19 passes whatever is typed into a slider's
    number box (or set by PNG Info / a loaded settings file) straight through: sizes that
    aren't multiples of 8 make diffusers raise, a seed ≥ 2⁶⁴ makes torch raise, and a
    steps×strength below 1 leaves img2img with nothing to do. Returns the cleaned values
    plus a list of human-readable adjustments."""
    fixes = []
    steps_i = int(round(_num(steps, DEFAULT_STEPS)))
    batch_i = int(round(_num(batch, 1)))
    cfg_f = _num(cfg, DEFAULT_CFG)
    strength_f = _num(strength, 0.75)
    steps_c, batch_c = min(max(steps_i, 1), 150), min(max(batch_i, 1), 8)
    cfg_c, strength_c = min(max(cfg_f, 0.0), 30.0), min(max(strength_f, 0.01), 1.0)
    if steps_c != steps_i:
        fixes.append(f"steps {steps_i} → {steps_c}")
    if batch_c != batch_i:
        fixes.append(f"batch {batch_i} → {batch_c}")
    if cfg_c != cfg_f:
        fixes.append(f"CFG {cfg_f:g} → {cfg_c:g}")
    if strength_c != strength_f:
        fixes.append(f"strength {strength_f:g} → {strength_c:g}")
    size = []
    for label, v, dflt in (("width", width, DEFAULT_WIDTH), ("height", height, DEFAULT_HEIGHT)):
        raw = int(round(_num(v, dflt)))
        c = min(max(int(round(raw / 8)) * 8, 256), 2048)       # diffusers needs multiples of 8
        if c != raw:
            fixes.append(f"{label} {raw} → {c}")
        size.append(c)
    s = _num(seed, -1)
    seed_i = -1 if s < 0 else int(s)
    if seed_i >= 2**32:                                          # A1111 seeds are 32-bit too
        fixes.append(f"seed {seed_i} → {seed_i % 2**32}")
        seed_i %= 2**32
    if img2img and int(steps_c * strength_c) < 1:
        # ceil(1 / strength) on the real strength (the rounded one left 0 steps at 1 × 0.495)
        import math
        need = min(150, max(1, math.ceil(1 / max(strength_c, 1e-3))))
        while int(need * strength_c) < 1 and need < 150:
            need += 1
        fixes.append(f"steps {steps_c} → {need} (img2img runs steps × strength of them)")
        steps_c = need
    return steps_c, cfg_c, size[0], size[1], batch_c, seed_i, strength_c, fixes


# Quality tags per model family (what the models were trained with). Added only when the
# prompt has none of that family's own quality tags; merged, so nothing is duplicated.
_FAMILY_QUALITY = {
    # Illustrious / NoobAI: aesthetic + quality tags from their training captions
    "illustrious": ("masterpiece, best quality, amazing quality, very aesthetic, absurdres",
                    "worst quality, low quality, bad quality, lowres, bad anatomy, jpeg artifacts, "
                    "signature, watermark"),
    "pony": ("score_9, score_8_up, score_7_up", "score_4, score_5, score_6"),
    "sdxl": ("masterpiece, best quality", "worst quality, low quality"),
    "sd15": ("masterpiece, best quality", "worst quality, low quality, lowres"),
}
_QUALITY_MARKERS = {
    "illustrious": ("masterpiece", "best quality", "amazing quality", "very aesthetic"),
    "pony": ("score_9", "score_8", "score_7"),
    "sdxl": ("masterpiece", "best quality"),
    "sd15": ("masterpiece", "best quality"),
}


def _family_quality(family: str, prompt: str, negative: str) -> tuple[str, str]:
    """(quality tags to add, negative tags to add) for a model family — each only when the
    prompt / negative has none of that family's own quality tags yet."""
    fam = family if family in _FAMILY_QUALITY else "sd15"
    pos, neg = _FAMILY_QUALITY[fam]
    p, n = (prompt or "").lower(), (negative or "").lower()
    add_pos = "" if any(m in p for m in _QUALITY_MARKERS[fam]) else pos
    neg_markers = ("score_4", "score_5") if fam == "pony" else ("worst quality", "low quality")
    add_neg = "" if any(m in n for m in neg_markers) else neg
    return add_pos, add_neg


def _clean_extra(extra: dict | None) -> dict:
    """CLIP skip / variation / hires-fix settings made safe (UI values, saved sessions and restored images can all hold junk): backend/records.clean_extra with this app's upscaler."""
    return _records.clean_extra(extra, upscaler.available_methods)


# LMS / PNDM filled SDXL pictures with iridescent colour noise until 2026-10-02 (4th-order Adams–Bashforth is unstable on
# SDXL-family models — backend/sampling lowers the order at the end and caps it at 2 there). PNDM and Heun are clean since,
# but leave more grain in smooth areas than DPM++ / Euler (measured, 3 seeds: chroma high-pass 0.63 / 0.65 vs 0.52–0.56) —
# the result says so
_SDXL_BAD_SAMPLERS = ("PNDM", "Heun")
_SDXL_FAMILIES = ("sdxl", "pony", "illustrious")

_XY_AXES = ["none", "CFG", "Steps", "Sampler", "Seed", "LoRA 1 weight", "CLIP skip", "Hires denoise",
            "Face denoise", "Hand denoise", "Eye denoise", "PAG scale", "CFG rescale", "Prompt S/R", "Checkpoint"]


def _xy_values(axis: str, text: str) -> tuple[list, str]:
    """Values for one X/Y axis from "a, b, c" (numbers also as ranges "4-10:2" / "1-5").
    Returns (values, error message)."""
    if not axis or axis == "none":
        return [None], ""
    raw = [v.strip() for v in str(text or "").split(",") if v.strip()]
    if not raw:
        return [], f"Enter values for {axis}, separated by commas."
    if axis == "Prompt S/R":
        return raw, ""
    if axis == "Checkpoint":
        out = []
        for v in raw:
            key = v.lower().removesuffix(".safetensors")
            hits = [p for n, p in list_checkpoints() if key in n.lower()]
            exact = [p for p in hits if Path(p).stem.lower() == key]
            if not hits:
                return [], f"No checkpoint matches “{html.escape(v)}”."
            out.append((exact or hits)[0])
        return out, ""
    if axis == "Sampler":
        out = []
        for v in raw:
            m = next((n for n in SCHEDULER_MAP if n.lower() == v.lower()), None) or \
                next((n for n in SCHEDULER_MAP if v.lower() in n.lower()), None)
            if not m:
                return [], f"Unknown sampler “{html.escape(v)}” — choose from: {', '.join(SCHEDULER_MAP)}"
            out.append(m)
        return out, ""
    vals = []
    for v in raw:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?:\s*:\s*(\d+(?:\.\d+)?))?", v)
        try:
            if m:
                a, b, st = float(m.group(1)), float(m.group(2)), float(m.group(3) or 1)
                if st <= 0:
                    return [], f"“{html.escape(v)}”: the step after “:” must be above 0 ({axis})."
                # "9-5" counts down (it used to give no values and crash the grid after the model load)
                x, sign = a, (1 if b >= a else -1)
                while (x <= b + 1e-9 if sign > 0 else x >= b - 1e-9) and len(vals) < 64:
                    vals.append(x); x += sign * st
            else:
                vals.append(float(v))
        except ValueError:
            return [], f"“{html.escape(v)}” isn't a number ({axis})."
    if axis in ("Steps", "Seed", "CLIP skip"):
        vals = [int(round(v)) for v in vals]
    if not vals:
        return [], f"No values for {axis}."
    return vals, ""


def _xy_grid(cells: list, xlabels: list, ylabels: list, title: str):
    """Labelled grid image (rows = Y values, columns = X values) on a dark background."""
    from PIL import Image as _Image, ImageDraw, ImageFont
    w0, h0 = cells[0].size
    f = min(1.0, 512 / max(w0, h0))
    cw, ch = int(w0 * f), int(h0 * f)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = small = ImageFont.load_default()
    left = 170 if any(ylabels) else 10
    top = 70
    cols, rows = len(xlabels), len(ylabels)
    grid = _Image.new("RGB", (left + cols * (cw + 6) + 4, top + rows * (ch + 6) + 4), (30, 30, 46))
    d = ImageDraw.Draw(grid)
    d.text((10, 8), title[:160], fill=(205, 214, 244), font=small)
    for c, lab in enumerate(xlabels):
        d.text((left + c * (cw + 6) + 6, 36), str(lab)[:40], fill=(137, 180, 250), font=font)
    for r, lab in enumerate(ylabels):
        if lab:
            d.text((10, top + r * (ch + 6) + ch // 2 - 10), str(lab)[:18], fill=(166, 227, 161), font=font)
    for i, im in enumerate(cells):
        r, c = divmod(i, cols)
        grid.paste(im.convert("RGB").resize((cw, ch), _Image.LANCZOS), (left + c * (cw + 6), top + r * (ch + 6)))
    return grid


def _carry_params(src_img, note: str | None):
    """PngInfo that keeps a source image's generation settings (A1111 text + our record) and
    appends what was done to it."""
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    src = getattr(src_img, "info", None) or {}
    params = src.get("parameters")
    if isinstance(params, bytes):
        params = params.decode("utf-8", "replace")
    if params:
        info.add_text("parameters", f"{params}\n{note}" if note else params)
    if src.get("imagegen"):
        info.add_itxt("imagegen", str(src["imagegen"]))
    return info


def _thermal_wait(progress=None) -> float:
    """Settings → 🌡 pause while hot: wait until the laptop is cooler before a picture or a post pass (Stop ends the wait).
    Returns the seconds waited."""
    from backend import thermal

    def say(text):
        try:
            if progress is not None:
                progress(0, desc=text)
        except Exception:
            pass
    return thermal.wait_cool(_generation_abort.is_set, say)


def _score_pause(progress=None, every: float = 10.0):
    """The `pause` callback of image_score.score_images: the 🌡 pause-while-hot wait before the first picture and then at most
    every `every` seconds of scoring (a zone read costs ~1 s). The ⭐ rating is CPU work and the zone follows the CPU die: with ORT's
    default threads a few seconds of it read 95–96 °C on the laptop (round 9). `pause.waited[0]` collects the seconds paused."""
    last, waited = [None], [0.0]

    def pause(k, n):
        if last[0] is not None and time.time() - last[0] < every:
            return
        waited[0] += _thermal_wait(progress)
        last[0] = time.time()
    pause.waited = waited
    return pause


def _eye_pass_skipped(pipe) -> bool:
    """The eye pass is skipped on Pony-family checkpoints: it softened them at every denoise (round 5, −28 % eye detail)."""
    return str(getattr(pipe, "model_family", "") or "").lower() == "pony"


_HAIR_COVER = re.compile(r"\b(hat|cap|bonnet|beret|veil|tiara|crown|headdress|headband|visor|beanie|goggles|headwear|hood|helmet|wig|bun|buns|"
                         r"hair bun|ponytail|twintails|braid|braids|hairband|mask on head)\b", re.I)


def _identity_margins(card: dict, outfits, n_seeds: int, n: int) -> list:
    """Per picture of an outfit batch (outfit-major: n_seeds pictures per outfit) the CCIP flag margin: identity_score.Z_FLAG_HAIR for an outfit with a hat /
    hairstyle (CCIP reads those as part of the character: 46 % of her own pictures in such outfits fell below the normal margin, 13 % below this one),
    None (the normal margin) for the others."""
    from backend import identity_score as _ids
    texts = card.get("outfits") or {}
    return [(_ids.Z_FLAG_HAIR if _HAIR_COVER.search(texts.get(outfits[min(i // max(1, n_seeds), len(outfits) - 1)], "") or "") else None)
            for i in range(n)] if outfits else [None] * n


def _flag_text(card: dict, outfit: str, flags) -> str:
    """The ⭐ flags of one picture as the outfit batch message shows them. A "not her?" flag on an outfit whose text changes the hair or head
    (hat, ponytail, buns, veil …) says so: CCIP reads hair and headwear as part of the character — round 6 flagged 15 of the 33 outfits of the v3 card with
    12 daylight references (every picture was her by eye; references that include such outfits flag 11 %)."""
    txt = ", ".join(flags)
    if any(str(f).startswith("not her?") for f in flags) and _HAIR_COVER.search((card.get("outfits") or {}).get(outfit, "") or ""):
        txt += " — a hat / hairstyle in this outfit lowers it: check by eye"
    return txt


def _inpaint_steps(steps, denoise) -> int:
    """Schedule length for an inpaint pass so that int(length × denoise) = the Steps value (A1111's behaviour): the
    pass used to run int(steps × denoise), so at 12 steps 0.75 and 0.8 were the same picture and 0.4 ran 4 steps."""
    import math
    steps, den = int(steps), max(0.05, float(denoise))
    n = math.ceil(steps / den)
    while n > 1 and int((n - 1) * den) >= steps:     # float rounding can make ceil() one too long
        n -= 1
    while int(n * den) < steps:
        n += 1
    return min(150, n)


def _editor_parts(val):
    """(background image, mask of everything painted) from a gr.ImageEditor value."""
    import numpy as _np
    if not isinstance(val, dict) or val.get("background") is None:
        return None, None
    bg = val["background"]
    W, H = bg.size
    m = _np.zeros((H, W), dtype=_np.uint8)
    for layer in val.get("layers") or []:
        if layer is not None:
            a = _np.array(layer.convert("RGBA").resize((W, H)))[:, :, 3]
            m = _np.maximum(m, (a > 10).astype(_np.uint8) * 255)
    return bg.convert("RGB"), (Image.fromarray(m, "L") if m.any() else None)


def _upscale_to(img, width: int, height: int, method: str = "Lanczos"):
    """img resized to exactly width×height: Lanczos, or Real-ESRGAN ×2/×4 then Lanczos to size
    (falls back to Lanczos if the model can't run)."""
    from PIL import Image as _Image
    img = img.convert("RGB")
    if method and method != "Lanczos":
        try:
            factor = 2 if max(width / img.width, height / img.height) <= 2 else 4
            big, _ = upscaler.upscale(img, factor, method)
            return big.resize((width, height), _Image.LANCZOS)
        except Exception as e:
            print(f"[Hires] {method} failed ({e}) — using Lanczos")
    return img.resize((width, height), _Image.LANCZOS)


def _fit_init_image(img, max_side: int = 2048, min_side: int = 64):
    """An img2img input scaled into a size the pipelines can handle: a 4000 px photo would
    need far more VRAM than any consumer card has (WDDM then pages to system RAM and the
    run crawls), and a tiny icon collapses to nothing in latent space."""
    from PIL import Image as _Image
    if not hasattr(img, "size"):
        return img, []
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
    if min(w, h) * scale < min_side:
        scale = min_side / max(1, min(w, h))
    if scale == 1.0:
        return img, []
    nw, nh = max(8, round(w * scale)), max(8, round(h * scale))
    out = img.resize((nw, nh), _Image.LANCZOS)
    note = f"init image {w}×{h} → {nw}×{nh}"
    # a sliver (1×2000) can't meet both limits: lifting its short side used to make it 64×128000
    if max(nw, nh) > max_side:
        cw, ch = min(nw, max_side), min(nh, max_side)
        x0, y0 = (nw - cw) // 2, (nh - ch) // 2
        out = out.crop((x0, y0, x0 + cw, y0 + ch))
        note += f", centre-cropped to {cw}×{ch}"
    out.info = dict(getattr(img, "info", {}) or {})
    return out, [note]


def _load_generation_settings(name: str):
    """Load saved settings. Returns tuple for all Gradio outputs or None."""
    name = _safe_name(name)
    if not name or name.casefold() in _RESERVED_SETTINGS:
        return [gr.update()] * 9
    f = SETTINGS_DIR / f"{name}.json"
    if not f.is_file():
        return [gr.update()] * 9
    try:
        d = _json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            return [gr.update()] * 9
        steps, cfg, width, height, batch, seed, _, _ = _clean_gen_args(
            d.get("steps", DEFAULT_STEPS), d.get("cfg_scale", DEFAULT_CFG),
            d.get("width", DEFAULT_WIDTH), d.get("height", DEFAULT_HEIGHT),
            d.get("batch_size", 1), d.get("seed", -1))
        d.update(steps=steps, cfg_scale=cfg, width=width, height=height, batch_size=batch, seed=seed)
        return [
            gr.update(value=str(d.get("prompt") or "")),
            gr.update(value=str(d.get("negative_prompt") or "")),
            gr.update(value=d.get("scheduler") if d.get("scheduler") in SCHEDULER_MAP else "DPM++ 2M Karras"),
            gr.update(value=d.get("steps", DEFAULT_STEPS)),
            gr.update(value=d.get("cfg_scale", DEFAULT_CFG)),
            gr.update(value=d.get("width", DEFAULT_WIDTH)),
            gr.update(value=d.get("height", DEFAULT_HEIGHT)),
            gr.update(value=d.get("batch_size", 1)),
            gr.update(value=d.get("seed", -1)),
        ]
    except Exception:
        return [gr.update()] * 9


# ═════════════════════════════════════════════════════════════════════════════
# Tab 1: Image Generation (txt2img + img2img)
# ═════════════════════════════════════════════════════════════════════════════
def _build_generate_tab():
    _ls = _read_last_session()   # restore where the user left off
    _lora_paths = {p for _, p in list_loras()}
    _ls_loras = (_ls.get("loras") or []) + [["none", None]] * 3

    def _ls_lora(i, default_w):
        path, w = _ls_loras[i][0], _ls_loras[i][1]
        ok = path in _lora_paths          # file may have been moved/deleted since
        return (path if ok else "none"), (w if ok and w is not None else default_w)
    with gr.Tab("🎨 Generate", id="generate"):
        gr.HTML(
            '<p style="color:#a6adc8;font-size:13px;">'
            'Generate images with Stable Diffusion (SD 1.x / SD 2.x / SDXL) '
            'using ZLUDA on AMD GPU or DirectML / CPU.</p>'
        )

        with gr.Row():
            # ── Left panel ──────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                _ckpt_choices = _refresh_checkpoints()
                _ckpt_values = [v for _, v in _ckpt_choices]
                _init_ckpt = (_ls.get("model") if _ls.get("model") in _ckpt_values
                              else (_ckpt_values[0] if _ckpt_values else ""))
                model_dd = gr.Dropdown(
                    label="Checkpoint",
                    choices=_ckpt_choices,
                    value=_init_ckpt,
                    allow_custom_value=True,
                    info="Local .safetensors / .ckpt or HuggingFace ID. "
                         "Loaded automatically when you press Generate.",
                )
                ckpt_info_html = gr.HTML(get_model_info(_init_ckpt) if _init_ckpt else "")
                _vae_choices = ["none"] + list_vaes()
                _init_vae = _ls.get("vae") if _ls.get("vae") in [
                    c[1] if isinstance(c, tuple) else c for c in _vae_choices] else "none"
                vae_dd = gr.Dropdown(
                    label="VAE (optional)",
                    choices=_vae_choices,
                    value=_init_vae,
                    info="Override the checkpoint's built-in VAE (e.g. vae-ft-mse for SD 1.5 = sharper colours).",
                )
                vae_info_html = gr.HTML("")
                load_model_btn  = gr.Button("⏏ Preload model", variant="secondary", size="sm")
                model_status    = gr.HTML(
                    '<p style="color:#a6adc8;">No model loaded yet — it loads automatically on '
                    'your first Generate (SD 1.5 ~20 s, SDXL ~30 s). Preload to do it now.</p>'
                    + _kernel_cache_note()
                )

                gr.Markdown("---")
                gr.HTML('<p style="color:#a6adc8;font-size:13px;margin:0 0 4px;">'
                        'LoRAs picked below are applied automatically when you Generate '
                        '(➕ Apply just does it right away). ✖ clears a slot.</p>')

                lora_dd = gr.Dropdown(
                    label="LoRA Slot 1 (Primary — Character)",
                    choices=_lora_choices(_init_ckpt),
                    value=_ls_lora(0, 0.8)[0],
                    info="Character LoRA — WHO is in the image. Slots 2–3 are for style/effects.",
                )
                lora_info1 = gr.HTML("")
                lora_weight = gr.Slider(0.1, 1.5, value=_ls_lora(0, 0.8)[1], step=0.05, label="Weight 1",
                                        info="0.7–0.85 recommended · >1.0 risks artifacts / burned colours.")
                with gr.Row():
                    apply_lora_btn  = gr.Button("➕ Apply Slot 1", size="sm")
                    remove_lora1_btn = gr.Button("✖ Remove Slot 1", size="sm")

                # slots 2–3 are for styles / extras: folded away unless the last session used one
                with gr.Accordion("🎨 LoRA slots 2–3 — style / extra",
                                  open=any(_ls_lora(i, 0.7)[0] != "none" for i in (1, 2))):
                    lora_dd2 = gr.Dropdown(
                        label="LoRA Slot 2 (Style / Effect)",
                        choices=_lora_choices(_init_ckpt),
                        value=_ls_lora(1, 0.7)[0],
                        info="Style / lighting / concept LoRA — HOW the image looks.",
                    )
                    lora_info2 = gr.HTML("")
                    lora_weight2 = gr.Slider(0.1, 1.5, value=_ls_lora(1, 0.7)[1], step=0.05, label="Weight 2",
                                             info="0.5–0.7 recommended · keep below the character LoRA's weight.")
                    with gr.Row():
                        apply_lora_btn2  = gr.Button("➕ Apply Slot 2", size="sm")
                        remove_lora2_btn = gr.Button("✖ Remove Slot 2", size="sm")

                    lora_dd3 = gr.Dropdown(
                        label="LoRA Slot 3 (Extra / Accent)",
                        choices=_lora_choices(_init_ckpt),
                        value=_ls_lora(2, 0.7)[0],
                        info="Secondary effect. With 3 LoRAs keep the weights' sum ≤ 2.0.",
                    )
                    lora_info3 = gr.HTML("")
                    lora_weight3 = gr.Slider(0.1, 1.5, value=_ls_lora(2, 0.7)[1], step=0.05, label="Weight 3",
                                             info="0.3–0.5 recommended when all 3 slots are used.")
                    with gr.Row():
                        apply_lora_btn3  = gr.Button("➕ Apply Slot 3", size="sm")
                        remove_lora3_btn = gr.Button("✖ Remove Slot 3", size="sm")

                remove_lora_btn = gr.Button("🗑 Clear All LoRAs", size="sm", variant="secondary")
                active_loras_html = gr.HTML("")
                lora_status  = gr.HTML("")
                compat_html  = gr.HTML("")   # compatibility warning
                triggers_html = gr.HTML("")  # status of the Civitai trigger-word fetch
                # keyword chips for what the last session had selected
                _kw_slots = [(lbl, v) for lbl, v in (("Model", _init_ckpt), ("S1", _ls_lora(0, 0.8)[0]),
                                                     ("S2", _ls_lora(1, 0.7)[0]), ("S3", _ls_lora(2, 0.7)[0]))
                             if v and v != "none"]
                try:
                    _kw_s, _kw_t, _kw_n = chips_for(_kw_slots)
                except Exception:
                    _kw_s, _kw_t, _kw_n = [], [], ""
                kw_note = gr.HTML(_kw_n)
                kw_ds = gr.Dataset(
                    label="🏷️ Keywords from the checkpoint and LoRA slots — click to add",
                    components=["textbox"], samples=_kw_s or [["-"]], type="index", visible=bool(_kw_s),
                    samples_per_page=40,
                )
                kw_tags_state = gr.State(_kw_t)
                kw_core_btn = gr.Button("✨ Add character tags (trigger words + tags in ≥ 50 % of its training images)",
                                        size="sm", visible=any(lbl != "Model" for lbl, _ in _kw_slots) and bool(_kw_s))
                with gr.Row():
                    fetch_triggers_btn = gr.Button(
                        "🔍 Fetch trigger words from Civitai", size="sm",
                        variant="secondary",
                    )
                fetch_triggers_status = gr.HTML("")

                with gr.Accordion("🎴 Character cards — one click to a full character setup", open=False):
                    from backend.character_cards import list_cards as _list_cards
                    _cards0 = _list_cards()
                    card_dd = gr.Dropdown(label="Character", choices=["(none)"] + _cards0,
                                          value=_cards0[0] if _cards0 else "(none)",
                                          info="Saved in settings/characters/ (checkpoint, LoRAs, tags, outfits, size…). "
                                               "📌 profiles: 🎴 Load uses the one for the checkpoint selected now")
                    from backend.character_cards import load_card as _load_card
                    _card0 = _load_card(_cards0[0]) if _cards0 else None
                    _outs0 = ["(no outfit tags)"] + list((_card0 or {}).get("outfits") or {})
                    outfit_dd = gr.Dropdown(label="Outfit", choices=_outs0, value=_outs0[1] if len(_outs0) > 1 else _outs0[0])
                    card_scene_txt = gr.Textbox(label="Scene / extra tags", lines=1,
                                                placeholder="__pose__, __expression__, __background__")
                    card_load_btn = gr.Button("🎴 Load → checkpoint, LoRAs, prompt, settings", variant="primary",
                                              size="sm")
                    card_name_txt = gr.Textbox(label="Card name", lines=1, value=_cards0[0] if _cards0 else "")
                    with gr.Row():
                        card_save_btn = gr.Button("💾 Save current setup", size="sm")
                        card_build_btn = gr.Button("🧩 Build from LoRA slot 1", size="sm")
                    card_profile_btn = gr.Button("📌 Save as this checkpoint's profile", size="sm")
                    with gr.Accordion("🧬 Learn her look — so the ⭐ rating can flag a face that isn't hers", open=False):
                        card_ident_files = gr.File(label="Pictures of her (3 or more, faces ≥ 100 px — pose and lighting don't matter; include the hats / hairstyles you want judged)",
                                                   file_count="multiple", file_types=["image"])
                        card_ident_btn = gr.Button("🧬 Learn from these pictures (first use downloads CCIP, 150 MB)", size="sm")
                    card_status = gr.HTML("")
                    with gr.Row():
                        card_batch_seeds = gr.Number(value=2, precision=0, minimum=1, maximum=8,
                                                     label="Seeds per outfit", scale=1)
                        card_batch_resume = gr.Checkbox(value=True, label="Skip pairs already made", scale=2,
                                                        info="Same settings + outfit + seed → reuse the saved image")
                    card_batch_btn = gr.Button("🎴▶ Generate every outfit (current settings, same seeds)",
                                               size="sm", variant="secondary")

                gr.Markdown("---")
                refresh_btn = gr.Button("🔄 Refresh model lists")

            # ── Right panel — generation controls ───────────────────────────
            with gr.Column(scale=3):
                with gr.Row(equal_height=False):
                    # ── Controls column ──────────────────────────────────────
                    with gr.Column(scale=5, min_width=420):
                        prompt_txt = gr.Textbox(
                            label="Prompt",
                            placeholder="score_9, score_8_up, masterpiece, best quality, 1girl, solo, ...",
                            lines=3,
                            value=_ls.get("prompt", ""),
                            info="Most important tags first. (tag:1.2) boosts, (tag:0.8) weakens. Ctrl+Enter generates.",
                        )
                        neg_prompt_txt = gr.Textbox(
                            label="Negative Prompt",
                            value=_ls.get("negative_prompt", DEFAULT_NEGATIVE),
                            lines=2,
                            info="What to avoid. Pony/IL: add score_4, score_5, score_6. 20–40 tags is plenty.",
                        )
                        auto_quality_cb = gr.Checkbox(
                            label="Auto-add quality tags",
                            value=_ls.get("auto_quality", True),
                            info="Adds the quality tags your model family was trained with (Illustrious/NoobAI: "
                                 "masterpiece … very aesthetic, absurdres · Pony: score_9… · SD 1.5: masterpiece, "
                                 "best quality) and matching negative tags — only if you have none.",
                        )
                        with gr.Row():
                            # an estimate until the tokenizer has loaded in the background
                            token_html = gr.HTML(token_report_html(
                                _ls.get("prompt", ""), _ls.get("negative_prompt", DEFAULT_NEGATIVE),
                                None if tokenizer_ready() else "estimate"))
                        # Danbooru tag autocomplete for the tag being typed at the end of the prompt
                        tag_ac_ds = gr.Dataset(label="🔤 Danbooru tags — click to complete", components=["textbox"],
                                               samples=[["-"]], type="index", visible=False, samples_per_page=12)
                        tag_ac_state = gr.State([])
                        with gr.Row():
                            tidy_btn = gr.Button("🧹 Tidy prompts — merge duplicate tags (strongest weight wins)",
                                                 size="sm", variant="secondary", scale=3)
                            spell_btn = gr.Button("💡 Fix Danbooru spellings", size="sm", variant="secondary",
                                                  scale=2)
                        with gr.Accordion("🎲 Wildcards — different picks for every image", open=False):
                            from backend.wildcards import list_wildcards as _list_wc
                            gr.HTML('<p style="color:#a6adc8;font-size:13px;margin:0 0 4px;">'
                                    '<code>{smile|pout|grin}</code> picks one per image · <code>{2$$a|b|c}</code> '
                                    'picks two · <code>{3::a|b}</code> makes a 3× likelier · <code>__outfit__</code> '
                                    'picks a line from <code>wildcards/outfit.txt</code> (add your own .txt files there '
                                    'or in <code>models/wildcards/</code>). Picks come from each image\'s seed, so '
                                    'the same seed gives the same picks; the resolved prompt is saved in the image.</p>')
                            _wc0 = _list_wc()
                            # type="index" + a State: with type="values" Gradio 4.19 looks the click up in
                            # the samples the Dataset was *built* with, so after 🔄 it inserted the wrong one
                            wc_ds = gr.Dataset(label="Click to add to the prompt", components=["textbox"],
                                               samples=[[f"__{n}__"] for n in _wc0] or [["-"]], samples_per_page=40,
                                               type="index")
                            wc_names_state = gr.State([f"__{n}__" for n in _wc0])
                            with gr.Row():
                                wc_preview_btn = gr.Button("👁 Preview 4 picks of this prompt", size="sm")
                                wc_refresh_btn = gr.Button("🔄 Reload wildcard files", size="sm")
                            wc_preview_html = gr.HTML("")

                        with gr.Row():
                            generate_btn = gr.Button("✨ Generate  (Ctrl+Enter)", variant="primary",
                                                     size="lg", scale=4, elem_id="gen-btn")
                            stop_btn     = gr.Button("⏹ Stop", variant="secondary", size="lg", scale=1)
                        gen_info     = gr.HTML("")

                        scheduler_dd = gr.Dropdown(
                            label="Sampler / Scheduler",
                            choices=list(SCHEDULER_MAP.keys()),
                            value=_ls.get("scheduler", "DPM++ 2M Karras"),
                            info="DPM++ 2M Karras: all-rounder · DPM++ 2M AYS: same quality in 10–12 steps · PNDM / Heun: a little more grain on SDXL",
                        )
                        with gr.Row():
                            steps_sl    = gr.Slider(1, 150, value=_ls.get("steps", DEFAULT_STEPS), step=1,  label="Steps",
                                                    info="AYS: 10–12 · Karras / Euler: 20–30")
                            cfg_sl      = gr.Slider(1, 30,  value=_ls.get("cfg_scale", DEFAULT_CFG), step=0.5, label="CFG Scale",
                                                    info="Pony / IL: 5–7 · SD 1.5: 7–9")

                        with gr.Row():
                            size_preset_dd = gr.Dropdown(
                                label="Size preset", choices=_SIZE_PRESETS, value=None, scale=3,
                                info="Native resolutions per model family.",
                            )
                            with gr.Column(scale=1, min_width=130):
                                swap_size_btn = gr.Button("⇄ Swap W/H", size="sm")

                        with gr.Row():
                            width_sl    = gr.Slider(256, 2048, value=_ls.get("width", DEFAULT_WIDTH), step=64, label="Width",
                                                    info="Multiple of 64.")
                            height_sl   = gr.Slider(256, 2048, value=_ls.get("height", DEFAULT_HEIGHT), step=64, label="Height",
                                                    info="SD 1.5 ≈ 512 · SDXL ≈ 1024.")
                            batch_sl    = gr.Slider(1, 8,     value=_ls.get("batch_size", 1), step=1, label="Batch Size",
                                                    info="Each image gets its own seed.")

                        with gr.Row():
                            seed_num    = gr.Number(value=-1, label="Seed (-1 = random)", precision=0, scale=3,
                                                    info="The seed used is saved in every PNG.")
                            with gr.Column(scale=1, min_width=130):
                                seed_reuse_btn  = gr.Button("♻️ Last seed", size="sm")
                                seed_random_btn = gr.Button("🎲 Random", size="sm")
                        with gr.Row():
                            clip_skip_rb = gr.Radio(
                                [1, 2], value=_ls.get("clip_skip") if _ls.get("clip_skip") in (1, 2) else 1,
                                label="CLIP skip",
                                info="2 = what most SD 1.5 anime checkpoints expect (A1111 “Clip skip: 2”). SDXL ignores it.")
                        from backend import polish as _polish
                        with gr.Row():
                            polish_dd = gr.Dropdown(
                                ["(off)", "Auto"] + list(_polish.SHOTS), value="(off)", scale=3,
                                label="✨ Polish — fill hires fix / face / hand / eye detail for this kind of picture",
                                info="Sets the controls below to the recipe measured for the framing (Help → ✨ Polish). "
                                     "Auto reads the framing tags in the prompt. Change anything afterwards.")
                            polish_note = gr.HTML("")
                        with gr.Accordion("🔍 Hires fix — generate small, then refine at a higher resolution",
                                          open=bool(_ls.get("hires_on"))):
                            hires_cb = gr.Checkbox(
                                label="Enable hires fix", value=bool(_ls.get("hires_on", False)),
                                info="Composes at the model's native size (no doubled bodies), upscales, then "
                                     "re-draws details at the big size. About 2–3× the time. txt2img only.")
                            _hires_ups = ["Lanczos"] + [m for m in upscaler.available_methods() if m != "Lanczos"]
                            with gr.Row():
                                hires_scale_sl = gr.Slider(1.1, 2.5, value=_ls.get("hires_scale", 1.5), step=0.05,
                                                           label="Upscale by", info="1.5× is the sweet spot on 12 GB.")
                                hires_denoise_sl = gr.Slider(0.1, 0.8, value=_ls.get("hires_denoise", 0.45), step=0.05,
                                                             label="Hires denoise",
                                                             info="0.3 keeps it · 0.45 adds detail · 0.6+ changes it.")
                            with gr.Row():
                                hires_steps_sl = gr.Slider(4, 60, value=_ls.get("hires_steps", 15), step=1,
                                                           label="Hires steps",
                                                           info="8 is enough for a full body · wide shots: 14")
                                hires_up_dd = gr.Dropdown(
                                    _hires_ups, label="Hires upscaler",
                                    value=_ls.get("hires_upscaler") if _ls.get("hires_upscaler") in _hires_ups else "Lanczos",
                                    info="Lanczos is instant; Real-ESRGAN gives sharper line art (a few seconds more).")
                        with gr.Accordion("✨ Face & hand detail — re-draw faces / hands at full resolution "
                                          "(ADetailer-style)", open=bool(_ls.get("fd_on") or _ls.get("hd_on") or _ls.get("ed_on"))):
                            fd_cb = gr.Checkbox(
                                label="Enable face detail", value=bool(_ls.get("fd_on", False)),
                                info="Re-draws each face at the model's native size: small faces in full-body or group "
                                     "shots get proper eyes and mouths · ~5–15 s per face")
                            with gr.Row():
                                fd_denoise_sl = gr.Slider(0.1, 0.8, value=_ls.get("fd_denoise", 0.4), step=0.05,
                                                          label="Face denoise",
                                                          info="0.3 = touch-up · 0.4 = fix details · 0.8 = a different face · small faces are raised "
                                                               "automatically (≤ 100 px: 0.55, ≤ 60 px: 0.65)")
                                fd_mode_rb = gr.Radio(["auto", "anime", "photo"],
                                                      value=_ls.get("fd_mode") if _ls.get("fd_mode") in
                                                      ("auto", "anime", "photo") else "auto",
                                                      label="Face detector")
                            fd_prompt_txt = gr.Textbox(label="Extra face prompt (optional)",
                                                       value=_ls.get("fd_prompt", ""),
                                                       placeholder="e.g. detailed eyes, beautiful face",
                                                       info="Added to your prompt for the face pass only.")
                            with gr.Row():
                                hd_cb = gr.Checkbox(
                                    label="✋ Also re-draw hands", value=bool(_ls.get("hd_on", False)),
                                    info="Re-draws each hand at native size, before the faces · cleans up smudged fingers, "
                                         "never fixes a finger count (use 🖐 in Inpaint) · ~5–15 s per hand")
                                hd_denoise_sl = gr.Slider(0.1, 0.7, value=_ls.get("hd_denoise", 0.35), step=0.05,
                                                          label="Hand denoise",
                                                          info="0.3–0.35 = touch-up · 0.45–0.55 = more hand-shaped, can invent things · "
                                                               "a broken hand: another seed, or 🖐 Re-draw hand in Inpaint")
                            with gr.Row():
                                ed_cb = gr.Checkbox(
                                    label="👁 Also re-draw eyes", value=bool(_ls.get("ed_on", False)),
                                    info="Re-draws both eyes of each face after the face pass; the colour stays in the iris "
                                         "(bangs and skin keep theirs) · ~5–25 s per face · skipped on Pony")
                                ed_denoise_sl = gr.Slider(0.1, 0.6, value=_ls.get("ed_denoise", 0.4), step=0.05,
                                                          label="Eye denoise",
                                                          info="0.25–0.3 = tidy up · 0.4 = more iris / pupil detail · 0.5+ = stray marks · "
                                                               "pupils need the pixels: hires / upscale first")
                        with gr.Accordion("🎚 Quality boosters — PAG, FreeU, CFG rescale",
                                          open=bool(_ls.get("pag_scale") or _ls.get("freeu"))):
                            with gr.Row():
                                pag_sl = gr.Slider(0, 6, value=_ls.get("pag_scale", 0), step=0.5, label="PAG scale",
                                                   info="Perturbed-attention guidance: cleaner structure, anatomy and "
                                                        "backgrounds. 0 = off · 2–3 recommended · ~1.5× the time.")
                                cfg_rescale_sl = gr.Slider(0, 1, value=_ls.get("cfg_rescale", 0), step=0.05,
                                                           label="CFG rescale",
                                                           info="Tames burned colours at high CFG. 0 = auto: 0.7 for "
                                                                "v-prediction models (NoobAI v-pred…), else off.")
                            freeu_cb = gr.Checkbox(label="FreeU", value=bool(_ls.get("freeu", False)),
                                                   info="Re-weights UNet features: more detail and contrast at no "
                                                        "cost; can over-saturate some anime models.")
                        with gr.Accordion("🔀 Variations — small changes to an image you like", open=False):
                            with gr.Row():
                                var_seed_num = gr.Number(value=-1, precision=0, label="Variation seed (-1 = random)")
                                var_strength_sl = gr.Slider(
                                    0, 1, value=0, step=0.01, label="Variation strength",
                                    info="0 = off · 0.05–0.1 = a close relative (pose and details shift, the character stays) · "
                                         "0.25+ = mostly a new picture · 1 = the variation seed's own image. Keep the main seed fixed.")
                        last_seed_state = gr.State([])      # seed of each image in the gallery
                        selected_idx_state = gr.State(0)    # gallery image the user clicked

                        _gen_total_mb = get_total_vram_mb()
                        _gen_vram_bar = gr.HTML(
                            vram_bar_html(
                                estimate_generate("", DEFAULT_WIDTH, DEFAULT_HEIGHT, 1),
                                _gen_total_mb,
                                "Est. VRAM",
                            )
                        )

                        # img2img toggle
                        with gr.Accordion("🖼 Image-to-Image (optional)", open=False) as i2i_acc:
                            init_image  = gr.Image(label="Input Image", type="pil")
                            strength_sl = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="Denoise Strength",
                                                    info="How much to change the input: 0.3 subtle · 0.5 restyle · 0.8+ mostly new. "
                                                         "Karras / AYS samplers change less at the same value (+0.15). "
                                                         "Recolouring an outfit: same seed + edited prompt; 🖌 Inpaint for small things (pin, expression).")
                            use_i2i_cb  = gr.Checkbox(label="Use img2img mode", value=False,
                                                      info="Start from the image above instead of pure noise.")
                            restore_cb  = gr.Checkbox(
                                label="Restore settings from dropped images", value=True,
                                info="An image made by this app (or A1111 / Forge / Civitai) brings back its "
                                     "checkpoint, VAE, LoRAs + weights, prompts, sampler, steps, CFG, size and seed.")
                            i2i_restore_html = gr.HTML("")
                            recreate_btn = gr.Button("🔁 Recreate it exactly (img2img off → Generate)",
                                                     variant="primary", size="sm", visible=False)
                            with gr.Row():
                                tag_i2i_btn = gr.Button("🏷 Interrogate — write this image's tags into the prompt "
                                                        "(WD14)", size="sm", variant="secondary", scale=3)
                                tag_thr_sl = gr.Slider(0.2, 0.8, value=0.35, step=0.05, label="Tag threshold",
                                                       scale=1, info="lower = more tags")
                            tag_i2i_html = gr.HTML("")
                        with gr.Accordion("🖌 Inpaint — paint what to change", open=False):
                            inp_editor = gr.ImageEditor(
                                label="Image — paint over the part to redraw", type="pil",
                                sources=["upload", "clipboard"], transforms=[],
                                brush=gr.Brush(default_size=40, colors=["#ffffff"], default_color="#ffffff",
                                               color_mode="fixed"),
                                eraser=gr.Eraser(default_size=40))
                            with gr.Row():
                                inp_denoise_sl = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="Inpaint denoise",
                                                           info="0.5 = adjust (expression) · 0.8 = redraw · 1.0 = ignore what's there · "
                                                                "to swap an accessory use 🦋 Swap below (0.9–0.95 leaves the old shape as lace)")
                                inp_pad_sl = gr.Slider(0, 256, value=48, step=8, label="Context padding (px)",
                                                       info="Surroundings the model sees around the painted area.")
                            gr.HTML('<p style="color:#a6adc8;font-size:13px;margin:2px 0;">Uses the prompt, '
                                    'model, LoRAs, sampler, steps, CFG and seed above. Only the painted area '
                                    'changes; it is redrawn at the model\'s native resolution, so small areas '
                                    '(hands, faces) get full detail. Describe what should be there.</p>')
                            inp_btn = gr.Button("🖌 Inpaint", variant="primary", size="sm")
                            with gr.Row():
                                inp_swap_txt = gr.Textbox(value="", label="🦋 Swap an accessory — what should be there instead?", scale=3,
                                                          placeholder="black butterfly hair ornament",
                                                          info="Paint over the whole old accessory (a hat: its crown above the head too), type the new one: "
                                                               "hair ornament, earrings, choker, hat · one pass at 1.0 · not there? another seed")
                                inp_swap_btn = gr.Button("🦋 Swap", size="sm", scale=1)
                            with gr.Row():
                                inp_hand_tries = gr.Slider(1, 8, value=4, step=1, label="🖐 Re-draw a hand — tries", scale=3,
                                                           info="Paint over the broken hand (nothing painted: the biggest detected hand) · each try has its own seed, "
                                                                "pick the best · about every other try is a correct hand, 4 tries give one in 10 of 11 · "
                                                                "a third come back changed (another glove or prop)")
                                inp_hand_btn = gr.Button("🖐 Re-draw hand", size="sm", scale=1)


                            gr.Markdown("---")
                            gr.HTML(
                                '<p style="color:#a6adc8;font-size:13px;margin:2px 0 6px;">'
                                '<b>img2img Tips:</b> Start with Denoise Strength 0.4-0.6 for refinement. '
                                'Use the prompt to steer what changes. Lower strength = closer to original. '
                                'Great for fixing anatomy, adding detail, or changing lighting on a generated image.</p>'
                            )

                            # ── img2img Enhancer Tags ──────────────────────────────
                            _i2i_btn_list: list[tuple] = []
                            with gr.Accordion("🔧 img2img Enhancer Tags", open=False):
                                gr.HTML(
                                    '<p style="color:#a6adc8;font-size:13px;margin:0 0 4px;">'
                                    'Click to append enhancement tags to your prompt.</p>'
                                )
                                for _cat, _i2i_tags in I2I_ENHANCER_TAGS.items():
                                    gr.HTML(
                                        f'<p style="font-size:13px;font-weight:bold;'
                                        f'color:#89b4fa;margin:6px 0 2px;">{_cat}</p>'
                                    )
                                    for _chunk in range(0, len(_i2i_tags), 6):
                                        with gr.Row():
                                            for _tag in _i2i_tags[_chunk:_chunk + 6]:
                                                _btn = gr.Button(_tag, size="sm")
                                                _i2i_btn_list.append((_btn, _tag))

                                gr.HTML('<hr style="margin:6px 0;opacity:0.3;">')
                                with gr.Row():
                                    danbooru_meta_btn = gr.Button(
                                        "🌐 Fetch Danbooru Quality Tags", size="sm", variant="secondary"
                                    )
                                    danbooru_general_btn = gr.Button(
                                        "🌐 Fetch Danbooru General Tags", size="sm", variant="secondary"
                                    )
                                danbooru_tags_html = gr.HTML("")

                        # ── Auto-Loop Controls ───────────────────────────────────────
                        with gr.Accordion("🔄 Auto-Loop — Continuous Generation", open=False):
                            gr.HTML(
                                '<p style="color:#a6adc8;font-size:13px;margin:0 0 6px;">'
                                'Start looping to automatically generate batch after batch with the '
                                'current settings. Each batch gets a random seed. Images are saved to '
                                '<code>outputs/</code>. Stop anytime with the Stop Loop button.</p>'
                            )
                            with gr.Row():
                                loop_max_batches = gr.Number(
                                    label="Max Batches", value=0, precision=0,
                                    info="0 = infinite (run until stopped). Otherwise stops after N batches.",
                                )
                                loop_delay = gr.Slider(
                                    0, 60, value=5, step=1, label="Delay (seconds)",
                                    info="Pause between batches. Gives GPU a breather and lets you check results.",
                                )
                            with gr.Row():
                                loop_start_btn = gr.Button("▶ Start Loop", variant="primary", size="sm")
                                loop_stop_btn  = gr.Button("⏹ Stop Loop", variant="stop", size="sm")
                            loop_status = gr.HTML("")

                        # ── X/Y comparison grid ───────────────────────────────────
                        with gr.Accordion("📊 X/Y grid — compare settings side by side", open=False):
                            gr.HTML('<p style="color:#a6adc8;font-size:13px;margin:0 0 6px;">One fixed seed, '
                                    'every combination of the values below (current settings for everything '
                                    'else). Numbers: <code>5, 7, 9</code> or ranges <code>4-10:2</code>. '
                                    '<b>Prompt S/R</b>: the first value is a word in your prompt, the others '
                                    'replace it (<code>red hair, blue hair, green hair</code>). Each cell is '
                                    'saved like a normal image; the grid is saved too.</p>')
                            with gr.Row():
                                xy_x_axis = gr.Dropdown(_XY_AXES, value="CFG", label="X axis")
                                xy_x_vals = gr.Textbox(value="5, 7, 9", label="X values")
                            with gr.Row():
                                xy_y_axis = gr.Dropdown(_XY_AXES, value="none", label="Y axis")
                                xy_y_vals = gr.Textbox(value="", label="Y values")
                            xy_btn = gr.Button("📊 Generate grid", variant="primary", size="sm")

                        # ── Quick Reference Guide ──────────────────────────────────
                        with gr.Accordion("📖 Settings Guide — How to Get Good Images", open=False):
                            gr.HTML('''
        <div style="font-size:13px; color:#cdd6f4; line-height:1.6;">
        <h4 style="color:#cba6f7; margin:0 0 8px;">🎯 Character LoRA Cheat Sheet</h4>
        <table style="width:100%; border-collapse:collapse; font-size:13px; margin-bottom:12px;">
        <tr style="background:#313244;"><th style="padding:4px 8px; text-align:left;">Setting</th><th style="padding:4px 8px;">Recommended</th><th style="padding:4px 8px;">Notes</th></tr>
        <tr><td style="padding:4px 8px;">Character LoRA weight</td><td style="padding:4px 8px; text-align:center;"><b>0.70 – 0.85</b></td><td style="padding:4px 8px;">Higher = more faithful but may distort</td></tr>
        <tr style="background:#313244;"><td style="padding:4px 8px;">Style/Effect LoRA weight</td><td style="padding:4px 8px; text-align:center;"><b>0.40 – 0.65</b></td><td style="padding:4px 8px;">Keep LOWER than character to preserve likeness</td></tr>
        <tr><td style="padding:4px 8px;">CFG Scale (Pony/IL)</td><td style="padding:4px 8px; text-align:center;"><b>5.5 – 7.0</b></td><td style="padding:4px 8px;">Lower than SD 1.5. Over 8 = oversaturated</td></tr>
        <tr style="background:#313244;"><td style="padding:4px 8px;">CFG Scale (SD 1.5)</td><td style="padding:4px 8px; text-align:center;"><b>7.0 – 9.0</b></td><td style="padding:4px 8px;">SD 1.5 needs higher CFG than SDXL</td></tr>
        <tr><td style="padding:4px 8px;">Steps</td><td style="padding:4px 8px; text-align:center;"><b>AYS 10–12 · Karras 20–30</b></td><td style="padding:4px 8px;">More ≠ better: AYS 12 matches 25 Karras steps</td></tr>
        <tr style="background:#313244;"><td style="padding:4px 8px;">Resolution (SDXL portrait)</td><td style="padding:4px 8px; text-align:center;"><b>832×1216</b></td><td style="padding:4px 8px;">Or 768×1344 for taller composition</td></tr>
        <tr style="background:#313244;"><td style="padding:4px 8px;">Sampler</td><td style="padding:4px 8px; text-align:center;"><b>DPM++ 2M Karras</b></td><td style="padding:4px 8px;">Euler a = more variety · DDIM = consistent img2img · UniPC = fewer steps</td></tr>
<tr><td style="padding:4px 8px;">img2img denoise</td><td style="padding:4px 8px; text-align:center;"><b>0.4 – 0.6</b></td><td style="padding:4px 8px;">0.2–0.4 fixes details · 0.5–0.7 restyles · 0.8+ mostly regenerates</td></tr>
<tr style="background:#313244;"><td style="padding:4px 8px;">Seed</td><td style="padding:4px 8px; text-align:center;"><b>-1 → ♻️</b></td><td style="padding:4px 8px;">Explore with random seeds, then ♻️ reuse the one you like and tweak the prompt</td></tr>
<tr><td style="padding:4px 8px;">Resolution (SD 1.5 portrait)</td><td style="padding:4px 8px; text-align:center;"><b>512×768</b></td><td style="padding:4px 8px;">Never exceed 768px on SD 1.5</td></tr>
        </table>

        <h4 style="color:#cba6f7; margin:0 0 8px;">📝 Prompt Structure (order matters!)</h4>
        <ol style="margin:0 0 12px; padding-left:20px;">
        <li><b>Quality tags</b> — <code>score_9, score_8_up, masterpiece, best quality</code> (Pony/IL only)</li>
        <li><b>Character trigger</b> — <code>my_character</code> or whatever the LoRA specifies</li>
        <li><b>Character features</b> — <code>red eyes, grey hair, long hair, french_braid</code></li>
        <li><b>Subject/pose</b> — <code>1girl, solo, full body, standing, looking at viewer</code></li>
        <li><b>Scene/action</b> — <code>standing in a park, sunset lighting</code></li>
        <li><b>Style/lighting</b> — <code>cinematic lighting, vibrant colors, sharp focus</code></li>
        </ol>

        <h4 style="color:#cba6f7; margin:0 0 8px;">⚠️ Common Mistakes</h4>
        <ul style="margin:0; padding-left:20px;">
        <li><b>LoRA weight too high (>1.0)</b> → Distorted face, burned colors, artifacts</li>
        <li><b>CFG too high (>10 on Pony)</b> → Oversaturated, "deep-fried" look</li>
        <li><b>Too many LoRAs at high weights</b> → Muddy, conflicting styles. Lower slots 2-3 first</li>
        <li><b>Wrong trigger words</b> → Character not recognized. Check Civitai page for exact triggers</li>
        <li><b>(tag: 2.0) emphasis</b> → Too strong! Use 1.1-1.3 max for emphasis</li>
        </ul>
        </div>
        ''')

                        # ── Prompt Presets ──────────────────────────────────────────
                        with gr.Accordion("💡 Prompt Presets", open=False):
                            gr.HTML(
                                '<p style="color:#a6adc8;font-size:13px;margin:0 0 6px;">'
                                'Select a preset and click Replace or Append. '
                                'Presets are quality boosters for each model family.</p>'
                            )
                            preset_dd = gr.Dropdown(
                                label="Preset",
                                choices=list(_PROMPT_PRESETS.keys()),
                                value=list(_PROMPT_PRESETS.keys())[0],
                                interactive=True,
                            )
                            with gr.Row():
                                preset_replace_btn = gr.Button("↩ Replace prompts", size="sm")
                                preset_append_btn  = gr.Button("＋ Append to prompts", size="sm")

                        # ── My Saved Prompts ────────────────────────────────────────
                        with gr.Accordion("📁 My Saved Prompts", open=False):
                            gr.HTML(
                                '<p style="color:#a6adc8;font-size:13px;margin:0 0 6px;">'
                                'Save / load your own positive prompts. '
                                'Presets are stored separately for SD 1.5 and XL/Pony/IL.</p>'
                            )
                            user_preset_dd = gr.Dropdown(
                                label="Saved Prompts",
                                choices=_list_user_presets(),
                                interactive=True,
                            )
                            with gr.Row():
                                user_load_btn = gr.Button("📂 Load", size="sm")
                                user_delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop")
                            with gr.Row():
                                user_preset_name = gr.Textbox(
                                    label="Preset name",
                                    placeholder="e.g. Cozy portrait",
                                    scale=3,
                                )
                                user_save_btn = gr.Button("💾 Save Prompt", size="sm", scale=1)
                            user_preset_status = gr.HTML("")

                        # ── Save / Load Generation Settings ─────────────────────────
                        with gr.Accordion("⚙️ Save / Load Settings", open=False):
                            gr.HTML(
                                '<p style="color:#a6adc8;font-size:13px;margin:0 0 6px;">'
                                'Saves the checkpoint, VAE, LoRA slots and weights, prompts, sampler, steps, '
                                'CFG, size, seed, hires fix, face detail and boosters — and the image selected '
                                'in the gallery as its example.</p>'
                            )
                            settings_dd = gr.Dropdown(
                                label="Saved Settings",
                                choices=_list_saved_settings(),
                                interactive=True,
                            )
                            settings_preview = gr.Image(label="Made with these settings", height=260,
                                                        interactive=False, visible=False)
                            with gr.Row():
                                settings_load_btn = gr.Button("📂 Load Settings", size="sm")
                            with gr.Row():
                                settings_name_txt = gr.Textbox(
                                    label="Settings name",
                                    placeholder="e.g. my_portrait_setup",
                                    scale=3,
                                )
                                settings_save_btn = gr.Button("💾 Save Settings", size="sm", scale=1)
                            settings_status = gr.HTML("")

                        # ── Quick Tag Bar ────────────────────────────────────────────
                        _tag_btn_list: list[tuple] = []   # (button, tag_text) — wired below
                        with gr.Accordion("🏷️ Quick Tags", open=False):
                            gr.HTML(
                                '<p style="color:#a6adc8;font-size:13px;margin:0 0 4px;">'
                                'Click any tag to insert it into the selected prompt box. '
                                'Toggle the target with the radio below.</p>'
                            )
                            tag_target = gr.Radio(
                                ["➕ Positive", "➖ Negative"],
                                value="➕ Positive",
                                label="Insert into",
                                interactive=True,
                            )
                            for _cat_name, _tags in _TAG_CATEGORIES.items():
                                gr.HTML(
                                    f'<p style="font-size:13px;font-weight:bold;'
                                    f'color:#cba6f7;margin:6px 0 2px;">{_cat_name}</p>'
                                )
                                for _chunk_start in range(0, len(_tags), 8):
                                    with gr.Row():
                                        for _tag in _tags[_chunk_start:_chunk_start + 8]:
                                            _btn = gr.Button(_tag, size="sm")
                                            _tag_btn_list.append((_btn, _tag))

                        # ── SmartSplit status banner (shown when active) ─────────────
                        smartsplit_banner = gr.HTML(_build_smartsplit_banner())

                    # ── Output column (next to the controls, no scrolling) ────
                    with gr.Column(scale=4, min_width=360, elem_id="gen-output-col"):
                        output_gallery = gr.Gallery(
                            label="Output",
                            columns=3,
                            object_fit="contain",
                            height=640,
                            preview=True,
                            show_download_button=True,
                        )
                        last_generated_images = gr.State([])
                        selected_gallery_image = gr.State(None)

                        with gr.Row():
                            gal_send_i2i_btn = gr.Button("🖼 Send to img2img", size="sm")
                            gal_send_up_btn  = gr.Button("🔍 Send to Upscale", size="sm")
                            open_outputs_btn = gr.Button("📂 Show in folder", size="sm")
                            gal_more_btn = gr.Button("🔀 More like this", size="sm")
                        gal_status = gr.HTML("")

    # ── Event wiring ──────────────────────────────────────────────────────────
    def do_load_model(model_path, vae_path, _from_generate=False):
        global sd, _smartsplit_pipe, _smartsplit_cfg
        disk = _hf_disk_problem(model_path)
        if disk:
            return f'<p style="color:#f38ba8;">❌ {html.escape(disk)}</p>', gr.update()
        from backend.sd_pipeline import _is_sdxl

        # Detect model type BEFORE loading
        is_xl = _is_sdxl(model_path)

        # Swap the active pipeline to the correct type
        if is_xl:
            if not isinstance(sd, SDXLPipeline):
                # Switching from SD 1.5 → SDXL: unload SD 1.5 pipeline
                _sd15._unload()
                sd = _sdxl
        else:
            if isinstance(sd, SDXLPipeline):
                # Switching from SDXL → SD 1.5: unload SDXL pipeline
                _sdxl._unload()
                sd = _sd15

        # Snapshot active LoRAs — sd._unload() clears them during model reload
        saved_loras = dict(sd._lora_adapters)
        # Preload button: abort any in-progress generation to avoid racing the reload.
        # When Generate itself loads the model, leave the flag alone — resetting it
        # here used to swallow a Stop pressed while the model was loading.
        if not _from_generate:
            _generation_abort.set()
        vae, vae_warn = _effective_vae(model_path, vae_path)
        msg = sd.load_model(model_path, vae)
        if not _from_generate:
            _generation_abort.clear()

        # Auto-rebuild SmartSplit pipe — ONLY for SD 1.5 models
        if _smartsplit_cfg.enabled and sd.pipe is not None:
            if is_xl:
                _smartsplit_pipe = None
                msg += " ⚠️ SmartSplit disabled (SDXL not supported, using standard path)"
            else:
                try:
                    from pathlib import Path as _P
                    stem = _P(model_path).stem if model_path else "model"
                    _smartsplit_pipe = SmartSplitPipeline(sd.pipe, _smartsplit_cfg, stem)
                except Exception as e:
                    print(f"[SmartSplit] Failed to rebuild pipeline: {e}")
                    _smartsplit_pipe = None

        # Re-apply LoRAs that were active before the switch
        lora_status = ""
        if saved_loras and sd.pipe is not None:
            sd._lora_adapters = saved_loras
            try:
                sd._reload_all_loras()
                names = [v[0] for v in saved_loras.values()]
                lora_status = f"<br>🔁 Re-applied LoRAs: {', '.join(names)}"
            except Exception as e:
                print(f"[LoRA] Re-apply failed after model switch: {e}")
                sd._lora_adapters = {}
                lora_status = "<br>⚠️ Previous LoRAs dropped (incompatible with new model)"

        # Resolution / quality tag hint based on model family (none when the load failed or was refused: the family
        # would still be the previous model's)
        family = sd.model_family if sd.pipe is not None else None
        res_hint = ""
        if family == "pony":
            res_hint = (
                "<br>🐴 <b>Pony model detected</b> — use <b>1024×1024</b> or <b>832×1216</b>. "
                "Quality tags: <code>score_9, score_8_up, score_7_up</code>"
            )
        elif family == "illustrious":
            res_hint = (
                "<br>🎨 <b>Illustrious model detected</b> — use <b>1024×1024</b> or <b>832×1216</b>. "
                "Quality tags: <code>masterpiece, best quality</code>"
            )
        elif family == "sdxl":
            res_hint = (
                "<br>📐 <b>SDXL model detected</b> — use <b>1024×1024</b> or <b>832×1216</b>. "
                "512px will produce low-quality output."
            )
        elif family == "sd15":
            res_hint = "<br>📐 SD 1.5 — use <b>512×512</b> or <b>512×768</b>."

        color = "#a6e3a1" if "✅" in msg else "#f38ba8"
        status_html = f'<p style="color:{color};">{msg}{lora_status}{res_hint}{vae_warn}</p>'
        # Refresh user preset dropdown for the new model family
        return status_html, gr.update(choices=_list_user_presets())

    def _active_loras_html() -> str:
        if not sd.loaded_loras:
            return ""
        badges = "".join(
            f'<span style="background:#a6e3a1;color:#1e1e2e;padding:2px 8px;'
            f'border-radius:9px;margin:2px;font-size:13px;">Slot {i+1}: {n}</span>'
            for i, (_, v) in enumerate(sorted(sd._lora_adapters.items()))
            for n in [v[0]]
        )
        return f'<p style="margin:4px 0;">{badges}</p>'

    def do_apply_lora(lora_path, weight, slot=0):
        if not lora_path or lora_path == "none":
            return '<p style="color:#f38ba8;">Select a LoRA first.</p>', _active_loras_html()
        bad = _lora_mismatch(lora_path)
        if bad:
            return f'<p style="color:#f38ba8;">❌ {bad}</p>', _active_loras_html()
        msg = sd.load_lora(lora_path, weight, slot=slot)
        color = "#a6e3a1" if "✅" in msg else "#f38ba8"
        return f'<p style="color:{color};">{msg}</p>', _active_loras_html()

    def do_remove_lora(slot):
        msg = sd.remove_lora(slot)
        return f'<p style="color:#a6e3a1;">{msg}</p>', _active_loras_html()

    def do_remove_loras():
        msg = sd.unload_loras()
        return f'<p style="color:#a6e3a1;">{msg}</p>', _active_loras_html()

    def do_stop():
        _autoloop_active.clear()
        _generation_abort.set()
        return ('<p style="color:#fab387;">⏹ Stopped. (If a model was loading, it finishes loading '
                'and stays ready; a running generation ends after its current step.)</p>')

    def _detail_pass(kind, imgs, seeds, prompt, neg_prompt, steps, cfg, scheduler, ex, progress):
        """Face, hand or eye detail on every image (same seed per image, low denoise)."""
        import math
        from backend.detail_tools import face_detail, hand_detail, eye_detail, detail_schedule_steps
        face = kind == "face"
        den = {"face": ex["fd_denoise"], "hand": ex["hd_denoise"], "eye": ex["ed_denoise"]}[kind]
        label, icon, what = {"face": ("Face", "✨", "face"), "hand": ("Hand", "✋", "hand"),
                             "eye": ("Eye", "👁", "eye")}[kind]
        t0, out, found, errors = time.time(), [], [], []
        # ~half the main steps actually run (ADetailer runs steps × denoise). SDXL A/B at 1024²:
        # 0.8× (~22 steps) ~45 s per face, 0.5× (~14) ~28 s, faces no worse
        fsteps = detail_schedule_steps(kind, steps, den)              # the eye pass runs 8 steps instead of 10 (round 8: equal quality, 21 % cheaper)
        for i, im in enumerate(imgs):
            if _generation_abort.is_set():
                raise _GenerationAborted()
            progress(0, desc=f"{label} detail {i + 1}/{len(imgs)}: finding {what}s…")

            def cb(step, total, i=i):
                if _generation_abort.is_set():
                    raise _GenerationAborted()
                progress(step / total, desc=f"{label} detail {i + 1}/{len(imgs)}: step {step}/{total}")
            try:
                kw = dict(denoise=den, steps=fsteps, cfg=cfg, seed=seeds[i] if i < len(seeds) else seeds[-1],
                          scheduler=scheduler, clip_skip=ex["clip_skip"], step_callback=cb)
                if face:
                    res, n = face_detail(sd, im, prompt, neg_prompt, mode=ex["fd_mode"], face_prompt=ex["fd_prompt"],
                                         **kw)
                elif kind == "eye":
                    res, n = eye_detail(sd, im, prompt, neg_prompt, **kw)
                else:
                    res, n = hand_detail(sd, im, prompt, neg_prompt, **kw)
            except _GenerationAborted:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                errors.append(str(e).splitlines()[0][:160] if str(e) else type(e).__name__)
                res, n = im, 0
            if hasattr(im, "info"):
                res.info = dict(im.info)
            out.append(res); found.append(n)
        note = (f'<br>{icon} {label} detail: {sum(found)} {what}(s) re-drawn '
                f'({", ".join(str(n) for n in found)}) · denoise {den:g} · {time.time() - t0:.1f}s'
                if sum(found) else f'<br>{icon} {label} detail: no {what}s found' if not errors else '')
        if kind == "hand" and not sum(found) and not errors:
            from backend.detail_tools import hand_detector_available
            if not hand_detector_available():
                note = "<br>✋ Hand detail skipped: the hand detector couldn't be downloaded (retried in 5 min)."
        if kind == "eye" and not sum(found) and not errors:
            from backend.detail_tools import eye_detector_available
            if not eye_detector_available():
                note = "<br>👁 Eye detail skipped: the eye detector couldn't be loaded (retried in 5 min)."
        if errors:   # it used to say "no faces found" when the pass had failed
            note += (f'<br><span style="color:#f38ba8;">{icon} {label} detail failed on {len(errors)} image(s): '
                     f'{html.escape(errors[0])}</span>')
        return out, note

    def _hires_pass(imgs, seeds, prompt, neg_prompt, cfg, scheduler, ex, progress):
        """Hires fix: upscale each first-pass image and re-draw it with img2img at that size
        (same seed). SD 1.5 is capped at 1536 px and SDXL at 2048 px on the long side."""
        import math
        t0 = time.time()
        xl = sd.model_family in ("sdxl", "pony", "illustrious")
        cap = 2048 if xl else 1536
        steps2 = min(150, math.ceil(ex["hires_steps"] / ex["hires_denoise"]))   # img2img runs steps × denoise
        out, size, ups = [], None, []
        for i, im in enumerate(imgs):
            w, h = im.size
            tw, th = w * ex["hires_scale"], h * ex["hires_scale"]
            f = min(1.0, cap / max(tw, th))
            tw, th = int(round(tw * f / 8)) * 8, int(round(th * f / 8)) * 8
            size = (w, h, tw, th)
            if _generation_abort.is_set():      # Stop between phases, not only inside the sampler
                raise _GenerationAborted()
            progress(0, desc=f"Hires fix {i + 1}/{len(imgs)}: upscaling to {tw}×{th}…")
            ups.append(_upscale_to(im, tw, th, ex["hires_upscaler"]))
        # all upscales first, then free Real-ESRGAN's DirectML session before the big img2img passes
        if ex["hires_upscaler"] != "Lanczos":
            upscaler.release()
        for i, up in enumerate(ups):
            if _generation_abort.is_set():
                raise _GenerationAborted()

            def cb(step, total, i=i):
                if _generation_abort.is_set():
                    raise _GenerationAborted()
                # img2img runs int(total × denoise) steps and diffusers numbers them from 1 within
                # that slice (the old "step − (total − done)" stayed at 0 for the whole pass)
                done = max(1, int(total * ex["hires_denoise"]))
                k = min(max(0, step), done)
                progress(k / done, desc=f"Hires fix {i + 1}/{len(imgs)}: step {k}/{done}")
            r, _ = sd.img2img(up, prompt, neg_prompt, ex["hires_denoise"], steps2, cfg,
                              seeds[i] if i < len(seeds) else seeds[-1], scheduler,
                              step_callback=cb, clip_skip=ex["clip_skip"])
            out.append(r[0] if r else up)
        w, h, tw, th = size
        note = (f'<br>🔍 Hires fix: {w}×{h} → {tw}×{th} · denoise {ex["hires_denoise"]:g} · '
                f'{ex["hires_steps"]} steps · {html.escape(ex["hires_upscaler"])} · {time.time() - t0:.1f}s')
        return out, note

    def do_generate(
        prompt, neg_prompt, scheduler, steps, cfg, width, height, batch,
        seed, init_img, strength, use_i2i, auto_quality=True, extra=None, progress=gr.Progress(),
        template=None,
    ):
        global _smartsplit_pipe, _smartsplit_cfg
        # (the abort flag is cleared by the caller when the run starts — clearing it
        #  here would discard a Stop pressed while the model was loading)
        if _generation_abort.is_set():
            return [], '<p style="color:#fab387;">⏹ Generation stopped.</p>', []
        from backend.sampling import reset_power_samples
        reset_power_samples()                          # the "looks power-limited" hint is about this picture's passes
        cool_wait = [_thermal_wait(progress)]          # Settings → 🌡 pause while hot (off unless set)
        if _generation_abort.is_set():
            return [], '<p style="color:#fab387;">⏹ Generation stopped.</p>', []
        steps, cfg, width, height, batch, seed, strength, fixes = _clean_gen_args(
            steps, cfg, width, height, batch, seed, strength, img2img=bool(use_i2i and init_img is not None))
        if use_i2i and init_img is None:
            fixes.append("img2img is on but no image is loaded — made a new image from the prompt instead")
        prompt, neg_prompt = prompt or "", neg_prompt or ""
        ex = _clean_extra(extra)
        # PAG / FreeU / CFG rescale are read by backend.sampling.run_pipe for every pass
        # (txt2img, hires, face detail) of this run; cfg_rescale 0 = auto (0.7 for v-pred)
        sd.boosters = {"pag": ex["pag_scale"], "freeu": ex["freeu"], "cfg_rescale": ex["cfg_rescale"] or None}
        from backend import wildcards as _wc
        if template is None and (_wc.is_dynamic(prompt) or _wc.is_dynamic(neg_prompt)):
            # Dynamic prompt: each image gets its own picks (from its own seed), one at a time
            if seed < 0:
                import random
                seed = random.randint(0, 2**32 - 1 - batch)
            n = 1 if (use_i2i and init_img is not None) else batch
            all_imgs, infos, seeds_used, missing = [], [], [], []
            for i in range(n):
                s_i = (seed + i) % 2**32
                p_i = _wc.resolve(prompt, s_i, missing)
                n_i = _wc.resolve(neg_prompt, s_i, missing)
                ex_i = dict(ex, var_seed=(ex["var_seed"] + i) % 2**32 if ex["var_seed"] >= 0 else -1)
                imgs_i, info_i, _ = do_generate(
                    p_i, n_i, scheduler, steps, cfg, width, height, 1, s_i, init_img, strength, use_i2i,
                    auto_quality=auto_quality, extra=ex_i, progress=progress,
                    template={"prompt": prompt, "negative": neg_prompt})
                if not imgs_i:          # stopped or failed: report it and stop the batch
                    infos.append(info_i)
                    break
                all_imgs += imgs_i
                seeds_used += list(getattr(sd, "last_seeds", None) or [s_i])
                infos.append(f'<p style="color:#cba6f7;font-size:13px;margin:2px 0;">🎲 #{i + 1}: '
                             f'<code>{html.escape(p_i)}</code></p>' + info_i)
            sd.last_seeds = seeds_used
            sd.boosters = {}
            if missing:
                infos.insert(0, '<p style="color:#f9e2af;font-size:13px;">⚠ Unknown wildcard(s): '
                             + ", ".join(f"<code>__{html.escape(m)}__</code>" for m in missing)
                             + " — add a .txt file to ImageGenApp/wildcards/ or models/wildcards/</p>")
            return all_imgs, "".join(infos), all_imgs
        if use_i2i and init_img is not None:
            # SD 1.5 attention at 2048 px needs ~4 GB per head slice and makes nothing better
            xl = sd.model_family in ("sdxl", "pony", "illustrious")
            init_img, img_fix = _fit_init_image(init_img, max_side=2048 if xl else 1280)
            fixes += img_fix
        fixes_note = (f'<br><span style="color:#f9e2af;">Adjusted: {"; ".join(fixes)}</span>'
                      if fixes else "")
        if scheduler in _SDXL_BAD_SAMPLERS and sd.model_family in _SDXL_FAMILIES:
            fixes_note += (f'<br><span style="color:#fab387;">ℹ {scheduler} leaves a little more grain in skies and '
                           f'backgrounds than DPM++ 2M AYS / Karras on SDXL models.</span>')

        # ── Optionally add the quality tags each model family was trained with ─
        added_tags = added_neg = ""
        if auto_quality:
            added_tags, added_neg = _family_quality(sd.model_family, prompt, neg_prompt)
            if added_tags:
                prompt = merge_prompts(added_tags, prompt)
            if added_neg:
                neg_prompt = merge_prompts(neg_prompt, added_neg)
        tags_note = ((f'<br><span style="color:#9399b2;">Auto-added quality tags: <code>{added_tags}</code>'
                      + (f' · negative: <code>{added_neg}</code>' if added_neg else "") + '</span>')
                     if (added_tags or added_neg) else "") + fixes_note

        try:
            # ── SmartSplit path ────────────────────────────────────────────────
            if _smartsplit_cfg.enabled and _smartsplit_pipe is not None and not use_i2i:
                from backend.sd_pipeline import _load_scheduler
                _load_scheduler(_smartsplit_pipe.pipe, scheduler)
                if seed < 0:           # pick it here so the saved metadata has the real seed
                    import random
                    seed = random.randint(0, 2**32 - 1)

                def prog_cb(pct, msg):
                    if _generation_abort.is_set():
                        raise _GenerationAborted()
                    progress(pct, desc=msg)

                imgs, timing = _smartsplit_pipe.generate(
                    prompt=prompt, negative_prompt=neg_prompt,
                    width=width, height=height, steps=steps, cfg_scale=cfg,
                    seed=seed, scheduler_fn=None, batch_size=batch,
                    progress_callback=prog_cb,
                )
                _save_outputs(imgs, dict(
                    mode="txt2img", prompt=prompt, negative_prompt=neg_prompt,
                    steps=steps, cfg_scale=cfg, seeds=[seed], scheduler=scheduler,
                    width=width, height=height,
                    **({"prompt_template": template["prompt"],
                        "negative_template": template["negative"] or None} if template else {}),
                ), pipe=sd)
                # SmartSplit runs plain txt2img only: say which settings it didn't apply (audit F-27)
                skipped = [lbl for lbl, on in (("hires fix", ex["hires_on"]), ("face detail", ex["fd_on"]),
                                                ("hand detail", ex["hd_on"]), ("eye detail", ex["ed_on"]),
                                                ("PAG", ex["pag_scale"] > 0),
                                                ("FreeU", ex["freeu"]), ("CFG rescale", ex["cfg_rescale"] > 0),
                                                ("CLIP skip", ex["clip_skip"] > 1),
                                                ("variation seed", ex["var_strength"] > 0)) if on]
                skip_note = (f'<br><span style="color:#fab387;">SmartSplit does not apply: {", ".join(skipped)} — '
                             f'switch SmartSplit off in Settings to use them.</span>' if skipped else "")
                info_html = (
                    f'<p style="color:#a6adc8;font-size:13px;">'
                    f'SmartSplit | Seed:{seed} | Steps:{steps} | {width}×{height}<br>'
                    f'{_smartsplit_pipe.timing_html(timing)}{skip_note}{tags_note}</p>'
                )
                return imgs or [], info_html, imgs or []

            # ── Standard single-GPU path ───────────────────────────────────────
            progress(0, desc="Encoding prompts…")

            def _std_progress(step, total):
                if _generation_abort.is_set():
                    raise _GenerationAborted()
                progress(step / total, desc=f"Step {step}/{total}")

            if use_i2i and init_img is not None:
                imgs, info = sd.img2img(
                    init_img, prompt, neg_prompt, strength, steps, cfg, seed, scheduler,
                    step_callback=_std_progress, clip_skip=ex["clip_skip"],
                )
            else:
                imgs, info = sd.txt2img(
                    prompt, neg_prompt, width, height, steps, cfg, seed, scheduler, batch,
                    step_callback=_std_progress, clip_skip=ex["clip_skip"],
                    var_seed=ex["var_seed"], var_strength=ex["var_strength"],
                )
            seeds = list(getattr(sd, "last_seeds", None) or [seed])
            var_seeds = list(getattr(sd, "last_var_seeds", None) or []) if ex["var_strength"] > 0 else []
            hires_note = fd_note = hd_note = ed_note = ""
            # Stop / an error during a later pass keeps the images of the last finished stage (they
            # used to be thrown away with the whole run — a 25 s base image lost to a hires OOM).
            halted = ""

            def _post_pass(label, run):
                nonlocal imgs, halted
                try:
                    cool_wait[0] += _thermal_wait(progress)
                    if _generation_abort.is_set():
                        raise _GenerationAborted()
                    imgs, note = run()
                    return note
                except _GenerationAborted:
                    halted = f"⏹ Stopped during {label}"
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    if "out of memory" in str(e).lower():
                        import gc, torch
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    halted = f"❌ {label} failed ({html.escape(str(e).splitlines()[0][:140] if str(e) else type(e).__name__)})"
                finally:
                    sd.last_seeds = seeds        # the img2img passes overwrite them
                return ""

            if ex["hires_on"] and imgs and not (use_i2i and init_img is not None):
                # PAG shapes the composition in the first pass; in the hires pass it only cost time
                # (1248×1824: ~130 s instead of ~75) and VRAM (peak 10.9 of 12 GB, batch 3)
                _b = dict(getattr(sd, "boosters", None) or {})
                sd.boosters = dict(_b, pag=0)
                try:
                    hires_note = _post_pass("hires fix", lambda: _hires_pass(
                        imgs, seeds, prompt, neg_prompt, cfg, scheduler, ex, progress))
                finally:
                    sd.boosters = _b
            # hands first: a hand next to the face (peace sign, chin rest) would otherwise be
            # repainted over the finished face's edge
            if ex["hd_on"] and imgs and not halted:
                hd_note = _post_pass("hand detail", lambda: _detail_pass(
                    "hand", imgs, seeds, prompt, neg_prompt, steps, cfg, scheduler, ex, progress))
            if ex["fd_on"] and imgs and not halted:
                fd_note = _post_pass("face detail", lambda: _detail_pass(
                    "face", imgs, seeds, prompt, neg_prompt, steps, cfg, scheduler, ex, progress))
            # eyes last: the face pass would otherwise redraw them again at its lower resolution
            ed_skip = ""
            if ex["ed_on"] and imgs and not halted and _eye_pass_skipped(sd):
                # round 5: the eye pass softened the Pony checkpoint by 28 % at every denoise (4 bases) — skip, and say so
                ed_skip = '<br>👁 Eye detail skipped: it blurs Pony-family models (measured −28 % eye detail).'
            elif ex["ed_on"] and imgs and not halted:
                ed_note = _post_pass("eye detail", lambda: _detail_pass(
                    "eye", imgs, seeds, prompt, neg_prompt, steps, cfg, scheduler, ex, progress))
            halt_note = (f'<br><span style="color:#fab387;">{halted} — saved the images from before it.</span>'
                         if halted else "")
            info_html = (f'<p style="color:#a6adc8;font-size:13px;">{info}{hires_note}{hd_note}{fd_note}{ed_note}{ed_skip}'
                         f'{"<br>🌡 Paused " + _fmt_elapsed(cool_wait[0]) + " to let the laptop cool" if cool_wait[0] >= 1 else ""}'
                         f'{halt_note}{tags_note}</p>')
            if use_i2i and init_img is not None and imgs:
                width, height = imgs[0].size   # img2img keeps the input's size
            i2i = bool(use_i2i and init_img is not None)
            src = (getattr(init_img, "info", None) or {}).get("saved_path") if i2i else None
            from backend.sampling import ran_as
            saved = _save_outputs(imgs, dict(
                mode="img2img" if i2i else "txt2img",
                prompt=prompt, negative_prompt=neg_prompt,
                steps=steps, cfg_scale=cfg, seeds=seeds, scheduler=ran_as(scheduler),
                width=width, height=height,
                clip_skip=ex["clip_skip"] if ex["clip_skip"] > 1 else None,
                **({"var_seeds": var_seeds, "var_strength": ex["var_strength"]} if var_seeds else {}),
                **({"hires": {"scale": ex["hires_scale"], "denoise": ex["hires_denoise"],
                              "steps": ex["hires_steps"], "upscaler": ex["hires_upscaler"]}}
                   if hires_note else {}),
                **({"face_detail": {"denoise": ex["fd_denoise"], "detector": ex["fd_mode"],
                                    "prompt": ex["fd_prompt"]}} if fd_note else {}),
                **({"hand_detail": {"denoise": ex["hd_denoise"]}} if hd_note else {}),
                **({"eye_detail": {"denoise": ex["ed_denoise"], **({"style": "natural"} if _eye_style_now() == "natural" else {})}} if ed_note else {}),
                **({"strength": strength, "source_image": Path(src).name if src else None} if i2i else {}),
                **({"prompt_template": template["prompt"],
                    "negative_template": template["negative"] or None} if template else {}),
                **_booster_record(sd, ex),
            ), pipe=sd)
            if saved:
                info_html += (f'<p style="color:#9399b2;font-size:13px;margin:2px 0;">'
                              f'Saved: {", ".join(p.name for p in saved)}</p>')
            try:
                import torch as _t
                if "cuda" in str(sd.device) and _t.cuda.is_available():
                    dev = _t.device(sd.device)
                    total_gb = _t.cuda.get_device_properties(dev).total_memory / 2**30
                    alloc_gb = _t.cuda.memory_allocated(dev) / 2**30
                    peak_gb  = _t.cuda.max_memory_reserved(dev) / 2**30
                    info_html += (
                        f'<p style="color:#9399b2;font-size:13px;margin:2px 0;">'
                        f'VRAM: {alloc_gb:.1f} GB held by the model · peak {peak_gb:.1f} of '
                        f'{total_gb:.1f} GB during this run</p>'
                    )
            except Exception as e:
                print(f"[VRAM] could not read memory stats: {e}")
            try:
                from backend.sampling import power_hint_html
                info_html += power_hint_html()         # a hint only (cool mode on + slow steps): Settings → Power profile
            except Exception as e:
                print(f"[Power] hint failed: {e}")
            return imgs or [], info_html, imgs or []

        except _GenerationAborted:
            return [], '<p style="color:#fab387;">⏹ Generation stopped.</p>', []
        except Exception as e:
            err = str(e)
            if "out of memory" in err.lower():
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return [], '<p style="color:#f38ba8;font-size:13px;">❌ GPU Out of Memory! Try lowering resolution or batch size.</p>', []
            return [], f'<p style="color:#f38ba8;font-size:13px;">❌ Generation failed: {err}</p>', []
        finally:
            # The abort flag is deliberately NOT cleared here: the outfit batch and the X/Y grid check it
            # after every image, and a Stop that landed in a post pass (hires / hand / face — minutes on
            # SDXL) used to disappear with the image's `finally`, so the batch carried on. Every top-level
            # run clears it when it starts.
            sd.boosters = {}      # the Bridge / Inpaint buttons must not inherit this run's PAG / FreeU

    # ── Auto-Loop generator (yields after each batch) ────────────────────────
    def do_autoloop(
        model_path, vae_path, lora1, w1, lora2, w2, lora3, w3, auto_quality,
        prompt, neg_prompt, scheduler, steps, cfg, width, height, batch,
        seed_input, init_img, strength, use_i2i, max_batches, delay,
        clip_skip=1, var_seed=-1, var_strength=0, hires_on=False, hires_scale=1.5,
        hires_denoise=0.45, hires_steps=15, hires_upscaler="Lanczos",
        fd_on=False, fd_denoise=0.4, fd_mode="auto", fd_prompt="",
        pag_scale=0.0, freeu=False, cfg_rescale=0.0, hd_on=False, hd_denoise=0.35, ed_on=False, ed_denoise=0.4,
    ):
        extra = _clean_extra(dict(clip_skip=clip_skip, var_seed=var_seed, var_strength=var_strength,
                                  hires_on=hires_on, hires_scale=hires_scale, hires_denoise=hires_denoise,
                                  hires_steps=hires_steps, hires_upscaler=hires_upscaler, fd_on=fd_on,
                                  fd_denoise=fd_denoise, fd_mode=fd_mode, fd_prompt=fd_prompt,
                                  pag_scale=pag_scale, freeu=freeu, cfg_rescale=cfg_rescale,
                                  hd_on=hd_on, hd_denoise=hd_denoise, ed_on=ed_on, ed_denoise=ed_denoise))
        import random
        # Guard against double-start
        if _autoloop_active.is_set():
            busy = ('<p style="color:#f38ba8;">❌ Loop already running. '
                    'Stop it first before starting a new one.</p>')
            yield [], busy, busy, gr.update(), gr.update()
            return

        # a cleared number box arrives as None (float(None) used to escape with the flag still set)
        w1, w2, w3 = (_num(w, 0.8) for w in (w1, w2, w3))
        max_batches, delay = _num(max_batches, 0), _num(delay, 0)
        # Mark the loop active *before* loading so Stop Loop works during the load too
        _autoloop_active.set()
        _generation_abort.clear()
        loading = '<p style="color:#89dceb;">⏳ Loading the selected model…</p>'
        yield [], loading, loading, gr.update(), gr.update()
        try:
            ok, status = _ensure_model(model_path, vae_path)
            err = "" if not ok else _sync_loras([(lora1, w1), (lora2, w2), (lora3, w3)])
        except Exception as e:
            ok, status, err = True, None, html.escape(str(e) or type(e).__name__)
        if not ok or err or not _autoloop_active.is_set():
            _autoloop_active.clear()
            _generation_abort.clear()
            if not ok:
                msg = status if isinstance(status, str) else '<p style="color:#f38ba8;">❌ Model failed to load.</p>'
            elif err:
                msg = f'<p style="color:#f38ba8;">❌ LoRA problem: {err}</p>'
            else:
                msg = '<p style="color:#fab387;">⏹ Auto-Loop stopped before the first batch.</p>'
            yield [], msg, msg, gr.update(), gr.update()
            return
        max_batches = int(max_batches) if max_batches else 0
        delay = max(0, float(delay))
        batch_num = 0
        total_images = 0
        last_error = ""
        start_time = time.time()

        try:
            while _autoloop_active.is_set():
                batch_num += 1
                if max_batches > 0 and batch_num > max_batches:
                    break

                # Random seed each batch
                seed = random.randint(0, 2**32 - 1)
                status = (
                    f'<p style="color:#89dceb;font-size:13px;">'
                    f'🔄 <b>Auto-Loop</b> — Batch {batch_num}'
                    f'{f"/{max_batches}" if max_batches > 0 else " (∞)"}'
                    f' | Total images: {total_images}'
                    f' | Running {_fmt_elapsed(time.time() - start_time)}'
                    f' | Seed: {seed}</p>'
                )
                # Yield status before generation starts (keep the previous batch visible)
                yield gr.update(), status, status, gr.update(), gr.update()

                try:
                    imgs, info_html, _ = do_generate(
                        prompt, neg_prompt, scheduler, steps, cfg,
                        width, height, batch, seed,
                        init_img, strength, use_i2i, auto_quality=auto_quality, extra=extra,
                    )
                    total_images += len(imgs)
                except _GenerationAborted:
                    break
                except Exception as exc:
                    last_error = f"Batch {batch_num} failed: {exc}"
                    break
                if not imgs and _autoloop_active.is_set():
                    if _generation_abort.is_set():      # the Stop button (not Stop Loop) ended this batch
                        break
                    # do_generate reports failures (OOM, bad model…) as HTML, not exceptions
                    last_error = f"Batch {batch_num} produced no images."
                    yield [], info_html, info_html, gr.update(), gr.update()
                    break

                if not _autoloop_active.is_set():
                    break

                done_status = (
                    f'<p style="color:#a6e3a1;font-size:13px;">'
                    f'🔄 <b>Auto-Loop</b> — Batch {batch_num}'
                    f'{f"/{max_batches}" if max_batches > 0 else " (∞)"}'
                    f' done | Total images: {total_images}'
                    f' | Running {_fmt_elapsed(time.time() - start_time)}'
                    f'</p>{info_html}'
                )
                yield imgs, done_status, done_status, imgs, list(getattr(sd, "last_seeds", []) or [])

                # Delay between batches (check abort every 0.5s)
                if _autoloop_active.is_set() and delay > 0:
                    wait_end = time.time() + delay
                    while time.time() < wait_end and _autoloop_active.is_set():
                        time.sleep(0.5)

        finally:
            _autoloop_active.clear()
            _generation_abort.clear()

        elapsed = _fmt_elapsed(time.time() - start_time)
        done = batch_num - (1 if last_error else 0)
        if max_batches > 0:
            done = min(done, max_batches)
        if last_error:
            final_html = (
                f'<p style="color:#f38ba8;font-size:13px;">❌ <b>Auto-Loop stopped</b> — {last_error} '
                f'({total_images} images saved in {elapsed})</p>'
            )
        else:
            final_html = (
                f'<p style="color:#cba6f7;font-size:13px;">'
                f'✅ <b>Auto-Loop finished</b> — {done} batch(es), '
                f'{total_images} images in {elapsed} (saved to outputs/)</p>'
            )
        # Keep the gallery showing the last batch instead of blanking it
        yield gr.update(), gr.update(), final_html, gr.update(), gr.update()

    def do_stop_loop():
        _autoloop_active.clear()
        _generation_abort.set()
        return '<p style="color:#fab387;">⏹ Stopping auto-loop (aborting current batch)…</p>'

    def do_refresh(model_val=None):
        lora_choices = _lora_choices(model_val)
        return (
            gr.update(choices=_refresh_checkpoints()),
            gr.update(choices=lora_choices),
            gr.update(choices=lora_choices),
            gr.update(choices=lora_choices),
            gr.update(choices=["none"] + list_vaes()),
        )

    load_model_btn.click(
        lambda m: f'<p style="color:#89b4fa;">⏳ {_load_desc(m)}</p>' if m else gr.update(),
        [model_dd], [model_status], queue=False,
    ).then(do_load_model, [model_dd, vae_dd], [model_status, user_preset_dd])
    _lora_out = [lora_status, active_loras_html]

    def _apply_slot(slot):
        @gpu_job
        def _fn(model_path, vae_path, lora_path, weight, progress=gr.Progress()):
            ok, status = _ensure_model(model_path, vae_path, progress)
            if not ok:
                return (status if isinstance(status, str) else ""), _active_loras_html()
            return do_apply_lora(lora_path, weight, slot=slot)
        return _fn
    for _slot, (_btn, _dd, _w) in enumerate(((apply_lora_btn, lora_dd, lora_weight),
                                             (apply_lora_btn2, lora_dd2, lora_weight2),
                                             (apply_lora_btn3, lora_dd3, lora_weight3))):
        _btn.click(_apply_slot(_slot), [model_dd, vae_dd, _dd, _w], _lora_out)
    # Removing a slot also clears its dropdown — otherwise the next Generate
    # (which syncs LoRAs to the dropdowns) would silently re-apply it.
    remove_lora1_btn.click(gpu_job(lambda: (*do_remove_lora(0), "none")), [], _lora_out + [lora_dd])
    remove_lora2_btn.click(gpu_job(lambda: (*do_remove_lora(1), "none")), [], _lora_out + [lora_dd2])
    remove_lora3_btn.click(gpu_job(lambda: (*do_remove_lora(2), "none")), [], _lora_out + [lora_dd3])
    remove_lora_btn.click(gpu_job(lambda: (*do_remove_loras(), "none", "none", "none")), [],
                          _lora_out + [lora_dd, lora_dd2, lora_dd3])

    # ── Compatibility check ────────────────────────────────────────────────
    _COMPAT_LABEL = {
        "sd15": "SD 1.5", "sd2": "SD 2.x", "sdxl": "SDXL",
        "pony": "Pony (SDXL)", "illustrious": "Illustrious (SDXL)",
        "flux": "FLUX", "unknown": "?",
    }
    # All SDXL-family types are cross-compatible for LoRAs
    _SDXL_FAMILY = {"sdxl", "pony", "illustrious"}
    _COMPAT_NOTE = {
        ("sd15", "sdxl"): "❌ SD 1.5 model + SDXL LoRA — will likely crash or produce garbage.",
        ("sdxl", "sd15"): "❌ SDXL model + SD 1.5 LoRA — will likely crash or produce garbage.",
        ("sd15", "sd2"):  "❌ SD 1.5 model + SD 2.x LoRA — incompatible architectures.",
        ("sd2",  "sd15"): "❌ SD 2.x model + SD 1.5 LoRA — incompatible architectures.",
        ("sd15", "flux"): "❌ SD 1.5 model + FLUX LoRA — not compatible.",
        ("sdxl", "flux"): "❌ SDXL model + FLUX LoRA — not compatible.",
        ("flux", "sd15"): "❌ FLUX model + SD 1.5 LoRA — not compatible.",
        ("flux", "sdxl"): "❌ FLUX model + SDXL LoRA — not compatible.",
    }

    def do_check_compat(model_val, *lora_vals):
        """One line per LoRA slot that has something in it; problems first."""
        lines = [_compat_line(model_val, lv, i + 1) for i, lv in enumerate(lora_vals)]
        lines = [l for l in lines if l]
        lines.sort(key=lambda l: "✅" in l)
        return "".join(lines)

    def _compat_line(model_val, lora_val, slot):
        if not lora_val or lora_val == "none" or not model_val:
            return ""
        slot_txt = f"Slot {slot} ({Path(str(lora_val)).stem}): "
        mb = _infer_base(model_val)
        lb = _infer_base(lora_val)
        # Normalize SDXL family for compat checking
        mb_norm = "sdxl" if mb in _SDXL_FAMILY else mb
        lb_norm = "sdxl" if lb in _SDXL_FAMILY else lb
        key = (mb_norm, lb_norm)
        # same architecture, different lineage: loads fine, but identities / concepts drift
        # (Illustrious character LoRAs on Pony checkpoints gave the character another face and extra features)
        if {mb, lb} == {"pony", "illustrious"}:
            ml = _COMPAT_LABEL.get(mb, mb); ll = _COMPAT_LABEL.get(lb, lb)
            return (f'<p style="color:#f9e2af;margin:4px 0;">{slot_txt}⚠️ {ll} LoRA on a {ml} checkpoint: it '
                    f'loads and runs, but characters and concepts often come out wrong (different face, '
                    f'extra features). Prefer a {ml} version of this LoRA or a {ll} checkpoint.</p>')
        if key in _COMPAT_NOTE:
            return (
                f'<p style="color:#f38ba8;margin:4px 0;">{slot_txt}{_COMPAT_NOTE[key]}</p>'
            )
        if mb_norm == "unknown" or lb_norm == "unknown":
            ml = _COMPAT_LABEL.get(mb, mb); ll = _COMPAT_LABEL.get(lb, lb)
            return (
                f'<p style="color:#f9e2af;margin:4px 0;">{slot_txt}⚠️ Cannot verify compatibility '
                f'(model={ml}, LoRA={ll}) — check manually.</p>'
            )
        ml = _COMPAT_LABEL.get(mb, mb); ll = _COMPAT_LABEL.get(lb, lb)
        return f'<p style="color:#a6e3a1;margin:4px 0;">{slot_txt}✅ {ll} LoRA fits this {ml} model.</p>'

    _compat_in = [model_dd, lora_dd, lora_dd2, lora_dd3]
    for _c in _compat_in:
        _c.change(do_check_compat, _compat_in, [compat_html])

    # Pasted a Civitai / A1111 prompt with <lora:name:0.8> tags? Diffusers would just read
    # them as words — move them into free LoRA slots and take them out of the prompt.
    def on_prompt_lora_tags(prompt, d1, w1, d2, w2, d3, w3):
        found, clean = _loras_from_meta(prompt, None)
        if not found:
            return (gr.update(),) * 8
        local = {Path(n).stem.lower(): p for n, p in list_loras()}
        slots = [[d1, w1], [d2, w2], [d3, w3]]
        placed, missing, no_room = [], [], []
        for name, w in found:
            path = local.get(name.lower().removesuffix(".safetensors"))
            if path is None:
                missing.append(name)
                continue
            w = min(1.5, max(0.1, w))
            same = next((s for s in slots if s[0] == path), None)
            free = same or next((s for s in slots if not s[0] or s[0] == "none"), None)
            if free is None:
                no_room.append(Path(path).stem)
                continue
            free[0], free[1] = path, w
            placed.append(f"{Path(path).stem} ×{w:g}")
        parts = []
        if placed:
            parts.append("🧬 Moved from the prompt into the LoRA slots: " + ", ".join(placed))
        if missing:
            parts.append("not found locally (tag removed): <b>"
                         + ", ".join(html.escape(m) for m in missing) + "</b>")
        if no_room:
            parts.append("no free slot for: " + ", ".join(no_room))
        note = ('<p style="color:#89b4fa;font-size:13px;margin:4px 0;">'
                + " · ".join(parts) + "</p>")
        return (clean, *[v for s in slots for v in s], note)

    prompt_txt.input(
        on_prompt_lora_tags,
        [prompt_txt, lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3],
        [prompt_txt, lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3, lora_status],
        queue=False,
    )

    # ── Model metadata info panels ─────────────────────────────────────────
    model_dd.change(get_model_info, [model_dd], [ckpt_info_html])
    vae_dd.change(get_model_info, [vae_dd], [vae_info_html])
    lora_dd.change(get_model_info, [lora_dd], [lora_info1])
    lora_dd2.change(get_model_info, [lora_dd2], [lora_info2])
    lora_dd3.change(get_model_info, [lora_dd3], [lora_info3])

    # ── Trigger words / top training tags ─────────────────────────────────
    # (Trigger words now appear as clickable keyword chips — see on_keywords below.)

    # ── Token counter + tidy ──────────────────────────────────────────────
    def _tokenizer():
        return getattr(getattr(sd, "pipe", None), "tokenizer", None)

    def _triggers_of(*loras):
        out = []
        for v in loras:
            kw = lora_keywords(v) if v and v != "none" else None
            if kw:
                out += kw.triggers + kw.likely
        return out

    def on_prompt_tokens(pos, neg, l1=None, l2=None, l3=None):
        return token_report_html(pos or "", neg or "", _tokenizer(), _triggers_of(l1, l2, l3))

    _tok_in = [prompt_txt, neg_prompt_txt, lora_dd, lora_dd2, lora_dd3]
    for _t in _tok_in:
        _t.change(on_prompt_tokens, _tok_in, [token_html], queue=False, trigger_mode="always_last")

    def do_tidy(pos, neg):
        new_pos, d1 = tidy_prompt(pos or "")
        new_neg, d2 = tidy_prompt(neg or "")
        merged = d1 + [f"(neg) {d}" for d in d2]
        note = ('<p style="font-size:13px;color:#a6e3a1;margin:2px 0;">🧹 Merged: '
                + html.escape("; ".join(merged)) + "</p>") if merged else \
            '<p style="font-size:13px;color:#a6adc8;margin:2px 0;">🧹 No duplicate tags.</p>'
        return new_pos, new_neg, token_report_html(new_pos, new_neg, _tokenizer()) + note
    tidy_btn.click(do_tidy, [prompt_txt, neg_prompt_txt], [prompt_txt, neg_prompt_txt, token_html],
                   queue=False)

    # ── Keyword chips (checkpoint + 3 LoRA slots) ─────────────────────────
    def on_keywords(model_val, l1, l2, l3):
        slots = [(lbl, v) for lbl, v in (("Model", model_val), ("S1", l1), ("S2", l2), ("S3", l3))
                 if v and v != "none"]
        try:
            samples, tags, note = chips_for(slots)
        except Exception as e:
            print(f"[Keywords] {e}")
            samples, tags, note = [], [], ""
        has = bool(samples)
        return (gr.update(samples=samples or [["-"]], visible=has), tags, note,
                gr.update(visible=any(lora_keywords(v) and (lora_keywords(v).triggers or lora_keywords(v).tags)
                                      for lbl, v in slots if lbl != "Model")))
    _kw_in = [model_dd, lora_dd, lora_dd2, lora_dd3]
    for _c in _kw_in:
        _c.change(on_keywords, _kw_in, [kw_ds, kw_tags_state, kw_note, kw_core_btn])

    def on_keyword_click(idx, tags, pos):
        if idx is None or not tags or not 0 <= idx < len(tags):
            return gr.update()
        tag, front = tags[idx]
        # trigger words / creator prompts go right after the quality tags (the first chunk
        # steers the image most); ordinary tags are merged in at the end
        return insert_after_quality(pos or "", tag) if front else merge_prompts(pos or "", tag)

    kw_ds.click(on_keyword_click, [kw_ds, kw_tags_state, prompt_txt], [prompt_txt])

    def do_core_tags(pos, l1, l2, l3):
        add = []
        for v in (l1, l2, l3):
            kw = lora_keywords(v) if v and v != "none" else None
            if kw:
                add += kw.core()
        return insert_after_quality(pos or "", ", ".join(add)) if add else gr.update()
    kw_core_btn.click(do_core_tags, [prompt_txt, lora_dd, lora_dd2, lora_dd3], [prompt_txt])

    # ── Fetch trigger words from Civitai (hash lookup, no re-download) ────
    def do_fetch_triggers(model_val, lora_val, progress=gr.Progress()):
        import config as _cfg
        from backend.model_manager import list_checkpoints, list_loras
        from backend.trigger_reader import batch_fetch_sidecars

        all_paths = [p for _, p in list_checkpoints()] + [p for _, p in list_loras()]
        if not all_paths:
            return '<p style="color:#f9e2af;">No local models found.</p>', ""

        progress(0, desc="Hashing files and querying Civitai…")
        results = batch_fetch_sidecars(
            all_paths,
            api_key=_cfg.CIVITAI_API_KEY or "",
            on_progress=lambda done, total, name: progress(
                done / total if total else 0,
                desc=f"[{done}/{total}] {name}",
            ),
        )
        ok  = sum(1 for v in results.values() if "✅" in v)
        nf  = sum(1 for v in results.values() if "not on Civitai" in v)
        err = sum(1 for v in results.values() if "❌" in v)
        status = (
            f'<p style="color:#a6e3a1;">✅ Fetched {ok} sidecar(s) | '
            f'⚠️ {nf} not on Civitai | ❌ {err} errors</p>'
        )
        return status, ""

    fetch_triggers_btn.click(
        do_fetch_triggers,
        [model_dd, lora_dd],
        [fetch_triggers_status, triggers_html],
    ).then(on_keywords, [model_dd, lora_dd, lora_dd2, lora_dd3],
           [kw_ds, kw_tags_state, kw_note, kw_core_btn])

    # ── Prompt presets ─────────────────────────────────────────────────────
    def do_replace_preset(preset_name):
        p = _PROMPT_PRESETS.get(preset_name, {})
        pos = p.get("pos", "")
        neg = p.get("neg", "")
        # Also apply recommended generation settings if the preset includes them
        out = [pos, neg]
        out.append(gr.update(value=p["steps"]) if "steps" in p else gr.update())
        out.append(gr.update(value=p["cfg"]) if "cfg" in p else gr.update())
        out.append(gr.update(value=p["width"]) if "width" in p else gr.update())
        out.append(gr.update(value=p["height"]) if "height" in p else gr.update())
        return out

    def do_append_preset(preset_name, cur_pos, cur_neg):
        p = _PROMPT_PRESETS.get(preset_name, {})
        add_pos = p.get("pos", "")
        add_neg = p.get("neg", "")
        # every tag once, at its strongest weight — presets that share tags don't stack them
        return merge_prompts(cur_pos, add_pos), merge_prompts(cur_neg, add_neg)

    preset_replace_btn.click(do_replace_preset, [preset_dd],
                             [prompt_txt, neg_prompt_txt, steps_sl, cfg_sl, width_sl, height_sl])
    preset_append_btn.click(do_append_preset,  [preset_dd, prompt_txt, neg_prompt_txt], [prompt_txt, neg_prompt_txt])

    # ── My Saved Prompts wiring ───────────────────────────────────────────
    def do_load_user_preset(name):
        if not name:
            return gr.update(), ""
        text = _load_user_preset(name)
        return gr.update(value=text) if text else gr.update(), ""

    def do_save_user_preset(name, prompt):
        status = _save_user_preset(name, prompt)
        new_choices = _list_user_presets()
        return gr.update(choices=new_choices, value=name), status

    def do_delete_user_preset(name):
        status = _delete_user_preset(name)
        new_choices = _list_user_presets()
        return gr.update(choices=new_choices, value=None), status

    user_load_btn.click(do_load_user_preset, [user_preset_dd], [prompt_txt, user_preset_status])
    user_save_btn.click(do_save_user_preset, [user_preset_name, prompt_txt], [user_preset_dd, user_preset_status])
    user_delete_btn.click(do_delete_user_preset, [user_preset_dd], [user_preset_dd, user_preset_status])

    # ── Settings save/load wiring ─────────────────────────────────────────
    _EXTRA_KEYS = ["clip_skip", "var_seed", "var_strength", "hires_on", "hires_scale", "hires_denoise", "hires_steps",
                   "hires_upscaler", "fd_on", "fd_denoise", "fd_mode", "fd_prompt", "pag_scale", "freeu", "cfg_rescale",
                   "hd_on", "hd_denoise", "ed_on", "ed_denoise"]

    def do_save_settings(prompt, neg, sched, steps, cfg, w, h, batch, seed, name,
                         model, vae, l1, w1, l2, w2, l3, w3, auto_q, sel_img, imgs, *extras):
        target = sel_img if sel_img is not None else (imgs[0] if imgs else None)
        example = (getattr(target, "info", None) or {}).get("saved_path") if target is not None else None
        status = _save_generation_settings(prompt, neg, sched, steps, cfg, w, h, batch, seed, name,
                                           model=model, vae=vae, lora_slots=[(l1, w1), (l2, w2), (l3, w3)],
                                           auto_quality=auto_q, extra=dict(zip(_EXTRA_KEYS, extras)),
                                           example=example)
        if status.startswith("❌"):
            return gr.update(), status
        new_choices = _list_saved_settings()
        sname = _safe_name(name.strip() or "last_settings")   # the file's real name (Load looks it up)
        return gr.update(choices=new_choices, value=sname), status

    def do_load_settings(name):
        updates = _load_generation_settings(name)
        if not name:
            return updates + [gr.update()] * (1 + 6 + 2 + len(_EXTRA_KEYS)) + [""]
        model_upd, notes = gr.update(), []
        try:
            d = _json.loads((SETTINGS_DIR / f"{_safe_name(name)}.json").read_text(encoding="utf-8"))
        except Exception:
            d = {}
        saved_model = d.get("model") or ""
        if saved_model:
            # match by file name: saved paths may come from another PC / folder
            want = Path(saved_model).name.lower()
            match = next((p for n, p in list_checkpoints() if n.lower() == want), None)
            if match:
                model_upd = match
                notes.append(f"model <b>{Path(match).stem}</b>")
            else:
                notes.append(f'⚠ model <b>{Path(saved_model).stem}</b> not found locally')
        # LoRA slots + weights (format 2), else the names older files list (+ "lora_weights" if any)
        lora_ups = [gr.update()] * 6
        slots = d.get("lora_slots")
        if not isinstance(slots, list) and d.get("loras"):
            lw = d.get("lora_weights") if isinstance(d.get("lora_weights"), dict) else {}
            slots = [[n, next((v for k, v in lw.items() if Path(k).stem.lower() == Path(str(n)).stem.lower()), 0.8)]
                     for n in d["loras"] if isinstance(n, str)]
        if isinstance(slots, list):
            local = list_loras()
            vals, missing = [], []
            for item in (slots + [["none", 0.8]] * 3)[:3]:
                n, w = (item + [0.8])[:2] if isinstance(item, list) and item else ("none", 0.8)
                if not n or n == "none":
                    vals += ["none", gr.update()]
                    continue
                want = Path(str(n)).stem.lower()
                path = next((pth for fn, pth in local if Path(fn).stem.lower() == want), None)
                if path:
                    vals += [path, min(1.5, max(0.1, float(_num(w, 0.8))))]
                else:
                    missing.append(Path(str(n)).stem); vals += ["none", gr.update()]
            lora_ups = vals
            used = [Path(str(v)).stem for v in vals[0::2] if isinstance(v, str) and v != "none"]
            if used:
                notes.append("LoRAs " + ", ".join(used))
            if missing:
                notes.append("⚠ LoRA not found: " + ", ".join(missing))
        vae_upd = gr.update()
        if d.get("vae"):
            vae_upd = "none" if d["vae"] == "none" else next(
                (pth for fn, pth in list_vaes() if fn.lower() == str(d["vae"]).lower()), gr.update())
        aq_upd = bool(d["auto_quality"]) if isinstance(d.get("auto_quality"), bool) else gr.update()
        extra_ups = [gr.update()] * len(_EXTRA_KEYS)
        if isinstance(d.get("extra"), dict):
            ex = _clean_extra(d["extra"])
            extra_ups = [ex[k] for k in _EXTRA_KEYS]
            on = [lbl for k, lbl in (("hires_on", "hires fix"), ("fd_on", "face detail"), ("hd_on", "hand detail"),
                                        ("ed_on", "eye detail"), ("freeu", "FreeU")) if ex[k]]
            if ex["pag_scale"]:
                on.append(f"PAG {ex['pag_scale']:g}")
            if on:
                notes.append(", ".join(on))
        msg = f"✅ Loaded '{name}'" + (" · " + " · ".join(notes) if notes else "")
        return updates + [model_upd, *lora_ups, vae_upd, aq_upd, *extra_ups, msg]

    _settings_extra = [clip_skip_rb, var_seed_num, var_strength_sl, hires_cb, hires_scale_sl, hires_denoise_sl,
                       hires_steps_sl, hires_up_dd, fd_cb, fd_denoise_sl, fd_mode_rb, fd_prompt_txt,
                       pag_sl, freeu_cb, cfg_rescale_sl, hd_cb, hd_denoise_sl, ed_cb, ed_denoise_sl]
    settings_save_btn.click(
        do_save_settings,
        [prompt_txt, neg_prompt_txt, scheduler_dd, steps_sl, cfg_sl,
         width_sl, height_sl, batch_sl, seed_num, settings_name_txt,
         model_dd, vae_dd, lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3, auto_quality_cb,
         selected_gallery_image, last_generated_images, *_settings_extra],
        [settings_dd, settings_status],
    )

    def on_settings_pick(name):
        """Thumbnail of the image a saved setting made (its "example_image" in outputs/)."""
        try:
            d = _json.loads((SETTINGS_DIR / f"{_safe_name(name)}.json").read_text(encoding="utf-8")) if name else {}
        except Exception:
            d = {}
        ex = Path(str(d.get("example_image") or "")).name if isinstance(d, dict) else ""
        f = OUTPUTS_DIR / ex if ex else None
        return gr.update(value=str(f), visible=True) if f and f.is_file() else gr.update(value=None, visible=False)

    settings_dd.change(on_settings_pick, [settings_dd], [settings_preview])
    settings_load_btn.click(
        do_load_settings,
        [settings_dd],
        [prompt_txt, neg_prompt_txt, scheduler_dd, steps_sl, cfg_sl,
         width_sl, height_sl, batch_sl, seed_num, model_dd,
         lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3, vae_dd, auto_quality_cb,
         *_settings_extra, settings_status],
    )

    # ── Quick tag chips ────────────────────────────────────────────────────
    def _make_tag_appender(tag: str):
        def _append(target: str, pos: str, neg: str):
            # merged, not appended: a tag already there keeps its place (and the stronger weight)
            if "Positive" in target:
                return merge_prompts(pos, tag), neg
            return pos, merge_prompts(neg, tag)
        return _append

    for _btn, _tag in _tag_btn_list:
        _btn.click(
            _make_tag_appender(_tag),
            [tag_target, prompt_txt, neg_prompt_txt],
            [prompt_txt, neg_prompt_txt],
        )

    # i2i enhancer tags always append to positive prompt
    def _i2i_append(tag, pos):
        return merge_prompts(pos, tag)

    for _btn, _tag in _i2i_btn_list:
        _btn.click(
            lambda pos, t=_tag: _i2i_append(t, pos),
            [prompt_txt],
            [prompt_txt],
        )

    danbooru_meta_btn.click(
        lambda: fetch_danbooru_tags_html(category_id=5, limit=50),
        [],
        [danbooru_tags_html],
    )
    danbooru_general_btn.click(
        lambda: fetch_danbooru_tags_html(category_id=0, limit=60),
        [],
        [danbooru_tags_html],
    )

    def _norm_vae(v):
        return None if (not v or v == "none") else v

    def _effective_vae(model_path, vae_path):
        """(VAE to load, warning): a VAE of the other family (an SD 1.5 VAE with an SDXL checkpoint or the
        other way round) loads without an error but decodes garbage — the checkpoint's own VAE is used then.
        The dropdown keeps its value across model switches, so this happened silently."""
        from backend.model_manager import vae_family
        from backend.sd_pipeline import _is_sdxl
        vae = _norm_vae(vae_path)
        if not vae or not model_path:
            return vae, ""
        fam = vae_family(vae)
        want = "sdxl" if _is_sdxl(model_path) else "sd1"
        if fam and fam != want:
            return None, (f'<br><span style="color:#f9e2af;">⚠ {html.escape(Path(str(vae)).name)} is an '
                          f'{"SDXL" if fam == "sdxl" else "SD 1.5"} VAE and this is an '
                          f'{"SDXL" if want == "sdxl" else "SD 1.5"} checkpoint — using the checkpoint\'s own VAE '
                          f'(a mismatched VAE decodes garbage). Set VAE to none or pick a matching one.</span>')
        return vae, ""

    def _load_desc(model_path) -> str:
        """What's about to happen when this checkpoint loads (a HF repo not yet cached downloads first)."""
        gb = _hf_download_gb(model_path)
        if gb:
            return (f"Downloading {model_path} from Hugging Face (~{gb:g} GB, one time only) — "
                    f"progress shows in the launcher console window…")
        is_repo = not Path(str(model_path)).exists() and "/" in str(model_path)
        name = str(model_path) if is_repo else Path(str(model_path)).stem
        if _kernel_cache_note():
            return (f"Loading {name}… First run on this PC: the GPU is compiling its kernels once "
                    f"— the first image can take 10–15 minutes. Later ones take seconds.")
        return f"Loading {name}… (first load of a model takes 20–40 s)"

    def _ensure_model(model_path, vae_path, progress=None):
        """Load the checkpoint/VAE selected in the UI if it isn't the active one.
        Returns (ok, status_html_or_update)."""
        if not model_path:
            return False, '<p style="color:#f38ba8;">❌ Pick a checkpoint first.</p>'
        if (sd.pipe is not None and sd.current_model == model_path
                and _norm_vae(sd._last_vae_path) == _effective_vae(model_path, vae_path)[0]):
            return True, gr.update()
        if progress is not None:
            progress(0, desc=_load_desc(model_path))
        status_html, _ = do_load_model(model_path, vae_path, _from_generate=True)
        return sd.pipe is not None and sd.current_model == model_path, status_html

    _ARCH_LABEL = {"sd1": "SD 1.5", "sd2": "SD 2.x", "sdxl": "SDXL / Pony / Illustrious"}

    def _lora_mismatch(lora_path) -> str:
        """Plain-English error if a LoRA can't work with the loaded model ('' if fine/unknown)."""
        from backend.model_manager import lora_arch
        arch = lora_arch(lora_path)
        if arch is None or sd.pipe is None:
            return ""
        model_arch = "sdxl" if isinstance(sd, SDXLPipeline) else "sd1"
        if (arch == "sdxl") == (model_arch == "sdxl"):
            return ""
        return (f"<b>{Path(lora_path).stem}</b> is a {_ARCH_LABEL[arch]} LoRA, but the loaded model "
                f"<b>{Path(str(sd.current_model)).stem}</b> is {_ARCH_LABEL[model_arch]}. "
                f"Pick a matching LoRA or checkpoint.")

    def _short_err(e) -> str:
        msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        return msg if len(msg) <= 240 else msg[:240] + "…"

    def _sync_loras(specs, progress=None):
        """Make the active LoRAs match the three slot dropdowns + weights, so choosing
        a LoRA (or moving its weight) takes effect without pressing Apply.
        Returns an error string, or "" when OK. One reload, only if something changed."""
        want = {i: (Path(p).name, p, float(w)) for i, (p, w) in enumerate(specs)
                if p and p != "none"}
        norm = lambda d: {k: (v[1], round(v[2], 3)) for k, v in d.items()}
        if norm(want) == norm(sd._lora_adapters):
            return ""
        for slot, (_, path, _) in sorted(want.items()):
            bad = _lora_mismatch(path)
            if bad:
                return f"Slot {slot + 1}: {bad}"
        if progress is not None:
            progress(0, desc="Applying LoRAs…")
        if want and sd._clean_unet_state is None:
            sd._snapshot_clean_state()          # model is still LoRA-free here
        old = dict(sd._lora_adapters)
        sd._lora_adapters = want
        try:
            sd._reload_all_loras()
            return ""
        except Exception as e:
            sd._lora_adapters = old
            try:
                sd._reload_all_loras()
            except Exception:
                pass
            return _short_err(e)

    def do_generate_ui(model_path, vae_path, lora1, w1, lora2, w2, lora3, w3,
                       auto_quality, prompt, neg_prompt, scheduler,
                       steps, cfg, width, height, batch, seed, init_img, strength, use_i2i,
                       clip_skip=1, var_seed=-1, var_strength=0, hires_on=False, hires_scale=1.5,
                       hires_denoise=0.45, hires_steps=15, hires_upscaler="Lanczos",
                       fd_on=False, fd_denoise=0.4, fd_mode="auto", fd_prompt="",
                       pag_scale=0.0, freeu=False, cfg_rescale=0.0, hd_on=False, hd_denoise=0.35, ed_on=False, ed_denoise=0.4,
                       progress=gr.Progress()):
        _generation_abort.clear()          # a new run starts; Stop from here on counts
        extra = _clean_extra(dict(clip_skip=clip_skip, var_seed=var_seed, var_strength=var_strength,
                                  hires_on=hires_on, hires_scale=hires_scale, hires_denoise=hires_denoise,
                                  hires_steps=hires_steps, hires_upscaler=hires_upscaler, fd_on=fd_on,
                                  fd_denoise=fd_denoise, fd_mode=fd_mode, fd_prompt=fd_prompt,
                                  pag_scale=pag_scale, freeu=freeu, cfg_rescale=cfg_rescale,
                                  hd_on=hd_on, hd_denoise=hd_denoise, ed_on=ed_on, ed_denoise=ed_denoise))
        w1, w2, w3 = (_num(w, 0.8) for w in (w1, w2, w3))
        ok, status = _ensure_model(model_path, vae_path, progress)
        if _generation_abort.is_set():
            return ([], '<p style="color:#fab387;">⏹ Stopped (the model is loaded and ready).</p>',
                    [], status, gr.update(), _active_loras_html())
        if not ok:
            return ([], status if isinstance(status, str) else "", [], status, gr.update(),
                    _active_loras_html())
        err = _sync_loras([(lora1, w1), (lora2, w2), (lora3, w3)], progress)
        if err:
            msg = f'<p style="color:#f38ba8;">❌ LoRA problem: {err}</p>'
            return [], msg, [], status, gr.update(), _active_loras_html()
        imgs, info_html, last = do_generate(
            prompt, neg_prompt, scheduler, steps, cfg, width, height, batch, seed,
            init_img, strength, use_i2i, auto_quality=auto_quality, extra=extra, progress=progress)
        steps, cfg, width, height, batch, seed, _, _ = _clean_gen_args(
            steps, cfg, width, height, batch, seed)
        _write_last_session(dict(
            model=model_path, vae=vae_path or "none", prompt=prompt, negative_prompt=neg_prompt,
            scheduler=scheduler, steps=steps, cfg_scale=cfg, width=width,
            height=height, batch_size=batch, seed=seed,
            auto_quality=bool(auto_quality),
            loras=[[lora1 or "none", float(w1)], [lora2 or "none", float(w2)], [lora3 or "none", float(w3)]],
            clip_skip=extra["clip_skip"], hires_on=extra["hires_on"], hires_scale=extra["hires_scale"],
            hires_denoise=extra["hires_denoise"], hires_steps=extra["hires_steps"],
            hires_upscaler=extra["hires_upscaler"], fd_on=extra["fd_on"], fd_denoise=extra["fd_denoise"],
            fd_mode=extra["fd_mode"], fd_prompt=extra["fd_prompt"],
            pag_scale=extra["pag_scale"], freeu=extra["freeu"], cfg_rescale=extra["cfg_rescale"],
            hd_on=extra["hd_on"], hd_denoise=extra["hd_denoise"],
            ed_on=extra["ed_on"], ed_denoise=extra["ed_denoise"],
        ))
        seeds = getattr(sd, "last_seeds", None) or []
        return (imgs, info_html, last, status, (list(seeds) if imgs else gr.update()),
                _active_loras_html())

    _gen_inputs = [
        model_dd, vae_dd, lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3,
        auto_quality_cb,
        prompt_txt, neg_prompt_txt, scheduler_dd, steps_sl, cfg_sl,
        width_sl, height_sl, batch_sl, seed_num,
        init_image, strength_sl, use_i2i_cb,
        clip_skip_rb, var_seed_num, var_strength_sl, hires_cb, hires_scale_sl, hires_denoise_sl,
        hires_steps_sl, hires_up_dd, fd_cb, fd_denoise_sl, fd_mode_rb, fd_prompt_txt,
        pag_sl, freeu_cb, cfg_rescale_sl, hd_cb, hd_denoise_sl, ed_cb, ed_denoise_sl,
    ]
    gen_event = generate_btn.click(
        lambda: (None, 0), None, [selected_gallery_image, selected_idx_state],
    ).then(
        do_generate_ui, _gen_inputs,
        [output_gallery, gen_info, last_generated_images, model_status, last_seed_state,
         active_loras_html],
    )
    # Stop only sets the abort flag (no cancels=): cancelling the task handed the GPU slot to the next
    # job while this one was still running. The job ends at its next step / phase check.
    stop_btn.click(do_stop, [], [gen_info])

    def do_xy_grid(model_path, vae_path, lora1, w1, lora2, w2, lora3, w3, auto_quality, prompt, neg_prompt,
                   scheduler, steps, cfg, width, height, batch, seed, init_img, strength, use_i2i,
                   clip_skip, var_seed, var_strength, hires_on, hires_scale, hires_denoise, hires_steps,
                   hires_upscaler, fd_on, fd_denoise, fd_mode, fd_prompt, pag_scale, freeu, cfg_rescale,
                   hd_on, hd_denoise, ed_on, ed_denoise, x_axis, x_text, y_axis, y_text,
                   progress=gr.Progress()):
        import random
        _generation_abort.clear()
        w1, w2, w3 = (_num(w, 0.8) for w in (w1, w2, w3))     # cleared boxes arrive as None
        err_html = lambda m: ([], f'<p style="color:#f38ba8;">❌ {m}</p>', gr.update(), gr.update(), gr.update(),
                              gr.update())
        xs, e1 = _xy_values(x_axis, x_text)
        ys, e2 = _xy_values(y_axis, y_text)
        if e1 or e2 or x_axis in (None, "none"):
            return err_html(e1 or e2 or "Pick an X axis.")
        if len(xs) * len(ys) > 48:
            return err_html(f"{len(xs)} × {len(ys)} = {len(xs) * len(ys)} images — keep it at 48 or fewer.")
        for axis, vals in ((x_axis, xs), (y_axis, ys)):
            if axis == "Prompt S/R" and vals[0] not in (prompt or ""):
                return err_html(f"Prompt S/R: “{html.escape(vals[0])}” isn't in the prompt.")
            if axis == "LoRA 1 weight" and (not lora1 or lora1 == "none"):
                return err_html("LoRA 1 weight: pick a LoRA in slot 1 first.")
        ok, status = _ensure_model(model_path, vae_path, progress)
        if not ok:
            return [], (status if isinstance(status, str) else ""), status, gr.update(), gr.update(), gr.update()
        seed = int(_num(seed, -1))
        seed = random.randint(0, 2**32 - 1) if seed < 0 else seed
        base_extra = dict(clip_skip=clip_skip, var_seed=var_seed, var_strength=var_strength, hires_on=hires_on,
                          hires_scale=hires_scale, hires_denoise=hires_denoise, hires_steps=hires_steps,
                          hires_upscaler=hires_upscaler, fd_on=fd_on, fd_denoise=fd_denoise, fd_mode=fd_mode,
                          fd_prompt=fd_prompt, pag_scale=pag_scale, freeu=freeu, cfg_rescale=cfg_rescale,
                          hd_on=hd_on, hd_denoise=hd_denoise, ed_on=ed_on, ed_denoise=ed_denoise)
        cells, t0, lora_w_now, model_now = [], time.time(), None, model_path
        cell_seeds = []
        total = len(xs) * len(ys)
        try:
            for yi, yv in enumerate(ys):
                for xi, xv in enumerate(xs):
                    if _generation_abort.is_set():
                        raise _GenerationAborted()
                    p = dict(prompt=prompt or "", scheduler=scheduler, steps=steps, cfg=cfg, seed=seed,
                             extra=dict(base_extra), lw=w1, model=model_path)
                    for axis, v, vals in ((x_axis, xv, xs), (y_axis, yv, ys)):
                        if axis == "CFG": p["cfg"] = v
                        elif axis == "Steps": p["steps"] = v
                        elif axis == "Sampler": p["scheduler"] = v
                        elif axis == "Seed": p["seed"] = v
                        elif axis == "LoRA 1 weight": p["lw"] = v
                        elif axis == "CLIP skip": p["extra"]["clip_skip"] = v
                        elif axis == "Hires denoise": p["extra"].update(hires_on=True, hires_denoise=v)
                        elif axis == "Face denoise": p["extra"].update(fd_on=True, fd_denoise=v)
                        elif axis == "Hand denoise": p["extra"].update(hd_on=True, hd_denoise=v)
                        elif axis == "Eye denoise": p["extra"].update(ed_on=True, ed_denoise=v)
                        elif axis == "PAG scale": p["extra"]["pag_scale"] = v
                        elif axis == "CFG rescale": p["extra"]["cfg_rescale"] = v
                        elif axis == "Prompt S/R": p["prompt"] = p["prompt"].replace(vals[0], v)
                        elif axis == "Checkpoint": p["model"] = v
                    if p["model"] != model_now:           # Checkpoint axis: switch models
                        progress((len(cells)) / total, desc=f"Loading {Path(str(p['model'])).stem}…")
                        ok_m, st_m = _ensure_model(p["model"], vae_path, progress)
                        if not ok_m:
                            return err_html(f"Could not load {html.escape(Path(str(p['model'])).stem)}")
                        model_now, lora_w_now = p["model"], None
                    if p["lw"] != lora_w_now:
                        err = _sync_loras([(lora1, p["lw"]), (lora2, w2), (lora3, w3)], progress)
                        if err:
                            return err_html(f"LoRA problem: {err}")
                        lora_w_now = p["lw"]
                    n = len(cells) + 1
                    progress((n - 1) / total, desc=f"Grid {n}/{total}")
                    imgs, info, _ = do_generate(p["prompt"], neg_prompt, p["scheduler"], p["steps"], p["cfg"],
                                                width, height, 1, p["seed"], init_img, strength, use_i2i,
                                                auto_quality=auto_quality, extra=p["extra"])
                    if not imgs:
                        return [], info, gr.update(), gr.update(), gr.update(), gr.update()
                    cells.append(imgs[0])
                    cell_seeds.append((getattr(sd, "last_seeds", None) or [p["seed"]])[0])
        except _GenerationAborted:
            if not cells:
                return ([], '<p style="color:#fab387;">⏹ Grid stopped.</p>', gr.update(), gr.update(), gr.update(),
                        gr.update())
        finally:
            if lora_w_now is not None and lora_w_now != w1:
                _sync_loras([(lora1, w1), (lora2, w2), (lora3, w3)])
        fmt = lambda a, v: "" if a in (None, "none") else Path(str(v)).stem[:24] if a == "Checkpoint" else \
            f"{a} {v:g}" if isinstance(v, float) else f"{v}" if a in ("Sampler", "Prompt S/R") else f"{a} {v}"
        xl = [fmt(x_axis, v) for v in xs]
        yl = [fmt(y_axis, v) for v in ys]
        cols = len(xs)
        n_real = len(cells)
        while len(cells) % cols:                  # stopped mid-row: pad with blanks (for the grid picture only)
            from PIL import Image as _I
            cells.append(_I.new("RGB", cells[0].size, (30, 30, 46)))
        # (a varied seed / checkpoint is in the axis labels — the base value would be wrong here)
        axes_used = {x_axis, y_axis}
        title = " · ".join([*([f"seed {seed}"] if "Seed" not in axes_used else []),
                            *([Path(str(sd.current_model)).stem] if "Checkpoint" not in axes_used else []),
                            f"X: {x_axis}"]) + (f" · Y: {y_axis}" if y_axis not in (None, "none") else "")
        grid = _xy_grid(cells, xl, yl[: len(cells) // cols], title)
        from PIL.PngImagePlugin import PngInfo
        info = PngInfo()
        info.add_text("parameters", f"{prompt}\nNegative prompt: {neg_prompt}\nX/Y grid: {title}; "
                      f"X values: {', '.join(xl)}" + (f"; Y values: {', '.join(yl)}" if any(yl) else ""))
        gpath = _unique_output(f"grid_{int(time.time())}")
        grid.save(gpath, pnginfo=info)
        grid.info["saved_path"] = str(gpath)
        msg = (f'<p style="color:#a6e3a1;font-size:13px;">📊 Grid of {n_real} images in '
               f'{_fmt_elapsed(time.time() - t0)} — saved as {gpath.name} (each cell is saved too)'
               + (" — ⏹ stopped early" if _generation_abort.is_set() else "") + '.</p>')
        # the gallery shows the grid first, then the cells: keep the image list and seeds aligned with it
        # (Send to img2img / Show in folder / Last seed used the previous run's list before)
        shown = [grid] + cells[:n_real]
        return (shown, msg, status, _active_loras_html(), shown,
                [cell_seeds[0] if cell_seeds else seed] + cell_seeds)

    def do_inpaint_ui(model_path, vae_path, lora1, w1, lora2, w2, lora3, w3, prompt, neg_prompt, scheduler,
                      steps, cfg, seed, clip_skip, editor, denoise, padding, progress=gr.Progress()):
        from backend.detail_tools import inpaint_region
        _generation_abort.clear()
        image, mask = _editor_parts(editor)
        if image is None:
            return [], '<p style="color:#f38ba8;">❌ Upload an image in the Inpaint box first.</p>', \
                gr.update(), gr.update(), gr.update()
        if mask is None:
            return [], '<p style="color:#f38ba8;">❌ Paint over the part you want redrawn.</p>', \
                gr.update(), gr.update(), gr.update()
        image, fixed = _fit_init_image(image, max_side=2048)
        if fixed:
            mask = mask.resize(image.size, Image.NEAREST)
        steps, cfg, _w, _h, _b, seed, denoise, fixes = _clean_gen_args(steps, cfg, 512, 512, 1, seed, denoise,
                                                                      img2img=True)
        ok, status = _ensure_model(model_path, vae_path, progress)
        if not ok:
            return [], (status if isinstance(status, str) else ""), status, gr.update(), gr.update()
        err = _sync_loras([(lora1, _num(w1, 0.8)), (lora2, _num(w2, 0.7)), (lora3, _num(w3, 0.7))], progress)
        if err:
            return [], f'<p style="color:#f38ba8;">❌ LoRA problem: {err}</p>', status, gr.update(), gr.update()
        cs = 2 if int(_num(clip_skip, 1)) >= 2 else 1

        def cb(step, total):
            if _generation_abort.is_set():
                raise _GenerationAborted()
            progress(step / total, desc=f"Inpainting: step {step}/{total}")
        t0 = time.time()
        try:
            out, used = inpaint_region(sd, image, mask, prompt or "", neg_prompt or "", steps=_inpaint_steps(steps, denoise), cfg=cfg,
                                       denoise=denoise, seed=seed, scheduler=scheduler, clip_skip=cs,
                                       padding=int(_num(padding, 48)), step_callback=cb)
        except _GenerationAborted:
            return [], '<p style="color:#fab387;">⏹ Inpaint stopped.</p>', status, gr.update(), gr.update()
        except Exception as e:
            return [], f'<p style="color:#f38ba8;">❌ Inpaint failed: {html.escape(str(e)[:300])}</p>', \
                status, gr.update(), gr.update()
        saved = _save_outputs([out], dict(mode="inpaint", prompt=prompt or "", negative_prompt=neg_prompt or "",
                                          steps=steps, cfg_scale=cfg, seeds=[used], scheduler=scheduler,
                                          width=out.width, height=out.height, strength=denoise,
                                          inpaint_padding=int(_num(padding, 48)),
                                          clip_skip=cs if cs > 1 else None), pipe=sd)
        msg = (f'<p style="color:#a6adc8;font-size:13px;">🖌 Inpainted (denoise {denoise:g}, seed {used}) in '
               f'{time.time() - t0:.1f}s · Saved: {saved[0].name if saved else "-"}'
               + (f'<br><span style="color:#f9e2af;">Adjusted: {"; ".join(fixes)}</span>' if fixes else "")
               + "</p>")
        return [out], msg, status, [out], [used]

    @gpu_job
    def do_swap_ui(model_path, vae_path, lora1, w1, lora2, w2, lora3, w3, prompt, neg_prompt, scheduler,
                   steps, cfg, seed, clip_skip, editor, padding, item, progress=gr.Progress()):
        """🦋 Swap a hair accessory: paint over the old one, name the new one — one inpaint at denoise 1.0 with a hair-only
        prompt (detail_tools.swap_item). At 0.9–0.95 the old shape still shows through as lace."""
        from backend.detail_tools import SWAP_MIN_PADDING, swap_item, swap_prompt_from
        _generation_abort.clear()
        item = (item or "").strip()
        image, mask = _editor_parts(editor)
        if image is None:
            return [], '<p style="color:#f38ba8;">❌ Upload an image in the Inpaint box first.</p>', \
                gr.update(), gr.update(), gr.update()
        if mask is None:
            return [], '<p style="color:#f38ba8;">❌ Paint over the accessory you want to replace.</p>', \
                gr.update(), gr.update(), gr.update()
        if not item:
            return [], '<p style="color:#f38ba8;">❌ Type what should be there instead (e.g. black butterfly hair ornament).</p>', \
                gr.update(), gr.update(), gr.update()
        image, fixed = _fit_init_image(image, max_side=2048)
        if fixed:
            mask = mask.resize(image.size, Image.NEAREST)
        steps, cfg, _w, _h, _b, seed, _d, fixes = _clean_gen_args(steps, cfg, 512, 512, 1, seed, 1.0, img2img=True)
        ok, status = _ensure_model(model_path, vae_path, progress)
        if not ok:
            return [], (status if isinstance(status, str) else ""), status, gr.update(), gr.update()
        err = _sync_loras([(lora1, _num(w1, 0.8)), (lora2, _num(w2, 0.7)), (lora3, _num(w3, 0.7))], progress)
        if err:
            return [], f'<p style="color:#f38ba8;">❌ LoRA problem: {err}</p>', status, gr.update(), gr.update()
        cs = 2 if int(_num(clip_skip, 1)) >= 2 else 1
        pad = max(int(_num(padding, 48)), SWAP_MIN_PADDING)

        def cb(step, total):
            if _generation_abort.is_set():
                raise _GenerationAborted()
            progress(step / total, desc=f"Swapping: step {step}/{total}")
        t0 = time.time()
        try:
            out, used = swap_item(sd, image, mask, item, prompt or "", neg_prompt or "", steps=steps, cfg=cfg, seed=seed,
                                  scheduler=scheduler, clip_skip=cs, padding=pad, step_callback=cb)
        except _GenerationAborted:
            return [], '<p style="color:#fab387;">⏹ Swap stopped.</p>', status, gr.update(), gr.update()
        except Exception as e:
            return [], f'<p style="color:#f38ba8;">❌ Swap failed: {html.escape(str(e)[:300])}</p>', \
                status, gr.update(), gr.update()
        saved = _save_outputs([out], dict(mode="inpaint", prompt=swap_prompt_from(prompt or "", item), negative_prompt=neg_prompt or "",
                                          steps=steps, cfg_scale=cfg, seeds=[used], scheduler=scheduler,
                                          width=out.width, height=out.height, strength=1.0, inpaint_padding=pad,
                                          inpaint_swap=item, clip_skip=cs if cs > 1 else None), pipe=sd)
        msg = (f'<p style="color:#a6adc8;font-size:13px;">🦋 Swapped for “{html.escape(item)}” (seed {used}) in '
               f'{time.time() - t0:.1f}s · Saved: {saved[0].name if saved else "-"}'
               + (f'<br><span style="color:#f9e2af;">Adjusted: {"; ".join(fixes)}</span>' if fixes else "")
               + "</p>")
        return [out], msg, status, [out], [used]

    @gpu_job
    def do_hand_redraw_ui(model_path, vae_path, lora1, w1, lora2, w2, lora3, w3, prompt, neg_prompt, scheduler,
                          steps, cfg, seed, clip_skip, editor, tries, progress=gr.Progress()):
        """🖐 Re-draw a hand: several re-draws (seed, seed + 1, …) of the painted or the biggest detected hand at 0.7 with a
        hand-only prompt (detail_tools.redraw_hand); the gallery shows the original first, then every try, each saved."""
        from backend.detail_tools import HAND_REDRAW_DENOISE, detect_hands, hand_mask, hand_prompt_from, redraw_hand
        _generation_abort.clear()
        image, mask = _editor_parts(editor)
        if image is None:
            return [], '<p style="color:#f38ba8;">❌ Upload an image in the Inpaint box first.</p>',                 gr.update(), gr.update(), gr.update()
        image, fixed = _fit_init_image(image, max_side=2048)
        if mask is not None and fixed:
            mask = mask.resize(image.size, Image.NEAREST)
        found = ""
        if mask is None:
            hands = detect_hands(image, 6)
            if not hands:
                return [], ('<p style="color:#f38ba8;">❌ No hand found — paint over the hand you want re-drawn.</p>'),                     gr.update(), gr.update(), gr.update()
            mask = hand_mask(image.size, hands[0])
            found = f" (the biggest of {len(hands)} detected hand(s))" if len(hands) > 1 else " (the detected hand)"
        steps, cfg, _w, _h, _b, seed, _d, fixes = _clean_gen_args(steps, cfg, 512, 512, 1, seed, HAND_REDRAW_DENOISE, img2img=True)
        n = int(min(8, max(1, _num(tries, 4))))
        ok, status = _ensure_model(model_path, vae_path, progress)
        if not ok:
            return [], (status if isinstance(status, str) else ""), status, gr.update(), gr.update()
        err = _sync_loras([(lora1, _num(w1, 0.8)), (lora2, _num(w2, 0.7)), (lora3, _num(w3, 0.7))], progress)
        if err:
            return [], f'<p style="color:#f38ba8;">❌ LoRA problem: {err}</p>', status, gr.update(), gr.update()
        cs = 2 if int(_num(clip_skip, 1)) >= 2 else 1
        cur = [0]

        def on_try(k):
            cur[0] = k
            _thermal_wait(progress)
            if _generation_abort.is_set():
                raise _GenerationAborted()

        def cb(step, total):
            if _generation_abort.is_set():
                raise _GenerationAborted()
            progress(step / total, desc=f"Hand try {cur[0] + 1}/{n}: step {step}/{total}")
        t0 = time.time()
        done, results = [], []            # redraw_hand appends every try to `done` as it finishes: a Stop or an error at try k keeps the k − 1 finished tries
        try:
            results = redraw_hand(sd, image, mask, prompt or "", neg_prompt or "", tries=n, steps=steps, cfg=cfg, seed=seed,
                                  scheduler=scheduler, clip_skip=cs, step_callback=cb, on_try=on_try, out=done)
            stopped = ""
        except _GenerationAborted:
            results, stopped = done, f" · ⏹ stopped after {len(done)} of {n} tries"
        except Exception as e:
            if not done:
                return [], f'<p style="color:#f38ba8;">❌ Hand re-draw failed: {html.escape(str(e)[:300])}</p>',                 status, gr.update(), gr.update()
            results, stopped = done, f" · ⚠ stopped at try {len(done) + 1} of {n}: {html.escape(str(e)[:120])}"
        if not results:
            return [], '<p style="color:#fab387;">⏹ Hand re-draw stopped.</p>', status, gr.update(), gr.update()
        imgs = [r[0] for r in results]
        used = [r[1] for r in results]
        saved = _save_outputs(imgs, dict(mode="inpaint", prompt=hand_prompt_from(prompt or ""), negative_prompt=neg_prompt or "",
                                         steps=steps, cfg_scale=cfg, seeds=used, scheduler=scheduler,
                                         width=imgs[0].width, height=imgs[0].height, strength=HAND_REDRAW_DENOISE, inpaint_hand_redraw=True,
                                         clip_skip=cs if cs > 1 else None), pipe=sd)
        shown = [image] + imgs
        msg = (f'<p style="color:#a6adc8;font-size:13px;">🖐 {len(imgs)} re-draw(s) of the hand{found} in {time.time() - t0:.1f}s '
               f'(seeds {used[0]}–{used[-1]}){stopped} — the first picture is the original; keep the try you like '
               f'(all saved: {html.escape(", ".join(p.name for p in saved[:2]))}{" …" if len(saved) > 2 else ""})'
               + (f'<br><span style="color:#f9e2af;">Adjusted: {"; ".join(fixes)}</span>' if fixes else "")
               + "</p>")
        return shown, msg, status, shown, [used[0]] + used

    inp_hand_btn.click(
        lambda: (None, 0), None, [selected_gallery_image, selected_idx_state],
    ).then(
        do_hand_redraw_ui,
        [model_dd, vae_dd, lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3, prompt_txt,
         neg_prompt_txt, scheduler_dd, steps_sl, cfg_sl, seed_num, clip_skip_rb, inp_editor, inp_hand_tries],
        [output_gallery, gen_info, model_status, last_generated_images, last_seed_state],
    )

    inp_swap_event = inp_swap_btn.click(
        lambda: (None, 0), None, [selected_gallery_image, selected_idx_state],
    ).then(
        do_swap_ui,
        [model_dd, vae_dd, lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3, prompt_txt,
         neg_prompt_txt, scheduler_dd, steps_sl, cfg_sl, seed_num, clip_skip_rb, inp_editor, inp_pad_sl, inp_swap_txt],
        [output_gallery, gen_info, model_status, last_generated_images, last_seed_state],
    )

    inp_event = inp_btn.click(
        lambda: (None, 0), None, [selected_gallery_image, selected_idx_state],
    ).then(
        do_inpaint_ui,
        [model_dd, vae_dd, lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3, prompt_txt,
         neg_prompt_txt, scheduler_dd, steps_sl, cfg_sl, seed_num, clip_skip_rb, inp_editor, inp_denoise_sl,
         inp_pad_sl],
        [output_gallery, gen_info, model_status, last_generated_images, last_seed_state],
    )

    xy_event = xy_btn.click(
        lambda: (None, 0), None, [selected_gallery_image, selected_idx_state],
    ).then(
        do_xy_grid, _gen_inputs + [xy_x_axis, xy_x_vals, xy_y_axis, xy_y_vals],
        [output_gallery, gen_info, model_status, active_loras_html, last_generated_images, last_seed_state],
    )
    # Recreate a dropped image: the same run as Generate, with img2img switched off
    rec_event = recreate_btn.click(
        lambda: (False, None, 0), None, [use_i2i_cb, selected_gallery_image, selected_idx_state],
    ).then(
        do_generate_ui, _gen_inputs,
        [output_gallery, gen_info, last_generated_images, model_status, last_seed_state,
         active_loras_html],
    )

    # ── Seed / size helpers ──────────────────────────────────────────────
    seed_random_btn.click(lambda: -1, None, seed_num)
    def do_reuse_seed(seeds, idx):
        """Seed of the gallery image you clicked (the first one if none was clicked)."""
        if not seeds:
            return gr.update(), '<p style="color:#f5c6a0;font-size:13px;">Generate something first.</p>'
        i = idx if isinstance(idx, int) and 0 <= idx < len(seeds) else 0
        return seeds[i], (f'<p style="color:#a6e3a1;font-size:13px;">♻️ Seed {seeds[i]} '
                          f'(image {i + 1} of {len(seeds)}) — the first image of your next run will match it.</p>')
    seed_reuse_btn.click(do_reuse_seed, [last_seed_state, selected_idx_state], [seed_num, gal_status])

    def do_more_like(seeds, idx, cur_strength):
        """Same seed + a random variation seed: the next run keeps the picked image's
        composition and changes details (raise Variation strength for bigger changes)."""
        if not seeds:
            return (gr.update(),) * 4 + ('<p style="color:#f5c6a0;font-size:13px;">Generate something first.</p>',)
        i = idx if isinstance(idx, int) and 0 <= idx < len(seeds) else 0
        st = cur_strength if cur_strength and 0 < float(cur_strength) <= 0.5 else 0.1
        return (seeds[i], -1, st, False,
                f'<p style="color:#a6e3a1;font-size:13px;">🔀 Seed {seeds[i]} + variation strength {st:g} — '
                f'press Generate (a batch gives several variations at once).</p>')
    gal_more_btn.click(do_more_like, [last_seed_state, selected_idx_state, var_strength_sl],
                       [seed_num, var_seed_num, var_strength_sl, use_i2i_cb, gal_status])
    swap_size_btn.click(lambda w, h: (h, w), [width_sl, height_sl], [width_sl, height_sl])

    def on_size_preset(label):
        wh = _parse_size_preset(label)
        return (wh[0], wh[1]) if wh else (gr.update(), gr.update())
    size_preset_dd.change(on_size_preset, [size_preset_dd], [width_sl, height_sl])

    # ✨ Polish: the detail passes' controls take the recipe for the picture's framing (backend/polish.py)
    _polish_outs = [hires_cb, hires_scale_sl, hires_denoise_sl, hires_steps_sl, hires_up_dd, fd_cb, fd_denoise_sl, hd_cb,
                    hd_denoise_sl, ed_cb, ed_denoise_sl]
    _polish_keys = ["hires_on", "hires_scale", "hires_denoise", "hires_steps", "hires_upscaler", "fd_on", "fd_denoise",
                    "hd_on", "hd_denoise", "ed_on", "ed_denoise"]

    def on_polish(choice, prompt):
        from backend import polish as _pl
        name = _pl.shot_type(prompt) if choice == "Auto" else choice
        r = _pl.recipe(name)
        if not r:
            return [gr.update()] * len(_polish_outs) + [""]
        return ([gr.update(value=r[k]) if k in r else gr.update() for k in _polish_keys]
                + [f'<p style="color:#a6e3a1;font-size:13px;">✨ {html.escape(name)}: {html.escape(_pl.summary(name))} '
                   f'— about {_pl.time_factor(name):g}× the time of the plain picture.</p>'])
    polish_dd.input(on_polish, [polish_dd, prompt_txt], _polish_outs + [polish_note])

    # ── Dropping an image into img2img brings back how it was made ───────────
    _restore_outputs = [prompt_txt, neg_prompt_txt, scheduler_dd, steps_sl, cfg_sl, width_sl, height_sl,
                        batch_sl, seed_num, model_dd, vae_dd, lora_dd, lora_weight, lora_dd2, lora_weight2,
                        lora_dd3, lora_weight3, strength_sl, use_i2i_cb, i2i_restore_html,
                        clip_skip_rb, var_seed_num, var_strength_sl, hires_cb, hires_scale_sl,
                        hires_denoise_sl, hires_steps_sl, hires_up_dd, fd_cb, fd_denoise_sl, fd_mode_rb,
                        fd_prompt_txt, pag_sl, freeu_cb, cfg_rescale_sl, hd_cb, hd_denoise_sl, ed_cb, ed_denoise_sl, recreate_btn]

    def on_i2i_drop(img, restore):
        keep = [gr.update()] * 18
        tail = [gr.update()] * 19 + [gr.update(visible=False)]
        if img is None:
            return (*keep, gr.update(), "", *tail)
        if not restore:
            return (*keep, True, "", *tail)
        try:
            meta = read_png_info(img)
        except Exception:
            meta = {}
        if not (meta.get("prompt") or meta.get("imagegen")):
            return (*keep, True, '<p style="font-size:13px;color:#a6adc8;">No generation settings in this image '
                                  '— img2img mode is on; write a prompt for it.</p>', *tail)
        plan = _restore_plan(meta)
        u = lambda v: gr.update() if v is None else v
        lora_ups = [gr.update()] * 6
        if plan["loras"] is not None:
            slots = (plan["loras"] + [("none", None)] * 3)[:3]
            lora_ups = [v for path, w in slots for v in (path, gr.update() if w is None else w)]
        if plan["mode"] == "img2img":
            how = ("It was itself an img2img result, so to recreate it exactly you need its source image "
                   "(and strength " + str(plan.get("strength")) + "); ")
        elif plan["mode"] == "inpaint":      # a txt2img run isn't the inpainted picture (audit F-20)
            how = ("It is an inpaint result: recreating it needs its source image and the painted mask; ")
        else:
            how = ("<b>To recreate it exactly:</b> untick “Use img2img mode” and press Generate. "
                   "Keep it ticked to make variations of it (strength 0.3–0.5 keeps the composition); ")
        note = ('<p style="font-size:13px;color:#a6e3a1;margin:4px 0;">📋 Restored '
                + ("exactly (ImageGen Studio record)" if plan["exact"] else "from its A1111-style parameters")
                + ": " + _plan_summary(plan) + '</p><p style="font-size:13px;color:#a6adc8;margin:2px 0;">'
                + how + "batch size was set to 1 so the seed matches.</p>")
        strength = gr.update()
        if plan["mode"] not in ("img2img", "inpaint"):
            strength = gr.update()                      # keep the user's variation strength
        elif plan.get("strength") is not None:
            strength = min(1.0, max(0.1, float(plan["strength"])))
        return (u(plan["prompt"]), u(plan["negative_prompt"]), u(plan["scheduler"]), u(plan["steps"]),
                u(plan["cfg_scale"]), u(plan["width"]), u(plan["height"]), 1, u(plan["seed"]),
                u(plan["model"]), u(plan["vae"]), *lora_ups, strength, True, note,
                *_plan_extra_updates(plan), gr.update(visible=plan["mode"] not in ("img2img", "inpaint")))

    init_image.upload(on_i2i_drop, [init_image, restore_cb], _restore_outputs)

    def on_model_pick(model_path, w, h):
        """Switching between SD 1.5 and SDXL-class models: move the size to that
        family's native resolution if the current one clearly belongs to the other."""
        from backend.sd_pipeline import _is_sdxl
        try:
            w, h = int(w), int(h)
            xl = _is_sdxl(model_path) if model_path else None
        except Exception:
            return gr.update(), gr.update()
        if xl and max(w, h) <= 768:
            return (832, 1216) if h > w else (1216, 832) if w > h else (1024, 1024)
        if xl is False and min(w, h) >= 1024:
            return (512, 768) if h > w else (768, 512) if w > h else (512, 512)
        return gr.update(), gr.update()
    model_dd.change(on_model_pick, [model_dd, width_sl, height_sl], [width_sl, height_sl])
    # Matching LoRAs first in all three slots (selected values are kept)
    model_dd.change(lambda m: (gr.update(choices=_lora_choices(m)),) * 3,
                    [model_dd], [lora_dd, lora_dd2, lora_dd3], queue=False)

    def do_open_outputs(sel=None, imgs=None):
        """Open Explorer with the selected (else first) image highlighted, or just the folder."""
        target = sel if sel is not None else (imgs[0] if imgs else None)
        saved = getattr(target, "info", {}).get("saved_path") if target is not None else None
        try:
            if saved and Path(saved).is_file():
                import subprocess
                subprocess.Popen(f'explorer /select,"{Path(saved)}"')
            else:
                os.startfile(str(OUTPUTS_DIR))   # Windows Explorer
            return ""
        except Exception as e:
            return f'<p style="color:#f38ba8;font-size:13px;">Could not open folder: {e}<br>{OUTPUTS_DIR}</p>'
    open_outputs_btn.click(do_open_outputs, [selected_gallery_image, last_generated_images], gal_status)

    # ── Gallery actions wiring ───────────────────────────────────────────
    def on_gallery_select(evt: gr.SelectData, imgs, seeds):
        i = evt.index if isinstance(evt.index, int) else 0
        img = imgs[i] if imgs and i < len(imgs) else None
        note = (f'<p style="color:#a6adc8;font-size:13px;">Image {i + 1} selected — seed '
                f'<b>{seeds[i]}</b>. ♻️ Last seed reuses it; Send to… sends it.</p>'
                if seeds and i < len(seeds) else "")
        return img, i, note

    output_gallery.select(on_gallery_select, [last_generated_images, last_seed_state],
                          [selected_gallery_image, selected_idx_state, gal_status])

    def do_send_gal_to_i2i(sel, imgs):
        target = sel if sel is not None else (imgs[0] if imgs else None)
        if target is None:
            return (gr.update(), gr.update(),
                    '<p style="color:#f38ba8;font-size:13px;">⚠ No image available to send.</p>')
        return (target, True,
                '<p style="color:#a6e3a1;font-size:13px;">✅ Image sent to img2img — img2img mode is on '
                '(see the open Image-to-Image section). Adjust Denoise Strength, then Generate.</p>')

    gal_send_i2i_btn.click(
        do_send_gal_to_i2i,
        [selected_gallery_image, last_generated_images],
        [init_image, use_i2i_cb, gal_status],
    ).then(None, None, None, js=_OPEN_I2I_JS)

    # ── Auto-Loop wiring ─────────────────────────────────────────────────
    loop_event = loop_start_btn.click(
        lambda: (None, 0), None, [selected_gallery_image, selected_idx_state],
    ).then(
        do_autoloop,
        [
            model_dd, vae_dd, lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3,
            auto_quality_cb,
            prompt_txt, neg_prompt_txt, scheduler_dd, steps_sl, cfg_sl,
            width_sl, height_sl, batch_sl, seed_num,
            init_image, strength_sl, use_i2i_cb,
            loop_max_batches, loop_delay,
            clip_skip_rb, var_seed_num, var_strength_sl, hires_cb, hires_scale_sl, hires_denoise_sl,
            hires_steps_sl, hires_up_dd, fd_cb, fd_denoise_sl, fd_mode_rb, fd_prompt_txt,
            pag_sl, freeu_cb, cfg_rescale_sl, hd_cb, hd_denoise_sl, ed_cb, ed_denoise_sl,
        ],
        [output_gallery, gen_info, loop_status, last_generated_images, last_seed_state],
    )
    # no cancels=: a cancelled task gives its GPU slot away while its thread still runs
    loop_stop_btn.click(do_stop_loop, [], [loop_status])
    refresh_btn.click(do_refresh, [model_dd], [model_dd, lora_dd, lora_dd2, lora_dd3, vae_dd])

    def _update_gen_vram(model, width, height, batch, lora, vae, use_i2i):
        used = estimate_generate(
            model_name=model or "",
            width=int(_num(width, 512)), height=int(_num(height, 512)), batch=int(_num(batch, 1)),
            use_lora=(lora and lora != "none"),
            use_external_vae=(vae and vae != "none"),
            use_img2img=bool(use_i2i),
        )
        return vram_bar_html(used, _gen_total_mb, "Est. VRAM")

    _gen_vram_inputs = [model_dd, width_sl, height_sl, batch_sl, lora_dd, vae_dd, use_i2i_cb]
    for _w in _gen_vram_inputs:
        _w.change(_update_gen_vram, _gen_vram_inputs, [_gen_vram_bar])

    # ── Danbooru tag autocomplete ──────────────────────────────────────────
    _tag_list_loading = []

    def _partial_tag(prompt):
        """The tag being typed: text after the last comma / line break (None if it's done)."""
        p = prompt or ""
        if not p or p[-1] in ",\n)]>}":
            return None
        cut = max(p.rfind(","), p.rfind("\n"))
        part = p[cut + 1:].lstrip(" (")
        return part if 2 <= len(part) <= 40 and "<" not in part and "__" not in part and "{" not in part else None

    def do_tag_suggest(prompt):
        from backend import danbooru_tags as dt
        part = _partial_tag(prompt)
        if part is None:
            return gr.update(visible=False), []
        if not dt.load(download=False):
            if not _tag_list_loading:          # fetch the ~300 KB list once, in the background
                _tag_list_loading.append(1)
                threading.Thread(target=dt.load, daemon=True).start()
            return gr.update(visible=False), []
        sug = [(t, n) for t, n in dt.suggest(part) if t != part.strip().lower()]
        if not sug:
            return gr.update(visible=False), []
        fmt = lambda n: f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else str(n)
        return (gr.update(samples=[[f"{t}  ·  {fmt(n)}"] for t, n in sug], visible=True),
                [t for t, _ in sug])

    def do_tag_complete(prompt, idx, tags):
        try:
            tag = tags[int(idx)]
        except (TypeError, ValueError, IndexError):
            return gr.update(), gr.update(visible=False), []
        part = _partial_tag(prompt)
        head = (prompt or "")[: len(prompt or "") - len(part)] if part else (prompt or "")
        # keep a "(" the user opened for weighting; escape brackets inside the tag itself
        esc = tag.replace("(", "\\(").replace(")", "\\)")
        return head + esc + ", ", gr.update(visible=False), []

    def do_fix_spellings(pos, neg):
        from backend.prompt_tools import apply_danbooru_fixes
        p2, f1 = apply_danbooru_fixes(pos or "")
        n2, f2 = apply_danbooru_fixes(neg or "")
        fixes = f1 + f2
        note = ('<p style="font-size:13px;color:#a6e3a1;margin:2px 0;">💡 Fixed: '
                + ", ".join(f"{html.escape(a)} → <b>{html.escape(b)}</b>" for a, b in fixes) + "</p>") if fixes else \
            '<p style="font-size:13px;color:#a6adc8;margin:2px 0;">💡 No near-miss Danbooru tags found.</p>'
        return p2, n2, note

    spell_btn.click(do_fix_spellings, [prompt_txt, neg_prompt_txt], [prompt_txt, neg_prompt_txt, gen_info])

    prompt_txt.input(do_tag_suggest, [prompt_txt], [tag_ac_ds, tag_ac_state],
                     trigger_mode="always_last", show_progress="hidden")
    tag_ac_ds.click(do_tag_complete, [prompt_txt, tag_ac_ds, tag_ac_state], [prompt_txt, tag_ac_ds, tag_ac_state])

    # ── Wildcards ───────────────────────────────────────────────────────────
    def do_wc_add(prompt, idx, names):
        try:
            tag = str(names[int(idx)]).strip()
        except (TypeError, ValueError, IndexError):
            return gr.update()
        return merge_prompts(prompt or "", tag) if tag else gr.update()

    def do_wc_preview(prompt, neg):
        from backend import wildcards as _wc
        import random as _r
        if not (_wc.is_dynamic(prompt) or _wc.is_dynamic(neg)):
            return ('<p style="color:#a6adc8;font-size:13px;">No {a|b} groups or __wildcards__ in the prompt — '
                    'every image gets the same prompt.</p>')
        rows, missing = [], []
        for _ in range(4):
            sd_ = _r.randint(0, 2**32 - 1)
            rows.append(f'<li><code>{html.escape(_wc.resolve(prompt, sd_, missing))}</code></li>')
        warn = ("<br>⚠ Unknown: " + ", ".join(f"<code>__{html.escape(m)}__</code>" for m in missing)
                if missing else "")
        return f'<ul style="font-size:13px;color:#cdd6f4;margin:2px 0;">{"".join(rows)}</ul>{warn}'

    def do_wc_refresh():
        from backend.wildcards import list_wildcards
        names = [f"__{n}__" for n in list_wildcards()]
        return gr.update(samples=[[n] for n in names] or [["-"]]), names

    wc_ds.click(do_wc_add, [prompt_txt, wc_ds, wc_names_state], [prompt_txt])
    wc_preview_btn.click(do_wc_preview, [prompt_txt, neg_prompt_txt], [wc_preview_html])
    wc_refresh_btn.click(do_wc_refresh, [], [wc_ds, wc_names_state])

    # ── WD14 interrogate ────────────────────────────────────────────────────
    def do_interrogate(img, prompt, thr, progress=gr.Progress()):
        if img is None:
            return gr.update(), '<p style="color:#f9e2af;font-size:13px;">⚠ Drop an image first.</p>'
        try:
            from backend.wd_tagger import tag_image, tags_text
            res = tag_image(img, general_threshold=float(thr or 0.35), progress=progress)
        except Exception as e:
            import traceback; traceback.print_exc()
            return gr.update(), f'<p style="color:#f38ba8;font-size:13px;">❌ Tagger failed: {html.escape(str(e))}</p>'
        tags = tags_text(res)
        rating = max(res["rating"].items(), key=lambda kv: kv[1])[0] if res["rating"] else "?"
        chars = ", ".join(f"{t} {p:.0%}" for t, p in res["character"]) or "none recognised"
        old = (f'<details><summary>previous prompt</summary><code>{html.escape(prompt)}</code></details>'
               if (prompt or "").strip() else "")
        return tags, (f'<p style="color:#a6e3a1;font-size:13px;margin:2px 0;">🏷 {len(res["general"])} tags · '
                      f'character: {html.escape(chars)} · rating: {rating}</p>{old}')

    tag_i2i_btn.click(do_interrogate, [init_image, prompt_txt, tag_thr_sl], [prompt_txt, tag_i2i_html])

    # ── Character cards ─────────────────────────────────────────────────────
    _NO_OUTFIT = "(no outfit tags)"

    def _card_summary(card) -> str:
        if not card:
            return ""
        loras = ", ".join(f"{Path(l['file']).stem} ×{l['weight']:g}" for l in card["loras"]) or "no LoRA"
        return (f'<p style="color:#a6adc8;font-size:13px;margin:2px 0;">🎴 <b>{html.escape(card["name"])}</b> · '
                f'{html.escape(card.get("checkpoint") or "current checkpoint")} · {html.escape(loras)} · '
                f'{len(card["outfits"])} outfit(s)'
                + (f' · 📌 {len(card["profiles"])} checkpoint profile(s)' if card.get("profiles") else "")
                + (f' · 🧬 face learned from {card["identity"]["n"]} pictures' if card.get("identity") else "")
                + f'<br><code>{html.escape(card["tags"])}</code></p>')

    def on_card_pick(name):
        from backend.character_cards import load_card
        card = load_card(name) if name and name != "(none)" else None
        if not card:
            return gr.update(choices=[_NO_OUTFIT], value=_NO_OUTFIT), gr.update(), ""
        outs = [_NO_OUTFIT] + list(card["outfits"])
        return (gr.update(choices=outs, value=outs[1] if len(outs) > 1 else _NO_OUTFIT),
                card["name"], _card_summary(card))

    def _find_file(fname, items):
        want = (fname or "").lower()
        return next((path for n, path in items if n.lower() == want), None) if want else None

    def do_card_load(name, outfit, scene, neg, cur_model=None):
        from backend.character_cards import load_card, card_prompt, card_for_checkpoint
        from backend.model_manager import list_checkpoints, list_loras, list_vaes
        n_out = 25
        card = load_card(name) if name and name != "(none)" else None
        if not card:
            return (*[gr.update()] * n_out, '<p style="color:#f9e2af;font-size:13px;">⚠ Pick a card first.</p>')
        card, profile = card_for_checkpoint(card, cur_model)    # the checkpoint selected now may have its own profile
        notes = []
        model = gr.update()
        if card.get("checkpoint") and not profile:               # a profile keeps the checkpoint the user picked
            path = _find_file(card["checkpoint"], list_checkpoints())
            if path:
                model = path
            else:
                notes.append(f"checkpoint {card['checkpoint']} isn't installed (kept the current one)")
        vae = gr.update()
        if card.get("vae"):
            vae = _find_file(card["vae"], list_vaes()) or gr.update()
        lora_vals = []
        loras = list_loras()
        for l in (card["loras"] + [None] * 3)[:3]:
            if l is None:
                lora_vals += ["none", gr.update()]
                continue
            path = _find_file(l["file"], loras)
            if not path:
                notes.append(f"LoRA {l['file']} isn't installed")
                lora_vals += ["none", gr.update()]
            else:
                lora_vals += [path, max(0.1, min(1.5, l["weight"]))]
        prompt = card_prompt(card, outfit if outfit != _NO_OUTFIT else None, scene or "")
        negative = merge_prompts(neg or "", card["negative"]) if card.get("negative") else gr.update()
        g = lambda k: card[k] if k in card else gr.update()
        # boosters only when the profile has them (face / hand detail: 0 = off, else the denoise strength)
        fd, hd, ed = card.get("face_detail"), card.get("hand_detail"), card.get("eye_detail")
        boost = [g("pag"), g("cfg_rescale"), g("freeu"),
                 gr.update() if fd is None else fd > 0, gr.update() if not fd else fd,
                 gr.update() if hd is None else hd > 0, gr.update() if not hd else hd,
                 gr.update() if ed is None else ed > 0, gr.update() if not ed else ed]
        if card.get("eye_style"):                                  # the card's own eye look (round / natural); absent = leave the current one
            from backend.detail_tools import set_eye_style
            notes.append(f"👁 eye style {set_eye_style(card['eye_style'])}")
        msg = (f'<p style="color:#a6e3a1;font-size:13px;margin:2px 0;">✅ Loaded {html.escape(card["name"])}'
               + (f' — {html.escape(outfit)}' if outfit and outfit != _NO_OUTFIT else "")
               + (f' · 📌 profile for {html.escape(profile)}' if profile else "") + "</p>"
               + (f'<p style="color:#f9e2af;font-size:13px;margin:2px 0;">⚠ {html.escape("; ".join(notes))}</p>'
                  if notes else ""))
        return (model, vae, *lora_vals, prompt, negative, g("width"), g("height"), g("cfg"), g("steps"),
                g("scheduler") if card.get("scheduler") in SCHEDULER_MAP else gr.update(),
                g("clip_skip") if card.get("clip_skip") in (1, 2) else gr.update(), *boost, msg)

    def _card_choices(value):
        from backend.character_cards import list_cards
        return gr.update(choices=["(none)"] + list_cards(), value=value)

    def do_card_save(name, model, vae, l1, w1, l2, w2, l3, w3, prompt, neg, w, h, cfg, steps, sched, cs):
        from backend.character_cards import load_card, save_card, safe_name
        n = safe_name(name)
        if not n:
            return gr.update(), gr.update(), '<p style="color:#f9e2af;font-size:13px;">⚠ Enter a card name.</p>'
        old = load_card(n) or {}
        card = {"name": n, "checkpoint": Path(str(model)).name if model and Path(str(model)).is_file() else None,
                "vae": Path(str(vae)).name if vae and vae != "none" and Path(str(vae)).is_file() else None,
                "loras": [{"file": Path(str(l)).name, "weight": wt} for l, wt in ((l1, w1), (l2, w2), (l3, w3))
                          if l and l != "none"],
                "tags": prompt or "", "outfits": old.get("outfits") or {}, "negative": neg or "",
                "width": w, "height": h, "cfg": cfg, "steps": steps, "scheduler": sched, "clip_skip": cs,
                "profiles": old.get("profiles") or {}, "identity": old.get("identity"), "notes": old.get("notes")}
        try:
            save_card(card)
        except (OSError, ValueError) as e:
            return gr.update(), gr.update(), f'<p style="color:#f38ba8;font-size:13px;">❌ {html.escape(str(e))}</p>'
        outs = [_NO_OUTFIT] + list(card["outfits"])
        return (_card_choices(n), gr.update(choices=outs, value=_NO_OUTFIT),
                f'<p style="color:#a6e3a1;font-size:13px;">💾 Saved card <b>{html.escape(n)}</b> (the whole '
                f'prompt is its tags; outfits kept). Edit settings/characters/{html.escape(n)}.json to fine-tune.</p>')

    def _eye_style_now() -> str:
        from backend.detail_tools import EYE_STYLE
        return EYE_STYLE["mode"]

    def do_card_profile(name, model, vae, l1, w1, l2, w2, l3, w3, neg, w, h, cfg, steps, sched, cs,
                        pag, freeu, rescale, fd_on, fd_den, hd_on, hd_den, ed_on=False, ed_den=0.4):
        """Remember the current LoRA weights, settings and boosters as the card's profile for the selected
        checkpoint (🎴 Load applies it whenever that checkpoint is selected)."""
        from backend.character_cards import load_card, save_card
        card = load_card(name) if name and name != "(none)" else None
        if not card:
            return '<p style="color:#f9e2af;font-size:13px;">⚠ Pick a card first.</p>'
        if not (model and Path(str(model)).is_file()):
            return '<p style="color:#f9e2af;font-size:13px;">⚠ Select the checkpoint this profile is for first.</p>'
        key = Path(str(model)).name
        card["profiles"] = {**(card.get("profiles") or {}), key: {
            "loras": [{"file": Path(str(l)).name, "weight": wt} for l, wt in ((l1, w1), (l2, w2), (l3, w3)) if l and l != "none"],
            "vae": Path(str(vae)).name if vae and vae != "none" and Path(str(vae)).is_file() else None,
            "negative": neg or "", "width": w, "height": h, "cfg": cfg, "steps": steps, "scheduler": sched,
            "clip_skip": cs, "pag": pag or 0.0, "freeu": bool(freeu), "cfg_rescale": rescale or 0.0,
            "face_detail": (fd_den or 0.0) if fd_on else 0.0, "hand_detail": (hd_den or 0.0) if hd_on else 0.0,
            "eye_detail": (ed_den or 0.0) if ed_on else 0.0, "eye_style": _eye_style_now()}}
        try:
            save_card(card)
        except (OSError, ValueError) as e:
            return f'<p style="color:#f38ba8;font-size:13px;">❌ {html.escape(str(e))}</p>'
        return (f'<p style="color:#a6e3a1;font-size:13px;">📌 Saved the profile of <b>{html.escape(card["name"])}</b> for '
                f'<b>{html.escape(key)}</b> — 🎴 Load applies it whenever this checkpoint is selected.</p>')

    def do_card_identity(name, files, progress=gr.Progress()):
        """🧬 Learn the selected card's look from pictures of the character (CPU: anime face detector + CCIP); the ⭐ rating of an
        outfit batch then flags a face that is further from them than her own pictures are. The first use downloads CCIP (150 MB, OpenRAIL)."""
        from PIL import Image as _I
        from backend.character_cards import load_card, save_card
        from backend import ccip as _ccip, identity_score as _ids
        warn = lambda m: f'<p style="color:#f9e2af;font-size:13px;">⚠ {m}</p>'
        card = load_card(name) if name and name != "(none)" else None
        if not card:
            return warn("Pick a card first.")
        paths = [getattr(f, "name", f) for f in (files or [])]
        if len(paths) < _ids.MIN_REFS:
            return warn(f"Add at least {_ids.MIN_REFS} pictures of her.")
        if not _ccip.available():
            progress(0, desc=f"Downloading CCIP ({_ccip.SIZE_MB} MB, once)…")
            try:
                _ccip.download()
            except Exception as e:
                return warn(f"Could not download CCIP: {html.escape(str(e).splitlines()[0][:160] if str(e) else type(e).__name__)}")

        def pictures():
            for i, p in enumerate(paths):
                progress(i / len(paths), desc=f"Reading picture {i + 1}/{len(paths)}")
                try:
                    with _I.open(p) as im:
                        yield im.convert("RGB")
                except Exception:
                    continue
        ident = _ids.build_identity(pictures())
        if not ident:
            return warn(f"Fewer than {_ids.MIN_REFS} of those pictures have a face of at least {_ids.MIN_FACE_PX} px that the anime face "
                        f"detector finds — nothing learned.")
        card["identity"] = ident
        try:
            save_card(card)
        except (OSError, ValueError) as e:
            return f'<p style="color:#f38ba8;font-size:13px;">❌ {html.escape(str(e))}</p>'
        return (f'<p style="color:#a6e3a1;font-size:13px;">🧬 Learned <b>{html.escape(card["name"])}</b> from {ident["n"]} of {len(paths)} '
                f'pictures (no face, or under {_ids.MIN_FACE_PX} px, are skipped; they agree to {ident["mean"]:.2f} ± {ident["sd"]:.2f}). '
                f'“🎴▶ Generate every outfit” now flags a face that is further from her than her own pictures are — about 1 in 30 of hers, '
                f'up to 1 in 5 of a no-LoRA look-alike, another girl every time.</p>')

    def do_card_build(name, lora, weight, model):
        from backend.character_cards import card_from_lora, load_card, save_card, safe_name
        if not lora or lora == "none":
            return gr.update(), gr.update(), gr.update(), \
                '<p style="color:#f9e2af;font-size:13px;">⚠ Pick the character LoRA in slot 1 first.</p>'
        card = card_from_lora(lora, float(weight or 0.8),
                              model if model and Path(str(model)).is_file() else None,
                              name=safe_name(name) or None)
        if not card:
            return gr.update(), gr.update(), gr.update(), \
                '<p style="color:#f38ba8;font-size:13px;">❌ Could not read that LoRA.</p>'
        if load_card(card["name"]):
            return gr.update(), gr.update(), gr.update(), (
                f'<p style="color:#f9e2af;font-size:13px;">⚠ A card called {html.escape(card["name"])} already '
                f'exists — type another name to build a new one.</p>')
        save_card(card)
        outs = [_NO_OUTFIT] + list(card["outfits"])
        return (_card_choices(card["name"]), gr.update(choices=outs, value=outs[1] if len(outs) > 1 else _NO_OUTFIT),
                card["name"], _card_summary(card).replace("🎴", "🧩 Built"))

    # .input, not .change: Save/Build set the dropdown themselves and report their own result
    card_dd.input(on_card_pick, [card_dd], [outfit_dd, card_name_txt, card_status])
    card_load_btn.click(
        do_card_load,
        [card_dd, outfit_dd, card_scene_txt, neg_prompt_txt, model_dd],
        [model_dd, vae_dd, lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3,
         prompt_txt, neg_prompt_txt, width_sl, height_sl, cfg_sl, steps_sl, scheduler_dd, clip_skip_rb,
         pag_sl, cfg_rescale_sl, freeu_cb, fd_cb, fd_denoise_sl, hd_cb, hd_denoise_sl, ed_cb, ed_denoise_sl,
         card_status])
    card_profile_btn.click(
        do_card_profile,
        [card_dd, model_dd, vae_dd, lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3,
         neg_prompt_txt, width_sl, height_sl, cfg_sl, steps_sl, scheduler_dd, clip_skip_rb,
         pag_sl, freeu_cb, cfg_rescale_sl, fd_cb, fd_denoise_sl, hd_cb, hd_denoise_sl, ed_cb, ed_denoise_sl],
        [card_status])
    card_save_btn.click(
        do_card_save,
        [card_name_txt, model_dd, vae_dd, lora_dd, lora_weight, lora_dd2, lora_weight2, lora_dd3, lora_weight3,
         prompt_txt, neg_prompt_txt, width_sl, height_sl, cfg_sl, steps_sl, scheduler_dd, clip_skip_rb],
        [card_dd, outfit_dd, card_status])
    card_ident_btn.click(do_card_identity, [card_dd, card_ident_files], [card_status])
    card_build_btn.click(do_card_build, [card_name_txt, lora_dd, lora_weight, model_dd],
                         [card_dd, outfit_dd, card_name_txt, card_status])

    # ── Outfit batch: every outfit of a card × N seeds as one job (resumable) ──
    @gpu_job
    def do_card_batch(card_name, scene, seeds_per, resume, model_path, vae_path, lora1, w1, lora2, w2, lora3, w3,
                      auto_quality, prompt, neg_prompt, scheduler, steps, cfg, width, height, batch, seed, init_img,
                      strength, use_i2i, clip_skip, var_seed, var_strength, hires_on, hires_scale, hires_denoise,
                      hires_steps, hires_upscaler, fd_on, fd_denoise, fd_mode, fd_prompt, pag_scale, freeu,
                      cfg_rescale, hd_on, hd_denoise, ed_on=False, ed_denoise=0.4, progress=gr.Progress()):
        """Generate every outfit of the card with the current settings (after 🎴 Load) and the same seeds for
        each outfit, then a labelled contact sheet. Finished outfit/seed pairs are remembered per card
        (outputs/card_batches/<card>.json) and skipped next time while the settings are unchanged."""
        import hashlib, random
        from PIL import Image as _I
        from backend.character_cards import load_card, card_prompt, safe_name
        _generation_abort.clear()
        keep = gr.update()
        say = lambda m, c="#f38ba8": [gr.update(), f'<p style="color:{c};font-size:13px;">{m}</p>', keep, keep,
                                      keep, keep]
        card = load_card(card_name) if card_name and card_name != "(none)" else None
        if not card:
            yield say("❌ Pick a character card first.")
            return
        outfits = list(card.get("outfits") or {}) or [None]
        n_seeds = int(min(8, max(1, _num(seeds_per, 2))))
        if len(outfits) * n_seeds > 96:
            yield say(f"❌ {len(outfits)} outfits × {n_seeds} seeds = {len(outfits) * n_seeds} images — keep it at 96 "
                      f"or fewer.")
            return
        w1, w2, w3 = (_num(w, 0.8) for w in (w1, w2, w3))
        ok, status = _ensure_model(model_path, vae_path, progress)
        if not ok:
            yield [gr.update(), status if isinstance(status, str) else "", keep, keep, status, keep]
            return
        err = _sync_loras([(lora1, w1), (lora2, w2), (lora3, w3)], progress)
        if err:
            yield say(f"❌ LoRA problem: {err}")
            return
        extra = _clean_extra(dict(clip_skip=clip_skip, var_seed=var_seed, var_strength=var_strength, hires_on=hires_on,
                                  hires_scale=hires_scale, hires_denoise=hires_denoise, hires_steps=hires_steps,
                                  hires_upscaler=hires_upscaler, fd_on=fd_on, fd_denoise=fd_denoise, fd_mode=fd_mode,
                                  fd_prompt=fd_prompt, pag_scale=pag_scale, freeu=freeu, cfg_rescale=cfg_rescale,
                                  hd_on=hd_on, hd_denoise=hd_denoise, ed_on=ed_on, ed_denoise=ed_denoise))
        idx_path = OUTPUTS_DIR / "card_batches" / f"{safe_name(card['name']) or 'card'}.json"
        try:
            index = _json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
        except Exception:
            index = {}
        same = _json.dumps([Path(str(model_path)).name, vae_path, [(Path(str(l)).name, w) for l, w in
                            ((lora1, w1), (lora2, w2), (lora3, w3)) if l and l != "none"], neg_prompt, scheduler,
                            steps, cfg, width, height, bool(auto_quality), extra], sort_keys=True, default=str)
        # seed -1 picks a random base seed — which made "Skip pairs already made" useless after a Stop or a
        # crash (every run drew new seeds). While a batch of these exact settings is unfinished, its base seed
        # is remembered in the index ("__pending__") and reused when Skip is on; a finished batch forgets it.
        pend = index.get("__pending__") if isinstance(index.get("__pending__"), dict) else {}
        pend_key = hashlib.sha1(f"{same}|{card['name']}|{scene or ''}|{n_seeds}".encode("utf-8")).hexdigest()[:16]
        user_seed = int(_num(seed, -1))
        track = user_seed < 0 and bool(resume)          # a random base seed that Skip should be able to find again
        if user_seed >= 0:
            base = user_seed
        elif resume and str(pend.get(pend_key, "")).isdigit():
            base = int(pend[pend_key]) % 2**32
        else:
            base = random.randint(0, 2**32 - 1)
        seeds = [(base + k) % 2**32 for k in range(n_seeds)]
        if track:
            pend[pend_key] = base
            index["__pending__"] = pend
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            idx_path.write_text(_json.dumps(index, indent=1), encoding="utf-8")
        cells, cell_seeds, t0, made, reused = [], [], time.time(), 0, 0
        total = len(outfits) * n_seeds
        try:
            for oi, outfit in enumerate(outfits):
                p = card_prompt(card, outfit, scene or "")
                for s_ in seeds:
                    if _generation_abort.is_set():
                        raise _GenerationAborted()
                    key = hashlib.sha1(f"{same}|{p}|{s_}".encode("utf-8")).hexdigest()[:16]
                    old = OUTPUTS_DIR / index.get(key, "~")
                    if resume and key in index and old.is_file():
                        im = _I.open(old); im.load(); im.info["saved_path"] = str(old)
                        cells.append(im); cell_seeds.append(s_); reused += 1
                        continue
                    progress(len(cells) / total, desc=f"{outfit or card['name']} · seed {s_} "
                                                      f"({len(cells) + 1}/{total})")
                    imgs, info, _ = do_generate(p, neg_prompt, scheduler, steps, cfg, width, height, 1, s_, None,
                                                0.5, False, auto_quality=auto_quality, extra=dict(extra))
                    if not imgs:
                        if _generation_abort.is_set():
                            raise _GenerationAborted()
                        yield [cells or gr.update(), info, keep, keep, keep, keep]
                        return
                    im = imgs[0]
                    cells.append(im); cell_seeds.append(s_); made += 1
                    saved = (getattr(im, "info", None) or {}).get("saved_path")
                    # a Stop / error inside hires, hand or face detail leaves the previous stage's image saved
                    # ("…saved the images from before it"): show it, but don't remember it as done — Skip would
                    # keep it without that pass for good
                    if saved and "saved the images from before it" not in info:
                        index[key] = Path(saved).name
                        idx_path.parent.mkdir(parents=True, exist_ok=True)
                        idx_path.write_text(_json.dumps(index, indent=1), encoding="utf-8")
                    yield [list(cells), f'<p style="color:#89dceb;font-size:13px;">🎴 {len(cells)}/{total} · '
                                        f'{html.escape(outfit or card["name"])} · seed {s_}</p>',
                           list(cells), list(cell_seeds), keep, keep]
        except _GenerationAborted:
            if not cells:
                yield say("⏹ Outfit batch stopped.", "#fab387")
                return
        if track and not _generation_abort.is_set() and len(cells) == total and pend.pop(pend_key, None) is not None:
            idx_path.write_text(_json.dumps(index, indent=1), encoding="utf-8")      # finished: forget the seed
        rows = (len(cells) + n_seeds - 1) // n_seeds
        n_real = len(cells)
        # ⭐ rating (CPU, only the checks whose models are already cached): one face / ≤ 2 hands, the card's hair / eye
        # colours and hair pin on the head crop (scene colour leaking into the eyes was the first thing to go), colour
        # noise. Written onto the contact sheet; the pictures themselves stay clean.
        scored, sheet_cells, looked, face_checked = [], list(cells), False, False
        score_pause = _score_pause(progress)
        try:
            from backend import image_score as _is
            can = _is.available()
            if n_real and any(can.values()):
                scored = _is.score_images(cells[:n_real], card.get("tags", ""), identity=card.get("identity"), pause=score_pause,
                                          identity_z=_identity_margins(card, outfits, n_seeds, n_real))
                sheet_cells = [_is.badge(c, r) for c, r in zip(cells, scored)]
                looked = bool(can["look"] and _is.ic.traits(card.get("tags", "")))
                from backend import identity_score as _ids
                face_checked = bool(card.get("identity") and card["identity"].get("model") == _ids.MODEL_TAG
                                    and can.get("identity") and can.get("faces"))      # learned with another model: not judged, so no "match"
        except Exception as e:
            print(f"[Score] skipped: {e}")
        while len(cells) % n_seeds:                     # stopped mid-row: pad with blanks (contact sheet only)
            cells.append(_I.new("RGB", cells[0].size, (30, 30, 46)))
            sheet_cells.append(cells[-1])
        labels = [o or "(no outfit)" for o in outfits][:rows]      # rows follow the outfit order
        sheet = _xy_grid(sheet_cells, [f"seed {s_}" for s_ in seeds], labels,
                         f"{card['name']} · {len(labels)} outfit(s) × {n_seeds} seed(s)")
        spath = _unique_output(f"outfits_{safe_name(card['name']) or 'card'}_{int(time.time())}")
        sheet.save(spath)
        sheet.info["saved_path"] = str(spath)
        shown = [sheet] + cells[:n_real]
        ident = ""
        if scored:
            bad = [(outfits[i // n_seeds] or card["name"], cell_seeds[i], r["flags"]) for i, r in enumerate(scored)
                   if r["flags"] and i < len(cell_seeds)]
            ident = f"<br>⭐ {sum(r['stars'] for r in scored) / len(scored):.1f}/5 on average"
            if bad:
                ident += ('<br><span style="color:#f9e2af;">⚠ possibly off-model or damaged: '
                          + "; ".join(f'{html.escape(str(o))} · seed {s_} ({html.escape(_flag_text(card, o, f))})' for o, s_, f in bad[:8])
                          + (f" … and {len(bad) - 8} more" if len(bad) > 8 else "") + "</span>")
            elif looked and face_checked:
                ident += f"<br>✅ hair / eye colours (WD14) and faces (CCIP) match the card in all {n_real} image(s)"
            elif looked:
                ident += f"<br>✅ hair / eye colours match the card in all {n_real} image(s) (WD14)"
            elif face_checked:
                ident += f"<br>✅ faces match the card in all {n_real} image(s) (CCIP)"
            if score_pause.waited[0] >= 1:
                ident += f"<br>🌡 the rating paused {score_pause.waited[0]:.0f} s while the laptop cooled down"
        msg = (f'<p style="color:#a6e3a1;font-size:13px;">🎴 {html.escape(card["name"])}: {made} new + {reused} '
               f'reused image(s) in {_fmt_elapsed(time.time() - t0)} — contact sheet {spath.name}'
               + (" (stopped early)" if _generation_abort.is_set() else "") + ident + "</p>")
        yield [shown, msg, shown, [cell_seeds[0] if cell_seeds else base] + cell_seeds, status, _active_loras_html()]

    card_batch_btn.click(
        lambda: (None, 0), None, [selected_gallery_image, selected_idx_state],
    ).then(
        do_card_batch, [card_dd, card_scene_txt, card_batch_seeds, card_batch_resume] + _gen_inputs,
        [output_gallery, gen_info, last_generated_images, last_seed_state, model_status, active_loras_html],
    )

    gen_controls = {
        "model": model_dd,
        "vae": vae_dd,
        "extra": [clip_skip_rb, var_seed_num, var_strength_sl, hires_cb, hires_scale_sl,
                  hires_denoise_sl, hires_steps_sl, hires_up_dd, fd_cb, fd_denoise_sl, fd_mode_rb,
                  fd_prompt_txt, pag_sl, freeu_cb, cfg_rescale_sl, hd_cb, hd_denoise_sl, ed_cb, ed_denoise_sl],
        "lora_dd1": lora_dd,
        "lora_w1": lora_weight,
        "lora_w2": lora_weight2,
        "lora_w3": lora_weight3,
        "lora_dd2": lora_dd2,
        "lora_dd3": lora_dd3,
        "prompt": prompt_txt,
        "neg_prompt": neg_prompt_txt,
        "scheduler": scheduler_dd,
        "steps": steps_sl,
        "cfg": cfg_sl,
        "width": width_sl,
        "height": height_sl,
        "seed": seed_num,
        "init_image": init_image,
        "use_i2i": use_i2i_cb,
        "gallery": output_gallery,
        "last_images": last_generated_images,
        "selected_image": selected_gallery_image,
        "gal_send_up_btn": gal_send_up_btn,
        "gal_status": gal_status,
    }
    return model_dd, lora_dd, vae_dd, gen_controls


# ═════════════════════════════════════════════════════════════════════════════
# Tab: SD 1.5 → SDXL Bridge
# ═════════════════════════════════════════════════════════════════════════════
def _build_bridge_tab():
    with gr.Tab("🌉 SD→SDXL Bridge"):
        gr.HTML(
            '<p style="color:#a6adc8;font-size:13px;">'
            '<b>SD 1.5 → SDXL img2img Bridge</b> — Use your SD 1.5 LoRAs to generate a base '
            'image, then automatically refine it with SDXL for higher detail and resolution. '
            'This is the best way to use SD 1.5 character LoRAs at SDXL quality.</p>'
        )
        with gr.Accordion("📖 How the Bridge Works", open=False):
            gr.HTML('''
<div style="font-size:13px;color:#cdd6f4;line-height:1.5;">
<ol style="padding-left:18px;margin:4px 0;">
<li><b>Stage 1 — SD 1.5 generates a base image</b> at 512×512 (or 512×768) using your SD 1.5 model + LoRA.
   The character/style from your LoRA is baked into this base.</li>
<li><b>Stage 2 — SDXL refines via img2img</b> at higher resolution (832×1216, etc.) with a low denoise
   strength (0.3–0.5). SDXL "redraws" the SD 1.5 output, adding its superior detail while
   preserving the character likeness from your LoRA.</li>
</ol>
<p style="margin:4px 0;"><b>Best practices:</b></p>
<ul style="padding-left:18px;margin:4px 0;">
<li><b>Denoise 0.30–0.40</b>: Preserves character likeness best. Higher = more SDXL influence.</li>
<li><b>Same prompt</b> for both stages (auto-applied). Adjust SDXL prompt if needed.</li>
<li><b>CFG 5–7</b> for SDXL stage. Lower than standalone SDXL since the base image provides structure.</li>
</ul>
</div>''')

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Stage 1 — SD 1.5 Base")
                from backend.sd_pipeline import _is_sdxl as _br_is_xl
                _br_all = _refresh_checkpoints()
                _br_sd15 = [c for c in _br_all if not _br_is_xl(c[1])]
                _br_xl = [c for c in _br_all if _br_is_xl(c[1])]
                _bs = _read_last_bridge()   # restore the last Bridge run's settings

                def _bsv(key, default, allowed=None):
                    v = _bs.get(key, default)
                    return v if (allowed is None or v in allowed) else default
                br_sd15_model = gr.Dropdown(
                    choices=_br_sd15,
                    value=_bsv("sd15_model", _br_sd15[0][1] if _br_sd15 else "",
                               [c[1] for c in _br_sd15]),
                    label="SD 1.5 Model",
                    info="Loaded automatically if it isn't already. LoRAs you applied to this "
                         "same model in the Generate tab are kept.",
                )
                br_prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    value=_bsv("prompt", ""),
                    placeholder="Character trigger, description, quality tags…",
                    info="Used for both stages. SDXL prompt can be overridden below.",
                )
                br_neg = gr.Textbox(
                    label="Negative Prompt",
                    lines=2,
                    value=_bsv("neg", DEFAULT_NEGATIVE),
                )
                with gr.Row():
                    br_sd15_steps = gr.Slider(10, 60, value=_bsv("sd15_steps", 30), step=1, label="SD 1.5 Steps")
                    br_sd15_cfg   = gr.Slider(1, 15, value=_bsv("sd15_cfg", 7.5), step=0.5, label="SD 1.5 CFG")
                with gr.Row():
                    br_sd15_w = gr.Slider(256, 768, value=_bsv("sd15_w", 512), step=64, label="Width")
                    br_sd15_h = gr.Slider(256, 768, value=_bsv("sd15_h", 768), step=64, label="Height")
                br_sd15_sched = gr.Dropdown(
                    choices=list(SCHEDULER_MAP.keys()),
                    value=_bsv("sd15_sched", "DPM++ 2M Karras", list(SCHEDULER_MAP)),
                    label="Scheduler",
                )
                br_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

            with gr.Column(scale=1):
                gr.Markdown("### Stage 2 — SDXL Refinement")
                br_sdxl_model = gr.Dropdown(
                    choices=_br_xl,
                    value=_bsv("sdxl_model", _br_xl[0][1] if _br_xl else "", [c[1] for c in _br_xl]),
                    label="SDXL / Pony / Illustrious Model",
                    info="Loaded automatically if it isn't already. On 12 GB cards the SD 1.5 "
                         "model is freed for this stage and reloads (with its LoRAs) next run.",
                )
                br_sdxl_prompt = gr.Textbox(
                    label="SDXL Prompt Override (optional)",
                    lines=2,
                    value=_bsv("sdxl_prompt", ""),
                    placeholder="Leave empty to reuse Stage 1 prompt",
                    info="If set, this replaces the prompt for SDXL. Otherwise Stage 1 prompt is used.",
                )
                with gr.Row():
                    br_sdxl_steps = gr.Slider(10, 60, value=_bsv("sdxl_steps", 30), step=1, label="SDXL Steps")
                    br_sdxl_cfg   = gr.Slider(1, 12, value=_bsv("sdxl_cfg", 6.0), step=0.5, label="SDXL CFG")
                with gr.Row():
                    br_sdxl_w = gr.Slider(512, 1536, value=_bsv("sdxl_w", 832), step=64, label="Target Width")
                    br_sdxl_h = gr.Slider(512, 1536, value=_bsv("sdxl_h", 1216), step=64, label="Target Height")
                br_denoise = gr.Slider(
                    0.1, 0.8, value=_bsv("denoise", 0.35), step=0.05,
                    label="SDXL Denoise Strength",
                    info="How much SDXL changes the SD 1.5 base. "
                         "0.30 = subtle refinement (best for character likeness). "
                         "0.50 = moderate redraw. 0.70+ = heavy reinterpretation.",
                )
                br_sdxl_sched = gr.Dropdown(
                    choices=list(SCHEDULER_MAP.keys()),
                    value=_bsv("sdxl_sched", "DPM++ 2M Karras", list(SCHEDULER_MAP)),
                    label="SDXL Scheduler",
                )

        with gr.Row():
            br_run_btn  = gr.Button("🌉 Run Bridge (SD 1.5 → SDXL)", variant="primary", size="lg")
            br_stop_btn = gr.Button("⏹ Stop", variant="secondary", size="sm")
        br_status = gr.HTML("")

        with gr.Row():
            br_stage1_out = gr.Image(label="Stage 1 — SD 1.5 Base", height=400)
            br_stage2_out = gr.Image(label="Stage 2 — SDXL Refined", height=400)
        br_gallery = gr.Gallery(label="Bridge Results", columns=4, height=300,
                                show_download_button=True)

        # ── Bridge callback ──────────────────────────────────────────────────
        def do_bridge(
            sd15_model, prompt, neg, sd15_steps, sd15_cfg, sd15_w, sd15_h, sd15_sched,
            seed, sdxl_model_choice, sdxl_prompt, sdxl_steps, sdxl_cfg,
            sdxl_w, sdxl_h, denoise, sdxl_sched, progress=gr.Progress(),
        ):
            _generation_abort.clear()
            try:   # remembered for the next launch (seed is not — it starts random)
                _LAST_BRIDGE_FILE.write_text(_json.dumps(dict(
                    sd15_model=sd15_model, prompt=prompt, neg=neg, sd15_steps=sd15_steps,
                    sd15_cfg=sd15_cfg, sd15_w=sd15_w, sd15_h=sd15_h, sd15_sched=sd15_sched,
                    sdxl_model=sdxl_model_choice, sdxl_prompt=sdxl_prompt, sdxl_steps=sdxl_steps,
                    sdxl_cfg=sdxl_cfg, sdxl_w=sdxl_w, sdxl_h=sdxl_h, denoise=denoise,
                    sdxl_sched=sdxl_sched), indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                print(f"[Bridge] Could not save settings: {e}")

            # same guards as Generate: multiples of 8, 32-bit seed, steps × denoise ≥ 1
            sd15_steps, sd15_cfg, sd15_w, sd15_h, _, seed_val, _, _ = _clean_gen_args(
                sd15_steps, sd15_cfg, sd15_w, sd15_h, 1, seed)
            sdxl_steps, sdxl_cfg, sdxl_w, sdxl_h, _, _, denoise, _ = _clean_gen_args(
                sdxl_steps, sdxl_cfg, sdxl_w, sdxl_h, 1, -1, denoise, img2img=True)
            prompt, neg = prompt or "", neg or ""

            def _vram_log(tag):
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        print(f"[Bridge] {tag}: allocated {_t.cuda.memory_allocated() / 2**30:.2f} GiB, "
                              f"reserved {_t.cuda.memory_reserved() / 2**30:.2f} GiB")
                except Exception:
                    pass

            # ── Stage 1: SD 1.5 txt2img ──────────────────────────────────────
            _vram_log("start")
            try:   # drop cached blocks left by the previous run's SDXL stage
                import torch as _t
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
            except Exception:
                pass
            if not sd15_model:
                return None, None, [], '<p style="color:#f38ba8;">❌ Pick an SD 1.5 model.</p>'
            if _sd15.current_model != sd15_model or _sd15.pipe is None:
                progress(0.0, desc=f"Loading {Path(str(sd15_model)).stem}…")
                msg = _sd15.load_model(str(sd15_model))
                if _sd15.pipe is None:
                    return None, None, [], f'<p style="color:#f38ba8;">{msg}</p>'
                # Re-apply the LoRAs this model had before the previous run freed it
                saved = _bridge_sd15_loras.get(str(sd15_model))
                if saved:
                    progress(0.0, desc="Re-applying SD 1.5 LoRAs…")
                    _sd15._lora_adapters = dict(saved)
                    _sd15._snapshot_clean_state()
                    try:
                        _sd15._reload_all_loras()
                    except Exception as e:
                        print(f"[Bridge] Could not re-apply LoRAs: {e}")
                        _sd15._lora_adapters = {}
            # <lora:name:0.8> tags in the Bridge prompt → apply them to the SD 1.5 model
            # (free slots; a LoRA already in a slot just gets the new weight)
            tag_loras, prompt = _loras_from_meta(prompt, None)
            tag_notes = []
            if tag_loras:
                from backend.model_manager import lora_arch
                local = {Path(n).stem.lower(): p for n, p in list_loras()}
                for name, w in tag_loras:
                    path = local.get(name.lower().removesuffix(".safetensors"))
                    if path is None:
                        tag_notes.append(f"{html.escape(name)} not found locally")
                        continue
                    if lora_arch(path) == "sdxl":
                        tag_notes.append(f"{Path(path).stem} is an SDXL LoRA — skipped for SD 1.5")
                        continue
                    slot = next((k for k, v in _sd15._lora_adapters.items() if v[1] == path),
                                next((k for k in range(3) if k not in _sd15._lora_adapters), None))
                    if slot is None:
                        tag_notes.append(f"no free LoRA slot for {Path(path).stem}")
                        continue
                    progress(0.0, desc=f"Applying LoRA {Path(path).stem}…")
                    res = _sd15.load_lora(path, min(1.5, max(0.1, w)), slot)
                    tag_notes.append(res.lstrip("✅ ").split(" | ")[0] if res.startswith("✅")
                                     else html.escape(res))
            progress(0.0, desc="Stage 1: Generating SD 1.5 base…")
            _vram_log("SD 1.5 ready")
            try:
                import torch as _t
                small_card = (_t.cuda.is_available() and
                              _t.cuda.get_device_properties(0).total_memory < 14 * 2**30)
            except Exception:
                small_card = False
            # With the SDXL model resident, SD 1.5's untiled VAE decode peaks at ~11.9 of
            # 12 GB and tips Windows into spilling; tiled decode leaves ~1 GB headroom.
            _sd15.force_tiled_decode = small_card and _sdxl.pipe is not None

            def _s1_prog(step, total):
                if _generation_abort.is_set():
                    raise _GenerationAborted()
                progress(0.0 + 0.4 * (step / total),
                         desc=f"Stage 1: Step {step}/{total}")

            try:
                s1_imgs, s1_info = _sd15.txt2img(
                    prompt, neg, int(sd15_w), int(sd15_h),
                    int(sd15_steps), float(sd15_cfg), seed_val, sd15_sched,
                    batch_size=1, step_callback=_s1_prog,
                )
            except _GenerationAborted:
                return None, None, [], '<p style="color:#fab387;">⏹ Stopped at Stage 1.</p>'
            except Exception as e:
                return None, None, [], f'<p style="color:#f38ba8;">❌ Stage 1 failed: {e}</p>'
            finally:
                _sd15.force_tiled_decode = False

            if not s1_imgs:
                return None, None, [], '<p style="color:#f38ba8;">❌ Stage 1 produced no images.</p>'

            base_image = s1_imgs[0]
            # Carry the seed actually used into stage 2 and the saved metadata
            seed_val = (_sd15.last_seeds or [seed_val])[0]
            # Stage 1 is saved now, while its model and LoRAs are still loaded (the record
            # reads them from the pipeline; on small cards SD 1.5 is freed below)
            s1_saved = _save_outputs([base_image], dict(
                mode="txt2img", prompt=prompt, negative_prompt=neg, steps=sd15_steps,
                cfg_scale=sd15_cfg, seeds=[seed_val], scheduler=sd15_sched,
                width=sd15_w, height=sd15_h, bridge_stage=1,
            ), pipe=_sd15)

            # SD 1.5 + SDXL + SDXL's working memory don't fit in 12 GB: Windows would
            # spill to system RAM (stage 2 went from ~6 s to ~60 s). Free SD 1.5 on
            # smaller cards; it reloads (with its LoRAs) at the start of the next run.
            if small_card:
                _bridge_sd15_loras[str(_sd15.current_model)] = dict(_sd15._lora_adapters)
                progress(0.44, desc="Freeing SD 1.5 VRAM for the SDXL stage…")
                _sd15._unload()
                _vram_log("SD 1.5 freed")
            else:
                _vram_log("SD 1.5 kept (large card)")

            # ── Stage 2: SDXL img2img ────────────────────────────────────────
            progress(0.45, desc="Stage 2: Preparing SDXL…")

            # Load the chosen SDXL model if it isn't the one in memory
            if _sdxl.pipe is None or (sdxl_model_choice and _sdxl.current_model != sdxl_model_choice):
                if sdxl_model_choice:
                    # Extract path from tuple if needed (dropdown returns tuple value)
                    model_path = sdxl_model_choice
                    if isinstance(model_path, (list, tuple)):
                        model_path = model_path[1] if len(model_path) > 1 else model_path[0]
                    progress(0.45, desc="Stage 2: Loading SDXL model…")
                    try:
                        _sdxl.load_model(str(model_path))
                    except Exception as e:
                        return (
                            base_image, None, [base_image],
                            f'<p style="color:#f38ba8;">❌ Failed to load SDXL model: {e}</p>'
                        )
                else:
                    return (
                        base_image, None, [base_image],
                        '<p style="color:#f38ba8;">❌ No SDXL model loaded. '
                        'Select one above or load in the Generate tab.</p>'
                    )

            _vram_log("SDXL ready")
            progress(0.48, desc="Stage 2: Refining with SDXL…")

            # Use override prompt if provided, else reuse stage 1 prompt
            xl_prompt = sdxl_prompt.strip() if sdxl_prompt and sdxl_prompt.strip() else prompt

            def _s2_prog(step, total):
                if _generation_abort.is_set():
                    raise _GenerationAborted()
                progress(0.45 + 0.5 * (step / total),
                         desc=f"Stage 2: Step {step}/{total}")

            try:
                # Resize base image to SDXL target
                from PIL import Image as _PILImage
                resized = base_image.resize((int(sdxl_w), int(sdxl_h)),
                                            _PILImage.LANCZOS)

                s2_imgs, s2_info = _sdxl.img2img(
                    resized, xl_prompt, neg, float(denoise),
                    int(sdxl_steps), float(sdxl_cfg), seed_val, sdxl_sched,
                    step_callback=_s2_prog,
                )
            except _GenerationAborted:
                return base_image, None, [base_image], '<p style="color:#fab387;">⏹ Stopped at Stage 2.</p>'
            except Exception as e:
                return base_image, None, [base_image], f'<p style="color:#f38ba8;">❌ Stage 2 failed: {e}</p>'

            refined = s2_imgs[0] if s2_imgs else None

            # Save stage 2 as the img2img it is (stage 1 was saved above)
            all_imgs = [base_image] + (s2_imgs or [])
            if s2_imgs:
                _save_outputs(s2_imgs, dict(
                    mode="img2img", prompt=xl_prompt, negative_prompt=neg, steps=sdxl_steps,
                    cfg_scale=sdxl_cfg, seeds=[seed_val], scheduler=sdxl_sched,
                    width=sdxl_w, height=sdxl_h, strength=denoise, bridge_stage=2,
                    source_image=s1_saved[0].name if s1_saved else None,
                ), pipe=_sdxl)

            progress(1.0, desc="✅ Bridge complete!")
            status = (
                f'<p style="color:#a6e3a1;">✅ Bridge complete! '
                f'Stage 1: {s1_info}<br>Stage 2: {s2_info}</p>'
            )
            if tag_notes:
                status += ('<p style="color:#89b4fa;font-size:13px;">🧬 LoRA tags from the prompt: '
                           + " · ".join(tag_notes) + "</p>")
            # stage 2 runs on whatever LoRAs the resident SDXL model has from the Generate tab (round 3, F5)
            xl_loras = [str(v[0]) for _, v in sorted(_sdxl._lora_adapters.items())] if s2_imgs else []
            if xl_loras:
                status += ('<p style="color:#f9e2af;font-size:13px;">ℹ Stage 2 used the SDXL LoRAs still applied from '
                           'the Generate tab: ' + html.escape(", ".join(xl_loras))
                           + " — clear them there for a plain SDXL refine.</p>")
            return base_image, refined, all_imgs, status

        br_event = br_run_btn.click(
            do_bridge,
            [
                br_sd15_model, br_prompt, br_neg, br_sd15_steps, br_sd15_cfg,
                br_sd15_w, br_sd15_h, br_sd15_sched, br_seed,
                br_sdxl_model, br_sdxl_prompt, br_sdxl_steps, br_sdxl_cfg,
                br_sdxl_w, br_sdxl_h, br_denoise, br_sdxl_sched,
            ],
            [br_stage1_out, br_stage2_out, br_gallery, br_status],
        )

        def _br_stop():
            _generation_abort.set()
            return '<p style="color:#fab387;">⏹ Stopping…</p>'

        br_stop_btn.click(_br_stop, [], [br_status])


# ═════════════════════════════════════════════════════════════════════════════
# Tab 2: Image Upscaling
# ═════════════════════════════════════════════════════════════════════════════
def _build_upscale_tab():
    with gr.Tab("🔍 Upscale", id="upscale"):
        gr.HTML(
            '<p style="color:#a6adc8;font-size:13px;">'
            'Upscale any image 2×–8× using Lanczos (CPU) or '
            'Real-ESRGAN (ONNX/PyTorch, auto-downloads ~67 MB model on first use).</p>'
        )

        with gr.Row():
            with gr.Column():
                up_input   = gr.Image(label="Input Image", type="pil")
                up_scale   = gr.Radio([2, 4, 8], value=4, label="Scale Factor",
                                      info="Multiplier for output resolution. 4× on a 512px image → 2048px.")
                up_method  = gr.Dropdown(
                    choices=upscaler.available_methods(),
                    value=upscaler.available_methods()[-1],
                    label="Upscale Method",
                    info="Lanczos = fast CPU-only. Real-ESRGAN = AI upscaler, "
                         "sharper details. ONNX variant uses DirectML GPU.",
                )
                _up_total_mb = get_total_vram_mb()
                _up_vram_bar = gr.HTML(
                    vram_bar_html(
                        estimate_upscale(512, 512, 4, upscaler.available_methods()[-1]),
                        _up_total_mb,
                        "Est. VRAM",
                    )
                )
                with gr.Row():
                    up_sd_cb = gr.Checkbox(
                        value=False, label="✨ SD detail pass (tiled, loaded model)", scale=2,
                        info="After upscaling, re-draws the image in native-size tiles at low denoise with the model "
                             "loaded on the Generate tab — real detail instead of smooth upscaling. Uses the image's "
                             "own prompt when it has one. ~10–30 s per tile.")
                    up_sd_den = gr.Slider(0.15, 0.5, value=0.3, step=0.05, label="Detail denoise", scale=1,
                                          info="0.25 subtle · 0.35 more detail · higher can put faces into tiles")
                upscale_btn = gr.Button("🔍 Upscale", variant="primary")
                up_info     = gr.HTML("")
                with gr.Accordion("📁 Batch — upscale a whole folder", open=False):
                    up_folder = gr.Textbox(label="Folder with images",
                                           placeholder=r"e.g. D:\pictures\to_upscale  (PNG / JPG / WebP)",
                                           info="Uses the scale and method above. Results go to "
                                                "outputs/upscaled_<folder>/ with the originals' settings kept.")
                    up_batch_btn = gr.Button("📁 Upscale folder", variant="secondary")
                    up_batch_info = gr.HTML("")

            with gr.Column():
                up_output = gr.Image(label="Upscaled Output", type="pil", show_download_button=True)

    def do_upscale(image, scale, method, sd_detail=False, sd_denoise=0.3, progress=gr.Progress()):
        if image is None:
            return None, '<p style="color:#f38ba8;">Please upload an image.</p>'

        def prog_cb(pct, msg):
            progress(pct, desc=msg)

        scale = max(1, int(_num(scale, 2)))
        ow, oh = image.size[0] * scale, image.size[1] * scale
        # Beyond ~64 MP the result is a multi-hundred-MB PNG that Pillow itself refuses to
        # reopen (decompression-bomb guard at 89 MP) — and it would take many minutes.
        if ow * oh > 64_000_000:
            return None, (f'<p style="color:#f38ba8;">❌ {image.size[0]}×{image.size[1]} at {scale}× would be '
                          f'{ow}×{oh} ({ow * oh / 1e6:.0f} MP) — too large. Use a smaller scale or '
                          f'a smaller image (up to ~64 MP output).</p>')
        if image.mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")
        try:
            out, info = upscaler.upscale(image, int(scale), method, prog_cb)
        except Exception as e:
            msg = html.escape((str(e).strip().splitlines() or [type(e).__name__])[0][:300])
            hint = " Try a smaller scale or Lanczos." if "memory" in msg.lower() else ""
            return None, f'<p style="color:#f38ba8;">❌ Upscale failed: {msg}.{hint}</p>'
        detail_note = ""
        if sd_detail:
            if sd.pipe is None:
                detail_note = ('<br><span style="color:#f9e2af;">✨ SD detail pass skipped: load a model on the '
                               'Generate tab first.</span>')
            elif out.size[0] * out.size[1] > 16_000_000:
                detail_note = ('<br><span style="color:#f9e2af;">✨ SD detail pass skipped: over 16 MP would take '
                               'too many tiles — use a smaller scale.</span>')
            else:
                from backend.detail_tools import tiled_detail
                from backend.png_info import read_image_metadata
                m = read_image_metadata(image) if getattr(image, "info", None) else {}
                prompt = m.get("prompt") or "masterpiece, best quality, highres, detailed"
                neg = m.get("negative_prompt") or "lowres, blurry, worst quality"
                sched = m.get("sampler") if m.get("sampler") in SCHEDULER_MAP else "DPM++ 2M"
                seed = int(_num(m.get("seed"), 0)) % 2**32
                t1 = time.time()
                _generation_abort.clear()

                def tcb(k, n, s_, t_):
                    if _generation_abort.is_set():
                        raise _GenerationAborted()
                    progress((k + s_ / max(1, t_)) / n, desc=f"Detail tile {k + 1}/{n}: step {s_}/{t_}")
                try:
                    out = tiled_detail(sd, out, prompt, neg, denoise=float(_num(sd_denoise, 0.3)), steps=20,
                                       cfg=6.0, seed=seed, scheduler=sched, clip_skip=int(_num(m.get("clip_skip"), 1)),
                                       step_callback=tcb)
                    method = f"{method} + SD detail {float(_num(sd_denoise, 0.3)):g}"
                    detail_note = (f'<br>✨ SD detail pass ({Path(str(sd.current_model)).stem}) · '
                                   f'{time.time() - t1:.0f}s')
                except _GenerationAborted:
                    detail_note = '<br><span style="color:#fab387;">⏹ SD detail pass stopped — saved the plain upscale.</span>'
                except Exception as e:
                    detail_note = (f'<br><span style="color:#f38ba8;">✨ SD detail pass failed: '
                                   f'{html.escape(str(e).splitlines()[0][:160] if str(e) else type(e).__name__)}</span>')
        # Auto-save, keeping the source's generation settings (A1111 text + our record) in the PNG
        src_info = getattr(image, "info", {}) or {}
        meta = _carry_params(image, f"Upscaled: {int(scale)}× {method}")
        # Name it after the generated source when known: 1790…_seed42_0_4x.png
        src = src_info.get("saved_path")
        base = f"{Path(src).stem}_{int(scale)}x" if src else f"upscale_{int(time.time())}"
        path = _unique_output(base)
        out.save(path, pnginfo=meta)
        out.info["saved_path"] = str(path)
        return out, (f'<p style="color:#a6adc8;font-size:13px;">{info}{detail_note}<br>'
                     f'Saved: outputs/{path.name}</p>')

    upscale_btn.click(do_upscale, [up_input, up_scale, up_method, up_sd_cb, up_sd_den], [up_output, up_info])

    def do_upscale_folder(folder, scale, method, progress=gr.Progress()):
        from PIL import Image as _Image
        src = Path(str(folder or "").strip().strip('"'))
        if not str(folder or "").strip() or not src.is_dir():
            return '<p style="color:#f38ba8;">❌ Enter an existing folder.</p>'
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        if not files:
            return '<p style="color:#f38ba8;">❌ No PNG / JPG / WebP images in that folder.</p>'
        scale = max(1, int(_num(scale, 2)))
        dest = OUTPUTS_DIR / f"upscaled_{_safe_name(src.name) or 'folder'}"
        dest.mkdir(parents=True, exist_ok=True)
        done, skipped, t0 = 0, [], time.time()
        for i, f in enumerate(files):
            progress(i / len(files), desc=f"{i + 1}/{len(files)}: {f.name}")
            try:
                with _Image.open(f) as im:
                    im.load()
                    meta = _carry_params(im, f"Upscaled: {scale}× {method}")
                    img = im.convert("RGB")
                if img.width * img.height * scale * scale > 64_000_000:
                    skipped.append(f"{f.name} (too large)")
                    continue
                out, _ = upscaler.upscale(img, scale, method)
                target = dest / f"{f.stem}_{scale}x.png"
                n = 1
                while target.exists():
                    target = dest / f"{f.stem}_{scale}x_{n}.png"; n += 1
                out.save(target, pnginfo=meta)
                done += 1
            except Exception as e:
                skipped.append(f"{f.name} ({html.escape(str(e)[:60])})")
        msg = (f'<p style="color:#a6e3a1;font-size:13px;">✅ Upscaled {done} of {len(files)} images in '
               f'{_fmt_elapsed(time.time() - t0)} → <code>outputs/{dest.name}/</code></p>')
        if skipped:
            msg += ('<p style="color:#f9e2af;font-size:13px;">Skipped: ' + ", ".join(skipped[:10])
                    + (" …" if len(skipped) > 10 else "") + "</p>")
        return msg

    up_batch_btn.click(do_upscale_folder, [up_folder, up_scale, up_method], [up_batch_info])

    def _update_up_vram(image, scale, method):
        w, h = (image.size if image is not None else (512, 512))
        used = estimate_upscale(w, h, int(scale), method or "")
        return vram_bar_html(used, _up_total_mb, "Est. VRAM")

    for _w in [up_scale, up_method]:
        _w.change(_update_up_vram, [up_input, up_scale, up_method], [_up_vram_bar])
    up_input.change(_update_up_vram, [up_input, up_scale, up_method], [_up_vram_bar])
    return up_input


# ═════════════════════════════════════════════════════════════════════════════
# Tab: PNG Info & Metadata Dispatch
# ═════════════════════════════════════════════════════════════════════════════
def _build_png_info_tab(gen_controls: dict, up_input: gr.Image):
    with gr.Tab("📄 PNG Info", id="pnginfo"):
        gr.HTML(
            '<p style="color:#a6adc8;font-size:13px;">'
            'Inspect generation parameters embedded in PNG images, and transfer '
            'them directly to <b>Generate</b>, <b>img2img</b>, or <b>Upscale</b>.</p>'
        )
        with gr.Row():
            with gr.Column(scale=1):
                png_input = gr.Image(label="Source Image", type="pil", sources=["upload", "clipboard"], height=420)
                with gr.Row():
                    send_gen_btn = gr.Button("🚀 Send to Generate", variant="primary")
                    send_i2i_btn = gr.Button("🖼 Send to img2img", variant="secondary")
                    send_up_btn = gr.Button("🔍 Send to Upscale", variant="secondary")
                png_tag_btn = gr.Button("🏷 Interrogate (WD14) → tags into the Generate prompt", size="sm")
                png_action_status = gr.HTML("")

            with gr.Column(scale=1):
                png_details_html = gr.HTML('<div style="color:#9399b2;padding:20px;text-align:center;">Drop or upload an image to inspect its metadata.</div>')
                with gr.Accordion("Raw Parameters", open=False):
                    png_raw_text = gr.Code(label="Raw Chunk Content", language="markdown", lines=8)

        def on_image_change(img):
            if img is None:
                return '<div style="color:#9399b2;padding:20px;text-align:center;">Drop or upload an image to inspect its metadata.</div>', ""
            meta = read_png_info(img)
            html = format_png_info_html(meta)
            return html, meta.get("raw_text") or "No parameters chunk found."

        png_input.change(on_image_change, [png_input], [png_details_html, png_raw_text])

        def do_send_to_gen(img):
            if img is None:
                return (*[gr.update()] * 16,
                        '<p style="color:#f38ba8;font-size:13px;">⚠ No image loaded in PNG Info.</p>',
                        *[gr.update()] * 19)
            plan = _restore_plan(read_png_info(img))
            u = lambda v: gr.update() if v is None else v
            lora_ups = [gr.update()] * 6
            if plan["loras"] is not None:
                slots = (plan["loras"] + [("none", None)] * 3)[:3]
                lora_ups = [v for path, w in slots for v in (path, gr.update() if w is None else w)]
            status_msg = ('<p style="color:#a6e3a1;font-size:13px;">✅ Sent to Generate: '
                          + (_plan_summary(plan) or "prompt only (no settings in this image)") + "</p>")
            return (u(plan["prompt"]), u(plan["negative_prompt"]), u(plan["scheduler"]), u(plan["steps"]),
                    u(plan["cfg_scale"]), u(plan["width"]), u(plan["height"]), u(plan["seed"]),
                    u(plan["model"]), u(plan["vae"]), *lora_ups, status_msg, *_plan_extra_updates(plan))

        send_gen_btn.click(
            do_send_to_gen,
            [png_input],
            [
                gen_controls["prompt"],
                gen_controls["neg_prompt"],
                gen_controls["scheduler"],
                gen_controls["steps"],
                gen_controls["cfg"],
                gen_controls["width"],
                gen_controls["height"],
                gen_controls["seed"],
                gen_controls["model"],
                gen_controls["vae"],
                gen_controls["lora_dd1"], gen_controls["lora_w1"],
                gen_controls["lora_dd2"], gen_controls["lora_w2"],
                gen_controls["lora_dd3"], gen_controls["lora_w3"],
                png_action_status,
                *gen_controls["extra"],
            ],
        )

        def do_png_tags(img, progress=gr.Progress()):
            if img is None:
                return gr.update(), '<p style="color:#f38ba8;font-size:13px;">⚠ No image loaded in PNG Info.</p>'
            try:
                from backend.wd_tagger import tag_image, tags_text
                res = tag_image(img, progress=progress)
            except Exception as e:
                return gr.update(), f'<p style="color:#f38ba8;font-size:13px;">❌ Tagger failed: {html.escape(str(e))}</p>'
            chars = ", ".join(f"{t} {p:.0%}" for t, p in res["character"]) or "none recognised"
            return tags_text(res), (f'<p style="color:#a6e3a1;font-size:13px;">🏷 {len(res["general"])} tags sent '
                                    f'to Generate · character: {html.escape(chars)}</p>')

        png_tag_btn.click(do_png_tags, [png_input], [gen_controls["prompt"], png_action_status])

        def do_send_to_i2i(img):
            if img is None:
                return gr.update(), gr.update(), '<p style="color:#f38ba8;font-size:13px;">⚠ No image loaded in PNG Info.</p>'
            return img, True, '<p style="color:#a6e3a1;font-size:13px;">✅ Sent image to img2img and enabled img2img mode!</p>'

        send_i2i_btn.click(
            do_send_to_i2i,
            [png_input],
            [gen_controls["init_image"], gen_controls["use_i2i"], png_action_status],
        ).then(None, None, None, js=_OPEN_I2I_JS)

        def do_send_to_up(img):
            if img is None:
                return gr.update(), '<p style="color:#f38ba8;font-size:13px;">⚠ No image loaded in PNG Info.</p>'
            return img, '<p style="color:#a6e3a1;font-size:13px;">✅ Sent image to Upscale tab!</p>'

        send_up_btn.click(
            do_send_to_up,
            [png_input],
            [up_input, png_action_status],
        )
    return {"to_generate": [send_gen_btn, send_i2i_btn], "to_upscale": [send_up_btn], "input": png_input}


# ═════════════════════════════════════════════════════════════════════════════
# Tab 3: Watermark / Logo Remover
# ═════════════════════════════════════════════════════════════════════════════
def _build_watermark_tab():
    from backend.watermark_remover import (
        detect_all, build_mask, merge_masks,
        overlay_detections, regions_to_text,
        inpaint_opencv, inpaint_lama, inpaint_sd,
        is_ocr_available, is_lama_available,
    )

    _ocr_note = (
        "✅ EasyOCR installed — text detection enabled."
        if is_ocr_available() else
        "⚠ EasyOCR not installed. Install with:  <code>pip install easyocr</code>  "
        "Corner-watermark heuristic is still available."
    )
    _lama_note = (
        "✅ LaMa ONNX available (auto-downloads ~100 MB on first use)."
        if is_lama_available() else
        "⚠ onnxruntime not found — LaMa unavailable."
    )

    with gr.Tab("🗑️ Watermark Remover"):
        gr.HTML(
            '<p style="color:#a6adc8;font-size:13px;">'
            'Detect and remove watermarks, logos, and embedded text from images. '
            'Uses OCR scanning to find text regions plus an edge-density heuristic '
            'for corner logos. <b>LaMa</b> (default) gives the cleanest result for almost all '
            'watermarks; OpenCV is instant but only suits thin text; SD repaints the area with '
            'your loaded model and may invent new details.</p>'
            f'<p style="color:#a6adc8;font-size:13px;">{_ocr_note}&nbsp; {_lama_note}</p>'
        )

        # ── State: detected regions list ─────────────────────────────────────
        _regions_state = gr.State(value=[])

        with gr.Row():
            # ── Left column: controls ─────────────────────────────────────────
            with gr.Column(scale=1):
                wm_input = gr.ImageEditor(
                    label="Input Image — draw over regions to remove",
                    type="pil",
                    sources=["upload", "clipboard"],
                    transforms=[],
                    brush=gr.Brush(default_size=20, colors=["#ff0000"],
                                   default_color="#ff0000",
                                   color_mode="fixed"),
                    eraser=gr.Eraser(default_size=20),
                )
                gr.HTML(
                    '<p style="color:#a6adc8;font-size:13px;">'
                    '🖌️ Upload an image, then <b>paint red</b> over any '
                    'extra areas you want removed. Auto-Detect finds text '
                    'automatically; your brush marks are merged with it.</p>'
                )

                with gr.Accordion("🔎 Detection Settings", open=True):
                    wm_manual_only = gr.Checkbox(
                        value=False,
                        label="✏️ Manual mask only (no auto-detect)",
                        info="Disable OCR + corner scanning. Only your brush strokes "
                             "are used as the removal mask. Great for manga, emoji, "
                             "or images where you want to keep some text.",
                    )
                    wm_conf   = gr.Slider(0.1, 0.9, value=0.3, step=0.05,
                                          label="OCR Confidence Threshold",
                                          info="Minimum confidence for OCR to report a text region. "
                                               "Lower = catches more but may include false positives.")
                    wm_corner = gr.Slider(0.05, 0.30, value=0.12, step=0.01,
                                          label="Corner Scan Area (% of image edge)",
                                          info="How far from each corner to scan for watermarks. "
                                               "12% means the outer 12% of each edge is checked.")
                    wm_langs  = gr.Dropdown(
                        # EasyOCR combines English with ONE of these scripts only — "en+ch_sim+ja+ko"
                        # always raised (audit F-22)
                        choices=["en", "en+ch_sim", "en+ja", "en+ko"],
                        value="en",
                        label="OCR Languages",
                        info="Add languages if watermark text is non-English",
                    )
                    wm_use_ocr    = gr.Checkbox(value=True,  label="Use OCR text detection",
                                                info="Detect text/logo regions via EasyOCR "
                                                     "(GPU under ZLUDA, ~1 s after the first run).")
                    wm_use_corner = gr.Checkbox(value=True,  label="Use corner heuristic",
                                                info="Scan image corners for edge-density anomalies "
                                                     "(logos, copyright marks). Very fast.")

                    # Hide OCR-related controls when manual-only is checked
                    def _toggle_detect_controls(manual):
                        vis = gr.update(visible=not manual)
                        return vis, vis, vis, vis, vis
                    wm_manual_only.change(
                        _toggle_detect_controls, [wm_manual_only],
                        [wm_conf, wm_corner, wm_langs, wm_use_ocr, wm_use_corner],
                    )

                detect_btn = gr.Button("🔍 Auto-Detect Watermarks", variant="secondary")
                wm_detect_info = gr.Textbox(
                    label="Detected Regions",
                    value="Upload an image and click Auto-Detect.",
                    lines=5, interactive=False,
                )

                gr.Markdown("---")
                gr.Markdown("### 🗑️ Removal Settings")

                wm_dilation = gr.Slider(0, 60, value=20, step=1,
                                        label="Mask Dilation (px) — expands each region",
                                        info="Pixels to expand each detected region outward. "
                                             "Ensures the full watermark + halo is covered. "
                                             "20px is a good default; increase for large logos.")
                wm_method = gr.Dropdown(
                    choices=[
                        "OpenCV TELEA (fast)",
                        "OpenCV NS (smoother)",
                        "LaMa ONNX (high quality)",
                        "SD Inpainting (creative — may add detail)",
                    ],
                    value="LaMa ONNX (high quality)" if is_lama_available()
                          else "OpenCV TELEA (fast)",
                    label="Removal Method",
                    info="LaMa: best for most watermarks (~7 s). OpenCV: instant, thin text only — "
                         "smears large areas. SD: repaints with the model loaded in Generate (~10 s), "
                         "can hallucinate objects.",
                )
                wm_sd_prompt = gr.Textbox(
                    label="SD Inpainting Prompt (SD method only)",
                    value="seamless background, clean, no text, no watermark",
                    visible=False,
                    info="Describes what should replace the watermark. Only used "
                         "with the SD Inpainting method.",
                )
                wm_method.change(
                    lambda m: gr.update(visible="SD" in m),
                    [wm_method], [wm_sd_prompt],
                )

                _wm_total_mb = get_total_vram_mb()
                _wm_vram_bar = gr.HTML(
                    vram_bar_html(
                        estimate_watermark("LaMa ONNX (high quality)"),
                        _wm_total_mb,
                        "Est. VRAM",
                    )
                )

                remove_btn = gr.Button("🗑️ Remove Watermark", variant="primary")
                wm_info    = gr.HTML("")
                with gr.Accordion("📁 Batch — clean a whole folder (auto-detect)", open=False):
                    wm_folder = gr.Textbox(
                        label="Folder with images",
                        placeholder=r"e.g. D:\datasets\my_character\raw",
                        info="Every PNG / JPG / WebP: auto-detect with the settings above, then remove with the "
                             "method above (LaMa recommended). Images where nothing is found are copied "
                             "unchanged. Results go to outputs/cleaned_<folder>/ — handy for LoRA training sets.")
                    wm_batch_btn = gr.Button("📁 Clean folder", variant="secondary")
                    wm_batch_info = gr.HTML("")

            # ── Right column: previews ────────────────────────────────────────
            with gr.Column(scale=1):
                wm_preview = gr.Image(
                    label="Detection Preview (cyan = OCR, orange = corner)",
                    type="pil",
                )
                wm_output  = gr.Image(
                    label="Cleaned Output",
                    type="pil",
                    show_download_button=True,
                )

    # ── Helpers to extract bg image and brush mask from ImageEditor ──────────
    def _extract_bg(editor_val):
        """Return the background PIL image from an ImageEditor value."""
        if editor_val is None:
            return None
        if isinstance(editor_val, dict):
            bg = editor_val.get("background")
            if bg is not None:
                return bg.convert("RGB") if hasattr(bg, "convert") else None
            return None
        # Plain PIL image (shouldn't happen but be safe)
        return editor_val.convert("RGB") if hasattr(editor_val, "convert") else None

    def _extract_drawn_mask(editor_val):
        """
        Merge all brush layers from ImageEditor into a single L-mode mask.
        Any non-transparent pixel in any layer → white in the mask.
        Returns None if nothing was drawn.
        """
        if not isinstance(editor_val, dict):
            return None
        layers = editor_val.get("layers") or []
        bg = editor_val.get("background")
        if not layers or bg is None:
            return None
        import numpy as _np
        W, H = bg.size
        combined = _np.zeros((H, W), dtype=_np.uint8)
        for layer in layers:
            if layer is None:
                continue
            rgba = layer.convert("RGBA").resize((W, H), Image.LANCZOS)
            alpha = _np.array(rgba)[:, :, 3]
            combined = _np.maximum(combined, (alpha > 10).astype(_np.uint8) * 255)
        if combined.max() == 0:
            return None
        return Image.fromarray(combined, mode="L")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def do_detect(editor_val, conf, corner_pct, langs_str, use_ocr, use_corner,
                  manual_only):
        if manual_only:
            return (
                [], None,
                '<p style="color:#89dceb;">✏️ Manual mask only — '
                'draw red over the areas you want removed, then click Remove. '
                'Auto-detect is disabled.</p>',
                "Manual mask mode — no auto-detection.",
            )
        image = _extract_bg(editor_val)
        if image is None:
            return (
                [], None,
                '<p style="color:#f38ba8;">Please upload an image first.</p>',
                "Upload an image and click Auto-Detect.",
            )
        # Parse language string (e.g. "en+ch_sim" → ["en", "ch_sim"])
        langs = _ocr_langs(langs_str)

        try:
            regions = detect_all(
                image,
                min_confidence=conf,
                corner_pct=corner_pct,
                languages=langs if use_ocr else None,
                use_ocr=use_ocr,
                use_corner=use_corner,
            )
        except Exception as e:
            msg = html.escape((str(e).strip().splitlines() or [type(e).__name__])[0][:300])
            return (
                [], None,
                f'<p style="color:#f38ba8;">❌ Detection failed: {msg}. Untick OCR (the first OCR run '
                f'downloads its model, so it needs internet) or draw the areas with the brush.</p>',
                "Detection failed.",
            )
        preview = overlay_detections(image, regions) if regions else image
        info_txt = regions_to_text(regions)
        status   = (
            f'<p style="color:#a6e3a1;">✅ Detected {len(regions)} region(s). '
            f'Adjust settings if needed, then click Remove.</p>'
            if regions else
            '<p style="color:#f5c6a0;">⚠ No regions detected. '
            'Try lowering the confidence threshold, '
            'or use the Manual Mask to draw regions.</p>'
        )
        return regions, preview, status, info_txt

    detect_btn.click(
        do_detect,
        [wm_input, wm_conf, wm_corner, wm_langs,
         wm_use_ocr, wm_use_corner, wm_manual_only],
        [_regions_state, wm_preview, wm_info, wm_detect_info],
    )

    def do_remove(editor_val, regions, dilation, method, sd_prompt,
                  manual_only, progress=gr.Progress()):
        image = _extract_bg(editor_val)
        manual_mask = _extract_drawn_mask(editor_val)
        if image is None:
            return None, '<p style="color:#f38ba8;">Please upload an image.</p>'

        def prog_cb(pct, msg):
            progress(pct, desc=msg)

        if manual_only:
            # Manual-only: use ONLY the brush-drawn mask, ignore auto-detected regions
            mask = manual_mask
            if mask is None:
                from PIL import Image as _PILImage
                mask = _PILImage.new("L", image.size, 0)
        else:
            # Build mask from detected regions + merge with brush strokes
            auto_mask = build_mask(image, regions or [], dilation=int(_num(dilation, 8)))
            mask = merge_masks(auto_mask, manual_mask)

        # Check that mask is non-empty
        import numpy as _np
        if _np.array(mask).max() == 0:
            hint = ("Draw red over the areas you want removed."
                    if manual_only else
                    "Run Auto-Detect first or draw regions manually.")
            return image, (
                f'<p style="color:#f5c6a0;">⚠ Mask is empty — no regions to remove. '
                f'{hint}</p>'
            )

        try:
            m = method.lower()
            if "lama" in m:
                out = inpaint_lama(image, mask, progress_callback=prog_cb)
            elif "sd" in m:
                out = inpaint_sd(image, mask, sd, prompt=sd_prompt,
                                 progress_callback=prog_cb)
            elif "ns" in m:
                out = inpaint_opencv(image, mask, method="ns")
            else:
                out = inpaint_opencv(image, mask, method="telea")
        except Exception as e:
            return image, f'<p style="color:#f38ba8;">❌ Removal failed: {e}</p>'

        # Auto-save (named after the source image when it came from this app)
        src = (getattr(image, "info", {}) or {}).get("saved_path")
        path = _unique_output(f"{Path(src).stem}_clean" if src else f"dewatermark_{int(time.time())}")
        out.save(path, pnginfo=_carry_params(image, f"Watermark removed: {method}"))

        return out, (
            f'<p style="color:#a6e3a1;">✅ Watermark removed using {method}. '
            f'Saved to outputs/{path.name}</p>'
        )

    def do_remove_folder(folder, conf, corner_pct, langs_str, use_ocr, use_corner, dilation, method,
                         sd_prompt, progress=gr.Progress()):
        from PIL import Image as _Image
        src = Path(str(folder or "").strip().strip('"'))
        if not str(folder or "").strip() or not src.is_dir():
            return '<p style="color:#f38ba8;">❌ Enter an existing folder.</p>'
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        if not files:
            return '<p style="color:#f38ba8;">❌ No PNG / JPG / WebP images in that folder.</p>'
        if not (use_ocr or use_corner):
            return '<p style="color:#f38ba8;">❌ Turn on OCR and/or the corner heuristic — batch mode can only auto-detect.</p>'
        langs = _ocr_langs(langs_str)
        dest = OUTPUTS_DIR / f"cleaned_{_safe_name(src.name) or 'folder'}"
        dest.mkdir(parents=True, exist_ok=True)
        cleaned, untouched, failed, t0 = 0, 0, [], time.time()
        m = (method or "").lower()
        for i, f in enumerate(files):
            progress(i / len(files), desc=f"{i + 1}/{len(files)}: {f.name}")
            try:
                with _Image.open(f) as im:
                    im.load()
                    img = im.convert("RGB")
                    img.info = dict(im.info)
                regions = detect_all(img, min_confidence=float(conf), corner_pct=float(corner_pct),
                                     languages=langs if use_ocr else None, use_ocr=bool(use_ocr),
                                     use_corner=bool(use_corner))
                target = dest / f"{f.stem}.png"
                if not regions:
                    img.save(target, pnginfo=_carry_params(img, None))
                    untouched += 1
                    continue
                mask = build_mask(img, regions, dilation=int(_num(dilation, 8)))
                if "lama" in m:
                    out = inpaint_lama(img, mask)
                elif "sd" in m:
                    out = inpaint_sd(img, mask, sd, prompt=sd_prompt)
                elif "ns" in m:
                    out = inpaint_opencv(img, mask, method="ns")
                else:
                    out = inpaint_opencv(img, mask, method="telea")
                out.save(target, pnginfo=_carry_params(img, f"Watermark removed: {method}"))
                cleaned += 1
            except Exception as e:
                failed.append(f"{f.name} ({html.escape(str(e)[:60])})")
        msg = (f'<p style="color:#a6e3a1;font-size:13px;">✅ {cleaned} cleaned, {untouched} had nothing to remove '
               f'(copied as-is) — {_fmt_elapsed(time.time() - t0)} → <code>outputs/{dest.name}/</code></p>')
        if failed:
            msg += '<p style="color:#f38ba8;font-size:13px;">Failed: ' + ", ".join(failed[:10]) + "</p>"
        return msg

    wm_batch_btn.click(
        do_remove_folder,
        [wm_folder, wm_conf, wm_corner, wm_langs, wm_use_ocr, wm_use_corner, wm_dilation, wm_method, wm_sd_prompt],
        [wm_batch_info],
    )

    remove_btn.click(
        do_remove,
        [wm_input, _regions_state, wm_dilation, wm_method,
         wm_sd_prompt, wm_manual_only],
        [wm_output, wm_info],
    )

    def _update_wm_vram(method, editor_val):
        img = _extract_bg(editor_val)
        w, h = (img.size if img is not None else (1024, 1024))
        used = estimate_watermark(method or "", w, h)
        return vram_bar_html(used, _wm_total_mb, "Est. VRAM")

    wm_method.change(_update_wm_vram, [wm_method, wm_input], [_wm_vram_bar])
    wm_input.change(_update_wm_vram, [wm_method, wm_input], [_wm_vram_bar])


# ═════════════════════════════════════════════════════════════════════════════
# Tab 4: Civitai Hub
# ═════════════════════════════════════════════════════════════════════════════
def _build_civitai_tab(model_dd, lora_dd, vae_dd, more_lora_dds=()):
    global _search_results, _current_model

    with gr.Tab("🌐 Civitai Hub"):
        gr.HTML(
            '<p style="color:#a6adc8;font-size:13px;">'
            'Browse and download Stable Diffusion models and LoRAs from Civitai. '
            'An API key is required for NSFW content and higher download speeds.</p>'
        )

        with gr.Row():
            search_box  = gr.Textbox(label="Name search", placeholder="searches model name/description…", scale=3)
            tag_box     = gr.Textbox(label="Tag", placeholder="e.g. soft lighting, smile, cherry blossoms…", scale=2,
                                     info="Searches Civitai tags — how popular models are actually indexed.")
            type_dd     = gr.Dropdown(
                ["Checkpoint", "LoRA", "LoCon / LyCORIS", "DoRA", "TextualInversion", "VAE", "Upscaler", "All"],
                value="LoRA", label="Type", scale=1,
                info="LoRA = standard LoRAs. LoCon/LyCORIS = most SDXL/Illustrious style LoRAs.",
            )
            sort_dd     = gr.Dropdown(
                ["Most Downloaded", "Highest Rated", "Newest"],
                value="Most Downloaded", label="Sort", scale=1,
                info="How to sort search results from Civitai.",
            )
            base_dd     = gr.Dropdown(
                ["", "SD 1.5", "SDXL 1.0", "SD 2.1", "Pony", "Illustrious", "NoobAI", "Flux.1 D", "Flux.1 S"],
                value="", label="Base Model", scale=1,
                info="Filter by base architecture. Leave blank for all.",
            )
            nsfw_dd     = gr.Dropdown(
                ["SFW only", "R18+ / NSFW", "Explicit (XXX)"],
                value="SFW only", label="Content", scale=1,
                info="R18+/Explicit requires a Civitai API key set below.",
            )
            search_btn  = gr.Button("🔍 Search", variant="primary", scale=1)

        with gr.Row():
            api_key_box = gr.Textbox(
                label="Civitai API Key (required for NSFW/explicit content)",
                # never put the stored key into the component: Gradio serves every value in /config
                placeholder=("●●● key saved — paste a new one to replace it, or type clear"
                             if CIVITAI_API_KEY else "Paste your key from civitai.com/user/account → API Keys…"),
                type="password",
                value="",
                scale=5,
            )
            api_key_btn = gr.Button("Set Key", scale=1)
        api_key_status = gr.HTML("")

        # Results grid: 4 columns × 5 rows = 20 cards, each with a Select button
        cards = []
        card_btns = []
        for _row in range(5):
            with gr.Row():
                for _col in range(4):
                    with gr.Column(scale=1):
                        cards.append(gr.HTML(""))
                        card_btns.append(gr.Button("📥 Select", size="sm", variant="secondary", visible=False))

        # Pagination controls
        with gr.Row():
            prev_page_btn = gr.Button("◀ Previous Page", size="sm", scale=1, interactive=False)
            with gr.Column(scale=2):
                page_label = gr.HTML('<p style="text-align:center;color:#9399b2;font-size:13px;">Page 1</p>')
            next_page_btn = gr.Button("Next Page ▶", size="sm", scale=1, interactive=False)

        gr.Markdown("---")

        # Model detail + version selector— shown after clicking Select on a card
        with gr.Row():
            with gr.Column(scale=2):
                detail_html  = gr.HTML('<p style="color:#a6adc8;font-size:13px;">Click <b>📥 Select</b> on any card above to load details and choose a version to download.</p>')
            with gr.Column(scale=1):
                version_dd   = gr.Dropdown([""], label="Version", value="", interactive=True)
                dl_btn       = gr.Button("⬇ Download Selected Version", variant="primary")
                dl_status    = gr.HTML("")
                dl_progress  = gr.Slider(0, 1, value=0, label="Download Progress", interactive=False)

        # Model index picker (hidden state to track which card was clicked)
        selected_idx = gr.State(value=-1)

    # ── Search callback ────────────────────────────────────────────────────────
    # Civitai API type values differ from display labels
    _CIVITAI_TYPE = {
        "LoRA": "LORA",
        "LoCon / LyCORIS": "LoCon",
        "TextualInversion": "TextualInversion",
        "All": "",
    }

    def _render_page():
        """Render the current page of _search_all_results into card HTML + pagination."""
        global _search_results, _search_page
        start = _search_page * _RESULTS_PER_PAGE
        end = start + _RESULTS_PER_PAGE
        _search_results = _search_all_results[start:end]
        total_pages = max(1, -(-len(_search_all_results) // _RESULTS_PER_PAGE))  # ceil div

        html_updates = []
        btn_updates  = []
        for i in range(_RESULTS_PER_PAGE):
            if i < len(_search_results):
                html_updates.append(gr.update(value=civitai.format_model_card(_search_results[i])))
                btn_updates.append(gr.update(visible=True))
            else:
                html_updates.append(gr.update(value=""))
                btn_updates.append(gr.update(visible=False))

        if _search_all_results:
            page_html = (
                f'<p style="text-align:center;color:#a6adc8;font-size:13px;">'
                f'Page {_search_page + 1} / {total_pages}'
                f' &nbsp;({len(_search_all_results)} results total)</p>'
            )
        else:
            page_html = (
                '<p style="text-align:center;color:#f5c6a0;font-size:13px;">'
                'No models found — try fewer or different words, another type / base model, '
                'or a wider content rating.</p>'
            )
        prev_upd = gr.update(interactive=(_search_page > 0))
        next_upd = gr.update(interactive=(_search_page < total_pages - 1))

        return html_updates + btn_updates + [page_html, prev_upd, next_upd]

    def do_search(query, tag, model_type, sort, base_model, nsfw_filter):
        global _search_all_results, _search_page
        nsfw     = nsfw_filter in ("R18+ / NSFW", "Explicit (XXX)")
        api_type = _CIVITAI_TYPE.get(model_type, model_type)
        q        = query.strip()
        t        = tag.strip()

        # Fetch up to 100 results from Civitai API
        common = dict(model_type=api_type, sort=sort, base_model=base_model,
                      limit=100, nsfw=nsfw)

        try:
            seen: dict[int, dict] = {}

            # ── Call 1: name/description search (+ explicit tag filter if set) ──
            data1 = civitai.search_models(query=q, tag=t, **common)
            for item in data1.get("items", []):
                seen[item["id"]] = item
            total = data1.get("metadata", {}).get("totalItems", 0)

            # ── Call 2: treat the name query as a tag search too ──────────────
            if q and not t:
                try:
                    data2 = civitai.search_models(query="", tag=q, **common)
                    for item in data2.get("items", []):
                        if item["id"] not in seen:
                            seen[item["id"]] = item
                    total = max(total, data2.get("metadata", {}).get("totalItems", 0))
                except Exception:
                    pass

        except Exception as e:
            err = (f'<p style="text-align:center;color:#f38ba8;font-size:13px;">❌ Civitai search failed: '
                   f'{html.escape(_civ_scrub(e))}<br>Check your internet connection (or try again — '
                   f'Civitai is sometimes slow).</p>')
            return ([gr.update(value="")] * _RESULTS_PER_PAGE + [gr.update(visible=False)] * _RESULTS_PER_PAGE
                    + [err, gr.update(interactive=False), gr.update(interactive=False)])

        # Enforce the filters here too: Civitai has changed which parameter names it honours
        # before (type= → types=), and an ignored filter shouldn't leak other kinds of models in.
        if api_type:
            seen = {k: v for k, v in seen.items() if v.get("type") == api_type}
        if base_model:
            seen = {k: v for k, v in seen.items()
                    if any(mv.get("baseModel") == base_model for mv in (v.get("modelVersions") or []))}

        # Sort merged pool according to user's chosen sort order
        if sort == "Highest Rated":
            merged = sorted(seen.values(),
                            key=lambda x: (x.get("stats") or {}).get("thumbsUpCount") or 0,
                            reverse=True)
        elif sort == "Newest":
            merged = sorted(seen.values(),
                            key=lambda x: x.get("publishedAt") or "",
                            reverse=True)
        else:
            merged = sorted(seen.values(),
                            key=lambda x: (x.get("stats") or {}).get("downloadCount") or 0,
                            reverse=True)

        _search_all_results = merged
        _search_page = 0
        return _render_page()

    def do_next_page():
        global _search_page
        _search_page += 1
        return _render_page()

    def do_prev_page():
        global _search_page
        _search_page = max(0, _search_page - 1)
        return _render_page()

    _page_outputs = cards + card_btns + [page_label, prev_page_btn, next_page_btn]

    search_btn.click(
        do_search,
        [search_box, tag_box, type_dd, sort_dd, base_dd, nsfw_dd],
        _page_outputs,
    )
    next_page_btn.click(do_next_page, [], _page_outputs)
    prev_page_btn.click(do_prev_page, [], _page_outputs)

    def do_set_api_key(key):
        k = (key or "").strip()
        if not k:        # the box is empty by design (the saved key isn't shown): keep it
            return ('<p style="color:#a6adc8;">No new key entered — the saved key (if any) stays. '
                    'Type <code>clear</code> to remove it.</p>')
        if k.lower() == "clear":
            civitai.set_api_key("")
            _persist_key("civitai", "")
            os.environ.pop("CIVITAI_API_KEY", None)
            return '<p style="color:#a6adc8;">API key removed.</p>'
        civitai.set_api_key(k)
        _persist_key("civitai", k)
        os.environ["CIVITAI_API_KEY"] = k
        return '<p style="color:#a6e3a1;">✅ API key saved (persists across restarts) — NSFW/explicit results unlocked.</p>'

    api_key_btn.click(do_set_api_key, [api_key_box], [api_key_status])

    # ── Card select callbacks ─────────────────────────────────────────────────
    def _make_card_select(idx: int):
        def on_select():
            global _current_model
            if idx >= len(_search_results):
                return (
                    '<p style="color:#a6adc8;">No model at this index.</p>',
                    gr.update(choices=[], value=None),
                )
            _current_model = _search_results[idx]
            choices = civitai.get_version_choices(_current_model)
            label_choices = [f"{v} (id={vid})" for v, vid in choices]
            detail = civitai.format_model_card(_current_model)
            return detail, gr.update(choices=label_choices, value=label_choices[0] if choices else "")
        return on_select

    for i, btn in enumerate(card_btns):
        btn.click(_make_card_select(i), [], [detail_html, version_dd])

    # ── Download callback ─────────────────────────────────────────────────────
    def do_download(version_label, current_ckpt=None, progress=gr.Progress()):
        global _current_model
        _no_update = (gr.update(), gr.update(), gr.update())
        if _current_model is None:
            return ('<p style="color:#f38ba8;">Select a model first.</p>', gr.update(value=0), *_no_update,
                    *[gr.update()] * len(more_lora_dds))
        if not version_label:
            return ('<p style="color:#f38ba8;">Select a version.</p>', gr.update(value=0), *_no_update,
                    *[gr.update()] * len(more_lora_dds))

        try:
            # Extract version id from label "(id=12345)"
            vid = int(version_label.split("id=")[-1].rstrip(")"))
        except Exception:
            return ('<p style="color:#f38ba8;">Could not parse version id.</p>', gr.update(value=0), *_no_update,
                    *[gr.update()] * len(more_lora_dds))

        try:
            version_data = civitai.get_model_version(vid)
        except Exception as e:
            return (f'<p style="color:#f38ba8;">API error: {html.escape(_civ_scrub(e))}</p>', gr.update(value=0),
                    *_no_update,
                    *[gr.update()] * len(more_lora_dds))

        def prog_cb(pct, msg):
            progress(pct, desc=msg)

        ok, result = civitai.download_model_version(_current_model, version_data, prog_cb)

        if ok:
            fname = Path(result).name
            folder = Path(result).parent.name
            companions = getattr(civitai, "last_companion_embeddings", [])
            msg = (
                f'<p style="color:#a6e3a1;">✅ Downloaded: {fname}<br>'
                f'Saved to: <code>{folder}/</code></p>'
            )
            if Path(result).parent.parent.name == "other":
                msg += ('<p style="color:#f9e2af;font-size:13px;margin-top:4px;">'
                        "This type isn't used by ImageGen Studio — it's kept in "
                        f'<code>models/other/{folder}/</code> for other tools.</p>')
            if companions:
                emb_list = ", ".join(f"<code>{n}</code>" for n in companions)
                msg += (
                    f'<p style="color:#89dceb;font-size:13px;margin-top:4px;">'
                    f'📎 Companion embeddings ({len(companions)}): {emb_list}<br>'
                    f'Saved to: <code>embeddings/</code> — auto-loaded when generating.</p>'
                )
            # Auto-refresh Generate tab dropdowns
            _ck = _refresh_checkpoints()
            model_upd = gr.update(choices=_ck)
            # First local checkpoint (or the picker still on an HF default)? Select it.
            if (str(Path(result)) in {v for _, v in _ck}
                    and not (current_ckpt and Path(str(current_ckpt)).is_file())):
                model_upd = gr.update(choices=_ck, value=str(Path(result)))
                msg += ('<p style="color:#89b4fa;font-size:13px;margin-top:4px;">'
                        'Selected as the checkpoint on the 🎨 Generate tab.</p>')
            lora_upd  = gr.update(choices=_lora_choices())
            vae_upd   = gr.update(choices=["none"] + list_vaes())
        else:
            msg = f'<p style="color:#f38ba8;">❌ {result}</p>'
            model_upd = gr.update()
            lora_upd  = gr.update()
            vae_upd   = gr.update()

        return (msg, gr.update(value=1.0 if ok else 0.0), model_upd, lora_upd, vae_upd,
                *[lora_upd] * len(more_lora_dds))

    dl_btn.click(do_download, [version_dd, model_dd],
                 [dl_status, dl_progress, model_dd, lora_dd, vae_dd, *more_lora_dds])


# ═════════════════════════════════════════════════════════════════════════════
# Tab 4: 🎓 Train LoRA
# ═════════════════════════════════════════════════════════════════════════════

# ── inline help CSS (re-uses Help tab palette) ─────────────────────────────
_TRAIN_CSS = (
    '<style>'
    '.tr-note{background:#1e1e2e;border-left:3px solid #fab387;padding:8px 12px;'
    'margin:6px 0;border-radius:0 6px 6px 0;font-size:13px;color:#a6adc8;line-height:1.55}'
    '.tr-tip{background:#1e1e2e;border-left:3px solid #a6e3a1;padding:8px 12px;'
    'margin:6px 0;border-radius:0 6px 6px 0;font-size:13px;color:#a6adc8;line-height:1.55}'
    '.tr-warn{background:#1e1e2e;border-left:3px solid #f38ba8;padding:8px 12px;'
    'margin:6px 0;border-radius:0 6px 6px 0;font-size:13px;color:#a6adc8;line-height:1.55}'
    '.tr-h{color:#cba6f7;font-weight:700;font-size:14px;margin:10px 0 4px}'
    '.tr-dim{color:#9399b2;font-size:13px}'
    '</style>'
)

# Module-level trainer state (same pattern as sd / upscaler singletons)
_trainer = None          # LoRATrainer instance (lives across callbacks)
_train_dataset = None    # last scan_images() result
_train_assignments = None  # last bucket assignments

def _build_train_tab(gen_lora_dds=()):
    # Default base model = the one last used in Generate; resolution follows its family
    from backend.sd_pipeline import _is_sdxl
    _tr_last_model = _read_last_session().get("model") or ""
    _tr_default_res = ("512 (SD 1.5)" if _tr_last_model and not _is_sdxl(_tr_last_model)
                       else "1024 (SDXL / Pony / Illustrious)")
    global _trainer, _train_dataset, _train_assignments
    from backend.model_manager import list_checkpoints

    with gr.Tab("🎓 Train LoRA"):
        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML(
            _TRAIN_CSS +
            '<div style="background:#313244;padding:12px 16px;border-radius:8px;margin-bottom:6px">'
            '<span style="color:#cba6f7;font-size:16px;font-weight:700">'
            '🎓 LoRA Training Studio</span><br>'
            '<span style="color:#a6adc8;font-size:13px">'
            'Train your own LoRA (Low-Rank Adaptation) weights for SD 1.5 or SDXL / Pony / Illustrious. '
            'The trained LoRA is saved as a standard <code>.safetensors</code> file that works with '
            'the Generate tab, Civitai community LoRAs, and other tools like ComfyUI.</span>'
            '</div>'
        )
        gr.HTML(
            '<div class="tr-note">'
            '<b>How LoRA training works — the short version:</b><br>'
            '① Collect 10–200 images of a character / style / concept.<br>'
            '② Caption each image with a trigger word + description tags.<br>'
            '③ The trainer injects tiny "adapter" matrices into the frozen base model, '
            'then teaches <em>only those adapters</em> to reproduce your images when the trigger word is used.<br>'
            '④ Result: a small .safetensors file (5–200 MB) you load in the Generate tab.'
            '</div>'
        )

        # ════════════════════════════════════════════════════════════════════
        # STEP 1 — Dataset
        # ════════════════════════════════════════════════════════════════════
        with gr.Accordion("📁 Step 1 — Prepare Dataset", open=True):
            gr.HTML(
                '<div class="tr-tip">'
                '<b>What makes a good training dataset?</b><br>'
                '• <b>Character LoRA:</b> 15–50 images of the character in varied poses, '
                'angles, outfits, and backgrounds. Solo images work best — '
                'crop out other characters.<br>'
                '• <b>Style LoRA:</b> 30–200 images in the target art style. '
                'Consistent style matters more than quantity.<br>'
                '• <b>Concept LoRA:</b> 10–30 images of the concept (e.g., a specific item, '
                'location, composition).'
                '</div>'
            )
            gr.HTML(
                '<div class="tr-tip">'
                '<b>🛏️ Body pillows & extreme aspect ratios:</b> '
                'Images are <em>not</em> forced to a square crop! This tool uses '
                '<b>aspect-ratio bucketing</b> — each image is assigned to a resolution bucket '
                'that matches its shape (e.g., a 16:5 body pillow → 960×320 bucket for SD 1.5, '
                '1920×640 for SDXL). Only a tiny center crop happens if the exact ratio doesn\'t '
                'exist. <b>Your wide/tall images are preserved.</b>'
                '</div>'
            )
            with gr.Row():
                # Folders (≤2 levels, e.g. my_character/raw) that directly hold images;
                # skips the generated _prepared_* folders
                _ds_root = APP_DIR / "training_datasets"
                _img_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                _ds_choices = sorted(
                    str(d) for d in ([*_ds_root.glob("*"), *_ds_root.glob("*/*")] if _ds_root.exists() else [])
                    if d.is_dir() and not d.name.startswith("_")
                    and any(f.suffix.lower() in _img_ext for f in d.iterdir()))
                train_folder = gr.Dropdown(
                    label="📂 Image folder",
                    choices=_ds_choices,
                    value=_ds_choices[0] if _ds_choices else "",   # never None: crashes the Gradio 4.19 filter
                    allow_custom_value=True,
                    info="Pick a folder from training_datasets/ or paste any folder path "
                         "containing your raw training images (.png, .jpg, .webp …).",
                    scale=4)
                scan_btn = gr.Button("🔍 Scan", scale=1)
            scan_status = gr.HTML("")

            with gr.Row():
                trigger_word_txt = gr.Textbox(
                    label="✨ Trigger word",
                    placeholder="my_character",
                    info="A unique keyword prepended to every caption. "
                         "You'll type this in the Generate tab to activate the LoRA.",
                    scale=2)
                train_res_dd = gr.Dropdown(
                    label="Target resolution",
                    choices=["512 (SD 1.5)", "768 (SD 1.5 hi-res)", "1024 (SDXL / Pony / Illustrious)"],
                    value=_tr_default_res,
                    info="Set automatically from the base model (Step 3). 768 = SD 1.5 hi-res.",
                    scale=2)
            gr.HTML(
                '<div class="tr-note">'
                '<b>Trigger word tips:</b><br>'
                '• Pick something that doesn\'t appear in normal English — '
                'e.g. a character name like <code>my_character</code>, or an invented word like <code>xyz_style</code>.<br>'
                '• <b>Character LoRAs:</b> also include trait tags in each caption — '
                '<code>my_character, red eyes, grey hair, long hair, dress</code> — '
                'so the model learns to associate the trigger with those features.<br>'
                '• The trigger word will be auto-prepended to every caption.'
                '</div>'
            )
            bucket_stats_html = gr.HTML("")
            prepare_btn = gr.Button("✅ Prepare Dataset (resize + write captions)", variant="primary")
            prepare_status = gr.HTML("")

        # ════════════════════════════════════════════════════════════════════
        # STEP 2 — Captioning
        # ════════════════════════════════════════════════════════════════════
        with gr.Accordion("🏷️ Step 2 — Caption / Tag Images", open=False):
            gr.HTML(
                '<div class="tr-tip">'
                '<b>Why captions matter:</b><br>'
                '• Each training image needs a <code>.txt</code> file describing its content.<br>'
                '• The model learns: <em>"when the user types these words → produce this image."</em><br>'
                '• <b>Character LoRA rule of thumb:</b> start with the trigger word, '
                'then list visual elements — hair color, eye color, outfit, pose, background.<br>'
                '• <b>Style LoRA:</b> describe the content (not the style) so the model associates '
                'the trigger with the style, not specific subjects.<br>'
                '• Captions should be <b>comma-separated tags</b> for anime LoRAs '
                '(e.g. <code>my_character, 1girl, red eyes, grey hair, standing, park</code>) '
                'or <b>natural language</b> for realistic/photo LoRAs '
                '(e.g. <code>my_character, a woman with red eyes and grey hair standing in a park</code>).'
                '</div>'
            )
            with gr.Row():
                caption_prefix_txt = gr.Textbox(
                    label="BLIP prefix hint",
                    value="an anime character",
                    info="Optional context for the captioner (e.g. 'an anime character', 'a photo of')",
                    scale=2)
                autocap_btn = gr.Button("🤖 Auto-Caption All (BLIP)", scale=1)
                unload_blip_btn = gr.Button("🗑️ Free BLIP RAM", scale=1)
            with gr.Row():
                wdtag_btn = gr.Button("🏷 Auto-Tag All (WD14 Danbooru tags — best for anime)", variant="primary",
                                      scale=3)
                wdtag_thr_sl = gr.Slider(0.2, 0.8, value=0.35, step=0.05, label="Tag threshold", scale=1)
            caption_status = gr.HTML("")
            gr.HTML(
                '<div class="tr-note">'
                '<b>How auto-captioning works:</b> '
                'BLIP (a vision-language model, ~1 GB) generates a natural-language description '
                'for each image. The trigger word is auto-prepended. '
                'You can then edit individual captions below. '
                'After captioning, click <b>Free BLIP RAM</b> to reclaim memory before training.'
                '</div>'
            )
            with gr.Row():
                caption_img_dd = gr.Dropdown(
                    label="Select image to edit caption",
                    choices=[], interactive=True, scale=3)
                caption_refresh_btn = gr.Button("🔄 Refresh", scale=0, min_width=110)
            caption_preview = gr.Image(label="Preview", height=256, interactive=False)
            caption_txt = gr.Textbox(
                label="Caption (editable)",
                lines=3,
                info="Edit the caption for the selected image. Saved as a .txt file alongside the image.")
            caption_save_btn = gr.Button("💾 Save Caption")
            cap_save_status = gr.HTML("")

        # ════════════════════════════════════════════════════════════════════
        # STEP 3 — Training Config
        # ════════════════════════════════════════════════════════════════════
        with gr.Accordion("⚙️ Step 3 — Training Configuration", open=False):
            gr.HTML(
                '<div class="tr-note">'
                '<b>Quick-start recommendations</b> (character LoRA on SDXL):<br>'
                'Rank 16 · Alpha 16 · LR 1e-4 · 10–20 epochs · Train text encoder ON · Cosine scheduler.<br>'
                'For SD 1.5: same settings but resolution 512.'
                '</div>'
            )
            with gr.Row():
                _tr_ckpts = _checkpoint_choices()
                _tr_last = _tr_last_model
                train_model_dd = gr.Dropdown(
                    label="🧠 Base model (checkpoint)",
                    choices=_tr_ckpts,
                    value=(_tr_last if _tr_last in [p for _, p in _tr_ckpts]
                           else (_tr_ckpts[0][1] if _tr_ckpts else "")),
                    info="The model your LoRA will be trained against. "
                         "Must match your target — e.g. train on an Illustrious checkpoint "
                         "if you want to use the LoRA with Illustrious models.",
                    scale=3)
                train_model_refresh = gr.Button("🔄 Refresh", scale=0, min_width=110)

            with gr.Row():
                lora_rank_sl = gr.Slider(
                    4, 128, value=16, step=4,
                    label="LoRA Rank (dim)",
                    info="Number of rank dimensions. Higher = more capacity but larger file. "
                         "8 = lightweight, 16 = standard, 32+ = detailed styles.")
                lora_alpha_sl = gr.Slider(
                    1, 128, value=16, step=1,
                    label="LoRA Alpha",
                    info="Scaling factor. Rule of thumb: set equal to rank. "
                         "Higher alpha/rank ratio = stronger LoRA effect per unit weight.")

            with gr.Row():
                lr_num = gr.Number(
                    value=1e-4, label="Learning rate",
                    info="How fast the model learns. 1e-4 = safe default. "
                         "Too high → overfitting (copies images exactly). "
                         "Too low → undertrained (LoRA has no effect).")
                te_lr_num = gr.Number(
                    value=5e-5, label="Text encoder LR",
                    info="Separate learning rate for text encoder. "
                         "Usually 0.5× the UNet LR. Teaches the model what your trigger word means.")

            with gr.Row():
                epochs_sl = gr.Slider(
                    1, 100, value=10, step=1,
                    label="Epochs",
                    info="How many times the trainer sees every image. "
                         "10–20 for character LoRAs. More images → fewer epochs needed.")
                train_te_cb = gr.Checkbox(
                    value=True, label="Train text encoder",
                    info="Recommended for character LoRAs — teaches the model what "
                         "your trigger word means. Disable for pure style LoRAs.")

            gr.HTML(
                '<div class="tr-tip">'
                '<b>Understanding epochs vs. steps:</b><br>'
                'If you have 30 images and set 10 epochs, the trainer runs 30 × 10 = 300 steps. '
                'Each step processes one image. More images with fewer epochs often generalises better '
                'than few images with many epochs (which risks <em>overfitting</em> — the LoRA memorises '
                'your images pixel-for-pixel instead of learning the concept).'
                '</div>'
            )

            with gr.Accordion("🔧 Advanced Settings", open=False):
                gr.HTML(
                    '<div class="tr-note">'
                    'These defaults work well for most LoRAs. Only change them if you '
                    'know what you\'re doing or want to experiment.'
                    '</div>'
                )
                with gr.Row():
                    target_dd = gr.Dropdown(
                        label="Target layers",
                        choices=["attn (attention only — fast, standard)",
                                 "full (attention + convolutions — more thorough)"],
                        value="attn (attention only — fast, standard)",
                        info="'attn' targets Q/K/V/Out attention layers. "
                             "'full' also includes ResNet conv layers for richer spatial detail.")
                    sched_dd = gr.Dropdown(
                        label="LR scheduler",
                        choices=["cosine", "constant", "linear"],
                        value="cosine",
                        info="How the learning rate changes during training. "
                             "'cosine' gently decreases LR — usually best.")
                with gr.Row():
                    grad_accum_sl = gr.Slider(
                        1, 8, value=1, step=1,
                        label="Gradient accumulation",
                        info="Simulates larger batch size. "
                             "2 = accumulates gradients over 2 images before each update.")
                    warmup_sl = gr.Slider(
                        0.0, 0.2, value=0.05, step=0.01,
                        label="Warmup ratio",
                        info="Fraction of total steps with gradually increasing LR. "
                             "0.05 = first 5% of training ramps up slowly.")
                with gr.Row():
                    grad_ckpt_cb = gr.Checkbox(
                        value=True, label="Gradient checkpointing",
                        info="Trades ~15% speed for much lower memory. Keep ON for CPU/DML training.")
                    train_clip_skip_rb = gr.Radio(
                        [1, 2], value=2, label="CLIP skip (SD 1.5)",
                        info="2 = what anime SD 1.5 checkpoints are used with (generate with CLIP skip 2 too). "
                             "SDXL ignores it.")
                    flip_cb = gr.Checkbox(
                        value=False, label="Random horizontal flip",
                        info="Doubles effective dataset size by randomly mirroring images. "
                             "Disable for asymmetric subjects (text, logos, specific hand positions).")
                with gr.Row():
                    save_every_sl = gr.Slider(
                        0, 50, value=0, step=1,
                        label="Save checkpoint every N epochs",
                        info="0 = save only the final LoRA. "
                             ">0 = also save intermediate checkpoints (useful for comparing epochs).")
                    seed_num = gr.Number(
                        value=42, label="Training seed",
                        info="Random seed for reproducibility. Same seed + same data = identical LoRA.")
                with gr.Row():
                    train_device_dd = gr.Dropdown(
                        label="Training device",
                        choices=["auto", "cuda", "cpu", "directml"],
                        value="auto",
                        info="Auto = CUDA/ZLUDA GPU if available, else CPU. "
                             "CUDA/ZLUDA gives ~5× speedup (launch via launch.bat). "
                             "DirectML backward is broken for UNet — expert use only. "
                             "CPU always works (uses your RAM).")

            with gr.Row():
                output_name_txt = gr.Textbox(
                    value="my_lora", label="Output filename",
                    info="Saved to models/loras/{name}.safetensors",
                    scale=2)

            gr.HTML(
                '<div class="tr-warn">'
                '<b>⚠ Memory & time estimates (CPU training):</b><br>'
                '• <b>SD 1.5 LoRA</b> — ~4 GB RAM, ~2–5 sec/step → 30 images × 10 epochs ≈ 15–25 min<br>'
                '• <b>SDXL LoRA</b> — ~12 GB RAM, ~10–30 sec/step → 30 images × 10 epochs ≈ 50–150 min<br>'
                'You have 128 GB RAM so memory is not a concern. '
                'DirectML on the RX 6800 XT should be 3–10× faster if the backward pass works.'
                '</div>'
            )

        # ════════════════════════════════════════════════════════════════════
        # STEP 4 — Train
        # ════════════════════════════════════════════════════════════════════
        with gr.Accordion("🚀 Step 4 — Train!", open=False):
            gr.HTML(
                '<div class="tr-note">'
                '<b>Training flow:</b><br>'
                '① <b>Prepare</b> — loads the base model, injects LoRA adapters, '
                'pre-encodes all images with the VAE (latent caching), '
                'and pre-encodes all captions with the text encoder. This takes 1–5 min.<br>'
                '② <b>Start Training</b> — the training loop runs. You\'ll see loss values decrease '
                'over time. When done the LoRA is saved to <code>models/loras/</code>.<br>'
                '③ <b>Use it!</b> — go to the Generate tab, select your new LoRA from the dropdown, '
                'type your trigger word in the prompt, and generate!'
                '</div>'
            )
            with gr.Row():
                train_prepare_btn = gr.Button("📦 Prepare (load model + cache)", variant="secondary", scale=1)
                train_start_btn = gr.Button("🚀 Start Training", variant="primary", scale=1)
                train_stop_btn = gr.Button("⏹ Stop", variant="stop", scale=0)
            train_progress = gr.HTML("")
            train_log = gr.HTML("")

            gr.HTML(
                '<div class="tr-tip">'
                '<b>After training completes:</b><br>'
                '• Your LoRA is saved as '
                '<code>models/loras/&lt;output_name&gt;.safetensors</code><br>'
                '• Go to <b>🎨 Generate</b> → select it from the LoRA dropdown<br>'
                '• Type your trigger word (e.g. <code>my_character</code>) in the positive prompt<br>'
                '• <b>Pro tip:</b> start with weight 0.7–0.8 and increase to 1.0 if the effect is subtle. '
                'If images look "fried" / oversaturated, lower the weight or reduce epochs.'
                '</div>'
            )
            gr.HTML(
                '<div class="tr-warn">'
                '<b>Troubleshooting trained LoRAs:</b><br>'
                '• <b>LoRA has no visible effect:</b> '
                'Check that you\'re using the trigger word in your prompt. Try weight 1.0. '
                'Make sure the base model matches the training base '
                '(e.g., don\'t use an Illustrious-trained LoRA on a Pony model).<br>'
                '• <b>Images look exactly like training data:</b> '
                'Overfitting — reduce epochs, lower learning rate, or add more training images.<br>'
                '• <b>Colors / anatomy are broken:</b> '
                'Learning rate too high — try 5e-5. '
                'Or the training data had inconsistent quality.<br>'
                '• <b>LoRA only works at weight 1.0:</b> '
                'Alpha is too low relative to rank. Try alpha = rank × 2.'
                '</div>'
            )

        # ════════════════════════════════════════════════════════════════════
        # EVENT HANDLERS
        # ════════════════════════════════════════════════════════════════════

        def do_scan(folder, res_choice):
            global _train_dataset, _train_assignments
            if not folder or not Path(folder).is_dir():
                return '<p style="color:#f38ba8">❌ Folder not found.</p>', ""
            from backend.dataset_manager import scan_images, compute_bucket_assignments
            _train_dataset = scan_images(folder)
            if not _train_dataset:
                return '<p style="color:#f38ba8">❌ No images found.</p>', ""

            base_res = 512
            if "1024" in res_choice:
                base_res = 1024
            elif "768" in res_choice:
                base_res = 768

            result = compute_bucket_assignments(_train_dataset, base_res=base_res)
            _train_assignments = result["assignments"]

            # Build stats HTML
            total = len(_train_assignments)
            capt_count = sum(1 for a in _train_assignments if a.get("caption"))
            max_crop = max(a["crop_pct"] for a in _train_assignments)
            avg_crop = sum(a["crop_pct"] for a in _train_assignments) / total

            bkt_html = '<div class="tr-h">Bucket Assignments</div>'
            bkt_html += f'<p style="color:#a6e3a1;font-size:13px">✅ {total} images scanned · '
            bkt_html += f'{capt_count} have captions · '
            bkt_html += f'avg crop {avg_crop:.1f}% · max crop {max_crop:.1f}%</p>'
            bkt_html += '<table style="width:100%;font-size:13px;border-collapse:collapse;margin:6px 0">'
            bkt_html += ('<tr style="background:#313244"><th style="padding:4px 8px;text-align:left">Bucket</th>'
                         '<th style="padding:4px 8px;text-align:left">Ratio</th>'
                         '<th style="padding:4px 8px;text-align:left">Images</th></tr>')
            for bkt, cnt in sorted(result["stats"].items(),
                                   key=lambda x: -x[1]):
                w, h = bkt.split("×")
                ratio = int(w) / max(int(h), 1)
                bkt_html += (f'<tr><td style="padding:3px 8px;color:#89b4fa">{bkt}</td>'
                             f'<td style="padding:3px 8px;color:#9399b2">{ratio:.2f}</td>'
                             f'<td style="padding:3px 8px">{cnt}</td></tr>')
            bkt_html += '</table>'
            if max_crop > 15:
                bkt_html += (f'<div class="tr-warn">Some images lose >{max_crop:.0f}% to cropping. '
                             'Consider cropping them manually to a cleaner aspect ratio.</div>')
            return f'<p style="color:#a6e3a1">✅ Found {total} images</p>', bkt_html

        def do_prepare(folder, trigger, res_choice, progress=gr.Progress()):
            global _train_assignments
            if not _train_assignments:
                return "❌ Scan the folder first."
            from backend.dataset_manager import prepare_dataset
            base_res = 1024 if "1024" in res_choice else (768 if "768" in res_choice else 512)
            out_dir = Path(folder) / f"_prepared_{base_res}"

            def cb(cur, tot, msg):
                progress(cur / max(tot, 1), desc=msg)

            msg = prepare_dataset(_train_assignments, out_dir, trigger, callback=cb)
            return f'<p style="color:#a6e3a1">{msg}<br>'  \
                   f'<span class="tr-dim">Use this folder for training: <code>{out_dir}</code></span></p>'

        def do_autocaption(folder, trigger, prefix, progress=gr.Progress()):
            if not folder or not Path(folder).is_dir():
                return "❌ Set folder first."
            from backend.auto_tagger import caption_batch
            from backend.dataset_manager import scan_images, write_caption
            imgs = scan_images(folder)
            if not imgs:
                return "❌ No images."

            def cb(cur, tot, msg):
                progress(cur / max(tot, 1), desc=msg)

            results = caption_batch(
                [i["path"] for i in imgs],
                trigger_word=trigger, prefix=prefix, callback=cb)
            for img in imgs:
                fn = Path(img["path"]).name
                if fn in results:
                    write_caption(img["path"], results[fn])
            return f'<p style="color:#a6e3a1">✅ Captioned {len(results)} images</p>'

        def do_wdtag(folder, trigger, thr, progress=gr.Progress()):
            if not folder or not Path(folder).is_dir():
                return "❌ Set folder first."
            from backend.dataset_manager import scan_images, write_caption
            from backend.wd_tagger import tag_image, tags_text
            from PIL import Image as _Img
            imgs = scan_images(folder)
            if not imgs:
                return "❌ No images."
            trig = (trigger or "").strip().strip(",")
            done, failed = 0, []
            for i, it in enumerate(imgs):
                try:
                    progress((i + 1) / len(imgs), desc=f"Tagging {Path(it['path']).name}")
                except Exception:
                    pass
                try:
                    with _Img.open(it["path"]) as im:
                        tags = tags_text(tag_image(im, general_threshold=float(thr or 0.35)), exclude=trig)
                except Exception as e:
                    failed.append(f"{Path(it['path']).name}: {e}")
                    continue
                write_caption(it["path"], f"{trig}, {tags}" if trig and tags else (trig or tags))
                done += 1
            err = (f'<br><span style="color:#f38ba8;">{len(failed)} failed: {html.escape(failed[0])}</span>'
                   if failed else "")
            return (f'<p style="color:#a6e3a1">✅ Tagged {done} images (trigger word first, then Danbooru tags)'
                    f'{err}</p>')

        def do_unload_blip():
            from backend.auto_tagger import unload_blip
            unload_blip()
            return '<p style="color:#a6e3a1">✅ BLIP model freed</p>'

        def do_refresh_cap_list(folder):
            if not folder or not Path(folder).is_dir():
                return gr.update(choices=[])
            from backend.dataset_manager import scan_images
            imgs = scan_images(folder)
            return gr.update(choices=[i["filename"] for i in imgs])

        def do_select_caption(folder, filename):
            if not folder or not filename:
                return None, ""
            from backend.dataset_manager import read_caption
            p = Path(folder) / filename
            if not p.exists():
                return None, ""
            cap = read_caption(str(p))
            try:
                img = Image.open(p)
                return img, cap
            except Exception:
                return None, cap

        def do_save_caption(folder, filename, caption):
            if not folder or not filename:
                return "❌ No image selected."
            from backend.dataset_manager import write_caption
            p = Path(folder) / filename
            if not p.exists():
                return "❌ Image not found."
            write_caption(str(p), caption)
            return f'<p style="color:#a6e3a1">✅ Caption saved for {filename}</p>'

        def do_refresh_models():
            return gr.update(choices=_checkpoint_choices())

        def _parse_res(res_choice):
            if "1024" in res_choice:
                return 1024
            if "768" in res_choice:
                return 768
            return 512

        @gpu_job
        def do_train_prepare(
            folder, trigger, res_choice, model_path,
            rank, alpha, lr, te_lr, epochs, train_te,
            target, sched, grad_accum, warmup, grad_ckpt, flip,
            save_every, seed, device_choice, output_name, train_clip_skip=2,
            progress=gr.Progress(),
        ):
            global _trainer
            if not model_path:
                return "❌ Select a base model.", ""
            if not folder or not Path(folder).is_dir():
                return "❌ Set dataset folder.", ""

            from backend.lora_trainer import TrainingConfig, LoRATrainer, safe_output_name
            from config import LORAS_DIR

            # Training loads its own copy of the base model; a generation model left in
            # VRAM (6.6 GB for SDXL) would push the card into shared memory and crawl.
            freed = [p.current_model for p in (_sd15, _sdxl) if p.pipe is not None]
            for _p in (_sd15, _sdxl):
                if _p.pipe is not None:
                    _p._unload()
            global _smartsplit_pipe
            _smartsplit_pipe = None       # it holds its own copy of the model (and its VRAM)
            output_name = safe_output_name(output_name)

            # If a _prepared_ subfolder exists, use it
            base_res = _parse_res(res_choice)
            prep_dir = Path(folder) / f"_prepared_{base_res}"
            ds_dir = str(prep_dir) if prep_dir.is_dir() else folder

            cfg = TrainingConfig(
                base_model=model_path,
                lora_rank=int(rank),
                lora_alpha=int(alpha),
                train_te=train_te,
                target_coverage="full" if "full" in target else "attn",
                dataset_dir=ds_dir,
                trigger_word=trigger,
                resolution=base_res,
                learning_rate=float(lr),
                te_learning_rate=float(te_lr),
                lr_scheduler=sched,
                epochs=int(epochs),
                gradient_accumulation=int(grad_accum),
                warmup_ratio=float(warmup),
                gradient_checkpointing=grad_ckpt,
                flip_augment=flip,
                save_every_n_epochs=int(save_every),
                seed=int(seed),
                device=device_choice,
                output_dir=str(LORAS_DIR),
                output_name=output_name,
                clip_skip=2 if str(train_clip_skip) == "2" else 1,
            )

            if _trainer is not None:
                _trainer.cleanup()

            _trainer = LoRATrainer(cfg)

            def cb(cur, tot, msg):
                progress(cur / max(tot, 1), desc=msg)

            msg = _trainer.prepare(callback=cb)
            log = (f'<p style="font-size:13px;color:#a6adc8">'
                   f'Model: {Path(model_path).stem} ({cfg.model_type.upper()})<br>'
                   f'Dataset: {ds_dir} ({len(_trainer.items)} images)<br>'
                   f'LoRA rank={rank} alpha={alpha} | LR={lr} | TE={train_te}<br>'
                   f'Epochs: {epochs} | Device: {device_choice}'
                   + (f'<br>Unloaded the Generate tab model(s) to free VRAM: '
                      f'{", ".join(Path(str(m)).stem for m in freed)} — they reload '
                      f'automatically on your next Generate.' if freed else '') + '</p>')
            color = "#a6e3a1" if "✅" in msg else "#f38ba8"
            return f'<p style="color:{color}">{msg}</p>', log

        def do_train_start(progress=gr.Progress()):
            global _trainer
            no_upd = [gr.update()] * len(gen_lora_dds)
            if _trainer is None:
                return ('<p style="color:#f38ba8">❌ Click 📦 Prepare first.</p>', "", *no_upd)
            def cb(pct, msg):
                progress(pct, desc=msg)

            try:
                result = _trainer.train(callback=cb)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return (f'<p style="color:#f38ba8">❌ Training failed: {e}</p>', "", *no_upd)
            color = "#a6e3a1" if "✅" in result else ("#fab387" if "⏹" in result else "#f38ba8")
            # New LoRA shows up in the Generate tab's slots right away
            lora_upd = gr.update(choices=_lora_choices())
            extra = ('<p class="tr-dim">It is now listed in the LoRA slots of the 🎨 Generate tab '
                     '— remember to add your trigger word to the prompt.</p>' if "✅" in result else "")
            return (f'<p style="color:{color}">{result}</p>{extra}', "", *[lora_upd] * len(gen_lora_dds))

        def do_train_stop():
            global _trainer
            if _trainer:
                _trainer.abort()
            return '<p style="color:#fab387">⏹ Stop signal sent — will finish current step</p>'

        # ── Wiring ─────────────────────────────────────────────────────────
        def on_train_model(model_path, res_choice):
            """Keep the target resolution matched to the base model's family."""
            from backend.sd_pipeline import _is_sdxl
            if not model_path:
                return gr.update()
            if _is_sdxl(model_path):
                return "1024 (SDXL / Pony / Illustrious)"
            return res_choice if res_choice and "SD 1.5" in res_choice else "512 (SD 1.5)"
        train_model_dd.change(on_train_model, [train_model_dd, train_res_dd], [train_res_dd])

        scan_btn.click(do_scan, [train_folder, train_res_dd],
                       [scan_status, bucket_stats_html])
        prepare_btn.click(do_prepare, [train_folder, trigger_word_txt, train_res_dd],
                          [prepare_status])

        wdtag_btn.click(do_wdtag, [train_folder, trigger_word_txt, wdtag_thr_sl], [caption_status])
        autocap_btn.click(
            do_autocaption,
            [train_folder, trigger_word_txt, caption_prefix_txt],
            [caption_status])
        unload_blip_btn.click(do_unload_blip, [], [caption_status])

        caption_refresh_btn.click(do_refresh_cap_list, [train_folder], [caption_img_dd])
        scan_btn.click(do_refresh_cap_list, [train_folder], [caption_img_dd])
        caption_img_dd.change(do_select_caption, [train_folder, caption_img_dd],
                              [caption_preview, caption_txt])
        caption_save_btn.click(do_save_caption,
                               [train_folder, caption_img_dd, caption_txt],
                               [cap_save_status])

        train_model_refresh.click(do_refresh_models, [], [train_model_dd])

        _train_config_inputs = [
            train_folder, trigger_word_txt, train_res_dd, train_model_dd,
            lora_rank_sl, lora_alpha_sl, lr_num, te_lr_num,
            epochs_sl, train_te_cb,
            target_dd, sched_dd, grad_accum_sl, warmup_sl,
            grad_ckpt_cb, flip_cb, save_every_sl, seed_num,
            train_device_dd, output_name_txt, train_clip_skip_rb,
        ]
        train_prepare_btn.click(do_train_prepare, _train_config_inputs,
                                [train_progress, train_log])
        train_start_btn.click(do_train_start, [], [train_progress, train_log, *gen_lora_dds])
        train_stop_btn.click(do_train_stop, [], [train_progress])


# ═════════════════════════════════════════════════════════════════════════════
# Tab 4: Settings
# ═════════════════════════════════════════════════════════════════════════════
def _build_history_tab(gen_controls: dict, up_input, png_input) -> dict:
    """🗂 History: browse, search and star everything in outputs/ (index: backend/history.py)."""
    from backend import history as H
    PAGE = 48
    with gr.Tab("🗂 History", id="history") as tab:
        gr.HTML('<p style="color:#a6adc8;font-size:13px;">Everything in <code>outputs/</code>, newest first. Search '
                'by prompt words, model, LoRA, seed or sampler (all words must match). Click an image for its '
                'settings; ⭐ marks favourites. <b>📄 Open in PNG Info</b> → <b>Send to Generate</b> restores '
                'everything it was made with.</p>')
        with gr.Row():
            h_text = gr.Textbox(label="Search", placeholder="e.g. red dress beach   or   9001", scale=4)
            h_model = gr.Dropdown(["All"], value="All", label="Model", scale=2)
            h_lora = gr.Dropdown(["All"], value="All", label="LoRA", scale=2)
            with gr.Column(scale=1, min_width=120):
                h_fav = gr.Checkbox(label="⭐ only", value=False)
                h_refresh = gr.Button("🔄 Search", size="sm", variant="primary")
        h_status = gr.HTML("")
        h_gallery = gr.Gallery(label="Outputs", columns=8, height=620, object_fit="contain", allow_preview=False,
                               show_label=False)
        with gr.Row():
            h_prev = gr.Button("◀ Newer", size="sm")
            h_page = gr.HTML('<p style="text-align:center;color:#a6adc8;"></p>')
            h_next = gr.Button("Older ▶", size="sm")
        h_names, h_pageno, h_sel = gr.State([]), gr.State(0), gr.State(None)
        h_info = gr.HTML("")
        with gr.Row():
            h_star = gr.Button("⭐ Favourite / unfavourite", size="sm")
            h_png = gr.Button("📄 Open in PNG Info", size="sm", variant="primary")
            h_i2i = gr.Button("🖼 Send to img2img", size="sm")
            h_up = gr.Button("🔍 Send to Upscale", size="sm")
            h_open = gr.Button("📂 Show in folder", size="sm")

    def _page(names, page):
        page = max(0, min(int(page or 0), max(0, (len(names) - 1) // PAGE)))
        favs = H.favourites()
        items = []
        for n in names[page * PAGE:(page + 1) * PAGE]:
            t = H.thumbnail(n)
            if t:
                items.append((t, ("⭐ " if n in favs else "") + n.rsplit(".", 1)[0][-22:]))
        label = (f'<p style="text-align:center;color:#a6adc8;font-size:13px;">page {page + 1} of '
                 f'{max(1, (len(names) + PAGE - 1) // PAGE)} · {len(names)} image(s)</p>')
        return items, label, page

    def do_hist_search(text, model, lora, fav_only):
        idx = H.build_index()
        names = H.search(idx, text, model, lora, bool(fav_only))
        models = sorted({e.get("model") for e in idx.values() if e.get("model")}, key=str.lower)
        loras = sorted({l for e in idx.values() for l in (e.get("loras") or [])}, key=str.lower)
        items, label, page = _page(names, 0)
        status = (f'<p style="color:#a6adc8;font-size:13px;">{len(names)} of {len(idx)} image(s) match.</p>'
                  if idx else '<p style="color:#a6adc8;">No images in outputs/ yet.</p>')
        return (items, label, names, page, None, "", status,
                gr.update(choices=["All"] + models, value=model if model in models else "All"),
                gr.update(choices=["All"] + loras, value=lora if lora in loras else "All"))

    def do_hist_page(names, page, delta):
        items, label, page = _page(names, int(page or 0) + delta)
        return items, label, page, None, ""

    def _info(name):
        if not name:
            return ""
        from backend.png_info import read_png_info
        p = OUTPUTS_DIR / name
        if not p.is_file():
            return '<p style="color:#f38ba8;">That file is gone.</p>'
        star = "⭐ favourite · " if name in H.favourites() else ""
        return (f'<p style="color:#cdd6f4;font-size:13px;margin:4px 0;">{star}<b>{html.escape(name)}</b></p>'
                + format_png_info_html(read_png_info(str(p))))

    def do_hist_select(names, page, evt: gr.SelectData):
        i = int(page or 0) * PAGE + int(evt.index)
        name = names[i] if 0 <= i < len(names) else None
        return name, _info(name)

    def do_hist_star(name, names, page):
        if not name:
            return gr.update(), "", gr.update()
        H.toggle_favourite(name)
        items, label, _ = _page(names, page)
        return items, _info(name), label

    def _load(name):
        p = OUTPUTS_DIR / str(name or "")
        if not name or not p.is_file():
            return None
        im = Image.open(p); im.load()
        im.info["saved_path"] = str(p)
        return im

    def do_hist_open(name):
        if name:
            p = OUTPUTS_DIR / name
            if p.is_file():
                import subprocess
                subprocess.Popen(f'explorer /select,"{p}"')
        return gr.update()

    search_in = [h_text, h_model, h_lora, h_fav]
    search_out = [h_gallery, h_page, h_names, h_pageno, h_sel, h_info, h_status, h_model, h_lora]
    for ev in (tab.select, h_refresh.click, h_text.submit, h_fav.input, h_model.input, h_lora.input):
        ev(do_hist_search, search_in, search_out)
    h_prev.click(lambda n, p: do_hist_page(n, p, -1), [h_names, h_pageno], [h_gallery, h_page, h_pageno, h_sel, h_info])
    h_next.click(lambda n, p: do_hist_page(n, p, +1), [h_names, h_pageno], [h_gallery, h_page, h_pageno, h_sel, h_info])
    h_gallery.select(do_hist_select, [h_names, h_pageno], [h_sel, h_info])
    h_star.click(do_hist_star, [h_sel, h_names, h_pageno], [h_gallery, h_info, h_page])
    h_open.click(do_hist_open, [h_sel], [h_status])
    h_png.click(lambda n: _load(n) or gr.update(), [h_sel], [png_input])
    h_i2i.click(lambda n: (_load(n) or gr.update(), True) if n else (gr.update(), gr.update()),
                [h_sel], [gen_controls["init_image"], gen_controls["use_i2i"]])
    h_up.click(lambda n: _load(n) or gr.update(), [h_sel], [up_input])
    return {"to_png": h_png, "to_generate": h_i2i, "to_upscale": h_up}


def _build_settings_tab():
    profile = get_profile()

    with gr.Tab("⚙️ Settings"):
        with gr.Row():
            with gr.Column():
                from backend.ui_settings import build_prefs_section
                build_prefs_section(sys.modules[__name__])
                gr.Markdown("### 🖥 GPU")

                # ── GPU picker ──────────────────────────────────────────────
                gpu_choices = [_gpu_choice_label(g) for g in profile.gpus] or ["CPU (no GPU detected)"]
                best_idx    = next(
                    (i for i, g in enumerate(profile.gpus) if g == profile.best_gpu), 0
                )
                gpu_radio = gr.Radio(
                    choices=gpu_choices,
                    value=gpu_choices[best_idx] if gpu_choices else None,
                    label="Active GPU for inference",
                    info="★ = auto-selected best GPU. You can override this.",
                )
                gpu_status = gr.HTML("")

                # ── GPU notes (warnings for RDNA4, iGPU, etc.) ─────────────
                gpu_notes_html = gr.HTML(
                    _build_gpu_notes_html(profile)
                )


                gr.Markdown("### 📂 Output folder")
                gr.HTML(f'<p style="color:#a6adc8;font-size:14px;">Generated images are saved to '
                        f'<code>{OUTPUTS_DIR}</code></p>')
                settings_open_out = gr.Button("📂 Open output folder", size="sm")
                settings_open_msg = gr.HTML("")

                def _open_outputs():
                    try:
                        os.startfile(str(OUTPUTS_DIR))
                        return ""
                    except Exception as e:
                        return f'<p style="color:#f38ba8;">Could not open the folder: {html.escape(str(e))}</p>'
                settings_open_out.click(_open_outputs, None, settings_open_msg)

                gr.Markdown("### 🔑 Hugging Face token (optional)")
                hf_token_txt = gr.Textbox(
                    label="HF Token",
                    value="",                    # stored tokens stay out of Gradio's /config
                    type="password",
                    placeholder="●●● token saved — paste a new one to replace it" if HF_TOKEN else "hf_…",
                    info="HuggingFace access token — only for gated repos (e.g. FLUX.1-dev, SD 3.x); the "
                         "built-in SD 1.5 / SDXL base downloads don't need one. "
                         "Get one at huggingface.co/settings/tokens.",
                )
                save_hf_btn    = gr.Button("Save HF Token")
                hf_token_status = gr.HTML("")

            with gr.Column():
                gr.Markdown("### 📁 Installed models")
                inventory_txt = gr.Textbox(
                    value=model_dir_summary(),
                    label="Installed Models",
                    lines=6, interactive=False,
                )
                gr.Button("🔄 Refresh inventory").click(
                    lambda: model_dir_summary(), [], [inventory_txt]
                )

                with gr.Accordion("🔍 Hardware report", open=False):
                    hw_report = gr.Textbox(
                        value=profile.summary,
                        label="Detected Hardware",
                        lines=12,
                        interactive=False,
                    )
                    refresh_hw_btn = gr.Button("🔄 Re-detect Hardware")

                gr.Markdown("### 🔗 Links")
                gr.HTML(
                    '<p style="font-size:14px;line-height:1.9;">'
                    '<a href="https://civitai.com" target="_blank">Civitai — models &amp; LoRAs ↗</a><br>'
                    '<a href="https://huggingface.co/models?pipeline_tag=text-to-image" target="_blank">Hugging Face models ↗</a><br>'
                    '<a href="https://github.com/vosen/ZLUDA" target="_blank">ZLUDA (CUDA on AMD) ↗</a>'
                    '</p>'
                )

        with gr.Accordion("🧪 Advanced — experimental features (not needed on most PCs)", open=False):
            # ── NPU (only shown when one is detected) ─────────────────────
            if profile.npu.available:
                gr.Markdown("### ⚡ AMD XDNA NPU (Ryzen AI)")
                gr.HTML(_npu_badge(profile.npu))
                gr.Textbox(value=_build_npu_info(profile.npu), lines=5,
                           interactive=False, label="NPU Details")

            # Accelerated text encoding for SDXL (iGPU DML)
            _has_dml = "DmlExecutionProvider" in __import__("onnxruntime").get_available_providers()
            npu_sdxl_te = gr.Checkbox(
                label="⚡ Accelerated SDXL text encoding (iGPU)",
                value=False,
                interactive=_has_dml,
                info=(
                    "Offloads CLIP-L + OpenCLIP-G text encoders to the integrated GPU "
                    "via ONNX Runtime DirectML during SDXL/Pony/Illustrious generation. "
                    "Mainly helps the DirectML backend — under ZLUDA the text encoders "
                    "already run on the dGPU. Reduces CPU prompt encoding from ~7s to ~100ms (780M). "
                    "ONNX models are exported on first use and cached."
                    if _has_dml else
                    "DirectML not available — install onnxruntime-directml."
                ),
            )

            # ── SmartSplit — heterogeneous pipeline parallelism ──────────
            gr.Markdown("### 🔀 SmartSplit — Multi-Unit Pipeline")
            _cap = _get_smartsplit_cap()
            gr.HTML(_build_smartsplit_cap_html(_cap))

            if _cap.possible:
                if _cap.dgpu_dml_idx >= 0:
                    gr.HTML(
                        f'<p style="color:#cba6f7;font-size:13px;margin:4px 0 8px;">'
                        f'🎮 <b>DirectML split mode</b> — no ZLUDA required.<br>'
                        f'UNet stays on <b>{_cap.dgpu_name}</b> (dGPU DirectML).<br>'
                        f'Text Encoder + VAE offloaded to <b>CPU</b>, freeing ~3 GB dGPU VRAM.<br>'
                        f'Device dropdowns below are ignored in this mode.</p>'
                    )
                smartsplit_enable = gr.Checkbox(
                    label="Enable SmartSplit (iGPU + NPU + dGPU)",
                    value=False,
                    info=(
                        "Distributes pipeline stages across all available compute units: "
                        "text encoder → NPU/iGPU, UNet → dGPU, VAE decode → iGPU. "
                        "Frees up to ~3-4 GB extra dGPU VRAM."
                    ),
                )

                te_device_dd = gr.Dropdown(
                    choices=_smartsplit_te_choices(_cap),
                    value=_smartsplit_te_choices(_cap)[0],
                    label="Text Encoder device",
                    info="Where to run the CLIP text encoder. NPU frees ~1 GB "
                         "dGPU VRAM. iGPU is the next best option.",
                )
                vae_device_dd = gr.Dropdown(
                    choices=_smartsplit_vae_choices(_cap),
                    value=_smartsplit_vae_choices(_cap)[0],
                    label="VAE Decode device",
                    info="Where to decode latents to pixels. iGPU frees ~2 GB "
                         "dGPU VRAM. CPU works but is slower.",
                )
                apply_split_btn = gr.Button("⚙ Apply SmartSplit Config", variant="secondary")
                split_status    = gr.HTML("")

                def on_apply_split(enabled, te_dev, vae_dev):
                    global _smartsplit_cfg, _smartsplit_pipe
                    _cap_now = _get_smartsplit_cap()
                    is_dml_split = enabled and _cap_now.dgpu_dml_idx >= 0

                    if is_dml_split:
                        # DirectML multi-GPU mode — auto-configured, no ZLUDA needed
                        _smartsplit_cfg = SmartSplitConfig(
                            enabled=True,
                            text_encoder_device="igpu_directml",
                            unet_device="dgpu_directml",
                            vae_device="igpu_directml",
                            dgpu_dml_idx=_cap_now.dgpu_dml_idx,
                            igpu_dml_idx=_cap_now.igpu_dml_idx,
                        )
                    else:
                        _smartsplit_cfg = SmartSplitConfig(
                            enabled=enabled,
                            text_encoder_device=te_dev,
                            unet_device="cuda",
                            vae_device=vae_dev,
                        )

                    # SmartSplit only applies to SD 1.5 pipeline
                    _sd15.unet_only_dml = is_dml_split

                    _smartsplit_pipe = None
                    reload_msg = ""
                    # SmartSplit only works with SD 1.5 — skip if SDXL model is active
                    if enabled and isinstance(sd, SDXLPipeline):
                        msg = '<p style="color:#f5c6a0;">⚠ SmartSplit not supported for SDXL/Pony/Illustrious. Load an SD 1.5 model first.</p>'
                    elif enabled and sd.current_model:
                        # Force a reload so device placement takes effect immediately.
                        # torch_directml cannot free VRAM after allocation, so we must
                        # reload the model with the correct placement from the start.
                        prev = sd.current_model
                        # Preserve active LoRAs so we can re-apply them after reload.
                        saved_adapters = dict(sd._lora_adapters)
                        sd.current_model = None  # bypass "already loaded" guard
                        r = sd.load_model(prev, sd._last_vae_path)
                        reload_msg = f" (model reloaded: {Path(prev).stem})"
                        # Re-apply LoRAs that were active before the reload.
                        if sd.pipe is not None and saved_adapters:
                            sd._lora_adapters = saved_adapters
                            try:
                                sd._reload_all_loras()
                            except Exception as e:
                                print(f"[SmartSplit] LoRA re-apply after reload failed: {e}")
                        if sd.pipe is not None:
                            try:
                                stem = Path(prev).stem
                                _smartsplit_pipe = SmartSplitPipeline(sd.pipe, _smartsplit_cfg, stem)
                                msg = f'<p style="color:#a6e3a1;">✅ {_smartsplit_cfg.describe()}{reload_msg}</p>'
                            except Exception as e:
                                msg = f'<p style="color:#f38ba8;">❌ SmartSplit init failed: {e}</p>'
                        else:
                            msg = f'<p style="color:#f38ba8;">❌ Model reload failed: {r}</p>'
                    elif enabled:
                        msg = '<p style="color:#f5c6a0;">⚠ Load a model in Generate first, then re-apply.</p>'
                    else:
                        msg = '<p style="color:#a6adc8;">SmartSplit disabled.</p>'
                    return msg

                apply_split_btn.click(
                    on_apply_split,
                    [smartsplit_enable, te_device_dd, vae_device_dd],
                    [split_status],
                )

            else:
                gr.HTML(
                    f'<p style="color:#9399b2;font-size:13px;">'
                    f'SmartSplit not available on this system.<br>'
                    f'{_cap.reason}</p>'
                )

            # ── Accelerated TE toggle handler ─────────────────────────────
            def on_accel_te_toggle(enabled):
                from backend.sdxl_pipeline import set_sdxl_npu_te
                set_sdxl_npu_te(enabled)
                state = "✅ enabled — iGPU DML" if enabled else "disabled"
                return gr.update(info=f"Accelerated text encoding {state}")

            npu_sdxl_te.change(
                on_accel_te_toggle,
                [npu_sdxl_te],
                [npu_sdxl_te],
            )


    # ── GPU change handler ────────────────────────────────────────────────────
    def on_gpu_select(choice):
        import config as _cfg
        # Parse index from label "GPU 0: ..."  or "★ GPU 0: ..."
        g = next((g for g in profile.gpus if f"GPU {g.index}:" in (choice or "")), None)
        if g is None:
            return '<p style="color:#f38ba8;">❌ Unknown GPU selection.</p>'
        if g.backend == "cuda_zluda":
            set_active_gpu(max(g.hip_index, 0), backend="cuda")
        elif g.backend == "directml":
            # DirectML numbers devices in its own order — find this GPU by name
            try:
                import torch_directml
                names = [torch_directml.device_name(i).rstrip("\x00")
                         for i in range(torch_directml.device_count())]
                dml_idx = next(i for i, n in enumerate(names)
                               if n.lower() == g.name.lower())
            except Exception:
                return (f'<p style="color:#f38ba8;">❌ {g.name} is not available through DirectML '
                        f'(install torch-directml).</p>')
            set_active_gpu(dml_idx, backend="directml")
        else:
            set_active_gpu(0, backend="cpu")
        for _p in (_sd15, _sdxl):     # weights live on the old device
            _p._unload()
        global _smartsplit_pipe
        _smartsplit_pipe = None       # built for the old device
        dev = _cfg.DEVICE
        slow = (' iGPUs are much slower than the dGPU for image generation.'
                if g.is_igpu else '')
        return (f'<p style="color:#a6e3a1;">✅ Now using {g.name} ({dev}). The model reloads '
                f'on your next Generate.{slow}</p>')

    gpu_radio.change(on_gpu_select, [gpu_radio], [gpu_status])

    def on_refresh_hw():
        from backend import hardware_detector as _hd
        _hd._PROFILE = None  # clear cache
        p = _hd.get_profile()
        return p.summary

    refresh_hw_btn.click(on_refresh_hw, [], [hw_report])

    def save_hf_token(token):
        t = token.strip()
        if t:
            _persist_key("huggingface", t)
            os.environ["HF_TOKEN"] = t
            os.environ["HUGGING_FACE_HUB_TOKEN"] = t
            try:
                from huggingface_hub import login
                login(token=t, add_to_git_credential=False)
                return '<p style="color:#a6e3a1;">✅ HF token saved (persists across restarts).</p>'
            except Exception as e:
                return f'<p style="color:#f38ba8;">⚠ Saved to disk but HF login failed: {e}</p>'
        return '<p style="color:#f38ba8;">Token is empty.</p>'

    save_hf_btn.click(save_hf_token, [hf_token_txt], [hf_token_status])


# ── Settings tab helper renderers ─────────────────────────────────────────────
def _build_gpu_notes_html(profile) -> str:
    notes = []
    for g in profile.gpus:
        if g.notes:
            notes.append(f'<li><b>GPU {g.index} ({g.name}):</b> {g.notes}</li>')
    if not notes:
        return ""
    return (
        '<ul style="font-size:13px;color:#a6adc8;padding-left:16px;margin:4px 0;">'
        + "".join(notes)
        + "</ul>"
    )


def _build_npu_info(npu: NPUInfo) -> str:
    if not npu.name or npu.name == "Not detected":
        return "No XDNA NPU detected on this system."
    lines = [f"Name    : {npu.name}", f"Backend : {npu.backend or 'N/A'}"]
    if npu.notes:
        lines.append("")
        lines.append(npu.notes)
    if npu.available:
        lines += [
            "",
            "Supported offload targets:",
            "  • Text Encoder (CLIP) — low latency, frees GPU VRAM",
            "  • VAE Decode — useful on systems with limited dGPU VRAM",
            "",
            "Enable via: Settings → SmartSplit → set Text Encoder to 'npu'",
            "NPU inference runs via the Ryzen AI conda env subprocess bridge.",
        ]
    return "\n".join(lines)


# ── SmartSplit helpers ─────────────────────────────────────────────────────────
_cached_smartsplit_cap: SmartSplitCapability | None = None

def _get_smartsplit_cap() -> SmartSplitCapability:
    global _cached_smartsplit_cap
    if _cached_smartsplit_cap is None:
        _cached_smartsplit_cap = detect_smartsplit_capability()
    return _cached_smartsplit_cap


def _build_smartsplit_cap_html(cap: SmartSplitCapability) -> str:
    if not cap.possible:
        return ""
    parts = []
    if cap.dgpu_available:
        parts.append(f'<span style="color:#89b4fa;">dGPU: {cap.dgpu_name}</span>')
    if cap.igpu_name:
        parts.append(f'<span style="color:#a6e3a1;">iGPU: {cap.igpu_name}</span>')
    if cap.npu_available:
        parts.append(f'<span style="color:#cba6f7;">NPU: {cap.npu_name}</span>')
    inner = " &nbsp;|&nbsp; ".join(parts)
    return (
        f'<div style="background:#181825;border:1px solid #313244;border-radius:8px;'
        f'padding:8px 12px;font-size:13px;margin:4px 0;">'
        f'🔀 <b>SmartSplit capable:</b>&nbsp; {inner}'
        f'</div>'
    )


def _build_smartsplit_banner() -> str:
    cap = _get_smartsplit_cap()
    if not (cap.possible and _smartsplit_cfg.enabled):
        return ""
    _style = ('background:#1e3a2e;border:1px solid #40a060;border-radius:6px;'
              'padding:4px 10px;font-size:13px;color:#a6e3a1;margin:4px 0;')
    if _smartsplit_cfg.unet_device == "dgpu_directml":
        dml_dgpu = cap.dgpu_name or f"DML:{_smartsplit_cfg.dgpu_dml_idx}"
        return (
            f'<div style="{_style}">'
            f'🔀 SmartSplit active: Text+VAE→CPU  |  UNet→{dml_dgpu}'
            f'</div>'
        )
    return (
        f'<div style="{_style}">'
        f'🔀 SmartSplit active: '
        f'Text→{_smartsplit_cfg.text_encoder_device.upper()}  '
        f'UNet→dGPU  '
        f'VAE→{_smartsplit_cfg.vae_device.upper()}'
        f'</div>'
    )


def _smartsplit_te_choices(cap: SmartSplitCapability) -> list[str]:
    choices = []
    if cap.npu_available:      choices.append("npu")
    if cap.igpu_te_available:  choices.append("igpu_directml")
    choices.append("cpu")
    return choices


def _smartsplit_vae_choices(cap: SmartSplitCapability) -> list[str]:
    choices = []
    if cap.igpu_vae_available: choices.append("igpu_directml")
    choices.append("cpu")
    return choices




_loras_from_meta = _records.loras_from_meta


_match_local = _records.match_local


def _restore_plan(meta: dict) -> dict:
    """What to put in the Generate controls to make an image again, from read_png_info() (backend/records.restore_plan with this app's model lists and sampler table)."""
    plan = _records.restore_plan(meta, list_checkpoints=list_checkpoints, list_vaes=list_vaes, list_loras=list_loras, scheduler_map=SCHEDULER_MAP)
    ed = plan.get("eye_detail") or {}
    if ed:                                                                  # the eye style is a global (Settings / card), not a control: say when it differs from the picture's
        from backend.detail_tools import EYE_STYLE
        pic = "natural" if ed.get("style") == "natural" else "round"
        if pic != EYE_STYLE["mode"]:
            plan["notes"].append(f"this picture's eye pass used the {pic} eye style (Settings → Eye style is {EYE_STYLE['mode']})")
    return plan


def _plan_extra_updates(plan: dict) -> list:
    """CLIP skip, variation seed/strength, hires-fix and detail-pass controls for a restore plan (backend/records.plan_extra_updates; `gr.update()` = keep the control)."""
    return _records.plan_extra_updates(plan, _clean_extra, gr.update())


def _plan_summary(plan: dict) -> str:
    bits = []
    if plan.get("model"):
        bits.append(f"model <b>{html.escape(Path(str(plan['model'])).stem)}</b>")
    if plan.get("vae") and plan["vae"] != "none":
        bits.append(f"VAE {html.escape(Path(plan['vae']).stem)}")
    if plan.get("loras"):
        bits.append("LoRAs " + ", ".join(f"{html.escape(Path(p).stem)} ×{w:g}" for p, w in plan["loras"]))
    for k, lbl in (("seed", "seed"), ("steps", "steps"), ("cfg_scale", "CFG")):
        if plan.get(k) is not None:
            bits.append(f"{lbl} {plan[k]}")
    if plan.get("clip_skip", 1) > 1:
        bits.append(f"CLIP skip {plan['clip_skip']}")
    if plan.get("var_strength"):
        bits.append(f"variation {plan['var_seed']} ×{plan['var_strength']:g}")
    if plan.get("hires"):
        h = plan["hires"]
        bits.append(f"hires fix ×{h.get('scale')} (denoise {h.get('denoise')})")
    if plan.get("face_detail"):
        bits.append(f"face detail (denoise {plan['face_detail'].get('denoise')})")
    if plan.get("hand_detail"):
        bits.append(f"hand detail (denoise {plan['hand_detail'].get('denoise')})")
    if plan.get("eye_detail"):
        bits.append(f"eye detail (denoise {plan['eye_detail'].get('denoise')}{', natural style' if plan['eye_detail'].get('style') == 'natural' else ''})")
    if plan.get("scheduler"):
        bits.append(plan["scheduler"])
    out = ", ".join(bits)
    if plan["missing"]:
        out += (' · <span style="color:#f38ba8;">not found locally: '
                + ", ".join(html.escape(m) for m in plan["missing"]) + "</span>")
    if plan["notes"]:
        out += " · " + "; ".join(plan["notes"])
    return out


def _kernel_cache_note() -> str:
    """Warning HTML when ZLUDA hasn't compiled its GPU kernels on this PC yet ('' otherwise).
    Measured on the RX 6800M: with an empty cache the first SD 1.5 image took ~15 min
    (every kernel compiles once); afterwards it's seconds. Nothing tells users that."""
    if not str(DEVICE).startswith("cuda"):
        return ""
    db = Path(os.environ.get("LOCALAPPDATA", "")) / "zluda" / "ComputeCache" / "zluda.db"
    try:
        if db.exists() and db.stat().st_size > 20 * 2**20:
            return ""
    except OSError:
        return ""
    return ('<p style="color:#f9e2af;margin-top:6px;">⏳ <b>First run on this PC:</b> the GPU compiles '
            'its kernels once, so the <b>first image can take 10–15 minutes</b> (it may look stuck — '
            'it isn\'t). After that, images take seconds. SDXL compiles a few more the first time.</p>')


def _hf_download_gb(model_path) -> float | None:
    """GB a HuggingFace repo ID will download on first load (fp16 weights), None if it's a
    local file or already cached."""
    p = str(model_path or "")
    if not p or Path(p).exists() or "/" not in p:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        # model_index.json alone proves nothing: from_single_file caches the repo's configs
        if any(isinstance(try_to_load_from_cache(p, f), str) for f in (
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "unet/diffusion_pytorch_model.safetensors")):
            return None
    except Exception:
        pass
    return 7.0 if "xl" in p.lower() else 2.0


def _hf_disk_problem(model_path) -> str:
    """Error text if a pending HuggingFace download won't fit on the cache drive, else ''."""
    gb = _hf_download_gb(model_path)
    if not gb:
        return ""
    try:
        import shutil
        from huggingface_hub import constants
        cache = Path(constants.HF_HUB_CACHE)
        cache.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(cache).free / 2**30
    except Exception:
        return ""
    if free >= gb + 1:
        return ""
    return (f"{model_path} needs a ~{gb:g} GB download, but only {free:.1f} GB is free on "
            f"{cache.anchor}. Free up space or pick a local checkpoint.")


def _unique_output(base: str) -> Path:
    """outputs/<base>.png, or <base>_1.png, _2… — never overwrites an earlier result."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path, n = OUTPUTS_DIR / f"{base}.png", 1
    while path.exists():
        path, n = OUTPUTS_DIR / f"{base}_{n}.png", n + 1
    return path


def _booster_record(pipe, ex: dict) -> dict:
    """PAG / FreeU / CFG rescale as they were applied (auto CFG rescale resolved), only
    the ones in use — so a plain image's record stays as before."""
    from backend.sampling import boosters
    b = boosters(pipe)
    out = {}
    if b["pag"] > 0:
        out["pag_scale"] = b["pag"]
    if b["freeu"]:
        out["freeu"] = True
    if b["cfg_rescale"] > 0:
        out["cfg_rescale"] = round(b["cfg_rescale"], 3)
    return out


_PREFS_FILE = SETTINGS_DIR / "_prefs.json"


def _load_prefs() -> dict:
    try:
        return _json.loads(_PREFS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_pref(key: str, value) -> None:
    prefs = _load_prefs(); prefs[key] = value
    try:
        _PREFS_FILE.write_text(_json.dumps(prefs, indent=1), encoding="utf-8")
    except OSError as e:
        print(f"[Prefs] could not save {key}: {e}")


from backend.ui_settings import POWER_LABELS as _POWER_LABELS, POWER_INFO as _POWER_INFO, power_choice as _power_choice      # the Settings tab's preference block lives there


_gen_record = _records.gen_record


def _params_text(rec: dict) -> str:
    """A1111 'parameters' text (read by A1111/Forge/Civitai and by PNG Info)."""
    text = rec.get("prompt", "") or ""
    if rec.get("negative_prompt"):
        text += f"\nNegative prompt: {rec['negative_prompt']}"
    parts = []
    for key, label in (("steps", "Steps"), ("scheduler", "Sampler"), ("cfg_scale", "CFG scale"), ("seed", "Seed")):
        if rec.get(key) is not None:
            parts.append(f"{label}: {rec[key]}")
    if rec.get("width") and rec.get("height"):
        parts.append(f"Size: {rec['width']}x{rec['height']}")
    if rec.get("var_seed") is not None and rec.get("var_strength"):
        parts.append(f"Variation seed: {rec['var_seed']}, Variation seed strength: {rec['var_strength']:g}")
    if rec.get("clip_skip"):
        parts.append(f"Clip skip: {rec['clip_skip']}")
    hires = rec.get("hires") or {}
    if hires:
        parts.append(f"Hires upscale: {hires['scale']:g}, Hires steps: {hires['steps']}, "
                     f"Hires upscaler: {hires['upscaler']}, Denoising strength: {hires['denoise']:g}")
    fd = rec.get("face_detail") or {}
    if fd:
        parts.append(f"Face detail: denoise {fd['denoise']:g} ({fd['detector']})")
    if (rec.get("hand_detail") or {}).get("denoise") is not None:
        parts.append(f"Hand detail: denoise {_num(rec['hand_detail']['denoise'], 0.35):g}")
    if (rec.get("eye_detail") or {}).get("denoise") is not None:
        parts.append(f"Eye detail: denoise {_num(rec['eye_detail']['denoise'], 0.3):g}")
        if rec["eye_detail"].get("style") == "natural":
            parts.append("Eye style: natural")
    if rec.get("emphasis") == "compel":
        parts.append("Emphasis: Compel")
    if rec.get("pag_scale"):
        parts.append(f"PAG scale: {rec['pag_scale']:g}")
    if rec.get("freeu"):
        parts.append("FreeU: on")
    if rec.get("cfg_rescale"):
        parts.append(f"CFG rescale: {rec['cfg_rescale']:g}")
    if rec.get("mode") == "inpaint":
        parts.append(f"Inpaint: denoise {rec.get('strength')}, padding {rec.get('inpaint_padding')}")
    model = rec.get("model") or {}
    if model.get("sha256_10"):
        parts.append(f"Model hash: {model['sha256_10']}")
    if model.get("file"):
        f = str(model["file"])
        parts.append(f"Model: {Path(f).stem if f.lower().endswith(('.safetensors', '.ckpt', '.pt', '.bin')) else f}")
    if rec.get("vae"):
        parts.append(f"VAE: {rec['vae']['file']}")
    if rec.get("strength") is not None:
        parts.append(f"Denoising strength: {rec['strength']}")
    if rec.get("source_image"):
        parts.append(f"img2img source: {rec['source_image']}")
    parts.append("Version: ImageGen Studio")
    if rec.get("loras"):   # last: PNG Info reads "LoRAs:" to the end of the line
        parts.append("LoRAs: " + ", ".join(f"{Path(l['file']).stem}:{l['weight']:g}" for l in rec["loras"]))
    return text + ("\n" + ", ".join(parts) if parts else "")


def _save_outputs(images: list, meta: dict | None = None, pipe=None) -> list[Path]:
    """Save images with their generation record: A1111 'parameters' text plus an 'imagegen'
    JSON chunk with the exact model / VAE / LoRA files and weights. meta['seeds'] holds the
    seed actually used for each image (so every file is reproducible on its own)."""
    from PIL.PngImagePlugin import PngInfo
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    seeds = list(meta.pop("seeds", None) or [meta.pop("seed", -1)])
    var_seeds = list(meta.pop("var_seeds", None) or [])
    if "img2img_strength" in meta:
        meta["strength"] = meta.pop("img2img_strength")
    legacy_model, legacy_loras = meta.pop("model", None), meta.pop("loras", None)
    ts = int(time.time())
    saved = []
    for i, img in enumerate(images):
        seed_i = seeds[i] if i < len(seeds) else seeds[-1]
        extra_i = {"var_seed": var_seeds[i] if i < len(var_seeds) else var_seeds[-1]} if var_seeds else {}
        rec = _gen_record(pipe, **dict(meta, seed=seed_i, **extra_i))
        if pipe is None and legacy_model:
            rec["model"] = {"file": legacy_model}
        if pipe is None and legacy_loras:
            rec["loras"] = [{"file": n, "weight": w} for n, w in _loras_from_meta("", legacy_loras)[0]]
        if not rec.get("width") and hasattr(img, "size"):
            rec["width"], rec["height"] = img.size
        pnginfo = PngInfo()
        params_text = _params_text(rec)
        pnginfo.add_text("parameters", params_text)
        pnginfo.add_itxt("imagegen", _json.dumps(rec, ensure_ascii=False))
        img.info["parameters"] = params_text   # travels with the image to Upscale / img2img
        img.info["imagegen"] = _json.dumps(rec, ensure_ascii=False)
        path = OUTPUTS_DIR / f"{ts}_seed{seed_i}_{i}.png"
        n = 1
        while path.exists():   # two runs in the same second with the same seed
            path = OUTPUTS_DIR / f"{ts}_seed{seed_i}_{i}_{n}.png"
            n += 1
        img.save(path, pnginfo=pnginfo)
        img.info["saved_path"] = str(path)   # lets "📂 Show in folder" highlight this file
        saved.append(path)
    return saved


_ARCH_TAG = {"sd1": "SD 1.5", "sd2": "SD 2", "sdxl": "SDXL"}


def _tagged(items: list, arch_fn) -> list:
    """(name, path) → ("SDXL · name", path): the base model is read from the file header,
    so an SD 1.5 LoRA is easy to tell apart from an SDXL one before anything loads."""
    out = []
    for name, path in items:
        tag = _ARCH_TAG.get(arch_fn(path))
        out.append((f"{tag} · {name}" if tag else name, path))
    return out


def _checkpoint_choices() -> list:
    from backend.model_manager import checkpoint_arch
    return _tagged(list_checkpoints(), checkpoint_arch)


def _lora_choices(model_path=None) -> list:
    """LoRA dropdown choices. With a checkpoint given, the LoRAs made for its model family
    come first (alphabetical within each group), so the ones that can work are on top."""
    from backend.model_manager import lora_arch
    items = _tagged(list_loras(), lora_arch)
    if model_path:
        from backend.sd_pipeline import _is_sdxl
        try:
            want = "SDXL" if _is_sdxl(str(model_path)) else "SD 1.5"
        except Exception:
            want = None
        if want:
            items.sort(key=lambda c: not c[0].startswith(want + " ·"))   # stable: keeps A→Z
    return ["none"] + items


def _refresh_checkpoints() -> list:
    """Return checkpoints + common HF defaults if nothing local is installed."""
    local = _checkpoint_choices()  # list of ("SDXL · file", full_path) tuples
    local_values = {v for _, v in local}
    defaults = [
        # runwayml/… is only a redirect now and stabilityai/…-2-1 is no longer public (401)
        ("SD 1.5 base — Hugging Face, ~2 GB download", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
        ("SDXL base — Hugging Face, ~7 GB download", "stabilityai/stable-diffusion-xl-base-1.0"),
    ]
    return local + [d for d in defaults if d[1] not in local_values]


# ═════════════════════════════════════════════════════════════════════════════
# Build & launch
# ═════════════════════════════════════════════════════════════════════════════
# Ctrl+Enter (Cmd+Enter on Mac) presses Generate while the Generate tab is visible
_KEYBOARD_JS = """
() => {
  // The whole UI is styled for dark mode (inline colours included). Gradio otherwise
  // follows the OS light/dark setting, which gave white panels with light-grey text.
  const keepDark = () => { if (!document.body.classList.contains('dark')) document.body.classList.add('dark'); };
  keepDark();
  new MutationObserver(keepDark).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      const b = document.getElementById('gen-btn');
      if (b && b.offsetParent !== null) { e.preventDefault(); b.click(); }
    }
  });
  // Finished while you were in another tab? Say so in the tab title until you come back.
  // Gradio 4.19 marks the outputs of a running event with the "pending" class.
  const baseTitle = document.title;
  let busy = false, busySince = 0;
  const check = () => {
    const now = !!document.querySelector('.pending');
    if (now && !busy) busySince = Date.now();
    if (!now && busy && document.hidden && Date.now() - busySince > 3000) {
      document.title = '✅ Done — ' + baseTitle;
    }
    busy = now;
  };
  new MutationObserver(check).observe(document.body,
    { subtree: true, attributes: true, attributeFilter: ['class'], childList: true });
  const reset = () => { if (document.title !== baseTitle && !document.hidden) document.title = baseTitle; };
  document.addEventListener('visibilitychange', reset);
  window.addEventListener('focus', reset);
  document.addEventListener('pointerdown', reset, true);
  document.addEventListener('keydown', reset, true);
}
"""


def build_app() -> gr.Blocks:
    css = """
    body, .gradio-container { background: #1e1e2e !important; color: #cdd6f4 !important; }
    .gradio-container { max-width: 1920px !important; }   /* use the screen; gallery needs the room */
    .gr-button-primary { background: #89b4fa !important; color: #1e1e2e !important; }
    .gr-button-secondary { border: 1px solid #45475a !important; color: #cdd6f4 !important; }
    .gr-input, .gr-dropdown, textarea, input[type=text] {
        background: #181825 !important; color: #cdd6f4 !important;
        border: 1px solid #45475a !important;
    }
    .gr-panel { background: #1e1e2e !important; }
    .tab-nav button { color: #cdd6f4 !important; }
    .tab-nav button.selected { border-bottom: 2px solid #89b4fa !important; }

    /* the output column stays in view while the long controls column scrolls */
    /* (overflow: clip clips like hidden but isn't a scroll container, so sticky works against the page) */
    .gradio-container { overflow: clip !important; }
    #gen-output-col { position: sticky; top: 8px; align-self: flex-start; }

    /* ── Readability & accessibility ─────────────────────────────────────── */
    .gradio-container { font-size: 15px; }
    .tab-nav button { font-size: 15px !important; padding: 8px 14px !important; }
    [data-testid="block-info"], .info, span.info { font-size: 13px !important; color: #bac2de !important; }
    a { color: #89b4fa; text-decoration: underline; }
    /* Keyboard users can see where focus is */
    :focus-visible { outline: 3px solid #89b4fa !important; outline-offset: 2px !important; }
    button:focus-visible, [role="tab"]:focus-visible { box-shadow: 0 0 0 3px #89b4fa !important; }
    /* Respect the OS "reduce motion" setting */
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important;
                               transition-duration: .01ms !important; scroll-behavior: auto !important; }
    }
    """

    with gr.Blocks(
        title="ImageGen Studio",
        analytics_enabled=False,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=["ui-monospace", "Consolas", "monospace"],
        ),
        css=css,
        js=_KEYBOARD_JS,
    ) as app:
        _profile = get_profile()
        gr.HTML(
            f'<div style="display:flex;align-items:center;gap:12px;padding:12px 0;flex-wrap:wrap;">'
            f'<span style="font-size:28px;font-weight:700;color:#cdd6f4;">🦞 ImageGen Studio</span>'
            f'&nbsp;&nbsp;{_device_badge()}'
            + (f'&nbsp;{_npu_badge(_profile.npu)}' if _profile.npu.available else '')
            + f'<span style="color:#9399b2;font-size:13px;margin-left:auto;">'
            f'Diffusers · ZLUDA · ROCm</span>'
            f'</div>'
        )

        with gr.Tabs() as main_tabs:
            _gen_model_dd, _gen_lora_dd, _gen_vae_dd, _gen_ctrl = _build_generate_tab()
            _build_bridge_tab()
            _up_input = _build_upscale_tab()
            _build_watermark_tab()
            _png_btns = _build_png_info_tab(_gen_ctrl, _up_input)
            _build_civitai_tab(_gen_model_dd, _gen_lora_dd, _gen_vae_dd,
                               (_gen_ctrl["lora_dd2"], _gen_ctrl["lora_dd3"]))
            _build_train_tab((_gen_lora_dd, _gen_ctrl["lora_dd2"], _gen_ctrl["lora_dd3"]))
            _hist_btns = _build_history_tab(_gen_ctrl, _up_input, _png_btns["input"])
            _build_settings_tab()
            build_help_tab()

        # "Send to …" buttons take you to where the image went
        _TO_TOP_JS = "() => window.scrollTo({top: 0, behavior: 'smooth'})"
        for _b in _png_btns["to_generate"]:
            _b.click(lambda: gr.Tabs(selected="generate"), None, main_tabs)
        _hist_btns["to_png"].click(lambda: gr.Tabs(selected="pnginfo"), None, main_tabs)
        _hist_btns["to_generate"].click(lambda: gr.Tabs(selected="generate"), None, main_tabs)
        _hist_btns["to_upscale"].click(lambda: gr.Tabs(selected="upscale"), None, main_tabs)
        for _b in _png_btns["to_upscale"]:
            _b.click(lambda: gr.Tabs(selected="upscale"), None, main_tabs).then(
                None, None, None, js=_TO_TOP_JS)

        # Wire Gallery -> Upscale button
        def do_send_gal_to_up(sel, imgs):
            target = sel if sel is not None else (imgs[0] if imgs else None)
            if target is None:
                return gr.update(), '<p style="color:#f38ba8;font-size:13px;">⚠ No image available to send.</p>'
            return target, '<p style="color:#a6e3a1;font-size:13px;">✅ Image sent to Upscale tab!</p>'

        _gen_ctrl["gal_send_up_btn"].click(
            do_send_gal_to_up,
            [_gen_ctrl["selected_image"], _gen_ctrl["last_images"]],
            [_up_input, _gen_ctrl["gal_status"]],
        ).then(
            lambda imgs: gr.Tabs(selected="upscale") if imgs else gr.update(),
            [_gen_ctrl["last_images"]], main_tabs,
        ).then(None, None, None, js=_TO_TOP_JS)

    _serialize_gpu_events(app)
    return app


# Events that load, change or run the pipelines / the GPU. Gradio 4 limits concurrency per event, so
# Generate could start while an X/Y grid, Auto-Loop or Bridge run was still going: both threads then
# loaded models / fused LoRAs on the same pipelines and the process died (access violation in
# safetensors load_file, 2026-09-30). One shared queue slot runs them one after another; Stop,
# typing helpers and the CPU-only taggers stay outside it.
_GPU_FNS = {"do_generate_ui", "do_xy_grid", "do_autoloop", "do_bridge", "do_inpaint_ui", "do_load_model",
            "do_upscale", "do_upscale_folder", "do_detect", "do_remove", "do_remove_folder", "do_train_start",
            "do_autocaption", "do_unload_blip", "on_apply_split", "on_gpu_select", "on_accel_te_toggle"}


def _serialize_gpu_events(blocks) -> int:
    n = 0
    for f in blocks.fns:
        if f.fn is not None and (getattr(f.fn, "__name__", "") in _GPU_FNS or getattr(f.fn, "_gpu_job", False)):
            f.concurrency_id, f.concurrency_limit = "gpu", 1
            if not getattr(f.fn, "_gpu_locked", False):
                f.fn = _locked(f.fn)
                f.fn._gpu_locked = f.fn._gpu_job = True
            n += 1
    return n


def _exit_with_parent() -> None:
    """Desktop window (Electron) sets IMAGEGEN_PARENT_PID: exit when that process is gone,
    so a crashed or force-closed window can't leave the backend holding the GPU."""
    pid = os.environ.get("IMAGEGEN_PARENT_PID", "")
    if not pid.isdigit() or sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Explicit types: with ctypes' default (signed 32-bit int) INFINITE = 0xFFFFFFFF overflows
    # and the watcher thread died silently.
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    k32.WaitForSingleObject.restype = wintypes.DWORD
    handle = k32.OpenProcess(0x00100000, False, int(pid))   # SYNCHRONIZE
    if not handle:
        print(f"[Desktop] Parent process {pid} already gone — exiting.", flush=True)
        os._exit(0)

    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)

    def _watch():
        try:
            k32.WaitForSingleObject(handle, 0xFFFFFFFF)      # blocks until the window exits
            # stdout is a pipe to the (now dead) window: printing raises, so it must not
            # be able to stop the termination below.
            print("[Desktop] Window closed — stopping the backend.", flush=True)
        except BaseException:
            pass
        finally:
            # Not os._exit(): that runs DLL unload, where ZLUDA can hang (measured — the
            # process stayed alive holding the GPU). TerminateProcess skips it.
            k32.TerminateProcess(k32.GetCurrentProcess(), 0)
    threading.Thread(target=_watch, name="parent-watch", daemon=True).start()
    print(f"[Desktop] Will stop when the desktop window (PID {pid}) closes.", flush=True)


def main():
    args = _ARGS or _parse_args()
    _exit_with_parent()
    app = build_app()
    app.queue(max_size=50)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
        show_api=False,
        quiet=True,
    )


if __name__ == "__main__":
    main()
