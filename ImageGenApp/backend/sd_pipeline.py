"""
backend/sd_pipeline.py
Stable Diffusion inference — supports SD 1.x, SD 2.x, SDXL.
Works via CUDA / ZLUDA (AMD ROCm emulation) / DirectML / CPU.
"""

from __future__ import annotations

import gc
import os
import re
import time
from pathlib import Path
from typing import Optional

from PIL import Image

# ── Windows: disable HF hub symlinks (requires Developer Mode otherwise) ───────
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    import huggingface_hub.file_download as _hf_dl
    # are_symlinks_supported() falsely returns True on Windows when test uses
    # a directory target but actual file symlinks require elevated privileges.
    _hf_dl.are_symlinks_supported = lambda *a, **kw: False
except Exception:
    pass

import config as _cfg
from backend.prompt_syntax import plain_prompt
from config import (
    CHECKPOINTS_DIR, LORAS_DIR, EMBEDDINGS_DIR,
    DEFAULT_STEPS, DEFAULT_CFG, DEFAULT_WIDTH, DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE, get_torch_dtype,
)

# Resolve device lazily so Settings-tab GPU changes take effect
def _device() -> str:
    return _cfg.DEVICE

# Constant adapter name — always reuse the same slot so PEFT never accumulates
# multiple default_0/default_1/… adapters with conflicting ranks.
_LORA_ADAPTER = "lora"

_TI_EXTS = {".pt", ".safetensors", ".bin"}

def _load_embeddings(pipe) -> None:
    """Auto-load textual-inversion embeddings from the correct arch subdir."""
    from config import EMBEDDINGS_SD15_DIR, EMBEDDINGS_SDXL_DIR

    is_sdxl = hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None
    arch_dir = EMBEDDINGS_SDXL_DIR if is_sdxl else EMBEDDINGS_SD15_DIR
    arch_label = "SDXL" if is_sdxl else "SD 1.5"

    emb_files = []
    # Primary: architecture-specific subdir
    if arch_dir.exists():
        emb_files += [f for f in arch_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in _TI_EXTS]
    # Fallback: flat embeddings/ dir (legacy) — only for SD 1.5 to avoid cross-arch loading
    if not is_sdxl and EMBEDDINGS_DIR.exists():
        emb_files += [f for f in EMBEDDINGS_DIR.iterdir()
                      if f.is_file() and f.suffix.lower() in _TI_EXTS]

    if not emb_files:
        return
    loaded = []
    for ef in emb_files:
        if not _is_ti_embedding(ef):
            continue
        if not _ti_matches_pipeline(ef, pipe):
            print(f"[Embedding] Skipped {ef.name}: dimension mismatch with {arch_label} model")
            continue
        try:
            token = ef.stem
            pipe.load_textual_inversion(str(ef), token=token)
            loaded.append(token)
        except Exception as e:
            print(f"[Embedding] Skipped {ef.name}: {e}")
    if loaded:
        print(f"[Embedding] Loaded {len(loaded)} {arch_label} embedding(s): {', '.join(loaded)}")


def _st_header(path: Path) -> dict:
    """A .safetensors file's tensor table (name → {dtype, shape, …}) without loading any weights.
    The companion-TI scan used to load every small file in the LoRA folders on each LoRA change."""
    import json, struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if not 2 <= n < 64 << 20:
            raise ValueError("not a safetensors header")
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def _torch_load_safe(path: Path):
    """torch.load without running pickled code (weights_only): embeddings are plain tensors/dicts;
    a .pt that needs more isn't loaded at all (weights_only=False executed any pickle)."""
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _is_ti_embedding(path: Path) -> bool:
    """Check if a .pt/.safetensors file is a textual-inversion embedding (not a LoRA)."""
    import torch
    ext = path.suffix.lower()
    try:
        if ext == ".pt":
            data = _torch_load_safe(path)
            if isinstance(data, dict):
                keys = set(data.keys())
                # TI embeddings have 'string_to_param' or a single key with a tensor
                if "string_to_param" in keys:
                    return True
                # If it has lora keys, it's a LoRA not a TI
                if any("lora" in k.lower() for k in keys):
                    return False
                # Single-key dict with a tensor value = TI embedding
                if len(keys) == 1:
                    val = next(iter(data.values()))
                    return isinstance(val, torch.Tensor) and val.dim() <= 2
            elif isinstance(data, torch.Tensor):
                return data.dim() <= 2
            return False
        elif ext == ".safetensors":
            keys = set(_st_header(path))
            if any("lora" in k.lower() for k in keys):
                return False
            # TI safetensors: typically 1 key "emb_params" or similar
            return len(keys) <= 3
        return False
    except Exception:
        return False


def _ti_matches_pipeline(path: Path, pipe) -> bool:
    """Check if a TI embedding's dimension matches the pipeline's text encoder."""
    import torch
    try:
        expected_dim = pipe.text_encoder.get_input_embeddings().weight.shape[1]
    except Exception:
        return True  # can't check, let load_textual_inversion handle it

    try:
        ext = path.suffix.lower()
        if ext == ".pt":
            data = _torch_load_safe(path)
            if isinstance(data, dict):
                if "string_to_param" in data:
                    tensor = next(iter(data["string_to_param"].values()))
                else:
                    tensor = next(iter(data.values()))
            else:
                tensor = data
            if isinstance(tensor, torch.Tensor):
                emb_dim = tensor.shape[-1]
                return emb_dim == expected_dim
        elif ext == ".safetensors":
            shapes = [v.get("shape") for v in _st_header(path).values() if isinstance(v, dict)]
            if shapes and shapes[0]:
                return shapes[0][-1] == expected_dim
    except Exception:
        pass
    return True  # if we can't check, let it try


def _load_companion_ti(pipe, lora_path: str) -> list[str]:
    """
    Auto-detect and load companion textual-inversion embeddings for a LoRA.
    Checks for .pt files sitting in the same directory as the LoRA file
    (users often drop companion TI embeddings next to their LoRAs).
    Also checks embeddings/ for any not yet loaded.
    Only loads files that are actual TI embeddings (not LoRAs) and match
    the current pipeline's embedding dimension (SD 1.5 vs SDXL).
    Returns list of loaded token names.
    """
    lora_p = Path(lora_path)
    loaded = []
    seen: set[str] = set()

    # Collect tokens already in the tokenizer to avoid double-loading
    try:
        existing_tokens = set(pipe.tokenizer.get_vocab().keys())
    except Exception:
        existing_tokens = set()

    def _try_load(ef: Path):
        if ef.name in seen or ef.stem in existing_tokens:
            return
        seen.add(ef.name)
        # Validate it's a real TI embedding, not a LoRA
        if not _is_ti_embedding(ef):
            return
        # Check dimension compatibility (SD 1.5 = 768, SDXL = 768/1280)
        if not _ti_matches_pipeline(ef, pipe):
            print(f"[Embedding] Skipped {ef.name}: dimension mismatch with current model")
            return
        try:
            token = ef.stem
            pipe.load_textual_inversion(str(ef), token=token)
            loaded.append(token)
            print(f"[Embedding] Auto-loaded companion TI: {ef.name} (token: {token})")
        except Exception as e:
            print(f"[Embedding] Skipped {ef.name}: {e}")

    # Check same directory as LoRA for .pt files (common pattern on Civitai)
    lora_dir = lora_p.parent
    if lora_dir.exists():
        for f in lora_dir.iterdir():
            if f.is_file() and f.suffix.lower() in _TI_EXTS and f != lora_p:
                # Skip large files (>50 MB) — those are LoRAs, not embeddings
                try:
                    if f.stat().st_size > 50 * 1024 * 1024:
                        continue
                except OSError:
                    continue
                _try_load(f)

    # Also pick up arch-specific embeddings not loaded at model-load time
    from config import EMBEDDINGS_SD15_DIR, EMBEDDINGS_SDXL_DIR
    is_sdxl = hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None
    arch_dir = EMBEDDINGS_SDXL_DIR if is_sdxl else EMBEDDINGS_SD15_DIR
    scan_dirs = [arch_dir] if is_sdxl else [arch_dir, EMBEDDINGS_DIR]
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for f in scan_dir.iterdir():
            if f.is_file() and f.suffix.lower() in _TI_EXTS:
                try:
                    if f.stat().st_size > 50 * 1024 * 1024:
                        continue
                except OSError:
                    continue
                _try_load(f)

    return loaded


def _strip_lora_layers(model) -> None:
    """
    Walk the module tree and replace every PEFT LoraLayer wrapper with its
    plain base nn.Linear / Conv2d.

    Why not use PEFT's delete_adapter() or unload_lora_weights()?
    diffusers uses inject_adapter_in_model() which patches individual layers
    rather than wrapping the whole model as a PeftModel.  The patched UNet
    therefore lacks a real .delete_adapter() method, so those calls silently
    fail.  Walking the tree and using setattr() bypasses all PEFT API and
    reliably removes every wrapper.

    After fuse_lora() the merged weights are already in get_base_layer().weight,
    so replacing LoraLayer → base_layer is lossless for fused adapters.
    For un-fused adapters (e.g. called via unload_loras) the base_layer holds
    the original pre-LoRA weights — which is exactly what we want.
    """
    for name, child in list(model.named_children()):
        if hasattr(child, "get_base_layer"):
            # PEFT tuner layer (LoraLinear, LoraConv2d, …) → swap in the base module
            setattr(model, name, child.get_base_layer())
        else:
            _strip_lora_layers(child)

# ── Scheduler map ──────────────────────────────────────────────────────────────
# (names → diffusers classes; the options per name live in backend/sampling.py)
from backend.sampling import SCHEDULERS as _SCHEDULERS
SCHEDULER_MAP: dict[str, str] = {name: cls for name, (cls, _) in _SCHEDULERS.items()}


def _load_scheduler(pipe, name: str):
    """Swap the scheduler on an existing pipeline (Karras / AYS / v-prediction handled in
    backend/sampling.make_scheduler)."""
    from backend.sampling import make_scheduler
    pipe.scheduler = make_scheduler(pipe, name)
    return pipe


def _read_sidecar_base(model_path: str) -> str | None:
    """Read baseModel from Civitai sidecar JSON next to a model file."""
    import json as _json
    from pathlib import Path as _P
    sidecar = _P(str(model_path) + ".civitai.json")
    if not sidecar.exists():
        return None
    try:
        return _json.loads(sidecar.read_text(encoding="utf-8")).get("baseModel", "").lower()
    except Exception:
        return None


# load_model(vae_path=KEEP_VAE): "whatever VAE is loaded" (Bridge, internal reloads). An explicit
# VAE (a path, or None / "none" for the checkpoint's own) that differs forces a reload — before, a
# VAE-only change returned "Already loaded" and the old VAE silently stayed (audit F-03).
KEEP_VAE = object()


def vae_key(v) -> str | None:
    return None if v is None or v is KEEP_VAE or str(v).strip().lower() in ("", "none") else str(v)


def _from_pretrained(cls, repo_id: str, **kwargs):
    """Load a HuggingFace diffusers repo, preferring its fp16 weights when we run in fp16
    (half the download: SD 1.5 ~2 GB instead of ~5 GB, SDXL ~7 GB instead of ~14 GB)."""
    import torch
    if kwargs.get("torch_dtype") == torch.float16:
        try:
            return cls.from_pretrained(repo_id, variant="fp16", **kwargs)
        except (OSError, ValueError) as e:   # repo has no fp16 variant files
            print(f"[SD] No fp16 variant for {repo_id} ({type(e).__name__}) — loading full weights")
    return cls.from_pretrained(repo_id, **kwargs)


def _is_sdxl(model_path_or_id: str) -> bool:
    """Detect SDXL-class model (SDXL/Pony/Illustrious) from sidecar or name."""
    import re
    # 1) Civitai sidecar — most reliable
    bm = _read_sidecar_base(model_path_or_id)
    if bm:
        return any(k in bm for k in ("sdxl", "xl", "pony", "illustrious", "noob"))
    # 2) Tensor names in the checkpoint header — catches SDXL files without "xl" in the name
    from backend.model_manager import checkpoint_arch
    arch = checkpoint_arch(model_path_or_id)
    if arch:
        return arch == "sdxl"
    # 3) Filename heuristics (HuggingFace IDs, .ckpt files) — use word-boundary for "xl" to avoid false positives
    #    (e.g. "example.safetensors" should NOT match)
    name = str(model_path_or_id).lower()
    if any(k in name for k in (
        "sdxl", "pony", "illustrious", "noobai",
        "juggernaut-xl", "playground-v2",
    )):
        return True
    stem = Path(name).stem
    return bool(re.search(r'(?:^|[-_ .])xl(?:[-_ .]|$)', stem))


def _model_family(model_path_or_id: str) -> str:
    """Classify model into family: 'pony', 'illustrious', 'sdxl', or 'sd15'."""
    import re
    # 1) Civitai sidecar
    bm = _read_sidecar_base(model_path_or_id)
    if bm:
        if "pony" in bm:
            return "pony"
        if "illustrious" in bm or "noob" in bm:
            return "illustrious"
        if "sdxl" in bm or "xl" in bm:
            return "sdxl"
        if "sd 1" in bm or "sd1" in bm:
            return "sd15"
    # 2) Checkpoint header: SD 1.x/2.x can't be Pony/Illustrious whatever the name says
    from backend.model_manager import checkpoint_arch
    arch = checkpoint_arch(model_path_or_id)
    if arch in ("sd1", "sd2"):
        return "sd15"
    # 3) Filename heuristics (tell SDXL sub-families apart)
    name = str(model_path_or_id).lower()
    if "pony" in name:
        return "pony"
    if "illustrious" in name or "noobai" in name:
        return "illustrious"
    if re.search(r'(?:^|[-_ ])il(?:l)?(?:[-_ .]|$)', Path(name).stem):
        return "illustrious"
    if arch == "sdxl" or _is_sdxl(name):
        return "sdxl"
    return "sd15"


def _cpu_vae_decode_sd(vae, latents) -> list:
    """Decode latents on CPU with multi-threading and adaptive tiling for SD 1.5."""
    import os
    import torch
    from PIL import Image as _Img
    import numpy as np

    try:
        max_threads = os.cpu_count() or 8
        if torch.get_num_threads() < max_threads:
            torch.set_num_threads(max_threads)
    except Exception:
        pass

    t0 = time.time()
    scale = getattr(vae.config, "scaling_factor", 0.18215)
    lat_cpu = latents.cpu().float()
    b, c, h, w = lat_cpu.shape

    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    vae.cpu().float()

    need_tiling = (h > 128 or w > 128 or b > 2)
    if need_tiling:
        vae.enable_tiling()
        print(f"[SD] Decoding latents on CPU (tiled, {w*8}×{h*8})...")
    else:
        vae.disable_tiling()
        print(f"[SD] Decoding latents on CPU (direct untiled {torch.get_num_threads()} threads, {w*8}×{h*8})...")

    try:
        with torch.no_grad():
            decoded = vae.decode(lat_cpu / scale, return_dict=False)[0]
    finally:
        vae.to(device=vae_device, dtype=vae_dtype)

    imgs = []
    arr = decoded.clamp(-1, 1).cpu().float().numpy()
    arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    for i in range(arr.shape[0]):
        imgs.append(_Img.fromarray(arr[i].transpose(1, 2, 0)))
    print(f"[SD] CPU VAE decode done ({time.time() - t0:.1f}s)")
    return imgs


def _gpu_vae_decode(vae, latents, tag: str) -> list | None:
    """
    Decode latents on the GPU the VAE already lives on (ZLUDA path, cuDNN off).
    Returns None on OOM or NaN so the caller can fall back to CPU decode.
    Measured on RX 6800M: SD 1.5 512×768 ~1s (vs ~10s CPU),
    SDXL 832×1216 tiled ~5s (vs ~21s CPU).
    """
    import torch
    t0 = time.time()
    try:
        with torch.no_grad():
            decoded = vae.decode(
                latents.to(device=vae.device, dtype=vae.dtype) / vae.config.scaling_factor,
                return_dict=False,
            )[0]
        if torch.isnan(decoded).any():
            print(f"[{tag}] GPU VAE decode produced NaN ({vae.dtype})")
            return None
    except torch.cuda.OutOfMemoryError:
        print(f"[{tag}] GPU VAE decode out of memory ({vae.dtype})")
        return None
    finally:
        torch.cuda.empty_cache()
    arr = ((decoded.float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
    arr = arr.permute(0, 2, 3, 1).cpu().numpy()
    print(f"[{tag}] GPU VAE decode done ({time.time() - t0:.1f}s)")
    return [Image.fromarray(a) for a in arr]


def _vram_reset(device: str) -> None:
    import torch
    if "cuda" in device:
        torch.cuda.reset_peak_memory_stats()


def _vram_spill_note(device: str) -> str:
    """Windows (WDDM) doesn't OOM when VRAM is over-committed: it pages to shared
    system memory and every step gets ~20x slower. Flag it in the info line."""
    import torch
    if "cuda" not in device:
        return ""
    total = torch.cuda.get_device_properties(torch.device(device)).total_memory
    peak = torch.cuda.max_memory_reserved()
    if peak <= total * 0.97:
        return ""
    msg = (f" | ⚠ VRAM over-committed ({peak / 2**30:.1f}/{total / 2**30:.1f} GiB) — "
           "spilled into system RAM, which is very slow; lower resolution or batch size")
    print(f"[VRAM]{msg}")
    return msg


def _build_embeds(pipe, prompt: str, neg_prompt: str, device: str, clip_skip: int = 1) -> dict:
    """
    Encode prompts with Compel (chunk-and-concatenate for >77 tokens).
    Returns a dict of kwargs to splat into pipe() — either prompt_embeds or
    raw strings when Compel is unavailable or the text encoder is on DML.
    SD 1.5 only — SDXL uses _build_sdxl_embeds() in sdxl_pipeline.py.
    """
    # CLIP skip 2 (A1111 "Clip skip: 2") = the text encoder's penultimate layer — what most
    # SD 1.5 anime checkpoints were trained with. diffusers counts it as clip_skip=1.
    skip_kw = {"clip_skip": clip_skip - 1} if clip_skip and clip_skip > 1 else {}
    try:
        from compel import Compel, ReturnedEmbeddingsType
    except ImportError:
        return {"prompt": plain_prompt(prompt), "negative_prompt": plain_prompt(neg_prompt), **skip_kw}

    # Clear this frame's references on the way out: something in the Compel call
    # path captures the call stack, which kept this frame — and through `pipe` the
    # whole pipeline — alive after the model was unloaded (the Bridge leaked 2 GB
    # of VRAM per run and pushed a 12 GB card into shared memory).
    c = None
    try:
        te_device = str(next(pipe.text_encoder.parameters()).device)
        if "privateuseone" in te_device:
            return {"prompt": plain_prompt(prompt), "negative_prompt": plain_prompt(neg_prompt), **skip_kw}
        # fp32 for the encode (see sdxl_pipeline._build_sdxl_embeds): fp16 encoding on ZLUDA
        # isn't deterministic run to run; fp16→fp32→fp16 is lossless
        import torch
        te_half = pipe.text_encoder.dtype == torch.float16
        if te_half:
            pipe.text_encoder.float()
        c = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            truncate_long_prompts=False,
            returned_embeddings_type=(ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
                                      if clip_skip and clip_skip > 1 else
                                      ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED),
        )
        # 75-token chunks cut at tag boundaries (BREAK = new chunk), each encoded with
        # Compel and concatenated — a tag is never split across two chunks.
        from backend.prompt_tools import encode_chunked, pad_to_same_chunks
        pos = encode_chunked(c, prompt, pipe.tokenizer)
        neg = encode_chunked(c, neg_prompt, pipe.tokenizer)
        pos, neg = pad_to_same_chunks(c, pos, neg)
        if te_half:
            pos, neg = pos.half(), neg.half()
        seq_len = pos.shape[1]
        if seq_len > 77:
            print(f"[Compel] Long prompt encoded in {seq_len // 77} chunks of 77 tokens ✓")
        return dict(prompt_embeds=pos, negative_prompt_embeds=neg)
    except Exception as e:
        print(f"[Compel] Encoding failed ({e}), falling back to raw 77-token strings")
        return {"prompt": plain_prompt(prompt), "negative_prompt": plain_prompt(neg_prompt), **skip_kw}
    finally:
        try:
            import torch
            if pipe is not None and pipe.text_encoder.dtype == torch.float32 and pipe.unet.dtype == torch.float16:
                pipe.text_encoder.half()          # back to the pipeline's dtype, also after an error
        except Exception:
            pass
        pipe = c = None




class SDPipeline:
    """
    Wraps diffusers SD 1.x / 2.x pipelines with:
    - LoRA loading / unloading
    - Scheduler hot-swap
    - ZLUDA / CUDA / DirectML / CPU device routing
    SDXL/Pony/Illustrious use SDXLPipeline in sdxl_pipeline.py instead.
    """

    def __init__(self):
        self.device      = _device()
        self.dtype       = get_torch_dtype(_device())
        self.pipe        = None          # txt2img pipeline
        self.img2img_pipe = None         # img2img pipeline (loaded on demand)
        self.current_model: str | None  = None
        self.loaded_loras: list[str]    = []
        self._lora_adapters: dict[int, tuple[str, str, float]] = {}  # slot → (name, path, weight)
        self.is_sdxl     = False
        self.model_family = "sd15"
        self.last_seeds: list[int] = []
        self.last_var_seeds: list[int] = []
        # Tiled decode needs ~1 GB less VRAM; the Bridge turns this on while an SDXL
        # model shares the card (colour shift is irrelevant there — SDXL redraws it)
        self.force_tiled_decode = False
        # SmartSplit DirectML flag: when True, only UNet is placed on DirectML at load time.
        # text_encoder + VAE remain on CPU so DirectML never allocates VRAM for them.
        self.unet_only_dml: bool = False
        self._last_vae_path: str | None = None
        # CPU-pinned snapshots of base model weights taken right after load_model(),
        # before any LoRA is ever fused.  Used by _restore_clean_state() to guarantee
        # _reload_all_loras() always starts from the true pre-LoRA checkpoint.
        self._clean_unet_state: dict | None = None
        self._clean_te_state:   dict | None = None

    # ── Model loading ──────────────────────────────────────────────────────────
    def load_model(self, model_path_or_id: str, vae_path=KEEP_VAE) -> str:
        """
        Load an SD 1.x/2.x model from a local .safetensors path or HuggingFace ID.
        SDXL models are handled by SDXLPipeline — this class should never receive them.
        """
        import torch
        from diffusers import StableDiffusionPipeline, AutoencoderKL

        if model_path_or_id == self.current_model and self.pipe is not None and (
                vae_path is KEEP_VAE or vae_key(vae_path) == vae_key(self._last_vae_path)):
            return f"✅ Already loaded: {Path(model_path_or_id).name}"
        vae_path = vae_key(vae_path)

        self.device = _device()
        self.dtype  = get_torch_dtype(self.device)
        self._unload()
        from backend.model_manager import commit_problem
        low = commit_problem(model_path_or_id)          # a refusal instead of a crash inside safetensors when commit runs out
        if low:
            return f"❌ {low}"

        path = Path(model_path_or_id)
        is_local = path.exists()
        self.is_sdxl = False
        self.model_family = _model_family(model_path_or_id)

        common_kwargs = dict(
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        try:
            if is_local and path.suffix in (".safetensors", ".ckpt"):
                self.pipe = StableDiffusionPipeline.from_single_file(
                    str(path), **common_kwargs
                )
            else:
                self.pipe = _from_pretrained(
                    StableDiffusionPipeline, model_path_or_id, **common_kwargs
                )

            # Optionally swap VAE
            if vae_path and Path(vae_path).exists():
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=self.dtype)
                self.pipe.vae = vae
            self._last_vae_path = vae_path

            # Device placement
            if "privateuseone" in self.device:
                import torch_directml
                dml_dev = torch_directml.device()
                if self.unet_only_dml:
                    # SmartSplit DML mode: only UNet+VAE go to DirectML.
                    self.pipe.unet = self.pipe.unet.to(dml_dev)
                    self.pipe.vae  = self.pipe.vae.to(dml_dev)
                    print(f"[Load] UNet+VAE → DirectML  |  Text → CPU (SmartSplit mode)")
                else:
                    self.pipe = self.pipe.to(dml_dev)
            else:
                self.pipe = self.pipe.to(self.device)

            # Memory efficiency helpers
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing(1)
            if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()
            if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
                self.pipe.vae.enable_slicing()
            if "cuda" in self.device and hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass

            self.current_model = model_path_or_id
            self.loaded_loras  = []
            self._lora_adapters = {}
            self.img2img_pipe  = None
            from backend.sampling import detect_prediction, configure_prediction
            self.prediction = detect_prediction(model_path_or_id) if is_local else {}
            configure_prediction(self.pipe, self.prediction)
            _load_embeddings(self.pipe)
            # LoRA restore snapshot is taken lazily on first load_lora():
            # saves a full CPU copy of the weights (~6.6 GB for SDXL) when no LoRA is used.
            model_name = path.stem if is_local else model_path_or_id
            return f"✅ Loaded: {model_name} (SD, {self.device})"

        except Exception as e:
            self._unload()
            from backend.model_manager import safetensors_problem
            bad = safetensors_problem(model_path_or_id)   # say *why*, not just "unable to load"
            return f"❌ {bad}" if bad else f"❌ Failed to load model: {e}"

    def _unload(self):
        """Release GPU / CPU memory held by the pipeline."""
        import torch
        self.pipe         = None
        self.img2img_pipe = None
        # the cached inpaint pipe (face detail / inpaint) holds the old UNet, TEs and VAE:
        # left in place it kept ~2.5 GB of VRAM after switching SD 1.5 → SDXL
        self._inpaint_pipe = None
        self._pag_pipes = None
        self.prediction = {}
        if getattr(self, "_snap", None) is not None:
            self._snap.close()
        self._snap = None
        self.current_model = None
        self.loaded_loras  = []
        self._lora_adapters = {}
        self._clean_unet_state = None
        self._clean_te_state   = None
        from backend.prompt_tools import release_parser_cache
        release_parser_cache()        # it can hold Compel → the text encoders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _snapshot_clean_state(self):
        """Start LoRA-restore snapshots for a clean model. They fill lazily: just before a LoRA
        is fused, the weights of the layers it changes are copied to the CPU (backend/lora_snapshot)
        — a full copy of every weight made the SDXL pipeline hold 6.8 GB of RAM."""
        if self.pipe is None:
            return
        from backend.lora_snapshot import Snapshot
        if getattr(self, "_snap", None) is not None:
            self._snap.close()
        self._snap = Snapshot()
        self._clean_unet_state, self._clean_te_state = {}, None   # marker only: "snapshots started"
        # the text encoder in full (246 MB, to disk): load_lora_weights can rewrite layers of it
        # before anything is fused (seen on SDXL's CLIP-L) — only a copy taken now restores it exactly
        if getattr(self.pipe, "text_encoder", None) is not None:
            for n, p in self.pipe.text_encoder.named_parameters():
                self._snap.add("text_encoder", n, p)
            self._snap.spill()

    def _restore_clean_state(self):
        """Put every weight a LoRA changed back from the snapshot (RAM part + temp files)."""
        if self.pipe is None or getattr(self, "_snap", None) is None:
            return
        self._snap.restore(self.pipe)

    # ── LoRA management ────────────────────────────────────────────────────────
    def load_lora(self, lora_path: str, weight: float = 0.8, slot: int = 0) -> str:
        if self.pipe is None:
            return "❌ Load a base model first."
        if self._clean_unet_state is None:
            self._snapshot_clean_state()  # model is still LoRA-free here
        prev = self._lora_adapters.get(slot)
        try:
            self._lora_adapters[slot] = (Path(lora_path).name, lora_path, weight)
            self._reload_all_loras()
            names = [v[1][0] for v in sorted(self._lora_adapters.items())]
            return (f"✅ Slot {slot+1}: {Path(lora_path).name} (×{weight:.2f})"
                    + (f" | Active: {', '.join(names)}" if len(names) > 1 else ""))
        except Exception as e:
            # Don't leave the failed LoRA registered (every later reload would retry it)
            # or the model half-reloaded: go back to the previous slot contents.
            if prev is None:
                self._lora_adapters.pop(slot, None)
            else:
                self._lora_adapters[slot] = prev
            try:
                self._reload_all_loras()
            except Exception as e2:
                print(f"[LoRA] Restore after failed load also failed: {e2}")
            from backend.model_manager import safetensors_problem
            bad = safetensors_problem(lora_path)
            return f"❌ {bad}" if bad else f"❌ LoRA error: {e}"

    def remove_lora(self, slot: int) -> str:
        if slot not in self._lora_adapters:
            return f"Slot {slot+1} is empty."
        name = self._lora_adapters.pop(slot)[0]
        self._reload_all_loras()
        remaining = [v[1][0] for v in sorted(self._lora_adapters.items())]
        return (f"✅ Removed: {name}"
                + (f" | Active: {', '.join(remaining)}" if remaining else " | No LoRAs active"))

    def _purge_peft(self):
        """
        Completely remove all PEFT adapter layers from UNet and text encoder.

        Strategy (belt-and-suspenders):
        1. Call diffusers' unload_lora_weights() for high-level bookkeeping.
        2. Call _strip_lora_layers() on each component — this directly replaces
           every LoraLayer wrapper with its base nn.Linear/Conv2d via setattr(),
           bypassing PEFT's API entirely.  Reliable even when delete_adapter()
           is not available (inject_adapter_in_model path, not PeftModel path).
        3. Clear the peft_config / _hf_peft_config_loaded attributes so the
           next load_lora_weights() creates a truly fresh adapter.

        Critical note: this must be called BEFORE _restore_clean_state().
        With PEFT wrappers present, named_parameters() returns paths like
        "proj_in.base_layer.weight" but the snapshot was saved as "proj_in.weight",
        so the copy_() loop finds no matches and silently restores nothing.
        Stripping the wrappers first lets the paths realign.
        """
        if self.pipe is None:
            return
        # Step 1: diffusers high-level cleanup (handles internal bookkeeping flags)
        try:
            self.pipe.unload_lora_weights()
        except Exception:
            pass
        # Step 2 + 3: strip any lingering wrappers and clear PEFT attributes
        for attr in ("unet", "text_encoder", "text_encoder_2"):
            model = getattr(self.pipe, attr, None)
            if model is None:
                continue
            _strip_lora_layers(model)
            for peft_attr in ("peft_config", "_hf_peft_config_loaded"):
                try:
                    delattr(model, peft_attr)
                except AttributeError:
                    pass

    def _reload_all_loras(self):
        """
        Restore model to its pre-LoRA state (via CPU snapshot), then fuse each
        active LoRA at its slot weight.  Using fuse+unload (not set_adapters)
        because PEFT inference hooks don't work reliably on DirectML.

        LoRA fusion is always done on CPU to avoid device mismatches — diffusers'
        load_lora_weights() places LoRA tensors on CPU by default, and fuse_lora()
        needs all tensors on the same device.

        Order matters:
          _purge_peft() FIRST  — strips LoraLayer wrappers so named_parameters()
                                  returns plain "proj_in.weight" paths again.
          _restore_clean_state() SECOND — snapshot keys now match, weights restored.
        Reversing the order silently breaks restore (param-path mismatch).
        """
        # 1. Strip PEFT wrappers so named_parameters() paths match the snapshot
        self._purge_peft()

        # DML: move to CPU for fusion. CUDA/ZLUDA fuses in place — Module.cpu()
        # on large models crashes inside ZLUDA (access violation).
        orig_device = self.device
        needs_move = "privateuseone" in orig_device and self.pipe is not None
        if needs_move:
            self.pipe.unet.cpu()
            if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
                self.pipe.text_encoder.cpu()

        try:
            # 2. Restore base weights from CPU snapshot
            self._restore_clean_state()

            self.loaded_loras = []

            if not self._lora_adapters:
                return

            # 3. Fuse each LoRA in slot order onto the restored base weights.
            #    Always use the same adapter name (_LORA_ADAPTER) so PEFT never
            #    creates parallel default_0/default_1/… slots with conflicting ranks.
            for slot in sorted(self._lora_adapters.keys()):
                name_str, path, weight = self._lora_adapters[slot]
                from backend.model_manager import lycoris_kind
                if lycoris_kind(path):   # LoHa / LoKr: diffusers can't load these, fuse ourselves
                    from backend.lycoris import apply_lycoris
                    from backend.lora_snapshot import param_snapshotter
                    n, skipped = apply_lycoris(self.pipe, path, weight,
                                               before_change=param_snapshotter(self.pipe, self._snap))
                    self.loaded_loras.append(name_str)
                    print(f"[LoRA] Slot {slot+1}: {name_str} (LyCORIS, {n} layers"
                          f"{f', {skipped} unmatched' if skipped else ''}) fused at ×{weight:.2f}")
                    continue
                try:
                    self.pipe.load_lora_weights(path, adapter_name=_LORA_ADAPTER)
                except Exception as e:
                    err = str(e)
                    if "Target modules" in err and "not found in the base model" in err:
                        raise ValueError(
                            f"LoRA '{Path(path).name}' format is incompatible with this model "
                            "(layer names don't match). It may have been trained on a different "
                            "architecture or tool. Try a different LoRA version."
                        ) from None
                    raise
                from backend.lora_snapshot import snapshot_wrapped
                snapshot_wrapped(self.pipe, self._snap)   # clean copies of what fuse changes
                self.pipe.fuse_lora(lora_scale=weight)
                self._purge_peft()  # strip wrappers; fused weights stay in base layers
                self.loaded_loras.append(name_str)
                print(f"[LoRA] Slot {slot+1}: {name_str} fused at ×{weight:.2f}")
            if getattr(self, "_snap", None) is not None:
                self._snap.spill()          # restore copies → temp file, off the RAM
        finally:
            # Always move back to original device
            if needs_move:
                self.pipe.unet.to(orig_device)
                if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
                    self.pipe.text_encoder.to(orig_device)

        # Load companion TI embeddings after fusion + device restore
        for slot in sorted(self._lora_adapters.keys()):
            _, path, _ = self._lora_adapters[slot]
            _load_companion_ti(self.pipe, path)

    def unload_loras(self) -> str:
        if self.pipe is None:
            return "No model loaded."
        # Purge PEFT first so _restore_clean_state() finds correct param paths
        self._purge_peft()
        self._restore_clean_state()
        self._lora_adapters = {}
        self.loaded_loras = []
        return "✅ LoRAs unloaded — model restored to base weights."

    # ── Text-to-image ──────────────────────────────────────────────────────────
    def txt2img(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG,
        seed: int = -1,
        scheduler: str = "DPM++ 2M Karras",
        batch_size: int = 1,
        step_callback=None,
        clip_skip: int = 1,
        var_seed: int = -1,
        var_strength: float = 0.0,
    ) -> tuple[list[Image.Image], str]:
        """
        Run text-to-image generation. Returns (images, info_string).
        step_callback(step, total) — optional progress callback called each step.
        """
        if self.pipe is None:
            return [], "❌ No model loaded. Pick a checkpoint and click Load Model (Generate tab)."

        import torch

        _load_scheduler(self.pipe, scheduler)
        generator, seeds = _make_generators(seed, self.device, max(1, int(batch_size)))
        self.last_seeds = seeds
        used_seed = seeds[0]
        latents, self.last_var_seeds = variation_latents(
            self.pipe, generator, var_seed, var_strength, max(1, int(batch_size)), width, height, self.device)
        lat_kw = {"latents": latents} if latents is not None else {}

        cb_kwargs = {}
        if step_callback is not None:
            def _pipe_callback(pipe, i, t, callback_kwargs):
                step_callback(i + 1, steps)
                return callback_kwargs
            cb_kwargs = {"callback_on_step_end": _pipe_callback}

        _vram_reset(self.device)
        t0 = time.time()
        embeds = _build_embeds(self.pipe, prompt, negative_prompt, self.device, clip_skip)

        # ZLUDA: decode latents ourselves (GPU first, CPU fallback) — see _decode_latents
        use_cpu_vae = "cuda" in self.device
        if use_cpu_vae:
            embeds["output_type"] = "latent"

        try:
            from backend.sampling import run_pipe
            result = run_pipe(
                self, self.pipe, "txt2img",
                **embeds,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                num_images_per_prompt=batch_size,
                **lat_kw,
                **cb_kwargs,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                oom_msg = (
                    f"❌ GPU Out of Memory (OOM)! "
                    f"Resolution {width}×{height} with batch={batch_size} exceeded VRAM. "
                    f"Try reducing resolution (e.g. 512×512 or 512×768) or reducing batch size to 1."
                )
                print(f"[SD] {oom_msg}")
                return [], oom_msg
            raise

        if use_cpu_vae:
            images = self._decode_latents(self.pipe.vae, result.images)
        else:
            images = result.images
        elapsed = time.time() - t0
        info = (
            f"{_seed_label(seeds)} | Steps: {steps} | CFG: {cfg_scale} | "
            f"Scheduler: {scheduler} | Size: {width}×{height} | "
            f"Time: {elapsed:.1f}s | Device: {self.device}"
        )
        info += _vram_spill_note(self.device)
        meta = dict(
            prompt=prompt, negative_prompt=negative_prompt,
            steps=steps, cfg_scale=cfg_scale, seed=used_seed,
            scheduler=scheduler, width=width, height=height,
            model=Path(self.current_model).stem if self.current_model else "",
            loras=", ".join(self.loaded_loras) if self.loaded_loras else "",
        )
        return images, info

    def _decode_latents(self, vae, latents) -> list[Image.Image]:
        """GPU decode (untiled up to 1024px — tiling visibly shifts SD 1.5 colours), CPU fallback."""
        _, _, h, w = latents.shape
        if h <= 128 and w <= 128 and not self.force_tiled_decode:
            vae.disable_tiling()
        else:
            vae.enable_tiling()
        images = _gpu_vae_decode(vae, latents, "SD")
        return images if images is not None else _cpu_vae_decode_sd(vae, latents)

    # ── Image-to-image ─────────────────────────────────────────────────────────
    def img2img(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE,
        strength: float = 0.75,
        steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG,
        seed: int = -1,
        scheduler: str = "DPM++ 2M Karras",
        step_callback=None,
        clip_skip: int = 1,
    ) -> tuple[list[Image.Image], str]:
        """
        Run image-to-image generation. Returns (images, info_string).
        step_callback(step, total) — optional progress callback called each step.
        """
        if self.pipe is None:
            return [], "❌ No model loaded."

        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        # Build (or reuse) the img2img pipeline from the same weights
        if self.img2img_pipe is None:
            # from_pipe defaults to torch_dtype=float32 and casts the *shared* UNet/TEs,
            # silently doubling VRAM (SDXL spills to shared memory) for txt2img too.
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(self.pipe, torch_dtype=self.dtype)
            if "privateuseone" in self.device:
                import torch_directml
                if self.unet_only_dml:
                    # SmartSplit: components shared with self.pipe, already on
                    # correct devices. Do NOT call .to(dml_dev).
                    pass
                else:
                    self.img2img_pipe = self.img2img_pipe.to(torch_directml.device())
            else:
                self.img2img_pipe = self.img2img_pipe.to(self.device)
            self.img2img_pipe.enable_attention_slicing(1)
            self.img2img_pipe.vae.enable_tiling()
            self.img2img_pipe.vae.enable_slicing()

        _load_scheduler(self.img2img_pipe, scheduler)
        generator, used_seed = _make_generator(seed, self.device)
        self.last_seeds = [used_seed]
        # Round UP to next multiple of 8 (avoids silently clipping content)
        init_image = init_image.convert("RGB").resize(
            ((init_image.width  + 7) // 8 * 8,
             (init_image.height + 7) // 8 * 8),
            Image.LANCZOS,
        )

        cb_kwargs = {}
        if step_callback is not None:
            def _pipe_callback(pipe, i, t, callback_kwargs):
                step_callback(i + 1, steps)
                return callback_kwargs
            cb_kwargs = {"callback_on_step_end": _pipe_callback}

        _vram_reset(self.device)
        t0 = time.time()
        embeds = _build_embeds(self.img2img_pipe, prompt, negative_prompt, self.device, clip_skip)

        # ZLUDA: decode latents ourselves (GPU first, CPU fallback) — see _decode_latents
        use_cpu_vae = "cuda" in self.device
        if use_cpu_vae:
            embeds["output_type"] = "latent"

        from backend.sampling import run_pipe
        result = run_pipe(
            self, self.img2img_pipe, "img2img",
            **embeds,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            **cb_kwargs,
        )

        if use_cpu_vae:
            images = self._decode_latents(self.img2img_pipe.vae, result.images)
        else:
            images = result.images
        elapsed = time.time() - t0

        info = (
            f"img2img | Strength: {strength} | Seed: {used_seed} | "
            f"Steps: {steps} | CFG: {cfg_scale} | Time: {elapsed:.1f}s"
        )
        info += _vram_spill_note(self.device)
        meta = dict(
            prompt=prompt, negative_prompt=negative_prompt,
            steps=steps, cfg_scale=cfg_scale, seed=used_seed,
            scheduler=scheduler, model=Path(self.current_model).stem if self.current_model else "",
            loras=", ".join(self.loaded_loras) if self.loaded_loras else "",
            img2img_strength=strength,
        )
        return images, info


# ── Helpers ────────────────────────────────────────────────────────────────────
def _make_generators(seed: int, device: str, n: int) -> tuple:
    """One generator per image, seeded seed, seed+1, … so every image in a batch
    can be reproduced on its own (batch 1 with that seed). n=1 matches _make_generator."""
    gen, first = _make_generator(seed, device)
    gens = [gen] + [torch_generator(device).manual_seed((first + i) % 2**32) for i in range(1, n)]
    return gens, [(first + i) % 2**32 for i in range(n)]


def _slerp(t: float, a, b):
    """Spherical interpolation between two noise tensors (keeps the noise's statistics —
    a plain mix would shrink its variance and wash the image out)."""
    import torch
    af, bf = a.flatten(1).float(), b.flatten(1).float()
    dot = (af / af.norm(dim=1, keepdim=True) * bf / bf.norm(dim=1, keepdim=True)).sum(1).clamp(-1, 1)
    omega = torch.acos(dot)[:, None]
    so = torch.sin(omega)
    if float(so.abs().min()) < 1e-6:
        out = (1 - t) * af + t * bf
    else:
        out = torch.sin((1 - t) * omega) / so * af + torch.sin(t * omega) / so * bf
    return out.reshape(a.shape).to(a.dtype)


def variation_latents(pipe, generators, var_seed, var_strength: float, batch: int,
                      width: int, height: int, device: str):
    """Initial latents for "variations": the image's own noise (same generator the pipeline
    would use, so strength 0 is identical) blended toward a second seed's noise.
    Returns (latents, variation seeds) — or (None, []) when variations are off."""
    if not var_strength or var_strength <= 0:
        return None, []
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    if var_seed is None or int(var_seed) < 0:
        var_seed = int(torch.randint(0, 2**32, (1,)).item())
    var_seeds = [(int(var_seed) + i) % 2**32 for i in range(batch)]
    gens = generators if isinstance(generators, list) else [generators]
    dtype = pipe.unet.dtype
    ch = pipe.unet.config.in_channels
    shape = (1, ch, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
    dev = torch.device(device if "privateuseone" not in device else "cpu")
    lat = []
    for i in range(batch):
        base = randn_tensor(shape, generator=gens[i], device=dev, dtype=dtype)
        other = randn_tensor(shape, generator=torch_generator(device).manual_seed(var_seeds[i]), device=dev, dtype=dtype)
        # the blend runs on the CPU: acos/sin on these tiny tensors would make ZLUDA compile
        # new GPU kernels (minutes the first time) for no speed gain
        mixed = _slerp(min(1.0, float(var_strength)), base.float().cpu(), other.float().cpu())
        lat.append(mixed.to(dev, dtype))
    return torch.cat(lat), var_seeds


def torch_generator(device: str):
    import torch
    dev = "cpu" if ("privateuseone" in device or device == "cpu") else device.split(":")[0]
    return torch.Generator(device=dev)


def _seed_label(seeds: list[int]) -> str:
    return f"Seed: {seeds[0]}" if len(seeds) == 1 else f"Seeds: {seeds[0]}…{seeds[-1]}"


def _make_generator(seed: int, device: str) -> tuple:
    """Return (generator, actual_seed). If seed == -1, generates a random seed."""
    import torch
    if seed is None or seed == -1:
        seed = int(torch.randint(0, 2**32, (1,)).item())
    # DirectML and some ZLUDA setups don't support CUDA generators — use CPU
    dev = "cpu" if ("privateuseone" in device or device == "cpu") else device.split(":")[0]
    return torch.Generator(device=dev).manual_seed(seed), seed


def _save_images(images: list[Image.Image], meta: dict | None = None):
    from config import OUTPUTS_DIR
    from PIL.PngImagePlugin import PngInfo
    ts = int(time.time())
    for i, img in enumerate(images):
        pnginfo = PngInfo()
        if meta:
            # A1111-compatible "parameters" chunk so SD viewers can parse it
            params_text = meta.get("prompt", "")
            neg = meta.get("negative_prompt", "")
            if neg:
                params_text += f"\nNegative prompt: {neg}"
            detail_parts = []
            for k in ("steps", "cfg_scale", "seed", "scheduler", "width", "height"):
                if k in meta:
                    label = {"cfg_scale": "CFG scale", "scheduler": "Sampler"}.get(k, k.capitalize())
                    detail_parts.append(f"{label}: {meta[k]}")
            if "model" in meta:
                detail_parts.append(f"Model: {meta['model']}")
            if "loras" in meta and meta["loras"]:
                detail_parts.append(f"LoRAs: {meta['loras']}")
            if detail_parts:
                params_text += "\n" + ", ".join(detail_parts)
            pnginfo.add_text("parameters", params_text)
        img.save(OUTPUTS_DIR / f"{ts}_{i}.png", pnginfo=pnginfo)
