"""
backend/sdxl_pipeline.py
SDXL / Pony / Illustrious inference — separated from SD 1.5 to avoid cross-contamination.
Handles: model loading, device placement (ZLUDA: everything on the dGPU;
DML: UNet on GPU, text+VAE on CPU), LoRA fuse/unfuse, Compel long-prompt encoding,
iGPU text encoding, manual VAE decode (tiled GPU on ZLUDA with CPU fallback).
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import config as _cfg
from backend.prompt_syntax import plain_prompt
from config import (
    CHECKPOINTS_DIR, LORAS_DIR,
    DEFAULT_STEPS, DEFAULT_CFG, DEFAULT_WIDTH, DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE, get_torch_dtype,
)
from backend.sd_pipeline import (
    _strip_lora_layers, _load_scheduler, _make_generator, _make_generators, _seed_label, _gpu_vae_decode, _vram_reset, _vram_spill_note,
    _is_sdxl, _model_family, _load_embeddings, _load_companion_ti, SCHEDULER_MAP,
)

_LORA_ADAPTER = "sdxl_lora"


def _device() -> str:
    return _cfg.DEVICE


def _find_igpu_device() -> "str | None":
    """
    Return a DirectML device string for the iGPU (if one exists besides the dGPU).
    Uses hardware_detector to find an iGPU, then maps to a torch_directml device index.
    Returns None if no usable iGPU is found.
    """
    try:
        import torch_directml
        from backend.hardware_detector import detect_hardware
        hw = detect_hardware()
        # Find iGPU that's NOT the currently active dGPU
        igpu = next((g for g in hw.gpus if g.is_igpu and g.backend == "directml"), None)
        if igpu is None:
            return None
        # torch_directml enumerates in the same order as WMIC/registry
        idx = igpu.index
        if idx < torch_directml.device_count():
            return str(torch_directml.device(idx))
    except Exception:
        pass
    return None


# Cache iGPU device detection (expensive — calls WMI)
_igpu_dev: str | None = ...  # sentinel: not yet probed


def _get_igpu_device() -> str | None:
    global _igpu_dev
    if _igpu_dev is ...:
        _igpu_dev = _find_igpu_device()
        if _igpu_dev:
            print(f"[SDXL] iGPU detected for text encoding: {_igpu_dev}")
        else:
            print("[SDXL] No iGPU found — text encoding will use CPU (float32)")
    return _igpu_dev


_SDXL_EMBED_CACHE: dict[tuple[str, str, str], dict] = {}


def _clear_embed_cache():
    global _SDXL_EMBED_CACHE
    _SDXL_EMBED_CACHE.clear()


def _build_sdxl_embeds(pipe, prompt: str, neg_prompt: str) -> dict:
    """
    Encode prompts via Compel for SDXL dual text encoders with LRU caching.
    """
    try:
        from compel import Compel, ReturnedEmbeddingsType
    except ImportError:
        print("[SDXL] Compel not installed — prompts will be truncated to 77 tokens")
        return {"prompt": plain_prompt(prompt), "negative_prompt": plain_prompt(neg_prompt)}

    # Check embedding cache to avoid recomputing identical prompts (saves 7s on CPU)
    model_key = str(getattr(pipe, "_model_name_or_path", ""))
    cache_key = (prompt.strip(), neg_prompt.strip(), model_key)
    if cache_key in _SDXL_EMBED_CACHE:
        cached = _SDXL_EMBED_CACHE[cache_key]
        print(f"[SDXL Compel] Embedding cache hit! Reusing precomputed prompt embeddings (0.0s) ✓")
        return {k: v.clone() for k, v in cached.items()}

    c = None
    # Clear this frame's references on the way out: something in the Compel call
    # path captures the call stack, which kept this frame — and through `pipe` the
    # whole pipeline — alive after the model was unloaded (the Bridge leaked 2 GB
    # of VRAM per run and pushed a 12 GB card into shared memory).
    try:
        import torch
        te_device = str(next(pipe.text_encoder.parameters()).device)
        if "privateuseone" in te_device:
            print("[SDXL] Text encoder is on DML — cannot use Compel, truncating to 77 tokens")
            return {"prompt": plain_prompt(prompt), "negative_prompt": plain_prompt(neg_prompt)}

        # ── On the CPU, cast TEs to float32 (fp16 matmuls are slow there) ──
        # On the GPU fp16 gives identical embeddings (cosine 1.00000) and skips a
        # 2.8 GB fp32 copy of the encoders plus ~3 s of first-run kernel compilation.
        on_cpu = te_device == "cpu"
        te1_was_half = on_cpu and (pipe.text_encoder.dtype == torch.float16)
        te2_was_half = on_cpu and (pipe.text_encoder_2.dtype == torch.float16)
        if te1_was_half:
            pipe.text_encoder.float()
        if te2_was_half:
            pipe.text_encoder_2.float()

        c = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        # 75-token chunks cut at tag boundaries (BREAK = new chunk); pooled embedding
        # from the first chunk, as A1111 does
        from backend.prompt_tools import encode_chunked, pad_to_same_chunks
        pos, pos_pool = encode_chunked(c, prompt, pipe.tokenizer, sdxl=True)
        neg, neg_pool = encode_chunked(c, neg_prompt, pipe.tokenizer, sdxl=True)
        pos, neg = pad_to_same_chunks(c, pos, neg, sdxl=True)

        # ── Restore TEs to float16 BEFORE returning ──
        if te1_was_half:
            pipe.text_encoder.half()
        if te2_was_half:
            pipe.text_encoder_2.half()

        # Cast embeddings to float16 to match pipeline expectations
        pos = pos.half()
        neg = neg.half()
        pos_pool = pos_pool.half()
        neg_pool = neg_pool.half()

        seq_len = pos.shape[1]
        if seq_len > 77:
            print(f"[SDXL Compel] Long prompt: {seq_len} tokens across {seq_len // 77} chunks ✓")

        res = dict(
            prompt_embeds=pos,
            negative_prompt_embeds=neg,
            pooled_prompt_embeds=pos_pool,
            negative_pooled_prompt_embeds=neg_pool,
        )
        # Store in LRU cache (limit to 16 entries)
        if len(_SDXL_EMBED_CACHE) >= 16:
            _SDXL_EMBED_CACHE.pop(next(iter(_SDXL_EMBED_CACHE)))
        _SDXL_EMBED_CACHE[cache_key] = {k: v.clone() for k, v in res.items()}
        return res
    except Exception as e:
        # Safety: restore TEs to float16 even on error (only the CPU path casts them)
        try:
            if pipe.text_encoder.dtype == torch.float32 and pipe.unet.dtype == torch.float16:
                pipe.text_encoder.half()
                pipe.text_encoder_2.half()
        except Exception:
            pass
        print(f"[SDXL Compel] Failed ({e}), falling back to raw 77-token strings")
        return {"prompt": plain_prompt(prompt), "negative_prompt": plain_prompt(neg_prompt)}
    finally:
        pipe = c = None


# ══════════════════════════════════════════════════════════════════════════════
# SDXL accelerated text encoding — iGPU (DML) or NPU (VitisAI)
# Offloads CLIP-L + OpenCLIP-G from CPU to the iGPU via ONNX Runtime DML,
# reducing prompt encoding from ~7s to ~100ms on a 780M (mainly for the DML backend).
# ══════════════════════════════════════════════════════════════════════════════

log = logging.getLogger("sdxl_accel_te")

# Module-level state
_accel_te_enabled: bool = False
_te1_session = None  # cached ORT InferenceSession for CLIP-L
_te2_session = None  # cached ORT InferenceSession for OpenCLIP-G
_cached_model_stem: str | None = None  # invalidate sessions on model switch

ONNX_CACHE_DIR = _cfg.APP_DIR / "onnx_cache"
ONNX_CACHE_DIR.mkdir(exist_ok=True)

def _dml_igpu_id() -> int:
    """DML device index of the iGPU. DML order is not fixed: on some laptops
    index 0 is the discrete GPU (e.g. an RX 6800M), so a hardcoded 0 would put the text encoders on the
    same dGPU the UNet is using."""
    igpu = _get_igpu_device()
    if igpu is None:
        raise RuntimeError("no iGPU found for accelerated text encoding")
    return int(igpu.split(":")[1])


def set_sdxl_npu_te(enabled: bool):
    """Toggle accelerated text encoding for SDXL pipeline (iGPU DML or NPU)."""
    global _accel_te_enabled
    _accel_te_enabled = enabled
    print(f"[SDXL Accel-TE] {'Enabled' if enabled else 'Disabled'} (iGPU DML)")


def _force_eager_attention(model):
    """Replace SDPA attention with eager (math) attention for ONNX export compatibility."""
    from transformers.models.clip.modeling_clip import CLIPSdpaAttention, CLIPAttention
    for name, module in model.named_modules():
        if isinstance(module, CLIPSdpaAttention):
            # CLIPSdpaAttention inherits from CLIPAttention; just swap the forward method
            module.forward = CLIPAttention.forward.__get__(module, type(module))


def _export_sdxl_te1_to_onnx(pipe, model_stem: str) -> Path:
    """Export SDXL CLIP-L text encoder to ONNX (penultimate hidden states)."""
    model_dir = ONNX_CACHE_DIR / model_stem
    model_dir.mkdir(exist_ok=True)
    onnx_path = model_dir / "sdxl_te1.onnx"
    if onnx_path.exists():
        return onnx_path

    import torch

    print(f"[SDXL Accel-TE] Exporting CLIP-L text encoder → {onnx_path}")
    te = pipe.text_encoder.eval()
    orig_dev = next(te.parameters()).device
    te.cpu().float()
    _force_eager_attention(te)

    class _TE1Penultimate(torch.nn.Module):
        """Wrapper: returns penultimate hidden state from CLIP-L."""
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids):
            out = self.encoder(input_ids, output_hidden_states=True)
            return out.hidden_states[-2]  # penultimate layer

    wrapper = _TE1Penultimate(te)
    max_len = pipe.tokenizer.model_max_length
    dummy = torch.zeros(1, max_len, dtype=torch.long)

    import warnings
    with warnings.catch_warnings():
        # Suppress harmless ONNX tracer warnings about constant tensor→bool checks
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Converting a tensor to a Python boolean.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.onnx")
        # TracerWarning from torch.jit (subclass of UserWarning but emitted as TracerWarning)
        try:
            from torch.jit import TracerWarning
            warnings.filterwarnings("ignore", category=TracerWarning)
        except ImportError:
            pass
        torch.onnx.export(
            wrapper, (dummy,), str(onnx_path),
            input_names=["input_ids"],
            output_names=["hidden_states"],
            dynamic_axes={"input_ids": {0: "batch"}, "hidden_states": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
    te.half().to(orig_dev)
    print(f"[SDXL Accel-TE] CLIP-L export done ({onnx_path.stat().st_size // 1024 // 1024} MB)")
    return onnx_path


def _export_sdxl_te2_to_onnx(pipe, model_stem: str) -> Path:
    """Export SDXL OpenCLIP-G text encoder to ONNX (penultimate hidden + pooled)."""
    model_dir = ONNX_CACHE_DIR / model_stem
    model_dir.mkdir(exist_ok=True)
    onnx_path = model_dir / "sdxl_te2.onnx"
    if onnx_path.exists():
        return onnx_path

    import torch

    print(f"[SDXL Accel-TE] Exporting OpenCLIP-G text encoder → {onnx_path}")
    te = pipe.text_encoder_2.eval()
    orig_dev = next(te.parameters()).device
    te.cpu().float()
    _force_eager_attention(te)

    class _TE2PenultimatePooled(torch.nn.Module):
        """Wrapper: returns penultimate hidden state + pooled text embedding."""
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids):
            out = self.encoder(input_ids, output_hidden_states=True)
            return out.hidden_states[-2], out.text_embeds

    wrapper = _TE2PenultimatePooled(te)
    max_len = pipe.tokenizer_2.model_max_length
    dummy = torch.zeros(1, max_len, dtype=torch.long)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Converting a tensor to a Python boolean.*")
        warnings.filterwarnings("ignore", category=UserWarning, message=".*aten::index operator.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.onnx")
        try:
            from torch.jit import TracerWarning
            warnings.filterwarnings("ignore", category=TracerWarning)
        except ImportError:
            pass
        torch.onnx.export(
            wrapper, (dummy,), str(onnx_path),
            input_names=["input_ids"],
            output_names=["hidden_states", "text_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "hidden_states": {0: "batch"},
                "text_embeds": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    te.half().to(orig_dev)

    # Total size includes external data files
    total_bytes = sum(f.stat().st_size for f in model_dir.iterdir())
    print(f"[SDXL Accel-TE] OpenCLIP-G export done ({total_bytes // 1024 // 1024} MB total)")
    return onnx_path


def _tokenize_chunked(tokenizer, text: str, max_len: int = 77):
    """
    Tokenize text into 77-token chunks for long prompt support.
    Returns list of numpy arrays, each shape (1, max_len).
    """
    # Tokenize without truncation to get full token list
    full = tokenizer(
        text, add_special_tokens=False, return_tensors="np",
    )["input_ids"][0]  # shape (N,)

    bos = tokenizer.bos_token_id or 49406
    eos = tokenizer.eos_token_id or 49407
    usable = max_len - 2  # reserve BOS + EOS slots

    if len(full) <= usable:
        # Fits in one chunk — standard tokenization with padding
        ids = tokenizer(
            [text], padding="max_length", max_length=max_len,
            truncation=True, return_tensors="np",
        )["input_ids"].astype(np.int64)
        return [ids]

    # Split into usable-token chunks
    chunks = []
    for start in range(0, len(full), usable):
        chunk_tokens = full[start:start + usable]
        # Pad with EOS to fill max_len
        padded = np.full(max_len, eos, dtype=np.int64)
        padded[0] = bos
        padded[1:1 + len(chunk_tokens)] = chunk_tokens
        padded[1 + len(chunk_tokens)] = eos
        chunks.append(padded.reshape(1, max_len))

    return chunks


def _get_ort_sessions(pipe, model_stem: str):
    """Get or create cached ORT InferenceSession pair for CLIP-L + OpenCLIP-G."""
    global _te1_session, _te2_session, _cached_model_stem

    if _cached_model_stem == model_stem and _te1_session and _te2_session:
        return _te1_session, _te2_session

    # Invalidate on model switch
    _te1_session = _te2_session = None
    _cached_model_stem = model_stem

    import onnxruntime as ort

    te1_path = _export_sdxl_te1_to_onnx(pipe, model_stem)
    te2_path = _export_sdxl_te2_to_onnx(pipe, model_stem)

    dml_id = _dml_igpu_id()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ("DmlExecutionProvider", {"device_id": dml_id}),
        "CPUExecutionProvider",
    ]

    print(f"[SDXL Accel-TE] Loading CLIP-L session (DML device {dml_id})...")
    _te1_session = ort.InferenceSession(str(te1_path), opts, providers=providers)
    print(f"[SDXL Accel-TE] Loading OpenCLIP-G session (DML device {dml_id})...")
    _te2_session = ort.InferenceSession(str(te2_path), opts, providers=providers)
    print(f"[SDXL Accel-TE] Sessions ready — {_te1_session.get_providers()[0]}")

    return _te1_session, _te2_session


def _build_sdxl_accel_embeds(pipe, prompt: str, neg_prompt: str, model_stem: str) -> dict | None:
    """
    Encode SDXL prompts on the iGPU via ONNX Runtime DirectML.
    Supports long prompts via chunked encoding (>77 tokens split into segments).

    Returns embeds dict matching Compel output format, or None on failure.
    """
    import torch

    try:
        sess1, sess2 = _get_ort_sessions(pipe, model_stem)
    except Exception as e:
        print(f"[SDXL Accel-TE] Session creation failed: {e}")
        return None

    t0 = time.time()
    max_len_1 = pipe.tokenizer.model_max_length    # 77
    max_len_2 = pipe.tokenizer_2.model_max_length  # 77

    def _encode_text(text: str):
        """Encode one text through both TEs on iGPU with chunked long prompt support."""
        chunks_1 = _tokenize_chunked(pipe.tokenizer, text, max_len_1)
        chunks_2 = _tokenize_chunked(pipe.tokenizer_2, text, max_len_2)
        n_chunks = max(len(chunks_1), len(chunks_2))

        while len(chunks_1) < n_chunks:
            chunks_1.append(chunks_1[-1])
        while len(chunks_2) < n_chunks:
            chunks_2.append(chunks_2[-1])

        all_hs1, all_hs2 = [], []
        pooled = None

        for i in range(n_chunks):
            out1 = sess1.run(None, {"input_ids": chunks_1[i]})
            all_hs1.append(out1[0])  # (1, 77, 768)

            out2 = sess2.run(None, {"input_ids": chunks_2[i]})
            all_hs2.append(out2[0])  # (1, 77, 1280)
            if pooled is None:
                pooled = out2[1]     # (1, 1280)

        hs1 = np.concatenate(all_hs1, axis=1)
        hs2 = np.concatenate(all_hs2, axis=1)
        hidden = np.concatenate([hs1, hs2], axis=-1)  # (1, N*77, 2048)
        return hidden, pooled

    try:
        pos_hidden, pos_pooled = _encode_text(prompt)
        neg_hidden, neg_pooled = _encode_text(neg_prompt)
    except Exception as e:
        print(f"[SDXL Accel-TE] Inference failed: {e}")
        return None

    # Pad to same sequence length
    max_seq = max(pos_hidden.shape[1], neg_hidden.shape[1])
    if pos_hidden.shape[1] < max_seq:
        pad = np.zeros((1, max_seq - pos_hidden.shape[1], 2048), dtype=pos_hidden.dtype)
        pos_hidden = np.concatenate([pos_hidden, pad], axis=1)
    if neg_hidden.shape[1] < max_seq:
        pad = np.zeros((1, max_seq - neg_hidden.shape[1], 2048), dtype=neg_hidden.dtype)
        neg_hidden = np.concatenate([neg_hidden, pad], axis=1)

    pos_embeds = torch.from_numpy(pos_hidden).half()
    neg_embeds = torch.from_numpy(neg_hidden).half()
    pos_pool = torch.from_numpy(pos_pooled).half()
    neg_pool = torch.from_numpy(neg_pooled).half()

    elapsed = time.time() - t0
    n_chunks_pos = len(_tokenize_chunked(pipe.tokenizer, prompt, max_len_1))
    print(f"[SDXL Accel-TE] Encoding done ({elapsed*1000:.0f}ms, {n_chunks_pos} chunk(s), "
          f"seq_len={pos_embeds.shape[1]}, device=iGPU DML)")

    return dict(
        prompt_embeds=pos_embeds,
        negative_prompt_embeds=neg_embeds,
        pooled_prompt_embeds=pos_pool,
        negative_pooled_prompt_embeds=neg_pool,
    )


def _cpu_vae_decode(vae, latents) -> list[Image.Image]:
    """Decode latents on CPU with multi-threading and adaptive untiled decoding for 1.7x-2x faster inference."""
    import os
    import torch
    from PIL import Image as _Img

    # Ensure CPU worker threads are utilized for conv2d math
    try:
        max_threads = os.cpu_count() or 8
        if torch.get_num_threads() < max_threads:
            torch.set_num_threads(max_threads)
    except Exception:
        pass

    scale = getattr(vae.config, "scaling_factor", 0.13025)
    lat_cpu = latents.cpu().float()
    b, c, h, w = lat_cpu.shape

    # Save original device and dtype so we can restore after decode
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    # CPU needs float32
    vae.cpu().float()

    # Adaptive tiling: only tile when latent is large (>160x160, i.e. >1280x1280 image) or batch > 2
    need_tiling = (h > 160 or w > 160 or b > 2)
    if need_tiling:
        vae.enable_tiling()
        vae.enable_slicing()
        vae.tile_sample_min_size = 512
        vae.tile_latent_min_size = 64
        print(f"[SDXL] Decoding latents on CPU (tiled, {w*8}×{h*8})...")
    else:
        vae.disable_tiling()
        vae.disable_slicing()
        print(f"[SDXL] Decoding latents on CPU (direct untiled {torch.get_num_threads()} threads, {w*8}×{h*8})...")

    t0 = time.time()
    try:
        with torch.no_grad():
            decoded = vae.decode(lat_cpu / scale).sample

        decoded = ((decoded / 2 + 0.5).clamp(0, 1) * 255).byte()
        images = [
            _Img.fromarray(decoded[i].permute(1, 2, 0).cpu().numpy())
            for i in range(decoded.shape[0])
        ]
        elapsed = time.time() - t0
        print(f"[SDXL] VAE decode done ({elapsed:.1f}s)")
    finally:
        # Restore to original dtype and device
        vae.to(device=vae_device, dtype=vae_dtype)

    del lat_cpu, decoded
    gc.collect()
    return images


# ── DirectML UNet patches ─────────────────────────────────────────────────────
# DirectML has incomplete op coverage: torch.exp, sin, cos inside
# get_timestep_embedding() silently fall back to CPU, producing CPU tensors
# that crash when they reach DML linear layers.
#
# Strategy: two-layer defence —
#   1. Monkey-patch get_time_embed / get_aug_embed to run Timesteps on CPU
#   2. register_forward_pre_hook on time_embedding & add_embedding to guard
#      every tensor INPUT reaching those modules (catches anything we missed)

def _patch_unet_for_dml(unet):
    """
    Patch UNet for DirectML compatibility.
    Called once after loading the UNet onto DML.
    """
    import types
    import torch

    target_device = next(unet.parameters()).device
    target_dtype = next(unet.parameters()).dtype
    target_dev_str = str(target_device)

    # ── Layer 1: Monkey-patch get_time_embed / get_aug_embed ────────────

    def _safe_get_time_embed(self, sample, timestep):
        """CPU-safe timestep embedding for DirectML."""
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device="cpu")
        else:
            timesteps = timesteps.detach().cpu().float()
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)          # sinusoidal on CPU — always works
        # Two-step move: cast dtype on CPU first (DML .to(dtype) can silently
        # move tensors to CPU), then move to DML device.
        t_emb = t_emb.to(dtype=sample.dtype)       # fp32 → fp16 on CPU
        t_emb = t_emb.to(device=sample.device)     # CPU → DML
        return t_emb

    def _safe_get_aug_embed(self, emb, encoder_hidden_states, added_cond_kwargs):
        """CPU-safe SDXL text_time addition embedding for DirectML."""
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
        if self.config.addition_embed_type != "text_time":
            return UNet2DConditionModel.get_aug_embed(
                self, emb, encoder_hidden_states, added_cond_kwargs
            )
        text_embeds = added_cond_kwargs["text_embeds"]
        time_ids = added_cond_kwargs["time_ids"]
        # Run add_time_proj on CPU to avoid DML op fallbacks
        time_embeds = self.add_time_proj(time_ids.cpu().flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        time_embeds = time_embeds.to(dtype=text_embeds.dtype).to(device=text_embeds.device)
        add_embeds = torch.cat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)
        return aug_emb

    unet.get_time_embed = types.MethodType(_safe_get_time_embed, unet)
    unet.get_aug_embed = types.MethodType(_safe_get_aug_embed, unet)

    # ── Layer 2: Comprehensive device guards ───────────────────────────
    # DirectML silently falls back to CPU for unsupported ops (e.g.
    # aten::_slow_conv2d_forward).  This produces CPU tensors that crash
    # the next DML module.  Fix: register hooks on EVERY sub-module so
    # that (a) Conv2d outputs are moved back to DML after CPU fallback,
    # and (b) all other modules get DML inputs even if upstream was CPU.

    _fallback_warned = set()          # only warn once per module name

    def _output_to_dml(module, _input, output):
        """Post-hook: move Conv2d/GroupNorm output back to DML after CPU fallback."""
        if isinstance(output, torch.Tensor) and output.device.type != "privateuseone":
            name = getattr(module, "_dml_hook_name", module.__class__.__name__)
            if name not in _fallback_warned:
                _fallback_warned.add(name)
                print(f"[SDXL DML] {name}: CPU fallback detected → moving output to DML")
            return output.to(dtype=target_dtype, device=target_device)
        return output

    def _input_to_dml(module, args):
        """Pre-hook: ensure all tensor inputs are on DML."""
        fixed = list(args)
        changed = False
        for i, a in enumerate(fixed):
            if torch.is_tensor(a) and a.device.type != "privateuseone":
                fixed[i] = a.to(dtype=target_dtype, device=target_device)
                changed = True
        return tuple(fixed) if changed else args

    # Register hooks on every sub-module of the UNet
    hook_count = 0
    for name, mod in unet.named_modules():
        if not name:
            continue  # skip UNet top-level
        mod._dml_hook_name = name
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            # Output hook: catch CPU fallback from _slow_conv2d_forward
            mod.register_forward_hook(_output_to_dml)
            hook_count += 1
        elif isinstance(mod, (torch.nn.GroupNorm, torch.nn.LayerNorm,
                              torch.nn.Linear)):
            # Pre-hook: ensure DML inputs for ops that can't handle CPU tensors
            mod.register_forward_pre_hook(_input_to_dml)
            hook_count += 1

    print(f"[SDXL] UNet patched for DirectML (device={target_dev_str}, "
          f"dtype={target_dtype}, {hook_count} device-guard hooks)")


def _patch_scheduler_for_dml(scheduler, dml_device):
    """
    Wrap scheduler methods for DirectML compatibility.

    DML silently falls back to CPU for many ops used in scheduler math
    (torch.exp, torch.pow, torch.sqrt, etc.).  Instead of trying to keep
    scheduler state on DML, we keep everything on CPU — the natural home
    for scheduler math — and shuttle latents / noise_pred between CPU and
    DML at each step boundary.

    Cost: ~128 KiB per transfer per step for 1024×1024 SDXL latents.
    Over 50 steps this adds < 2 ms total at PCIe 3.0 speeds — negligible.
    """
    import types
    import torch

    _Cls = type(scheduler)
    _orig_set_timesteps = _Cls.set_timesteps
    _orig_step = _Cls.step
    _orig_scale = _Cls.scale_model_input
    _orig_add_noise = getattr(_Cls, "add_noise", None)

    # ── set_timesteps: force CPU for all scheduler state tensors ──────
    def _set_timesteps_cpu(self, *args, device=None, **kwargs):
        _orig_set_timesteps(self, *args, device="cpu", **kwargs)

    # ── step: run on CPU, shuttle results back to DML ─────────────────
    def _step_cpu(self, model_output, timestep, sample, *args, **kwargs):
        need_dml = (model_output.device.type == "privateuseone"
                    or sample.device.type == "privateuseone")
        mo = model_output.cpu() if model_output.device.type == "privateuseone" else model_output
        s = sample.cpu() if sample.device.type == "privateuseone" else sample
        ts = (timestep.cpu()
              if torch.is_tensor(timestep) and timestep.device.type == "privateuseone"
              else timestep)

        result = _orig_step(self, mo, ts, s, *args, **kwargs)

        if not need_dml:
            return result

        # Move result tensors back to DML
        if isinstance(result, tuple):
            return tuple(
                r.to(dml_device) if torch.is_tensor(r) else r
                for r in result
            )
        if hasattr(result, "prev_sample") and torch.is_tensor(result.prev_sample):
            result.prev_sample = result.prev_sample.to(dml_device)
        return result

    # ── scale_model_input: some schedulers (Euler) divide by sigma ────
    def _scale_cpu(self, sample, *args, **kwargs):
        need_dml = sample.device.type == "privateuseone"
        s = sample.cpu() if need_dml else sample
        # Handle DML timestep arg if present
        cpu_args = tuple(
            a.cpu() if torch.is_tensor(a) and a.device.type == "privateuseone" else a
            for a in args
        )
        result = _orig_scale(self, s, *cpu_args, **kwargs)
        if need_dml and torch.is_tensor(result):
            return result.to(dml_device)
        return result

    scheduler.set_timesteps = types.MethodType(_set_timesteps_cpu, scheduler)
    scheduler.step = types.MethodType(_step_cpu, scheduler)
    scheduler.scale_model_input = types.MethodType(_scale_cpu, scheduler)

    # ── add_noise: used by img2img to noise the init latents ──────────
    if _orig_add_noise is not None:
        def _add_noise_cpu(self, original_samples, noise, timesteps):
            need_dml = original_samples.device.type == "privateuseone"
            os_ = original_samples.cpu() if need_dml else original_samples
            n = noise.cpu() if noise.device.type == "privateuseone" else noise
            ts = (timesteps.cpu()
                  if torch.is_tensor(timesteps) and timesteps.device.type == "privateuseone"
                  else timesteps)
            result = _orig_add_noise(self, os_, n, ts)
            if need_dml and torch.is_tensor(result):
                return result.to(dml_device)
            return result
        scheduler.add_noise = types.MethodType(_add_noise_cpu, scheduler)

    print("[SDXL DML] Scheduler patched: CPU math + auto DML↔CPU transfer")


class SDXLPipeline:
    """
    Dedicated SDXL / Pony / Illustrious pipeline.
    Keeps SD 1.5 code paths in SDPipeline completely untouched.

    Device strategy on DirectML:
      - UNet → DML GPU (bulk of compute)
      - Text encoders → CPU (required for Compel, ~2.5 GB RAM)
      - VAE → CPU (avoids OOM after UNet fills VRAM, ~300 MB RAM)
    """

    def __init__(self):
        self.device = _device()
        self.dtype = get_torch_dtype(_device())
        self.pipe = None
        self.img2img_pipe = None
        self.current_model: str | None = None
        self.loaded_loras: list[str] = []
        self._lora_adapters: dict[int, tuple[str, str, float]] = {}
        self.is_sdxl = True  # always True for this class
        self.model_family = "sdxl"
        self.last_seeds: list[int] = []
        self.last_var_seeds: list[int] = []
        self.unet_only_dml: bool = False  # not used by SDXL pipeline, kept for API compat
        self._last_vae_path: str | None = None
        self._clean_unet_state: dict | None = None
        self._clean_te_state: dict | None = None
        self._clean_te2_state: dict | None = None
        self._dml_dev = None  # cached DML device object
        self._vae_needs_fp32 = False  # set when this model's VAE overflows in fp16

    # ── Model loading ──────────────────────────────────────────────────────────
    def load_model(self, model_path_or_id: str, vae_path: str | None = None) -> str:
        import torch
        from diffusers import StableDiffusionXLPipeline, AutoencoderKL

        if model_path_or_id == self.current_model:
            return f"✅ Already loaded: {Path(model_path_or_id).name}"

        self.device = _device()
        self.dtype = get_torch_dtype(self.device)
        self._unload()

        path = Path(model_path_or_id)
        is_local = path.exists()
        self.model_family = _model_family(model_path_or_id)

        # SDXL has no safety_checker — don't pass those kwargs
        common_kwargs = dict(torch_dtype=self.dtype)

        try:
            if is_local and path.suffix in (".safetensors", ".ckpt"):
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    str(path), **common_kwargs
                )
            else:
                from backend.sd_pipeline import _from_pretrained
                self.pipe = _from_pretrained(
                    StableDiffusionXLPipeline, model_path_or_id, **common_kwargs
                )

            # Optionally swap VAE
            if vae_path and Path(vae_path).exists():
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=self.dtype)
                self.pipe.vae = vae
            self._last_vae_path = vae_path

            # ── Device placement ──
            if "privateuseone" in self.device:
                import torch_directml
                dml_dev = torch_directml.device()
                # UNet-only on GPU; text encoders + VAE stay on CPU in float16.
                # TEs are temporarily cast to float32 during Compel encoding
                # (CPU has no float16 HW) and restored to float16 before pipeline call.
                # text_encoder_2.dtype MUST be float16 at pipeline call time — it's
                # used as reference dtype for timestep embeds, add_time_ids, latents.
                self.pipe.unet = self.pipe.unet.to(dml_dev)
                _patch_unet_for_dml(self.pipe.unet)
                self._dml_dev = dml_dev
                _patch_scheduler_for_dml(self.pipe.scheduler, dml_dev)
                print(f"[SDXL Load] UNet → DirectML (fp16)  |  Text+VAE → CPU (fp16)")
            else:
                self.pipe = self.pipe.to(self.device)

            # Memory efficiency
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing(1)
            if hasattr(self.pipe, "vae"):
                self.pipe.vae.enable_tiling()
                self.pipe.vae.tile_sample_min_size = 512
                self.pipe.vae.tile_latent_min_size = 64
                self.pipe.vae.enable_slicing()

            self.current_model = model_path_or_id
            self.loaded_loras = []
            self._lora_adapters = {}
            self.img2img_pipe = None
            from backend.sampling import detect_prediction, configure_prediction
            self.prediction = detect_prediction(model_path_or_id) if is_local else {}
            configure_prediction(self.pipe, self.prediction)
            if self.prediction.get("v_pred"):
                print(f"[SDXL Load] v-prediction checkpoint (zero-terminal SNR: {self.prediction['zero_snr']})")
            _load_embeddings(self.pipe)
            # LoRA restore snapshot is taken lazily on first load_lora():
            # saves a full CPU copy of the weights (~6.6 GB for SDXL) when no LoRA is used.

            model_name = path.stem if is_local else model_path_or_id
            fam = self.model_family.capitalize() + (" · v-pred" if self.prediction.get("v_pred") else "")
            return f"✅ Loaded: {model_name} ({fam}, {self.device})"

        except Exception as e:
            self._unload()
            from backend.model_manager import safetensors_problem
            bad = safetensors_problem(model_path_or_id)   # say *why*, not just "unable to load"
            return f"❌ {bad}" if bad else f"❌ Failed to load model: {e}"

    def _unload(self):
        import torch
        self._vae_needs_fp32 = False
        self.pipe = None
        self.img2img_pipe = None
        # the cached inpaint pipe (face detail / inpaint) holds the old UNet, TEs and VAE:
        # left in place it kept ~2.5 GB of VRAM after switching SD 1.5 → SDXL
        self._inpaint_pipe = None
        self._pag_pipes = None
        self.prediction = {}
        self.current_model = None
        self.loaded_loras = []
        self._lora_adapters = {}
        self._clean_unet_state = None
        self._clean_te_state = None
        self._clean_te2_state = None
        _clear_embed_cache()
        from backend.prompt_tools import release_parser_cache
        release_parser_cache()        # it can hold Compel → the text encoders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Snapshot / restore for LoRA ────────────────────────────────────────────
    def _snapshot_clean_state(self):
        if self.pipe is None:
            return
        import torch
        with torch.no_grad():
            self._clean_unet_state = {
                n: p.detach().cpu().clone()
                for n, p in self.pipe.unet.named_parameters()
            }
            if self.pipe.text_encoder is not None:
                self._clean_te_state = {
                    n: p.detach().cpu().clone()
                    for n, p in self.pipe.text_encoder.named_parameters()
                }
            if hasattr(self.pipe, "text_encoder_2") and self.pipe.text_encoder_2 is not None:
                self._clean_te2_state = {
                    n: p.detach().cpu().clone()
                    for n, p in self.pipe.text_encoder_2.named_parameters()
                }
        total = sum(v.numel() for v in self._clean_unet_state.values()) / 1e6
        # Save fingerprint for comparison during generation
        self._clean_fingerprint = []
        for n, v in self._clean_unet_state.items():
            if 'attn1.to_q.weight' in n:
                self._clean_fingerprint.append(v.float().norm().item())
                if len(self._clean_fingerprint) >= 2:
                    break
        print(f"[SDXL LoRA] Snapshot saved ({total:.0f}M UNet params on CPU)")

    def _restore_clean_state(self):
        if self.pipe is None or not self._clean_unet_state:
            return
        import torch
        with torch.no_grad():
            restored, missed = 0, 0
            for name, param in self.pipe.unet.named_parameters():
                if name in self._clean_unet_state:
                    param.data.copy_(self._clean_unet_state[name].to(param.device))
                    restored += 1
                else:
                    missed += 1
            if missed:
                print(f"[SDXL LoRA] WARNING: {missed} UNet params not in snapshot (restored {restored})")

            if self._clean_te_state and self.pipe.text_encoder is not None:
                for name, param in self.pipe.text_encoder.named_parameters():
                    if name in self._clean_te_state:
                        param.data.copy_(self._clean_te_state[name].to(param.device))
            if self._clean_te2_state and getattr(self.pipe, "text_encoder_2", None) is not None:
                for name, param in self.pipe.text_encoder_2.named_parameters():
                    if name in self._clean_te2_state:
                        param.data.copy_(self._clean_te2_state[name].to(param.device))

    # ── LoRA management ────────────────────────────────────────────────────────
    def load_lora(self, lora_path: str, weight: float = 0.8, slot: int = 0) -> str:
        if self.pipe is None:
            return "❌ Load a base model first."
        if self._clean_unet_state is None:
            self._snapshot_clean_state()  # model is still LoRA-free here
        old_entry = self._lora_adapters.get(slot)
        try:
            self._lora_adapters[slot] = (Path(lora_path).name, lora_path, weight)
            self._reload_all_loras()
            names = [v[0] for v in sorted(self._lora_adapters.values())]
            return (f"✅ Slot {slot+1}: {Path(lora_path).name} (×{weight:.2f})"
                    + (f" | Active: {', '.join(names)}" if len(names) > 1 else ""))
        except Exception as e:
            # Roll back: restore previous slot entry (or remove it)
            if old_entry is not None:
                self._lora_adapters[slot] = old_entry
            else:
                self._lora_adapters.pop(slot, None)
            # Attempt to restore clean state so pipeline isn't corrupted
            try:
                self._reload_all_loras()
            except Exception:
                pass
            from backend.model_manager import safetensors_problem
            bad = safetensors_problem(lora_path)
            return f"❌ {bad}" if bad else f"❌ LoRA error: {e}"

    def remove_lora(self, slot: int) -> str:
        if slot not in self._lora_adapters:
            return f"Slot {slot+1} is empty."
        name = self._lora_adapters.pop(slot)[0]
        self._reload_all_loras()
        remaining = [v[0] for v in sorted(self._lora_adapters.values())]
        return (f"✅ Removed: {name}"
                + (f" | Active: {', '.join(remaining)}" if remaining else " | No LoRAs active"))

    def _purge_peft(self):
        if self.pipe is None:
            return
        try:
            self.pipe.unload_lora_weights()
        except Exception:
            pass
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
        import torch

        _clear_embed_cache()
        self._purge_peft()
        self.loaded_loras = []

        if not self._lora_adapters:
            self._restore_clean_state()
            # Re-register DML hooks after _purge_peft may have stripped modules
            if self._dml_dev is not None:
                for mod in self.pipe.unet.modules():
                    mod._forward_hooks.clear()
                    mod._forward_pre_hooks.clear()
                _patch_unet_for_dml(self.pipe.unet)
            return

        # ── DML: fuse on CPU (PEFT ops unreliable on DirectML). CUDA/ZLUDA fuses
        # in place: moving the 5 GB UNet to CPU crashes inside ZLUDA (access
        # violation in Module.cpu()), and it's ~13 GB of PCIe traffic per change.
        dml_dev = self._dml_dev
        if dml_dev is not None:
            self.pipe.unet.cpu()

        try:
            self._restore_clean_state()  # snapshot → current device

            # Sample a few reference params to verify fusion changes weights
            _verify_keys = []
            for n, p in self.pipe.unet.named_parameters():
                if 'attn1.to_q.weight' in n:
                    _verify_keys.append(n)
                    if len(_verify_keys) >= 3:
                        break
            _pre_norms = {k: self.pipe.unet.get_parameter(k).detach().float().norm().item()
                          for k in _verify_keys}

            for slot in sorted(self._lora_adapters.keys()):
                name_str, path, weight = self._lora_adapters[slot]
                from backend.model_manager import lycoris_kind
                if lycoris_kind(path):   # LoHa / LoKr: diffusers can't load these, fuse ourselves
                    from backend.lycoris import apply_lycoris
                    n, skipped = apply_lycoris(self.pipe, path, weight)
                    self.loaded_loras.append(name_str)
                    print(f"[SDXL LoRA] Slot {slot+1}: {name_str} (LyCORIS, {n} layers"
                          f"{f', {skipped} unmatched' if skipped else ''}) fused at ×{weight:.2f}")
                    continue
                try:
                    self.pipe.load_lora_weights(path, adapter_name=_LORA_ADAPTER)
                except Exception as e:
                    err = str(e)
                    if "Target modules" in err and "not found in the base model" in err:
                        raise ValueError(
                            f"LoRA '{Path(path).name}' is incompatible with this SDXL model "
                            "(layer names don't match). It may be an SD 1.5 LoRA."
                        ) from None
                    raise

                self.pipe.fuse_lora(lora_scale=weight)
                self._purge_peft()
                self.loaded_loras.append(name_str)
                print(f"[SDXL LoRA] Slot {slot+1}: {name_str} fused at ×{weight:.2f}")

            # ── Verify fusion (only warn on failure) ──
            diffs = []
            for k in _verify_keys:
                try:
                    post = self.pipe.unet.get_parameter(k).detach().float().norm().item()
                    diffs.append(abs(post - _pre_norms[k]))
                except Exception:
                    pass
            if diffs and sum(diffs) / len(diffs) < 1e-6:
                print("[SDXL LoRA] ⚠ WARNING: UNet weights appear UNCHANGED after fusion!")

            if self._clean_te2_state and self.pipe.text_encoder_2 is not None:
                te_diffs = []
                for n, p in self.pipe.text_encoder_2.named_parameters():
                    if n in self._clean_te2_state and 'self_attn.q_proj.weight' in n:
                        d = (p.detach().cpu().float() - self._clean_te2_state[n].float()).norm().item()
                        te_diffs.append(d)
                        if len(te_diffs) >= 3:
                            break
                if te_diffs and sum(te_diffs) / len(te_diffs) < 1e-6:
                    print("[SDXL LoRA] ⚠ WARNING: TE2 weights appear UNCHANGED after fusion!")

        finally:
            # ALWAYS restore model to original device, even on failure
            if dml_dev is not None:
                self.pipe.unet.to(dml_dev)
                for mod in self.pipe.unet.modules():
                    mod._forward_hooks.clear()
                    mod._forward_pre_hooks.clear()
                _patch_unet_for_dml(self.pipe.unet)

                # Verify DML round-trip preserved fused weights
                for k in _verify_keys:
                    try:
                        post = self.pipe.unet.get_parameter(k).detach().cpu().float().norm().item()
                        if abs(post - _pre_norms[k]) < 1e-6:
                            print("[SDXL LoRA] ⚠ WARNING: DML round-trip LOST fused weights!")
                            break
                    except Exception:
                        pass

        # Load companion TI embeddings after fusion + device restore
        for slot in sorted(self._lora_adapters.keys()):
            _, path, _ = self._lora_adapters[slot]
            _load_companion_ti(self.pipe, path)

    def unload_loras(self) -> str:
        if self.pipe is None:
            return "No model loaded."
        self._purge_peft()
        self._restore_clean_state()
        self._lora_adapters = {}
        self.loaded_loras = []
        # Re-register hooks — _purge_peft replaced module objects
        if self._dml_dev is not None:
            for mod in self.pipe.unet.modules():
                mod._forward_hooks.clear()
                mod._forward_pre_hooks.clear()
            _patch_unet_for_dml(self.pipe.unet)
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
        clip_skip: int = 1,                    # SDXL always uses the penultimate layer already
        var_seed: int = -1,
        var_strength: float = 0.0,
    ) -> tuple[list[Image.Image], str]:
        if self.pipe is None:
            return [], "❌ No model loaded. Pick a checkpoint and click Load Model (Generate tab)."

        import torch

        _load_scheduler(self.pipe, scheduler)
        # _load_scheduler creates a NEW scheduler — re-patch for DML
        if self._dml_dev is not None:
            _patch_scheduler_for_dml(self.pipe.scheduler, self._dml_dev)
        generator, seeds = _make_generators(seed, self.device, max(1, int(batch_size)))
        self.last_seeds = seeds
        used_seed = seeds[0]
        from backend.sd_pipeline import variation_latents
        latents, self.last_var_seeds = variation_latents(
            self.pipe, generator, var_seed, var_strength, max(1, int(batch_size)), width, height, self.device)
        lat_kw = {"latents": latents} if latents is not None else {}
        # Decode latents ourselves: GPU (ZLUDA) with CPU fallback, or CPU (DML)
        use_cpu_vae = "privateuseone" in self.device or "cuda" in self.device

        # ── Step callback with console + Gradio progress ──
        cb_kwargs = {}
        if step_callback is not None:
            def _pipe_callback(pipe, i, t, callback_kwargs):
                # The GPU runs asynchronously: without this the bar reaches the last step
                # while ~half the work is still queued (SDXL batch 2: 20/20 at 24 s, done at
                # 55 s) — and Stop has to wait for the whole queue to drain. (Costs ~0.6 %;
                # SD 1.5 skips it: its steps are short and the sync would cost ~4 %.)
                if "cuda" in str(self.device):
                    torch.cuda.synchronize()
                step_callback(i + 1, steps)
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"[SDXL] Step {i+1}/{steps}")
                return callback_kwargs
            cb_kwargs = {"callback_on_step_end": _pipe_callback}

        _vram_reset(self.device)
        t0 = time.time()

        # ── Generation summary ──
        lora_summary = " | ".join(
            f"{name}×{w:.2f}" for _, (name, _, w) in sorted(self._lora_adapters.items())
        ) or "none"
        print(f"\n{'='*72}")
        print(f"[SDXL Generate] seed={used_seed} | {width}×{height} | steps={steps} | "
              f"CFG={cfg_scale} | batch={batch_size} | sched={scheduler}")
        print(f"[SDXL Generate] LoRAs: {lora_summary}")
        print(f"[SDXL Generate] Prompt: {prompt[:200]}{'…' if len(prompt)>200 else ''}")
        # Quick UNet weight fingerprint to confirm LoRA state
        _fp_vals = []
        for n, p in self.pipe.unet.named_parameters():
            if 'attn1.to_q.weight' in n:
                _fp_vals.append(f"{p.detach().cpu().float().norm().item():.4f}")
                if len(_fp_vals) >= 2:
                    break
        if _fp_vals:
            clean_fp = ", ".join(f"{v:.4f}" for v in getattr(self, '_clean_fingerprint', []))
            print(f"[SDXL Generate] UNet fingerprint: {', '.join(_fp_vals)} "
                  f"(clean: {clean_fp or 'N/A'})")
        print(f"{'='*72}")

        # ── Prompt encoding: try iGPU DML first, fall back to CPU Compel ─────
        embeds = None
        if _accel_te_enabled:
            print(f"[SDXL] Encoding prompts (iGPU DML)...")
            _stem = Path(self.current_model).stem if self.current_model else "sdxl"
            embeds = _build_sdxl_accel_embeds(self.pipe, prompt, negative_prompt, _stem)
            if embeds is None:
                print(f"[SDXL] Accelerated encoding failed, falling back to CPU...")

        if embeds is None:
            _te_dev = next(self.pipe.text_encoder.parameters()).device
            print(f"[SDXL] Encoding prompts ({'GPU' if _te_dev.type == 'cuda' else _te_dev})...")
            embeds = _build_sdxl_embeds(self.pipe, prompt, negative_prompt)

        t_enc = time.time() - t0
        print(f"[SDXL] Prompt encoding done ({t_enc:.1f}s), starting denoising ({steps} steps)...")

        if use_cpu_vae:
            embeds["output_type"] = "latent"
        if "cuda" in self.device:
            # Drop cached blocks from the previous run: on Windows an over-full
            # dGPU spills into shared memory instead of raising OOM.
            torch.cuda.empty_cache()

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
                    f"Try reducing resolution (e.g. 768×768 or 832×1216) or reducing batch size to 1."
                )
                print(f"[SDXL] {oom_msg}")
                return [], oom_msg
            raise

        if "cuda" in self.device:
            torch.cuda.synchronize()  # pipe() returns before the GPU queue drains
        t_denoise = time.time() - t0 - t_enc
        print(f"[SDXL] Denoising done ({t_denoise:.1f}s)")

        if use_cpu_vae:
            images = self._decode_latents(self.pipe.vae, result.images)
        else:
            images = result.images

        elapsed = time.time() - t0
        info = (
            f"{_seed_label(seeds)} | Steps: {steps} | CFG: {cfg_scale} | "
            f"Scheduler: {scheduler} | Size: {width}×{height} | "
            f"Time: {elapsed:.1f}s (encode {t_enc:.0f}s + denoise {t_denoise:.0f}s + VAE {elapsed - t_enc - t_denoise:.0f}s) | "
            f"Device: {self.device} | Family: {self.model_family}"
        )
        info += _vram_spill_note(self.device)
        # app.py _save_outputs() handles saving with seed in filename — don't double-save
        return images, info

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
        clip_skip: int = 1,                    # (SDXL already uses the penultimate layer)
    ) -> tuple[list[Image.Image], str]:
        if self.pipe is None:
            return [], "❌ No model loaded."

        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline

        if self.img2img_pipe is None:
            # from_pipe defaults to torch_dtype=float32 and casts the *shared* UNet/TEs,
            # silently doubling VRAM (SDXL spills to shared memory) for txt2img too.
            self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(self.pipe, torch_dtype=self.dtype)
            # Components shared with txt2img pipe — already on correct devices.
            self.img2img_pipe.enable_attention_slicing(1)
            self.img2img_pipe.vae.enable_tiling()
            self.img2img_pipe.vae.tile_sample_min_size = 512
            self.img2img_pipe.vae.tile_latent_min_size = 64
            self.img2img_pipe.vae.enable_slicing()

        _load_scheduler(self.img2img_pipe, scheduler)
        if self._dml_dev is not None:
            _patch_scheduler_for_dml(self.img2img_pipe.scheduler, self._dml_dev)
        generator, used_seed = _make_generator(seed, self.device)
        self.last_seeds = [used_seed]
        # Decode latents ourselves: GPU (ZLUDA) with CPU fallback, or CPU (DML)
        use_cpu_vae = "privateuseone" in self.device or "cuda" in self.device

        init_image = init_image.convert("RGB").resize(
            ((init_image.width + 7) // 8 * 8,
             (init_image.height + 7) // 8 * 8),
            Image.LANCZOS,
        )

        cb_kwargs = {}
        if step_callback is not None:
            def _pipe_callback(pipe, i, t, callback_kwargs):
                # The GPU runs asynchronously: without this the bar reaches the last step
                # while ~half the work is still queued (SDXL batch 2: 20/20 at 24 s, done at
                # 55 s) — and Stop has to wait for the whole queue to drain. (Costs ~0.6 %;
                # SD 1.5 skips it: its steps are short and the sync would cost ~4 %.)
                if "cuda" in str(self.device):
                    torch.cuda.synchronize()
                step_callback(i + 1, steps)
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"[SDXL i2i] Step {i+1}/{steps}")
                return callback_kwargs
            cb_kwargs = {"callback_on_step_end": _pipe_callback}

        _vram_reset(self.device)
        t0 = time.time()
        embeds = None
        if _accel_te_enabled:
            _stem = Path(self.current_model).stem if self.current_model else "sdxl"
            embeds = _build_sdxl_accel_embeds(self.img2img_pipe, prompt, negative_prompt, _stem)
        if embeds is None:
            embeds = _build_sdxl_embeds(self.img2img_pipe, prompt, negative_prompt)
        if use_cpu_vae:
            embeds["output_type"] = "latent"
        if "privateuseone" in self.device:
            # DML keeps the VAE on CPU, which can't do float16 conv2d — cast to
            # float32 for encoding the init image AND for the manual decode.
            # (On ZLUDA the VAE is on the GPU; diffusers upcasts for encode itself.)
            vae = self.img2img_pipe.vae
            vae.float()
            vae.enable_tiling()
            vae.tile_sample_min_size = 512
            vae.tile_latent_min_size = 64

        try:
            with torch.no_grad():
                if "cuda" in self.device:
                    init_image = self._encode_image(init_image, generator)
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
        finally:
            # Always restore VAE to float16 — pipeline uses text_encoder_2.dtype
            # as reference dtype, and VAE must not stay float32 across calls.
            if use_cpu_vae:
                self.img2img_pipe.vae.half()

        elapsed = time.time() - t0
        info = (
            f"img2img | Strength: {strength} | Seed: {used_seed} | "
            f"Steps: {steps} | CFG: {cfg_scale} | Time: {elapsed:.1f}s"
        )
        info += _vram_spill_note(self.device)
        # app.py _save_outputs() handles saving — don't double-save
        return images, info

    def _encode_image(self, image: Image.Image, generator):
        """ZLUDA: fp16 tiled GPU encode, then free the scratch before denoising.
        Letting diffusers encode upcasts the VAE to fp32 at full resolution; with
        cuDNN off the im2col buffers push VRAM into shared system memory, and
        Windows doesn't OOM — every following step just crawls over PCIe."""
        import torch
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import retrieve_latents
        vae = self.img2img_pipe.vae
        vae.enable_tiling()
        vae.tile_sample_min_size = 512
        vae.tile_latent_min_size = 64
        x = self.img2img_pipe.image_processor.preprocess(image).to(vae.device)
        torch.cuda.empty_cache()   # start from a clean pool: fp32 scratch is 2× fp16
        try:
            # fp32 only if fp16 overflows (remembered per model, shared with decode)
            dtypes = (torch.float32,) if self._vae_needs_fp32 else (self.dtype, torch.float32)
            for dtype in dtypes:
                vae.to(dtype=dtype)
                tile = 256 if dtype == torch.float32 else 512   # keep fp32 scratch small
                vae.tile_sample_min_size = tile
                vae.tile_latent_min_size = tile // 8
                with torch.no_grad():
                    lat = retrieve_latents(vae.encode(x.to(dtype)), generator=generator)
                if not torch.isnan(lat).any():
                    break
                print("[SDXL i2i] fp16 VAE encode produced NaN — retrying in fp32")
                self._vae_needs_fp32 = True
        finally:
            vae.to(dtype=self.dtype)
            vae.tile_sample_min_size = 512
            vae.tile_latent_min_size = 64
            del x
            torch.cuda.empty_cache()
        # 4-channel input is treated as ready-made latents by the img2img pipeline
        return (lat * vae.config.scaling_factor).to(self.dtype)

    def _decode_latents(self, vae, latents) -> list[Image.Image]:
        """ZLUDA: tiled decode on the GPU (untiled needs >4 GB of im2col scratch with
        cuDNN off). fp16 first (~4 s); many SDXL checkpoints ship the original SDXL VAE,
        which overflows in fp16 → NaN, so retry in fp32 with 256 px tiles (~8 s, fits
        12 GB) and remember that for this model. CPU (~22 s) is the last resort.
        DML: CPU decode."""
        import torch
        if "cuda" in self.device:
            attempts = [] if self._vae_needs_fp32 else [(self.dtype, 512)]
            attempts.append((torch.float32, 256))
            vae.enable_tiling()
            vae.enable_slicing()
            try:
                for dtype, tile in attempts:
                    vae.to(dtype=dtype)
                    vae.tile_sample_min_size = tile
                    vae.tile_latent_min_size = tile // 8
                    images = _gpu_vae_decode(vae, latents, "SDXL")
                    if images is not None:
                        if dtype == torch.float32 and not self._vae_needs_fp32:
                            self._vae_needs_fp32 = True
                            print("[SDXL] This model's VAE needs fp32 — using it from now on")
                        return images
            finally:
                vae.to(dtype=self.dtype)
                vae.tile_sample_min_size = 512
                vae.tile_latent_min_size = 64
            print("[SDXL] GPU VAE decode failed — decoding on CPU")
        return _cpu_vae_decode(vae, latents)
