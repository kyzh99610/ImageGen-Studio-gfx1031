"""
LoRA training for Stable Diffusion 1.5 and SDXL / Pony / Illustrious.

Produces standard kohya-format safetensors files that are directly loadable
by ``pipe.load_lora_weights()`` and the ImageGenApp pipeline.

Supports:
  - Aspect-ratio bucketing  (no forced square crop)
  - Latent caching           (VAE encodes once → training only runs UNet)
  - Text-encoder LoRA        (optional, recommended for character LoRAs)
  - Gradient checkpointing   (lower VRAM at ~15 % speed cost)
  - CPU or DirectML training (auto-detected)
  - Progress callback        (drives Gradio progress bar)
"""

from __future__ import annotations

import gc
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

from backend.dataset_manager import (
    compute_bucket_assignments,
    read_caption,
    resize_and_crop,
    scan_images,
)

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    base_model: str = ""           # checkpoint path
    model_type: str = ""           # "sd15" or "sdxl" (auto-detected)

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_rank: int = 16
    lora_alpha: int = 16
    train_unet: bool = True
    train_te: bool = True          # also train text encoder(s)
    # "attn" = attention layers only,  "full" = + ResNet convs + I/O convs
    target_coverage: str = "attn"

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset_dir: str = ""          # folder with prepared images + .txt captions
    trigger_word: str = ""
    resolution: int = 512          # 512 for SD 1.5, 1024 for SDXL
    use_bucketing: bool = True
    flip_augment: bool = False     # random horizontal flip

    # ── Training ───────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    te_learning_rate: float = 5e-5 # separate LR for text encoder
    lr_scheduler: str = "cosine"   # constant | cosine | linear
    epochs: int = 10
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05     # fraction of total steps
    seed: int = 42

    # ── Memory ─────────────────────────────────────────────────────────────
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    mixed_precision: bool = False  # fp16 mixed precision (DML only)

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: str = ""
    output_name: str = "my_lora"
    save_every_n_epochs: int = 0   # 0 = final only

    # ── Device ─────────────────────────────────────────────────────────────
    device: str = "auto"           # "auto" (CUDA/ZLUDA→CPU), "cuda", "directml", or "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

_UNET_ATTN_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0",
    "proj_in", "proj_out",
]
_UNET_FULL_TARGETS = _UNET_ATTN_TARGETS + [
    "conv1", "conv2", "conv_shortcut",
    "time_emb_proj", "conv_in", "conv_out",
]
_TE_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


def _best_cuda_device() -> int:
    """Pick the best CUDA device: prefer IMAGEGEN_GPU env, else dGPU over iGPU."""
    try:
        env_idx = os.environ.get("IMAGEGEN_GPU")
        if env_idx is not None:
            idx = int(env_idx)
            if 0 <= idx < torch.cuda.device_count():
                return idx
    except (ValueError, TypeError):
        pass

    n = torch.cuda.device_count()
    if n <= 1:
        return 0
    # Prefer discrete GPUs (RX series) over iGPUs (780M, Vega, 610M)
    igpu_keywords = ["780M", "Vega", "610M", "gfx1103", "gfx1102"]
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        if not any(kw.lower() in name.lower() for kw in igpu_keywords):
            return i  # first non-iGPU device
    return 0


def _get_device(cfg: TrainingConfig) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")

    # "auto" tries: CUDA/ZLUDA → CPU  (DML backward is broken for UNet)
    # "cuda" forces CUDA/ZLUDA
    # "directml" forces DML (will likely crash on backward — expert use only)
    if cfg.device in ("cuda", "auto") and torch.cuda.is_available():
        # Pick the best CUDA device (dGPU over iGPU)
        dev_idx = _best_cuda_device()
        dev = torch.device(f"cuda:{dev_idx}")
        name = torch.cuda.get_device_name(dev_idx)
        print(f"[Train] Using CUDA device {dev_idx}: {name}")
        # ZLUDA on AMD: disable flash/mem-efficient SDPA (needs cutlass, unavailable on HIP)
        if "ZLUDA" in name or "AMD" in name or "Radeon" in name:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            print("[Train] Disabled flash/mem-efficient SDPA for AMD/ZLUDA compatibility")
        return dev

    if cfg.device == "directml":
        if _check_dml_backward():
            import torch_directml
            dev = torch_directml.device()
            print(f"[Train] Using DirectML device: {dev}")
            print("[Train] ⚠ DirectML training is experimental — may crash on full UNet backward")
            return dev
        else:
            print("[Train] ⚠ DirectML backward() smoke test failed — falling back to CPU")

    if cfg.device == "auto":
        print("[Train] Auto-detect: no GPU training available, using CPU")
        if not torch.cuda.is_available():
            _diagnose_gpu()
        print("[Train]   Tip: Launch via launch.bat to enable ZLUDA → CUDA GPU training")
    return torch.device("cpu")


def _diagnose_gpu():
    """Print helpful GPU diagnostic info when CUDA isn't available."""
    try:
        import ctypes, struct
        hip_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "ZLUDA", "amdhip64.dll"),
            "amdhip64.dll",
        ]
        hip = None
        for p in hip_paths:
            try:
                hip = ctypes.CDLL(p)
                break
            except OSError:
                continue
        if hip and hip.hipInit(0) == 0:
            count = ctypes.c_int(0)
            hip.hipGetDeviceCount(ctypes.byref(count))
            for i in range(count.value):
                buf = ctypes.create_string_buffer(4096)
                hip.hipGetDeviceProperties(buf, i)
                name = buf.raw[:256].split(b'\x00')[0].decode()
                vram = struct.unpack_from('Q', buf.raw, 256)[0]
                print(f"[Train]   HIP device {i}: {name} ({vram/1024**3:.0f} GB)")
            if count.value > 0:
                print("[Train]   HIP runtime works but ZLUDA cuInit failed.")
                print("[Train]   ZLUDA needs gfx1030/gfx1031 (RDNA 2 dGPU).")
                print("[Train]   If your dGPU isn't listed above, try ROCm 6.x HIP SDK.")
    except Exception:
        pass


def _check_dml_backward() -> bool:
    """Realistic smoke test: does DML handle GroupNorm + Conv2d backward?
    Note: This may pass but full UNet backward can still fail on DML."""
    try:
        import torch_directml
        dev = torch_directml.device()
        model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 3, padding=1),
            torch.nn.GroupNorm(8, 32),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 4, 3, padding=1),
        ).to(dev)
        x = torch.randn(1, 4, 8, 8, device=dev)
        loss = model(x).sum()
        loss.backward()
        del model, x, loss
        return True
    except Exception:
        return False


def _lr_lambda(step: int, total: int, warmup: int, schedule: str):
    """Returns multiplier for LambdaLR."""
    if step < warmup:
        return max(1e-6, step / max(warmup, 1))
    progress = (step - warmup) / max(total - warmup, 1)
    if schedule == "cosine":
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    if schedule == "linear":
        return max(0.0, 1.0 - progress)
    return 1.0  # constant


# ═════════════════════════════════════════════════════════════════════════════
# Kohya-format save
# ═════════════════════════════════════════════════════════════════════════════

def _peft_state_to_kohya(
    peft_sd: dict[str, torch.Tensor],
    prefix: str,          # "lora_unet" | "lora_te1" | "lora_te2"
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Convert a PEFT state-dict to kohya naming.

    PEFT key:  ``base_model.model.{path}.lora_A.default.weight``
    Kohya key: ``lora_unet_{path_underscored}.lora_down.weight``
    """
    out: dict[str, torch.Tensor] = {}
    seen_layers: set[str] = set()

    for key, val in peft_sd.items():
        clean = key.replace("base_model.model.", "").replace(".default", "")
        # e.g. "down_blocks.0.attentions.0...to_q.lora_A.weight"
        parts = clean.rsplit(".", 2)       # [layer_path, "lora_A", "weight"]
        if len(parts) < 3:
            continue
        layer_path, direction, _ = parts

        # Convert direction
        if direction == "lora_A":
            kohya_dir = "lora_down"
        elif direction == "lora_B":
            kohya_dir = "lora_up"
        else:
            continue

        kohya_layer = f"{prefix}_" + layer_path.replace(".", "_")
        out[f"{kohya_layer}.{kohya_dir}.weight"] = val.contiguous().cpu()

        # Add alpha tensor (once per layer)
        if kohya_layer not in seen_layers:
            seen_layers.add(kohya_layer)
            out[f"{kohya_layer}.alpha"] = torch.tensor(alpha, dtype=torch.float32)

    return out


def _unique_path(path: Path) -> Path:
    """path, or path with _2/_3/… appended to the stem if it already exists."""
    if not path.exists():
        return path
    n = 2
    while (candidate := path.with_name(f"{path.stem}_{n}{path.suffix}")).exists():
        n += 1
    return candidate


def safe_output_name(name: str) -> str:
    """File-system-safe LoRA name (Windows forbids <>:"/\|?* and trailing dots/spaces)."""
    import re
    n = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (name or "").strip()).strip(". ")
    if n.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                                   *(f"LPT{i}" for i in range(1, 10))}:
        n = "_" + n                     # reserved device names (CON, NUL…) can't be files
    return n[:120] or "my_lora"


def save_lora(
    unet,
    text_encoders: list | None,
    output_path: str | Path,
    cfg: TrainingConfig,
):
    """Save trained LoRA weights as kohya-compatible safetensors."""
    from peft import get_peft_model_state_dict

    state = {}

    # UNet
    unet_sd = get_peft_model_state_dict(unet)
    state.update(_peft_state_to_kohya(unet_sd, "lora_unet", cfg.lora_alpha))

    # Text encoder(s)
    if text_encoders:
        prefixes = ["lora_te1"] if len(text_encoders) == 1 else ["lora_te1", "lora_te2"]
        for te, pfx in zip(text_encoders, prefixes):
            try:
                te_sd = get_peft_model_state_dict(te)
                state.update(_peft_state_to_kohya(te_sd, pfx, cfg.lora_alpha))
            except Exception as e:
                print(f"[Train] ⚠ WARNING: Failed to save {pfx} weights: {e}")

    metadata = {
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(cfg.lora_rank),
        "ss_network_alpha": str(cfg.lora_alpha),
        "ss_base_model": Path(cfg.base_model).stem,
        "ss_training_comment": f"Trained by ImageGenApp | trigger: {cfg.trigger_word}",
        "ss_output_name": cfg.output_name,
        "modelspec.architecture": "stable-diffusion-xl-v1-base" if cfg.model_type == "sdxl" else "stable-diffusion-v1",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output_path), metadata=metadata)
    n = len(state)
    print(f"[Train] Saved LoRA → {output_path}  ({n} tensors)")


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class _TrainItem:
    __slots__ = ("image_path", "caption", "bucket_w", "bucket_h")

    def __init__(self, image_path: str, caption: str, bw: int, bh: int):
        self.image_path = image_path
        self.caption = caption
        self.bucket_w = bw
        self.bucket_h = bh


def _build_items(cfg: TrainingConfig) -> list[_TrainItem]:
    """Scan dataset_dir and build training items with bucket assignments."""
    images = scan_images(cfg.dataset_dir)
    if not images:
        raise ValueError(f"No images found in {cfg.dataset_dir}")

    if cfg.use_bucketing:
        result = compute_bucket_assignments(
            images, base_res=cfg.resolution, min_dim=192, step=64)
        items = []
        for a in result["assignments"]:
            items.append(_TrainItem(a["path"], a["caption"],
                                    a["bucket_w"], a["bucket_h"]))
        # Print bucket distribution
        for bkt, cnt in sorted(result["stats"].items()):
            print(f"  Bucket {bkt}: {cnt} image(s)")
        return items
    else:
        r = cfg.resolution
        return [_TrainItem(i["path"], i["caption"], r, r) for i in images]


# ═════════════════════════════════════════════════════════════════════════════
# Latent & text-embedding caching
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _cache_latents_for_items(
    items: list[_TrainItem],
    vae,
    flip_augment: bool = False,
    callback: Callable | None = None,
) -> list[torch.Tensor]:
    """Encode every image with the VAE once.  Returns list of latent tensors."""
    from torchvision import transforms as T

    vae.eval()
    was_half = (vae.dtype == torch.float16)
    if was_half:
        vae.float()

    scaling = float(vae.config.scaling_factor)
    latents: list[torch.Tensor] = []

    for i, item in enumerate(items):
        if callback:
            callback(i, len(items), f"Caching latent {i+1}/{len(items)}")

        img = resize_and_crop(item.image_path, item.bucket_w, item.bucket_h)
        if flip_augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        t = T.ToTensor()(img).unsqueeze(0)         # [1,3,H,W] in [0,1]
        t = t * 2.0 - 1.0                          # normalise to [-1,1]
        lat = vae.encode(t.to(vae.device)).latent_dist.mode() * scaling
        latents.append(lat.squeeze(0).cpu())        # [4,h,w]

    if was_half:
        vae.half()

    return latents


@torch.no_grad()
def _cache_text_embeds(
    items: list[_TrainItem],
    tokenizer, text_encoder,
    tokenizer_2=None, text_encoder_2=None,
    is_sdxl: bool = False,
    trigger_word: str = "",
    callback: Callable | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Pre-compute text embeddings for every caption."""

    text_encoder.eval()
    if text_encoder_2 is not None:
        text_encoder_2.eval()

    embeds: list[dict[str, torch.Tensor]] = []
    max_len = tokenizer.model_max_length

    for i, item in enumerate(items):
        if callback:
            callback(i, len(items), f"Encoding caption {i+1}/{len(items)}")

        cap = item.caption
        # Prepend trigger word if not already present
        if trigger_word and not cap.lower().startswith(trigger_word.lower()):
            cap = f"{trigger_word}, {cap}" if cap else trigger_word
        tok = tokenizer(cap, padding="max_length", truncation=True,
                        max_length=max_len, return_tensors="pt")
        ids = tok.input_ids
        enc_out = text_encoder(ids.to(text_encoder.device), output_hidden_states=True)

        if is_sdxl and text_encoder_2 is not None and tokenizer_2 is not None:
            tok2 = tokenizer_2(cap, padding="max_length", truncation=True,
                               max_length=tokenizer_2.model_max_length,
                               return_tensors="pt")
            enc2_out = text_encoder_2(
                tok2.input_ids.to(text_encoder_2.device),
                output_hidden_states=True)

            # SDXL uses penultimate hidden states concatenated
            hidden1 = enc_out.hidden_states[-2].squeeze(0).cpu()
            hidden2 = enc2_out.hidden_states[-2].squeeze(0).cpu()
            pooled = enc2_out[0].squeeze(0).cpu()   # pooled from TE2

            embeds.append({
                "encoder_hidden_states": torch.cat([hidden1, hidden2], dim=-1),
                "pooled": pooled,
                "original_size": (item.bucket_h, item.bucket_w),
            })
        else:
            embeds.append({
                "encoder_hidden_states": enc_out[0].squeeze(0).cpu(),
            })

    return embeds


# ═════════════════════════════════════════════════════════════════════════════
# Main trainer
# ═════════════════════════════════════════════════════════════════════════════

class LoRATrainer:
    """Train a LoRA adapter and save as safetensors."""

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self._abort = False

        # Populated by prepare()
        self.pipe: Any = None
        self.unet: Any = None
        self.vae: Any = None
        self.text_encoder: Any = None
        self.text_encoder_2: Any = None
        self.tokenizer: Any = None
        self.tokenizer_2: Any = None
        self.noise_scheduler: Any = None
        self.items: list[_TrainItem] = []
        self.latent_cache: list[torch.Tensor] = []
        self.embed_cache: list[dict[str, torch.Tensor]] = []
        self._trained_te_list: list = []

    # ── public API ─────────────────────────────────────────────────────────

    def abort(self):
        self._abort = True

    def prepare(self, callback: Callable | None = None) -> str:
        """Load base model, inject LoRA, cache latents + embeddings."""
        self._abort = False
        try:
            self._load_base_model(callback)
            self._build_items(callback)
            self._inject_lora()
            self._cache(callback)
            n_params = sum(p.numel() for p in self.unet.parameters()
                          if p.requires_grad)
            te_params = 0
            for te in self._trained_te_list:
                te_params += sum(p.numel() for p in te.parameters()
                                if p.requires_grad)
            return (f"✅ Ready — {len(self.items)} images, "
                    f"{n_params/1e6:.1f}M UNet + {te_params/1e6:.1f}M TE trainable params")
        except Exception as e:
            self.cleanup()
            return f"❌ Prepare failed: {e}"

    def train(self, callback: Callable | None = None) -> str:
        """Main training loop.  Returns status message."""
        self._abort = False
        if not self.items:
            return "❌ Call prepare() first."

        cfg = self.cfg
        device = _get_device(cfg)
        is_sdxl = cfg.model_type == "sdxl"
        use_cached_embeds = bool(self.embed_cache)
        use_cached_latents = bool(self.latent_cache)

        # Move UNet to training device
        self.unet.to(device)
        self.unet.train()
        for te in self._trained_te_list:
            te.to(device)
            te.train()

        if cfg.gradient_checkpointing:
            try:
                self.unet.enable_gradient_checkpointing()
                for te in self._trained_te_list:
                    if hasattr(te, "gradient_checkpointing_enable"):
                        te.gradient_checkpointing_enable()
            except Exception as e:
                print(f"[Train] ⚠ Gradient checkpointing failed ({e}), continuing without it")

        # Optimizer — separate param groups for UNet and TE
        param_groups = []
        if cfg.train_unet:
            param_groups.append({
                "params": [p for p in self.unet.parameters() if p.requires_grad],
                "lr": cfg.learning_rate,
            })
        for te in self._trained_te_list:
            param_groups.append({
                "params": [p for p in te.parameters() if p.requires_grad],
                "lr": cfg.te_learning_rate,
            })

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

        # LR scheduler
        total_steps = len(self.items) * cfg.epochs // cfg.gradient_accumulation
        warmup = int(total_steps * cfg.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda s: _lr_lambda(s, total_steps, warmup, cfg.lr_scheduler),
        )

        # Seed
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # For on-the-fly VAE encoding
        vae_scaling = None
        if not use_cached_latents and self.vae is not None:
            vae_scaling = float(self.vae.config.scaling_factor)
            self.vae.eval()

        # ── Training loop ──────────────────────────────────────────────────
        global_step = 0
        forward_count = 0
        log_loss = 0.0
        best_loss = float("inf")
        t_start = time.time()
        prediction_type = self.noise_scheduler.config.prediction_type
        num_timesteps = self.noise_scheduler.config.num_train_timesteps

        print(f"\n[Train] Starting: {cfg.epochs} epochs × {len(self.items)} images "
              f"= {total_steps} steps  (device={device})")
        print(f"[Train] Prediction type: {prediction_type}")
        if not use_cached_embeds:
            print("[Train] TE forward: LIVE (training text encoder)")

        for epoch in range(cfg.epochs):
            if self._abort:
                break

            indices = list(range(len(self.items)))
            random.shuffle(indices)
            epoch_loss = 0.0

            for idx_in_epoch, data_idx in enumerate(indices):
                if self._abort:
                    break

                # --- Latents ---
                if use_cached_latents:
                    latents = self.latent_cache[data_idx].unsqueeze(0).to(device)
                else:
                    # On-the-fly VAE encode
                    from torchvision import transforms as T
                    item = self.items[data_idx]
                    img = resize_and_crop(item.image_path, item.bucket_w, item.bucket_h)
                    if cfg.flip_augment and random.random() > 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    t_img = T.ToTensor()(img).unsqueeze(0) * 2.0 - 1.0
                    with torch.no_grad():
                        latents = self.vae.encode(
                            t_img.to(self.vae.device)
                        ).latent_dist.mode() * vae_scaling
                    latents = latents.to(device)

                # --- Text embeddings ---
                if use_cached_embeds:
                    embed_dict = self.embed_cache[data_idx]
                    enc_hidden = embed_dict["encoder_hidden_states"].unsqueeze(0).to(device)
                else:
                    # Live TE forward (so TE LoRA gets gradients)
                    embed_dict, enc_hidden = self._forward_te(data_idx, device, is_sdxl)

                # Noise + timestep
                noise = torch.randn_like(latents)
                t = torch.randint(0, num_timesteps, (1,), device=device).long()
                noisy = self.noise_scheduler.add_noise(latents, noise, t)

                # UNet forward
                if is_sdxl:
                    pooled = embed_dict["pooled"].unsqueeze(0).to(device)
                    orig_h, orig_w = embed_dict["original_size"]
                    add_time_ids = torch.tensor(
                        [[orig_h, orig_w, 0, 0, orig_h, orig_w]],
                        dtype=latents.dtype, device=device)
                    added_cond = {
                        "text_embeds": pooled,
                        "time_ids": add_time_ids,
                    }
                    pred = self.unet(noisy, t, enc_hidden,
                                     added_cond_kwargs=added_cond).sample
                else:
                    pred = self.unet(noisy, t, enc_hidden).sample

                # Target
                if prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, t)
                else:
                    target = noise

                loss = F.mse_loss(pred.float(), target.float())
                loss = loss / cfg.gradient_accumulation
                loss.backward()

                step_loss = loss.item() * cfg.gradient_accumulation
                epoch_loss += step_loss
                log_loss += step_loss
                forward_count += 1

                # Optimizer step
                if (idx_in_epoch + 1) % cfg.gradient_accumulation == 0:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for g in param_groups for p in g["params"]],
                            cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if callback:
                        pct = global_step / max(total_steps, 1)
                        elapsed = time.time() - t_start
                        eta = elapsed / max(global_step, 1) * (total_steps - global_step)
                        avg = log_loss / max(forward_count, 1)
                        callback(pct,
                                 f"Epoch {epoch+1}/{cfg.epochs} | "
                                 f"Step {global_step}/{total_steps} | "
                                 f"Loss {step_loss:.4f} (avg {avg:.4f}) | "
                                 f"ETA {eta/60:.0f}m")

            # Flush any trailing accumulated gradients at epoch boundary
            if len(indices) % cfg.gradient_accumulation != 0:
                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in param_groups for p in g["params"]],
                        cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # End of epoch
            avg_epoch = epoch_loss / max(len(indices), 1)
            print(f"[Train] Epoch {epoch+1}/{cfg.epochs} — "
                  f"avg loss {avg_epoch:.4f} — "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

            if avg_epoch < best_loss:
                best_loss = avg_epoch

            # Periodic save
            if (cfg.save_every_n_epochs > 0
                    and (epoch + 1) % cfg.save_every_n_epochs == 0
                    and epoch + 1 < cfg.epochs):
                ep_path = _unique_path(Path(cfg.output_dir) / f"{cfg.output_name}_ep{epoch+1}.safetensors")
                save_lora(self.unet, self._trained_te_list, ep_path, cfg)

        # ── Cleanup ────────────────────────────────────────────────────────
        self.unet.cpu()
        for te in self._trained_te_list:
            te.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        if self._abort:
            return f"⏹ Aborted at step {global_step}/{total_steps} ({elapsed/60:.1f}m)"

        # Final save
        # Never overwrite an existing LoRA (the default name is "my_lora"): add _2, _3 …
        out_path = _unique_path(Path(cfg.output_dir) / f"{cfg.output_name}.safetensors")
        save_lora(self.unet, self._trained_te_list, out_path, cfg)
        return (f"✅ Training complete — {global_step} steps in {elapsed/60:.1f}m — "
                f"best loss {best_loss:.4f} — saved → {out_path.name}")

    def _forward_te(self, data_idx: int, device: torch.device, is_sdxl: bool):
        """Run text encoder(s) forward for a single item (live, with gradients)."""
        item = self.items[data_idx]
        cap = item.caption
        trigger = self.cfg.trigger_word
        if trigger and not cap.lower().startswith(trigger.lower()):
            cap = f"{trigger}, {cap}" if cap else trigger

        max_len = self.tokenizer.model_max_length
        tok = self.tokenizer(cap, padding="max_length", truncation=True,
                             max_length=max_len, return_tensors="pt")
        ids = tok.input_ids.to(device)
        enc_out = self.text_encoder(ids, output_hidden_states=True)

        if is_sdxl and self.text_encoder_2 is not None and self.tokenizer_2 is not None:
            tok2 = self.tokenizer_2(cap, padding="max_length", truncation=True,
                                    max_length=self.tokenizer_2.model_max_length,
                                    return_tensors="pt")
            enc2_out = self.text_encoder_2(
                tok2.input_ids.to(device), output_hidden_states=True)
            hidden1 = enc_out.hidden_states[-2].squeeze(0)
            hidden2 = enc2_out.hidden_states[-2].squeeze(0)
            pooled = enc2_out[0].squeeze(0)
            enc_hidden = torch.cat([hidden1, hidden2], dim=-1).unsqueeze(0)
            embed_dict = {
                "encoder_hidden_states": enc_hidden.squeeze(0),
                "pooled": pooled,
                "original_size": (item.bucket_h, item.bucket_w),
            }
            return embed_dict, enc_hidden
        else:
            enc_hidden = enc_out[0]
            embed_dict = {"encoder_hidden_states": enc_hidden.squeeze(0)}
            return embed_dict, enc_hidden

    def cleanup(self):
        """Release all model memory."""
        for attr in ("pipe", "unet", "vae", "text_encoder", "text_encoder_2",
                      "tokenizer", "tokenizer_2", "noise_scheduler"):
            setattr(self, attr, None)
        self.items = []
        self.latent_cache.clear()
        self.embed_cache.clear()
        self._trained_te_list.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── internals ──────────────────────────────────────────────────────────

    def _load_base_model(self, callback: Callable | None):
        cfg = self.cfg
        path = Path(cfg.base_model)

        if callback:
            callback(0, 5, "Loading base model …")

        # Auto-detect model type from filename / sidecar
        from backend.sd_pipeline import _is_sdxl
        is_xl = _is_sdxl(str(path))
        cfg.model_type = "sdxl" if is_xl else "sd15"
        cfg.resolution = 1024 if is_xl else 512

        if is_xl:
            from diffusers import StableDiffusionXLPipeline
            PipeCls = StableDiffusionXLPipeline
        else:
            from diffusers import StableDiffusionPipeline
            PipeCls = StableDiffusionPipeline

        load_kw: dict[str, Any] = {"torch_dtype": torch.float32}
        if path.is_file():
            self.pipe = PipeCls.from_single_file(str(path), **load_kw)
        else:
            self.pipe = PipeCls.from_pretrained(str(path), **load_kw)

        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder_2 = getattr(self.pipe, "text_encoder_2", None)
        self.tokenizer_2 = getattr(self.pipe, "tokenizer_2", None)

        # Noise scheduler for training (DDPM, not the inference scheduler)
        from diffusers import DDPMScheduler
        self.noise_scheduler = DDPMScheduler.from_config(
            self.pipe.scheduler.config)

        # Freeze everything
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)

        print(f"[Train] Loaded {path.name} as {cfg.model_type.upper()} "
              f"(UNet {sum(p.numel() for p in self.unet.parameters())/1e6:.0f}M params)")

    def _build_items(self, callback: Callable | None):
        if callback:
            callback(1, 5, "Scanning dataset …")
        self.items = _build_items(self.cfg)
        print(f"[Train] {len(self.items)} training images")

    def _inject_lora(self):
        from peft import LoraConfig, get_peft_model

        cfg = self.cfg
        targets = (_UNET_FULL_TARGETS if cfg.target_coverage == "full"
                   else _UNET_ATTN_TARGETS)

        lora_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=targets,
            lora_dropout=0.0,
            bias="none",
        )
        self.unet = get_peft_model(self.unet, lora_cfg)
        self.unet.print_trainable_parameters()

        self._trained_te_list = []
        if cfg.train_te:
            te_cfg = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                target_modules=_TE_TARGETS,
                lora_dropout=0.0,
                bias="none",
            )
            self.text_encoder = get_peft_model(self.text_encoder, te_cfg)
            self._trained_te_list.append(self.text_encoder)
            if self.text_encoder_2 is not None:
                self.text_encoder_2 = get_peft_model(self.text_encoder_2, te_cfg)
                self._trained_te_list.append(self.text_encoder_2)

    def _cache(self, callback: Callable | None):
        cfg = self.cfg

        if callback:
            callback(2, 5, "Caching latents (VAE encode) …")

        if cfg.cache_latents:
            self.latent_cache = _cache_latents_for_items(
                self.items, self.vae, cfg.flip_augment,
                callback=callback)
        else:
            # Placeholder — on-the-fly encoding in train loop
            self.latent_cache = []

        if callback:
            callback(3, 5, "Caching text embeddings …")

        # Only cache text embeddings when NOT training the text encoder.
        # When train_te=True, TE must run forward in the training loop
        # so its LoRA weights receive gradients.
        if cfg.train_te and self._trained_te_list:
            self.embed_cache = []  # will run TE forward in train loop
            print("[Train] TE training enabled — text embeddings will be computed live")
        else:
            self.embed_cache = _cache_text_embeds(
                self.items,
                self.tokenizer, self.text_encoder,
                self.tokenizer_2, self.text_encoder_2,
                is_sdxl=(cfg.model_type == "sdxl"),
                trigger_word=cfg.trigger_word,
                callback=callback,
            )

        # Free VAE — not needed during training when latents are cached
        if cfg.cache_latents:
            self.vae.cpu()
            del self.vae
            self.vae = None
            gc.collect()

        if callback:
            callback(4, 5, "Ready!")
