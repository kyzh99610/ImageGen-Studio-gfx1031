"""
backend/smartsplit_pipeline.py
Heterogeneous pipeline parallelism — splits Stable Diffusion stages across
the available compute units on the same system.

Inspired by AMD SmartAccess Video (which splits video codec work across iGPU
and dGPU media engines), this does the same for AI inference stages:

  Stage 1 — Text Encoder (CLIP)  → XDNA NPU  | iGPU (DirectML) | CPU
  Stage 2 — UNet denoising       → dGPU (ZLUDA/CUDA)  ← always here
  Stage 3 — VAE Decode           → iGPU (DirectML ONNX) | CPU

Benefits on the 8700G+6800XT workstation (128 GB DDR5):
  • Frees ~1.5 GB dGPU VRAM (text encoder off-loaded)
  • Frees ~2 GB  dGPU VRAM (VAE decoder off-loaded)
  • Total: up to 3-4 GB more dGPU headroom → higher resolution / larger batches
  • NPU + iGPU run in parallel with the UNet when their stages overlap

Requirements:
  • onnxruntime             (always — for ONNX VAE)
  • onnxruntime-directml    (iGPU VAE decode via DirectML)
  • onnxruntime-vitisai     (NPU text encoder — optional)
  • AMD Ryzen AI driver     (NPU only)
  • diffusers, torch        (UNet on dGPU via ZLUDA)
"""

from __future__ import annotations

import copy
import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from config import APP_DIR, get_torch_dtype

ONNX_CACHE_DIR = APP_DIR / "onnx_cache"
ONNX_CACHE_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Capability detection
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SmartSplitCapability:
    """Describes what split is possible on this machine."""
    possible:          bool  = False

    # Text encoder options (ranked best→worst)
    npu_available:     bool  = False   # VitisAI EP detected
    igpu_te_available: bool  = False   # DirectML available for text enc
    cpu_te_available:  bool  = True    # always

    # VAE options
    igpu_vae_available: bool = False   # DirectML ONNX VAE
    cpu_vae_available:  bool = True    # always

    # Primary compute
    dgpu_available:    bool  = False   # ZLUDA / CUDA dGPU

    igpu_name:         str   = ""
    dgpu_name:         str   = ""
    npu_name:          str   = ""
    dgpu_dml_idx:      int   = -1      # DirectML device index for dGPU (-1 = use CUDA)
    igpu_dml_idx:      int   = -1      # DirectML device index for iGPU
    reason:            str   = ""      # why not possible (if possible=False)


def detect_smartsplit_capability() -> SmartSplitCapability:
    """
    Probe the system and return what SmartSplit modes are achievable.
    Only marks possible=True when there is a meaningful split worth doing
    (i.e. both a dGPU and at least one secondary compute unit are present).
    """
    cap = SmartSplitCapability()

    # ── dGPU via ZLUDA/CUDA ────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            cap.dgpu_available = True
            cap.dgpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    if not cap.dgpu_available:
        # ── Try DirectML multi-GPU split (works without ZLUDA/CUDA) ──────────
        try:
            import torch_directml as _tdml
            _n = _tdml.device_count()
            if _n >= 2:
                _IGPU_KW = ("780m", "760m", "880m", "680m", "vega 8", "vega 11",
                            "610m", "618m", "radeon(tm) graphics", "780", "760")
                _dml_names = [(i, _tdml.device_name(i)) for i in range(_n)]
                igpu_i = next(
                    (i for i, nm in _dml_names
                     if any(k in nm.lower() for k in _IGPU_KW)), -1
                )
                dgpu_i = next((i for i, nm in _dml_names if i != igpu_i), -1)
                if igpu_i >= 0 and dgpu_i >= 0:
                    cap.dgpu_available     = True
                    cap.dgpu_dml_idx       = dgpu_i
                    cap.igpu_dml_idx       = igpu_i
                    cap.dgpu_name          = _dml_names[dgpu_i][1]
                    cap.igpu_name          = _dml_names[igpu_i][1]
                    cap.igpu_vae_available = True
                    cap.igpu_te_available  = True
        except Exception:
            pass

        if not cap.dgpu_available:
            # No dGPU via any path
            try:
                from backend.hardware_detector import get_profile
                p = get_profile()
                dgpu = next((g for g in p.gpus if g.is_dgpu), None)
                if dgpu:
                    cap.reason = (
                        f"dGPU detected ({dgpu.name}) but CUDA/ZLUDA is not available. "
                        "SmartSplit requires CUDA or ZLUDA for UNet inference on the dGPU. "
                        "Your system uses DirectML which runs the full pipeline on one device. "
                        "This does not affect normal image generation."
                    )
                else:
                    cap.reason = "No dGPU detected — SmartSplit requires a dedicated GPU."
            except Exception:
                cap.reason = "No dGPU detected via CUDA/ZLUDA — SmartSplit requires a primary dGPU."
            return cap

    # ── iGPU via DirectML ─────────────────────────────────────────────────
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            cap.igpu_vae_available = True
            cap.igpu_te_available  = True
            # Try to get iGPU name from hardware_detector
            try:
                from backend.hardware_detector import get_profile
                p = get_profile()
                igpu = next((g for g in p.gpus if g.is_igpu), None)
                cap.igpu_name = igpu.name if igpu else "AMD iGPU (DirectML)"
            except Exception:
                cap.igpu_name = "AMD iGPU (DirectML)"
    except ImportError:
        pass

    # ── XDNA NPU via VitisAI ─────────────────────────────────────────────
    try:
        import onnxruntime as ort
        if "VitisAIExecutionProvider" in ort.get_available_providers():
            cap.npu_available = True
            cap.npu_name = "AMD XDNA NPU (VitisAI EP active)"
    except Exception:
        pass

    # If VitisAI EP isn't in our onnxruntime, try the subprocess bridge
    # (SDK conda env has Python 3.12 with VitisAI EP)
    if not cap.npu_available:
        try:
            from backend.npu_bridge import check_npu_bridge_available
            bridge_ok, bridge_msg = check_npu_bridge_available()
            if bridge_ok:
                cap.npu_available = True
                cap.npu_name = "AMD XDNA NPU (via bridge → Ryzen AI conda env)"
            else:
                # Fall back to hardware_detector for info display
                from backend.hardware_detector import get_profile
                npu_info = get_profile().npu      # the profile already ran _detect_npu (a PowerShell query, ~1 s)
                if npu_info.available:
                    cap.npu_name = f"{npu_info.name} (bridge unavailable: {bridge_msg})"
                elif npu_info.name != "Not detected":
                    cap.npu_name = f"{npu_info.name} (not ready)"
        except Exception:
            pass

    # Not worthwhile if no secondary device exists
    has_secondary = cap.igpu_vae_available or cap.npu_available
    if not has_secondary:
        cap.reason = (
            "No secondary compute unit detected (iGPU DirectML or XDNA NPU).\n"
            "Install onnxruntime-directml for iGPU VAE offload, or\n"
            "onnxruntime-vitisai for NPU text encoder offload."
        )
        return cap

    cap.possible = True
    return cap


# ══════════════════════════════════════════════════════════════════════════════
# Stage assignment config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SmartSplitConfig:
    """User-facing configuration — which device handles which stage."""
    text_encoder_device: str = "cpu"          # "npu" | "igpu_directml" | "cpu"
    unet_device:         str = "cuda"         # "cuda" (ZLUDA/CUDA) | "dgpu_directml"
    vae_device:          str = "cpu"          # "igpu_directml" | "cpu"
    enabled:             bool = False
    dgpu_dml_idx:        int  = -1            # DirectML device index for dGPU
    igpu_dml_idx:        int  = -1            # DirectML device index for iGPU

    @classmethod
    def auto(cls, cap: SmartSplitCapability) -> "SmartSplitConfig":
        """Build an optimal config from detected capabilities."""
        if not cap.possible:
            return cls(enabled=False)
        if cap.dgpu_dml_idx >= 0:
            # DirectML multi-GPU split (no ZLUDA needed)
            # DML path always runs TE on CPU (via diffusers encode_prompt),
            # NPU bridge requires ONNX export which is only in the CUDA path.
            return cls(
                enabled=True,
                text_encoder_device="cpu",
                unet_device="dgpu_directml",
                vae_device="cpu",
                dgpu_dml_idx=cap.dgpu_dml_idx,
                igpu_dml_idx=cap.igpu_dml_idx,
            )
        te = "npu" if cap.npu_available else ("igpu_directml" if cap.igpu_te_available else "cpu")
        vae = "igpu_directml" if cap.igpu_vae_available else "cpu"
        return cls(
            text_encoder_device=te,
            unet_device="cuda",
            vae_device=vae,
            enabled=True,
        )

    def describe(self) -> str:
        if not self.enabled:
            return "SmartSplit disabled — all stages run on primary GPU."
        if self.unet_device == "dgpu_directml":
            return (
                f"SmartSplit enabled (DirectML split):\n"
                f"  Text Encoder → {self.text_encoder_device.upper()}  (frees ~1 GB dGPU VRAM)\n"
                f"  UNet         → dGPU DirectML (device {self.dgpu_dml_idx})\n"
                f"  VAE Decode   → {self.vae_device.upper()}  (frees ~2 GB dGPU VRAM)\n"
            )
        return (
            f"SmartSplit enabled:\n"
            f"  Text Encoder → {self.text_encoder_device.upper()}\n"
            f"  UNet         → {self.unet_device.upper()} (dGPU)\n"
            f"  VAE Decode   → {self.vae_device.upper()}\n"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ONNX export helpers
# ══════════════════════════════════════════════════════════════════════════════

def _export_text_encoder_to_onnx(pipe, model_stem: str) -> Path:
    """Export the CLIP text encoder to ONNX (cached by model name)."""
    onnx_path = ONNX_CACHE_DIR / f"{model_stem}_text_encoder.onnx"
    if onnx_path.exists():
        return onnx_path

    print(f"[SmartSplit] Exporting text encoder to ONNX → {onnx_path}")
    import torch
    te = pipe.text_encoder.eval()

    # ONNX export must run on CPU — temporarily move there
    original_device = next(te.parameters()).device
    te.cpu()

    tokenizer = pipe.tokenizer

    dummy = tokenizer(
        ["dummy"], padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"]

    torch.onnx.export(
        te,
        (input_ids,),
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={"input_ids": {0: "batch"}, "last_hidden_state": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    # Restore original device
    te.to(original_device)
    print("[SmartSplit] Text encoder ONNX export done.")
    return onnx_path


def _export_vae_decoder_to_onnx(pipe, model_stem: str) -> Path:
    """Export the VAE decoder to ONNX (cached by model name)."""
    onnx_path = ONNX_CACHE_DIR / f"{model_stem}_vae_decoder.onnx"
    if onnx_path.exists():
        return onnx_path

    print(f"[SmartSplit] Exporting VAE decoder to ONNX → {onnx_path}")
    import torch
    from diffusers.models.autoencoders.vae import Decoder

    vae = pipe.vae.eval()

    # ONNX export must run on CPU — temporarily move there
    original_device = next(vae.parameters()).device
    vae.cpu()

    # Wrap decode step (latents → image)
    class VAEDecodeWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents).sample

    wrapper = VAEDecodeWrapper(vae)
    latent_channels = vae.config.latent_channels
    dummy_latent = torch.randn(1, latent_channels, 64, 64)

    torch.onnx.export(
        wrapper,
        (dummy_latent,),
        str(onnx_path),
        input_names=["latents"],
        output_names=["images"],
        dynamic_axes={
            "latents": {0: "batch", 2: "height", 3: "width"},
            "images":  {0: "batch", 2: "img_height", 3: "img_width"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    # Restore original device
    vae.to(original_device)
    print("[SmartSplit] VAE decoder ONNX export done.")
    return onnx_path


def _ort_session(onnx_path: Path, ep: str):
    """Create an onnxruntime InferenceSession with the requested EP."""
    import onnxruntime as ort

    ep_map = {
        "npu":           "VitisAIExecutionProvider",
        "igpu_directml": "DmlExecutionProvider",
        "cpu":           "CPUExecutionProvider",
    }
    provider = ep_map.get(ep, "CPUExecutionProvider")
    available = ort.get_available_providers()

    if provider not in available:
        print(f"[SmartSplit] {provider} not available, falling back to CPU")
        provider = "CPUExecutionProvider"

    return ort.InferenceSession(str(onnx_path), providers=[provider])


# ══════════════════════════════════════════════════════════════════════════════
# NPU session proxy — wraps the subprocess bridge as an ort-like session
# ══════════════════════════════════════════════════════════════════════════════

class _NPUSessionProxy:
    """Quacks like an ort.InferenceSession but runs on NPU via subprocess bridge."""

    def __init__(self, bridge, onnx_path: str):
        self._bridge = bridge
        self._onnx_path = onnx_path

    def run(self, output_names, inputs: dict) -> list:
        outputs, elapsed, provider = self._bridge.run(self._onnx_path, inputs)
        return outputs


# ══════════════════════════════════════════════════════════════════════════════
# Main SmartSplit pipeline
# ══════════════════════════════════════════════════════════════════════════════

class SmartSplitPipeline:
    """
    Wraps a loaded diffusers pipeline and redistributes stages to secondary
    compute units (iGPU via DirectML, XDNA NPU via VitisAI).

    Typical 8700G workstation usage:
        Text encoder  → XDNA NPU  (frees ~1.5 GB dGPU VRAM)
        UNet          → RX 6800 XT (the heavy lifter, all steps)
        VAE decoder   → Radeon 780M iGPU via DirectML (frees ~2 GB dGPU VRAM)
    """

    def __init__(self, pipe, config: SmartSplitConfig, model_stem: str = "model"):
        self.pipe       = pipe
        self.cfg        = config
        self.stem       = model_stem

        self._te_session  = None   # ONNX session for text encoder
        self._vae_session = None   # ONNX session for VAE decoder
        self._npu_bridge  = None   # NPU subprocess bridge (VitisAI via conda env)
        self._dml_dgpu    = None   # DirectML device object for dGPU (non-CUDA path)
        self._dml_igpu    = None   # DirectML device object for iGPU

        self._setup()

    def _setup(self):
        """Export ONNX models and create sessions for offloaded stages."""
        if not self.cfg.enabled:
            return

        # ── DirectML split: UNet+VAE on dGPU DirectML, Text on CPU ──────────────
        if self.cfg.unet_device == "dgpu_directml":
            try:
                import torch_directml as _tdml
                self._dml_dgpu = _tdml.device(self.cfg.dgpu_dml_idx)
                self._dml_igpu = None
                te_dev   = next(self.pipe.text_encoder.parameters()).device
                vae_dev  = next(self.pipe.vae.parameters()).device
                unet_dev = next(self.pipe.unet.parameters()).device
                # Ensure attention slicing is 1-head to prevent OOM in custom loop
                self.pipe.enable_attention_slicing(1)
                print(f"[SmartSplit] UNet+VAE → {unet_dev} (dGPU DirectML)")
                print(f"[SmartSplit] Text     → CPU  (saves ~500 MB dGPU VRAM)")
                if "cpu" not in str(te_dev).lower():
                    print(f"[SmartSplit] ⚠ text_encoder is on {te_dev} — "
                          f"reload the model to apply SmartSplit VRAM savings.")
            except Exception as e:
                print(f"[SmartSplit] Setup failed: {e}")
                self._dml_dgpu = None
            return  # No ONNX export for DirectML path

        if self.cfg.text_encoder_device in ("npu", "igpu_directml"):
            try:
                onnx_path = _export_text_encoder_to_onnx(self.pipe, self.stem)
                if self.cfg.text_encoder_device == "npu":
                    # Try NPU bridge (subprocess → conda env VitisAI EP)
                    self._te_session = self._setup_npu_te(onnx_path)
                if self._te_session is None:
                    # Fall back to local ort session (iGPU DirectML or CPU)
                    fallback = self.cfg.text_encoder_device
                    self._te_session = _ort_session(onnx_path, fallback)
                print(f"[SmartSplit] Text encoder → {self.cfg.text_encoder_device}")
            except Exception as e:
                print(f"[SmartSplit] Text encoder offload failed ({e}), using dGPU")
                self._te_session = None

        if self.cfg.vae_device == "igpu_directml":
            try:
                onnx_path = _export_vae_decoder_to_onnx(self.pipe, self.stem)
                self._vae_session = _ort_session(onnx_path, "igpu_directml")
                print(f"[SmartSplit] VAE decoder  → igpu_directml")
            except Exception as e:
                print(f"[SmartSplit] VAE offload failed ({e}), using dGPU")
                self._vae_session = None

    # ── NPU bridge setup ─────────────────────────────────────────────────────
    def _setup_npu_te(self, onnx_path: Path):
        """Start NPU bridge and return a proxy that quacks like an ort session."""
        try:
            from backend.npu_bridge import get_npu_bridge
            bridge = get_npu_bridge()
            if not bridge.is_running:
                msg = bridge.start()
                print(f"[SmartSplit] {msg}")
            if bridge.is_running:
                self._npu_bridge = bridge
                self._npu_te_onnx = str(onnx_path)
                # Return a sentinel so _te_session is truthy
                return _NPUSessionProxy(bridge, str(onnx_path))
        except Exception as e:
            print(f"[SmartSplit] NPU bridge setup failed: {e}")
        return None

    # ── Text encoding ──────────────────────────────────────────────────────────
    def encode_prompt(self, prompt: str, negative_prompt: str, device: str):
        """
        Encode text using offloaded session (NPU/iGPU) or fall back to the
        standard diffusers encode_prompt on the dGPU.
        """
        import torch

        if self._dml_dgpu is not None:
            # DirectML split — text encoder is on CPU; encode there, move embeds to dGPU
            try:
                _r = self.pipe.encode_prompt(
                    prompt, device="cpu", num_images_per_prompt=1,
                    do_classifier_free_guidance=True, negative_prompt=negative_prompt,
                )
                pos, neg = _r[0], _r[1]  # SD 1.5 returns 2, SDXL returns 4
                return torch.cat([neg, pos]).to(self._dml_dgpu)
            except Exception as e:
                print(f"[SmartSplit] TE encode failed ({e}), using main device")

        if self._te_session is not None:
            tokenizer = self.pipe.tokenizer
            max_len   = tokenizer.model_max_length

            def _tokenize(text: str) -> np.ndarray:
                ids = tokenizer(
                    [text], padding="max_length", max_length=max_len,
                    truncation=True, return_tensors="np",
                )["input_ids"].astype(np.int64)
                return ids

            pos_ids = _tokenize(prompt)
            neg_ids = _tokenize(negative_prompt)

            # Run on NPU/iGPU
            pos_embeds = self._te_session.run(None, {"input_ids": pos_ids})[0]
            neg_embeds = self._te_session.run(None, {"input_ids": neg_ids})[0]

            # Convert back to torch tensors on dGPU
            pos_t = torch.from_numpy(pos_embeds).to(device, dtype=torch.float16)
            neg_t = torch.from_numpy(neg_embeds).to(device, dtype=torch.float16)
            return torch.cat([neg_t, pos_t])   # classifier-free guidance format
        else:
            # Standard diffusers encoding on main device or CPU
            te_dev = next(self.pipe.text_encoder.parameters()).device
            if self.cfg.text_encoder_device == "cpu":
                if "cpu" not in str(te_dev).lower():
                    self.pipe.text_encoder = self.pipe.text_encoder.to("cpu")
                _r = self.pipe.encode_prompt(
                    prompt, device="cpu", num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                pos, neg = _r[0].to(device), _r[1].to(device)
                return torch.cat([neg, pos])
            else:
                _r = self.pipe.encode_prompt(
                    prompt, device=device, num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                pos, neg = _r[0], _r[1]  # SD 1.5 returns 2, SDXL returns 4
                return torch.cat([neg, pos])

    # ── VAE decoding ──────────────────────────────────────────────────────────
    def decode_latents(self, latents) -> list[Image.Image]:
        """Decode latents using iGPU DirectML ONNX session or dGPU."""
        import torch

        if self._dml_dgpu is not None:
            # DirectML split — VAE is on DML, decode latents directly on DML.
            # Enable tiling to avoid OOM at high resolutions (1024×768 etc.)
            try:
                scale = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
                self.pipe.vae.enable_tiling()
                self.pipe.vae.enable_slicing()
                with torch.no_grad():
                    decoded = self.pipe.vae.decode(latents / scale).sample
                self.pipe.vae.disable_tiling()
                decoded = ((decoded / 2 + 0.5).clamp(0, 1) * 255).byte()
                return [
                    Image.fromarray(decoded[i].permute(1, 2, 0).cpu().numpy())
                    for i in range(decoded.shape[0])
                ]
            except Exception as e:
                print(f"[SmartSplit] VAE decode failed on GPU ({e}), falling back to CPU VAE decode (slow but safe)...")
                try:
                    scale = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
                    # Cast to float32: VAE was loaded in fp16 for DML, but CPU needs fp32
                    vae_cpu = copy.deepcopy(self.pipe.vae).to("cpu").float()
                    lat_cpu = latents.cpu().float()
                    vae_cpu.enable_tiling()
                    vae_cpu.enable_slicing()
                    with torch.no_grad():
                        decoded = vae_cpu.decode(lat_cpu / scale).sample
                    del vae_cpu, lat_cpu
                    gc.collect()
                    decoded = ((decoded / 2 + 0.5).clamp(0, 1) * 255).byte()
                    return [
                        Image.fromarray(decoded[i].permute(1, 2, 0).numpy())
                        for i in range(decoded.shape[0])
                    ]
                except Exception as e2:
                    print(f"[SmartSplit] CPU VAE fallback also failed: {e2}")
                    raise

        if self._vae_session is not None:
            # Transfer latents from dGPU to CPU numpy for iGPU ONNX session
            scale = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
            lat_np = (latents.cpu().float().numpy() / scale).astype(np.float32)
            out = self._vae_session.run(None, {"latents": lat_np})[0]  # (N,3,H,W)
            images = []
            for img_arr in out:
                arr = np.clip((img_arr.transpose(1, 2, 0) + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
                images.append(Image.fromarray(arr))
            return images
        else:
            # Standard VAE on dGPU
            scale = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
            with torch.no_grad():
                decoded = self.pipe.vae.decode(latents / scale).sample
            decoded = ((decoded / 2 + 0.5).clamp(0, 1) * 255).byte()
            return [
                Image.fromarray(decoded[i].permute(1, 2, 0).cpu().numpy())
                for i in range(decoded.shape[0])
            ]

    # ── Full generation ───────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        scheduler_fn,
        batch_size: int = 1,
        progress_callback: Callable | None = None,
    ) -> tuple[list[Image.Image], dict[str, float]]:
        """
        Run the full pipeline with stages split across compute units.
        Returns (images, timing_dict).
        """
        import torch

        timing: dict[str, float] = {}

        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())

        # ── DirectML path ──────────────────────────────────────────────────────
        # Text encoder is on CPU; UNet + VAE are on DirectML.
        # We pre-encode text on CPU, then hand off to the standard diffusers
        # pipeline which runs denoising + VAE decode entirely on the DML device.
        # This avoids any DML↔CPU tensor transfers mid-generation.
        if self._dml_dgpu is not None:
            # Stage 1: Text encoding on CPU using compel (supports >77 tokens)
            t0 = time.time()
            if progress_callback:
                progress_callback(0.03, "[Stage 1/3] Text encoding → CPU")
            try:
                from backend.sd_pipeline import _build_embeds
                _e = _build_embeds(self.pipe, prompt, negative_prompt, "cpu")
                if "prompt_embeds" in _e:
                    pos_embeds, neg_embeds = _e["prompt_embeds"], _e["negative_prompt_embeds"]
                    # Repeat for batch_size (compel returns single-item)
                    if batch_size > 1:
                        pos_embeds = pos_embeds.repeat(batch_size, 1, 1)
                        neg_embeds = neg_embeds.repeat(batch_size, 1, 1)
                else:
                    raise RuntimeError("compel unavailable")
            except Exception as e:
                print(f"[SmartSplit] Compel encoding failed ({e}), falling back to encode_prompt")
                _r = self.pipe.encode_prompt(
                    prompt, device="cpu", num_images_per_prompt=batch_size,
                    do_classifier_free_guidance=True, negative_prompt=negative_prompt,
                )
                pos_embeds, neg_embeds = _r[0], _r[1]
            timing["text_encode_s"] = time.time() - t0

            # Stage 2: UNet denoising on DML — custom loop avoids diffusers
            # internal dtype casts that break torch_directml (DMLTensor assert).
            # Sequential CFG: run uncond and cond passes separately to halve
            # peak VRAM per UNet call (critical for high-res + DML).
            t0 = time.time()
            dml_device = self._dml_dgpu
            dtype = next(self.pipe.unet.parameters()).dtype
            latent_h = height // 8
            latent_w = width  // 8
            channels = self.pipe.unet.config.in_channels

            # Randn on CPU (DML Generator unsupported), then move to DML
            latents = torch.randn(
                (batch_size, channels, latent_h, latent_w),
                generator=torch.Generator("cpu").manual_seed(seed),
                device="cpu", dtype=dtype,
            ).to(dml_device)

            scheduler = self.pipe.scheduler
            scheduler.set_timesteps(steps, device=dml_device)
            latents = latents * scheduler.init_noise_sigma

            # Pre-move embeds to DML (kept separate for sequential CFG)
            neg_dml = neg_embeds.to(dml_device)
            pos_dml = pos_embeds.to(dml_device)
            n_steps = len(scheduler.timesteps)

            if progress_callback:
                progress_callback(0.05, "[Stage 2/3] Starting UNet denoising → dGPU DirectML")

            for i, t in enumerate(scheduler.timesteps):
                print(f"[SmartSplit] step {i+1}/{n_steps}", end="\r", flush=True)
                if progress_callback:
                    progress_callback(
                        0.05 + 0.85 * (i / n_steps),
                        f"[Stage 2/3] UNet step {i+1}/{n_steps} → dGPU DirectML",
                    )
                t_dml = t.to(dml_device)
                lat_scaled = scheduler.scale_model_input(latents, t)
                # Two separate passes instead of doubled batch — halves peak VRAM
                with torch.no_grad():
                    noise_uncond = self.pipe.unet(
                        lat_scaled, t_dml,
                        encoder_hidden_states=neg_dml,
                        return_dict=False,
                    )[0]
                    noise_cond = self.pipe.unet(
                        lat_scaled, t_dml,
                        encoder_hidden_states=pos_dml,
                        return_dict=False,
                    )[0]
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            print()
            timing["unet_s"] = time.time() - t0

            # Stage 3: VAE decode — VAE is on DML, latents already on DML
            t0 = time.time()
            if progress_callback:
                progress_callback(0.92, "[Stage 3/3] VAE decode → dGPU DirectML")
            images = self.decode_latents(latents)
            timing["vae_decode_s"] = time.time() - t0
            timing["total_s"] = sum(timing.values())

            if progress_callback:
                progress_callback(1.0, "✅ Done!")

            return images, timing

        # ── ONNX / CUDA path (NPU/iGPU split) ─────────────────────────────────
        device = next(self.pipe.unet.parameters()).device
        dtype  = next(self.pipe.unet.parameters()).dtype
        gen_device = "cpu" if ("privateuseone" in str(device) or "directml" in str(device)) else str(device).split(":")[0]
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        # ── Stage 1: Text encoding ─────────────────────────────────────────
        t0 = time.time()
        if progress_callback:
            progress_callback(0.0, f"[Stage 1/3] Text encoding → {self.cfg.text_encoder_device.upper()}")
        prompt_embeds = self.encode_prompt(prompt, negative_prompt, str(device))
        # Expand for batch
        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(batch_size, dim=0)
        timing["text_encode_s"] = time.time() - t0

        # ── Stage 2: UNet denoising on dGPU ───────────────────────────────
        t0 = time.time()
        latent_h = height // 8
        latent_w = width  // 8
        channels = self.pipe.unet.config.in_channels
        latents = torch.randn(
            (batch_size, channels, latent_h, latent_w),
            generator=generator, device=device, dtype=dtype,
        )

        # Scale by initial noise sigma
        pipe_scheduler = self.pipe.scheduler
        pipe_scheduler.set_timesteps(steps, device=device)
        latents = latents * pipe_scheduler.init_noise_sigma

        n_steps = len(pipe_scheduler.timesteps)
        for i, t in enumerate(pipe_scheduler.timesteps):
            if progress_callback:
                progress_callback(
                    0.1 + 0.75 * (i / n_steps),
                    f"[Stage 2/3] UNet step {i+1}/{n_steps} → {self.cfg.unet_device.upper()}",
                )
            print(f"[SmartSplit] step {i+1}/{n_steps}", end="\r", flush=True)
            lat_input = torch.cat([latents] * 2)
            lat_input = pipe_scheduler.scale_model_input(lat_input, t)

            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    lat_input, t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            latents = pipe_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        timing["unet_s"] = time.time() - t0

        # ── Stage 3: VAE decode ───────────────────────────────────────────
        t0 = time.time()
        if progress_callback:
            progress_callback(0.9, f"[Stage 3/3] VAE decode → {self.cfg.vae_device.upper()}")
        images = self.decode_latents(latents)
        timing["vae_decode_s"] = time.time() - t0

        timing["total_s"] = sum(timing.values())
        return images, timing

    def stage_summary(self) -> str:
        """Return a one-line description of the active split for display."""
        if not self.cfg.enabled:
            return "SmartSplit: off"
        te  = self.cfg.text_encoder_device.upper()
        un  = self.cfg.unet_device.upper()
        vae = self.cfg.vae_device.upper()
        return f"SmartSplit: Text→{te}  UNet→{un}  VAE→{vae}"

    def timing_html(self, timing: dict[str, float]) -> str:
        te  = timing.get("text_encode_s", 0)
        un  = timing.get("unet_s", 0)
        vae = timing.get("vae_decode_s", 0)
        tot = timing.get("total_s", 0)
        return (
            f'<span style="font-size:13px;color:#a6adc8;">'
            f'Text({self.cfg.text_encoder_device.upper()}):{te:.2f}s  '
            f'UNet({self.cfg.unet_device.upper()}):{un:.1f}s  '
            f'VAE({self.cfg.vae_device.upper()}):{vae:.2f}s  '
            f'Total:{tot:.1f}s</span>'
        )
