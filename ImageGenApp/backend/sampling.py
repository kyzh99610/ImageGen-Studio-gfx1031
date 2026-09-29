"""
Sampling upgrades shared by the SD 1.5 and SDXL pipelines:

- Schedulers as A1111 names them. "Karras" really uses Karras sigmas now (before
  2026-09-28 "DPM++ 2M Karras" was plain DPM++ 2M — images saved with the old record
  format restore as "DPM++ 2M"), plus DPM++ 2M SDE and Align-Your-Steps (AYS).
- AYS: NVIDIA's optimised 10-step noise schedules (diffusers' AysSchedules), stretched
  log-linearly to any step count — about the quality of 25 Karras steps in 10–12.
- V-prediction checkpoints (NoobAI-XL v-pred, …): found from the `v_pred` / `ztsnr` keys
  a trainer writes into the checkpoint (or "vpred" in the file name) — the scheduler then
  predicts v, with zero-terminal-SNR betas and trailing timesteps; CFG rescale 0.7 by default.
- PAG (perturbed-attention guidance), FreeU and CFG rescale, applied per call by run_pipe().
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np

# name → (diffusers class, extra config)
SCHEDULERS: dict[str, tuple[str, dict]] = {
    "DPM++ 2M Karras":     ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "DPM++ 2M":            ("DPMSolverMultistepScheduler", {}),
    "DPM++ 2M SDE Karras": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++",
                                                            "use_karras_sigmas": True}),
    "DPM++ 2M AYS":        ("DPMSolverMultistepScheduler", {"_ays": True}),
    "DPM++ SDE Karras":    ("DPMSolverSDEScheduler", {"use_karras_sigmas": True}),
    "DPM++ SDE":           ("DPMSolverSDEScheduler", {}),
    "Euler a":             ("EulerAncestralDiscreteScheduler", {}),
    "Euler":               ("EulerDiscreteScheduler", {}),
    "Euler AYS":           ("EulerDiscreteScheduler", {"_ays": True}),
    "DDIM":                ("DDIMScheduler", {}),
    "PNDM":                ("PNDMScheduler", {}),
    "LMS":                 ("LMSDiscreteScheduler", {}),
    "Heun":                ("HeunDiscreteScheduler", {}),
    "UniPC":               ("UniPCMultistepScheduler", {}),
}
# records written before the Karras fix (format < 2) used these names for the plain samplers
LEGACY_NAMES = {"DPM++ 2M Karras": "DPM++ 2M", "DPM++ SDE Karras": "DPM++ SDE"}


# ── Align Your Steps ──────────────────────────────────────────────────────────
def ays_timesteps(n: int, xl: bool, alphas_cumprod) -> list[int]:
    """n decreasing timesteps following NVIDIA's AYS schedule (defined for 10 steps;
    other counts are interpolated log-linearly in sigma, as the paper suggests)."""
    from diffusers.schedulers import AysSchedules
    table = AysSchedules["StableDiffusionXLTimesteps" if xl else "StableDiffusionTimesteps"]
    n = max(1, int(n))
    if n == len(table):
        return list(table)
    ac = np.asarray(alphas_cumprod, dtype=np.float64)
    log_sig = np.log(np.sqrt((1 - ac) / np.clip(ac, 1e-12, None)))
    src = log_sig[np.asarray(table)]
    want = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(table)), src)
    ts = [int(np.argmin(np.abs(log_sig - s))) for s in want]
    for i in range(1, n):                 # strictly decreasing (duplicates at large n)
        ts[i] = min(ts[i], ts[i - 1] - 1)
    return [max(0, t) for t in ts]


def _use_ays(scheduler, xl: bool):
    """Make scheduler.set_timesteps(n) use AYS timesteps (every pipeline, img2img slicing
    included, goes through set_timesteps — so nothing else needs to know)."""
    orig = scheduler.set_timesteps

    def set_timesteps(num_inference_steps=None, device=None, timesteps=None, **kw):
        if timesteps is None and num_inference_steps:
            timesteps = ays_timesteps(num_inference_steps, xl, scheduler.alphas_cumprod.cpu().numpy())
            return orig(device=device, timesteps=timesteps, **kw)
        return orig(num_inference_steps, device=device, timesteps=timesteps, **kw)

    scheduler.set_timesteps = set_timesteps
    scheduler._ays = True


def make_scheduler(pipe, name: str):
    """A fresh scheduler for `name` from the pipeline's config (keeps its prediction type)."""
    import diffusers
    cls_name, extra = SCHEDULERS.get(name, SCHEDULERS["DPM++ 2M Karras"])
    extra = dict(extra)
    ays = extra.pop("_ays", False)
    cfg = dict(pipe.scheduler.config)
    if cls_name in ("DPMSolverMultistepScheduler", "UniPCMultistepScheduler"):
        extra["lower_order_final"] = True       # no off-by-one IndexError on the last step
    zero_snr = bool(cfg.get("rescale_betas_zero_snr"))
    if zero_snr:
        # Karras / AYS sigmas start from sigma_max, which zero-terminal SNR makes infinite
        extra.pop("use_karras_sigmas", None)
        ays = False
        extra["timestep_spacing"] = "trailing"
    # a previous scheduler's options must not leak into this one
    for k in ("use_karras_sigmas", "algorithm_type"):
        if k not in extra and k in cfg:
            cfg.pop(k)
    cls = getattr(diffusers, cls_name)
    try:
        sched = cls.from_config(cfg, **extra)
    except ImportError as e:
        # DPMSolverSDEScheduler needs torchsde (in requirements.txt; older installs lack it)
        print(f"[Sampler] {name} unavailable ({e}); using DPM++ 2M Karras. Re-run install.bat to add it.")
        return make_scheduler(pipe, "DPM++ 2M Karras")
    if ays:
        _use_ays(sched, xl=hasattr(pipe, "text_encoder_2"))
    return sched


# ── V-prediction ──────────────────────────────────────────────────────────────
def detect_prediction(path: str) -> dict:
    """{'v_pred': bool, 'zero_snr': bool} from a checkpoint's safetensors header
    (NoobAI-style `v_pred` / `ztsnr` marker keys) or, failing that, its file name."""
    out = {"v_pred": False, "zero_snr": False}
    p = Path(str(path))
    name = p.name.lower()
    if re.search(r"v[-_ ]?pred", name):
        out.update(v_pred=True, zero_snr=True)
    if p.suffix.lower() == ".safetensors" and p.is_file():
        try:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                if 2 < n < 200 << 20:
                    hdr = json.loads(f.read(n))
                    keys = set(hdr)
                    meta = hdr.get("__metadata__") or {}
                    if "v_pred" in keys or str(meta.get("modelspec.prediction_type", "")).lower() in ("v", "v_prediction"):
                        out["v_pred"] = True
                    if "ztsnr" in keys:
                        out["zero_snr"] = True
                    if out["v_pred"] and "ztsnr" not in keys and "v_pred" in keys:
                        out["zero_snr"] = True    # v_pred-marked SDXL releases are all ZTSNR-trained
        except (OSError, ValueError, struct.error):
            pass
    return out


def configure_prediction(pipe, pred: dict) -> None:
    """Put a v-prediction checkpoint's needs into the scheduler config, so every scheduler
    made from it later (make_scheduler) keeps them."""
    if not pred.get("v_pred"):
        return
    upd = {"prediction_type": "v_prediction"}
    if pred.get("zero_snr"):
        upd.update(rescale_betas_zero_snr=True, timestep_spacing="trailing")
    # as keyword arguments: from_config() resets keys listed in the config's hidden
    # "_use_default_values" to their defaults, even when the dict itself overrides them
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, **upd)


# ── Per-call boosters: PAG, FreeU, CFG rescale ────────────────────────────────
# FreeU factors (keyed by "is SDXL"), tuned on anime checkpoints (2026-09-28, hassakuXL Illustrious /
# hassaku SD 1.5, same seeds). The paper's SDXL values (b1 1.3, b2 1.4, s1 0.9, s2 0.2) doubled the
# saturation (63 → 135), clipped 10 % of pixels (vs 2 %) and bent the composition; these keep the
# picture and palette and add a little contrast / prompt adherence.
_FREEU = {True: dict(b1=1.05, b2=1.1, s1=0.95, s2=0.8), False: dict(b1=1.1, b2=1.2, s1=0.9, s2=0.6)}
_gpu_fft_ok = True


def _patch_fourier_filter():
    """FreeU filters skip features with an FFT; ZLUDA has no cuFFT (CUFFT_NOT_SUPPORTED),
    so after the first failure the (small: 2×1280×38×26 for SDXL) FFT runs on the CPU."""
    import diffusers.utils.torch_utils as tu
    if getattr(tu.fourier_filter, "_cpu_fallback", False):
        return
    orig = tu.fourier_filter

    def fourier_filter(x_in, threshold, scale):
        global _gpu_fft_ok
        if x_in.device.type != "cpu" and _gpu_fft_ok:
            try:
                return orig(x_in, threshold, scale)
            except RuntimeError as e:
                if "fft" not in str(e).lower():
                    raise
                _gpu_fft_ok = False
                print(f"[FreeU] GPU FFT unavailable ({e}); filtering on the CPU")
        if x_in.device.type == "cpu":
            return orig(x_in, threshold, scale)
        return orig(x_in.detach().float().cpu(), threshold, scale).to(x_in.device, x_in.dtype)

    fourier_filter._cpu_fallback = True
    tu.fourier_filter = fourier_filter


def _pag_class(xl: bool, kind: str):
    import diffusers
    name = {("txt2img", True): "StableDiffusionXLPAGPipeline",
            ("img2img", True): "StableDiffusionXLPAGImg2ImgPipeline",
            ("txt2img", False): "StableDiffusionPAGPipeline",
            ("img2img", False): "StableDiffusionPAGImg2ImgPipeline"}.get((kind, xl))
    return getattr(diffusers, name, None) if name else None


_PARAMS: dict[type, set] = {}


def _call_params(cls) -> set:
    if cls not in _PARAMS:
        import inspect
        _PARAMS[cls] = set(inspect.signature(cls.__call__).parameters)
    return _PARAMS[cls]


def boosters(sdp) -> dict:
    """The pipeline wrapper's current settings (set by the app before a run)."""
    b = dict(getattr(sdp, "boosters", None) or {})
    pred = getattr(sdp, "prediction", {}) or {}
    rescale = b.get("cfg_rescale")
    if rescale is None or rescale < 0:
        rescale = 0.7 if pred.get("v_pred") else 0.0   # "auto"
    return {"pag": max(0.0, float(b.get("pag") or 0.0)), "freeu": bool(b.get("freeu")),
            "cfg_rescale": min(1.0, float(rescale))}


def run_pipe(sdp, pipe, kind: str, **call):
    """Call a diffusers pipeline with PAG / FreeU / CFG rescale applied as set on `sdp`.
    kind: txt2img / img2img / inpaint (PAG isn't used for inpaint)."""
    import torch
    b = boosters(sdp)
    xl = hasattr(pipe, "text_encoder_2")
    unet = pipe.unet
    if b["cfg_rescale"] > 0:
        call["guidance_rescale"] = b["cfg_rescale"]
    if b["freeu"]:
        _patch_fourier_filter()
        unet.enable_freeu(**_FREEU[xl])
    target = pipe
    if b["pag"] > 0 and kind in ("txt2img", "img2img"):
        cls = _pag_class(xl, kind)
        if cls is not None:
            cache = getattr(sdp, "_pag_pipes", None)
            if cache is None or cache.get("unet") is not unet:
                cache = sdp._pag_pipes = {"unet": unet}
            if kind not in cache:
                cache[kind] = cls.from_pipe(pipe, pag_applied_layers=["mid"], torch_dtype=unet.dtype)
            target = cache[kind]
            target.scheduler = pipe.scheduler            # the sampler chosen for this run
            call["pag_scale"] = b["pag"]
    # SD 1.5 img2img / inpaint have no guidance_rescale: pass only what the target accepts
    if "guidance_rescale" in call and "guidance_rescale" not in _call_params(type(target)):
        call.pop("guidance_rescale")
    # PAG swaps attention processors on the shared UNet and only restores them when a run
    # finishes; a Stop mid-run must not leave them in place for the next plain run.
    procs = dict(unet.attn_processors) if target is not pipe else None
    try:
        with torch.no_grad():
            return target(**call)
    finally:
        if procs is not None:
            unet.set_attn_processor(procs)
        if b["freeu"]:
            unet.disable_freeu()
        target = None
