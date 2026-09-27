"""
detail_tools.py — inpainting and the face-detail pass (ADetailer-style), both done on a crop
around the area at the model's native resolution with the loaded model, its LoRAs and the
user's prompt.

  inpaint_region()  repaint the masked part of an image ("only masked", A1111 style): the
                    crop is scaled so its long side is 512 (SD 1.5) / 1024 (SDXL), inpainted,
                    scaled back and blended through a feathered mask — pixels away from the
                    mask never change.
  detect_faces()    anime faces (nagadomi's lbpcascade_animeface, MIT, ~250 KB, downloaded
                    once) and photo faces (OpenCV's bundled Haar cascade).
  face_detail()     every detected face re-drawn at native resolution with low denoise —
                    small faces in full-body shots come out with proper eyes and mouths.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import MODELS_DIR

ANIME_CASCADE = MODELS_DIR / "detectors" / "lbpcascade_animeface.xml"
_ANIME_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
_ANIME_SHA256 = "9376d30ac38db6bda2a68b88b3b76bbd7e6aa33af47f7f5c76bc88ca75f1ce30"
_cascades: dict = {}


# ── Inpainting ────────────────────────────────────────────────────────────────
def _embeds(sdp, pipe, prompt: str, negative: str, clip_skip: int) -> dict:
    """Prompt embeddings exactly as txt2img builds them (Compel weights, long prompts,
    CLIP skip for SD 1.5, pooled embeddings for SDXL)."""
    if getattr(sdp, "is_sdxl", False):
        from backend.sdxl_pipeline import _build_sdxl_embeds
        return _build_sdxl_embeds(pipe, prompt, negative)
    from backend.sd_pipeline import _build_embeds
    return _build_embeds(pipe, prompt, negative, sdp.device, clip_skip)


def _crop_box(mask_np: np.ndarray, W: int, H: int, pad: int, min_side: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask_np)
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    min_side = min(min_side, W, H)
    if x2 - x1 < min_side:
        cx = (x1 + x2) // 2
        x1 = max(0, min(W - min_side, cx - min_side // 2)); x2 = x1 + min_side
    if y2 - y1 < min_side:
        cy = (y1 + y2) // 2
        y1 = max(0, min(H - min_side, cy - min_side // 2)); y2 = y1 + min_side
    return x1, y1, x2, y2


def inpaint_region(sdp, image: Image.Image, mask: Image.Image, prompt: str, negative: str = "", *,
                   steps: int = 25, cfg: float = 7.0, denoise: float = 0.75, seed: int = -1,
                   scheduler: str = "DPM++ 2M Karras", clip_skip: int = 1, padding: int = 48,
                   min_context: int | None = None, feather: int = 4, step_callback=None) -> tuple[Image.Image, int]:
    """Repaint the white part of `mask` with the loaded model. Returns (image, seed used)."""
    import torch
    from backend.sd_pipeline import _load_scheduler, _make_generator

    if sdp is None or sdp.pipe is None:
        raise RuntimeError("No model loaded.")
    image = image.convert("RGB")
    W, H = image.size
    mask = mask.convert("L").resize((W, H), Image.NEAREST)
    m_np = np.array(mask) > 127
    if not m_np.any():
        return image, seed
    xl = bool(getattr(sdp, "is_sdxl", False))
    native = 1024 if xl else 512
    if xl:
        from diffusers import StableDiffusionXLInpaintPipeline as _Inpaint
    else:
        from diffusers import StableDiffusionInpaintPipeline as _Inpaint

    box = _crop_box(m_np, W, H, int(padding), min_context if min_context is not None else native // 2)
    crop, crop_mask = image.crop(box), mask.crop(box)
    cw, ch = crop.size
    k = native / max(cw, ch)
    rw, rh = max(64, int(round(cw * k / 8)) * 8), max(64, int(round(ch * k / 8)) * 8)

    # from_pipe shares the UNet (LoRAs included); dtype must be passed or it upcasts to fp32
    pipe = getattr(sdp, "_inpaint_pipe", None)
    if pipe is None or getattr(pipe, "unet", None) is not sdp.pipe.unet:
        pipe = _Inpaint.from_pipe(sdp.pipe, torch_dtype=sdp.dtype)
        sdp._inpaint_pipe = pipe
    _load_scheduler(pipe, scheduler)
    generator, used_seed = _make_generator(seed, sdp.device)
    embeds = _embeds(sdp, pipe, prompt, negative, clip_skip)
    cb = {}
    if step_callback is not None:
        def _cb(p, i, t, kw):
            step_callback(i + 1, steps)
            return kw
        cb = {"callback_on_step_end": _cb}
    with torch.inference_mode():
        lat = pipe(**embeds, image=crop.resize((rw, rh), Image.LANCZOS),
                   mask_image=crop_mask.resize((rw, rh), Image.NEAREST), width=rw, height=rh,
                   strength=float(denoise), num_inference_steps=int(steps), guidance_scale=float(cfg),
                   generator=generator, output_type="latent", **cb).images
    result = sdp._decode_latents(pipe.vae, lat)[0].resize((cw, ch), Image.LANCZOS)
    soft = crop_mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(feather))
    out = image.copy()
    out.paste(Image.composite(result, crop, soft), box[:2])
    return out, used_seed


# ── Faces ─────────────────────────────────────────────────────────────────────
def _anime_cascade_path() -> Path | None:
    if ANIME_CASCADE.exists():
        return ANIME_CASCADE
    try:
        import httpx
        data = httpx.get(_ANIME_URL, timeout=20, follow_redirects=True).content
        if hashlib.sha256(data).hexdigest() != _ANIME_SHA256:
            print("[FaceDetail] anime face detector download didn't match its checksum — skipped")
            return None
        ANIME_CASCADE.parent.mkdir(parents=True, exist_ok=True)
        ANIME_CASCADE.write_bytes(data)
        return ANIME_CASCADE
    except Exception as e:
        print(f"[FaceDetail] could not download the anime face detector: {e}")
        return None


def _cascade(kind: str):
    import cv2
    if kind not in _cascades:
        path = _anime_cascade_path() if kind == "anime" else \
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        c = cv2.CascadeClassifier(str(path)) if path and Path(path).exists() else None
        _cascades[kind] = c if c is not None and not c.empty() else None
    return _cascades[kind]


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


def detect_faces(image: Image.Image, mode: str = "auto", max_faces: int = 4,
                 min_frac: float = 0.03) -> list[tuple[int, int, int, int]]:
    """Face boxes (x1, y1, x2, y2), biggest first. mode: auto / anime / photo."""
    import cv2
    gray = cv2.equalizeHist(np.array(image.convert("L")))
    W, H = image.size
    min_px = max(24, int(min(W, H) * min_frac))
    boxes = []
    for kind in (["anime", "photo"] if mode == "auto" else [mode]):
        c = _cascade(kind)
        if c is None:
            continue
        found = c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7 if kind == "anime" else 8,
                                   minSize=(min_px, min_px))
        boxes += [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in (found if len(found) else [])]
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    keep = []
    for b in boxes:
        # overlapping boxes: keep the bigger one. Boxes far smaller than the main face are
        # nearly always false hits (windows, buttons, background patterns) — a face-prompted
        # repaint there would draw a face into the scenery.
        if all(_iou(b, k) < 0.3 for k in keep) and (not keep or (b[2] - b[0]) >= 0.45 * (keep[0][2] - keep[0][0])):
            keep.append(b)
    return keep[: max(0, int(max_faces))]


def face_detail(sdp, image: Image.Image, prompt: str, negative: str = "", *, denoise: float = 0.4,
                steps: int = 20, cfg: float = 7.0, seed: int = -1, scheduler: str = "DPM++ 2M Karras",
                clip_skip: int = 1, mode: str = "auto", face_prompt: str = "", max_faces: int = 4,
                step_callback=None) -> tuple[Image.Image, int]:
    """Re-draw each detected face at native resolution. Returns (image, faces fixed)."""
    faces = detect_faces(image, mode, max_faces)
    if not faces:
        return image, 0
    W, H = image.size
    out = image
    for n, (x1, y1, x2, y2) in enumerate(faces):
        fw, fh = x2 - x1, y2 - y1
        # mask: the face plus a margin (hairline, chin); crop context: about 2× the face
        mx, my = int(fw * 0.15), int(fh * 0.2)
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).ellipse([x1 - mx, y1 - my, x2 + mx, y2 + my], fill=255)
        p = f"{prompt}, {face_prompt}" if face_prompt else prompt
        out, _ = inpaint_region(sdp, out, mask, p, negative, steps=steps, cfg=cfg, denoise=denoise,
                                seed=(seed + n) % 2**32 if seed is not None and seed >= 0 else -1,
                                scheduler=scheduler, clip_skip=clip_skip, padding=int(max(fw, fh) * 0.5),
                                min_context=0, feather=max(3, fw // 20), step_callback=step_callback)
    return out, len(faces)
