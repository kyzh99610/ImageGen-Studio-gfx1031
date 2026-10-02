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
# deepghs anime face detector (YOLOv8n, MIT, 12 MB ONNX; F1 0.94 at conf 0.278 per its threshold.json).
# The LBP cascade found a test character's face (bangs over one eye) in only 16 of 28 generations.
_YOLO_REPO, _YOLO_FILE = "deepghs/anime_face_detection", "face_detect_v1.4_n/model.onnx"
_YOLO_SHA256 = "fd860b650a4377046842c3cd80d01b0b408bdfbdb4acee5759630f82c6ef04a9"
# its threshold.json says 0.278 (best F1 on its test set); on this app's outputs anime faces
# scored 0.81–0.88 and fox / panda photos 0.38–0.51, so a stricter cut keeps animals out
_YOLO_CONF = 0.55
_yolo = {}
# deepghs anime hand detector (YOLOv8s, OpenRAIL, 44 MB ONNX; F1 0.79 at 0.395 per its threshold.json)
_HAND_REPO, _HAND_FILE = "deepghs/anime_hand_detection", "hand_detect_v1.0_s/model.onnx"
_HAND_SHA256 = "408750ad39645fcdc0c5e774aa45a73941b2e785fc5611fb7d3d9790a41899c0"
_HAND_CONF = 0.45
_hand_yolo = {}
_cascades: dict = {}


# ── Inpainting ────────────────────────────────────────────────────────────────
def _embeds(sdp, pipe, prompt: str, negative: str, clip_skip: int) -> dict:
    """Prompt embeddings exactly as txt2img builds them (Compel weights, long prompts,
    CLIP skip for SD 1.5, pooled embeddings for SDXL)."""
    # (frames on the Compel call path get captured; don't let this one keep `pipe` alive)
    try:
        if getattr(sdp, "is_sdxl", False):
            from backend.sdxl_pipeline import _build_sdxl_embeds
            return _build_sdxl_embeds(pipe, prompt, negative)
        from backend.sd_pipeline import _build_embeds
        return _build_embeds(pipe, prompt, negative, sdp.device, clip_skip)
    finally:
        pipe = sdp = None


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
    from backend.sampling import uniform_variant

    if sdp is None or sdp.pipe is None:
        raise RuntimeError("No model loaded.")
    # Karras / AYS start an inpaint pass at much lower noise for the same denoise (AYS 12 at 0.95: sigma 7.4, evenly
    # spaced 9.1; at 0.75: 2.9 vs 4.1), so a big flat colour couldn't be changed at all — denoise means what it says,
    # as in the face / hand / tiled passes (2026-10-02)
    scheduler = uniform_variant(scheduler)
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
    embeds = lat = None
    # Like the txt2img encoders: this frame must not keep `pipe` (→ the old UNet / TEs) alive
    # once the model is unloaded — after face detail, a switch to another model left
    # 1.6–1.8 GB of VRAM held.
    try:
        _load_scheduler(pipe, scheduler)
        generator, used_seed = _make_generator(seed, sdp.device)
        embeds = _embeds(sdp, pipe, prompt, negative, clip_skip)
        cb = {}
        if step_callback is not None:
            eff = max(1, int(int(steps) * float(denoise)))   # the steps the inpaint pass really runs

            def _cb(p, i, t, kw):
                step_callback(min(i + 1, eff), eff)
                return kw
            cb = {"callback_on_step_end": _cb}
        from backend.sampling import run_pipe      # FreeU / CFG rescale as in the main pass
        # The SDXL inpaint pipeline encodes the crop itself and, because the SDXL VAE config says
        # force_upcast, casts the whole VAE to fp32 first: with native (non-cuDNN) convs a 512 px fp32
        # tile needs a 1.12 GiB scratch buffer, which failed on every face-detail pass with 3 GB free.
        # Encode in fp16 unless this checkpoint's VAE really needs fp32 (then 256 px tiles).
        vae = pipe.vae
        saved_vae = (vae.config.get("force_upcast"), getattr(vae, "tile_sample_min_size", None),
                     getattr(vae, "tile_latent_min_size", None))
        if xl and saved_vae[0] is not None:
            need32 = bool(getattr(sdp, "_vae_needs_fp32", False))
            vae.register_to_config(force_upcast=need32)
            if need32 and saved_vae[1]:
                vae.tile_sample_min_size, vae.tile_latent_min_size = 256, 32
        try:
            with torch.inference_mode():
                lat = run_pipe(sdp, pipe, "inpaint", **embeds, image=crop.resize((rw, rh), Image.LANCZOS),
                               mask_image=crop_mask.resize((rw, rh), Image.NEAREST), width=rw, height=rh,
                               strength=float(denoise), num_inference_steps=int(steps), guidance_scale=float(cfg),
                               generator=generator, output_type="latent", **cb).images
        finally:
            if xl and saved_vae[0] is not None:
                vae.register_to_config(force_upcast=saved_vae[0])
                if saved_vae[1]:
                    vae.tile_sample_min_size, vae.tile_latent_min_size = saved_vae[1], saved_vae[2]
            vae = None
        result = sdp._decode_latents(pipe.vae, lat)[0].resize((cw, ch), Image.LANCZOS)
    finally:
        pipe = embeds = lat = cb = None
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


# A real face, cut out with some context and enlarged so it is ~160 px wide, draws many raw
# (minNeighbors=0) cascade hits on it; hands, clothing folds and scenery draw 0–3. Measured on 15
# SDXL outputs: faces 6–49 hits, false detections 0–3.
_MIN_CONFIRM_HITS = 4


def _confirm_hits(image: Image.Image, box, kind: str) -> int:
    """Raw hits of the `kind` cascade centred inside `box`, on an enlarged crop around it."""
    import cv2
    c = _cascade(kind)
    if c is None:
        return 0
    x1, y1, x2, y2 = box
    fw = max(1, x2 - x1)
    pad = fw // 2
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    crop = image.crop((cx1, cy1, min(image.width, x2 + pad), min(image.height, y2 + pad)))
    sc = 160 / fw
    crop = crop.resize((max(1, int(crop.width * sc)), max(1, int(crop.height * sc))), Image.BICUBIC)
    g = cv2.equalizeHist(np.array(crop.convert("L")))
    found = c.detectMultiScale(g, scaleFactor=1.05, minNeighbors=0, minSize=(80, 80))
    bx1, by1, bx2, by2 = (x1 - cx1) * sc, (y1 - cy1) * sc, (x2 - cx1) * sc, (y2 - cy1) * sc
    return sum(1 for x, y, w, h in (found if len(found) else [])
               if bx1 <= x + w / 2 <= bx2 and by1 <= y + h / 2 <= by2)


def _onnx_session(cache: dict, repo: str, file: str, sha256: str, what: str, fallback: str):
    """An ONNX detector from the HF hub (downloaded once, checksum-checked, ORT CPU), or None."""
    import time
    # a failed download is retried after 5 minutes (it used to stay off until a restart, and the
    # hand pass then said "no hands found"; audit F-18)
    if "sess" in cache and (cache["sess"] is not None or time.time() - cache.get("failed_at", time.time()) < 300):
        return cache["sess"]
    sess = None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo, file)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
            print(f"[Detail] {what} model didn't match its checksum — {fallback}")
        else:
            import onnxruntime as ort
            opts = ort.SessionOptions(); opts.log_severity_level = 3
            sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[Detail] {what} model unavailable ({e}) — {fallback}")
    cache["sess"] = sess
    if sess is None:
        cache["failed_at"] = time.time()
    else:
        cache.pop("failed_at", None)
    return sess


def _yolo_session():
    """The ONNX face detector, or None (then the cascades are used)."""
    return _onnx_session(_yolo, _YOLO_REPO, _YOLO_FILE, _YOLO_SHA256, "anime face", "using the cascade")


def _yolo_faces(image: Image.Image, conf: float = _YOLO_CONF, side: int = 640) -> list[tuple[int, int, int, int]]:
    """Face boxes from the YOLO detector (empty if it isn't available)."""
    return [b for b, _s in _yolo_run(_yolo_session(), image, conf, side)]


def _yolo_run(sess, image: Image.Image, conf: float, side: int = 640) -> list[tuple[tuple[int, int, int, int], float]]:
    """(box, score) from a one-class YOLOv8 ONNX export; empty without a session."""
    import cv2
    if sess is None:
        return []
    W, H = image.size
    k = side / max(W, H)
    nw, nh = max(32, round(W * k / 32) * 32), max(32, round(H * k / 32) * 32)
    x = np.asarray(image.convert("RGB").resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
    out = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
    out = out.T if out.shape[0] == 5 else out  # YOLOv8 export: (4 box + 1 class, anchors) → (anchors, 5)
    out = out[out[:, 4] >= conf]
    if not len(out):
        return []
    sx, sy = W / nw, H / nh
    boxes = [[float((cx - w / 2) * sx), float((cy - h / 2) * sy), float(w * sx), float(h * sy)]
             for cx, cy, w, h in out[:, :4]]
    idx = cv2.dnn.NMSBoxes(boxes, out[:, 4].astype(float).tolist(), conf, 0.5)
    res = []
    for i in (np.array(idx).reshape(-1) if len(idx) else []):
        bx, by, bw, bh = boxes[int(i)]
        res.append(((max(0, int(bx)), max(0, int(by)), min(W, int(bx + bw)), min(H, int(by + bh))),
                    float(out[int(i), 4])))
    return res


def detect_faces(image: Image.Image, mode: str = "auto", max_faces: int = 4,
                 min_frac: float = 0.03) -> list[tuple[int, int, int, int]]:
    """Face boxes (x1, y1, x2, y2), biggest first. mode: auto / anime / photo.
    Anime / auto: the YOLO anime face detector; the cascades below are photo mode, and the
    fallback when the model can't be downloaded."""
    import cv2
    W, H = image.size
    min_px = max(24, int(min(W, H) * min_frac))
    if mode in ("auto", "anime") and _yolo_session() is not None:
        # the model is the only detector here: when it sees no face there is none — the cascade
        # fallback found lighthouses, water and leaves in pictures without people
        found = [b for b in _yolo_faces(image) if b[2] - b[0] >= min_px]
        if not found:
            return []
        if found:
            found.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
            keep = []
            for b in found:       # same overlap / tiny-box rules as the cascade path
                if any(_iou(b, k) >= 0.3 or (k[0] <= (b[0] + b[2]) / 2 <= k[2] and k[1] <= (b[1] + b[3]) / 2 <= k[3])
                       for k in keep):
                    continue
                if keep and (b[2] - b[0]) < 0.45 * (keep[0][2] - keep[0][0]):
                    continue
                keep.append(b)
            return keep[: max(0, int(max_faces))]
    gray = cv2.equalizeHist(np.array(image.convert("L")))
    boxes = []
    for kind in (["anime", "photo"] if mode == "auto" else [mode]):
        c = _cascade(kind)
        if c is None:
            continue
        found = c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7 if kind == "anime" else 8,
                                   minSize=(min_px, min_px))
        boxes += [((int(x), int(y), int(x + w), int(y + h)), kind) for x, y, w, h in (found if len(found) else [])]
    boxes.sort(key=lambda bk: -(bk[0][2] - bk[0][0]) * (bk[0][3] - bk[0][1]))
    # Confirm each box on an enlarged crop. The photo (Haar) cascade finds hands, folds and
    # bodies in anime pictures and confirms them itself (a hand on a railing: 56 hits), so in
    # auto mode the anime cascade confirms every box; only when it confirms none (a photo)
    # are Haar boxes confirmed by Haar.
    if mode == "auto":
        cand = [b for b, _k in boxes if _confirm_hits(image, b, "anime") >= _MIN_CONFIRM_HITS]
        if not cand:
            cand = [b for b, k in boxes if k == "photo" and _confirm_hits(image, b, "photo") >= _MIN_CONFIRM_HITS]
    else:
        cand = [b for b, k in boxes if _confirm_hits(image, b, k) >= _MIN_CONFIRM_HITS]
    keep = []
    for b in cand:
        # overlapping boxes: keep the bigger one. Boxes far smaller than the main face are
        # nearly always false hits (windows, buttons, background patterns) — a face-prompted
        # repaint there would draw a face into the scenery.
        # (or its centre is inside one: two boxes on one face overlapped by only 28 %)
        if any(_iou(b, k) >= 0.3 or (k[0] <= (b[0] + b[2]) / 2 <= k[2] and k[1] <= (b[1] + b[3]) / 2 <= k[3])
               for k in keep):
            continue
        if keep and (b[2] - b[0]) < 0.45 * (keep[0][2] - keep[0][0]):
            continue
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
    from backend.sampling import uniform_variant
    scheduler = uniform_variant(scheduler)   # see sampling._UNIFORM
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


# ── Hands ─────────────────────────────────────────────────────────────────────
def _hand_session():
    return _onnx_session(_hand_yolo, _HAND_REPO, _HAND_FILE, _HAND_SHA256, "anime hand", "hand detail is off")


def hand_detector_available() -> bool:
    return _hand_session() is not None


def detect_hands(image: Image.Image, max_hands: int = 6, conf: float = _HAND_CONF,
                 min_frac: float = 0.02) -> list[tuple[int, int, int, int]]:
    """Hand boxes (x1, y1, x2, y2), biggest first; empty when the detector isn't available.
    Gloved hands count; so do some hand-shaped things (ship turrets scored 0.78) — the hand
    pass keeps its denoise low so such a box is only touched up, not turned into a hand."""
    W, H = image.size
    min_px = max(12, int(min(W, H) * min_frac))
    found = [b for b, _s in _yolo_run(_hand_session(), image, conf)
             if min(b[2] - b[0], b[3] - b[1]) >= min_px]
    found.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    keep = []
    for b in found:
        if any(_iou(b, k) >= 0.3 or (k[0] <= (b[0] + b[2]) / 2 <= k[2] and k[1] <= (b[1] + b[3]) / 2 <= k[3])
               for k in keep):
            continue
        keep.append(b)
    return keep[: max(0, int(max_hands))]


def hand_detail(sdp, image: Image.Image, prompt: str, negative: str = "", *, denoise: float = 0.35,
                steps: int = 20, cfg: float = 7.0, seed: int = -1, scheduler: str = "DPM++ 2M Karras",
                clip_skip: int = 1, hand_prompt: str = "", max_hands: int = 6,
                step_callback=None) -> tuple[Image.Image, int]:
    """Re-draw each detected hand at native resolution. Returns (image, hands fixed)."""
    hands = detect_hands(image, max_hands)
    if not hands:
        return image, 0
    from backend.sampling import uniform_variant
    scheduler = uniform_variant(scheduler)   # see sampling._UNIFORM
    W, H = image.size
    out = image
    for n, (x1, y1, x2, y2) in enumerate(hands):
        hw, hh = x2 - x1, y2 - y1
        # a rounded box (an ellipse would clip fingertips in the corners) plus a small margin
        mx, my = int(hw * 0.12), int(hh * 0.12)
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).rounded_rectangle([x1 - mx, y1 - my, x2 + mx, y2 + my],
                                               radius=max(2, min(hw, hh) // 4), fill=255)
        p = f"{prompt}, {hand_prompt}" if hand_prompt else prompt
        out, _ = inpaint_region(sdp, out, mask, p, negative, steps=steps, cfg=cfg, denoise=denoise,
                                seed=(seed + 101 + n) % 2**32 if seed is not None and seed >= 0 else -1,
                                scheduler=scheduler, clip_skip=clip_skip, padding=int(max(hw, hh) * 0.6),
                                min_context=0, feather=max(3, hw // 12), step_callback=step_callback)
    return out, len(hands)


# ── Tiled "SD upscale" detail pass ────────────────────────────────────────────
def tile_boxes(W: int, H: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Overlapping tile boxes (x1, y1, x2, y2) of at most tile×tile covering W×H; edge tiles are shifted
    inwards so every tile is full-size when the image is at least that big."""
    def starts(n):
        if n <= tile:
            return [0]
        step = tile - overlap
        s = list(range(0, n - tile, step)) + [n - tile]
        return sorted(set(s))
    return [(x, y, min(W, x + tile), min(H, y + tile)) for y in starts(H) for x in starts(W)]


def _ramp_mask(w: int, h: int, left: int, top: int, right: int, bottom: int) -> Image.Image:
    """Alpha mask that fades in over `left`/`top`/`right`/`bottom` pixels (0 where it shouldn't fade)."""
    a = np.ones((h, w), np.float32)
    if left:
        a[:, :left] *= np.linspace(0, 1, left, dtype=np.float32)[None, :]
    if right:
        a[:, w - right:] *= np.linspace(1, 0, right, dtype=np.float32)[None, :]
    if top:
        a[:top, :] *= np.linspace(0, 1, top, dtype=np.float32)[:, None]
    if bottom:
        a[h - bottom:, :] *= np.linspace(1, 0, bottom, dtype=np.float32)[:, None]
    return Image.fromarray((a * 255).round().astype(np.uint8), "L")


def tiled_detail(sdp, image: Image.Image, prompt: str, negative: str = "", *, denoise: float = 0.3,
                 steps: int = 20, cfg: float = 7.0, seed: int = 0, scheduler: str = "DPM++ 2M Karras",
                 clip_skip: int = 1, overlap: int = 96, step_callback=None) -> Image.Image:
    """Re-draw an (already upscaled) image tile by tile at the model's native size with low denoise, blending
    the overlaps — adds detail beyond what hires fix can reach (it stops at 1536 / 2048 px). Each tile gets
    the whole prompt, so keep denoise low (0.25–0.35) or faces may appear in the scenery."""
    from backend.sampling import uniform_variant
    if sdp is None or sdp.pipe is None:
        raise RuntimeError("No model loaded.")
    image = image.convert("RGB")
    W, H = image.size
    native = 1024 if getattr(sdp, "is_sdxl", False) else 512
    tile = min(native, W, H) // 8 * 8
    ov = max(0, min(int(overlap), tile // 3))
    boxes = tile_boxes(W, H, tile, ov)
    out = image.copy()
    scheduler = uniform_variant(scheduler)     # denoise means what it says (see sampling._UNIFORM)
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        crop = out.crop((x1, y1, x2, y2))      # tiles see the already-refined neighbours they overlap
        cw, ch = crop.size
        res, _ = sdp.img2img(crop, prompt, negative, denoise, steps, cfg, (seed + k) % 2**32, scheduler,
                             step_callback=(lambda s, t, k=k: step_callback(k, len(boxes), s, t))
                             if step_callback else None, clip_skip=clip_skip)
        tile_img = res[0].resize((cw, ch), Image.LANCZOS) if res[0].size != (cw, ch) else res[0]
        mask = _ramp_mask(cw, ch, ov if x1 > 0 else 0, ov if y1 > 0 else 0,
                          ov if x2 < W else 0, ov if y2 < H else 0)
        out.paste(tile_img, (x1, y1), mask)
    return out
