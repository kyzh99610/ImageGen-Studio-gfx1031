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
import re
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
# deepghs anime eye detector (YOLOv8s, OpenRAIL, 44.6 MB ONNX). Run on the face crop, not the whole picture: on a
# full-body 1664×2432 picture it found one eye of two, on the face crop both (scores 0.63–0.78 on her eyes).
_EYE_REPO, _EYE_FILE = "deepghs/anime_eye_detection", "eye_detect_v1.0_s/model.onnx"
_EYE_SHA256 = "7c5f0259103cc407e2a0f0b047a51271246b9525d27920315295a54367c5c583"
_EYE_LOCAL = MODELS_DIR / "detectors" / "anime_eye_detect_v1.0_s.onnx"
_EYE_CONF = 0.4
_EYE_RESCUE_CONF = 0.25      # the second eye of a face may be found down to this score (see detect_eyes)
_eye_yolo = {}
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
                   min_context: int | None = None, feather: int = 4, step_callback=None,
                   native: int | None = None) -> tuple[Image.Image, int]:
    """Repaint the white part of `mask` with the loaded model. Returns (image, seed used).
    `native` overrides the long side the crop is scaled to (default 512 SD 1.5 / 1024 SDXL)."""
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
    native = int(native) if native else (1024 if xl else 512)
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
        need32 = bool(getattr(sdp, "_vae_needs_fp32", False)) if xl else False
        for attempt in (0, 1):
            generator, used_seed = _make_generator(seed, sdp.device)       # same noise again on the retry
            vae = pipe.vae
            saved_vae = (vae.config.get("force_upcast"), getattr(vae, "tile_sample_min_size", None),
                         getattr(vae, "tile_latent_min_size", None))
            if xl and saved_vae[0] is not None:
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
            if bool(torch.isfinite(lat).all()):
                break
            # A NoobAI checkpoint (VAE stored in bf16) overflows in an fp16 encode: every latent came back NaN and the redraw was a black
            # patch. The fp32 need is otherwise only learned by a failed fp16 *decode*, which a pass that runs first in a
            # fresh process (Inpaint tab, SD detail, a card batch's eye pass) never makes — learn it here and go again.
            if attempt == 0 and xl and not need32:
                need32 = True
                sdp._vae_needs_fp32 = True
                print("[Inpaint] NaN latents after an fp16 VAE encode — this model's VAE needs fp32; retrying", flush=True)
                continue
            raise RuntimeError("the sampler produced NaN latents, so nothing was changed")
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


def _onnx_session(cache: dict, repo: str, file: str, sha256: str, what: str, fallback: str, local: Path | None = None):
    """An ONNX detector from the HF hub (downloaded once, checksum-checked, ORT CPU), or None.
    `local`: a copy placed in models/detectors/ by hand is used first when its checksum matches."""
    import time
    # a failed download is retried after 5 minutes (it used to stay off until a restart, and the
    # hand pass then said "no hands found"; audit F-18)
    if "sess" in cache and (cache["sess"] is not None or time.time() - cache.get("failed_at", time.time()) < 300):
        return cache["sess"]
    sess = None
    try:
        if local is not None and Path(local).is_file() and \
                hashlib.sha256(Path(local).read_bytes()).hexdigest() == sha256:
            path = str(local)
        else:
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


def size_aware_denoise(denoise: float, face_px: float) -> float:
    """Face-pass denoise for a face `face_px` wide. Measured on hassakuXL 832×1216 / 1216×832, identity (0 = generic look-alike, 1 = her on a
    portrait) paired against 0.35 on the same pictures: round 4 — a ~50 px face (wide shot) +2.0 at 0.35 but +2.6 at 0.65, and 0.8 turned a
    tiny face into somebody else; round 5 — 62–90 px (n 4) 0.55 +0.48 ± 0.14 (the old blend to 0.35 at 110 px: +0.32), 90–130 px (n 8) 0.55
    +0.21 ± 0.09 / 0.65 +0.31 ± 0.14 (the old rule left them at 0.35: +0.02), ≥ 130 px (n 3) no difference; at ≤ 104 px the hair pin came back more
    often at 0.55 (pin probability 0.39 → 0.63 at 62–90 px, 0.62 → 0.78 at 90–130 px). So faces ≤ 100 px get at least 0.55, faces ≥ 150 px the
    slider's value, linear in between; never above max(slider, 0.65). Round 6: faces ≤ 62 px (n 4) 0.65 over 0.55 +0.50 ± 0.24 (3 of 4
    better), the direction of round 4's +0.57 (0.65 over 0.35, 3/3 seeds) — so the tiniest faces (≤ 60 px) get 0.65, blending to 0.55 at 100 px."""
    lo, hi, small, tiny, tiny_px = 100.0, 150.0, 0.55, 0.65, 60.0
    if denoise >= tiny or face_px >= hi:
        return denoise
    if face_px <= tiny_px:
        want = tiny
    elif face_px <= lo:
        want = tiny + (small - tiny) * (face_px - tiny_px) / (lo - tiny_px)
    else:
        want = small + (denoise - small) * (face_px - lo) / (hi - lo)
    return round(min(max(denoise, 0.65), max(denoise, want)), 3)


def face_detail(sdp, image: Image.Image, prompt: str, negative: str = "", *, denoise: float = 0.4,
                steps: int = 20, cfg: float = 7.0, seed: int = -1, scheduler: str = "DPM++ 2M Karras",
                clip_skip: int = 1, mode: str = "auto", face_prompt: str = "", max_faces: int = 4,
                size_aware: bool = True, step_callback=None) -> tuple[Image.Image, int]:
    """Re-draw each detected face at native resolution. Returns (image, faces fixed).
    size_aware: tiny faces get a higher denoise (`size_aware_denoise`); the slider is the value for normal faces."""
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
        den = size_aware_denoise(denoise, fw) if size_aware else denoise
        out, _ = inpaint_region(sdp, out, mask, p, negative, steps=steps, cfg=cfg, denoise=den,
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


# ── Eyes ──────────────────────────────────────────────────────────────────────
def _eye_session():
    return _onnx_session(_eye_yolo, _EYE_REPO, _EYE_FILE, _EYE_SHA256, "anime eye", "eye detail is off",
                         local=_EYE_LOCAL)


def eye_detector_available() -> bool:
    return _eye_session() is not None


def detect_eyes(image: Image.Image, faces=None, max_faces: int = 4, conf: float = _EYE_CONF,
                rescue: float | None = _EYE_RESCUE_CONF) -> list[list[tuple[int, int, int, int]]]:
    """Eye boxes per face: [[eye, eye], …] in the order of `faces` (detected when not given); a face whose
    eyes aren't found gets []. The detector runs on each face crop (+25 % margin) — on a whole full-body picture
    it missed eyes it found on the crop — and keeps at most two plausible boxes per face: width 6–50 % of the
    face, centre inside the face box and above 80 % of its height (no mouths, no eyes of a second face).
    `rescue`: a face with only one eye above `conf` may take a weaker box (score ≥ rescue) at least a quarter of
    the face width away from it — the far eye of a three-quarter view or the one under her bangs (on 1,056 faces
    the far eye was found at 0.27–0.38 in 12 of 12 cases; none of those boxes was a false one)."""
    sess = _eye_session()
    if sess is None:
        return []
    if faces is None:
        faces = detect_faces(image, "anime", max_faces)
    W, H = image.size
    res = []
    low = min(conf, rescue) if rescue else conf
    for x1, y1, x2, y2 in faces:
        fw, fh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, x1 - fw // 4), max(0, y1 - fh // 4)
        cx2, cy2 = min(W, x2 + fw // 4), min(H, y2 + fh // 4)
        found = []
        for (bx1, by1, bx2, by2), sc in _yolo_run(sess, image.crop((cx1, cy1, cx2, cy2)), low):
            b = (bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1)
            mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if not (0.06 * fw <= b[2] - b[0] <= 0.5 * fw and x1 <= mx <= x2 and y1 <= my <= y1 + 0.8 * fh):
                continue
            found.append((b, sc))
        found.sort(key=lambda bs: -bs[1])
        keep = []
        for b, sc in found:
            if sc >= conf and not any(_iou(b, k) >= 0.2 for k in keep):
                keep.append(b)
        if len(keep) == 1 and rescue and rescue < conf:
            kx = (keep[0][0] + keep[0][2]) / 2
            for b, sc in found:
                if sc >= conf or sc < rescue or _iou(b, keep[0]) >= 0.2 or abs((b[0] + b[2]) / 2 - kx) < 0.25 * fw:
                    continue
                keep.append(b)
                break
        res.append(sorted(keep[:2]))
    return res


def colour_guard(before: Image.Image, after: Image.Image, iris_boxes, region=None,
                 min_chroma: float = 12.0) -> Image.Image:
    """`after` with its colour (Lab a/b) kept only where colour belongs: inside the eye boxes and only on pixels
    that were already colourful in `before` (the iris). Everywhere else — grey bangs over the eye, eyelids, skin —
    the brightness detail comes from `after` but the colour stays `before`'s. Stops a red iris prompt from tinting
    the hair and face around it (seen on bangs over one eye). `region` limits the work (x1, y1, x2, y2)."""
    import cv2
    W, H = after.size
    rx1, ry1, rx2, ry2 = region or (0, 0, W, H)
    rx1, ry1, rx2, ry2 = max(0, rx1), max(0, ry1), min(W, rx2), min(H, ry2)
    if rx2 <= rx1 or ry2 <= ry1:
        return after
    b = cv2.cvtColor(np.asarray(before.convert("RGB").crop((rx1, ry1, rx2, ry2))), cv2.COLOR_RGB2LAB).astype(np.float32)
    a = cv2.cvtColor(np.asarray(after.convert("RGB").crop((rx1, ry1, rx2, ry2))), cv2.COLOR_RGB2LAB).astype(np.float32)
    core = Image.new("L", (rx2 - rx1, ry2 - ry1), 0)
    d = ImageDraw.Draw(core)
    for x1, y1, x2, y2 in iris_boxes:
        d.ellipse([x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1], fill=255)
    chroma = np.hypot(b[..., 1] - 128.0, b[..., 2] - 128.0)
    allow = (np.asarray(core) > 0) & (chroma >= min_chroma)
    allow = cv2.dilate(allow.astype(np.uint8), np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(allow.astype(np.float32), (0, 0), 1.0)[..., None]
    out = a.copy()
    out[..., 1:] = alpha * a[..., 1:] + (1 - alpha) * b[..., 1:]
    rgb = cv2.cvtColor(np.clip(out, 0, 255).round().astype(np.uint8), cv2.COLOR_LAB2RGB)
    res = after.convert("RGB").copy()
    res.paste(Image.fromarray(rgb), (rx1, ry1))
    return res


# the eye pass's own prompt (round 5, five Illustrious / NoobAI / Pony checkpoints, paired against the full prompt + three eye words):
# quality + subject + the tags that describe the eyes, the expression and the gaze, then words that ask for the structures that should
# show. Without "sparkling eyes, eye highlights, limbal ring": those drew 4-point star sparkles into the pupils in 3 of 8 pictures (3x crops).
_EYE_KEEP = re.compile(r"eye|pupil|iris|lash|tsurime|jitome|tareme|sanpaku|heterochromia|glasses|eyewear|goggles|monocle|tears|crying|wink|"
                       r"smile|expression|blush|looking|bangs|hair over|hair between|eyebrow|makeup|eyeshadow|eyeliner|"
                       r"@_@|\+_\+|\^_\^|>_<|=_=|;_;|o_o|0_0|x_x", re.I)          # (the kaomoji are eyes too)
EYE_DETAIL_WORDS = "detailed eyes, beautiful detailed eyes, eyelashes, detailed pupils, iris detail"
# "detailed pupils" alone drew bold outlined slit pupils (one checkpoint 4 of 4, Pony, some hassakuXL). With "round pupils" in the prompt and
# "slit pupils, cat eyes" in the negative the pupils stay round and the iris keeps its shading (8 pictures, 3x crops; hassakuXL's eye sharpness over
# the untouched base 1.12 -> 1.01) — unless the prompt asks for a pupil shape itself (slit / heart-shaped / @_@ …): then the pass leaves it alone.
EYE_ROUND = "round pupils"
EYE_ROUND_NEGATIVE = "slit pupils, cat eyes"
_PUPIL_SHAPE = re.compile(r"slit|cat eyes|reptil|snake|dragon|goat|animal eyes|-shaped|shaped pupils|horizontal pupils|vertical pupils|rectangular|no pupils|"
                          r"blank eyes|empty eyes|mesmeriz|hypnosis|spiral|ringed eyes|@_@|\+_\+|solid circle eyes", re.I)


def _eye_tags(prompt: str) -> list[str]:
    """The tags of `prompt` the eye pass keeps: quality, subject, and everything that describes the eyes, the expression and the gaze."""
    from backend.prompt_tools import _is_quality, split_tags
    keep = []
    for tag in split_tags(prompt or ""):
        core = re.sub(r"[()\\]", "", tag).lower()
        if _is_quality(tag) or re.fullmatch(r"\d?(girl|girls|boy|boys)|solo", core.split(":")[0].strip()) or _EYE_KEEP.search(core):
            keep.append(tag)
    return keep


def eye_prompt_from(prompt: str) -> str:
    """The prompt the eye pass redraws with: the quality and subject tags of `prompt`, its eye / expression / gaze tags and EYE_DETAIL_WORDS
    + "round pupils" (the scene, outfit, hair and accessory tags stay out — the crop is only the eyes)."""
    from backend.prompt_tools import join_tags
    keep = _eye_tags(prompt)
    own_shape = _PUPIL_SHAPE.search(", ".join(keep))
    return join_tags(keep + [EYE_DETAIL_WORDS if own_shape else f"{EYE_DETAIL_WORDS}, {EYE_ROUND}"])


def eye_negative_from(prompt: str, negative: str = "") -> str:
    """`negative` + "slit pupils, cat eyes" (see EYE_ROUND), except when the prompt names a pupil shape itself."""
    from backend.prompt_tools import merge_prompts
    if _PUPIL_SHAPE.search(", ".join(_eye_tags(prompt))):
        return negative
    return merge_prompts(negative or "", EYE_ROUND_NEGATIVE)


# Executed steps of the detail passes: about half the main steps (ADetailer runs steps × denoise), at least 10 — and 8 for the eye pass. Round 8 (an Illustrious checkpoint + a character LoRA, 2 scenes × 2 seeds, CCIP / pin /
# eye energy / by eye): the eye pass at 8 steps equals the one at 10 (CCIP −0.001, eye energy ×0.98, pin ±0.00) and is 21 % cheaper; at 6 steps it loses 4 % eye energy.
DETAIL_MIN_STEPS = {"face": 10, "hand": 10, "eye": 8}


def detail_schedule_steps(kind: str, steps: int, denoise: float) -> int:
    """Schedule length of a face / hand / eye detail pass: ceil(max(the pass's minimum, main steps × 0.5) / denoise), so that int(length × denoise) steps run."""
    import math
    return min(150, math.ceil(max(DETAIL_MIN_STEPS.get(kind, 10), int(steps) * 0.5) / max(0.05, float(denoise))))


EYE_DENOISE = 0.4        # default strength of the eye pass: with the eye-only prompt Laplacian energy of the eyes +17 % (0.3: +6 %) over the old whole-prompt pass at 0.3, no colour bleed


def eye_detail(sdp, image: Image.Image, prompt: str, negative: str = "", *, denoise: float = EYE_DENOISE,
               steps: int = 20, cfg: float = 7.0, seed: int = -1, scheduler: str = "DPM++ 2M Karras",
               clip_skip: int = 1, eye_prompt: str = "", max_faces: int = 4, guard: bool = True,
               step_callback=None) -> tuple[Image.Image, int]:
    """Re-draw the eyes of each face at high resolution: one inpaint per face covering both eyes (so they match),
    crop ≈ the eye pair + one eye width around it, then `colour_guard` so the iris colour can't bleed into bangs,
    lids or skin. Returns (image, eyes re-drawn)."""
    groups = [g for g in detect_eyes(image, max_faces=max_faces) if g]
    if not groups:
        return image, 0
    from backend.sampling import uniform_variant
    scheduler = uniform_variant(scheduler)
    W, H = image.size
    out = image
    p = eye_prompt or eye_prompt_from(prompt)       # `eye_prompt` given = the whole prompt of the redraw (and the negative is used as given)
    negative = negative if eye_prompt else eye_negative_from(prompt, negative)
    for n, eyes in enumerate(groups):
        mask = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(mask)
        ew = max(x2 - x1 for x1, _y1, x2, _y2 in eyes)
        for x1, y1, x2, y2 in eyes:
            mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.35)      # lashes and lids
            d.ellipse([x1 - mx, y1 - my, x2 + mx, y2 + my], fill=255)
        before = out
        out, _ = inpaint_region(sdp, out, mask, p, negative, steps=steps, cfg=cfg, denoise=denoise,
                                seed=(seed + 211 + n) % 2**32 if seed is not None and seed >= 0 else -1,
                                scheduler=scheduler, clip_skip=clip_skip, padding=int(ew * 1.2),
                                min_context=0, feather=max(2, ew // 10), step_callback=step_callback)
        if guard:
            gx1 = min(e[0] for e in eyes) - ew; gy1 = min(e[1] for e in eyes) - ew
            gx2 = max(e[2] for e in eyes) + ew; gy2 = max(e[3] for e in eyes) + ew
            out = colour_guard(before, out, eyes, (gx1, gy1, gx2, gy2))
    return out, sum(len(g) for g in groups)


# ── Swap a hair accessory ─────────────────────────────────────────────────────
# 🖌 Inpaint over an old ornament at 0.9–0.95 leaves its lattice showing as lace (round 6: a butterfly in 8 of 8, the old shape in all 8); at 1.0
# nothing of it survives (denoise 1.0 starts from noise inside the mask). But the whole prompt then draws its own scene into the crop — the canon
# pin's "lace, mesh" tags gave a net round / instead of the butterfly on 2 of 4 seeds — so the swap prompts with what belongs in the crop only:
# the quality, subject and hair tags of the prompt (never its tags about the accessory that is there now) and the new item. Round 7 (hassakuXL,
# the X-lattice clip → a black butterfly, 4 seeds): one pass at 1.0 with that prompt = a clean butterfly on every seed; erasing the old clip first
# (1.0, hair-only prompt) and drawing at 0.9 gave a second butterfly or a web, never a better one.
SWAP_DENOISE = 1.0
SWAP_MIN_PADDING = 64        # the model needs hair / face round the spot to place and size the item
_SWAP_KEEP = re.compile(r"hair|bangs|strand|ponytail|braid|twintail|sidelock|ahoge|forehead", re.I)
_SWAP_OLD = re.compile(r"ornament|clip|hairpin|hair ?stick|barrette|hair ?band|headband|headdress|ribbon|\bbow\b|scrunchie|tiara|crown|\bhat\b|bonnet|"
                       r"\bveil\b|flower|butterfly|\blace\b|\bmesh\b|accessor", re.I)


# Round 8 (an Illustrious checkpoint + a character LoRA, three cowboy pictures): earrings and a choker work with this hair-tag prompt as it is. A hat needs the whole hat PAINTED: a mask that only covered the hair gave a hat in 1 of 3; with a
# realistic mask (the crown reaching ~100 px above the head) the hair tags are what keeps the model from drawing a second head — without them ("1girl, solo, beret") it drew a tiny girl wearing the beret into the mask.
def swap_prompt_from(prompt: str, item: str) -> str:
    """The prompt of a swap pass: the quality and subject tags of `prompt` and its hair tags (minus the ones that name an accessory), then
    `item` at weight 1.2 — the scene, outfit and the old accessory stay out, the crop is only hair."""
    from backend.prompt_tools import _is_quality, join_tags, split_tags
    keep = []
    for tag in split_tags(prompt or ""):
        core = re.sub(r"[()\\]", "", tag).lower()
        if _SWAP_OLD.search(core):
            continue
        if _is_quality(tag) or re.fullmatch(r"\d?(girl|girls|boy|boys)|solo", core.split(":")[0].strip()) or _SWAP_KEEP.search(core):
            keep.append(tag)
    return join_tags(keep + [f"({item.strip()}:1.2)"])


def swap_item(sdp, image: Image.Image, mask: Image.Image, item: str, prompt: str, negative: str = "", *,
              steps: int = 12, cfg: float = 6.0, seed: int = -1, scheduler: str = "DPM++ 2M Karras",
              clip_skip: int = 1, padding: int = SWAP_MIN_PADDING, step_callback=None) -> tuple[Image.Image, int]:
    """Draw `item` (e.g. "black butterfly hair ornament") where `mask` is painted — over an old accessory: one inpaint at denoise 1.0 with
    `swap_prompt_from`. Returns (image, seed used). `steps` is the number of sampling steps (the schedule length at 1.0)."""
    return inpaint_region(sdp, image, mask, swap_prompt_from(prompt, item), negative, steps=steps, cfg=cfg, denoise=SWAP_DENOISE, seed=seed,
                          scheduler=scheduler, clip_skip=clip_skip, padding=max(int(padding), SWAP_MIN_PADDING), step_callback=step_callback)


# ── Re-draw a broken hand (several tries to pick from) ────────────────────────
# Round 7, 11 genuinely broken hands (10 SD 1.5, 1 SDXL): the hand pass at 0.35–0.55 never fixed a finger count or a tangle; a re-draw
# of the hand at 0.8 with a hand-only prompt fixed 2 of 10 (+4 plausible but changed), 1.0 invented objects. So one try rarely works —
# the useful tool makes several (seed, seed+1, …) and lets the user pick.
HAND_REDRAW_DENOISE = 0.8
HAND_EXTRA = "detailed hands, perfect hands, five fingers, detailed fingers, hand focus"
_HAND_KEEP = re.compile(r"hand|finger|holding|peace|\bv\b|gesture|fist|nail|glove|wrist|bracelet|ring\b|fan|cup|mug|phone|flower|bouquet|"
                        r"umbrella|reaching|outstretched|interlocked|adjusting|pointing|waving|sleeve|cuff", re.I)


def hand_prompt_from(prompt: str) -> str:
    """The prompt of a hand re-draw: the quality and subject tags of `prompt`, its tags about hands / what they hold / sleeves, then
    the hand words (the scene, face and outfit stay out: the crop is only the hand)."""
    from backend.prompt_tools import _is_quality, join_tags, split_tags
    keep = []
    for tag in split_tags(prompt or ""):
        core = re.sub(r"[()\\]", "", tag).lower()
        if _is_quality(tag) or re.fullmatch(r"\d?(girl|girls|boy|boys)|solo", core.split(":")[0].strip()) or _HAND_KEEP.search(core):
            keep.append(tag)
    return join_tags(keep + [HAND_EXTRA])


def hand_mask(size: tuple[int, int], box) -> Image.Image:
    """The hand pass's mask for a hand box: a rounded box + 12 % (an ellipse clips fingertips)."""
    W, H = size
    x1, y1, x2, y2 = box
    hw, hh = x2 - x1, y2 - y1
    mx, my = int(hw * 0.12), int(hh * 0.12)
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rounded_rectangle([x1 - mx, y1 - my, x2 + mx, y2 + my], radius=max(2, min(hw, hh) // 4), fill=255)
    return m


# Round 9 (SD 1.5, ~220 tries, blind by-eye ratings): on hands next to her face (cheek / chin) the guard-less mask repaints part of her face — 4.7–7.4 / 255 mean change inside the face
# box against 0.5–0.7 with the face taken out of the mask — and the hands are as good (13 of 24 correct against 11 of 24). Adding "face, head, extra face" to the negative changed
# details but not the outcome (the same tries still drew a face), and a hand prompt without "1girl, solo" cut the whole-face tries on heart_98902 from 5/40 to 1/40 but lowered
# the correct-rate (pooled with three controls 22/64 against 26/64): neither is shipped.
HAND_FACE_MASK = True
_HAND_FACE_GROW, _HAND_FACE_KEEP = 1.15, 0.25


def _face_model_cached() -> bool:
    """The anime face detector is already downloaded: a hand re-draw uses it only then (no download from this path; the face pass fetches it)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(_YOLO_REPO, _YOLO_FILE), str)
    except Exception:
        return False


def face_free_mask(image: Image.Image, mask: Image.Image, pad: float = 0.6, faces=None) -> tuple[Image.Image, list]:
    """(mask, faces that reach into the re-draw's crop). The crop of a hand re-draw is the mask's box + `pad` × its side; a face inside it is
    taken out of the mask (its ellipse, grown by 15 %), so the model never paints over her face. A mask that would lose more than 75 % of its
    area (a hand held over her face) keeps its shape. Without the face model (or when detection fails) the mask is returned as it was."""
    m = np.array(mask.convert("L").resize(image.size, Image.NEAREST)) > 127
    if not m.any():
        return mask, []
    ys, xs = np.nonzero(m)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    p = int(max(x2 - x1, y2 - y1, 1) * pad)
    crop = (x1 - p, y1 - p, x2 + p, y2 + p)
    if faces is None:
        try:
            faces = detect_faces(image, "anime") if _face_model_cached() else []
        except Exception:
            faces = []
    inside = [f for f in faces if f[0] < crop[2] and f[2] > crop[0] and f[1] < crop[3] and f[3] > crop[1]]
    if not inside:
        return mask, []
    el = Image.new("L", image.size, 0)
    d = ImageDraw.Draw(el)
    for fx1, fy1, fx2, fy2 in inside:
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        hw, hh = (fx2 - fx1) / 2 * _HAND_FACE_GROW, (fy2 - fy1) / 2 * _HAND_FACE_GROW
        d.ellipse([cx - hw, cy - hh, cx + hw, cy + hh], fill=255)
    keep = m & (np.array(el) == 0)
    if keep.sum() < _HAND_FACE_KEEP * m.sum():
        return mask, inside
    return Image.fromarray((keep * 255).astype(np.uint8), "L"), inside


def redraw_hand(sdp, image: Image.Image, mask: Image.Image, prompt: str, negative: str = "", *, tries: int = 4,
                steps: int = 12, cfg: float = 6.0, seed: int = -1, scheduler: str = "DPM++ 2M Karras", clip_skip: int = 1,
                step_callback=None, on_try=None, out: list | None = None) -> list[tuple[Image.Image, int]]:
    """`tries` re-draws of the painted hand at denoise 0.8 with `hand_prompt_from`, seeds seed, seed + 1, … (a random base for
    seed -1). Returns [(image, seed)]. Steps as the hand pass measured them (round 7): ceil(max(10, steps × 0.5) / 0.8).
    Every finished try is appended to `out` (pass your own list) as soon as it is done: a Stop or an error at try k raises out of
    this function, and the k − 1 finished tries are still in `out`."""
    import math
    import random
    from backend.sampling import uniform_variant
    m = np.array(mask.convert("L").resize(image.size, Image.NEAREST)) > 127
    if not m.any():
        return []
    ys, xs = np.nonzero(m)
    side = max(int(xs.max() - xs.min()), int(ys.max() - ys.min()), 1)
    base = int(seed) if seed is not None and int(seed) >= 0 else random.randint(0, 2**32 - 1)
    p = hand_prompt_from(prompt)
    if HAND_FACE_MASK:                                     # a face next to the hand stays as it is (the crop padding below still follows the whole hand)
        mask, _faces = face_free_mask(image, mask)
    run_steps = min(150, math.ceil(max(10, int(steps) * 0.5) / HAND_REDRAW_DENOISE))
    out = [] if out is None else out
    for k in range(max(1, min(8, int(tries)))):
        if on_try is not None:
            on_try(k)
        img, used = inpaint_region(sdp, image, mask, p, negative, steps=run_steps, cfg=cfg, denoise=HAND_REDRAW_DENOISE,
                                   seed=(base + k) % 2**32, scheduler=uniform_variant(scheduler), clip_skip=clip_skip,
                                   padding=int(side * 0.6), min_context=0, feather=max(3, side // 12), step_callback=step_callback)
        out.append((img, used))
    return out


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
