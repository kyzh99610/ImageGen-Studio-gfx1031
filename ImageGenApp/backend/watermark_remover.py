"""
backend/watermark_remover.py

OCR-based text / logo detection and inpainting for watermark removal.
Handles both corner watermarks and embedded text (e.g. "ROG x Hatsune Miku").

Detection methods:
  - EasyOCR          — text + logo bounding boxes (pip install easyocr)
  - Corner heuristic — edge-density scan of image corners

Inpainting methods:
  - OpenCV TELEA / NS — fast, always available, good for thin text
  - LaMa ONNX         — high quality for large areas (auto-downloads ~100 MB)
  - SD Inpainting     — best quality, uses the loaded diffusers pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from config import APP_DIR

_INPAINT_DIR = APP_DIR / "models" / "inpainting"
_LAMA_PATH   = _INPAINT_DIR / "inpainting_lama_2025jan.onnx"
_LAMA_REPO   = "opencv/inpainting_lama"
_LAMA_FILE   = "inpainting_lama_2025jan.onnx"

# Cached ONNX session (created once on first use)
_lama_session = None

# ── Availability checks ────────────────────────────────────────────────────────

def is_ocr_available() -> bool:
    try:
        import easyocr  # noqa
        return True
    except ImportError:
        return False


def is_lama_available() -> bool:
    try:
        import onnxruntime  # noqa
        return True  # ONNX present; model auto-downloads on first use
    except ImportError:
        return False


# ── OCR reader (module-level lazy singleton) ───────────────────────────────────

_ocr_reader = None
_ocr_langs: list[str] | None = None


def _get_ocr_reader(langs: list[str]):
    global _ocr_reader, _ocr_langs
    if _ocr_reader is None or _ocr_langs != langs:
        import easyocr
        try:
            import torch
            use_gpu = torch.cuda.is_available()
            # EasyOCR's cuDNN RNN is broken under ZLUDA (MIOpen). config.py
            # disables cuDNN on AMD, and then PyTorch's native LSTM works:
            # same text, ~4× faster than CPU on the RX 6800M.
            # Only fall back to CPU if MIOpen was re-enabled (IMAGEGEN_CUDNN=1).
            if use_gpu and torch.backends.cudnn.enabled:
                name = torch.cuda.get_device_name(0)
                if "ZLUDA" in name or "AMD" in name or "Radeon" in name:
                    use_gpu = False
        except Exception:
            use_gpu = False
        _ocr_reader = easyocr.Reader(langs, gpu=use_gpu)
        _ocr_langs  = list(langs)
    return _ocr_reader


def _get_lama_session():
    """Return the cached ONNX LaMa session, creating it on first call."""
    global _lama_session
    if _lama_session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3   # suppress [W] initializer noise
        _lama_session = ort.InferenceSession(
            str(_LAMA_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],   # DML crashes on this model
        )
    return _lama_session


# ── Detection ──────────────────────────────────────────────────────────────────

def detect_text_regions(
    image: Image.Image,
    min_confidence: float = 0.3,
    languages: list[str] | None = None,
) -> list[dict]:
    """
    Detect text bounding boxes using EasyOCR.
    Returns list of {"bbox": (x1,y1,x2,y2), "text": str, "confidence": float,
                      "source": "ocr"}.
    Returns [] if EasyOCR is not installed.
    """
    if not is_ocr_available():
        return []

    langs  = languages or ["en"]
    reader = _get_ocr_reader(langs)
    img_np = np.array(image.convert("RGB"))

    raw = reader.readtext(img_np)
    regions = []
    for (bbox_pts, text, conf) in raw:
        if conf < min_confidence:
            continue
        # bbox_pts: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        xs = [int(p[0]) for p in bbox_pts]
        ys = [int(p[1]) for p in bbox_pts]
        regions.append({
            "bbox":       (min(xs), min(ys), max(xs), max(ys)),
            "text":       text,
            "confidence": round(conf, 2),
            "source":     "ocr",
        })
    return regions


def detect_corner_regions(
    image: Image.Image,
    corner_pct: float = 0.12,
    _edges: "np.ndarray | None" = None,
) -> list[dict]:
    """
    Heuristic corner scan: edges in a corner strip are compared to the centre.
    Corners with significantly higher edge density likely contain watermarks/logos.
    Returns list of {"bbox", "text", "confidence", "source": "corner"}.
    _edges: pre-computed Canny array (pass from detect_all to avoid recomputation).
    """
    import cv2
    img_np = np.array(image.convert("L"))
    h, w   = img_np.shape
    cw = max(16, int(w * corner_pct))
    ch = max(16, int(h * corner_pct))

    edges = _edges if _edges is not None else cv2.Canny(img_np, 50, 150)

    # Centre reference density (middle 50% of image)
    cy1, cy2 = h // 4, 3 * h // 4
    cx1, cx2 = w // 4, 3 * w // 4
    centre_density = float(edges[cy1:cy2, cx1:cx2].mean()) + 1e-6

    corners = [
        ("top-left",     0,      0,      cw,     ch),
        ("top-right",    w - cw, 0,      w,      ch),
        ("bottom-left",  0,      h - ch, cw,     h),
        ("bottom-right", w - cw, h - ch, w,      h),
    ]

    regions = []
    for (name, x1, y1, x2, y2) in corners:
        density = float(edges[y1:y2, x1:x2].mean())
        if density > centre_density * 1.5 and density > 5:
            regions.append({
                "bbox":       (x1, y1, x2, y2),
                "text":       f"[{name} watermark]",
                "confidence": round(min(1.0, density / centre_density), 2),
                "source":     "corner",
            })
    return regions


def detect_all(
    image: Image.Image,
    min_confidence: float = 0.3,
    corner_pct: float = 0.12,
    languages: list[str] | None = None,
    use_ocr: bool = True,
    use_corner: bool = True,
    merge_gap_px: int = 30,
) -> list[dict]:
    """Run all enabled detectors, expand to logo boundaries, merge, deduplicate."""
    import cv2
    # Pre-compute Canny once — shared by corner detection and logo boundary expansion
    _gray  = np.array(image.convert("L"))
    _edges = cv2.Canny(_gray, 50, 150)

    regions: list[dict] = []
    if use_ocr:
        regions += detect_text_regions(image, min_confidence, languages)
    if use_corner:
        regions += detect_corner_regions(image, corner_pct, _edges=_edges)
    # Expand OCR text boxes to encompass surrounding logo/graphic elements
    regions = _expand_to_logo_boundary(image, regions, _edges=_edges)
    regions = _merge_nearby(regions, gap_px=merge_gap_px)
    return _dedup_regions(regions)


def _expand_to_logo_boundary(
    image: Image.Image,
    regions: list[dict],
    search_mult: float = 2.0,
    edge_thresh: int = 30,
    _edges: "np.ndarray | None" = None,
) -> list[dict]:
    """
    For each OCR-detected text region, search outward for enclosing logo
    contours using edge detection. This catches graphic elements (icons,
    borders, @ symbols, company logos) that surround detected text but
    aren't detected by OCR themselves.
    _edges: pre-computed Canny array (pass from detect_all to avoid recomputation).
    """
    import cv2

    if not regions:
        return regions

    gray = np.array(image.convert("L"))
    H, W = gray.shape
    edges = _edges if _edges is not None else cv2.Canny(gray, 50, 150)

    expanded = []
    for r in regions:
        if r.get("source") != "ocr":
            expanded.append(r)
            continue

        x1, y1, x2, y2 = r["bbox"]
        bw = x2 - x1
        bh = y2 - y1

        # Search area: expand the text bbox by search_mult on each side
        sx1 = max(0, x1 - int(bw * search_mult))
        sy1 = max(0, y1 - int(bh * search_mult))
        sx2 = min(W, x2 + int(bw * search_mult))
        sy2 = min(H, y2 + int(bh * search_mult))

        # Find contours in the search area
        roi = edges[sy1:sy2, sx1:sx2]
        # Dilate to connect nearby edges into logo outlines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        roi_dilated = cv2.dilate(roi, kernel, iterations=2)

        contours, _ = cv2.findContours(
            roi_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find the contour that contains the original text bbox centre
        cx = (x1 + x2) // 2 - sx1
        cy = (y1 + y2) // 2 - sy1
        best = r["bbox"]

        for cnt in contours:
            # Check if text centre is inside this contour
            if cv2.pointPolygonTest(cnt, (float(cx), float(cy)), False) >= 0:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                # Convert back to image coordinates
                nx1 = sx1 + rx
                ny1 = sy1 + ry
                nx2 = sx1 + rx + rw
                ny2 = sy1 + ry + rh
                # Only expand to a *logo-sized* contour: an icon beside the text or a
                # frame around it. On photos the contour around the text centre is
                # often the subject itself (a 10×-area limit let "SAMPLE" grab 65 %
                # of the image), so bound each dimension as well as the area.
                cnt_area = rw * rh
                txt_area = bw * bh
                if (txt_area < cnt_area < txt_area * 4
                        and rw <= bw * 2 and rh <= bh * 3):
                    best = (
                        min(best[0], nx1),
                        min(best[1], ny1),
                        max(best[2], nx2),
                        max(best[3], ny2),
                    )

        expanded.append({**r, "bbox": best})
    return expanded


def _merge_nearby(regions: list[dict], gap_px: int = 30) -> list[dict]:
    """
    Iteratively merge bounding boxes that are within gap_px pixels of each other.
    This fuses logo containers and their embedded text into a single region,
    preventing the logo background from being left behind after inpainting.
    """
    if not regions:
        return regions
    changed = True
    result = list(regions)
    while changed:
        changed = False
        new_result: list[dict] = []
        used: set[int] = set()
        for i, a in enumerate(result):
            if i in used:
                continue
            ax1, ay1, ax2, ay2 = a["bbox"]
            merged = dict(a)
            for j in range(i + 1, len(result)):
                if j in used:
                    continue
                bx1, by1, bx2, by2 = result[j]["bbox"]
                # Expand each box by gap_px and check overlap
                if (ax2 + gap_px >= bx1 and bx2 + gap_px >= ax1 and
                        ay2 + gap_px >= by1 and by2 + gap_px >= ay1):
                    ax1 = min(ax1, bx1)
                    ay1 = min(ay1, by1)
                    ax2 = max(ax2, bx2)
                    ay2 = max(ay2, by2)
                    merged["bbox"] = (ax1, ay1, ax2, ay2)
                    merged["confidence"] = max(
                        merged.get("confidence", 0),
                        result[j].get("confidence", 0),
                    )
                    t_a = merged.get("text", "")
                    t_b = result[j].get("text", "")
                    merged["text"] = (t_a + " " + t_b).strip()
                    used.add(j)
                    changed = True
            merged["bbox"] = (ax1, ay1, ax2, ay2)
            new_result.append(merged)
        result = new_result
    return result


def _dedup_regions(regions: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Remove duplicate/highly-overlapping bounding boxes (keep highest confidence)."""
    kept: list[dict] = []
    for r in sorted(regions, key=lambda x: x["confidence"], reverse=True):
        dominated = False
        for k in kept:
            if _iou(r["bbox"], k["bbox"]) > iou_thresh:
                dominated = True
                break
        if not dominated:
            kept.append(r)
    return kept


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# ── Mask + Visualization ───────────────────────────────────────────────────────

def build_mask(
    image: Image.Image,
    regions: list[dict],
    dilation: int = 12,
) -> Image.Image:
    """
    Build a grayscale mask (255 = inpaint / remove, 0 = keep) from region dicts.
    Each bounding box is expanded by *dilation* pixels on all sides.
    """
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    W, H = image.size
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        x1 = max(0, x1 - dilation)
        y1 = max(0, y1 - dilation)
        x2 = min(W, x2 + dilation)
        y2 = min(H, y2 + dilation)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def merge_masks(base: Image.Image, drawn: Image.Image | None) -> Image.Image:
    """
    Merge an auto-detected mask with a user-drawn mask.
    *drawn* may be None or an RGBA/RGB image where non-black = inpaint region.
    """
    if drawn is None:
        return base
    drawn_l = drawn.convert("L")
    if drawn_l.size != base.size:
        drawn_l = drawn_l.resize(base.size, Image.LANCZOS)
    combined = Image.new("L", base.size, 0)
    combined.paste(base)
    # any non-zero pixel in drawn also marks as inpaint
    drawn_np  = np.array(drawn_l)
    base_np   = np.array(combined)
    merged_np = np.clip(base_np.astype(int) + (drawn_np > 10).astype(int) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(merged_np, mode="L")


def overlay_detections(image: Image.Image, regions: list[dict]) -> Image.Image:
    """
    Return a copy of *image* with detected regions outlined.
    OCR regions → cyan; corner heuristic regions → orange.
    """
    out     = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        if r.get("source") == "ocr":
            fill    = (0, 220, 200, 60)
            outline = (0, 220, 200, 220)
        else:
            fill    = (255, 160, 0, 60)
            outline = (255, 160, 0, 220)
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline)
        label = (r.get("text") or "")[:28]
        conf  = r.get("confidence", 0)
        draw.text((x1 + 3, y1 + 2), f"{label} ({conf:.0%})", fill=(255, 255, 200, 230))

    return Image.alpha_composite(out, overlay).convert("RGB")


def regions_to_text(regions: list[dict]) -> str:
    """Format detected regions as a human-readable summary string."""
    if not regions:
        return "No watermarks / text regions detected."
    lines = [f"Found {len(regions)} region(s):"]
    for i, r in enumerate(regions, 1):
        src  = r.get("source", "?")
        text = (r.get("text") or "").strip()
        conf = r.get("confidence", 0)
        bbox = r["bbox"]
        lines.append(
            f"  {i}. [{src}] \"{text}\" — conf {conf:.0%} "
            f"@ ({bbox[0]},{bbox[1]})–({bbox[2]},{bbox[3]})"
        )
    return "\n".join(lines)


# ── Inpainting ─────────────────────────────────────────────────────────────────

def inpaint_opencv(
    image: Image.Image,
    mask: Image.Image,
    method: str = "telea",
) -> Image.Image:
    """
    OpenCV inpainting — always available, fast, works best for thin text.
    method: "telea" (Telea, default) | "ns" (Navier-Stokes, smoother texture).
    """
    import cv2
    img_np   = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask_np  = np.array(mask.convert("L"))
    flag     = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    result   = cv2.inpaint(img_np, mask_np, inpaintRadius=4, flags=flag)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def inpaint_lama(
    image: Image.Image,
    mask: Image.Image,
    progress_callback=None,
) -> Image.Image:
    """
    LaMa (Large Mask inpainting) via ONNX Runtime.
    Uses patch-based processing: crops just the masked region + padding,
    runs LaMa at 512×512, then pastes the result back into the full-res image.
    This preserves the original resolution everywhere outside the inpainted area.
    DmlExecutionProvider is skipped — it crashes on this model's MatMul ops.
    """
    import cv2

    if not _LAMA_PATH.exists():
        if progress_callback:
            progress_callback(0.0, "Downloading LaMa ONNX model (~88 MB)…")
        _download_lama()

    if progress_callback:
        progress_callback(0.15, "Loading LaMa ONNX session…")

    session = _get_lama_session()

    W, H = image.size
    mask_np = np.array(mask.convert("L"))

    # Find bounding box of the mask region
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return image  # nothing to inpaint

    x1_m, y1_m = int(xs.min()), int(ys.min())
    x2_m, y2_m = int(xs.max()), int(ys.max())

    # Pad crop region (50% of bbox on each side, min 32 px) for context
    pw = max(32, (x2_m - x1_m) // 2)
    ph = max(32, (y2_m - y1_m) // 2)
    cx1 = max(0, x1_m - pw)
    cy1 = max(0, y1_m - ph)
    cx2 = min(W, x2_m + pw)
    cy2 = min(H, y2_m + ph)

    # Crop image and mask to the patch
    crop_img  = np.array(image.convert("RGB").crop((cx1, cy1, cx2, cy2)))
    crop_mask = np.array(mask.convert("L") .crop((cx1, cy1, cx2, cy2)))
    ch_orig, cw_orig = crop_img.shape[:2]

    # Pre-process crop: resize to 512×512, BGR float32 [0,1]
    img_bgr = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
    img_512 = cv2.resize(img_bgr, (512, 512)).astype(np.float32) / 255.0
    img_t   = img_512.transpose(2, 0, 1)[np.newaxis]           # 1,3,512,512

    msk_512 = cv2.resize(crop_mask, (512, 512),
                         interpolation=cv2.INTER_NEAREST)
    msk_bin = (msk_512 > 0).astype(np.float32)
    msk_t   = msk_bin[np.newaxis, np.newaxis]                   # 1,1,512,512

    if progress_callback:
        progress_callback(0.35, "Running LaMa inference…")

    out_t  = session.run(None, {"image": img_t, "mask": msk_t})[0]  # 1,3,512,512
    raw = out_t[0].transpose(1, 2, 0)                                # H,W,3 BGR float

    # Normalise to [0,1] float32 — keep float throughout to avoid
    # precision loss from repeated uint8 floor rounding (causes darkening).
    if raw.max() > 1.5:
        raw = raw / 255.0
    raw = raw.clip(0.0, 1.0)

    # Resize at float32 precision using bicubic (better quality than bilinear)
    out_crop_bgr_f = cv2.resize(raw, (cw_orig, ch_orig),
                                interpolation=cv2.INTER_CUBIC)
    out_crop_rgb = (out_crop_bgr_f[:, :, ::-1] * 255.0).clip(0, 255).astype(np.uint8)

    # ── Composite: Poisson seamless blending (primary) ────────────────────────
    # cv2.seamlessClone adjusts colour gradients at the mask boundary so the
    # inpainted region matches the surroundings — eliminates the darker halo
    # that Gaussian alpha-blending causes.
    # IMPORTANT: `center` must be the bounding-box center of the mask in the
    # crop, NOT the image center.  seamlessClone maps the mask-bbox centre in
    # src to `center` in dst — using (w//2, h//2) shifts content to the wrong
    # place whenever the watermark is not at the exact middle of the crop.
    mask_uint8 = (crop_mask > 0).astype(np.uint8) * 255
    try:
        # Erode 1 px so the mask never touches the crop border (seamlessClone
        # requirement — touching border causes it to corrupt the whole image).
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_safe = cv2.erode(mask_uint8, k, iterations=1)
        if mask_safe.max() == 0:
            raise ValueError("mask eroded to empty")
        mys, mxs = np.where(mask_safe > 0)
        center = (int((int(mxs.min()) + int(mxs.max())) // 2),
                  int((int(mys.min()) + int(mys.max())) // 2))
        composite = cv2.seamlessClone(
            out_crop_rgb, crop_img, mask_safe, center, cv2.NORMAL_CLONE
        )
    except Exception:
        # Fallback: feathered alpha blend (correct position, slight halo risk)
        blur_r = max(3, int(min(ch_orig, cw_orig) * 0.03)) | 1
        alpha  = cv2.GaussianBlur(
            mask_uint8.astype(np.float32) / 255.0, (blur_r, blur_r), 0
        )[:, :, np.newaxis]
        composite = (
            out_crop_rgb.astype(np.float32) * alpha
            + crop_img.astype(np.float32) * (1.0 - alpha)
        ).clip(0, 255).astype(np.uint8)

    # Paste the composited crop back into a full-resolution copy
    result = image.convert("RGB").copy()
    result.paste(Image.fromarray(composite), (cx1, cy1))

    if progress_callback:
        progress_callback(1.0, "LaMa inpainting complete.")
    return result


def inpaint_sd(
    image: Image.Image,
    mask: Image.Image,
    sd_pipeline,            # backend.sd_pipeline.SDPipeline / SDXLPipeline instance
    prompt: str = "",
    progress_callback=None,
    steps: int = 30,
) -> Image.Image:
    """
    Inpaint with the loaded Stable Diffusion model (SD 1.5 or SDXL).
    Works on a crop around the mask at the model's native size and blends the
    result back through a feathered mask, so pixels away from the watermark are
    untouched and large photos don't get pushed through SD at full resolution.
    """
    import torch
    from PIL import ImageFilter

    if sd_pipeline is None or sd_pipeline.pipe is None:
        raise RuntimeError(
            "No model loaded. Load a checkpoint in the Generate tab first."
        )
    m_np = np.array(mask.convert("L")) > 0
    if not m_np.any():
        return image
    image = image.convert("RGB")
    W, H = image.size

    is_xl = getattr(sd_pipeline, "is_sdxl", False)
    # Concrete classes: diffusers' AutoPipelineForInpainting imports every pipeline
    # family and fails on transformers 4.44 (CogView4 needs GlmModel).
    if is_xl:
        from diffusers import StableDiffusionXLInpaintPipeline as _Inpaint
        native = 1024
    else:
        from diffusers import StableDiffusionInpaintPipeline as _Inpaint
        native = 512

    # ── Crop: mask bbox + context padding, at least native/2 px on a side ──
    ys, xs = np.nonzero(m_np)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = max(32, int(0.25 * max(x2 - x1, y2 - y1)))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
    min_side = min(native // 2, W, H)
    if x2 - x1 < min_side:
        cx = (x1 + x2) // 2
        x1 = max(0, min(W - min_side, cx - min_side // 2)); x2 = x1 + min_side
    if y2 - y1 < min_side:
        cy = (y1 + y2) // 2
        y1 = max(0, min(H - min_side, cy - min_side // 2)); y2 = y1 + min_side
    box = (int(x1), int(y1), int(x2), int(y2))
    crop, crop_mask = image.crop(box), mask.convert("L").crop(box)

    # Scale the crop so its long side is the model's native size (multiple of 8)
    cw, ch = crop.size
    k = native / max(cw, ch)
    rw, rh = max(64, int(round(cw * k / 8)) * 8), max(64, int(round(ch * k / 8)) * 8)

    if progress_callback:
        progress_callback(0.05, f"Building {'SDXL' if is_xl else 'SD 1.5'} inpainting pipeline…")
    # from_pipe defaults to float32 and would upcast the shared generation model
    inp_pipe = _Inpaint.from_pipe(sd_pipeline.pipe, torch_dtype=sd_pipeline.dtype)

    def _cb(pipe, i, t, kw):
        if progress_callback:
            progress_callback(0.1 + 0.75 * (i + 1) / steps, f"SD inpainting step {i + 1}/{steps}")
        return kw

    if progress_callback:
        progress_callback(0.1, f"Inpainting a {cw}×{ch} region at {rw}×{rh}…")
    with torch.inference_mode():
        out = inp_pipe(
            prompt=prompt or "seamless background, clean, no text, no watermark",
            negative_prompt="text, watermark, logo, letters, signature",
            image=crop.resize((rw, rh), Image.LANCZOS),
            mask_image=crop_mask.resize((rw, rh), Image.NEAREST),
            width=rw, height=rh,
            num_inference_steps=steps,
            guidance_scale=7.5,
            output_type="latent",
            callback_on_step_end=_cb,
        ).images
    if progress_callback:
        progress_callback(0.9, "Decoding…")
    # Reuse the pipeline's own decode path (GPU, tiled for SDXL, CPU fallback on NaN/OOM)
    result = sd_pipeline._decode_latents(sd_pipeline.pipe.vae, out)[0].resize((cw, ch), Image.LANCZOS)

    # ── Blend back: feathered mask so the seam is invisible ────────────────
    feather = crop_mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(4))
    patch = Image.composite(result, crop, feather)
    final = image.copy()
    final.paste(patch, box[:2])
    if progress_callback:
        progress_callback(1.0, "SD inpainting complete.")
    return final


# ── LaMa model download ────────────────────────────────────────────────────────

def _download_lama():
    """Download LaMa ONNX model via huggingface_hub (uses HF_TOKEN if set)."""
    try:
        from huggingface_hub import hf_hub_download
        import os
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        _INPAINT_DIR.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=_LAMA_REPO,
            filename=_LAMA_FILE,
            local_dir=str(_INPAINT_DIR),
            token=token,
        )
        print(f"[LaMa] Downloaded to {path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download LaMa model '{_LAMA_REPO}/{_LAMA_FILE}': {e}\n"
            f"Manually place the .onnx file in: {_INPAINT_DIR}"
        )
