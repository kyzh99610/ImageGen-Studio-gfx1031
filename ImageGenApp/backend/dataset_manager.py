"""
Dataset preparation for LoRA training.

Handles:
- Image scanning and validation
- Aspect ratio bucketing (supports extreme ratios like 16:5 body pillows)
- Smart resizing with minimal center crop
- Caption / tag file management (.txt sidecar files)
- Dataset statistics and bucket preview
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from PIL import Image

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


# ═════════════════════════════════════════════════════════════════════════════
# Image scanning
# ═════════════════════════════════════════════════════════════════════════════

def scan_images(folder: str | Path) -> list[dict]:
    """Scan a folder for images and return metadata for each.

    Returns list of dicts with keys:
        path, filename, width, height, aspect_ratio, caption, size_kb
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []

    results = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in _IMG_EXTS:
            continue
        try:
            with Image.open(f) as img:
                w, h = img.size
        except Exception:
            continue

        caption = read_caption(f)
        results.append({
            "path": str(f),
            "filename": f.name,
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 3),
            "caption": caption,
            "size_kb": round(f.stat().st_size / 1024, 1),
        })
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Aspect-ratio bucketing
# ═════════════════════════════════════════════════════════════════════════════

def generate_buckets(
    base_res: int = 512,
    min_dim: int = 192,
    max_dim: int = 0,
    step: int = 64,
) -> list[tuple[int, int]]:
    """Generate aspect ratio buckets around base_res² total pixels.

    Body pillow (16:5 = 3.2 ratio) gets a bucket like 960×320 (SD 1.5)
    or 1920×640 (SDXL). No content-destroying crop needed.

    Args:
        base_res: 512 for SD 1.5, 1024 for SDXL.
        min_dim:  Smallest allowed dimension.  Lower → more extreme ratios.
        max_dim:  Largest allowed dimension.  0 = auto (base_res × 3).
        step:     Dimension granularity (must divide into bucket dims).
    """
    if max_dim <= 0:
        max_dim = base_res * 3
    target_area = base_res * base_res
    max_area = int(target_area * 1.5)

    buckets: set[tuple[int, int]] = set()
    for w in range(min_dim, max_dim + 1, step):
        h = round(target_area / w / step) * step
        if min_dim <= h <= max_dim and w * h <= max_area:
            buckets.add((w, h))

    return sorted(buckets, key=lambda b: b[0] / b[1])


def assign_bucket(
    img_w: int, img_h: int,
    buckets: list[tuple[int, int]],
) -> tuple[int, int]:
    """Pick the bucket whose aspect ratio is closest to the image's."""
    img_ar = img_w / max(img_h, 1)
    best, best_diff = buckets[len(buckets) // 2], float("inf")
    for bw, bh in buckets:
        diff = abs(math.log(max(img_ar, 0.01)) - math.log(bw / max(bh, 1)))
        if diff < best_diff:
            best_diff = diff
            best = (bw, bh)
    return best


def compute_crop_pct(img_w: int, img_h: int, bw: int, bh: int) -> float:
    """How much of the image is lost to center-crop (0–100 %)."""
    img_ar = img_w / max(img_h, 1)
    bkt_ar = bw / max(bh, 1)
    if img_ar > bkt_ar:
        scale = bh / max(img_h, 1)
        return max(0.0, (img_w * scale - bw) / max(img_w * scale, 1) * 100)
    else:
        scale = bw / max(img_w, 1)
        return max(0.0, (img_h * scale - bh) / max(img_h * scale, 1) * 100)


def compute_bucket_assignments(
    images: list[dict],
    base_res: int = 512,
    min_dim: int = 192,
    step: int = 64,
) -> dict:
    """Compute bucket assignments for every image.

    Returns:
        buckets:     list of (w, h)
        assignments: list of dicts  (original + bucket_w, bucket_h, crop_pct)
        stats:       dict  bucket_label → count
    """
    buckets = generate_buckets(base_res, min_dim, step=step)
    assignments, bucket_counts = [], {}

    for img in images:
        bw, bh = assign_bucket(img["width"], img["height"], buckets)
        crop = compute_crop_pct(img["width"], img["height"], bw, bh)
        assignments.append({**img, "bucket_w": bw, "bucket_h": bh,
                            "crop_pct": round(crop, 1)})
        key = f"{bw}×{bh}"
        bucket_counts[key] = bucket_counts.get(key, 0) + 1

    return {"buckets": buckets, "assignments": assignments, "stats": bucket_counts}


# ═════════════════════════════════════════════════════════════════════════════
# Image resizing
# ═════════════════════════════════════════════════════════════════════════════

def resize_and_crop(
    img_path: str | Path,
    target_w: int,
    target_h: int,
) -> Image.Image:
    """Resize to bucket with minimal center crop.

    1.  Scale so the *shorter* dimension fills the target exactly.
    2.  Center-crop the *longer* dimension.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


# ═════════════════════════════════════════════════════════════════════════════
# Caption I/O
# ═════════════════════════════════════════════════════════════════════════════

def read_caption(image_path: str | Path) -> str:
    p = Path(image_path).with_suffix(".txt")
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""


def write_caption(image_path: str | Path, caption: str):
    Path(image_path).with_suffix(".txt").write_text(
        caption.strip(), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# Full dataset preparation
# ═════════════════════════════════════════════════════════════════════════════

def prepare_dataset(
    assignments: list[dict],
    output_dir: str | Path,
    trigger_word: str = "",
    callback: Callable | None = None,
) -> str:
    """Resize images into buckets, write caption .txt files.

    Produces a flat directory of  ``{stem}.png`` + ``{stem}.txt`` pairs
    ready for the training loop.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(assignments)

    for i, item in enumerate(assignments):
        if callback:
            callback(i, total, f"Processing {item['filename']}")
        try:
            img = resize_and_crop(item["path"], item["bucket_w"], item["bucket_h"])
            stem = Path(item["filename"]).stem
            img.save(output_dir / f"{stem}.png", "PNG")

            caption = item.get("caption", "")
            if trigger_word and trigger_word.lower() not in caption.lower():
                caption = f"{trigger_word}, {caption}" if caption else trigger_word
            (output_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
        except Exception as e:
            print(f"[Dataset] Error processing {item['filename']}: {e}")

    if callback:
        callback(total, total, "Done!")
    return f"✅ Prepared {total} images → {output_dir}"
