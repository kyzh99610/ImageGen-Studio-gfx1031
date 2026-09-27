"""
Auto-captioning for LoRA training datasets.

Uses BLIP from Hugging Face ``transformers`` for general image captioning.
Trigger word is prepended to every caption for character LoRAs.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

from PIL import Image

_blip_model = None
_blip_processor = None


def _load_blip():
    """Lazy-load BLIP captioning model (~1 GB on first download)."""
    global _blip_model, _blip_processor
    if _blip_model is not None:
        return

    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    model_id = "Salesforce/blip-image-captioning-base"
    print("[AutoTag] Loading BLIP model …")
    _blip_processor = BlipProcessor.from_pretrained(model_id)
    _blip_model = BlipForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32)
    _blip_model.eval()
    print("[AutoTag] BLIP ready")


def caption_image(image_path: str | Path, prefix: str = "") -> str:
    """Generate a natural-language caption for one image.

    Args:
        image_path: Path to an image file.
        prefix: Optional conditional prefix (e.g. ``"an anime character"``).
    """
    import torch
    _load_blip()

    img = Image.open(image_path).convert("RGB")
    if prefix:
        inputs = _blip_processor(img, text=prefix, return_tensors="pt")
    else:
        inputs = _blip_processor(img, return_tensors="pt")

    with torch.no_grad():
        ids = _blip_model.generate(
            **inputs, max_new_tokens=75, num_beams=3,
            repetition_penalty=1.5)
    return _blip_processor.decode(ids[0], skip_special_tokens=True).strip()


def caption_batch(
    image_paths: list[str | Path],
    trigger_word: str = "",
    prefix: str = "",
    callback: Callable | None = None,
) -> dict[str, str]:
    """Caption many images.  Returns ``{filename: caption}``."""
    results: dict[str, str] = {}
    total = len(image_paths)
    for i, p in enumerate(image_paths):
        p = Path(p)
        if callback:
            callback(i, total, f"Captioning {p.name}")
        try:
            cap = caption_image(p, prefix=prefix)
            if trigger_word:
                cap = f"{trigger_word}, {cap}"
            results[p.name] = cap
        except Exception as e:
            print(f"[AutoTag] Error on {p.name}: {e}")
            results[p.name] = trigger_word or ""
    if callback:
        callback(total, total, "Done!")
    return results


def unload_blip():
    """Free BLIP from RAM."""
    global _blip_model, _blip_processor
    _blip_model = None
    _blip_processor = None
    gc.collect()
