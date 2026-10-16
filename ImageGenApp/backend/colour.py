"""🎨 Colour strength: a post-hoc colour scale of a finished picture (round 25). Pure (numpy + OpenCV), no model, no GPU.

`scale_colour(image, strength)` multiplies the CIE-LAB chroma (a / b) of every pixel by `strength` (1.0 = unchanged, 0.8 = 20 % less colour) and leaves the lightness L alone — so shading, line art and
contrast stay as drawn and a skin tone keeps its hue (only gets paler); an RGB / HSV saturation scale shifts the lightness of the colours it touches (measured in round 25).
It runs once, on the finished picture, after every pass (hires / face / hand / eye / max detail), so no pass reads a scaled picture."""
from __future__ import annotations

import numpy as np
from PIL import Image

METHOD = "lab-chroma"
MIN_STRENGTH = 0.5


def clamp_strength(value, default: float = 1.0) -> float:
    """A strength in [MIN_STRENGTH, 1.0]; anything unreadable or above 1 means 'off' (1.0)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:                                  # NaN
        return default
    return min(1.0, max(MIN_STRENGTH, v))


def scale_colour(image: Image.Image, strength: float) -> Image.Image:
    """The picture with its LAB chroma scaled by `strength`; strength >= 1 returns the picture itself. Alpha is kept."""
    s = clamp_strength(strength)
    if s >= 1.0:
        return image
    import cv2
    alpha = image.getchannel("A") if image.mode == "RGBA" else None
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)                    # float32: L 0..100, a / b about -128..127
    lab[..., 1:] *= s
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)
    res = Image.fromarray(np.rint(out * 255.0).astype(np.uint8), "RGB")
    if alpha is not None:
        res.putalpha(alpha)
    return res
