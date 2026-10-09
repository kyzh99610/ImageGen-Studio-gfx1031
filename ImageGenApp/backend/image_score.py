"""
image_score.py — ⭐ for generated pictures: the cheap CPU checks that separated good from broken ones in the audits.
Nothing is downloaded here: a check only runs when its model is already in the Hugging Face cache.

  * anatomy   — the anime face / hand detectors (backend/detail_tools): exactly one face, at most two hands
  * look      — WD14 on the *head crop* (in the whole 832×1216 picture a hair pin or the eyes are a few pixels): the hair /
                eye colours the character card names (backend/identity_check) and its hair accessory ("butterfly hair
                ornament": WD14 on the head crop separated pictures with and without the pin at AUC 0.99)
  * artefacts — saturated colour specks in desaturated areas: the iridescent noise an unstable sampler leaves

A flag costs a star (no face at all: two); 5 stars = nothing found, 1 = several problems.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from backend import identity_check as ic
from backend.prompt_tools import parse_tag, split_tags

# card tags that name a hair accessory (WD14 knows these as *_hair_ornament, hairclip, hair_bow …)
ACCESSORY_WORDS = ("hair ornament", "hairclip", "hair clip", "hairpin", "hair pin", "hair flower", "hair bow",
                   "hair ribbon", "hairband", "hair bell", "hair stick")
ACCESSORY_THRESHOLD = 0.5          # WD14 probability on the head crop under which the pin counts as missing
SPECK_LIMIT = 0.0005               # share of specks in the picture above which colour noise is reported
HEAD_SCALE = 2.4                   # head crop = this many face widths


def accessories(tags: str) -> list[tuple[str, list[str]]]:
    """[(card tag, [WD14 tag names it may appear as, most specific first])] for the hair accessories a card's tags name:
    "black butterfly hair ornament" → black_butterfly_hair_ornament, butterfly_hair_ornament (not the bare hair_ornament,
    which any other ornament would satisfy)."""
    out, seen = [], set()
    for raw in split_tags(tags or ""):
        key = parse_tag(raw)[0]
        if key in seen or not any(w in key for w in ACCESSORY_WORDS):
            continue
        seen.add(key)
        words = key.split()
        floor = 3 if len(words) >= 3 else len(words)
        out.append((key, ["_".join(words[i:]) for i in range(0, len(words) - floor + 1)]))
    return out


def head_crop(img: Image.Image, box, scale: float = HEAD_SCALE) -> Image.Image:
    """A square crop around a face box, `scale` face widths wide, kept inside the picture."""
    W, H = img.size
    x1, y1, x2, y2 = box
    side = int(min(max(x2 - x1, y2 - y1) * scale, W, H))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    left = min(max(0, cx - side // 2), W - side)
    top = min(max(0, cy - side // 2), H - side)
    return img.crop((left, top, left + side, top + side))


def colour_specks(img: Image.Image) -> float:
    """Share of pixels that are saturated specks in a desaturated neighbourhood (grey / white fabric, hair, background)."""
    import cv2
    a = np.asarray(img.convert("RGB"), np.uint8)
    k = 1216.0 / max(a.shape[:2])
    if abs(k - 1) > 0.02:
        a = cv2.resize(a, (round(a.shape[1] * k), round(a.shape[0] * k)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.float32)
    S, V = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    near = cv2.medianBlur((S * 255).astype(np.uint8), 7).astype(np.float32) / 255.0
    return float(((S > 0.45) & (near < 0.18) & (V > 0.25)).mean())


def _ids():
    from backend import identity_score
    return identity_score


def _cached(repo: str, file: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo, file), str)
    except Exception:
        return False


def available() -> dict:
    """Which checks can run here (models already cached): {'look': WD14, 'faces': …, 'hands': …, 'identity': CCIP}."""
    from backend import detail_tools as dt
    from backend import identity_score as ids
    return {"look": ic.available(), "faces": _cached(dt._YOLO_REPO, dt._YOLO_FILE), "hands": _cached(dt._HAND_REPO, dt._HAND_FILE),
            "identity": ids.available()}


def score_images(images, tags: str, *, probs_fn=None, faces_fn=None, hands_fn=None, speck_fn=colour_specks,
                 identity=None, feature_fn=None, pause=None, identity_z=None) -> list[dict]:
    """Per image {'stars': 1..5, 'flags': ['no face found', 'grey hair 0.12', 'no butterfly hair ornament 0.31', …]}.
    The detectors default to the app's own, used only for the checks `available()` reports; pass the *_fn arguments to
    replace them (tests). A check that can't run is simply skipped. `identity` = a card's identity (backend/identity_score):
    a face further from her references than her own pictures are (and at least 100 px wide) costs a star.
    `identity_z` = per picture the number of spreads below her references' mean that flags it (None / missing = identity_score.Z_FLAG; the outfit
    batch passes Z_FLAG_HAIR for outfits with a hat or hairstyle).
    `pause(k, n)` is called before picture k of n (the app's "pause while hot": scoring is CPU work, and on the laptop the
    CPU die is what the thermal zone follows)."""
    can = available() if not (probs_fn and faces_fn and hands_fn) else {"look": True, "faces": True, "hands": True}
    ident = identity if identity and (feature_fn is not None or can.get("identity", True) and _ids().available()) else None
    if probs_fn is None and can["look"]:
        probs_fn = ic.probs
    if faces_fn is None and can["faces"]:
        from backend.detail_tools import detect_faces
        faces_fn = lambda im: detect_faces(im, "anime")
    if hands_fn is None and can["hands"]:
        from backend.detail_tools import detect_hands
        hands_fn = detect_hands
    traits, acc = ic.traits(tags), accessories(tags)
    out, images = [], list(images)
    for k, im in enumerate(images):
        if pause is not None:
            pause(k, len(images))
        stars, flags = 5, []
        faces = faces_fn(im) if faces_fn else None
        if faces is not None:
            if not faces:
                stars -= 2
                flags.append("no face found")
            elif len(faces) > 1:
                stars -= 1
                flags.append(f"{len(faces)} faces")
        if hands_fn:
            n = len(hands_fn(im))
            if n > 2:
                stars -= 1
                flags.append(f"{n} hands")
        if probs_fn and (traits or acc):
            p = probs_fn(head_crop(im, faces[0]) if faces else im)
            for kind, (colour, names) in traits.items():
                v = max(p.get(n, 0.0) for n in names)
                if v < ic.THRESHOLD:
                    stars -= 1
                    flags.append(f"{colour} {kind} {v:.2f}")
            for key, names in acc:
                names = [n for n in names if n in p]
                if not names:                       # WD14 has no tag for it: can't judge
                    continue
                v = max(p[n] for n in names)
                if v < ACCESSORY_THRESHOLD:
                    stars -= 1
                    flags.append(f"no {key} {v:.2f}")
        if ident and faces:
            r = _ids().head_feature(im, box=faces[0], feature_fn=feature_fn, min_face_px=_ids().MIN_FACE_PX)      # a small face is not judged
            zf = identity_z[k] if identity_z is not None and k < len(identity_z) and identity_z[k] else _ids().Z_FLAG
            res = _ids().check(ident, r[0], r[1], z_flag=zf) if r is not None else None
            if res and res["flag"]:
                stars -= 1
                flags.append(f"not her? face {res['cos']:.2f} (her pictures {res['mean']:.2f})")
        if speck_fn and speck_fn(im) > SPECK_LIMIT:
            stars -= 1
            flags.append("colour noise")
        out.append({"stars": max(1, stars), "flags": flags})
    return out


def badge(img: Image.Image, result: dict) -> Image.Image:
    """A copy of the picture with its rating (n/5) and flags written into the corner — for the contact sheet."""
    from PIL import ImageDraw, ImageFont
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    size = max(18, im.width // 14)
    try:
        font, small = ImageFont.truetype("arial.ttf", size), ImageFont.truetype("arial.ttf", max(12, size // 2))
    except Exception:
        font = small = ImageFont.load_default()
    colour = {5: (166, 227, 161), 4: (166, 227, 161), 3: (249, 226, 175)}.get(result["stars"], (243, 139, 168))
    d.rectangle((0, 0, size * 3, int(size * 1.4)), fill=(17, 17, 27))
    d.text((6, 2), f"{result['stars']}/5", fill=colour, font=font)
    for i, f in enumerate(result["flags"][:4]):
        d.text((6, int(size * 1.5) + i * (size // 2 + 6)), f[:40], fill=(17, 17, 27), font=small, stroke_width=2, stroke_fill=colour)
    return im
