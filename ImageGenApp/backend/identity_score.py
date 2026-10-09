"""
identity_score.py — does this picture's face look like the card's character?  (CPU only, CCIP anime character re-identification, backend/ccip)

A card may carry an `identity`: the mean CCIP feature of the head crops of pictures of the character (chosen by the user, ≥ 3) and how much those pictures
agree. The ⭐ rating of the 🎴 outfit batch compares every new picture's head crop with it and flags a face that is further from them than her own
pictures are.

Calibrated on 247 existing pictures (130 daylight portraits of her + poses / night / neon / sunset / small faces, and a no-LoRA girl with her tags, another character with
her tags, other girls), random reference sets of 6–40 pictures, her held-out pictures against them (faces ≥ 100 px):
  · scene-invariant — difference to the references (the model's own (1 − cos) / 2): portraits 0.040, night 0.045, sunset 0.043, poses 0.041, a girl with other
    hair / eyes 0.39. (WD14's tagger feature put her in other scenes *below* a generic girl in the portrait scene and could not be used.)
  · her false alarms 3–4 % (daylight-only references) / 1–2 % (varied) at z < −2 with the floor below; the no-LoRA look-alike flagged 16–19 % / 3–9 %,
    another character 32–38 % / 4–20 %, other girls 100 %. AUC her vs look-alike 0.85–0.89, vs another character 0.94–0.96, with 6 references already.
  · faces under 100 px are not judged: her differences there are 0.07–0.18 (blurred faces), against 0.04.
So it catches a different character, a merged character and sometimes a lost LoRA (a generic look-alike is only caught about one time in five): a warning, not a verdict.
"""
from __future__ import annotations

import base64

import numpy as np
from PIL import Image

from backend.ccip import MODEL_TAG

HEAD_SCALE = 2.4          # head crop = this many face widths (as image_score)
MIN_REFS = 3              # fewer reference pictures say nothing about a character
MAX_REFS = 64
DIM = 768                 # CCIP feature size
MIN_FACE_PX = 100         # smaller faces give unreliable features: not used as references, never judged
SD_FLOOR = 0.02           # the references' own spread can't count as smaller than this (near-identical pictures would make a razor-thin tolerance)
Z_FLAG = 2.0              # flagged when the cosine is this many spreads below the references' own mean
Z_FLAG_HAIR = 5.0         # … for a picture whose outfit changes the hair / head (hat, ponytail, buns, veil …): CCIP reads hair and headwear as part of the
                          # character. Round 13, her 66 outfit pictures (all her by eye, 12 references of daylight pictures): of the 46 with such an outfit
                          # 21 (46 %) fell below z −2, 14 below −3, 10 below −4, 6 (13 %) below −5, 4 below −6 (worst −7.8); the 20 others 3 (15 %) below −2, none below
                          # −3. Another girl (other hair / eyes) sits at z −29 … −43, so that stays caught at any margin.


def available() -> bool:
    """The CCIP model is already in the Hugging Face cache (never downloads)."""
    try:
        from backend import ccip
        return ccip.available()
    except Exception:
        return False


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / (float(np.linalg.norm(v)) + 1e-9)


def encode_vec(v) -> str:
    """A 768-d vector as base64 float16 (keeps a card file small)."""
    return base64.b64encode(np.asarray(v, np.float16).astype("<f2").tobytes()).decode("ascii")


def decode_vec(s) -> np.ndarray | None:
    """The unit vector stored by `encode_vec`; None when `s` isn't one (hand-edited / damaged card)."""
    try:
        raw = base64.b64decode(str(s), validate=True)
        v = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    except Exception:
        return None
    if v.shape != (DIM,) or not np.isfinite(v).all() or not float(np.linalg.norm(v)) > 0:
        return None
    return _unit(v)


def head_crop(img: Image.Image, box, scale: float = HEAD_SCALE) -> Image.Image:
    from backend.image_score import head_crop as _hc
    return _hc(img, box, scale)


def head_feature(img: Image.Image, box=None, faces_fn=None, feature_fn=None, min_face_px: int = 0):
    """(unit CCIP feature of the head crop, face width in px) for the largest face (`box` given, or found with the anime face detector);
    None when there is no face, the face is under `min_face_px` (no feature is computed for it) or the model isn't available here."""
    if box is None:
        if faces_fn is None:
            from backend.detail_tools import detect_faces
            faces_fn = lambda im: detect_faces(im, "anime")
        faces = faces_fn(img)
        if not faces:
            return None
        box = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    if int(box[2] - box[0]) < min_face_px:
        return None
    if feature_fn is None:
        from backend import ccip
        feature_fn = ccip.features
    f = feature_fn(head_crop(img, box))
    return None if f is None else (_unit(f), int(box[2] - box[0]))


def build_identity(images, faces_fn=None, feature_fn=None) -> dict | None:
    """The `identity` of a card from pictures of the character (one face each, the largest is used; pictures without a face, or with a face under
    MIN_FACE_PX, are skipped).  {'model', 'n', 'centroid' (base64 float16), 'mean' / 'sd' (each picture's cosine to the centroid of the others)};
    None when fewer than MIN_REFS usable pictures."""
    feats = []
    for im in images:                               # a generator is fine: pictures are opened one at a time
        r = head_feature(im, faces_fn=faces_fn, feature_fn=feature_fn, min_face_px=MIN_FACE_PX)
        if r is not None:
            feats.append(r[0])
            if len(feats) >= MAX_REFS:
                break
    if len(feats) < MIN_REFS:
        return None
    F = np.stack(feats)
    loo = [float(F[i] @ _unit(np.delete(F, i, 0).mean(0))) for i in range(len(F))]
    return {"model": MODEL_TAG, "n": len(F), "centroid": encode_vec(_unit(F.mean(0))),
            "mean": round(float(np.mean(loo)), 4), "sd": round(float(np.std(loo)), 4)}


def clean_identity(d) -> dict | None:
    """A card's `identity` with only valid fields (None when there is no usable centroid)."""
    if not isinstance(d, dict) or decode_vec(d.get("centroid")) is None:
        return None
    num = lambda v, lo, hi, dflt: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= float(v) <= hi else dflt
    return {"model": d["model"] if isinstance(d.get("model"), str) else MODEL_TAG,
            "n": int(num(d.get("n"), 1, 100000, MIN_REFS)), "centroid": d["centroid"],
            "mean": round(num(d.get("mean"), -1.0, 1.0, 0.95), 4), "sd": round(num(d.get("sd"), 0.0, 1.0, 0.02), 4)}


def check(identity: dict, feat, face_px: float | None = None, z_flag: float = Z_FLAG) -> dict | None:
    """{'cos', 'mean', 'z', 'judged', 'flag'} of a picture's head feature against a card's identity — None when the identity is unusable (damaged, or
    learned with another model) or there is no feature. `judged` is False, and `flag` never set, for a face under MIN_FACE_PX (None = size unknown: judged)."""
    centroid = decode_vec((identity or {}).get("centroid"))
    if centroid is None or feat is None or identity.get("model") != MODEL_TAG:
        return None
    cos = float(_unit(feat) @ centroid)
    mean, sd = float(identity.get("mean", 0.95)), float(identity.get("sd", SD_FLOOR))
    z = (cos - mean) / max(sd, SD_FLOOR)
    judged = face_px is None or face_px >= MIN_FACE_PX
    return {"cos": round(cos, 3), "mean": round(mean, 3), "z": round(z, 2), "judged": judged, "flag": bool(judged and z < -z_flag)}
