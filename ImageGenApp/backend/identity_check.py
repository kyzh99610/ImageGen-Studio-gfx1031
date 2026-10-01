"""
identity_check.py — is she still herself? For each generated image: the WD14 probability (CPU, backend/wd_tagger) of the
hair and eye colours that the character card's tags name, so the 🎴 outfit batch can flag off-model results — scene colour
leaking into eyes / hair, a LoRA that lost the face. WD14 says "grey hair" for what Civitai calls silver, and anime LoRA
outfits rarely hide the eyes, so a probability under ~0.35 is a real miss (her own curated images score 0.85 / 0.87 on
average, the weakest 0.29–0.38 were closed or shaded eyes).

Nothing is downloaded here: the check only runs when the WD14 model is already in the Hugging Face cache.
"""
from __future__ import annotations

import re

# card word → the WD14 tag names that mean it (Danbooru has no "silver hair": WD14 calls it grey_hair / white_hair)
_HAIR = {
    "grey": ("grey_hair", "white_hair"), "gray": ("grey_hair", "white_hair"), "silver": ("grey_hair", "white_hair"),
    "white": ("white_hair", "grey_hair"), "blonde": ("blonde_hair",), "brown": ("brown_hair", "light_brown_hair"),
    "black": ("black_hair",), "blue": ("blue_hair", "light_blue_hair", "dark_blue_hair"), "pink": ("pink_hair",),
    "red": ("red_hair",), "orange": ("orange_hair",), "purple": ("purple_hair",), "green": ("green_hair",),
    "aqua": ("aqua_hair",),
}
_EYES = ("red", "blue", "green", "purple", "yellow", "pink", "brown", "black", "aqua", "orange", "white", "grey")

THRESHOLD = 0.35


def traits(tags: str) -> dict:
    """{'hair': ('grey', ['grey_hair', 'white_hair']), 'eyes': ('red', ['red_eyes'])} from a card's tag string
    ("1girl, red eyes, grey hair, long hair, …"); a trait the tags don't name is absent."""
    out: dict = {}
    for colour, kind in re.findall(r"\b([a-z]+)[ _](hair|eyes)\b", (tags or "").lower()):
        if kind == "hair" and "hair" not in out and colour in _HAIR:
            out["hair"] = (colour, list(_HAIR[colour]))
        elif kind == "eyes" and "eyes" not in out and colour in _EYES:
            out["eyes"] = (colour, [f"{colour}_eyes"])
    return out


def available() -> bool:
    """True when the WD14 model is already in the local cache (never triggers a download)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from backend.wd_tagger import REPO
        return isinstance(try_to_load_from_cache(REPO, "model.onnx"), str)
    except Exception:
        return False


def probs(img) -> dict:
    """{WD14 tag name: probability} for every tag of the vocabulary."""
    from backend import wd_tagger as W
    sess = W._load()
    inp = sess.get_inputs()[0]
    size = int(inp.shape[1]) if isinstance(inp.shape[1], int) else 448
    p = sess.run(None, {inp.name: W._prepare(img, size)})[0][0]
    return {name: float(v) for (name, _cat), v in zip(W._labels, p)}


def check(images, tags: str, threshold: float = THRESHOLD, probs_fn=None) -> list[dict]:
    """Per image {'hair': p, 'eyes': p, 'flags': ['red eyes 0.12', …]} for the traits the tags name ({} when none)."""
    t = traits(tags)
    if not t:
        return [{} for _ in images]
    fn = probs_fn or probs
    out = []
    for im in images:
        p = fn(im)
        r: dict = {"flags": []}
        for kind, (colour, names) in t.items():
            v = max(p.get(n, 0.0) for n in names)
            r[kind] = round(v, 2)
            if v < threshold:
                r["flags"].append(f"{colour} {kind} {v:.2f}")
        out.append(r)
    return out
