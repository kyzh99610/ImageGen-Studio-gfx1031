"""
Character cards: one JSON file per character in settings/characters/ with everything that
makes that character come out right — checkpoint, LoRAs + weights, the character's own tags,
named outfits (tag sets), an optional negative, size, CFG, steps, sampler and CLIP skip.
"Build from LoRA" fills one in from the LoRA's Civitai trigger prompts and training tags.
A card may also carry `profiles`: per-checkpoint overrides (LoRA weights, CFG, sampler, boosters …) that
🎴 Load applies when the checkpoint selected in the Generate tab has one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

try:
    from config import SETTINGS_DIR
except Exception:                                   # standalone use / tests
    SETTINGS_DIR = Path(__file__).resolve().parent.parent / "settings"

CARDS_DIR = SETTINGS_DIR / "characters"

# tag words that describe the character's body (kept with the character, not an outfit)
_BODY_WORDS = ("hair", "eyes", "bangs", "ahoge", "ears", "tail", "horns", "wings", "mole", "freckles",
               "sidelocks", "eyebrows", "fang", "skin", "halo", "pupils", "eyelashes")
# a signature hair pin is part of the look, but taggers miss small ornaments (one character LoRA's training captions
# have "butterfly hair ornament" in 58 % of the images): these tags count from half the images, not 60 %
_ACCESSORY_WORDS = ("hair ornament", "hairclip", "hair clip", "hairpin", "hair pin", "hair flower", "hair bow",
                    "hair ribbon", "hairband", "hair bell", "hair stick")
# words that name clothing (used to name outfits after their most telling pieces)
_CLOTHES_WORDS = ("dress", "bikini", "swimsuit", "jacket", "coat", "uniform", "kimono", "yukata", "shirt",
                  "skirt", "shorts", "pants", "headwear", "hat", "headdress", "choker", "gloves", "thighhighs",
                  "pantyhose", "leotard", "apron", "cape", "armor", "hoodie", "sweater", "boots", "bodysuit",
                  "cardigan", "vest", "necktie", "ribbon", "veil", "maid", "serafuku", "hairband", "bow")
_OPTIONAL_KEYS = ("vae", "negative", "width", "height", "cfg", "steps", "scheduler", "clip_skip", "notes")


def safe_name(name) -> str:
    """A card name usable as a file name ('' if nothing is left)."""
    n = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name or "")).strip().lstrip("._ ").rstrip(". ")
    if n.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                                   *(f"LPT{i}" for i in range(1, 10))}:
        n = "_" + n
    return n[:80]


def _num(v, lo, hi, cast):
    try:
        v = cast(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if lo <= v <= hi else None


def _clean_loras(raw) -> list:
    loras = []
    for l in raw or []:
        if isinstance(l, dict) and isinstance(l.get("file"), str) and l["file"].strip():
            w = _num(l.get("weight", 0.8), -3.0, 3.0, float)
            loras.append({"file": Path(l["file"]).name, "weight": 0.8 if w is None else round(w, 3)})
    return loras[:3]


def _fill_settings(d: dict, out: dict) -> dict:
    """The optional numeric / text settings of a card or a profile, sanitised, into `out`."""
    as_int = lambda v: int(round(float(v)))      # "832.7" / 1216.0 from a hand-edited file
    for k, lo, hi, cast in (("width", 256, 2048, as_int), ("height", 256, 2048, as_int), ("cfg", 1.0, 30.0, float),
                            ("steps", 1, 150, as_int), ("clip_skip", 1, 4, as_int)):
        if d.get(k) is not None:
            v = _num(d.get(k), lo, hi, cast)
            if v is not None:
                out[k] = (v // 8 * 8) if k in ("width", "height") else v
    for k in ("vae", "negative", "scheduler", "notes"):
        if isinstance(d.get(k), str) and d[k].strip():
            out[k] = Path(d[k]).name if k == "vae" else d[k].strip()
    return out


# quality boosters a per-checkpoint profile may carry (0 = off; face / hand detail = the denoise strength)
_BOOSTERS = (("pag", 0.0, 6.0), ("cfg_rescale", 0.0, 1.0), ("face_detail", 0.0, 0.8), ("hand_detail", 0.0, 0.7))


def _clean_profile(p) -> dict | None:
    """One checkpoint's overrides of the card (LoRAs as a whole list, settings, boosters); None if nothing valid is left."""
    if not isinstance(p, dict):
        return None
    out = {"loras": _clean_loras(p["loras"])} if isinstance(p.get("loras"), list) else {}
    _fill_settings(p, out)
    for k, lo, hi in _BOOSTERS:
        if p.get(k) is not None:
            v = _num(p.get(k), lo, hi, float)
            if v is not None:
                v = round(v, 3)
                out[k] = max(v, 0.1) if k.endswith("_detail") and v > 0 else v      # the sliders start at 0.1
    if isinstance(p.get("freeu"), bool):
        out["freeu"] = p["freeu"]
    return out or None


def card_for_checkpoint(card: dict, checkpoint) -> tuple[dict, str | None]:
    """The card with the profile for `checkpoint` (file name or path) laid over it, and that profile's key;
    (card, None) when the card has no profile for it."""
    want = Path(str(checkpoint or "")).name.lower()
    for key, prof in (card.get("profiles") or {}).items():
        if want and (key.lower() == want or Path(key).stem.lower() == Path(want).stem):
            merged = {k: v for k, v in card.items() if k != "profiles"}
            merged.update(prof)                       # a profile's LoRA list replaces the card's as a whole
            merged["checkpoint"] = key
            return merged, key
    return card, None


def clean_card(d) -> dict | None:
    """A card with only known, sane fields (None if it isn't a card at all)."""
    if not isinstance(d, dict):
        return None
    name = safe_name(d.get("name"))
    if not name:
        return None
    loras = _clean_loras(d.get("loras"))
    outfits = {}
    raw = d.get("outfits") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(k, str) and k.strip() and isinstance(v, str):
                outfits[k.strip()[:80]] = v.strip()
    card = {"name": name,
            "checkpoint": Path(d["checkpoint"]).name if isinstance(d.get("checkpoint"), str) and d["checkpoint"] else None,
            "loras": loras,
            "tags": d.get("tags") if isinstance(d.get("tags"), str) else "",
            "outfits": outfits}
    _fill_settings(d, card)
    profiles = {}
    if isinstance(d.get("profiles"), dict):
        for ck, p in d["profiles"].items():
            prof = _clean_profile(p)
            if isinstance(ck, str) and Path(ck).name and prof:
                profiles[Path(ck).name] = prof
    if profiles:
        card["profiles"] = profiles
    return card


def list_cards() -> list[str]:
    if not CARDS_DIR.is_dir():
        return []
    return sorted((p.stem for p in CARDS_DIR.glob("*.json")), key=str.lower)


def load_card(name: str) -> dict | None:
    n = safe_name(name)
    f = CARDS_DIR / f"{n}.json"
    if not n or not f.is_file():
        return None
    try:
        card = clean_card(json.loads(f.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    if card:
        card["name"] = n            # the file name is the card's name
    return card


def save_card(card: dict) -> Path:
    c = clean_card(card)
    if not c:
        raise ValueError("A character card needs a name.")
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    f = CARDS_DIR / f"{c['name']}.json"
    f.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    return f


def _k(t: str) -> str:
    return " ".join(t.replace("_", " ").lower().split())


def outfit_label(tags: list[str]) -> str:
    """'dress · black gloves · black jacket' — the clothing tags of an outfit, else its first tags."""
    clothes = [t for t in tags if any(w in _k(t).split() or _k(t).endswith(w) for w in _CLOTHES_WORDS)]
    return " · ".join((clothes or tags)[:3]) or "outfit"


def card_from_lora(lora_path: str, weight: float = 0.8, checkpoint: str | None = None,
                   name: str | None = None) -> dict | None:
    """A card built from a LoRA: its trigger words + body tags as the character's tags, and
    each of the creator's multi-tag trigger prompts (minus those tags) as an outfit."""
    from backend.lora_keywords import lora_keywords
    from backend.prompt_tools import split_tags
    kw = lora_keywords(lora_path)
    if kw is None:
        return None
    tags = list(kw.triggers) + [t for t in kw.likely if t not in kw.triggers]
    for t, cov in kw.tags:
        need = 0.5 if any(w in _k(t) for w in _ACCESSORY_WORDS) else 0.6
        if cov >= need and any(w in _k(t) for w in _BODY_WORDS) and _k(t) not in {_k(x) for x in tags}:
            tags.append(t)
        if len(tags) >= 10:
            break
    # "1girl" / "1boy" / "solo" when (nearly) every training image has them
    subject = [t for t, cov in kw.tags if cov >= 0.6 and _k(t) in ("1girl", "1boy", "solo", "2girls")]
    tags = subject + [t for t in tags if _k(t) not in {_k(x) for x in subject}]
    have = {_k(t) for t in tags}
    outfits = {}
    for ph in kw.phrases:
        rest = [t for t in split_tags(ph) if _k(t) not in have]
        if rest:
            label = outfit_label(rest)
            while label in outfits:
                label += "′"
            outfits[label] = ", ".join(rest)
    stem = Path(str(lora_path)).stem
    nice = re.sub(r"^(il|pony|xl|sdxl|sd15)[_\- ]+", "", stem, flags=re.I)
    return clean_card({
        "name": name or nice.replace("_", " ").strip() or stem,
        "checkpoint": Path(checkpoint).name if checkpoint and Path(str(checkpoint)).suffix else None,
        "loras": [{"file": Path(str(lora_path)).name, "weight": weight}],
        "tags": ", ".join(tags),
        "outfits": outfits,
    })


def card_prompt(card: dict, outfit: str | None = None, scene: str = "") -> str:
    """The card's character tags + the chosen outfit + anything else (scene, pose…)."""
    from backend.prompt_tools import merge_prompts
    parts = [card.get("tags", "")]
    if outfit and outfit in (card.get("outfits") or {}):
        parts.append(card["outfits"][outfit])
    parts.append(scene or "")
    parts = [p for p in parts if p and p.strip()]
    return merge_prompts(*parts) if parts else ""
