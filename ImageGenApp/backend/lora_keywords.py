"""
lora_keywords.py — the words a LoRA (or checkpoint) responds to, ranked by how strongly it
learned them, so the Generate tab can offer them as one-click chips.

Sources, best first:
  1. Civitai sidecar (<file>.civitai.json → trainedWords): the creator's trigger words.
     Often whole prompts ("mychar, red eyes, grey hair") — split into tags.
  2. kohya training metadata in the .safetensors header: ss_tag_frequency (how many training
     images carried each caption tag) and ss_dataset_dirs (images per folder).
     coverage = images with the tag / training images. A tag present in ~every image is what
     the LoRA ties its concept to — for a character LoRA that's the name, hair, eyes and
     outfit; the more consistent a tag was, the more reliably it recalls the character.
     Without Civitai trigger words, rare-ish tags (not generic danbooru tags) with ≥ 90 %
     coverage are shown as likely triggers.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

# Caption tags that say nothing about what a LoRA learned (quality/meta/rating tags)
_META = {
    "highres", "absurdres", "lowres", "commentary", "commentary request", "english commentary",
    "translated", "translation request", "signature", "watermark", "artist name", "dated",
    "web address", "twitter username", "patreon username", "official art", "bad id", "bad pixiv id",
    "masterpiece", "best quality", "high quality", "4k", "8k", "monochrome", "greyscale",
    "simple background", "white background", "transparent background", "multiple views",
}
# Common tags that are never *the* trigger even at 100 % coverage
_GENERIC = {
    "1girl", "1boy", "2girls", "solo", "solo focus", "breasts", "large breasts", "medium breasts",
    "small breasts", "long hair", "short hair", "looking at viewer", "blush", "smile", "open mouth",
    "simple background", "navel", "thighs", "bangs",
    "closed mouth", "upper body", "full body", "standing", "sitting", "indoors", "outdoors", "hair between eyes",
}


# "white hair", "red eyes", "maid dress": attributes, not names — shown as chips, never as the trigger
_ATTRIBUTE_NOUNS = {"hair", "eyes", "dress", "skin", "breasts", "ears", "tail", "hairband", "gloves",
                    "thighhighs", "shirt", "skirt", "uniform", "background", "focus", "horns", "wings"}


@dataclass
class Keywords:
    name: str
    triggers: list[str] = field(default_factory=list)       # creator / inferred trigger words
    phrases: list[str] = field(default_factory=list)        # creator's multi-tag trigger prompts (outfits…)
    likely: list[str] = field(default_factory=list)         # name-like tags in ≥ 90 % of the training images
    tags: list[tuple[str, float]] = field(default_factory=list)   # (tag, coverage 0–1), best first
    n_images: int = 0
    inferred: bool = False                                  # triggers guessed from coverage
    source: str = ""                                        # "civitai" / "training tags" / ""

    def core(self, min_cov: float = 0.5, limit: int = 12) -> list[str]:
        """Triggers + the tags present in at least min_cov of the training images."""
        out = list(self.triggers) + [t for t in self.likely if t not in self.triggers]
        for t, c in self.tags:
            if c >= min_cov and t not in out and _key(t) not in _META:
                out.append(t)
            if len(out) >= limit + len(self.triggers):
                break
        return out


def _key(t: str) -> str:
    return " ".join(t.replace("_", " ").replace("\\(", "(").replace("\\)", ")").lower().split())


def _header_meta(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 2 or n > 100 << 20:
                return {}
            meta = json.loads(f.read(n)).get("__metadata__") or {}
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _sidecar(path: Path) -> dict:
    for p in (Path(str(path) + ".civitai.json"), path.with_suffix(".civitai.json")):
        if p.is_file():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
    return {}


_CACHE: dict[tuple, Keywords] = {}


def lora_keywords(path: str) -> Keywords | None:
    """Keywords for a .safetensors LoRA/checkpoint (None for 'none' / missing files)."""
    if not path or path == "none":
        return None
    p = Path(str(path))
    if not p.is_file():
        return None
    side = Path(str(p) + ".civitai.json")
    key = (str(p), p.stat().st_mtime, side.stat().st_mtime if side.exists() else 0)
    if key in _CACHE:
        return _CACHE[key]
    kw = Keywords(name=p.stem)

    sc = _sidecar(p)
    from backend.prompt_tools import split_tags
    phrases = [split_tags(ph) for ph in (sc.get("trainedWords") or []) if isinstance(ph, str) and ph.strip()]
    phrases = [ph for ph in phrases if ph]
    if phrases:
        kw.source = "civitai"
        multi = [ph for ph in phrases if len(ph) > 1]
        # Tags every multi-tag phrase shares are the trigger (the character); each whole
        # phrase is one of the creator's ready-made prompts (usually one per outfit).
        shared = [t for t in multi[0] if all(_key(t) in {_key(x) for x in ph} for ph in multi[1:])] if multi else []
        for t in shared + [ph[0] for ph in phrases if len(ph) == 1]:
            if _key(t) not in {_key(x) for x in kw.triggers}:
                kw.triggers.append(t)
        if not kw.triggers and multi:
            kw.triggers.append(multi[0][0])
        seen_ph = set()
        for ph in multi:
            txt = ", ".join(ph)
            if txt.lower() not in seen_ph:
                seen_ph.add(txt.lower())
                kw.phrases.append(txt)

    meta = _header_meta(p) if p.suffix.lower() == ".safetensors" else {}
    raw = meta.get("ss_tag_frequency")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = None
    counts: dict[str, int] = {}
    spelling: dict[str, str] = {}
    if isinstance(raw, dict):
        for ds in raw.values():
            if not isinstance(ds, dict):
                continue
            for t, c in ds.items():
                t = str(t).strip()
                k = _key(t)
                if not k or len(k) > 60 or k.startswith(("rating", "score ")):
                    continue
                try:
                    counts[k] = counts.get(k, 0) + int(c)
                except (TypeError, ValueError):
                    continue
                spelling.setdefault(k, t)
    if counts:
        n = 0
        dirs = meta.get("ss_dataset_dirs")
        try:
            dirs = json.loads(dirs) if isinstance(dirs, str) else dirs
            n = sum(int(v.get("img_count", 0)) for v in (dirs or {}).values() if isinstance(v, dict))
        except Exception:
            n = 0
        n = max(n, max(counts.values()))       # a tag can't be in more images than exist
        kw.n_images = n
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        kw.tags = [(spelling[k], min(1.0, c / n)) for k, c in ranked if k not in _META][:60]
        if not kw.source:
            kw.source = "training tags"
        # Name-like tags in (almost) every training image are what the LoRA actually learned —
        # even when Civitai lists another spelling (a caption typo is the real trigger then)
        kw.likely = [t for t, c in kw.tags if c >= 0.9 and _key(t) not in _GENERIC
                     and _key(t).split(" ")[-1] not in _ATTRIBUTE_NOUNS
                     and _key(t) not in {_key(x) for x in kw.triggers}][:3]
        if not kw.triggers and kw.likely:
            kw.triggers, kw.likely, kw.inferred = kw.likely[:2], kw.likely[2:], True
    _CACHE[key] = kw
    return kw


def chips_for(slots: list[tuple[str, str]]) -> tuple[list[list[str]], list[list], str]:
    """Chip labels + the tag each chip inserts, for [(slot label, file path), …].
    Returns (dataset samples, [[tag, goes_up_front], …], a short HTML note)."""
    samples, tags, notes = [], [], []
    seen: set[str] = set()
    for label, path in slots:
        kw = lora_keywords(path)
        if kw is None:
            continue
        if not kw.triggers and not kw.tags:
            if label == "Model":          # most checkpoints carry no keyword data — not worth a note
                continue
            notes.append(f"{label} <b>{_esc(kw.name)}</b>: no keyword data (try 🔍 Fetch trigger words from Civitai)")
            continue
        for i, ph in enumerate(kw.phrases):
            n = len(ph.split(","))
            samples.append([f"{label} 📋 prompt {i + 1}: {ph[:48]}{'…' if len(ph) > 48 else ''} ({n} tags)"])
            tags.append([ph, True])
        for t in kw.triggers:
            if _key(t) in seen:
                continue
            seen.add(_key(t))
            samples.append([f"{label} {'🔑' if not kw.inferred else '🗝'} {t}"])
            tags.append([t, True])
        for t in kw.likely:
            if _key(t) not in seen:
                seen.add(_key(t))
                samples.append([f"{label} 🗝 {t}"])
                tags.append([t, True])
        for t, c in kw.tags[:24]:
            if _key(t) in seen or c < 0.15:
                continue
            seen.add(_key(t))
            samples.append([f"{label} {t} · {c:.0%}"])
            tags.append([t, False])
        src = ("Civitai trigger words" + (f" + {len(kw.phrases)} ready-made prompt(s) 📋" if kw.phrases else "")
               if kw.source == "civitai" else
               "likely triggers (in ≥90 % of its training images)" if kw.inferred else kw.source)
        if kw.n_images:
            src += f", tags from {kw.n_images} training images"
        notes.append(f"{label} <b>{_esc(kw.name)}</b>: {src}")
    note = ""
    if notes:
        note = ('<p style="font-size:13px;color:#a6adc8;margin:2px 0;">🏷️ Click a keyword to add it to the '
                'prompt (duplicates are merged, the stronger weight wins). 🔑 = trigger word, 🗝 = likely trigger, '
                '📋 = the creator’s full prompt (e.g. an outfit), '
                '% = share of the training images that had the tag — higher means the LoRA ties it to its '
                'subject more strongly.<br>' + "<br>".join(notes) + "</p>")
    return samples, tags, note


def _esc(s: str) -> str:
    from html import escape
    return escape(str(s))
