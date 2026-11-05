"""
neg_hygiene.py — 🧹 Negative hygiene: before a picture is made, drop the negative tags that fight what the prompt asks for,
and keep each repeated negative tag once at its strongest weight (pure: no UI, no GPU).

Why: long negatives collect tags from presets, cards, family quality tags and pasted prompts. A negative tag whose words
are all inside a prompt tag (naming its head word) works against the prompt — "long hair" in the negative while the prompt says "very long hair",
"hair ornament" against "butterfly hair ornament", "blurry" against "depth of field" — and every duplicate costs tokens
(a negative past 75 tokens is encoded in more chunks: slower, and each chunk dilutes the others).

Rules (per picture, on the resolved prompt):
- a negative tag is dropped when all of its words are words of one prompt tag ("navel" against "covered navel", but
  NOT "exposed navel": "exposed" is not in the prompt tag), or when the prompt names one of its aliases
  (DEFAULT_ALIASES + settings/neg_conflicts.json: {"negative tag": ["prompt tag", ...]});
- prompt tags that negate ("no gloves", "without hat") never cause a drop; LoRA tags, BREAK and embeddings are left alone;
- a negative tag written more than once is kept once, where it has its strongest weight; groups "(a, b, c:1.3)" are
  looked into (each item counts with the group's weight) and written back as groups.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from backend.prompt_tools import parse_tag, split_tags, join_tags

CONFIG = {"on": True}        # Settings → 🧹 Negative hygiene (saved as "neg_hygiene" in settings/_prefs.json)

# negative tag -> prompt tags that make it a conflict although they share no words
DEFAULT_ALIASES = {
    "blurry": ["depth of field", "bokeh", "blurry background", "blurry foreground", "motion blur", "chromatic aberration"],
    "see-through": ["sheer", "transparent", "translucent"],
    "greyscale": ["monochrome"], "grayscale": ["monochrome"], "monochrome": ["greyscale", "grayscale"],
    "simple background": ["white background", "black background", "grey background", "gradient background"],
    "nsfw": [],
}
MAX_TAG_WORDS = 4      # longer prompt items are phrases, not tags: they never drop a negative tag
_NEGATING = {"no", "not", "without", "non", "never"}
_WORDS = re.compile(r"[a-z0-9']+")


def set_on(v) -> bool:
    CONFIG["on"] = bool(v)
    return CONFIG["on"]


def words(key: str) -> frozenset:
    return frozenset(_WORDS.findall(key.lower()))


def load_aliases(settings_dir=None) -> dict:
    out = {k: list(v) for k, v in DEFAULT_ALIASES.items()}
    if settings_dir is not None:
        try:
            d = json.loads((Path(settings_dir) / "neg_conflicts.json").read_text(encoding="utf-8-sig"))
            for k, v in (d.items() if isinstance(d, dict) else ()):
                if isinstance(v, list):
                    out.setdefault(parse_tag(str(k))[0], []).extend(str(x) for x in v)
        except Exception:
            pass
    return out


def _flat(text: str, w: float = 1.0):
    """(key, effective weight, raw) for every item, looking into groups; groups yield their items."""
    for raw in split_tags(text or ""):
        if raw == "BREAK" or raw.startswith("<"):
            continue
        key, tw, core = parse_tag(raw)
        inner = split_tags(core) if core != raw or "," in core else [raw]
        if len(inner) > 1:
            yield from _flat(core, w * tw)
        elif key:
            yield key, w * tw, raw


def _positive_index(prompt: str):
    sets, keys = [], set()
    for key, _w, _raw in _flat(prompt):
        ws = _WORDS.findall(key.lower())
        # a negation anywhere ("no gloves", "nylon not rubber") asks for the thing to be absent: it never drops the negative
        # tag that keeps it out (before 2026-11-03 only a leading one counted, and 7 presets' negatives lost "rubber")
        if not ws or any(w in _NEGATING for w in ws) or len(ws) > MAX_TAG_WORDS:   # prose ("red iris visible under the bangs") has no head word
            continue
        sets.append((frozenset(ws), ws[-1]))      # the tag's words and its head word (the last one: 'covered NAVEL')
        keys.add(key)
    return sets, keys


def conflicts(nkey: str, pos_sets, pos_keys, aliases) -> bool:
    nw = words(nkey)
    if not nw:
        return False
    # all of the negative tag's words are in one prompt tag AND it names that tag's head word: 'navel' / 'covered navel',
    # 'long hair' / 'very long hair' - but not 'black hair' / 'black hair streak' or 'red wine' / 'wine red eyes'
    if any(nw <= ps and head in nw for ps, head in pos_sets):
        return True
    for a in aliases.get(nkey, ()):
        ak = parse_tag(a)[0]
        aw = words(ak)
        if ak in pos_keys or (aw and any(aw <= ps for ps, _h in pos_sets)):
            return True
    return False


def prune_negative(prompt: str, negative: str, aliases=None) -> tuple[str, list[str], int]:
    """(cleaned negative, the dropped tags (as written), how many repeats were merged). Unchanged text when nothing applies."""
    if not negative or not negative.strip():
        return negative or "", [], 0
    aliases = DEFAULT_ALIASES if aliases is None else aliases
    pos_sets, pos_keys = _positive_index(prompt)
    best: dict[str, float] = {}
    for key, w, _raw in _flat(negative):
        best[key] = max(w, best.get(key, w))
    dropped, merged, emitted = [], [0], set()

    def walk(text: str, outer: float) -> list[str]:
        out = []
        for raw in split_tags(text):
            if raw == "BREAK" or raw.startswith("<"):
                out.append(raw)
                continue
            key, tw, core = parse_tag(raw)
            inner = split_tags(core) if "," in core else [raw]
            if len(inner) > 1:                                   # a weighted group: filter its items, keep it a group
                kept = walk(core, outer * tw)
                kept = [k for k in kept if k != "BREAK"]
                if kept:
                    out.append(f"({', '.join(kept)}:{round(tw, 3):g})" if abs(tw - 1.0) > 1e-6 else ", ".join(kept))
                continue
            if not key:
                continue
            eff = outer * tw
            if conflicts(key, pos_sets, pos_keys, aliases):
                dropped.append(raw)
                continue
            if key in emitted or eff < best.get(key, eff) - 1e-6:
                merged[0] += 1
                continue
            emitted.add(key)
            out.append(raw)
        return out

    tags = walk(negative, 1.0)
    if not dropped and not merged[0]:
        return negative, [], 0
    cleaned = []
    for t in tags:
        if t == "BREAK" and (not cleaned or cleaned[-1] == "BREAK"):
            continue
        cleaned.append(t)
    while cleaned and cleaned[-1] == "BREAK":
        cleaned.pop()
    return join_tags(cleaned), list(dict.fromkeys(dropped)), merged[0]


def dedupe_prompt(prompt: str) -> tuple[str, int]:
    """(the prompt with each repeated tag once, how many repeats were merged). A repeat runs where the tag first appears, at its
    strongest weight (`prompt_tools.merge_prompts`: a wildcard group written twice stays two picks, BREAK and LoRA tags are kept);
    weighted groups "(a, b:1.2)" count as one item. The text is returned unchanged when nothing repeats.
    Why: the auto-quality tags already merged the prompt this way when they were added, so a prompt that wrote its own quality
    tags kept its repeats and one that didn't lost them; and a repeat is not emphasis — write one weight instead."""
    from backend.prompt_tools import merge_prompts, _DYNAMIC
    seen, n = set(), 0
    for raw in split_tags(prompt or ""):
        if raw == "BREAK" or _DYNAMIC.search(raw):
            continue
        k = parse_tag(raw)[0]
        if k:
            n += k in seen
            seen.add(k)
    return (merge_prompts(prompt), n) if n else (prompt or "", 0)


def note_html(dropped: list[str], merged: int, prompt_merged: int = 0) -> str:
    """The result line's 🧹 note, in both languages (backend/i18n.L: the page shows the chosen one)."""
    if not dropped and not merged and not prompt_merged:
        return ""
    import html
    from backend.i18n import L
    out = []
    if prompt_merged:
        out.append(L(f"🧹 Prompt: {prompt_merged} repeated tag(s) ran once, at their strongest weight",
                     f"🧹 正向提示词：{prompt_merged} 个重复的标签只用了一次（取最强的权重）"))
    parts_en, parts_zh = [], []
    if dropped:
        shown = html.escape(", ".join(dropped[:8]) + (f" … (+{len(dropped) - 8})" if len(dropped) > 8 else ""))
        parts_en.append(f"dropped {len(dropped)} tag(s) the prompt asks for: <code>{shown}</code>")
        parts_zh.append(f"去掉了 {len(dropped)} 个正向提示词要画的标签：<code>{shown}</code>")
    if merged:
        parts_en.append(f"merged {merged} repeated tag(s)")
        parts_zh.append(f"合并了 {merged} 个重复的标签")
    if parts_en:
        out.append(L(f"🧹 Negative: {' · '.join(parts_en)}", f"🧹 反向提示词：{' · '.join(parts_zh)}"))
    return "".join(f'<br><span style="color:#9399b2;">{o}</span>' for o in out)
