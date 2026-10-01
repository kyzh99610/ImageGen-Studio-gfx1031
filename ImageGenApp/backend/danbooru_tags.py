"""
Danbooru tag list (from the WD14 tagger's selected_tags.csv: ~8,100 general + ~2,750
character tags with post counts; ~300 KB, fetched on first use) for:
- autocomplete: tags starting with what is being typed, most-used first;
- "did you mean": a tag that isn't a Danbooru tag but is one typo away from one
  ("long haired" → "long hair"). Anime checkpoints trained on Danbooru captions
  (Illustrious, NoobAI, Pony, most SD 1.5 anime models) respond to the exact tags.
Tags the list doesn't know and that aren't close to one (character names, quality tags,
free text) are left alone.
"""
from __future__ import annotations

import csv
import difflib
import threading
from functools import lru_cache

REPO = "SmilingWolf/wd-vit-tagger-v3"
_lock = threading.Lock()
_tags: list[tuple[str, int, int]] | None = None     # (tag with spaces, category 0/4, count) by count
_known: set[str] = set()
_by_first: dict[str, list[int]] = {}                 # first letter → indexes into _tags (general only)


def _key(t: str) -> str:
    return " ".join(t.replace("_", " ").replace("\\(", "(").replace("\\)", ")").lower().split())


def load(download: bool = True) -> bool:
    """Load the list (from the HF cache; downloads ~300 KB once if allowed)."""
    global _tags
    with _lock:
        if _tags is not None:
            return bool(_tags)
        try:
            from huggingface_hub import hf_hub_download
            try:
                path = hf_hub_download(REPO, "selected_tags.csv", local_files_only=True)
            except Exception:
                if not download:
                    return False
                path = hf_hub_download(REPO, "selected_tags.csv")
            rows = []
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    cat = int(r["category"])
                    if cat in (0, 4):
                        rows.append((_key(r["name"]), cat, int(r["count"])))
        except Exception as e:
            print(f"[Tags] Danbooru tag list unavailable: {e}")
            _tags = []
            return False
        rows.sort(key=lambda x: -x[2])
        _tags = rows
        _known.update(t for t, _, _ in rows)
        for i, (t, cat, _) in enumerate(rows):
            if cat == 0 and t:
                _by_first.setdefault(t[0], []).append(i)
        return True


def is_known(tag: str) -> bool:
    return _key(tag) in _known


def suggest(partial: str, limit: int = 12) -> list[tuple[str, int]]:
    """(tag, post count) for tags starting with `partial` (any word of a multi-word tag
    counts after the ones that start with it), most-used first."""
    if not load(download=False):
        return []
    k = _key(partial)
    if len(k) < 2:
        return []
    starts, inner = [], []
    for t, _, n in _tags:                 # already sorted by count
        if t.startswith(k):
            starts.append((t, n))
            if len(starts) >= limit:
                break
        elif len(inner) < limit and (" " + k) in (" " + t):
            inner.append((t, n))
    return (starts + inner)[:limit]


def _is_qualified(typed: str, known: str) -> bool:
    """`typed` is `known` with extra words in front / behind ("black butterfly hair ornament" = the tag
    "butterfly hair ornament" + a colour): a deliberate qualifier, not a typo — the bare tag would drop it."""
    a, b = typed.split(), known.split()
    return len(a) > len(b) and any(a[i:i + len(b)] == b for i in range(len(a) - len(b) + 1))


@lru_cache(maxsize=4096)
def did_you_mean(tag: str) -> str | None:
    """A Danbooru tag one small typo away from `tag` (None if `tag` is fine or nothing is close)."""
    if not load(download=False):
        return None
    k = _key(tag)
    if not k or k in _known or len(k) < 4:
        return None
    pool = [_tags[i][0] for i in _by_first.get(k[0], [])]
    m = difflib.get_close_matches(k, pool, n=1, cutoff=0.88)
    return m[0] if m and m[0] != k and not _is_qualified(k, m[0]) else None
