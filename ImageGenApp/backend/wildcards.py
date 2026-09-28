"""
Dynamic prompts: `{a|b|c}` picks one option, `{2$$a|b|c}` picks two (joined with ", "),
`{3::a|b}` weights an option, and `__name__` picks a random line from wildcards/name.txt
(sub-folders: `__hair/color__`). Picks are made per image from that image's seed, so the
same seed gives the same prompt again. Groups without "|" (`{tag}`) are left untouched.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

try:
    import config as _cfg
    WILDCARD_DIRS = [_cfg.APP_DIR / "wildcards", _cfg.MODELS_DIR / "wildcards"]
except Exception:                                   # standalone use / tests
    _here = Path(__file__).resolve().parent.parent
    WILDCARD_DIRS = [_here / "wildcards", _here / "models" / "wildcards"]

_WILD = re.compile(r"__([A-Za-z0-9][\w\-/ ]*?)__")
_MAX_DEPTH = 20
_cache: dict[Path, tuple[float, list[str]]] = {}


def list_wildcards() -> list[str]:
    """Names usable as __name__ (sub-folders use "/"), first directory wins."""
    names: dict[str, None] = {}
    for d in WILDCARD_DIRS:
        if d.is_dir():
            for f in sorted(d.rglob("*.txt")):
                names.setdefault(f.relative_to(d).with_suffix("").as_posix(), None)
    return list(names)


def _lines(name: str) -> list[str] | None:
    rel = Path(*name.strip().split("/")).with_suffix(".txt")
    if ".." in rel.parts:
        return None
    for d in WILDCARD_DIRS:
        f = d / rel
        if f.is_file():
            mt = f.stat().st_mtime
            hit = _cache.get(f)
            if hit and hit[0] == mt:
                return hit[1]
            text = f.read_bytes().decode("utf-8", "replace")
            lines = [l.strip() for l in text.splitlines()]
            lines = [l for l in lines if l and not l.startswith("#")]
            _cache[f] = (mt, lines)
            return lines
    return None


def _innermost_group(s: str) -> tuple[int, int] | None:
    """(start, end) of the first {...} that has no braces inside; escaped \\{ \\} are skipped."""
    start = None
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            start = i
        elif ch == "}" and start is not None:
            return start, i
        i += 1
    return None


def _split_options(body: str) -> list[str]:
    out, buf, i = [], [], 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            buf.append(body[i:i + 2]); i += 2; continue
        if body[i] == "|":
            out.append("".join(buf)); buf = []
        else:
            buf.append(body[i])
        i += 1
    out.append("".join(buf))
    return out


def _pick_group(body: str, rng: random.Random) -> str:
    count = 1
    m = re.match(r"^\s*(\d+)(?:-(\d+))?\$\$", body)
    if m:
        lo = int(m.group(1)); hi = int(m.group(2)) if m.group(2) else lo
        count = rng.randint(min(lo, hi), max(lo, hi))
        body = body[m.end():]
    opts, weights = [], []
    for o in _split_options(body):
        wm = re.match(r"^\s*(\d*\.?\d+)::", o)
        w = float(wm.group(1)) if wm else 1.0
        opts.append((o[wm.end():] if wm else o).strip())
        weights.append(max(w, 0.0))
    if not opts or count <= 0:
        return ""
    if count == 1:
        return rng.choices(opts, weights=weights)[0] if sum(weights) > 0 else rng.choice(opts)
    idx = list(range(len(opts)))
    chosen = []
    for _ in range(min(count, len(opts))):       # without replacement, weighted
        ws = [weights[j] or 1e-9 for j in idx]
        j = rng.choices(idx, weights=ws)[0]
        idx.remove(j); chosen.append(opts[j])
    return ", ".join(c for c in chosen if c)


def is_dynamic(text: str) -> bool:
    if not text:
        return False
    if _WILD.search(text):
        return True
    s = text
    while (g := _innermost_group(s)) is not None:
        if "|" in s[g[0] + 1:g[1]].replace("\\|", ""):
            return True
        s = s[:g[0]] + "\x00" + s[g[0] + 1:g[1]] + "\x01" + s[g[1] + 1:]
    return False


def resolve(text: str, seed: int, missing: list | None = None) -> str:
    """The prompt with every {…|…} and __name__ replaced, chosen from `seed`.
    Unknown wildcards stay as written; their names are appended to `missing`."""
    if not is_dynamic(text):
        return text
    rng = random.Random(f"wildcards:{int(seed)}")
    s = text
    for _ in range(_MAX_DEPTH):
        changed = False

        def wild(m):
            nonlocal changed
            lines = _lines(m.group(1))
            if not lines:
                if missing is not None and m.group(1) not in missing:
                    missing.append(m.group(1))
                return "\x02" + m.group(1) + "\x03"          # protect, restored below
            changed = True
            return rng.choice(lines)

        s = _WILD.sub(wild, s)
        # innermost groups first, so options may contain groups and wildcards
        out = s
        guard = 0
        while (g := _innermost_group(out)) is not None and guard < 10000:
            guard += 1
            body = out[g[0] + 1:g[1]]
            if "|" in body.replace("\\|", "") or re.match(r"^\s*\d+(?:-\d+)?\$\$", body):
                rep = _pick_group(body, rng); changed = True
            else:
                rep = "\x04" + body + "\x05"                  # literal {tag}: keep
            out = out[:g[0]] + rep + out[g[1] + 1:]
        s = out
        if not changed or not is_dynamic(s):
            break
    s = s.replace("\x02", "__").replace("\x03", "__").replace("\x04", "{").replace("\x05", "}")
    # tidy the commas an empty pick leaves behind: "a, , b" → "a, b"
    s = re.sub(r"(,\s*){2,}", ", ", s)
    return re.sub(r"^\s*,\s*|\s*,\s*$", "", s)
