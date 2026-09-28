"""
prompt_tools.py — tag-level prompt handling: merge without duplicates, count CLIP tokens,
and split long prompts into 75-token chunks at tag boundaries.

Why:
  • Presets / quick tags / LoRA keywords used to be appended as text, so "masterpiece" or
    "(best quality:1.2)" could end up in the prompt three times at different weights.
    merge_prompts() keeps every tag once, at its STRONGEST weight, in first-seen order.
  • CLIP reads 75 tokens at a time. Long prompts are encoded chunk by chunk (nothing is
    dropped), but Compel cut chunks at exactly 75 tokens — often in the middle of a tag,
    so half of "long silver hair" landed in the next chunk with a different context.
    chunk_prompt() moves the whole tag to the next chunk instead (like A1111), and BREAK
    starts a new chunk on purpose.

Tag syntax understood (A1111 + Compel): (x), ((x)), (x:1.3), [x], (x)1.3, x++, x--,
<lora:name:0.8>, \\( escaped \\) parentheses, BREAK.
"""
from __future__ import annotations

import re
from functools import lru_cache

CHUNK_TOKENS = 75            # CLIP window 77 minus the start/end tokens
_A1111_W = re.compile(r"^(.*?):\s*(-?\d*\.?\d+)\s*$", re.S)
_COMPEL_W = re.compile(r"^\((.*)\)(\d*\.?\d+)$", re.S)
_LORA = re.compile(r"^<(lora|lyco|hypernet):([^:>]+)(?::([-\d.]+))?[^>]*>$", re.I)
_BREAK = re.compile(r"\bBREAK\b")


# ── Splitting ─────────────────────────────────────────────────────────────────
def split_tags(prompt: str) -> list[str]:
    """Comma-separated tags, respecting brackets and escapes: "(a, b:1.2), c" → 2 tags.
    A standalone BREAK becomes its own "BREAK" item."""
    out: list[str] = []
    for i, seg in enumerate(_BREAK.split(prompt or "")):
        if i:
            out.append("BREAK")
        buf, depth, j = [], 0, 0
        while j < len(seg):
            ch = seg[j]
            if ch == "\\" and j + 1 < len(seg):
                buf.append(seg[j:j + 2]); j += 2; continue
            if ch in "([<{":
                depth += 1
            elif ch in ")]>}" and depth:
                depth -= 1
            if ch in ",\n" and depth == 0:
                out.append("".join(buf)); buf = []
            else:
                buf.append(ch)
            j += 1
        out.append("".join(buf))
    return [t.strip() for t in out if t.strip()]


def join_tags(tags: list[str]) -> str:
    text = ", ".join(tags)
    return text.replace(", BREAK, ", " BREAK ").replace("BREAK, ", "BREAK ").replace(", BREAK", " BREAK")


def _outer_pair(s: str, open_ch: str, close_ch: str) -> bool:
    """True if s is wrapped by ONE matching pair: "(a) (b)" is not."""
    if not (s.startswith(open_ch) and s.endswith(close_ch)):
        return False
    depth, i = 0, 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2; continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
        i += 1
    return depth == 0


def parse_tag(tag: str) -> tuple[str, float, str]:
    """(key, weight, core text). key identifies the tag regardless of weighting and of
    "_" vs " " / letter case; core is the text without any weight syntax."""
    t = tag.strip()
    m = _LORA.match(t)
    if m:
        w = float(m.group(3)) if m.group(3) and re.fullmatch(r"-?\d*\.?\d+", m.group(3)) else 1.0
        return f"<{m.group(1).lower()}:{m.group(2).strip().lower()}>", w, t
    w = 1.0
    for _ in range(50):
        m = _COMPEL_W.match(t)
        if m and _outer_pair("(" + m.group(1) + ")", "(", ")"):
            w *= float(m.group(2)); t = m.group(1).strip(); continue
        pm = re.match(r"^(.*?)(\++|-+)$", t)
        if pm and pm.group(1) and not pm.group(1).endswith(("\\", " ")) and pm.group(1)[-1] not in "+-":
            n = len(pm.group(2))
            w *= (1.1 ** n) if pm.group(2)[0] == "+" else (0.9 ** n)
            t = pm.group(1).strip()          # "(x:1.2)++": the loop unwraps the rest
            continue
        if _outer_pair(t, "(", ")"):
            inner = t[1:-1].strip()
            m = _A1111_W.match(inner)
            if m and not m.group(1).endswith("\\"):
                w *= float(m.group(2)); t = m.group(1).strip()
            else:
                w *= 1.1; t = inner
            continue
        if _outer_pair(t, "[", "]"):
            w /= 1.1; t = t[1:-1].strip(); continue
        break
    key = re.sub(r"\s+", " ", t.replace("\\(", "(").replace("\\)", ")").replace("_", " ")).strip().lower()
    return key, w, t


def format_tag(core: str, weight: float) -> str:
    if core.startswith("<"):
        return core
    if abs(weight - 1.0) < 0.005:
        return core
    return f"({core}:{round(weight, 2):g})"


_DYNAMIC = re.compile(r"\{[^{}]*\||(?<![\w])__[A-Za-z0-9][\w\-/ ]*?__(?![\w])")


def _base_keys(base: str) -> list[tuple[str, float, str]]:
    """(key, weight, raw) for the base prompt's tags. A wildcard group written twice
    ("{a|b|c}, {a|b|c}") means two picks, so repeats of one get their own keys."""
    out, seen = [], set()
    for raw in split_tags(base):
        if raw == "BREAK":
            continue
        k, w, _ = parse_tag(raw)
        if k in seen and _DYNAMIC.search(raw):
            k = f"{k}#{len(out)}"
        seen.add(k)
        out.append((k, w, raw))
    return out


def merge_prompts(base: str, *additions: str) -> str:
    """base + additions, every tag once at its strongest weight, first-seen order.
    Unchanged tags keep the user's own spelling; BREAK markers are kept."""
    # Fast path: nothing in the additions changes an existing tag → keep the user's text as
    # typed (line breaks, spacing) and just append the new tags.
    base_w: dict[str, float] = {}
    bk = _base_keys(base)
    for k, w, _ in bk:
        base_w[k] = max(w, base_w.get(k, w))
    if len(base_w) == len(bk):   # base has no duplicates
        new, new_w, touches = [], {}, False
        for text in additions:
            for raw in split_tags(text):
                if raw == "BREAK":
                    touches = True; break
                k, w, _ = parse_tag(raw)
                if k in base_w:
                    if w > base_w[k] + 1e-6:
                        touches = True; break
                elif k in new_w:
                    if w > new_w[k] + 1e-6:          # the additions repeat a tag more strongly
                        touches = True; break
                elif k:
                    new.append(raw); new_w[k] = w
            if touches:
                break
        if not touches:
            head = (base or "").rstrip().rstrip(",").rstrip()
            return head + (", " if head and new else "") + ", ".join(new) if new else (base or "")
    order: list[str] = []
    best: dict[str, tuple[float, str]] = {}
    for n_text, text in enumerate((base, *additions)):
        seen_here: set[str] = set()
        for raw in split_tags(text):
            if raw == "BREAK":
                k = f"BREAK#{len(order)}"
                order.append(k); best[k] = (1.0, "BREAK")
                continue
            key, w, core = parse_tag(raw)
            if not key:
                continue
            if n_text == 0 and key in seen_here and _DYNAMIC.search(raw):
                key = f"{key}#{len(order)}"          # "{a|b}, {a|b}" = two picks: keep both
            seen_here.add(key)
            if key not in best:
                order.append(key); best[key] = (w, raw)
            elif w > best[key][0] + 1e-6:
                best[key] = (w, raw if not raw.startswith("<") else raw)
    # drop BREAKs at the start/end or doubled up
    tags = [best[k][1] for k in order]
    cleaned: list[str] = []
    for t in tags:
        if t == "BREAK" and (not cleaned or cleaned[-1] == "BREAK"):
            continue
        cleaned.append(t)
    while cleaned and cleaned[-1] == "BREAK":
        cleaned.pop()
    return join_tags(cleaned)


def tidy_prompt(prompt: str) -> tuple[str, list[str]]:
    """merge_prompts on one prompt + a human-readable list of what was merged."""
    seen: dict[str, list[str]] = {}
    for raw in split_tags(prompt):
        if raw != "BREAK" and not _DYNAMIC.search(raw):
            seen.setdefault(parse_tag(raw)[0], []).append(raw)
    dupes = [f"{v[0]} ×{len(v)}" if len(set(v)) == 1 else " / ".join(v)
             for v in seen.values() if len(v) > 1]
    return merge_prompts(prompt), dupes


# ── Tokens & chunks ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _cached_tokenizer():
    """CLIP-L tokenizer from the local HF cache (SD 1.5 and SDXL share its vocabulary)."""
    try:
        from transformers import CLIPTokenizer
        for repo in ("stable-diffusion-v1-5/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0",
                     "openai/clip-vit-large-patch14"):
            try:
                return CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer" if "/" in repo and "openai" not in repo
                                                     else "", local_files_only=True, clean_up_tokenization_spaces=True)
            except Exception:
                continue
    except Exception:
        pass
    return None


@lru_cache(maxsize=8192)
def _plain(tag: str) -> str:
    """Tag text as CLIP sees it (weight syntax removed)."""
    if tag.startswith("<"):
        return ""
    from backend.prompt_syntax import plain_prompt
    return plain_prompt(tag) if any(c in tag for c in "()[]+-") else tag


def preload_tokenizer() -> None:
    """Load the CLIP tokenizer in the background (~2 s) so start-up doesn't wait for it."""
    import threading
    threading.Thread(target=_cached_tokenizer, name="tokenizer-preload", daemon=True).start()


def tokenizer_ready() -> bool:
    return _cached_tokenizer.cache_info().currsize > 0


def count_tokens(text: str, tokenizer=None) -> int:
    """CLIP tokens in text. tokenizer="estimate" counts roughly without loading anything."""
    tok = None if tokenizer == "estimate" else (tokenizer or _cached_tokenizer())
    if not text:
        return 0
    if tok is None:                       # rough fallback: ~1.3 tokens per word + punctuation
        return int(len(re.findall(r"\w+", text)) * 1.3 + text.count(","))
    key = (id(tok), text)                 # the counter runs on every keystroke: cache per tag
    n = _TOKEN_CACHE.get(key)
    if n is None:
        n = len(tok(text, add_special_tokens=False, truncation=False).input_ids)
        if len(_TOKEN_CACHE) > 20000:
            _TOKEN_CACHE.clear()
        _TOKEN_CACHE[key] = n
    return n


_TOKEN_CACHE: dict[tuple[int, str], int] = {}


def chunk_prompt(prompt: str, tokenizer=None, limit: int = CHUNK_TOKENS) -> list[list[str]]:
    """Tags grouped into CLIP chunks of ≤ limit tokens, never splitting a tag (one tag longer
    than a whole chunk gets its own chunk). BREAK forces a new chunk. <lora:…> tags are
    dropped (they aren't text)."""
    chunks: list[list[str]] = [[]]
    used = 0
    for tag in split_tags(prompt):
        if tag == "BREAK":
            if chunks[-1]:
                chunks.append([]); used = 0
            continue
        if _LORA.match(tag):
            continue
        n = count_tokens(_plain(tag), tokenizer) + (1 if chunks[-1] else 0)   # + the comma
        if chunks[-1] and used + n > limit:
            chunks.append([]); used = 0
            n -= 1
        chunks[-1].append(tag)
        used += n
    return [c for c in chunks if c] or [[]]


def token_report(prompt: str, tokenizer=None) -> dict:
    chunks = chunk_prompt(prompt, tokenizer)
    sizes = [count_tokens(_plain(", ".join(c)) if len(c) == 1 else ", ".join(_plain(t) for t in c), tokenizer)
             for c in chunks]
    return {"tokens": sum(sizes), "chunks": len(chunks) if any(chunks) else 0, "sizes": sizes,
            "starts": [c[0] for c in chunks if c],
            "exact": tokenizer != "estimate" and (tokenizer or _cached_tokenizer()) is not None}


def prompt_warnings(prompt: str, negative: str = "", important=(), tokenizer=None) -> list[str]:
    """Things that quietly weaken a prompt: a LoRA trigger word pushed out of the first
    chunk (it steers far less there), and a tag in both prompts (they cancel out)."""
    warn = []
    chunks = chunk_prompt(prompt, tokenizer)
    if len(chunks) > 1 and important:
        later = {parse_tag(t)[0] for c in chunks[1:] for t in c}
        first = {parse_tag(t)[0] for t in chunks[0]}
        moved = [t for t in important if parse_tag(t)[0] in later and parse_tag(t)[0] not in first]
        if moved:
            warn.append("trigger word(s) " + ", ".join(f"<b>{_esc(t)}</b>" for t in moved[:4])
                        + " are past the first 75 tokens — move them to the front (or click the chip) "
                          "for a stronger effect")
    pos = {parse_tag(t)[0]: t for t in split_tags(prompt) if t != "BREAK"}
    both = [pos[k] for k in (parse_tag(t)[0] for t in split_tags(negative) if t != "BREAK") if k in pos]
    if both:
        warn.append("in both prompts (they cancel out): " + ", ".join(f"<b>{_esc(t)}</b>" for t in both[:6]))
    return warn


def token_report_html(prompt: str, negative: str = "", tokenizer=None, important=()) -> str:
    """One-line summary under the prompt boxes (+ warnings)."""
    def one(label, text):
        r = token_report(text, tokenizer)
        if not r["tokens"]:
            return f"{label}: empty"
        approx = "" if r["exact"] else "≈"
        if r["chunks"] <= 1:
            return f"{label}: {approx}{r['tokens']}/75 token{'s' if r['tokens'] != 1 else ''}"
        s = f"{label}: {approx}{r['tokens']} tokens"
        if r["chunks"] > 1:
            parts = " | ".join(f"{n}" for n in r["sizes"])
            starts = " · ".join(f"#{i + 2} starts at <i>{_esc(t[:28])}</i>" for i, t in enumerate(r["starts"][1:]))
            s += f" → {r['chunks']} chunks [{parts}] ({starts})"
        return s
    note = ('<span style="color:#9399b2;"> · every 75 tokens is a separate CLIP chunk; nothing is '
            'cut off, but tags in the first chunk steer the image most. BREAK starts a new chunk.</span>')
    warn = "".join(f'<br><span style="color:#f9e2af;">⚠ {w}</span>'
                   for w in prompt_warnings(prompt, negative, important, tokenizer))
    fixes = danbooru_hints(prompt) + danbooru_hints(negative)
    hint = ('<br><span style="color:#89dceb;">💡 Danbooru spelling (anime models learned these exact tags): '
            + ", ".join(f"<i>{_esc(a)}</i> → <b>{_esc(b)}</b>" for a, b in fixes) + "</span>") if fixes else ""
    return (f'<p style="font-size:13px;color:#bac2de;margin:2px 0;">📏 {one("Prompt", prompt)} · '
            f'{one("Negative", negative)}{note}{warn}{hint}</p>')


def danbooru_hints(prompt: str, limit: int = 6) -> list[tuple[str, str]]:
    """(tag as written, the Danbooru tag it's a near-miss of) — only when the tag list is
    already on disk (no download from here) and only for plain tags."""
    try:
        from backend.danbooru_tags import did_you_mean
    except Exception:
        return []
    out = []
    tags = split_tags(prompt or "")
    if tags and (prompt or "").rstrip()[-1:] not in (",", "\n", ""):
        tags = tags[:-1]            # still being typed ("thigh" on its way to "thighhighs")
    for raw in tags:
        if raw == "BREAK" or raw.startswith("<") or "{" in raw or "__" in raw:
            continue
        core = parse_tag(raw)[2]
        if not core or any(c in core for c in "()[]|:"):
            continue
        m = did_you_mean(core)
        if m and (core, m) not in out:
            out.append((core, m))
        if len(out) >= limit:
            break
    return out


def apply_danbooru_fixes(prompt: str) -> tuple[str, list[tuple[str, str]]]:
    """The prompt with every danbooru_hints() near-miss replaced (weights kept:
    "(long haired:1.2)" → "(long hair:1.2)"), and the list of changes."""
    fixes = danbooru_hints(prompt + ",", limit=100)
    if not fixes:
        return prompt, []
    out = []
    for raw in split_tags(prompt):
        core = parse_tag(raw)[2] if raw != "BREAK" else ""
        new = next((b for a, b in fixes if a == core), None)
        out.append(raw.replace(core, new.replace("(", "\\(").replace(")", "\\)"), 1) if new else raw)
    return join_tags(out), fixes


def _esc(s: str) -> str:
    from html import escape
    return escape(s)


def release_parser_cache():
    """Empty pyparsing's packrat cache. Compel parses prompts with pyparsing, and packrat
    caching (switched on globally by some library) stores parse exceptions *with their
    tracebacks* — so the Compel call frames, the Compel object and through it the text
    encoders stayed alive after the model was unloaded (1.6 GB of VRAM for SDXL)."""
    try:
        import pyparsing
        pyparsing.ParserElement.reset_cache()
    except Exception:
        pass


def encode_chunked(compel, prompt: str, tokenizer=None, sdxl: bool = False):
    """Encode each tag-aligned chunk with Compel and concatenate: [1, 77·k, dim]
    (+ the first chunk's pooled embedding for SDXL, as A1111 does)."""
    import torch
    from backend.prompt_syntax import a1111_to_compel
    # Frames on the Compel call path are kept by pyparsing's packrat cache (see
    # release_parser_cache); a `compel` local here would keep the text encoders alive.
    try:
        chunks = chunk_prompt(prompt, tokenizer)
        embs, pooled = [], None
        for tags in chunks:
            out = compel(a1111_to_compel(", ".join(tags)))
            if sdxl:
                e, p = out
                pooled = p if pooled is None else pooled
            else:
                e = out
            embs.append(e)
        emb = torch.cat(embs, dim=1) if len(embs) > 1 else embs[0]
        return (emb, pooled) if sdxl else emb
    finally:
        compel = tokenizer = out = None
        release_parser_cache()


def pad_to_same_chunks(compel, a, b, sdxl: bool = False):
    """Pad the shorter conditioning with empty-prompt chunks (A1111 behaviour)."""
    import torch
    if a.shape[1] == b.shape[1]:
        return a, b
    try:
        empty = compel("")
    finally:
        compel = None          # (see encode_chunked)
        release_parser_cache()
    empty = empty[0] if sdxl else empty
    empty = empty.to(a.dtype)
    def pad(x, n):
        reps = (n - x.shape[1]) // empty.shape[1]
        return torch.cat([x] + [empty] * reps, dim=1) if reps > 0 else x
    n = max(a.shape[1], b.shape[1])
    return pad(a, n), pad(b, n)


# ── Placement ─────────────────────────────────────────────────────────────────
_QUALITY = {
    "masterpiece", "best quality", "high quality", "amazing quality", "very aesthetic", "aesthetic",
    "absurdres", "highres", "ultra-detailed", "ultra detailed", "highly detailed", "extremely detailed",
    "8k", "4k", "8k uhd", "8k resolution", "raw photo", "newest", "detailed", "hdr", "sharp focus",
}


def _is_quality(tag: str) -> bool:
    k = parse_tag(tag)[0]
    return k in _QUALITY or k.startswith(("score ", "source ", "rating "))


def insert_after_quality(prompt: str, add: str) -> str:
    """Put `add` (trigger words, a character's tags) right after the leading quality tags —
    still inside the first CLIP chunk, which steers the image most — instead of at the end.
    Tags already present move up to that spot (keeping the stronger weight)."""
    tags = split_tags(prompt)
    i = 0
    while i < len(tags) and tags[i] != "BREAK" and _is_quality(tags[i]):
        i += 1
    return merge_prompts(join_tags(tags[:i] + split_tags(add) + tags[i:]))
