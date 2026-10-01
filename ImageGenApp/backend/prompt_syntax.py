"""
prompt_syntax.py — translate A1111 / Civitai prompt weighting into Compel syntax.

Compel (used for >77-token prompts and weighting) does not understand A1111 syntax:
"(best quality:1.2)" came through as the literal tokens "best quality : 1 . 2" with no
weight, "((masterpiece))" got no emphasis and "[blurry]" no de-emphasis. Every Civitai
prompt and the app's own presets use that syntax, so it is converted before encoding:

    (x)        → (x)1.1          ((x))  → ((x)1.1)1.1   (nested weights multiply)
    (x:1.3)    → (x)1.3          [x]    → (x)0.9091
    \\( \\)      → kept (literal parentheses)

Text already in Compel syntax — "(x)1.2", "(x)+", "word++" — is left alone.
"""
from __future__ import annotations

import re

# A number right after ")" is a Compel weight only when a word doesn't continue: in
# "(solo)1girl" the "1" belongs to "1girl" (Compel would read weight 1 + "girl").
_COMPEL_WEIGHT = re.compile(r"(?:[+-]+|\d*\.?\d+)(?![\w.])")
_A1111_WEIGHT = re.compile(r"^(.*):\s*(\d*\.?\d+)\s*$", re.S)
_EXTRA_NET = re.compile(r"<(?:lora|lyco|hypernet):[^>]*>", re.I)


def _fmt(w: float) -> str:
    return f"{w:.4f}".rstrip("0").rstrip(".")


def _parse(s: str, i: int, closer: str | None) -> tuple[str, int, bool]:
    buf: list[str] = []
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):          # escaped bracket stays literal
            buf.append(s[i:i + 2])
            i += 2
            continue
        if ch in "([":
            inner, j, closed = _parse(s, i + 1, ")" if ch == "(" else "]")
            if not closed:                          # unbalanced: keep as typed
                buf.append(ch + inner)
                i = j
                continue
            if ch == "(":
                if _COMPEL_WEIGHT.match(s, j):      # already Compel: "(x)1.2" / "(x)+"
                    buf.append(f"({inner})")
                    i = j
                    continue
                m = _A1111_WEIGHT.match(inner)
                body, w = (m.group(1), float(m.group(2))) if m else (inner, 1.1)
            else:
                body, w = inner, 1 / 1.1
            buf.append(f"({body}){_fmt(w)}")
            if j < len(s) and (s[j].isalnum() or s[j] == "."):
                buf.append(" ")                     # "(x)1girl" → "(x)1.1 1girl", not "1.11girl"
            i = j
            continue
        if closer is not None and ch == closer:
            return "".join(buf), i + 1, True
        buf.append(ch)
        i += 1
    return "".join(buf), i, closer is None


# Compel reads "+"/"-" right after a word as a weight ("word+" = ×1.1), so the Danbooru kaomoji
# "+_+" and "-_-" became the text "+_" / "-_" with a weight (audit F-08). A sign that ends a
# non-word (the char before it isn't a letter, digit, ")" or another sign) is escaped here, and the
# patched Fragment below drops the backslash again. Compel's own "word++" keeps working.
_TRAILING_SIGN = re.compile(r"(?<=[^A-Za-z0-9)+\-\\])([+-]+)(?=\s|[,.)\]\"=:]|$)")


def _protect_trailing_signs(prompt: str) -> str:
    if "+" not in prompt and "-" not in prompt:
        return prompt
    return _TRAILING_SIGN.sub(lambda m: "".join("\\" + c for c in m.group(1)), prompt)


def _patch_compel_fragment() -> None:
    """Compel unescapes only backslash-( ) " in fragment text; drop the backslash of + / - too."""
    try:
        from compel import prompt_parser as P
    except Exception:
        return
    if getattr(P.Fragment, "_imagegen_unescape", False):
        return
    orig = P.Fragment.__init__

    def __init__(self, text, weight=1):
        orig(self, text, weight)
        if "\\" in self.text:
            self.text = self.text.replace("\\+", "+").replace("\\-", "-")
    P.Fragment.__init__ = __init__
    P.Fragment._imagegen_unescape = True


_patch_compel_fragment()


def a1111_to_compel(prompt: str) -> str:
    """Convert A1111-style emphasis to Compel weights (no-op for prompts without brackets)."""
    if not prompt:
        return ""
    # A1111's BREAK starts a new 75-token chunk; Compel has no equivalent and would
    # encode the word "break", so treat it as a separator.
    prompt = re.sub(r"\s*\bBREAK\b\s*", ", ", prompt)
    # <lora:name:0.8> / <lyco:…> / <hypernet:…> are instructions for A1111, not text. The
    # Generate tab moves them into the LoRA slots; anything still here (Bridge prompt,
    # loaded presets, Auto-Loop) would be encoded as "< lora : name : 0 . 8 >".
    if "<" in prompt:
        tags = _EXTRA_NET.findall(prompt)
        if tags:
            print(f"[Prompt] Ignored {len(tags)} <lora:…> tag(s) in the prompt text — "
                  "use the LoRA slots (Generate tab) to apply LoRAs")
            prompt = _EXTRA_NET.sub("", prompt)
            prompt = re.sub(r"\s*,\s*(,\s*)+", ", ", prompt).strip().strip(",").strip()
    prompt = _protect_trailing_signs(prompt)
    if "(" not in prompt and "[" not in prompt:
        return prompt
    try:
        out, _, _ = _parse(prompt, 0, None)
    except RecursionError:                          # hundreds of nested brackets: drop them
        return re.sub(r"(?<!\\)[()\[\]]", "", prompt)
    return out


def plain_prompt(prompt: str) -> str:
    """The prompt's words without any weighting syntax — for paths that hand raw text to
    diffusers (DirectML text encoder, Compel failure), where "(x:1.2)" would otherwise be
    encoded as the tokens "( x : 1 . 2 )"."""
    if not prompt:
        return prompt or ""
    try:
        from compel.prompt_parser import PromptParser
        conj = PromptParser().parse_conjunction(a1111_to_compel(prompt))
        return ", ".join(p for p in (" ".join(f.text for f in getattr(pr, "children", []))
                                     for pr in conj.prompts) if p).replace(" , ", ", ")
    except Exception:
        return prompt
