"""
backend/trigger_reader.py
Reads trigger words and top training tags from local model files.

Sources (in priority order):
1. .civitai.json sidecar — saved at download time, contains `trainedWords`
   set by the model creator on Civitai.
2. .safetensors header metadata — kohya_ss embeds `ss_tag_frequency` with
   every tag and its training frequency.  Top tags ≈ the concepts the LoRA
   specialises in.
"""

from __future__ import annotations

import hashlib
import json
import struct
from html import escape as _esc
from pathlib import Path
from typing import Any, Callable


# Tags so generic they appear in almost every anime/photo dataset and give
# no useful information about *this specific* model.
_SKIP_TAGS: frozenset[str] = frozenset({
    "1girl", "solo", "looking at viewer", "smile", "open mouth",
    "long hair", "short hair", "blonde hair", "brown hair", "black hair",
    "white hair", "blue hair", "pink hair", "red hair",
    "simple background", "white background", "blue background",
    "breasts", "large breasts", "medium breasts", "small breasts",
    "multiple views", "full body", "upper body", "close-up",
    "highres", "absurdres", "commentary", "english commentary",
    "traditional media", "signature", "dated", "watermark",
    "official art", "official alternate costume",
    "rating:explicit", "rating:questionable", "rating:safe",
    "score_9", "score_8", "score_7",
})


def _read_safetensors_meta(path: Path) -> dict[str, Any]:
    """
    Read the `__metadata__` dict from a .safetensors file header
    without loading the tensors.  Returns {} on any error.
    """
    try:
        with open(path, "rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) < 8:
                return {}
            header_len: int = struct.unpack("<Q", raw_len)[0]
            if header_len == 0 or header_len > 50_000_000:   # skip obviously bad files
                return {}
            header_bytes = fh.read(header_len)
        header: dict = json.loads(header_bytes)
        return header.get("__metadata__", {})
    except Exception:
        return {}


def _extract_top_tags(meta: dict[str, Any], top_n: int = 30) -> list[tuple[str, int]]:
    """
    Extract the most-trained tags from `ss_tag_frequency`.
    Returns a list of (tag, count) sorted by count descending.
    """
    raw = meta.get("ss_tag_frequency", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, dict):
        return []

    merged: dict[str, int] = {}
    for dataset_tags in raw.values():
        if isinstance(dataset_tags, dict):
            for tag, count in dataset_tags.items():
                try:
                    merged[tag.strip()] = merged.get(tag.strip(), 0) + int(count)
                except (TypeError, ValueError):
                    pass

    filtered = [
        (tag, cnt) for tag, cnt in merged.items()
        if tag not in _SKIP_TAGS and len(tag) > 1
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:top_n]


def read_model_triggers(model_path: str) -> dict[str, Any]:
    """
    Return trigger / training info for a single model file.

    Result dict keys:
        trained_words  : list[str]   — creator-specified activation tokens
        top_tags       : list[tuple] — (tag, train_count) from safetensors header
        base_model     : str         — e.g. "SD 1.5", "SDXL 1.0"
        model_name     : str         — human name from Civitai sidecar
        source         : str         — where the info came from ("sidecar", "header", "none")
    """
    path = Path(model_path)
    result: dict[str, Any] = {
        "trained_words": [],
        "top_tags":      [],
        "base_model":    "",
        "model_name":    path.stem,
        "source":        "none",
    }

    if not path.exists():
        return result

    # ── 1. Civitai sidecar (.civitai.json) ────────────────────────────────────
    # fetch_civitai_sidecar saves as "model.safetensors.civitai.json" (appended).
    # Check that format first; fall back to stripped ".civitai.json" for legacy files.
    sidecar = Path(str(path) + ".civitai.json")
    if not sidecar.exists():
        sidecar2 = path.with_suffix("").with_suffix(".civitai.json")
        if sidecar2.exists():
            sidecar = sidecar2

    if sidecar.exists():
        try:
            data: dict = json.loads(sidecar.read_text(encoding="utf-8"))
            tw = data.get("trainedWords") or []
            result["trained_words"] = [w.strip() for w in tw if isinstance(w, str) and w.strip()]
            result["base_model"]    = str(data.get("baseModel") or "")
            result["model_name"]    = str((data.get("model") or {}).get("name") or path.stem)
            result["source"]        = "sidecar"
        except Exception:
            pass

    # ── 2. .safetensors header ─────────────────────────────────────────────────
    if path.suffix == ".safetensors":
        meta = _read_safetensors_meta(path)
        if meta:
            result["top_tags"] = _extract_top_tags(meta)
            if not result["base_model"]:
                result["base_model"] = meta.get("ss_base_model_version", "")
            if result["source"] == "none":
                result["source"] = "header"

    return result


def render_triggers_html(checkpoint_path: str, lora_path: str) -> str:
    """
    Build an HTML summary of trigger words + top training tags for the
    currently loaded checkpoint and/or LoRA.  Returns "" when nothing useful
    is found.
    """
    parts: list[str] = []

    def _badge(text: str, color: str) -> str:
        return (
            f'<span style="background:{color};color:#1e1e2e;padding:1px 7px;'
            f'margin:2px 2px;border-radius:10px;font-size:13px;display:inline-block;'
            f'white-space:nowrap;">{_esc(text)}</span>'
        )

    # Checkpoint info (trigger words only — training tags for a full checkpoint are huge)
    if checkpoint_path and checkpoint_path != "none":
        ckpt = read_model_triggers(checkpoint_path)
        if ckpt["trained_words"]:
            badges = " ".join(_badge(w, "#f9e2af") for w in ckpt["trained_words"][:20])
            parts.append(
                f'<div style="margin-bottom:6px;">'
                f'<span style="color:#f9e2af;font-size:13px;font-weight:bold;">'
                f'🔑 Checkpoint trigger words</span><br>{badges}</div>'
            )

    # LoRA info (trigger words + top training tags)
    if lora_path and lora_path != "none":
        lora = read_model_triggers(lora_path)
        lora_header = (
            f'<span style="color:#cba6f7;font-size:13px;font-weight:bold;">'
            f'🔑 LoRA: {_esc(lora["model_name"])}'
            + (f' — {_esc(lora["base_model"])}' if lora["base_model"] else "")
            + "</span>"
        )

        if lora["trained_words"]:
            badges = " ".join(_badge(w, "#cba6f7") for w in lora["trained_words"][:20])
            parts.append(
                f'<div style="margin-bottom:4px;">{lora_header}<br>'
                f'<span style="color:#a6adc8;font-size:13px;">Trigger words:</span><br>'
                f'{badges}</div>'
            )

        if lora["top_tags"]:
            # Show top 25 tags with frequency as tooltip via title=
            tag_badges = " ".join(
                f'<span style="background:#313244;color:#cdd6f4;padding:1px 6px;'
                f'margin:2px 2px;border-radius:10px;font-size:13px;display:inline-block;'
                f'white-space:nowrap;cursor:default;" title="trained {cnt}×">{_esc(tag)}</span>'
                for tag, cnt in lora["top_tags"][:25]
            )
            src_note = "(from safetensors metadata)" if lora["source"] in ("header", "sidecar") else ""
            parts.append(
                f'<div style="margin-bottom:4px;">'
                + ("" if lora["trained_words"] else lora_header + "<br>")
                + f'<span style="color:#a6adc8;font-size:13px;">'
                  f'Top training concepts {src_note}:</span><br>'
                + tag_badges
                + "</div>"
            )

        if not lora["trained_words"] and not lora["top_tags"]:
            parts.append(
                f'<div style="color:#9399b2;font-size:13px;">'
                f'ℹ️ No trigger word data found for this LoRA '
                f'(no .civitai.json sidecar and no ss_tag_frequency in header).</div>'
            )

    return "\n".join(parts)


# ── Civitai hash-based sidecar fetch ──────────────────────────────────────────

def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest().upper()


def fetch_civitai_sidecar(model_path: str, api_key: str = "") -> dict[str, Any]:
    """
    Look up a local model file on Civitai by its SHA-256 hash and save a
    .civitai.json sidecar next to the file.

    Returns the result dict from read_model_triggers() (now with trainedWords
    filled in from Civitai), or raises on network / not-found errors.
    """
    import httpx

    path = Path(model_path)
    sidecar = Path(str(path) + ".civitai.json")

    if sidecar.exists():
        return read_model_triggers(model_path)   # already fetched

    sha = _sha256_of_file(path)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    resp = httpx.get(
        f"https://civitai.com/api/v1/model-versions/by-hash/{sha}",
        headers=headers,
        timeout=15,
        follow_redirects=True,
    )
    resp.raise_for_status()
    data = resp.json()

    sidecar.write_text(
        json.dumps({
            "trainedWords": data.get("trainedWords", []),
            "baseModel":    data.get("baseModel", ""),
            "model": {
                "name":  (data.get("model") or {}).get("name", ""),
                "type":  (data.get("model") or {}).get("type", ""),
                "nsfw":  (data.get("model") or {}).get("nsfw", False),
                "tags":  (data.get("model") or {}).get("tags", []),
                "description": (data.get("model") or {}).get("description", ""),
            },
            "civitai_version_id": data.get("id"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return read_model_triggers(model_path)


def batch_fetch_sidecars(
    paths: list[str],
    api_key: str = "",
    on_progress: "Callable[[int, int, str], None] | None" = None,
) -> dict[str, str]:
    """
    Fetch Civitai sidecars for a list of model files that don't already have one.
    Skips .ckpt files (hash lookup unreliable for legacy format).

    Returns {path: status_message} for every file.
    on_progress(done, total, current_filename) is called after each file.
    """
    results: dict[str, str] = {}
    to_fetch = [
        p for p in paths
        if not Path(str(p) + ".civitai.json").exists()
        and Path(p).suffix == ".safetensors"
    ]
    total = len(to_fetch)
    for i, p in enumerate(to_fetch):
        name = Path(p).name
        if on_progress:
            on_progress(i, total, name)
        try:
            fetch_civitai_sidecar(p, api_key)
            results[p] = "✅ fetched"
        except Exception as exc:
            msg = str(exc)
            if "404" in msg:
                results[p] = "⚠️ not on Civitai"
            elif "401" in msg or "403" in msg:
                results[p] = "🔒 auth required — add API key"
            else:
                results[p] = f"❌ {msg[:60]}"
    return results


# ── Model specialty display ────────────────────────────────────────────────────

_SPECIALTY_KEYWORDS: dict[str, str] = {
    # tag keyword → human label (emoji + text)
    "anime":        "🎌 Anime",
    "manga":        "📖 Manga",
    "realistic":    "📷 Realistic",
    "photorealistic": "📷 Photorealistic",
    "illustration": "🖌 Illustration",
    "cartoon":      "🎠 Cartoon",
    "3d":           "🧊 3D",
    "fantasy":      "🔮 Fantasy",
    "nsfw":         "🔞 NSFW",
    "furry":        "🐾 Furry",
    "landscape":    "🌄 Landscape",
    "portrait":     "🧑 Portrait",
    "scifi":        "🚀 Sci-Fi",
    "horror":       "💀 Horror",
}


def get_model_specialty_html(model_path: str) -> str:
    """
    Return coloured badges showing a model's specialty (anime/realistic/NSFW etc.)
    read from its .civitai.json sidecar.  Returns "" when no sidecar exists.
    """
    path = Path(model_path)
    if not path.exists():
        return ""

    sidecar = Path(str(path) + ".civitai.json")
    if not sidecar.exists():
        sidecar2 = path.with_suffix("").with_suffix(".civitai.json")
        if sidecar2.exists():
            sidecar = sidecar2

    if not sidecar.exists():
        return ""

    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return ""

    if not isinstance(data, dict):
        return ""
    model_info = data.get("model") or {}
    if not isinstance(model_info, dict):
        return ""
    tags: list[str] = [(t.get("name", "") if isinstance(t, dict) else str(t)).lower()
                       for t in (model_info.get("tags") or [])]
    nsfw: bool = bool(model_info.get("nsfw", False))
    model_type: str = _esc(str(model_info.get("type") or ""))
    name: str = _esc(str(model_info.get("name") or ""))

    if not tags and not nsfw and not name:
        return ""

    badges: list[str] = []

    # Model type badge
    if model_type:
        badges.append(
            f'<span style="background:#313244;color:#a6adc8;padding:1px 7px;'
            f'margin:2px;border-radius:10px;font-size:13px;">{model_type}</span>'
        )

    # NSFW badge
    if nsfw:
        badges.append(
            '<span style="background:#f38ba8;color:#1e1e2e;padding:1px 7px;'
            'margin:2px;border-radius:10px;font-size:13px;font-weight:bold;">🔞 NSFW</span>'
        )

    # Specialty tags
    seen: set[str] = set()
    for tag in tags:
        for kw, label in _SPECIALTY_KEYWORDS.items():
            if kw in tag and label not in seen:
                color = "#f38ba8" if "🔞" in label else "#cba6f7"
                badges.append(
                    f'<span style="background:{color};color:#1e1e2e;padding:1px 7px;'
                    f'margin:2px;border-radius:10px;font-size:13px;">{label}</span>'
                )
                seen.add(label)

    if not badges:
        return ""

    header = f'<span style="color:#a6adc8;font-size:13px;">Civitai: </span>' if not name else \
             f'<span style="color:#cdd6f4;font-size:13px;font-weight:bold;">{name}</span> '
    return (
        f'<div style="margin:2px 0 4px;">{header}'
        + "".join(badges) + "</div>"
    )
