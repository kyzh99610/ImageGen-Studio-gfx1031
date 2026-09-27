"""
backend/tag_fetcher.py
Curated img2img enhancement tag sets + optional Danbooru live-tag fetch.
"""
from __future__ import annotations

# ── Curated img2img enhancer tags ─────────────────────────────────────────────
# Organized for use in the 🖼️ img2img Enhancers panel.
I2I_ENHANCER_TAGS: dict[str, list[str]] = {
    "✨ Quality": [
        "masterpiece", "best quality", "ultra detailed", "highly detailed",
        "8k", "hdr", "photorealistic", "hyperrealistic", "sharp focus",
        "high resolution", "absurdres",
    ],
    "👁 Fix Details": [
        "perfect anatomy", "correct anatomy", "perfect hands", "detailed fingers",
        "perfect face", "symmetrical face", "detailed eyes", "beautiful eyes",
        "detailed skin", "skin pores",
    ],
    "💡 Lighting": [
        "perfect lighting", "cinematic lighting", "soft lighting",
        "volumetric lighting", "rim lighting", "dramatic lighting",
        "natural lighting", "studio lighting",
    ],
    "🎨 Style Boost": [
        "digital art", "anime style", "manga style", "illustration",
        "concept art", "oil painting", "watercolor", "cel shading",
    ],
    "📐 Composition": [
        "rule of thirds", "depth of field", "bokeh", "wide angle shot",
        "close up", "portrait", "dutch angle", "low angle",
    ],
    "🧹 Clean Up": [
        "solo", "simple background", "white background", "clean background",
        "remove artifacts", "no extra limbs", "no blur",
    ],
}

# ── Danbooru live tag fetch ────────────────────────────────────────────────────
_DANBOORU_CACHE: dict[int, list[str]] = {}   # category_id → tag list


def fetch_danbooru_tags(
    category_id: int = 5,
    limit: int = 50,
    timeout: float = 8.0,
) -> list[str]:
    """
    Fetch top tags from Danbooru public API by category.
    category_id: 0=general, 1=artist, 3=copyright, 4=character, 5=meta (quality/technical)

    Results are cached per session. Returns [] on failure.
    """
    if category_id in _DANBOORU_CACHE:
        return _DANBOORU_CACHE[category_id]
    try:
        import httpx
        resp = httpx.get(
            "https://danbooru.donmai.us/tags.json",
            params={
                "search[category]": category_id,
                "search[order]": "count",
                "search[hide_empty]": "true",
                "limit": limit,
            },
            timeout=timeout,
            headers={"User-Agent": "ImageGenApp/1.0"},
        )
        resp.raise_for_status()
        tags = [t["name"].replace("_", " ") for t in resp.json() if t.get("name")]
        _DANBOORU_CACHE[category_id] = tags
        return tags
    except Exception:
        return []


def fetch_danbooru_tags_html(category_id: int = 0, limit: int = 60) -> str:
    """
    Return an HTML blob of clickable tag badges fetched from Danbooru.
    Intended for Gradio gr.HTML output.
    """
    tags = fetch_danbooru_tags(category_id, limit)
    if not tags:
        return '<p style="color:#f38ba8;font-size:13px;">Could not reach Danbooru API.</p>'
    badges = " ".join(
        f'<span class="db-tag" style="background:#313244;color:#cdd6f4;padding:2px 8px;'
        f'margin:2px;border-radius:12px;font-size:13px;display:inline-block;'
        f'cursor:pointer;white-space:nowrap;" title="click to copy">{t}</span>'
        for t in tags
    )
    return (
        f'<p style="font-size:13px;color:#a6adc8;margin:2px 0 4px;">'
        f'Top {len(tags)} Danbooru general tags (click to copy):</p>' + badges
    )
