"""
history.py — an index of everything in outputs/ for the 🗂 History tab: prompt, model, LoRAs, seed,
sampler and size of each PNG, read from its text chunks only (no pixel decoding), cached in
settings/_history_index.json by file name + mtime, plus small JPEG thumbnails in outputs/.thumbs/.
Favourites live in settings/_favourites.json.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from PIL import Image

from config import OUTPUTS_DIR, SETTINGS_DIR

INDEX_FILE = SETTINGS_DIR / "_history_index.json"
FAV_FILE = SETTINGS_DIR / "_favourites.json"
THUMBS = OUTPUTS_DIR / ".thumbs"
THUMB_SIDE = 256
_lock = threading.Lock()


def _entry(p: Path) -> dict:
    from backend.png_info import read_image_metadata
    with Image.open(p) as im:          # text chunks only — decoding 2000 PNGs took a minute
        m = read_image_metadata(im)
    loras = m.get("loras") or ""
    rec = m.get("imagegen") or {}
    return {
        "mtime": p.stat().st_mtime,
        "prompt": (m.get("prompt") or "")[:2000],
        "negative": (m.get("negative_prompt") or "")[:500],
        "model": Path(str(m.get("model") or "")).stem,
        "loras": [x.split(":")[0].strip() for x in str(loras).split(",") if x.strip()],
        "seed": m.get("seed"),
        "sampler": m.get("sampler") or "",
        "size": f"{m.get('width')}×{m.get('height')}",
        "mode": rec.get("mode") or "",
    }


def build_index(outputs: Path | None = None) -> dict[str, dict]:
    """{file name: entry} for every PNG in outputs/ (not sub-folders), refreshing changed files only."""
    outputs = Path(outputs or OUTPUTS_DIR)
    with _lock:
        try:
            index = json.loads(INDEX_FILE.read_text(encoding="utf-8")) if INDEX_FILE.exists() else {}
        except Exception:
            index = {}
        seen, changed = set(), False
        for p in outputs.glob("*.png"):
            seen.add(p.name)
            try:
                mt = p.stat().st_mtime
                old = index.get(p.name)
                if old and abs(old.get("mtime", 0) - mt) < 1e-3:
                    continue
                index[p.name] = _entry(p)
                changed = True
            except Exception as e:                   # a damaged file must not break the browser
                index[p.name] = {"mtime": 0, "prompt": "", "negative": "", "model": "", "loras": [], "seed": None,
                                 "sampler": "", "size": "", "mode": "", "error": str(e)[:120]}
                changed = True
        for gone in set(index) - seen:
            del index[gone]
            changed = True
        if changed:
            try:
                INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
        return index


def favourites() -> set[str]:
    try:
        return set(json.loads(FAV_FILE.read_text(encoding="utf-8")))
    except Exception:
        return set()


def toggle_favourite(name: str) -> bool:
    """Flip a file's star; returns the new state."""
    with _lock:
        favs = favourites()
        on = name not in favs
        (favs.add if on else favs.discard)(name)
        FAV_FILE.write_text(json.dumps(sorted(favs), ensure_ascii=False), encoding="utf-8")
        return on


def search(index: dict, text: str = "", model: str = "", lora: str = "", favs_only: bool = False,
           favs: set | None = None) -> list[str]:
    """File names matching every word of `text` (in prompt / model / LoRAs / seed / sampler / name), the model
    and LoRA filters and optionally only favourites — newest first."""
    words = [w for w in (text or "").lower().split() if w]
    favs = favs if favs is not None else favourites()
    out = []
    for name, e in index.items():
        if favs_only and name not in favs:
            continue
        if model and model != "All" and e.get("model") != model:
            continue
        if lora and lora != "All" and lora not in (e.get("loras") or []):
            continue
        if words:
            hay = " ".join([name, e.get("prompt", ""), e.get("model", ""), " ".join(e.get("loras") or []),
                            str(e.get("seed")), e.get("sampler", "")]).lower()
            if not all(w in hay for w in words):
                continue
        out.append(name)
    out.sort(key=lambda n: -index[n].get("mtime", 0))
    return out


def _placeholder() -> str | None:
    """A grey tile with a cross for files that can't be thumbnailed (gone or damaged). The gallery must keep
    one tile per name: it used to skip them, and every tile after the gap then mapped to the wrong
    names[page × PAGE + index] when clicked."""
    dst = THUMBS / "_unreadable.jpg"
    try:
        if not dst.exists():
            THUMBS.mkdir(parents=True, exist_ok=True)
            from PIL import ImageDraw
            im = Image.new("RGB", (THUMB_SIDE, THUMB_SIDE), (49, 50, 68))
            d = ImageDraw.Draw(im)
            m = THUMB_SIDE // 4
            d.line((m, m, THUMB_SIDE - m, THUMB_SIDE - m), fill=(137, 142, 170), width=6)
            d.line((THUMB_SIDE - m, m, m, THUMB_SIDE - m), fill=(137, 142, 170), width=6)
            im.save(dst, "JPEG", quality=85)
        return str(dst)
    except Exception:
        return None


def thumbnail(name: str, outputs: Path | None = None) -> str | None:
    """Path of a ≤256 px JPEG thumbnail (made once, remade when the image changes); a placeholder tile when
    the file is gone or damaged."""
    src = Path(outputs or OUTPUTS_DIR) / name
    if not src.is_file():
        return _placeholder()
    THUMBS.mkdir(parents=True, exist_ok=True)
    dst = THUMBS / (src.stem + ".jpg")
    try:
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            with Image.open(src) as im:
                im.thumbnail((THUMB_SIDE, THUMB_SIDE))
                im.convert("RGB").save(dst, "JPEG", quality=85)
        return str(dst)
    except Exception:
        return _placeholder()
