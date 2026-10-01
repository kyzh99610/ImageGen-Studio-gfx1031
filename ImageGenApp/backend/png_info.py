"""
backend/png_info.py
Extract and parse generation metadata from PNG/JPEG images.
Supports Automatic1111, diffusers, ComfyUI, NovelAI, and ImageGen Studio formats.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from PIL import Image


def read_image_metadata(image_or_path: Image.Image | str | Path) -> dict[str, Any]:
    """
    Read generation metadata from a PIL Image or image file path.
    Returns a dictionary of parsed parameters:
      - prompt: str
      - negative_prompt: str
      - steps: int | None
      - sampler: str | None
      - cfg_scale: float | None
      - seed: int | None
      - width: int | None
      - height: int | None
      - model: str | None
      - loras: str | None
      - raw_text: str
    """
    if isinstance(image_or_path, (str, Path)):
        # Read and close: an open handle keeps the file locked on Windows
        with Image.open(str(image_or_path)) as f:
            f.load()
            img = f.copy()
            img.info = dict(f.info)
    else:
        img = image_or_path

    raw_text = ""
    info = getattr(img, "info", None) or {}
    for key in ("parameters", "Comment", "prompt", "Description"):
        val = info.get(key)
        if isinstance(val, bytes):
            val = val.decode("utf-8", "replace")
        if val:
            raw_text = str(val)
            break
    if not raw_text:
        raw_text = _exif_user_comment(img)

    # NovelAI keeps its settings as JSON in "Comment"; ComfyUI stores a node graph in
    # "prompt". Neither is the A1111 text format parsed below.
    novelai = _parse_json(info.get("Comment")) if "Comment" in info else None
    if novelai is not None and isinstance(novelai, dict) and "prompt" in novelai:
        return _from_novelai(img, novelai, str(info.get("Comment")), info.get("Description"))
    if raw_text.lstrip().startswith("{") and _parse_json(raw_text) is not None:
        return _empty_result(img, raw_text, note="ComfyUI workflow (node graph) — not A1111 parameters")

    result: dict[str, Any] = {
        "prompt": "",
        "negative_prompt": "",
        "steps": None,
        "sampler": None,
        "cfg_scale": None,
        "seed": None,
        "width": img.width if hasattr(img, "width") else 512,
        "height": img.height if hasattr(img, "height") else 512,
        "model": None,
        "loras": None,
        "raw_text": raw_text,
    }

    if not raw_text and "imagegen" not in info:   # a record alone is enough (audit F-21)
        return result

    # Standard A1111 / ImageGen parameters format:
    # <Positive Prompt>
    # Negative prompt: <Negative Prompt>
    # Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7.0, Seed: 42, Size: 512x512, Model: xyz...
    lines = raw_text.strip().split("\n")
    param_line_idx = -1
    neg_line_idx = -1

    for i, line in enumerate(lines):
        line_s = line.strip()
        if line_s.startswith("Negative prompt:"):
            neg_line_idx = i
        elif re.search(r"\bSteps:\s*\d+", line_s, re.I):
            param_line_idx = i

    # Extract positive prompt
    if neg_line_idx != -1:
        result["prompt"] = "\n".join(lines[:neg_line_idx]).strip()
        if param_line_idx != -1 and param_line_idx > neg_line_idx:
            neg_content = "\n".join(lines[neg_line_idx:param_line_idx])
            result["negative_prompt"] = re.sub(r"^Negative prompt:\s*", "", neg_content, flags=re.I).strip()
        else:
            neg_content = "\n".join(lines[neg_line_idx:])
            result["negative_prompt"] = re.sub(r"^Negative prompt:\s*", "", neg_content, flags=re.I).strip()
    elif param_line_idx != -1:
        result["prompt"] = "\n".join(lines[:param_line_idx]).strip()
    else:
        result["prompt"] = raw_text.strip()

    # Extract key-value pairs from parameter line
    param_text = lines[param_line_idx] if param_line_idx != -1 else raw_text
    
    # Steps
    m_steps = re.search(r"\bSteps:\s*(\d+)", param_text, re.I)
    if m_steps:
        result["steps"] = int(m_steps.group(1))

    # Sampler
    m_sampler = re.search(r"\b(?:Sampler|Scheduler):\s*([^,]+)", param_text, re.I)
    if m_sampler:
        result["sampler"] = m_sampler.group(1).strip()
        # A1111 1.9+ / Forge split "DPM++ 2M Karras" into "Sampler: DPM++ 2M, Schedule type: Karras"
        m_sched = re.search(r"\bSchedule type:\s*([^,]+)", param_text, re.I)
        sched = m_sched.group(1).strip().lower() if m_sched else ""
        suffix = {"karras": "Karras", "align your steps": "AYS"}.get(sched)
        if suffix and not result["sampler"].lower().endswith(suffix.lower()):
            result["sampler"] += f" {suffix}"

    # CFG scale
    m_cfg = re.search(r"\bCFG\s*scale:\s*([0-9.]+)", param_text, re.I)
    if m_cfg:
        try:
            result["cfg_scale"] = float(m_cfg.group(1))
        except ValueError:
            pass

    # Seed
    m_seed = re.search(r"\bSeed:\s*(-?\d+)", param_text, re.I)
    if m_seed:
        result["seed"] = int(m_seed.group(1))

    # Size (e.g. Size: 832x1216 or 512×768)
    m_size = re.search(r"\bSize:\s*(\d+)[x×](\d+)", param_text, re.I)
    if m_size:
        result["width"] = int(m_size.group(1))
        result["height"] = int(m_size.group(2))

    # Model
    m_model = re.search(r"\bModel:\s*([^,]+)", param_text, re.I)
    if m_model:
        result["model"] = m_model.group(1).strip()

    # LoRAs
    m_loras = re.search(r"\bLoRAs:\s*([^,\n]+(?:,\s*[^,\n]+)*)", param_text, re.I)
    if m_loras:
        result["loras"] = m_loras.group(1).strip()

    # Older ImageGen Studio files wrote "Width: 512, Height: 768" instead of "Size:"
    if not m_size:
        mw = re.search(r"\bWidth:\s*(\d+)", param_text)
        mh = re.search(r"\bHeight:\s*(\d+)", param_text)
        if mw and mh:
            result["width"], result["height"] = int(mw.group(1)), int(mh.group(1))
    m_vae = re.search(r"\bVAE:\s*([^,\n]+)", param_text)
    if m_vae:
        result["vae"] = m_vae.group(1).strip()
    m_den = re.search(r"\bDenoising strength:\s*([0-9.]+)", param_text)
    if m_den:
        try:
            result["strength"] = float(m_den.group(1))
        except ValueError:
            pass
    m_hash = re.search(r"\bModel hash:\s*([0-9a-fA-F]{8,})", param_text)
    if m_hash:
        result["model_hash"] = m_hash.group(1).lower()

    # CLIP skip, variation seed and hires fix (A1111 names)
    m = re.search(r"\bClip skip:\s*(\d+)", param_text)
    if m:
        result["clip_skip"] = int(m.group(1))
    m = re.search(r"\bVariation seed:\s*(\d+)", param_text)
    m2 = re.search(r"\bVariation seed strength:\s*([0-9.]+)", param_text)
    if m and m2:
        result["var_seed"], result["var_strength"] = int(m.group(1)), float(m2.group(1))
    m = re.search(r"\bHires upscale:\s*([0-9.]+)", param_text)
    if m:
        hs = re.search(r"\bHires steps:\s*(\d+)", param_text)
        hu = re.search(r"\bHires upscaler:\s*([^,\n]+)", param_text)
        result["hires"] = {"scale": float(m.group(1)),
                           "steps": int(hs.group(1)) if hs else None,
                           "upscaler": hu.group(1).strip() if hu else "Lanczos",
                           "denoise": result.pop("strength", None)}   # A1111: "Denoising strength" = hires denoise

    m = re.search(r"\bFace detail: denoise ([0-9.]+) \((\w+)\)", param_text)
    if m:
        result["face_detail"] = {"denoise": float(m.group(1)), "detector": m.group(2), "prompt": ""}
    m = re.search(r"\bHand detail: denoise ([0-9.]+)", param_text)
    if m:
        result["hand_detail"] = {"denoise": float(m.group(1))}

    # ImageGen Studio's own record: exact model / VAE / LoRA files and weights
    rec = _parse_json(info.get("imagegen")) if "imagegen" in info else None
    if isinstance(rec, dict):
        result["imagegen"] = rec
        for k in ("prompt", "negative_prompt", "steps", "seed", "width", "height", "strength"):
            if rec.get(k) not in (None, ""):
                result[k] = rec[k]
        if rec.get("cfg_scale") is not None:
            result["cfg_scale"] = rec["cfg_scale"]
        if rec.get("scheduler"):
            result["sampler"] = rec["scheduler"]
        if (rec.get("model") or {}).get("file"):
            result["model"] = rec["model"]["file"]
            if rec["model"].get("sha256_10"):
                result["model_hash"] = rec["model"]["sha256_10"]
        if rec.get("loras"):
            def _w(v):
                try:
                    return f"{float(v):g}"
                except (TypeError, ValueError):     # a damaged record ("abc") must not break reading it
                    return "0.8"
            result["loras"] = ", ".join(f"{Path(str(l['file'])).stem}:{_w(l.get('weight', 0.8))}"
                                        for l in rec["loras"] if isinstance(l, dict) and l.get("file"))
        if rec.get("vae"):
            result["vae"] = (rec["vae"] or {}).get("file")
        for k in ("clip_skip", "var_seed", "var_strength", "hires", "face_detail", "hand_detail", "inpaint_padding",
                  "pag_scale", "freeu", "cfg_rescale"):
            if rec.get(k) is not None:
                result[k] = rec[k]
        if rec.get("hires") and rec.get("mode") != "img2img":
            result.pop("strength", None)

    return result


def _parse_json(val):
    import json
    try:
        return json.loads(val.decode("utf-8", "replace") if isinstance(val, bytes) else str(val))
    except Exception:
        return None


def _exif_user_comment(img) -> str:
    """A1111 saves JPEG/WebP parameters as EXIF UserComment (tag 0x9286). It lives in the
    Exif sub-IFD (0x8769), not the base IFD, and starts with an 8-byte charset header."""
    try:
        exif = img.getexif()
    except Exception:
        return ""
    if not exif:
        return ""
    val = None
    try:
        val = exif.get_ifd(0x8769).get(0x9286)
    except Exception:
        pass
    if val is None:
        val = exif.get(0x9286)
    if val is None:
        return ""
    nul = chr(0)
    if isinstance(val, bytes):
        head, body = val[:8], val[8:]
        if head.startswith(b"UNICODE"):
            # piexif (A1111) writes UTF-16BE; some tools write UTF-16LE: keep the
            # decoding that yields fewer unprintable characters
            def junk(t):
                return sum(not (c.isprintable() or c in "\r\n\t") for c in t)
            text = min((body.decode(enc, "replace") for enc in ("utf-16-be", "utf-16-le")), key=junk)
            return text.strip(nul).strip()
        if head.startswith(b"ASCII") or head == bytes(8):
            return body.decode("utf-8", "replace").strip(nul).strip()
        return val.decode("utf-8", "replace").strip(nul).strip()
    return str(val).strip(nul).strip()


def _empty_result(img, raw_text: str = "", note: str = "") -> dict[str, Any]:
    return {
        "prompt": "", "negative_prompt": "", "steps": None, "sampler": None,
        "cfg_scale": None, "seed": None,
        "width": getattr(img, "width", None), "height": getattr(img, "height", None),
        "model": None, "loras": None, "raw_text": raw_text, "note": note,
    }


def _from_novelai(img, d: dict, raw: str, description) -> dict[str, Any]:
    r = _empty_result(img, raw)
    r["prompt"] = str(d.get("prompt") or description or "")
    r["negative_prompt"] = str(d.get("uc") or "")
    for key, dst, cast in (("steps", "steps", int), ("scale", "cfg_scale", float),
                           ("seed", "seed", int), ("width", "width", int), ("height", "height", int)):
        try:
            if d.get(key) is not None:
                r[dst] = cast(d[key])
        except (TypeError, ValueError):
            pass
    if d.get("sampler"):
        r["sampler"] = str(d["sampler"])
    r["model"] = "NovelAI"
    return r


def format_png_info_html(meta: dict[str, Any]) -> str:
    """Format parsed metadata into a high-readability HTML presentation."""
    from html import escape
    meta = {k: (escape(v) if isinstance(v, str) else v) for k, v in meta.items()}
    if meta.get("note") and not meta.get("prompt"):
        return f'<p style="color:#9399b2;font-size:13px;padding:12px;">{meta["note"]}. See Raw Parameters.</p>'
    if not meta.get("raw_text") and not meta.get("prompt"):
        return '<p style="color:#9399b2;font-size:13px;padding:12px;">No generation metadata detected in this image.</p>'

    badges = []
    if meta.get("model"):
        badges.append(f'<span style="background:#313244;color:#cdd6f4;padding:3px 8px;border-radius:4px;font-size:13px;">📦 Model: <b>{meta["model"]}</b></span>')
    if meta.get("sampler"):
        badges.append(f'<span style="background:#313244;color:#89b4fa;padding:3px 8px;border-radius:4px;font-size:13px;">🎛️ Sampler: <b>{meta["sampler"]}</b></span>')
    if meta.get("steps") is not None:
        badges.append(f'<span style="background:#313244;color:#a6e3a1;padding:3px 8px;border-radius:4px;font-size:13px;">👣 Steps: <b>{meta["steps"]}</b></span>')
    if meta.get("cfg_scale") is not None:
        badges.append(f'<span style="background:#313244;color:#f9e2af;padding:3px 8px;border-radius:4px;font-size:13px;">🎯 CFG: <b>{meta["cfg_scale"]}</b></span>')
    if meta.get("seed") is not None:
        badges.append(f'<span style="background:#313244;color:#fab387;padding:3px 8px;border-radius:4px;font-size:13px;">🎲 Seed: <b>{meta["seed"]}</b></span>')
    if meta.get("width") and meta.get("height"):
        badges.append(f'<span style="background:#313244;color:#cba6f7;padding:3px 8px;border-radius:4px;font-size:13px;">📐 Size: <b>{meta["width"]}×{meta["height"]}</b></span>')
    if meta.get("loras"):
        badges.append(f'<span style="background:#313244;color:#f38ba8;padding:3px 8px;border-radius:4px;font-size:13px;">🧬 LoRAs: <b>{meta["loras"]}</b></span>')

    if meta.get("vae"):
        badges.append(f'<span style="background:#313244;color:#94e2d5;padding:3px 8px;border-radius:4px;font-size:13px;">🎨 VAE: <b>{meta["vae"]}</b></span>')
    if meta.get("strength") is not None:
        badges.append(f'<span style="background:#313244;color:#fab387;padding:3px 8px;border-radius:4px;font-size:13px;">🖼 img2img strength: <b>{meta["strength"]}</b></span>')
    if meta.get("clip_skip") and int(meta["clip_skip"]) > 1:
        badges.append(f'<span style="background:#313244;color:#b4befe;padding:3px 8px;border-radius:4px;font-size:13px;">✂️ CLIP skip: <b>{meta["clip_skip"]}</b></span>')
    if meta.get("var_strength"):
        badges.append(f'<span style="background:#313244;color:#f5c2e7;padding:3px 8px;border-radius:4px;font-size:13px;">🔀 Variation: <b>{meta.get("var_seed")} ×{meta["var_strength"]}</b></span>')
    if isinstance(meta.get("hires"), dict):
        h = meta["hires"]
        badges.append(f'<span style="background:#313244;color:#89dceb;padding:3px 8px;border-radius:4px;font-size:13px;">🔍 Hires fix: <b>×{h.get("scale")} · denoise {h.get("denoise")} · {h.get("steps")} steps</b></span>')
    if isinstance(meta.get("face_detail"), dict):
        badges.append(f'<span style="background:#313244;color:#f9e2af;padding:3px 8px;border-radius:4px;font-size:13px;">✨ Face detail: <b>denoise {meta["face_detail"].get("denoise")}</b></span>')
    if isinstance(meta.get("hand_detail"), dict):
        badges.append(f'<span style="background:#313244;color:#f9e2af;padding:3px 8px;border-radius:4px;font-size:13px;">✋ Hand detail: <b>denoise {meta["hand_detail"].get("denoise")}</b></span>')
    if meta.get("model_hash"):
        badges.append(f'<span style="background:#313244;color:#a6adc8;padding:3px 8px;border-radius:4px;font-size:13px;">#️⃣ Model hash: <b>{meta["model_hash"]}</b></span>')
    badge_html = " ".join(badges)

    html = f"""
    <div style="background:#1e1e2e;padding:12px;border-radius:8px;border:1px solid #313244;margin:8px 0;">
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
            {badge_html}
        </div>
        <div style="margin-bottom:8px;">
            <b style="color:#a6e3a1;font-size:13px;">➕ Prompt:</b>
            <div style="background:#181825;padding:8px;border-radius:6px;font-size:13px;color:#cdd6f4;white-space:pre-wrap;max-height:120px;overflow-y:auto;border:1px solid #313244;">{meta.get("prompt") or "(empty)"}</div>
        </div>
        <div style="margin-bottom:4px;">
            <b style="color:#f38ba8;font-size:13px;">➖ Negative Prompt:</b>
            <div style="background:#181825;padding:8px;border-radius:6px;font-size:13px;color:#bac2de;white-space:pre-wrap;max-height:80px;overflow-y:auto;border:1px solid #313244;">{meta.get("negative_prompt") or "(empty)"}</div>
        </div>
    </div>
    """
    return html


# Alias for convenience
read_png_info = read_image_metadata
