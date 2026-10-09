"""
records.py — the "imagegen" record of a picture and the restore plan made from it (PNG Info → Send to Generate, dropping a picture into img2img, the Upscale tab's carry-over).
Pure helpers moved out of app.py in round 9 (nothing here touches the UI): app.py keeps thin wrappers / aliases with the same names (`_gen_record`, `_restore_plan`, `_clean_extra`,
`_plan_extra_updates`, `_loras_from_meta`, `_match_local`, `_num`) and hands them what only it owns — the model lists, the upscaler's methods, the sampler table and `gr.update()`.
"""
from __future__ import annotations

import re
from pathlib import Path

LORA_TAG = re.compile(r"<lora:([^:>]+)(?::([-\d.]+))?[^>]*>", re.I)       # A1111 <lora:name:weight>


def num(v, default: float) -> float:
    """float(v), or default for None / '' / NaN / text."""
    try:
        f = float(v)
        return default if f != f or f in (float("inf"), float("-inf")) else f
    except (TypeError, ValueError):
        return default


def clean_extra(extra: dict | None, available_methods) -> dict:
    """CLIP skip / variation / hires-fix settings made safe (UI values, saved sessions and
    restored images can all hold junk)."""
    e = dict(extra or {})
    cs = int(round(num(e.get("clip_skip"), 1)))
    vs = num(e.get("var_seed"), -1)
    method = str(e.get("hires_upscaler") or "Lanczos")
    if method != "Lanczos" and method not in available_methods():
        method = "Lanczos"
    return {
        "clip_skip": 2 if cs >= 2 else 1,
        "var_seed": -1 if vs < 0 else int(vs) % 2**32,
        "var_strength": min(1.0, max(0.0, num(e.get("var_strength"), 0.0))),
        "hires_on": bool(e.get("hires_on")),
        "hires_scale": min(2.5, max(1.05, num(e.get("hires_scale"), 1.5))),
        "hires_denoise": min(0.9, max(0.05, num(e.get("hires_denoise"), 0.45))),
        "hires_steps": int(min(150, max(1, round(num(e.get("hires_steps"), 15))))),
        "hires_upscaler": method,
        "fd_on": bool(e.get("fd_on")),
        "fd_denoise": min(0.8, max(0.1, num(e.get("fd_denoise"), 0.4))),
        "fd_mode": e.get("fd_mode") if e.get("fd_mode") in ("auto", "anime", "photo") else "auto",
        "fd_prompt": str(e.get("fd_prompt") or "")[:500],
        "pag_scale": min(6.0, max(0.0, num(e.get("pag_scale"), 0.0))),
        "freeu": bool(e.get("freeu")),
        "cfg_rescale": min(1.0, max(0.0, num(e.get("cfg_rescale"), 0.0))),
        "hd_on": bool(e.get("hd_on")),
        "hd_denoise": min(0.7, max(0.1, num(e.get("hd_denoise"), 0.35))),
        "ed_on": bool(e.get("ed_on")),
        "ed_denoise": min(0.6, max(0.1, num(e.get("ed_denoise"), 0.4))),      # = detail_tools.EYE_DENOISE
        "md_on": bool(e.get("md_on")),                                        # round 14: Real-ESRGAN 2x right before the eye pass
    }


def loras_from_meta(prompt: str, loras_field: str | None):
    """[(name, weight)] from A1111 <lora:…> prompt tags or our 'LoRAs:' field,
    plus the prompt with the tags taken out."""
    found = [(m.group(1).strip(), num(m.group(2), 0.8) if m.group(2) else 0.8)
             for m in LORA_TAG.finditer(prompt or "")]
    clean = LORA_TAG.sub("", prompt or "")
    clean = re.sub(r"\s*,\s*(,\s*)+", ", ", clean)          # ", ," left behind by a tag
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip().strip(",").strip()
    if not found and loras_field:
        for part in loras_field.split(","):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"(.+?)(?:\s*\(×?([\d.]+)\)|:([-\d.]+))?$", part)
            name, w = m.group(1).strip(), m.group(2) or m.group(3)
            found.append((name, num(w, 0.8) if w else 0.8))
    return found, clean


def match_local(name: str | None, items: list, hash10: str | None = None) -> str | None:
    """A local file for a recorded model/VAE/LoRA name: exact file name, then stem (case-
    insensitive), then — if the file was renamed — its cached SHA-256 prefix."""
    if not name and not hash10:
        return None
    paths = [p for _, p in items]
    if name:
        n = Path(str(name)).name.lower()
        stem = Path(n).stem if Path(n).suffix in (".safetensors", ".ckpt", ".pt", ".bin") else n
        for label, path in items:
            if Path(path).name.lower() == n:
                return path
        for label, path in items:
            if Path(path).stem.lower() == stem:
                return path
    if hash10:
        from backend.model_hash import find_by_hash
        return find_by_hash(hash10, paths)
    return None


def restore_plan(meta: dict, *, list_checkpoints, list_vaes, list_loras, scheduler_map) -> dict:
    """What to put in the Generate controls to make an image again, from read_png_info():
    exact files and weights from our 'imagegen' record when present, else A1111 text."""
    rec = meta.get("imagegen") or {}
    plan: dict = {"notes": [], "missing": []}
    lora_found, clean = loras_from_meta(meta.get("prompt") or "", None if rec else meta.get("loras"))
    plan["prompt"] = clean if lora_found else (meta.get("prompt") or None)
    plan["negative_prompt"] = meta.get("negative_prompt") or None
    sampler = (meta.get("sampler") or "").lower()
    plan["scheduler"] = next((n for n in scheduler_map if n.lower() == sampler), None) or next(
        (n for n in scheduler_map if sampler and (sampler in n.lower() or n.lower() in sampler)), None)
    # before record format 2 this app's "DPM++ 2M Karras" / "DPM++ SDE Karras" ran without
    # Karras sigmas — restore what actually made the image
    from backend.sampling import LEGACY_NAMES
    if rec and int(num(rec.get("format"), 1)) < 2 and plan["scheduler"] in LEGACY_NAMES:
        plan["scheduler"] = LEGACY_NAMES[plan["scheduler"]]
    for k in ("steps", "cfg_scale", "seed", "width", "height", "strength"):
        plan[k] = meta.get(k)
    # checkpoint
    mfile = (rec.get("model") or {}).get("file") or meta.get("model")
    mhash = (rec.get("model") or {}).get("sha256_10") or meta.get("model_hash")
    plan["model"] = match_local(mfile, list_checkpoints(), mhash)
    if mfile and not plan["model"]:
        if re.fullmatch(r"[\w.-]+/[\w.-]+", str(mfile)):             # a Hugging Face repo ID
            plan["model"] = str(mfile)
        else:
            plan["missing"].append(f"checkpoint {mfile}")
    # VAE
    vfile = (rec.get("vae") or {}).get("file") if rec else meta.get("vae")
    if rec and rec.get("vae") is None and "vae" in rec:
        plan["vae"] = "none"
    elif vfile:
        plan["vae"] = match_local(vfile, list_vaes(), (rec.get("vae") or {}).get("sha256_10"))
        if not plan["vae"]:
            plan["missing"].append(f"VAE {vfile}")
    else:
        plan["vae"] = None
    # LoRAs: exact files + weights from our record, else prompt tags / "LoRAs:" field
    wanted = ([(l.get("file"), l.get("weight", 0.8), l.get("sha256_10")) for l in rec.get("loras") or []
               if isinstance(l, dict)] if rec.get("loras") is not None else
              [(n, w, None) for n, w in lora_found])
    loras = []
    for name, w, h in wanted:
        path = match_local(name if Path(str(name)).suffix else f"{name}.safetensors", list_loras(), h) \
            or match_local(name, list_loras(), h)
        if path:
            loras.append((path, min(1.5, max(0.1, float(num(w, 0.8))))))
        else:
            plan["missing"].append(f"LoRA {name}")
    if len(loras) > 3:
        plan["notes"].append("only 3 LoRA slots — skipped " + ", ".join(Path(p).stem for p, _ in loras[3:]))
    plan["loras"] = loras[:3] if (wanted or rec) else None      # None = leave the slots alone
    # CLIP skip / variation / hires: part of reproducing the image, so always set (defaults = off)
    plan["clip_skip"] = 2 if int(num(meta.get("clip_skip"), 1)) >= 2 else 1
    plan["var_seed"] = meta.get("var_seed") if meta.get("var_strength") else -1
    plan["var_strength"] = float(num(meta.get("var_strength"), 0.0)) if meta.get("var_seed") is not None else 0.0
    plan["hires"] = meta.get("hires") if isinstance(meta.get("hires"), dict) else None
    plan["face_detail"] = meta.get("face_detail") if isinstance(meta.get("face_detail"), dict) else None
    plan["hand_detail"] = meta.get("hand_detail") if isinstance(meta.get("hand_detail"), dict) else None
    plan["eye_detail"] = meta.get("eye_detail") if isinstance(meta.get("eye_detail"), dict) else None
    plan["max_detail"] = meta.get("max_detail") if isinstance(meta.get("max_detail"), dict) else None
    plan["pag_scale"] = float(num(meta.get("pag_scale"), 0.0))
    plan["freeu"] = bool(meta.get("freeu"))
    plan["cfg_rescale"] = float(num(meta.get("cfg_rescale"), 0.0))
    plan["mode"] = rec.get("mode") or ("img2img" if meta.get("strength") is not None else "txt2img")
    # weighted prompts look different under the other weighting mode: say so (images from before the
    # A1111 mode existed were made with Compel's)
    from backend.prompt_syntax import EMPHASIS
    made = meta.get("emphasis") or (rec.get("emphasis") if rec else None) or ("compel" if rec else "a1111")
    if made != EMPHASIS["mode"] and re.search(r"\([^()]*:\s*\d*\.?\d+\s*\)|\(\(|\[", str(meta.get("prompt") or "")):
        plan["notes"].append(f"made with {'Compel' if made == 'compel' else 'A1111'} prompt weighting — switch "
                             f"Settings → Prompt weights to reproduce it exactly")
    plan["exact"] = bool(rec)
    return plan


def plan_extra_updates(plan: dict, clean_extra, keep) -> list:
    """CLIP skip, variation seed/strength and hires-fix controls for a restore plan
    (hires sliders keep their values when the image didn't use hires fix)."""
    h = plan.get("hires") or {}
    ex = clean_extra(dict(clip_skip=plan.get("clip_skip"), var_seed=plan.get("var_seed"),
                           var_strength=plan.get("var_strength"), hires_on=bool(h),
                           hires_scale=h.get("scale"), hires_denoise=h.get("denoise"),
                           hires_steps=h.get("steps"), hires_upscaler=h.get("upscaler")))
    fd = plan.get("face_detail") or {}
    fdx = clean_extra(dict(fd_on=bool(fd), fd_denoise=fd.get("denoise"), fd_mode=fd.get("detector"),
                            fd_prompt=fd.get("prompt")))
    hd = plan.get("hand_detail") or {}
    hdx = clean_extra(dict(hd_on=bool(hd), hd_denoise=hd.get("denoise")))
    ed = plan.get("eye_detail") or {}
    edx = clean_extra(dict(ed_on=bool(ed), ed_denoise=ed.get("denoise")))
    return [ex["clip_skip"], ex["var_seed"], ex["var_strength"], bool(h),
            ex["hires_scale"] if h else keep, ex["hires_denoise"] if h else keep,
            ex["hires_steps"] if h else keep, ex["hires_upscaler"] if h else keep,
            bool(fd), fdx["fd_denoise"] if fd else keep, fdx["fd_mode"] if fd else keep,
            fdx["fd_prompt"] if fd else keep,
            # boosters are part of how the image looks: always set (off when the record has none)
            *(lambda b: [b["pag_scale"], b["freeu"], b["cfg_rescale"]])(clean_extra(dict(
                pag_scale=plan.get("pag_scale"), freeu=plan.get("freeu"), cfg_rescale=plan.get("cfg_rescale")))),
            bool(hd), hdx["hd_denoise"] if hd else keep,
            bool(ed), edx["ed_denoise"] if ed else keep,
            bool(plan.get("max_detail"))]


def gen_record(pipe, **settings) -> dict:
    """Everything needed to make an image again: settings + the exact model, VAE and LoRA
    files (name, AutoV2 hash when known) that were loaded in `pipe`."""
    from backend.model_hash import autov2, hash_later
    rec = {"app": "ImageGen Studio", "format": 2}   # 2: Karras samplers use Karras sigmas
    from backend.prompt_syntax import EMPHASIS
    rec["emphasis"] = EMPHASIS["mode"]               # how (tag:1.2) weights were applied
    rec.update({k: v for k, v in settings.items() if v is not None})
    if pipe is not None:
        mp = str(getattr(pipe, "current_model", "") or "")
        vp = getattr(pipe, "_last_vae_path", None)
        loras = [(name, path, w) for _, (name, path, w) in sorted(getattr(pipe, "_lora_adapters", {}).items())]
        hash_later(mp, vp, *[path for _, path, _ in loras])
        # a local file → its name; a Hugging Face repo ID ("org/name-1.0") → kept whole
        rec["model"] = {"file": Path(mp).name if Path(mp).is_file() else mp,
                        "family": getattr(pipe, "model_family", ""), "sha256_10": autov2(mp)}
        if (getattr(pipe, "prediction", None) or {}).get("v_pred"):
            rec["model"]["prediction"] = "v"
        rec["vae"] = {"file": Path(vp).name, "sha256_10": autov2(vp)} if vp else None
        rec["loras"] = [{"file": Path(path).name, "weight": round(float(w), 3), "sha256_10": autov2(path)}
                        for name, path, w in loras]
    return rec
