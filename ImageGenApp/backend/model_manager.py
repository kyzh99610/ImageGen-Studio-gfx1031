"""
backend/model_manager.py
Scan local model directories and enumerate available checkpoints, LoRAs, VAEs,
upscalers, and embeddings for use in the UI dropdowns.
"""

from __future__ import annotations

from pathlib import Path

from config import CHECKPOINTS_DIR, LORAS_DIR, VAE_DIR, UPSCALERS_DIR, EMBEDDINGS_DIR

_SUPPORTED_EXTS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx"}


def _scan(directory: Path) -> list[tuple[str, str]]:
    if not directory.exists():
        return []
    return sorted(
        ((p.name, str(p)) for p in directory.rglob("*")
         if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS),
        key=lambda t: t[0].lower(),   # case-insensitive, like Explorer
    )


_ARCH_CACHE: dict[tuple[str, float], str | None] = {}
_LORA_ARCH_CACHE: dict[tuple[str, float], str | None] = {}


def safetensors_problem(path: str) -> str:
    """Plain-English reason a .safetensors file can't be loaded ('' if it looks intact):
    the header records where every tensor ends, so a cut-short download or copy shows up
    as a file smaller than its own header says it should be."""
    import json, struct
    p = Path(str(path))
    if p.suffix.lower() != ".safetensors" or not p.is_file():
        return ""
    size = p.stat().st_size
    try:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 2 or 8 + n > size or n > 200 << 20:
                return f"{p.name} isn't a valid .safetensors file (damaged header)"
            hdr = json.loads(f.read(n))
        end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
    except Exception:
        return f"{p.name} isn't a valid .safetensors file (unreadable header)"
    need = 8 + n + end
    if size < need:
        return (f"{p.name} is incomplete — {size / 2**30:.2f} of {need / 2**30:.2f} GB "
                f"(the download or copy was cut short). Download it again.")
    return ""


def lycoris_kind(path: str) -> str | None:
    """'LoHa' / 'LoKr' if the file uses a LyCORIS format diffusers can't load, else None."""
    import json, struct
    p = Path(str(path))
    if p.suffix.lower() != ".safetensors" or not p.is_file():
        return None
    try:
        with open(p, "rb") as f:
            keys = " ".join(json.loads(f.read(struct.unpack("<Q", f.read(8))[0])).keys())
    except Exception:
        return None
    return "LoHa" if ".hada_w1" in keys else "LoKr" if ".lokr_w" in keys else None


def lora_arch(path: str) -> str | None:
    """'sdxl' / 'sd2' / 'sd1' for a LoRA, from tensor names + shapes in the safetensors
    header (no weights loaded). Cross-attention to_k input width is the tell:
    768 = SD 1.x (CLIP-L), 1024 = SD 2.x (OpenCLIP-H), 2048 = SDXL (L+G). None if unsure."""
    import json, re, struct
    p = Path(str(path))
    if p.suffix.lower() != ".safetensors" or not p.is_file():
        return None
    key = (str(p), p.stat().st_mtime)
    if key not in _LORA_ARCH_CACHE:
        arch = None
        try:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            names = [k for k in hdr if k != "__metadata__"]
            if any(("lora_te2_" in k) or ("text_encoder_2" in k) for k in names):
                arch = "sdxl"
            else:
                for k in names:
                    # kohya: ..._attn2_to_k.lora_down.weight · diffusers/PEFT: ...attn2.to_k.lora_A.weight
                    # LyCORIS LoHa: ..._attn2_to_k.hada_w1_b  ([rank, in_features])
                    if re.search(r"attn2[._]to_k\.((lora_down|lora_A|lora\.down)\.weight|hada_w1_b)$", k):
                        width = hdr[k]["shape"][-1]
                        arch = {768: "sd1", 1024: "sd2", 2048: "sdxl"}.get(width)
                        break
        except Exception:
            pass
        _LORA_ARCH_CACHE[key] = arch
    return _LORA_ARCH_CACHE[key]


def checkpoint_arch(path: str) -> str | None:
    """'sdxl' / 'sd2' / 'sd1' from a checkpoint's tensor names — reads only the
    safetensors JSON header, not the weights. None if unknown / not safetensors.
    Authoritative where filename heuristics fail (e.g. SDXL models without 'xl' in the name)."""
    import json, struct
    p = Path(str(path))
    if p.suffix.lower() != ".safetensors" or not p.is_file():
        return None
    key = (str(p), p.stat().st_mtime)
    if key not in _ARCH_CACHE:
        arch = None
        try:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                names = " ".join(json.loads(f.read(n)).keys())
            if "conditioner.embedders.1" in names:
                arch = "sdxl"
            elif "cond_stage_model.model.transformer" in names:
                arch = "sd2"
            elif "cond_stage_model.transformer" in names:
                arch = "sd1"
        except Exception:
            pass
        _ARCH_CACHE[key] = arch
    return _ARCH_CACHE[key]


_VAE_FAMILY: dict = {}


def vae_family(path: str) -> str | None:
    """'sdxl' / 'sd1' for a VAE file (or a checkpoint's built-in VAE), None if it can't be told.
    Both VAEs have identical tensor shapes — an SD 1.5 VAE loads into SDXL without an error and decodes
    garbage — but their quant_conv bias differs by an order of magnitude: |b| ≈ 43.8 for every SDXL VAE
    here (base, Pony, Illustrious, NoobAI), 4.4–6.4 for SD 1.x ones (ft-mse, anything, orangemix …).
    Reads just that tensor from the file (no mmap of a 6 GB checkpoint)."""
    import json as _j, struct
    import numpy as np
    p = Path(str(path))
    key = (str(p), p.stat().st_mtime if p.is_file() else 0)
    if key in _VAE_FAMILY:
        return _VAE_FAMILY[key]
    fam = None
    try:
        if p.suffix.lower() == ".safetensors":
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = _j.loads(f.read(n))
                name = next((k for k in ("quant_conv.bias", "first_stage_model.quant_conv.bias",
                                         "vae.quant_conv.bias") if k in hdr), None)
                if name:
                    info = hdr[name]
                    a, b = info["data_offsets"]
                    f.seek(8 + n + a)
                    raw = f.read(b - a)
                    if info["dtype"] == "BF16":
                        t = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
                    else:
                        t = np.frombuffer(raw, {"F16": np.float16, "F32": np.float32}[info["dtype"]]).astype(np.float32)
                    fam = "sdxl" if float(np.linalg.norm(t)) > 20 else "sd1"
    except Exception:
        fam = None
    _VAE_FAMILY[key] = fam
    return fam


def list_checkpoints() -> list[str]:
    return _scan(CHECKPOINTS_DIR)


def list_loras() -> list[str]:
    return _scan(LORAS_DIR)


def list_vaes() -> list[str]:
    return _scan(VAE_DIR)


def list_upscalers() -> list:
    models = _scan(UPSCALERS_DIR)
    # Always offer Lanczos (PIL built-in)
    return [("Lanczos (built-in)", "Lanczos (built-in)")] + models


def list_embeddings() -> list[str]:
    return _scan(EMBEDDINGS_DIR)


def list_all() -> dict[str, list[str]]:
    return {
        "checkpoints": list_checkpoints(),
        "loras":       list_loras(),
        "vaes":        list_vaes(),
        "upscalers":   list_upscalers(),
    }


def short_name(path: str) -> str:
    """Return the file stem (no extension, no directory)."""
    return Path(path).stem


def _read_sf_metadata(path: str) -> dict:
    """Read JSON metadata header from a .safetensors file without loading tensors."""
    import json, struct
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n)).get("__metadata__", {})
    except Exception:
        return {}


def _detect_sd_version(meta: dict, path: str) -> str:
    """Infer SD version from Civitai sidecar, safetensors metadata, or filename."""
    import json as _json, re
    # 1) Try Civitai sidecar JSON first
    sidecar = Path(str(path) + ".civitai.json")
    if sidecar.exists():
        try:
            bm = str(_json.loads(sidecar.read_text(encoding="utf-8")).get("baseModel") or "").lower()
            if "pony" in bm:
                return "Pony (SDXL)"
            if "illustrious" in bm or "noob" in bm:
                return "Illustrious (SDXL)"
            if "sdxl" in bm or "xl" in bm:
                return "SDXL"
            if "sd 1" in bm or "sd1" in bm:
                return "SD 1.x"
            if "sd 2" in bm or "sd2" in bm:
                return "SD 2.x"
            if "flux" in bm:
                return "FLUX"
        except Exception:
            pass
    # 2) Tensor names (authoritative for SD1/SD2 vs SDXL) — LoRAs via their cross-attention width
    #    (checkpoint_arch alone is None for LoRAs, and 7 of 22 SDXL LoRAs showed "SD 1.x"; F-16)
    hdr = checkpoint_arch(path) or lora_arch(path)
    if hdr == "sd1":
        return "SD 1.x"
    if hdr == "sd2":
        return "SD 2.x"
    if hdr == "sdxl":          # the base a LoRA was trained on names its lineage
        base = str(meta.get("ss_sd_model_name", "")).lower()
        if "pony" in base or re.search(r"(^|\D)290640(\D|$)", base):
            return "Pony (SDXL)"
        if any(x in base for x in ("illustrious", "noob", "wai")):
            return "Illustrious (SDXL)"
    # 3) Safetensors metadata / filename (also tells Pony / Illustrious apart)
    arch = (meta.get("modelspec.architecture") or
            meta.get("ss_base_model_version") or "").lower()
    name = Path(path).name.lower()
    if "pony" in name or "pony" in arch:
        return "Pony (SDXL)"
    if any(x in name for x in ("illustrious", "noobai")) or "illustrious" in arch:
        return "Illustrious (SDXL)"
    # "IL" / "ill" as standalone token
    if re.search(r'(?:^|[-_ ])il(?:l)?(?:[-_ .]|$)', name):
        return "Illustrious (SDXL)"
    if "sdxl" in arch or "xl_base" in arch or "sdxl" in name or "_xl" in name:
        return "SDXL"
    if hdr == "sdxl":
        return "SDXL"
    if "sd_v2" in arch or "v2" in arch:
        return "SD 2.x"
    if "sd_v1" in arch or "v1" in arch:
        return "SD 1.x"
    if "xl" in name:
        return "SDXL"
    return "SD 1.x"


def get_model_info(path: str) -> str:
    """
    Return an HTML snippet with file size, SD version, training metadata,
    and Civitai specialty tags (when sidecar available).
    Returns empty string for 'none' / missing paths.
    """
    if not path or path == "none" or not Path(path).exists():
        return ""
    try:
        from backend.trigger_reader import get_model_specialty_html
        size_mb = Path(path).stat().st_size / 1024 / 1024
        size_str = f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.0f} MB"

        meta = _read_sf_metadata(path)
        sd_ver = _detect_sd_version(meta, path)

        parts = [f"<b>{size_str}</b>", sd_ver]

        # LoRA-specific fields
        steps = meta.get("ss_steps") or meta.get("ss_max_train_steps")
        try:
            if steps:
                parts.append(f"steps: {int(float(steps)):,}")
        except (TypeError, ValueError):
            pass
        rank = meta.get("ss_network_dim")
        if rank:
            parts.append(f"rank: {rank}")
        alpha = meta.get("ss_network_alpha")
        if alpha:
            parts.append(f"α: {alpha}")
        epochs = meta.get("ss_epoch")
        if epochs:
            parts.append(f"epoch: {epochs}")

        # Generic modelspec fields
        title = meta.get("modelspec.title") or meta.get("ss_output_name")
        if title:
            from html import escape
            parts.insert(0, f'<span style="font-weight:bold;color:#cdd6f4;">{escape(str(title))}</span>')

        base_html = (
            '<p style="font-size:13px;color:#a6adc8;margin:2px 0 4px;line-height:1.4;">'
            + " &nbsp;·&nbsp; ".join(parts) + "</p>"
        )

        # Append Civitai specialty badges (only if sidecar exists)
        specialty_html = get_model_specialty_html(path)
        return base_html + specialty_html
    except Exception:
        return ""


def model_dir_summary() -> str:
    """Human-readable summary of what is installed."""
    m = list_all()
    lines = [
        f"Checkpoints : {len(m['checkpoints'])}",
        f"LoRAs       : {len(m['loras'])}",
        f"VAEs        : {len(m['vaes'])}",
        f"Upscalers   : {len(m['upscalers']) - 1}",  # subtract Lanczos
    ]
    return "\n".join(lines)
