"""
lycoris.py — apply LyCORIS LoHa / LoKr files (kohya naming) by fusing them straight
into the model weights.

diffusers 0.36 only loads plain LoRA/LoCon (lora_down/lora_up). LoHa and LoKr store the
weight change differently, so we rebuild each layer's ΔW here (same maths as the LyCORIS
reference / ComfyUI) and add it to the matching nn.Linear / nn.Conv2d weight. The
pipelines restore their CPU snapshot before every LoRA reload, so this is undone the
same way a fused normal LoRA is.
"""
from __future__ import annotations

from pathlib import Path

import torch

# kohya key prefix → pipeline attribute holding that module tree
_PREFIXES = (
    ("lora_unet_", "unet"),
    ("lora_te_", "text_encoder"),      # SD 1.x single text encoder
    ("lora_te1_", "text_encoder"),     # SDXL CLIP-L
    ("lora_te2_", "text_encoder_2"),   # SDXL OpenCLIP-G
)


def _module_map(pipe) -> dict[str, torch.nn.Module]:
    """kohya layer name (lora_unet_down_blocks_0_…_to_k) → module with a .weight."""
    out = {}
    for prefix, attr in _PREFIXES:
        root = getattr(pipe, attr, None)
        if root is None:
            continue
        for name, mod in root.named_modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
                out[prefix + name.replace(".", "_")] = mod
    return out


def _cp(t, wa, wb):
    """Rebuild a Tucker-decomposed conv weight: [out, in, k, k]."""
    return torch.einsum("i j k l, j r, i p -> p r k l", t, wb, wa)


def _delta(g: dict[str, torch.Tensor]) -> tuple[torch.Tensor, float]:
    """ΔW (unscaled) and its alpha/rank scale for one layer's tensors."""
    alpha = float(g["alpha"]) if "alpha" in g else None
    if "hada_w1_a" in g:                                           # LoHa
        if "hada_t1" in g:
            w1 = _cp(g["hada_t1"], g["hada_w1_a"], g["hada_w1_b"])
            w2 = _cp(g["hada_t2"], g["hada_w2_a"], g["hada_w2_b"])
        else:
            w1 = g["hada_w1_a"] @ g["hada_w1_b"]
            w2 = g["hada_w2_a"] @ g["hada_w2_b"]
        dim = g["hada_w1_b"].shape[0]
        return w1 * w2, (alpha / dim if alpha is not None else 1.0)
    if any(k.startswith("lokr_") for k in g):                      # LoKr
        dim = None
        if "lokr_w1" in g:
            w1 = g["lokr_w1"]
        else:
            w1 = g["lokr_w1_a"] @ g["lokr_w1_b"]
            dim = g["lokr_w1_b"].shape[0]
        if "lokr_w2" in g:
            w2 = g["lokr_w2"]
        elif "lokr_t2" in g:
            w2 = _cp(g["lokr_t2"], g["lokr_w2_a"], g["lokr_w2_b"])
            dim = g["lokr_w2_b"].shape[0]
        else:
            w2 = g["lokr_w2_a"] @ g["lokr_w2_b"]
            dim = g["lokr_w2_b"].shape[0]
        if w2.dim() == 4 and w1.dim() == 2:
            w1 = w1[:, :, None, None]
        scale = alpha / dim if (alpha is not None and dim) else 1.0
        return torch.kron(w1, w2), scale
    if "lora_down.weight" in g:                                    # plain LoRA / LoCon layer
        up, down = g["lora_up.weight"], g["lora_down.weight"]
        rank = down.shape[0]
        if down.dim() == 4:
            d = torch.einsum("o r a b, r i k l -> o i k l", up, down) if up.dim() == 4 \
                else (up @ down.flatten(1)).reshape(up.shape[0], *down.shape[1:])
        else:
            d = up @ down
        return d, (alpha / rank if alpha is not None else 1.0)
    raise ValueError(f"unsupported LyCORIS layer (keys: {', '.join(sorted(g))})")


@torch.no_grad()
def apply_lycoris(pipe, path: str, weight: float) -> tuple[int, int]:
    """Fuse a LoHa/LoKr file into pipe's UNet/text encoders at `weight`.
    Returns (layers applied, layers with no matching module)."""
    from safetensors.torch import load_file
    sd = load_file(str(path), device="cpu")
    if any(k.endswith("dora_scale") for k in sd):
        raise ValueError(f"{Path(path).name} uses DoRA, which isn't supported")
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        layer, _, part = k.partition(".")
        groups.setdefault(layer, {})[part] = v
    modules = _module_map(pipe)
    applied = skipped = 0
    for layer, g in groups.items():
        mod = modules.get(layer)
        if mod is None:
            skipped += 1
            continue
        d, scale = _delta({k: v.float() for k, v in g.items()})
        w = mod.weight
        if d.numel() != w.numel():
            raise ValueError(
                f"{Path(path).name} doesn't fit this model ({layer}: {tuple(d.shape)} vs "
                f"{tuple(w.shape)}) — it was made for a different base model.")
        w.add_((d.reshape(w.shape) * (scale * weight)).to(device=w.device, dtype=w.dtype))
        applied += 1
    if applied == 0:
        raise ValueError(f"{Path(path).name}: none of its layers match this model "
                         f"(different base model or naming).")
    return applied, skipped
