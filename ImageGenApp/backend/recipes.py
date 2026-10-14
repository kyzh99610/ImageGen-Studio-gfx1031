"""⚡ Use the measured recipe: the sampler settings that were measured per checkpoint family (pure data, no UI).

Only families with measurements are listed — a family without one keeps the user's settings (the button says so).
Illustrious / NoobAI: AYS 12 steps, CFG 6, PAG 2 (A1111 prompt weights). Measured on five Illustrious / NoobAI checkpoints with one character LoRA over rounds 3–15:
the card recipe 'sits on a plateau' — every lever tried alone (CFG 5, LoRA 0.9, Euler a 20 steps, 768×1344, CFG rescale, PAG 3, weighted eye tags) was no better or worse, paired on fresh seeds;
10-step bases, no PAG (−24 % whole-picture detail) and Compel weights cost something.
"""
from __future__ import annotations

FAMILY_LABEL = {"illustrious": "Illustrious / NoobAI", "pony": "Pony", "sdxl": "SDXL base", "sd15": "SD 1.5"}

MEASURED: dict[str, dict] = {
    "illustrious": dict(scheduler="DPM++ 2M AYS", steps=12, cfg=6.0, pag=2.0,
                        basis="measured on Illustrious / NoobAI anime checkpoints"),
}


def family_label(family: str | None) -> str:
    return FAMILY_LABEL.get(str(family or "").lower(), str(family or "an unknown family"))


def recipe_for(family: str | None) -> dict | None:
    """A copy of the measured recipe of a family, or None when nothing was measured for it."""
    r = MEASURED.get(str(family or "").lower())
    return dict(r) if r else None


def summary(r: dict) -> str:
    return f"{r['scheduler']} · {r['steps']} steps · CFG {r['cfg']:g} · PAG {r['pag']:g}"
