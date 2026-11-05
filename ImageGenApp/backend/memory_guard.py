"""
memory_guard.py - lets cached helper models go when Windows commit (RAM + page file) runs short. On native ROCm every GPU byte is commit too, and what
the app keeps loaded for speed (Real-ESRGAN, EasyOCR, LaMa, BLIP, the accelerated text encoders) added up: a session that had used them took free commit
to 0.8 GB at the next model (RX 9070 XT, 32 GB RAM, the default page file). Each registers a release function; make_room(gb) calls them, cheapest to
reload first and the Generate tab's model last, until that much is free. Whatever is released loads again on its next use.
"""
from __future__ import annotations

import gc

_RELEASE: dict[str, tuple[int, object]] = {}


def register(name: str, release, last: bool = False) -> None:
    """`release()` drops the model and returns True when something was loaded. `last`: only when nothing else is left (the generation model)."""
    _RELEASE[name] = (1 if last else 0, release)


def make_room(gb: float, keep: tuple = ()) -> list[str]:
    """Release cached models until `gb` of commit is free or nothing is left; returns the names released, in order."""
    from backend.model_manager import avail_commit_gb
    freed = []
    for name, (_, release) in sorted(_RELEASE.items(), key=lambda kv: kv[1][0]):
        free = avail_commit_gb()
        if free is None or free >= gb:
            break
        if name in keep:
            continue
        try:
            if release():
                freed.append(name)
        except Exception as e:
            print(f"[Memory] releasing {name} failed: {e}")
            continue
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    if freed:
        print(f"[Memory] free commit was under {gb:.1f} GB: released {', '.join(freed)} (each loads again when it is used)")
    return freed
