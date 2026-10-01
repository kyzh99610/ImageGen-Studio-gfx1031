"""
LoRA-restore snapshots: the clean weights of the layers a LoRA changes, kept on DISK.

The pipelines used to copy the whole UNet + text encoders to RAM before the first LoRA (SDXL:
3.4 B params, 6.8 GB). Even only the layers a character LoRA touches were 5.65 GB. That RAM (and
its Windows commit charge) stayed while the model was loaded, and the next SDXL checkpoint load —
safetensors maps the 6.5 GB file copy-on-write, which is charged as commit too — ran a 31 GB PC
with a small pagefile out of commit: the process died with an access violation inside
safetensors.load_file. Now each layer is copied once, just before it is first fused, and spilled
to a temporary safetensors file when the fusion is done; restoring reads it back tensor by tensor.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

COMPONENTS = ("unet", "text_encoder", "text_encoder_2")
_DIR = Path(tempfile.gettempdir()) / "imagegen_lora_snapshots"


def _restore_param(param, saved) -> None:
    """Copy a snapshot tensor back into `param`. A token-embedding table that a textual
    inversion grew after the snapshot (companion TIs load after the first LoRA fuse) gets its
    original rows back and keeps the new tokens' rows — a plain copy_() raised a size error."""
    saved = saved.to(param.device)
    if param.shape == saved.shape:
        param.data.copy_(saved)
    elif param.dim() == 2 and param.shape[1] == saved.shape[1] and param.shape[0] > saved.shape[0]:
        param.data[:saved.shape[0]].copy_(saved)
    else:
        print(f"[LoRA] Snapshot shape {tuple(saved.shape)} doesn't fit {tuple(param.shape)} — left as is")


def _cleanup_stale(max_age_h: float = 12.0):
    """Files of processes that crashed / were killed (never reused across processes)."""
    try:
        for f in _DIR.glob("*.safetensors"):
            pid = int(f.name.split("_")[0]) if f.name.split("_")[0].isdigit() else -1
            alive = False
            if pid > 0 and pid != os.getpid():
                try:
                    import psutil
                    alive = psutil.pid_exists(pid)
                except Exception:
                    alive = time.time() - f.stat().st_mtime < max_age_h * 3600
            if pid != os.getpid() and not alive:
                f.unlink(missing_ok=True)
    except Exception:
        pass


class Snapshot:
    """Clean copies of weights, by component → parameter name. New ones collect in RAM until
    spill() writes them to a file; has() / restore() look in both."""

    def __init__(self):
        self.mem: dict[str, dict] = {c: {} for c in COMPONENTS}
        self.files: list[tuple[Path, dict[str, set]]] = []
        self._ram = 0

    def has(self, comp: str, key: str) -> bool:
        return key in self.mem[comp] or any(key in keys.get(comp, ()) for _, keys in self.files)

    def add(self, comp: str, key: str, tensor) -> None:
        if not self.has(comp, key):
            t = self.mem[comp][key] = tensor.detach().cpu().clone()
            self._ram += t.nbytes
            if self._ram > 512 << 20:
                self.spill()               # keep the RAM part small while a big LoRA is snapshotted

    def spill(self, chunk_bytes: int = 512 << 20) -> None:
        """Move the RAM part to temporary files, ≤ chunk_bytes each: save_file() builds a file's
        whole buffer in memory first — one 5.6 GB file raised MemoryError with commit near its
        limit. Each chunk's tensors are dropped as soon as it is written."""
        if not any(self.mem.values()):
            return
        from safetensors.torch import save_file
        _DIR.mkdir(parents=True, exist_ok=True)
        if not self.files:
            _cleanup_stale()
        items = [(c, k) for c, d in self.mem.items() for k in list(d)]
        while items:
            chunk, size = [], 0
            while items and (not chunk or size + self.mem[items[0][0]][items[0][1]].nbytes <= chunk_bytes):
                c, k = items.pop(0)
                chunk.append((c, k)); size += self.mem[c][k].nbytes
            path = _DIR / f"{os.getpid()}_{id(self)}_{len(self.files)}.safetensors"
            save_file({f"{c}::{k}": self.mem[c][k].contiguous() for c, k in chunk}, str(path))
            keys: dict[str, set] = {}
            for c, k in chunk:
                keys.setdefault(c, set()).add(k)
                del self.mem[c][k]                 # frees this chunk's RAM right away
            self.files.append((path, keys))
        self._ram = 0

    def restore(self, pipe) -> int:
        import torch
        params = {}
        for comp in COMPONENTS:
            model = getattr(pipe, comp, None)
            if model is not None:
                params[comp] = dict(model.named_parameters())
        n = 0
        with torch.no_grad():
            for comp, d in self.mem.items():
                for k, v in d.items():
                    p = params.get(comp, {}).get(k)
                    if p is not None:
                        _restore_param(p, v); n += 1
            if self.files:
                from safetensors import safe_open
                for path, _keys in self.files:
                    with safe_open(str(path), framework="pt", device="cpu") as f:
                        for name in f.keys():
                            comp, _, k = name.partition("::")
                            p = params.get(comp, {}).get(k)
                            if p is not None:
                                _restore_param(p, f.get_tensor(name)); n += 1
        return n

    def close(self) -> None:
        self.mem = {c: {} for c in COMPONENTS}
        self._ram = 0
        for path, _ in self.files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self.files = []

    def size_gb(self) -> float:
        ram = sum(v.numel() * v.element_size() for d in self.mem.values() for v in d.values())
        disk = sum(p.stat().st_size for p, _ in self.files if p.exists())
        return (ram + disk) / 2**30

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def snapshot_wrapped(pipe, snap: Snapshot) -> int:
    """After load_lora_weights(), before fuse_lora(): every PEFT-wrapped layer's clean weights,
    under the name they have once the wrapper is stripped. Returns params added."""
    added = 0
    for comp in COMPONENTS:
        model = getattr(pipe, comp, None)
        if model is None:
            continue
        for name, mod in model.named_modules():
            base = getattr(mod, "base_layer", None)
            if base is None or not hasattr(mod, "lora_A"):
                continue
            for pn, p in base.named_parameters(recurse=False):
                key = f"{name}.{pn}"
                if not snap.has(comp, key):
                    snap.add(comp, key, p); added += 1
    return added


def param_snapshotter(pipe, snap: Snapshot):
    """Callback for in-place changes (LyCORIS fusion): snapshot a parameter once."""
    where = {}
    for comp in COMPONENTS:
        model = getattr(pipe, comp, None)
        if model is not None:
            for n, p in model.named_parameters():
                where[id(p)] = (comp, n)

    def snap_param(p):
        hit = where.get(id(p))
        if hit is not None:
            snap.add(hit[0], hit[1], p)
    return snap_param
