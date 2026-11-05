"""
onnx_tools.py - torch.onnx.export for both Python environments. From torch 2.9 on (native ROCm has 2.14) it defaults to the dynamo exporter, which needs onnxscript, and every export
failed there; the TorchScript exporter is the one these exports were written and checked for (torch 2.4 under ZLUDA).
"""
from __future__ import annotations

import inspect
from pathlib import Path


EXPORT_COMMIT_GB = 6.0      # a VAE decoder export peaked at 4.5 GB of commit (no_grad, native ROCm)


def export_onnx(model, args, path, **kwargs) -> None:
    """torch.onnx.export with the TorchScript exporter; a failed export leaves no file behind (the callers treat an existing file as a finished export)."""
    import torch
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    from backend.memory_guard import make_room
    make_room(EXPORT_COMMIT_GB, keep=("the Generate tab's model",))   # the model being exported must stay
    try:
        # without no_grad the tracer kept the activations' autograd graph: a VAE decoder export peaked at 11 GB of commit and kept 10 GB
        # afterwards (native ROCm, SD 1.5); with it 4.5 GB and 3 GB
        with torch.no_grad():
            torch.onnx.export(model, args, str(path), **kwargs)
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise
