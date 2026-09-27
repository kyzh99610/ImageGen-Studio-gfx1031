"""
selftest_zluda.py — verify the GPU backend gives *correct* results, not just "no crash".

Runs the ops Stable Diffusion depends on (GEMM, conv, attention, GroupNorm) on
the GPU and compares each against a float32 CPU reference. A broken backend
(e.g. MIOpen convolutions under ZLUDA on gfx1031, which return NaN/garbage)
fails here instead of producing black images later.

Usage (from ImageGenApp):
    run_zluda.bat selftest_zluda.py            # app config (cuDNN off on AMD)
    run_zluda.bat selftest_zluda.py --cudnn    # force MIOpen on, to compare
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if "--cudnn" in sys.argv:
    os.environ["IMAGEGEN_CUDNN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

import config

FAILED = 0


def check(name: str, fn, *cpu_args, tol: float = 2e-2):
    """Run fn on GPU (fp16) and CPU (fp32); compare relative error."""
    global FAILED
    dev = config.DEVICE
    gpu_args = [a.to(dev, torch.float16) if torch.is_tensor(a) else a for a in cpu_args]
    try:
        torch.cuda.synchronize()
        t = time.time()
        out = fn(*gpu_args)
        torch.cuda.synchronize()
        ms = (time.time() - t) * 1000
        ref = fn(*cpu_args)
        out = out.float().cpu()
        if torch.isnan(out).any() or torch.isinf(out).any():
            raise AssertionError("NaN/Inf in GPU output")
        rel = ((out - ref).norm() / ref.norm().clamp_min(1e-6)).item()
        ok = rel < tol
        print(f"  {'✅' if ok else '❌'} {name:42s} rel.err {rel:.2e}  {ms:7.1f} ms")
        FAILED += not ok
    except Exception as e:  # report and keep going
        print(f"  ❌ {name:42s} {type(e).__name__}: {str(e)[:90]}")
        FAILED += 1


def main() -> int:
    print("=" * 72)
    print(" ZLUDA / GPU backend self-test")
    print("=" * 72)
    print(f"  device       : {config.DEVICE}")
    if not config.DEVICE.startswith("cuda"):
        print("  ❌ CUDA/ZLUDA device not active — launch via run_zluda.bat / launch.bat")
        return 1
    print(f"  gpu          : {torch.cuda.get_device_name()}")
    print(f"  torch        : {torch.__version__}")
    print(f"  cuDNN/MIOpen : {'ON' if torch.backends.cudnn.enabled else 'off (native convs)'}")
    for k in ("ROCM_ARCH", "HIP_PATH", "ROCBLAS_TENSILE_LIBPATH", "DISABLE_ADDMM_CUDA_LT"):
        print(f"  {k:24s}: {os.environ.get(k, '(unset)')}")
    print()

    torch.manual_seed(0)
    r = lambda *s: torch.randn(*s)

    a, b = r(1024, 1024), r(1024, 1024)
    check("matmul 1024² (rocBLAS)", torch.matmul, a, b)
    # Real SD shapes: rocBLAS picks a kernel per shape, so a square test alone can pass
    # while e.g. CLIP's 77-token GEMMs have no working kernel.
    for m, k_, n in ((77, 768, 3072), (4096, 640, 640), (1024, 1280, 5120), (154, 2048, 1280)):
        check(f"linear+bias {m}×{k_}→{n} (addmm / cublasLt)",
              lambda x, w, bias: F.linear(x, w, bias), r(m, k_), r(n, k_) * 0.03, r(n))
    check("batched matmul 16×(77×64 @ 64×77)", torch.bmm, r(16, 77, 64), r(16, 64, 77))

    for c, h, w in ((320, 64, 64), (1280, 16, 16)):
        x, k = r(2, c, h, w), r(c, c, 3, 3) * 0.02
        check(f"conv3x3 UNet-sized {c}×{h}×{w}", lambda x, k: F.conv2d(x, k, padding=1), x, k)
    x, k = r(1, 128, 256, 256), r(128, 128, 3, 3) * 0.02
    check("conv3x3 VAE-sized 128×256×256", lambda x, k: F.conv2d(x, k, padding=1), x, k)

    q = r(2, 8, 1024, 40)
    check("attention (SDPA math) 8×1024×40",
          lambda q: F.scaled_dot_product_attention(q, q, q), q)

    x, g, bb = r(2, 320, 32, 32), torch.ones(320), torch.zeros(320)
    check("groupnorm 32 groups", lambda x, g, bb: F.group_norm(x, 32, g, bb), x, g, bb)
    x, g, bb = r(2, 77, 768), torch.ones(768), torch.zeros(768)
    check("layernorm 77×768 (text encoder)",
          lambda x, g, bb: F.layer_norm(x, (768,), g, bb), x, g, bb)
    x, k = r(2, 640, 32, 32), r(320, 640, 1, 1) * 0.04
    check("conv1x1 640→320", lambda x, k: F.conv2d(x, k), x, k)
    check("nearest upsample ×2 (UNet)",
          lambda x: F.interpolate(x, scale_factor=2.0, mode="nearest"), r(2, 320, 32, 32))

    print()
    if FAILED:
        print(f"❌ {FAILED} check(s) failed — GPU results are wrong on this backend.")
        if torch.backends.cudnn.enabled:
            print("   cuDNN/MIOpen is ON; rerun without --cudnn / IMAGEGEN_CUDNN.")
        return 1
    print("✅ All checks passed — GPU output matches CPU reference.")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # ZLUDA processes can hang during interpreter shutdown
