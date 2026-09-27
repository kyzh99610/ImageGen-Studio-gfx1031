"""
rocm_env.py — pick the HIP/ROCm runtime and rocBLAS kernels that actually work for the
installed AMD GPU, *before* ZLUDA/torch start (standard library only).

Measured on the RX 6800M (gfx1031, 2026-09-26):
  • therock_sdk (bundled, ~2× faster) and the system HIP SDK 6.4 both run under ZLUDA v6.
  • gfx1031 needs the separately bundled gfx1031 rocBLAS kernels: therock's own gfx1031
    files fail every GEMM, and HIP SDK 6.4 has none (the process crashes).
  • HSA_OVERRIDE_GFX_VERSION does nothing on Windows (a bogus value passes the self-test).
Other architectures must NOT get the gfx1031-only library, and therock only carries RDNA 1/2
kernels (gfx101x/103x) — RDNA 3/4 (gfx110x/gfx120x) need the system HIP SDK 6.4+.

Which GPU: IMAGEGEN_GPU (HIP device index) when set, otherwise the best one hipInfo lists —
a discrete card before an integrated one, then the most memory. HIP device order is also
ZLUDA's CUDA order, so the index doubles as the torch device index.

CLI (used by the launchers):  python backend/rocm_env.py --bat   → prints `set "K=V"` lines
                              python backend/rocm_env.py --json  → prints the decision
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = APP_DIR.parent
THEROCK_BIN = REPO_DIR / "therock_sdk" / "bin"
GFX1031_LIB = REPO_DIR / "_gfx1031_rocm624" / "rocm gfx1031 for rocm 6.2.4" / "library"
SYSTEM_ROCM = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "AMD" / "ROCm"


def _system_roots() -> list[Path]:
    """Folders holding versioned HIP SDKs (…\\ROCm\\6.4): the default install location plus
    wherever HIP_PATH* points (the HIP SDK installer sets them; the user may have picked
    another drive)."""
    roots = [SYSTEM_ROCM]
    for k, v in os.environ.items():
        if k.upper().startswith("HIP_PATH") and v:
            p = Path(v.rstrip("\\/"))
            if re.fullmatch(r"\d+\.\d+", p.name) and p.parent not in roots:
                roots.append(p.parent)
    return roots


def _candidates() -> list[tuple[str, Path]]:
    """(label, bin dir) of usable runtimes: therock first, then system ROCm 6.x newest
    first, then 7.x (ZLUDA v6 is only verified on 6.x)."""
    out: list[tuple[str, Path]] = []
    if (THEROCK_BIN / "rocblas.dll").exists():
        out.append(("therock", THEROCK_BIN))
    vers, seen = [], set()
    for root in _system_roots():
        try:
            dirs = list(root.iterdir()) if root.is_dir() else []
        except OSError:
            dirs = []
        for d in dirs:
            m = re.fullmatch(r"(\d+)\.(\d+)", d.name)
            key = str(d).lower()
            if m and key not in seen and (d / "bin" / "rocblas.dll").exists():
                seen.add(key)
                vers.append(((int(m.group(1)), int(m.group(2))), d / "bin"))
    vers.sort(key=lambda v: (v[0][0] != 6, [-x for x in v[0]]))  # 6.x first, newest first
    out += [(f"{ver[0]}.{ver[1]}", p) for ver, p in vers]
    return out


def parse_hipinfo(txt: str) -> list[dict]:
    """hipInfo.exe output → [{name, arch, mem_gb, integrated}] in HIP device order."""
    devs = []
    for block in re.split(r"^device#\s*\d+\s*$", txt, flags=re.M)[1:]:
        arch = re.search(r"^gcnArchName:\s*(gfx[0-9a-f]+)", block, re.M)
        if not arch:
            continue
        name = re.search(r"^Name:\s*(.+?)\s*$", block, re.M)
        mem = re.search(r"^totalGlobalMem:\s*([\d.]+)\s*([GM])B", block, re.M)
        integ = re.search(r"^isIntegrated:\s*(\d+)", block, re.M)
        devs.append({
            "name": name.group(1) if name else "",
            "arch": arch.group(1),
            "mem_gb": (float(mem.group(1)) / (1 if mem.group(2) == "G" else 1024)) if mem else 0.0,
            "integrated": bool(integ and integ.group(1) != "0"),
        })
    if not devs:   # older hipInfo without "device#" headers
        names = re.findall(r"^Name:\s*(.+?)\s*$", txt, re.M)
        archs = re.findall(r"^gcnArchName:\s*(gfx[0-9a-f]+)", txt, re.M)
        devs = [{"name": names[i] if len(names) == len(archs) else "", "arch": a,
                 "mem_gb": 0.0, "integrated": False} for i, a in enumerate(archs)]
    return devs


def gpu_devices(bin_dirs: list[Path] | None = None) -> list[dict]:
    """The GPUs HIP can use, in HIP device order, via hipInfo.exe ([] without a runtime)."""
    for b in bin_dirs if bin_dirs is not None else [p for _, p in _candidates()]:
        exe = b / "hipInfo.exe"
        if not exe.exists():
            continue
        try:
            env = dict(os.environ, PATH=f"{b};{os.environ.get('PATH', '')}")
            txt = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30,
                                 env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        except Exception:
            continue
        devs = parse_hipinfo(txt or "")
        if devs:
            return devs
    return []


def gpu_archs(bin_dirs: list[Path] | None = None) -> list[tuple[str, str]]:
    """[(name, gfx arch)] of the GPUs HIP can use, in HIP device order."""
    return [(d["name"], d["arch"]) for d in gpu_devices(bin_dirs)]


def best_index(devs: list[dict]) -> int:
    """Discrete before integrated, then the most memory; ties keep HIP order."""
    if not devs:
        return 0
    return max(range(len(devs)), key=lambda i: (not devs[i]["integrated"], devs[i]["mem_gb"], -i))


def _lib_supports(bin_dir: Path, arch: str) -> bool:
    lib = bin_dir / "rocblas" / "library"
    try:
        return lib.is_dir() and any(arch in f.name for f in lib.iterdir())
    except OSError:
        return False


def choose(gpu_index: int | None = None) -> dict:
    """Decide runtime + rocBLAS kernels for HIP device `gpu_index` (None = the best GPU)."""
    cands = _candidates()
    devs = gpu_devices([p for _, p in cands])
    if gpu_index is None:
        idx = best_index(devs)
    else:
        idx = gpu_index if 0 <= gpu_index < len(devs) else 0
    dev = devs[idx] if idx < len(devs) else {"name": "", "arch": ""}
    arch, name = dev["arch"], dev["name"]
    res = {"arch": arch, "gpu": name, "gpu_index": idx, "runtime": "", "rocm_bin": "",
           "hip_path": "", "tensile": "", "note": ""}
    if gpu_index is not None and devs and not 0 <= gpu_index < len(devs):
        res["note"] = f"GPU {gpu_index} doesn't exist (HIP sees {len(devs)}) - using GPU 0"
    for label, b in cands:
        if arch == "gfx1031":
            ok = GFX1031_LIB.is_dir()          # both runtimes work with the gfx1031 kernels
        elif label == "therock":
            ok = bool(arch) and _lib_supports(b, arch)
        else:
            ok = not arch or _lib_supports(b, arch)
        if ok:
            res.update(runtime=label, rocm_bin=str(b),
                       hip_path=str(b.parent) + ("" if label == "therock" else "\\"))
            break
    if not res["runtime"] and cands:
        label, b = cands[0]
        res.update(runtime=label, rocm_bin=str(b), hip_path=str(b.parent),
                   note=f"no runtime has rocBLAS kernels for {arch or 'this GPU'} - GEMMs may fail")
    if arch == "gfx1031" and GFX1031_LIB.is_dir():
        res["tensile"] = str(GFX1031_LIB)
    elif arch == "gfx1031":
        res["note"] = "gfx1031 GPU but the bundled gfx1031 rocBLAS kernels are missing"
    return res


def _bat_value(v) -> str:
    # `set "K=V"` is safe for spaces/&/(), but a % would be expanded when the file is called
    return str(v).replace("%", "%%")


def _bat(res: dict) -> str:
    lines = [f'set "ROCM_ARCH={_bat_value(res["arch"])}"', f'set "ROCM_RUNTIME={_bat_value(res["runtime"])}"',
             f'set "ROCM_GPU_INDEX={res.get("gpu_index", 0)}"']
    if res["rocm_bin"]:
        lines += [f'set "ROCM_BIN={_bat_value(res["rocm_bin"])}"', f'set "HIP_PATH={_bat_value(res["hip_path"])}"']
    if res["tensile"]:
        lines.append(f'set "ROCBLAS_TENSILE_LIBPATH={_bat_value(res["tensile"])}"')
    if res["note"]:
        lines.append(f'set "ROCM_NOTE={_bat_value(res["note"])}"')
    return "\n".join(lines)


def _env_index() -> int | None:
    v = str(os.environ.get("IMAGEGEN_GPU", "")).strip()
    try:
        return int(v) if v else None
    except ValueError:
        return None


if __name__ == "__main__":
    r = choose(_env_index())
    print(_bat(r) if "--bat" in sys.argv else json.dumps(r, indent=2))
