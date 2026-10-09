"""
backend/hardware_detector.py
Enumerate every compute device available on this machine and rank them
for Stable Diffusion workloads.

Supported hardware families:
  AMD dGPU   — RX 6800M, RX 6800 XT, RX 9070 XT (RDNA 2/3/4)
  AMD iGPU   — Vega (Zen 3), 610M RDNA2 (Zen 5), 780M RDNA3 (Zen 4)
  XDNA NPU   — Ryzen AI (8700G/Strix), for attention/VAE offload via Ryzen AI SDK
  DirectML   — All AMD/Intel/NVIDIA via onnxruntime-directml
  CPU        — Always available fallback

The module never crashes — every detection step is wrapped in a try/except
so missing drivers or packages just produce "unavailable" entries.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUInfo:
    index:       int
    name:        str
    vram_mb:     int            # 0 = unknown / shared RAM
    backend:     str            # "cuda_zluda" | "directml" | "cpu"
    hip_index:   int = -1       # HIP_VISIBLE_DEVICES index (for ZLUDA)
    is_igpu:     bool = False
    is_dgpu:     bool = False
    rdna_gen:    int = 0        # 1/2/3/4 or 0=unknown
    score:       int = 0        # Higher = more preferred for SD
    notes:       str = ""


@dataclass
class NPUInfo:
    available:   bool = False
    name:        str  = ""
    backend:     str  = ""      # "vitisai" | "vitisai_sdk" | "ryzenai" | "pending" | ""
    notes:       str  = ""


@dataclass
class HardwareProfile:
    gpus:        list[GPUInfo] = field(default_factory=list)
    npu:         NPUInfo       = field(default_factory=NPUInfo)
    cpu_name:    str           = ""
    ram_gb:      int           = 0
    best_gpu:    Optional[GPUInfo] = None  # Top-ranked GPU for SD
    summary:     str           = ""


# ── RDNA generation heuristic ──────────────────────────────────────────────────
_RDNA_PATTERNS: list[tuple[re.Pattern, int, bool]] = [
    # (pattern, rdna_gen, is_igpu)
    (re.compile(r"RX\s*(6[0-9]{3}|6[89][0-9]{2})", re.I), 2, False),   # RX 6xxx dGPU RDNA2
    (re.compile(r"RX\s*7[0-9]{3}", re.I),                  3, False),   # RX 7xxx dGPU RDNA3
    (re.compile(r"RX\s*9[0-9]{3}", re.I),                  4, False),   # RX 9xxx dGPU RDNA4
    (re.compile(r"610M",           re.I),                   2, True),    # 610M iGPU RDNA2
    (re.compile(r"680M|660M",      re.I),                   2, True),    # Zen3+ iGPU RDNA2
    (re.compile(r"780M|760M|740M", re.I),                   3, True),    # Zen4 iGPU RDNA3
    (re.compile(r"890M|880M|870M|860M|850M|840M|830M|Strix Halo|Hawk Point", re.I), 3, True),  # Zen5 iGPU RDNA3.5
    (re.compile(r"Vega",           re.I),                   1, True),    # Vega iGPU (Zen3 and earlier)
]

def _classify_gpu(name: str) -> tuple[int, bool]:
    """Returns (rdna_gen, is_igpu). 0 = unknown."""
    for pat, gen, igpu in _RDNA_PATTERNS:
        if pat.search(name):
            return gen, igpu
    # Generic AMD iGPU markers (use regex to handle trademark symbols)
    for pattern in (r"radeon\s*\S*\s*graphics", r"integrated", r"internal"):
        if re.search(pattern, name, re.I):
            return 0, True
    return 0, False


def _score_gpu(g: GPUInfo) -> int:
    """Higher score = better for SD. Used to sort and pick best_gpu."""
    s = 0
    # VRAM weight (most important)
    s += min(g.vram_mb // 1024, 24) * 100      # up to 2400 pts
    # dGPU > iGPU
    if g.is_dgpu:   s += 500
    if g.is_igpu:   s -= 100
    # RDNA generation bonus
    s += g.rdna_gen * 50
    return s


# ── Registry-based VRAM query (accurate for >4 GB AMD/NVIDIA GPUs) ─────────────
def _winreg_vram() -> dict[str, int]:
    """
    Returns {DriverDesc: vram_bytes} from the Windows GPU display driver registry.
    Uses HardwareInformation.qwMemorySize (64-bit), which is accurate for GPUs
    with >4 GB VRAM unlike WMIC's 32-bit AdapterRAM field.
    """
    try:
        import winreg
        results: dict[str, int] = {}
        key_path = (
            r"SYSTEM\CurrentControlSet\Control\Class"
            r"\{4d36e968-e325-11ce-bfc1-08002be10318}"
        )
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as base:
            i = 0
            while True:
                try:
                    sub_name = winreg.EnumKey(base, i)
                    i += 1
                    try:
                        with winreg.OpenKey(base, sub_name) as sub:
                            try:
                                mem_data, _ = winreg.QueryValueEx(
                                    sub, "HardwareInformation.qwMemorySize"
                                )
                                # REG_QWORD (type 11) → Python int; REG_BINARY → bytes
                                if isinstance(mem_data, int):
                                    vram = mem_data
                                elif isinstance(mem_data, bytes) and len(mem_data) >= 8:
                                    vram = int.from_bytes(mem_data[:8], "little")
                                else:
                                    continue
                                desc, _ = winreg.QueryValueEx(sub, "DriverDesc")
                                if vram > 0:
                                    results[desc] = vram
                            except OSError:
                                pass
                    except OSError:
                        pass
                except OSError:
                    break
        return results
    except Exception:
        return {}


# ── WMIC-based GPU enumeration (Windows, no extra packages needed) ─────────────
def _wmic_gpus() -> list[tuple[str, int]]:
    """
    Returns list of (gpu_name, vram_bytes) from WMIC, with VRAM corrected via
    the registry when WMIC's 32-bit AdapterRAM field underreports (e.g. AMD GPUs
    with >4 GB VRAM show up as 3 GB in WMIC but 12 GB in the registry).
    """
    reg_vram = _winreg_vram()  # {name: bytes}, 64-bit accurate

    # Generic words that appear in nearly every GPU name — exclude from matching
    _SKIP = {"amd", "nvidia", "intel", "radeon", "geforce", "graphics",
             "display", "adapter", "integrated"}

    def _best_vram(name: str, wmic_bytes: int) -> int:
        """Return registry VRAM if it's meaningfully larger than WMIC's value."""
        name_lower = name.lower()
        for reg_name, reg_bytes in reg_vram.items():
            # Only use model-specific tokens (e.g. "6800M", "3080") for matching
            specific = {w.lower() for w in reg_name.split()
                        if len(w) > 3 and w.lower() not in _SKIP}
            if specific and any(w in name_lower for w in specific):
                if reg_bytes > wmic_bytes * 1.5:  # registry reports ≥50% more
                    return reg_bytes
        return wmic_bytes

    # Try wmic.exe first (Windows 10 / early Win11), then fall back to
    # Get-CimInstance (wmic.exe was removed in Windows 11 24H2+).
    def _parse_wmic(text: str) -> list[tuple[str, int]]:
        results = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[1].isdigit():
                vram = int(parts[1])
                name = parts[2]
                if name and name.lower() not in ("name", ""):
                    vram = _best_vram(name, vram)
                    results.append((name, vram))
        return results

    try:
        out = subprocess.check_output(
            ["wmic", "path", "win32_VideoController",
             "get", "Name,AdapterRAM", "/format:csv"],
            timeout=10, stderr=subprocess.DEVNULL, text=True,
        )
        results = _parse_wmic(out)
        if results:
            return results
    except Exception:
        pass

    # PowerShell fallback (Windows 11 24H2+ where wmic.exe is absent)
    try:
        import json
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController "
             "| Select-Object Name,AdapterRAM | ConvertTo-Json"],
            timeout=10, stderr=subprocess.DEVNULL, text=True,
        )
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        results = []
        for entry in data:
            name = str(entry.get("Name") or "").strip()
            vram = int(entry.get("AdapterRAM") or 0)
            if name and name.lower() not in ("name", ""):
                vram = _best_vram(name, vram)
                results.append((name, vram))
        return results
    except Exception:
        return []


# ── CUDA / ZLUDA device enumeration ───────────────────────────────────────────
def _cuda_gpus() -> list[tuple[str, int]]:
    """Returns list of (device_name, vram_bytes) via torch.cuda."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return [
            (
                torch.cuda.get_device_name(i),
                torch.cuda.get_device_properties(i).total_memory,
            )
            for i in range(torch.cuda.device_count())
        ]
    except Exception:
        return []


# ── DirectML enumeration ───────────────────────────────────────────────────────
def _directml_available() -> bool:
    try:
        import onnxruntime as ort
        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


# ── XDNA / Ryzen AI NPU detection ─────────────────────────────────────────────
def _detect_npu() -> NPUInfo:
    """
    Detect AMD XDNA NPU (Ryzen AI 8040/8700G/Strix/Hawk Point etc.).
    Checks for VitisAI EP in onnxruntime, then Ryzen AI package, then
    falls back to looking for the NPU in WMIC device list.
    """
    # 1. VitisAI Execution Provider in onnxruntime
    try:
        import onnxruntime as ort
        if "VitisAIExecutionProvider" in ort.get_available_providers():
            return NPUInfo(
                available=True,
                name="AMD XDNA NPU (VitisAI EP)",
                backend="vitisai",
                notes="Use for text encoder / VAE decode acceleration",
            )
    except Exception:
        pass

    # 2. AMD Ryzen AI package
    try:
        import ryzenai  # type: ignore  # noqa
        return NPUInfo(
            available=True,
            name="AMD XDNA NPU (ryzenai SDK)",
            backend="ryzenai",
            notes="Ryzen AI SDK detected",
        )
    except ImportError:
        pass

    # 2.5 Ryzen AI SDK installation folder (covers SDK installed for Python 3.12+
    #     where the wheel can't be imported from this Python 3.10 process)
    try:
        ryzenai_path = os.environ.get("RYZEN_AI_INSTALLATION_PATH", "")
        if not ryzenai_path:
            for ver in ("1.7.0", "1.6.1", "1.6.0", "1.5.0"):
                p = rf"C:\Program Files\RyzenAI\{ver}"
                if os.path.isdir(p):
                    ryzenai_path = p
                    break
        if ryzenai_path:
            vitisai_dll = os.path.join(ryzenai_path, "deployment",
                                       "onnxruntime_providers_vitisai.dll")
            if os.path.isfile(vitisai_dll):
                ver_tag = os.path.basename(ryzenai_path)
                return NPUInfo(
                    available=True,
                    name="AMD XDNA NPU (Ryzen AI SDK)",
                    backend="vitisai_sdk",
                    notes=(
                        f"Ryzen AI SDK {ver_tag} installed. "
                        "NPU ready — inference runs via the ryzen-ai conda env "
                        f"({ryzenai_path})."
                    ),
                )
    except Exception:
        pass

    # 3. PnP device list— look for AMD NPU / IPU / Compute Accelerator
    # Try Get-PnpDevice first (available on Win11 where wmic.exe is gone),
    # then fall back to legacy wmic win32_PnPEntity.
    def _parse_npu_name(lines: list[str]) -> str | None:
        for line in lines:
            line = line.strip()
            if line and "name" not in line.lower():
                return line
        return None

    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-PnpDevice -Class ComputeAccelerator -ErrorAction SilentlyContinue "
             "| Where-Object { $_.FriendlyName -match 'NPU|IPU|AI|Compute' } "
             "| Select-Object -ExpandProperty FriendlyName"],
            timeout=10, stderr=subprocess.DEVNULL, text=True,
        )
        npu_name = _parse_npu_name(out.splitlines())
        if npu_name:
            return NPUInfo(
                available=True,
                name=npu_name,
                backend="pending",
                notes=(
                    "NPU hardware found but Ryzen AI SDK not installed. "
                    "Install AMD Ryzen AI Software: "
                    "https://ryzenai.docs.amd.com/en/latest/inst.html"
                ),
            )
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["wmic", "path", "win32_PnPEntity",
             "where", "Name like '%AMD%NPU%' or Name like '%AMD%IPU%'",
             "get", "Name"],
            timeout=10, stderr=subprocess.DEVNULL, text=True,
        )
        npu_name = _parse_npu_name(out.splitlines())
        if npu_name:
            return NPUInfo(
                available=True,
                name=npu_name,
                backend="pending",
                notes=(
                    "NPU hardware found but driver/SDK not fully installed. "
                    "Install AMD Ryzen AI Software: "
                    "https://ryzenai.docs.amd.com/en/latest/inst.html"
                ),
            )
    except Exception:
        pass

    # 4. Check CPU name — if it's a known NPU-bearing chip, suggest it
    try:
        cpu = _cpu_name()
        npu_cpus = re.compile(
            r"(8[0-9]{3}[GHU]|9[0-9]{3}H|Strix|Hawk Point|Phoenix)", re.I
        )
        if npu_cpus.search(cpu):
            return NPUInfo(
                available=False,
                name="AMD XDNA NPU (driver/SDK not detected)",
                backend="",
                notes=(
                    f"Your CPU ({cpu}) has an XDNA NPU but the Ryzen AI driver was not found.\n"
                    "Install AMD Ryzen AI Software to enable NPU inference:\n"
                    "https://ryzenai.docs.amd.com/en/latest/inst.html"
                ),
            )
    except Exception:
        pass

    return NPUInfo(available=False, name="Not detected", backend="")


# ── CPU / RAM info ────────────────────────────────────────────────────────────
def _cpu_name() -> str:
    # platform.processor() returns a generic "AMD64 Family X Model Y" string on
    # Windows 11 — use the branded name (e.g. "AMD Ryzen 7 8700G"). The registry holds the same
    # string Win32_Processor reports and answers in under a millisecond; a PowerShell start-up
    # costs ~1.5 s and this runs three times per app start (measured, round 13).
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if name:
            return name
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"],
            timeout=8, stderr=subprocess.DEVNULL, text=True,
        ).strip().splitlines()
        if out:
            return out[0].strip()
    except Exception:
        pass
    try:
        import platform
        return platform.processor() or ""
    except Exception:
        return ""


def _ram_gb() -> int:
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys // (1024 ** 3)
    except Exception:
        return 0


# ── Main detection entry point ────────────────────────────────────────────────
def detect_hardware() -> HardwareProfile:
    """
    Build a complete HardwareProfile for the current machine.
    Safe to call multiple times (cached after first call via module-level singleton).
    """
    profile = HardwareProfile()
    profile.cpu_name = _cpu_name()
    profile.ram_gb   = _ram_gb()
    profile.npu      = _detect_npu()

    # ── Step 1: Try CUDA/ZLUDA first (best performance path) ──────────────────
    cuda_devs = _cuda_gpus()
    for i, (name, vram) in enumerate(cuda_devs):
        rdna_gen, is_igpu = _classify_gpu(name)
        g = GPUInfo(
            index=i, name=name, vram_mb=vram // (1024 * 1024),
            backend="cuda_zluda", hip_index=i,
            is_igpu=is_igpu, is_dgpu=not is_igpu,
            rdna_gen=rdna_gen,
        )
        g.score = _score_gpu(g)
        profile.gpus.append(g)

    # ── Step 2: Augment with WMIC data (fills in iGPUs CUDA may not expose) ───
    wmic_devs = _wmic_gpus()
    # ZLUDA reports "AMD Radeon RX 6800M [ZLUDA]"; Windows says "AMD Radeon RX 6800M"
    _norm = lambda n: n.lower().replace("[zluda]", "").strip()
    seen_names = {_norm(g.name) for g in profile.gpus}
    dml_avail  = _directml_available()

    for name, vram in wmic_devs:
        if any(k in name.lower() for k in ("amd", "radeon", "vega", "intel", "nvidia")):
            if _norm(name) not in seen_names:
                rdna_gen, is_igpu = _classify_gpu(name)
                g = GPUInfo(
                    index=len(profile.gpus),
                    name=name,
                    vram_mb=vram // (1024 * 1024),
                    backend="directml" if dml_avail else "cpu",
                    is_igpu=is_igpu, is_dgpu=not is_igpu,
                    rdna_gen=rdna_gen,
                )
                g.score = _score_gpu(g)  # will be recalculated after iGPU VRAM correction
                profile.gpus.append(g)
                seen_names.add(_norm(name))

    # ── Step 3: Annotate iGPU VRAM ────────────────────────────────────────────
    # iGPUs use shared system RAM, not dedicated VRAM.  The Windows registry
    # (and WMIC) often reports the BIOS-allocated shared memory size (e.g.
    # 16 GB on a 128 GB system) as "dedicated VRAM", which is misleading for
    # ML workload planning.  Cap iGPU VRAM to a realistic ML-usable estimate.
    for g in profile.gpus:
        if g.is_igpu:
            estimated = min(16384, (profile.ram_gb * 1024) // 4)
            if g.vram_mb <= 512 or g.vram_mb >= profile.ram_gb * 1024 // 8:
                # Registry value is missing, suspiciously large, or matches
                # shared RAM — replace with conservative estimate
                g.vram_mb = estimated
            g.notes += f"Shared RAM iGPU (~{estimated // 1024} GB usable for ML). "

        # Add RDNA gen notes
        if g.rdna_gen == 4:
            g.notes += "RDNA 4 — ZLUDA support is experimental; use DirectML if issues arise. "
        elif g.rdna_gen == 1:
            g.notes += "Vega iGPU — limited compute; CPU fallback may be faster for large models. "

    # ── Step 4: Recalculate scores after VRAM correction, sort, pick best ────
    for g in profile.gpus:
        g.score = _score_gpu(g)
        if g.backend != "cuda_zluda":
            g.score -= 50  # slight penalty vs CUDA path
    profile.gpus.sort(key=lambda g: g.score, reverse=True)
    profile.best_gpu = profile.gpus[0] if profile.gpus else None

    # ── Step 5: Build human-readable summary ──────────────────────────────────
    lines = [f"CPU   : {profile.cpu_name}", f"RAM   : {profile.ram_gb} GB"]
    for g in profile.gpus:
        tag = "★" if g == profile.best_gpu else " "
        vram_str = f"{round(g.vram_mb / 1024)} GB" if g.vram_mb >= 1024 else f"{g.vram_mb} MB"
        if g.is_igpu:
            vram_str += " shared"
        igpu_tag = "[iGPU]" if g.is_igpu else "[dGPU]"
        lines.append(
            f"{tag} GPU {g.index}: {g.name} {igpu_tag} "
            f"{vram_str} VRAM · RDNA{g.rdna_gen or '?'} · {g.backend}"
        )
    if profile.npu.available:
        lines.append(f"NPU   : {profile.npu.name} ✅")
    else:
        lines.append(f"NPU   : {profile.npu.name}")
    profile.summary = "\n".join(lines)

    return profile


# Module-level singleton — detected once on import
_PROFILE: HardwareProfile | None = None


def get_profile() -> HardwareProfile:
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = detect_hardware()
    return _PROFILE
