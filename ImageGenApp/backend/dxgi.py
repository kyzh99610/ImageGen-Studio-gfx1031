"""
dxgi.py - the GPUs in DXGI order, which is how ONNX Runtime's DmlExecutionProvider numbers them (`device_id` = EnumAdapters1 index).
torch-directml has an order of its own: never hand its numbers to ONNX Runtime (see dml_index). Without a device_id ONNX Runtime
takes the high-performance GPU, so a SmartSplit "iGPU" session on the laptop ran on the RX 6800M.
Stdlib only (ctypes): works without torch-directml.
"""
from __future__ import annotations

import ctypes
import sys

_DXGI_ADAPTER_FLAG_SOFTWARE = 2


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]


class _AdapterDesc1(ctypes.Structure):          # DXGI_ADAPTER_DESC1
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("LuidLow", ctypes.c_uint32), ("LuidHigh", ctypes.c_int32), ("Flags", ctypes.c_uint)]


# IID_IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
_IID_FACTORY1 = _GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))


def _method(obj: int, index: int, restype, *argtypes):
    """Method `index` of a COM object's vtable (IUnknown 0-2, IDXGIObject 3-6, IDXGIFactory 7-11 / IDXGIAdapter 7-9, then the *1 methods)."""
    vtbl = ctypes.cast(ctypes.c_void_p(obj), ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])


def adapters() -> list[dict]:
    """[{index, name, vendor, vram_mb, software}] in DXGI order ([] off Windows or when DXGI can't be reached)."""
    if sys.platform != "win32":
        return []
    try:
        create = ctypes.WinDLL("dxgi").CreateDXGIFactory1
    except (OSError, AttributeError):
        return []
    create.restype = ctypes.c_long
    create.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    factory = ctypes.c_void_p()
    if create(ctypes.byref(_IID_FACTORY1), ctypes.byref(factory)) != 0 or not factory.value:
        return []
    out = []
    try:
        enum_adapters1 = _method(factory.value, 12, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        for i in range(64):
            adapter = ctypes.c_void_p()
            if enum_adapters1(factory.value, i, ctypes.byref(adapter)) != 0 or not adapter.value:
                break           # DXGI_ERROR_NOT_FOUND: no more adapters
            try:
                desc = _AdapterDesc1()
                if _method(adapter.value, 10, ctypes.c_long, ctypes.POINTER(_AdapterDesc1))(adapter.value, ctypes.byref(desc)) == 0:
                    out.append({"index": i, "name": desc.Description, "vendor": desc.VendorId, "device": desc.DeviceId,
                                "vram_mb": desc.DedicatedVideoMemory // 2**20,
                                "luid": f"luid_0x{desc.LuidHigh & 0xFFFFFFFF:08X}_0x{desc.LuidLow:08X}",     # as the GPU performance counters name it
                                "software": bool(desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE)})
            finally:
                _method(adapter.value, 2, ctypes.c_ulong)(adapter.value)        # Release
    finally:
        _method(factory.value, 2, ctypes.c_ulong)(factory.value)
    return out


# GCN / Vega integrated GPUs by DXGI DeviceId (AMD): Raven 15DD, Picasso / Dali 15D8, Renoir 1636, Lucienne 164C, Cezanne 1638, Barcelo 15E7,
# Carrizo / Bristol Ridge 9874, Stoney 98E4. The RX 6800M laptop's (Cezanne; Windows names it only "AMD Radeon(TM) Graphics", so the name
# can't tell) ran SmartSplit's SD 1.5 VAE decode in 8 s, the RX 6800M's 0.5 s, and then hung: DXGI 887A0006 and a driver reset of the GPU
# that drives the laptop's screen (LiveKernelEvent 141), 2026-11-03. Nothing is offloaded to one.
GCN_IGPU_IDS = frozenset({0x15DD, 0x15D8, 0x1636, 0x164C, 0x1638, 0x15E7, 0x9874, 0x98E4})


def offload_igpu() -> tuple[int | None, str]:
    """(DirectML device_id of the iGPU that SmartSplit / accelerated text encoding may use, "") or (None, why not)."""
    from backend.hardware_detector import _classify_gpu
    for a in adapters():
        if a["software"] or not _classify_gpu(a["name"])[1]:
            continue
        if a.get("vendor") == 0x1002 and a.get("device") in GCN_IGPU_IDS:
            return None, (f"the iGPU ({a['name']}) is a Vega-class GPU: an SD 1.5 VAE decode took it 8 s against the dGPU's 0.5 s "
                          "and then hung it (a driver reset of the GPU that drives the screen)")
        return a["index"], ""
    return None, "no iGPU among the DirectML adapters"


def dml_index(igpu: bool) -> int | None:
    """ONNX Runtime's DmlExecutionProvider `device_id` of the integrated (igpu=True) or the discrete GPU, by name
    (hardware_detector._classify_gpu); None when there is none. That id is the DXGI number, NOT torch-directml's: measured on the
    laptop (2026-11-03, the panel on the iGPU) DXGI and device_id 0 = the Vega iGPU, 1 = the RX 6800M, while torch-directml calls the
    RX 6800M 0 — the app used torch-directml's numbers, so its "dGPU" upscaler session ran on the iGPU and its "iGPU" text encoders on
    the dGPU. With no device_id ONNX Runtime takes the RX 6800M. The order can follow the display setup, so ask each time."""
    from backend.hardware_detector import _classify_gpu
    for a in adapters():
        if not a["software"] and _classify_gpu(a["name"])[1] == igpu:
            return a["index"]
    return None
