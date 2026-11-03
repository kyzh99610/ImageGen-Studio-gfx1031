"""
dxgi.py - the PC's graphics adapters in DXGI order (IDXGIFactory1::EnumAdapters1): the order in which DirectML and ONNX Runtime's
DmlExecutionProvider number them (`device_id`). torch-directml gave these names before; native ROCm's environment has no torch-directml.
Standard library only (ctypes); Windows only, [] elsewhere.
"""
from __future__ import annotations

import ctypes
import os


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort), ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint), ("SubSysId", ctypes.c_uint),
                ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t), ("LuidLow", ctypes.c_ulong), ("LuidHigh", ctypes.c_long), ("Flags", ctypes.c_uint)]


_IID_FACTORY1 = _GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
_SOFTWARE = 2                    # DXGI_ADAPTER_FLAG_SOFTWARE: the Microsoft Basic Render Driver


def _method(obj, index, *argtypes):
    """A COM method by its vtable slot (IUnknown 0-2, IDXGIObject 3-6, then the interface's own)."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[index])


def adapters() -> list[tuple[str, bool]]:
    """[(name, is_software)] in DXGI order; [] when DXGI can't be asked."""
    if os.name != "nt":
        return []
    out = []
    try:
        factory = ctypes.c_void_p()
        if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(_IID_FACTORY1), ctypes.byref(factory)) != 0:
            return []
        try:
            for i in range(16):
                adapter = ctypes.c_void_p()
                if _method(factory, 12, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(factory, i, ctypes.byref(adapter)) != 0:
                    break                                         # DXGI_ERROR_NOT_FOUND: no more adapters
                desc = _AdapterDesc1()
                _method(adapter, 10, ctypes.POINTER(_AdapterDesc1))(adapter, ctypes.byref(desc))
                out.append((desc.Description, bool(desc.Flags & _SOFTWARE)))
                _method(adapter, 2)(adapter)                      # Release
        finally:
            _method(factory, 2)(factory)
    except Exception:
        return []
    return out


def dml_index(igpu: bool) -> int | None:
    """DirectML device_id of the first hardware adapter that is (igpu=True) or is not (False) an integrated GPU; None when there is none."""
    from backend.hardware_detector import _classify_gpu
    for i, (name, software) in enumerate(adapters()):
        if not software and _classify_gpu(name)[1] == igpu:
            return i
    return None
