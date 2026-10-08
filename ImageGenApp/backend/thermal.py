"""
thermal.py — read the Windows thermal zone and pause GPU jobs while the laptop is too hot.

Why: the RX 6800M laptop switched itself off five times in four days under minutes of full GPU load (EventLog 6008, no
bluescreen); its thermal zone (\\_TZ.tz01) pinned at 95.9–96.9 °C for 3–4 min before each cut and never reads higher.
Cool mode (sampling.COOL) lowers the load; this module adds a hard brake between pictures / passes: when the zone is at or
above `limit` °C the job waits (Stop still works) until it is at or below `limit - 8`. Windows only (PDH counter
"\\Thermal Zone Information(*)\\Temperature", Kelvin); elsewhere `read_temp()` returns None and nothing waits.
"""
from __future__ import annotations

import time

GUARD = {"limit": 0.0}           # 0 = off; set from Settings (saved as "thermal_limit" in settings/_prefs.json)
_RESUME_GAP = 8.0
_MAX_WAIT = 900.0                # never wait longer than 15 min for one cool-down


def set_limit(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = 0.0
    if f != f or f < 60:
        f = 0.0                                  # off (values below 60 °C would never let a laptop run)
    GUARD["limit"] = min(99.0, f)
    return GUARD["limit"]


def apply_saved(prefs: dict, sampling, environ=None) -> bool:
    """Start-up: the saved cool factor and pause limit (Settings). Not under the test suite (IMAGEGEN_TESTS=1): the suite must not depend on a user's
    saved choices — a pause limit makes every fake generate in it read the live sensor and wait while the laptop is hot. Returns whether they were applied."""
    import os
    if (os.environ if environ is None else environ).get("IMAGEGEN_TESTS") == "1":
        return False
    sampling.set_cool((prefs or {}).get("cool", 0.0))
    set_limit((prefs or {}).get("thermal_limit", 0.0))
    return True


def read_temp() -> float | None:
    """Hottest Windows thermal zone in °C, or None when it can't be read."""
    try:
        import ctypes
        from ctypes import wintypes
        pdh = ctypes.WinDLL("pdh")
    except Exception:
        return None

    class _FMT(ctypes.Structure):
        _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]

    class _ITEM(ctypes.Structure):
        _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _FMT)]

    q, c = wintypes.HANDLE(), wintypes.HANDLE()
    try:
        if pdh.PdhOpenQueryW(None, None, ctypes.byref(q)) != 0:
            return None
        if pdh.PdhAddEnglishCounterW(q, "\\Thermal Zone Information(*)\\Temperature", None, ctypes.byref(c)) != 0:
            return None
        if pdh.PdhCollectQueryData(q) != 0:
            return None
        size, count = wintypes.DWORD(0), wintypes.DWORD(0)
        PDH_FMT_DOUBLE = 0x00000200
        pdh.PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count), None)
        if not size.value:
            return None
        buf = (ctypes.c_byte * size.value)()
        if pdh.PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count), buf) != 0:
            return None
        items = ctypes.cast(buf, ctypes.POINTER(_ITEM))
        vals = [items[i].FmtValue.doubleValue for i in range(count.value) if items[i].FmtValue.CStatus in (0, 1)]
        vals = [v - 273.15 for v in vals if 200 < v < 400]          # the counter is in Kelvin
        return max(vals) if vals else None
    except Exception:
        return None
    finally:
        try:
            if q:
                pdh.PdhCloseQuery(q)
        except Exception:
            pass


# Inside a sampling pass only an emergency brake: act at limit + 6, resume at the limit. Round 6 measured the in-pass check
# with the between-pieces rule (act at the limit, resume at limit - 8): every hires pass turned stop-go (the zone rises ~1 °C/s
# at 31 % duty, so it hit 88 within ~40 s and then waited 1-3 min for 80), 8.7 instead of 3.9 min per Full-body picture, and
# the peaks stayed at 95.9 — what keeps the laptop safe is the smooth per-step pause plus a cool start of every piece.
IN_PASS_MARGIN = 6.0
_PLATEAU = 20.0         # between pieces: no new low for this long below the limit = the laptop's idle level, go on


def wait_cool(should_stop=None, say=None, read=read_temp, sleep=time.sleep, clock=time.monotonic,
              in_pass: bool = False) -> float:
    """When the guard is on and the laptop is at or above the limit, wait until it is `limit - 8` or cooler (or Stop, or
    15 min). in_pass: the emergency brake used between sampling steps — act at limit + 6, resume at the limit.
    `say(text)` reports progress. Returns the seconds waited (0 when nothing was needed)."""
    limit = GUARD["limit"]
    if not limit:
        return 0.0
    trigger, resume = (limit + IN_PASS_MARGIN, limit) if in_pass else (limit, limit - _RESUME_GAP)
    t = read()
    if t is None or t < trigger:
        return 0.0
    t0 = clock()
    best, best_at = t, t0
    while t is not None and t > resume and clock() - t0 < _MAX_WAIT:
        if should_stop is not None and should_stop():
            break
        # Between pieces: also go on once the zone is under the limit and has stopped falling (no new low by ≥ 1 °C in
        # 20 s) — the laptop has reached its idle level. Round 8: with an idle floor of 84–88 °C the wait for 80 took 35 %
        # of a hand re-draw call (25 waits of 3–44 s); the brake inside a pass keeps its own rule.
        if t < best - 1.0:
            best, best_at = t, clock()
        elif not in_pass and t < limit and clock() - best_at >= _PLATEAU:
            break
        if say is not None:
            say(f"🌡 Cooling down: {t:.0f} °C — waiting for {resume:.0f} °C before the next step")
        sleep(2.0)
        t = read()
    return clock() - t0
