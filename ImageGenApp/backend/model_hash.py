"""
model_hash.py — SHA-256 of model files, computed once in the background and cached.

Used for the A1111-compatible "Model hash" in saved images (AutoV2 = first 10 hex digits of
the file's SHA-256 — the same hash Civitai shows and matches uploads by) and to find a model
again after it was renamed. Hashing a 7 GB SDXL file takes ~10–20 s, so it never blocks a
generation: hash_later() queues it, cached() answers instantly (None until it's done).
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from config import SETTINGS_DIR

_CACHE_FILE = SETTINGS_DIR / "_hash_cache.json"
_lock = threading.Lock()
_cache: dict[str, dict] | None = None
_queue: list[str] = []
_worker: threading.Thread | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            if not isinstance(_cache, dict):
                _cache = {}
        except Exception:
            _cache = {}
    return _cache


def _sig(p: Path) -> tuple[int, float]:
    st = p.stat()
    return st.st_size, round(st.st_mtime, 3)


def cached(path: str | None) -> str | None:
    """Full SHA-256 (hex) if already known for this exact file version, else None."""
    if not path:
        return None
    p = Path(str(path))
    try:
        size, mtime = _sig(p)
    except OSError:
        return None
    with _lock:
        e = _load().get(str(p.resolve()).lower())
    if e and e.get("size") == size and e.get("mtime") == mtime:
        return e.get("sha256")
    return None


def autov2(path: str | None) -> str | None:
    h = cached(path)
    return h[:10] if h else None


def compute(path: str) -> str | None:
    p = Path(str(path))
    try:
        size, mtime = _sig(p)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
    except OSError:
        return None
    digest = h.hexdigest()
    with _lock:
        c = _load()
        c[str(p.resolve()).lower()] = {"sha256": digest, "size": size, "mtime": mtime, "name": p.name}
        try:
            _CACHE_FILE.write_text(json.dumps(c, indent=1), encoding="utf-8")
        except OSError:
            pass
    return digest


def _run():
    global _worker
    while True:
        with _lock:
            if not _queue:
                _worker = None
                return
            path = _queue.pop(0)
        if cached(path) is None:
            compute(path)


def hash_later(*paths: str | None) -> None:
    """Queue files for background hashing (local files only; repeats are ignored)."""
    global _worker
    with _lock:
        for p in paths:
            if p and p != "none" and Path(str(p)).is_file() and str(p) not in _queue:
                _queue.append(str(p))
        if _queue and _worker is None:
            _worker = threading.Thread(target=_run, name="model-hash", daemon=True)
            _worker.start()


def find_by_hash(prefix: str, candidates: list[str]) -> str | None:
    """A local file (from candidates) whose cached SHA-256 starts with prefix."""
    prefix = (prefix or "").lower()
    if len(prefix) < 8:
        return None
    for c in candidates:
        h = cached(c)
        if h and h.startswith(prefix):
            return c
    return None
