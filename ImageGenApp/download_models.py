"""
download_models.py — Pre-download popular SD 1.5 models into models/
Run via:  download_models.bat   (or python download_models.py)

Models downloaded:
  checkpoints/
    dreamshaper_8.safetensors                  ~2.0 GB  (versatile, artistic + anime)
    Realistic_Vision_V6.0_NV_B1_fp16.safetensors ~2.0 GB  (photorealistic)
    AbsoluteRealityV1.6525_pruned.safetensors  ~2.0 GB  (hyper-realistic)
  vae/
    vae-ft-mse-840000.safetensors              ~335 MB  (SD 1.5 standard VAE)
"""

from __future__ import annotations
import sys, os, time
from pathlib import Path

# ── Resolve paths ──────────────────────────────────────────────────────────────
APP_DIR   = Path(__file__).parent.resolve()
CKPT_DIR  = APP_DIR / "models" / "checkpoints"
VAE_DIR   = APP_DIR / "models" / "vae"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
VAE_DIR.mkdir(parents=True, exist_ok=True)

# ── Model manifest ─────────────────────────────────────────────────────────────
MODELS = [
    {
        "name":  "DreamShaper 8",
        "url":   "https://huggingface.co/Yntec/Dreamshaper8/resolve/main/dreamshaper_8.safetensors",
        "dest":  CKPT_DIR / "dreamshaper_8.safetensors",
        "size":  "~2.0 GB",
    },
    {
        "name":  "Realistic Vision V6.0 NV (fp16)",
        "url":   "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_fp16.safetensors",
        "dest":  CKPT_DIR / "Realistic_Vision_V6.0_NV_B1_fp16.safetensors",
        "size":  "~2.0 GB",
    },
    {
        "name":  "AbsoluteReality V1.6525 (pruned)",
        "url":   "https://huggingface.co/Lykon/AbsoluteReality/resolve/main/AbsoluteRealityV1.6525_pruned.safetensors",
        "dest":  CKPT_DIR / "AbsoluteRealityV1.6525_pruned.safetensors",
        "size":  "~2.0 GB",
    },
    {
        "name":  "SD 1.5 VAE (ft-mse)",
        "url":   "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
        "dest":  VAE_DIR / "vae-ft-mse-840000.safetensors",
        "size":  "~335 MB",
    },
]


def _bar(done: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return f"{done / 1_048_576:.1f} MB"
    pct = done / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:5.1f}%  {done/1_048_576:.1f}/{total/1_048_576:.1f} MB"


def download(url: str, dest: Path, name: str) -> bool:
    import requests

    # Resume support: check existing size
    existing = dest.stat().st_size if dest.exists() else 0
    headers  = {"Range": f"bytes={existing}-"} if existing else {}

    try:
        r = requests.get(url, headers=headers, stream=True, timeout=30,
                         allow_redirects=True)
    except Exception as e:
        print(f"  ERROR connecting: {e}")
        return False

    if r.status_code == 416:
        # Already fully downloaded (server says range not satisfiable)
        print(f"  ✅ Already complete ({existing/1_048_576:.1f} MB)")
        return True
    if r.status_code not in (200, 206):
        print(f"  ERROR: HTTP {r.status_code}")
        return False

    total = int(r.headers.get("Content-Length", 0))
    if r.status_code == 206:
        # Resuming — total is remaining bytes
        total += existing
    mode  = "ab" if existing and r.status_code == 206 else "wb"
    done  = existing if mode == "ab" else 0

    t0 = time.time()
    with open(dest, mode) as f:
        for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            f.write(chunk)
            done += len(chunk)
            elapsed = max(time.time() - t0, 0.001)
            speed   = (done - existing) / elapsed / 1_048_576
            print(f"\r  {_bar(done, total)}  {speed:.1f} MB/s  ", end="", flush=True)
    print()
    return True


def main():
    print("=" * 60)
    print("  ImageGen Studio — Model Downloader")
    print("=" * 60)

    total_models = len(MODELS)
    ok = 0
    for i, m in enumerate(MODELS, 1):
        dest: Path = m["dest"]
        name: str  = m["name"]
        url:  str  = m["url"]
        size: str  = m["size"]

        print(f"\n[{i}/{total_models}] {name}  ({size})")
        print(f"  → {dest.name}")

        if dest.exists() and dest.stat().st_size > 100_000_000:
            print(f"  ✅ Already downloaded ({dest.stat().st_size/1_048_576:.0f} MB)")
            ok += 1
            continue

        success = download(url, dest, name)
        if success:
            print(f"  ✅ Saved to {dest}")
            ok += 1
        else:
            print(f"  ❌ Failed — you can retry by running this script again")

    print(f"\n{'=' * 60}")
    print(f"  Done: {ok}/{total_models} models ready")
    print(f"  Checkpoints: {CKPT_DIR}")
    print(f"  VAEs:        {VAE_DIR}")
    print("=" * 60)
    if ok < total_models:
        print("\n  Re-run this script to resume any failed downloads.")


if __name__ == "__main__":
    main()
