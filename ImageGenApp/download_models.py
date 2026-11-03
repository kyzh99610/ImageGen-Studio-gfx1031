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

Each file comes from a fixed Hugging Face commit, goes to <name>.part first (an interrupted download resumes there) and gets its
real name only once its size and SHA-256 match.
"""

from __future__ import annotations
import hashlib, sys, time
from pathlib import Path

# ── Resolve paths ──────────────────────────────────────────────────────────────
APP_DIR   = Path(__file__).parent.resolve()
CKPT_DIR  = APP_DIR / "models" / "checkpoints"
VAE_DIR   = APP_DIR / "models" / "vae"

# ── Model manifest ─────────────────────────────────────────────────────────────
HF = "https://huggingface.co"
MODELS = [
    {
        "name":   "DreamShaper 8",
        "url":    f"{HF}/Yntec/Dreamshaper8/resolve/9b7fee79ae78b56b182d03bc17ab5f73446c8f97/dreamshaper_8.safetensors",
        "dest":   CKPT_DIR / "dreamshaper_8.safetensors",
        "bytes":  2_132_625_894,
        "sha256": "879db523c30d3b9017143d56705015e15a2cb5628762c11d086fed9538abd7fd",
    },
    {
        "name":   "Realistic Vision V6.0 NV (fp16)",
        "url":    f"{HF}/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/9a857a696b9aabbf509073e0aa55ec8200b6ef7d/Realistic_Vision_V6.0_NV_B1_fp16.safetensors",
        "dest":   CKPT_DIR / "Realistic_Vision_V6.0_NV_B1_fp16.safetensors",
        "bytes":  2_132_625_894,
        "sha256": "c48bfd159cd7a6507b128685e963c398fa72399cefafaf603781df50ce836cc7",
    },
    {
        "name":   "AbsoluteReality V1.6525 (pruned)",
        "url":    f"{HF}/Lykon/AbsoluteReality/resolve/df7c3efce8f54d02cf73a21fcecd908a0fde2898/AbsoluteRealityV1.6525_pruned.safetensors",
        "dest":   CKPT_DIR / "AbsoluteRealityV1.6525_pruned.safetensors",
        "bytes":  2_132_625_432,
        "sha256": "be1d90c4abb7bb0183f267f899f38b44112ad6ef9a757a6723514ea4e9be15dc",
    },
    {
        "name":   "SD 1.5 VAE (ft-mse)",
        "url":    f"{HF}/stabilityai/sd-vae-ft-mse-original/resolve/629b3ad3030ce36e15e70c5db7d91df0d60c627f/vae-ft-mse-840000-ema-pruned.safetensors",
        "dest":   VAE_DIR / "vae-ft-mse-840000.safetensors",
        "bytes":  334_641_190,
        "sha256": "735e4c3a447a3255760d7f86845f09f937809baa529c17370d83e4c3758f3c75",
    },
]


def _bar(done: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return f"{done / 1_048_576:.1f} MB"
    pct = done / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:5.1f}%  {done/1_048_576:.1f}/{total/1_048_576:.1f} MB"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int, sha256: str) -> bool:
    """Fetch into dest.part (resuming what is there), check size and SHA-256, then rename to dest."""
    import requests

    part = dest.with_name(dest.name + ".part")
    existing = part.stat().st_size if part.exists() else 0
    if existing > size:                                   # longer than the real file: not resumable
        part.unlink()
        existing = 0
    if existing < size:
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True)
        except Exception as e:
            print(f"  ERROR connecting: {e}")
            return False
        if r.status_code not in (200, 206):
            print(f"  ERROR: HTTP {r.status_code}")
            return False
        mode = "ab" if existing and r.status_code == 206 else "wb"   # 200: the server sent the whole file again
        done = existing if mode == "ab" else 0
        t0 = time.time()
        try:
            with open(part, mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)
                    done += len(chunk)
                    elapsed = max(time.time() - t0, 0.001)
                    print(f"\r  {_bar(done, size)}  {(done - existing) / elapsed / 1_048_576:.1f} MB/s  ", end="", flush=True)
        except Exception as e:
            print(f"\n  ERROR while downloading: {e}")
            return False
        print()
    got = part.stat().st_size
    if got != size:
        print(f"  ERROR: {got:,} bytes, expected {size:,} (run again to resume)")
        return False
    print("  Checking SHA-256 …")
    digest = _sha256(part)
    if digest != sha256:
        part.unlink()                                     # a resume can't repair it
        print(f"  ERROR: SHA-256 {digest[:16]}… is not the expected {sha256[:16]}… — the download was removed, run again")
        return False
    part.replace(dest)
    return True


def main() -> int:
    print("=" * 60)
    print("  ImageGen Studio — Model Downloader")
    print("=" * 60)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    VAE_DIR.mkdir(parents=True, exist_ok=True)

    total_models = len(MODELS)
    ok = 0
    for i, m in enumerate(MODELS, 1):
        dest: Path = m["dest"]
        print(f"\n[{i}/{total_models}] {m['name']}  ({m['bytes'] / 2**30:.1f} GB)")
        print(f"  → {dest.name}")

        if dest.exists():
            have = dest.stat().st_size
            if have == m["bytes"]:
                print(f"  ✅ Already downloaded ({have / 1_048_576:.0f} MB)")
                ok += 1
                continue
            part = dest.with_name(dest.name + ".part")
            if have < m["bytes"] and not part.exists():
                dest.replace(part)                        # a half download from an older version of this script: resume it
                print(f"  Resuming an incomplete file ({have / 1_048_576:.0f} MB)")
            else:
                print(f"  ⚠ {dest.name} exists with a different size ({have:,} bytes) — left as it is; move it away to download this one")
                continue

        if download(m["url"], dest, m["bytes"], m["sha256"]):
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
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
