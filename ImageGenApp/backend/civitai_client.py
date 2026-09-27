"""
backend/civitai_client.py
Civitai REST API v1 client — search models, retrieve version metadata,
and download checkpoints / LoRAs with real-time progress.
Docs: https://developer.civitai.com/docs/api/public-rest
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import httpx

from config import (
    CIVITAI_API_BASE, CIVITAI_API_KEY,
    MODELS_DIR, CHECKPOINTS_DIR, LORAS_DIR, VAE_DIR, UPSCALERS_DIR,
    EMBEDDINGS_DIR, EMBEDDINGS_SD15_DIR, EMBEDDINGS_SDXL_DIR,
)

# Map Civitai model type → local directory
_TYPE_TO_DIR: dict[str, Path] = {
    "Checkpoint":        CHECKPOINTS_DIR,
    "TextualInversion":  EMBEDDINGS_DIR,
    "Hypernetwork":      CHECKPOINTS_DIR / ".." / "hypernetworks",
    "AestheticGradient": MODELS_DIR / "other" / "aestheticgradient",
    "LORA":              LORAS_DIR,
    "LoCon":             LORAS_DIR,
    "DoRA":              LORAS_DIR,
    "VAE":               VAE_DIR,
    "Upscaler":          UPSCALERS_DIR,
    # Types this app can't use go to models/other/<type>/ — in checkpoints/ a motion
    # module (.safetensors) would show up in the Checkpoint list and fail to load.
    "MotionModule":      MODELS_DIR / "other" / "motionmodule",
    "Wildcards":         MODELS_DIR / "other" / "wildcards",
    "Poses":             MODELS_DIR / "other" / "poses",
    "Controlnet":        CHECKPOINTS_DIR / ".." / "controlnet",
    "Classifier":        MODELS_DIR / "other" / "classifier",
    "Other":             MODELS_DIR / "other" / "other",
}


def _safe_filename(name: str, fallback: str = "model.safetensors") -> str:
    """A file name from the API, made safe to save on Windows: no folders ("../x"),
    no characters Windows forbids (<>:"/\\|?*), no reserved device names (CON, NUL…)."""
    import re
    name = str(name or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name).strip().rstrip(". ")
    stem = name.split(".")[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}:
        name = "_" + name
    return name[:200] or fallback


def _esc(v) -> str:
    from html import escape
    return escape(str(v if v is not None else ""))


class CivitaiClient:
    """
    Thin async-friendly wrapper around the Civitai public REST API.
    All methods are synchronous but use httpx for proper timeouts.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or CIVITAI_API_KEY
        self._client = self._make_client()

    def _make_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=CIVITAI_API_BASE,
            timeout=30.0,
            headers=self._build_headers(),
            follow_redirects=True,
        )

    def _build_headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def set_api_key(self, key: str):
        self.api_key = key.strip() or None
        # Recreate the client so the Authorization header is guaranteed fresh
        self._client = self._make_client()

    # ── Search ─────────────────────────────────────────────────────────────────
    def search_models(
        self,
        query: str = "",
        tag: str = "",
        model_type: str = "Checkpoint",
        sort: str = "Most Downloaded",
        period: str = "AllTime",
        limit: int = 20,
        page: int = 1,
        nsfw: bool = False,
        base_model: str = "",
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """
        Returns raw API response dict:
          { items: [...], metadata: { totalItems, currentPage, pageSize, nextCursor, ... } }

        Note: Civitai API requires cursor-based pagination when a query is
        provided.  The ``page`` param is only sent for non-query browsing.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "sort": sort,
            "period": period,
            "nsfw": str(nsfw).lower(),  # Civitai expects lowercase "true"/"false"
        }

        # Pass API key as query param too — Civitai supports both auth methods,
        # and query-param auth is more reliable than header injection.
        if self.api_key:
            params["token"] = self.api_key

        if query:
            params["query"] = query
            if cursor:
                params["cursor"] = cursor
        elif cursor:
            # cursor-based pagination (next page token) takes priority over page number
            params["cursor"] = cursor
        else:
            params["page"] = page
        if tag:
            params["tag"] = tag
        # Plural names since Civitai's 2026 API change: the singular `type=` / `baseModel=`
        # are now silently ignored (a LoRA search returned mostly checkpoints).
        if model_type and model_type != "All":
            params["types"] = model_type
        if base_model:
            params["baseModels"] = base_model

        resp = self._client.get("/models", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_model(self, model_id: int) -> dict[str, Any]:
        resp = self._client.get(f"/models/{model_id}")
        resp.raise_for_status()
        return resp.json()

    def get_model_version(self, version_id: int) -> dict[str, Any]:
        resp = self._client.get(f"/model-versions/{version_id}")
        resp.raise_for_status()
        return resp.json()

    # ── Download ───────────────────────────────────────────────────────────────
    def download_model_version(
        self,
        model: dict[str, Any],
        version: dict[str, Any],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[bool, str]:
        """
        Download the primary file of a model version plus any companion
        textual-inversion embedding files (.pt / small .safetensors).
        Returns (success, local_path_or_error).
        self.last_companion_embeddings is set to the list of downloaded names.
        """
        self.last_companion_embeddings: list[str] = []
        files = version.get("files", [])
        if not files:
            return False, "No files found for this version."

        # Prefer: primary+safetensors > primary > any safetensors > first file
        files = [f for f in files if isinstance(f, dict)]
        if not files:
            return False, "No files found for this version."
        primary_sf  = next((f for f in files if f.get("primary") and str(f.get("name", "")).endswith(".safetensors")), None)
        primary_any = next((f for f in files if f.get("primary")), None)
        any_sf      = next((f for f in files if str(f.get("name", "")).endswith(".safetensors")), None)
        primary     = primary_sf or primary_any or any_sf or files[0]

        url       = primary.get("downloadUrl") or primary.get("url")
        if not url:
            return False, "Civitai didn't give a download link for this file."
        file_name = _safe_filename(primary.get("name"), f"civitai_{version.get('id', 'model')}.safetensors")
        model_type = model.get("type", "Checkpoint")
        save_dir   = _TYPE_TO_DIR.get(model_type, MODELS_DIR / "other" / str(model_type).lower())
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path  = save_dir / file_name
        # Stream into a .part file and rename when complete, so an interrupted
        # download never leaves a truncated model under its real name (which the
        # "already exists" check below would then treat as a finished download).
        part_path  = save_dir / (file_name + ".part")

        already_existed = save_path.exists()

        if already_existed:
            # Primary exists — check for missing companion embeddings
            self.last_companion_embeddings = self._download_companion_embeddings(
                files, primary, version, progress_callback
            )
            return True, str(save_path)

        # Stream download with progress + real-time speed/ETA
        try:
            total = int((primary.get("sizeKB") or 0) * 1024)
            # A .part left by a dropped connection is resumed with an HTTP Range request
            resume_from = part_path.stat().st_size if part_path.exists() else 0
            headers = self._build_headers()
            if resume_from:
                headers["Range"] = f"bytes={resume_from}-"
            downloaded = 0
            t_start = time.monotonic()

            with httpx.stream(
                "GET", url,
                headers=headers,
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, read=300.0),
            ) as resp:
                if resp.status_code == 404:
                    return False, (
                        "File not found (404) — this model was likely removed from Civitai "
                        "or the download link expired. Try refreshing the model page."
                    )
                if resp.status_code in (401, 403):
                    return False, (
                        f"Civitai refused the download (HTTP {resp.status_code}). This file needs an "
                        "API key — paste yours in the API key box above (free at civitai.com → "
                        "Account settings → API Keys)."
                    )
                if resp.status_code == 416 and resume_from:
                    # The server won't serve that range (file changed?) — start over cleanly
                    part_path.unlink(missing_ok=True)
                    return self.download_model_version(model, version, progress_callback)
                resp.raise_for_status()
                if "text/html" in resp.headers.get("content-type", ""):
                    return False, (
                        "Civitai sent a web page instead of the file — usually a login/age gate. "
                        "Set your Civitai API key above and try again."
                    )
                body_len = int(resp.headers.get("content-length", 0) or 0)
                if resume_from and resp.status_code == 206:
                    downloaded = resume_from                 # append to what we already have
                    content_len = resume_from + body_len if body_len else 0
                    print(f"[Civitai] Resuming {file_name} at {resume_from / 1e6:.1f} MB")
                else:
                    resume_from = 0                          # server ignored Range: start over
                    content_len = body_len or total
                # Fail up front instead of after gigabytes when the drive can't hold the file
                import shutil
                free = shutil.disk_usage(save_dir).free
                still_needed = (content_len - downloaded) if content_len else 0
                if still_needed and still_needed + (512 << 20) > free:
                    return False, (
                        f"Not enough disk space on {Path(save_dir).anchor or save_dir}: this file needs "
                        f"{still_needed / 2**30:.1f} GB more but only {free / 2**30:.1f} GB is free. "
                        "Free up some space (or delete models you don't use) and try again."
                    )

                with open(part_path, "ab" if resume_from else "wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=1 << 17):  # 128 KB
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and content_len:
                            elapsed = time.monotonic() - t_start
                            avg_speed = (downloaded - resume_from) / elapsed if elapsed > 0.1 else 0
                            pct      = downloaded / content_len
                            mb_done  = downloaded / 1_000_000
                            mb_total = content_len / 1_000_000
                            remaining = (content_len - downloaded) / avg_speed if avg_speed > 0 else 0

                            speed_str = (
                                f"{avg_speed/1_000_000:.1f} MB/s" if avg_speed >= 1_000_000
                                else f"{avg_speed/1_000:.0f} KB/s"
                            )
                            eta_str = (
                                f"{remaining:.0f}s"      if remaining < 60
                                else f"{remaining/60:.1f} min" if remaining < 3600
                                else f"{remaining/3600:.1f} hr"
                            )
                            elapsed_str = (
                                f"{elapsed:.0f}s"      if elapsed < 60
                                else f"{elapsed/60:.1f} min" if elapsed < 3600
                                else f"{elapsed/3600:.1f} hr"
                            )
                            progress_callback(
                                pct,
                                f"⬇ {mb_done:.1f} / {mb_total:.1f} MB  —  {speed_str}  —  {elapsed_str} elapsed  —  ETA {eta_str}",
                            )

            if content_len and downloaded != content_len:
                if downloaded > content_len:
                    part_path.unlink(missing_ok=True)
                    return False, "Download came out larger than expected — discarded. Try again."
                return False, (f"Download incomplete ({downloaded / 1e6:.0f} of "
                               f"{content_len / 1e6:.0f} MB) — connection dropped. "
                               "Press Download again to resume where it stopped.")
            part_path.replace(save_path)

            # Final progress message with total time
            total_elapsed = time.monotonic() - t_start
            if progress_callback:
                elapsed_str = (
                    f"{total_elapsed:.1f}s" if total_elapsed < 60
                    else f"{total_elapsed/60:.1f} min" if total_elapsed < 3600
                    else f"{total_elapsed/3600:.1f} hr"
                )
                mb_total = downloaded / 1_000_000
                avg_speed = (downloaded - resume_from) / total_elapsed if total_elapsed > 0.1 else 0
                speed_str = (
                    f"{avg_speed/1_000_000:.1f} MB/s" if avg_speed >= 1_000_000
                    else f"{avg_speed/1_000:.0f} KB/s"
                )
                progress_callback(1.0, f"✅ {mb_total:.1f} MB in {elapsed_str} ({speed_str} avg)")

            # Save sidecar metadata for trigger word lookup
            sidecar = Path(str(save_path) + ".civitai.json")
            try:
                sidecar.write_text(
                    __import__("json").dumps({
                        "trainedWords": version.get("trainedWords", []),
                        "baseModel":    version.get("baseModel", ""),
                        "model":        {"name": model.get("name", ""), "type": model.get("type", "")},
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

            # Download companion TI embedding files
            self.last_companion_embeddings = self._download_companion_embeddings(
                files, primary, version, progress_callback
            )
            return True, str(save_path)

        except httpx.HTTPStatusError as e:
            part_path.unlink(missing_ok=True)
            if e.response.status_code == 404:
                return False, "File not found (404) — model may have been removed from Civitai."
            return False, f"Download failed: HTTP {e.response.status_code}"
        except httpx.TransportError as e:   # network dropped: keep the .part for resuming
            have = part_path.stat().st_size / 1e6 if part_path.exists() else 0
            return False, (f"Download interrupted ({type(e).__name__}) after {have:.1f} MB — "
                           "press Download again to resume where it stopped.")
        except BaseException as e:   # includes Gradio cancelling the event mid-download
            part_path.unlink(missing_ok=True)
            if not isinstance(e, Exception):
                raise
            return False, f"Download failed: {e}"

    def _download_companion_embeddings(
        self,
        files: list[dict],
        primary: dict,
        version: dict,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[str]:
        """
        Download companion textual-inversion embedding files from the same
        model version into the architecture-specific subdir (sd15/ or sdxl/).
        Returns list of downloaded filenames.
        """
        _TI_EXTS = {".pt", ".safetensors", ".bin"}
        _MAX_TI_SIZE_KB = 50_000  # 50 MB — TI embeddings are tiny, skip huge files

        companions = []
        for f in files:
            if f is primary:
                continue
            name = f.get("name") or ""
            ext = Path(name).suffix.lower()
            size_kb = f.get("sizeKB") or 0
            if ext in _TI_EXTS and size_kb < _MAX_TI_SIZE_KB:
                companions.append(f)

        if not companions:
            return []

        # Pick the right subdir based on the version's baseModel
        base = (version.get("baseModel") or "").lower()
        is_sdxl = any(k in base for k in ("sdxl", "pony", "illustrious", "noob", "il "))
        save_dir = EMBEDDINGS_SDXL_DIR if is_sdxl else EMBEDDINGS_SD15_DIR
        arch_label = "sdxl" if is_sdxl else "sd15"
        save_dir.mkdir(parents=True, exist_ok=True)

        downloaded_names = []
        for f in companions:
            fname = _safe_filename(f.get("name"), "")
            if not fname:
                continue
            dest = save_dir / fname
            # Also check old flat dir to avoid re-downloading moved files
            if dest.exists() or (EMBEDDINGS_DIR / fname).exists():
                downloaded_names.append(fname)
                continue
            url = f.get("downloadUrl") or f.get("url")
            if not url:
                continue
            try:
                if progress_callback:
                    progress_callback(0.99, f"⬇ Downloading companion embedding: {fname}")
                resp = httpx.get(
                    url,
                    headers=self._build_headers(),
                    follow_redirects=True,
                    timeout=httpx.Timeout(30.0, read=120.0),
                )
                resp.raise_for_status()
                if "text/html" in resp.headers.get("content-type", ""):
                    raise RuntimeError("got a web page (login/age gate) instead of the file")
                dest.write_bytes(resp.content)
                downloaded_names.append(fname)
                print(f"[Civitai] Companion embedding saved: {fname} → embeddings/{arch_label}/")
            except Exception as e:
                print(f"[Civitai] Failed to download companion embedding {fname}: {e}")

        return downloaded_names

    # ── Format helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def format_model_card(model: dict[str, Any]) -> str:
        """Return an HTML model card for display in Gradio."""
        name       = _esc(model.get("name") or "Unknown")
        model_type = _esc(model.get("type") or "")
        stats      = model.get("stats") or {}
        dl         = stats.get("downloadCount") or 0
        thumbs_up  = stats.get("thumbsUpCount") or 0
        thumbs_dn  = stats.get("thumbsDownCount") or 0
        # Compute approval percentage (Civitai removed numeric rating)
        total_votes = thumbs_up + thumbs_dn
        approval    = (thumbs_up / total_votes * 100) if total_votes > 0 else 0
        versions   = [v for v in (model.get("modelVersions") or []) if isinstance(v, dict)]
        base_model = _esc(versions[0].get("baseModel") or "") if versions else ""
        creator    = _esc((model.get("creator") or {}).get("username") or "unknown")
        model_id   = _esc(model.get("id") or "")

        # Get first preview image
        preview_url = ""
        if versions:
            imgs = versions[0].get("images") or []
            if imgs and isinstance(imgs[0], dict):
                preview_url = _esc(imgs[0].get("url") or "")

        img_html = (
            f'<img src="{preview_url}" style="width:100%;height:200px;border-radius:6px;'
            f'object-fit:cover;display:block;" />'
            if preview_url else
            '<div style="height:200px;background:#1e1e2e;border-radius:6px;'
            'display:flex;align-items:center;justify-content:center;'
            'color:#a6adc8;font-size:13px;">No preview</div>'
        )

        # Count companion embedding files across versions
        companion_count = 0
        if versions:
            for f in versions[0].get("files") or []:
                fname = f.get("name") or ""
                ext = Path(fname).suffix.lower()
                size_kb = f.get("sizeKB") or 0
                is_primary = f.get("primary", False)
                if not is_primary and ext in (".pt", ".safetensors", ".bin") and size_kb < 50_000:
                    companion_count += 1

        companion_badge = ""
        if companion_count:
            companion_badge = (
                f'<span style="font-size:13px;color:#89dceb;background:#1e3a4a;'
                f'border-radius:4px;padding:2px 5px;margin-left:4px;">'
                f'📎 +{companion_count} embedding{"s" if companion_count > 1 else ""}</span>'
            )

        return f"""
<div style="border:1px solid #333;border-radius:10px;padding:10px;
            background:#16161e;font-family:sans-serif;color:#cdd6f4;">
  {img_html}
  <div style="margin-top:8px;display:flex;align-items:flex-start;gap:6px;">
    <b style="font-size:13px;flex:1;overflow:hidden;display:-webkit-box;
              -webkit-line-clamp:2;-webkit-box-orient:vertical;">{name}</b>
    <span style="flex-shrink:0;font-size:13px;color:#fff;background:#313244;
                 border-radius:4px;padding:2px 5px;">{model_type}</span>
    {companion_badge}
  </div>
  <div style="font-size:13px;color:#a6adc8;margin-top:4px;
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
    {base_model} · {creator}
  </div>
  <div style="font-size:13px;margin-top:6px;display:flex;justify-content:space-between;align-items:center;">
    <span>⬇ {dl:,} &nbsp;👍 {thumbs_up:,} ({approval:.0f}%)</span>
    <a href="https://civitai.com/models/{model_id}" target="_blank"
       style="color:#89b4fa;text-decoration:none;">Civitai ↗</a>
  </div>
</div>"""

    @staticmethod
    def get_version_choices(model: dict[str, Any]) -> list[tuple[str, int]]:
        """Return list of (display_name, version_id) for a model's versions."""
        return [
            (f"{v.get('name') or 'v?'}  [{v.get('baseModel') or ''}]", v["id"])
            for v in (model.get("modelVersions") or [])
            if isinstance(v, dict) and v.get("id") is not None
        ]
