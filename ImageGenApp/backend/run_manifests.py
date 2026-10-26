"""
run_manifests.py — recipes of pictures that carry no metadata of their own but were listed by the run that made them.

Some generators (scripts, other forks of this app, batch bots) save plain PNGs and keep each picture's recipe in a manifest file
next to them: `<run folder>/manifest*.jsonl`, one JSON record per picture, `name` = the picture's file stem, with its prompt,
negative, checkpoint, LoRAs, sampler, steps, CFG, size, seed and hires settings. Point Settings → "Run manifest folders" (pref
`manifest_dirs`) at the folders that hold such run folders, and a picture dropped into img2img (or PNG Info) that has no
parameters of its own is looked up there: by its file name (names repeat across runs, so the candidates' own pictures are compared
with the dropped one), or, when the browser did not pass a name, by its pixels against the runs' best/ pictures.

`to_meta(record)` turns a record into the dict `records.restore_plan` reads, so the Generate tab gets the run's txt2img settings back;
`advice(...)` says what a 12 GB card should change (at most 3 LoRA slots, the strongest kept; files that are not installed here and
the local ones of the same name family; the time per picture).
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

_LOCK = threading.Lock()
_INDEX: dict = {"stamp": None, "names": {}}
_FP: dict = {}                       # path -> (mtime, fingerprint)
FP_SIZE = 24


def dirs_from_prefs(settings_dir) -> list[Path]:
    try:
        p = json.loads((Path(settings_dir) / "_prefs.json").read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = p.get("manifest_dirs") or []
    if isinstance(raw, str):
        raw = [x for x in re.split(r"[;\n]", raw)]
    return [Path(str(x).strip()) for x in raw if str(x).strip() and Path(str(x).strip()).is_dir()]


def _manifests(dirs) -> list[Path]:
    out = []
    for d in dirs:
        d = Path(d)
        out += sorted(d.glob("manifest*.jsonl")) + sorted(d.glob("*/manifest*.jsonl"))
    return out


def index(dirs) -> dict:
    """{record name: [(manifest path, line number)]}, rebuilt when a manifest changes (63 MB of JSON ≈ a few seconds, once)."""
    files = _manifests(dirs)
    stamp = tuple((str(f), f.stat().st_mtime_ns) for f in files)
    with _LOCK:
        if _INDEX["stamp"] == stamp:
            return _INDEX["names"]
        names: dict = {}
        for f in files:
            with open(f, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    m = re.search(r'"name"\s*:\s*"([^"]+)"', line[:400])
                    if m:
                        names.setdefault(m.group(1), []).append((str(f), i))
        _INDEX.update(stamp=stamp, names=names)
        return names


def _record(path: str, line_no: int) -> dict | None:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i == line_no:
                try:
                    r = json.loads(line)
                except ValueError:
                    return None
                return r if isinstance(r, dict) else None
    return None


def match_names(stem: str, names: dict) -> list[str]:
    """Record names for a file stem: exact; else the longest record name the stem starts with (a suffix like '_h40' was added);
    else the names that start with the stem."""
    if not stem:
        return []
    if stem in names:
        return [stem]
    pre = [n for n in names if stem.startswith(n)]
    if pre:
        return [max(pre, key=len)]
    return sorted(n for n in names if n.startswith(stem))[:20]


def fingerprint(img) -> list[int]:
    from PIL import Image
    g = img.convert("L").resize((FP_SIZE, FP_SIZE), Image.BILINEAR)
    return list(g.getdata())


def _fp_file(path: Path):
    from PIL import Image
    mt = path.stat().st_mtime_ns
    hit = _FP.get(str(path))
    if hit and hit[0] == mt:
        return hit[1]
    with Image.open(path) as im:
        fp = fingerprint(im)
    _FP[str(path)] = (mt, fp)
    return fp


def distance(a, b) -> float:
    """Mean absolute difference of two fingerprints (0 = the same picture; another picture of the same scene ≈ 15-40)."""
    return sum(abs(x - y) for x, y in zip(a, b)) / max(1, len(a))


def _pictures_named(run_dir: Path, stem: str) -> list[Path]:
    return [p for p in [run_dir / f"{stem}.png"] + list(run_dir.glob(f"*/{stem}.png")) if p.is_file()]


def find(img, name: str, dirs, max_dist: float = 12.0) -> dict | None:
    """{record, run, picture, distance, how} for a dropped picture, or None. `name` = its original file name (may be '')."""
    if not dirs:
        return None
    names = index(dirs)
    fp = fingerprint(img) if img is not None else None
    stem = Path(str(name or "")).stem
    best = None
    for n in match_names(stem, names):
        for path, line in names[n]:
            run = Path(path).parent
            pics = _pictures_named(run, stem) or _pictures_named(run, n)
            d = min((distance(fp, _fp_file(p)) for p in pics), default=None) if fp is not None and pics else None
            cand = dict(name=n, path=path, line=line, run=run, picture=pics[0] if pics else None, distance=d, how="name")
            key = (d if d is not None else 999.0, -Path(path).stat().st_mtime)
            if best is None or key < best[0]:
                best = (key, cand)
    if best is None and fp is not None:                     # no usable name: the runs' best/ pictures by their pixels
        by_run: dict = {}
        for n, locs in names.items():
            for q, l in locs:
                by_run.setdefault(str(Path(q).parent), {})[n] = (q, l)
        for run in dict.fromkeys(m.parent for m in _manifests(dirs)):
            rn = by_run.get(str(run), {})
            for p in sorted((run / "best").glob("*.png")):
                d = distance(fp, _fp_file(p))
                if d <= max_dist and (best is None or d < best[0][0]):
                    hit = match_names(p.stem, rn)
                    if hit:
                        path, line = rn[hit[0]]
                        best = ((d, 0), dict(name=hit[0], path=path, line=line, run=run, picture=p, distance=d, how="pixels"))
    if best is None:
        return None
    c = best[1]
    if c["distance"] is not None and c["distance"] > 40:     # the name matched but it is clearly another picture: say so, still use it
        c["how"] += " (the picture itself differs: another version of that scene)"
    rec = _record(c["path"], c["line"])
    if not rec:
        return None
    c["record"] = rec
    return c


def _ck_name(v) -> str:
    n = Path(str(v or "").replace("\\", "/")).name
    return n if not n or Path(n).suffix else n + ".safetensors"


def to_meta(rec: dict) -> dict:
    """The record as read_png_info's dict (what records.restore_plan reads). LoRAs strongest first: only 3 slots."""
    loras = sorted([(str(n), float(w)) for n, w in (rec.get("loras") or []) if n], key=lambda t: -abs(t[1]))
    h = rec.get("hires") if isinstance(rec.get("hires"), dict) else {}
    det = rec.get("detail") if isinstance(rec.get("detail"), dict) else {}
    meta = {"prompt": str(rec.get("prompt") or ""), "negative_prompt": str(rec.get("negative") or rec.get("negative_prompt") or ""),
            "sampler": str(rec.get("scheduler") or rec.get("sampler") or ""), "steps": rec.get("steps"), "cfg_scale": rec.get("cfg_scale"),
            "seed": rec.get("seed"), "width": rec.get("width"), "height": rec.get("height"), "model": _ck_name(rec.get("ckpt") or rec.get("model")),
            "loras": ", ".join(f"{n}:{w:g}" for n, w in loras), "pag_scale": float(rec.get("pag_scale") or 0.0),
            "emphasis": str(rec.get("weighting_mode") or "a1111")}
    if h and h.get("applied", True) and h.get("factor"):
        meta["hires"] = {"scale": float(h.get("factor") or 1.5), "denoise": float(h.get("denoise") or 0.4),
                         "steps": int(h.get("steps") or rec.get("steps") or 12), "upscaler": str(h.get("upscaler") or "Real-ESRGAN (ONNX)")}
    if det.get("faces_refined"):
        meta["face_detail"] = {"denoise": float(det.get("face_denoise") or 0.28)}
    if det.get("hands_refined"):
        meta["hand_detail"] = {"denoise": float(det.get("hand_denoise") or 0.55)}
    if det.get("eyes_refined"):
        meta["eye_detail"] = {"denoise": float(det.get("eye_denoise") or 0.4), "style": "unknown"}     # its own eye pass; the style is not recorded
    return meta


def advice(rec: dict, local_checkpoints, local_loras, seconds: float | None = None) -> list[str]:
    """What to change on this machine (a 12 GB card): LoRAs beyond 3 slots, files not installed (+ local ones that look related), time."""
    out = []
    loras = sorted([(str(n), float(w)) for n, w in (rec.get("loras") or []) if n], key=lambda t: -abs(t[1]))
    if len(loras) > 3:
        out.append("it used " + str(len(loras)) + " LoRAs; the 3 strongest are in the slots, left out: "
                   + ", ".join(f"{n} {w:g}" for n, w in loras[3:]) + " (a 12 GB card runs SDXL + 3 fused LoRAs fine; more slots would not fit the UI)")
    ck_name = Path(_ck_name(rec.get("ckpt") or rec.get("model"))).stem
    ck = ck_name.lower()
    local_ck = [Path(str(n)).stem for n in local_checkpoints]
    if ck and ck not in (x.lower() for x in local_ck):
        tok = [t for t in re.split(r"[^a-z]+", ck) if len(t) >= 4]
        near = [x for x in local_ck if any(t in x.lower() for t in tok)][:3]
        out.append(f"checkpoint {ck_name} is not installed" + (f"; on disk with a related name: {', '.join(near)}" if near else
                                                           "; pick an Illustrious / NoobAI checkpoint you have"))
    local_lo = {Path(str(n)).stem.lower() for n in local_loras}
    miss = [n for n, _ in loras[:3] if Path(n).stem.lower() not in local_lo]
    if miss:
        out.append("LoRA not installed: " + ", ".join(miss) + " (download it, or leave the slot empty)")
    hs = int((rec.get("hires") or {}).get("steps") or rec.get("steps") or 0) if isinstance(rec.get("hires"), dict) and rec["hires"].get("factor") else 0
    if hs > 14:
        out.append(f"its hires fix ran {hs} steps (as many as the base); this app's measured hires recipe uses 12 at about the same "
                   "denoise — lower Hires steps to 12 for speed (12 against {hs} was not compared here)".replace("{hs}", str(hs)))
    if seconds:
        out.append(f"about {seconds / 60:.1f} min per picture here at full speed (its steps, size, prompt length and passes)")
    return out
