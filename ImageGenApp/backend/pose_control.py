"""
pose_control.py — 🧍 Pose control: an OpenPose ControlNet for SDXL-family checkpoints, a CPU skeleton extractor (DWPose) and saved poses.

Round 33 measured it (AGENTS.md): on her four weakest poses a ControlNet raises the pose match from 4 to 18 of 24 pictures at strength 0.8, adds ~10 s per picture
and needs PAG off (PAG 2 next to the pose model reserves 12.7 GB on a 12 GB card). Everything here is off unless the Generate tab's checkbox is ticked.

* Skeleton extraction: DWPose (yzd-v/DWPose, Apache-2.0) on the CPU through onnxruntime — a YOLOX person box + a 133-keypoint SimCC model — drawn in the OpenPose style
  (own renderer, no controlnet_aux). Two files in models/detectors/, downloaded on request (SHA-256 pinned), nothing downloads by itself.
* Saved poses: settings/poses/<name>.json (+ a preview png) holds the keypoints in the source picture's pixels. The skeleton is drawn at generation time at the size of the picture
  (a different aspect ratio is fitted, not stretched) and legs can be hidden (a cowboy shot has no knees: DWPose invents them).
* ControlNet models: every folder of models/controlnet/ with a config.json and a .safetensors file. The default is the Apache-2.0 one listed in MODELS.

This module is pure (no Gradio, no torch at import); backend/sampling.py runs the ControlNet pipeline, app.py wires the controls.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from config import MODELS_DIR, SETTINGS_DIR

POSES_DIR = SETTINGS_DIR / "poses"
STARTER_DIR = Path(__file__).resolve().parent.parent / "poses"      # the built-in starter poses (keypoints only, shipped with the app; never written to)
STAR = "★ "                                                          # prefix of a built-in pose in the Pose list
# Keeping the 2.5 GB pose model loaded between the pictures of a batch (round 35) is OFF: measured on the 12 GB RX 6800M it SPILLS ~1 GB into shared GPU memory (shared 1,316 MB against
# 231-253 MB when it is released after every picture) and saved nothing (49.2 s against 43.9 s per picture over 6 X/Y cells, noise from the heat pauses). The machinery in
# sampling.run_pipe stays for cards with more VRAM (untested): set True to enable it.
KEEP_RESIDENT = False
CONTROLNET_DIR = MODELS_DIR / "controlnet"
DETECTOR_DIR = MODELS_DIR / "detectors"

DEFAULT_MODEL = "xinsir_openpose_sdxl_1.0"
TYPE_POSE, TYPE_COMP = "Pose (skeleton)", "Composition (depth)"      # the values of the 🧍 box's Control type radio (= the extra key pc_type)
STRENGTH_MIN, STRENGTH_MAX, STRENGTH_DEFAULT = 0.4, 1.0, 0.8
CURRENT = "(uploaded picture)"          # the label of the pose made from the picture in the upload box (stored as _current)
_CURRENT_FILE = "_current"
BODY_MIN_POINTS = 4                     # fewer confident body points than this = no person found
_THR = 0.3                              # keypoint confidence for drawing / counting

# ── files that can be downloaded (all SHA-256 pinned; a download that does not match is deleted) ──────────────────────────────────────────
DWPOSE = {
    "dwpose_yolox_l.onnx": dict(repo="yzd-v/DWPose", file="yolox_l.onnx", mb=217,
                                sha256="7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411"),
    "dwpose_dw-ll_ucoco_384.onnx": dict(repo="yzd-v/DWPose", file="dw-ll_ucoco_384.onnx", mb=134,
                                        sha256="724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843"),
}
# folder name -> repo, licence, files {local name: (repo file name, sha256)}
MODELS = {
    DEFAULT_MODEL: dict(repo="xinsir/controlnet-openpose-sdxl-1.0", license="Apache-2.0", mb=2386, kind="pose",
                        files={"config.json": ("config.json", "b5b8f9a7619c9452f19407ef73a01ce3f061fcf932de43ddbaca336d12d02801"),
                               "diffusion_pytorch_model.safetensors": ("diffusion_pytorch_model.safetensors",
                                                                       "b8524e557a7df60d081f5d4a0eb109967d107df217943bf88c2d99b9ebcc06c5")}),
    # round 37: the public depth ControlNet of the 🧍 Composition mode (same author and licence as the default pose model)
    "xinsir_depth_sdxl_1.0": dict(repo="xinsir/controlnet-depth-sdxl-1.0", license="Apache-2.0", mb=2386, kind="depth",
                                  files={"config.json": ("config.json", "b5b8f9a7619c9452f19407ef73a01ce3f061fcf932de43ddbaca336d12d02801"),
                                         "diffusion_pytorch_model.safetensors": ("diffusion_pytorch_model.safetensors",
                                                                                 "ea3770af4c33ea8ba8dccc0106ec5d5c5c6e504eb382119a3ccac3c9d973d344")}),
}
# PRIVATE-ONLY: models without a stated licence are listed by backend/pose_models_private.py, which the public port deletes (this block stays harmless without it).
try:                                                       # pragma: no cover - present only in the private repo
    from backend.pose_models_private import PRIVATE_MODELS as _PRIVATE
    MODELS.update(_PRIVATE)
except Exception:
    pass

_SESSIONS: dict = {}


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def clean_type(v) -> str:
    """The 🧍 control type: Composition (depth) for anything that starts with 'comp', else Pose (skeleton)."""
    return TYPE_COMP if str(v or "").strip().lower().startswith("comp") else TYPE_POSE


def clamp_strength(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return STRENGTH_DEFAULT
    if x != x:
        return STRENGTH_DEFAULT
    return min(STRENGTH_MAX, max(STRENGTH_MIN, x))


def safe_pose_name(name) -> str:
    """A pose name usable as a file name ('' if nothing is left); leading dots / underscores are stripped (`_current` is reserved)."""
    n = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name or "")).strip().lstrip("._ ").rstrip(". ")
    if n.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        n = "_" + n
    return n[:60]


def _file_of(name: str) -> str:
    """The file stem of a pose name shown in the UI ('(uploaded picture)' is `_current`)."""
    name = str(name or "").strip()
    if name == CURRENT:
        return _CURRENT_FILE
    return safe_pose_name(name)


def _round_kp(a) -> list:
    return [[round(float(x), 1), round(float(y), 1), round(float(s), 2)] for x, y, s in np.asarray(a, np.float32).reshape(-1, 3)]


def pose_hash(pose: dict) -> str:
    """10 hex digits that identify the keypoints (the record's `source`)."""
    core = json.dumps({k: pose.get(k) for k in ("size", "body", "lh", "rh")}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(core.encode()).hexdigest()[:10]


# ── saved poses ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def starter_poses() -> dict[str, str]:
    """The built-in poses: label ('★ title') -> file stem in STARTER_DIR. A file is {"title", "size", "body", "lh", "rh"}: keypoints in the pixels of a `size` canvas, no picture."""
    out = {}
    try:
        for p in sorted(STARTER_DIR.glob("*.json"), key=lambda q: q.stem.lower()):
            try:
                title = str(json.loads(p.read_text(encoding="utf-8")).get("title") or p.stem.replace("_", " ")).strip()
            except (OSError, ValueError, AttributeError):
                continue
            out[STAR + title] = p.stem
    except OSError:
        pass
    return out


def is_starter(name) -> bool:
    return str(name or "").startswith(STAR.strip())


def list_poses() -> list[str]:
    """Names for the Pose list: the uploaded picture's pose (CURRENT), the user's saved poses, then the built-in ★ poses."""
    out = []
    try:
        for p in sorted(POSES_DIR.glob("*.json"), key=lambda q: q.stem.lower()):
            out.append(CURRENT if p.stem == _CURRENT_FILE else p.stem)
    except OSError:
        pass
    mine = sorted(out, key=lambda n: (n != CURRENT, n.lower()))
    return mine + sorted(starter_poses(), key=str.lower)


def _load_starter(name: str) -> dict | None:
    stem = starter_poses().get(str(name or "").strip())
    if not stem:
        return None
    try:
        data = json.loads((STARTER_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not (isinstance(data, dict) and isinstance(data.get("body"), list) and len(data["body"]) == 18):
        return None
    data["hash"] = pose_hash(data)
    return data


def save_pose(name: str, pose: dict, preview: Image.Image | None = None) -> str:
    """Store a pose; returns the name it is listed under. `pose` = extract()'s dict (or an image pose: {"image": True, "size": [w, h]} + `preview` = the picture)."""
    if is_starter(name):
        raise ValueError("A name starting with ★ is reserved for the built-in poses — pick another name.")
    stem = _file_of(name)
    if not stem:
        raise ValueError("Give the pose a name.")
    POSES_DIR.mkdir(parents=True, exist_ok=True)
    data = dict(pose)
    if data.get("image"):
        if preview is None:
            raise ValueError("An image pose needs its picture.")
        data["hash"] = hashlib.sha256(preview.convert("RGB").tobytes()).hexdigest()[:10]
    else:
        data["hash"] = pose_hash(data)
    tmp = POSES_DIR / f"{stem}.json.part"
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, POSES_DIR / f"{stem}.json")
    if preview is not None:
        preview.convert("RGB").save(POSES_DIR / f"{stem}.png")
    return CURRENT if stem == _CURRENT_FILE else stem


def load_pose(name: str) -> dict | None:
    if is_starter(name):
        return _load_starter(name)
    stem = _file_of(name)
    if not stem:
        return None
    try:
        data = json.loads((POSES_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("image"):
        try:
            data["_picture"] = Image.open(POSES_DIR / f"{stem}.png").convert("RGB")
        except OSError:
            return None
    elif not (isinstance(data.get("body"), list) and len(data["body"]) == 18):
        return None
    return data


def delete_pose(name: str) -> bool:
    if is_starter(name):
        return False                                        # the built-in poses stay
    stem = _file_of(name)
    if not stem:
        return False
    gone = False
    for ext in (".json", ".png"):
        try:
            (POSES_DIR / f"{stem}{ext}").unlink()
            gone = True
        except OSError:
            pass
    return gone


def preview_of(name: str, size: int = 384) -> Image.Image | None:
    """A small skeleton picture of a saved pose (for the preview box)."""
    pose = load_pose(name)
    if pose is None:
        return None
    w, h = (pose.get("size") or [832, 1216])[:2]
    img = render(pose, int(w), int(h))
    img.thumbnail((size, size))
    return img


def ensure_pose_from_record(rec: dict | None) -> str | None:
    """A record's `pose_control.keypoints` as a saved pose (restoring a picture on a machine that lacks the pose). Returns the pose's name, or None."""
    if not isinstance(rec, dict):
        return None
    kp = rec.get("keypoints")
    if not (isinstance(kp, dict) and isinstance(kp.get("body"), list) and len(kp["body"]) == 18):
        return None
    pose = {"v": 1, "size": list(kp.get("size") or [832, 1216])[:2], "body": kp["body"], "lh": kp.get("lh") or [], "rh": kp.get("rh") or []}
    h = pose_hash(pose)
    named = [n for n in list_poses() if n != CURRENT]
    for n in ([str(rec["pose"])] if rec.get("pose") else []) + named + [CURRENT]:        # the record's own pose first, then a named pose with the same keypoints, then the uploaded picture's
        p = load_pose(n) if n else None
        if p is not None and p.get("hash") == h:
            return n
    name = f"restored_{h[:6]}"
    save_pose(name, pose, preview_of_pose(pose))
    return name


def preview_of_pose(pose: dict, size: int = 384) -> Image.Image:
    w, h = (pose.get("size") or [832, 1216])[:2]
    img = render(pose, int(w), int(h))
    img.thumbnail((size, size))
    return img


# ── rendering ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
_LIMBS = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
_COL = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
_HAND = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18],
         [18, 19], [19, 20]]
_LEG_POINTS = (9, 10, 12, 13)           # OpenPose body order: knees and ankles


def _hsv(h):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
    return [int(r * 255), int(g * 255), int(b * 255)]


def _layout(src_w: float, src_h: float, W: int, H: int):
    """Scale + offset that put a source picture's pixels on a W×H canvas: the same aspect ratio (±3 %) fills it, another one is fitted and centred."""
    if abs((src_w / src_h) - (W / H)) / (W / H) < 0.03:
        return W / src_w, H / src_h, 0.0, 0.0
    s = min(W / src_w, H / src_h)
    return s, s, (W - src_w * s) / 2, (H - src_h * s) / 2


def render(pose: dict, W: int, H: int, hide_legs: bool = False) -> Image.Image:
    """The OpenPose-style control picture (RGB, W×H, black background) of a stored pose."""
    import cv2
    W, H = max(64, int(W)), max(64, int(H))
    sw0, sh0 = (pose.get("size") or [W, H])[:2]
    sx, sy, ox, oy = _layout(float(sw0), float(sh0), W, H)
    if pose.get("image"):
        pic = pose.get("_picture")
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        if pic is not None:
            fit = pic.resize((max(1, round(pic.width * sx)), max(1, round(pic.height * sy))), Image.LANCZOS)
            canvas.paste(fit, (round(ox), round(oy)))
        return canvas
    sw = max(3, round(8 * H / 1216))
    canvas = np.zeros((H, W, 3), np.uint8)
    b = np.array(pose["body"], np.float32).reshape(18, 3)
    if hide_legs:
        for i in _LEG_POINTS:
            b[i, 2] = 0.0
    b[:, 0] = b[:, 0] * sx + ox
    b[:, 1] = b[:, 1] * sy + oy
    for i, (a, c) in enumerate(_LIMBS):
        p, q = b[a - 1], b[c - 1]
        if p[2] < _THR or q[2] < _THR:
            continue
        X = np.array([p[1], q[1]])
        Y = np.array([p[0], q[0]])
        length = math.hypot(X[0] - X[1], Y[0] - Y[1])
        ang = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        poly = cv2.ellipse2Poly((int(Y.mean()), int(X.mean())), (int(length / 2), sw), int(ang), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, _COL[i])
    canvas = (canvas * 0.6).astype(np.uint8)
    for i in range(18):
        if b[i][2] >= _THR:
            cv2.circle(canvas, (int(b[i][0]), int(b[i][1])), max(3, sw), _COL[i], -1)
    for key in ("lh", "rh"):
        arr = pose.get(key)
        if not arr or len(arr) != 21:
            continue
        h = np.array(arr, np.float32).reshape(21, 3)
        h[:, 0] = h[:, 0] * sx + ox
        h[:, 1] = h[:, 1] * sy + oy
        for ie, (a, c) in enumerate(_HAND):
            if h[a][2] >= _THR and h[c][2] >= _THR:
                cv2.line(canvas, (int(h[a][0]), int(h[a][1])), (int(h[c][0]), int(h[c][1])), _hsv(ie / len(_HAND)), max(2, sw // 3))
        for k in range(21):
            if h[k][2] >= _THR:
                cv2.circle(canvas, (int(h[k][0]), int(h[k][1])), max(2, sw // 3), (255, 0, 0), -1)
    return Image.fromarray(canvas)


# ── downloads and file checks ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def missing_dwpose() -> list[str]:
    return [n for n in DWPOSE if not (DETECTOR_DIR / n).is_file()]


def dwpose_available() -> bool:
    return not missing_dwpose()


def kind_of(name: str) -> str:
    """What a ControlNet folder is for: 'pose', 'depth', or 'any' (an unknown model is offered for both). A registered model says so; a local folder by its name."""
    m = MODELS.get(str(name or ""))
    if m and m.get("kind"):
        return m["kind"]
    n = str(name or "").lower()
    return "depth" if "depth" in n else "pose" if "pose" in n else "any"


def list_models(kind: str | None = None) -> list[str]:
    """Folders of models/controlnet/ that hold a config.json and a .safetensors file (the default one first). `kind` ('pose' / 'depth') keeps the ones for that use
    (plus the unclassified ones)."""
    out = []
    try:
        for d in sorted(CONTROLNET_DIR.iterdir(), key=lambda p: p.name.lower()):
            if d.is_dir() and (d / "config.json").is_file() and any(d.glob("*.safetensors")):
                if kind is None or kind_of(d.name) in (kind, "any"):
                    out.append(d.name)
    except OSError:
        pass
    first = {"depth": "xinsir_depth_sdxl_1.0"}.get(kind or "", DEFAULT_MODEL)
    return sorted(out, key=lambda n: (n != first, n.lower()))


def model_path(name: str) -> Path | None:
    name = (name or "").strip() or DEFAULT_MODEL
    if not re.fullmatch(r"[\w.\- ]{1,80}", name):
        return None
    d = CONTROLNET_DIR / name
    return d if (d / "config.json").is_file() and any(d.glob("*.safetensors")) else None


def missing_downloads() -> list[str]:
    """What the download button would fetch (DWPose files and the default ControlNet), as short labels."""
    out = [f"DWPose {DWPOSE[n]['mb']} MB" for n in missing_dwpose()]
    if model_path(DEFAULT_MODEL) is None and DEFAULT_MODEL in MODELS:
        out.append(f"ControlNet {MODELS[DEFAULT_MODEL]['mb'] / 1000:.1f} GB")
    return out


def _fetch(repo: str, repo_file: str, dest: Path, sha256: str) -> None:
    """One pinned file from the Hugging Face hub into `dest` (written as .part, checked, then renamed; a file that does not match is deleted)."""
    from huggingface_hub import hf_hub_download
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = dest.parent / ".download"
    try:
        got = Path(hf_hub_download(repo, repo_file, local_dir=str(tmp_dir)))
        if _sha256(got) != sha256:
            try:
                got.unlink()
            except OSError:
                pass
            raise RuntimeError(f"{dest.name}: the downloaded file does not match its expected SHA-256 — it was deleted.")
        os.replace(got, dest)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def download_dwpose(say=lambda t: None) -> str:
    done = []
    for name, d in DWPOSE.items():
        if (DETECTOR_DIR / name).is_file():
            continue
        say(f"Downloading {name} ({d['mb']} MB)…")
        _fetch(d["repo"], d["file"], DETECTOR_DIR / name, d["sha256"])
        done.append(name)
    _SESSIONS.clear()
    return "DWPose files ready." if not done else f"Downloaded {', '.join(done)}."


def download_model(key: str = DEFAULT_MODEL, say=lambda t: None) -> str:
    m = MODELS.get(key)
    if m is None:
        raise ValueError(f"Unknown ControlNet model '{key}'.")
    for local, (repo_file, sha) in m["files"].items():
        dest = CONTROLNET_DIR / key / local
        if dest.is_file():
            continue
        say(f"Downloading {key}/{local}…")
        _fetch(m["repo"], repo_file, dest, sha)
    return f"{key} ready ({m.get('license', 'licence not stated')})."


# ── skeleton extraction (DWPose, CPU) ───────────────────────────────────────────────────────────────────────────────────────────────────
def _make_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _session(file: str):
    path = DETECTOR_DIR / file
    key = (str(path), path.stat().st_mtime_ns if path.exists() else 0)
    if key not in _SESSIONS:
        if DWPOSE.get(file) and _sha256(path) != DWPOSE[file]["sha256"]:
            raise RuntimeError(f"{file} does not match its expected SHA-256 — delete it and use the download button.")
        _SESSIONS.clear()
        _SESSIONS[key] = _make_session(path)
    return _SESSIONS[key]


def _nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


def detect_people(bgr, thr: float = 0.3) -> list:
    """YOLOX-L person boxes [(xyxy array, score)] in the picture's pixels."""
    import cv2
    S = _session("dwpose_yolox_l.onnx")
    H, W = bgr.shape[:2]
    size = (640, 640)
    pad = np.ones((size[0], size[1], 3), np.uint8) * 114
    r = min(size[0] / H, size[1] / W)
    pad[:int(H * r), :int(W * r)] = cv2.resize(bgr, (max(1, int(W * r)), max(1, int(H * r))), interpolation=cv2.INTER_LINEAR)
    x = np.ascontiguousarray(pad.transpose(2, 0, 1), dtype=np.float32)[None]
    out = np.array(S.run(None, {S.get_inputs()[0].name: x})[0], dtype=np.float32)
    grids, strides = [], []
    for stride in (8, 16, 32):
        hs, ws = size[0] // stride, size[1] // stride
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
        strides.append(np.full((1, hs * ws, 1), stride))
    grids = np.concatenate(grids, 1)
    strides = np.concatenate(strides, 1)
    out[..., :2] = (out[..., :2] + grids) * strides
    out[..., 2:4] = np.exp(out[..., 2:4]) * strides
    p = out[0]
    boxes = p[:, :4]
    scores = p[:, 4:5] * p[:, 5:]
    xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1) / r
    cls = scores.argmax(1)
    cs = scores[np.arange(len(cls)), cls]
    ok = cs > 0.1
    xyxy, cs, cls = xyxy[ok], cs[ok], cls[ok]
    if len(cs) == 0:
        return []
    keep = _nms(xyxy, cs, 0.45)
    return [(xyxy[i], float(cs[i])) for i in keep if cls[i] == 0 and cs[i] > thr]


def _third(a, b):
    d = a - b
    return b + np.array([-d[1], d[0]])


def _keypoints(bgr, box) -> np.ndarray:
    """133 whole-body keypoints (x, y, confidence) of one person box. The model wants RGB input (BGR gave low confidences)."""
    import cv2
    S = _session("dwpose_dw-ll_ucoco_384.onnx")
    iw, ih = 288, 384
    x0, y0, x1, y1 = [float(v) for v in box]
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5])
    scale = np.array([x1 - x0, y1 - y0]) * 1.25
    asp = iw / ih
    scale = np.array([scale[0], scale[0] / asp]) if scale[0] > scale[1] * asp else np.array([scale[1] * asp, scale[1]])
    src = np.zeros((3, 2), np.float32)
    dst = np.zeros((3, 2), np.float32)
    src[0] = center
    src[1] = center + np.array([0, scale[0] * -0.5])
    src[2] = _third(src[0], src[1])
    dst[0] = [iw * 0.5, ih * 0.5]
    dst[1] = dst[0] + np.array([0, iw * -0.5])
    dst[2] = _third(dst[0], dst[1])
    M = cv2.getAffineTransform(src, dst)
    img = cv2.warpAffine(bgr, M, (iw, ih), flags=cv2.INTER_LINEAR).astype(np.float32)[:, :, ::-1]
    img = (img - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
    inp = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None]
    sx, sy = S.run(None, {S.get_inputs()[0].name: inp})
    sx, sy = np.asarray(sx)[0], np.asarray(sy)[0]
    xl = sx.argmax(1).astype(np.float32)
    yl = sy.argmax(1).astype(np.float32)
    vals = np.minimum(sx.max(1), sy.max(1))
    kp = np.stack([xl, yl], 1) / 2.0
    kp = kp / np.array([iw, ih]) * scale + center - 0.5 * scale
    return np.concatenate([kp, vals[:, None]], 1)


_OP = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]       # OpenPose body order from COCO-17 (+ the neck = the shoulders' mean)


def extract(img: Image.Image) -> tuple[dict | None, str]:
    """The skeleton of the biggest person in a picture: (pose dict, message). Raises FileNotFoundError when the DWPose files are missing."""
    miss = missing_dwpose()
    if miss:
        raise FileNotFoundError("The DWPose files are missing: " + ", ".join(miss))
    rgb = np.array(img.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()
    people = detect_people(bgr)
    if not people:
        return None, "No person found in the picture — use one that shows the whole figure clearly (or tick 'already a pose image')."
    box, score = max(people, key=lambda bs: float((bs[0][2] - bs[0][0]) * (bs[0][3] - bs[0][1])) * bs[1])
    kp = _keypoints(bgr, box)
    body = np.zeros((18, 3), np.float32)
    for i, j in enumerate(_OP):
        if j is None:
            a, b = kp[5], kp[6]
            body[i] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2])]
        else:
            body[i] = kp[j]
    if int((body[:, 2] >= _THR).sum()) < BODY_MIN_POINTS:
        return None, "A person was found but too little of the body is visible to make a skeleton."
    pose = {"v": 1, "size": [int(rgb.shape[1]), int(rgb.shape[0])], "body": _round_kp(body), "lh": _round_kp(kp[91:112]), "rh": _round_kp(kp[112:133])}
    pose["hash"] = pose_hash(pose)
    msg = f"Skeleton found ({int((body[:, 2] >= _THR).sum())} of 18 body points)"
    if len(people) > 1:
        msg += f" — {len(people)} people in the picture, the biggest was used"
    return pose, msg + "."


# ── what a Generate run needs ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def build_control(ex: dict, width: int, height: int) -> tuple[dict | None, dict | None, str]:
    """(control dict for sd.control, the record's `pose_control` entry, error text) for an `ex` with pc_on. The error is '' when everything is there."""
    name = str(ex.get("pc_pose") or "").strip()
    pose = load_pose(name) if name else None
    if pose is None:
        return None, None, ("Pose control is on but no pose is chosen — pick a saved pose or upload a picture in the 🧍 Pose control box." if not name
                            else f"Pose control: the pose '{name}' was not found (settings/poses).")
    mdl_name = (ex.get("pc_model") or "").strip() or DEFAULT_MODEL
    if kind_of(mdl_name) == "depth":
        return None, None, f"Pose control: '{mdl_name}' is a depth model — pick an OpenPose model, or switch the control type to Composition (depth)."
    mdl = model_path(mdl_name)
    if mdl is None:
        have = missing_downloads()
        return None, None, (f"Pose control: the ControlNet model '{mdl_name}' is not in models/controlnet/. "
                            + ("Use the ⬇ Download button in the 🧍 Pose control box." if mdl_name == DEFAULT_MODEL and have else "Pick another model or download the default one."))
    legs = bool(ex.get("pc_legs"))
    strength = clamp_strength(ex.get("pc_strength"))
    image = render(pose, width, height, hide_legs=legs)
    rec = {"model": mdl.name, "strength": strength, "pose": name, "source": pose.get("hash") or pose_hash(pose), "legs": "hidden" if legs else "shown"}
    if not pose.get("image"):
        rec["keypoints"] = {k: pose.get(k) for k in ("size", "body", "lh", "rh")}
    ctl = {"image": image, "scale": strength, "model": str(mdl), "name": name}
    return ctl, rec, ""
