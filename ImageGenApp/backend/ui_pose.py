"""
ui_pose.py - the Generate tab's 🧍 Pose / Composition control accordion (round 34, composition added in round 37): its components and its handlers (extract a skeleton or a depth map from a
picture, save / delete poses, download the pinned files). `build_section(ns, ls)` is called from app._build_generate_tab right after the colour-strength accordion; `ns` is the app module
(not used for state: the section only reads and writes backend.pose_control / backend.composition), `ls` the last session's settings. It returns the six controls that travel with every
Generate (`extras`: checkbox, pose, strength, model, legs, control type), in the order of the extra keys pc_on / pc_pose / pc_strength / pc_model / pc_legs / pc_type.

The control type radio switches the box between two modes that share the checkbox, the picture upload, the strength slider and the model list: Pose (skeleton: DWPose, saved poses, hide
legs) and Composition (depth: Depth-Anything-V2-Small, strength 0.4-0.8 default 0.5, depth ControlNets). The pose-only controls hide in Composition mode; saved poses and the ★ starters stay
skeleton-only (the depth map of the uploaded picture is kept as settings/composition/_current.png, never in settings/poses).
"""
from __future__ import annotations

import html
from types import SimpleNamespace

import gradio as gr

from backend import composition as comp
from backend import pose_control as pc


def _ok(t: str) -> str:
    return f'<p style="color:#a6e3a1;font-size:13px;">✅ {html.escape(t)}</p>'


def _warn(t: str) -> str:
    return f'<p style="color:#fab387;font-size:13px;">⚠ {html.escape(t)}</p>'


def _files_status() -> str:
    parts = []
    parts.append("DWPose: " + ("✅" if pc.dwpose_available() else "❌ missing"))
    models = pc.list_models("pose")
    parts.append("OpenPose ControlNet: " + (", ".join(models) if models else "❌ none in models/controlnet/"))
    parts.append("depth estimator: " + ("✅" if comp.estimator_available() else "❌ missing"))
    depth = pc.list_models("depth")
    parts.append("depth ControlNet: " + (", ".join(depth) if depth else "❌ none in models/controlnet/"))
    return '<p style="color:#9399b2;font-size:12px;">' + html.escape(" · ".join(parts)) + "</p>"


DEPTH_ESTIMATOR_CHOICE = f"Depth estimator (Depth-Anything-V2-Small, {comp.ESTIMATORS[comp.ESTIMATOR_FILE]['mb']} MB, {comp.ESTIMATORS[comp.ESTIMATOR_FILE]['license']})"


def _download_choices() -> list[str]:
    out = [f"DWPose skeleton files ({sum(d['mb'] for d in pc.DWPOSE.values())} MB)", DEPTH_ESTIMATOR_CHOICE]
    for key, m in pc.MODELS.items():
        out.append(f"{key} ({m['mb'] / 1000:.1f} GB, {m.get('license', 'licence not stated')})")
    return out


def _model_for(kind: str, current: str) -> tuple[list[str], str]:
    """(the model list for a mode, the value to show): the current one when it fits the mode, else the mode's default, else the first."""
    models = pc.list_models(kind)
    default = comp.DEFAULT_MODEL if kind == "depth" else pc.DEFAULT_MODEL
    cur = str(current or "")
    return models, cur if cur in models else (default if default in models else (models[0] if models else ""))


def _depth_preview(depth, size: int = 384):
    img = depth.convert("RGB")
    img.thumbnail((size, size))
    return img


def build_section(ns, ls: dict) -> SimpleNamespace:
    ls = ls or {}
    ptype = pc.clean_type(ls.get("pc_type"))
    is_comp = ptype == pc.TYPE_COMP
    models, model_val = _model_for("depth" if is_comp else "pose", ls.get("pc_model"))
    s_min, s_max, s_def = (comp.STRENGTH_MIN, comp.STRENGTH_MAX, comp.STRENGTH_DEFAULT) if is_comp else (pc.STRENGTH_MIN, pc.STRENGTH_MAX, pc.STRENGTH_DEFAULT)
    strength0 = (comp.clamp_strength if is_comp else pc.clamp_strength)(ls.get("pc_strength")) if ls.get("pc_strength") is not None else s_def
    with gr.Accordion("🧍 Pose / composition control — pick the pose or copy a picture's composition, keep her look", open=bool(ls.get("pc_on"))):
        pc_cb = gr.Checkbox(label="Use pose / composition control", value=bool(ls.get("pc_on")),
                            info="SDXL / Illustrious / Pony only · ≈ +10 s per picture · PAG is switched off while it is on · "
                                 "Pose: pose match 4 → 18 of 24 on weak poses, pair it with Polish “— hands” · "
                                 "Composition: the figure, framing and hands of a picture, but its clothes leak in ~1 of 4")
        type_rd = gr.Radio([pc.TYPE_POSE, pc.TYPE_COMP], value=ptype, label="Control type",
                           info="Pose = a skeleton (poses are saved, ★ starters) · Composition = the depth map of the picture: same figure and framing, new outfit (strength 0.5, never saved)")
        with gr.Row():
            img = gr.Image(label="Composition picture — its depth map is made on the CPU" if is_comp else "Pose picture — a picture of a person (the skeleton is read on the CPU)",
                           type="pil", sources=["upload", "clipboard"], height=280)
            preview = gr.Image(label="Depth map, flattened (what the model gets)" if is_comp else "Skeleton (what the model gets)", interactive=False, height=280,
                               show_download_button=False, value=None)
        with gr.Row():
            already_cb = gr.Checkbox(label="The picture is already a depth map (used as it is)" if is_comp else "The picture is already a pose image (skeleton on black)", value=False)
            extract_btn = gr.Button("🗺 Make the depth map" if is_comp else "🦴 Read skeleton from the picture", size="sm")
        with gr.Row(visible=not is_comp) as pose_row:
            pose_dd = gr.Dropdown(pc.list_poses(), value=str(ls.get("pc_pose") or ""), allow_custom_value=True, label="Pose",
                                  info=f"★ = built-in starter poses · your saved poses (settings/poses) · “{pc.CURRENT}” = the picture above")
            save_name = gr.Textbox(label="Save the chosen pose as…", placeholder="e.g. reaching, v-sign", max_lines=1, scale=1)
        with gr.Row(visible=not is_comp) as pose_btns:
            save_btn = gr.Button("💾 Save pose", size="sm")
            del_btn = gr.Button("🗑 Delete pose", size="sm")
            refresh_btn = gr.Button("🔄", size="sm", scale=0, min_width=40)
        with gr.Row():
            strength_sl = gr.Slider(s_min, s_max, value=strength0, step=0.05, label="Pose / composition strength",
                                    info="Pose: 0.8 = measured sweet spot (0.4 / 0.6 follow the pose far less), 1.0 holds it tightest · Composition: 0.5 default, 0.4-0.8 (higher copies more of the source's clothes)")
            legs_cb = gr.Checkbox(label="Hide legs", value=bool(ls.get("pc_legs")), visible=not is_comp,
                                  info="For cowboy shots and portraits: the detector invents knees the picture does not show")
        model_dd = gr.Dropdown(models, value=model_val, allow_custom_value=True, label="ControlNet model",
                               info="Folders of models/controlnet/ that suit the control type · the defaults are Apache-2.0")
        files_html = gr.HTML(_files_status())
        with gr.Row():
            dl_choices = _download_choices()
            dl_dd = gr.Dropdown(dl_choices, value=DEPTH_ESTIMATOR_CHOICE if is_comp else dl_choices[0], allow_custom_value=False,
                                label="Download (SHA-256 checked, nothing downloads by itself)")
            dl_btn = gr.Button("⬇ Download", size="sm")
        status = gr.HTML("")

    def on_pose_extract(picture, already, ptype_in=pc.TYPE_POSE):
        if picture is None:
            return gr.update(), gr.update(), _warn("Upload a picture first.")
        if pc.clean_type(ptype_in) == pc.TYPE_COMP:
            try:
                depth = picture.convert("L") if already else comp.depth_map(picture)
            except FileNotFoundError:
                return gr.update(), gr.update(), _warn("The depth estimator file is missing — pick “Depth estimator” in the Download box below and press ⬇ Download "
                                                      "(or tick “already a depth map”).")
            except Exception as e:
                print(f"[Composition] depth map failed: {type(e).__name__}: {e}")
                return gr.update(), gr.update(), _warn(f"Could not make a depth map: {str(e).splitlines()[0][:160] if str(e) else type(e).__name__}")
            comp.save_current(depth)
            return _depth_preview(comp.flatten(depth)), gr.update(), _ok("Depth map ready — Generate with the box ticked copies this picture's figure and framing (the picture itself is not stored in the record).")
        try:
            if already:
                from PIL import Image
                pic = picture.convert("RGB")
                pc.save_pose(pc.CURRENT, {"v": 1, "image": True, "size": [pic.width, pic.height]}, preview=pic)
                shown = pc.preview_of(pc.CURRENT)
                return shown, gr.update(choices=pc.list_poses(), value=pc.CURRENT), _ok("The picture is used as the pose image as it is.")
            pose, msg = pc.extract(picture)
        except FileNotFoundError:
            return gr.update(), gr.update(), _warn("The DWPose files are missing — pick “DWPose skeleton files” in the Download box below and press ⬇ Download "
                                                  "(or tick “already a pose image”).")
        except Exception as e:
            print(f"[Pose] extraction failed: {type(e).__name__}: {e}")
            return gr.update(), gr.update(), _warn(f"Could not read a skeleton: {str(e).splitlines()[0][:160] if str(e) else type(e).__name__}")
        if pose is None:
            return gr.update(), gr.update(), _warn(msg)
        pc.save_pose(pc.CURRENT, pose, preview=pc.preview_of_pose(pose))
        return pc.preview_of(pc.CURRENT), gr.update(choices=pc.list_poses(), value=pc.CURRENT), _ok(msg)

    def on_pose_select(name):
        shown = pc.preview_of(name) if str(name or "").strip() else None
        return shown if shown is not None else gr.update()

    def on_pose_save(name, selected):
        stem = pc.safe_pose_name(name)
        if not stem:
            return gr.update(), _warn("Type a name for the pose first.")
        src = pc.load_pose(selected or pc.CURRENT)
        if src is None:
            return gr.update(), _warn("Choose a pose (or read one from a picture) first.")
        prev = src.pop("_picture", None) if src.get("image") else pc.preview_of_pose(src)
        pc.save_pose(stem, src, preview=prev)
        return gr.update(choices=pc.list_poses(), value=stem), _ok(f"Saved as “{stem}”.")

    def on_pose_delete(selected):
        if pc.is_starter(selected):
            return gr.update(), _warn("The ★ starter poses stay — save a copy under your own name to change it.")
        if not str(selected or "").strip() or not pc.delete_pose(selected):
            return gr.update(), _warn("Nothing to delete — choose a saved pose.")
        left = pc.list_poses()
        return gr.update(choices=left, value=left[0] if left else ""), _ok(f"Deleted “{selected}”.")

    def on_pose_refresh(ptype_in=pc.TYPE_POSE):
        kind = "depth" if pc.clean_type(ptype_in) == pc.TYPE_COMP else "pose"
        return gr.update(choices=pc.list_poses()), gr.update(choices=pc.list_models(kind)), _files_status()

    def on_pose_download(which, ptype_in=pc.TYPE_POSE, progress=gr.Progress()):
        def say(t):
            try:
                progress(0, desc=t)
            except Exception:
                pass
        key = ""
        try:
            which = str(which or "")
            if which.startswith("DWPose"):
                msg = pc.download_dwpose(say)
            elif which.startswith("Depth estimator"):
                msg = comp.download_estimator(say)
            else:
                key = which.split(" (")[0]
                msg = pc.download_model(key, say)
        except Exception as e:
            print(f"[Pose] download failed: {type(e).__name__}: {e}")
            return _warn(f"Download failed: {str(e).splitlines()[0][:200] if str(e) else type(e).__name__}"), gr.update(), _files_status()
        kind = "depth" if pc.clean_type(ptype_in) == pc.TYPE_COMP else "pose"
        models_now = pc.list_models(kind)
        if key and key in models_now and key in (pc.DEFAULT_MODEL, comp.DEFAULT_MODEL):
            model_upd = gr.update(choices=models_now, value=key)
        else:
            model_upd = gr.update(choices=models_now)
        return _ok(msg), model_upd, _files_status()

    def on_type_change(ptype_in, pose_name, cur_model):
        """The control type radio: the strength range / default, the model list, the pose-only controls and the labels follow the mode (a UI-only listener, no API endpoint)."""
        t = pc.clean_type(ptype_in)
        c = t == pc.TYPE_COMP
        mdls, val = _model_for("depth" if c else "pose", cur_model)
        if c:
            depth = comp.load_current()
            shown = _depth_preview(comp.flatten(depth)) if depth is not None else None
        else:
            shown = pc.preview_of(pose_name) if str(pose_name or "").strip() else None
        return (gr.update(minimum=comp.STRENGTH_MIN if c else pc.STRENGTH_MIN, maximum=comp.STRENGTH_MAX if c else pc.STRENGTH_MAX,
                          value=comp.STRENGTH_DEFAULT if c else pc.STRENGTH_DEFAULT),
                gr.update(choices=mdls, value=val),
                gr.update(visible=not c), gr.update(visible=not c), gr.update(visible=not c),
                gr.update(label="Composition picture — its depth map is made on the CPU" if c else "Pose picture — a picture of a person (the skeleton is read on the CPU)"),
                gr.update(label="Depth map, flattened (what the model gets)" if c else "Skeleton (what the model gets)", value=shown),
                gr.update(label="The picture is already a depth map (used as it is)" if c else "The picture is already a pose image (skeleton on black)"),
                gr.update(value="🗺 Make the depth map" if c else "🦴 Read skeleton from the picture"),
                gr.update(value=DEPTH_ESTIMATOR_CHOICE if c else _download_choices()[0]))

    img.upload(on_pose_extract, [img, already_cb, type_rd], [preview, pose_dd, status])
    extract_btn.click(on_pose_extract, [img, already_cb, type_rd], [preview, pose_dd, status])
    pose_dd.change(on_pose_select, [pose_dd], [preview], show_progress="hidden")
    save_btn.click(on_pose_save, [save_name, pose_dd], [pose_dd, status])
    del_btn.click(on_pose_delete, [pose_dd], [pose_dd, status])
    refresh_btn.click(on_pose_refresh, [type_rd], [pose_dd, model_dd, files_html])
    dl_btn.click(on_pose_download, [dl_dd, type_rd], [status, model_dd, files_html])
    type_rd.input(on_type_change, [type_rd, pose_dd, model_dd], [strength_sl, model_dd, pose_row, pose_btns, legs_cb, img, preview, already_cb, extract_btn, dl_dd],
                  api_name=False, show_progress="hidden")
    return SimpleNamespace(extras=[pc_cb, pose_dd, strength_sl, model_dd, legs_cb, type_rd], preview=preview, status=status)
