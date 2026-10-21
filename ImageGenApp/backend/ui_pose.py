"""
ui_pose.py - the Generate tab's 🧍 Pose control accordion (round 34): its components and its handlers (extract a skeleton from a picture, save / delete poses, download the pinned
files). `build_section(ns, ls)` is called from app._build_generate_tab right after the colour-strength accordion; `ns` is the app module (not used for state: the section only reads
and writes backend.pose_control), `ls` the last session's settings. It returns the five controls that travel with every Generate (`extras`: checkbox, pose, strength, model, legs),
in the order of the extra keys pc_on / pc_pose / pc_strength / pc_model / pc_legs.
"""
from __future__ import annotations

import html
from types import SimpleNamespace

import gradio as gr

from backend import pose_control as pc


def _ok(t: str) -> str:
    return f'<p style="color:#a6e3a1;font-size:13px;">✅ {html.escape(t)}</p>'


def _warn(t: str) -> str:
    return f'<p style="color:#fab387;font-size:13px;">⚠ {html.escape(t)}</p>'


def _files_status() -> str:
    parts = []
    parts.append("DWPose: " + ("✅" if pc.dwpose_available() else "❌ missing"))
    models = pc.list_models()
    parts.append("ControlNet: " + (", ".join(models) if models else "❌ none in models/controlnet/"))
    return '<p style="color:#9399b2;font-size:12px;">' + html.escape(" · ".join(parts)) + "</p>"


def _download_choices() -> list[str]:
    out = [f"DWPose skeleton files ({sum(d['mb'] for d in pc.DWPOSE.values())} MB)"]
    for key, m in pc.MODELS.items():
        out.append(f"{key} ({m['mb'] / 1000:.1f} GB, {m.get('license', 'licence not stated')})")
    return out


def build_section(ns, ls: dict) -> SimpleNamespace:
    ls = ls or {}
    models = pc.list_models()
    model_val = str(ls.get("pc_model") or "") or (pc.DEFAULT_MODEL if pc.DEFAULT_MODEL in models else (models[0] if models else ""))
    with gr.Accordion("🧍 Pose control — pick the pose, keep her look", open=bool(ls.get("pc_on"))):
        pc_cb = gr.Checkbox(label="Use pose control", value=bool(ls.get("pc_on")),
                            info="SDXL / Illustrious / Pony only · pose match 4 → 18 of 24 pictures on weak poses (xinsir, strength 0.8) · ≈ +10 s over a picture without PAG (+3 s over PAG 2) · "
                                 "PAG is switched off while it is on · keep the pose words in the prompt · hands break more: pair it with Polish “— hands”")
        with gr.Row():
            img = gr.Image(label="Pose picture — a picture of a person (the skeleton is read on the CPU)", type="pil", sources=["upload", "clipboard"],
                           height=280)
            preview = gr.Image(label="Skeleton (what the model gets)", interactive=False, height=280, show_download_button=False, value=None)
        with gr.Row():
            already_cb = gr.Checkbox(label="The picture is already a pose image (skeleton on black)", value=False)
            extract_btn = gr.Button("🦴 Read skeleton from the picture", size="sm")
        with gr.Row():
            pose_dd = gr.Dropdown(pc.list_poses(), value=str(ls.get("pc_pose") or ""), allow_custom_value=True, label="Pose",
                                  info=f"Saved poses (settings/poses) · “{pc.CURRENT}” = the picture above")
            save_name = gr.Textbox(label="Save the chosen pose as…", placeholder="e.g. reaching, v-sign", max_lines=1, scale=1)
        with gr.Row():
            save_btn = gr.Button("💾 Save pose", size="sm")
            del_btn = gr.Button("🗑 Delete pose", size="sm")
            refresh_btn = gr.Button("🔄", size="sm", scale=0, min_width=40)
        with gr.Row():
            strength_sl = gr.Slider(pc.STRENGTH_MIN, pc.STRENGTH_MAX, value=pc.clamp_strength(ls.get("pc_strength")), step=0.05, label="Pose strength",
                                    info="0.8 = measured sweet spot (0.4 / 0.6 follow the pose far less) · 1.0 holds it tightest")
            legs_cb = gr.Checkbox(label="Hide legs", value=bool(ls.get("pc_legs")),
                                  info="For cowboy shots and portraits: the detector invents knees the picture does not show")
        model_dd = gr.Dropdown(models, value=model_val, allow_custom_value=True, label="ControlNet model",
                               info="Folders of models/controlnet/ · the default is Apache-2.0")
        files_html = gr.HTML(_files_status())
        with gr.Row():
            dl_dd = gr.Dropdown(_download_choices(), value=_download_choices()[0], allow_custom_value=False, label="Download (SHA-256 checked, nothing downloads by itself)")
            dl_btn = gr.Button("⬇ Download", size="sm")
        status = gr.HTML("")

    def on_pose_extract(picture, already):
        if picture is None:
            return gr.update(), gr.update(), _warn("Upload a picture first.")
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
        if not str(selected or "").strip() or not pc.delete_pose(selected):
            return gr.update(), _warn("Nothing to delete — choose a saved pose.")
        left = pc.list_poses()
        return gr.update(choices=left, value=left[0] if left else ""), _ok(f"Deleted “{selected}”.")

    def on_pose_refresh():
        return gr.update(choices=pc.list_poses()), gr.update(choices=pc.list_models()), _files_status()

    def on_pose_download(which, progress=gr.Progress()):
        def say(t):
            try:
                progress(0, desc=t)
            except Exception:
                pass
        try:
            which = str(which or "")
            key = ""
            if which.startswith("DWPose"):
                msg = pc.download_dwpose(say)
            else:
                key = which.split(" (")[0]
                msg = pc.download_model(key, say)
        except Exception as e:
            print(f"[Pose] download failed: {type(e).__name__}: {e}")
            return _warn(f"Download failed: {str(e).splitlines()[0][:200] if str(e) else type(e).__name__}"), gr.update(), _files_status()
        models_now = pc.list_models()
        if not which.startswith("DWPose") and key == pc.DEFAULT_MODEL and key in models_now:
            model_upd = gr.update(choices=models_now, value=key)
        else:
            model_upd = gr.update(choices=models_now)
        return _ok(msg), model_upd, _files_status()

    img.upload(on_pose_extract, [img, already_cb], [preview, pose_dd, status])
    extract_btn.click(on_pose_extract, [img, already_cb], [preview, pose_dd, status])
    pose_dd.change(on_pose_select, [pose_dd], [preview], show_progress="hidden")
    save_btn.click(on_pose_save, [save_name, pose_dd], [pose_dd, status])
    del_btn.click(on_pose_delete, [pose_dd], [pose_dd, status])
    refresh_btn.click(on_pose_refresh, None, [pose_dd, model_dd, files_html])
    dl_btn.click(on_pose_download, [dl_dd], [status, model_dd, files_html])
    return SimpleNamespace(extras=[pc_cb, pose_dd, strength_sl, model_dd, legs_cb], preview=preview, status=status)
