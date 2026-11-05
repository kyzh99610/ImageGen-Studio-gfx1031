"""
ui_settings.py - the preference block of the Settings tab (prompt weights, eye style, power profile, cool mode, pause while hot): its components and its handlers,
moved out of app.py in round 12 (no behaviour change). `build_prefs_section(ns)` is called from app._build_settings_tab inside its column; `ns` is the app module, so the handlers
reach the app's own helpers at call time (`ns._save_pref` - the tests patch it on the app module) and nothing here imports app.
"""
from __future__ import annotations

import html

import gradio as gr
from backend.i18n import T          # 中文 / English: the Chinese of a hint (backend/i18n.py)

POWER_LABELS = {"full": "Full power (Turbo, unlimited CPU): cool 1.5 + pause above 88 °C",
                "limited": "Power-limited (USB-C charger, or Manual with a CPU limit): cool 0 + pause above 88 °C",
                "custom": "Custom (the two sliders below)"}
POWER_INFO = T("Sets the two sliders below for the way the laptop is powered. Full power = cool 1.5 + pause above 88 °C, the cautious default. "
               "Power-limited = cool 0 + pause above 88 °C: faster, and it stayed under 94 °C on a slow charger or a CPU limit. Numbers: Help → Cool mode.",
               "按笔记本的供电方式设置下面两个滑块。满功率 = 降温 1.5 + 超过 88 °C 暂停，最稳妥的默认值。"
               "限功率 = 降温 0 + 超过 88 °C 暂停：更快，在慢充或限制 CPU 功率时温度没超过 94 °C。数据见帮助 → 降温模式。")


def pause_status(limit, temp_now) -> str:
    """The Pause-while-hot status line. With no thermal zone nothing can ever wait, so a limit must not read as protection."""
    if not limit:
        return '<p style="color:#a6adc8;font-size:13px;">Pause while hot: off</p>'
    if temp_now is None:
        return f'<p style="color:#f9e2af;font-size:13px;">⚠ Saved {limit:g} °C, but this PC reports no thermal zone: nothing will wait.</p>'
    return f'<p style="color:#a6e3a1;font-size:13px;">✅ Pausing above {limit:g} °C (resume at {limit - 8:g} °C)</p>'


def power_choice():
    """The Power-profile radio value for the live cool factor + pause limit (a gr.update for the Settings radio)."""
    from backend import sampling, thermal
    return gr.update(value=POWER_LABELS[thermal.profile_of(sampling.COOL["factor"], thermal.GUARD["limit"])])


def build_prefs_section(ns) -> None:
    from backend.prompt_syntax import EMPHASIS as _EMPH, set_emphasis as _set_emph, MODE_LABELS as _EMPH_LABELS
    gr.Markdown("### ⚖️ Prompt weights")
    _emph_choices = {"a1111_normkeep": "Norm-keep (default)", "a1111": "A1111 (old)", "compel": "Compel (stronger)"}
    emph_rb = gr.Radio(
        list(_emph_choices.values()),
        value=_emph_choices.get(_EMPH["mode"], "Norm-keep (default)"),
        label="How (tag:1.2) weights are applied",
        info=T("Norm-keep = a weight lifts its tag over the tags beside it and the rest of the prompt keeps its strength (recommended). A1111 (old) = A1111 / Forge's own maths: on SDXL a chunk full of weighted tags can come out weaker or much stronger than written, and high weights on skin / hair phrases drew blobs. Compel = a weighted tag counts ~5× more, and weighted background or colour tags bleed into the character. Remembered for the next start.",
               "Norm-keep = 加权只让这个标签比旁边的标签更突出，提示词其余部分的强度不变（推荐）。A1111 (old) = A1111 / Forge 原本的算法：在 SDXL 上，一段里加权的标签一多，整段可能比写的弱、也可能强很多；皮肤 / 头发描述用高权重时画出过色块。Compel = 加权的标签作用大约强 5 倍，背景或颜色标签加权后会染到角色身上。下次启动仍然有效。"))
    emph_status = gr.HTML("")

    def on_emphasis(choice):
        mode = _set_emph(choice)
        ns._save_pref("emphasis", mode)
        return f'<p style="color:#a6e3a1;font-size:13px;">✅ Prompt weights: {_EMPH_LABELS[mode]}</p>'
    emph_rb.change(on_emphasis, [emph_rb], [emph_status])
    from backend.detail_tools import EYE_STYLE as _EYE_STYLE, set_eye_style as _set_eye_style_ui
    gr.Markdown("### 👁 Eye style")
    eye_rb = gr.Radio(
        ["Round pupils (default)", "Natural"],
        value="Natural" if _EYE_STYLE["mode"] == "natural" else "Round pupils (default)",
        label="How the eye pass (✨ Polish, 👁 Also re-draw eyes) draws pupils",
        info=T("Round pupils = round pupils with a clear highlight and iris shading (the default). Natural = no pupil words: the eye pass cleans up and leaves the pupils as the picture drew them, softer. A character card can carry its own choice. Remembered for the next start.", "圆瞳孔 = 有清楚高光和虹膜阴影的圆瞳孔（默认）。自然 = 不加瞳孔描述：修眼睛时只做整理，瞳孔保持原图的样子，更柔和。角色卡可以带自己的选择。下次启动仍然有效。"))
    eye_status = gr.HTML("")

    def on_eye_style(choice):
        mode = _set_eye_style_ui("natural" if str(choice).startswith("Natural") else "round")
        ns._save_pref("eye_style", mode)
        return f'<p style="color:#a6e3a1;font-size:13px;">✅ Eye style: {mode}</p>'
    eye_rb.change(on_eye_style, [eye_rb], [eye_status])
    from backend import neg_hygiene as _nh
    gr.Markdown("### 🧹 Negative hygiene")
    nh_cb = gr.Checkbox(
        value=lambda: _nh.CONFIG["on"], label="Drop the negative tags the prompt asks for, keep repeated negative tags once",   # a callable: read again at every page load
        info=T("Before each picture, leaves out the negative tags that fight the prompt (\"long hair\" when the prompt says \"very long hair\", \"blurry\" against \"depth of field\"; more in settings/neg_conflicts.json) and keeps a repeated tag once. The result line lists what was left out. Remembered for the next start.", "每张图之前，去掉和正向提示词冲突的反向标签（正向写了 \"very long hair\" 就去掉 \"long hair\"，\"blurry\" 对 \"depth of field\"；更多见 settings/neg_conflicts.json），重复的标签只留一个。结果里会列出去掉了什么。下次启动仍然有效。"))
    nh_status = gr.HTML("")

    def on_neg_hygiene(v):
        on = _nh.set_on(v)
        ns._save_pref("neg_hygiene", on)
        return (f'<p style="color:#a6e3a1;font-size:13px;">✅ Negative hygiene {"on" if on else "off"}</p>')
    nh_cb.input(on_neg_hygiene, [nh_cb], [nh_status])      # .input: the user's click only (a page load refreshing the value must not write the pref)
    from backend import sampling as _smp
    from backend import thermal as _thm
    gr.Markdown("### 🌡 Power profile")
    power_rb = gr.Radio(
        list(POWER_LABELS.values()),
        value=POWER_LABELS[_thm.profile_of(_smp.COOL["factor"], _thm.GUARD["limit"])],
        label="How the laptop is powered (sets the two sliders below)",
        info=POWER_INFO)
    power_status = gr.HTML("")
    gr.Markdown("### 🌡 Cool mode")
    cool_sl = gr.Slider(0, 3, value=_smp.COOL["factor"], step=0.25,
                        label="Pause after every sampling step (× the step's time)",
                        info=T("0 = off. For laptops that shut down under long GPU work: 1.5 = the GPU works ~40 % of the time. Pictures stay the same; the drawing part takes (1 + value)× as long. Remembered for the next start.", "0 = 关闭。给长时间跑显卡会关机的笔记本用：1.5 = 显卡大约只工作 40 % 的时间。图不变；画图部分的时间变成 (1 + 这个值) 倍。下次启动仍然有效。"))
    cool_status = gr.HTML("")

    def on_cool(v):
        f = _smp.set_cool(v)
        ns._save_pref("cool", f)
        return (f'<p style="color:#a6e3a1;font-size:13px;">✅ Cool mode: pause {f:g}× the step time</p>' if f
                else '<p style="color:#a6adc8;font-size:13px;">Cool mode off</p>'), power_choice()
    cool_sl.release(on_cool, [cool_sl], [cool_status, power_rb])
    _t_now = _thm.read_temp()
    therm_sl = gr.Slider(0, 96, value=_thm.GUARD["limit"], step=1,
                         label="Pause while hotter than (°C) — 0 = off",
                         info=T("Before each picture and each hires / face / eye pass, waits until the thermal zone is 8 °C below this. "
                                "The RX 6800M laptop switched off after minutes at 96 °C; 90 is sensible there. "
                                + (f"Now: {_t_now:.0f} °C." if _t_now is not None else "This PC reports no temperature, so it never waits.")
                                + " Remembered for the next start.",
                                "每张图、每次高清修复 / 修脸 / 修眼睛之前，等温度降到比这个值低 8 °C。RX 6800M 笔记本在 96 °C 跑几分钟就会关机；那台机器设 90 比较合适。"
                                + (f"现在：{_t_now:.0f} °C。" if _t_now is not None else "这台电脑读不到温度，所以永远不会等。")
                                + "下次启动仍然有效。"))
    therm_status = gr.HTML("")

    def on_therm(v):
        f = _thm.set_limit(v)
        ns._save_pref("thermal_limit", f)
        return pause_status(f, _t_now), power_choice()
    therm_sl.release(on_therm, [therm_sl], [therm_status, power_rb])

    def on_power_profile(choice):
        name = next((k for k, v in POWER_LABELS.items() if v == choice), "custom")
        got = _thm.set_profile(name, _smp)
        if got is None:                                          # "Custom": the sliders stay as they are
            return ('<p style="color:#a6adc8;font-size:13px;">Custom: set the two sliders below.</p>',
                    gr.update(), gr.update())
        cool, limit = got
        ns._save_pref("cool", cool)
        ns._save_pref("thermal_limit", limit)
        wait = (f'wait above {limit:g} °C (resume at {limit - 8:g} °C)' if _t_now is not None
                else f'⚠ no wait above {limit:g} °C: this PC reports no thermal zone')
        return (f'<p style="color:#a6e3a1;font-size:13px;">✅ {html.escape(POWER_LABELS[name])}: pause {cool:g}× the step time, '
                f'{wait}</p>', gr.update(value=cool), gr.update(value=limit))
    power_rb.input(on_power_profile, [power_rb], [power_status, cool_sl, therm_sl])

    # 📇 Run manifest folders: pictures of runs that keep their recipes in manifest*.jsonl (backend/run_manifests.py) — dropped into
    # img2img they bring their run's txt2img settings back. A UI-only listener: no endpoint.
    def _dirs_now():
        try:
            import json as _j
            v = _j.loads((ns.SETTINGS_DIR / "_prefs.json").read_text(encoding="utf-8")).get("manifest_dirs") or []
        except Exception:
            v = []
        return "\n".join(v) if isinstance(v, list) else str(v)
    man_tb = gr.Textbox(value=_dirs_now, lines=2, label="📇 Run manifest folders (one per line)",
                        info=T("Folders with run folders that hold manifest*.jsonl (one record per picture). A picture from such a run dropped into img2img brings back that run's prompt, model, LoRAs, sampler, size, seed and passes.", "包含运行文件夹的目录，里面有 manifest*.jsonl（每张图一条记录）。把这种运行里的图拖进图生图，就能恢复那次运行的提示词、模型、LoRA、采样器、尺寸、种子和细化步骤。"))
    man_status = gr.HTML("")

    def on_manifest_dirs(text):
        from pathlib import Path as _P
        dirs = [l.strip().strip('"') for l in str(text or "").splitlines() if l.strip()]
        ns._save_pref("manifest_dirs", dirs)
        bad = [d for d in dirs if not _P(d).is_dir()]
        from backend import run_manifests as _rm
        n = len(_rm.index([_P(d) for d in dirs if _P(d).is_dir()])) if dirs else 0
        return (f'<p style="color:#a6e3a1;font-size:13px;">✅ {len(dirs) - len(bad)} folder(s), {n} pictures listed in their manifests</p>'
                + (f'<p style="color:#f9e2af;font-size:13px;">⚠ not a folder: {html.escape(", ".join(bad))}</p>' if bad else ""))
    man_tb.blur(on_manifest_dirs, [man_tb], [man_status], api_name=False)
