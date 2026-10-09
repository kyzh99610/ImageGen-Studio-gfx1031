"""
ui_settings.py - the preference block of the Settings tab (prompt weights, eye style, power profile, cool mode, pause while hot): its components and its handlers,
moved out of app.py in round 12 (no behaviour change). `build_prefs_section(ns)` is called from app._build_settings_tab inside its column; `ns` is the app module, so the handlers
reach the app's own helpers at call time (`ns._save_pref` - the tests patch it on the app module) and nothing here imports app.
"""
from __future__ import annotations

import html

import gradio as gr

POWER_LABELS = {"full": "Full power (Turbo / Manual): cool 1.5 + pause above 88 °C",
                "limited": "Power-limited (Windows \"Performance\" scheme / USB-C charger): cool 0 + pause above 88 °C",
                "custom": "Custom (the two sliders below)"}
POWER_INFO = ("Full power = what the app ships: cool 1.5 + pause above 88 °C. Power-limited is for a SLOW charger only — the Windows \"Performance\" scheme on a USB-C charger "
              "(a sampling step takes ~4 s per megapixel instead of ~2): cool 0 + pause above 88 did a ✨ Polish Full-body picture in 202 s instead of 388 s, and 8 pictures back to back "
              "stayed under 94 °C. On the full-power charger it is NOT safe: 8 % of the readings over 42 minutes were at 94 °C or more (max 95.9), with a 30 s run, and the pause waited 70 % of the time: keep Full power there. "
              "Custom = move the two sliders below. Saved for the next start.")


def power_choice():
    """The Power-profile radio value for the live cool factor + pause limit (a gr.update for the Settings radio)."""
    from backend import sampling, thermal
    return gr.update(value=POWER_LABELS[thermal.profile_of(sampling.COOL["factor"], thermal.GUARD["limit"])])


def build_prefs_section(ns) -> None:
    from backend.prompt_syntax import EMPHASIS as _EMPH, set_emphasis as _set_emph
    gr.Markdown("### ⚖️ Prompt weights")
    emph_rb = gr.Radio(
        ["A1111 (default)", "Compel (stronger)"],
        value="Compel (stronger)" if _EMPH["mode"] == "compel" else "A1111 (default)",
        label="How (tag:1.2) weights are applied",
        info="A1111 = like A1111 / Forge / Civitai (weight, then keep the overall strength). Compel = "
             "the app's old behaviour: a weighted tag counts ~5× more — weighted background / colour "
             "tags bleached their colour into the character. Saved for the next start.")
    emph_status = gr.HTML("")

    def on_emphasis(choice):
        mode = _set_emph("compel" if str(choice).startswith("Compel") else "a1111")
        ns._save_pref("emphasis", mode)
        return f'<p style="color:#a6e3a1;font-size:13px;">✅ Prompt weights: {mode}</p>'
    emph_rb.change(on_emphasis, [emph_rb], [emph_status])
    from backend.detail_tools import EYE_STYLE as _EYE_STYLE, set_eye_style as _set_eye_style_ui
    gr.Markdown("### 👁 Eye style")
    eye_rb = gr.Radio(
        ["Round pupils (default)", "Natural"],
        value="Natural" if _EYE_STYLE["mode"] == "natural" else "Round pupils (default)",
        label="How the eye pass (✨ Polish, 👁 Also re-draw eyes) draws pupils",
        info="Round pupils = the shipped words (\"detailed pupils, round pupils\", and \"slit pupils, cat eyes\" in the negative): round pupils with a clear highlight "
             "and iris shading, a small dark pupil dot. Natural = no pupil words at all (the negative stays): the pass cleans the eyes and leaves the pupil to the base "
             "picture — softer and less contrasty (the same eye words minus \"round pupils\" alone drew slit pupils in 5 of 6 test pictures, so that is not offered). "
             "A character card can carry its own choice (🎴 Load applies it, 📌 saves the current one). Saved for the next start.")
    eye_status = gr.HTML("")

    def on_eye_style(choice):
        mode = _set_eye_style_ui("natural" if str(choice).startswith("Natural") else "round")
        ns._save_pref("eye_style", mode)
        return f'<p style="color:#a6e3a1;font-size:13px;">✅ Eye style: {mode}</p>'
    eye_rb.change(on_eye_style, [eye_rb], [eye_status])
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
                        info="0 = off. For laptops that shut down under long GPU runs: 1.5 = the GPU works ~40 % of "
                             "the time (measured: 5 h without a power-off on the RX 6800M laptop, which powered off "
                             "after 3–4 min at full load). Pictures stay identical; generation takes (1 + value)× as "
                             "long for the sampling part. Saved for the next start.")
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
                         info="Before each picture and each hires / face / eye pass, wait until the laptop's thermal "
                              "zone is 8 °C below this. The RX 6800M laptop switched off after minutes at 96 °C; 90 "
                              "is a sensible value there. "
                              + (f"Now: {_t_now:.0f} °C." if _t_now is not None else "This PC reports no thermal zone.")
                              + " Saved for the next start.")
    therm_status = gr.HTML("")

    def on_therm(v):
        f = _thm.set_limit(v)
        ns._save_pref("thermal_limit", f)
        return (f'<p style="color:#a6e3a1;font-size:13px;">✅ Pausing above {f:g} °C (resume at {f - 8:g} °C)</p>'
                if f else '<p style="color:#a6adc8;font-size:13px;">Pause while hot: off</p>'), power_choice()
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
        return (f'<p style="color:#a6e3a1;font-size:13px;">✅ {html.escape(POWER_LABELS[name])}: pause {cool:g}× the step time, '
                f'wait above {limit:g} °C (resume at {limit - 8:g} °C)</p>', gr.update(value=cool), gr.update(value=limit))
    power_rb.input(on_power_profile, [power_rb], [power_status, cool_sl, therm_sl])
