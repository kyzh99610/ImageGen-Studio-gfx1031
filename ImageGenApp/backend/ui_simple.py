"""ui_simple.py — the 🪄 一键出图 / Quick picture tab: a few plain choices and one button (the plan is backend/simple_mode.py).

`build_tab(ns)` makes the tab (it comes first in the page); `wire(ns, ui, main_tabs)` connects it once the Generate tab exists, because a run
is that tab's own Generate (`ns._GEN_HELPERS["generate_ui"]`, the raw function: this tab's event holds the GPU lock) and "Open in Generate"
fills that tab's controls. `ns` is the app module, looked up at call time (the tests patch it). One named endpoint, `/do_simple_generate`
(a GPU job); the preview, the follow-ups and the list refreshes have none.
"""
from __future__ import annotations

import html
import inspect
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from backend import simple_mode as SM
from backend.i18n import L, T, LANG_ARG_JS

LAST_FILE_NAME = "_simple_last.json"      # the tab's last choices (settings/, "_" = never listed as a saved setting)

# shown texts of the radio choices (backend/simple_mode keeps the values)
for _en, _zh in (("Saved setting", "已保存的设置"), ("Character card", "角色卡"), ("Just a prompt", "只写一句话"), ("Auto", "自动"),
                 ("Portrait", "头像"), ("Cowboy shot (thighs up)", "半身（到大腿）"), ("Full body", "全身"), ("Wide shot", "远景"),
                 ("As saved", "按设置"), ("Fast", "快"), ("Standard", "标准"), ("Best (slower)", "精细（更慢）"), ("Tall", "竖图"),
                 ("Square", "方图"), ("Wide", "横图"), ("Vivid", "鲜艳"), ("Natural", "自然"), ("Soft", "柔和"),
                 ("Softer lines", "线条柔和"), ("Painterly", "手绘感"), ("Drawing style", "画风"),
                 (SM.FOLLOW, "（按设置）"), (SM.RANDOM_OUTFIT, "（随机服装）"), (SM.NO_OUTFIT, "（不加服装）"),
                 # the tab's labels and buttons
                 ("🪄 Quick picture", "🪄 一键出图"), ("Start from", "从哪开始"), ("Saved setting", "已保存的设置"), ("Character card", "角色卡"),
                 ("Outfit", "服装"), ("Model", "模型"), ("What to draw", "想画什么"), ("Add variety (a wildcard file)", "加点变化（通配符文件）"),
                 ("Framing", "取景"), ("Detail", "精细程度"), ("Picture shape", "画面形状"), ("Colour", "色彩"), ("How many pictures", "张数"),
                 ("Seed", "种子"), ("Check every picture afterwards", "出图后自动检查每张图"), ("✨ Make the pictures", "✨ 生成"),
                 ("⏹ Stop", "⏹ 停止"), ("🔄 Refresh the lists", "🔄 刷新列表"), ("Pictures", "图片"),
                 ("🔀 More like the picked one", "🔀 再来几张类似的"), ("⬆ Redo the picked one in more detail", "⬆ 更精细地重画这张"),
                 ("🎲 New pictures", "🎲 换一批"), ("⭐ Favourite the picked one", "⭐ 收藏这张"), ("🔧 Open in Generate", "🔧 在高级页打开")):
    T(_en, _zh)

_P = '<p style="font-size:14px;margin:4px 0;line-height:1.6;color:{c};">{t}</p>'
OK, WARN, ERR, MUTED = "#a6e3a1", "#f9e2af", "#f38ba8", "#a6adc8"


def _p(en, zh, colour=MUTED, tag="p"):
    return _P.format(c=colour, t=L(en, zh))


def _e(s) -> str:
    return html.escape(str(s))


# ── data the plan needs (the app's own lists and readers) ─────────────────────────────────────────────────────
def _env(ns):
    from backend import character_cards as CC, recipes as RC, settings_info as SI, zh_tags as ZT
    from backend.sd_pipeline import _model_family
    from backend.model_manager import list_checkpoints, list_loras, list_vaes, lora_arch
    last = ns._read_last_session() if hasattr(ns, "_read_last_session") else {}
    ckpts = list_checkpoints()
    default = last.get("model") if isinstance(last.get("model"), str) and Path(last.get("model") or "").is_file() else (ckpts[0][1] if ckpts else "")

    def family(path):
        try:
            return _model_family(str(path)) if path else ""
        except Exception:
            return ""
    return SimpleNamespace(read_setting=lambda n: SI.read(ns.SETTINGS_DIR, ns._settings_name(n)) if ns._settings_name(n) else None,
                           load_card=CC.load_card, card_for_checkpoint=CC.card_for_checkpoint, card_prompt=CC.card_prompt,
                           checkpoints=ckpts, loras=list_loras(), vaes=list_vaes(), family=family, lora_arch=lora_arch, recipe_for=RC.recipe_for,
                           default_model=default, to_tags=ZT.to_tags,
                           defaults=dict(scheduler=last.get("scheduler") or "DPM++ 2M Karras", steps=ns.DEFAULT_STEPS, cfg=ns.DEFAULT_CFG))


def _times_file(ns):
    return ns.SETTINGS_DIR / SM.TIMES_FILE_NAME


def _read_last(ns) -> dict:
    try:
        d = json.loads((ns.SETTINGS_DIR / LAST_FILE_NAME).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_last(ns, c: dict) -> None:
    try:
        keep = {k: c[k] for k in ("source", "setting", "card", "outfit", "model", "variety", "framing", "detail", "shape", "colour", "look", "count")}
        (ns.SETTINGS_DIR / LAST_FILE_NAME).write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    except (OSError, KeyError):
        pass


def _choices(source, setting, card, outfit, model, what, variety, framing, detail, shape, colour, look, count, seed):
    return dict(source=source, setting=setting, card=card, outfit=outfit, model=model, what=what, variety=variety, framing=framing,
                detail=detail, shape=shape, colour=colour, look=look, count=count, seed=seed)


# ── the plan box ("what I'll do") ────────────────────────────────────────────────────────────────────────────
def plan_html(p: dict, measured: float | None = None) -> str:
    i = p["info"]
    if p["errors"]:
        return "".join(_p(en, zh, ERR) for en, zh in p["errors"]) + "".join(_p(en, zh, WARN) for en, zh in p["notes"])
    lora_en = "".join(f" + {_e(n)} {w:g}" for n, w in i["loras"])
    rows = []
    rows.append(L(f"<b>Model</b>: {_e(i['model'])}{lora_en}", f"<b>模型</b>：{_e(i['model'])}{lora_en}"))
    src_en, src_zh = i["source"]
    what = f" + <code>{_e(i['what'])}</code>" if i["what"] else ""
    rows.append(L(f"<b>Content</b>: {_e(src_en)}{what}", f"<b>内容</b>：{_e(src_zh)}{what}"))
    fr_en = {v: k for k, v in SM.FRAMINGS}[i["framing"]]
    eyes = i.get("eyes", True)          # False on a Pony checkpoint: Generate skips the eye pass
    if SM.PO.is_auto(i["choice"]):
        how_en = f"each picture gets the recipe for its own framing ({SM.recipe_words(i['picks'], False, eyes)})"
        how_zh = f"每张图按自己的取景选配方（{SM.recipe_words(i['picks'], True, eyes)}）"
    else:
        how_en, how_zh = SM.recipe_words(i["picks"], False, eyes), SM.recipe_words(i["picks"], True, eyes)
    rows.append(L(f"<b>Framing</b>: {_e(fr_en)} → {_e(how_en)}", f"<b>取景</b>：{SM.FRAMING_ZH[i['framing']]} → {_e(how_zh)}"))
    w, h = i["size"]
    seed_en = "a new seed each time" if i["seed"] < 0 else f"seed {i['seed']}"
    seed_zh = "每次换种子" if i["seed"] < 0 else f"种子 {i['seed']}"
    col = "" if i["colour"] in (None, 1.0) else f" · colour {i['colour']:g}"
    col_zh = "" if i["colour"] in (None, 1.0) else f" · 色彩 {i['colour']:g}"
    rows.append(L(f"<b>Size</b>: {w}×{h} · {i['count']} picture(s) · {seed_en}{col}", f"<b>尺寸</b>：{w}×{h} · {i['count']} 张 · {seed_zh}{col_zh}"))
    if i.get("sampler"):
        sch, st, cfg, pag = i["sampler"]
        look = i.get("look", "follow")
        look_en = {v: k for k, v in SM.LOOKS}.get(look, look)
        pag_en, pag_zh = (f"PAG {float(pag):g}", f"PAG {float(pag):g}") if pag else ("no PAG", "不开 PAG")
        rows.append(L(f"<b>Drawing</b>: {_e(look_en)} — {_e(sch)} · {st} steps · CFG {float(cfg):g} · {pag_en}",
                      f"<b>画风</b>：{SM.LOOK_ZH.get(look, look)} — {_e(sch)} · {st} 步 · CFG {float(cfg):g} · {pag_zh}"))
    if measured:
        t_en, t_zh = f"about {measured:.0f} s a picture here last time", f"上次在这台电脑上每张约 {measured:.0f} 秒"
    else:
        t_en, t_zh = f"about {i['factor']:g}× the time of a plain picture", f"约为普通出图的 {i['factor']:g} 倍"
    rows.append(L(f"<b>Time</b>: {t_en}", f"<b>用时</b>：{t_zh}"))
    rows.append(L(f"<b>Steps</b>: {SM.steps_text(i['picks'], False, i['passes'], eyes)}", f"<b>步骤</b>：{SM.steps_text(i['picks'], True, i['passes'], eyes)}"))
    head = _p("What I'll do", "我将要做", "#cba6f7")
    body = "".join(f'<li style="margin:2px 0;">{r}</li>' for r in rows)
    notes = "".join(_p(en, zh, WARN) for en, zh in p["notes"])
    return (f'<div style="border:1px solid #45475a;border-radius:10px;padding:8px 14px;background:#181825;">{head}'
            f'<ul style="margin:4px 0 4px 18px;padding:0;font-size:14px;line-height:1.6;color:#cdd6f4;">{body}</ul>{notes}</div>')


# ── what the after-picture check found, in words ─────────────────────────────────────────────────────────────
def _flag_words(flag: str) -> tuple[str, str]:
    f = str(flag)
    if f == "no face found":
        return "no face found", "没找到脸"
    if f.endswith(" faces"):
        return f"{f} in the picture", f"画里有 {f.split()[0]} 张脸"
    if f.endswith(" hands"):
        return f"{f} found", f"找到 {f.split()[0]} 只手"
    if f.startswith("not her?"):
        return "the face may not look like the character", "脸可能不像这个角色"
    if f.startswith("no "):
        return f"missing: {f[3:].rsplit(' ', 1)[0]}", f"缺少：{f[3:].rsplit(' ', 1)[0]}"
    return f, f


def check_html(scored: list, files: list) -> tuple[str, list[int]]:
    """Per picture: stars and what looks off; returns (html, the picture numbers that need a look)."""
    from backend.image_score import HAND_NOTE, SPECK_NOTE
    rows, bad = [], []
    for k, s in enumerate(scored, 1):
        en_bits, zh_bits = [], []
        for f in s.get("flags") or []:
            en, zh = _flag_words(f)
            en_bits.append(en); zh_bits.append(zh)
        for n in s.get("notes") or []:
            if n == HAND_NOTE:
                en_bits.append("a hand may be broken"); zh_bits.append("可能有一只手画坏了")
            elif n == SPECK_NOTE:
                en_bits.append("colour specks (lace or noise?)"); zh_bits.append("有彩色斑点（蕾丝还是噪点？）")
        stars = "⭐" * int(s.get("stars", 0))
        if s.get("flags") or HAND_NOTE in (s.get("notes") or []):      # the speck note fires on lace and fur too: shown, not counted as a problem
            bad.append(k)
        if en_bits:
            rows.append(L(f"Picture {k} {stars}: " + _e("; ".join(en_bits)), f"第 {k} 张 {stars}：" + _e("；".join(zh_bits))))
        else:
            rows.append(L(f"Picture {k} {stars}: looks fine", f"第 {k} 张 {stars}：看起来没问题"))
    body = "".join(f'<li style="margin:2px 0;">{r}</li>' for r in rows)
    return f'<ul style="margin:4px 0 4px 18px;padding:0;font-size:14px;line-height:1.6;color:#cdd6f4;">{body}</ul>', bad


def result_html(n: int, secs: float, check: str, bad: list[int], details: str) -> str:
    out = _p(f"Done: {n} picture(s) in {secs:.0f} s ({secs / max(n, 1):.0f} s each). They are saved with their settings, like every picture.",
             f"完成：{n} 张，用时 {secs:.0f} 秒（每张 {secs / max(n, 1):.0f} 秒）。图片和参数一起保存了，和其它图片一样。", OK)
    out += check
    if bad:
        nums = "、".join(str(k) for k in bad)
        out += _p(f"Next: click a picture that looks off (picture {nums.replace('、', ', ')}) and press ⬆ Redo the picked one in more detail, "
                  f"or 🔀 More like the picked one for small changes. For a broken hand, 🔧 Open in Generate → 🖌 Inpaint → 🖐 Re-draw hand.",
                  f"下一步：点一下有问题的图（第 {nums} 张），再按「⬆ 更精细地重画这张」，或者按「🔀 再来几张类似的」做小改动。"
                  f"手画坏了：按「🔧 在高级页打开」→ 局部重绘 → 🖐 重画手。", WARN)
    else:
        out += _p("Next: click the picture you like best, then 🔀 More like the picked one, ⬆ Redo the picked one in more detail, "
                  "or ⭐ to keep it in History's favourites.",
                  "下一步：点你最喜欢的那张，再按「🔀 再来几张类似的」「⬆ 更精细地重画这张」，或者按 ⭐ 收藏到历史记录里。")
    if details:
        out += (f'<details style="margin-top:6px;"><summary style="cursor:pointer;color:{MUTED};font-size:13px;">'
                f'{L("Technical details (from Generate)", "技术细节（来自生成页）")}</summary>{details}</details>')
    return out


# ── the tab ──────────────────────────────────────────────────────────────────────────────────────────────────
def build_tab(ns) -> dict:
    from backend.character_cards import list_cards
    from backend.model_manager import list_checkpoints
    from backend.wildcards import list_wildcards
    last = _read_last(ns)
    settings = ns._list_saved_settings()
    cards = list_cards()
    ckpts = [n for n, _ in list_checkpoints()]
    wcs = [w for w in list_wildcards() if SM.usable_wildcard(w)]
    get = lambda k, ok, d: last.get(k) if last.get(k) in ok else d
    src0 = get("source", SM._values(SM.SOURCES), "setting" if settings else ("card" if cards else "prompt"))
    card0 = get("card", cards, cards[0] if cards else "")
    with gr.Tab("🪄 Quick picture", id="simple"):
        gr.HTML(_p("Pick where the picture starts, say what to draw, press the button. The app uses its measured settings for the rest; "
                   "🎨 Generate (the next tab) has every control if you want to change one.",
                   "选好从哪开始、想画什么，按一下按钮就行。其余参数用 app 实测过的设置；想自己调某个参数，下一个标签页「🎨 生成」里全都有。"))
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=340):
                source = gr.Radio(SM.SOURCES, value=src0, label="Start from",
                                  info=T("A saved setting runs as you saved it; a character card brings its model, LoRAs and look; "
                                         "'just a prompt' uses the model's measured recipe.",
                                         "已保存的设置：按你保存的样子出图；角色卡：带上它的模型、LoRA 和长相；只写一句话：用这个模型实测过的配方。"))
                setting = gr.Dropdown(settings or [""], value=get("setting", settings, settings[0] if settings else ""), label="Saved setting",
                                      info=T("Every saved setting, by folder (adult ones too).", "所有已保存的设置，按文件夹分组（包括成人设置）。"),
                                      visible=src0 == "setting", filterable=True)
                with gr.Row(visible=src0 == "card") as card_row:
                    card = gr.Dropdown(cards or [""], value=card0, label="Character card", scale=3)
                    outs = _outfits(card0)
                    outfit = gr.Dropdown(outs, value=get("outfit", outs, SM.RANDOM_OUTFIT), label="Outfit", scale=3,
                                         info=T("Random = a different outfit of the card for every picture.", "随机 = 每张图换一套角色卡里的服装。"))
                model = gr.Dropdown([SM.FOLLOW] + ckpts, value=get("model", [SM.FOLLOW] + ckpts, SM.FOLLOW), label="Model",
                                    info=T("'(as saved)' = the setting's or the card's model; with 'just a prompt' the last model you used.",
                                           "“（按设置）”= 用设置或角色卡里的模型；“只写一句话”时用你上次用的模型。"), filterable=True)
                what = gr.Textbox(label="What to draw", lines=2, value="",
                                  placeholder=T("beach, looking back, smile", "海边，回头看，微笑"),
                                  info=T("Optional. Added to the setting's or card's prompt. Everyday Chinese words are turned into tags; "
                                         "words the models don't know are listed, never guessed.",
                                         "可以不填。会加在设置或角色卡的提示词后面。常用中文词会自动换成模型认识的标签；不认识的词会列出来，不会乱猜。"))
                variety = gr.Dropdown([SM.NO_WILDCARD] + wcs, value=get("variety", [SM.NO_WILDCARD] + wcs, SM.NO_WILDCARD),
                                      label="Add variety (a wildcard file)", filterable=True,
                                      info=T("Each picture takes a different line of the file you pick (your own files in models/wildcards too).",
                                             "每张图从选的文件里随机取一行（也包括 models/wildcards 里你自己的文件）。"))
                framing = gr.Radio(SM.FRAMINGS, value=get("framing", SM._values(SM.FRAMINGS), "auto"), label="Framing",
                                   info=T("Auto: each picture's own prompt decides. Another choice replaces the prompt's framing words and runs that framing's recipe.",
                                          "自动：按每张图自己的提示词决定。选别的：替换提示词里的取景词，并用这种取景的配方。"))
                detail = gr.Radio(SM.DETAILS, value=get("detail", SM._values(SM.DETAILS), "follow"), label="Detail",
                                  info=T("As saved: a saved setting's own passes (a card or a prompt: Standard). Fast: eyes only. Standard: the measured "
                                         "recipe for the framing. Best: hires and 2× detail where that measured better, 3-4× slower.",
                                         "按设置：用设置里自己的步骤（角色卡和只写一句话时等于标准）。快：只修眼睛。标准：这种取景实测过的配方。"
                                         "精细：在实测更好的地方加高清修复和 2 倍细节，慢 3–4 倍。"))
                # new users start on the softer lines (2026-11-03: the measured 12-step PAG recipe looked stiff to the user); a remembered choice stays
                look = gr.Radio(SM.LOOKS, value=get("look", SM._values(SM.LOOKS), "soft"), label="Drawing style",
                                info=T("As saved: the setting's or card's own sampler (the measured 12-step recipe with PAG 2: thick outlines, hard light). "
                                       "Softer lines: 28 steps, CFG 5, no PAG: thinner outlines and gentler shading, the character as recognisable as before. "
                                       "Painterly: Euler a, 28 steps: the softest, a little less like the character in the test. "
                                       "Illustrious / NoobAI models only; other models keep their settings.",
                                       "按设置：用设置或角色卡自己的采样（实测的 12 步 + PAG 2 配方：描边粗、明暗硬）。"
                                       "线条柔和：28 步、CFG 5、不开 PAG，描边更细、明暗更柔，角色和原来一样好认。"
                                       "手绘感：Euler a 28 步，最柔，测试里稍微没那么像角色。"
                                       "只对 Illustrious / NoobAI 模型生效，其他模型保持原设置。"))
                with gr.Row():
                    shape = gr.Radio(SM.SHAPES, value=get("shape", SM._values(SM.SHAPES), "follow"), label="Picture shape")
                    colour = gr.Radio(SM.COLOURS, value=get("colour", SM._values(SM.COLOURS), "follow"), label="Colour",
                                      info=T("Soft = paler colours, same drawing.", "柔和 = 颜色淡一些，画的内容不变。"))
                with gr.Row():
                    count = gr.Slider(1, 8, value=int(last.get("count", 2)) if str(last.get("count", "")).isdigit() else 2, step=1,
                                      label="How many pictures")
                    seed = gr.Number(value=-1, precision=0, label="Seed",
                                     info=T("-1 = new pictures every time; a number repeats the same picture.", "-1 = 每次都是新图；填一个数字 = 重画同一张。"))
                check = gr.Checkbox(value=True, label="Check every picture afterwards",
                                    info=T("Stars, faces and hands, with the models already on this PC (nothing downloads); 2-5 s a picture.",
                                           "打星、数脸、看手，只用电脑上已有的模型（不会下载）；每张多 2–5 秒。"))
                lang = gr.Textbox(value="zh", visible=False, label="lang", show_label=False)      # 'zh' / 'en': set by the page (js=) when the button is pressed
                with gr.Row():
                    go = gr.Button("✨ Make the pictures", variant="primary", size="lg", scale=3, elem_id="simple-go")
                    stop = gr.Button("⏹ Stop", size="lg", scale=1)
                refresh = gr.Button("🔄 Refresh the lists", size="sm")
            with gr.Column(scale=6, min_width=420):
                plan_box = gr.HTML("")
                gallery = gr.Gallery(label="Pictures", columns=2, height=640, object_fit="contain", preview=False, show_label=False)
                result = gr.HTML("")
                with gr.Row():
                    more = gr.Button("🔀 More like the picked one", size="sm")
                    finer = gr.Button("⬆ Redo the picked one in more detail", size="sm")
                    new = gr.Button("🎲 New pictures", size="sm")
                with gr.Row():
                    fav = gr.Button("⭐ Favourite the picked one", size="sm")
                    to_adv = gr.Button("🔧 Open in Generate", size="sm")
                follow = gr.HTML("")
        state = gr.State({})
        picked = gr.State(0)
    return dict(source=source, setting=setting, card_row=card_row, card=card, outfit=outfit, model=model, what=what, variety=variety,
                framing=framing, detail=detail, shape=shape, colour=colour, look=look, count=count, seed=seed, check=check, lang=lang, go=go,
                stop=stop, refresh=refresh, plan_box=plan_box, gallery=gallery, result=result, more=more, finer=finer, new=new, fav=fav,
                to_adv=to_adv, follow=follow, state=state, picked=picked)


def _outfits(card_name) -> list[str]:
    from backend.character_cards import load_card
    c = load_card(card_name) if card_name else None
    return [SM.RANDOM_OUTFIT, SM.NO_OUTFIT] + list((c or {}).get("outfits") or {})


def _variation(state: dict, i: int) -> tuple[int, float]:
    """(variation seed, strength) of picture i of the last run; (-1, 0.0) when it is not a variation (or an older state)."""
    v = (state or {}).get("variations") or []
    try:
        vs, st = v[i]
        return int(vs), float(st)
    except (IndexError, TypeError, ValueError):
        return -1, 0.0


def _gen_kwargs(ns, args: dict) -> dict:
    """The plan's arguments as do_generate_ui takes them: the extra settings made safe, unknown keys dropped."""
    fn = ns._GEN_HELPERS["generate_ui"]
    names = [n for n in inspect.signature(fn).parameters if n != "progress"]
    extra_keys = [k for k in names if k not in SM_BASE]
    ex = ns._clean_extra({k: args[k] for k in extra_keys if k in args})
    out = {k: args[k] for k in SM_BASE if k in args}
    out.update({k: ex[k] for k in extra_keys if k in ex})
    return {k: out[k] for k in names if k in out}


SM_BASE = ("model_path", "vae_path", "lora1", "w1", "lora2", "w2", "lora3", "w3", "auto_quality", "prompt", "neg_prompt", "scheduler",
           "steps", "cfg", "width", "height", "batch", "seed", "init_img", "strength", "use_i2i")


def run_plan(ns, p: dict, check: bool, progress, calls: list[dict] | None = None):
    """Run a plan through Generate (one call, or the `calls` overrides one after another); returns (pictures, result html, state)."""
    kw = _gen_kwargs(ns, p["args"])
    calls = calls or [{}]
    ns._SAVE_LOG = []
    t0 = time.time()
    imgs, seeds, variations, details, status = [], [], [], [], ""
    try:
        for ov in calls:
            call = dict(kw, **ov)
            out = ns._GEN_HELPERS["generate_ui"](**call, progress=progress)
            got, info, _last, status, sd_seeds, _loras = out
            details.append(info if isinstance(info, str) else "")
            if not got:
                break
            imgs += list(got)
            seeds += list(sd_seeds) if isinstance(sd_seeds, (list, tuple)) else []
            vs, st = int(call.get("var_seed", -1)), float(call.get("var_strength", 0) or 0)
            # a batch of one call draws var_seed, var_seed + 1, … (sd_pipeline.variation_latents); no variation: (-1, 0)
            variations += [((vs + k) % 2**32 if vs >= 0 and st > 0 else -1, st if vs >= 0 else 0.0) for k in range(len(got))]
            if ns._generation_abort.is_set():
                break
        files = [str(f) for f in (ns._SAVE_LOG or [])]
    finally:
        ns._SAVE_LOG = None
    secs = time.time() - t0
    state = dict(choices=p["choices"], seeds=seeds, variations=variations, files=files,
                 info={k: v for k, v in p["info"].items() if k != "picks"})
    if not imgs:
        if ns._generation_abort.is_set():
            return [], _p("⏹ Stopped before a picture was finished: nothing new was saved.", "⏹ 已停止：这次还没有画完的图，没有保存新图片。", WARN), state
        msg = "".join(details) or (status if isinstance(status, str) else "")
        return [], (_p("No picture was made. What Generate said:", "没有出图。生成页的提示：", ERR) + msg), state
    if not ns._generation_abort.is_set() and len(imgs) == sum(int(dict(kw, **ov).get("batch", 1)) for ov in calls):
        SM.record_time(_times_file(ns), p["info"], secs / len(imgs))
    chk, bad = "", []
    if check:
        try:
            from backend import image_score as IS
            card = None
            if p["choices"]["source"] == "card":
                from backend.character_cards import load_card
                card = load_card(p["choices"]["card"])
            scored = IS.score_images(imgs, (card or {}).get("tags") or p["args"]["prompt"], identity=(card or {}).get("identity"),
                                     pause=ns._score_pause(progress))
            chk, bad = check_html(scored, files)
        except Exception as e:
            print(f"[Quick picture] check failed: {e}")
            chk = _p(f"The check could not run ({e}).", f"检查没能运行（{e}）。", WARN)
    return imgs, result_html(len(imgs), secs, chk, bad, "".join(details)), state


def wire(ns, ui: dict, main_tabs) -> None:
    """Connect the tab's events (after the Generate tab is built)."""
    inputs = [ui[k] for k in ("source", "setting", "card", "outfit", "model", "what", "variety", "framing", "detail", "shape", "colour", "look",
                              "count", "seed")]

    def make_plan(*vals):
        return SM.plan(_choices(*vals), _env(ns))

    def on_preview(*vals):
        try:
            p = make_plan(*vals)
            return plan_html(p, SM.read_times(_times_file(ns)).get(SM.time_key(p["info"])))
        except Exception as e:
            return _p(f"Could not build the plan: {_e(e)}", f"没能生成计划：{_e(e)}", ERR)

    def do_simple_generate(source, setting, card, outfit, model, what, variety, framing, detail, shape, colour, look, count, seed, check, lang,
                           progress=gr.Progress()):
        """🪄 Quick picture: the plan from the plain choices, run through Generate. lang = 'zh' / 'en', the page's language (its js= sets it);
        the HTML answer carries both languages either way, so an HTTP client may pass either."""
        vals = (source, setting, card, outfit, model, what, variety, framing, detail, shape, colour, look, count, seed)
        p = make_plan(*vals)
        box = plan_html(p, SM.read_times(_times_file(ns)).get(SM.time_key(p["info"])))
        if p["errors"]:
            return gr.update(), box, gr.update(), box
        _write_last(ns, p["choices"])
        imgs, res, state = run_plan(ns, p, bool(check), progress)
        return imgs, res, state, box

    do_simple_generate._gpu_job = True

    def do_simple_follow(kind, state, idx, check, progress=gr.Progress()):
        """🔀 More like the picked one (same seed, small variations) / ⬆ Redo it in more detail (same seed, Best, one picture).
        A picked variation is redone as itself; 🔀 on a variation makes new variations of the same base picture (a variation of a variation
        is not something the pipeline can draw), which look alike at strength 0.1."""
        seeds = (state or {}).get("seeds") or []
        if not seeds:
            return gr.update(), _p("Make some pictures first.", "请先出几张图。", WARN), gr.update()
        i = idx if isinstance(idx, int) and 0 <= idx < len(seeds) else 0
        c = dict(state["choices"], seed=int(seeds[i]))
        vs, st = _variation(state, i)
        if kind == "finer":
            c.update(count=1, detail="best")
            p = SM.plan(c, _env(ns))
            calls = [dict(var_seed=vs, var_strength=st)] if st > 0 else [{}]     # the picked picture itself, variation included
        else:
            p = SM.plan(c, _env(ns))
            base = random.randint(0, 2**32 - 1 - 8)
            calls = [dict(batch=1, var_seed=base + k, var_strength=0.1) for k in range(int(c.get("count", 2)))]
        if p["errors"]:
            return gr.update(), "".join(_p(en, zh, ERR) for en, zh in p["errors"]), gr.update()
        imgs, res, new_state = run_plan(ns, p, bool(check), progress, calls)
        return imgs, res, new_state

    def do_simple_more(state, idx, check, progress=gr.Progress()):
        return do_simple_follow("more", state, idx, check, progress)

    def do_simple_finer(state, idx, check, progress=gr.Progress()):
        return do_simple_follow("finer", state, idx, check, progress)

    do_simple_more._gpu_job = do_simple_finer._gpu_job = True

    def on_source(s):
        return gr.update(visible=s == "setting"), gr.update(visible=s == "card")

    def on_card(name):
        outs = _outfits(name)
        return gr.update(choices=outs, value=SM.RANDOM_OUTFIT)

    def on_refresh(setting_now, card_now, model_now, variety_now):
        from backend.character_cards import list_cards
        from backend.model_manager import list_checkpoints
        from backend.wildcards import list_wildcards
        s, c = ns._list_saved_settings(), list_cards()
        m = [SM.FOLLOW] + [n for n, _ in list_checkpoints()]
        w = [SM.NO_WILDCARD] + [x for x in list_wildcards() if SM.usable_wildcard(x)]
        keep = lambda v, xs: v if v in xs else (xs[0] if xs else "")
        return (gr.update(choices=s or [""], value=keep(setting_now, s)), gr.update(choices=c or [""], value=keep(card_now, c)),
                gr.update(choices=m, value=keep(model_now, m)), gr.update(choices=w, value=keep(variety_now, w)))

    def on_pick(evt: gr.SelectData):
        return int(evt.index) if isinstance(evt.index, int) else 0

    def on_fav(state, idx):
        files = (state or {}).get("files") or []
        if not files:
            return _p("Make some pictures first.", "请先出几张图。", WARN)
        i = idx if isinstance(idx, int) and 0 <= idx < len(files) else 0
        from backend import history as H
        name = Path(files[i]).name
        if Path(files[i]).parent != Path(ns.OUTPUTS_DIR):
            return _p(f"Picture {i + 1} is in a sub-folder ({_e(Path(files[i]).parent.name)}), which History doesn't list.",
                      f"第 {i + 1} 张在子文件夹（{_e(Path(files[i]).parent.name)}）里，历史记录不列出子文件夹。", WARN)
        on = H.toggle_favourite(name)
        return _p(f"⭐ Picture {i + 1} is a favourite now." if on else f"Picture {i + 1} is no longer a favourite.",
                  f"⭐ 第 {i + 1} 张已收藏。" if on else f"第 {i + 1} 张已取消收藏。", OK)

    def on_to_advanced(state, idx):
        """🔧 Open in Generate: the Generate tab's controls take the plan of the last run with the picked picture's seed (img2img stays as it is)."""
        gen_inputs = ns._GEN_HELPERS["gen_inputs"]
        keep = [gr.update()] * len(gen_inputs)
        if not (state or {}).get("choices"):
            return keep + [_p("Make some pictures first.", "请先出几张图。", WARN)]
        seeds = state.get("seeds") or []
        c = dict(state["choices"])
        if seeds:
            c["seed"] = int(seeds[idx if isinstance(idx, int) and 0 <= idx < len(seeds) else 0])
        p = SM.plan(c, _env(ns))
        kw = _gen_kwargs(ns, p["args"])
        if seeds:
            vs, st = _variation(state, idx if isinstance(idx, int) and 0 <= idx < len(seeds) else 0)
            if st > 0:
                kw.update(var_seed=vs, var_strength=st)              # the picked picture is a variation: the Generate tab gets it too
        fn = ns._GEN_HELPERS["generate_ui"]
        names = [n for n in inspect.signature(fn).parameters if n != "progress"]
        vals = []
        for n in names[:len(gen_inputs)]:
            vals.append(gr.update() if n in ("init_img", "strength", "use_i2i") else kw.get(n, gr.update()))
        return vals + [_p("The Generate tab now holds this plan: change any control there and press Generate.",
                          "「🎨 生成」页已经填好这次的参数：在那里改任何一项，再按生成。", OK)]

    for comp in inputs:
        evt = comp.input if comp is ui["what"] else comp.change
        evt(on_preview, inputs, ui["plan_box"], api_name=False, show_progress="hidden")
    ui["source"].change(on_source, ui["source"], [ui["setting"], ui["card_row"]], api_name=False)
    ui["card"].change(on_card, ui["card"], ui["outfit"], api_name=False)
    ui["refresh"].click(on_refresh, [ui["setting"], ui["card"], ui["model"], ui["variety"]],
                        [ui["setting"], ui["card"], ui["model"], ui["variety"]], api_name=False)
    run_inputs = inputs + [ui["check"], ui["lang"]]
    # Gradio 4.19 hands the js the inputs followed by more values: keep the inputs, set the last one (the language)
    n_in = len(run_inputs)
    lang_js = f"(...a) => {{ const b = a.slice(0, {n_in}); b[{n_in - 1}] = (" + LANG_ARG_JS + ")(); return b; }"
    ui["go"].click(do_simple_generate, run_inputs, [ui["gallery"], ui["result"], ui["state"], ui["plan_box"]],
                   api_name="do_simple_generate", js=lang_js)
    ui["new"].click(lambda: -1, None, ui["seed"], api_name=False).then(
        do_simple_generate, run_inputs, [ui["gallery"], ui["result"], ui["state"], ui["plan_box"]], api_name=False, js=lang_js)
    ui["more"].click(do_simple_more, [ui["state"], ui["picked"], ui["check"]], [ui["gallery"], ui["result"], ui["state"]], api_name=False)
    ui["finer"].click(do_simple_finer, [ui["state"], ui["picked"], ui["check"]], [ui["gallery"], ui["result"], ui["state"]], api_name=False)
    ui["stop"].click(lambda: ns._GEN_HELPERS["stop"](), None, ui["follow"], api_name=False)
    ui["gallery"].select(on_pick, None, ui["picked"], api_name=False)
    ui["gallery"].change(lambda: 0, None, ui["picked"], api_name=False)
    ui["fav"].click(on_fav, [ui["state"], ui["picked"]], ui["follow"], api_name=False)
    gen_inputs = ns._GEN_HELPERS["gen_inputs"]
    ui["to_adv"].click(on_to_advanced, [ui["state"], ui["picked"]], gen_inputs + [ui["follow"]], api_name=False).then(
        lambda st: gr.Tabs(selected="generate") if (st or {}).get("choices") else gr.update(), ui["state"], main_tabs, api_name=False)
    ui["plan_box"].value = on_preview(*[c.value for c in inputs])
