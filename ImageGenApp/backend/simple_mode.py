"""simple_mode.py — 🪄 一键出图 (Quick picture): a few plain choices -> every argument of the Generate tab, using the measured defaults.

Nothing here invents a recipe. The choices map onto what was measured and shipped before:
  * where the picture starts: a saved setting (any group, adult ones included, exactly as saved), a character card with its profile for the
    chosen checkpoint (backend/character_cards), or just a prompt on a checkpoint with ⚡ the measured recipe of its family (backend/recipes);
  * framing + detail -> ✨ Polish (backend/polish): Auto picks per picture from that picture's own prompt; a fixed framing puts its tag in the
    prompt and runs its recipe; Fast = the eye pass only, Best = Auto — best (the slower recipe measured better for each framing);
  * shape -> the family's usual sizes, colour -> 🎨 colour strength (the user's own card measured best at 0.75-0.8).
The plan is a dict of `do_generate_ui`'s arguments, so the picture, its record, ⭐, History and restore work as for any other picture.
This module only reads: the settings / cards / file lists come in through `env` (a SimpleNamespace), which keeps it testable on the CPU.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from backend import polish as PO
from backend import wildcards as WC
from backend.prompt_tools import join_tags, parse_tag, split_tags

FOLLOW = "（按设置）as saved"        # a dropdown shows its value as it is (the page does not translate choice lists): both languages            # the model / shape / colour of the setting or card
RANDOM_OUTFIT = "（随机服装）random outfit"        # a dropdown shows its value as it is (the page does not translate choice lists): both languages
NO_OUTFIT = "（不加服装）no outfit"        # a dropdown shows its value as it is (the page does not translate choice lists): both languages
NO_WILDCARD = "（无）none"        # a dropdown shows its value as it is (the page does not translate choice lists): both languages

# (shown text, value): the shown English is translated by the page (backend/ui_zh.ZH), the value is what the API takes
SOURCES = [("Saved setting", "setting"), ("Character card", "card"), ("Just a prompt", "prompt")]
FRAMINGS = [("Auto", "auto"), ("Portrait", "portrait"), ("Cowboy shot (thighs up)", "cowboy"), ("Full body", "full"), ("Wide shot", "wide")]
# "As saved": a saved setting runs its own passes / Polish exactly as saved (a card or a plain prompt has none: Standard)
DETAILS = [("As saved", "follow"), ("Fast", "fast"), ("Standard", "standard"), ("Best (slower)", "best")]
SHAPES = [("As saved", "follow"), ("Tall", "tall"), ("Square", "square"), ("Wide", "wide")]
COLOURS = [("As saved", "follow"), ("Vivid", "vivid"), ("Natural", "natural"), ("Soft", "soft")]

FRAMING_TAG = {"portrait": "upper body", "cowboy": "cowboy shot", "full": "full body", "wide": "wide shot"}
FRAMING_ZH = {"auto": "自动", "portrait": "头像", "cowboy": "半身（到大腿）", "full": "全身", "wide": "远景"}
STANDARD = {"portrait": "Portrait", "cowboy": "Cowboy shot — hires", "full": "Full body", "wide": "Wide shot"}
# Fast: the plain picture + the eye pass (0.4, the measured default); no hires, no face / hand pass
FAST = dict(hires_on=False, fd_on=False, hd_on=False, ed_on=True, ed_denoise=0.4, md_on=False)
COLOUR_VALUE = {"vivid": 1.0, "natural": 0.85, "soft": 0.75}
SIZES = {"sdxl": {"tall": (832, 1216), "square": (1024, 1024), "wide": (1216, 832)},
         "sd15": {"tall": (512, 768), "square": (512, 512), "wide": (768, 512)}}
_SDXL_FAMILIES = ("sdxl", "pony", "illustrious")

RECIPE_ZH = {"Portrait": "眼睛精修", "Portrait — hires": "高清修复 + 眼睛", "Cowboy shot": "眼睛精修", "Cowboy shot — hires": "高清修复 + 眼睛",
             "Full body": "高清修复 + 眼睛", "Wide shot": "高清修复 + 脸 + 眼睛", "Full body — max detail": "高清修复 + 2 倍放大 + 眼睛",
             "Wide shot — max detail": "高清修复 + 脸 + 2 倍放大 + 眼睛", "Cowboy shot — hands": "手 + 眼睛", "Full body — hands": "高清修复 + 手 + 眼睛"}

_FRAME_RX = [rx for _, rx in PO._FRAMES]


def _values(pairs) -> list[str]:
    return [v for _, v in pairs]


def clean_choices(c: dict) -> dict:
    """The choices made safe (an HTTP client can send anything)."""
    pick = lambda v, pairs, d: v if v in _values(pairs) else d
    try:
        count = int(float(c.get("count", 1)))
    except (TypeError, ValueError):
        count = 1
    try:
        seed = int(float(c.get("seed", -1)))
    except (TypeError, ValueError, OverflowError):
        seed = -1
    return dict(source=pick(c.get("source"), SOURCES, "setting"), setting=str(c.get("setting") or "").strip(),
                card=str(c.get("card") or "").strip(), outfit=str(c.get("outfit") or RANDOM_OUTFIT),
                model=str(c.get("model") or FOLLOW), what=str(c.get("what") or "")[:2000],
                variety=str(c.get("variety") or NO_WILDCARD), framing=pick(c.get("framing"), FRAMINGS, "auto"),
                detail=pick(c.get("detail"), DETAILS, "follow"), shape=pick(c.get("shape"), SHAPES, "follow"),
                colour=pick(c.get("colour"), COLOURS, "follow"), count=min(8, max(1, count)),
                seed=seed if 0 <= seed < 2**32 else -1)


def _find(name, pairs) -> str | None:
    """A local file (path) by file name or stem, case ignored; pairs = [(file name, path)]."""
    if not name or name in ("none", FOLLOW):
        return None
    want = Path(str(name)).name.lower()
    stem = Path(want).stem
    return next((p for n, p in pairs if n.lower() == want), None) or next((p for n, p in pairs if Path(n).stem.lower() == stem), None)


def strip_framing(prompt: str) -> tuple[str, list[str]]:
    """The prompt without its framing tags (a chosen framing replaces them); returns (prompt, the removed tags). Wildcards stay."""
    keep, gone = [], []
    for t in split_tags(prompt or ""):
        core = parse_tag(t)[2] if t != "BREAK" else t
        core = re.sub(r"[()\[\]\\]", " ", core).strip()
        if t != "BREAK" and "__" not in t and "{" not in t and any(rx.fullmatch(core) for rx in _FRAME_RX):
            gone.append(t)
        else:
            keep.append(t)
    return join_tags(keep), gone


def _join(*parts: str) -> str:
    return ", ".join(p.strip().strip(",").strip() for p in parts if p and p.strip().strip(","))


def recipe_choice(framing: str, detail: str) -> str:
    """The ✨ Polish value of a plan: Auto / Auto — best / a fixed recipe / (off) for Fast."""
    if detail == "fast":
        return "(off)"
    if framing == "auto":
        return PO.AUTO_BEST if detail == "best" else "Auto"
    name = STANDARD[framing]
    return PO.best_of(name) if detail == "best" else name


def recipe_extra(framing: str, detail: str) -> dict:
    """The detail-pass settings the plan sets (Auto sets its own per picture inside Generate). The 2× stage (md_on) is switched off
    unless the recipe itself sets it: the Polish recipes leave md_on alone, so a saved setting's Max detail would otherwise ride along."""
    if detail == "fast":
        return dict(FAST)
    ch = recipe_choice(framing, detail)
    return dict(md_on=False) if PO.is_auto(ch) else dict(dict(md_on=False), **PO.recipe(ch))


# what the wildcard parser resolves as __name__ (backend/wildcards._WILD): a letter or digit first, no dots
_WILDCARD_NAME = re.compile(r"[A-Za-z0-9][\w\-/ ]*")


def usable_wildcard(name: str) -> bool:
    return bool(_WILDCARD_NAME.fullmatch(str(name or "")))


def _top_level_pipes(text: str) -> str:
    """An outfit's own '|' outside braces would split the random-outfit group: make it a comma ({red|blue} dress stays a group)."""
    out, depth = [], 0
    for ch in text:
        depth += ch == "{"
        depth -= ch == "}" and depth > 0
        out.append(", " if ch == "|" and depth == 0 else ch)
    return "".join(out)


def picks(prompt: str, choice: str, draws: int = 20) -> dict:
    """What Polish will run for this prompt: {recipe: share}; "(off)" = the passes the plan's own settings switch on ("passes").
    Auto on a wildcard prompt: drawn over seeds 0..draws-1."""
    if choice == "(off)":
        return {"passes": 1.0}
    if not PO.is_auto(choice):
        return {choice: 1.0}
    if not WC.is_dynamic(prompt):
        return {PO.auto_pick(choice, prompt): 1.0}
    out: dict = {}
    for s in range(draws):
        n = PO.auto_pick(choice, WC.resolve(prompt, s))
        out[n] = out.get(n, 0) + 1 / draws
    return out


def passes_factor(ex: dict) -> float:
    """A rough time factor of switched-on passes (the pieces of polish.TIME_FACTOR: hires ~+1.7, face ~+0.3, hands ~+0.7, 2x ~+1.5, eyes ~+0.3)."""
    return round(1.0 + 1.7 * bool(ex.get("hires_on")) + 0.3 * bool(ex.get("fd_on")) + 0.7 * bool(ex.get("hd_on"))
                 + 1.5 * bool(ex.get("md_on")) + 0.3 * bool(ex.get("ed_on")), 1)


def time_factor(pk: dict, ex: dict | None = None) -> float:
    """The plan's time against the plain picture (polish.TIME_FACTOR: the measured pieces added up)."""
    return sum((passes_factor(FAST if n == "fast" else ex or {}) if n in ("passes", "fast") else PO.time_factor(n)) * w for n, w in pk.items())


def _note(notes, en, zh):
    notes.append((en, zh))


def plan(choices: dict, env) -> dict:
    """{'args': do_generate_ui's keyword arguments, 'notes': [(en, zh)], 'errors': [(en, zh)], 'info': what the plan box shows}.

    env: read_setting(name) -> dict | None, load_card(name) -> dict | None, card_for_checkpoint(card, ckpt) -> (card, profile key),
    card_prompt(card, outfit, scene) -> str, checkpoints / loras / vaes = [(file name, path)], family(path) -> str,
    recipe_for(family) -> dict | None, default_model -> path or '', defaults = {scheduler, steps, cfg}, to_tags(text) -> (tags, unknown)."""
    c = clean_choices(choices)
    notes, errors = [], []
    a = dict(model_path="", vae_path="none", lora1="none", w1=0.8, lora2="none", w2=0.7, lora3="none", w3=0.7, auto_quality=True,
             prompt="", neg_prompt="", scheduler=env.defaults["scheduler"], steps=env.defaults["steps"], cfg=env.defaults["cfg"],
             width=0, height=0, batch=c["count"], seed=c["seed"], init_img=None, strength=0.75, use_i2i=False)
    extra: dict = {}
    source_label = ("", "")
    what, unknown = env.to_tags(c["what"]) if c["what"].strip() else ("", [])
    if unknown:
        _note(notes, "These words aren't known to the models and were left out: " + "、".join(unknown) + ". Write them in English or as Danbooru tags.",
              "这些词模型看不懂，已经略过：" + "、".join(unknown) + "。可以改用英文或 Danbooru 标签。")
    chosen_model = _find(c["model"], env.checkpoints) if c["model"] != FOLLOW else None
    if c["model"] != FOLLOW and not chosen_model:
        _note(errors, f"The model {c['model']} isn't installed.", f"找不到模型 {c['model']}。")

    if c["source"] == "setting":
        d = env.read_setting(c["setting"]) if c["setting"] else None
        if not d:
            _note(errors, "Pick a saved setting first." if not c["setting"] else f"Couldn't read the saved setting '{c['setting']}'.",
                  "请先选一个已保存的设置。" if not c["setting"] else f"读不出已保存的设置“{c['setting']}”。")
        else:
            source_label = (f"the saved setting {c['setting']}", f"已保存的设置 {c['setting']}")
            saved = _find(d.get("model"), env.checkpoints)
            if d.get("model") and not saved and not chosen_model:
                _note(errors, f"The setting's model {Path(str(d['model'])).name} isn't installed: pick a model.",
                      f"这个设置用的模型 {Path(str(d['model'])).name} 没有安装，请选一个模型。")
            a["model_path"] = chosen_model or saved or env.default_model
            if chosen_model and saved and env.family(chosen_model) != env.family(saved):
                _note(notes, "The chosen model is another family than the setting's: its LoRAs may not fit.",
                      "选的模型和设置原来的模型不是同一类，设置里的 LoRA 可能用不了。")
            slots = d.get("lora_slots")
            if not isinstance(slots, list) and isinstance(d.get("loras"), list):
                lw = d.get("lora_weights") if isinstance(d.get("lora_weights"), dict) else {}
                slots = [[n, next((v for k, v in lw.items() if Path(k).stem.lower() == Path(str(n)).stem.lower()), 0.8)]
                         for n in d["loras"] if isinstance(n, str)]
            missing = []
            for i, item in enumerate((slots or [])[:3]):
                n, w = (list(item) + [0.8])[:2] if isinstance(item, (list, tuple)) and item else ("none", 0.8)
                if not n or n == "none":
                    continue
                path = _find(n, env.loras)
                if path:
                    a[f"lora{i + 1}"], a[f"w{i + 1}"] = path, _w(w)
                else:
                    missing.append(Path(str(n)).stem)
            if missing:
                _note(notes, "Not installed, left out: LoRA " + ", ".join(missing) + ".", "没有安装，已略过：LoRA " + "、".join(missing) + "。")
            if d.get("vae"):
                a["vae_path"] = "none" if d["vae"] == "none" else (_find(d["vae"], env.vaes) or "none")
            a["prompt"] = str(d.get("prompt") or "")
            a["neg_prompt"] = str(d.get("negative_prompt") or "")
            for k, dk in (("scheduler", "scheduler"), ("steps", "steps"), ("cfg", "cfg_scale"), ("width", "width"), ("height", "height")):
                if d.get(dk) not in (None, ""):
                    a[k] = d[dk]
            if isinstance(d.get("auto_quality"), bool):
                a["auto_quality"] = d["auto_quality"]
            extra = dict(d["extra"]) if isinstance(d.get("extra"), dict) else {}
            a["prompt"] = _join(a["prompt"], what)

    elif c["source"] == "card":
        card = env.load_card(c["card"]) if c["card"] else None
        if not card:
            _note(errors, "Pick a character card first." if not c["card"] else f"Couldn't read the character card '{c['card']}'.",
                  "请先选一张角色卡。" if not c["card"] else f"读不出角色卡“{c['card']}”。")
        else:
            source_label = (f"the character card {card['name']}", f"角色卡 {card['name']}")
            card_ckpt = _find(card.get("checkpoint"), env.checkpoints)
            want = chosen_model or card_ckpt
            card, profile = env.card_for_checkpoint(card, want or card.get("checkpoint"))
            if profile:
                _note(notes, f"The card's settings for {Path(profile).stem} are used.", f"用角色卡为 {Path(profile).stem} 存的配置。")
            if not want:
                _note(errors, f"The card's model {card.get('checkpoint') or '(none)'} isn't installed: pick a model.",
                      f"角色卡用的模型 {card.get('checkpoint') or '（无）'} 没有安装，请选一个模型。")
            a["model_path"] = want or env.default_model
            if card.get("vae"):
                a["vae_path"] = _find(card["vae"], env.vaes) or "none"
            missing = []
            for i, l in enumerate((card.get("loras") or [])[:3]):
                path = _find(l.get("file"), env.loras)
                if path:
                    a[f"lora{i + 1}"], a[f"w{i + 1}"] = path, _w(l.get("weight", 0.8))
                else:
                    missing.append(Path(str(l.get("file"))).stem)
            if missing:
                _note(notes, "Not installed, left out: LoRA " + ", ".join(missing) + ".", "没有安装，已略过：LoRA " + "、".join(missing) + "。")
            outfits = card.get("outfits") or {}
            outfit = c["outfit"]
            vals = [v.strip() for v in outfits.values() if v.strip()]
            if outfit == RANDOM_OUTFIT and vals:
                # one outfit per picture: a wildcard group, picked from each picture's seed like any {a|b}; a single outfit goes in as it is
                # (a one-option {x} is not a wildcard: its braces would reach the model)
                group = vals[0] if len(vals) == 1 else "{" + "|".join(_top_level_pipes(v) for v in vals) + "}"
                a["prompt"] = _join(env.card_prompt(card, None, ""), group, what)
            else:
                a["prompt"] = env.card_prompt(card, outfit if outfit in outfits else None, what)
            a["neg_prompt"] = card.get("negative") or ""
            fam = env.family(a["model_path"]) if a["model_path"] else ""
            r = env.recipe_for(fam) or {}
            a["scheduler"] = card.get("scheduler") or r.get("scheduler") or a["scheduler"]
            a["steps"] = card.get("steps") or r.get("steps") or a["steps"]
            a["cfg"] = card.get("cfg") or r.get("cfg") or a["cfg"]
            a["width"], a["height"] = card.get("width") or 0, card.get("height") or 0
            extra = dict(pag_scale=card["pag"] if card.get("pag") is not None else r.get("pag", 0.0),
                         cfg_rescale=card.get("cfg_rescale") or 0.0, freeu=bool(card.get("freeu")),
                         clip_skip=card.get("clip_skip") or 1)
            for key, on, den in (("face_detail", "fd_on", "fd_denoise"), ("hand_detail", "hd_on", "hd_denoise"), ("eye_detail", "ed_on", "ed_denoise")):
                if card.get(key):
                    extra[on], extra[den] = True, card[key]
            if card.get("colour") is not None:
                extra["cs"] = card["colour"]

    else:
        source_label = ("your prompt", "你写的内容")
        a["model_path"] = chosen_model or env.default_model
        if not a["model_path"]:
            _note(errors, "Pick a model first.", "请先选一个模型。")
        if not what and c["variety"] == NO_WILDCARD:
            _note(errors, "Write what to draw first.", "请先写想画什么。")
        fam = env.family(a["model_path"]) if a["model_path"] else ""
        r = env.recipe_for(fam)
        if r:
            a.update(scheduler=r["scheduler"], steps=r["steps"], cfg=r["cfg"])
            extra["pag_scale"] = r["pag"]
        a["prompt"] = _join("1girl, solo" if what and not re.search(r"\b(\d\+?(girl|boy)s?|solo|no humans|multiple)\b", what) else "", what)

    if c["variety"] != NO_WILDCARD:
        name = c["variety"].strip()
        if usable_wildcard(name):
            a["prompt"] = _join(a["prompt"], f"__{name}__")
        else:
            _note(notes, f"The wildcard file '{name}' can't be used: a name must start with a letter or digit and hold no dots. It was left out.",
                  f"通配符文件“{name}”用不了：名字要以字母或数字开头，不能有点号。已略过。")

    # framing: a chosen framing replaces the prompt's own framing tags and runs its recipe
    if c["framing"] != "auto":
        a["prompt"], gone = strip_framing(a["prompt"])
        a["prompt"] = _join(a["prompt"], FRAMING_TAG[c["framing"]])
        if gone:
            _note(notes, "Framing tags replaced: " + ", ".join(gone) + f" → {FRAMING_TAG[c['framing']]}.",
                  "取景标签已替换：" + "、".join(gone) + f" → {FRAMING_TAG[c['framing']]}。")
    if c["detail"] == "follow" and c["source"] == "setting":
        choice = PO.clean_choice(extra.get("polish"))          # the setting's own Polish and passes, untouched
    else:
        detail = "standard" if c["detail"] == "follow" else c["detail"]
        choice = recipe_choice(c["framing"], detail)
        extra.update(recipe_extra(c["framing"], detail))
    extra["polish"] = choice
    extra.setdefault("var_seed", -1)
    extra["var_strength"] = 0.0

    fam = env.family(a["model_path"]) if a["model_path"] else ""
    sizes = SIZES["sdxl" if fam in _SDXL_FAMILIES else "sd15"]
    if c["shape"] != "follow" or not (a["width"] and a["height"]):
        a["width"], a["height"] = sizes["tall" if c["shape"] == "follow" else c["shape"]]
    if c["colour"] != "follow":
        extra["cs"] = COLOUR_VALUE[c["colour"]]
    a.update(extra)

    pk = {"fast": 1.0} if c["detail"] == "fast" else picks(a["prompt"], choice)
    info = dict(source=source_label, model=Path(str(a["model_path"])).stem if a["model_path"] else "",
                loras=[(Path(str(a[f"lora{i}"])).stem, a[f"w{i}"]) for i in (1, 2, 3) if a[f"lora{i}"] not in ("none", "", None)],
                what=what, choice=choice, picks=pk, factor=round(time_factor(pk, extra), 1), passes={k: extra.get(k) for k in FAST}, size=(a["width"], a["height"]), count=c["count"],
                seed=c["seed"], framing=c["framing"], detail=c["detail"], family=fam,
                colour=extra.get("cs"), dynamic=WC.is_dynamic(a["prompt"]))
    return dict(args=a, notes=notes, errors=errors, info=info, choices=c)


def _w(v) -> float:
    try:
        return min(1.5, max(0.1, float(v)))
    except (TypeError, ValueError):
        return 0.8


def steps_text(pk: dict, zh: bool, passes: dict | None = None) -> str:
    """The stages the picture goes through, in words (from the most likely recipe; "passes" = the plan's own switched-on passes)."""
    top = max(pk.items(), key=lambda kv: kv[1])[0] if pk else "passes"
    r = dict(FAST) if top == "fast" else dict(passes or {}) if top == "passes" else PO.recipe(top)
    stages = ["加载模型", "画底图"] if zh else ["load the model", "draw the picture"]
    if r.get("hires_on"):
        stages.append("高清修复" if zh else "hires fix")
    if r.get("hd_on"):
        stages.append("修手" if zh else "hands")
    if r.get("fd_on"):
        stages.append("修脸" if zh else "face")
    if r.get("md_on"):
        stages.append("2 倍放大" if zh else "2× upscale")
    if r.get("ed_on"):
        stages.append("修眼睛" if zh else "eyes")
    stages.append("保存" if zh else "save")
    return " → ".join(stages)


def recipe_words(pk: dict, zh: bool) -> str:
    """'Cowboy shot — hires 80 %, Full body 20 %' in words."""
    parts = []
    for n, w in sorted(pk.items(), key=lambda kv: -kv[1]):
        name = (("按设置里的步骤" if zh else "the passes as set") if n == "passes" else ("只修眼睛" if zh else "eyes only") if n == "fast"
                else (RECIPE_ZH.get(n, n) if zh else (PO.summary(n) or n)))
        parts.append(name + (f" {round(100 * w)} %" if len(pk) > 1 else ""))
    return ("；" if zh else "; ").join(parts)


# seconds per picture measured on this PC, per plan (model, size, recipe): shown in the plan box the next time
TIMES_FILE_NAME = "_simple_times.json"


def time_key(info: dict) -> str:
    return f"{info['model']}|{info['size'][0]}x{info['size'][1]}|{info['choice']}|{info['detail']}"


def _raw_times(path) -> dict:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def read_times(path) -> dict:
    """{plan key: seconds per picture} = the median of the last runs (a run that loaded the model is slower; the median outvotes it)."""
    out = {}
    for k, v in _raw_times(path).items():
        xs = sorted(float(x) for x in (v if isinstance(v, list) else [v]) if isinstance(x, (int, float)))
        if xs:
            out[k] = xs[(len(xs) - 1) // 2]           # the lower middle: with [warm, cold] the warm run
    return out


def record_time(path, info: dict, secs_per_picture: float) -> None:
    d = _raw_times(path)
    old = d.get(time_key(info))
    xs = (old if isinstance(old, list) else [old] if isinstance(old, (int, float)) else []) + [round(float(secs_per_picture), 1)]
    d[time_key(info)] = xs[-5:]
    try:
        Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
