"""
help_content.py — Bilingual (EN/中文) Help tab for ImageGen Studio.

All Chinese translations use terminology established by the Chinese SD/AI community
(Bilibili, Civitai CN, LiblibAI, etc.) — no runtime LLM required.
"""

# ── Shared CSS injected once ─────────────────────────────────────────────────
_CSS = """
<style>
.hg-wrap { font-family: "Segoe UI", sans-serif; color: #cdd6f4; line-height: 1.6; }
.hg-h2  { font-size:17px; font-weight:700; color:#cba6f7; margin:16px 0 8px; border-bottom:1px solid #313244; padding-bottom:4px; }
.hg-h3  { font-size:14px; font-weight:700; color:#89b4fa; margin:12px 0 4px; }
.hg-table { width:100%; border-collapse:collapse; margin:6px 0 14px; font-size:13px; }
.hg-table th { background:#313244; color:#cba6f7; padding:6px 10px; text-align:left; font-weight:600; }
.hg-table td { padding:5px 10px; border-bottom:1px solid #1e1e2e; vertical-align:top; }
.hg-table tr:hover td { background:#1e1e2e; }
.hg-code { font-family: "Cascadia Code","Consolas",monospace; background:#181825; color:#a6e3a1;
           padding:2px 6px; border-radius:4px; font-size:13px; }
.hg-tag  { display:inline-block; background:#313244; color:#f5c2e7; padding:1px 7px;
           border-radius:9px; font-size:13px; margin:1px 2px; }
.hg-note { background:#1e1e2e; border-left:3px solid #fab387; padding:8px 12px;
           margin:8px 0; border-radius:0 6px 6px 0; font-size:13px; color:#a6adc8; }
.hg-tip  { background:#1e1e2e; border-left:3px solid #a6e3a1; padding:8px 12px;
           margin:8px 0; border-radius:0 6px 6px 0; font-size:13px; color:#a6adc8; }
.cn { color:#89dceb; }
</style>
"""

def _wrap(html: str) -> str:
    return _CSS + f'<div class="hg-wrap">{html}</div>'

def _table(headers: list[str], rows: list[list[str]]) -> str:
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for row in rows:
        tds = "".join(f"<td>{c}</td>" for c in row)
        trs += f"<tr>{tds}</tr>"
    return f'<table class="hg-table"><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>'

def _code(s: str) -> str:
    return f'<span class="hg-code">{s}</span>'

def _cn(s: str) -> str:
    return f'<span class="cn">{s}</span>'

def _note(s: str) -> str:
    return f'<div class="hg-note">{s}</div>'

def _tip(s: str) -> str:
    return f'<div class="hg-tip">💡 {s}</div>'


# ═════════════════════════════════════════════════════════════════════════════
# Section 1 — Prompt Syntax
# ═════════════════════════════════════════════════════════════════════════════
def quickstart_html() -> str:
    h = '<div class="hg-h2">🚀 Quick Start &nbsp;<span class="cn">快速上手</span></div>'
    h += _table(
        ["Step / 步骤", "What to do / 操作", _cn("说明")],
        [
            ["<b>1. Pick a checkpoint</b>",
             "🎨 Generate → <b>Checkpoint</b>. SD 1.5 models suit 512×768; SDXL / Pony / Illustrious suit "
             "832×1216 — the size switches automatically when you change family.",
             _cn("在“生成”页选择模型。切换 SD1.5 / SDXL 时分辨率会自动调整。")],
            ["<b>2. Write a prompt</b>",
             "Most important tags first. Leave <b>Auto-add quality tags</b> on unless you write your own.",
             _cn("重要的标签放前面。除非自己写质量标签，否则保持“自动添加质量标签”开启。")],
            ["<b>3. Generate</b>",
             "Click <b>✨ Generate</b> or press <b>Ctrl+Enter</b>. The model loads by itself; the first "
             "image of a session is slower (~1 min) while the GPU kernels load.",
             _cn("点击生成或按 Ctrl+Enter。模型会自动加载；每次启动后的第一张图较慢（约1分钟）。")],
            ["<b>4. Add LoRAs</b>",
             "Pick them in the 3 LoRA slots — they are applied on Generate. A LoRA for the wrong model "
             "family (SD 1.5 vs SDXL) is refused with a clear message.",
             _cn("在3个LoRA槽位中选择即可，生成时自动应用。与模型不匹配的LoRA会被拒绝并提示原因。")],
            ["<b>5. Keep what you like</b>",
             "Images are saved to <code>outputs/</code> with all settings inside (📂 Show in folder highlights the selected image in Explorer). "
             "♻️ Last seed reuses it; 📄 PNG Info reads any saved image (also A1111/Civitai ones) back into Generate, LoRAs included.",
             _cn("图片连同参数保存在 outputs/（📂 在文件夹中显示：在资源管理器中选中当前图片）。♻️ 复用种子；“PNG信息”页可把旧图（包括 A1111/Civitai 图片）的参数连同 LoRA 导回生成页。")],
            ["<b>6. Refine</b>",
             "🖼 Send to img2img (denoise 0.4–0.6) or 🔍 Send to Upscale.",
             _cn("发送到图生图（重绘幅度0.4–0.6）或发送到放大。")],
        ]
    )
    h += _table(
        ["If… / 如果…", "Then / 那么", _cn("说明")],
        [
            ["Info line shows <b>⚠ VRAM over-committed</b>",
             "The GPU ran out of memory and Windows moved data to system RAM (very slow). "
             "Lower the size or batch.",
             _cn("显存不足，Windows 改用系统内存（非常慢）。请降低分辨率或批量。")],
            ["Images are black or noisy",
             "Check the LoRA matches the model; then run <code>run_zluda.bat selftest_zluda.py</code> "
             "in the ImageGenApp folder.",
             _cn("先确认LoRA与模型匹配；再在 ImageGenApp 目录运行 run_zluda.bat selftest_zluda.py。")],
            ["You restart the app",
             "Your last prompt, model, LoRA slots and settings come back automatically (the Bridge tab too).",
             _cn("上次的提示词、模型、LoRA 和参数会自动恢复（桥接页同样）。")],
            ["A model / LoRA says <b>incomplete</b> or <b>isn't a valid .safetensors file</b>",
             "The file was cut short or damaged (interrupted download or copy). Download it again.",
             _cn("文件不完整或已损坏（下载或复制被中断），请重新下载。")],
            ["A Civitai download says <b>interrupted</b>",
             "Press Download again — it continues where it stopped instead of starting over.",
             _cn("再次点击下载即可从中断处继续，无需从头开始。")],
            ["You pasted a prompt with <code>&lt;lora:…&gt;</code> tags",
             "They move into free LoRA slots by themselves; LoRAs you don't have are listed.",
             _cn("标签会自动移入空闲 LoRA 槽位；本地没有的 LoRA 会被列出。")],
            ["You launch the app a second time",
             "It just reopens the running one in your browser.",
             _cn("会直接在浏览器中打开已在运行的实例。")],
        ]
    )
    return _wrap(h)


def prompt_syntax_html() -> str:
    h = '<div class="hg-h2">📝 Prompt Syntax &nbsp;<span class="cn">提示词语法</span></div>'

    h += '<div class="hg-h3">Basics &nbsp;<span class="cn">基础规则</span></div>'
    h += _table(
        ["Rule / 规则", "English", _cn("中文说明")],
        [
            ["Comma-separated / 逗号分隔",
             "Tags separated by <code>,</code>. Earlier tags = higher weight.",
             _cn("用英文逗号分隔标签。靠前的标签权重更高，越靠后影响越弱。")],
            ["Word order matters / 词序影响权重",
             "Put the most important concepts first.",
             _cn("最重要的词放最前面，例如先写风格，再写角色，最后写细节。")],
            ["Language / 语言",
             "English tags give best results on most models. Chinese characters are largely ignored.",
             _cn("绝大多数模型只识别英文标签，中文字符通常被忽略或效果极差。")],
            ["Case / 大小写",
             "Lowercase preferred. <code>1girl</code> not <code>1Girl</code>.",
             _cn("统一使用小写字母，例如写 1girl 而非 1Girl。")],
        ]
    )

    h += '<div class="hg-h3">Weight Syntax &nbsp;<span class="cn">权重语法</span></div>'
    h += _table(
        ["Syntax / 语法", "Multiplier / 倍率", "Example / 示例", _cn("中文说明")],
        [
            [_code("(tag:1.3)"), "×1.3", _code("(smile:1.3)"),    _cn("精确指定权重，推荐范围 0.5–1.5")],
            [_code("(tag)"),     "×1.1", _code("(masterpiece)"),   _cn("圆括号每层 ×1.1 倍")],
            [_code("((tag))"),   "×1.21", _code("((blush))"),      _cn("双括号 ×1.1² ≈ ×1.21")],
            [_code("[tag]"),     "×0.9", _code("[blurry]"),        _cn("方括号降低权重 ×0.9，负面词常用")],
            [_code("[[tag]]"),   "×0.81", _code("[[ugly]]"),       _cn("双方括号 ×0.9² ≈ ×0.81")],
            [_code("(tag:0.5)"), "×0.5", _code("(text:0.5)"),     _cn("权重低于 1.0 表示减弱影响")],
        ]
    )
    h += _note("⚠️ Weights above 1.5 or below 0.3 often cause artifacts. Keep it between 0.5–1.4 for clean results. &nbsp;"
               + _cn("权重超过 1.5 或低于 0.3 容易产生图像异常，建议控制在 0.5–1.4 之间。"))

    h += '<div class="hg-h3">Long Prompts, BREAK & Tokens &nbsp;<span class="cn">长提示词、BREAK 与 token</span></div>'
    h += _table(
        ["Keyword / 功能", "Effect / 效果", _cn("中文说明")],
        [
            [_code("BREAK"),
             "Starts a new 75-token chunk: the tags after it are encoded separately.",
             _cn("开始新的75-token段：其后的标签单独编码。")],
            ["📏 token counter",
             "Under the prompts: tokens used, how many chunks, and where each chunk starts. "
             "Nothing is cut off — every 75 tokens become one more chunk — but tags in the "
             "<b>first</b> chunk steer the image most. A tag is never split between chunks.",
             _cn("提示词下方显示 token 数、分段数及每段起点。不会截断，但第一段影响最大；标签不会被拆开。")],
            ["⚠ warnings",
             "A LoRA trigger word past the first 75 tokens, or a tag in both the prompt and the negative prompt.",
             _cn("LoRA 触发词不在第一段，或同一标签同时出现在正负提示词中时提示。")],
            ["🧹 Tidy prompts",
             "Merges duplicate tags (masterpiece / (masterpiece:1.3) / masterpiece++) and keeps the strongest weight.",
             _cn("合并重复标签，保留最高权重。")],
            ["Presets & tag chips",
             "Adding a preset or quick tag never duplicates a tag: an existing one keeps its place and the stronger weight wins.",
             _cn("叠加预设或快捷标签时不会重复，同名标签保留更高权重。")],
            [_code("AND"), _code("[a:b:0.5]"), _cn("不支持：A1111 的 AND、ADDBASE/ADDCOMM、[a:b:0.5] 调度和 [a|b] 轮换在本程序中会被当作普通文字或降权处理。")],
        ]
    )
    h += _note("Not supported here (they are A1111 WebUI features): <b>AND</b>, <b>ADDBASE / ADDCOMM</b>, "
               "prompt scheduling <code>[a:b:0.5]</code> and alternation <code>[a|b]</code> — square brackets "
               "only mean \"weaker\" (×0.91). "
               + _cn("本程序不支持 AND、ADDBASE/ADDCOMM、[a:b:0.5] 调度与 [a|b] 轮换；方括号只表示降低权重（×0.91）。"))

    h += '<div class="hg-h3">LoRA Keywords &nbsp;<span class="cn">LoRA 关键词</span></div>'
    h += _tip("Pick a LoRA and its keywords appear as chips under the LoRA slots — click one to add it. "
              "🔑 = trigger word from Civitai, 🗝 = likely trigger (a name-like tag in ≥90 % of its training images — "
              "sometimes the real spelling differs from Civitai's), 📋 = the creator's full prompt (e.g. one outfit), "
              "% = share of training images with that tag: the higher, the more the LoRA ties it to its subject. "
              "<b>✨ Add character tags</b> adds the triggers plus every tag in at least half of the training images, "
              "right after your quality tags. "
              + _cn("选择 LoRA 后，关键词以按钮形式出现，点击即可加入提示词。🔑 触发词，🗝 可能的触发词，📋 作者的完整提示词，"
                    "% 为训练图片中含该标签的比例。✨ 一键加入触发词及出现率≥50%的标签。"))

    h += '<div class="hg-h3">Reproducing an Image &nbsp;<span class="cn">复现图片</span></div>'
    h += _tip("Every saved PNG records the checkpoint, VAE, LoRAs and their weights, prompts, sampler, steps, CFG, "
              "size and seed (A1111-compatible text + an exact ImageGen Studio record). Drop it into "
              "<b>Image-to-Image</b> and everything comes back; untick “Use img2img mode” and Generate to make the "
              "same image again (the GPU isn't bit-exact, so expect differences under ~2/255 per pixel). "
              + _cn("每张图片都记录了模型、VAE、LoRA 及权重、提示词、采样器、步数、CFG、尺寸和种子。拖入图生图即可全部恢复；"
                    "取消勾选图生图模式再生成即可复现原图（GPU 非逐位一致，像素差约 2/255 以内）。"))
    return _wrap(h)


# ═════════════════════════════════════════════════════════════════════════════
# Section 2 — Generation Parameters
# ═════════════════════════════════════════════════════════════════════════════
def parameters_html() -> str:
    h = '<div class="hg-h2">⚙️ Generation Parameters &nbsp;<span class="cn">生成参数</span></div>'

    h += _table(
        ["Parameter / 参数", "Recommended / 推荐值", "Effect / 效果", _cn("中文说明")],
        [
            ["<b>CFG Scale</b><br><small>提示词相关性 / 引导强度</small>",
             "7 – 9",
             "How literally the model follows your prompt. Low = creative, high = rigid.",
             _cn("数值越高越严格遵循提示词，越低模型越自由发挥。7–9 是最常用的平衡点，超过 15 通常过饱和。")],

            ["<b>Steps</b><br><small>采样步数</small>",
             "20 – 30",
             "Denoising iterations. More steps = finer details, diminishing returns past 30–40.",
             _cn("去噪迭代次数。20步已是较好质量，超过40步收益明显递减。DPM++ 系列采样器20步即可出好图。")],

            ["<b>Seed</b><br><small>随机种子</small>",
             "-1 (random)",
             "-1 = random new image every time. The seed actually used is saved in every PNG "
             "(batch images get seed, seed+1, …). Press ♻️ Last seed to lock it in, then tweak the prompt.",
             _cn("-1 表示每次随机生成。每张PNG都会保存实际使用的种子（批量时依次为 种子、种子+1…）。"
                 "点 ♻️ 复用上次种子即可锁定，再微调提示词。")],

            ["<b>Width / Height</b><br><small>宽度 / 高度</small>",
             "SD1.5: 512×512<br>SDXL: 1024×1024",
             "Must be multiples of 64. Wrong resolution for model causes warped anatomy.",
             _cn("必须是64的倍数。SD1.5在512分辨率训练，SDXL在1024训练。用错尺寸会导致人物变形或出现多头。")],

            ["<b>Batch Size</b><br><small>批次大小</small>",
             "1 – 4",
             "Images generated in parallel. Each image adds ~2 GB VRAM.",
             _cn("同时生成的图片数量。每张额外图片约多占2GB显存，显存不足时降低此值。")],

            ["<b>Denoise Strength</b><br><small>去噪强度（img2img）</small>",
             "0.5 – 0.75",
             "img2img only. 0.3 = subtle edit, 0.75 = major repaint, 1.0 = ignores input.",
             _cn("仅img2img模式。0.3接近原图微调，0.75大幅重绘，1.0完全忽略输入图。")],
        ]
    )

    h += '<div class="hg-h3">Schedulers (Samplers) &nbsp;<span class="cn">采样器</span></div>'
    h += _table(
        ["Scheduler / 采样器", "Speed / 速度", "Quality / 质量", _cn("适用场景")],
        [
            [_code("DPM++ 2M Karras"),     "⚡⚡⚡", "★★★★★", _cn("最推荐，20–30步即可出高质量图，通用首选。")],
            [_code("DPM++ 2M AYS"),        "⚡⚡⚡⚡", "★★★★★", _cn("AYS 采样计划：10–12 步约等于常规 25 步。")],
            [_code("DPM++ 2M"),            "⚡⚡⚡", "★★★★",  _cn("不带 Karras 的原版，旧记录（格式1）的“Karras”即此。")],
            [_code("DPM++ 2M SDE Karras"), "⚡⚡",   "★★★★★", _cn("随机性更强、细节更丰富。")],
            [_code("DPM++ SDE Karras"),    "⚡",     "★★★★★", _cn("每步两次计算，质量极高但慢一倍。")],
            [_code("DPM++ SDE"),           "⚡",     "★★★★",  _cn("同上，不带 Karras。")],
            [_code("Euler a"),             "⚡⚡⚡", "★★★★",  _cn("多样性好，每步加噪，适合探索风格。")],
            [_code("Euler"),               "⚡⚡⚡", "★★★★",  _cn("稳定，收敛快，适合固定构图的复现。")],
            [_code("Euler AYS"),           "⚡⚡⚡⚡", "★★★★",  _cn("Euler + AYS 采样计划，少步数。")],
            [_code("UniPC"),               "⚡⚡⚡", "★★★★",  _cn("快速收敛，少步（10步）效果好。")],
            [_code("Heun"),                "⚡",     "★★★★",  _cn("每步两次计算，慢但精确。")],
            [_code("DDIM"),                "⚡⚡",   "★★★",   _cn("经典算法，img2img效果稳定，已被DPM系列超越。")],
            [_code("PNDM"),                "⚡⚡",   "★★★",   _cn("即 A1111 的 PLMS，老牌算法，已基本被DPM替代。")],
            [_code("LMS"),                 "⚡⚡",   "★★★",   _cn("线性多步采样，质量一般，已较少使用。")],
        ]
    )
    h += _note("💡 Start with <b>DPM++ 2M Karras</b> at 20 steps. Only change if you need specific characteristics. "
               + _cn("新手直接用 DPM++ 2M Karras + 20步，这是最稳定的组合。"))
    h += '<div class="hg-h3">Hires Fix, Variations, CLIP Skip & X/Y Grid &nbsp;<span class="cn">高清修复、变体、CLIP跳层与对比网格</span></div>'
    h += _table(
        ["Setting / 设置", "What it does / 作用", _cn("中文说明")],
        [
            ["🔍 Hires fix",
             "Generates at the model's native size (good composition, no doubled bodies), upscales ×1.5–2, then "
             "re-draws details with img2img at the big size (denoise 0.35–0.5). SD 1.5 512×768 → 768×1152 in ~30 s; "
             "SDXL 832×1216 → 1248×1824 in ~2 min (fits 12 GB).",
             _cn("先按原生尺寸构图，放大后再以图生图重绘细节（降噪0.35–0.5）。")],
            ["🔀 Variations",
             "Keeps the main seed and blends in a second seed's noise: 0.05–0.1 = a close relative (pose and details shift, the "
             "character stays), 0.25+ = mostly a new picture. "
             "“🔀 More like this” under the gallery sets it up for the image you picked.",
             _cn("保持主种子并混入变体种子：0.05–0.1 为相近变体（姿势与细节会变，角色不变），0.25 以上基本是新图。图库下“More like this”一键设置。")],
            ["✂️ CLIP skip",
             "2 = use the text encoder's second-to-last layer — what most SD 1.5 anime checkpoints were trained "
             "with (A1111 “Clip skip: 2”). SDXL always does this already.",
             _cn("2 = 使用文本编码器倒数第二层，多数 SD1.5 动漫模型推荐。SDXL 不受影响。")],
            ["✨ Face detail",
             "Finds faces and re-draws each one at the model's native size — fixes small or messy faces. Measured: "
             "no gain once the face is ≥ ~25 % of the picture width, a clear one below ~20 % (denoise 0.35), and for "
             "small faces 0.55 beats 0.35 (+0.2 to +0.5 identity at 60–130 px, and the hair pin comes back more often; a wide "
             "shot, ~6 %, gains most), so faces ≤ ~100 px get at least 0.55 automatically, tiny ones ≤ ~60 px 0.65 (blended down to the slider's value "
             "at ~150 px; the slider is the value for big faces) — 0.8 can turn the face into somebody else; hires "
             "fix 1.5× + face detail was best for the smallest faces. The pass uses your whole prompt for every face, so with two "
             "character LoRAs in one picture it cannot keep the characters apart. Works after hires fix and in "
             "img2img too.",
             _cn("检测人脸并以模型原生分辨率重绘，修复小脸/乱脸。实测：脸宽 ≥ 画面约 25% 时无明显收益，低于约 20% 明显改善（降噪 0.35），"
                 "小脸用 0.55 比 0.35 更好（60–130 像素处身份相似度 +0.2～+0.5，发饰也更常出现，远景约 6% 收益最大），≤ 约 100 像素的脸会自动提高到至少 0.55，≤ 约 60 像素的提高到 0.65（约 150 像素以上用滑块值；0.8 可能把脸画成别人），最小的脸配合高清修复 1.5× 最佳；每张脸都用整段提示词，"
                 "含两个角色 LoRA 的图无法区分角色。")],
            ["✋ Hand detail",
             "Same idea for hands (anime hand detector, gloves count), run before the faces: 0.35 cleans up "
             "smudged fingers. Measured on 11 genuinely broken hands (10 on SD 1.5, 1 on SDXL): 0.35–0.55 turned blobs and mitten fists into hand-shaped fists in about "
             "half of them and **never fixed a finger count or an untangled pair** (a peace sign with three fingers stayed at three in all six passes), and at 0.45–0.55 it once drew a "
             "small figure into the hand's box; a hand-only prompt, more context and hires fix first changed little (hires first fixed 1 of 10). Re-drawing the hand at 0.8 "
             "gave a correct hand in 2 of 10 (plus 4 plausible ones with another glove, prop or pose); at 1.0 it invents objects. For a broken hand: another seed first, or 🖌 Inpaint "
             "over the hand at ~0.8 with 'detailed hands, five fingers' and a few seeds. The pass may touch up hand-shaped things (a ship's turret) — at low denoise they stay what they were. Face and "
             "hand passes use the evenly spaced version of your sampler (Karras / AYS at the same denoise "
             "changed almost nothing).",
             _cn("对手部做同样的重绘（动漫手部检测，手套也算），在脸部之前进行：0.35 清理手指。实测 11 只真正画坏的手：0.35–0.55 约一半能把团块/连指拳变成有手形的拳头，"
                 "但从未修正手指数量或缠在一起的双手；0.8 重画 10 只中有 2 只得到正确的手（另有 4 只合理但手套/道具/姿势变了），1.0 会凭空画出物品。"
                 "手画坏了：先换种子，或用 🖌 Inpaint 在约 0.8 下涂抹手部并多试几个种子。脸/手重绘使用均匀步长版本的采样器。")],
            ["👁 Eye detail",
             "Finds the eyes (anime eye detector, on each face) and re-draws both eyes of a face in one pass at high "
             "resolution, after the face pass, with a prompt of only the eye tags (colour, gaze, expression, glasses…) plus “detailed eyes, "
             "detailed pupils, iris detail”: 0.3 tidies a malformed eye (one under bangs), the default 0.4 adds iris / pupil structure "
             "(Laplacian energy of the eyes +17 % over the old pass, no colour bleed; closed or half-closed eyes, glasses and bangs stay as they were), "
             "0.5+ redraws the eye and can add stray marks. At ~45 px per eye (832×1216) it is a cleanup — pupil and iris rings need pixels: "
             "after hires fix or a 2× upscale the same pass draws them (about +12 % more detail again). "
             "The prompt also says “round pupils” and the negative “slit pupils, cat eyes” — without them “detailed pupils” drew bold "
             "outlined slit pupils on some checkpoints; if your own prompt names a pupil shape (slit pupils, heart-shaped pupils, @_@ …) "
             "the pass leaves the shape to it. Pony models come out softer after any eye pass (about −27 % sharpness at 0.25–0.4): leave it off there. "
             "A colour guard keeps the new colour inside the iris: bangs over an eye, lids and skin keep their own "
             "colour and only get sharper, so a red iris can't tint the hair.",
             _cn("检测眼睛并在脸部重绘之后以高分辨率重绘双眼，提示词只含眼睛相关标签加“detailed eyes, detailed pupils, iris detail”：0.3 修正变形的眼睛，默认 0.4 增加虹膜/瞳孔结构"
                 "（眼部细节能量比旧方法 +17%，无串色；闭眼/半闭眼、眼镜、刘海保持不变），0.5 以上会重画眼睛并可能出现杂点；832×1216 图中眼睛约 45 像素，"
                 "瞳孔与虹膜环需要像素：先高清修复或 2× 放大再重绘可画出（再多约 12% 细节）。"
                 "提示词还含“round pupils”，反向提示词含“slit pupils, cat eyes”——否则“detailed pupils”在部分模型上会画出粗黑边的竖瞳；"
                 "若你的提示词自己指定了瞳孔形状（slit pupils、heart-shaped pupils、@_@ 等），则保持你的设定。Pony 系模型做任何眼部重绘都会变软"
                 "（0.25–0.4 时清晰度约 −27%）：建议对 Pony 关闭。"
                 "颜色保护只让虹膜内取新颜色，遮眼的刘海、眼睑和皮肤保持原色，不会被红色虹膜染色。")],
            ["🌡 Cool mode / pause while hot (Settings)",
             "For laptops that switch themselves off under long GPU runs. Cool mode pauses after every sampling step (1.5 = the GPU "
             "works ~40 % of the time; same pictures, slower). Pause while hot waits before each picture and each hires / face / eye "
             "pass until the thermal zone is 8 °C under the limit you set (90 suits the RX 6800M laptop, which switched off after "
             "minutes at 96 °C). Stop ends any wait. Both are off by default and saved for the next start.",
             _cn("笔记本长时间满载会自动关机时使用。冷却模式在每个采样步之后暂停（1.5 = GPU 约 40% 时间工作，画面不变，速度变慢）；"
                 "过热暂停会在每张图和每个高清修复/脸/眼重绘之前等待，直到温度比设定值低 8 °C（RX 6800M 笔记本建议 90）。停止按钮可结束等待。默认关闭，设置会保存。")],
            ["✨ Polish (Generate tab)",
             "The dropdown above Hires fix — Portrait / Cowboy shot / Full body / Wide shot, or Auto (reads the framing tags in the prompt) — "
             "fills hires fix, face detail and eye detail with the recipe for that framing: eyes only for a portrait or cowboy shot (the face is "
             "≥ ~24 % of the width, where face detail changes nothing), hires 1.5× + eyes for a full body, hires 1.5× + face + eyes for a wide shot. "
             "The note shows how many times the plain picture's time it costs (≈ 1.4× / 1.4× / 4.1× / 4.8×; measured chains came out cheaper). "
             "Eye detail is skipped on Pony models (it blurs them). Measured (2 seeds each): at a cowboy shot "
             "(face 24–28 %) nothing in the chain changes identity — it is insurance; for a full body the plain face is often damaged and hires 1.5× + eyes "
             "fixes it best (identity score 0.937 → 0.967, the hair pin .42 → .90, face ×1.3 sharper) — the face pass on top of hires adds nothing. "
             "Hands are left out: the hand pass cleans texture but never fixed a finger count. Change any control afterwards.",
             _cn("高清修复上方的下拉框：人像 / 牛仔镜头 / 全身 / 远景（或“自动”，读取提示词里的构图标签）按构图一键填好高清修复、脸部与眼部重绘："
                 "人像和牛仔镜头只重绘眼睛（脸宽 ≥ 约 24%，脸部重绘无效），全身 高清修复 1.5×+眼，远景 高清修复 1.5×+脸+眼；提示条显示约为普通出图的几倍时间（实测更省时）。"
                 "实测（各 2 个种子）：牛仔镜头（脸宽 24–28%）整条链几乎不改变相似度，只是保险；全身图的原始脸常有损坏，高清修复 1.5×+眼睛修得最好"
                 "（相似度 0.937→0.967，发饰 .42→.90，脸部清晰度 ×1.3），在其上再做脸部重绘没有额外收益。"
                 "不含手部重绘（只改善纹理，不能修正手指数）。之后仍可调整任何控件。")],
            ["⚖️ Prompt weights & colour bleeding",
             "(tag:1.3) works like A1111 now (Settings → Prompt weights): the app used to apply weights 2–5× harder, so "
             "weighted background / lighting / colour tags flooded their colour onto the character. If a scene colour "
             "still bleeds, lower that tag's weight.",
             _cn("权重 (tag:1.3) 现与 A1111 相同（设置 → Prompt weights）；之前权重过强，背景/光照颜色会染到角色身上。"
                 "仍有串色时降低场景标签的权重。")],
            ["🎲 Samplers on SDXL",
             "LMS and PNDM used to fill SDXL pictures with colour noise (4th-order multistep methods are unstable on "
             "SDXL); on SDXL-family models (SDXL, Pony, Illustrious, NoobAI) they now run as 2nd-order methods and are "
             "clean. PNDM and Heun still leave a little more grain in skies and backgrounds than DPM++ 2M Karras / AYS "
             "or Euler — the result line says so on SDXL models.",
             _cn("LMS、PNDM 以前会在 SDXL 上产生彩色噪点（四阶多步法在 SDXL 上不稳定）；在 SDXL 系列模型（SDXL / Pony / "
                 "Illustrious / NoobAI）上它们现在按二阶运行，已正常。"
                 "PNDM 和 Heun 在天空/背景处仍比 DPM++ 2M Karras / AYS 或 Euler 略多颗粒。")],
            ["🎴 Every outfit of a card",
             "Character cards → “🎴▶ Generate every outfit”: each outfit × N seeds (the same seeds for every outfit) "
             "with the current settings, then a labelled contact sheet. Pairs already made with the same settings "
             "are skipped, so a stopped run resumes. Every picture gets a ⭐ rating (n/5) on the sheet from CPU checks "
             "whose models are already cached: one face / at most two hands, the card's hair and eye colours and its "
             "hair pin on the head crop, colour noise — with the reasons for anything below 5. After “🧬 Learn her look” (3 or more "
             "pictures of her with a face ≥ 100 px — pose and lighting don't matter, but hair and headwear do: include the hats, ponytails and buns of the "
             "outfits you will judge; the first use downloads the 150 MB CCIP anime-character model) it also flags a face that is further from her than her own pictures are. Measured on 247 existing "
             "pictures: her own held-out pictures flagged ~3 %, a no-LoRA look-alike with her tags up to ~1 in 5, another character with her "
             "tags up to ~1 in 3 (tighter references catch more), another girl every time; faces under 100 px are not judged. A warning, not a verdict. With 12 plain daylight references 15 of the 33 outfits of the v3 card were flagged although every picture was her "
             "(a hat, ponytail or buns shifts CCIP's fingerprint; references that include such outfits flagged 11 %, and the message then says \"check by eye\").",
             _cn("角色卡 →“生成所有服装”：每套服装 × N 个种子（种子相同便于对比），最后生成对比图；已生成的会跳过，可断点续跑。"
                 "对比图上每张图有 ⭐ 评分（n/5）：一张脸/至多两只手、角色卡的发色瞳色与发饰、彩色噪点（仅用已缓存的模型在 CPU 上检查）。"
                 "用“🧬 Learn her look”（3 张以上她的图，脸宽 ≥ 100 像素，姿势/光线不限，但发型和头饰有影响：请包含要检查的服装里的帽子、马尾、丸子头；首次使用会下载 150 MB 的 CCIP 动漫角色识别模型）学习后还会标出比她本人图更不像她的脸（247 张现有图实测：她自己的图误报约 3%，同标签无 LoRA 的相似脸最多约 1/5，同标签的其他角色最多约 1/3（参考图越相近检出越多），其他女孩 100%；小于 100 像素的脸不判断）。仅作提示，不是定论。参考图只有普通日光照时，v3 角色卡 33 套服装里有 15 套被标记（其实每张都是她）：帽子、马尾、丸子头会改变 CCIP 指纹；参考图包含这类服装后误报降到 11%，提示会写“请肉眼检查”。")],
            ["🗂 History",
             "Search everything in outputs/ by prompt words, model, LoRA or seed; ⭐ favourites; open an image in PNG "
             "Info → Send to Generate to restore exactly how it was made.",
             _cn("历史：按提示词、模型、LoRA、种子搜索全部输出，可收藏，一键在 PNG Info 中恢复参数。")],
            ["✨ SD detail pass (Upscale tab)",
             "After upscaling, re-draws the image in native-size tiles at low denoise (0.25–0.35) with the loaded "
             "model — real detail beyond hires fix's 1536 / 2048 px limit. Higher denoise can put faces into tiles.",
             _cn("放大后用已加载模型分块低降噪重绘，超越高清修复的尺寸上限；降噪过高可能在分块里画出多余的脸。")],
            ["🖌 Inpaint",
             "Paint over any area and describe what should be there; only that area is redrawn, at native "
             "resolution, with your model, LoRAs and seed. Good for small things — measured: to swap a hair "
             "ornament use 🦋 Swap (paint over the whole old one — what is outside the paint stays —, type the new one: a single pass at denoise 1.0 with a hair-only prompt — at 0.9–0.95 the old shape still shows through as lace, 0.75–0.85 gives a hybrid), an expression ~0.5 "
             "deepens a smile and ~0.65–0.8 opens a closed mouth (waiIllustrious opens it at ~0.5). It uses the evenly spaced version of your sampler so "
             "denoise means what it says, and the Steps value is what runs at any denoise (as in A1111). A garment recolour can't be done this way (up to 0.95 the colour stays, "
             "1.0 over a body-sized area draws a new figure) — use same seed + edited prompt.",
             _cn("涂抹要修改的区域并描述内容，仅重绘该区域。适合小物件（换发饰请用 🦋 Swap：涂抹旧饰品、输入新饰品，降噪 1.0、仅用发型提示词；0.9–0.95 旧饰品的线条仍会隐约残留；表情约 0.5 微调、约 0.65–0.8 张嘴）；"
                 "任何降噪下都按设定步数运行；换衣服颜色请用同种子+改提示词。")],
            ["⏩ DPM++ 2M AYS",
             "NVIDIA's Align-Your-Steps schedule: about the quality of 25 steps in 10–12. The Karras samplers "
             "really use Karras sigmas (as in A1111).",
             _cn("AYS 采样计划：10–12 步约等于常规 25 步的质量。")],
            ["🎚 PAG / FreeU / CFG rescale",
             "PAG 2–3: cleaner structure and backgrounds (~1.5–2× time). FreeU: more detail and contrast. CFG rescale: "
             "tames burned colours; automatic 0.7 for v-prediction models (NoobAI v-pred).",
             _cn("PAG 改善结构与背景；FreeU 增加细节；CFG rescale 抑制过饱和，v-pred 模型自动 0.7。")],
            ["🔤 Danbooru tags",
             "Autocomplete chips while typing (most-used first) and near-miss hints (long haired → long hair); "
             "“💡 Fix Danbooru spellings” applies them.",
             _cn("输入时补全 Danbooru 标签，并提示拼写相近的正确标签。")],
            ["🎲 Wildcards",
             "{a|b|c} picks one option per image, {2$$a|b|c} two, {3::a|b} weights a; __outfit__ picks a line from "
             "wildcards/outfit.txt (add your own .txt files). Picks follow each image's seed.",
             _cn("{a|b|c} 每张图随机选一项；__outfit__ 从 wildcards/outfit.txt 随机取一行。随种子固定。")],
            ["🎴 Character cards",
             "Checkpoint + LoRAs + character tags + outfits + size/CFG/steps in one click. “Build from LoRA slot 1” "
             "turns a character LoRA's trigger words and Civitai example prompts into a card with outfits. "
             "“📌 Save as this checkpoint's profile” remembers the LoRA weights, CFG, sampler and boosters (PAG, face "
             "detail …) that suit the selected checkpoint; 🎴 Load applies the profile of whichever checkpoint is "
             "selected, and the card's defaults for the others.",
             _cn("一键载入角色的模型、LoRA、角色标签与服装。可由 LoRA 自动生成。“📌 保存为当前模型的配置”记住该模型适用的 LoRA 权重、"
                 "CFG、采样器与画质增强；载入时按当前所选模型套用对应配置。")],
            ["🏷 Interrogate (WD14)",
             "Reads an image's Danbooru tags into the prompt (img2img, PNG Info); in Train LoRA it tags a whole "
             "dataset. ~1 s per image on the CPU.",
             _cn("WD14 反推图片的 Danbooru 标签；训练页可批量打标。")],
            ["📊 X/Y grid",
             "Same seed, every combination of two settings (CFG, steps, sampler, seed, LoRA 1 weight, CLIP skip, "
             "hires denoise, checkpoint, or Prompt S/R word swaps) in one labelled grid image.",
             _cn("同一种子下对比两组参数的所有组合，生成带标签的网格图。")],
        ]
    )
    h += _tip("All of these are saved in the image and come back when you drop it into Image-to-Image. "
              + _cn("以上设置都会写入图片，拖入图生图即可恢复。"))
    return _wrap(h)


# ═════════════════════════════════════════════════════════════════════════════
# Section 3 — Model Architecture
# ═════════════════════════════════════════════════════════════════════════════
def models_html() -> str:
    h = '<div class="hg-h2">🧩 Model Architecture &nbsp;<span class="cn">模型架构</span></div>'

    h += _table(
        ["Model / 模型", "Base Res. / 基础分辨率", "LoRA Type / LoRA类型", _cn("特点与说明")],
        [
            ["<b>SD 1.5</b>",
             "512 × 512",
             "SD1 LoRA",
             _cn("最广泛的生态，Civitai上绝大多数LoRA是SD1.5格式。显存占用低（~4GB），速度快。")],
            ["<b>SD 2.x</b>",
             "768 × 768",
             "SD2 LoRA",
             _cn("改进版，更好的解剖结构，但LoRA生态较少，已基本被SDXL取代。")],
            ["<b>SDXL</b>",
             "1024 × 1024",
             "SDXL LoRA",
             _cn("高分辨率、高质量，显存需求较高（~8GB+）。Illustrious/NoobAI等二次元大模型基于此架构。")],
            ["<b>Illustrious / NoobAI</b>",
             "1024 × 1024",
             "SDXL LoRA",
             _cn("基于SDXL微调的高质量二次元模型，需要 score_9, score_8_up 等质量标签。")],
            ["<b>Pony Diffusion V6</b>",
             "1024 × 1024",
             "SDXL / Pony LoRA",
             _cn("以兽人/拟人/二次元为主，使用 score_9 score_8_up 质量打分标签体系。")],
            ["<b>FLUX</b>",
             "1024 × 1024",
             "FLUX LoRA (专用)",
             _cn("最新架构，图像质量极高，推理速度慢，显存需求大（~12GB+）。")],
        ]
    )

    h += _note("⚠️ LoRAs are <b>not cross-compatible</b> between architectures. "
               "SD1.5 LoRA on an SDXL model will crash or produce garbage. "
               "Every checkpoint and LoRA in the lists is labelled <b>SD 1.5</b> or <b>SDXL</b> (read from the file), "
               "and the app warns you as soon as a slot doesn't match. &nbsp;"
               + _cn("不同架构的LoRA不能混用！SD1.5的LoRA用在SDXL模型上会崩溃或生成乱图。列表中每个模型和LoRA都标有 SD 1.5 / SDXL（从文件读取），"
                     "槽位不匹配时会立即提示。"))

    h += '<div class="hg-h3">Pony / NoobAI / Illustrious Quality Tags &nbsp;<span class="cn">质量打分标签</span></div>'
    h += _table(
        ["Positive Tag / 正向标签", "Negative Tag / 负向标签", _cn("说明")],
        [
            [_code("score_9"),          _code("score_1"),   _cn("最高/最低质量分。score_9 对这类模型至关重要。")],
            [_code("score_8_up"),       _code("score_2"),   _cn("8分及以上质量范围。")],
            [_code("score_7_up"),       _code("score_3"),   _cn("7分及以上，可选。")],
            [_code("masterpiece"),      _code("score_4"),   _cn("杰作，与score_9配合使用效果更佳。")],
            [_code("best quality"),     "",                 _cn("最佳质量，通用正向标签。")],
        ]
    )
    h += _tip("For Pony/NoobAI/Illustrious, <b>always start positive prompts</b> with: "
              + _code("score_9, score_8_up, score_7_up, masterpiece, best quality")
              + _cn("  — 这类模型不加分数标签出图质量会明显下降。"))
    return _wrap(h)


# ═════════════════════════════════════════════════════════════════════════════
# Section 4 — LoRA & VAE
# ═════════════════════════════════════════════════════════════════════════════
def lora_vae_html() -> str:
    h = '<div class="hg-h2">🔗 LoRA & VAE</div>'

    h += '<div class="hg-h3">LoRA (Low-Rank Adaptation) &nbsp;<span class="cn">低秩适应微调模型</span></div>'
    h += _table(
        ["Concept / 概念", "Detail / 详解", _cn("中文说明")],
        [
            ["What is LoRA? / 什么是LoRA",
             "Small supplementary model that modifies the base checkpoint's output style/subject/behavior.",
             _cn("LoRA是附加在大模型上的小型权重文件，用于微调风格、角色或特定内容，不需要重新训练整个模型。")],
            ["Weight (强度)",
             "0.5 = subtle influence. 0.8 = recommended default. 1.0 = full effect. &gt;1.0 = exaggerated.",
             _cn("0.5~0.8融合自然，1.0完全效果，超过1.0会过度强调甚至变形。通常0.7~0.9是最佳范围。")],
            ["Trigger Words / 触发词",
             "Many LoRAs require specific keywords to activate. Check the trigger words panel.",
             _cn("很多LoRA需要特定触发词才能激活效果。查看左侧触发词面板或在Civitai页面查看。")],
            ["Multi-LoRA / 多LoRA叠加",
             "Up to 3 LoRAs at once, each at its own weight (they are merged into the model one after another).",
             _cn("最多同时叠加3个LoRA，各自独立权重（依次合并进模型）。")],
            ["LoCon / LyCORIS",
             "Advanced LoRA variants with more parameter coverage. Treat them the same as LoRA — "
             "LoHa and LoKr files work too (the app merges them itself).",
             _cn("LoCon/LyCORIS是更高级的LoRA变体，覆盖更多网络层，使用方法与LoRA相同；LoHa、LoKr 格式同样可用（由本应用自行合并）。")],
            [_code("&lt;lora:name:0.8&gt;"),
             "Civitai / A1111 prompts carry LoRAs as tags. Paste such a prompt into Generate and the tags "
             "move into free LoRA slots; the Bridge applies them to its SD 1.5 stage; PNG Info → Generate "
             "fills the slots from them.",
             _cn("Civitai/A1111 的提示词用标签写 LoRA。粘贴到生成页时标签会自动移入空闲的 LoRA 槽位；"
                 "桥接页会把它们用于 SD 1.5 阶段；PNG信息 → 生成 也会据此填好槽位。")],
            ["DoRA",
             "Direction + Rank decomposition. Newer format, better style fidelity.",
             _cn("DoRA是新格式LoRA，方向+秩分解，风格还原度更高。")],
        ]
    )

    h += '<div class="hg-h3">VAE (Variational Auto-Encoder) &nbsp;<span class="cn">变分自编码器</span></div>'
    h += _table(
        ["Concept / 概念", "Detail / 详解", _cn("中文说明")],
        [
            ["What is VAE? / 什么是VAE",
             "The decoder that converts latent vectors to final pixel images. Affects color/saturation.",
             _cn("VAE是将潜空间向量解码为最终像素图像的组件，直接影响颜色饱和度、肤色还原和细节清晰度。")],
            [_code("mse840000 / vae-ft-mse"),
             "Most common, fixes washed-out colors on SD1.5 models.",
             _cn("最常用的通用VAE，修复SD1.5模型色彩偏灰偏淡的问题，推荐SD1.5模型配套使用。")],
            [_code("840000-ema / kl-f8"),
             "Richer colors, slightly more saturated. Popular for anime.",
             _cn("色彩更鲜艳，动漫风格模型常用。")],
            ["Built-in VAE / 内置VAE",
             "Most modern SDXL and Pony models have a baked-in VAE — leave the VAE dropdown as 'none'.",
             _cn("大多数现代SDXL/Pony模型已内置VAE，下拉框保持'none'即可，不需要额外选择。")],
            ["NSFW Tip / 成人内容提示",
             "If anime NSFW looks gray/washed, add a VAE like mse840000.",
             _cn("如果动漫成人图颜色发灰，给SD1.5模型加一个mse840000 VAE即可解决。")],
        ]
    )
    return _wrap(h)


# ═════════════════════════════════════════════════════════════════════════════
# Section 4b — LoRA Training Guide
# ═════════════════════════════════════════════════════════════════════════════
def lora_training_html() -> str:
    h = '<div class="hg-h2">🎓 LoRA Training Guide &nbsp;<span class="cn">LoRA训练指南</span></div>'

    # ── Overview ────────────────────────────────────────────────────────────
    h += '<div class="hg-h3">What is LoRA training? &nbsp;<span class="cn">什么是LoRA训练?</span></div>'
    h += ('<p style="font-size:13px;color:#cdd6f4;line-height:1.6">'
          'LoRA training teaches a pre-trained Stable Diffusion model a new concept '
          '(character, style, object) by adding small "adapter" weight matrices '
          'to the frozen base model. Only the adapters are trained — the base model '
          'stays untouched. The result is a small <code>.safetensors</code> file (5–200 MB) '
          'that you load in the Generate tab.'
          '<br><span class="cn">LoRA训练通过向冻结的基础模型添加小型"适配器"权重矩阵来教模型新概念'
          '（角色、风格、物体）。只训练适配器——基础模型不变。结果是一个小的safetensors文件，在生成标签页中加载。</span></p>')

    # ── Dataset preparation ────────────────────────────────────────────────
    h += '<div class="hg-h3">Dataset Preparation &nbsp;<span class="cn">数据集准备</span></div>'
    h += _table(
        ["Topic / 主题", "Advice / 建议", _cn("中文说明")],
        [
            ["Image count / 图片数量",
             "Character: 15–50 images. Style: 30–200. More diverse > more quantity.",
             _cn("角色LoRA准备15-50张图，风格LoRA准备30-200张。多样性比数量更重要。")],
            ["Image quality / 图片质量",
             "High-res, clean images. Remove watermarks, text overlays, low-quality shots.",
             _cn("使用高清、干净的图片。去除水印、文字覆盖、低质量图片。")],
            ["Solo character / 单人角色",
             "For character LoRAs, use solo images. Crop out other characters.",
             _cn("角色LoRA尽量使用单人图片，裁掉其他角色。")],
            ["Aspect ratios / 宽高比",
             "Mixed ratios are fine! Bucketing handles squares, portraits, landscapes, "
             "and even body pillows (16:5) without content loss.",
             _cn("混合宽高比没问题！分桶系统处理正方形、竖图、横图，甚至抱枕(16:5)都不会裁切内容。")],
            ["File formats / 文件格式",
             "PNG, JPG, WebP, BMP, TIFF all supported.",
             _cn("支持PNG、JPG、WebP、BMP、TIFF格式。")],
        ]
    )

    # ── Captioning ─────────────────────────────────────────────────────────
    h += '<div class="hg-h3">Captioning &nbsp;<span class="cn">标注/打标</span></div>'
    h += _table(
        ["Topic / 主题", "Advice / 建议", _cn("中文说明")],
        [
            ["Trigger word / 触发词",
             "A unique keyword that activates the LoRA. "
             "Use the character name or an invented word. Prepended to every caption.",
             _cn("一个唯一的关键词用来激活LoRA。使用角色名或自创词。会自动添加到每个标注前面。")],
            ["Character captions / 角色标注",
             _code("my_character, 1girl, red eyes, grey hair, long hair, dress, standing, park") +
             "<br>List the trigger, then visual features, then scene/pose/background.",
             _cn("先写触发词，再写视觉特征（发色、眼色、服装），最后写场景/姿势/背景。")],
            ["Style captions / 风格标注",
             "Describe the <em>content</em> (not the style). "
             "The model will learn to associate the trigger word with the visual style itself.",
             _cn("描述内容而非风格。模型会自动学会将触发词与视觉风格关联。")],
            ["Auto-caption (BLIP) / 自动标注",
             "BLIP generates a starting caption. Always review and edit — "
             "it may miss character-specific details.",
             _cn("BLIP生成初始标注。务必检查和编辑——它可能遗漏角色特定细节。")],
            ["Caption file / 标注文件",
             "Each image <code>photo.png</code> needs a <code>photo.txt</code> alongside it.",
             _cn("每张图片photo.png需要旁边有一个photo.txt文件。")],
        ]
    )
    h += _tip("For character LoRAs, include trait tags like hair/eye color in <em>every</em> caption. "
              "This teaches the model to always reproduce those features when the trigger is used."
              "<br>" + _cn("角色LoRA中，每个标注都要包含发色、眼色等特征标签，这样模型才能学会用触发词时始终还原这些特征。"))

    # ── Training parameters ────────────────────────────────────────────────
    h += '<div class="hg-h3">Training Parameters &nbsp;<span class="cn">训练参数</span></div>'
    h += _table(
        ["Parameter / 参数", "Recommended / 推荐值", "Explanation / 解释"],
        [
            ["Rank (dim) / 秩",
             "16 (standard)<br>8 (lightweight)<br>32+ (complex styles)",
             "Number of adapter dimensions. Higher = more capacity, larger file. "
             + _cn("适配器维度数。越高容量越大，文件越大。16是最常用值。")],
            ["Alpha / α",
             "= rank (e.g. 16)",
             "Scaling factor. <code>alpha / rank</code> controls the LoRA's strength. "
             "alpha = rank → scale = 1.0 (neutral). "
             + _cn("缩放因子。alpha/rank控制LoRA强度。alpha=rank时缩放为1.0（中性）。")],
            ["Learning rate / 学习率",
             "UNet: 1e-4<br>TE: 5e-5",
             "Too high → overfitting (memorises images). Too low → no effect. "
             + _cn("太高→过拟合（记住图片）。太低→没有效果。1e-4是安全默认值。")],
            ["Epochs / 训练轮次",
             "10–20 (character)<br>5–10 (style)",
             "How many times the trainer sees every image. More images → fewer epochs. "
             + _cn("训练器看每张图片的次数。图片越多需要的轮次越少。")],
            ["Train text encoder / 训练文本编码器",
             "✅ ON for characters<br>❌ OFF for styles",
             "Teaches the trigger word meaning. Essential for character LoRAs. "
             + _cn("教会模型触发词的含义。角色LoRA必须开启。风格LoRA可以关闭。")],
            ["Target layers / 目标层",
             "attn (fast)<br>full (thorough)",
             "'attn' = Q/K/V/Out attention. 'full' = + conv layers for spatial detail. "
             + _cn("attn=只训练注意力层（快）。full=加上卷积层，空间细节更好。")],
            ["LR scheduler / 学习率调度",
             "cosine",
             "Cosine gently decreases LR over training. Usually best. "
             + _cn("余弦调度平缓降低学习率，通常最佳。")],
            ["Gradient checkpointing / 梯度检查点",
             "✅ ON",
             "Saves memory at ~15% speed cost. Always enable for CPU training. "
             + _cn("节省内存，速度降低约15%。CPU训练时始终开启。")],
        ]
    )

    # ── Troubleshooting ────────────────────────────────────────────────────
    h += '<div class="hg-h3">Troubleshooting / 常见问题 &nbsp;<span class="cn">故障排除</span></div>'
    h += _table(
        ["Problem / 问题", "Cause / 原因", "Fix / 解决方案"],
        [
            ["LoRA has no effect / LoRA没有效果",
             "Forgot trigger word, weight too low, or base model mismatch.",
             "Use trigger word in prompt, try weight 1.0, check base model compatibility. "
             + _cn("在提示词中使用触发词，权重设为1.0，检查基础模型兼容性。")],
            ["Copies training images exactly / 完全复制训练图片",
             "Overfitting — too many epochs or LR too high.",
             "Reduce epochs (10→5), lower LR (1e-4→5e-5), add more training images. "
             + _cn("减少轮次，降低学习率，增加训练图片。")],
            ["Broken colors / anatomy / 颜色或结构损坏",
             "LR too high or bad training data.",
             "Try 5e-5 LR. Remove low-quality images from dataset. "
             + _cn("尝试5e-5学习率。从数据集中移除低质量图片。")],
            ["Only works at weight 1.0 / 只在权重1.0时有效",
             "Alpha too low relative to rank.",
             "Set alpha = rank × 2 (e.g. rank 16, alpha 32). "
             + _cn("设置alpha为rank的2倍。")],
            ["Works on training base but not other models / 仅在训练基础模型上有效",
             "Weight space drift between model versions.",
             "Train on the exact model you plan to use. Illustrious v0.1 ≠ v1.60. "
             + _cn("在你打算使用的确切模型上训练。不同版本的模型权重空间不同。")],
        ]
    )

    # ── Body pillows / extreme aspect ratios ───────────────────────────────
    h += '<div class="hg-h3">Body Pillows &amp; Extreme Ratios &nbsp;<span class="cn">抱枕与极端宽高比</span></div>'
    h += ('<p style="font-size:13px;color:#cdd6f4;line-height:1.6">'
          'Aspect-ratio bucketing assigns each image to a resolution "bucket" '
          'that matches its shape. Only a tiny center crop (if any) is applied — '
          'your wide or tall images are preserved.</p>')
    h += _table(
        ["Aspect Ratio / 比例", "SD 1.5 Bucket / 分桶", "SDXL Bucket / 分桶", _cn("说明")],
        [
            ["1:1 (square / 正方形)", "512×512", "1024×1024", _cn("标准正方形")],
            ["2:3 (portrait / 竖图)", "448×640", "896×1280", _cn("常见竖构图")],
            ["3:2 (landscape / 横图)", "640×448", "1280×896", _cn("常见横构图")],
            ["16:9 (widescreen / 宽屏)", "704×384", "1408×768", _cn("宽屏比例")],
            ["16:5 (body pillow / 抱枕)", "960×320", "1920×640",
             _cn("抱枕等超宽图——不会被裁切！")],
            ["1:3 (tall strip / 竖条)", "320×960", "640×1920", _cn("极窄竖构图")],
        ]
    )
    h += _note("The bucket system generates sizes in steps of 64 pixels. "
               "If your image's exact ratio doesn't have a bucket, the closest one is used "
               "with minimal center cropping (typically &lt;5%)."
               "<br>" + _cn("分桶系统以64像素为步长生成尺寸。如果你的图片精确比例没有对应分桶，"
                            "会使用最接近的分桶，只有极小的居中裁切（通常<5%）。"))

    return _wrap(h)



def quality_tags_html() -> str:
    h = '<div class="hg-h2">✨ Quality Tag Cheatsheet &nbsp;<span class="cn">质量标签速查表</span></div>'

    h += '<div class="hg-h3">Universal Quality Boosters &nbsp;<span class="cn">通用质量正向标签</span></div>'
    h += _table(
        ["Tag / 标签", _cn("含义"), "Notes / 备注"],
        [
            [_code("masterpiece"),        _cn("杰作"),              "Top-tier overall quality signal."],
            [_code("best quality"),       _cn("最佳质量"),          "Paired with masterpiece for strongest effect."],
            [_code("ultra-detailed"),     _cn("超精细细节"),        "Forces high detail rendering."],
            [_code("8k uhd"),             _cn("8K超高清"),          "Upscale-quality signal, more detail."],
            [_code("sharp focus"),        _cn("锐利对焦"),          "Reduces blurriness. Good for portraits."],
            [_code("HDR"),                _cn("高动态范围"),        "Richer contrast and color range."],
            [_code("film grain"),         _cn("胶片颗粒感"),        "Adds cinematic texture."],
            [_code("bokeh"),              _cn("背景虚化/散景"),     "Background blur effect."],
            [_code("cinematic lighting"), _cn("电影级打光"),        "Dramatic professional lighting."],
            [_code("RAW photo"),          _cn("RAW格式照片质感"),   "Forces photorealistic output."],
            [_code("DSLR"),               _cn("单反相机拍摄质感"),  "Realistic camera photo feel."],
            [_code("professional"),       _cn("专业级"),            "General quality booster."],
            [_code("highly detailed"),    _cn("高度精细"),          "More rendering detail everywhere."],
            [_code("intricate details"),  _cn("精致细节"),          "Forces micro-detail rendering."],
        ]
    )

    h += '<div class="hg-h3">Universal Negative Tags &nbsp;<span class="cn">通用负向标签</span></div>'
    h += _table(
        ["Tag / 标签", _cn("含义"), "What it prevents / 防止的问题"],
        [
            [_code("(worst quality:1.4)"),  _cn("最差质量"),        "Strongly discourages low-quality outputs."],
            [_code("(low quality:1.4)"),    _cn("低质量"),          "Discourages blurry/flat results."],
            [_code("normal quality"),        _cn("普通质量"),        "Blocks mediocre mid-tier results."],
            [_code("jpeg artifacts"),        _cn("JPEG压缩噪点"),    "Prevents compression artifacts."],
            [_code("blurry"),                _cn("模糊"),            "Forces sharper output."],
            [_code("out of focus"),          _cn("失焦"),            "Prevents soft/unfocused rendering."],
            [_code("extra fingers"),         _cn("多余手指"),        "Common defect — extra fingers on hands."],
            [_code("missing fingers"),       _cn("缺少手指"),        "Prevents finger count errors."],
            [_code("fused fingers"),         _cn("手指粘连"),        "Prevents merged/webbed fingers."],
            [_code("too many fingers"),      _cn("手指过多"),        "Variant of the finger defect."],
            [_code("(mutated hands:1.3)"),   _cn("变形的手"),        "Broad hand deformity prevention."],
            [_code("poorly drawn hands"),    _cn("手部绘制粗糙"),    "Improves overall hand quality."],
            [_code("poorly drawn face"),     _cn("面部绘制粗糙"),    "Improves face rendering quality."],
            [_code("bad anatomy"),           _cn("解剖结构错误"),    "General anatomy errors."],
            [_code("extra limbs"),           _cn("多余肢体"),        "Prevents extra arms/legs."],
            [_code("missing limbs"),         _cn("缺少肢体"),        "Prevents missing arms/legs."],
            [_code("(deformed:1.3)"),        _cn("变形"),            "Weighted deformity prevention."],
            [_code("disfigured"),            _cn("面目全非"),        "Severe deformity prevention."],
            [_code("watermark"),             _code("水印"),          "Prevents watermark text in image."],
            [_code("signature"),             _cn("签名"),            "Prevents artist signature."],
            [_code("text"),                  _cn("文字"),            "Prevents random text in image."],
            [_code("cropped"),               _cn("裁剪/截断"),       "Prevents cut-off compositions."],
            [_code("out of frame"),          _cn("出画面"),          "Keeps subject within frame."],
        ]
    )
    h += _note("💡 Weights like <b>(worst quality:1.4)</b> DO work — the model will actively avoid "
               "those features. Don't go above 2.0 or it may invert. Negative prompt weights 1.2–1.5 are sweet spot. "
               + _cn("负向提示词加权重是有效果的！(worst quality:1.4) 会让模型更强烈地回避低质量输出。"
                      "权重建议在1.2–1.5之间，超过2.0可能反效果。"))
    return _wrap(h)


# ═════════════════════════════════════════════════════════════════════════════
# Section 6 — Style & Camera Tags
# ═════════════════════════════════════════════════════════════════════════════
def style_tags_html() -> str:
    h = '<div class="hg-h2">🎨 Style & Camera Tags &nbsp;<span class="cn">风格与镜头标签</span></div>'

    h += '<div class="hg-h3">Art Style Tags &nbsp;<span class="cn">画风标签</span></div>'
    h += _table(
        ["Tag / 标签", _cn("中文"), "Effect / 效果"],
        [
            [_code("anime"),           _cn("动漫"),        "Japanese animation style."],
            [_code("manga"),           _cn("漫画"),        "Black and white comic style."],
            [_code("photorealistic"),  _cn("写实照片风"),  "Photo-like realism."],
            [_code("3D render"),       _cn("3D渲染"),       "CGI rendered look."],
            [_code("illustration"),    _cn("插画"),         "Digital art illustration style."],
            [_code("oil painting"),    _cn("油画"),         "Traditional oil painting texture."],
            [_code("watercolor"),      _cn("水彩"),         "Soft watercolor art style."],
            [_code("cel shading"),     _cn("赛璐璐着色"),   "Flat cartoon shading like anime game."],
            [_code("concept art"),     _cn("概念艺术"),     "Professional design/fantasy art feel."],
            [_code("pixel art"),       _cn("像素艺术"),     "Retro pixelated style."],
        ]
    )

    h += '<div class="hg-h3">Camera & Composition Tags &nbsp;<span class="cn">构图与镜头标签</span></div>'
    h += _table(
        ["Tag / 标签", _cn("中文"), "Effect / 效果"],
        [
            [_code("close-up"),        _cn("特写"),        "Tight crop on face/detail."],
            [_code("portrait"),        _cn("人像"),        "Head and shoulders framing."],
            [_code("full body"),       _cn("全身"),        "Full body shot."],
            [_code("upper body"),      _cn("上半身"),      "Chest and above."],
            [_code("from above"),      _cn("俯视"),        "Bird's-eye / top-down angle."],
            [_code("from below"),      _cn("仰视"),        "Upward angle, exaggerates legs/height."],
            [_code("from behind"),     _cn("背面"),        "View from behind the character."],
            [_code("dutch angle"),     _cn("荷兰角"),      "Tilted/dramatic camera angle."],
            [_code("wide shot"),       _cn("广角"),        "Environmental/landscape framing."],
            [_code("pov"),             _cn("第一人称视角"), "Point-of-view shot."],
            [_code("looking at viewer"), _cn("直视观众"),  "Character makes eye contact with camera."],
        ]
    )

    return _wrap(h)


# ═════════════════════════════════════════════════════════════════════════════
# Section 7 — SmartSplit & Hardware
# ═════════════════════════════════════════════════════════════════════════════
def hardware_html() -> str:
    h = '<div class="hg-h2">🔀 SmartSplit & Hardware &nbsp;<span class="cn">多设备并行加速</span></div>'

    h += _table(
        ["Component / 组件", "Role / 职责", _cn("说明")],
        [
            ["<b>ZLUDA v6</b>",
             "Translates CUDA calls to AMD HIP so AMD GPUs appear as NVIDIA to PyTorch. Primary backend.",
             _cn("将CUDA指令翻译为AMD HIP，让PyTorch/Diffusers把你的AMD GPU当作NVIDIA使用。由launch.bat自动启用。主要后端。")],
            ["<b>dGPU (RX 6800M / RX 6800 XT / RX 9070 XT)</b>",
             "Runs text encoders, UNet and VAE decode via ZLUDA. cuDNN/MIOpen is disabled on AMD "
             "(its convolutions return wrong results under ZLUDA); PyTorch's native convolutions are used instead.",
             _cn("通过ZLUDA运行文本编码、UNet去噪和VAE解码。AMD上禁用cuDNN/MIOpen（其卷积在ZLUDA下结果错误），改用PyTorch原生卷积。")],
            ["<b>iGPU (780M / Vega / 610M)</b>",
             "Optional SDXL text encoding via ONNX DirectML — mainly useful on the DirectML backend "
             "(under ZLUDA the dGPU already encodes prompts in well under a second). Also used for VAE in SmartSplit (SD 1.5).",
             _cn("可选：通过ONNX DirectML运行SDXL文本编码——主要对DirectML后端有用（ZLUDA下独显编码提示词不到1秒）。SmartSplit模式下也可运行VAE解码。")],
            ["<b>NPU (XDNA1)</b>",
             "Detected but cannot run full CLIP models (hardware context creation fails). iGPU provides the acceleration instead.",
             _cn("已检测到但无法运行完整CLIP模型（硬件上下文创建失败）。改由集显iGPU通过DML提供加速。")],
            ["<b>DirectML</b>",
             "Microsoft's GPU API, works on any DX12 GPU. Used for iGPU text encoding and as ZLUDA fallback.",
             _cn("微软通用GPU接口，支持所有DX12显卡。用于集显文本编码，也是ZLUDA不可用时的备用方案。")],
        ]
    )

    h += '<div class="hg-h3">Accelerated SDXL Text Encoding &nbsp;<span class="cn">SDXL加速文本编码</span></div>'
    h += """<ol style="color:#cdd6f4;font-size:13px;line-height:2;">
<li>Go to <b>⚙️ Settings</b> tab → check <b>⚡ Accelerated SDXL text encoding (iGPU)</b>.</li>
<li>CLIP-L and OpenCLIP-G are exported to ONNX once per model (cached in <code>onnx_cache/</code>).</li>
<li>Subsequent runs use the integrated GPU via DirectML (the app finds it by name — DML device order differs per PC).
    On a 780M: ~115ms total vs ~6.4s CPU = <b>56× speedup</b>.</li>
<li>Quality is identical: cosine similarity >0.9998 vs CPU baseline.</li>
<li>Supports long prompts (>77 tokens): 75-token chunks cut at tag boundaries; BREAK starts a new one.</li>
<li>Under ZLUDA this is rarely needed — the text encoders already run on the dGPU.</li>
</ol>
<div class="hg-note">""" + _cn(
        "启用步骤：进入设置标签页 → 勾选⚡加速SDXL文本编码 → 首次使用时自动导出ONNX（每模型一次）→ "
        "后续运行使用集显（780M上比CPU快56倍）。输出质量不变。ZLUDA下文本编码已在独显运行，通常无需开启。"
    ) + "</div>"

    h += '<div class="hg-h3">GPU Self-Test &nbsp;<span class="cn">GPU自检</span></div>'
    h += """<ol style="color:#cdd6f4;font-size:13px;line-height:2;">
<li>Black/noisy images or NaN errors after a driver, ROCm or ZLUDA update? Run
    <code>run_zluda.bat selftest_zluda.py</code> in the <code>ImageGenApp</code> folder.</li>
<li>It compares GPU results for matrix multiply, convolution, attention and GroupNorm against the CPU
    and fails loudly if the GPU backend returns wrong numbers.</li>
<li>If generation suddenly becomes very slow, check the info line for <b>⚠ VRAM over-committed</b> —
    Windows spills to system RAM instead of reporting out-of-memory. Lower the resolution or batch size.</li>
</ol>
<div class="hg-note">""" + _cn(
        "驱动/ROCm/ZLUDA更新后出现黑图或NaN？在ImageGenApp目录运行 run_zluda.bat selftest_zluda.py，"
        "它会对比GPU与CPU的计算结果。若生成突然变得极慢，查看信息栏是否出现“VRAM over-committed”——"
        "Windows显存不足时不会报错，而是溢出到系统内存，请降低分辨率或批量。"
    ) + "</div>"

    h += '<div class="hg-h3">SmartSplit (SD 1.5 Only) &nbsp;<span class="cn">多设备分流（仅SD 1.5）</span></div>'
    h += """<ol style="color:#cdd6f4;font-size:13px;line-height:2;">
<li>Launch via <code>launch.bat</code> (ZLUDA must be active for dGPU detection).</li>
<li>Go to <b>⚙️ Settings</b> tab → <b>SmartSplit</b> section.</li>
<li>If it shows as <em>capable</em>, check <b>Enable SmartSplit</b>.</li>
<li>Choose Text Encoder device (CPU / iGPU) and VAE device (CPU).</li>
<li>Click <b>Apply SmartSplit Config</b>.</li>
<li>Load any <b>SD 1.5</b> model — the SmartSplit pipe auto-builds.</li>
<li>⚠️ SmartSplit is <b>not used for SDXL</b> — SDXL uses the accelerated text encoding above instead.</li>
</ol>
<div class="hg-note">""" + _cn(
        "SmartSplit仅适用于SD 1.5模型，将文本编码器和VAE分流到CPU/集显，释放独显显存。"
        "SDXL模型使用上面的加速文本编码方案。"
    ) + "</div>"

    h += '<div class="hg-h3">VRAM Requirements &nbsp;<span class="cn">显存需求参考</span></div>'
    h += _table(
        ["Config / 配置", "Est. VRAM / 预估显存", _cn("说明")],
        [
            ["SD1.5, 512×512, no LoRA",    "~3.5 GB",  _cn("最低配置，适合4GB显卡。")],
            ["SD1.5, 768×768, + LoRA",     "~5.5 GB",  _cn("常用配置。")],
            ["SDXL, 1024×1024, no LoRA",   "~8 GB",    _cn("需要8GB+显卡。")],
            ["SDXL, 1024×1024, + LoRA",    "~10 GB",   _cn("建议12GB+显卡。")],
            ["SDXL, 1024×1024, 3× LoRA",   "~12 GB",   _cn("多LoRA叠加显存占用显著增加。")],
            ["FLUX, 1024×1024",            "~12–16 GB", _cn("FLUX架构显存需求极高。")],
        ]
    )
    return _wrap(h)


# ═════════════════════════════════════════════════════════════════════════════
# Section 8 — Glossary
# ═════════════════════════════════════════════════════════════════════════════
def glossary_html() -> str:
    h = '<div class="hg-h2">📖 Glossary &nbsp;<span class="cn">术语表</span></div>'
    h += _table(
        ["Term / 术语", _cn("中文"), "Definition / 定义"],
        [
            ["Checkpoint",        _cn("大模型/基础模型"),  "The main model file (.safetensors / .ckpt). Everything else builds on top."],
            ["LoRA",              _cn("低秩适应微调"),     "Small add-on weight file that tweaks style/subject/behavior."],
            ["LyCORIS / LoCon",   _cn("高级LoRA变体"),     "Advanced LoRA variants with broader network coverage."],
            ["VAE",               _cn("变分自编码器"),     "Decoder that converts latents to pixels. Affects color."],
            ["Latent space",      _cn("潜空间"),           "Compressed mathematical representation of images the model works in."],
            ["Denoising",         _cn("去噪"),             "The core SD process: iteratively remove noise to form an image."],
            ["CFG Scale",         _cn("提示词引导强度"),   "How strictly the model follows your prompt vs its own creativity."],
            ["Sampler / Scheduler", _cn("采样器"),         "Algorithm controlling how denoising steps proceed."],
            ["Seed",              _cn("随机种子"),         "Number that initializes noise. Same seed + same settings = same image."],
            ["Embedding / TI",    _cn("文本嵌入/Textual Inversion"), "Another type of fine-tuning stored as small vector files."],
            ["Safetensors",       _cn("安全张量格式"),     "Safer model file format (.safetensors) replacing .ckpt/.pt."],
            ["SDXL",              _cn("SD扩展版/大分辨率版"), "Stable Diffusion XL — higher resolution, two-stage model."],
            ["img2img",           _cn("图生图"),           "Generate a new image using an existing image as starting point."],
            ["txt2img",           _cn("文生图"),           "Generate an image from text prompts alone."],
            ["Inpainting",        _cn("局部重绘"),         "Fill/edit masked regions of an existing image."],
            ["ControlNet",        _cn("控制网络"),         "Adds structural control (pose, depth, edges) to generation."],
            ["Trigger word",      _cn("触发词"),           "Keyword required to activate a LoRA's trained content."],
            ["PEFT",              _cn("参数高效微调"),     "Parameter-Efficient Fine-Tuning — the library powering multi-LoRA."],
            ["ZLUDA",             _cn("CUDA转AMD适配层"),  "AMD GPU CUDA emulation layer — makes AMD GPUs work with CUDA code."],
            ["DirectML",          _cn("微软通用GPU接口"),  "Microsoft's vendor-neutral GPU API for any DX12 capable GPU."],
            ["ONNX",              _cn("开放神经网络交换格式"), "Portable model format used for NPU/iGPU inference in SmartSplit."],
            ["UNet",              _cn("U型网络"),          "Core denoising network. The largest, most compute-intensive component."],
            ["CLIP",              _cn("对比语言-图像预训练"), "Text encoder that converts your prompt into vectors the model understands."],
        ]
    )
    return _wrap(h)


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point — called from app.py
# ═════════════════════════════════════════════════════════════════════════════
def build_help_tab():
    """Build the full ❓ Help 帮助 tab. Call inside a gr.Blocks context."""
    import gradio as gr

    with gr.Tab("❓ Help 帮助"):
        gr.HTML(
            '<div style="background:#313244;padding:10px 16px;border-radius:8px;margin-bottom:8px;">'
            '<span style="color:#cba6f7;font-size:15px;font-weight:700;">📚 ImageGen Studio — '
            'Full Reference Guide &nbsp; <span style="color:#89dceb;">完整使用指南</span></span><br>'
            '<span style="color:#a6adc8;font-size:13px;">'
            'All syntax, parameters, and concepts explained in English and Chinese. '
            '<span style="color:#89dceb;">所有语法、参数和概念的中英双语说明。</span></span></div>'
        )

        with gr.Accordion("🚀 Quick Start / 快速上手", open=True):
            gr.HTML(quickstart_html())

        with gr.Accordion("📝 Prompt Syntax / 提示词语法", open=False):
            gr.HTML(prompt_syntax_html())

        with gr.Accordion("⚙️ Generation Parameters / 生成参数", open=False):
            gr.HTML(parameters_html())

        with gr.Accordion("🧩 Model Architecture / 模型架构", open=False):
            gr.HTML(models_html())

        with gr.Accordion("🔗 LoRA & VAE", open=False):
            gr.HTML(lora_vae_html())

        with gr.Accordion("🎓 LoRA Training Guide / LoRA训练指南", open=False):
            gr.HTML(lora_training_html())

        with gr.Accordion("✨ Quality Tag Cheatsheet / 质量标签速查表", open=False):
            gr.HTML(quality_tags_html())

        with gr.Accordion("🎨 Style & Camera Tags / 风格与镜头标签", open=False):
            gr.HTML(style_tags_html())

        with gr.Accordion("🔀 SmartSplit & Hardware / 多设备分流与硬件", open=False):
            gr.HTML(hardware_html())

        with gr.Accordion("📖 Glossary / 术语表", open=False):
            gr.HTML(glossary_html())
