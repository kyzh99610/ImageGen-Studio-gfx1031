"""ui_messages.py — the Chinese of the app's messages and guide texts (status, result and error lines, the static notes on the tabs).

The server writes these in English inside HTML blocks; the page's 中文 / English switch (backend/i18n.py) shows the Chinese:
  * EXACT: a whole text fragment (the text between two HTML tags) -> its Chinese;
  * TEMPLATES: a fragment with values in it, written with {} where a value goes (the Chinese uses {} in the same order, or {0} {1} … to
    reorder) -> turned into a regex for the page. Only text inside HTML / Markdown blocks is matched against the templates.
A fragment that is not here stays English in both languages. Add a line here when a message is added or reworded.
"""
from __future__ import annotations

import html
import re

EXACT: dict[str, str] = {
    # Generate tab: notes and guides
    "Pick a checkpoint, write a prompt, press Generate (Ctrl+Enter). ✨ Polish sets the detail passes for your framing · the Help tab explains every control · the first picture of a session is slower (the GPU kernels load).":
        "选模型，写提示词，按生成（Ctrl+Enter）。✨ 精修会按取景设好细化步骤 · 帮助页解释每个控件 · 每次启动后的第一张图会慢一些（显卡要加载内核）。",
    "No model loaded yet — it loads automatically on your first Generate (SD 1.5 ~20 s, SDXL ~30 s). Preload to do it now.":
        "还没加载模型 — 第一次按生成时会自动加载（SD 1.5 约 20 秒，SDXL 约 30 秒）。按“预加载模型”可以现在就加载。",
    "LoRAs picked below are applied automatically when you Generate (➕ Apply just does it right away). ✖ clears a slot.":
        "下面选的 LoRA 会在生成时自动加上（➕ 应用 = 马上加上）。✖ 清空这个槽。",
    "Click any tag to insert it into the selected prompt box. Toggle the target with the radio below.": "点任意标签插入到选中的提示词框。用下面的单选切换插入到哪个框。",
    "Click to append enhancement tags to your prompt.": "点一下把增强标签加到提示词后面。",
    "Select a preset and click Replace or Append. Quality boosters and scene templates — the Help tab (Quality Tag Cheatsheet) explains the tags.":
        "选一个预设，再点“替换”或“追加”。有质量增强和场景模板 — 帮助页的“质量标签速查”解释了这些标签。",
    "Save / load your own positive prompts. Presets are stored separately for SD 1.5 and XL/Pony/IL.": "保存 / 读取你自己的正向提示词。SD 1.5 和 XL/Pony/IL 的预设分开存放。",
    "Saves the checkpoint, VAE, LoRA slots and weights, prompts, sampler, steps, CFG, size, seed, hires fix, face detail and boosters — and the image selected in the gallery as its example.":
        "保存模型、VAE、LoRA 和权重、提示词、采样器、步数、CFG、尺寸、种子、高清修复、脸部细化和质量增强 — 还会把图库里选中的图存为示例图。",
    "Start looping to automatically generate batch after batch with the current settings. Each batch gets a random seed. Images are saved to":
        "开始循环后，会用当前参数一批接一批地自动出图。每批用随机种子。图片保存到",
    ". Stop anytime with the Stop Loop button.": "。随时可以按“停止循环”停下。",
    "Start with Denoise Strength 0.4-0.6 for refinement. Use the prompt to steer what changes. Lower strength = closer to original. Great for fixing anatomy, adding detail, or changing lighting on a generated image.":
        "精修时重绘幅度从 0.4–0.6 开始。用提示词决定改什么。幅度越低越接近原图。适合修人体、加细节、改光线。",
    "img2img Tips:": "图生图小贴士：",
    "Uses the prompt, model, LoRAs, sampler, steps, CFG and seed above. Only the painted area changes; it is redrawn at the model's native resolution, so small areas (hands, faces) get full detail. Describe what should be there.":
        "使用上面的提示词、模型、LoRA、采样器、步数、CFG 和种子。只改涂抹的区域；它会按模型原生分辨率重画，所以小区域（手、脸）也有完整细节。写清楚那里应该画什么。",
    "One fixed seed, every combination of the values below (current settings for everything else). Numbers:": "固定一个种子，把下面的值两两组合各画一张（其它参数用当前设置）。数字：",
    "or ranges": "或范围",
    ": the first value is a word in your prompt, the others replace it, e.g.": "：第一个值是提示词里的一个词，后面的值依次替换它，例如",
    "does the same in the negative prompt. A value in quotes keeps its commas, e.g.": "在反向提示词里做同样的替换。带逗号的值用引号括起来，例如",
    "An empty value": "空值",
    "removes the words. Each cell is saved like a normal image; the grid is saved too.": "表示删掉这些词。每一格都像普通图片一样保存，整张网格也会保存。",
    "picks one per image ·": "每张图选一个 ·", "picks two ·": "选两个 ·", "makes a 3× likelier ·": "让 a 的概率高 3 倍 ·",
    "picks a line from": "从这个文件随机取一行：", "(add your own .txt files there or in": "（可以在那里或这个文件夹放你自己的 .txt 文件：",
    "). Picks come from each image's seed, so the same seed gives the same picks; the resolved prompt is saved in the image.":
        "）。选到哪一项由每张图的种子决定，所以同一个种子选到的一样；选好的提示词会存在图片里。",
    "No {a|b} groups or __wildcards__ in the prompt — every image gets the same prompt.": "提示词里没有 {a|b} 组或 __通配符__ — 每张图的提示词都一样。",
    "Generate something first.": "请先生成一张图。",
    "No generation settings in this image — img2img mode is on; write a prompt for it.": "这张图里没有生成参数 — 已打开图生图模式；请给它写提示词。",
    "No local models found.": "没有找到本地模型。",
    "First run on this PC:": "这台电脑第一次运行：",
    "the GPU compiles its kernels once, so the": "显卡要先编译一次内核，所以",
    "first image can take 10–15 minutes": "第一张图可能要 10–15 分钟",
    "(it may look stuck — it isn't). After that, images take seconds. SDXL compiles a few more the first time.": "（看起来像卡住了，其实没有）。之后每张图只要几秒。SDXL 第一次还要多编译一些。",
    "More: Help tab → Quick Start · Generation Parameters · Quality Tag Cheatsheet.": "更多：帮助页 → 快速上手 · 生成参数 · 质量标签速查。",
    "Explore with random seeds, then ♻️ reuse the one you like and tweak the prompt": "先用随机种子多试几张，再 ♻️ 复用喜欢的那张的种子，微调提示词",
    # the cheat sheet (Generate tab)
    "🎯 Character LoRA Cheat Sheet": "🎯 角色 LoRA 速查表", "Character LoRA weight": "角色 LoRA 权重", "Style/Effect LoRA weight": "风格 / 效果 LoRA 权重",
    "Higher = more faithful but may distort": "越高越像，但可能变形", "Keep LOWER than character to preserve likeness": "要比角色 LoRA 低，才能保住长相",
    "CFG Scale (Pony/IL)": "CFG（Pony / IL）", "CFG Scale (SD 1.5)": "CFG（SD 1.5）", "Lower than SD 1.5. Over 8 = oversaturated": "比 SD 1.5 低。超过 8 会过饱和",
    "SD 1.5 needs higher CFG than SDXL": "SD 1.5 要比 SDXL 用更高的 CFG", "Resolution (SDXL portrait)": "分辨率（SDXL 竖图）", "Or 768×1344 for taller composition": "想要更高的构图可以用 768×1344",
    "Resolution (SD 1.5 portrait)": "分辨率（SD 1.5 竖图）", "Never exceed 768px on SD 1.5": "SD 1.5 不要超过 768 像素", "More ≠ better: AYS 12 matches 25 Karras steps": "不是越多越好：AYS 12 步相当于 Karras 25 步",
    "📝 Prompt Structure (order matters!)": "📝 提示词结构（顺序很重要！）", "Quality tags": "质量标签", "Character trigger": "角色触发词", "Character features": "角色特征",
    "Scene/action": "场景 / 动作", "Subject/pose": "主体 / 姿势", "Style/lighting": "风格 / 光线", "or whatever the LoRA specifies": "或者 LoRA 页面要求的写法",
    "⚠️ Common Mistakes": "⚠️ 常见错误", "LoRA weight too high (>1.0)": "LoRA 权重太高（> 1.0）", "→ Distorted face, burned colors, artifacts": "→ 脸变形、颜色过曝、出现瑕疵",
    "CFG too high (>10 on Pony)": "CFG 太高（Pony 上 > 10）", "→ Oversaturated, \"deep-fried\" look": "→ 过饱和，像“油炸”过一样", "Wrong trigger words": "触发词写错",
    "→ Character not recognized. Check Civitai page for exact triggers": "→ 认不出角色。去 Civitai 页面查准确的触发词", "(tag: 2.0) emphasis": "(tag: 2.0) 强调",
    "→ Too strong! Use 1.1-1.3 max for emphasis": "→ 太强了！强调最多用 1.1–1.3", "Too many LoRAs at high weights": "多个 LoRA 权重都很高",
    "→ Muddy, conflicting styles. Lower slots 2-3 first": "→ 画面浑浊、风格打架。先降低第 2–3 槽", "(Pony/IL only)": "（只用于 Pony / IL）",
    "AYS 10–12 · Karras 20–30": "AYS 10–12 · Karras 20–30", "0.2–0.4 fixes details · 0.5–0.7 restyles · 0.8+ mostly regenerates": "0.2–0.4 修细节 · 0.5–0.7 换风格 · 0.8 以上基本重画",
    "img2img denoise": "图生图重绘幅度", "Euler a = more variety · DDIM = consistent img2img · UniPC = fewer steps": "Euler a = 变化更多 · DDIM = 图生图更稳定 · UniPC = 步数更少",
    "📐 SD 1.5 — use": "📐 SD 1.5 — 用", ". 512px will produce low-quality output.": "。512 像素出图质量会差。",
    "Illustrious model detected": "检测到 Illustrious 模型", "Pony model detected": "检测到 Pony 模型", "SDXL model detected": "检测到 SDXL 模型", ". Quality tags:": "。质量标签：",
    # Generate: results, stops, errors
    "⏳ Loading the selected model…": "⏳ 正在加载所选模型…",
    "⏹ Auto-Loop stopped before the first batch.": "⏹ 自动循环在第一批之前就停止了。", "⏹ Generation stopped.": "⏹ 已停止生成。", "⏹ Grid stopped.": "⏹ 网格已停止。",
    "⏹ Outfit batch stopped.": "⏹ 批量服装已停止。", "⏹ Stopped (the model is loaded and ready).": "⏹ 已停止（模型已加载好，随时可用）。",
    "⏹ Stopped. (If a model was loading, it finishes loading and stays ready; a running generation ends after its current step.)":
        "⏹ 已停止。（正在加载的模型会加载完并保持就绪；正在画的图会在当前这一步结束后停下。）",
    "⏹ Stopping auto-loop (aborting current batch)…": "⏹ 正在停止自动循环（中止当前这一批）…",
    "Auto-Loop": "自动循环", "Auto-Loop finished": "自动循环完成", "Auto-Loop stopped": "自动循环已停止",
    "⚠ Drop an image first.": "⚠ 请先拖入一张图。", "⚠ Enter a card name.": "⚠ 请填写角色卡名称。", "⚠ No image available to send.": "⚠ 没有可以发送的图片。",
    "⚠ Pick a card first.": "⚠ 请先选一张角色卡。", "⚠ Pick the character LoRA in slot 1 first.": "⚠ 请先在第 1 槽选角色 LoRA。",
    "⚠ Select the checkpoint this profile is for first.": "⚠ 请先选这份配置对应的模型。", "⚠ Unknown wildcard(s):": "⚠ 找不到的通配符：", "⚠ LoRA not found:": "⚠ 找不到 LoRA：",
    "⚠ Settings → Prompt weights is “Compel”; the recipe was measured with A1111 weights.": "⚠ 设置 → 提示词权重现在是 “Compel”；这个配方是用 A1111 权重实测的。",
    "⚠ its hires / detail / booster values are unreadable: those controls set to their defaults": "⚠ 它的高清修复 / 细化 / 增强参数读不出来：这些控件改成默认值",
    "⚠ possibly off-model or damaged:": "⚠ 可能不像角色或画坏了：", "⚠️ Previous LoRAs dropped (incompatible with new model)": "⚠️ 之前的 LoRA 已卸下（和新模型不兼容）",
    "⚡ No checkpoint is selected or loaded, so there is no family to look up; current settings kept.": "⚡ 没有选中或加载模型，查不到模型类别；保持当前设置。",
    "⚡ No measured recipe for": "⚡ 还没有实测配方：",
    "✅ Already loaded:": "✅ 已经加载：", "✅ Image sent to Upscale tab!": "✅ 图片已发送到放大页。",
    "✅ Image sent to img2img — img2img mode is on (see the open Image-to-Image section). Adjust Denoise Strength, then Generate.": "✅ 图片已发送到图生图 — 图生图模式已打开（见展开的“图生图”部分）。调好重绘幅度，再按生成。",
    "✅ LoRAs unloaded — model restored to base weights.": "✅ LoRA 已卸下 — 模型恢复到原始权重。",
    "✨ Auto: each picture gets the recipe for its own prompt (framing tags, a gesture pose, 🧍 pose control) when it is made — the recipe that ran is named under the picture and in its record.":
        "✨ 自动：每张图出图时按自己的提示词选配方（取景词、手势、🧍 姿势控制）— 实际用的配方会写在图片下方和它的记录里。",
    "✨ Polish Auto →": "✨ 精修 自动 →", "🎨 Colour strength": "🎨 色彩强度",
    "✋ Hand detail skipped: the hand detector couldn't be downloaded (retried in 5 min).": "✋ 跳过修手：手部检测器下载不了（5 分钟后重试）。",
    "👁 Eye detail skipped: it blurs Pony-family models (measured −28 % eye detail).": "👁 跳过修眼睛：它会让 Pony 类模型变糊（实测眼睛细节少 28 %）。",
    "👁 Eye detail skipped: the eye detector couldn't be loaded (retried in 5 min).": "👁 跳过修眼睛：眼部检测器加载不了（5 分钟后重试）。",
    "💡 No near-miss Danbooru tags found.": "💡 没有发现拼错的 Danbooru 标签。", "🧹 No duplicate tags.": "🧹 没有重复的标签。",
    "🔍 Hires fix skipped: img2img keeps the input picture's size (upscale it first, or use 🔬 Max detail).": "🔍 跳过高清修复：图生图保持输入图的尺寸（先放大它，或者用 🔬 最大细节）。",
    "❌ Could not read that LoRA.": "❌ 读不了这个 LoRA。", "❌ GPU Out of Memory! Try lowering resolution or batch size.": "❌ 显存不够！请降低分辨率或批量数。",
    "❌ Load a base model first.": "❌ 请先加载一个底模。", "❌ Loop already running. Stop it first before starting a new one.": "❌ 循环已经在运行。先停止它再开始新的。",
    "❌ Model failed to load.": "❌ 模型加载失败。", "❌ No model loaded.": "❌ 没有加载模型。", "❌ No model loaded. Pick a checkpoint and click Load Model (Generate tab).": "❌ 没有加载模型。请在生成页选模型再加载。",
    "❌ Pick a character card first.": "❌ 请先选一张角色卡。", "❌ Pick a checkpoint first.": "❌ 请先选一个模型。",
    "❌ FLUX model + SD 1.5 LoRA — not compatible.": "❌ FLUX 模型 + SD 1.5 LoRA — 不兼容。", "❌ FLUX model + SDXL LoRA — not compatible.": "❌ FLUX 模型 + SDXL LoRA — 不兼容。",
    "❌ SD 1.5 model + FLUX LoRA — not compatible.": "❌ SD 1.5 模型 + FLUX LoRA — 不兼容。", "❌ SD 1.5 model + SD 2.x LoRA — incompatible architectures.": "❌ SD 1.5 模型 + SD 2.x LoRA — 结构不兼容。",
    "❌ SD 1.5 model + SDXL LoRA — will likely crash or produce garbage.": "❌ SD 1.5 模型 + SDXL LoRA — 很可能崩溃或出乱码图。",
    "❌ SD 2.x model + SD 1.5 LoRA — incompatible architectures.": "❌ SD 2.x 模型 + SD 1.5 LoRA — 结构不兼容。", "❌ SDXL model + FLUX LoRA — not compatible.": "❌ SDXL 模型 + FLUX LoRA — 不兼容。",
    "❌ SDXL model + SD 1.5 LoRA — will likely crash or produce garbage.": "❌ SDXL 模型 + SD 1.5 LoRA — 很可能崩溃或出乱码图。",
    "Select a LoRA first.": "请先选一个 LoRA。", "not found locally": "本地没有找到", "Could not open folder:": "打不开文件夹：", "SmartSplit does not apply:": "多设备分流不适用于：",
    "Auto-added quality tags:": "自动加的质量标签：", "🔁 Re-applied LoRAs:": "🔁 重新加上的 LoRA：", "🧬 Moved from the prompt into the LoRA slots:": "🧬 从提示词移到 LoRA 槽里的：",
    "🌡 the rating paused": "🌡 打分时暂停了", "📝 a hand may be broken in": "📝 可能有手画坏了：", "📝 colour specks in": "📝 有彩色斑点：",
    "✅ faces match the card in all": "✅ 脸和角色卡全部吻合：", "✅ hair / eye colours match the card in all": "✅ 发色 / 瞳色和角色卡全部吻合：",
    "✅ hair / eye colours (WD14) and faces (CCIP) match the card in all": "✅ 发色 / 瞳色（WD14）和脸（CCIP）和角色卡全部吻合：",
    "💾 Saved card": "💾 已保存角色卡", "📌 Saved the profile of": "📌 已保存配置：", "⚠ A card called": "⚠ 已经有一张角色卡叫",
    "— 🎴 Load applies it whenever this checkpoint is selected.": "— 以后选中这个模型时，🎴 加载会用这份配置。",
    "✅ Settings saved as '": "✅ 设置已保存为 '", "❌ Could not load '": "❌ 读不出 '",
    "🔍 Hires fix:": "🔍 高清修复：", "🔬 Max detail: Real-ESRGAN 2× ·": "🔬 最大细节：Real-ESRGAN 2 倍 ·", "🧍 Composition control:": "🧍 构图控制：", "🧍 Pose control:": "🧍 姿势控制：",
    "📊 Grid of": "📊 网格：", "⏹ Stopped during": "⏹ 停在了：", "❌ Failed to load model:": "❌ 模型加载失败：", "❌ Generation failed:": "❌ 生成失败：",
    "❌ LoRA error:": "❌ LoRA 出错：", "❌ LoRA problem:": "❌ LoRA 出问题：", "❌ Tagger failed:": "❌ 打标签失败：", "⚠️ Cannot verify compatibility (model=": "⚠️ 没法确认兼容性（模型=",
    "❌ GPU Out of Memory (OOM)! Resolution": "❌ 显存不够！分辨率",
    "🧭 img2img is on at 0.4 (close to it: 0.2–0.3 closer, 0.5+ looser; Batch Size N = N variations). 🔁 Recreate runs its txt2img settings with its seed (another GPU and runtime: close, not identical). 💾 Save Settings under the name below keeps it as a base for txt2img.":
        "🧭 图生图已打开，幅度 0.4（0.2–0.3 更像原图，0.5 以上变化更大；批量数 N = N 个变体）。🔁 精确重现 = 用它的文生图参数和种子重画（不同显卡和运行环境：接近，不完全一样）。💾 按下面的名称保存设置，可以留作文生图的基础。",
}

TEMPLATES: list[tuple[str, str]] = [
    # the pipelines' result line (backend/sd_pipeline.py, sdxl_pipeline.py); a picture's own notes follow on the next lines
    ("Seed: {} | Steps: {} | CFG: {} | Scheduler: {} | Size: {} | Time: {}s (encode {}s + denoise {}s + VAE {}s) | Device: {} | Family: {}",
     "种子：{} | 步数：{} | CFG：{} | 采样器：{} | 尺寸：{} | 用时：{} 秒（编码 {} 秒 + 去噪 {} 秒 + 解码 {} 秒）| 设备：{} | 类别：{}"),
    ("Seeds: {} | Steps: {} | CFG: {} | Scheduler: {} | Size: {} | Time: {}s (encode {}s + denoise {}s + VAE {}s) | Device: {} | Family: {}",
     "种子：{} | 步数：{} | CFG：{} | 采样器：{} | 尺寸：{} | 用时：{} 秒（编码 {} 秒 + 去噪 {} 秒 + 解码 {} 秒）| 设备：{} | 类别：{}"),
    ("Seed: {} | Steps: {} | CFG: {} | Scheduler: {} | Size: {} | Time: {}s | Device: {}", "种子：{} | 步数：{} | CFG：{} | 采样器：{} | 尺寸：{} | 用时：{} 秒 | 设备：{}"),
    ("Seeds: {} | Steps: {} | CFG: {} | Scheduler: {} | Size: {} | Time: {}s | Device: {}", "种子：{} | 步数：{} | CFG：{} | 采样器：{} | 尺寸：{} | 用时：{} 秒 | 设备：{}"),
    ("img2img | Strength: {} | Seed: {} | Steps: {} | CFG: {} | Time: {}s", "图生图 | 幅度：{} | 种子：{} | 步数：{} | CFG：{} | 用时：{} 秒"),
    ("Could not open folder: {}", "打不开文件夹：{}"),
    ("SmartSplit does not apply: {} — switch SmartSplit off in Settings to use them.", "多设备分流不适用于：{} — 要用它们，请在设置里关掉多设备分流。"),
    ("SmartSplit | Seed:{} | Steps:{} | {}×{}", "多设备分流 | 种子：{} | 步数：{} | {}×{}"),
    ("VRAM: {} GB held by the model · peak {} of {} GB during this run", "显存：模型占用 {} GB · 这次最高用到 {} / {} GB"),
    ("{} {} detail failed on {} image(s): {}", "{} {} 细化在 {} 张图上失败：{}"),
    ("{} {} detail: no {}s found", "{} {} 细化：没找到 {}"),
    ("{} {} detail: {} {}(s) re-drawn ({}) · denoise {} · {}s", "{} {} 细化：重画了 {} 个 {}（{}）· 幅度 {} · {} 秒"),
    ("{} — saved the images from before it.", "{} — 之前画好的图已保存。"),
    ("— Batch {}{} done | Total images: {} | Running {}", "— 第 {}{} 批完成 | 共 {} 张 | 已运行 {}"),
    ("— Batch {}{} | Total images: {} | Running {} | Seed: {}", "— 第 {}{} 批 | 共 {} 张 | 已运行 {} | 种子：{}"),
    ("— {} ({} images saved in {})", "— {}（{} 张图，保存在 {}）"),
    ("— {} batch(es), {} images in {} (saved to outputs/)", "— {} 批，{} 张图，用时 {}（保存在 outputs/）"),
    ("ℹ {} leaves a little more grain in skies and backgrounds than DPM++ 2M AYS / Karras on SDXL models.", "ℹ 在 SDXL 模型上，{} 画的天空和背景比 DPM++ 2M AYS / Karras 颗粒感多一点。"),
    ("⏹ Stopped during {}", "⏹ 停在了：{}"),
    ("♻️ Seed {} (image {} of {}) — the first image of your next run will match it.", "♻️ 种子 {}（第 {} / {} 张）— 下次生成的第一张会和它一样。"),
    ("⚠ A card called {} already exists — type another name to build a new one.", "⚠ 已经有一张角色卡叫 {} — 想新建请换个名字。"),
    ("⚠ {} is an {} VAE and this is an {} checkpoint — using the checkpoint's own VAE (a mismatched VAE decodes garbage). Set VAE to none or pick a matching one.",
     "⚠ {} 是 {} 的 VAE，而这个是 {} 模型 — 改用模型自带的 VAE（不匹配的 VAE 会解出乱码）。请把 VAE 设为 none 或选一个匹配的。"),
    ("⚡ No measured recipe for {} yet; current settings kept.", "⚡ {} 还没有实测配方；保持当前设置。"),
    ("⚡ {}: {} — {}. Prompt, size, LoRAs and seed untouched.{}", "⚡ {}：{} — {}。提示词、尺寸、LoRA 和种子不变。{}"),
    ("✅ Already loaded: {}", "✅ 已经加载：{}"),
    ("✅ Fetched {} sidecar(s) | ⚠️ {} not on Civitai | ❌ {} errors", "✅ 取到 {} 个说明文件 | ⚠️ {} 个不在 Civitai 上 | ❌ {} 个出错"),
    ("✅ Loaded: {} (SD, {})", "✅ 已加载：{}（SD，{}）"),
    ("✅ Settings saved as '{}'", "✅ 设置已保存为 '{}'"),
    ("✅ faces match the card in all {} image(s) (CCIP)", "✅ 全部 {} 张图的脸都和角色卡吻合（CCIP）"),
    ("✅ hair / eye colours (WD14) and faces (CCIP) match the card in all {} image(s)", "✅ 全部 {} 张图的发色 / 瞳色（WD14）和脸（CCIP）都和角色卡吻合"),
    ("✅ hair / eye colours match the card in all {} image(s) (WD14)", "✅ 全部 {} 张图的发色 / 瞳色都和角色卡吻合（WD14）"),
    ("✨ Polish Auto → {}", "✨ 精修 自动 → {}"),
    ("✨ {}: {} — about {}× the time of the plain picture.{}", "✨ {}：{} — 用时约为普通出图的 {} 倍。{}"),
    ("❌ '{}' is a reserved name — choose another.", "❌ '{}' 是保留名称 — 请换一个。"),
    ("❌ Could not load '{}' ({}) — nothing was changed", "❌ 读不出 '{}'（{}）— 什么都没改"),
    ("❌ Failed to load model: {}", "❌ 模型加载失败：{}"),
    ("❌ GPU Out of Memory (OOM)! Resolution {}×{} with batch={} exceeded VRAM. Try reducing resolution (e.g. 512×512 or 512×768) or reducing batch size to 1.",
     "❌ 显存不够！分辨率 {}×{}、批量 {} 超出显存。请降低分辨率（例如 512×512 或 512×768）或把批量数改成 1。"),
    ("❌ GPU Out of Memory (OOM)! Resolution {}×{} with batch={} exceeded VRAM. Try reducing resolution (e.g. 768×768 or 832×1216) or reducing batch size to 1.",
     "❌ 显存不够！分辨率 {}×{}、批量 {} 超出显存。请降低分辨率（例如 768×768 或 832×1216）或把批量数改成 1。"),
    ("❌ Generation failed: {}", "❌ 生成失败：{}"),
    ("❌ LoRA error: {}", "❌ LoRA 出错：{}"),
    ("❌ LoRA problem: {}", "❌ LoRA 出问题：{}"),
    ("❌ Tagger failed: {}", "❌ 打标签失败：{}"),
    ("❌ {} outfits × {} seeds = {} images — keep it at 96 or fewer.", "❌ {} 套服装 × {} 个种子 = {} 张图 — 请控制在 96 张以内。"),
    ("⭐ {}/5 on average", "⭐ 平均 {}/5"),
    ("🌡 the rating paused {} s while the laptop cooled down", "🌡 打分时暂停了 {} 秒，让笔记本降温"),
    ("🎨 Colour strength {}", "🎨 色彩强度 {}"),
    ("🎴 {}: {} new + {} reused image(s) in {} — contact sheet {}", "🎴 {}：新画 {} 张 + 沿用 {} 张，用时 {} — 总览图 {}"),
    ("🏷 {} tags · character: {} · rating: {}", "🏷 {} 个标签 · 角色：{} · 评级：{}"),
    ("📊 Grid of {} images in {} — saved as {} (each cell is saved too)", "📊 {} 张图的网格，用时 {} — 保存为 {}（每一格也单独保存了）"),
    ("📝 a hand may be broken in {} picture(s) — look at {}; when this fires it is right about 9 times in 10, but it sees only about half of the broken hands; no star was taken",
     "📝 {} 张图里可能有手画坏了 — 看看 {}；这个提示大约十次有九次是对的，但只能发现大约一半画坏的手；没有扣星"),
    ("📝 colour specks in {} picture(s) — white lace, fur or (rarely) sampler noise: look at {}; no star was taken",
     "📝 {} 张图里有彩色斑点 — 可能是白色蕾丝、毛皮，偶尔是采样噪点：看看 {}；没有扣星"),
    ("🔀 Seed {} + variation strength {} — press Generate (a batch gives several variations at once).", "🔀 种子 {} + 变体强度 {} — 按生成（批量数大于 1 时一次出几个变体）。"),
    ("🔁 Re-applied LoRAs: {}", "🔁 重新加上的 LoRA：{}"),
    ("🔍 Hires fix: {}×{} → {}×{} · denoise {} · {} steps · {} · {}s", "🔍 高清修复：{}×{} → {}×{} · 幅度 {} · {} 步 · {} · {} 秒"),
    ("🔬 Max detail: Real-ESRGAN 2× · {}×{} → {}×{} · {}s", "🔬 最大细节：Real-ESRGAN 2 倍 · {}×{} → {}×{} · {} 秒"),
    ("🧍 Composition control: {} · strength {}", "🧍 构图控制：{} · 强度 {}"),
    ("🧍 Pose control: {} · {} · strength {}", "🧍 姿势控制：{} · {} · 强度 {}"),
    ("🧍 {} is not used for img2img — ignored.", "🧍 {} 不用于图生图 — 已忽略。"),
    ("{}⚠️ Cannot verify compatibility (model={}, LoRA={}) — check manually.", "{}⚠️ 没法确认兼容性（模型={}，LoRA={}）— 请自己检查。"),
    ("{}⚠️ {} LoRA on a {} checkpoint: it loads and runs, but characters and concepts often come out wrong (different face, extra features). Prefer a {} version of this LoRA or a {} checkpoint.",
     "{}⚠️ {} 的 LoRA 用在 {} 模型上：能加载能运行，但角色和概念经常画不对（脸不一样、多出东西）。最好用这个 LoRA 的 {} 版本，或换 {} 模型。"),
    ("{}✅ {} LoRA fits this {} model.", "{}✅ {} 的 LoRA 适合这个 {} 模型。"),
    ("Image {} selected — seed", "已选第 {} 张 — 种子"),
    ("(the whole prompt is its tags; outfits kept). Edit settings/characters/{}.json to fine-tune.", "（整段提示词作为它的标签；服装保留）。想细调就编辑 settings/characters/{}.json。"),
    ("from {} of {} pictures (no face, or under {} px, are skipped; they agree to {} ± {}). “🎴▶ Generate every outfit” now flags a face that is further from her than her own pictures are — about 1 in 30 of hers, up to 1 in 5 of a no-LoRA look-alike.",
     "来自 {} / {} 张图（没有脸或小于 {} 像素的跳过；它们彼此相似度 {} ± {}）。现在“🎴▶ 生成全部服装”会标出比她自己的图更不像她的脸 — 她本人的图约 1/30 会被标，没用 LoRA 的相似角色最多 1/5。"),
]


# ── the other tabs: guides and messages ──────────────────────────────────────────────────────────────────────
EXACT.update({
    "\"when the user types these words → produce this image.\"": "“用户写这些词时 → 画出这样的图。”", "(character LoRA on SDXL):": "（SDXL 上的角色 LoRA）：",
    "(dGPU DirectML).": "（独显 DirectML）。", ") in the positive prompt": "）写进正向提示词", ", freeing ~3 GB dGPU VRAM.": "，给独显省出约 3 GB 显存。",
    "(default) gives the cleanest result for almost all watermarks; OpenCV is instant but only suits thin text; SD repaints the area with your loaded model and may invent new details.":
        "（默认）几乎所有水印都去得最干净；OpenCV 瞬间完成但只适合细文字；SD 用已加载的模型重画那块区域，可能画出新的东西。",
    ", newest first. Search by prompt words, model, LoRA, seed or sampler (all words must match). Click an image for its settings; ⭐ marks favourites.":
        "，最新的在前。可以按提示词、模型、LoRA、种子或采样器搜索（所有词都要匹配）。点图片看它的参数；⭐ 表示收藏。",
    ", or an invented word like": "，或者自创的词，例如", "10–30 images of the concept (e.g., a specific item, location, composition).": "这个概念的 10–30 张图（例如某个物品、地点、构图）。",
    "15–50 images of the character in varied poses, angles, outfits, and backgrounds. Solo images work best — crop out other characters.": "这个角色的 15–50 张图，姿势、角度、服装、背景各不相同。单人图最好 — 把其他角色裁掉。",
    "30–200 images in the target art style. Consistent style matters more than quantity.": "目标画风的 30–200 张图。风格统一比数量重要。",
    ": Preserves character likeness best. Higher = more SDXL influence.": "：最能保住角色长相。越高 SDXL 的影响越大。", "API error:": "API 出错：", "API key removed.": "API 密钥已删除。",
    "After training completes:": "训练完成后：", "Alpha is too low relative to rank. Try alpha = rank × 2.": "alpha 相对秩太低。试试 alpha = 秩 × 2。", "Best practices:": "建议：",
    "Browse and download Stable Diffusion models and LoRAs from Civitai. An API key is required for NSFW content and higher download speeds.": "从 Civitai 浏览和下载 Stable Diffusion 模型和 LoRA。看 NSFW 内容和更快的下载速度需要 API 密钥。",
    "Character LoRA rule of thumb:": "角色 LoRA 的经验：", "Character LoRA:": "角色 LoRA：", "Character LoRAs:": "角色 LoRA：",
    "Check that you're using the trigger word in your prompt. Try weight 1.0. Make sure the base model matches the training base (e.g., don't use an Illustrious-trained LoRA on a Pony model).":
        "检查提示词里有没有写触发词。试试权重 1.0。确认底模和训练时的底模是同一类（例如不要把 Illustrious 训练的 LoRA 用在 Pony 模型上）。",
    "Check your internet connection (or try again — Civitai is sometimes slow).": "请检查网络（或者再试一次 — Civitai 有时会慢）。", "Civitai — models & LoRAs ↗": "Civitai — 模型和 LoRA ↗",
    "Colors / anatomy are broken:": "颜色 / 人体画坏了：", "Concept LoRA:": "概念 LoRA：", "Corner-watermark heuristic is still available.": "角落水印检测仍然可用。",
    "Could not open the folder:": "打不开文件夹：", "Could not parse version id.": "读不懂版本编号。",
    "Detect and remove watermarks, logos, and embedded text from images. Uses OCR scanning to find text regions plus an edge-density heuristic for corner logos.": "找出并去掉图片里的水印、logo 和文字。用 OCR 找文字区域，再用边缘密度检测角落里的 logo。",
    "Device dropdowns below are ignored in this mode.": "这个模式下会忽略下面的设备选择。", "DirectML split mode": "DirectML 分流模式", "Everything in": "这里列出",
    "For SD 1.5: same settings but resolution 512.": "SD 1.5：同样的设置，但分辨率用 512。", "Generated images are saved to": "生成的图片保存在",
    "How LoRA training works — the short version:": "LoRA 训练是怎么回事 — 简单说：", "Hugging Face models ↗": "Hugging Face 模型 ↗",
    "Images look exactly like training data:": "画出来和训练图一模一样：", "Inspect generation parameters embedded in PNG images, and transfer them directly to": "查看 PNG 图片里存的生成参数，并直接发送到",
    "Learning rate too high — try 5e-5. Or the training data had inconsistent quality.": "学习率太高 — 试试 5e-5。也可能是训练图质量参差不齐。", "LoRA has no visible effect:": "LoRA 看不出效果：",
    "LoRA only works at weight 1.0:": "LoRA 只有在权重 1.0 时才有效：", "No images in outputs/ yet.": "outputs/ 里还没有图片。", "No model at this index.": "这个位置没有模型。",
    "No models found — try fewer or different words, another type / base model, or a wider content rating.": "没找到模型 — 试试少写几个词或换词、换类型 / 底模，或放宽内容分级。",
    "No new key entered — the saved key (if any) stays. Type": "没有输入新密钥 — 已保存的密钥（如果有）保持不变。输入",
    "Overfitting — reduce epochs, lower learning rate, or add more training images.": "过拟合 — 减少轮数、降低学习率，或者多加训练图。",
    "Please upload an image first.": "请先上传一张图片。", "Please upload an image.": "请上传一张图片。", "Pro tip:": "小技巧：", "Quick-start recommendations": "快速上手推荐",
    "Rank 16 · Alpha 16 · LR 1e-4 · 300–800 steps (images × epochs: 16 images → 20–50 epochs) · Train text encoder ON · Cosine scheduler.": "秩 16 · Alpha 16 · 学习率 1e-4 · 300–800 步（图片数 × 轮数：16 张图 → 20–50 轮）· 训练文本编码器打开 · Cosine 调度。",
    "SD 1.5 LoRA": "SD 1.5 LoRA", "SD 1.5 → SDXL img2img Bridge": "SD 1.5 → SDXL 图生图桥接", "SDXL LoRA": "SDXL LoRA", "Same prompt": "同一个提示词", "Saved to:": "保存到：",
    "Saved: outputs/": "已保存：outputs/", "Select a model first.": "请先选一个模型。", "Select a version.": "请选一个版本。", "Selected as the checkpoint on the 🎨 Generate tab.": "已在 🎨 生成页选为模型。",
    "Send to Generate": "发送到生成页", "SmartSplit disabled.": "多设备分流已关闭。", "SmartSplit not available on this system.": "这台电脑用不了多设备分流。",
    "Stage 1 — SD 1.5 generates a base image": "第 1 阶段 — SD 1.5 画一张底图", "Stage 2 — SDXL refines via img2img": "第 2 阶段 — SDXL 用图生图精修", "Start Training": "开始训练",
    "Style LoRA:": "画风 LoRA：", "Text Encoder + VAE offloaded to": "文本编码器和 VAE 放到", "That file is gone.": "那个文件已经不在了。",
    "This type isn't used by ImageGen Studio — it's kept in": "ImageGen Studio 用不到这种类型 — 它被放在",
    "Token is empty.": "令牌是空的。", "Train your own LoRA (Low-Rank Adaptation) weights for SD 1.5 or SDXL / Pony / Illustrious. The trained LoRA is saved as a standard": "为 SD 1.5 或 SDXL / Pony / Illustrious 训练你自己的 LoRA。训练好的 LoRA 存成标准的",
    "Training flow:": "训练流程：", "Trigger word tips:": "触发词小贴士：", "Troubleshooting trained LoRAs:": "训练出的 LoRA 有问题时：", "UNet stays on": "UNet 留在",
    "Understanding epochs vs. steps:": "轮数和步数的区别：", "Upscale any image 2×–8× using Lanczos (CPU) or Real-ESRGAN (ONNX/PyTorch, auto-downloads ~67 MB model on first use).": "用 Lanczos（CPU）或 Real-ESRGAN（第一次用会自动下载约 67 MB 的模型）把任何图片放大 2–8 倍。",
    "Use it!": "用起来！", "What makes a good training dataset?": "怎样的训练集才好？", "Why captions matter:": "为什么标注很重要：", "ZLUDA (CUDA on AMD) ↗": "ZLUDA（在 AMD 上跑 CUDA）↗",
    "also include trait tags in each caption —": "每条标注也要写上特征标签 —", "comma-separated tags": "逗号分隔的标签", "natural language": "自然语言",
    "describe the content (not the style) so the model associates the trigger with the style, not specific subjects.": "描述内容（不是画风），这样模型会把触发词和画风联系起来，而不是和具体的东西联系起来。",
    "file describing its content.": "文件描述它的内容。", "file that works with the Generate tab, Civitai community LoRAs, and other tools like ComfyUI.": "文件，可以在生成页、Civitai 社区和 ComfyUI 等工具里使用。",
    "for SDXL stage. Lower than standalone SDXL since the base image provides structure.": "用于 SDXL 阶段。比单独用 SDXL 时低，因为底图已经定好了结构。",
    "for anime LoRAs (e.g.": "适合动漫 LoRA（例如", "for both stages (auto-applied). Adjust SDXL prompt if needed.": "两个阶段都用（自动加上）。需要时调整 SDXL 提示词。", "for other tools.": "给其它工具用。",
    "for realistic/photo LoRAs (e.g.": "适合写实 / 照片 LoRA（例如", "not found locally:": "本地没有找到：", "on any card above to load details and choose a version to download.": "点上面任意卡片，查看详情并选择要下载的版本。",
    "only those adapters": "只训练这些适配器", "over any extra areas you want removed. Auto-Detect finds text automatically; your brush marks are merged with it.": "涂在你还想去掉的地方。自动检测会找出文字；你涂的区域会和它合并。",
    "paint red": "用红色涂", "restores everything it was made with.": "恢复画它时用的所有参数。",
    "start with the trigger word, then list visual elements — hair color, eye color, outfit, pose, background.": "先写触发词，再列出看得到的东西 — 发色、瞳色、服装、姿势、背景。",
    "start with weight 0.7–0.8 and increase to 1.0 if the effect is subtle. If images look \"fried\" / oversaturated, lower the weight or reduce epochs.": "权重先用 0.7–0.8，效果不明显再加到 1.0。图看起来“过曝 / 过饱和”就降低权重或减少轮数。",
    "to remove it.": "可以删除它。", "to reproduce your images when the trigger word is used.": "在写触发词时画出你的图。",
    "— Use your SD 1.5 LoRAs to generate a base image, then automatically refine it with SDXL for higher detail and resolution. This is the best way to use SD 1.5 character LoRAs at SDXL quality.":
        "— 用你的 SD 1.5 LoRA 画底图，再自动用 SDXL 精修，得到更高的细节和分辨率。这是让 SD 1.5 角色 LoRA 达到 SDXL 质量的最好办法。",
    "— auto-loaded when generating.": "— 生成时自动加载。", "— go to the Generate tab, select your new LoRA from the dropdown, type your trigger word in the prompt, and generate!": "— 去生成页，在下拉框里选你的新 LoRA，在提示词里写触发词，然后生成！",
    "— loads the base model, injects LoRA adapters, pre-encodes all images with the VAE (latent caching), and pre-encodes all captions with the text encoder. This takes 1–5 min.":
        "— 加载底模，加入 LoRA 适配器，用 VAE 预先编码所有图片（潜空间缓存），用文本编码器预先编码所有标注。需要 1–5 分钟。",
    "— no ZLUDA required.": "— 不需要 ZLUDA。", "— so the model learns to associate the trigger with those features.": "— 这样模型会把触发词和这些特征联系起来。",
    "— the LoRA memorises your images pixel-for-pixel instead of learning the concept).": "— LoRA 会把图逐像素背下来，而不是学会这个概念）。",
    "— the training loop runs. You'll see loss values decrease over time. When done the LoRA is saved to": "— 开始训练。你会看到 loss 值慢慢下降。完成后 LoRA 保存到",
    "— ~12 GB RAM, ~10–30 sec/step → 30 images × 10 epochs ≈ 50–150 min": "— 约 12 GB 内存，每步约 10–30 秒 → 30 张 × 10 轮 ≈ 50–150 分钟",
    "— ~4 GB RAM, ~2–5 sec/step → 30 images × 10 epochs ≈ 15–25 min": "— 约 4 GB 内存，每步约 2–5 秒 → 30 张 × 10 轮 ≈ 15–25 分钟",
    "• Captions should be": "• 标注应该是", "• Each training image needs a": "• 每张训练图都需要一个", "• Go to": "• 去",
    "• Pick something that doesn't appear in normal English — e.g. a character name like": "• 选一个普通英文里不会出现的词 — 例如角色名",
    "• The model learns:": "• 模型学到的是：", "• The trigger word will be auto-prepended to every caption.": "• 触发词会自动加在每条标注的最前面。", "• Type your trigger word (e.g.": "• 写上你的触发词（例如",
    "• Your LoRA is saved as": "• 你的 LoRA 保存为", "→ select it from the LoRA dropdown": "→ 在 LoRA 下拉框里选它",
    "ℹ Free memory (RAM + page file) was short, so the Bridge held one model at a time; a larger page file avoids the reload.": "ℹ 可用内存（内存 + 页面文件）不够，所以桥接一次只放一个模型；加大页面文件可以避免重新加载。",
    "ℹ Stage 2 used the SDXL LoRAs still applied from the Generate tab:": "ℹ 第 2 阶段用了生成页上还加着的 SDXL LoRA：",
    "⏹ SD detail pass stopped — saved the plain upscale.": "⏹ SD 细节重画已停止 — 保存了普通放大的图。", "⏹ Stopped at Stage 1.": "⏹ 停在第 1 阶段。", "⏹ Stopped at Stage 2.": "⏹ 停在第 2 阶段。",
    "⏹ Stopped before picture": "⏹ 停在这张图之前：", "⏹ Stopped — nothing was saved.": "⏹ 已停止 — 什么都没保存。",
    "① Collect 10–200 images of a character / style / concept.": "① 收集一个角色 / 画风 / 概念的 10–200 张图。", "② Caption each image with a trigger word + description tags.": "② 给每张图写标注：触发词 + 描述标签。",
    "③ The trainer injects tiny \"adapter\" matrices into the frozen base model, then teaches": "③ 训练器在冻结的底模里加入很小的“适配器”矩阵，然后教会",
    "④ Result: a small .safetensors file (5–200 MB) you load in the Generate tab.": "④ 结果：一个小的 .safetensors 文件（5–200 MB），在生成页加载使用。",
    "⚠ EasyOCR not installed. Install with:": "⚠ 没有安装 EasyOCR。安装方法：", "⚠ It really runs: Text Encoder →": "⚠ 实际运行：文本编码器 →", "⚠ Load a model in Generate first, then re-apply.": "⚠ 先在生成页加载模型，再重新应用。",
    "⚠ Mask is empty — no regions to remove.": "⚠ 蒙版是空的 — 没有要去掉的区域。", "⚠ Memory & time estimates (CPU training):": "⚠ 内存和时间估计（CPU 训练）：", "⚠ No image loaded in PNG Info.": "⚠ 图片信息页没有载入图片。",
    "⚠ No regions detected. Try lowering the confidence threshold, or use the Manual Mask to draw regions.": "⚠ 没找到区域。试试降低置信度阈值，或者用手动蒙版自己涂。",
    "⚠ Saved to disk but HF login failed:": "⚠ 已存到硬盘，但 HF 登录失败：", "⚠ SmartSplit not supported for SDXL/Pony/Illustrious. Load an SD 1.5 model first.": "⚠ 多设备分流不支持 SDXL/Pony/Illustrious。请先加载一个 SD 1.5 模型。",
    "⚠ onnxruntime not found — LaMa unavailable.": "⚠ 找不到 onnxruntime — LaMa 用不了。",
    "✅ API key saved (persists across restarts) — NSFW/explicit results unlocked.": "✅ API 密钥已保存（重启后仍有效）— 可以看 NSFW / 露骨结果了。", "✅ Bridge complete!": "✅ 桥接完成！", "✅ Bridge complete! Stage 1:": "✅ 桥接完成！第 1 阶段：",
    "✅ Deleted preset '": "✅ 已删除预设 '", "✅ EasyOCR installed — text detection enabled.": "✅ EasyOCR 已安装 — 可以检测文字了。", "✅ HF token saved (persists across restarts).": "✅ HF 令牌已保存（重启后仍有效）。",
    "✅ LaMa ONNX available (auto-downloads ~100 MB on first use).": "✅ LaMa ONNX 可用（第一次使用会自动下载约 100 MB）。", "✅ Now using": "✅ 现在使用", "✅ Saved preset '": "✅ 已保存预设 '",
    "✅ Sent image to Upscale tab!": "✅ 图片已发送到放大页。", "✅ Sent image to img2img and enabled img2img mode!": "✅ 图片已发送到图生图，并打开了图生图模式。", "✅ Sent to Generate": "✅ 已发送到生成页",
    "✅ Watermark removed using": "✅ 去水印完成，用的是", "✏️ Manual mask only — draw red over the areas you want removed, then click Remove. Auto-detect is disabled.": "✏️ 只用手动蒙版 — 用红色涂上要去掉的地方，再点“去除”。自动检测已关闭。",
    "✨ SD detail pass (": "✨ SD 细节重画（", "✨ SD detail pass failed:": "✨ SD 细节重画失败：", "✨ SD detail pass skipped: load a model on the Generate tab first.": "✨ 跳过 SD 细节重画：请先在生成页加载模型。",
    "✨ SD detail pass skipped: over 16 MP would take too many tiles — use a smaller scale.": "✨ 跳过 SD 细节重画：超过 16 MP 要分太多块 — 请用小一点的倍数。",
    "❌ Civitai search failed:": "❌ Civitai 搜索失败：", "❌ Detection failed:": "❌ 检测失败：", "❌ Enter an existing folder.": "❌ 请填写一个存在的文件夹。", "❌ Failed to load SDXL model:": "❌ SDXL 模型加载失败：",
    "❌ Model reload failed:": "❌ 模型重新加载失败：", "❌ No PNG / JPG / WebP images in that folder.": "❌ 那个文件夹里没有 PNG / JPG / WebP 图片。", "❌ No SDXL model loaded. Select one above or load in the Generate tab.": "❌ 没有加载 SDXL 模型。请在上面选一个，或在生成页加载。",
    "❌ None of the": "❌ 一个都没有：", "❌ Pick an SD 1.5 model.": "❌ 请选一个 SD 1.5 模型。", "❌ Preset name cannot be empty.": "❌ 预设名称不能为空。", "❌ Prompt is empty — nothing to save.": "❌ 提示词是空的 — 没东西可存。",
    "❌ Removal failed:": "❌ 去除失败：", "❌ SmartSplit init failed:": "❌ 多设备分流初始化失败：", "❌ Stage 1 failed:": "❌ 第 1 阶段失败：", "❌ Stage 1 produced no images.": "❌ 第 1 阶段没有出图。",
    "❌ Stage 2 failed:": "❌ 第 2 阶段失败：", "❌ Turn on OCR and/or the corner heuristic — batch mode can only auto-detect.": "❌ 请打开 OCR 或角落检测 — 批量模式只能自动检测。", "❌ Unknown GPU selection.": "❌ 不认识这个 GPU 选择。",
    "❌ Upscale failed:": "❌ 放大失败：", "⭐ No favourites yet — star pictures in the list above first.": "⭐ 还没有收藏 — 先在上面的列表里给图片加星。", "🎓 LoRA Training Studio": "🎓 LoRA 训练工作室",
    "🎲 Picks for seed": "🎲 这个种子抽到的：", "📎 Companion embeddings (": "📎 配套的嵌入（", "🖌️ Upload an image, then": "🖌️ 上传一张图片，然后", "🧬 LoRA tags from the prompt:": "🧬 提示词里的 LoRA 标签：",
    "#️⃣ Model hash:": "#️⃣ 模型哈希：", "2× Real-ESRGAN": "2 倍 Real-ESRGAN", "Content rule:": "内容规则：", "Cool mode off": "降温模式关闭", "Custom: set the two sliders below.": "自定义：自己设置下面两个滑块。",
    "Explore: from round 2,": "探索：从第 2 轮起，", "Explore: off (every picture gets a new seed).": "探索：关闭（每张图都用新种子）。",
    "It is now listed in the LoRA slots of the 🎨 Generate tab — remember to add your trigger word to the prompt. The training model was released from memory (Prepare again to train another LoRA).":
        "它现在出现在 🎨 生成页的 LoRA 槽里了 — 记得在提示词里写触发词。训练用的模型已经从内存里卸下（要训练另一个 LoRA，请重新准备）。",
    "Latest run:": "最近一次运行：", "Let the trigger carry the character: left out of every caption (tag, captions):": "让触发词承载角色：从每条标注里省略的标签（标签，标注数）：",
    "LoRA rank=": "LoRA 秩=", "No generation metadata detected in this image.": "这张图里没有生成参数。", "One round ≈": "一轮约", "Pause while hot: off": "过热暂停：关闭",
    "The training model was released from memory — Prepare again to train.": "训练用的模型已经从内存里卸下 — 要训练请重新准备。", "Unloaded the Generate tab model(s) to free VRAM:": "为了腾出显存，卸下了生成页的模型：",
    "Use this folder for training:": "用这个文件夹训练：", "there for every picture.": "每张图都在那里。", "⏹ Stop signal sent — will finish current step": "⏹ 已发出停止信号 — 当前这一步完成后停下",
    "⏹ Stopping after the current saved setting (STOP written to": "⏹ 当前这个设置画完后停止（STOP 写入了", "⚠ A loop is already running (": "⚠ 已经有一个循环在运行（", "⚠ Pick or save a plan first.": "⚠ 请先选或保存一个计划。",
    "⚠ Saved settings not found:": "⚠ 找不到已保存的设置：", "⚠ no wait above": "⚠ 不会因温度等待：", "⚠ not a folder:": "⚠ 不是文件夹：", "✂️ CLIP skip:": "✂️ CLIP 跳层：", "✅ BLIP model freed": "✅ BLIP 模型已释放",
    "✅ Caption saved for": "✅ 已保存标注：", "✅ Cool mode: pause": "✅ 降温模式：暂停", "✅ Eye style:": "✅ 眼睛样式：", "✅ Negative hygiene": "✅ 反向提示词清理", "✅ Pausing above": "✅ 超过这个温度暂停：",
    "✅ Prompt weights:": "✅ 提示词权重：", "✅ Saved plan “": "✅ 已保存计划“", "✋ Hand detail:": "✋ 修手：", "✨ Face detail:": "✨ 修脸：", "❌ Click 📦 Prepare first.": "❌ 请先点 📦 准备训练。",
    "❌ Folder not found.": "❌ 找不到文件夹。", "❌ Image not found.": "❌ 找不到图片。", "❌ No image selected.": "❌ 没有选中图片。", "❌ No images found.": "❌ 没有找到图片。", "❌ No images.": "❌ 没有图片。",
    "❌ Scan the folder first.": "❌ 请先扫描文件夹。", "❌ Select a base model.": "❌ 请选一个底模。", "❌ Set dataset folder.": "❌ 请设置数据集文件夹。", "❌ Set folder first.": "❌ 请先设置文件夹。",
    "❌ Training failed:": "❌ 训练失败：", "🎨 Colour strength:": "🎨 色彩强度：", "👁 Eye detail:": "👁 修眼睛：", "🔬 Max detail:": "🔬 最大细节：", "🖼 img2img strength:": "🖼 图生图幅度：",
    "GPU training (ZLUDA) on an RX 6800M: SD 1.5 ~1.4 s per step (≈ 7 GB VRAM); SDXL ~6 s per step with the base model held in fp16 (≈ 10 GB VRAM, Prepare ~7 min); the first run on a PC compiles kernels for ~6 minutes. SDXL training needs a GPU with at least 11 GB.":
        "在 RX 6800M 上用显卡训练（ZLUDA）：SD 1.5 每步约 1.4 秒（约 7 GB 显存）；SDXL 每步约 6 秒，底模用 fp16（约 10 GB 显存，准备约 7 分钟）；一台电脑第一次运行要编译约 6 分钟内核。训练 SDXL 需要至少 11 GB 显存的显卡。",
    "If you have 30 images and set 10 epochs, the trainer runs 30 × 10 = 300 steps. Character LoRAs usually need several hundred to ~1000 steps. Each step processes one image. More images with fewer epochs often generalises better than few images with many epochs (which risks":
        "如果有 30 张图、设 10 轮，训练器就跑 30 × 10 = 300 步。角色 LoRA 一般要几百到约 1000 步。每一步处理一张图。图多轮少，通常比图少轮多学得更好（后者容易",
    "at 512×512 (or 512×768) using your SD 1.5 model + LoRA. The character/style from your LoRA is baked into this base.": "用 512×512（或 512×768）和你的 SD 1.5 模型 + LoRA。LoRA 里的角色 / 画风就画进了这张底图。",
    "at higher resolution (832×1216, etc.) with a low denoise strength (0.3–0.5). SDXL \"redraws\" the SD 1.5 output, adding its superior detail while preserving the character likeness from your LoRA.":
        "用更高的分辨率（832×1216 等）和较低的重绘幅度（0.3–0.5）。SDXL 把 SD 1.5 的图“重画”一遍，加上更好的细节，同时保住 LoRA 带来的角色长相。",
    "pictures. ‘~’ = a rough estimate with the heat pause (no loop run has timed that setting yet); settings that keep failing get parked and starred ones get more, so later rounds differ.":
        "张图。“~” = 包含过热暂停的粗略估计（还没有循环给这个设置计过时）；一直失败的设置会被搁置，加星的会多画，所以后面的轮次会不一样。",
    "⏹ Stop Loop": "⏹ 停止循环", "⏹ Stop after this picture": "⏹ 这张画完后停止", "📄 Open in PNG Info": "📄 在图片信息中打开",
})

# Gradio's own words in its upload boxes and editors (it would translate them only for a Chinese browser locale)
EXACT.update({
    "Drop Image Here": "把图片拖到这里", "- or -": "- 或 -", "Click to Upload": "点击上传", "Drop File Here": "把文件拖到这里",
    "Drop Video Here": "把视频拖到这里", "Drop Audio Here": "把音频拖到这里", "Upload": "上传", "Clear": "清除", "Undo": "撤销", "Redo": "重做",
    "Brush": "画笔", "Eraser": "橡皮擦", "Layers": "图层", "Crop": "裁剪", "Download": "下载", "Share": "分享", "Copy": "复制", "Copied!": "已复制",
    "Upload an image": "上传一张图片", "select the draw tool to start": "选择画笔工具开始画", "or": "或",
    "Built with Gradio": "由 Gradio 构建", "Clear canvas": "清空画布", "Upload button": "上传", "Paste button": "粘贴", "Image button": "图片", "Draw button": "画笔", "Erase button": "橡皮擦",
})

# static texts that stayed English in 中文 mode (the leftover check of 2026-11-03): headings, radio choices, guide fragments, Help cells.
# Technical names stay as they are (samplers, model families, file names, the tags you type, hardware names).
EXACT.update({
    # Generate: the tag panels, the pose box, the settings guide, the editor
    "✨ Quality": "✨ 质量", "👁 Fix Details": "👁 修细节", "💡 Lighting": "💡 光线", "🎨 Style Boost": "🎨 风格增强", "📐 Composition": "📐 构图",
    "🧹 Clean Up": "🧹 清理", "🔞 Enhance (R18)": "🔞 增强（R18）", "🎨 Style": "🎨 画风", "👤 Subject": "👤 主体", "🔞 NSFW Basics": "🔞 成人：基础",
    "💦 NSFW Acts": "💦 成人：动作", "🧬 Body": "🧬 身体", "🎭 Expression / Pose": "🎭 表情 / 姿势", "🎀 Outfit / Fetish": "🎀 服装 / 癖好",
    "🚫 Negative tags": "🚫 反向标签", "➕ Positive": "➕ 正向", "➖ Negative": "➖ 反向",
    "auto": "自动", "anime": "动漫", "photo": "照片", "Pose (skeleton)": "姿势（骨架）", "Composition (depth)": "构图（深度）",
    "Est. VRAM": "预估显存", "Setting": "设置项", "Recommended": "推荐值", "Notes": "说明", "Sampler": "采样器",
    # Bridge, Watermark, PNG Info, Civitai
    "Denoise 0.30–0.40": "重绘幅度 0.30–0.40", "Stage 2 — SDXL Refinement": "第 2 步 — SDXL 细化", "🗑️ Removal Settings": "🗑️ 去除设置",
    "Generate": "生成", "img2img": "图生图", ", or": "，或", "Upscale": "放大",
    "Drop or upload an image to inspect its metadata.": "拖入或上传一张图片，查看它带的生成参数。", "Click": "点击",
    # Train
    "🛏️ Body pillows & extreme aspect ratios:": "🛏️ 抱枕图和极端长宽比：", "Images are": "图片", "not": "不会",
    "forced to a square crop! This tool uses": "被强制裁成正方形！这个工具用", "aspect-ratio bucketing": "长宽比分桶",
    "— each image is assigned to a resolution bucket that matches its shape (e.g., a 16:5 body pillow → 960×320 bucket for SD 1.5, 1920×640 for SDXL). Only a tiny center crop happens if the exact ratio doesn't exist.":
        "——每张图会分到和它形状相符的分辨率桶里（例如 16:5 的抱枕图 → SD 1.5 用 960×320，SDXL 用 1920×640）。只有找不到完全相同的比例时，才会在中间裁掉一点点。",
    "Your wide/tall images are preserved.": "宽图和长图都会保留原样。", ") or": "），或",
    "How auto-captioning works:": "自动标注是怎么做的：",
    "BLIP (a vision-language model, ~1 GB) generates a natural-language description for each image. The trigger word is auto-prepended. You can then edit individual captions below. After captioning, click":
        "BLIP（一个看图说话的模型，约 1 GB）会给每张图写一句自然语言描述，并自动在最前面加上触发词。之后可以在下面逐条修改。标注完以后，点",
    "Free BLIP RAM": "释放 BLIP 内存", "to reclaim memory before training.": "，在训练前把内存腾出来。", "overfitting": "过拟合",
    "These defaults work well for most LoRAs. Only change them if you know what you're doing or want to experiment.":
        "这些默认值适合大多数 LoRA。除非你清楚自己在做什么，或者想做实验，否则不用改。",
    "Prepare": "准备",
    # Studio Loop
    "Render your": "把你的", "saved settings": "已保存的设置",
    "round after round without you at the computer (overnight), score every picture and sort it into best / ok / review / reject under":
        "一轮接一轮地出图，不用你守在电脑前（比如整夜），给每张图打分，并分进 best / ok / review / reject，位置在",
    ". Settings that keep failing are parked, settings whose pictures you ⭐ (here or in History) get more. Stops at the stop time, the hour budget, or ⏹ Stop.":
        "。一直失败的设置会被搁置，你加过 ⭐ 的（在这里或历史记录里）会多画。到停止时间、用完小时数，或按 ⏹ 停止时结束。",
    "Evening:": "晚上：", "pick a plan (or pick saved settings / ➕ a whole folder and 💾 save it) →": "选一个计划（或者选几个已保存的设置 / ➕ 整个文件夹，再 💾 保存）→",
    "(what is missing, time per round, how many rounds fit) →": "（缺什么、每轮多久、能跑几轮）→", "Morning:": "早上：",
    "🔄 Status → browse best / ok / review / reject → ⭐ the pictures you like →": "🔄 状态 → 浏览 best / ok / review / reject → 给喜欢的图加 ⭐ →",
    "in the run folder lists what each setting got wrong.": "（在这次运行的文件夹里）列出每个设置哪里出了问题。",
    "saved setting": "已保存的设置", "per round": "每轮", "per picture": "每张", "checkpoint": "模型", "(off)": "（关闭）",
    "best": "best 最好", "ok": "ok 还行", "review": "review 待看", "reject": "reject 淘汰",
    # Settings
    "⚖️ Prompt weights": "⚖️ 提示词权重", "A1111 (default)": "A1111（默认）", "Compel (stronger)": "Compel（更强）",
    "Norm-keep (default)": "Norm-keep（默认）", "A1111 (old)": "A1111（旧算法）", "👁 Eye style": "👁 眼睛样式",
    "Round pupils (default)": "圆瞳孔（默认）", "🧹 Negative hygiene": "🧹 反向提示词清理", "🌡 Power profile": "🌡 功耗档位",
    "Full power (Turbo, unlimited CPU): cool 1.5 + pause above 88 °C": "满功率（Turbo，CPU 不限功率）：降温 1.5 + 超过 88 °C 暂停",
    "Power-limited (USB-C charger, or Manual with a CPU limit): cool 0 + pause above 88 °C": "限功率（USB-C 充电器，或手动模式限制了 CPU 功率）：降温 0 + 超过 88 °C 暂停",
    "Custom (the two sliders below)": "自定义（下面两个滑块）", "🌡 Cool mode": "🌡 降温模式", "🖥 GPU": "🖥 显卡",
    "RDNA 4 — runs best on native ROCm (install_rocm.bat): about twice ZLUDA's speed on the RX 9070 XT.": "RDNA 4——用原生 ROCm 最好（install_rocm.bat）：在 RX 9070 XT 上大约是 ZLUDA 的两倍速度。",
    "Vega iGPU — limited compute; CPU fallback may be faster for large models.": "Vega 核显——算力有限；大模型用 CPU 可能更快。",
    "📂 Output folder": "📂 输出文件夹", "🔑 Hugging Face token (optional)": "🔑 Hugging Face 令牌（可选）", "📁 Installed models": "📁 已安装的模型",
    "🔗 Links": "🔗 链接", "🔀 SmartSplit — Multi-Unit Pipeline": "🔀 多设备分流（SmartSplit）", "SmartSplit capable:": "可以用多设备分流：",
    # Help: quick start and troubleshooting
    "1. Pick a checkpoint": "1. 选一个模型", "2. Write a prompt": "2. 写提示词", "3. Generate": "3. 出图", "4. Add LoRAs": "4. 加 LoRA",
    "5. Keep what you like": "5. 留下喜欢的", "6. Refine": "6. 精修", "Info line shows": "信息行显示", "⚠ VRAM over-committed": "⚠ VRAM over-committed（显存超量）",
    "Images are black or noisy": "图片全黑或满是噪点", "You restart the app": "重启 app 之后", "A model / LoRA says": "模型 / LoRA 提示",
    "incomplete": "incomplete（不完整）", "isn't a valid .safetensors file": "isn't a valid .safetensors file（不是有效的 .safetensors 文件）",
    "A Civitai download says": "Civitai 下载提示", "interrupted": "interrupted（中断）", "You pasted a prompt with": "粘贴的提示词里有",
    "tags": "标签", "You launch the app a second time": "再启动一次 app",
    # Help: prompt syntax, parameters, models
    "📏 token counter": "📏 词元计数", "⚠ warnings": "⚠ 警告", "🧹 Tidy prompts": "🧹 整理提示词", "Presets & tag chips": "预设和标签按钮",
    "SDXL / Illustrious / Pony: 5 – 7 (6 = the measured recipe)": "SDXL / Illustrious / Pony：5 – 7（6 = 实测配方）", "-1 (random)": "-1（随机）",
    "Width / Height": "宽 / 高", "🔍 Hires fix": "🔍 高清修复", "🔀 Variations": "🔀 变体", "✂️ CLIP skip": "✂️ CLIP 跳层", "✨ Face detail": "✨ 修脸",
    "✋ Hand detail": "✋ 修手", "👁 Eye detail": "👁 修眼睛", "🌡 Cool mode / pause while hot (Settings)": "🌡 降温模式 / 过热暂停（设置）",
    "✨ Polish (Generate tab)": "✨ 精修 Polish（生成页）", "⚖️ Prompt weights & colour bleeding": "⚖️ 提示词权重与串色", "🎲 Samplers on SDXL": "🎲 SDXL 上的采样器",
    "🔁 Studio Loop (unattended rounds)": "🔁 工作室循环（无人值守，一轮轮出图）",
    "🧭 Before you press Start: what a setting / a plan will do": "🧭 按开始之前：一个设置 / 计划会做什么",
    "📇 Pictures from another run: drag one in, get its recipe back": "📇 别处出的图：拖进来，拿回它的配方",
    "🔓 Where content is filtered, and who controls it": "🔓 内容在哪里过滤，由谁决定", "🎴 Every outfit of a card": "🎴 角色卡的每套服装",
    "✨ SD detail pass (Upscale tab)": "✨ SD 细节重画（放大页）", "⚡ Use the measured recipe": "⚡ 使用实测配方",
    "🧍 Pose control (Generate tab, under Colour strength)": "🧍 姿势控制（生成页，色彩强度下面）", "🔬 Sharpen eyes (Upscale tab)": "🔬 锐化眼睛（放大页）",
    "🎚 PAG / FreeU / CFG rescale": "🎚 PAG / FreeU / CFG 重缩放", "🔤 Danbooru tags": "🔤 Danbooru 标签", "🎲 Wildcards": "🎲 通配符",
    "🎴 Character cards": "🎴 角色卡", "🗑 Watermark remover": "🗑 去水印", "🏷 Interrogate (WD14)": "🏷 反推标签（WD14）", "📊 X/Y grid": "📊 X/Y 对比网格",
    "(not supported here)": "（这里不支持）",
    # Help: LoRA training
    "16 (standard)": "16（标准）", "8 (lightweight)": "8（轻量）", "32+ (complex styles)": "32+（复杂画风）", "= rank (e.g. 16)": "= rank（例如 16）",
    "Judge by": "看", "steps": "步数", "= images × epochs: 300–800 for a 16-image character": "= 图片数 × 轮数：16 张图的角色约 300–800 步",
    "(20–50 epochs)": "（20–50 轮）", "✅ ON for characters": "✅ 角色：开", "❌ OFF for styles": "❌ 画风：关",
    "automatic: base model in fp16, LoRA in fp32": "自动：基础模型用 fp16，LoRA 用 fp32", "attn (fast)": "attn（快）", "full (thorough)": "full（全面）", "✅ ON": "✅ 开",
    "Forgot trigger word, weight too low, or base model mismatch.": "忘了写触发词、权重太低，或者基础模型不匹配。",
    "Overfitting — too many epochs or LR too high.": "过拟合——轮数太多或学习率太高。", "LR too high or bad training data.": "学习率太高，或训练数据不好。",
    "Alpha too low relative to rank.": "Alpha 相对 rank 太低。", "Weight space drift between model versions.": "不同版本的模型之间权重空间有偏移。",
    "Aspect-ratio bucketing assigns each image to a resolution \"bucket\" that matches its shape. Only a tiny center crop (if any) is applied — your wide or tall images are preserved.":
        "长宽比分桶会把每张图分到和它形状相符的分辨率“桶”里。最多只在中间裁掉一点点——宽图和长图都会保留。",
    # Help: quality / negative / style / camera tag notes (the tag itself stays: it is what you type)
    "Top-tier overall quality signal.": "最高档的整体质量信号。", "Paired with masterpiece for strongest effect.": "和 masterpiece 一起用效果最强。",
    "Forces high detail rendering.": "强制画出高细节。", "Upscale-quality signal, more detail.": "高清质量信号，细节更多。",
    "Reduces blurriness. Good for portraits.": "减少模糊，适合人像。", "Richer contrast and color range.": "对比度和色彩范围更丰富。",
    "Adds cinematic texture.": "加上电影感的颗粒质感。", "Background blur effect.": "背景虚化效果。", "Dramatic professional lighting.": "有戏剧感的专业打光。",
    "Forces photorealistic output.": "强制写实照片效果。", "Realistic camera photo feel.": "真实相机拍照的感觉。", "General quality booster.": "通用的质量增强。",
    "More rendering detail everywhere.": "到处都画出更多细节。", "Forces micro-detail rendering.": "强制画出微小细节。",
    "Strongly discourages low-quality outputs.": "强力避免低质量的图。", "Discourages blurry/flat results.": "避免模糊 / 平淡的结果。",
    "Blocks mediocre mid-tier results.": "挡掉平庸的中等结果。", "Prevents compression artifacts.": "避免压缩瑕疵。", "Forces sharper output.": "让画面更清晰。",
    "Prevents soft/unfocused rendering.": "避免发虚 / 失焦。", "Common defect — extra fingers on hands.": "常见毛病——手指多了。",
    "Prevents finger count errors.": "避免手指数量出错。", "Prevents merged/webbed fingers.": "避免手指粘连 / 长蹼。", "Variant of the finger defect.": "手指毛病的另一种写法。",
    "Broad hand deformity prevention.": "大范围避免手部变形。", "Improves overall hand quality.": "整体改善手的质量。", "Improves face rendering quality.": "改善脸的画质。",
    "General anatomy errors.": "一般的人体结构错误。", "Prevents extra arms/legs.": "避免多出胳膊 / 腿。", "Prevents missing arms/legs.": "避免缺胳膊 / 少腿。",
    "Weighted deformity prevention.": "加权避免变形。", "Severe deformity prevention.": "避免严重变形。", "Prevents watermark text in image.": "避免图里出现水印文字。",
    "Prevents artist signature.": "避免画师签名。", "Prevents random text in image.": "避免图里出现乱七八糟的文字。", "Prevents cut-off compositions.": "避免构图被裁掉。",
    "Keeps subject within frame.": "让主体留在画面里。", "Japanese animation style.": "日本动画风格。", "Black and white comic style.": "黑白漫画风格。",
    "Photo-like realism.": "照片般的写实。", "CGI rendered look.": "CG 渲染的效果。", "Digital art illustration style.": "数字插画风格。",
    "Traditional oil painting texture.": "传统油画质感。", "Soft watercolor art style.": "柔和的水彩画风。", "Flat cartoon shading like anime game.": "像动漫游戏那样的平涂卡通上色。",
    "Professional design/fantasy art feel.": "专业设定稿 / 奇幻美术的感觉。", "Retro pixelated style.": "复古像素风。", "Tight crop on face/detail.": "紧贴脸部 / 细节的特写。",
    "Head and shoulders framing.": "头和肩膀的取景。", "Full body shot.": "全身照。", "Chest and above.": "胸部以上。", "Bird's-eye / top-down angle.": "鸟瞰 / 从上往下的角度。",
    "Upward angle, exaggerates legs/height.": "仰视角度，显得腿长、个子高。", "View from behind the character.": "从角色背后看。", "Tilted/dramatic camera angle.": "倾斜 / 有戏剧感的镜头角度。",
    "Environmental/landscape framing.": "带环境 / 风景的取景。", "Point-of-view shot.": "第一人称视角。", "Character makes eye contact with camera.": "角色看着镜头。",
    "Flushed cheeks.": "脸颊泛红。", "Hearts in eyes, love/ecstasy expression.": "眼里有爱心，表示喜爱 / 陶醉的表情。", "More specific heart pupil tag.": "更具体的爱心瞳孔标签。",
    # Help: hardware
    "Go to": "进入", "tab → check": "标签页 → 勾选", "CLIP-L and OpenCLIP-G are exported to ONNX once per model (cached in": "CLIP-L 和 OpenCLIP-G 每个模型导出一次 ONNX（缓存在",
    "Subsequent runs use the integrated GPU via DirectML (the app finds it by name — DML device order differs per PC). On a 780M: ~115ms total vs ~6.4s CPU =":
        "之后通过 DirectML 用核显运行（app 按名字找它——每台电脑的 DML 设备顺序不一样）。在 780M 上：共约 115 毫秒，CPU 约 6.4 秒 =",
    "56× speedup": "快 56 倍", "Quality is identical: cosine similarity >0.9998 vs CPU baseline.": "质量完全一样：和 CPU 结果的余弦相似度 >0.9998。",
    "Supports long prompts (>77 tokens): 75-token chunks cut at tag boundaries; BREAK starts a new one.": "支持长提示词（>77 个词元）：在标签边界切成 75 个词元一段；BREAK 另起一段。",
    "On a GPU backend (ZLUDA or native ROCm) it stays off and says why: the text encoders already run on the card (~0.5 s for a new prompt, a repeated one is cached), and the iGPU would save under 0.3 s while its export and sessions hold ~6 GB of memory. It is for the DirectML backend.":
        "在显卡后端（ZLUDA 或原生 ROCm）上它保持关闭并说明原因：文本编码本来就在显卡上跑（新提示词约 0.5 秒，重复的会缓存），核显只能省不到 0.3 秒，"
        "导出和会话却要占约 6 GB 内存。它是给 DirectML 后端用的。",
    "Black/noisy images or NaN errors after a driver, ROCm or ZLUDA update? Run": "驱动、ROCm 或 ZLUDA 更新后出现黑图、噪点或 NaN 错误？运行",
    "on native ROCm) in the": "用于原生 ROCm），位置在", "folder.": "文件夹。",
    "It compares GPU results for matrix multiply, convolution, attention and GroupNorm against the CPU and fails loudly if the GPU backend returns wrong numbers.":
        "它把显卡算出的矩阵乘法、卷积、注意力和 GroupNorm 结果和 CPU 对比，显卡后端算错时会明确报错。",
    "If generation suddenly becomes very slow, check the info line for": "如果出图突然变得很慢，看看信息行里有没有",
    "— Windows spills to system RAM instead of reporting out-of-memory. Lower the resolution or batch size.": "——Windows 显存不够时不会报错，而是溢出到系统内存。请降低分辨率或批量。",
    "Launch via": "用", "(ZLUDA or native ROCm must be active for dGPU detection).": "启动（要检测到独显，必须启用 ZLUDA 或原生 ROCm）。", "tab →": "标签页 →",
    "section.": "那一节。", "If it shows as": "如果显示", "capable": "可用", ", check": "，勾选", "Enable SmartSplit": "启用多设备分流",
    "Choose Text Encoder device (CPU / iGPU — or NPU when one is detected) and VAE device (CPU / iGPU).": "选择文本编码器设备（CPU / 核显——检测到 NPU 时也可以选 NPU）和 VAE 设备（CPU / 核显）。",
    "Apply SmartSplit Config": "应用分流配置", "Load any": "加载任意一个", "model — the SmartSplit pipe auto-builds.": "模型——多设备分流管线会自动建好。",
    "⚠️ SmartSplit is": "⚠️ 多设备分流", "not used for SDXL": "不用于 SDXL", "— SDXL uses the accelerated text encoding above instead.": "——SDXL 改用上面的加速文本编码。",
    "⚠️ Nothing is offloaded to a": "⚠️ 不会把任何阶段分给", "Vega-class iGPU": "Vega 级核显",
    "(Ryzen 2000–5000 and 7030-series APUs and the older A-series ones, e.g. an RX 6800M laptop's): its SD 1.5 VAE decode took 8 s against the RX 6800M's 0.5 s and then hung it (a driver reset of the GPU that drives the screen). SmartSplit and the accelerated text encoding say so and stay off there.":
        "（锐龙 2000–5000 和 7030 系列 APU，以及更早的 A 系列，例如 RX 6800M 笔记本里的那块）：它的 SD 1.5 VAE 解码要 8 秒，RX 6800M 只要 0.5 秒，"
        "之后还卡死了它（负责显示的那块显卡触发了驱动重置）。多设备分流和加速文本编码会说明原因并保持关闭。",
    # Help: glossary definitions (the 中文 column is the term's name)
    "The main model file (.safetensors / .ckpt). Everything else builds on top.": "主模型文件（.safetensors / .ckpt），其它一切都建立在它上面。",
    "Small add-on weight file that tweaks style/subject/behavior.": "小的附加权重文件，用来调整画风 / 主体 / 行为。",
    "Advanced LoRA variants with broader network coverage.": "覆盖网络更广的高级 LoRA 变体。", "Decoder that converts latents to pixels. Affects color.": "把潜空间数据解码成像素的解码器，会影响颜色。",
    "Latent space": "潜空间", "Compressed mathematical representation of images the model works in.": "模型工作时所用的、压缩过的图像数学表示。",
    "Denoising": "去噪", "The core SD process: iteratively remove noise to form an image.": "SD 的核心过程：一步步去掉噪声，形成图像。",
    "How strictly the model follows your prompt vs its own creativity.": "模型在多大程度上照你的提示词画，而不是自由发挥。",
    "Algorithm controlling how denoising steps proceed.": "决定去噪步骤怎么进行的算法。", "Number that initializes noise. Same seed + same settings = same image.": "用来生成初始噪声的数字。相同种子 + 相同设置 = 相同的图。",
    "Another type of fine-tuning stored as small vector files.": "另一种微调方式，保存成很小的向量文件。", "Safer model file format (.safetensors) replacing .ckpt/.pt.": "更安全的模型文件格式（.safetensors），取代 .ckpt / .pt。",
    "Stable Diffusion XL — higher resolution, two-stage model.": "Stable Diffusion XL——分辨率更高的两段式模型。",
    "Generate a new image using an existing image as starting point.": "以一张现有的图为起点生成新图。", "Generate an image from text prompts alone.": "只用文字提示词生成图片。",
    "Inpainting": "局部重绘", "Fill/edit masked regions of an existing image.": "填充 / 修改现有图片里被蒙版盖住的区域。",
    "Adds structural control (pose, depth, edges) to generation.": "给出图加上结构控制（姿势、深度、边缘）。", "Trigger word": "触发词",
    "Keyword required to activate a LoRA's trained content.": "激活 LoRA 所学内容需要的关键词。", "Parameter-Efficient Fine-Tuning — the library powering multi-LoRA.": "参数高效微调——支撑多 LoRA 的库。",
    "AMD GPU CUDA emulation layer — makes AMD GPUs work with CUDA code.": "AMD 显卡的 CUDA 模拟层——让 AMD 显卡能跑 CUDA 代码。",
    "Microsoft's vendor-neutral GPU API for any DX12 capable GPU.": "微软的通用显卡接口，支持所有能跑 DX12 的显卡。",
    "Portable model format used for NPU/iGPU inference in SmartSplit.": "可移植的模型格式，多设备分流用它在 NPU / 核显上推理。",
    "Core denoising network. The largest, most compute-intensive component.": "核心的去噪网络，最大、最耗算力的部分。",
    "Text encoder that converts your prompt into vectors the model understands.": "文本编码器，把提示词变成模型能理解的向量。",
    "SD1.5, 512×512, no LoRA": "SD1.5，512×512，不带 LoRA", "SD1.5, 768×768, + LoRA": "SD1.5，768×768，+ LoRA",
    "SDXL, 1024×1024, no LoRA": "SDXL，1024×1024，不带 LoRA", "SDXL, 1024×1024, + LoRA": "SDXL，1024×1024，+ LoRA",
    "SDXL, 1024×1024, 3× LoRA": "SDXL，1024×1024，3 个 LoRA", "✨ Polish": "✨ 精修", "Cowboy shot — hires": "Cowboy shot — hires（半身 + 高清修复）",
})

TEMPLATES += [
    ("API error: {}", "API 出错：{}"), ("Could not open the folder: {}", "打不开文件夹：{}"), ("Page {} / {} ({} results total)", "第 {} / {} 页（共 {} 个结果）"),
    ("Saved: outputs/{}", "已保存：outputs/{}"), ("page {} of {} · {} image(s)", "第 {} / {} 页 · {} 张"),
    ("{} of {} image(s) match.", "{} / {} 张图吻合。"), ("⏹ Stopped before picture {} of {}; the {} finished one(s) are saved.", "⏹ 停在第 {} / {} 张之前；画好的 {} 张已保存。"),
    ("⚠ It really runs: Text Encoder → {}, VAE Decode → {}. The console says why.", "⚠ 实际运行：文本编码器 → {}，VAE 解码 → {}。原因见控制台。"),
    ("⚠ Mask is empty — no regions to remove. {}", "⚠ 蒙版是空的 — 没有要去掉的区域。{}"), ("⚠ Saved to disk but HF login failed: {}", "⚠ 已存到硬盘，但 HF 登录失败：{}"),
    ("⚠ {} used the loaded model (no record or checkpoint not installed):", "⚠ {} 用了已加载的模型（没有记录，或者模型没安装）："),
    ("⚠ {} — used the loaded model ({}); eyes may look a little different", "⚠ {} — 用了已加载的模型（{}）；眼睛可能稍有不同"),
    ("✅ Bridge complete! Stage 1: {}", "✅ 桥接完成！第 1 阶段：{}"), ("✅ Deleted preset '{}'", "✅ 已删除预设 '{}'"),
    ("✅ Detected {} region(s). Adjust settings if needed, then click Remove.", "✅ 找到 {} 块区域。需要的话调整设置，再点“去除”。"),
    ("✅ Now using {} ({}). The model reloads on your next Generate.{}", "✅ 现在使用 {}（{}）。下次生成时会重新加载模型。{}"),
    ("✅ Saved preset '{}' ({})", "✅ 已保存预设 '{}'（{}）"), ("✅ Upscaled {} of {} images in {} →", "✅ 放大了 {} / {} 张图，用时 {} →"),
    ("✅ Watermark removed using {}. Saved to outputs/{}", "✅ 用 {} 去掉了水印。保存到 outputs/{}"),
    ("✅ {} cleaned, {} had nothing to remove (copied as-is) — {} →", "✅ 清理了 {} 张，{} 张没有可去的（原样复制）— 用时 {} →"),
    ("✨ SD detail pass ({}) · {}s", "✨ SD 细节重画（{}）· {} 秒"), ("✨ SD detail pass failed: {}", "✨ SD 细节重画失败：{}"),
    ("❌ Civitai search failed: {}", "❌ Civitai 搜索失败：{}"),
    ("❌ Detection failed: {}. Untick OCR (the first OCR run downloads its model, so it needs internet) or draw the areas with the brush.", "❌ 检测失败：{}。取消勾选 OCR（第一次运行 OCR 要下载模型，需要联网），或者用画笔自己涂。"),
    ("❌ Failed to load SDXL model: {}", "❌ SDXL 模型加载失败：{}"), ("❌ Model reload failed: {}", "❌ 模型重新加载失败：{}"), ("❌ Preset '{}' not found.", "❌ 找不到预设 '{}'。"),
    ("❌ Removal failed: {}", "❌ 去除失败：{}"), ("❌ SmartSplit init failed: {}", "❌ 多设备分流初始化失败：{}"), ("❌ Stage 1 failed: {}", "❌ 第 1 阶段失败：{}"),
    ("❌ Stage 2 failed: {}", "❌ 第 2 阶段失败：{}"), ("❌ Upscale failed: {}.{}", "❌ 放大失败：{}。{}"), ("❌ {} is not available through DirectML (install torch-directml).", "❌ {} 用不了 DirectML（需要安装 torch-directml）。"),
    ("❌ {}×{} at {}× would be {}×{} ({} MP) — too large. Use a smaller scale or a smaller image (up to ~64 MP output).", "❌ {}×{} 放大 {} 倍会变成 {}×{}（{} MP）— 太大了。请用小一点的倍数或小一点的图（输出最多约 64 MP）。"),
    ("🎲 Picks for seed {}:", "🎲 种子 {} 抽到的："), ("🏷 {} tags sent to Generate · character: {}", "🏷 {} 个标签已发送到生成页 · 角色：{}"),
    ("📎 Companion embeddings ({}): {}", "📎 配套的嵌入（{}）：{}"), ("🔬 Sharpened {} of {} images in {} →", "🔬 锐化了 {} / {} 张图，用时 {} →"),
    ("🔬 Sharpened: {}×{} → {}×{} · {} eye(s) re-drawn (denoise {}{}) · {}s", "🔬 已锐化：{}×{} → {}×{} · 重画了 {} 只眼睛（幅度 {}{}）· {} 秒"),
    ("({} pictures); time left {} → about", "（{} 张）；剩余时间 {} → 大约"), ("Content rule: {}; artifacts ({} tags) are rejected; the age guard is always on.", "内容规则：{}；瑕疵（{} 个标签）会被拒绝；年龄保护始终开启。"),
    ("Epochs: {} | Device: {}", "轮数：{} | 设备：{}"), ("Explore: from round 2, {} % of each setting's pictures are close variations of its best one so far.", "探索：从第 2 轮起，每个设置有 {} % 的图是它目前最好那张的近似变体。"),
    ("LoRA rank={} alpha={} | LR={} | TE={}", "LoRA 秩={} alpha={} | 学习率={} | 文本编码器={}"),
    ("Unloaded the Generate tab model(s) to free VRAM: {} — they reload automatically on your next Generate.", "为了腾出显存，卸下了生成页的模型：{} — 下次生成时会自动重新加载。"),
    ("{}. See Raw Parameters.", "{}。见“原始参数”。"), ("×{} · denoise {} · {} steps", "×{} · 幅度 {} · {} 步"),
    ("⏹ Stopping after the current saved setting (STOP written to {}).", "⏹ 当前这个设置画完后停止（STOP 写入了 {}）。"), ("⚠ A loop is already running ({}). Stop it first.", "⚠ 已经有一个循环在运行（{}）。请先停止它。"),
    ("⚠ Saved settings not found: {}", "⚠ 找不到已保存的设置：{}"), ("⚠ Saved {} °C, but this PC reports no thermal zone: nothing will wait.", "⚠ 已保存 {} °C，但这台电脑读不到温度：不会等待。"),
    ("⚠ no wait above {} °C: this PC reports no thermal zone", "⚠ 不会在 {} °C 以上等待：这台电脑读不到温度"), ("⚠ not a folder: {}", "⚠ 不是文件夹：{}"),
    ("✅ Caption saved for {}", "✅ 已保存 {} 的标注"), ("✅ Captioned {} images", "✅ 已标注 {} 张图"), ("✅ Cool mode: pause {}× the step time", "✅ 降温模式：每步后暂停该步时间的 {} 倍"),
    ("✅ Eye style: {}", "✅ 眼睛样式：{}"), ("✅ Found {} images", "✅ 找到 {} 张图"), ("✅ Negative hygiene {}", "✅ 反向提示词清理 {}"), ("✅ Pausing above {} °C (resume at {} °C)", "✅ 超过 {} °C 暂停（降到 {} °C 继续）"),
    ("✅ Prompt weights: {}", "✅ 提示词权重：{}"), ("✅ Saved plan “{}”", "✅ 已保存计划“{}”"), ("✅ Tagged {} images (trigger word first, then Danbooru tags){}", "✅ 已给 {} 张图打标签（先写触发词，再写 Danbooru 标签）{}"),
    ("✅ {} folder(s), {} pictures listed in their manifests", "✅ {} 个文件夹，清单里列了 {} 张图"), ("✅ {} images scanned ·", "✅ 扫描了 {} 张图 ·"), ("✅ {}: pause {}× the step time, {}", "✅ {}：每步后暂停该步时间的 {} 倍，{}"),
    ("❌ Training failed: {}", "❌ 训练失败：{}"),
    ("🧹 Merged: {}", "🧹 已合并：{}"), ("{} min", "{} 分钟"), ("{} min ~", "{} 分钟 ~"), ("{} round(s)", "{} 轮"),
    ("Shared RAM iGPU (~{} GB usable for ML).", "共享内存的核显（约 {} GB 可用于 AI）。"),
    ("Shared RAM iGPU (~{} GB usable for ML). Vega iGPU — limited compute; CPU fallback may be faster for large models.",
     "共享内存的核显（约 {} GB 可用于 AI）。Vega 核显——算力有限；大模型用 CPU 可能更快。"),
    ("NPU: {} (via bridge → Ryzen AI environment)", "NPU：{}（经桥接 → Ryzen AI 环境）"),
    (", ~{} pictures. ‘~’ = a rough estimate with the heat pause (no loop run has timed that setting yet); settings that keep failing get parked and starred ones get more, so later rounds differ.",
     "，约 {} 张图。“~” = 包含过热暂停的粗略估计（还没有循环给这个设置计过时）；一直失败的设置会被搁置，加星的会多画，所以后面的轮次会不一样。"),
    ("❌ Prompt S/R: “{}” isn't in the prompt.", "❌ Prompt S/R：提示词里没有“{}”。"),
    ("❌ Negative S/R: “{}” isn't in the negative prompt.", "❌ Negative S/R：反向提示词里没有“{}”。"),
    ('❌ {}: the first value is the words to find (then the replacements; "a, b" in quotes keeps a comma, "" removes the words).',
     '❌ {}：第一个值是要找的词（后面是替换成的词；带逗号的值用引号括起来，"" 表示删掉这些词）。'),
]
# generic templates last: the first template that matches wins, so a broad one must not shadow a specific one
TEMPLATES += [("Page {}", "第 {} 页"), ("Saved: {}", "已保存：{}"), ("✅ Loaded: {} ({}, {})", "✅ 已加载：{}（{}，{}）")]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def exact() -> dict[str, str]:
    return {_norm(k): v for k, v in EXACT.items()}


_SLOT = re.compile(r"\{(\d*)\}")


def to_regex(template: str) -> str:
    """'❌ Generation failed: {}' -> a JS / Python regex for the whole text fragment (the values are lazy groups)."""
    parts = _SLOT.split(_norm(template))
    out = []
    for i, p in enumerate(parts):
        out.append(re.escape(p).replace("/", "\\/") if i % 2 == 0 else "([\\s\\S]*?)")
    return "^" + "".join(out) + "$"


def to_replacement(zh: str) -> str:
    """'❌ 生成失败：{}' -> '❌ 生成失败：$1' ({} in order, or {0}, {1} … to reorder)."""
    n = [0]

    def slot(m):
        if m.group(1):
            return f"${int(m.group(1)) + 1}"
        n[0] += 1
        return f"${n[0]}"
    return _SLOT.sub(slot, zh.replace("$", "$$"))


def patterns() -> list[tuple[str, str]]:
    return [(to_regex(en), to_replacement(zh)) for en, zh in TEMPLATES]


def translate(text: str) -> str | None:
    """What the page shows in Chinese for one text fragment (the same rules as the page's script; tests use it)."""
    t = _norm(text)
    ex = exact()
    if t in ex:
        return ex[t]
    for rx, rep in patterns():
        m = re.match(rx.replace("\\/", "/"), t)
        if m:
            return re.sub(r"\$(\d+)", lambda g: m.group(int(g.group(1))), rep.replace("$$", "\x00")).replace("\x00", "$")
    return None
