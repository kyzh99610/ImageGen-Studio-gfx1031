# 🦞 ImageGen Studio

A local AI image generation suite built for **AMD GPUs via ZLUDA v6 (ROCm/HIP)**, with GPU VAE decoding, **SmartSplit multi-device offload**, optional iGPU text encoding, and a built-in **Civitai Hub** for model discovery. Runs on Windows with AMD RDNA 2/3/4 dGPUs (primary target: RX 6800M, gfx1031), with optional iGPU and XDNA NPU support.

> Developer / AI-assistant notes live in [`../AGENTS.md`](../AGENTS.md).

---

## Getting started

Installation, launching (browser or desktop window), launch options, the GPU self-test, performance numbers,
supported GPUs and a feature overview are in the [main README](../README.md). This page describes each tab in
detail. (Don't also install plain `onnxruntime` — it and `onnxruntime-directml` write to the same folder;
re-run `install.bat` any time to repair packages.)

PowerShell users can also launch with `launch.ps1`:

```powershell
.\launch.ps1 -Port 8080 -Share   # custom port, public Gradio share link
.\launch.ps1 -Dml                # DirectML instead of ZLUDA
.\launch.ps1 -Cpu                # force CPU mode
```

`launch.bat --share` also creates a public Gradio link.

---

## Tabs

### 🎨 Generate

Generate images from text prompts using Stable Diffusion checkpoints. The output gallery sits right next to the prompt, so you never scroll to see results.

**Just pick and press Generate** (or **Ctrl+Enter**):
- The selected **checkpoint and VAE load automatically** — "Load Model" only preloads.
- **A1111 weighting works**: `(tag:1.2)`, `((tag))` and `[tag]` are translated for the Compel encoder (which otherwise read them as plain words); Compel's own `(tag)1.2` / `tag++` work too. `BREAK` acts as a separator, and where Compel can't run (DirectML text encoder) the weights are dropped instead of being read as words.
- **Working in another tab?** When a generation, Bridge run, upscale or download finishes while the app is in the background, the tab title changes to “✅ Done” until you come back.
- **Pasting a Civitai / A1111 prompt?** Its `<lora:name:0.8>` tags are moved into free LoRA slots (with their weights) and removed from the prompt.
- The **LoRAs shown in the three slots are applied automatically** with their current weights. Pick "none" or press ✖ to drop one. Every checkpoint and LoRA in the lists is labelled **SD 1.5** or **SDXL** (read from the file itself) and the LoRAs that fit the chosen checkpoint are listed first, all three slots are checked against the checkpoint as soon as you pick either one (LyCORIS LoHa / LoKr files work too — the app fuses them itself, since diffusers can't load them), and a LoRA made for the other model family (e.g. an SDXL LoRA on SD 1.5) is refused with a plain-English message.
- Switching between SD 1.5 and SDXL moves Width × Height to that family's native size. **Size presets** and **⇄ Swap** are under the size sliders.

**Seeds you can reuse:** every PNG stores the seed that was actually used, including random ones. In a batch, images get seed, seed+1, … so each is reproducible on its own. **♻️ Last seed** puts it back; **🎲** returns to random.

**Every image can be made again:** each PNG records the checkpoint, VAE, LoRAs with their weights, prompts, sampler, steps, CFG, size and seed (A1111-compatible text that Civitai/Forge read, plus an exact record for this app, including the A1111 model hash). **Drop it into Image-to-Image** and all of that comes back (models are found by file name, or by hash if you renamed them). Untick "Use img2img mode" and Generate to recreate it, or keep it ticked to make variations. The GPU isn't bit-exact, so a recreated image differs by under ~2/255 per pixel.

**Hires fix:** generate at the model's native size (good composition), then upscale ×1.5–2 and let the model re-draw details at the big size. SD 1.5 512×768 → 768×1152 takes about 30 s; SDXL 832×1216 → 1248×1824 about 2 min, and fits in 12 GB. Choose Lanczos, or Real-ESRGAN for sharper line art.

**✨ Face detail (ADetailer-style):** finds faces (a YOLO anime-face detector; photo mode uses OpenCV's face cascades) and redraws each one at the model's full resolution with low denoise. Small faces in full-body or group shots get proper eyes and mouths, at about 5–15 s per face on SD 1.5 and ~30 s on SDXL. You can add an extra face-only prompt. **✋ Also re-draw hands** does the same for hands (anime hand detector, gloves too) before the faces — it cleans up smudged fingers but can't reliably fix a finger count.

**🎴 Generate every outfit:** in the character-card panel, renders each outfit of the card × N seeds with the current settings (the same seeds for every outfit) and finishes with a labelled contact sheet. Pairs already made with the same settings are skipped, so a stopped run picks up where it left off.

**🗂 History tab:** every image in `outputs/`, newest first, searchable by prompt words, model, LoRA, seed or sampler, with ⭐ favourites; **📄 Open in PNG Info → Send to Generate** restores how an image was made. The index reads only PNG text chunks (≈ 8 s for 2,000 images the first time, instant afterwards) and keeps small thumbnails in `outputs/.thumbs/`.

**✨ SD detail pass (Upscale tab):** after the upscale, the image is re-drawn in overlapping native-size tiles at low denoise with the model loaded on the Generate tab — real detail where hires fix stops (1536 / 2048 px). It uses the image's own prompt when it has one.

**🖌 Inpaint:** paint over any part of an image and describe what should be there. Only the painted area changes, and it's redrawn at native resolution, so hands and faces get full detail. It uses your current model, LoRAs, sampler and seed.

**🎚 Samplers & quality boosters:**
- **DPM++ 2M AYS** (NVIDIA's *Align Your Steps* schedule) gets about the quality of 25 regular steps in 10–12. On SDXL that's ~17 s instead of ~35 s.
- The *Karras* samplers now really use Karras sigmas, as in A1111 and Forge.
- **PAG** (perturbed-attention guidance, 2–3) gives cleaner structure, line art and backgrounds, at ~1.5–2× the time.
- **FreeU** adds detail and contrast for free. Its settings are tuned for anime models: the paper's values over-saturated them.
- **CFG rescale** tames burned colours at high CFG.
- **V-prediction** checkpoints (e.g. NoobAI-XL v-pred) are detected and set up automatically, with CFG rescale 0.7.
- The X/Y grid has *PAG scale* and *CFG rescale* axes for comparisons.

**🔤 Danbooru tags:** anime checkpoints learned exact Danbooru tags.
- While you type, the most-used matching tags appear as chips (e.g. `thigh` → thighhighs 1.0M, thighs 466k…); click one to complete it.
- Near misses are pointed out under the prompt, like *long haired → long hair* or *thigh highs → thighhighs*.
- **💡 Fix Danbooru spellings** corrects them all and keeps your weights.

**🎴 Character cards:** save everything that makes a character come out right: checkpoint, LoRAs and weights, the character's tags, named outfits, negative prompt, size, CFG, steps, sampler and CLIP skip. Loading one sets all of it at once, with the outfit you pick and any scene tags you add. **🧩 Build from LoRA slot 1** makes a card from a character LoRA: its trigger words and hair/eye tags become the character, and each of the creator's example prompts on Civitai becomes an outfit. Cards are JSON files in `settings/characters/`.

**🎲 Wildcards:** `{smile|pout|grin}` picks one option per image, `{2$$a|b|c}` picks two, and `{3::a|b}` makes *a* three times likelier. `__outfit__` picks a random line from `wildcards/outfit.txt`. Starter lists ship for outfit, pose, expression, background, hair, eyes, lighting and camera; add your own `.txt` files there or in `models/wildcards/`. Each image of a batch gets its own picks from its own seed, so a seed gives the same picks again. The resolved prompt is saved in the image (and the template beside it), and **👁 Preview** shows four sample picks.

**🏷 Interrogate (WD14):** the WD14 anime tagger (wd-vit-tagger-v3, ~380 MB, downloaded on first use) reads an image and writes its Danbooru tags into the prompt. It's in the img2img section and in PNG Info, and runs on the CPU in about a second. In **Train LoRA**, *Auto-Tag All* captions a whole dataset with it (trigger word first). For anime training sets this works better than BLIP's sentences.

**Variations & CLIP skip:** keep a seed you like and add a *variation seed* at 0.05–0.2 strength for the same picture with small changes. **🔀 More like this** under the gallery sets that up for the image you picked. **CLIP skip 2** is what most SD 1.5 anime checkpoints expect.

**📊 X/Y grid:** compare two settings side by side with one seed. Axes: CFG, steps, sampler, seed, LoRA 1 weight, CLIP skip, hires denoise, checkpoint, or *Prompt S/R* (e.g. `red hair, blue hair, green hair`). You get one labelled grid, and every cell is also saved as a normal image.

**LoRA keywords you can click:** pick a LoRA and its keywords appear as chips under the slots. They include Civitai trigger words (🔑) and the creator's full prompts, e.g. one per outfit (📋). There are also the tags from its training captions with how many training images had them (%); the higher the share, the more strongly the LoRA ties that tag to its character. Likely triggers are marked 🗝: name-like tags in almost every training image, even when Civitai lists another spelling. **✨ Add character tags** adds the triggers plus every tag in at least half of the training images, right after your quality tags.

**Long prompts and duplicates:** a counter under the prompts shows tokens and 75-token chunks, and where each chunk starts. Nothing is cut off, and a tag is never split between chunks. `BREAK` starts a new chunk. It warns when a LoRA trigger word slipped out of the first chunk, or when a tag is in both the prompt and the negative prompt. Presets, quick tags and chips never add a tag twice: the stronger weight wins. **🧹 Tidy prompts** cleans up duplicates you typed yourself.

**Your session comes back:** prompt, negative prompt, model, VAE, LoRA slots and settings are restored the next time you start the app (the seed always starts random).

Also:
- **txt2img** and **img2img** (🖼 Send to img2img opens the img2img section with the image loaded)
- **14 samplers**: DPM++ 2M Karras, DPM++ 2M, DPM++ 2M SDE Karras, DPM++ 2M AYS, DPM++ SDE (Karras), Euler a, Euler, Euler AYS, DDIM, PNDM, LMS, Heun, UniPC
- **Batch generation** (1–8 images) and **Auto-Loop** for continuous generation with a configurable delay
- **Auto-add quality tags** checkbox — adds the quality tags each model family was trained with, plus matching negative tags: Illustrious/NoobAI `masterpiece, best quality, amazing quality, very aesthetic, absurdres`, Pony `score_9, score_8_up, score_7_up`, SD 1.5 `masterpiece, best quality`. Only added when your prompt has none; the exact prompt used is shown and saved
- **Prompt presets**, **My Saved Prompts**, **Save / Load Settings** (restores the checkpoint too), **Quick Tags**
- **Compel long-prompt encoding** with weighted/blended syntax `(word:1.3)`, `(word1|word2)`
- **📂 Show in folder** under the gallery opens Explorer with the selected image highlighted
- **Accelerated SDXL text encoding** (optional, Settings tab): CLIP-L and OpenCLIP-G on the integrated GPU via ONNX DirectML — mainly useful for the DirectML backend

Every setting has a one-line hint; the **📖 Settings Guide** accordion has the full cheat sheet.

### 🌉 SD→SDXL Bridge

Generate a base image with an **SD 1.5** model (and the SD 1.5 LoRAs you applied to it in Generate), then refine it with an **SDXL / Pony / Illustrious** model via img2img. Both model pickers list only the matching family and load automatically. On 12 GB cards the SD 1.5 model is freed before the SDXL stage (both plus SDXL's working memory don't fit, and Windows would spill to system RAM); it reloads with its LoRAs at the start of the next run, which costs a few seconds. The same seed is used for both stages and saved in the PNGs. `<lora:name:0.8>` tags in its prompt are applied to the SD 1.5 model (SDXL LoRAs are skipped with a note). Its settings (models, prompts, steps, sizes, strength — not the seed) are remembered for the next launch.

### 🔍 Upscale

Upscale images by 2×, 4×, or 8× using Lanczos (CPU), Real-ESRGAN ONNX (DirectML on the discrete GPU — ~2× faster than CPU), or Real-ESRGAN PyTorch. The GPU works in 256 px tiles so no single job trips Windows' ~2 s GPU watchdog; if the GPU ever returns a blank image the upscale is redone on the CPU automatically. The info line says which device was used. On the GPU the app makes a one-time fp16 copy of the model (~1.5× faster: 832×1216 → 2× in ~35 s instead of ~55 s, visually identical). Results are saved next to their source (`<source>_4x.png`, never overwriting) with the source's generation settings kept. **📁 Batch** upscales a whole folder into `outputs/upscaled_<folder>/`.

### 📄 PNG Info

Drop any image made by this app, A1111/Forge or Civitai to see its prompt and settings. **🚀 Send to Generate** fills in prompt, negative, sampler, steps, CFG, size, seed and checkpoint, and puts its VAE and LoRAs into the slots with their weights — both `<lora:name:0.8>` prompt tags and this app's `LoRAs:` field are understood; the tags are taken out of the prompt, and LoRAs you don't have locally are listed.

### 🗑️ Watermark Remover

Remove watermarks, logos, and overlaid text from images:

- **Auto-Detect** — EasyOCR (on the GPU under ZLUDA) finds text regions; corner heuristic catches corner logos. Detected regions are expanded to include surrounding graphic elements (borders, icons) and merged when close together.
- **Draw-on-image** — Paint red brush strokes directly on the image to mark additional regions for removal.
- **OCR-off mode** — Toggle OCR detection off to use draw-only selection, keeping desired text intact (useful for emoji, manga, etc.).
- **Auto-downscale** — Large images (>2048px) are automatically downscaled for the editor canvas. Full resolution preserved in output.
- **Removal methods:**
  - *LaMa ONNX* (default) — the cleanest result for almost every watermark (~7 s). Auto-downloads ~100 MB model on first use. CPU only (DML crashes on this model's MatMul ops).
  - *OpenCV TELEA/NS* — instant, but only good for thin text; large areas get smeared
  - *SD Inpainting* — repaints a crop around the mask with the model loaded in Generate (SD 1.5 or SDXL, ~10 s) and blends it back; creative, so it can invent objects
- **Poisson seamless blending** — Inpainted regions use `cv2.seamlessClone` for gradient-matched compositing; falls back to feathered Gaussian blend at image edges.
- **📁 Batch** — clean a whole folder (auto-detect + the chosen method) into `outputs/cleaned_<folder>/`; images where nothing is found are copied unchanged. Handy for LoRA training sets.
- **Saved automatically** — as `<source>_clean.png` when the image came from this app (else `dewatermark_<time>.png`); earlier results are never overwritten.

### 🌐 Civitai Hub

Browse, search, and download Stable Diffusion models and LoRAs from Civitai:

- **Filters**: type (Checkpoint, LoRA, LoCon/LyCORIS, DoRA, VAE, Upscaler), base model, content rating (SFW/NSFW/XXX)
- **Sort**: Most Downloaded, Highest Rated (by thumbs up), Newest
- **Pagination**: fetches up to 100 results per search, displayed 20 per page with ◀ Previous / Next ▶ navigation
- **Download progress**: live speed (MB/s), elapsed time, ETA, and completion summary
- **Safe downloads**: files are written as `name.part` and renamed only when complete and the right size — an interrupted download never leaves a broken model behind
- **Interrupted downloads resume**: if the connection drops, the partial file is kept and pressing Download again continues where it stopped (a cancelled download is discarded).
- **Where files go**: checkpoints, LoRA/LoCon/LyCORIS/DoRA, VAE, embeddings and upscalers to their folders; types the app can't use (motion modules, poses, wildcards…) to `models/other/<type>/`, so they never clutter the model lists
- **Auto-refresh**: downloaded models instantly appear in the Generate tab (checkpoint, VAE and all three LoRA slots);
  your first downloaded checkpoint is also selected for you

An API key (free from civitai.com → Account settings → API Keys) is required for NSFW/explicit content and some downloads; the app tells you when a download was refused for lack of one.

### 🎓 Train LoRA

Train your own LoRA adapters from scratch:

1. **Dataset Preparation** — Pick a folder from `training_datasets/` (or paste any path). Scans, assigns aspect-ratio buckets, resizes to training resolution. The resolution follows the base model you choose (SD 1.5 → 512, SDXL → 1024).
2. **Auto-Captioning** — BLIP generates initial captions. Built-in per-image caption editor with trigger word auto-prepend.
3. **Training** — Latent caching, live text encoder training, gradient checkpointing, cosine LR scheduling. Progress bar with loss tracking and ETA.
4. **Output** — Standard kohya-format `.safetensors` saved to `models/loras/` and listed in the Generate tab's LoRA slots right away. An existing LoRA with the same name is never overwritten (`name_2`, `name_3`, …).

Supports SD 1.5 and SDXL / Pony / Illustrious. GPU training via ZLUDA (SD 1.5, 25 images, 1 epoch ≈ 5 min on an RX 6800M); CPU training as safe fallback. Any model loaded in the Generate tab is unloaded first to free VRAM — it reloads on your next Generate.

### ⚙️ Settings

- GPU selection and hardware detection — pick the dGPU (ZLUDA) or the iGPU (DirectML); the model reloads on your next Generate
- **Accelerated SDXL Text Encoding** toggle (iGPU via DirectML)
- **SmartSplit** multi-device pipeline configuration (SD 1.5 only)
- Civitai API key and HuggingFace token management (persisted across restarts)
- Model inventory and VRAM estimation

### ❓ Help

Full bilingual (English/中文) reference guide covering prompt syntax, generation parameters, model architectures, LoRA/VAE usage, LoRA training, quality/style tag cheatsheets, SmartSplit & hardware, and a glossary.

---

## GPU / Acceleration Architecture

### ZLUDA v6 (Primary — dGPU)

ZLUDA v6 (downloaded by setup into `ZLUDA_v6\zluda\`) intercepts NVIDIA CUDA calls at runtime, forwarding them to AMD HIP/ROCm. Text encoders, the UNet (main denoising network) and the VAE decoder all run on the dGPU via ZLUDA.

**Requirements:**
1. **ROCm runtime** — AMD HIP SDK 6.x (6.4 tested); an optional `therock_sdk\` folder is preferred when present
2. **ZLUDA v6** — `ZLUDA_v6\zluda\`, injected via `launch.bat` (ZLUDA v5 fails every GPU operation on current drivers)
3. **gfx1031 only:** the community rocBLAS kernels in `_gfx1031_rocm624\` (AMD's HIP SDK has none for gfx1031)
4. **PyTorch CUDA 11.8 wheel** — Standard CUDA build; ZLUDA translates at runtime

**Why cuDNN is off:** under ZLUDA on RDNA 2 (gfx1031), AMD's MIOpen convolution library returns wrong results — NaN latents or garbage VAE output. The app disables cuDNN on AMD GPUs and uses PyTorch's own convolutions (through rocBLAS), which are correct and faster. This also lets the VAE decode on the GPU (SDXL ~4 s instead of ~21 s on CPU) with an automatic CPU fallback. Set `IMAGEGEN_CUDNN=1` only to experiment.

**VRAM over-commit:** Windows doesn't report out-of-memory when the GPU fills up — it silently spills into system RAM and generation becomes ~20× slower. If that happens the info line shows **⚠ VRAM over-committed**; lower the resolution or batch size.

**Verify GPU is detected:** Open **Settings** tab → device badge shows:
- 🟦 **CUDA / ZLUDA (ROCm)** — AMD GPU via ZLUDA ✅
- 🟩 **DirectML** — fallback via torch-directml
- 🟥 **CPU** — slowest fallback

### Accelerated SDXL Text Encoding (iGPU DML)

SDXL models use two text encoders: CLIP-L (768-dim, 12 layers, ~442 MB) and OpenCLIP-G (1280-dim, 32 layers, ~2.5 GB). Under ZLUDA they run on the dGPU with Compel (long-prompt support). On the DirectML backend they run on CPU (~6.4 seconds per prompt).

When **"⚡ Accelerated SDXL text encoding (iGPU)"** is enabled in Settings:
- Both encoders are exported to ONNX (once per model, cached in `onnx_cache/<model_stem>/`)
- Inference runs on the **integrated GPU** via ONNX Runtime DirectML. The app finds the iGPU by name, because DirectML device numbering differs between PCs (on some laptops device 0 is the discrete GPU).
- ~115ms total for both encoders on a 780M (56× faster than CPU)
- Cosine similarity vs CPU baseline: >0.9998 (effectively identical output)
- Supports chunked long prompts (>77 tokens split into multiple 77-token windows)

### SmartSplit (SD 1.5 Multi-Device Pipeline)

For SD 1.5 models, SmartSplit distributes components across devices to reduce dGPU VRAM pressure:
- **Text Encoder** → CPU (or iGPU via ONNX DML)
- **UNet** → dGPU via ZLUDA (or DirectML)
- **VAE Decode** → CPU (float32 for compatibility)

SmartSplit is **SD 1.5 only** — SDXL uses the standard single-GPU path with optional accelerated text encoding above.

### NPU Status (XDNA1, e.g. Ryzen 7 8700G)

The XDNA1 NPU is detected when the AMD Ryzen AI SDK (tested: 1.7.0) is installed. However:
- **VitisAI EP compiles CLIP models** (342s → 87 MB `.rai` cache file)
- **Hardware context creation fails** (`0xc01e0009`) — XDNA1 is designed for small INT8 quantized workloads, not 442 MB+ float32 transformer models
- The 780M iGPU via DML provides the acceleration instead (56× CPU speed)
- NPU tests in `run_tests.py` are skipped automatically on machines without the Ryzen AI SDK

## Civitai Integration

1. (Optional) Get a free API key at [civitai.com/user/account](https://civitai.com/user/account) → paste in **Civitai Hub** tab
2. Search by name and/or tag, filter by type and base model
3. Click **📥 Select** on any card, pick a version, hit **⬇ Download**
4. Models auto-save to the correct folder and appear in Generate tab dropdowns

| Type | Local Path |
|---|---|
| Checkpoint / SDXL | `models/checkpoints/` |
| LoRA / LyCORIS / DoRA | `models/loras/` |
| VAE | `models/vae/` |
| Upscaler | `models/upscalers/` |

**Civitai API note:** filters are `types=` / `baseModels=` (plural — the singular forms are silently ignored since 2026) and ratings come from `thumbsUpCount`/`thumbsDownCount` (the old `rating` field was removed). Results are also filtered locally.

---

## Directory Structure

```
ImageGenApp/
├── app.py                    # Main Gradio application (entry point)
├── config.py                 # Paths, device detection, ZLUDA/cuDNN setup
├── requirements.txt          # Python dependencies
├── launch.bat                # Windows launcher with ZLUDA v6 injection
├── launch.ps1                # PowerShell launcher
├── run_zluda.bat             # Run any script under the ZLUDA environment
├── install.bat               # One-time dependency installer
├── selftest_zluda.py         # GPU-vs-CPU correctness self-test
├── smoke_gpu.py              # GPU smoke test (run_zluda.bat smoke_gpu.py [--sdxl], app closed)
├── run_tests.py              # CPU test suite (NPU tests skip without the hardware)
├── download_models.bat/.py   # Starter model downloader
├── backend/                  # One module per feature (list with descriptions: ../AGENTS.md "Layout")
├── wildcards/                # Starter wildcard files (__outfit__, __pose__ …)
├── onnx_cache/               # Cached ONNX exports per model (auto-generated)
│   └── <model_stem>/         # sdxl_te1.onnx, sdxl_te2.onnx + external data
├── models/
│   ├── checkpoints/          # SD model files (.safetensors / .ckpt)
│   ├── loras/                # LoRA weight files (downloaded + trained)
│   ├── vae/                  # VAE files
│   ├── inpainting/           # LaMa ONNX model (auto-downloaded)
│   └── upscalers/            # Upscaler models (auto-downloaded on first use)
├── presets/                   # Prompt presets (sd15/ and xl/ subdirs)
├── training_datasets/         # LoRA training data (images + captions)
└── outputs/                   # Generated + upscaled images (auto-saved)
```

---

## Troubleshooting

Install and startup problems (first image slow, black images, VRAM spill, window flashes and closes) are in
the [main README](../README.md#troubleshooting).

| Problem | Solution |
|---|---|
| "CUDA not available" after launch | Ensure HIP SDK is installed and `launch.bat` is used (not `python app.py` directly) |
| Model fails to load | Check that `.safetensors` is not corrupted; try a HuggingFace model ID |
| Out of memory (OOM) | Reduce batch to 1, lower resolution, or restart app. Enable SmartSplit for SD 1.5. |
| App window doesn't close after Ctrl+C | ZLUDA shutdown hang — close the console window |
| Civitai search returns 0 results | Check your internet connection; Civitai is sometimes slow — try again |
| Civitai download stuck | Large models (2–10 GB) take time; an API key helps. Check ETA in progress bar. |
| ONNX export warnings on first run | Normal — TracerWarnings during ONNX export are suppressed; only appears once per model |
| Accelerated TE shows "Session creation failed" | DmlExecutionProvider not available; check onnxruntime version (need 1.23+) |
| Watermark remover white spots | Increase mask dilation to 30–40px; Poisson blending handles most edge artifacts |
| EasyOCR "Using CPU" message | Only expected on DirectML/CPU, or if `IMAGEGEN_CUDNN=1` is set. Under ZLUDA OCR runs on the GPU |
| Black images from generation | Check LoRA compatibility; mismatched base (e.g., SD1.5 LoRA on SDXL) causes this |
| Auto-loop stop kills current batch | By design — stop button aborts immediately, not after current batch finishes |
