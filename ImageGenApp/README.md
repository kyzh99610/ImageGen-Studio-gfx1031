# 🦞 ImageGen Studio

A local AI image generation suite built for **AMD GPUs via ZLUDA v6 (ROCm/HIP)**, with GPU VAE decoding, **SmartSplit multi-device offload**, optional iGPU text encoding, and a built-in **Civitai Hub** for model discovery. Runs on Windows with AMD RDNA 2/3/4 dGPUs (primary target: RX 6800M, gfx1031), with optional iGPU and XDNA NPU support.

> Developer / AI-assistant notes live in [`../AGENTS.md`](../AGENTS.md).

---

## Features

| Feature | Details |
|---|---|
| **Text-to-Image** | SD 1.x, SD 2.x, SDXL / Pony / Illustrious · 9 samplers · batching · seeds · auto-loop |
| **Image-to-Image** | Variable denoising strength, same model |
| **LoRA** | 3-slot LoRA fusion with per-slot weights · automatic LoRA/model compatibility check · SD 1.5 + SDXL |
| **LoRA Training** | Dataset prep, aspect-ratio bucketing, BLIP auto-captioning, SD 1.5 + SDXL training, kohya-compatible output |
| **Upscaling** | Lanczos (CPU) · Real-ESRGAN ONNX (DirectML on the discrete GPU) · Real-ESRGAN PyTorch |
| **Watermark Remover** | GPU OCR + corner detection · draw-on-image brush · OCR-off mode for selective removal · LaMa / OpenCV / SD inpainting |
| **Civitai Hub** | Search, preview, paginated results (100 per search, 20 per page), download with live speed/ETA tracking |
| **Prompt Presets** | Save/load prompt+settings presets · Quick-tag buttons |
| **ZLUDA v6** | AMD CUDA→HIP translation so PyTorch/Diffusers run unmodified (cuDNN/MIOpen disabled for correctness) |
| **GPU Self-Test** | `run_zluda.bat selftest_zluda.py` checks GPU results against the CPU after driver/ZLUDA updates |
| **Accelerated Text Encoding** | Optional: SDXL CLIP-L + OpenCLIP-G on the iGPU via ONNX DML (mainly for the DirectML backend) |
| **SmartSplit** | SD 1.5 multi-device pipeline: text encoder on CPU/iGPU, UNet on dGPU, VAE on CPU |
| **DirectML Fallback** | Runs on AMD / Intel / NVIDIA GPUs through DirectML when ZLUDA isn't available (slower) |
| **Auto-save** | All outputs saved to `outputs/` with seed + metadata in filename |
| **Bilingual Help** | Full in-app reference guide in English and Chinese (中文) |

---

## Quick Start (Windows)

### 1. Install (once)

See the [main README](../README.md#installation): install AMD HIP SDK 6.4, then run `installer\setup.bat`.
It downloads portable Python 3.10, ZLUDA v6 and the gfx1031 rocBLAS kernels, and runs `install.bat`
(PyTorch 2.4.1 CUDA 11.8 build + `requirements.txt`). Re-run `install.bat` any time to repair packages.
(Don't also install plain `onnxruntime` — it and `onnxruntime-directml` write to the same folder.)

### 2. Launch

**In your browser:**

```bat
launch.bat
```

The launcher automatically:
- Asks `backend\rocm_env.py` which AMD GPU to use (the best one, or `--gpu N`) and which HIP runtime has
  kernels for it (the installed HIP SDK 6.x, or an optional `therock_sdk` folder), and — for gfx1031 GPUs
  only — points rocBLAS at the gfx1031 kernel library that setup downloaded
- Runs `ZLUDA_v6\zluda\zluda.exe -- python app.py` (ZLUDA v6 stable) to translate CUDA to AMD HIP
- Serves the UI at `http://127.0.0.1:7860` — launching again while it runs just reopens that tab;
  if another program owns the port, the next free one is used (the console says which)

**Desktop window (optional):** with [Node.js](https://nodejs.org) installed, `cd desktop`,
`npm install`, `npm run package`, then `powershell -ExecutionPolicy Bypass -File make_shortcuts.ps1` adds
**ImageGen Studio** to the Desktop and Start menu. It opens its own window, shows a loading screen while the
backend starts, and stops the backend — freeing the GPU — when you close it (Ctrl +/− zooms the UI).

The **first generation after each launch is slower** (about a minute for SD 1.5) while ZLUDA loads its compiled GPU kernels; later generations run at full speed. **On a new PC (or after a ZLUDA/driver update) the very first image takes 10–15 minutes** while every GPU kernel compiles once — the Generate tab warns about this; it isn't stuck.

### Options

```bat
launch.bat --port 8080          # Custom port
launch.bat --share              # Public Gradio share link
launch.bat --cpu                # Force CPU (skip ZLUDA)
launch.bat --dml                # DirectML instead of ZLUDA
launch.bat --gpu 1              # Pick a different GPU index
launch.bat --no-browser         # Don't open a browser tab
launch.bat --help               # List all options
```

### Check the GPU backend

```bat
run_zluda.bat selftest_zluda.py
```

Runs matrix multiply, convolution, attention and GroupNorm on the GPU and compares them with the CPU. Use it after any driver, ROCm or ZLUDA change — it fails loudly instead of letting you discover a broken backend through black images.

### Performance (RX 6800M laptop, gfx1031, warm)

Measured with a faster TheRock ROCm runtime; with the standard AMD HIP SDK 6.4 expect roughly 2× these
times for SD 1.5 (see the main README for HIP SDK numbers).

| Task | Time |
|---|---|
| SD 1.5, 512×768, 20 steps | ~7.5 s |
| SD 1.5 img2img | ~5.5 s |
| SDXL / Pony / Illustrious, 832×1216, batch 2, 20 steps | ~55 s |
| SDXL img2img (strength 0.5, 16 steps) | ~15 s |
| SDXL img2img, checkpoints whose VAE needs fp32 (strength 0.35, 10 steps) | ~8 s |
| SD→SDXL Bridge (10 + 10 steps, 512×768 → 832×1216) | ~20 s per run |

```powershell
.\launch.ps1 -Port 8080 -Share
.\launch.ps1 -NoZluda            # Disable ZLUDA injection
.\launch.ps1 -Cpu                # Force CPU mode
```

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

**Your session comes back:** prompt, negative prompt, model, VAE, LoRA slots and settings are restored the next time you start the app (the seed always starts random).

Also:
- **txt2img** and **img2img** (🖼 Send to img2img opens the img2img section with the image loaded)
- **9 samplers**: DPM++ 2M Karras, DPM++ SDE Karras, Euler a, Euler, DDIM, PNDM, LMS, Heun, UniPC
- **Batch generation** (1–8 images) and **Auto-Loop** for continuous generation with a configurable delay
- **Auto-add quality tags** checkbox — prepends `masterpiece, best quality` (or `score_9, …` for Pony) when your prompt has none; the exact prompt used is shown and saved
- **Prompt presets**, **My Saved Prompts**, **Save / Load Settings** (restores the checkpoint too), **Quick Tags**
- **Compel long-prompt encoding** with weighted/blended syntax `(word:1.3)`, `(word1|word2)`
- **📂 Show in folder** under the gallery opens Explorer with the selected image highlighted
- **Accelerated SDXL text encoding** (optional, Settings tab): CLIP-L and OpenCLIP-G on the integrated GPU via ONNX DirectML — mainly useful for the DirectML backend

Every setting has a one-line hint; the **📖 Settings Guide** accordion has the full cheat sheet.

### 🌉 SD→SDXL Bridge

Generate a base image with an **SD 1.5** model (and the SD 1.5 LoRAs you applied to it in Generate), then refine it with an **SDXL / Pony / Illustrious** model via img2img. Both model pickers list only the matching family and load automatically. On 12 GB cards the SD 1.5 model is freed before the SDXL stage (both plus SDXL's working memory don't fit, and Windows would spill to system RAM); it reloads with its LoRAs at the start of the next run, which costs a few seconds. The same seed is used for both stages and saved in the PNGs. `<lora:name:0.8>` tags in its prompt are applied to the SD 1.5 model (SDXL LoRAs are skipped with a note). Its settings (models, prompts, steps, sizes, strength — not the seed) are remembered for the next launch.

### 🔍 Upscale

Upscale images by 2×, 4×, or 8× using Lanczos (CPU), Real-ESRGAN ONNX (DirectML on the discrete GPU — ~2× faster than CPU), or Real-ESRGAN PyTorch. The GPU works in 256 px tiles so no single job trips Windows' ~2 s GPU watchdog; if the GPU ever returns a blank image the upscale is redone on the CPU automatically. The info line says which device was used. On the GPU the app makes a one-time fp16 copy of the model (~1.5× faster: 832×1216 → 2× in ~35 s instead of ~55 s, visually identical). Results are saved next to their source (`<source>_4x.png`, never overwriting) with the source's generation settings kept.

### 📄 PNG Info

Drop any image made by this app, A1111/Forge or Civitai to see its prompt and settings. **🚀 Send to Generate** fills in prompt, negative, sampler, steps, CFG, size, seed and checkpoint, and puts its LoRAs into the three slots with their weights — both `<lora:name:0.8>` prompt tags and this app's `LoRAs:` field are understood; the tags are taken out of the prompt, and LoRAs you don't have locally are listed.

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
├── app.py                    # Main Gradio application (~3800 lines, entry point)
├── config.py                 # Paths, device detection, ZLUDA/cuDNN setup
├── requirements.txt          # Python dependencies
├── launch.bat                # Windows launcher with ZLUDA v6 injection
├── launch.ps1                # PowerShell launcher
├── run_zluda.bat             # Run any script under the ZLUDA environment
├── install.bat               # One-time dependency installer
├── selftest_zluda.py         # GPU-vs-CPU correctness self-test
├── run_tests.py              # Test suite (72 CPU tests: UI build, rocm_env/gfx1031, PNG Info, prompts, edge cases, NPU; NPU ones skip without hardware)
├── download_models.bat/.py   # Starter model downloader
├── backend/
│   ├── sd_pipeline.py        # SD 1.x inference (txt2img, img2img, LoRA)
│   ├── sdxl_pipeline.py      # SDXL inference + iGPU DML text encoding + LoRA
│   ├── smartsplit_pipeline.py# SmartSplit multi-device pipeline (SD 1.5 only)
│   ├── hardware_detector.py  # GPU/iGPU/NPU detection, VRAM scoring, registry query
│   ├── npu_bridge.py         # Subprocess bridge to VitisAI conda env (NPU inference)
│   ├── vram_estimator.py     # VRAM usage estimation for UI display
│   ├── lora_trainer.py       # LoRA training (SD 1.5 + SDXL, kohya output)
│   ├── dataset_manager.py    # Dataset scanning, aspect-ratio bucketing
│   ├── auto_tagger.py        # BLIP auto-captioning
│   ├── upscaler.py           # Multi-backend upscaling (Lanczos / ESRGAN)
│   ├── watermark_remover.py  # OCR detection + LaMa/OpenCV/SD inpainting
│   ├── civitai_client.py     # Civitai API v1 wrapper (search, download, progress)
│   ├── model_manager.py      # Local model directory scanner
│   ├── help_content.py       # Bilingual help tab content (EN/中文)
│   ├── tag_fetcher.py        # Civitai tag/trigger word fetcher
│   └── trigger_reader.py     # Sidecar .civitai.json trigger word reader
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

## Hardware

| GPU | Status |
|---|---|
| **gfx1031** — RX 6700 / 6700 XT / 6750 XT / 6750 GRE, RX 6700M / 6800M / 6850M XT | **Primary target**, fully tested on an RX 6800M 12 GB |
| gfx1030 — RX 6800 / 6800 XT / 6900 XT / 6950 XT | Works (HIP SDK has its kernels); less tested |
| RDNA 3 / 4 (RX 7000 / 9000) | Runtime choice is wired up; ZLUDA support there is still experimental |
| Anything else | DirectML fallback (`launch.bat --dml`), slower |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "CUDA not available" after launch | Ensure HIP SDK is installed and `launch.bat` is used (not `python app.py` directly) |
| Model fails to load | Check that `.safetensors` is not corrupted; try a HuggingFace model ID |
| Out of memory (OOM) | Reduce batch to 1, lower resolution, or restart app. Enable SmartSplit for SD 1.5. |
| Generation suddenly very slow / "⚠ VRAM over-committed" | GPU memory spilled into system RAM (Windows doesn't raise OOM). Lower resolution or batch size |
| First image after launch is slow | Normal — ZLUDA loads its compiled kernels once per session |
| Black / noisy images or NaN errors | Run `run_zluda.bat selftest_zluda.py`. Make sure `IMAGEGEN_CUDNN` isn't set. Then check LoRA/model compatibility |
| App window doesn't close after Ctrl+C | ZLUDA shutdown hang — close the console window |
| Civitai search returns 0 results | Check your internet connection; Civitai is sometimes slow — try again |
| Civitai download stuck | Large models (2–10 GB) take time; an API key helps. Check ETA in progress bar. |
| ONNX export warnings on first run | Normal — TracerWarnings during ONNX export are suppressed; only appears once per model |
| Accelerated TE shows "Session creation failed" | DmlExecutionProvider not available; check onnxruntime version (need 1.23+) |
| Watermark remover white spots | Increase mask dilation to 30–40px; Poisson blending handles most edge artifacts |
| EasyOCR "Using CPU" message | Only expected on DirectML/CPU, or if `IMAGEGEN_CUDNN=1` is set. Under ZLUDA OCR runs on the GPU |
| Black images from generation | Check LoRA compatibility; mismatched base (e.g., SD1.5 LoRA on SDXL) causes this |
| Auto-loop stop kills current batch | By design — stop button aborts immediately, not after current batch finishes |
