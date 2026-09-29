# ImageGen Studio — Stable Diffusion on gfx1031 AMD GPUs (Windows)

**Local Stable Diffusion 1.5 / SDXL / Pony / Illustrious image generation for the AMD GPUs AMD's Windows
tools leave out: the gfx1031 family** — RX 6700, RX 6700 XT, RX 6750 XT, RX 6750 GRE, RX 6700M, RX 6800M and
RX 6850M XT. It runs on your GPU (not DirectML, not the CPU), on plain Windows, with no WSL or Linux.

One setup script installs everything into this folder. You get a web UI (or an optional desktop window)
with txt2img, img2img, LoRAs, upscaling, LoRA training, a Civitai browser and more.

> Tested on an **RX 6800M (12 GB)** laptop: SD 1.5 512×768 in **~16 s** and SDXL 832×1216 in
> **~90 s** (20 steps, AMD HIP SDK 6.4). See [Performance](#performance).

---

## Why this exists

If you own a gfx1031 card and tried Stable Diffusion on Windows, you've probably hit one of these:

| Problem | Cause | What this project does |
|---|---|---|
| `hipErrorFileNotFound` / "Cannot read TensileLibrary…gfx1031" / instant crash on the first matrix multiply | AMD's **HIP SDK for Windows ships no rocBLAS kernels for gfx1031** (gfx1030 only) | Downloads community-built gfx1031 rocBLAS kernels ([likelovewant/ROCmLibs](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU), release v0.6.2.4) and points rocBLAS at them via `ROCBLAS_TENSILE_LIBPATH` — **only** when the GPU really is gfx1031 |
| `CUBLAS_STATUS_NOT_SUPPORTED` in every `Linear` layer | hipBLASLt has no gfx103x kernels | Sets `DISABLE_ADDMM_CUDA_LT=1` |
| Black images, NaN latents, garbage VAE output | MIOpen convolutions return wrong results under ZLUDA on gfx1031 | Disables cuDNN/MIOpen; PyTorch's own convolutions (via rocBLAS) are correct and faster |
| `HSA_OVERRIDE_GFX_VERSION=10.3.0` "fix" from Linux guides | Does nothing on Windows (verified) | Not used |
| DirectML works but is slow and memory-hungry | — | Uses [ZLUDA v6](https://github.com/vosen/ZLUDA): the regular CUDA build of PyTorch runs on the AMD GPU through HIP |
| "Is it working or broken?" | — | `selftest_zluda.py` compares GPU results with the CPU for every operation SD uses |

Every workaround was switched off in turn and checked with that self-test, so each one is known to be needed.
The notes are in [`AGENTS.md`](AGENTS.md#gfx1031-audit-2026-09-27--each-workaround-toggled-against-the-extended-self-test).

---

## Requirements

- **Windows 10 (1803+) or 11**, 64-bit
- **A gfx1031 GPU** (list above). Other AMD GPUs: see [Other GPUs](#other-gpus)
- A recent **AMD Adrenalin driver** (tested with 32.0.21045)
- **AMD HIP SDK 6.4** for Windows (free from AMD, see step 1)
- **~20 GB free disk space** for Python packages, runtimes and a couple of models. Each SDXL model adds ~7 GB.
- 16 GB RAM recommended

## Installation

1. **Install the AMD HIP SDK 6.4.** Download *"HIP SDK for Windows"*, **version 6.4** (not 7.x), from
   <https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html> and run it. You can untick the
   display driver in its installer if your driver is newer.
2. **Get this project:** use **Code → Download ZIP** and extract it (or `git clone` it), e.g. to `C:\ImageGen`.
   Avoid folder names containing `!` or `%`, because Windows batch files can't handle them.
3. **Run `installer\setup.bat`.** It downloads and verifies (SHA-256), all inside this folder:
   - Python 3.10.11 portable (python.org)
   - PyTorch 2.4.1 (CUDA 11.8 build) and the packages in `ImageGenApp\requirements.txt` (~5 GB)
   - ZLUDA v6 (`github.com/vosen/ZLUDA`, release `v6`)
   - the gfx1031 rocBLAS kernels (~8 MB)

   It then checks that it can see the HIP SDK and your GPU, and offers three starter SD 1.5 models (~6 GB).
   Expect 15–30 minutes, depending on your connection.
4. **Check the GPU** (optional, recommended):
   ```bat
   ImageGenApp\run_zluda.bat selftest_zluda.py
   ```
   `✅ All checks passed` means ZLUDA, the HIP SDK and the gfx1031 kernels work together.
5. **Start the app:** double-click `ImageGenApp\launch.bat`. Your browser opens at <http://127.0.0.1:7860>.

> ⏳ **The very first image takes 10–15 minutes.** ZLUDA compiles every GPU kernel once and caches them in
> `%LOCALAPPDATA%\zluda\ComputeCache`, and the app shows a notice while it does. It isn't stuck. After that,
> images take seconds. SDXL compiles a few more kernels the first time you use it. After a ZLUDA or
> driver update, this happens once more.

No models yet? Use the **🌐 Civitai Hub** tab to search for and download checkpoints and LoRAs from inside the
app, run `ImageGenApp\download_models.bat`, or put `.safetensors` files into `ImageGenApp\models\checkpoints`.

### Launch options

```bat
launch.bat --help            :: list everything
launch.bat --port 8080       :: another port (a busy port moves to the next free one automatically)
launch.bat --gpu 1           :: pick the HIP device (default: the best discrete GPU)
launch.bat --no-browser      :: don't open the browser
launch.bat --dml             :: DirectML instead of ZLUDA (any GPU, slower)
launch.bat --cpu             :: CPU only (very slow)
```

The console prints which GPU and runtime it picked, e.g.
`[INFO] GPU 0 gfx1031 - ROCm runtime: 6.4 (C:\Program Files\AMD\ROCm\6.4\bin)`.

### Optional: desktop window

With [Node.js](https://nodejs.org) installed:

```bat
cd desktop
npm install
npm run package
powershell -ExecutionPolicy Bypass -File make_shortcuts.ps1
```

This adds an **ImageGen Studio** shortcut to your Desktop and Start menu. It opens the app in its own
window, shows a loading screen while it starts, and frees the GPU when you close it.

---

## Features

- **Generate:** SD 1.x / SD 2.x / SDXL / Pony / Illustrious / NoobAI, 14 samplers, batches, reusable
  per-image seeds, Auto-Loop, A1111 prompt weighting (`(tag:1.2)`, `[tag]`, `BREAK`, `<lora:name:0.8>`)
- **Hires fix** (small pass → upscale → re-draw details: SDXL 832×1216 → 1248×1824 fits in 12 GB), **variation
  seeds** ("more like this"), **CLIP skip 2** for SD 1.5 anime models, and an **X/Y grid** to compare CFG / steps /
  sampler / LoRA weight / prompt words side by side.
- **✨ Face detail** (ADetailer-style: faces redrawn at full resolution) and **🖌 Inpaint** (paint an area,
  redraw only that part with your model and LoRAs).
- **Faster, cleaner sampling:** DPM++ 2M **AYS** (Align Your Steps: ~25-step quality in 10–12 steps, about 2×
  faster), real Karras samplers, **PAG**, FreeU, CFG rescale, and automatic setup for **v-prediction**
  checkpoints (NoobAI-XL v-pred). Danbooru tag autocomplete and spelling hints.
- **Anime helpers:** 🎴 character cards (checkpoint + LoRAs + tags + outfits in one click, built from a character
  LoRA's Civitai prompts), 🎲 wildcards (`{a|b}`, `__outfit__`: different picks per image, reproducible per seed),
  🏷 WD14 interrogate / dataset auto-tagging, and quality tags matched to the model family (Illustrious, Pony, SD 1.5).
- **Batch folders** for the upscaler and the watermark remover (auto-detect + LaMa), e.g. to clean a LoRA training set.
- **Reproducible images:** every PNG records the checkpoint, VAE, LoRAs + weights and all settings (A1111-compatible,
  incl. model hash). Drop one into img2img and everything comes back.
- **Clickable LoRA keywords:** trigger words, the creator's ready-made prompts, and the LoRA's training tags ranked by
  how many training images had them. One click adds a character's defining tags.
- **Long prompts done right:** 75-token chunks cut between tags (`BREAK` starts a new one), a live token counter,
  warnings when a trigger word falls out of the first chunk, and no duplicate tags when presets or tags are added
  (the stronger weight wins).
- **Three LoRA slots**, including LyCORIS LoHa/LoKr. SD 1.5 vs SDXL is read from the file header, and
  mismatches are refused with a clear message instead of producing black images.
- **img2img**, and an **SD 1.5 → SDXL Bridge** (compose with SD 1.5, refine with SDXL)
- **Upscale** (Real-ESRGAN on the GPU via DirectML, or Lanczos), **Watermark Remover** (OCR + LaMa inpainting)
- **PNG Info:** reads A1111 / Forge / Civitai (PNG and JPEG) and NovelAI metadata, and sends the settings to Generate
- **Civitai Hub:** search, filter, and download with resume support
- **LoRA training** (SD 1.5 / SDXL, kohya-compatible output)
- **VRAM estimate** before you generate, and a warning when Windows spills VRAM into system RAM (it doesn't raise
  out-of-memory, it just gets ~20× slower)
- Bilingual in-app help (English / 中文)

Full feature documentation: [`ImageGenApp/README.md`](ImageGenApp/README.md).

---

## Performance

RX 6800M laptop (gfx1031, 12 GB), 20 steps, after the one-time kernel compile. The second and later images of a
session are faster than the first, because ZLUDA loads its cached kernels on the first run.

| Task | AMD HIP SDK 6.4 (what setup installs) |
|---|---|
| SD 1.5, 512×768 | ~16 s |
| SDXL / Pony / Illustrious, 832×1216 | ~90 s (4.2 s/step, peak VRAM 9.7 GB, no spill) |

A faster ROCm runtime ([TheRock](https://github.com/ROCm/TheRock) nightly builds) roughly halves these times
(SD 1.5 7.4 s, SDXL 26.8 s on the same laptop). If a `therock_sdk\bin` folder with its `rocblas.dll` exists next
to `ImageGenApp`, the launcher prefers it. Assembling one is advanced and unsupported, so the HIP SDK path is
the recommended one.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[WARN] No usable AMD HIP runtime found` | Install AMD HIP SDK **6.4** (step 1), then start again |
| Self-test fails / black or noisy images | Make sure you started with `launch.bat` / `run_zluda.bat`, not `python app.py`, and that `IMAGEGEN_CUDNN` isn't set. Post the self-test output in an issue. |
| First image seems stuck | Normal on first use (10–15 min kernel compile). Watch Task Manager: the GPU is busy. |
| Generation suddenly very slow, "⚠ VRAM over-committed" | VRAM spilled into system RAM. Lower the resolution or batch size. |
| `.bat` window flashes and closes | Run it from a Command Prompt to see the error; check the folder path has no `!` or `%` |
| Browser stuck on "Loading…" | Open <http://127.0.0.1:7860> in a new tab, and turn off browser auto-translate for the page |
| App keeps running after closing the browser | Close the console window, or use the desktop window, which stops the backend itself |

## Other GPUs

`backend/rocm_env.py` reads your GPU's architecture from `hipInfo` and picks the right runtime and kernels:
- **gfx1030** (RX 6800 / 6800 XT / 6900 XT / 6950 XT) uses the HIP SDK's own kernels and doesn't get the gfx1031 library.
- **RDNA 3 / 4** get the HIP SDK runtime, but ZLUDA support there is still experimental.
- Anything else can use `launch.bat --dml` (DirectML).

Reports from other cards are welcome.

## Running the tests

```bat
python-3.10\python.exe ImageGenApp\run_tests.py
```

That runs 91 CPU tests; NPU tests skip without a Ryzen AI NPU. For the GPU, run `ImageGenApp\run_zluda.bat selftest_zluda.py`.

## Credits and licenses

This project's code is MIT-licensed (see [LICENSE](LICENSE)). It downloads, but does not include:
- [ZLUDA](https://github.com/vosen/ZLUDA) by Andrzej Janik and contributors (Apache-2.0 / MIT)
- gfx1031 rocBLAS kernels from [likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU),
  built from AMD's rocBLAS/Tensile (MIT) with "littlewu's logic"
- AMD HIP SDK (AMD's license, installed by you), Python (PSF), PyTorch, diffusers, Gradio, and the other packages in
  `requirements.txt` (each under its own license)
- Models you download have their own licenses. Check them on Civitai / Hugging Face.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This project isn't affiliated with AMD, NVIDIA or the ZLUDA
project. Use it responsibly and follow the laws where you live and the licenses of the models you use.
