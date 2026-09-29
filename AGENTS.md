# ImageGen Studio — Agent Context

Single source of truth for AI assistants (Copilot, Gemini, Claude, Codex…) working in this repo.
`CLAUDE.md`, `GEMINI.md` and `.github/copilot-instructions.md` only point here — edit this file, not them.
User-facing docs: `README.md` (install + gfx1031 background) and `ImageGenApp/README.md` (app features).

**Last updated: 2026-09-29**

---

## What this is

A Windows-only local Stable Diffusion app (Python 3.10 + Gradio 4 + diffusers) for **AMD GPUs via ZLUDA v6**
(CUDA → HIP). Supports SD 1.5, SDXL / Pony / Illustrious, LoRA, img2img, upscaling, watermark removal,
LoRA training, and a Civitai browser. Primary target: **RX 6800M (gfx1031, 12 GB)** laptop.

## Layout

```
<repo root>\                       ← runtimes/models are .gitignored (installer\setup.ps1 downloads them)
├── AGENTS.md                      ← this file
├── ImageGenApp/                   ← the app
│   ├── app.py                     ← Gradio Blocks UI (~6000 lines), all callbacks, _save_outputs()
│   ├── config.py                  ← paths, device detection, ZLUDA env/cuDNN setup (import first!)
│   ├── launch.bat / launch.ps1    ← main launchers (ZLUDA v6 + therock_sdk PATH setup)
│   ├── run_zluda.bat <script.py>  ← run any script under the launch.bat ZLUDA environment
│   ├── install.bat / requirements.txt
│   ├── selftest_zluda.py          ← GPU-vs-CPU correctness check (GEMM/conv/attention/GroupNorm)
│   ├── wildcards/                 ← starter wildcard files (__outfit__, __pose__…)
│   ├── run_tests.py               ← 92-test CPU suite (NPU/SmartSplit/accel-TE, UI build, LyCORIS, rocm_env, PNG Info,
│   │                                 prompt syntax, damaged files, edge cases in user input, launcher flags)
│   └── backend/
│       ├── sd_pipeline.py         ← SD 1.x: load, txt2img/img2img, LoRA, GPU VAE decode + VRAM spill check
│       ├── sdxl_pipeline.py       ← SDXL: same + iGPU DML text encoding, tiled GPU VAE, own img2img encode
│       ├── smartsplit_pipeline.py ← SD 1.5 multi-device (TE→CPU/iGPU, UNet→dGPU, VAE→CPU)
│       ├── hardware_detector.py   ← GPU/iGPU/NPU detection, _classify_gpu(), VRAM scoring
│       ├── upscaler.py            ← Lanczos / Real-ESRGAN ONNX (DML on the dGPU) / PyTorch ESRGAN
│       ├── watermark_remover.py   ← EasyOCR (GPU when cuDNN off) + LaMa/OpenCV/SD inpainting
│       ├── lora_trainer.py, dataset_manager.py, auto_tagger.py
│       ├── lycoris.py             ← LoHa/LoKr fusion (diffusers can't load them)
│       ├── png_info.py            ← read A1111 / Forge / Civitai / NovelAI / ComfyUI metadata + imagegen records
│       ├── prompt_syntax.py       ← A1111 (x:1.2)/((x))/[x] → Compel weights (Compel ignores A1111 syntax)
│       ├── prompt_tools.py        ← tag parse/merge (strongest weight wins), CLIP token count, 75-token chunks
│       │                             at tag boundaries (BREAK = new chunk), encode_chunked(), prompt warnings
│       ├── lora_keywords.py       ← trigger words / creator prompts / training-tag coverage → keyword chips
│       ├── detail_tools.py        ← inpaint_region (only-masked), detect_faces, face_detail (ADetailer-style)
│       ├── sampling.py            ← scheduler table (real Karras, AYS), v-pred detection, run_pipe: PAG/FreeU/CFG rescale
│       ├── danbooru_tags.py       ← Danbooru tag list: autocomplete, near-miss spelling hints
│       ├── wildcards.py           ← {a|b} / __name__ dynamic prompts, resolved per image seed
│       ├── character_cards.py     ← settings/characters/*.json: model + LoRAs + tags + outfits, build from LoRA
│       ├── wd_tagger.py           ← WD14 (wd-vit-tagger-v3 ONNX, CPU) image → Danbooru tags
│       ├── model_hash.py          ← background SHA-256 cache (settings/_hash_cache.json) → A1111 "Model hash"
│       ├── civitai_client.py, model_manager.py, tag_fetcher.py, trigger_reader.py
│       ├── npu_bridge.py, vram_estimator.py, rocm_env.py (runtime + rocBLAS kernels per GPU arch)
│       └── help_content.py        ← bilingual (EN/中文) in-app Help tab — keep in sync with behaviour
├── installer/                     ← setup.bat/ps1 (fresh-PC bootstrap: Python, packages, ZLUDA, gfx1031 kernels)
├── python-3.10/                   ← portable Python (use this, not system python)
├── ZLUDA_v6/zluda/                ← ZLUDA v6 (downloaded by setup)
├── therock_sdk/bin                ← OPTIONAL faster ROCm runtime (TheRock); used instead of the HIP SDK when present
└── _gfx1031_rocm624/…/library     ← gfx1031-ONLY rocBLAS Tensile kernels (downloaded by setup; set by rocm_env.py)
```

## Run & test

```bat
ImageGenApp\install.bat                         :: once
cd desktop && npm install && npm run package    :: optional desktop window (Electron 44)
ImageGenApp\launch.bat [--port N] [--share] [--cpu] [--dml] [--gpu N] [--no-browser] [--no-pause]
:: busy port: an ImageGen Studio already there is reopened, anything else → next free port
.\ImageGenApp\launch.ps1 [-Port N] [-Share] [-Cpu] [-NoZluda] [-Dml]
ImageGenApp\run_zluda.bat selftest_zluda.py     :: GPU correctness (exit 0 = OK); add --cudnn to test MIOpen
python-3.10\python.exe ImageGenApp\run_tests.py :: 87 pass + 5 skip on machines without a Ryzen AI NPU
installer\setup.bat                              :: fresh PC: Python, packages, ZLUDA v6, gfx1031 kernels (SHA-256 pinned)
```

Scripts that touch the GPU **must run under ZLUDA** (`run_zluda.bat`) with cwd = `ImageGenApp`
(model paths like `models/checkpoints/...` are cwd-relative) and must `import config` before building
pipelines — that's where cuDNN gets disabled. ZLUDA processes can hang on interpreter shutdown:
standalone scripts end with `os._exit(code)`.

## Hardware

Primary target and test machine: **RX 6800M (gfx1031, 12 GB)** laptop with a Vega 8 iGPU. The code also handles
multi-GPU PCs (iGPU + dGPU, e.g. 780M + RX 6800 XT) and an XDNA1 NPU, but those paths are less tested.

DML enumeration order differs per machine — never hardcode a DML index. Use
`sdxl_pipeline._dml_igpu_id()` (iGPU) or `upscaler._dml_dgpu_id()` (dGPU), both based on
`hardware_detector._classify_gpu(name)`.

---

## GPU backend (read before touching device code)

### Desktop window (`desktop/`, Electron 44)
`main.js` spawns `launch.bat --port N --no-browser --no-pause` hidden (same runtime selection as a normal launch),
shows `splash.html` with the backend's output until `GET /` answers, then loads the UI (`?__theme=dark`). It attaches
to an ImageGen Studio already running on 7860–7880 instead of starting a second backend (and leaves that one running).
Stopping: synchronous `taskkill /T` on close (an async one lost the race with app exit), plus `app.py` watches
`IMAGEGEN_PARENT_PID` and calls `TerminateProcess` on itself when the window dies — `os._exit` hangs in ZLUDA's DLL
unload, and printing first raises because stdout is a pipe to the dead window (both measured). `IMAGEGEN_SNAPSHOT_DIR`
makes it save PNGs of the splash and UI for testing. Logs: `ImageGenApp/logs/desktop-backend.log`.

### ZLUDA v6 — primary
ZLUDA **v6 stable** (2026-06-29, `ZLUDA_v6/VERSION.txt`) replaced v6-preview60 on 2026-09-27: self-test passes,
SD 1.5 512x768 7.4 s, SDXL 832x1216 26.8 s. A new ZLUDA version recompiles every kernel once (~14 min first image).
v7 is still preview-only.
`launch.bat` puts `therock_sdk\bin` then `ZLUDA_v6\zluda` on PATH and runs `zluda.exe -- python app.py`.
PyTorch (2.4.1+cu118 wheel) sees `cuda:0` named `"AMD Radeon RX 6800M [ZLUDA]"`.
Env set by launchers / `config.py`: `ROCBLAS_TENSILE_LIBPATH` (gfx1031 only, via `backend/rocm_env.py`),
`DISABLE_ADDMM_CUDA_LT=1` (hipBLASLt lacks gfx103x), `MIOPEN_*` cache/log vars, `HF_HOME=.hf_cache`.
**Do not set `HIP_VISIBLE_DEVICES` with v6** (breaks multi-GPU); `IMAGEGEN_GPU` selects the torch device.

### Fallback without therock_sdk (portable zip)
ZLUDA v6 + the system **HIP SDK 6.4** (`C:\Program Files\AMD\ROCm\6.4`) + the gfx1031 Tensile library passes
`selftest_zluda.py` and generates (SD 1.5 512x768: ~15 s vs ~8 s on therock). With ROCm 6.4's own rocBLAS library the
process crashes (0xC0000409) — the gfx1031 library is required. `launch.bat` / `launch.ps1` / `run_zluda.bat` pick
therock_sdk if present, else the newest system ROCm 6.x, and set `HIP_PATH` to match. ZLUDA v5 fails every op (CUBLAS_STATUS_ALLOC_FAILED, "operation not supported") on driver 32.0.21045, so only
v6 is supported. install.bat/launchers mention an `installer\hip_sdk_installer.exe` if someone places AMD's
installer there; otherwise users get the amd.com link.

### cuDNN/MIOpen is OFF on AMD — do not re-enable
`config.get_device()` sets, for ZLUDA/AMD devices:
- `torch.backends.cudnn.enabled = False` — MIOpen convolutions under ZLUDA on gfx1031 return garbage
  (rel. error 1.0 in `selftest_zluda.py --cudnn`), and NaN latents unless `AMD_SERIALIZE_KERNEL=3`
  (which costs ~1.4× per step). Native im2col + rocBLAS convs are correct and faster.
  `IMAGEGEN_CUDNN=1` re-enables MIOpen for experiments only.
- flash / mem-efficient SDPA disabled (they silently no-op under ZLUDA; math SDPA is used). No xFormers.
- `AMD_SERIALIZE_KERNEL` is **not** set anywhere any more. Setting it is a debugging tool only.

Consequences of native convs: big im2col scratch buffers. Untiled SDXL VAE decode at 832×1216 wants >4 GB,
so SDXL VAE is always tiled (512 px). ZLUDA compiles kernels once per PC into
`%LOCALAPPDATA%\zluda\ComputeCache\zluda.db` — with an empty cache the first SD 1.5 image took **~15 min**
(measured 2026-09-27); the Generate tab warns when the db is missing/small (`app._kernel_cache_note`).

### VAE decode
Both pipelines decode latents themselves (`output_type="latent"`), via `sd_pipeline._gpu_vae_decode()`.
SD 1.5: untiled ≤1024 px (tiling visibly shifts SD 1.5 colours) unless `force_tiled_decode` (Bridge).
SDXL: fp16 with 512 px tiles → on NaN **fp32 with 256 px tiles on the GPU** (~3 s; many checkpoints ship the
original SDXL VAE, which overflows in fp16) → CPU (~22 s) last. `SDXLPipeline._vae_needs_fp32` remembers the fp32
need per model (reset on unload) and is shared with img2img's `_encode_image`. DirectML still decodes on CPU.

### HuggingFace repo IDs
`sd_pipeline._from_pretrained()` asks for `variant="fp16"` (half the download) and falls back to full weights.
Fallback choices when no checkpoint is installed: `stable-diffusion-v1-5/stable-diffusion-v1-5` (runwayml/ is a redirect)
and SDXL base; `stabilityai/stable-diffusion-2-1` is no longer public.

### SDXL text encoders & progress
Under ZLUDA the TEs sit on the GPU and are encoded in fp16 (identical embeddings; the old per-prompt fp32 cast cost
~2.8 GB of VRAM and ~3 s on the first run). The fp32 cast is kept only for TEs on the CPU (DirectML backend).
SDXL step callbacks call `torch.cuda.synchronize()` so the progress bar and Stop track the real GPU position
(without it batch-2 832x1216 showed 20/20 at 24 s and finished at 55 s); SD 1.5 skips it (would cost ~4 %).

### img2img
`from_pipe()` in diffusers 0.36 defaults to `torch_dtype=float32` and casts the **shared** UNet/TEs — always pass
`torch_dtype=self.dtype`. On ZLUDA, SDXL img2img encodes the init image itself (`_encode_image`: fp16 tiled,
fp32 retry on NaN, then `empty_cache()`) and passes 4-channel latents to the pipeline.

### VRAM over-commit (Windows)
WDDM does **not** raise OOM when VRAM is full — it pages into shared system RAM and runs ~20× slower.
`_vram_spill_note()` appends "⚠ VRAM over-committed" to the info line when `max_memory_reserved` exceeds the card.
When something "hangs", check Task Manager → GPU → Shared GPU memory for python.exe.

### Unloading must actually free VRAM
**Root cause (found 2026-09-27):** Compel parses prompts with pyparsing, and pyparsing's *packrat cache* (switched
on globally by some library in the app process) stores parse exceptions with their tracebacks — so the Compel call
frames, their callers' frames and everything in their locals (`pipe`, the Compel object → the text encoders) stayed
alive after `_unload()`. `prompt_tools.release_parser_cache()` (`ParserElement.reset_cache()`) runs after every
chunked encode and in both `_unload()`s; the older `finally: pipe = c = None` clears stay as a second line.
Measured (`VRAM … held`): SD 1.5 → SDXL left 8.6 GB held instead of 6.6 (the SD 1.5 inpaint pipe too — `_unload()`
now drops `_inpaint_pipe`), SDXL → SD 1.5 3.6 GB instead of 2.0 (the SDXL TEs, 1.6 GB); both fixed. To debug another
leak: weakref the module before unloading, then walk `gc.get_referrers` inside the app (a plain script won't
reproduce it) — frames under `pyparsing/core.py _parseCache` point here.

### Reproducible outputs & restore
`app._save_outputs(images, meta, pipe=…)` writes A1111 `parameters` text (Steps, Sampler, CFG scale, Seed,
`Size: WxH`, Model hash (AutoV2, once `model_hash` has it — hashing runs in the background, so the first image
of a new model may lack it), Model, VAE, Denoising strength, `LoRAs:` last) **and** an `imagegen` iTXt JSON
record: mode, prompts, scheduler, steps, cfg_scale, seed, width, height, strength, model {file, family,
sha256_10}, vae {file}|null, loras [{file, weight, sha256_10}]. Built by `_gen_record()` from the pipeline that
made the image. The Bridge saves stage 1 (txt2img, SD 1.5) and stage 2 (img2img, SDXL) with separate records.
`_restore_plan(read_png_info(img))` maps a record (or plain A1111 text) back to local files: exact file name →
stem → cached hash (renamed files). Used by PNG Info → Send to Generate and by `init_image.upload`
(**upload, not change** — the gallery's "Send to img2img" must not overwrite a prompt the user edited since).
Same seed + settings on this GPU differ by ~1–1.7/255 run to run (native convs aren't bit-exact); a restored
regeneration measured 0.63/255.

### Inpaint, face detail, checkpoint grid axis (`backend/detail_tools.py`)
- `inpaint_region()`: "only masked" inpainting — crop = mask bbox + padding (≥ native/2 unless `min_context=0`),
  scaled so the long side is 512 (SD 1.5) / 1024 (SDXL), the concrete `StableDiffusion(XL)InpaintPipeline.from_pipe`
  (cached as `sdp._inpaint_pipe`, shares the LoRA-fused UNet; `torch_dtype` passed), prompt embeds from the same
  builders as txt2img (Compel, chunks, CLIP skip / SDXL pooled), feathered paste-back. Measured: pixels outside the
  mask identical (Δ 0.00).
- `detect_faces()`: nagadomi lbpcascade_animeface (MIT, downloaded once to models/detectors/, SHA-256 pinned) +
  OpenCV's bundled Haar frontal-face cascade; overlapping boxes (IoU ≥ 0.3 or centre inside) → bigger one; boxes
  < 45 % of the main face's width are dropped (windows/buttons were detected as faces on an SDXL ship scene).
  Every box is **confirmed** (`_confirm_hits`: raw minNeighbors=0 hits on a crop enlarged to a ~160 px face; faces
  6–49, false 0–3 on 15 SDXL outputs; ≥ 4 kept). Auto mode confirms all boxes with the *anime* cascade — Haar found
  hands/chairs/bodies in anime images and confirmed them itself — and uses Haar-confirmed boxes only when nothing is
  anime-confirmed (photos). A 505 px false box used to set the size bar and hide 4 real grid faces. Turned heads are
  often missed.
- `face_detail()`: per face an elliptical mask (face + 15–20 % margin), crop ≈ 2× the face, denoise 0.4 →
  small faces redrawn at ~native/2 px. SD 1.5 full-body 512×768: 1 face in 5.9 s, 1.8 % of pixels changed.
- Generate tab: "✨ Face detail" runs after hires fix on txt2img *and* img2img results; "🖌 Inpaint" accordion
  (gr.ImageEditor, `app._editor_parts()`), record mode "inpaint" + `inpaint_padding` (not recreatable without the
  source + mask). X/Y grid "Checkpoint" axis: values matched to local file names; each switch goes through
  `_ensure_model` and re-syncs LoRAs.

### Samplers, AYS, PAG, FreeU, v-prediction (`backend/sampling.py`)
- **"Karras" is real now.** Until 2026-09-28 `_load_scheduler` built `DPMSolverMultistepScheduler` without
  `use_karras_sigmas`, so "DPM++ 2M Karras" (the default) and "DPM++ SDE Karras" were the plain samplers — not what
  A1111 / Civitai metadata means. `SCHEDULERS` maps each name to (class, options); `make_scheduler()` drops options a
  previous scheduler left in the config (`use_karras_sigmas`, `algorithm_type`). Records are `format: 2` since; a
  format-1 record's "DPM++ 2M Karras" / "DPM++ SDE Karras" restores as "DPM++ 2M" / "DPM++ SDE" (`LEGACY_NAMES`),
  A1111 text keeps meaning real Karras. New: DPM++ 2M, DPM++ 2M SDE Karras, DPM++ 2M AYS, DPM++ SDE, Euler AYS.
- **AYS** (NVIDIA Align Your Steps, diffusers `AysSchedules`): exact 10-step tables for SD 1.5 / SDXL, other counts
  interpolated log-linearly in sigma and mapped back to strictly decreasing timesteps. Implemented by wrapping the
  scheduler instance's `set_timesteps`, so img2img / hires / inpaint (strength slicing) need nothing else. Character
  grid (hassakuXL, 832×1216): AYS 10 steps ≈ the 25-step images, plain DPM++ 2M at 10 is soft; 12 AYS steps = 17 s
  denoise vs ~35 s for 25 Karras steps.
- **`run_pipe(sdp, pipe, kind, **call)`** — every txt2img / img2img / inpaint call of both pipelines goes through it:
  `sdp.boosters` (set by `do_generate`, reset to `{}` in its `finally`, so Bridge / Inpaint buttons don't inherit
  them) → PAG (`StableDiffusion[XL]PAG[Img2Img]Pipeline.from_pipe(pag_applied_layers=["mid"])`, cached per UNet in
  `sdp._pag_pipes`, dropped in `_unload`), FreeU (`unet.enable_freeu`, off again in `finally`), CFG rescale
  (`guidance_rescale`, only passed when the pipeline class accepts it — SD 1.5 img2img / inpaint don't). PAG swaps
  attention processors on the shared UNet and diffusers restores them only at the end of a run: `run_pipe` restores
  them in `finally` (a Stop mid-PAG left the next plain run broken otherwise; measured Δ 0.00 after the fix).
  Timings (12 AYS steps, 832×1216): plain 17 s, PAG 2.5 29–34 s (batch 3 instead of 2), FreeU 22 s.
- **FreeU under ZLUDA:** its Fourier filter needs cuFFT → `CUFFT_NOT_SUPPORTED`. `_patch_fourier_filter()` wraps
  diffusers' `fourier_filter` (looked up as a module global by `apply_freeu`): after the first failure the FFT runs
  on the CPU (small skip tensors, +30 % time).
- **FreeU factors are tuned for anime checkpoints**, not the paper's: SDXL b1 1.05 / b2 1.1 / s1 0.95 / s2 0.8,
  SD 1.5 b1 1.1 / b2 1.2 / s1 0.9 / s2 0.6. Measured on hassakuXL (same seeds, 12 AYS steps): the paper's SDXL values
  (1.3/1.4/0.9/0.2) took mean saturation 63 → 135, clipped 10 % of pixels (vs 2 %) and bent the composition; the
  chosen ones 83 / 5.5 % and 92 / 2.0 % on two seeds. Stacking paper-FreeU + PAG + hires gave burned images.
- **Hires fix runs its second pass without PAG** (`do_generate` sets `pag=0` around `_hires_pass`): PAG shapes the
  composition in the first pass; in the 1248×1824 pass it cost ~130 s instead of ~80 s and peaked at 10.9 of 12 GB
  (batch 3). The record still lists the PAG scale (it describes the first pass).
- **V-prediction:** `detect_prediction()` reads the safetensors header — `v_pred` / `ztsnr` marker keys (NoobAI),
  `modelspec.prediction_type`, or "vpred" in the file name — and `configure_prediction()` sets
  `prediction_type=v_prediction`, `rescale_betas_zero_snr`, `timestep_spacing=trailing`. Pass them to `from_config`
  as **keyword arguments**: overrides inside the dict are reset by the config's hidden `_use_default_values`. With
  zero-terminal SNR, Karras / AYS are skipped (sigma_max is infinite). CFG rescale 0 = auto → 0.7 for v-pred. No
  v-pred checkpoint is installed here, so this path is unit-tested only.
- UI: "🎚 Quality boosters" accordion; X/Y axes "PAG scale", "CFG rescale"; A1111 text "PAG scale: …",
  "FreeU: on", "CFG rescale: …"; restore sets them (off when the record has none). The grid title leaves out the
  base seed / checkpoint when that is an axis.

### Danbooru tags (`backend/danbooru_tags.py`)
WD14's `selected_tags.csv` (~8,100 general + ~2,750 character tags with post counts; ~300 KB via hf_hub_download —
fetched in the background the first time someone types, never from the token counter). Autocomplete chips under the
prompt for the tag being typed (text after the last comma; `prompt_txt.input`, `trigger_mode="always_last"`),
brackets escaped on insert. "Did you mean": `difflib` cutoff 0.88 against general tags with the same first letter
(~9 ms for 13 tags, lru-cached) — only near misses ("long haired" → long hair, "thigh highs" → thighhighs); the tag
still being typed, quality tags, names and free text are left alone. Shown in the token counter;
"💡 Fix Danbooru spellings" applies them keeping weight syntax. Danbooru calls silver hair "grey hair" (alias),
the list has no aliases.

### Anime helpers: wildcards, character cards, WD14 tagger, family quality tags
- **Wildcards** (`backend/wildcards.py`): `{a|b}`, `{2$$a|b|c}`, `{3::a|b}` (weight), `__name__` = random line of
  `ImageGenApp/wildcards/name.txt` (shipped SFW starters: outfit, pose, expression, background, hair, eyes, lighting,
  camera) or `models/wildcards/` (user; first dir wins; sub-folders `__a/b__`). Innermost group first, ≤ 20 rounds (a
  self-referencing file stops there), `{tag}` without `|` and `\{…\}` stay literal, unknown `__x__` stay and are
  reported. RNG = `random.Random(f"wildcards:{seed}")` per image. `do_generate` (so Auto-Loop and the X/Y grid too)
  splits a dynamic prompt into batch-1 calls with seed, seed+1, … and records `prompt_template` /
  `negative_template` next to the resolved prompt (restore uses the resolved one). `split_tags` treats `{}` as
  brackets. The Bridge doesn't resolve them.
- **Character cards** (`backend/character_cards.py`, `settings/characters/<name>.json`, gitignored): checkpoint / VAE
  / ≤ 3 LoRAs by **file name** (found again with `list_checkpoints()` etc.), `tags`, named `outfits`, optional
  negative, size, CFG, steps, sampler, CLIP skip; `clean_card()` drops anything else. `card_from_lora()`: Civitai
  triggers + 🗝 likely triggers + body tags (hair/eyes/… ≥ 60 % coverage) + `1girl`/`solo` when ≥ 60 %; every
  multi-tag Civitai trigger prompt minus those tags = one outfit, named after its clothing tags. Build refuses to
  overwrite an existing card; Save keeps the card's outfits and stores the whole prompt as its tags.
- **WD14 tagger** (`backend/wd_tagger.py`): SmilingWolf/wd-vit-tagger-v3 (Apache-2.0, `model.onnx` 378 MB via
  hf_hub_download into `.hf_cache`), ORT **CPU** (0.8 s warm, doesn't touch VRAM). Input: RGBA on white, padded
  square, bicubic 448, **BGR** float32 0–255 NHWC. General ≥ 0.35, character ≥ 0.85, rating kept apart; `_` → space
  except kaomoji, brackets escaped. Used by 🏷 Interrogate (img2img accordion, PNG Info) and Train LoRA "Auto-Tag
  All (WD14)" (trigger first). Gradio 4.19's `gr.Progress` raised "list index out of range" when called from an API
  request before the first step — progress calls are wrapped (`_say`).
- **Family quality tags** (`app._family_quality`): Illustrious/NoobAI get `masterpiece, best quality, amazing
  quality, very aesthetic, absurdres` + `worst quality, low quality, bad quality, lowres, bad anatomy, jpeg
  artifacts, signature, watermark`; Pony `score_9, score_8_up, score_7_up` + `score_4, score_5, score_6`; each side
  only when none of that family's markers is there.
- Face detail runs ~half the main steps (`ceil(max(10, steps × 0.5) / denoise)`): SDXL A/B at 1024² — 0.8× (~22
  effective steps) ~45 s per face, 0.5× (~14) ~28 s, faces no worse.

### Hires fix, variations, CLIP skip, X/Y grid, batch folders
- **Hires fix** (`app._hires_pass`): txt2img at the entered size → `_upscale_to()` (Lanczos, or Real-ESRGAN ×2/×4 then
  Lanczos) → `sd.img2img(strength=denoise, steps=ceil(hires_steps/denoise))` with each image's own seed. Long side capped
  at 1536 (SD 1.5) / 2048 (SDXL). SDXL 832×1216 → 1248×1824 peaks at 10.1 of 12 GB, ~60 s for 12 steps. The img2img
  passes overwrite `sd.last_seeds` — restored afterwards. Record: `width/height` = first pass + `hires{scale,denoise,
  steps,upscaler}`; A1111 text "Hires upscale/steps/upscaler" + "Denoising strength" (= hires denoise, which
  png_info moves out of `strength`).
- **Variations** (`sd_pipeline.variation_latents`): the pipeline's own first noise (same generator → strength 0 is
  bit-identical) slerped toward `var_seed`'s noise, passed as `latents=`. The slerp runs on the **CPU**: acos/sin on
  the GPU made ZLUDA compile new kernels (a batch of 3 took 154 s instead of ~25 s).
- **CLIP skip** (SD 1.5): Compel `PENULTIMATE_HIDDEN_STATES_NORMALIZED`; the string fallback passes diffusers
  `clip_skip=1` (diffusers counts from 0). SDXL already uses the penultimate layer.
- `app._clean_extra()` sanitises all of these; `_plan_extra_updates()` restores them (drop into img2img / PNG Info).
- **X/Y grid** (`do_xy_grid`, `_xy_values`, `_xy_grid`): ≤ 48 cells, fixed seed, each cell via `do_generate` (saved
  with its record) + a labelled grid PNG. LoRA-weight axis re-syncs LoRAs per value and restores slot 1 afterwards.
- **Batch folders**: Upscale tab (`do_upscale_folder` → outputs/upscaled_<folder>/) and Watermark tab
  (`do_remove_folder`: auto-detect + chosen method; images with nothing found are copied → outputs/cleaned_<folder>/).
  `_carry_params()` keeps the source's parameters + imagegen record in both, and in the single-image remover.
- **LaMa** (`watermark_remover.inpaint_lama`): one 512² ONNX run per separate region, square context crops
  (mirror-padded, never stretched — a wide bottom caption squeezed into a square came back as a grey smear),
  feathered blend where the mask touches the crop/image edge (seamlessClone washes to grey there). OCR boxes are
  widened by ~1 character so "©"/"@" next to the text are removed too.

### Prompts: merge, chunks, keywords
`prompt_tools.merge_prompts()` is used by presets, quick tags, img2img enhancer tags, keyword chips and the
auto-quality tags: each tag once (key = lower-case, `_`→space, weight syntax removed), strongest weight wins,
first position kept. Long prompts: `encode_chunked()` encodes ≤75-token chunks cut at tag boundaries (Compel alone
cut mid-tag every 75 tokens and BREAK became a comma) and concatenates them; SDXL pooled = first chunk;
`pad_to_same_chunks()` pads the shorter side with empty-prompt chunks. Keyword chips (`gr.Dataset`, `type="index"`,
tags in a `gr.State` as `[tag, goes_up_front]`): triggers / 📋 creator prompts go after the leading quality tags
(`insert_after_quality`), other tags merge at the end. Coverage = caption tag count / training images
(`ss_dataset_dirs` img_count). Civitai trainedWords that are whole prompts: the tags all of them share become the
trigger. Name-like tags in ≥90 % of images are 🗝 likely triggers (e.g. a caption typo of the name).

### LoRA
The Generate tab applies whatever the three slot dropdowns show on every Generate/Auto-Loop (`_sync_loras`, one
reload only on change); ✖/Clear reset the dropdowns. `model_manager.lora_arch()` reads the cross-attention width from
the safetensors header (768 SD1 / 1024 SD2 / 2048 SDXL) so mismatches are refused before loading.
`model_manager.checkpoint_arch()` does the same for checkpoints (SDXL files without "xl" in the name).
LyCORIS LoHa (`hada_w*`, incl. Tucker `hada_t*`) / LoKr (`lokr_*`) can't be loaded by diffusers 0.36: `backend/lycoris.py`
rebuilds ΔW per layer and adds it to the kohya-named Linear/Conv2d weight (undone by the usual snapshot restore).
A failed LoRA load drops that slot and reloads the rest; weights were verified bit-identical afterwards.
Dropdown labels come from `app._checkpoint_choices()` / `_lora_choices()` ("SDXL · file"); values stay full paths,
and code that matches models by filename must use `list_checkpoints()`, not the labels.
Fuse+unload via PEFT, restore from a CPU snapshot taken lazily on first `load_lora()`.
CUDA/ZLUDA fuses **on the GPU**: `Module.cpu()` on the 5 GB SDXL UNet crashes inside ZLUDA (access violation).
DirectML still round-trips to CPU. Same-seed output differs run-to-run by ~0.4–0.7/255 (native convs aren't
bit-deterministic), so compare LoRA restore results against that noise floor.

### DirectML — fallback & iGPU
`torch_directml.device(i)` → `"privateuseone:i"`; match on `"privateuseone"`, never `"directml"`.
SDXL on DML: UNet on GPU with device-guard hooks, scheduler math on CPU, TEs + VAE on CPU.

### Accelerated SDXL text encoding (iGPU, optional)
CLIP-L + OpenCLIP-G exported once to `onnx_cache/<model_stem>/` (OpenCLIP-G > 2 GB → external data files),
run with ORT `DmlExecutionProvider` on the **iGPU** (`_dml_igpu_id()`). ~115 ms on a 780M vs ~6.4 s CPU,
cosine >0.9998. Export needs `_force_eager_attention()` (SDPA breaks ONNX export); long prompts chunked by
`_tokenize_chunked()`. Under ZLUDA the TEs already run on the dGPU, so this mainly helps the DML backend.

### Olive — removed (2026-09-27)
The Olive converter was convert-only: no generation path ever loaded its output. The Settings UI,
`backend/olive_optimizer.py`, the `olive-ai` package and 4.9 GB of `olive_models/` were deleted.

### SD→SDXL Bridge
On <14 GB cards stage 2 frees SD 1.5 first (its LoRAs are restored at the start of the next run) and stage 1 uses a
tiled VAE decode while SDXL is resident — otherwise the pair plus SDXL activations exceed 12 GB and WDDM spills.
`[Bridge] …` lines in the console log allocated/reserved VRAM per stage. Runs take ~20 s warm.

### SmartSplit & NPU
SmartSplit is SD 1.5 only. XDNA1 NPU (8700G) compiles CLIP via VitisAI but hardware-context creation fails
(`0xc01e0009`); the iGPU is used instead. NPU tests skip when the Ryzen AI conda env is absent.

### Measured on RX 6800M (2026-09-26)
Warm (2nd+ generation in a session): SD 1.5 512×768 20 steps: ~36 s → **7.5 s** · SD 1.5 img2img 5.5 s ·
SDXL 832×1216 batch 2, 20 steps: ~55 s · SDXL img2img: 250 s (fp32 spill bug) → 15 s ·
SDXL VAE 21 s (CPU) → ~4 s (fp32-VAE checkpoints ~3 s on GPU) · Bridge 56–88 s → ~20 s ·
EasyOCR 0.74 s → 0.17 s · ESRGAN ONNX 14.5 s → 7.7 s (512 px; 2026-09-26 re-measure 12 s, 832×1216 → ~55 s —
it always runs the 4× model, so 2× costs the same as 4×; full-size edge tiles were tried and are *slower*).
On DML the upscaler now uses `real_esrgan_x4_fp16.onnx` (made once by `upscaler._fp16_model()`, Resize/Identity kept fp32):
832×1216 → ~35 s, PSNR ≈ 57 dB vs fp32. Needs `onnx` + `onnxconverter-common`, else it stays on fp32.
SDXL 832×1216 batch 1: ~26 s denoise + 4 s VAE · SDXL img2img 0.5 strength: ~17 s · no VRAM spill (peak 9.4 GiB).
The **first** generation of each session is slow (SD 1.5 ~55 s) while ZLUDA loads/JITs kernels; the very first
run on a PC (empty `zluda.db`) is much slower — ~15 min for the first SD 1.5 image — while it compiles kernels.

---

## Conventions

- `config.py` is the single source of truth for paths and device selection; import it before torch pipelines.
- Service singletons (`sd`, `_sdxl`, `upscaler`, `civitai`) are created once at module level in `app.py`.
- Gradio callbacks return `gr.update()` / values; only the Civitai tab globals (`_search_results`,
  `_search_all_results`, `_search_page`, `_current_model`) are mutated.
- Pipelines own their memory: `_unload()` before loading another model.
- Every `gr.Dropdown` needs an explicit `value`; with `allow_custom_value=True` a `None` value crashes the Gradio
  4.19 frontend (`input_text.toLowerCase`) and the page never leaves "Loading…". `run_tests.py` builds the UI and
  fails on such dropdowns.
- Gradio 4.19 ignores server-side `gr.Accordion(open=...)` updates — open accordions with a `.then(js=...)` click.
- Never `sed -i` CRLF files in Git Bash: it rewrites them to LF. Edit with Python (bytes) or the Edit tool.
- `config.py` reconfigures stdout/stderr to UTF-8 with replacement: output is full of ✅/→ and redirected output
  (logs, pipes) uses cp1252, where `print` would raise.
- `.bat` files are **CP1252/ANSI**, not UTF-8, and literal `)` inside `if` blocks must be escaped `^)`.
  Edit them byte-wise; don't let an editor re-save them as UTF-8.
- Keep each file's existing line endings (`sdxl_pipeline.py` is CRLF, most others LF).
- API keys live only in `ImageGenApp/settings/api_keys.json` (gitignored).
  Never hardcode keys in launchers or scripts.
- Git: commit messages `type: short imperative summary`; runtimes/models/outputs are .gitignored.

## Package versions (python-3.10)

torch 2.4.1+cu118 · torchvision 0.19.1 · torch-directml 0.2.5 · diffusers 0.36.0 · transformers 4.44.0 ·
huggingface_hub 0.36.2 · accelerate 1.12.0 · peft 0.17.0 · compel 2.0.2 · torchsde 0.2.6 (DPM++ SDE samplers; `make_scheduler` falls back to DPM++ 2M Karras
without it) · gradio 4.19.2 (index.html patched) ·
**onnxruntime-directml 1.23.0 only** (plain `onnxruntime` shares the same package dir — never install both;
`rembg` will complain in `pip check`, harmless) · easyocr 1.7.2 · opencv-python-headless 4.9.0.80.

## Civitai API

Base `https://civitai.com/api/v1`, header `Authorization: Bearer <key>` (needed for NSFW/downloads).
Filter params are **`types=`** and **`baseModels=`** (plural). This flipped: in early 2026 only `type=` worked, but
re-tested 2026-09-26 the singular `type=` / `baseModel=` are *silently ignored* (a LoRA search returned mostly
checkpoints) while the plural forms filter correctly. `do_search` also filters the results locally, so a future
flip can't leak other model types in. Some items now carry `null` stats — use `or 0`. `stats.rating` was removed:
use `thumbsUpCount` / `thumbsDownCount`. `limit=100` max; UI paginates 20/page.
Endpoints: `/models?query=&types=Checkpoint&baseModels=Pony&sort=Most+Downloaded&limit=100`, `/models/{id}`, `/model-versions/{id}`.

## Watermark remover

Detection: EasyOCR text boxes → corner edge-density heuristic → logo expansion (Canny + contours) →
merge boxes within 30 px → drop >50 % IoU duplicates. Inpainting: OpenCV TELEA/NS, LaMa ONNX (CPU only, ~2 s —
DML still fails on its FFC MatMul with onnxruntime-directml 1.23, re-checked 2026-09-26), SD inpainting (uses loaded checkpoint). Blending: `cv2.seamlessClone`,
feathered Gaussian fallback at image borders.

---

## gfx1031 audit (2026-09-27) — each workaround toggled against the extended self-test

| Workaround | Verdict |
|---|---|
| gfx1031 rocBLAS Tensile library | **Required** — therock's own gfx1031 files fail every GEMM; HIP SDK 6.4 has none (crash). gfx1031-only, so `rocm_env.py` sets it only for gfx1031 |
| `DISABLE_ADDMM_CUDA_LT=1` | **Required** — without it every Linear+bias fails (`CUBLAS_STATUS_NOT_SUPPORTED`). The old self-test had no addmm case (false negative) |
| cuDNN/MIOpen off | **Required** (see above) |
| `HSA_OVERRIDE_GFX_VERSION=10.3.0` | **No effect on Windows** (bogus `9.0.0` passes) — removed |
| therock for every GPU | **Wrong for RDNA 3/4**: therock only has gfx101x/103x kernels. `rocm_env.py` picks the system HIP SDK 6.x for those |
| `HIP_VISIBLE_DEVICES` in `set_active_gpu` | v5-era; breaks ZLUDA v6 multi-GPU — removed |

`backend/rocm_env.py` (stdlib only) reads the arch from `hipInfo.exe`, chooses therock vs system ROCm 6.x by which rocBLAS
library has kernels for it, and emits `set` lines the launchers `call` from a per-launch temp file (a shared file once
went stale and made `launch.bat` pick a missing runtime). `run_tests.py` covers gfx1031/gfx1030/gfx1201/gfx1103.
Without `--gpu`/`IMAGEGEN_GPU` it picks the best HIP device itself (discrete before integrated, then most memory —
`hipInfo` reports `isIntegrated`/`totalGlobalMem`) and returns `ROCM_GPU_INDEX`, which the launchers pass on as
`IMAGEGEN_GPU`. Arguments are parsed **before** rocm_env runs (it used to run first and always decide for GPU 0,
and an explicit `--gpu 0` was overridden by auto-detection). `_gpu_detect.py` was removed.

### Where the runtime pieces come from (all verified byte-identical to the tested files)
| Piece | Source | SHA-256 of download |
|---|---|---|
| ZLUDA v6 | github.com/vosen/ZLUDA release `v6`, `zluda-windows-3fe1206.zip` | `fda8891c…7208a` |
| gfx1031 rocBLAS kernels | github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU release `v0.6.2.4`, `rocm.gfx1031.for.hip.sdk.6.2.4.littlewu.s.logic.7z` | `a1946df3…93b73` |
| Python 3.10.11 | python.org embeddable amd64 zip | `608619f8…7629d` |
| HIP SDK 6.4 | amd.com (EULA; not redistributed) | — |
`installer/setup.ps1` downloads and checks the first three; `tar.exe` (Windows 10 1803+) extracts the .7z.
`install.bat` installs torch cu118 then `requirements.txt` (pins = the working env; three OpenCV builds had piled up
in the dev env — only `opencv-python-headless` 4.9 is what loads and what is pinned).

## Fix history (don't reintroduce)

1. **ZLUDA v5 + ROCm 7.1** → `hipErrorNoDevice`; v5 is built for the ROCm 6 ABI. Use ZLUDA v6 (v5 since removed).
2. **Batch files crash instantly** → UTF-8 bytes / unescaped `)` in `if` blocks (see conventions).
3. **Gradio stuck on "Loading…/加载中"** → CDN refs removed from `gradio/templates/frontend/index.html`,
   `font_mono` set to local fonts, `<meta name="google" content="notranslate">`, explicit Dropdown values.
   Re-apply the index.html patches if gradio is upgraded.
4. **`cannot import name 'cached_download'`** → diffusers 0.25 vs new huggingface_hub; use diffusers 0.36.
5. **Chrome "C.DLL not found"** → `zluda_with.exe` injected into Chrome; launchers use PATH injection.
6. **Civitai 0 results / rating 0.0** → param names + thumbs counts (see Civitai API). The param names changed
   again by 2026-09-26 (`types=` / `baseModels=`); results are now also filtered locally.
7. **LoRA black images** → base mismatch (SD 1.5 LoRA on SDXL); UI shows a compatibility warning.
8. **ONNX TracerWarnings** on first export → suppressed; harmless.
9. **gfx1031 rocBLAS Tensile crash** (`hipErrorFileNotFound 301`) → 84 MB gfx1031 `rocblas.dll` in
   `therock_sdk\bin`, gfx1031 Tensile library via `ROCBLAS_TENSILE_LIBPATH`. (`HSA_OVERRIDE_GFX_VERSION` was part of
   this fix but is ignored on Windows — removed 2026-09-27; see gfx1031 audit below.)
10. **NaN latents / garbage VAE** → MIOpen under ZLUDA; cuDNN disabled (2026-09-25). The old
    `AMD_SERIALIZE_KERNEL=3` + CPU-VAE workarounds were removed.
11. **img2img makes everything slow** → `from_pipe()` fp32 upcast of shared modules (2026-09-25).
12. **SDXL LoRA load segfault** → `Module.cpu()` under ZLUDA; fuse on GPU (2026-09-25).
13. **Accel-TE ran on the dGPU on the laptop** → hardcoded DML id 0; now looked up by name.
14. **The old make_zip.ps1 would ship api_keys.json, LoRAs, outputs** → exclusion list fixed (2026-09-26); the
    script isn't in the repo any more — any packaging script must leave out `settings/`, `models/`, `outputs/`.
15. **Seeds saved as -1** → real per-image seeds (batch: seed, seed+1, …) in filenames and PNG metadata.
16. **Upscaler returned black images** → 512 px DML tiles exceeded Windows' ~2 s GPU watchdog (TDR); DML tiles
    capped at 256, CPU at 384, blank output retried on CPU.
17. **Unloaded models kept their VRAM** (frame-held pipeline, see above) — Bridge got slower every run.
18. **SD inpainting always failed** → `AutoPipelineForInpainting` imports CogView4 (needs transformers' GlmModel);
    use the concrete SD 1.5 / SDXL inpaint pipelines.
19. **Edge-case pass (2026-09-27)** — each found by a test in `run_tests.py`:
    Gradio 4.19 passes slider text boxes through unchecked, so `app._clean_gen_args` makes Generate / Auto-Loop /
    Bridge inputs safe (sizes → multiples of 8 in 256–2048, seed ≥ 2⁶⁴ crashed torch → wrapped to 32 bit,
    img2img steps×strength < 1 → steps raised, None/NaN → defaults) and reports each change under the result;
    `_fit_init_image` caps img2img inputs (SD 1.5 1280 px, SDXL 2048 px; a 4000 px photo spilled VRAM) and lifts
    tiny ones to 64 px; upscale refuses > 64 MP output. `prompt_syntax`: `(solo)1girl` lost its "1", hundreds of
    nested brackets raised RecursionError. PNG Info: prompts weren't HTML-escaped (`<lora:…>` vanished), A1111
    JPEG/WebP EXIF UserComment is in the Exif sub-IFD (was read from IFD0 → never found), NovelAI JSON and ComfyUI
    graphs are recognised, files are closed after reading. Civitai file names are sanitised (`../`, `:` …, CON),
    null `stats`/`modelVersions`/`images` no longer crash cards. Preset/settings names are sanitised and
    `API_KEYS` (case-insensitive on Windows) can no longer overwrite `api_keys.json`. VRAM estimate reads the
    checkpoint header (Pony/Illustrious were counted as SD 1.5). `IMAGEGEN_GPU=abc`/empty crashed `config` import.
    `.bat` launchers: `%VAR%` inside `( )` blocks broke for paths like `Program Files (x86)` (tested from
    `D:\IG Test (x86) & co\…`), `--port --no-pause` took `--no-pause` as the port, em-dashes in two .bat files.
20. **VRAM stayed held after switching models** (1.6–2 GB, SDXL peak 11.5 of 12 GB) → pyparsing's packrat cache kept
    Compel's frames (→ text encoders) alive, and `_unload()` kept the cached inpaint pipe. See "Unloading must
    actually free VRAM". Found by the 2026-09-27 regression run (`VRAM: … held` after SD 1.5 ↔ SDXL switches).
21. **"DPM++ 2M Karras" wasn't Karras** (no `use_karras_sigmas`) → see "Samplers, AYS, …"; old records map to
    "DPM++ 2M". Same section: FreeU crashed under ZLUDA (cuFFT) → CPU FFT fallback.

## Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| Batch file flashes and closes | CP1252/paren issue — run it from `cmd` to see the error |
| `CUDA not available` / device is cpu | Not launched via `launch.bat` / `run_zluda.bat`, or HIP runtime missing |
| Black / noisy images, NaN | Run `run_zluda.bat selftest_zluda.py`; check `IMAGEGEN_CUDNN` isn't set; LoRA/base mismatch |
| Generation suddenly very slow | VRAM spilled to shared memory — look for "⚠ VRAM over-committed", lower size/batch |
| First generation of a session is slow | Normal: ZLUDA loads kernels per process (~1 min); on a new PC ~15 min once (empty zluda.db) |
| Process won't exit | ZLUDA shutdown hang — close the window / `taskkill /F /T` the zluda.exe tree |
| "… is incomplete" / "isn't a valid .safetensors file" | Truncated/corrupt file (`model_manager.safetensors_problem()`) — re-download |
| Prompt weights seem ignored | Must go through `prompt_syntax.a1111_to_compel()`; Compel reads raw `(x:1.2)` as text |
| HF models land in `~/.cache/huggingface` | Something imported huggingface_hub before `config` set `HF_HOME` (app.py imports config before gradio) |
| Gradio stuck loading | index.html patches lost after a gradio upgrade |
| Accel-TE "Session creation failed" | No iGPU found or DML EP missing (`onnxruntime-directml`) |
| NPU tests skipped | Expected without the Ryzen AI SDK conda env |
