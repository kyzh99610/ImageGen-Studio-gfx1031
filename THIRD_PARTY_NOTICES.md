# Third-party components

ImageGen Studio's own code is under the MIT license ([LICENSE](LICENSE)). The repository contains **no**
third-party binaries. `installer/setup.ps1` downloads the following from their official locations and
checks each download against a pinned SHA-256:

| Component | Source | License |
|---|---|---|
| Python 3.10.11 (embeddable) | https://www.python.org/ftp/python/3.10.11/ | PSF License |
| pip (get-pip.py) | https://bootstrap.pypa.io/ | MIT |
| ZLUDA v6 (`zluda-windows-3fe1206.zip`) | https://github.com/vosen/ZLUDA/releases/tag/v6 | Apache-2.0 OR MIT |
| gfx1031 rocBLAS kernels (`rocm.gfx1031.for.hip.sdk.6.2.4.littlewu.s.logic.7z`) | https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.6.2.4 | built from AMD rocBLAS / Tensile (MIT); see that repository |

`ImageGenApp/install.bat` installs the Python packages in `ImageGenApp/requirements.txt` (PyTorch, diffusers,
transformers, Gradio, ONNX Runtime DirectML, EasyOCR, …) from PyPI and download.pytorch.org, each under its own license.

You install the **AMD HIP SDK** yourself, under AMD's license terms.

At runtime the app may download models on request: the anime face detector
[lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) (MIT, SHA-256 checked) for Face detail; the WD14 tagger
[SmilingWolf/wd-vit-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) (Apache-2.0) for Interrogate / Auto-Tag; Stable Diffusion checkpoints from Hugging Face or Civitai,
the Real-ESRGAN and LaMa ONNX models, and BLIP for auto-captioning. Each model has its own license. Check it
before using the outputs commercially.
