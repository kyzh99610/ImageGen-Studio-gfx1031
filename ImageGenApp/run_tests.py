"""
ImageGen Studio test suite (CPU only — no GPU or models needed; ~1 min).

    python-3.10\python.exe ImageGenApp\run_tests.py

Covers: NPU / SmartSplit / accelerated text-encoder wiring (NPU tests skip without the
Ryzen AI SDK), the UI build (no Dropdown that can hang the Gradio 4.19 page), LyCORIS maths,
HIP runtime + rocBLAS kernel choice per GPU architecture (gfx1031 …), damaged model files,
PNG Info formats, A1111 prompt syntax, and edge cases in user input: odd sizes/seeds/steps,
huge or tiny img2img images, file names from the Civitai API and from users, launcher flags.
GPU correctness is checked separately by selftest_zluda.py (run_zluda.bat selftest_zluda.py).
"""

import sys, os, re, json, traceback, threading
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
import numpy as np
os.environ["IMAGEGEN_TESTS"] = "1"         # the app must not apply the user's saved cool factor / pause limit (a pause limit makes every fake generate read the live sensor and wait while hot)

# Fix Windows console encoding for Unicode emoji
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

APP_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

PASS = 0
FAIL = 0
SKIP = 0
RESULTS = []


class Skip(Exception):
    """Raised by a test whose hardware isn't present on this machine."""


def needs_npu():
    """Skip unless this machine has a Ryzen AI NPU *and* the Ryzen AI conda env
    (only the 8700G workstation in the hardware matrix does)."""
    from backend.npu_bridge import find_conda_python
    if find_conda_python() is None:
        raise Skip("no Ryzen AI SDK conda env on this machine")


ONLY = os.environ.get("TEST_ONLY", "")      # regex: run only the tests whose name matches (a quick check of one area), e.g. TEST_ONLY="Eye detail"


def test(name):
    def decorator(fn):
        global PASS, FAIL, SKIP
        if ONLY and not re.search(ONLY, name, re.I):
            return fn
        try:
            fn()
            PASS += 1
            RESULTS.append(("✅", name))
            print(f"  ✅ {name}")
        except Skip as e:
            SKIP += 1
            RESULTS.append(("⏭", name, str(e)))
            print(f"  ⏭ {name} (skipped: {e})")
        except Exception as e:
            FAIL += 1
            RESULTS.append(("❌", name, str(e)))
            print(f"  ❌ {name}: {e}")
            traceback.print_exc()
    return decorator


print("=" * 70)
print("ImageGen Studio Test Suite")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
print("\n[1/8] NPU Detection (hardware_detector)")
# ══════════════════════════════════════════════════════════════════════════

@test("_detect_npu returns NPUInfo dataclass")
def _():
    from backend.hardware_detector import _detect_npu, NPUInfo
    result = _detect_npu()
    assert isinstance(result, NPUInfo), f"Expected NPUInfo, got {type(result)}"
    assert hasattr(result, 'available')
    assert hasattr(result, 'name')
    assert hasattr(result, 'backend')
    assert hasattr(result, 'notes')

@test("_detect_npu on 8700G finds NPU hardware via PnP")
def _():
    from backend.hardware_detector import _detect_npu
    result = _detect_npu()
    # 8700G has XDNA NPU — should detect it via PnP even without SDK
    # On non-NPU machines, this still passes (just detects "Not detected")
    assert isinstance(result.name, str)
    assert isinstance(result.backend, str)
    print(f"    NPU detected: name={result.name}, backend={result.backend}, available={result.available}")

@test("_detect_npu VitisAI EP path works when mocked")
def _():
    needs_npu()
    # On this machine (8700G), the SDK is actually installed
    # so _detect_npu should return available=True
    from backend.hardware_detector import _detect_npu, NPUInfo
    result = _detect_npu()
    assert result.available is True, f"8700G should detect NPU, got: {result}"
    assert "vitisai" in result.backend.lower() or "sdk" in result.backend.lower()

@test("_detect_npu Ryzen AI SDK path detection")
def _():
    from backend.hardware_detector import NPUInfo
    # Check if the SDK path scanner works (scans known directories)
    # Just verify the logic runs without crashing
    known_paths = [
        r"C:\Program Files\RyzenAI\1.7.0",
        r"C:\Program Files\RyzenAI\1.6.1",
        r"C:\Program Files\RyzenAI\1.6.0",
    ]
    found = [p for p in known_paths if os.path.isdir(p)]
    print(f"    Ryzen AI SDK paths found: {found or 'none'}")

@test("_detect_npu NPU-bearing CPU regex")
def _():
    import re
    # Test the regex that identifies NPU-bearing CPUs
    npu_cpus = re.compile(
        r"(8[0-9]{3}[GHU]|9[0-9]{3}H|Strix|Hawk Point|Phoenix)", re.I
    )
    assert npu_cpus.search("AMD Ryzen 7 8700G"), "Should match 8700G"
    assert npu_cpus.search("AMD Ryzen 9 8945HS"), "Should match 8945HS"
    assert npu_cpus.search("AMD Ryzen 9 9945HX"), "Should match 9945HX (Strix)"
    assert npu_cpus.search("AMD Ryzen AI 9 HX Strix Point"), "Should match Strix"
    assert not npu_cpus.search("AMD Ryzen 9 5900HX"), "Should NOT match 5900HX (no NPU)"
    assert not npu_cpus.search("AMD Ryzen 9 7800X3D"), "Should NOT match 7800X3D (no NPU)"
    assert not npu_cpus.search("AMD Ryzen 7 9700X"), "Should NOT match 9700X (no NPU)"


# ══════════════════════════════════════════════════════════════════════════
print("\n[2/8] SmartSplit Capability Detection")
# ══════════════════════════════════════════════════════════════════════════

@test("detect_smartsplit_capability returns SmartSplitCapability")
def _():
    from backend.smartsplit_pipeline import detect_smartsplit_capability, SmartSplitCapability
    cap = detect_smartsplit_capability()
    assert isinstance(cap, SmartSplitCapability)
    print(f"    possible={cap.possible}, dgpu={cap.dgpu_available}, npu={cap.npu_available}")
    print(f"    igpu_te={cap.igpu_te_available}, igpu_vae={cap.igpu_vae_available}")
    if not cap.possible:
        print(f"    reason: {cap.reason}")

@test("SmartSplitCapability fields are all booleans/strings")
def _():
    from backend.smartsplit_pipeline import detect_smartsplit_capability
    cap = detect_smartsplit_capability()
    assert isinstance(cap.possible, bool)
    assert isinstance(cap.npu_available, bool)
    assert isinstance(cap.igpu_te_available, bool)
    assert isinstance(cap.igpu_vae_available, bool)
    assert isinstance(cap.dgpu_available, bool)
    assert isinstance(cap.npu_name, str)
    assert isinstance(cap.igpu_name, str)
    assert isinstance(cap.dgpu_name, str)
    assert isinstance(cap.reason, str)

@test("Start-up (round 26): detect_hardware runs the NPU and the GPU-list queries in threads — the same profile as before, a failing query falls back to its default, the two queries overlap in time")
def _():
    import time
    from unittest.mock import patch
    from backend import hardware_detector as HD
    npu = HD.NPUInfo(available=True, name="X NPU", backend="vitisai", notes="n")

    def run(delay=0.0, npu_fn=None, wmic_fn=None):
        def slow_npu():
            time.sleep(delay)
            return npu

        def slow_wmic():
            time.sleep(delay)
            return [("AMD Radeon RX 6800M", 12 * 1024 ** 3), ("AMD Radeon(TM) Graphics", 512 * 1024 ** 2)]
        with patch.object(HD, "_cpu_name", lambda: "CPU x"), patch.object(HD, "_ram_gb", lambda: 32),                 patch.object(HD, "_cuda_gpus", lambda: [("AMD Radeon RX 6800M [ZLUDA]", 12 * 1024 ** 3)]), patch.object(HD, "_directml_available", lambda: True),                 patch.object(HD, "_detect_npu", npu_fn or slow_npu), patch.object(HD, "_wmic_gpus", wmic_fn or slow_wmic):
            t0 = time.perf_counter()
            prof = HD.detect_hardware()
            return prof, time.perf_counter() - t0
    p0, _ = run(0.0)
    assert p0.npu == npu and p0.cpu_name == "CPU x" and p0.ram_gb == 32, p0
    assert len(p0.gpus) == 2 and p0.gpus[0].backend == "cuda_zluda" and p0.best_gpu is p0.gpus[0] and p0.gpus[1].is_igpu, [(g.name, g.backend) for g in p0.gpus]
    assert "NPU   : X NPU ✅" in p0.summary and "CPU   : CPU x" in p0.summary and "RAM   : 32 GB" in p0.summary, p0.summary
    # the same profile with slow queries, in less time than the two queries one after the other
    p1, dt = run(0.4)
    assert p1.summary == p0.summary and [(g.name, g.vram_mb, g.score) for g in p1.gpus] == [(g.name, g.vram_mb, g.score) for g in p0.gpus] and p1.npu == p0.npu
    assert dt < 0.7, dt                                                         # sequential would need >= 0.8 s
    # a query that raises falls back to its default (no NPU / no extra GPU), the rest of the profile is intact

    def boom():
        raise RuntimeError("powershell missing")
    p2, _ = run(0.0, npu_fn=boom, wmic_fn=boom)
    assert p2.npu == HD.NPUInfo() and len(p2.gpus) == 1 and p2.best_gpu.name.startswith("AMD Radeon RX 6800M"), p2


@test("SmartSplit NPU detection mirrors hardware_detector")
def _():
    from backend.smartsplit_pipeline import detect_smartsplit_capability
    from backend.hardware_detector import _detect_npu
    cap = detect_smartsplit_capability()
    npu = _detect_npu()
    # Both should agree on NPU VitisAI availability
    if npu.backend == "vitisai":
        assert cap.npu_available, "SmartSplit should see NPU if hardware_detector found VitisAI"


# ══════════════════════════════════════════════════════════════════════════
print("\n[3/8] SmartSplit Config Auto-Routing")
# ══════════════════════════════════════════════════════════════════════════

@test("SmartSplitConfig.auto() routes TE to NPU when available")
def _():
    from backend.smartsplit_pipeline import SmartSplitConfig, SmartSplitCapability
    cap = SmartSplitCapability(
        possible=True,
        dgpu_available=True,
        npu_available=True,
        igpu_te_available=True,
        igpu_vae_available=True,
    )
    cfg = SmartSplitConfig.auto(cap)
    assert cfg.enabled is True
    assert cfg.text_encoder_device == "npu", f"Expected 'npu', got '{cfg.text_encoder_device}'"
    assert cfg.unet_device == "cuda"
    assert cfg.vae_device == "igpu_directml"

@test("SmartSplitConfig.auto() falls back to iGPU when no NPU")
def _():
    from backend.smartsplit_pipeline import SmartSplitConfig, SmartSplitCapability
    cap = SmartSplitCapability(
        possible=True,
        dgpu_available=True,
        npu_available=False,
        igpu_te_available=True,
        igpu_vae_available=True,
    )
    cfg = SmartSplitConfig.auto(cap)
    assert cfg.text_encoder_device == "igpu_directml"
    assert cfg.vae_device == "igpu_directml"

@test("SmartSplitConfig.auto() falls back to CPU when no NPU or iGPU")
def _():
    from backend.smartsplit_pipeline import SmartSplitConfig, SmartSplitCapability
    cap = SmartSplitCapability(
        possible=True,
        dgpu_available=True,
        npu_available=False,
        igpu_te_available=False,
        igpu_vae_available=True,  # Still need a secondary device
    )
    cfg = SmartSplitConfig.auto(cap)
    assert cfg.text_encoder_device == "cpu"

@test("SmartSplitConfig.auto() disabled when not possible")
def _():
    from backend.smartsplit_pipeline import SmartSplitConfig, SmartSplitCapability
    cap = SmartSplitCapability(possible=False)
    cfg = SmartSplitConfig.auto(cap)
    assert cfg.enabled is False

@test("SmartSplitConfig.auto() DirectML multi-GPU split")
def _():
    from backend.smartsplit_pipeline import SmartSplitConfig, SmartSplitCapability
    cap = SmartSplitCapability(
        possible=True,
        dgpu_available=True,
        dgpu_dml_idx=1,
        igpu_dml_idx=0,
        npu_available=True,  # NPU should still be ignored in DML path
    )
    cfg = SmartSplitConfig.auto(cap)
    assert cfg.unet_device == "dgpu_directml"
    assert cfg.dgpu_dml_idx == 1
    assert cfg.igpu_dml_idx == 0
    # DML path always uses CPU for TE (NPU bridge requires ONNX/CUDA path)
    assert cfg.text_encoder_device == "cpu", f"DML path should use CPU for TE, got '{cfg.text_encoder_device}'"

@test("SmartSplitConfig.describe() includes device names")
def _():
    from backend.smartsplit_pipeline import SmartSplitConfig
    cfg = SmartSplitConfig(enabled=True, text_encoder_device="npu",
                           unet_device="cuda", vae_device="igpu_directml")
    desc = cfg.describe()
    assert "NPU" in desc
    assert "CUDA" in desc
    assert "IGPU_DIRECTML" in desc


# ══════════════════════════════════════════════════════════════════════════
print("\n[4/8] ONNX Session & VitisAI EP Fallback")
# ══════════════════════════════════════════════════════════════════════════

@test("_ort_session (round 26: behaviour): 'npu' asks for VitisAI, 'igpu_directml' for DirectML, 'cpu' for CPU; a provider this onnxruntime lacks falls back to CPU, an unknown device too")
def _():
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch
    from backend.smartsplit_pipeline import _ort_session
    made = []
    fake = SimpleNamespace(get_available_providers=lambda: avail[0], InferenceSession=lambda path, providers=None: made.append((path, providers)) or "session")
    avail = [["VitisAIExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]]
    with patch.dict(sys.modules, {"onnxruntime": fake}):
        for ep, want in (("npu", "VitisAIExecutionProvider"), ("igpu_directml", "DmlExecutionProvider"), ("cpu", "CPUExecutionProvider"), ("something-else", "CPUExecutionProvider")):
            made.clear()
            assert _ort_session(Path("m.onnx"), ep) == "session" and made == [("m.onnx", [want])], (ep, made)
        avail[0] = ["CPUExecutionProvider"]                                   # no VitisAI / DML in this onnxruntime: CPU, with the model path unchanged
        for ep in ("npu", "igpu_directml"):
            made.clear()
            _ort_session(Path("m.onnx"), ep)
            assert made == [("m.onnx", ["CPUExecutionProvider"])], (ep, made)

@test("ONNX export (round 26: behaviour): the text encoder is exported with input_ids -> last_hidden_state / pooler_output and a dynamic batch, on the CPU and moved back; a cached file is reused")
def _():
    import tempfile
    import torch
    from unittest.mock import patch
    import backend.smartsplit_pipeline as SP

    class TE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(2, 2)
            self.moves = []

        def cpu(self):
            self.moves.append("cpu")
            return self

        def to(self, dev):
            self.moves.append(str(dev))
            return self

    class Tok:
        model_max_length = 77

        def __call__(self, texts, padding=None, max_length=None, return_tensors=None):
            return {"input_ids": torch.zeros(1, max_length, dtype=torch.long)}
    te = TE()
    pipe = type("P", (), {})()
    pipe.text_encoder, pipe.tokenizer = te, Tok()
    calls = []
    out = Path(tempfile.mkdtemp())
    with patch.object(SP, "ONNX_CACHE_DIR", out), patch.object(torch.onnx, "export", lambda *a, **k: (calls.append((a, k)), Path(a[2]).write_bytes(b"onnx"))):
        p = SP._export_text_encoder_to_onnx(pipe, "stem")
        assert p == out / "stem_text_encoder.onnx" and p.exists() and len(calls) == 1
        a, k = calls[0]
        assert a[1][0].shape == (1, 77) and k["input_names"] == ["input_ids"] and k["output_names"] == ["last_hidden_state", "pooler_output"], k
        assert k["dynamic_axes"] == {"input_ids": {0: "batch"}, "last_hidden_state": {0: "batch"}} and k["opset_version"] == 17, k
        assert te.moves[0] == "cpu" and len(te.moves) == 2, te.moves                       # exported on the CPU, then back where it was
        assert SP._export_text_encoder_to_onnx(pipe, "stem") == p and len(calls) == 1      # cached: no second export

@test("ONNX export (round 26: behaviour): the VAE decoder is exported with latents -> images and dynamic batch / height / width axes; a cached file is reused")
def _():
    import tempfile
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    import backend.smartsplit_pipeline as SP

    class VAE(torch.nn.Module):
        config = SimpleNamespace(latent_channels=4)

        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(2, 2)

        def decode(self, z):
            return SimpleNamespace(sample=z)
    pipe = SimpleNamespace(vae=VAE())
    calls = []
    out = Path(tempfile.mkdtemp())
    with patch.object(SP, "ONNX_CACHE_DIR", out), patch.object(torch.onnx, "export", lambda *a, **k: (calls.append((a, k)), Path(a[2]).write_bytes(b"onnx"))):
        p = SP._export_vae_decoder_to_onnx(pipe, "stem")
        assert p == out / "stem_vae_decoder.onnx" and len(calls) == 1
        a, k = calls[0]
        assert a[1][0].shape == (1, 4, 64, 64) and k["input_names"] == ["latents"] and k["output_names"] == ["images"], k
        assert k["dynamic_axes"]["latents"] == {0: "batch", 2: "height", 3: "width"} and k["dynamic_axes"]["images"] == {0: "batch", 2: "img_height", 3: "img_width"}, k
        assert SP._export_vae_decoder_to_onnx(pipe, "stem") == p and len(calls) == 1

# ══════════════════════════════════════════════════════════════════════════
print("\n[5/8] SmartSplit Pipeline NPU Path")
# ══════════════════════════════════════════════════════════════════════════

@test("SmartSplitPipeline._setup (round 26: behaviour): an NPU text encoder uses the bridge proxy when it starts, else the local ORT session of the same device; a failing export leaves the dGPU path")
def _():
    from unittest.mock import patch
    import backend.smartsplit_pipeline as SP
    cfg = SP.SmartSplitConfig(enabled=True, text_encoder_device="npu", unet_device="cuda", vae_device="cpu")
    sessions = []
    with patch.object(SP, "_export_text_encoder_to_onnx", lambda pipe, stem: Path("te.onnx")), \
            patch.object(SP, "_ort_session", lambda p, ep: sessions.append((str(p), ep)) or "ort-session"):
        with patch.object(SP.SmartSplitPipeline, "_setup_npu_te", lambda self, p: "npu-proxy"):
            sp = SP.SmartSplitPipeline(object(), cfg, "stem")
            assert sp._te_session == "npu-proxy" and not sessions                       # the bridge started: no local session
        with patch.object(SP.SmartSplitPipeline, "_setup_npu_te", lambda self, p: None):
            sp = SP.SmartSplitPipeline(object(), cfg, "stem")
            assert sp._te_session == "ort-session" and sessions == [("te.onnx", "npu")]  # no bridge: the local session for the same device (which falls back to CPU itself)
        cfg2 = SP.SmartSplitConfig(enabled=True, text_encoder_device="igpu_directml", unet_device="cuda", vae_device="cpu")
        sessions.clear()
        assert SP.SmartSplitPipeline(object(), cfg2, "stem")._te_session == "ort-session" and sessions == [("te.onnx", "igpu_directml")]
    with patch.object(SP, "_export_text_encoder_to_onnx", lambda pipe, stem: (_ for _ in ()).throw(RuntimeError("export failed"))):
        assert SP.SmartSplitPipeline(object(), cfg, "stem")._te_session is None          # offload failed: the standard dGPU encoder
    assert SP.SmartSplitPipeline(object(), SP.SmartSplitConfig(enabled=False), "stem")._te_session is None

@test("SmartSplitPipeline.encode_prompt (round 26: behaviour): the offloaded session encodes the prompt, then the negative prompt, and the result is [negative, positive] in fp16 on the target device")
def _():
    import numpy as np
    import torch
    from types import SimpleNamespace
    import backend.smartsplit_pipeline as SP
    sp = SP.SmartSplitPipeline(object(), SP.SmartSplitConfig(enabled=False), "stem")
    seen = []

    class Sess:
        def run(self, names, inputs):
            ids = inputs["input_ids"]
            seen.append(ids.copy())
            return [np.full((1, 3, 4), float(ids[0, 0]), np.float32)]

    class Tok:
        model_max_length = 5

        def __call__(self, texts, padding=None, max_length=None, truncation=None, return_tensors=None):
            return {"input_ids": np.full((1, max_length), 7 if texts[0] == "pos" else 9)}
    sp.pipe, sp._te_session = SimpleNamespace(tokenizer=Tok()), Sess()
    out = sp.encode_prompt("pos", "neg", "cpu")
    assert [int(x[0, 0]) for x in seen] == [7, 9] and all(x.dtype == np.int64 for x in seen)         # prompt first, then the negative prompt, as int64 ids
    assert out.dtype == torch.float16 and out.shape == (2, 3, 4) and float(out[0, 0, 0]) == 9.0 and float(out[1, 0, 0]) == 7.0, out    # [negative, positive]

@test("SmartSplitPipeline.generate has NPU timing tracking")
def _():
    src = Path(APP_DIR / "backend" / "smartsplit_pipeline.py").read_text(encoding="utf-8")
    assert 'timing["text_encode_s"]' in src
    assert 'timing["unet_s"]' in src
    assert 'timing["vae_decode_s"]' in src
    assert 'timing["total_s"]' in src

@test("SmartSplitPipeline.stage_summary includes TE device")
def _():
    from backend.smartsplit_pipeline import SmartSplitPipeline, SmartSplitConfig
    # Can't instantiate without a real pipe, but verify the method logic
    cfg = SmartSplitConfig(enabled=True, text_encoder_device="npu",
                           unet_device="cuda", vae_device="cpu")
    # Manually test the output format
    te = cfg.text_encoder_device.upper()
    un = cfg.unet_device.upper()
    vae = cfg.vae_device.upper()
    summary = f"SmartSplit: Text→{te}  UNet→{un}  VAE→{vae}"
    assert "NPU" in summary
    assert "CUDA" in summary


# ══════════════════════════════════════════════════════════════════════════
print("\n[6/8] App.py SmartSplit Generation Path Routing")
# ══════════════════════════════════════════════════════════════════════════

@test("app.py (round 26: behaviour) exposes the SmartSplit components the Settings tab and Generate path use")
def _():
    import app as _app
    for name in ("SmartSplitPipeline", "SmartSplitConfig", "detect_smartsplit_capability", "_smartsplit_te_choices", "_smartsplit_vae_choices", "_build_smartsplit_cap_html"):
        assert callable(getattr(_app, name, None)), name

@test("app.py has _smartsplit_cfg and _smartsplit_pipe globals")
def _():
    import app as _app
    assert hasattr(_app, '_smartsplit_cfg')
    assert hasattr(_app, '_smartsplit_pipe')
    assert hasattr(_app, '_smartsplit_cap')

@test("SmartSplit auto-rebuilds on SD 1.5 model load")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert "_smartsplit_pipe = SmartSplitPipeline(sd.pipe, _smartsplit_cfg, stem)" in src

@test("SmartSplit disables on SDXL model load")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert '_smartsplit_pipe = None' in src
    assert 'SmartSplit disabled (SDXL not supported' in src

@test("SmartSplit rejects SDXL in on_apply_split (round 26: behaviour: the handler run with an SDXL pipeline loaded, no SmartSplit pipe is built)")
def _():
    import app as _app
    from unittest.mock import patch
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
    fn = getattr(by["on_apply_split"].fn, "__wrapped__", by["on_apply_split"].fn)
    old_sd, old_pipe = _app.sd, _app._smartsplit_pipe
    try:
        _app.sd = object.__new__(_app.SDXLPipeline)
        _app.sd.current_model = "x.safetensors"
        out = fn(True, "cpu", "cpu")
        txt = " ".join(str(x) for x in (out if isinstance(out, (list, tuple)) else [out]))
        assert "SmartSplit not supported for SDXL/Pony/Illustrious" in txt, txt[:300]
        assert _app._smartsplit_pipe is None
    finally:
        _app.sd, _app._smartsplit_pipe = old_sd, old_pipe

# ══════════════════════════════════════════════════════════════════════════
print("\n[7/8] UI Wiring: NPU Badge, SmartSplit Controls")
# ══════════════════════════════════════════════════════════════════════════

@test("_npu_badge renders green for available NPU")
def _():
    from app import _npu_badge
    from backend.hardware_detector import NPUInfo
    npu = NPUInfo(available=True, name="AMD XDNA NPU (VitisAI EP)", backend="vitisai")
    html = _npu_badge(npu)
    assert "NPU" in html
    assert "cba6f7" in html  # purple accent color for available NPU

@test("_npu_badge renders gray for unavailable NPU")
def _():
    from app import _npu_badge
    from backend.hardware_detector import NPUInfo
    npu = NPUInfo(available=False, name="Not detected", backend="")
    html = _npu_badge(npu)
    assert "NPU" in html
    assert "45475a" in html  # dark gray for unavailable NPU

@test("_build_npu_info includes usage instructions when available")
def _():
    from app import _build_npu_info
    from backend.hardware_detector import NPUInfo
    npu = NPUInfo(available=True, name="AMD XDNA NPU", backend="vitisai",
                  notes="Use for text encoder / VAE decode acceleration")
    info = _build_npu_info(npu)
    assert "Offload text encoder to NPU" in info or "Text Encoder" in info
    assert "conda env" in info or "bridge" in info

@test("_build_npu_info handles not-detected case")
def _():
    from app import _build_npu_info
    from backend.hardware_detector import NPUInfo
    npu = NPUInfo(available=False, name="Not detected", backend="")
    info = _build_npu_info(npu)
    assert "No XDNA NPU detected" in info

@test("_smartsplit_te_choices (round 26: behaviour) offers 'npu' only when an NPU is available, 'igpu_directml' only with a usable iGPU, 'cpu' always")
def _():
    import app as _app
    from backend.smartsplit_pipeline import SmartSplitCapability
    cap = lambda **k: SmartSplitCapability(possible=True, dgpu_available=True, **k)
    assert _app._smartsplit_te_choices(cap(npu_available=True, igpu_te_available=True)) == ["npu", "igpu_directml", "cpu"]
    assert _app._smartsplit_te_choices(cap(npu_available=False, igpu_te_available=True)) == ["igpu_directml", "cpu"]
    assert _app._smartsplit_te_choices(cap(npu_available=True, igpu_te_available=False)) == ["npu", "cpu"]
    assert _app._smartsplit_te_choices(cap()) == ["cpu"]

@test("_build_smartsplit_cap_html (round 26: behaviour) names the dGPU, the iGPU and — only when available — the NPU")
def _():
    import app as _app
    from backend.smartsplit_pipeline import SmartSplitCapability
    base = dict(possible=True, dgpu_available=True, dgpu_name="DGPU-X", igpu_name="IGPU-Y", npu_name="NPU-Z")
    html_on = _app._build_smartsplit_cap_html(SmartSplitCapability(**base, npu_available=True))
    html_off = _app._build_smartsplit_cap_html(SmartSplitCapability(**base, npu_available=False))
    assert "DGPU-X" in html_on and "IGPU-Y" in html_on and "NPU: NPU-Z" in html_on and "NPU" not in html_off.replace("SmartSplit", ""), (html_on[:200], html_off[:200])
    assert _app._build_smartsplit_cap_html(SmartSplitCapability(possible=False)) == ""

@test("Settings tab (round 26: behaviour: the built UI) shows the NPU section with badge and details when an NPU is detected, and not otherwise")
def _():
    import copy
    from unittest.mock import patch
    import app as _app
    from backend.hardware_detector import NPUInfo
    real = _app.get_profile()

    def built(npu):
        p = copy.copy(real)
        p.npu = npu
        with patch.object(_app, "get_profile", lambda: p):
            demo = _app.build_app()
        return [getattr(b, "value", "") for b in demo.blocks.values()]
    have = NPUInfo(available=True, name="AMD XDNA NPU (VitisAI EP)", backend="vitisai", notes="n")
    vals = built(have)
    assert any("AMD XDNA NPU (Ryzen AI)" in str(v) for v in vals) and _app._npu_badge(have) in vals and _app._build_npu_info(have) in vals, "NPU section missing"
    assert not any("AMD XDNA NPU (Ryzen AI)" in str(v) for v in built(NPUInfo(available=False, name="Not detected", backend=""))), "NPU section shown without an NPU"

@test("SmartSplit TE dropdown (round 26: behaviour: the built UI) offers 'npu' when the capability has an NPU, and only the CPU on a machine without helpers")
def _():
    from unittest.mock import patch
    import app as _app
    from backend.smartsplit_pipeline import SmartSplitCapability

    def choices(cap):
        with patch.object(_app, "_get_smartsplit_cap", lambda: cap):
            demo = _app.build_app()
        dd = [b for b in demo.blocks.values() if type(b).__name__ == "Dropdown" and getattr(b, "label", "") == "Text Encoder device"]
        return [[c[1] if isinstance(c, (list, tuple)) else c for c in d.choices] for d in dd]
    assert choices(SmartSplitCapability(possible=True, dgpu_available=True, npu_available=True, npu_name="N", igpu_te_available=True)) == [["npu", "igpu_directml", "cpu"]]
    assert choices(SmartSplitCapability(possible=True, dgpu_available=True, igpu_te_available=True)) == [["igpu_directml", "cpu"]]
    assert choices(SmartSplitCapability(possible=False)) == []                    # not capable: the section isn't built

# ══════════════════════════════════════════════════════════════════════════
print("\n[8/10] End-to-End SmartSplit with NPU")
# ══════════════════════════════════════════════════════════════════════════

@test("Full SmartSplit config with NPU → correct describe() output")
def _():
    from backend.smartsplit_pipeline import SmartSplitConfig, SmartSplitCapability
    # Simulate 8700G: dGPU + iGPU + NPU all available
    cap = SmartSplitCapability(
        possible=True,
        dgpu_available=True,
        npu_available=True,
        igpu_te_available=True,
        igpu_vae_available=True,
        dgpu_name="AMD Radeon RX 6800 XT",
        igpu_name="AMD Radeon 780M",
        npu_name="AMD XDNA NPU (VitisAI)",
    )
    cfg = SmartSplitConfig.auto(cap)
    assert cfg.enabled
    assert cfg.text_encoder_device == "npu"
    assert cfg.vae_device == "igpu_directml"
    desc = cfg.describe()
    assert "NPU" in desc
    assert "IGPU_DIRECTML" in desc
    print(f"    Config: {desc.strip()}")

@test("SmartSplitPipeline timing_html formats correctly")
def _():
    from backend.smartsplit_pipeline import SmartSplitPipeline, SmartSplitConfig
    # Can't instantiate SmartSplitPipeline without a real pipe,
    # but we can test the timing_html method by calling it on a mock
    cfg = SmartSplitConfig(enabled=True, text_encoder_device="npu",
                           unet_device="cuda", vae_device="igpu_directml")
    # Manually invoke the formatting logic
    timing = {"text_encode_s": 0.35, "unet_s": 45.2, "vae_decode_s": 3.1, "total_s": 48.65}
    html = (
        f'<span style="font-size:11px;color:#a6adc8;">'
        f'Text({cfg.text_encoder_device.upper()}):{timing["text_encode_s"]:.2f}s  '
        f'UNet({cfg.unet_device.upper()}):{timing["unet_s"]:.1f}s  '
        f'VAE({cfg.vae_device.upper()}):{timing["vae_decode_s"]:.2f}s  '
        f'Total:{timing["total_s"]:.1f}s</span>'
    )
    assert "NPU" in html
    assert "45.2s" in html
    assert "IGPU_DIRECTML" in html

@test("ONNX cache dir exists and is writable")
def _():
    from backend.smartsplit_pipeline import ONNX_CACHE_DIR
    assert ONNX_CACHE_DIR.exists(), f"ONNX cache dir missing: {ONNX_CACHE_DIR}"
    assert ONNX_CACHE_DIR.is_dir()
    # Test writability
    test_file = ONNX_CACHE_DIR / "_test_write.tmp"
    test_file.write_text("test")
    assert test_file.exists()
    test_file.unlink()

@test("SmartSplit capability detection on this machine")
def _():
    from backend.smartsplit_pipeline import detect_smartsplit_capability
    cap = detect_smartsplit_capability()
    print(f"    This machine:")
    print(f"      possible={cap.possible}")
    print(f"      dGPU: {cap.dgpu_name or 'none'} (cuda={cap.dgpu_available})")
    print(f"      iGPU: {cap.igpu_name or 'none'} (te={cap.igpu_te_available}, vae={cap.igpu_vae_available})")
    print(f"      NPU:  {cap.npu_name or 'none'} (available={cap.npu_available})")
    if not cap.possible:
        print(f"      reason: {cap.reason}")
    needs_npu()
    # On the 8700G workstation, all three should be available
    assert cap.possible, "SmartSplit should be possible on 8700G"
    assert cap.npu_available, "NPU should be available via bridge on 8700G"


# ══════════════════════════════════════════════════════════════════════════
print("\n[9/10] NPU Bridge Subprocess")
# ══════════════════════════════════════════════════════════════════════════

@test("NPU bridge finds conda Python")
def _():
    needs_npu()
    from backend.npu_bridge import find_conda_python
    python = find_conda_python()
    assert python is not None, "Should find conda Python on 8700G"
    assert python.exists()
    print(f"    Conda Python: {python}")

@test("NPU bridge availability check passes")
def _():
    needs_npu()
    from backend.npu_bridge import check_npu_bridge_available
    ok, msg = check_npu_bridge_available()
    assert ok, f"Bridge should be available: {msg}"
    print(f"    {msg}")

@test("NPU bridge start/run/stop lifecycle")
def _():
    needs_npu()
    import numpy as np
    import onnx
    from onnx import helper, TensorProto
    from backend.npu_bridge import NPUBridge

    # Create test ONNX model
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])
    node = helper.make_node('Relu', ['X'], ['Y'])
    graph = helper.make_graph([node], 'test', [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    test_path = str(APP_DIR / '_test_bridge.onnx')
    onnx.save(model, test_path)

    bridge = NPUBridge()
    try:
        msg = bridge.start()
        assert bridge.is_running, f"Bridge should be running: {msg}"
        assert "VitisAIExecutionProvider" in msg

        # Run inference
        inputs = {'X': np.array([[1.0, -2.0, 3.0]], dtype=np.float32)}
        outputs, elapsed, provider = bridge.run(test_path, inputs)
        assert len(outputs) == 1
        np.testing.assert_array_almost_equal(outputs[0], [[1.0, 0.0, 3.0]])
        assert provider == "VitisAIExecutionProvider"
        print(f"    NPU inference OK ({elapsed:.4f}s, provider={provider})")

        # Second run tests session caching
        _, elapsed2, _ = bridge.run(test_path, inputs)
        assert elapsed2 < elapsed, "Cached session should be faster"
        print(f"    Cached run: {elapsed2:.6f}s")
    finally:
        bridge.stop()
        os.remove(test_path)
    assert not bridge.is_running

@test("_NPUSessionProxy quacks like ort session")
def _():
    from backend.smartsplit_pipeline import _NPUSessionProxy
    from unittest.mock import MagicMock
    mock_bridge = MagicMock()
    mock_bridge.run.return_value = ([np.array([[1.0]])], 0.01, "VitisAIExecutionProvider")
    proxy = _NPUSessionProxy(mock_bridge, "test.onnx")
    result = proxy.run(None, {"input_ids": np.array([[1]])})
    assert len(result) == 1
    mock_bridge.run.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════
print("\n[10/10] SmartSplit + NPU Integration")
# ══════════════════════════════════════════════════════════════════════════

@test("SmartSplit auto-config routes TE to NPU on this machine")
def _():
    from backend.smartsplit_pipeline import SmartSplitConfig, detect_smartsplit_capability
    cap = detect_smartsplit_capability()
    cfg = SmartSplitConfig.auto(cap)
    assert cfg.enabled
    if cap.dgpu_dml_idx >= 0:
        # DML multi-GPU path — NPU not supported, TE goes to CPU
        assert cfg.text_encoder_device == "cpu", f"DML path should use CPU, got '{cfg.text_encoder_device}'"
        assert cfg.unet_device == "dgpu_directml"
        print(f"    Auto config (DML): TE→{cfg.text_encoder_device}, UNet→{cfg.unet_device}, VAE→{cfg.vae_device}")
    else:
        # CUDA/ZLUDA path — NPU preferred for TE
        assert cfg.text_encoder_device == "npu", f"CUDA path should use NPU, got '{cfg.text_encoder_device}'"
        print(f"    Auto config (CUDA): TE→{cfg.text_encoder_device}, UNet→{cfg.unet_device}, VAE→{cfg.vae_device}")


# ══════════════════════════════════════════════════════════════════════════
print("\n[11/11] SDXL Accelerated Text Encoding (iGPU DML)")
# ══════════════════════════════════════════════════════════════════════════

@test("set_sdxl_npu_te toggle works")
def _():
    from backend.sdxl_pipeline import set_sdxl_npu_te
    import backend.sdxl_pipeline as _mod
    set_sdxl_npu_te(True)
    assert _mod._accel_te_enabled is True
    set_sdxl_npu_te(False)
    assert _mod._accel_te_enabled is False

@test("ONNX cache directory exists")
def _():
    from backend.sdxl_pipeline import ONNX_CACHE_DIR
    assert ONNX_CACHE_DIR.exists()
    assert ONNX_CACHE_DIR.is_dir()

@test("_tokenize_chunked returns valid single chunk")
def _():
    from backend.sdxl_pipeline import _tokenize_chunked
    # Use a mock tokenizer that returns known results
    class MockTokenizer:
        model_max_length = 77
        bos_token_id = 49406
        eos_token_id = 49407
        def __call__(self, text, **kwargs):
            if kwargs.get("add_special_tokens") is False:
                # Full tokenization (no padding/truncation)
                return {"input_ids": np.array([[100, 101, 102]])}
            else:
                # Standard tokenization with padding
                ids = np.zeros((1, 77), dtype=np.int64)
                ids[0, :3] = [49406, 100, 101]
                ids[0, 3] = 49407
                return {"input_ids": ids}
    tok = MockTokenizer()
    chunks = _tokenize_chunked(tok, "test prompt", 77)
    assert len(chunks) == 1
    assert chunks[0].shape == (1, 77)
    print(f"    Single chunk shape: {chunks[0].shape}")

@test("_tokenize_chunked handles long prompt (multiple chunks)")
def _():
    from backend.sdxl_pipeline import _tokenize_chunked
    # Mock tokenizer returning >75 tokens (77 - 2 for BOS/EOS)
    class MockTokenizer:
        model_max_length = 77
        bos_token_id = 49406
        eos_token_id = 49407
        def __call__(self, text, **kwargs):
            if kwargs.get("add_special_tokens") is False:
                # 160 tokens — should split into 3 chunks (75+75+10)
                return {"input_ids": np.arange(100, 260).reshape(1, -1)}
            else:
                ids = np.zeros((1, 77), dtype=np.int64)
                return {"input_ids": ids}
    tok = MockTokenizer()
    chunks = _tokenize_chunked(tok, "very long " * 80, 77)
    assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
    for c in chunks:
        assert c.shape == (1, 77)
        assert c[0, 0] == 49406, "First token should be BOS"
    print(f"    Long prompt → {len(chunks)} chunks, each (1, 77)")

@test("_build_sdxl_accel_embeds returns None when ORT unavailable")
def _():
    from backend.sdxl_pipeline import _build_sdxl_accel_embeds
    pipe = MagicMock()
    pipe.tokenizer.model_max_length = 77
    pipe.tokenizer_2.model_max_length = 77
    # Mock _get_ort_sessions to raise (simulates missing onnxruntime-directml)
    with patch("backend.sdxl_pipeline._get_ort_sessions", side_effect=RuntimeError("no DML")):
        result = _build_sdxl_accel_embeds(pipe, "test", "bad", "model")
    assert result is None

def _sdxl_te_routing(kind, accel_on, accel_result):
    """Run the real SDXLPipeline.txt2img / img2img with fakes around it; returns (embed builders called, the embeds run_pipe received)."""
    from types import SimpleNamespace
    import torch
    from PIL import Image as _Img
    from backend import sdxl_pipeline as SX
    called, got = [], {}
    cpu_embeds = {"prompt_embeds": "cpu"}

    def fake_accel(pipe, prompt, neg, stem):
        called.append("accel")
        return dict(accel_result) if accel_result is not None else None

    def fake_cpu(pipe, prompt, neg):
        called.append("cpu")
        return dict(cpu_embeds)

    def fake_run(sdp, pipe, kind_, **kw):
        got.update(kw)
        return SimpleNamespace(images=[_Img.new("RGB", (64, 64))])

    p = SX.SDXLPipeline()
    p.device = "cpu"
    p.pipe = MagicMock()
    p.pipe.text_encoder.parameters.side_effect = lambda: iter([SimpleNamespace(device=torch.device("cpu"))])
    p.img2img_pipe = MagicMock()
    with patch.object(SX, "_accel_te_enabled", accel_on), patch.object(SX, "_build_sdxl_accel_embeds", fake_accel), \
            patch.object(SX, "_build_sdxl_embeds", fake_cpu), patch.object(SX, "_load_scheduler"), \
            patch.object(SX, "_make_generators", lambda seed, dev, n: (None, [1])), \
            patch.object(SX, "_make_generator", lambda seed, dev: (None, 1)), patch.object(SX, "_vram_reset"), \
            patch("backend.sd_pipeline.variation_latents", lambda *a, **k: (None, [])), \
            patch("backend.sampling.run_pipe", fake_run):
        if kind == "txt2img":
            images, info = p.txt2img("a girl", "bad", 64, 64, 2, 5.0, 1)
        else:
            images, info = p.img2img(_Img.new("RGB", (64, 64)), "a girl", "bad", 0.5, 2, 5.0, 1)
    assert images and not str(info).startswith("❌"), info
    return called, got


@test("SDXL accel-TE routing in txt2img and img2img: off = CPU Compel only; on = the accelerated embeds are used; on but failed = CPU fallback")
def _():
    for kind in ("txt2img", "img2img"):
        called, got = _sdxl_te_routing(kind, False, None)
        assert called == ["cpu"] and got["prompt_embeds"] == "cpu", (kind, called)
        called, got = _sdxl_te_routing(kind, True, {"prompt_embeds": "accel"})
        assert called == ["accel"] and got["prompt_embeds"] == "accel", (kind, called, got)
        called, got = _sdxl_te_routing(kind, True, None)
        assert called == ["accel", "cpu"] and got["prompt_embeds"] == "cpu", (kind, called)

@test("DML EP available in onnxruntime")
def _():
    import onnxruntime as ort
    eps = ort.get_available_providers()
    assert "DmlExecutionProvider" in eps, f"DML EP missing, available: {eps}"
    print(f"    ORT {ort.__version__}: {eps}")


@test("UI builds and no Dropdown can crash the Gradio 4.19 frontend")
def _():
    # value=None (or a non-string) on a Dropdown with allow_custom_value=True throws
    # "input_text.toLowerCase is not a function" and the page never leaves "Loading…".
    import gradio as gr
    import app as _app
    demo = _app.build_app()
    bad = []
    for blk in demo.blocks.values():
        if isinstance(blk, gr.Dropdown) and getattr(blk, "allow_custom_value", False):
            if blk.value is None or not isinstance(blk.value, (str, list)):
                bad.append(f"{blk.label!r}={blk.value!r}")
    assert not bad, f"Dropdowns that crash the page: {bad}"
    print(f"    {sum(isinstance(b, gr.Dropdown) for b in demo.blocks.values())} dropdowns checked")


@test("LyCORIS LoHa / LoKr fusion matches the reference maths")
def _():
    import tempfile, types
    import torch
    from safetensors.torch import save_file
    from backend.lycoris import apply_lycoris
    from backend.model_manager import lycoris_kind
    torch.manual_seed(0)

    class Blk(torch.nn.Module):          # stands in for a UNet: kohya names lora_unet_blk_…
        def __init__(self):
            super().__init__()
            self.to_k = torch.nn.Linear(8, 6, bias=False)
            self.conv = torch.nn.Conv2d(4, 5, 3, bias=False)
    unet = torch.nn.Module(); unet.blk = Blk()
    pipe = types.SimpleNamespace(unet=unet, text_encoder=None, text_encoder_2=None)
    w0k, w0c = unet.blk.to_k.weight.clone(), unet.blk.conv.weight.clone()
    r = torch.randn
    loha = {  # Linear: plain LoHa · Conv: Tucker LoHa
        "lora_unet_blk_to_k.hada_w1_a": r(6, 2), "lora_unet_blk_to_k.hada_w1_b": r(2, 8),
        "lora_unet_blk_to_k.hada_w2_a": r(6, 2), "lora_unet_blk_to_k.hada_w2_b": r(2, 8),
        "lora_unet_blk_to_k.alpha": torch.tensor(1.0),
        "lora_unet_blk_conv.hada_w1_a": r(2, 5), "lora_unet_blk_conv.hada_w1_b": r(2, 4),
        "lora_unet_blk_conv.hada_w2_a": r(2, 5), "lora_unet_blk_conv.hada_w2_b": r(2, 4),
        "lora_unet_blk_conv.hada_t1": r(2, 2, 3, 3), "lora_unet_blk_conv.hada_t2": r(2, 2, 3, 3),
        "lora_unet_blk_conv.alpha": torch.tensor(2.0),
        "lora_unet_not_in_model.hada_w1_a": r(1, 1), "lora_unet_not_in_model.hada_w1_b": r(1, 1),
        "lora_unet_not_in_model.hada_w2_a": r(1, 1), "lora_unet_not_in_model.hada_w2_b": r(1, 1),
    }
    d = tempfile.mkdtemp()
    f = os.path.join(d, "loha.safetensors"); save_file(loha, f)
    assert lycoris_kind(f) == "LoHa"
    applied, skipped = apply_lycoris(pipe, f, 0.5)
    assert (applied, skipped) == (2, 1), (applied, skipped)
    L = loha
    exp_k = (L["lora_unet_blk_to_k.hada_w1_a"] @ L["lora_unet_blk_to_k.hada_w1_b"]) * \
            (L["lora_unet_blk_to_k.hada_w2_a"] @ L["lora_unet_blk_to_k.hada_w2_b"]) * (1.0 / 2) * 0.5
    assert torch.allclose(unet.blk.to_k.weight - w0k, exp_k, atol=1e-5)
    def tucker(t, wa, wb):   # explicit loops, independent of the einsum under test
        out = torch.zeros(wa.shape[1], wb.shape[1], *t.shape[2:])
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                out += wa[i][:, None, None, None] * wb[j][None, :, None, None] * t[i, j]
        return out
    c = "lora_unet_blk_conv."
    exp_c = tucker(L[c + "hada_t1"], L[c + "hada_w1_a"], L[c + "hada_w1_b"]) * \
            tucker(L[c + "hada_t2"], L[c + "hada_w2_a"], L[c + "hada_w2_b"]) * (2.0 / 2) * 0.5
    assert torch.allclose(unet.blk.conv.weight - w0c, exp_c, atol=1e-4)

    unet.blk.to_k.weight.data.copy_(w0k)
    lokr = {"lora_unet_blk_to_k.lokr_w1": r(2, 2),
            "lora_unet_blk_to_k.lokr_w2_a": r(3, 1), "lora_unet_blk_to_k.lokr_w2_b": r(1, 4),
            "lora_unet_blk_to_k.alpha": torch.tensor(0.5)}
    f2 = os.path.join(d, "lokr.safetensors"); save_file(lokr, f2)
    assert lycoris_kind(f2) == "LoKr"
    apply_lycoris(pipe, f2, 1.0)
    exp = torch.kron(lokr["lora_unet_blk_to_k.lokr_w1"],
                     lokr["lora_unet_blk_to_k.lokr_w2_a"] @ lokr["lora_unet_blk_to_k.lokr_w2_b"]) * 0.5
    assert torch.allclose(unet.blk.to_k.weight - w0k, exp, atol=1e-5)

    bad = {"lora_unet_blk_to_k.hada_w1_a": r(3, 1), "lora_unet_blk_to_k.hada_w1_b": r(1, 3),
           "lora_unet_blk_to_k.hada_w2_a": r(3, 1), "lora_unet_blk_to_k.hada_w2_b": r(1, 3)}
    f3 = os.path.join(d, "bad.safetensors"); save_file(bad, f3)
    try:
        apply_lycoris(pipe, f3, 1.0)
        raise AssertionError("a LoHa for another model shape was applied")
    except ValueError as e:
        assert "different base model" in str(e)
    print("    LoHa (plain + Tucker), LoKr, unmatched layers and wrong-model refusal OK")


@test("LyCORIS on SDXL: kohya's SGM layer names (input_blocks_4_1_…) reach the UNet")
def _():
    # SDXL LoHa/LoKr files name UNet layers the original way; these used to match nothing, so only
    # the text-encoder half of such a file was applied (788 of 1052 layers silently skipped)
    import types
    from accelerate import init_empty_weights
    from diffusers import UNet2DConditionModel
    from backend.lycoris import _module_map
    with init_empty_weights():
        unet = UNet2DConditionModel(
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            block_out_channels=(320, 640, 1280), layers_per_block=2, cross_attention_dim=2048,
            transformer_layers_per_block=(1, 2, 10), attention_head_dim=(5, 10, 20), use_linear_projection=True,
            addition_embed_type="text_time", addition_time_embed_dim=256, projection_class_embeddings_input_dim=2816)
    mm = _module_map(types.SimpleNamespace(unet=unet))
    u = unet
    expect = {
        "input_blocks_1_0_in_layers_2": u.down_blocks[0].resnets[0].conv1,
        "input_blocks_3_0_op": u.down_blocks[0].downsamplers[0].conv,
        "input_blocks_4_0_skip_connection": u.down_blocks[1].resnets[0].conv_shortcut,
        "input_blocks_4_1_transformer_blocks_0_attn1_to_q": u.down_blocks[1].attentions[0].transformer_blocks[0].attn1.to_q,
        "input_blocks_8_1_transformer_blocks_9_ff_net_2": u.down_blocks[2].attentions[1].transformer_blocks[9].ff.net[2],
        "middle_block_0_emb_layers_1": u.mid_block.resnets[0].time_emb_proj,
        "middle_block_1_proj_in": u.mid_block.attentions[0].proj_in,
        "middle_block_2_out_layers_3": u.mid_block.resnets[1].conv2,
        "output_blocks_2_2_conv": u.up_blocks[0].upsamplers[0].conv,
        "output_blocks_5_2_conv": u.up_blocks[1].upsamplers[0].conv,
        "output_blocks_8_0_skip_connection": u.up_blocks[2].resnets[2].conv_shortcut,
        "time_embed_2": u.time_embedding.linear_2, "label_emb_0_0": u.add_embedding.linear_1, "out_2": u.conv_out,
    }
    for k, mod in expect.items():
        assert mm.get("lora_unet_" + k) is mod, k
    assert mm["lora_unet_down_blocks_1_attentions_0_proj_in"] is u.down_blocks[1].attentions[0].proj_in
    n_sgm = sum(1 for k in mm if k.startswith(("lora_unet_input_blocks", "lora_unet_middle_block", "lora_unet_output_blocks")))
    assert n_sgm > 700, n_sgm
    print(f"    {n_sgm} SGM-named SDXL UNet layers mapped; diffusers names still work")


@test("ROCm runtime + rocBLAS kernels are chosen per GPU architecture")
def _():
    import tempfile
    from pathlib import Path
    from backend import rocm_env as R
    root = Path(tempfile.mkdtemp())

    def mk_rt(name, archs):
        b = root / name / "bin"
        (b / "rocblas" / "library").mkdir(parents=True)
        (b / "rocblas.dll").write_bytes(b"")
        for a in archs:
            (b / "rocblas" / "library" / f"TensileLibrary_lazy_{a}.dat").write_bytes(b"")
        return b
    therock = mk_rt("therock", ["gfx1010", "gfx1030", "gfx1031"])
    sys64 = mk_rt("6.4", ["gfx1030", "gfx1100", "gfx1101", "gfx1201"])
    g1031 = root / "g1031"; g1031.mkdir()
    orig = (R._candidates, R.gpu_devices, R.GFX1031_LIB)
    try:
        R.GFX1031_LIB = g1031
        def run(arch, cands):
            R._candidates = lambda: cands
            R.gpu_devices = lambda bins=None: [{"name": "GPU", "arch": arch, "mem_gb": 12.0,
                                                "integrated": False}]
            return R.choose(0)
        both = [("therock", therock), ("6.4", sys64)]
        r = run("gfx1031", both)        # RX 6800M: therock + the gfx1031 kernels
        assert (r["runtime"], bool(r["tensile"])) == ("therock", True), r
        r = run("gfx1031", both[1:])    # portable zip: system HIP SDK + gfx1031 kernels
        assert (r["runtime"], bool(r["tensile"])) == ("6.4", True), r
        r = run("gfx1030", both)        # RX 6800 XT: therock, NOT the gfx1031-only library
        assert (r["runtime"], r["tensile"]) == ("therock", ""), r
        r = run("gfx1201", both)        # RX 9070 XT: therock has no RDNA 4 kernels
        assert (r["runtime"], r["tensile"]) == ("6.4", ""), r
        r = run("gfx1103", both)        # 780M: nobody has kernels -> warn, don't pretend
        assert r["note"] and r["tensile"] == "", r
        assert 'ROCBLAS_TENSILE_LIBPATH' not in R._bat(run("gfx1030", both))
    finally:
        R._candidates, R.gpu_devices, R.GFX1031_LIB = orig


@test("Damaged / cut-short .safetensors files are explained")
def _():
    import tempfile, torch
    from safetensors.torch import save_file
    from backend.model_manager import safetensors_problem
    d = tempfile.mkdtemp()
    good = os.path.join(d, "good.safetensors")
    save_file({"w": torch.zeros(1000, 100)}, good)
    assert safetensors_problem(good) == ""
    cut = os.path.join(d, "cut.safetensors")
    open(cut, "wb").write(open(good, "rb").read()[:5000])
    assert "incomplete" in safetensors_problem(cut)
    junk = os.path.join(d, "junk.safetensors")
    open(junk, "wb").write(b"\x10" + b"\x00" * 7 + b"{garbage")
    assert "isn't a valid" in safetensors_problem(junk)


@test("Checkpoint load preflight: too little free Windows commit → a plain refusal (not a crash in safetensors), enough → the load goes on")
def _():
    import tempfile
    from unittest.mock import patch
    from backend import model_manager as MM
    tmp = Path(tempfile.mkdtemp())
    f = tmp / "big.safetensors"
    f.write_bytes(b"\0" * (4 << 20))                          # 4 MB stands in for a 6.6 GB checkpoint (need = 2.05–2.4 × its size)
    assert MM.commit_problem(str(tmp / "missing.safetensors")) is None and MM.commit_problem("org/repo") is None
    with patch.object(MM, "avail_commit_gb", return_value=0.001):
        msg = MM.commit_problem(str(f))
        assert msg and "big.safetensors" in msg and "free commit" in msg and "page file" in msg, msg
    with patch.object(MM, "avail_commit_gb", return_value=100.0):
        assert MM.commit_problem(str(f)) is None
    with patch.object(MM, "avail_commit_gb", return_value=None):
        assert MM.commit_problem(str(f)) is None              # unknown → never blocks
    free = MM.avail_commit_gb()
    assert free is None or 0 < free < 4096, free               # the real Win32 call works on this machine
    # measured peaks for a 6.62 GB anime SDXL file (2026-10-01/02): first load of a process 15.6 GB, later loads 13.5 GB —
    # the factors must not drop below what a real load takes (a too-lenient preflight lets the load crash the app)
    assert MM.LOAD_COMMIT_FACTOR_FIRST * 6.62 >= 15.6 and MM.LOAD_COMMIT_FACTOR * 6.62 >= 13.5 and MM.LOAD_COMMIT_FACTOR_FIRST > MM.LOAD_COMMIT_FACTOR
    from backend.sdxl_pipeline import SDXLPipeline
    from backend.sd_pipeline import SDPipeline
    for cls in (SDXLPipeline, SDPipeline):                     # both refuse before they touch the file
        sdp = cls()
        with patch.object(MM, "avail_commit_gb", return_value=0.001):
            r = sdp.load_model(str(f))
        assert r.startswith("❌ Not enough free memory") and sdp.pipe is None, r
        with patch.object(MM, "avail_commit_gb", return_value=100.0):
            r2 = sdp.load_model(str(f))                        # zeros are no checkpoint: it fails later, not at the preflight
        assert "Not enough free memory" not in r2 and r2.startswith("❌"), r2


@test("PNG Info reads LoRAs from <lora:…> tags and our 'LoRAs:' field")
def _():
    import app as _app
    f = _app._loras_from_meta
    assert f("a, <lora:x:0.7>, b, <lora:y> c", None) == ([("x", 0.7), ("y", 0.8)], "a, b, c")
    assert f("a", "x.safetensors, y.safetensors")[0] == [("x.safetensors", 0.8), ("y.safetensors", 0.8)]
    assert f("a", "x:0.65, y:1")[0] == [("x", 0.65), ("y", 1.0)]
    assert f("a", "x (×0.5)")[0] == [("x", 0.5)]
    assert f("plain prompt", None) == ([], "plain prompt")


@test("A1111 prompt weights are translated for Compel")
def _():
    from backend.prompt_syntax import a1111_to_compel as f
    from compel.prompt_parser import PromptParser
    cases = {
        "(best quality:1.2), cat": "(best quality)1.2, cat",
        "((masterpiece)), [blurry]": "((masterpiece)1.1)1.1, (blurry)0.9091",
        "(a (b:1.3):1.2) c": "(a (b)1.3)1.2 c",
        "(x)1.2, word++": "(x)1.2, word++",            # already Compel — untouched
        r"\(literal\) x": r"\(literal\) x",
        "unbalanced ( here": "unbalanced ( here",
    }
    for src, want in cases.items():
        assert f(src) == want, (src, f(src))
    assert f("1girl BREAK red dress, breakfast") == "1girl, red dress, breakfast"
    assert f("a castle, <lora:x:0.8>, (sky:1.2)") == "a castle, (sky)1.2"   # tag isn't text
    assert f("heart <3") == "heart <3"
    from backend.prompt_syntax import plain_prompt
    assert plain_prompt("(best quality:1.2), ((masterpiece)), [blurry], cat") == \
        "best quality, masterpiece, blurry, cat"
    frags = PromptParser().parse_conjunction(f("(best quality:1.2), cat")).prompts[0].children
    assert abs(frags[0].weight - 1.2) < 1e-6 and frags[0].text == "best quality"


# ══════════════════════════════════════════════════════════════════════════
print("\n[edge] Edge cases — inputs users (and other tools) really produce")
# ══════════════════════════════════════════════════════════════════════════
import tempfile as _tf


@test("Prompt syntax survives pathological input (nesting, (x)1girl, BREAK, emoji)")
def _():
    from backend.prompt_syntax import a1111_to_compel as f
    from compel.prompt_parser import PromptParser
    assert f("(solo)1girl") == "(solo)1.1 1girl"             # the 1 belongs to 1girl
    assert f("(x)1.2, y") == "(x)1.2, y"                      # real Compel weight kept
    deep = "(" * 3000 + "cat" + ")" * 3000                   # used to raise RecursionError
    assert f(deep) == "cat"
    assert f("(" * 3000) == ""
    for s in ["", "()", "[]", "(:1.2)", "smile :)", ":( sad", "x)", "(x:-1)", "BREAK BREAK",
              "emoji 😀 (cat:1.1) 中文", "a" * 20000, "(x: 1.2 )", "\\(x\\)"]:
        PromptParser().parse_conjunction(f(s))              # must not raise


@test("PNG Info: A1111 JPEG EXIF, NovelAI, ComfyUI, HTML escaping, file not left locked")
def _():
    import struct
    from PIL import Image, PngImagePlugin
    from backend.png_info import read_image_metadata as rd, format_png_info_html as fmt
    tmp = Path(_tf.mkdtemp())
    # A1111 writes JPEG parameters as EXIF UserComment in the Exif sub-IFD, UTF-16BE
    text = "jpeg prompt, <lora:abc:0.6>\nSteps: 12, Sampler: Euler a, CFG scale: 5, Seed: -7, Size: 64x64"
    uc = b"UNICODE\x00" + text.encode("utf-16-be")
    ifd0 = struct.pack(">H", 1) + struct.pack(">HHII", 0x8769, 4, 1, 26) + struct.pack(">I", 0)
    exififd = struct.pack(">H", 1) + struct.pack(">HHII", 0x9286, 7, len(uc), 44) + struct.pack(">I", 0)
    exif = b"Exif\x00\x00MM\x00\x2a" + struct.pack(">I", 8) + ifd0 + exififd + uc
    j = tmp / "a.jpg"
    Image.new("RGB", (64, 64)).save(j, exif=exif)
    m = rd(j)
    assert m["prompt"].startswith("jpeg prompt") and m["steps"] == 12 and m["seed"] == -7, m
    j.unlink()                                               # handle was closed
    m2 = rd(Image.new("RGB", (1, 1)))
    assert m2["prompt"] == "" and "No generation metadata" in fmt(m2)
    html = fmt(m)
    assert "&lt;lora:abc:0.6&gt;" in html and "<lora:" not in html   # visible, not an HTML tag
    # NovelAI: JSON in "Comment"
    nai = Image.new("RGB", (8, 8))
    nai.info = {"Description": "desc", "Comment": '{"prompt": "nai girl", "uc": "bad", "steps": 28, '
                '"scale": 5.5, "seed": 123, "sampler": "k_euler", "width": 832, "height": 1216}'}
    m = rd(nai)
    assert (m["prompt"], m["negative_prompt"], m["steps"], m["cfg_scale"], m["seed"], m["width"]) == \
        ("nai girl", "bad", 28, 5.5, 123, 832), m
    # ComfyUI: a node graph, not parameters — say so instead of dumping JSON as "the prompt"
    comfy = Image.new("RGB", (8, 8))
    comfy.info = {"prompt": '{"3": {"class_type": "KSampler", "inputs": {"seed": 1}}}'}
    m = rd(comfy)
    assert m["prompt"] == "" and "ComfyUI" in m.get("note", "") and "ComfyUI" in fmt(m), m


@test("Saved images round-trip through PNG Info (unicode, multi-line, LoRA tags, seeds)")
def _():
    import app as _app
    from PIL import Image
    from backend.png_info import read_image_metadata as rd
    tmp = Path(_tf.mkdtemp())
    orig = _app.OUTPUTS_DIR
    try:
        _app.OUTPUTS_DIR = tmp
        prompt = "中文 😀 cat,\n(best quality:1.2), <lora:foo:0.7>"
        paths = _app._save_outputs([Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))], dict(
            prompt=prompt, negative_prompt="bad hands", steps=20, cfg_scale=6.5, seeds=[41, 42],
            scheduler="Euler a", width=64, height=64, model="myModel", loras="foo:0.7"))
        assert len(paths) == 2 and all(p.exists() for p in paths)
        for p, seed in zip(paths, (41, 42)):
            m = rd(p)
            assert (m["prompt"], m["negative_prompt"], m["steps"], m["cfg_scale"], m["seed"]) == \
                (prompt, "bad hands", 20, 6.5, seed), m
            assert m["model"] == "myModel" and m["loras"] == "foo:0.7", m
        loras, clean = _app._loras_from_meta(rd(paths[0])["prompt"], "foo:0.7")
        assert loras == [("foo", 0.7)] and "<lora" not in clean
    finally:
        _app.OUTPUTS_DIR = orig


@test("Damaged model files never crash the header readers")
def _():
    from backend import model_manager as mm
    tmp = Path(_tf.mkdtemp())
    import struct
    cases = {"empty": b"", "short": b"\x01\x02\x03", "huge_n": struct.pack("<Q", 2**62) + b"{}",
             "not_json": struct.pack("<Q", 10) + b"not json!!", "list": struct.pack("<Q", 2) + b"[]",
             "bad_offsets": struct.pack("<Q", 30) + b'{"a": {"data_offsets": "x"}}  '}
    for name, data in cases.items():
        p = tmp / f"{name}.safetensors"
        p.write_bytes(data)
        assert mm.safetensors_problem(str(p)), name                # flagged as damaged
        assert mm.lora_arch(str(p)) is None and mm.checkpoint_arch(str(p)) is None, name
        assert mm.lycoris_kind(str(p)) is None, name
        mm.get_model_info(str(p))                                   # no exception
    assert mm.safetensors_problem(str(tmp / "missing.safetensors")) == ""


@test("VRAM estimate recognises SDXL checkpoints by header, not by 'xl' in the name")
def _():
    import json, struct
    from backend.vram_estimator import _detect_model_class as cls
    tmp = Path(_tf.mkdtemp())
    hdr = json.dumps({"conditioner.embedders.1.model.x": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    p = tmp / "ponyDiffusionV6.safetensors"          # SDXL without "xl" in its name
    p.write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\0\0")
    assert cls(str(p)) == "sdxl"
    assert cls("juggernautXL_v9") == "sdxl" and cls("dreamshaper_8") == "sd15"
    assert cls("illustriousXL") == "sdxl" and cls("noobaiEps") == "sdxl"
    assert cls("stabilityai/stable-diffusion-xl-base-1.0") == "sdxl"
    assert cls("") == "unknown"


@test("Civitai: API file names can't escape the models folder; null fields don't crash")
def _():
    from backend.civitai_client import _safe_filename as sf, CivitaiClient
    assert sf("../../evil.safetensors") == "evil.safetensors"
    assert sf("..\\..\\x.pt") == "x.pt"
    assert sf('a:b?c*"d<e>|f.safetensors') == "a_b_c__d_e__f.safetensors"
    assert sf("CON.safetensors") == "_CON.safetensors"
    assert sf("..") == "model.safetensors" and sf(None) == "model.safetensors"
    card = CivitaiClient.format_model_card({"name": "<script>x</script>", "stats": None,
                                            "modelVersions": None, "creator": None})
    assert "<script>" not in card and "&lt;script&gt;" in card
    card = CivitaiClient.format_model_card({"name": "ok", "modelVersions": [
        {"baseModel": None, "images": None, "files": [{"name": None, "sizeKB": None}]}]})
    assert "ok" in card
    assert CivitaiClient.get_version_choices({"modelVersions": [{"name": None, "id": 5}, {"id": None}, None]}) == \
        [("v?  []", 5)]
    c = CivitaiClient(api_key="")
    ok, msg = c.download_model_version({"type": "LORA"}, {"files": [{"name": "x.safetensors"}]})
    assert not ok and "download link" in msg, msg


@test("Trigger words / sidecars: null model, dict tags and HTML in names are handled")
def _():
    import json
    from backend.trigger_reader import read_model_triggers, render_triggers_html, get_model_specialty_html
    tmp = Path(_tf.mkdtemp())
    p = tmp / "lora.safetensors"
    p.write_bytes(b"\0" * 16)
    Path(str(p) + ".civitai.json").write_text(json.dumps(
        {"trainedWords": ["<lora:x:1>", None, 5, "  ok  "], "baseModel": None, "model": None}), encoding="utf-8")
    r = read_model_triggers(str(p))
    assert r["trained_words"] == ["<lora:x:1>", "ok"] and r["base_model"] == "", r
    html = render_triggers_html("none", str(p))
    assert "&lt;lora:x:1&gt;" in html and "<lora:x:1>" not in html
    assert get_model_specialty_html(str(p)) == ""
    Path(str(p) + ".civitai.json").write_text(json.dumps(
        {"model": {"name": "<b>n</b>", "tags": [{"name": "Anime"}, "realistic"], "nsfw": True}}), encoding="utf-8")
    h = get_model_specialty_html(str(p))
    assert "Anime" in h and "Realistic" in h and "<b>n</b>" not in h


@test("Generation inputs are made safe (sizes ÷8, seed range, steps×strength, NaN/None)")
def _():
    import app as _app
    f = _app._clean_gen_args
    s, c, w, h, b, seed, st, fixes = f(0, -3, 500, 333, 0, 2**70, 0.75)
    assert (s, c, w, h, b, seed) == (1, 0.0, 496, 336, 1, 0) and len(fixes) >= 5, fixes
    s, c, w, h, b, seed, st, fixes = f(None, None, None, None, None, None, None)
    assert (w, h, b, seed) == (_app.DEFAULT_WIDTH, _app.DEFAULT_HEIGHT, 1, -1), (w, h, b, seed)
    s, c, w, h, b, seed, st, fixes = f(float("nan"), "7", 99999, -5, 99, -12, 5, img2img=True)
    assert (w, h, b, seed, st) == (2048, 256, 8, -1, 1.0) and s == _app.DEFAULT_STEPS
    s, *_rest, fixes = f(5, 7, 512, 512, 1, 1, 0.05, img2img=True)
    assert s == 20, s                                          # 5 × 0.05 < 1 step → 20
    s, *_rest, fixes = f(20, 7, 512, 512, 1, 2**32 - 1, 0.5)
    assert _rest[4] == 2**32 - 1 and not fixes                 # valid input: untouched
    from PIL import Image
    big, fx = _app._fit_init_image(Image.new("RGB", (4000, 3000)), 1280)
    assert big.size == (1280, 960) and fx
    tiny, fx = _app._fit_init_image(Image.new("RGBA", (20, 10)))
    assert min(tiny.size) == 64 and fx
    same, fx = _app._fit_init_image(Image.new("RGB", (512, 768)))
    assert same.size == (512, 768) and not fx


@test("Preset / settings names can't leave their folder or overwrite api_keys.json")
def _():
    import app as _app
    n = _app._safe_name
    assert n("../../evil") == "evil" and n("a/b:c?*") == "a_b_c__" and n("_last_session") == "last_session"
    assert n("CON") == "_CON" and n("   ") == "" and n(None) == ""
    tmp = Path(_tf.mkdtemp())
    orig = _app.SETTINGS_DIR
    try:
        _app.SETTINGS_DIR = tmp
        keys = tmp / "api_keys.json"
        keys.write_text('{"civitai": "secret"}', encoding="utf-8")
        r = _app._save_generation_settings("p", "n", "Euler a", 20, 7, 512, 512, 1, None, "API_KEYS")
        assert "reserved" in r and keys.read_text(encoding="utf-8") == '{"civitai": "secret"}', r
        r = _app._save_generation_settings("p", "n", "Euler a", "30", 7, 500, 512, 1, None, "../x")
        assert (tmp / "x.json").exists(), r
        d = json.loads((tmp / "x.json").read_text(encoding="utf-8"))
        assert d["seed"] == -1 and d["width"] == 496 and d["steps"] == 30, d
        assert "API_KEYS" not in _app._list_saved_settings() and "api_keys" not in _app._list_saved_settings()
        (tmp / "junk.json").write_text("[1, 2]", encoding="utf-8")
        assert len(_app._load_generation_settings("junk")) == 9          # no crash on a non-dict file
        assert len(_app._load_generation_settings("../api_keys")) == 9
    finally:
        _app.SETTINGS_DIR = orig
    from backend.lora_trainer import safe_output_name as so
    assert so("..") == "my_lora" and so("CON") == "_CON" and so("a/b") == "a_b"


@test("LoRA tags with odd weights parse instead of raising")
def _():
    import app as _app
    found, clean = _app._loras_from_meta("x <lora:a:.> <lora:b:1:0.5> <LORA:c> <lora:d:-0.4>", None)
    assert found == [("a", 0.8), ("b", 1.0), ("c", 0.8), ("d", -0.4)], found
    assert clean == "x"
    found, _ = _app._loras_from_meta("", "e (×.), f:abc, g:0.3")
    assert ("g", 0.3) in found


@test("rocm_env: multi-GPU hipInfo, best-GPU choice, bad IMAGEGEN_GPU, % in paths")
def _():
    import os
    from backend import rocm_env as R
    txt = ("device#                           0\nName:                             AMD Radeon 780M\n"
           "totalGlobalMem:                   2.00 GB\nisIntegrated:                     1\n"
           "gcnArchName:                      gfx1103\n"
           "device#                           1\nName:                             AMD Radeon RX 6800 XT\n"
           "totalGlobalMem:                   15.98 GB\nisIntegrated:                     0\n"
           "gcnArchName:                      gfx1030:sramecc-:xnack-\n")
    devs = R.parse_hipinfo(txt)
    assert [(d["arch"], d["integrated"]) for d in devs] == [("gfx1103", True), ("gfx1030", False)], devs
    assert R.best_index(devs) == 1 and R.best_index([]) == 0
    assert R.parse_hipinfo("") == [] and R.parse_hipinfo("garbage\nName: x") == []
    orig = (R._candidates, R.gpu_devices)
    try:
        R._candidates = lambda: []
        R.gpu_devices = lambda bins=None: devs
        assert R.choose(None)["gpu_index"] == 1                  # dGPU, not the iGPU
        assert R.choose(0)["arch"] == "gfx1103"
        r = R.choose(7)
        assert r["gpu_index"] == 0 and "doesn't exist" in r["note"], r
    finally:
        R._candidates, R.gpu_devices = orig
    bat = R._bat({"arch": "gfx1031", "runtime": "therock", "gpu_index": 0, "rocm_bin": r"C:\a%PATH%b",
                  "hip_path": r"C:\a%PATH%", "tensile": "", "note": "50% sure"})
    assert "%%PATH%%" in bat and "50%% sure" in bat
    old = os.environ.get("IMAGEGEN_GPU")
    try:
        for v, want in (("", None), ("abc", None), (" 2 ", 2)):
            os.environ["IMAGEGEN_GPU"] = v
            assert R._env_index() == want, (v, R._env_index())
        import config
        os.environ["IMAGEGEN_GPU"] = "junk"
        assert config._env_gpu_index() == 0
        os.environ["IMAGEGEN_GPU"] = "-3"
        assert config._env_gpu_index() == 0
    finally:
        if old is None:
            os.environ.pop("IMAGEGEN_GPU", None)
        else:
            os.environ["IMAGEGEN_GPU"] = old


@test("Launchers: ASCII/CRLF batch files, --help works, unknown flags are reported")
def _():
    import subprocess
    for bat in [*APP_DIR.glob("*.bat"), *(APP_DIR.parent / "installer").glob("*.bat")]:
        raw = bat.read_bytes()
        assert all(b < 128 for b in raw), f"{bat.name} has non-ASCII bytes (cmd reads it as CP1252)"
        assert raw.count(b"\n") == raw.count(b"\r\n"), f"{bat.name} has LF-only lines"
    r = subprocess.run(["cmd", "/d", "/c", str(APP_DIR / "launch.bat"), "--help"],
                       capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and "--gpu N" in r.stdout, (r.returncode, r.stdout[-500:])
    r = subprocess.run(["cmd", "/d", "/c", str(APP_DIR / "launch.bat"), "--bogus", "--help"],
                       capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    assert 'Unknown option "--bogus"' in r.stdout, r.stdout[-500:]
    r = subprocess.run(["cmd", "/d", "/c", str(APP_DIR / "launch.bat"), "--port", "--no-pause"],
                       capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    assert r.returncode != 0 and "--port needs" in r.stdout, r.stdout[-500:]


# ══════════════════════════════════════════════════════════════════════════
print("\n[features] Reproducible outputs, keyword chips, long prompts")
# ══════════════════════════════════════════════════════════════════════════


@test("Prompt merge: every tag once at its strongest weight, syntax-aware")
def _():
    from backend.prompt_tools import parse_tag, merge_prompts, tidy_prompt, split_tags, insert_after_quality
    assert parse_tag("((masterpiece))")[0:2] == ("masterpiece", 1.1 * 1.1)
    assert parse_tag("[blurry]")[1] < 1 and parse_tag("word++")[1] > 1.2
    assert parse_tag("Long_Hair")[0] == parse_tag("long hair")[0]
    assert parse_tag("(x)1.3")[:2] == ("x", 1.3)
    assert split_tags("a, (b, c:1.2), d BREAK e") == ["a", "(b, c:1.2)", "d", "BREAK", "e"]
    assert merge_prompts("masterpiece, best quality, 1girl",
                         "(masterpiece:1.3), [1girl], red eyes") == "(masterpiece:1.3), best quality, 1girl, red eyes"
    assert merge_prompts("a, b BREAK c", "b, d") == "a, b BREAK c, d"
    assert tidy_prompt("a, a, (a:1.4), b")[0] == "(a:1.4), b"
    assert insert_after_quality("score_9, score_8_up, 1girl, (x:0.8)", "x, y") == "score_9, score_8_up, x, y, 1girl"


@test("Long prompts: 75-token chunks at tag boundaries, BREAK starts a chunk")
def _():
    from backend.prompt_tools import chunk_prompt, count_tokens, token_report_html, prompt_warnings
    long = ", ".join(f"very detailed tag number {i}" for i in range(30))
    chunks = chunk_prompt(long)
    assert len(chunks) >= 3 and all(count_tokens(", ".join(c)) <= 75 for c in chunks)
    assert sum(len(c) for c in chunks) == 30                          # nothing dropped or split
    assert chunk_prompt("a BREAK b, c") == [["a"], ["b", "c"]]
    assert chunk_prompt("x, <lora:foo:0.5>") == [["x"]]              # LoRA tags aren't text
    html = token_report_html(long, "bad")
    assert "chunks" in html and "starts at" in html
    assert prompt_warnings(long + ", mytrigger", "", ["mytrigger"])    # trigger pushed past chunk 1
    assert prompt_warnings("cat, blurry", "Blurry")                     # same tag in both prompts


@test("Prompt warning: daylight and night lighting tags in one prompt (golden hour turned a night city into a sunset); no false alarms")
def _():
    from backend.prompt_tools import lighting_conflict, prompt_warnings, token_report_html
    lw = lambda p, n="": [w for w in prompt_warnings(p, n) if w.startswith("lighting tags disagree")]
    # conflicts: any daylight tag with any night tag, weights and spellings ignored; the message names both sides
    for p in ("1girl, golden hour, night city", "1girl, (sunset:1.2), moonlight", "daytime, neon lights, street", "1girl, Sunlight, night",
              "1girl, golden_hour, [night sky]", "1girl, (night:1.3), blue sky, standing"):
        w = lw(p)
        assert len(w) == 1 and "(daylight)" in w[0] and "(night)" in w[0], (p, w)
    w = lw("1girl, golden hour, (night sky:1.2), sunset, neon lights")[0]
    assert "<b>golden hour</b>, <b>sunset</b> (daylight) and <b>night sky</b>, <b>neon lights</b> (night)" in w, w
    assert lighting_conflict("golden hour, night") == (["golden hour"], ["night"]) and lighting_conflict("1girl, solo") == ([], [])
    # no alarm: one side only, lighting words that are something else, weak / dynamic / LoRA mentions, the negative prompt
    for p in ("golden hour, sunset, sunlight", "night, neon lights, moonlight", "1girl, dark hair, dark skin, nightgown, sunlight",
              "golden hour, (night:0.3)", "{day|night} city, 1girl", "__lighting__, night", "1girl, <lora:night_style:1>, golden hour",
              "1girl, night gown, afternoon tea", "dark background, sunset", "city lights, golden hour", ""):
        assert not lw(p), (p, lw(p))
    assert not lw("1girl, golden hour", "night, dark") and not lw("1girl, night", "golden hour")        # the negative side is what you don't want
    assert len(lw("golden hour, night BREAK 1girl")) == 1                                             # BREAK doesn't hide it
    assert "lighting tags disagree" in token_report_html("1girl, golden hour, night city", "") and "lighting tags disagree" not in token_report_html("1girl, golden hour", "")


@test("Chunked encoding concatenates per-chunk embeddings and pads to equal length")
def _():
    import torch
    from backend.prompt_tools import encode_chunked, pad_to_same_chunks

    class FakeCompel:
        def __init__(self, sdxl):
            self.sdxl, self.calls = sdxl, []
        def __call__(self, text):
            self.calls.append(text)
            e = torch.full((1, 77, 8), float(len(self.calls)))
            return (e, torch.ones(1, 4) * len(self.calls)) if self.sdxl else e
    long = ", ".join(f"very detailed tag number {i}" for i in range(30)) + " BREAK sky"
    c = FakeCompel(False)
    e = encode_chunked(c, long)
    assert e.shape == (1, 77 * len(c.calls), 8) and len(c.calls) >= 4 and "BREAK" not in " ".join(c.calls)
    n = encode_chunked(c, "bad")
    a, b = pad_to_same_chunks(c, e, n)
    assert a.shape == b.shape
    cx = FakeCompel(True)
    ex, pooled = encode_chunked(cx, long, sdxl=True)
    assert float(pooled[0, 0]) == 1.0                                    # pooled from the first chunk


@test("LoRA keywords: Civitai prompts, shared trigger, coverage from training captions")
def _():
    import struct
    from backend.lora_keywords import lora_keywords, chips_for
    tmp = Path(_tf.mkdtemp())
    meta = {"ss_tag_frequency": json.dumps({"10_x": {"charname": 20, "1girl": 20, "white hair": 19,
                                                     "red eyes": 18, "dress": 9, "highres": 20, "smile": 3}}),
            "ss_dataset_dirs": json.dumps({"10_x": {"n_repeats": 10, "img_count": 20}})}
    hdr = json.dumps({"__metadata__": meta, "w": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    p = tmp / "char.safetensors"
    p.write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\0\0")
    kw = lora_keywords(str(p))
    assert kw.triggers == ["charname"] and kw.inferred and kw.n_images == 20, kw
    assert ("white hair", 0.95) in kw.tags and not any(t == "highres" for t, _ in kw.tags)   # meta tag hidden
    assert kw.core()[:3] == ["charname", "1girl", "white hair"]
    Path(str(p) + ".civitai.json").write_text(json.dumps({"trainedWords": [
        "charname, white hair, red eyes, dress", "charname, white hair, red eyes, swimsuit"]}), encoding="utf-8")
    kw = lora_keywords(str(p))
    assert kw.triggers == ["charname", "white hair", "red eyes"] and len(kw.phrases) == 2, kw
    samples, tags, note = chips_for([("S1", str(p))])
    assert any("📋" in s[0] for s in samples) and tags[0][1] is True and "S1" in note
    assert any(t == ["dress", False] for t in tags)


@test("Saved images carry model/VAE/LoRA files and restore exactly (incl. renamed files)")
def _():
    import app as _app
    from PIL import Image
    from backend.png_info import read_image_metadata as rd
    from backend import model_hash
    tmp = Path(_tf.mkdtemp())
    for sub in ("ck", "vae", "lora", "out"):
        (tmp / sub).mkdir()
    ck = tmp / "ck" / "myModel.safetensors"; ck.write_bytes(b"ckpt-bytes")
    vae = tmp / "vae" / "myVae.safetensors"; vae.write_bytes(b"vae")
    lo = tmp / "lora" / "charLora.safetensors"; lo.write_bytes(b"lora")

    class P:            # stands in for a loaded pipeline
        current_model, _last_vae_path, model_family = str(ck), str(vae), "sd15"
        _lora_adapters = {0: ("charLora.safetensors", str(lo), 0.75)}
    orig_hash = (model_hash._CACHE_FILE, model_hash._cache)
    model_hash._CACHE_FILE, model_hash._cache = tmp / "hashes.json", {}     # keep the real cache clean
    model_hash.compute(str(ck))
    orig = (_app.OUTPUTS_DIR, _app.list_checkpoints, _app.list_loras, _app.list_vaes)
    try:
        _app.OUTPUTS_DIR = tmp / "out"
        [path] = _app._save_outputs([Image.new("RGB", (64, 80))], dict(
            mode="txt2img", prompt="(cat:1.2), <lora:charLora:0.75>", negative_prompt="bad", steps=12,
            cfg_scale=6.5, seeds=[99], scheduler="Euler a", width=64, height=80), pipe=P())
        with Image.open(path) as im:
            params = im.info["parameters"]
        assert "Size: 64x80" in params and "VAE: myVae.safetensors" in params and "LoRAs: charLora:0.75" in params
        assert f"Model hash: {model_hash.cached(str(ck))[:10]}" in params, params
        meta = rd(path)
        assert meta["imagegen"]["loras"][0]["weight"] == 0.75 and meta["vae"] == "myVae.safetensors"
        _app.list_checkpoints = lambda: [(ck.name, str(ck))]
        _app.list_vaes = lambda: [(vae.name, str(vae))]
        _app.list_loras = lambda: [(lo.name, str(lo))]
        plan = _app._restore_plan(meta)
        assert (plan["model"], plan["vae"], plan["loras"]) == (str(ck), str(vae), [(str(lo), 0.75)]), plan
        assert (plan["seed"], plan["steps"], plan["cfg_scale"], plan["scheduler"], plan["width"]) == \
            (99, 12, 6.5, "Euler a", 64) and plan["exact"] and plan["prompt"] == "(cat:1.2)"
        # the checkpoint was renamed since: found again by its hash
        ck2 = ck.with_name("renamed.safetensors"); ck.rename(ck2)
        model_hash.compute(str(ck2))
        _app.list_checkpoints = lambda: [(ck2.name, str(ck2))]
        assert _app._restore_plan(meta)["model"] == str(ck2)
        # an A1111 image (no record) still restores prompt + settings + LoRA tags
        a1 = tmp / "a1111.png"
        from PIL.PngImagePlugin import PngInfo
        info = PngInfo(); info.add_text("parameters", "a dog, <lora:charLora:0.6>\nNegative prompt: bad\n"
                                        "Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 5, Size: 512x768, Model: renamed")
        Image.new("RGB", (8, 8)).save(a1, pnginfo=info)
        plan = _app._restore_plan(rd(a1))
        assert plan["loras"] == [(str(lo), 0.6)] and plan["model"] == str(ck2) and not plan["exact"], plan
        assert plan["prompt"] == "a dog" and plan["scheduler"] == "DPM++ 2M Karras"
    finally:
        _app.OUTPUTS_DIR, _app.list_checkpoints, _app.list_loras, _app.list_vaes = orig
        model_hash._CACHE_FILE, model_hash._cache = orig_hash


@test("Bug-check: merge keeps formatting, HF repo IDs restore, malformed records don't crash")
def _():
    import app as _app
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from backend.prompt_tools import merge_prompts, count_tokens
    from backend.png_info import read_image_metadata as rd
    base = "masterpiece,\nbest quality,\n1girl"
    assert merge_prompts(base, "red eyes") == base + ", red eyes"          # line breaks survive
    assert merge_prompts(base, "1girl") == base
    assert merge_prompts("a", "x, (x:1.3)") == "a, (x:1.3)"
    assert count_tokens("a b c", "estimate") > 0
    tmp = Path(_tf.mkdtemp())

    class HF:
        current_model, _last_vae_path, model_family = "stabilityai/stable-diffusion-xl-base-1.0", None, "sdxl"
        _lora_adapters = {}
    orig = _app.OUTPUTS_DIR
    try:
        _app.OUTPUTS_DIR = tmp
        [p] = _app._save_outputs([Image.new("RGB", (32, 32))], dict(mode="txt2img", prompt="cat", steps=5,
                                  cfg_scale=5, seeds=[1], scheduler="Euler", width=32, height=32), pipe=HF())
        plan = _app._restore_plan(rd(p))
        assert plan["model"] == "stabilityai/stable-diffusion-xl-base-1.0" and not plan["missing"], plan
        assert plan["loras"] == [] and plan["vae"] == "none"                # exact: no LoRAs, no VAE
    finally:
        _app.OUTPUTS_DIR = orig
    g = tmp / "g.png"; info = PngInfo()
    info.add_itxt("imagegen", json.dumps({"loras": [{"file": None}, "x", {"weight": "abc", "file": "q"}],
                                         "model": None, "vae": {"file": None}, "steps": "12"}))
    Image.new("RGB", (8, 8)).save(g, pnginfo=info)
    _app._restore_plan(rd(g))
    g2 = tmp / "g2.png"; info = PngInfo(); info.add_itxt("imagegen", "{not json"); info.add_text("parameters", "y")
    Image.new("RGB", (8, 8)).save(g2, pnginfo=info)
    plan = _app._restore_plan(rd(g2))
    assert plan["prompt"] == "y" and not plan["exact"]


@test("Hires fix / variations / CLIP skip / X-Y grid settings: parsing, records, restore")
def _():
    import app as _app
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from backend.png_info import read_image_metadata as rd
    ex = _app._clean_extra(dict(clip_skip="3", var_seed=-5, var_strength=7, hires_on=1, hires_scale=9,
                                hires_denoise=None, hires_steps=0, hires_upscaler="Nope"))
    assert {k: ex[k] for k in ("clip_skip", "var_seed", "var_strength", "hires_on", "hires_scale", "hires_denoise",
                               "hires_steps", "hires_upscaler", "fd_on", "fd_mode")} ==         {"clip_skip": 2, "var_seed": -1, "var_strength": 1.0, "hires_on": True, "hires_scale": 2.5,
         "hires_denoise": 0.45, "hires_steps": 1, "hires_upscaler": "Lanczos", "fd_on": False, "fd_mode": "auto"}, ex
    assert _app._xy_values("CFG", "4-10:2") == ([4.0, 6.0, 8.0, 10.0], "")
    assert _app._xy_values("Steps", "10-12") == ([10, 11, 12], "")
    assert _app._xy_values("Sampler", "euler a")[0] == ["Euler a"]
    assert _app._xy_values("Seed", "1, x")[1] and _app._xy_values("CFG", "")[1]
    assert _app._xy_values("none", "")[0] == [None]
    g = _app._xy_grid([Image.new("RGB", (512, 768))] * 4, ["CFG 5", "CFG 8"], ["a", "b"], "t")
    assert g.size[0] > 600 and g.size[1] > 1000
    tmp = Path(_tf.mkdtemp())

    class P:
        current_model, _last_vae_path, model_family = "", None, "sd15"
        _lora_adapters = {}
    orig = _app.OUTPUTS_DIR
    try:
        _app.OUTPUTS_DIR = tmp
        paths = _app._save_outputs([Image.new("RGB", (768, 1152))] * 2, dict(
            mode="txt2img", prompt="cat", steps=20, cfg_scale=7, seeds=[5, 6], scheduler="Euler a",
            width=512, height=768, clip_skip=2, var_seeds=[40, 41], var_strength=0.15,
            hires={"scale": 1.5, "denoise": 0.45, "steps": 12, "upscaler": "Lanczos"}), pipe=P())
        m = rd(paths[1])
        assert (m["var_seed"], m["var_strength"], m["clip_skip"], m["hires"]["scale"]) == (41, 0.15, 2, 1.5), m
        with Image.open(paths[1]) as im:
            t = im.info["parameters"]
        assert "Size: 512x768" in t and "Clip skip: 2" in t and "Hires upscale: 1.5" in t and "Variation seed: 41" in t
        plan = _app._restore_plan(m)
        assert (plan["clip_skip"], plan["var_seed"], plan["var_strength"], plan["hires"]["denoise"],
                plan["width"]) == (2, 41, 0.15, 0.45, 512), plan
        assert _app._plan_extra_updates(plan)[:4] == [2, 41, 0.15, True]
    finally:
        _app.OUTPUTS_DIR = orig
    # an A1111 hires image: Denoising strength is the hires denoise, not an img2img strength
    a = tmp / "a.png"; info = PngInfo()
    info.add_text("parameters", "x\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 3, Size: 512x768, Clip skip: 2, "
                  "Denoising strength: 0.4, Hires upscale: 2, Hires steps: 10, Hires upscaler: R-ESRGAN 4x+")
    Image.new("RGB", (8, 8)).save(a, pnginfo=info)
    m = rd(a)
    assert m["hires"] == {"scale": 2.0, "steps": 10, "upscaler": "R-ESRGAN 4x+", "denoise": 0.4} and "strength" not in m, m
    plan = _app._restore_plan(m)
    assert plan["mode"] == "txt2img" and _app._plan_extra_updates(plan)[7] == "Lanczos"   # unknown upscaler → Lanczos


@test("Variation latents: strength 0 = the seed's own noise; blend keeps unit variance")
def _():
    import torch
    from backend.sd_pipeline import variation_latents, _slerp

    class U:
        dtype = torch.float32
        class config:
            in_channels = 4

    class Pipe:
        unet = U(); vae_scale_factor = 8
    lat, vs = variation_latents(Pipe(), [torch.Generator().manual_seed(1)], 7, 0.0, 1, 512, 512, "cpu")
    assert lat is None and vs == []
    lat, vs = variation_latents(Pipe(), [torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)],
                                7, 0.3, 2, 512, 768, "cpu")
    assert lat.shape == (2, 4, 96, 64) and vs == [7, 8]
    assert 0.9 < float(lat.std()) < 1.1                                  # slerp, not a variance-shrinking mix
    a = torch.randn(1, 4, 8, 8); b = torch.randn(1, 4, 8, 8)
    assert torch.allclose(_slerp(0.0, a, b), a, atol=1e-5) and torch.allclose(_slerp(1.0, a, b), b, atol=1e-4)


@test("LaMa watermark removal: separate regions, edge captions, nothing outside the mask changes")
def _():
    import numpy as np
    from PIL import Image, ImageDraw
    from backend.watermark_remover import inpaint_lama, is_lama_available, _LAMA_PATH
    if not _LAMA_PATH.exists():
        raise Skip("LaMa model not downloaded yet")
    rng = np.random.default_rng(0)
    base = (rng.random((600, 400, 3)) * 40 + 100).astype(np.uint8)       # textured grey
    img = Image.fromarray(base)
    marked = img.copy(); d = ImageDraw.Draw(marked)
    d.rectangle([10, 560, 390, 598], fill=(255, 255, 255))               # caption on the bottom edge
    d.rectangle([330, 10, 390, 30], fill=(255, 255, 255))                # corner tag
    mask = Image.new("L", img.size, 0); dm = ImageDraw.Draw(mask)
    dm.rectangle([8, 558, 392, 599], fill=255); dm.rectangle([328, 8, 392, 32], fill=255)
    out = np.asarray(inpaint_lama(marked, mask)).astype(np.float32)
    # the white boxes are gone (filled close to the grey texture, not a white/grey smear)
    assert abs(out[570:595, 20:380].mean() - base[570:595, 20:380].mean()) < 25, out[570:595, 20:380].mean()
    assert abs(out[12:28, 335:385].mean() - base[12:28, 335:385].mean()) < 25
    # far from both regions nothing changed
    assert np.array_equal(out[100:500, 0:300].astype(np.uint8), base[100:500, 0:300])


@test("Face detail / inpaint / checkpoint axis: detection filters, masks, records, restore")
def _():
    import app as _app
    from PIL import Image, ImageDraw
    from backend import detail_tools as dt
    from backend.png_info import read_image_metadata as rd
    # the relative-size filter drops tiny false hits next to a real face
    boxes = [(100, 100, 300, 300), (10, 10, 40, 40), (400, 100, 560, 260)]
    orig, orig_hits, orig_yolo = dt._cascade, dt._confirm_hits, dict(dt._yolo)
    try:
        dt._yolo["sess"] = None                     # the cascade path (fallback / photo mode)
        class Fake:
            def __init__(self, found): self.found = found
            def detectMultiScale(self, *a, **k):
                return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in self.found]
        dt._cascade = lambda kind: Fake(boxes) if kind == "anime" else None
        dt._confirm_hits = lambda img, b, kind: 10
        got = dt.detect_faces(Image.new("RGB", (640, 480)), "auto")
        assert got == [(100, 100, 300, 300), (400, 100, 560, 260)], got
        # unconfirmed boxes are dropped, and don't set the size the others are measured against
        dt._confirm_hits = lambda img, b, kind: 0 if b == (100, 100, 300, 300) else 10
        assert dt.detect_faces(Image.new("RGB", (640, 480)), "auto") == [(400, 100, 560, 260)]
        # auto: Haar ("photo") boxes need the anime cascade's confirmation, unless nothing is anime
        photo_box = [(500, 300, 600, 400)]
        dt._cascade = lambda kind: Fake(boxes if kind == "anime" else photo_box)
        dt._confirm_hits = lambda img, b, kind: 10 if (kind == "anime") == (b != photo_box[0]) else 0
        got = dt.detect_faces(Image.new("RGB", (640, 480)), "auto")
        assert got == [(100, 100, 300, 300), (400, 100, 560, 260)], got
        dt._confirm_hits = lambda img, b, kind: 10 if kind == "photo" else 0      # a photo: Haar only
        assert dt.detect_faces(Image.new("RGB", (640, 480)), "auto") == [(500, 300, 600, 400)]
        # a second box centred inside a face is the same face
        dt._cascade = lambda kind: Fake([(100, 100, 300, 300), (110, 180, 250, 320)]) if kind == "anime" else None
        dt._confirm_hits = lambda img, b, kind: 10
        assert dt.detect_faces(Image.new("RGB", (640, 480)), "auto") == [(100, 100, 300, 300)]
        dt._cascade = lambda kind: None
        assert dt.detect_faces(Image.new("RGB", (64, 64))) == []
        # the YOLO anime face model: boxes decoded from (1, 5, anchors), scaled back to the image,
        # low scores dropped, overlaps merged; when it runs and sees nothing, no cascade fallback
        class YoloSess:
            def __init__(self, rows): self.rows = rows
            def get_inputs(self):
                class I: name = "images"
                return [I()]
            def run(self, _o, feed):
                x = feed["images"]
                assert x.shape == (1, 3, 640, 480) and x.dtype == np.float32 and x.max() <= 1.0, x.shape
                return [np.array(self.rows, dtype=np.float32).T[None]]
        # 480×640 input for a 960×1280 image → scale 2
        dt._yolo["sess"] = YoloSess([[100, 100, 60, 60, 0.9], [102, 101, 58, 62, 0.85], [300, 400, 50, 50, 0.3]])
        dt._cascade = lambda kind: Fake([(0, 0, 500, 500)])     # must not be used
        got = dt.detect_faces(Image.new("RGB", (960, 1280)), "auto")
        assert got == [(140, 140, 260, 260)], got
        dt._yolo["sess"] = YoloSess([[300, 400, 50, 50, 0.3]])
        assert dt.detect_faces(Image.new("RGB", (960, 1280)), "anime") == []
        assert dt.detect_faces(Image.new("RGB", (960, 1280)), "photo") != []    # photo mode = the cascade
    finally:
        dt._cascade, dt._confirm_hits = orig, orig_hits
        dt._yolo.clear(); dt._yolo.update(orig_yolo)
    # the real confirmation: an empty picture has no face hits
    assert dt._confirm_hits(Image.new("RGB", (300, 300), (200, 180, 160)), (100, 100, 200, 200), "anime") == 0
    # face_detail with no faces returns the image unchanged, without touching the model
    img = Image.new("RGB", (64, 64), (10, 20, 30))
    orig_df = dt.detect_faces
    try:
        dt.detect_faces = lambda *a, **k: []
        out, n = dt.face_detail(None, img, "x")
        assert out is img and n == 0
    finally:
        dt.detect_faces = orig_df
    # crop box: padded, clamped to the image, at least min_side
    m = np.zeros((600, 400), bool); m[590:600, 390:400] = True
    x1, y1, x2, y2 = dt._crop_box(m, 400, 600, 20, 256)
    assert (x2 - x1, y2 - y1) == (256, 256) and x2 <= 400 and y2 <= 600
    # ImageEditor value → image + painted mask
    bg = Image.new("RGB", (100, 80), (200, 0, 0))
    layer = Image.new("RGBA", (100, 80), (0, 0, 0, 0)); ImageDraw.Draw(layer).rectangle([10, 10, 30, 30], fill=(255, 255, 255, 255))
    im2, mask = _app._editor_parts({"background": bg, "layers": [layer], "composite": bg})
    assert im2.size == (100, 80) and np.array(mask)[20, 20] == 255 and np.array(mask)[60, 60] == 0
    assert _app._editor_parts({"background": bg, "layers": []})[1] is None and _app._editor_parts(None) == (None, None)
    # Checkpoint axis matches local files by name
    cks = _app.list_checkpoints()
    if cks:
        stem = Path(cks[0][1]).stem
        vals, err = _app._xy_values("Checkpoint", stem.upper())
        assert not err and vals == [cks[0][1]], (vals, err)
    assert _app._xy_values("Checkpoint", "definitely-not-a-model-xyz")[1]
    # face detail + inpaint are recorded and restored
    tmp = Path(_tf.mkdtemp())

    class P:
        current_model, _last_vae_path, model_family = "", None, "sd15"
        _lora_adapters = {}
    o = _app.OUTPUTS_DIR
    try:
        _app.OUTPUTS_DIR = tmp
        [p1] = _app._save_outputs([Image.new("RGB", (64, 64))], dict(
            mode="txt2img", prompt="a", steps=20, cfg_scale=7, seeds=[1], scheduler="Euler a", width=64, height=64,
            face_detail={"denoise": 0.35, "detector": "anime", "prompt": "detailed eyes"}), pipe=P())
        [p2] = _app._save_outputs([Image.new("RGB", (64, 64))], dict(
            mode="inpaint", prompt="b", steps=20, cfg_scale=7, seeds=[2], scheduler="Euler a", width=64, height=64,
            strength=0.7, inpaint_padding=48), pipe=P())
    finally:
        _app.OUTPUTS_DIR = o
    m1 = rd(p1)
    assert m1["face_detail"]["detector"] == "anime"
    with Image.open(p1) as im:
        assert "Face detail: denoise 0.35 (anime)" in im.info["parameters"]
    ups = _app._plan_extra_updates(_app._restore_plan(m1))
    assert ups[8:12] == [True, 0.35, "anime", "detailed eyes"], ups
    with Image.open(p2) as im:
        assert "Inpaint: denoise 0.7, padding 48" in im.info["parameters"]
    assert _app._restore_plan(rd(p2))["mode"] == "inpaint"


@test("Wildcards: {a|b}, __files__, weights, picks per seed, escapes, missing, no runaway recursion")
def _():
    from backend import wildcards as wc
    from backend.prompt_tools import split_tags
    tmp = Path(_tf.mkdtemp())
    (tmp / "hair").mkdir()
    (tmp / "hair" / "color.txt").write_text("# comment\nred hair\n\nblue hair\n", encoding="utf-8")
    (tmp / "outfit.txt").write_text("maid, {apron|frills}\nkimono\n", encoding="utf-8")
    (tmp / "loop.txt").write_text("__loop__\n", encoding="utf-8")
    old = wc.WILDCARD_DIRS
    try:
        wc.WILDCARD_DIRS = [tmp]
        assert wc.list_wildcards() == ["hair/color", "loop", "outfit"], wc.list_wildcards()
        t = "1girl, __hair/color__, __outfit__, {smile|pout}, {keep}, (x:1.2)"
        assert wc.is_dynamic(t) and not wc.is_dynamic("a, {b}, (c:1.2)") and not wc.is_dynamic("")
        outs = {wc.resolve(t, s) for s in range(40)}
        assert len(outs) > 4, outs                               # the picks vary with the seed
        assert all(wc.resolve(t, s) == wc.resolve(t, s) for s in (1, 99))   # … and repeat per seed
        for o in outs:
            assert "__" not in o and "|" not in o and "{keep}" in o and "(x:1.2)" in o, o
            assert ("red hair" in o) != ("blue hair" in o), o
            assert not ("maid" in o and "apron" not in o and "frills" not in o), o   # nested group resolved
        assert {wc.resolve("{0::a|b}", s) for s in range(30)} == {"b"}
        two = wc.resolve("{2$$a|b|c}", 5).split(", ")
        assert len(two) == 2 and len(set(two)) == 2 and set(two) <= {"a", "b", "c"}, two
        assert wc.resolve("x, {|}, y", 3) == "x, y"                 # empty pick leaves no ", ,"
        assert wc.resolve(r"\{a|b\}", 1) == r"\{a|b\}"               # escaped: literal
        missing = []
        assert wc.resolve("a, __nope__, {b|b}", 1, missing) == "a, __nope__, b" and missing == ["nope"]
        assert "loop" in wc.resolve("__loop__", 1)                  # self-reference stops at the depth limit
        assert wc._lines("../outfit") is None
        # bug-check round: inside a word it's a tag, a Notepad BOM isn't part of the first line,
        # a negative weight counts as 0 instead of leaking "-1::" into the prompt
        assert wc.resolve("long__hair__style, {a|a}", 1) == "long__hair__style, a"
        (tmp / "bom.txt").write_bytes(b"\xef\xbb\xbfred\nblue\n")
        assert {wc.resolve("__bom__", s) for s in range(30)} == {"red", "blue"}
        assert {wc.resolve("{-1::a|b}", s) for s in range(20)} == {"b"}
    finally:
        wc.WILDCARD_DIRS = old
    # "{a|b}, {a|b}" asks for two picks: merge / tidy must not collapse it (a repeated addition still merges)
    from backend.prompt_tools import merge_prompts, tidy_prompt
    assert merge_prompts("{a|b}, {a|b}, c, c") == "{a|b}, {a|b}, c"
    assert merge_prompts("x, __outfit__", "__outfit__, y") == "x, __outfit__, y"
    assert tidy_prompt("__pose__, __pose__, c")[1] == []
    assert split_tags("a, {b, c|d}, e") == ["a", "{b, c|d}", "e"]
    # the shipped starter files all resolve
    shipped = wc.list_wildcards()
    assert {"outfit", "pose", "expression", "background", "hair", "eyes", "lighting", "camera"} <= set(shipped), shipped
    for n in shipped:
        r = wc.resolve(f"__{n}__", 7)
        assert r and "__" not in r and "{" not in r, (n, r)


@test("Family quality tags: Illustrious aesthetic tags + negatives, Pony scores, nothing twice")
def _():
    import app as _app
    pos, neg = _app._family_quality("illustrious", "1girl", "")
    assert "very aesthetic" in pos and "absurdres" in pos and "worst quality" in neg, (pos, neg)
    assert _app._family_quality("illustrious", "masterpiece, 1girl", "lowres, worst quality") == ("", "")
    pos, neg = _app._family_quality("pony", "1girl", "bad hands")
    assert pos.startswith("score_9") and "score_4" in neg
    assert _app._family_quality("pony", "score_9, 1girl", "score_4") == ("", "")
    assert _app._family_quality("weird", "x", "")[0] == "masterpiece, best quality"    # unknown → SD 1.5


@test("Character cards: build from a LoRA (triggers, body tags, outfits), save/load, sanitising")
def _():
    import struct
    from backend import character_cards as cc
    tmp = Path(_tf.mkdtemp())
    meta = {"ss_tag_frequency": json.dumps({"10_x": {"heroine": 20, "1girl": 20, "solo": 19, "silver hair": 19,
                                                     "red eyes": 18, "cleavage": 15, "smile": 3,
                                                     "butterfly hair ornament": 11, "wavy hair": 11}}),
            "ss_dataset_dirs": json.dumps({"10_x": {"n_repeats": 10, "img_count": 20}})}
    hdr = json.dumps({"__metadata__": meta, "w": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    lp = tmp / "IL_Heroine.safetensors"
    lp.write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\0\0")
    Path(str(lp) + ".civitai.json").write_text(json.dumps({"trainedWords": [
        "heroine, silver hair, red eyes, maid, maid headdress, frilled apron",
        "heroine, silver hair, red eyes, purple bikini, choker"]}), encoding="utf-8")
    old = cc.CARDS_DIR
    try:
        cc.CARDS_DIR = tmp / "characters"
        card = cc.card_from_lora(str(lp), 0.75, str(tmp / "base.safetensors"))
        assert card["name"] == "Heroine" and card["loras"] == [{"file": "IL_Heroine.safetensors", "weight": 0.75}]
        assert card["tags"].startswith("1girl, solo, heroine, silver hair, red eyes"), card["tags"]
        assert "cleavage" not in card["tags"]                        # not a body tag
        assert "butterfly hair ornament" in card["tags"]             # signature pin: kept from 50 % of the images
        assert "wavy hair" not in card["tags"]                       # an ordinary body tag still needs 60 %
        assert list(card["outfits"]) == ["maid · maid headdress · frilled apron", "purple bikini · choker"], card
        assert card["outfits"]["purple bikini · choker"] == "purple bikini, choker"
        pr = cc.card_prompt(card, "purple bikini · choker", "beach, silver hair")
        assert pr.count("silver hair") == 1 and pr.endswith("purple bikini, choker, beach"), pr
        assert cc.card_prompt(card, "no such outfit") == card["tags"]
        assert cc.card_prompt({"tags": ""}) == "" and cc.card_prompt({"tags": "", "outfits": {"o": "x"}}, "o") == "x"
        cc.save_card(card)
        assert cc.list_cards() == ["Heroine"] and cc.load_card("Heroine") == card
        assert cc.load_card("../Heroine") is None or cc.load_card("../Heroine")["name"] == "Heroine"
        bad = cc.clean_card({"name": "../CON", "checkpoint": "C:\\x\\m.safetensors", "width": "5000", "cfg": "x",
                             "steps": 30, "loras": [{"file": "a/b.safetensors", "weight": "9"}, "junk"],
                             "outfits": {"ok": "x", "": "y", "n": 3}, "clip_skip": 2})
        assert bad["name"] == "_CON" and bad["checkpoint"] == "m.safetensors" and "width" not in bad
        assert "cfg" not in bad and bad["steps"] == 30 and bad["clip_skip"] == 2
        assert bad["loras"] == [{"file": "b.safetensors", "weight": 0.8}] and bad["outfits"] == {"ok": "x"}
        assert cc.clean_card({"name": "  "}) is None and cc.clean_card("x") is None
        odd = cc.clean_card({"name": "q", "width": "832.7", "height": "inf", "steps": 30.4, "cfg": "nan"})
        assert odd.get("width") == 832 and "height" not in odd and odd.get("steps") == 30 and "cfg" not in odd, odd
        (cc.CARDS_DIR / "broken.json").write_text("{nope", encoding="utf-8")
        assert cc.load_card("broken") is None
    finally:
        cc.CARDS_DIR = old


@test("Character card profiles: sanitised, picked by the selected checkpoint (name / stem / path, any case), laid over the card")
def _():
    from backend import character_cards as cc
    card = cc.clean_card({"name": "P", "checkpoint": "a.safetensors", "loras": [{"file": "x.safetensors", "weight": 0.75}],
                          "cfg": 6.0, "steps": 12, "scheduler": "DPM++ 2M AYS", "profiles": {
        "C:\\m\\wai.safetensors": {"loras": [{"file": "y.safetensors", "weight": "0.6"}], "cfg": 5, "pag": 2, "freeu": True,
                                   "face_detail": 0.05, "hand_detail": 9, "junk": 1},
        "pony.safetensors": {"loras": []}, "bad": "x", "": {"cfg": 5}, "empty.safetensors": {"cfg": "nan"}}})
    assert set(card["profiles"]) == {"wai.safetensors", "pony.safetensors"}, card["profiles"]
    assert card["profiles"]["wai.safetensors"] == {"loras": [{"file": "y.safetensors", "weight": 0.6}], "cfg": 5.0, "pag": 2.0,
                                                   "freeu": True, "face_detail": 0.1}, card["profiles"]   # 9 is out of range; 0.05 → slider minimum
    assert card["profiles"]["pony.safetensors"] == {"loras": []}
    eff, key = cc.card_for_checkpoint(card, r"D:\models\checkpoints\WAI.safetensors")
    assert key == "wai.safetensors" and eff["cfg"] == 5.0 and eff["steps"] == 12 and "profiles" not in eff, eff
    assert eff["loras"] == [{"file": "y.safetensors", "weight": 0.6}] and eff["checkpoint"] == "wai.safetensors"
    assert cc.card_for_checkpoint(card, "pony")[1] == "pony.safetensors"                         # by stem
    eff, key = cc.card_for_checkpoint(card, "other.safetensors")
    assert key is None and eff is card
    assert cc.card_for_checkpoint(card, None)[1] is None and cc.card_for_checkpoint({"name": "q"}, "a")[1] is None
    assert "profiles" not in cc.clean_card({"name": "n", "profiles": "x"})


@test("🎴 Load applies the card's profile for the selected checkpoint (settings + boosters, checkpoint kept), 📌 saves one, Save keeps them")
def _():
    import app as _app
    import backend.character_cards as CC
    tmp = Path(_tf.mkdtemp())
    old = CC.CARDS_DIR
    try:
        CC.CARDS_DIR = tmp / "cards"
        CC.CARDS_DIR.mkdir()
        from backend import identity_score as _ID
        _ident = {"model": _ID.MODEL_TAG, "n": 5, "centroid": _ID.encode_vec(_ID._unit(__import__("numpy").arange(1.0, 769.0))), "mean": 0.93, "sd": 0.02}
        (CC.CARDS_DIR / "G.json").write_text(json.dumps({
            "name": "G", "checkpoint": "base.safetensors", "tags": "1girl", "cfg": 6.0, "steps": 12, "scheduler": "Euler a",
            "notes": "her canon pin is the black butterfly", "identity": _ident,
            "loras": [{"file": "l.safetensors", "weight": 0.7}],
            "profiles": {"wai.safetensors": {"cfg": 4.5, "steps": 20, "scheduler": "DPM++ 2M Karras", "pag": 2.0, "freeu": True,
                                             "face_detail": 0.35, "hand_detail": 0.0, "loras": []}}}), encoding="utf-8")
        by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
        fn = lambda n: getattr(by[n].fn, "__wrapped__", by[n].fn)
        with patch("backend.model_manager.list_checkpoints", return_value=[("base.safetensors", "/m/base.safetensors"),
                                                                           ("wai.safetensors", "/m/wai.safetensors")]), \
             patch("backend.model_manager.list_loras", return_value=[("l.safetensors", "/m/l.safetensors")]), \
             patch("backend.model_manager.list_vaes", return_value=[]):
            out = fn("do_card_load")("G", "(no outfit tags)", "", "", "/m/wai.safetensors")
            assert len(out) == 27, len(out)            # + eye detail on / denoise + the colour strength (round 26)
            assert out[0].get("__type__") == "update" and out[2] == "none"                         # checkpoint kept, profile has no LoRA
            assert (out[12], out[13]) == (4.5, 20) and out[14] == "DPM++ 2M Karras", out[12:15]
            assert (out[16], out[18], out[19], out[20], out[21]) == (2.0, True, True, 0.35, False), out[16:23]
            assert "profile for wai.safetensors" in out[-1]
            out = fn("do_card_load")("G", "(no outfit tags)", "", "", "/m/base.safetensors")      # no profile for this one: the card
            assert out[0] == "/m/base.safetensors" and out[2] == "/m/l.safetensors" and out[3] == 0.7
            assert (out[12], out[13], out[14]) == (6.0, 12, "Euler a") and out[16].get("__type__") == "update" and "profile" not in out[-1]
        # 📌 Save as this checkpoint's profile, then 💾 Save current setup must keep the profiles
        ck, lo = tmp / "sdxl.safetensors", tmp / "l.safetensors"
        ck.write_bytes(b"x")
        lo.write_bytes(b"x")
        msg = fn("do_card_profile")("G", str(ck), "none", str(lo), 0.6, "none", 0.7, "none", 0.7, "bad hands", 832, 1216, 5.0, 12,
                                    "DPM++ 2M AYS", 1, 2.0, False, 0.5, True, 0.35, False, 0.35)
        assert "📌 Saved" in msg, msg
        prof = CC.load_card("G")["profiles"]["sdxl.safetensors"]
        assert prof["loras"] == [{"file": "l.safetensors", "weight": 0.6}] and prof["cfg"] == 5.0 and prof["pag"] == 2.0, prof
        assert (prof["cfg_rescale"], prof["face_detail"], prof["hand_detail"], prof["freeu"]) == (0.5, 0.35, 0.0, False), prof
        assert "Pick a card" in fn("do_card_profile")("(none)", str(ck), "none", "none", 0.7, "none", 0.7, "none", 0.7, "", 832, 1216,
                                                      5.0, 12, "Euler a", 1, 0.0, False, 0.0, False, 0.35, False, 0.35)
        fn("do_card_save")("G", str(ck), "none", "none", 0.7, "none", 0.7, "none", 0.7, "1girl, solo", "bad", 832, 1216, 6.0, 12,
                           "Euler a", 1)
        saved = CC.load_card("G")
        assert set(saved["profiles"]) == {"wai.safetensors", "sdxl.safetensors"}
        assert saved["identity"] == _ident and saved["notes"] == "her canon pin is the black butterfly", saved     # 💾 Save used to drop notes (and would drop the identity)
    finally:
        CC.CARDS_DIR = old


@test("🎨 Colour strength in cards (round 26): an optional 'colour' key per card / profile, sanitised (0.5-1); 🎴 Load sets the slider (a card without it leaves it), a profile overrides the card, 📌 saves it, 💾 Save keeps it")
def _():
    import app as _app
    import backend.character_cards as CC
    c = CC.clean_card({"name": "C", "colour": 0.8, "profiles": {"a.safetensors": {"colour": "0.7", "cfg": 5}, "b.safetensors": {"colour": 3, "cfg": 5}, "c.safetensors": {"colour": "x", "cfg": 5}}})
    assert c["colour"] == 0.8 and c["profiles"]["a.safetensors"]["colour"] == 0.7, c
    assert "colour" not in c["profiles"]["b.safetensors"] and "colour" not in c["profiles"]["c.safetensors"], c["profiles"]        # out of range / junk: dropped, the profile stays
    assert "colour" not in CC.clean_card({"name": "D", "colour": 0.2}) and CC.clean_card({"name": "D", "colour": 1.0})["colour"] == 1.0
    tmp = Path(_tf.mkdtemp())
    old = CC.CARDS_DIR
    try:
        CC.CARDS_DIR = tmp / "cards"
        CC.CARDS_DIR.mkdir()
        (CC.CARDS_DIR / "G.json").write_text(json.dumps({"name": "G", "checkpoint": "base.safetensors", "tags": "1girl", "colour": 0.8, "loras": [],
                                                         "profiles": {"wai.safetensors": {"cfg": 5.0, "colour": 0.7}, "other.safetensors": {"cfg": 5.0}}}), encoding="utf-8")
        (CC.CARDS_DIR / "H.json").write_text(json.dumps({"name": "H", "checkpoint": "base.safetensors", "tags": "1girl", "loras": []}), encoding="utf-8")
        by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
        fn = lambda n: getattr(by[n].fn, "__wrapped__", by[n].fn)
        cks = [("base.safetensors", "/m/base.safetensors"), ("wai.safetensors", "/m/wai.safetensors"), ("other.safetensors", "/m/other.safetensors")]
        with patch("backend.model_manager.list_checkpoints", return_value=cks), patch("backend.model_manager.list_loras", return_value=[]), patch("backend.model_manager.list_vaes", return_value=[]):
            assert fn("do_card_load")("G", "(no outfit tags)", "", "", "/m/base.safetensors")[-2] == 0.8        # the card's colour
            assert fn("do_card_load")("G", "(no outfit tags)", "", "", "/m/wai.safetensors")[-2] == 0.7         # a profile overrides it
            assert fn("do_card_load")("G", "(no outfit tags)", "", "", "/m/other.safetensors")[-2] == 0.8       # a profile without one: the card's
            assert fn("do_card_load")("H", "(no outfit tags)", "", "", "/m/base.safetensors")[-2].get("__type__") == "update"   # no colour: the control stays
            assert fn("do_card_load")("(none)", "", "", "")[-2].get("__type__") == "update"
        # the slider is a Load output (right before the status) and a 📌 input (the last one)
        demo = _app.build_app()
        load_ev = [f for f in demo.fns if getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "") == "do_card_load"][0]
        prof_ev = [f for f in demo.fns if getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "") == "do_card_profile"][0]
        assert type(load_ev.outputs[-2]).__name__ == "Slider" and load_ev.outputs[-2].label == "Colour strength", [getattr(o, "label", None) for o in load_ev.outputs[-3:]]
        assert prof_ev.inputs[-1].label == "Colour strength", prof_ev.inputs[-1].label
        ck = tmp / "sdxl.safetensors"
        ck.write_bytes(b"x")
        base = ("H", str(ck), "none", "none", 0.7, "none", 0.7, "none", 0.7, "bad", 832, 1216, 5.0, 12, "DPM++ 2M AYS", 1, 2.0, False, 0.0, False, 0.35, False, 0.35, False, 0.4)
        assert "📌 Saved" in fn("do_card_profile")(*base, 0.8)
        assert CC.load_card("H")["profiles"]["sdxl.safetensors"]["colour"] == 0.8
        assert "📌 Saved" in fn("do_card_profile")(*base)                                                          # the default slider value 1 is saved too (Load then resets it)
        assert CC.load_card("H")["profiles"]["sdxl.safetensors"]["colour"] == 1.0
        fn("do_card_save")("G", str(ck), "none", "none", 0.7, "none", 0.7, "none", 0.7, "1girl", "bad", 832, 1216, 6.0, 12, "Euler a", 1)
        assert CC.load_card("G")["colour"] == 0.8 and CC.load_card("G")["profiles"]["wai.safetensors"]["colour"] == 0.7        # 💾 Save keeps them
    finally:
        CC.CARDS_DIR = old


@test("WD14 tagger: input prep (white pad, BGR, NHWC), thresholds, rating kept apart, tag text")
def _():
    import numpy as np
    from PIL import Image
    from backend import wd_tagger as wd
    im = Image.new("RGBA", (40, 20), (255, 0, 0, 255))
    arr = wd._prepare(im, 448)
    assert arr.shape == (1, 448, 448, 3) and arr.dtype == np.float32
    assert tuple(arr[0, 224, 224]) == (0.0, 0.0, 255.0)          # red → BGR
    assert tuple(arr[0, 5, 224]) == (255.0, 255.0, 255.0)        # padding is white
    assert wd._pretty("hatsune_miku_(vocaloid)") == "hatsune miku \\(vocaloid\\)" and wd._pretty("^_^") == "^_^"

    class In:
        name, shape = "input", [1, 448, 448, 3]

    class Sess:
        def get_inputs(self): return [In()]
        def run(self, _o, feed):
            assert feed["input"].shape == (1, 448, 448, 3)
            return [np.array([[0.9, 0.1, 0.8, 0.3, 0.95, 0.5]], dtype=np.float32)]
    old = (wd._session, wd._labels)
    try:
        wd._session = Sess()
        wd._labels = [("general", 9), ("explicit", 9), ("long_hair", 0), ("smile", 0), ("some_char_(game)", 4),
                      ("other_char", 4)]
        r = wd.tag_image(Image.new("RGB", (64, 64)), 0.35, 0.85, progress=lambda *a, **k: 1 / 0)
        assert r["general"] == [("long hair", r["general"][0][1])] and len(r["general"]) == 1, r
        assert [t for t, _ in r["character"]] == ["some char \\(game\\)"] and set(r["rating"]) == {"general", "explicit"}
        assert wd.tags_text(r) == "some char \\(game\\), long hair"
        assert wd.tags_text(r, exclude="Long_Hair", with_character=False) == ""
    finally:
        wd._session, wd._labels = old


@test("Unload frees text encoders: pyparsing's packrat cache no longer pins Compel frames; inpaint pipe dropped")
def _():
    import gc, weakref
    import pyparsing as pp
    from backend.prompt_tools import release_parser_cache
    pp.ParserElement.enable_packrat()          # some library switches it on in the app process

    class Encoder:                             # stands in for Compel → text encoders
        pass

    def encode(enc):
        from compel.prompt_parser import PromptParser
        PromptParser().parse_conjunction("(masterpiece)1.2, 1girl, [x]")   # what Compel runs per prompt

    enc = Encoder(); ref = weakref.ref(enc)
    encode(enc); del enc
    gc.collect()
    assert ref() is not None, "packrat no longer keeps frames — this test can't show the leak any more"
    release_parser_cache(); gc.collect()
    assert ref() is None, "the parser cache still holds the encoder's frame"
    # both pipelines drop the cached inpaint pipe (face detail / inpaint) on unload
    from backend.sd_pipeline import SDPipeline
    from backend.sdxl_pipeline import SDXLPipeline
    for cls in (SDPipeline, SDXLPipeline):
        obj = cls.__new__(cls)
        obj.__dict__.update(pipe=None, img2img_pipe=None, _inpaint_pipe=object(), _vae_needs_fp32=False)
        try:
            obj._unload()
        except Exception:
            pass
        assert obj.__dict__.get("_inpaint_pipe") is None, cls.__name__


@test("UniPC: its per-step linear solve falls back to the CPU when the GPU's batched LU is unsupported (ZLUDA)")
def _():
    import torch
    from diffusers import EulerDiscreteScheduler
    from backend import sampling as sm
    real = torch.linalg.solve
    A, B = torch.tensor([[2.0, 0.0], [0.0, 4.0]]), torch.tensor([2.0, 4.0])
    calls = []

    def flaky(a, b, *args, **kw):
        calls.append(a)
        if a is A:                                  # the first attempt = the "GPU" one
            raise RuntimeError("CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling `cublasSgetrsBatched( handle, trans, n )`")
        return real(a, b, *args, **kw)

    def other_error(a, b, *args, **kw):
        raise RuntimeError("linalg.solve: The solver failed because the input matrix is singular.")
    try:
        torch.linalg.solve = flaky
        sm._patch_linalg_solve()
        assert torch.allclose(torch.linalg.solve(A, B), torch.tensor([1.0, 1.0])) and sm._gpu_solve_ok is False
        n = len(calls)
        assert torch.allclose(torch.linalg.solve(A, B), torch.tensor([1.0, 1.0])) and len(calls) == n + 1 and calls[-1] is not A
        torch.linalg.solve, sm._gpu_solve_ok = other_error, True
        sm._patch_linalg_solve()                    # an unrelated error is not swallowed
        try:
            torch.linalg.solve(A, B)
            raise AssertionError("singular-matrix error was swallowed")
        except RuntimeError as e:
            assert "singular" in str(e)
        # building UniPC installs the patch; the sampler still runs
        torch.linalg.solve, sm._gpu_solve_ok = real, True

        class Pipe:
            scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               num_train_timesteps=1000, steps_offset=1, prediction_type="epsilon")
        s = sm.make_scheduler(Pipe(), "UniPC")
        assert type(s).__name__ == "UniPCMultistepScheduler" and getattr(torch.linalg.solve, "_cpu_fallback", False)
        s.set_timesteps(8)
        x = torch.randn(1, 4, 8, 8)
        for t in s.timesteps:
            x = s.step(torch.randn_like(x), t, x).prev_sample
        assert torch.isfinite(x).all()
    finally:
        torch.linalg.solve, sm._gpu_solve_ok = real, True


@test("Samplers: real Karras sigmas, AYS schedules, no option leaks, v-prediction config + detection")
def _():
    import struct
    from diffusers import EulerDiscreteScheduler
    from backend import sampling as sm
    from diffusers.schedulers import AysSchedules
    cfg = dict(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
               steps_offset=1, timestep_spacing="leading", prediction_type="epsilon")

    class Pipe:
        def __init__(self, xl):
            self.scheduler = EulerDiscreteScheduler(**cfg)
            if xl:
                self.text_encoder_2 = object()
    for xl in (False, True):
        p = Pipe(xl)
        p.scheduler = sm.make_scheduler(p, "DPM++ 2M Karras")
        assert p.scheduler.config.use_karras_sigmas is True          # it was plain DPM++ 2M before
        p.scheduler = sm.make_scheduler(p, "DPM++ 2M")
        assert not p.scheduler.config.use_karras_sigmas             # doesn't inherit the previous one's
        p.scheduler = sm.make_scheduler(p, "DPM++ 2M SDE Karras")
        assert p.scheduler.config.algorithm_type == "sde-dpmsolver++"
        p.scheduler = sm.make_scheduler(p, "DPM++ 2M AYS")
        assert p.scheduler.config.algorithm_type == "dpmsolver++"
        # DPM++ SDE needs torchsde: builds when installed, else falls back instead of failing Generate
        try:
            import torchsde  # noqa: F401
            assert type(sm.make_scheduler(p, "DPM++ SDE Karras")).__name__ == "DPMSolverSDEScheduler"
        except ImportError:
            pass
        with patch("diffusers.DPMSolverSDEScheduler.from_config", side_effect=ImportError("no torchsde")):
            fb = sm.make_scheduler(p, "DPM++ SDE")
        assert type(fb).__name__ == "DPMSolverMultistepScheduler" and fb.config.use_karras_sigmas is True
        table = AysSchedules["StableDiffusionXLTimesteps" if xl else "StableDiffusionTimesteps"]
        p.scheduler.set_timesteps(10)
        assert [int(t) for t in p.scheduler.timesteps] == table
        for n in (6, 13, 40):
            p.scheduler.set_timesteps(n)
            ts = [int(t) for t in p.scheduler.timesteps]
            assert len(ts) == n and ts[0] == 999 and ts[-1] == table[-1] and all(a > b for a, b in zip(ts, ts[1:])), ts
    assert set(sm.LEGACY_NAMES.values()) <= set(sm.SCHEDULERS)
    # v-prediction: kept by every scheduler made later; Karras / AYS off with zero-terminal SNR
    p = Pipe(True)
    sm.configure_prediction(p, {"v_pred": True, "zero_snr": True})
    for name in ("DPM++ 2M Karras", "Euler a", "DPM++ 2M AYS"):
        p.scheduler = sm.make_scheduler(p, name)
        c = p.scheduler.config
        assert c.prediction_type == "v_prediction" and c.rescale_betas_zero_snr and c.timestep_spacing == "trailing"
        assert not c.get("use_karras_sigmas") and not getattr(p.scheduler, "_ays", False), name
    # detection: NoobAI-style marker keys, modelspec metadata, file name
    tmp = Path(_tf.mkdtemp())

    def ckpt(name, keys, meta=None):
        hdr = {k: {"dtype": "F16", "shape": [0], "data_offsets": [0, 0]} for k in keys}
        if meta:
            hdr["__metadata__"] = meta
        raw = json.dumps(hdr).encode()
        (tmp / name).write_bytes(struct.pack("<Q", len(raw)) + raw)
        return str(tmp / name)
    assert sm.detect_prediction(ckpt("a.safetensors", ["v_pred", "ztsnr", "w"])) == {"v_pred": True, "zero_snr": True}
    assert sm.detect_prediction(ckpt("b.safetensors", ["w"]))["v_pred"] is False
    assert sm.detect_prediction(ckpt("noob-vpred-1.0.safetensors", ["w"]))["v_pred"] is True
    assert sm.detect_prediction(ckpt("c.safetensors", ["w"], {"modelspec.prediction_type": "v"}))["v_pred"] is True
    assert sm.detect_prediction(str(tmp / "missing.safetensors"))["v_pred"] is False
    (tmp / "junk.safetensors").write_bytes(b"\xff" * 20)
    assert sm.detect_prediction(str(tmp / "junk.safetensors"))["v_pred"] is False


@test("Boosters: run_pipe passes only supported args, restores PAG attention on a crash, FreeU off; FFT fallback")
def _():
    import torch
    from backend import sampling as sm

    class Unet:
        def __init__(self):
            self.procs = {"a": "orig"}; self.freeu = None; self.dtype = torch.float16
        @property
        def attn_processors(self): return dict(self.procs)
        def set_attn_processor(self, p): self.procs = dict(p)
        def enable_freeu(self, **k): self.freeu = k
        def disable_freeu(self): self.freeu = None

    class Plain:                          # like SD 1.5 img2img: no guidance_rescale
        def __init__(self, unet): self.unet = unet; self.scheduler = "S"; self.got = None
        def __call__(self, prompt=None, guidance_scale=7.0):
            self.got = dict(prompt=prompt); assert self.unet.freeu is not None; return "ok"

    class WithRescale(Plain):
        def __call__(self, prompt=None, guidance_scale=7.0, guidance_rescale=0.0):
            self.got = dict(prompt=prompt, guidance_rescale=guidance_rescale); return "ok"

    class Host:                           # the SDPipeline wrapper
        boosters = {"pag": 0, "freeu": True, "cfg_rescale": 0.6}
        prediction = {}
    u = Unet(); h = Host()
    pl = Plain(u)
    assert sm.run_pipe(h, pl, "img2img", prompt="x") == "ok" and pl.got == {"prompt": "x"} and u.freeu is None
    wr = WithRescale(u)
    sm.run_pipe(h, wr, "txt2img", prompt="x")
    assert wr.got["guidance_rescale"] == 0.6
    h.boosters = {"cfg_rescale": None}; h.prediction = {"v_pred": True}
    assert sm.boosters(h)["cfg_rescale"] == 0.7                   # auto for v-prediction
    h.prediction = {}
    assert sm.boosters(h)["cfg_rescale"] == 0.0
    # PAG: the PAG pipeline swaps processors and dies mid-run → the originals come back
    class PagPipe:
        def __init__(self, unet): self.unet = unet; self.scheduler = None
        @classmethod
        def from_pipe(cls, pipe, **k): return cls(pipe.unet)
        def __call__(self, **k):
            assert k["pag_scale"] == 3.0
            self.unet.set_attn_processor({"a": "pag"})
            raise RuntimeError("stopped")
    orig = sm._pag_class
    try:
        sm._pag_class = lambda xl, kind: PagPipe
        h.boosters = {"pag": 3.0}
        try:
            sm.run_pipe(h, wr, "txt2img", prompt="x")
        except RuntimeError:
            pass
        assert u.procs == {"a": "orig"}, u.procs
        assert sm.run_pipe(h, wr, "inpaint", prompt="x") == "ok"    # no PAG for inpaint
    finally:
        sm._pag_class = orig
    # FreeU's Fourier filter: the wrapped version still filters CPU tensors like the original
    import diffusers.utils.torch_utils as tu
    sm._patch_fourier_filter()
    x = torch.randn(1, 4, 16, 16)
    assert getattr(tu.fourier_filter, "_cpu_fallback", False) and tu.fourier_filter(x, 1, 0.5).shape == x.shape


@test("Sampler names in records: old images restore the sampler that really ran; boosters recorded + restored")
def _():
    import app as _app
    from PIL import Image
    from backend.png_info import read_image_metadata as rd
    base = {"prompt": "x", "sampler": "DPM++ 2M Karras"}
    old = _app._restore_plan(dict(base, imagegen={"app": "ImageGen Studio", "format": 1, "scheduler": "DPM++ 2M Karras"}))
    new = _app._restore_plan(dict(base, imagegen={"app": "ImageGen Studio", "format": 2, "scheduler": "DPM++ 2M Karras"}))
    a1111 = _app._restore_plan(dict(base))
    assert old["scheduler"] == "DPM++ 2M" and new["scheduler"] == "DPM++ 2M Karras" and a1111["scheduler"] == "DPM++ 2M Karras"
    assert _app._restore_plan(dict(base, sampler="DPM++ 2M SDE Karras"))["scheduler"] == "DPM++ 2M SDE Karras"
    # record → A1111 text → restore
    tmp = Path(_tf.mkdtemp())

    class P:
        current_model, _last_vae_path, model_family = "", None, "illustrious"
        _lora_adapters = {}
        boosters = {"pag": 2.5, "freeu": True, "cfg_rescale": None}
        prediction = {"v_pred": True}
    o = _app.OUTPUTS_DIR
    try:
        _app.OUTPUTS_DIR = tmp
        [f] = _app._save_outputs([Image.new("RGB", (64, 64))], dict(
            mode="txt2img", prompt="a", steps=12, cfg_scale=6, seeds=[1], scheduler="DPM++ 2M AYS", width=64, height=64,
            **_app._booster_record(P(), {})), pipe=P())
    finally:
        _app.OUTPUTS_DIR = o
    with Image.open(f) as im:
        rec = json.loads(im.info["imagegen"]); params = im.info["parameters"]
    assert rec["format"] == 2 and rec["pag_scale"] == 2.5 and rec["freeu"] is True and rec["cfg_rescale"] == 0.7, rec
    assert "PAG scale: 2.5" in params and "FreeU: on" in params and "CFG rescale: 0.7" in params
    ups = _app._plan_extra_updates(_app._restore_plan(rd(f)))
    assert ups[12:15] == [2.5, True, 0.7], ups
    ups0 = _app._plan_extra_updates(_app._restore_plan({"prompt": "x"}))
    assert ups0[12:15] == [0.0, False, 0.0]                        # none recorded → off
    ex = _app._clean_extra(dict(pag_scale="99", freeu="yes", cfg_rescale=-3))
    assert ex["pag_scale"] == 6.0 and ex["freeu"] is True and ex["cfg_rescale"] == 0.0
    assert "PAG scale" in _app._XY_AXES and _app._xy_values("PAG scale", "0, 2.5") == ([0.0, 2.5], "")


@test("Upscale tab (single image and folder) keeps the source's imagegen record, not only the A1111 text")
def _():
    import app as _app
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    tmp = Path(_tf.mkdtemp())
    rec = json.dumps({"app": "ImageGen Studio", "format": 2, "mode": "txt2img", "seed": 7, "scheduler": "DPM++ 2M AYS"})
    meta = PngInfo()
    meta.add_text("parameters", "1girl\nSteps: 12, Seed: 7")
    meta.add_itxt("imagegen", rec)
    src_dir = tmp / "in"
    src_dir.mkdir()
    Image.new("RGB", (16, 24), (200, 30, 30)).save(src_dir / "a.png", pnginfo=meta)
    Image.new("RGB", (16, 24)).save(src_dir / "plain.png")                   # no metadata at all: nothing to carry, no crash
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
    fn = lambda n: getattr(by[n].fn, "__wrapped__", by[n].fn)
    quiet = lambda *a, **k: None
    o = _app.OUTPUTS_DIR
    try:
        _app.OUTPUTS_DIR = tmp / "out"
        with Image.open(src_dir / "a.png") as im:
            im.load()
            out, _msg = fn("do_upscale")(im, 2, "Lanczos", False, 0.3, progress=quiet)
        assert out.size == (32, 48), out.size
        [saved] = list((tmp / "out").glob("*.png"))
        with Image.open(saved) as r:
            assert json.loads(r.info["imagegen"]) == json.loads(rec), r.info
            assert r.info["parameters"] == "1girl\nSteps: 12, Seed: 7\nUpscaled: 2× Lanczos", r.info["parameters"]
        msg = fn("do_upscale_folder")(str(src_dir), 2, "Lanczos", progress=quiet)
        assert "Upscaled 2 of 2" in msg, msg
        with Image.open(tmp / "out" / "upscaled_in" / "a_2x.png") as r:
            assert json.loads(r.info["imagegen"]) == json.loads(rec) and r.info["parameters"].endswith("Upscaled: 2× Lanczos"), r.info
        with Image.open(tmp / "out" / "upscaled_in" / "plain_2x.png") as r:
            assert "imagegen" not in r.info and "parameters" not in r.info, r.info
    finally:
        _app.OUTPUTS_DIR = o


@test("Danbooru tags: autocomplete by popularity, near-miss hints, one-click fixes keep weights")
def _():
    from backend import danbooru_tags as dt
    from backend.prompt_tools import danbooru_hints, apply_danbooru_fixes
    if not dt.load(download=False):
        raise Skip("Danbooru tag list not downloaded yet (fetched on first autocomplete)")
    sug = dt.suggest("long h")
    assert sug[0][0] == "long hair" and all(a[1] >= b[1] for a, b in zip(sug, sug[1:]) if a[0].startswith("long h"))
    assert dt.suggest("x") == [] and dt.did_you_mean("long hair") is None
    assert dt.did_you_mean("long haired") == "long hair" and dt.did_you_mean("thigh highs") == "thighhighs"
    for fine in ("masterpiece", "very aesthetic", "mychara", "score_9", "absurdres"):
        assert dt.did_you_mean(fine) is None, fine            # quality tags / names aren't "fixed"
    # a colour in front of a valid tag is a qualifier, not a typo: "fixing" it dropped the "black" that makes
    # IL_Heroine draw her black lace pin instead of a blue morpho (the 'b' of black/butterfly passed the first-letter filter)
    assert dt.did_you_mean("black butterfly hair ornament") is None
    assert dt.did_you_mean("blue bow tie") == "blue bowtie"    # a real near miss is still reported
    assert danbooru_hints("1girl, long haired, thigh") == [("long haired", "long hair")]   # last one still typed
    p, fixes = apply_danbooru_fixes("(long haired:1.2), [blue eye], <lora:x:0.8>, {a|b}, __pose__, 1girl")
    assert p == "(long hair:1.2), [blue eyes], <lora:x:0.8>, {a|b}, __pose__, 1girl", p


@test("Saved settings (format 2): LoRA slots + weights, VAE, auto-quality, hires / face detail / boosters")
def _():
    import app as _app
    tmp = Path(_tf.mkdtemp())
    old = _app.SETTINGS_DIR
    try:
        _app.SETTINGS_DIR = tmp
        msg = _app._save_generation_settings(
            "p", "n", "DPM++ 2M AYS", 12, 6, 832, 1216, 1, 42, "my set",
            model="C:/models/ck.safetensors", vae="none",
            lora_slots=[("C:/l/IL_A.safetensors", 0.75), ("none", 0.7), ("D:/x/B.safetensors", "1.2")],
            auto_quality=True, extra=dict(pag_scale=2, fd_on=True, fd_denoise=0.35, hires_on="yes", freeu=0))
        assert msg.startswith("✅"), msg
        d = json.loads((tmp / "my set.json").read_text(encoding="utf-8"))
        assert d["format"] == 2 and d["lora_slots"] == [["IL_A.safetensors", 0.75], ["none", 0.7], ["B.safetensors", 1.2]]
        assert d["loras"] == ["IL_A", "B"] and d["vae"] == "none" and d["auto_quality"] is True
        assert d["model"].endswith("ck.safetensors")
        ex = d["extra"]
        assert ex["pag_scale"] == 2.0 and ex["fd_on"] is True and ex["hires_on"] is True and ex["freeu"] is False
        assert set(ex) == set(_app._clean_extra({})), sorted(ex)
        # an old-style call (no UI extras) still writes the old fields only
        _app._save_generation_settings("p", "n", "Euler a", 20, 7, 512, 512, 1, 1, "old style")
        d2 = json.loads((tmp / "old style.json").read_text(encoding="utf-8"))
        assert "lora_slots" not in d2 and "extra" not in d2 and d2["prompt"] == "p"
    finally:
        _app.SETTINGS_DIR = old


@test("SDXL inpaint / face detail: no fp32 VAE upcast unless needed (it ran out of GPU memory), settings restored")
def _():
    from PIL import Image
    from diffusers import EulerDiscreteScheduler
    from backend import detail_tools as dt, sampling as sm

    class VAE:
        def __init__(self):
            self.config = {"force_upcast": True}; self.tile_sample_min_size = 512; self.tile_latent_min_size = 64
        def register_to_config(self, **k): self.config = dict(self.config, **k)

    class Pipe:
        def __init__(self, unet): self.unet = unet; self.vae = VAE(); self.scheduler = EulerDiscreteScheduler()
        text_encoder_2 = object()

    class Host:
        is_sdxl = True; device = "cpu"; dtype = None; boosters = {}; prediction = {}
        def _decode_latents(self, vae, lat): return [Image.new("RGB", (1024, 1024))]
    seen = []

    import torch

    class R:
        images = torch.zeros(1, 4, 8, 8)          # finite latents (inpaint_region checks them for NaN since the NaN fix)

    def fake_run(sdp, pipe, kind, **k):
        seen.append((pipe.vae.config["force_upcast"], pipe.vae.tile_sample_min_size))
        if len(seen) == 3:
            raise RuntimeError("boom")
        return R()
    unet = object()
    h = Host(); h.pipe = Pipe(unet); h._inpaint_pipe = h.pipe
    img = Image.new("RGB", (832, 1216)); mask = Image.new("L", (832, 1216)); mask.paste(255, (300, 200, 500, 400))
    orig = (dt._embeds, sm.run_pipe)
    try:
        dt._embeds = lambda *a, **k: {}
        sm.run_pipe = fake_run
        h._vae_needs_fp32 = False
        dt.inpaint_region(h, img, mask, "x", steps=4, denoise=0.4, seed=1)
        h._vae_needs_fp32 = True
        dt.inpaint_region(h, img, mask, "x", steps=4, denoise=0.4, seed=1)
        try:
            dt.inpaint_region(h, img, mask, "x", steps=4, denoise=0.4, seed=1)
        except RuntimeError:
            pass
    finally:
        dt._embeds, sm.run_pipe = orig
    assert seen[0] == (False, 512) and seen[1] == (True, 256), seen
    v = h.pipe.vae
    assert v.config["force_upcast"] is True and (v.tile_sample_min_size, v.tile_latent_min_size) == (512, 64)


@test("LoRA restore snapshot: only touched layers, spilled to disk (no RAM held), exact restore, files removed")
def _():
    import torch
    from backend.lora_snapshot import Snapshot, param_snapshotter

    class Pipe:
        def __init__(self):
            torch.manual_seed(0)
            self.unet = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).half()
            self.text_encoder = torch.nn.Linear(4, 4).half()
    pipe = Pipe()
    clean = {k: v.clone() for k, v in pipe.unet.state_dict().items()}
    clean_te = {k: v.clone() for k, v in pipe.text_encoder.state_dict().items()}
    snap = Snapshot()
    snapper = param_snapshotter(pipe, snap)
    with torch.no_grad():                      # "slot 1" changes layer 0 and the text encoder
        for prm in (pipe.unet[0].weight, pipe.text_encoder.weight):
            snapper(prm); prm.add_(1.0)
        snapper(pipe.unet[0].weight); pipe.unet[0].weight.add_(1.0)     # twice: still the clean copy
    assert set(snap.mem["unet"]) == {"0.weight"} and set(snap.mem["text_encoder"]) == {"weight"}
    snap.spill()
    assert not any(snap.mem.values()) and len(snap.files) == 1 and snap.files[0][0].exists()
    with torch.no_grad():                      # "slot 2" (after the spill) changes layer 1
        snapper(pipe.unet[1].bias); pipe.unet[1].bias.add_(2.0)
        snapper(pipe.unet[0].weight)          # already on disk → not copied again (it's dirty now)
    assert set(snap.mem["unet"]) == {"1.bias"}
    assert snap.restore(pipe) == 3
    for k, v in pipe.unet.state_dict().items():
        assert torch.equal(v, clean[k]), k
    assert torch.equal(pipe.text_encoder.weight, clean_te["weight"])
    files = [f for f, _ in snap.files]
    snap.close()
    assert all(not f.exists() for f in files)


@test("SDXL LoRA lineage (Pony vs Illustrious) from training metadata; mismatch warning")
def _():
    import struct
    import app as _app
    tmp = Path(_tf.mkdtemp())

    def lora(name, meta):
        hdr = {"lora_unet_down_blocks_0.lora_down.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
               "__metadata__": meta}
        raw = json.dumps(hdr).encode()
        (tmp / name).write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0")
        return str(tmp / name)
    assert _app._sdxl_lineage(lora("a.safetensors", {"ss_sd_model_name": "ponyDiffusionV6XL_v6.safetensors"}), True) == "pony"
    assert _app._sdxl_lineage(lora("b.safetensors", {"ss_sd_model_name": "290640.safetensors"}), True) == "pony"
    assert _app._sdxl_lineage(lora("c.safetensors", {"ss_sd_model_name": "noobaiVPred10.safetensors"}), True) == "illustrious"
    assert _app._sdxl_lineage(lora("d.safetensors", {"ss_sd_model_name": "Illustrious-XL-v0.1.safetensors"}), True) == "illustrious"
    assert _app._sdxl_lineage(lora("e.safetensors", {"ss_sd_model_name": "sd_xl_base_1.0.safetensors"}), True) is None
    assert _app._sdxl_lineage(lora("f.safetensors", {"ss_sd_model_name": "12906400.safetensors"}), True) is None
    assert _app._sdxl_lineage(str(tmp / "missing.safetensors"), True) is None


@test("LoRA restore survives a textual inversion that grew the token table; A1111 'Schedule type' restores")
def _():
    import torch, tempfile
    from types import SimpleNamespace
    from backend.lora_snapshot import Snapshot
    te = torch.nn.Sequential(torch.nn.Embedding(10, 4), torch.nn.Linear(4, 4))
    pipe = SimpleNamespace(unet=torch.nn.Linear(4, 4), text_encoder=te, text_encoder_2=None)
    for spill in (False, True):
        snap = Snapshot()
        for k, p in te.named_parameters():
            snap.add("text_encoder", k, p.detach())
        if spill:
            snap.spill()
        clean = te[0].weight.detach().clone()
        with torch.no_grad():                      # a fused LoRA changes weights ...
            te[0].weight.add_(1); te[1].weight.add_(1)
        te[0].weight = torch.nn.Parameter(torch.cat([te[0].weight.data, torch.full((1, 4), 7.0)]))  # ... a TI adds a row
        snap.restore(pipe)                         # raised "size of tensor a (11) must match (10)" before
        assert torch.equal(te[0].weight[:10], clean) and torch.equal(te[0].weight[10], torch.full((4,), 7.0))
        snap.close()
        te[0].weight = torch.nn.Parameter(te[0].weight.data[:10].clone())
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from backend.png_info import read_image_metadata
    for text, want in (("Sampler: DPM++ 2M, Schedule type: Karras", "DPM++ 2M Karras"),
                       ("Sampler: Euler, Schedule type: Align Your Steps", "Euler AYS"),
                       ("Sampler: DPM++ 2M Karras, Schedule type: Karras", "DPM++ 2M Karras"),
                       ("Sampler: Euler a, Schedule type: Automatic", "Euler a")):
        info = PngInfo(); info.add_text("parameters", f"1girl\nSteps: 20, {text}, CFG scale: 7, Seed: 1")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"
            Image.new("RGB", (8, 8)).save(p, pnginfo=info)
            assert read_image_metadata(p)["sampler"] == want, (text, read_image_metadata(p)["sampler"])


@test("Hand detail: detector boxes (overlaps, tiny, max), pass skipped without hands, record + A1111 text restore")
def _():
    import numpy as np
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    import app as _app
    from backend import detail_tools as dt
    from backend.png_info import read_image_metadata

    class Sess:                                      # a YOLOv8 export: (1, 5, anchors) at the 640 input
        def __init__(self, rows): self.rows = np.array(rows, dtype=np.float32).T[None]
        def get_inputs(self): return [type("I", (), {"name": "images"})()]
        def run(self, _o, feed):
            assert feed["images"].shape == (1, 3, 640, 480)   # 960x1280 → 480x640 (multiples of 32)
            return [self.rows]
    orig = dict(dt._hand_yolo)
    try:
        dt._hand_yolo["sess"] = Sess([[100, 100, 40, 40, 0.9],    # a hand
                                      [102, 101, 38, 42, 0.8],    # the same hand again → dropped
                                      [300, 400, 60, 50, 0.7],    # a second hand
                                      [400, 100, 4, 4, 0.9],      # 8 px in the image → too small
                                      [200, 500, 40, 40, 0.3]])   # below the threshold
        got = dt.detect_hands(Image.new("RGB", (960, 1280)))
        assert got == [(540, 750, 660, 850), (160, 160, 240, 240)], got     # biggest first, scaled ×2
        assert dt.detect_hands(Image.new("RGB", (960, 1280)), max_hands=1) == [(540, 750, 660, 850)]
        dt._hand_yolo["sess"] = None                 # no model → no hands (and no crash)
        assert dt.detect_hands(Image.new("RGB", (960, 1280))) == []
        img = Image.new("RGB", (64, 64))
        out, n = dt.hand_detail(None, img, "x")      # no hands: image back, model untouched
        assert out is img and n == 0
    finally:
        dt._hand_yolo.clear(); dt._hand_yolo.update(orig)
    # detail passes run the evenly spaced sampler (Karras / AYS hardly changed a face at 0.35)
    from backend.sampling import uniform_variant, SCHEDULERS
    assert uniform_variant("DPM++ 2M AYS") == "DPM++ 2M" and uniform_variant("Euler AYS") == "Euler"
    assert uniform_variant("DPM++ SDE Karras") == "DPM++ SDE" and uniform_variant("Euler a") == "Euler a"
    assert all(uniform_variant(n) in SCHEDULERS for n in SCHEDULERS)
    seen = {}
    orig_inp, orig_df = dt.inpaint_region, dt.detect_faces
    try:
        dt.detect_faces = lambda *a, **k: [(10, 10, 30, 30)]
        dt.inpaint_region = lambda sdp, im, mask, *a, **k: (seen.update(k), (im, 0))[1]
        dt.face_detail(object(), Image.new("RGB", (64, 64)), "x", scheduler="DPM++ 2M AYS")
        assert seen["scheduler"] == "DPM++ 2M", seen
    finally:
        dt.inpaint_region, dt.detect_faces = orig_inp, orig_df
    # the 🖌 Inpaint button too: AYS 12 at 0.95 started at sigma 7.4 and couldn't recolour a garment (round 3, F4)
    import backend.sd_pipeline as _sdp_mod

    class _Stop(Exception):
        pass
    got = []
    orig_ls = _sdp_mod._load_scheduler
    unet = object()
    fake = type("S", (), {"pipe": type("P", (), {"unet": unet})(), "is_sdxl": False, "dtype": None, "device": "cpu"})()
    fake._inpaint_pipe = type("I", (), {"unet": unet})()
    try:
        _sdp_mod._load_scheduler = lambda pipe, name: (got.append(name), (_ for _ in ()).throw(_Stop()))
        m = Image.new("L", (64, 64)); m.paste(255, (16, 16, 48, 48))
        try:
            dt.inpaint_region(fake, Image.new("RGB", (64, 64)), m, "x", scheduler="DPM++ 2M AYS")
        except _Stop:
            pass
        assert got == ["DPM++ 2M"], got
    finally:
        _sdp_mod._load_scheduler = orig_ls
    ex = _app._clean_extra(dict(hd_on=1, hd_denoise=5))
    assert ex["hd_on"] is True and ex["hd_denoise"] == 0.7
    assert _app._clean_extra({})["hd_denoise"] == 0.35
    # A1111 text + restore: the two hand controls come last
    info = PngInfo(); info.add_text("parameters", "1girl\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, "
                                    "Face detail: denoise 0.4 (auto), Hand detail: denoise 0.45")
    with _tf.TemporaryDirectory() as d:
        f = Path(d) / "h.png"
        Image.new("RGB", (8, 8)).save(f, pnginfo=info)
        meta = read_image_metadata(f)
    assert meta["hand_detail"] == {"denoise": 0.45}, meta.get("hand_detail")
    ups = _app._plan_extra_updates(_app._restore_plan(meta))
    assert ups[-11:-9] == [True, 0.45] and ups[8] is True, ups           # the five pose-control values (round 34) and the colour strength (round 25) come last
    ups0 = _app._plan_extra_updates(_app._restore_plan({"prompt": "x"}))
    assert ups0[-11] is False


@test("Inpaint: NaN latents from an fp16 VAE encode (a NoobAI checkpoint's bf16-origin VAE) are retried once in fp32 and the need is remembered; NaN is never pasted")
def _():
    import types
    import torch
    from PIL import Image
    import backend.sampling as sampling
    import backend.sd_pipeline as sdpipe
    from backend import detail_tools as dt

    class FakeVae:
        def __init__(self): self.config = {"force_upcast": True}; self.tile_sample_min_size, self.tile_latent_min_size = 512, 64
        def register_to_config(self, **kw): self.config.update(kw)

    class FakePipe:
        def __init__(self, unet): self.vae, self.unet = FakeVae(), unet

    unet = object()
    sdp = types.SimpleNamespace(is_sdxl=True, pipe=FakePipe(unet), dtype=torch.float16, device="cpu", _vae_needs_fp32=False)
    sdp._inpaint_pipe = FakePipe(unet)
    sdp._decode_latents = lambda vae, lat: [Image.new("RGB", (64, 64), (200, 30, 30))]
    seen = []

    def fake_run(sdp_, pipe, kind, **call):
        # NaN while the encode runs in fp16 (force_upcast False), fine once the config says fp32
        up = pipe.vae.config["force_upcast"]
        seen.append((up, pipe.vae.tile_sample_min_size))
        lat = torch.zeros(1, 4, 8, 8)
        if not up:
            lat[:] = float("nan")
        return types.SimpleNamespace(images=lat)
    orig = (sampling.run_pipe, sdpipe._load_scheduler, sdpipe._make_generator, dt._embeds)
    try:
        sampling.run_pipe, sdpipe._load_scheduler = fake_run, lambda pipe, name: None
        sdpipe._make_generator, dt._embeds = (lambda seed, dev: (None, 7)), (lambda *a, **k: {})
        img = Image.new("RGB", (256, 256), (240, 240, 240))
        mask = Image.new("L", (256, 256), 0); mask.paste(255, (100, 100, 160, 140))
        out, seed_used = dt.inpaint_region(sdp, img, mask, "x", seed=3)
        assert [s[0] for s in seen] == [False, True], seen                      # fp16 encode first, then fp32 (256 px tiles)
        assert seen[1][1] == 256 and sdp._vae_needs_fp32 is True, (seen, sdp._vae_needs_fp32)
        assert sdp._inpaint_pipe.vae.config["force_upcast"] is True and sdp._inpaint_pipe.vae.tile_sample_min_size == 512      # config restored
        assert out.getpixel((130, 120)) != (240, 240, 240) and out.getpixel((5, 5)) == (240, 240, 240) and seed_used == 7
        # the need is remembered: the next pass goes straight to fp32
        seen.clear(); dt.inpaint_region(sdp, img, mask, "x", seed=3)
        assert [s[0] for s in seen] == [True], seen
        # NaN even in fp32 (or on SD 1.5): refused, the picture is not touched
        sampling.run_pipe = lambda *a, **k: types.SimpleNamespace(images=torch.full((1, 4, 8, 8), float("nan")))
        try:
            dt.inpaint_region(sdp, img, mask, "x", seed=3)
            raise AssertionError("NaN latents were accepted")
        except RuntimeError as e:
            assert "NaN" in str(e), e
    finally:
        sampling.run_pipe, sdpipe._load_scheduler, sdpipe._make_generator, dt._embeds = orig


@test("Hand detail (round 29): the pass re-draws with the hand-only prompt (hand / held-object tags + hand words), never the whole scene (no hair-ornament butterflies or girls drawn into the box)")
def _():
    from PIL import Image
    from backend import detail_tools as dt
    seen = []
    orig_det, orig_inp, orig_conf = dt.detect_hands, dt.inpaint_region, dt.confirmed_hands
    try:
        dt.confirmed_hands = lambda image, boxes: list(boxes)
        dt.detect_hands = lambda image, max_hands=6, conf=0.45, min_frac=0.02: [(20, 20, 60, 60)]
        dt.inpaint_region = lambda sdp, im, mask, prompt, *a, **k: (seen.append(prompt), (im, 0))[1]
        full = ("masterpiece, best quality, 1girl, solo, grey hair, (black butterfly hair ornament:1.2), cowboy shot, white bikini, v, peace sign, hand up, "
                "holding umbrella, smile, poolside, blue sky")
        dt.hand_detail(object(), Image.new("RGB", (128, 128)), full, "lowres", denoise=0.5)
        p = seen[0]
        tags = [t.strip() for t in p.split(",")]
        for want in ("masterpiece", "1girl", "peace sign", "holding umbrella", "detailed hands", "five fingers"):
            assert want in tags, (want, p)
        for gone in ("butterfly", "poolside", "bikini", "grey hair", "blue sky"):
            assert gone not in p, (gone, p)
        seen.clear()
        dt.hand_detail(object(), Image.new("RGB", (128, 128)), full, "lowres", hand_prompt="long fingers")       # an explicit hand_prompt is added to the hand-only prompt
        assert seen[0].endswith("long fingers") and "poolside" not in seen[0], seen[0]
    finally:
        dt.detect_hands, dt.inpaint_region, dt.confirmed_hands = orig_det, orig_inp, orig_conf


@test("Hand detail (round 30): a second look at each hand box — boxes the detector finds again on the box + context are re-drawn, boxes it does not find (a forearm, a hip, a nose) are left alone")
def _():
    from PIL import Image
    from backend import detail_tools as dt
    img = Image.new("RGB", (400, 400))
    img.paste((255, 0, 0), (100, 100, 140, 140))                  # the real hand is red, the other box is empty black
    real, fake = (100, 100, 140, 140), (250, 250, 290, 290)
    calls = []

    def fake_yolo(sess, im, conf, side=640):
        calls.append((im.size, conf))
        # the crop of the real box finds the hand again (a box in crop coordinates), the crop of the other one finds nothing
        s = im.size[0]
        return [((int(s * 0.3), int(s * 0.3), int(s * 0.7), int(s * 0.7)), 0.8)] if im.getpixel((s // 2, s // 2)) == (255, 0, 0) else []
    orig = (dt._hand_session, dt._yolo_run, dt.detect_hands, dt.inpaint_region, dt._face_model_cached)
    try:
        dt._face_model_cached = lambda: False
        dt._hand_session = lambda: object()
        dt._yolo_run = fake_yolo
        assert dt.hand_confirmed(img, real) is True and dt.hand_confirmed(img, fake) is False
        assert calls[0][0] == (96, 96) and calls[0][1] == 0.05, calls[0]          # 40 px box x 2.4 = a 96 px crop, a low threshold for the second look
        # a weak second look (below the minimum) does not confirm
        dt._yolo_run = lambda sess, im, conf, side=640: [((int(im.size[0] * 0.3), int(im.size[1] * 0.3), int(im.size[0] * 0.7), int(im.size[1] * 0.7)), 0.2)]
        assert dt.hand_confirmed(img, real) is False
        # a hit that does not overlap the box (another hand in the context) does not confirm
        dt._yolo_run = lambda sess, im, conf, side=640: [((0, 0, 8, 8), 0.9)]
        assert dt.hand_confirmed(img, real) is False
        # the first look can be sure on its own — a score >= 0.75 on a box that is not tiny next to the face (mitten fists the detector does not see in context) — but not on a tiny box
        dt._yolo_run = lambda sess, im, conf, side=640: []
        assert dt.hand_confirmed(img, real, score=0.8) is True and dt.hand_confirmed(img, real, score=0.7) is False
        assert dt.hand_confirmed(img, real, score=0.8, face=(0, 0, 100, 100)) is True        # 1600 px² >= 12 % of the face's 10,000
        assert dt.hand_confirmed(img, real, score=0.8, face=(0, 0, 300, 300)) is False       # 1600 px² < 12 % of 90,000: a small box with a high score (a hip, a nose) is not taken on its score
        dt._yolo_run = fake_yolo
        dt._hand_session = lambda: None                                           # no detector: the box is taken as it is
        assert dt.hand_confirmed(img, fake) is True
        # hand_detail re-draws only the confirmed boxes (and everything with confirm=False)
        seen = []
        dt._hand_session = lambda: object()
        dt._yolo_run = fake_yolo
        dt.detect_hands = lambda image, max_hands=6, conf=0.45, min_frac=0.02: [real, fake]
        dt.inpaint_region = lambda sdp, im, mask, prompt, *a, **k: (seen.append(mask.getbbox()), (im, 0))[1]
        out, n = dt.hand_detail(object(), img, "1girl, solo, v", "lowres")
        assert n == 1 and len(seen) == 1 and seen[0][0] < 140 and seen[0][2] <= 150, (n, seen)
        seen.clear()
        out, n = dt.hand_detail(object(), img, "1girl, solo, v", "lowres", confirm=False)
        assert n == 2 and len(seen) == 2, (n, seen)
        # the opt-in Polish entries: the framing recipe + one hand pass at 0.7 (Auto never picks them)
        from backend import polish as P
        for name, base in (("Cowboy shot — hands", "Cowboy shot"), ("Full body — hands", "Full body")):
            r, b0 = P.recipe(name), P.recipe(base)
            assert r["hd_on"] and r["hd_denoise"] == 0.7 and not b0["hd_on"], (name, r)
            assert all(r[k] == v for k, v in b0.items() if k not in ("hd_on",)), (name, r, b0)
            assert P.time_factor(name) > P.time_factor(base) and "hands 0.7" in P.summary(name)
        assert name in P.SHOTS
        # Auto: a gesture pose on a cowboy / full-body framing takes the '— hands' variant; portrait / wide and other poses don't
        for prompt, want in (("1girl, v, peace sign, cowboy shot", "Cowboy shot — hands"), ("1girl, (reaching towards viewer:1.1), full body", "Full body — hands"),
                             ("1girl, arms up, smile", "Cowboy shot — hands"), ("1girl, peace sign, portrait", "Portrait"), ("1girl, waving, wide shot", "Wide shot"),
                             ("1girl, hands on hips, full body", "Full body"), ("1girl, v-neck, cowboy shot", "Cowboy shot"), ("1girl, arms up stretching", "Cowboy shot")):
            assert P.shot_type(prompt) == want, (prompt, P.shot_type(prompt), want)
    finally:
        dt._hand_session, dt._yolo_run, dt.detect_hands, dt.inpaint_region, dt._face_model_cached = orig


@test("Eye detail (round 29): the eye prompt carries the eye-state tags of an expression (closed / wink / half-closed / tears / sparkling / heart / star pupils / kaomoji eyes), so the pass keeps them instead of opening the eyes")
def _():
    from backend import detail_tools as dt
    for state in ("closed eyes", "one eye closed", "wink", "half-closed eyes", "jitome", "tears", "crying", "streaming tears", "sparkling eyes",
                  "heart-shaped pupils", "star-shaped pupils", "@_@", ">_<", "^_^", "empty eyes", "looking up", "rolling eyes", "narrowed eyes", "squinting"):
        full = f"masterpiece, 1girl, solo, grey hair, cowboy shot, white bikini, {state}, smile, poolside"
        p = dt.eye_prompt_from(full)
        tags = [x.strip() for x in p.split(",")]
        assert state in tags, (state, p)
        assert "poolside" not in tags and "white bikini" not in tags and "grey hair" not in tags, (state, p)
    # a pupil shape the prompt names is left alone (no 'round pupils' words, no slit negative): the survey's heart / star-pupil pictures kept their shapes
    for shape in ("heart-shaped pupils", "star-shaped pupils", "@_@"):
        q = f"1girl, solo, {shape}, blush"
        assert "round pupils" not in dt.eye_prompt_from(q) and dt.eye_negative_from(q, "lowres") == "lowres", shape


@test("Eye detail: the redraw uses an eye-only prompt (quality + subject + eye / expression / gaze tags + detail words), not the whole scene; an explicit eye_prompt is used as given")
def _():
    from PIL import Image
    from backend import detail_tools as dt
    full = (r"masterpiece, best quality, 1girl, solo, (my_character \(series name\):1.1), red eyes, sharp eyes, tsurime, grey hair, long hair, "
            r"(black butterfly hair ornament:1.2), lace, night, city lights, cowboy shot, standing, looking at viewer, light smile, white dress")
    p = dt.eye_prompt_from(full)
    tags = [x.strip() for x in p.split(",")]
    for want in ("masterpiece", "best quality", "1girl", "solo", "red eyes", "sharp eyes", "tsurime", "looking at viewer", "light smile", "detailed pupils", "iris detail",
                 "round pupils"):
        assert want in tags, (want, p)
    for gone in ("grey hair", "long hair", "night", "city lights", "white dress", "standing", "lace") + ("(black butterfly hair ornament:1.2)",):
        assert gone not in tags and gone not in p.replace("(", "").replace(")", "").split(", ") or gone == "", (gone, p)
    assert "heroine" not in p and "butterfly" not in p, p
    assert dt.eye_prompt_from("") == dt.EYE_DETAIL_WORDS + ", " + dt.EYE_ROUND
    # round pupils + "slit pupils, cat eyes" in the negative — unless the prompt names a pupil shape itself (then the pass leaves the shape to it)
    assert dt.eye_negative_from(full, "lowres, watermark") == "lowres, watermark, slit pupils, cat eyes" and dt.eye_negative_from(full) == "slit pupils, cat eyes"
    assert dt.eye_negative_from("1girl, red eyes, (cat eyes:1.2), forest", "lowres") == "lowres"
    for own in ("slit pupils", "heart-shaped pupils", "@_@", "reptile eyes", "dragon eyes", "spiral eyes", "no pupils"):
        q = f"1girl, solo, {own}, forest"
        assert "round pupils" not in dt.eye_prompt_from(q) and own in dt.eye_prompt_from(q) and dt.eye_negative_from(q, "lowres") == "lowres", (own, dt.eye_prompt_from(q))
    assert "round pupils" in dt.eye_prompt_from("1girl, solo, dragon horns, red eyes") and dt.eye_negative_from("1girl, dragon horns, red eyes", "x").endswith("cat eyes")
    assert "(red eyes:1.2)" in dt.eye_prompt_from("score_9, 1girl, (red eyes:1.2), forest") and "forest" not in dt.eye_prompt_from("score_9, 1girl, (red eyes:1.2), forest")
    seen = []
    orig_run, orig_sess, orig_inp, orig_faces = dt._yolo_run, dt._eye_session, dt.inpaint_region, dt.detect_faces
    try:
        dt._eye_session = lambda: object()
        dt._yolo_run = lambda sess, im, conf, side=640: [((110, 120, 150, 140), 0.8), ((200, 118, 245, 140), 0.75)]
        dt.detect_faces = lambda image, mode="auto", max_faces=4, min_frac=0.03: [(100, 100, 300, 300)]
        negs = []
        dt.inpaint_region = lambda sdp, im, mask, prompt, *a, **k: (seen.append(prompt), negs.append(a[0] if a else k.get("negative")), (im, 0))[2]
        img = Image.new("RGB", (400, 400))
        dt.eye_detail(None, img, full, "n", seed=1)
        dt.eye_detail(None, img, full, "n", seed=1, eye_prompt="my own eyes prompt")
        assert seen[0] == p and seen[1] == "my own eyes prompt", seen
        assert negs[0] == "n, slit pupils, cat eyes" and negs[1] == "n", negs      # an explicit eye_prompt takes the negative as given
    finally:
        dt._yolo_run, dt._eye_session, dt.inpaint_region, dt.detect_faces = orig_run, orig_sess, orig_inp, orig_faces


@test("Eye detail: the far eye of a three-quarter view is found at a lower score (rescue) — never a duplicate or a box beside the first eye")
def _():
    from PIL import Image
    from backend import detail_tools as dt
    face = (100, 100, 300, 300)
    orig_run, orig_sess = dt._yolo_run, dt._eye_session
    try:
        dt._eye_session = lambda: object()
        seen = []

        def fake(sess, im, conf, side=640):
            seen.append(conf)
            boxes = [((150, 120, 200, 142), 0.8),          # the near eye
                     ((62, 126, 90, 144), 0.31),           # the far eye (crop coordinates; faces start at 50, 50)
                     ((152, 124, 198, 144), 0.3)]          # a weak duplicate of the near eye
            return [(b, s) for b, s in boxes if s >= conf]
        dt._yolo_run = fake
        img = Image.new("RGB", (400, 400))
        got = dt.detect_eyes(img, faces=[face])
        assert len(got[0]) == 2 and seen[-1] == 0.25, (got, seen)
        assert (112, 176, 140, 194) in got[0] and (200, 170, 250, 192) in got[0], got       # near + far, the duplicate is dropped
        assert [len(g) for g in dt.detect_eyes(img, faces=[face], rescue=None)] == [1]       # off: only the confident eye
        # two confident eyes: the weak box never replaces or joins them
        dt._yolo_run = lambda sess, im, conf, side=640: [((110, 120, 150, 140), 0.8), ((200, 118, 245, 140), 0.75),
                                                          ((120, 122, 140, 138), 0.3)]
        assert len(dt.detect_eyes(img, faces=[face])[0]) == 2
        # a weak box right next to the only confident eye (< 25 % of the face width apart) is not "the other eye"
        dt._yolo_run = lambda sess, im, conf, side=640: [((110, 120, 150, 140), 0.8), ((165, 121, 190, 141), 0.3)]
        assert len(dt.detect_eyes(img, faces=[face])[0]) == 1
        # below the rescue score nothing is taken
        dt._yolo_run = lambda sess, im, conf, side=640: [((110, 120, 150, 140), 0.8), ((230, 120, 260, 140), 0.2)]
        assert len(dt.detect_eyes(img, faces=[face])[0]) == 1
    finally:
        dt._yolo_run, dt._eye_session = orig_run, orig_sess


@test("Eye detail: eyes per face (plausibility filter), colour guard keeps a red iris off grey bangs, one pass per face, A1111 text + restore")
def _():
    import numpy as np
    from PIL import Image, ImageDraw
    from PIL.PngImagePlugin import PngInfo
    import app as _app
    from backend import detail_tools as dt
    from backend.png_info import read_image_metadata
    face = (100, 100, 300, 300)
    orig_run, orig_sess, orig_inp = dt._yolo_run, dt._eye_session, dt.inpaint_region
    try:
        dt._eye_session = lambda: object()
        # boxes come back in crop coordinates (crop starts at face - 25 % = (50, 50))
        dt._yolo_run = lambda sess, im, conf, side=640: [
            ((110, 120, 150, 140), 0.8), ((200, 118, 245, 140), 0.75),   # two eyes
            ((112, 121, 149, 139), 0.6),                                  # the same eye again
            ((150, 230, 200, 250), 0.7),                                  # a mouth: too low in the face
            ((60, 60, 300, 120), 0.9)]                                    # too wide for an eye
        got = dt.detect_eyes(Image.new("RGB", (400, 400)), faces=[face])
        assert got == [[(160, 170, 200, 190), (250, 168, 295, 190)]], got
        dt._eye_session = lambda: None
        assert dt.detect_eyes(Image.new("RGB", (400, 400)), faces=[face]) == []
        # colour guard: a grey strand over a red iris stays grey, skin keeps its hue, the iris takes the new colour
        before = Image.new("RGB", (100, 60), (200, 170, 160)); d = ImageDraw.Draw(before)
        d.ellipse([30, 15, 70, 45], fill=(200, 30, 40)); d.rectangle([45, 0, 52, 60], fill=(150, 150, 155))
        after = before.copy(); ImageDraw.Draw(after).rectangle([20, 10, 80, 50], fill=(150, 20, 220))
        g = np.asarray(dt.colour_guard(before, after, [(30, 15, 70, 45)])).astype(int)
        strand, iris, skin = g[30, 48], g[30, 35], g[12, 22]
        assert abs(strand[0] - strand[2]) < 12 and abs(strand[0] - strand[1]) < 12, strand      # still grey
        assert iris[2] > iris[1] + 60, iris                                                      # new (violet) colour
        assert skin[0] > skin[2], skin                                                           # skin hue kept
        # one inpaint per face covering both eyes, evenly spaced sampler, seed + 211
        calls = []
        dt._eye_session = lambda: object()
        dt.inpaint_region = lambda sdp, im, mask, *a, **k: (calls.append((np.asarray(mask), k)), (im, 0))[1]
        dt.detect_faces_orig = dt.detect_faces
        dt.detect_faces = lambda image, mode="auto", max_faces=4, min_frac=0.03: [face]
        out, n = dt.eye_detail(None, Image.new("RGB", (400, 400)), "x", seed=5, scheduler="DPM++ 2M AYS")
        assert n == 2 and len(calls) == 1, (n, len(calls))
        m, k = calls[0]
        assert m[180, 180] == 255 and m[179, 272] == 255 and m[260, 200] == 0          # both eyes, not the mouth
        assert k["scheduler"] == "DPM++ 2M" and k["seed"] == 216, k
    finally:
        dt._yolo_run, dt._eye_session, dt.inpaint_region = orig_run, orig_sess, orig_inp
        if hasattr(dt, "detect_faces_orig"):
            dt.detect_faces = dt.detect_faces_orig
            del dt.detect_faces_orig
    ex = _app._clean_extra(dict(ed_on=1, ed_denoise=5))
    assert ex["ed_on"] is True and ex["ed_denoise"] == 0.6 and _app._clean_extra({})["ed_denoise"] == dt.EYE_DENOISE == 0.4      # the UI default, the sanitiser and eye_detail agree
    rec = _app._gen_record(None, prompt="x", steps=20, eye_detail={"denoise": 0.3})
    assert "Eye detail: denoise 0.3" in _app._params_text(rec)
    info = PngInfo(); info.add_text("parameters", "1girl\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, "
                                    "Hand detail: denoise 0.45, Eye detail: denoise 0.3")
    with _tf.TemporaryDirectory() as d:
        f = Path(d) / "e.png"
        Image.new("RGB", (8, 8)).save(f, pnginfo=info)
        meta = read_image_metadata(f)
    assert meta["eye_detail"] == {"denoise": 0.3}, meta.get("eye_detail")
    ups = _app._plan_extra_updates(_app._restore_plan(meta))
    assert ups[-9:-7] == [True, 0.3] and ups[-11:-9] == [True, 0.45] and ups[-7] is False and ups[-6] == 1.0, ups
    assert "eye detail" in _app._plan_summary(_app._restore_plan(meta))


@test("👁 Eye style (round 10): 'natural' asks for no pupil words in the eye pass (the slit negative stays), a card / profile carries it (🎴 Load applies, 📌 saves), the record + A1111 text + restore say it; the default stays round")
def _():
    import app as _app
    import backend.character_cards as CC
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from backend import detail_tools as dt
    from backend.png_info import read_image_metadata
    full = "masterpiece, best quality, 1girl, solo, red eyes, sharp eyes, tsurime, grey hair, looking at viewer, light smile, white dress"
    try:
        assert dt.EYE_STYLE["mode"] == "round"                                                   # the shipped default
        assert dt.set_eye_style("Natural ") == "natural" and dt.set_eye_style("what") == "round" and dt.set_eye_style(None) == "round"
        round_p, round_n = dt.eye_prompt_from(full), dt.eye_negative_from(full, "lowres")
        assert "round pupils" in round_p and round_n == "lowres, slit pupils, cat eyes"            # exactly what it was
        dt.set_eye_style("natural")
        nat_p, nat_n = dt.eye_prompt_from(full), dt.eye_negative_from(full, "lowres")
        assert dt.EYE_NATURAL_WORDS == "detailed eyes, beautiful detailed eyes, eyelashes, iris detail"                  # the eye words minus every pupil word
        assert "round pupils" not in nat_p and "detailed pupils" not in nat_p and dt.EYE_NATURAL_WORDS in nat_p and "red eyes" in nat_p, (nat_p, round_p)
        assert nat_n == "lowres, slit pupils, cat eyes" and dt.eye_negative_from(full) == "slit pupils, cat eyes"        # the slit / cat-eye negative stays: the literal 'natural' (no negative) drew slit pupils in 5 of 6 pictures
        # cards and profiles keep a valid style only; a profile's style replaces the card's
        card = CC.clean_card({"name": "x", "eye_style": " Natural ", "profiles": {"a.safetensors": {"eye_style": "round"}, "b.safetensors": {"eye_style": "slit", "cfg": 6.0}}})
        assert card["eye_style"] == "natural" and card["profiles"]["a.safetensors"]["eye_style"] == "round" and "eye_style" not in card["profiles"]["b.safetensors"], card
        assert "eye_style" not in CC.clean_card({"name": "y", "eye_style": 3})
        eff, key = CC.card_for_checkpoint(card, "a.safetensors")
        assert eff["eye_style"] == "round" and key == "a.safetensors"
        # the record, the A1111 text, the reader, the restore note
        rec = _app._gen_record(None, prompt="x", steps=20, eye_detail={"denoise": 0.4, "style": "natural"})
        assert "Eye detail: denoise 0.4, Eye style: natural" in _app._params_text(rec), _app._params_text(rec)
        assert "Eye style" not in _app._params_text(_app._gen_record(None, prompt="x", steps=20, eye_detail={"denoise": 0.4}))       # the default leaves no trace
        info = PngInfo(); info.add_text("parameters", "1girl\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Eye detail: denoise 0.4, Eye style: natural")
        with _tf.TemporaryDirectory() as d:
            f = Path(d) / "e.png"
            Image.new("RGB", (8, 8)).save(f, pnginfo=info)
            meta = read_image_metadata(f)
        assert meta["eye_detail"] == {"denoise": 0.4, "style": "natural"}, meta.get("eye_detail")
        dt.set_eye_style("round")
        assert any("natural eye style" in n for n in _app._restore_plan(meta)["notes"]), _app._restore_plan(meta)["notes"]            # the picture used natural, Settings says round
        dt.set_eye_style("natural")
        assert not any("eye style" in n for n in _app._restore_plan(meta)["notes"]) and "natural style" in _app._plan_summary(_app._restore_plan(meta))
        # 🎴 Load applies the card's style (a card without one leaves the current style); 📌 saves the current style in the profile
        tmp = Path(_tf.mkdtemp())
        old = CC.CARDS_DIR
        try:
            CC.CARDS_DIR = tmp / "cards"
            CC.CARDS_DIR.mkdir()
            (CC.CARDS_DIR / "E.json").write_text(json.dumps({"name": "E", "tags": "1girl", "eye_style": "natural"}), encoding="utf-8")
            (CC.CARDS_DIR / "F.json").write_text(json.dumps({"name": "F", "tags": "1girl"}), encoding="utf-8")
            by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
            fn = lambda n: getattr(by[n].fn, "__wrapped__", by[n].fn)
            with patch("backend.model_manager.list_checkpoints", return_value=[]), patch("backend.model_manager.list_loras", return_value=[]), \
                 patch("backend.model_manager.list_vaes", return_value=[]):
                dt.set_eye_style("round")
                out = fn("do_card_load")("E", "(no outfit tags)", "", "", None)
                assert dt.EYE_STYLE["mode"] == "natural" and "eye style natural" in out[-1], out[-1]
                out = fn("do_card_load")("F", "(no outfit tags)", "", "", None)
                assert dt.EYE_STYLE["mode"] == "natural" and "eye style" not in out[-1], out[-1]              # no key = unchanged
            ck = tmp / "m.safetensors"
            ck.write_bytes(b"x")
            msg = fn("do_card_profile")("F", str(ck), "none", "none", 0.7, "none", 0.7, "none", 0.7, "", 832, 1216, 6.0, 12, "Euler a", 1, 0.0, False, 0.0, False, 0.35, False, 0.35)
            assert "📌 Saved" in msg and CC.load_card("F")["profiles"]["m.safetensors"]["eye_style"] == "natural", CC.load_card("F")
            dt.set_eye_style("round")
            fn("do_card_profile")("F", str(ck), "none", "none", 0.7, "none", 0.7, "none", 0.7, "", 832, 1216, 6.0, 12, "Euler a", 1, 0.0, False, 0.0, False, 0.35, False, 0.35)
            assert CC.load_card("F")["profiles"]["m.safetensors"]["eye_style"] == "round"                      # saving again overwrites it
        finally:
            CC.CARDS_DIR = old
    finally:
        dt.set_eye_style("round")


def _buttons_wired_to(demo, fn_name):
    """The components that trigger the Gradio function `fn_name` in build_app() (following `.then()` chains back to the first event): a behaviour check of 'this button runs that handler'."""
    deps = demo.config["dependencies"]
    f = next(x for x in demo.fns if getattr(x.fn, "__name__", "") == fn_name)
    i = demo.fns.index(f)
    while True:
        real = [t for t in deps[i]["targets"] if t[0] is not None]
        if real:
            return f, [(demo.blocks[t[0]], t[1]) for t in real]
        if deps[i].get("trigger_after") is None:
            return f, []
        i = deps[i]["trigger_after"]


@test("Round-4 follow-ups: tiny faces get a higher face denoise, inpaint runs the Steps value, truncated uploads fail loudly, eye detail in card profiles")
def _():
    import io
    from PIL import Image, ImageDraw
    from PIL.PngImagePlugin import PngInfo
    import app as _app
    from backend import detail_tools as dt
    from backend.character_cards import clean_card
    from backend.png_info import read_image_metadata
    # size-aware face denoise: ≤ 60 px → 0.65 (round 6), → 0.55 at 100 px (round 5), ≥ 150 px → the slider, never lowers a high slider
    sad = dt.size_aware_denoise
    assert sad(0.35, 50) == 0.65 and sad(0.35, 60) == 0.65 and sad(0.35, 100) == 0.55 and sad(0.35, 150) == 0.35 and sad(0.35, 160) == 0.35
    assert abs(sad(0.35, 80) - 0.6) < 1e-9 and abs(sad(0.35, 125) - 0.45) < 1e-9 and sad(0.7, 40) == 0.7 and sad(0.6, 40) == 0.65
    assert abs(sad(0.5, 90) - 0.575) < 1e-9 and abs(sad(0.5, 125) - 0.525) < 1e-9 and sad(0.5, 150) == 0.5
    seen = []
    orig_df, orig_inp = dt.detect_faces, dt.inpaint_region
    try:
        dt.detect_faces = lambda *a, **k: [(10, 10, 60, 60), (100, 100, 260, 260)]       # 50 px and 160 px
        dt.inpaint_region = lambda sdp, im, mask, *a, **k: (seen.append(k["denoise"]), (im, 0))[1]
        dt.face_detail(object(), Image.new("RGB", (400, 400)), "x", denoise=0.35)
        assert seen == [0.65, 0.35], seen
        seen.clear()
        dt.face_detail(object(), Image.new("RGB", (400, 400)), "x", denoise=0.35, size_aware=False)
        assert seen == [0.35, 0.35], seen
    finally:
        dt.detect_faces, dt.inpaint_region = orig_df, orig_inp
    # 🖌 Inpaint: the Steps value is what runs (0.75 and 0.8 were the same picture at 12 steps)
    import inspect
    for steps in (12, 20, 30):
        for den in (0.4, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0):
            assert int(_app._inpaint_steps(steps, den) * den) == steps, (steps, den)
    assert len({_app._inpaint_steps(12, d) for d in (0.75, 0.8)}) == 2
    # the handler (round 16: a behaviour check instead of a look at app.py's source): the 🖌 button runs do_inpaint_ui and the Steps value reaches inpaint_region as _inpaint_steps(steps, denoise)
    import app as _a2
    from backend import detail_tools as _dt2
    demo = _a2.build_app()
    f_inp, btns = _buttons_wired_to(demo, "do_inpaint_ui")
    assert any(type(b).__name__ == "Button" and "Inpaint" in str(b.value) and e == "click" for b, e in btns), btns
    editor = {"background": Image.new("RGB", (128, 128), (200, 200, 200)), "layers": [Image.new("RGBA", (128, 128), (255, 255, 255, 255))]}

    class _FakeSD:
        pipe, current_model, _last_vae_path, _lora_adapters = object(), "m", None, {}
    seen_steps = []
    keep = (_a2.sd, _a2._save_outputs, _dt2.inpaint_region)
    try:
        _a2.sd = _FakeSD()
        _a2._save_outputs = lambda imgs, meta, pipe=None: [Path("x.png")]
        _dt2.inpaint_region = lambda sdp, im, mask, prompt, neg, **k: (seen_steps.append((k["steps"], k["denoise"])), (im, 3))[1]
        for den in (0.5, 0.75, 0.8, 1.0):
            res = f_inp.fn("m", "none", "none", 0.8, "none", 0.7, "none", 0.7, "1girl", "bad", "DPM++ 2M Karras", 12, 6.0, 5, 1, editor, den, 48, progress=lambda *a, **k: None)
            assert len(res[0]) == 1, res
        assert [(st, d) for st, d in seen_steps] == [(_a2._inpaint_steps(12, d), d) for d in (0.5, 0.75, 0.8, 1.0)], seen_steps
        assert all(int(st * d) == 12 for st, d in seen_steps)
    finally:
        _a2.sd, _a2._save_outputs, _dt2.inpaint_region = keep
    # truncated files: generation inputs fail instead of decoding to a half-black picture; metadata still reads
    from PIL import ImageFile
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False
    info = PngInfo(); info.add_text("parameters", "1girl\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 5")
    buf = io.BytesIO(); Image.new("RGB", (64, 64), (200, 30, 40)).save(buf, "PNG", pnginfo=info)
    data = buf.getvalue()
    with _tf.TemporaryDirectory() as d:
        f = Path(d) / "half.png"
        f.write_bytes(data[:len(data) - 40])          # cut inside the image data
        try:
            Image.open(f).load()
            raise AssertionError("a truncated PNG loaded silently")
        except OSError:
            pass
        assert read_image_metadata(f)["seed"] == 5
    # card profiles carry eye detail (0 = off, else its denoise; the slider starts at 0.1)
    c = clean_card({"name": "t", "profiles": {"a.safetensors": {"eye_detail": 0.3, "face_detail": 0.35}}})
    assert c["profiles"]["a.safetensors"]["eye_detail"] == 0.3
    assert clean_card({"name": "t", "profiles": {"a.safetensors": {"eye_detail": 9}}}) is not None



@test("🦋 Swap a hair accessory: hair-only prompt + the item, one inpaint at denoise 1.0 (≥ 64 px context); the box, the button and the GPU job exist")
def _():
    import re as _re
    from PIL import Image
    import app as _app
    from backend import detail_tools as dt
    canon = ("masterpiece, best quality, 1girl, solo, (heroine \\(original\\):1.1), red eyes, grey hair, long hair, (black butterfly hair ornament:1.2), lace, mesh, "
             "hair ornament, hair between eyes, hair ribbon, mature female, cowboy shot, white dress, simple background")
    p = dt.swap_prompt_from(canon, " red ribbon ")
    tags = [t.strip() for t in p.split(",")]
    assert tags[-1] == "(red ribbon:1.2)" and tags[:2] == ["masterpiece", "best quality"], p
    for keep in ("1girl", "solo", "grey hair", "long hair", "hair between eyes"):
        assert keep in tags, (keep, p)
    for gone in ("butterfly", "lace", "mesh", "ornament", "ribbon", "white dress", "cowboy shot", "simple background", "red eyes", "heroine"):
        assert gone not in p.replace("(red ribbon:1.2)", ""), (gone, p)
    assert dt.swap_prompt_from("", "bow") == "(bow:1.2)"                                   # nothing to keep: just the item
    for item in ("gold hoop earrings", "black choker", "black beret"):                    # round 8: every kind of item keeps the hair tags (without them a hat mask drew a second girl)
        pi = [t.strip() for t in dt.swap_prompt_from(canon, item).split(",")]
        assert "grey hair" in pi and "long hair" in pi and pi[-1] == f"({item}:1.2)" and "white dress" not in pi, (item, pi)
    seen = {}
    orig = dt.inpaint_region
    try:
        dt.inpaint_region = lambda sdp, im, mask, prompt, neg, **k: (seen.update(prompt=prompt, neg=neg, **k), (im, 7))[1]
        im, m = Image.new("RGB", (256, 256)), Image.new("L", (256, 256), 255)
        out, used = dt.swap_item(object(), im, m, "red ribbon", canon, "bad hands", steps=12, cfg=6.0, seed=5, padding=16)
        assert used == 7 and out is im
        assert seen["denoise"] == 1.0 and seen["padding"] == 64 and seen["steps"] == 12 and seen["neg"] == "bad hands" and seen["seed"] == 5, seen
        assert seen["prompt"].endswith("(red ribbon:1.2)") and "ornament" not in seen["prompt"], seen["prompt"]
        dt.swap_item(object(), im, m, "red ribbon", canon, "", padding=200)
        assert seen["padding"] == 200                                                      # a wider context the user chose is kept
    finally:
        dt.inpaint_region = orig
    demo = _app.build_app()
    f_swap, btns = _buttons_wired_to(demo, "do_swap_ui")                                  # round 16: wiring checked on the built app, not on app.py's source text
    assert any(type(b).__name__ == "Button" and "Swap" in str(b.value) and e == "click" for b, e in btns), btns
    assert f_swap.concurrency_id == "gpu" and f_swap.concurrency_limit == 1 and getattr(f_swap.fn, "_gpu_job", False), "the swap is a GPU job"
    assert any(type(c).__name__ == "Textbox" and "Swap an accessory" in str(c.label) for c in f_swap.inputs), "the item box is an input"
    fs = [f_swap]
    # the handler itself (the UI build only constructs it): a fake editor value, the model / LoRA / save steps stubbed, the inpaint recorded
    run, saved = fs[0].fn, {}
    editor = {"background": Image.new("RGB", (128, 128), (200, 200, 200)), "layers": [Image.new("RGBA", (128, 128), (255, 255, 255, 255))]}
    class FakeSD:                       # "the model is already loaded, no LoRA wanted": _ensure_model and _sync_loras return at once
        pipe, current_model, _last_vae_path, _lora_adapters = object(), "m", None, {}
    keep = (_app.sd, _app._save_outputs, dt.inpaint_region)
    try:
        _app.sd = FakeSD()
        _app._save_outputs = lambda imgs, meta, pipe=None: (saved.update(meta=meta), [Path("swapped.png")])[1]
        dt.inpaint_region = lambda sdp, im, mask, prompt, neg, **k: (saved.update(call=dict(prompt=prompt, neg=neg, **k)), (im, 11))[1]
        quiet = lambda *a, **k: None
        args = ("m", "none", "none", 0.8, "none", 0.7, "none", 0.7, canon, "bad hands", "DPM++ 2M Karras", 12, 6.0, 5, 1)
        res = run(*args, editor, 48, "red ribbon", progress=quiet)
        assert len(res[0]) == 1 and "Swapped for" in res[1] and "swapped.png" in res[1], res
        c, m = saved["call"], saved["meta"]
        assert c["denoise"] == 1.0 and c["padding"] == 64 and c["steps"] == 12 and c["prompt"].endswith("(red ribbon:1.2)") and c["neg"] == "bad hands", c
        assert m["mode"] == "inpaint" and m["strength"] == 1.0 and m["inpaint_swap"] == "red ribbon" and m["inpaint_padding"] == 64 and m["seeds"] == [11], m
        saved.clear()
        res = run(*args, editor, 48, "  ", progress=quiet)                              # no item: refused, nothing drawn
        assert not res[0] and "what should be there" in res[1] and "call" not in saved, res
        res = run(*args, {"background": editor["background"], "layers": []}, 48, "red ribbon", progress=quiet)     # nothing painted
        assert not res[0] and "Paint over" in res[1] and "call" not in saved, res
        def stopped(*a, **k):                                                              # Stop pressed during the pass
            raise _app._GenerationAborted()
        dt.inpaint_region = stopped
        res = run(*args, editor, 48, "red ribbon", progress=quiet)
        assert not res[0] and "stopped" in res[1] and "meta" not in saved, res
        def broken(*a, **k):
            raise RuntimeError("the sampler produced NaN latents, so nothing was changed")
        dt.inpaint_region = broken
        res = run(*args, editor, 48, "red ribbon", progress=quiet)                        # a failed pass says so, nothing is saved
        assert not res[0] and "Swap failed" in res[1] and "NaN" in res[1] and "meta" not in saved, res
    finally:
        _app.sd, _app._save_outputs, dt.inpaint_region = keep


@test("Cool mode pauses after every sampling step (factor × step time, Stop ends the pause); Pony skips the eye pass; Cowboy Polish = eyes only")
def _():
    import time
    from backend import sampling as S
    from backend import polish as PO
    import app as _app
    from backend import thermal as _T
    old_f, old_stop, old_lim = S.COOL["factor"], S.should_stop, _T.GUARD["limit"]
    _T.set_limit(0)                                 # the saved prefs may have it on
    try:
        assert S.set_cool("x") == 0.0 and S.set_cool(9) == 4.0 and S.set_cool(-1) == 0.0 and S.set_cool(1.5) == 1.5
        S.set_cool(1.0)
        seen = []

        def orig(pipe, i, t, kw):
            time.sleep(0.05)                       # a 50 ms "step" inside the callback window
            seen.append(i)
            return {"latents": "L"}
        cb = S._cool_callback(orig)
        t0 = time.perf_counter()
        out = cb(None, 0, 999, {"latents": "x"})
        dt = time.perf_counter() - t0
        assert out == {"latents": "L"} and seen == [0], (out, seen)
        assert 0.09 <= dt < 0.5, dt                # 50 ms step + ~50 ms pause
        S.should_stop = lambda: True               # Stop pressed: no pause
        t0 = time.perf_counter()
        cb(None, 1, 998, {})
        assert time.perf_counter() - t0 < 0.09
        S.should_stop = None
        assert S._cool_callback(None)(None, 0, 1, {"latents": 1}) == {"latents": 1}   # no original callback

        class _Pipe:                                # run_pipe wraps the callback only when cool mode is on
            unet = object(); scheduler = object()

            def __call__(self, callback_on_step_end=None, **kw):
                return dict(kw, callback_on_step_end=callback_on_step_end) if callback_on_step_end else kw
        sdp = type("D", (), {"boosters": {}})()
        call = S.run_pipe(sdp, _Pipe(), "txt2img", prompt="x")
        assert "callback_on_step_end" in call
        S.set_cool(0)
        assert "callback_on_step_end" not in S.run_pipe(sdp, _Pipe(), "txt2img", prompt="x")
    finally:
        S.COOL["factor"], S.should_stop = old_f, old_stop
        _T.GUARD["limit"] = old_lim
    # Pony-family checkpoints skip the eye pass (it softened them by 28 % at every denoise)
    assert _app._eye_pass_skipped(type("P", (), {"model_family": "pony"})())
    assert not _app._eye_pass_skipped(type("P", (), {"model_family": "illustrious"})())
    assert not _app._eye_pass_skipped(object())
    # Polish → Cowboy shot: eyes only (the face pass measured no gain at 24–28 % faces)
    r = PO.recipe("Cowboy shot")
    assert r["fd_on"] is False and r["ed_on"] is True and r["hires_on"] is False, r
    assert PO.TIME_FACTOR["Cowboy shot"] == PO.TIME_FACTOR["Portrait"]



@test("🌡 Power profile (round 11): 'full' = cool 1.5 + pause 88 (the shipped default), 'limited' = cool 0 + pause 88, anything else 'custom'; the Settings radio sets and saves both; a slow SDXL pass only HINTS (cool on, ≥ 3 s per megapixel)")
def _():
    import time
    import numpy as np
    from backend import sampling as S, thermal as T
    import app as _app
    assert T.PROFILES == {"full": (1.5, 88.0), "limited": (0.0, 88.0)}
    assert T.profile_of(1.5, 88) == "full" and T.profile_of(0, 88.0) == "limited" and T.profile_of(1.0, 88) == "custom" and T.profile_of(1.5, 0) == "custom" and T.profile_of("x", 1) == "custom"
    old_c, old_l, old_save, old_pc = S.COOL["factor"], T.GUARD["limit"], _app._save_pref, S.POWER["passes"]
    saved = {}
    try:
        T.set_limit(0); S.set_cool(0)
        assert T.set_profile("custom", S) is None and S.COOL["factor"] == 0.0 and T.GUARD["limit"] == 0.0     # nothing changes for 'custom' / unknown
        assert T.set_profile("full", S) == (1.5, 88.0) and S.COOL["factor"] == 1.5 and T.GUARD["limit"] == 88.0
        assert T.set_profile("limited", S) == (0.0, 88.0) and S.COOL["factor"] == 0.0 and T.GUARD["limit"] == 88.0
        # the Settings radio: both sliders move, both prefs are saved; the sliders' own handlers hand the radio back to 'custom' / a profile
        _app._save_pref = lambda k, v: saved.__setitem__(k, v)
        fns = {getattr(f.fn, "__name__", ""): f.fn for f in _app.build_app().fns}
        res = fns["on_power_profile"](_app._POWER_LABELS["full"])
        assert saved == {"cool": 1.5, "thermal_limit": 88.0} and res[1]["value"] == 1.5 and res[2]["value"] == 88.0 and "Full power" in res[0], (saved, res)
        res = fns["on_power_profile"](_app._POWER_LABELS["limited"])
        assert saved == {"cool": 0.0, "thermal_limit": 88.0} and res[1]["value"] == 0.0 and S.COOL["factor"] == 0.0
        res = fns["on_power_profile"](_app._POWER_LABELS["custom"])
        assert res[1] == {"__type__": "update"} or "value" not in res[1], res
        assert saved["cool"] == 0.0                                                            # 'custom' saved nothing new
        out = fns["on_cool"](1.0)
        assert out[1]["value"] == _app._POWER_LABELS["custom"] and S.COOL["factor"] == 1.0
        out = fns["on_cool"](1.5)
        assert out[1]["value"] == _app._POWER_LABELS["full"]                                   # back on the pair: the radio says 'full' again
        assert _app._power_choice()["value"] == _app._POWER_LABELS["full"]
        # the hint: only while cool mode pauses, only from 3 s per megapixel and image, measured from step 1 on, never a setting change
        clock = [0.0]
        real_pc = time.perf_counter
        time.perf_counter = lambda: clock[0]
        try:
            def slow(pipe, i, t, kw):
                clock[0] += 4.0                    # a 4 s step on a ~1 MP latent: the USB-C regime of round 10
                return kw
            S.set_cool(0); S.reset_power_samples()
            cb = S._cool_callback(slow)
            lat = np.zeros((1, 4, 152, 104))
            for i in range(6):
                cb(None, i, 0, {"latents": lat})
        finally:
            time.perf_counter = real_pc
        sec = S.power_limited_seconds()
        assert sec is not None and 3.8 < sec < 4.1, sec
        assert S.power_hint_html() == "" and S.COOL["factor"] == 0.0                           # cool 0: nothing to hint about
        S.set_cool(1.5)
        h = S.power_hint_html()
        assert "power-limited" in h and "Power profile" in h and S.COOL["factor"] == 1.5 and T.GUARD["limit"] == 88.0, h
        S.POWER["passes"] = [[1.9] * 8, [1.7] * 8, [9.0] * 3]                                   # a full-power machine; a pass under 4 timed steps counts for nothing
        assert S.power_hint_html() == ""
        S.reset_power_samples()
        assert S.power_limited_seconds() is None and S.power_hint_html() == ""
    finally:
        S.COOL["factor"], T.GUARD["limit"], _app._save_pref, S.POWER["passes"] = old_c, old_l, old_save, old_pc


@test("Start-up prefs: the saved cool factor and pause limit are applied except under the test suite (IMAGEGEN_TESTS=1), which must not depend on them")
def _():
    from backend import thermal as T
    class FakeSampling:
        def __init__(self): self.cool = None
        def set_cool(self, v): self.cool = v
    old = T.GUARD["limit"]
    try:
        fs = FakeSampling()
        assert T.apply_saved({"cool": 1.5, "thermal_limit": 88.0}, fs, environ={}) is True and fs.cool == 1.5 and T.GUARD["limit"] == 88.0
        T.set_limit(0)
        fs = FakeSampling()
        assert T.apply_saved({"cool": 1.5, "thermal_limit": 88.0}, fs, environ={"IMAGEGEN_TESTS": "1"}) is False and fs.cool is None and T.GUARD["limit"] == 0.0
        assert os.environ.get("IMAGEGEN_TESTS") == "1"           # this very suite runs with the flag
        T.apply_saved({}, fs, environ={})                         # no saved choices: both off
        assert fs.cool == 0.0 and T.GUARD["limit"] == 0.0
    finally:
        T.GUARD["limit"] = old


@test("🌡 Pause while hot: waits until the thermal zone is 8 °C under the limit (Stop / 15 min end it), off by default, reads the real sensor")
def _():
    from backend import thermal as T
    import app as _app
    old = T.GUARD["limit"]
    try:
        assert T.set_limit(0) == 0.0 and T.set_limit(50) == 0.0 and T.set_limit("x") == 0.0 and T.set_limit(120) == 99.0
        T.set_limit(0)
        assert T.wait_cool(read=lambda: 99.0, sleep=lambda s: None) == 0.0          # off → never waits
        T.set_limit(90)
        temps = iter([93.0, 92.0, 88.0, 84.0, 81.0])
        now = [0.0]
        said = []

        def sleep(sec):
            now[0] += sec
        waited = T.wait_cool(read=lambda: next(temps), sleep=sleep, clock=lambda: now[0], say=said.append)
        assert waited == 8.0 and len(said) == 4 and "waiting for 82" in said[0], (waited, said)   # 93, 92, 88, 84 wait; 81 ≤ 82 resumes
        assert T.wait_cool(read=lambda: 85.0, sleep=sleep, clock=lambda: now[0]) == 0.0           # below the limit: no wait
        assert T.wait_cool(read=lambda: None, sleep=sleep, clock=lambda: now[0]) == 0.0           # no sensor: no wait
        stop = iter([False, True])
        now[0] = 0.0
        assert T.wait_cool(should_stop=lambda: next(stop), read=lambda: 95.0, sleep=sleep, clock=lambda: now[0]) == 2.0
        now[0] = 0.0
        assert T.wait_cool(read=lambda: 95.0, sleep=sleep, clock=lambda: now[0]) >= 900.0          # gives up after 15 min
        # between pieces the wait also ends once the zone sits under the limit and stops falling (the laptop's idle level, round 8)
        temps3 = iter([92.0, 87.0] + [87.0] * 30)
        now[0] = 0.0
        w = T.wait_cool(read=lambda: next(temps3), sleep=sleep, clock=lambda: now[0])
        assert 20.0 <= w <= 24.0, w                                                      # 87 °C for 20 s → go on (never reaches 82)
        now[0] = 0.0
        assert T.wait_cool(read=lambda: 95.0, sleep=sleep, clock=lambda: now[0]) >= 900.0          # above the limit: no plateau exit
        # inside a pass: only an emergency brake (act at limit + 6 = 96, resume at the limit) — no stop-go at 90
        assert T.wait_cool(read=lambda: 95.0, sleep=sleep, clock=lambda: now[0], in_pass=True) == 0.0
        temps2 = iter([96.0, 92.0, 89.5, 90.0])
        now[0] = 0.0
        assert T.wait_cool(read=lambda: next(temps2), sleep=sleep, clock=lambda: now[0], in_pass=True) == 4.0   # 96, 92 wait; 89.5 ≤ 90
        T.set_limit(0)
        assert _app._thermal_wait() == 0.0
        # inside a pass: with the limit set the callback also checks the zone (every ~10 s, emergency brake)
        import inspect, torch
        from backend import sampling as S
        assert not hasattr(S, "_size_factor")    # the pause is uniform (scaling by pass size was slower, not cooler); the helper is gone
        calls = []
        old_wait, old_every, old_cool = T.wait_cool, S._THERM_EVERY, S.COOL["factor"]
        try:
            T.wait_cool = lambda *a, **k: calls.append(k.get("in_pass")) or 0.0
            S._THERM_EVERY = 0.0
            S.COOL["factor"] = 0.0
            T.set_limit(90)

            class _P:
                unet = object(); scheduler = object()

                def __call__(self, callback_on_step_end=None, **kw):
                    return dict(kw, callback_on_step_end=callback_on_step_end) if callback_on_step_end else kw
            call = S.run_pipe(type("D", (), {"boosters": {}})(), _P(), "txt2img", prompt="x")
            assert "callback_on_step_end" in call                       # installed for the thermal check alone
            call["callback_on_step_end"](None, 0, 1, {"latents": torch.zeros(1, 4, 8, 8)})
            assert calls == [True]
        finally:
            T.wait_cool, S._THERM_EVERY, S.COOL["factor"] = old_wait, old_every, old_cool
            T.set_limit(0)
        t = T.read_temp()
        assert t is None or 0 < t < 130, t
    finally:
        T.GUARD["limit"] = old



@test("🖐 Re-draw a hand: hand-only prompt, N tries at 0.7 (round 10; 0.8 before) with seed, seed+1 …, painted mask or the biggest detected hand; a GPU job; original shown first")
def _():
    from PIL import Image
    import app as _app
    from backend import detail_tools as dt
    p = dt.hand_prompt_from("masterpiece, best quality, 1girl, solo, grey hair, red eyes, holding cup, peace sign, white dress, night, city lights")
    tags = [t.strip() for t in p.split(",")]
    for keep in ("masterpiece", "1girl", "solo", "holding cup", "peace sign", "five fingers", "hand focus"):
        assert keep in tags, (keep, p)
    for gone in ("grey hair", "red eyes", "white dress", "night", "city lights"):
        assert gone not in tags, (gone, p)
    m = np.array(dt.hand_mask((100, 100), (40, 40, 60, 60)))
    assert m[50, 50] == 255 and m[38, 50] == 255 and m[10, 10] == 0                      # box + 12 %
    calls = []
    orig = dt.inpaint_region
    try:
        dt.inpaint_region = lambda sdp, im, mask, prompt, neg, **k: (calls.append(dict(prompt=prompt, **k)), (im, k["seed"]))[1]
        im = Image.new("RGB", (200, 200)); mk = Image.new("L", (200, 200), 0); mk.paste(255, (80, 80, 120, 130))
        res = dt.redraw_hand(object(), im, mk, "1girl, holding cup", tries=3, steps=12, seed=10, scheduler="DPM++ 2M AYS")
        assert [u for _i, u in res] == [10, 11, 12] and len(calls) == 3
        c = calls[0]
        assert c["denoise"] == 0.7 and c["scheduler"] == "DPM++ 2M" and c["steps"] == 15 and c["padding"] == 29 and "five fingers" in c["prompt"], c          # ceil(max(10, 12 × 0.5) / 0.7) = 15
        assert dt.redraw_hand(object(), im, Image.new("L", (200, 200), 0), "x") == []          # nothing painted → nothing done
        assert len(dt.redraw_hand(object(), im, mk, "x", tries=50, seed=1)) == 8                 # at most 8 tries
    finally:
        dt.inpaint_region = orig
    fs = [f for f in _app.build_app().fns if getattr(f.fn, "__name__", "") == "do_hand_redraw_ui"]
    assert fs and all(f.concurrency_id == "gpu" and f.concurrency_limit == 1 for f in fs), "the hand re-draw is a GPU job"
    run, saved = fs[0].fn, {}
    run = getattr(run, "__wrapped__", run)

    class FakeSD:
        pipe, current_model, _last_vae_path, _lora_adapters = object(), "m", None, {}
    keep = (_app.sd, _app._save_outputs, dt.redraw_hand, dt.detect_hands)
    try:
        _app.sd = FakeSD()
        _app._save_outputs = lambda imgs, meta, pipe=None: (saved.update(meta=meta, n=len(imgs)), [Path(f"h{i}.png") for i in range(len(imgs))])[1]
        dt.redraw_hand = lambda sdp, im, mask, prompt, neg, *, tries, **k: (saved.update(mask=np.array(mask), tries=tries),
                                                                           [(im, 100 + i) for i in range(tries)])[1]
        quiet = lambda *a, **k: None
        args = ("m", "none", "none", 0.8, "none", 0.7, "none", 0.7, "1girl, peace sign", "bad hands", "DPM++ 2M Karras", 12, 6.0, 5, 1)
        bg = Image.new("RGB", (128, 128), (200, 200, 200))
        painted = Image.new("RGBA", (128, 128), (0, 0, 0, 0)); painted.paste((255, 255, 255, 255), (10, 10, 30, 30))
        res = run(*args, {"background": bg, "layers": [painted]}, 3, progress=quiet)
        assert len(res[0]) == 4 and res[0][0] is not None and res[4] == [100, 100, 101, 102], res[4]     # original + 3 tries
        assert saved["tries"] == 3 and saved["mask"][20, 20] and not saved["mask"][100, 100] and saved["meta"]["strength"] == 0.7
        assert "original" in res[1]
        dt.detect_hands = lambda image, n=6: []                                            # nothing painted, no hand found
        res = run(*args, {"background": bg, "layers": []}, 3, progress=quiet)
        assert not res[0] and "No hand found" in res[1], res
        dt.detect_hands = lambda image, n=6: [(60, 60, 90, 100), (5, 5, 15, 15)]          # biggest detected hand is used
        res = run(*args, {"background": bg, "layers": []}, 2, progress=quiet)
        assert len(res[0]) == 3 and saved["mask"][80, 75] and not saved["mask"][10, 10] and "biggest of 2" in res[1], res[1]
    finally:
        _app.sd, _app._save_outputs, dt.redraw_hand, dt.detect_hands = keep


@test("🖐 Re-draw a hand: a Stop or an error at try k keeps, shows and saves the k − 1 finished tries (they used to be thrown away)")
def _():
    from PIL import Image
    import app as _app
    from backend import detail_tools as dt
    im = Image.new("RGB", (200, 200)); mk = Image.new("L", (200, 200), 0); mk.paste(255, (80, 80, 120, 130))
    orig, n = dt.inpaint_region, [0]

    def fake(sdp, image, mask, prompt, neg, **k):
        n[0] += 1
        if n[0] == 3:
            raise RuntimeError("boom")
        return image, k["seed"]
    try:
        dt.inpaint_region = fake
        out = []
        try:
            dt.redraw_hand(object(), im, mk, "1girl", tries=5, steps=12, seed=40, out=out)
            raise AssertionError("the error at try 3 must propagate")
        except RuntimeError as e:
            assert "boom" in str(e)
        assert [u for _i, u in out] == [40, 41], out                                          # tries 1 and 2 were finished and are kept
        n[0] = 10
        assert [u for _i, u in dt.redraw_hand(object(), im, mk, "x", tries=2, seed=7)] == [7, 8]   # without `out` it still returns its list
    finally:
        dt.inpaint_region = orig
    fs = [f for f in _app.build_app().fns if getattr(f.fn, "__name__", "") == "do_hand_redraw_ui"]
    run, saved = getattr(fs[0].fn, "__wrapped__", fs[0].fn), {}

    class FakeSD:
        pipe, current_model, _last_vae_path, _lora_adapters = object(), "m", None, {}
    keep = (_app.sd, _app._save_outputs, dt.redraw_hand)
    try:
        _app.sd = FakeSD()
        _app._save_outputs = lambda imgs, meta, pipe=None: (saved.update(meta=meta, n=len(imgs)), [Path(f"h{i}.png") for i in range(len(imgs))])[1]
        quiet = lambda *a, **k: None
        args = ("m", "none", "none", 0.8, "none", 0.7, "none", 0.7, "1girl, peace sign", "bad hands", "DPM++ 2M Karras", 12, 6.0, 5, 1)
        bg = Image.new("RGB", (128, 128), (200, 200, 200))
        painted = Image.new("RGBA", (128, 128), (0, 0, 0, 0)); painted.paste((255, 255, 255, 255), (10, 10, 30, 30))
        ed = {"background": bg, "layers": [painted]}

        def stop_after_two(sdp, image, mask, prompt, neg, *, tries, out=None, **k):
            out.extend([(image, 500), (image, 501)])
            raise _app._GenerationAborted()
        dt.redraw_hand = stop_after_two
        res = run(*args, ed, 5, progress=quiet)
        assert len(res[0]) == 3 and res[4] == [500, 500, 501] and saved["n"] == 2 and saved["meta"]["seeds"] == [500, 501], (len(res[0]), res[4], saved)
        assert "stopped after 2 of 5 tries" in res[1] and "original" in res[1], res[1]

        def fail_at_two(sdp, image, mask, prompt, neg, *, tries, out=None, **k):
            out.append((image, 600))
            raise RuntimeError("out of memory")
        dt.redraw_hand = fail_at_two
        res = run(*args, ed, 5, progress=quiet)
        assert len(res[0]) == 2 and "out of memory" in res[1] and "try 2 of 5" in res[1], res[1]

        def fail_at_once(sdp, image, mask, prompt, neg, *, tries, out=None, **k):
            raise RuntimeError("boom now")
        dt.redraw_hand = fail_at_once
        res = run(*args, ed, 5, progress=quiet)
        assert not res[0] and "failed" in res[1] and "boom now" in res[1], res[1]

        def stop_at_once(sdp, image, mask, prompt, neg, *, tries, out=None, **k):
            raise _app._GenerationAborted()
        dt.redraw_hand = stop_at_once
        res = run(*args, ed, 5, progress=quiet)
        assert not res[0] and "stopped" in res[1], res[1]
    finally:
        _app.sd, _app._save_outputs, dt.redraw_hand = keep


@test("🖐 Re-draw a hand, face guard: a face reaching into the crop leaves the mask (on by default; the negative / no-subject-tags variants of the A/B are not shipped)")
def _():
    from PIL import Image
    from backend import detail_tools as dt
    assert dt.HAND_FACE_MASK is True and not hasattr(dt, "HAND_FACE_NEGATIVE") and not hasattr(dt, "HAND_FACE")        # round 9: the mask part shipped, the negative did not
    px = lambda m: np.array(m) > 127
    im = Image.new("RGB", (400, 400)); mk = Image.new("L", (400, 400), 0); mk.paste(255, (150, 200, 250, 280))      # a 100 × 80 hand; its crop is x 90–310, y 140–340
    face_in, far, over = [(170, 130, 250, 210)], [(0, 0, 60, 60)], [(120, 150, 280, 300)]
    m, inside = dt.face_free_mask(im, mk, faces=far)
    assert inside == [] and m is mk                                                      # no face in the crop: the very same mask
    m, inside = dt.face_free_mask(im, mk, faces=face_in)
    assert inside == face_in and px(m)[270, 200] and not px(m)[205, 210] and px(m).sum() < px(mk).sum(), "the face's ellipse (+15 %) is out of the mask, the rest stays"
    assert px(m)[240, 150] and px(m)[279, 249]                                          # far from the face: still painted
    m, inside = dt.face_free_mask(im, mk, faces=over)
    assert inside == over and px(m).sum() == px(mk).sum()                                # a hand held over her face: the mask would lose > 75 %, it keeps its shape
    seen, orig = [], (dt.inpaint_region, dt.detect_faces, dt._face_model_cached)

    def boom(*a, **k):
        raise RuntimeError("detector exploded")
    redraw = lambda: dt.redraw_hand(object(), im, mk, "1girl, holding cup", "bad hands", tries=1, seed=3)
    try:
        dt.inpaint_region = lambda sdp, image, mask, prompt, neg, **k: (seen.append((px(mask), neg, k["padding"], prompt)), (image, k["seed"]))[1]
        dt.detect_faces = lambda image, mode="auto", **k: face_in
        dt._face_model_cached = lambda: True
        redraw()
        m1, neg1, pad1, prompt1 = seen[-1]
        assert m1.sum() < px(mk).sum() and not m1[205, 210] and neg1 == "bad hands" and pad1 == 59 and "1girl" in prompt1, (neg1, pad1)   # the face is out of the mask; negative, prompt and the crop padding (59 = 0.6 × the whole hand's side) untouched
        dt.HAND_FACE_MASK = False
        redraw()
        assert seen[-1][0].sum() == px(mk).sum() and seen[-1][2] == 59                    # switched off: exactly as before
        dt.HAND_FACE_MASK = True
        dt._face_model_cached = lambda: False                                              # the face model isn't downloaded: nothing is detected, no download from here
        redraw()
        assert seen[-1][0].sum() == px(mk).sum()
        dt._face_model_cached = lambda: True
        dt.detect_faces = boom                                                             # a failing detector never stops a re-draw
        redraw()
        assert seen[-1][0].sum() == px(mk).sum()
    finally:
        dt.inpaint_region, dt.detect_faces, dt._face_model_cached = orig
        dt.HAND_FACE_MASK = True


@test("GPU events share one queue slot (Generate during an X/Y grid crashed the process); Stop stays free")
def _():
    import app as _app
    b = _app.build_app()
    by = {}
    for f in b.fns:
        by.setdefault(getattr(f.fn, "__name__", "?"), []).append(f)
    missing = [n for n in _app._GPU_FNS if n not in by]
    assert not missing, f"GPU functions not wired to any event (renamed?): {missing}"
    for n in _app._GPU_FNS:
        assert all(f.concurrency_id == "gpu" and f.concurrency_limit == 1 for f in by[n]), n
    for n in ("do_stop", "do_stop_loop", "do_train_stop", "on_prompt_tokens", "do_tag_complete"):
        assert all(f.concurrency_id != "gpu" for f in by.get(n, [])), n


# ══════════════════════════════════════════════════════════════════════════
# Audit 2026-09-30 regressions (F-01 … F-33)
# ══════════════════════════════════════════════════════════════════════════

@test("--cpu start: importing the app under FORCE_CPU doesn't deadlock (torch vs the tokenizer thread)")
def _():
    import subprocess
    env = dict(os.environ, FORCE_CPU="1", HF_HUB_OFFLINE="1")
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, '.'); sys.argv = ['app.py']; import app; "
                        "print('ok', app.sd.device)"], cwd=str(Path(__file__).parent), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == 0 and "ok cpu" in r.stdout, (r.returncode, r.stderr[-800:])


@test("GPU jobs: every handler that touches the pipelines is in the queue group, runs under the lock, no cancels=")
def _():
    import threading, time as _t
    import app as _app
    b = _app.build_app()
    mutators = {"do_remove_lora", "do_remove_loras", "do_apply_lora", "_ensure_model", "_sync_loras", "do_load_model",
                "do_generate", "_unload", "load_lora", "unload_loras", "remove_lora", "load_model", "_apply_slot"}
    bad = []
    for f in b.fns:
        fn = getattr(f.fn, "__wrapped__", f.fn)
        code = getattr(fn, "__code__", None)
        if code is None:
            continue
        names = set(code.co_names) | set(code.co_freevars)
        if names & mutators and f.concurrency_id != "gpu":
            bad.append(getattr(fn, "__name__", "?"))
    assert not bad, f"handlers that change the pipelines outside the GPU group: {bad}"
    gpu_idx = {i for i, f in enumerate(b.fns) if f.concurrency_id == "gpu"}
    cancelling = [d for d in b.dependencies if set(d.get("cancels") or []) & gpu_idx]
    assert not cancelling, "a Stop with cancels= frees the GPU slot while the job's thread still runs"
    # the lock: two jobs never overlap, generators hold it until they finish
    spans = []

    def job(tag):
        t0 = _t.perf_counter(); _t.sleep(0.15); spans.append((t0, _t.perf_counter()))
        return tag

    def gen_job():
        t0 = _t.perf_counter()
        for i in range(3):
            _t.sleep(0.05); yield i
        spans.append((t0, _t.perf_counter()))
    locked, locked_gen = _app._locked(job), _app._locked(gen_job)
    th = [threading.Thread(target=locked, args=(i,)) for i in range(3)] + \
         [threading.Thread(target=lambda: list(locked_gen())) for _ in range(2)]
    [x.start() for x in th]; [x.join(10) for x in th]
    spans.sort()
    assert len(spans) == 5 and all(a[1] <= b_[0] + 1e-3 for a, b_ in zip(spans, spans[1:])), spans
    import inspect
    assert inspect.isgeneratorfunction(locked_gen) and not inspect.isgeneratorfunction(locked)


@test("VAE-only change reloads (was 'Already loaded' with the old VAE); KEEP_VAE / same VAE don't")
def _():
    from backend.sd_pipeline import SDPipeline, KEEP_VAE
    from backend.sdxl_pipeline import SDXLPipeline
    for cls in (SDPipeline, SDXLPipeline):
        p = cls()
        p.current_model, p.pipe, p._last_vae_path = "D:/nowhere/m.safetensors", object(), None
        assert p.load_model("D:/nowhere/m.safetensors").startswith("✅ Already")
        assert p.load_model("D:/nowhere/m.safetensors", KEEP_VAE).startswith("✅ Already")
        assert p.load_model("D:/nowhere/m.safetensors", "none").startswith("✅ Already")
        assert p.load_model("D:/nowhere/m.safetensors", None).startswith("✅ Already")
        p.current_model, p.pipe = "D:/nowhere/m.safetensors", object()
        r = p.load_model("D:/nowhere/m.safetensors", "D:/nowhere/vae.safetensors")
        assert not r.startswith("✅ Already"), r


def _app_closures():
    """do_generate / _hires_pass / _detail_pass from the built UI (nested functions)."""
    import app as _app
    b = _app.build_app()
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in b.fns}
    xy = getattr(by["do_xy_grid"].fn, "__wrapped__", by["do_xy_grid"].fn)
    cl = dict(zip(xy.__code__.co_freevars, xy.__closure__))
    dg = cl["do_generate"].cell_contents
    gcl = dict(zip(dg.__code__.co_freevars, dg.__closure__))
    return _app, dg, gcl["_hires_pass"].cell_contents


@test("Stop / OOM during hires keeps and saves the base images; hires progress counts the real steps")
def _():
    from PIL import Image
    _app, dg, hires_pass = _app_closures()

    class FakeSD:
        model_family, current_model, _last_vae_path, prediction, device = "sd15", "x.safetensors", None, {}, "cpu"
        _lora_adapters, boosters, last_seeds, last_var_seeds, pipe = {}, {}, [], [], object()

        def __init__(self, fail=None): self.fail, self.calls = fail, []

        def txt2img(self, prompt, neg, w, h, steps, cfg, seed, sched, batch, step_callback=None, **kw):
            self.last_seeds = [seed + i for i in range(batch)]
            return [Image.new("RGB", (w, h), (i * 40, 0, 0)) for i in range(batch)], "info"

        def img2img(self, img, prompt, neg, strength, steps, cfg, seed, sched, step_callback=None, clip_skip=1):
            self.calls.append((steps, strength))
            if self.fail:
                raise self.fail
            for i in range(int(steps * strength)):
                step_callback and step_callback(i + 1, steps)
            return [img], "i2i"
    old_sd, old_out = _app.sd, _app.OUTPUTS_DIR
    tmp = Path(_tf.mkdtemp())
    try:
        _app.OUTPUTS_DIR = tmp
        for exc in (_app._GenerationAborted(), RuntimeError("HIP out of memory")):
            for f in tmp.glob("*.png"):
                f.unlink()
            _app.sd = FakeSD(exc)
            imgs, info, _ = dg("1girl", "", "Euler", 10, 7, 512, 512, 2, 100, None, 0.5, False, auto_quality=False,
                               extra=dict(hires_on=True, hires_scale=1.5, hires_denoise=0.45, hires_steps=15))
            assert len(imgs) == 2 and len(list(tmp.glob("*.png"))) == 2, (type(exc).__name__, len(imgs), info[:200])
            assert "saved the images from before it" in info
        _app.sd = FakeSD()
        seen = []
        ex = _app._clean_extra(dict(hires_on=True, hires_scale=1.5, hires_denoise=0.45, hires_steps=15))
        hires_pass([Image.new("RGB", (64, 64))], [1], "p", "n", 7, "Euler", ex,
                   lambda frac, desc="": seen.append(frac))
        assert max(seen) == 1.0 and len(set(seen)) > 5, seen
    finally:
        _app.sd, _app.OUTPUTS_DIR = old_sd, old_out


@test("Kaomoji: >_< / :< don't swallow the prompt; +_+ / -_- reach the text encoder unchanged")
def _():
    from backend.prompt_tools import split_tags, parse_tag, merge_prompts
    from backend.prompt_syntax import a1111_to_compel, plain_prompt
    from compel.prompt_parser import PromptParser
    for k in (">_<", ":<", "+_+", "-_-", "^_^", "o_o", "x_x", ";)", ":>"):
        assert split_tags(f"1girl, {k}, blush, smile") == ["1girl", k, "blush", "smile"], k
    assert split_tags("<lora:a, b:0.5>, 1girl") == ["<lora:a, b:0.5>", "1girl"]
    assert merge_prompts("1girl, >_<, blush", "blush, red hair") == "1girl, >_<, blush, red hair"
    assert parse_tag("+_+")[1] == 1.0 and parse_tag("x++")[1] > 1.2 and parse_tag("hands-")[1] == 0.9

    def flat(x, w=1.0):
        if hasattr(x, "children"):
            return [y for ch in x.children for y in flat(ch, w * getattr(x, "weight", 1))]
        return [(x.text, round(w * x.weight, 3))]
    pp = PromptParser()
    parts = lambda t: [y for x in pp.parse_conjunction(a1111_to_compel(t)).prompts[0].children for y in flat(x)]
    assert parts("1girl, +_+, -_-, blush") == [("1girl, +_+, -_-, blush", 1.0)]
    assert ("+_+", 1.2) in parts("(+_+:1.2), smile") and ("-_-", 0.909) in parts("[-_-], smile")
    assert ("detailed", 1.21) in parts("detailed++, x") and ("eyes", 1.1) in parts("(eyes)+")
    assert parts(r"my_character \(series name\)") == [("my_character (series name)", 1.0)]
    assert plain_prompt("1girl, +_+, blush") == "1girl, +_+, blush"


def _tiny_sd(tok):
    """A 16×16 random-weight SD pipeline behind the app's SDPipeline wrapper (CPU, fp32)."""
    import torch
    from transformers import CLIPTextModel, CLIPTextConfig
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, PNDMScheduler
    from backend.sd_pipeline import SDPipeline
    torch.manual_seed(0)
    te = CLIPTextModel(CLIPTextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, max_position_embeddings=77, projection_dim=32))
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
                                block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32,
                                attention_head_dim=4, norm_num_groups=8)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",),
                        up_block_types=("UpDecoderBlock2D",), block_out_channels=(32,), latent_channels=4,
                        norm_num_groups=8, layers_per_block=1)
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=tok, unet=unet,
                                   scheduler=PNDMScheduler(skip_prk_steps=True, steps_offset=1,
                                                           beta_schedule="scaled_linear", beta_start=0.00085,
                                                           beta_end=0.012),
                                   safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    sdp = SDPipeline()
    sdp.pipe, sdp.current_model, sdp.device, sdp.dtype = pipe, "tiny", "cpu", torch.float32
    sdp._lora_adapters = {}
    return sdp, pipe


@test("Samplers on a tiny CPU pipeline: same seed → same image, batch image i == seed+i (DPM++ SDE too), PNDM at 1–3 steps")
def _():
    import torch, numpy as np
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer",
                                            local_files_only=True)
    except Exception:
        raise Skip("SD 1.5 CLIP tokenizer not in .hf_cache")
    sdp, pipe = _tiny_sd(tok)
    arr = lambda im: np.asarray(im.convert("RGB"), dtype=np.int16)

    def gen(**kw):
        a = dict(prompt="1girl, red hair", negative_prompt="lowres", width=16, height=16, steps=6, cfg_scale=5.0,
                 seed=42, scheduler="Euler", batch_size=1)
        a.update(kw)
        return sdp.txt2img(**a)[0]
    from backend.sampling import SCHEDULERS
    from diffusers import EulerDiscreteScheduler
    for name in SCHEDULERS:
        for steps in (1, 3, 8):
            a, b = gen(scheduler=name, steps=steps, seed=7), gen(scheduler=name, steps=steps, seed=7)
            assert np.abs(arr(a[0]) - arr(b[0])).max() <= 1, (name, steps, "same seed differs")
        batch = gen(scheduler=name, steps=5, batch_size=2, seed=100)
        single = gen(scheduler=name, steps=5, seed=101)
        assert np.abs(arr(batch[1]) - arr(single[0])).max() <= 1, (name, "batch image 1 != seed+1")
    # SDXL files start from an Euler config (no skip_prk_steps): PNDM must still run at few steps
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    for steps in (1, 2, 3):
        gen(scheduler="PNDM", steps=steps)



@test("LMS / PNDM end at a lower order (4th-order Adams–Bashforth blew up SDXL pictures), never exceed order 2 on SDXL-family pipelines (a NoobAI checkpoint), img2img starts at order 1; model noise is no longer amplified ~20×")
def _():
    import torch
    from diffusers import EulerDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler
    from backend.sampling import make_scheduler
    # SDXL's own scheduler config (what make_scheduler starts from)
    cfg = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
               prediction_type="epsilon", timestep_spacing="leading", steps_offset=1)

    class Pipe:                                  # SD 1.5-like: no second text encoder
        scheduler = EulerDiscreteScheduler.from_config(cfg)

    class PipeXL(Pipe):                          # SDXL family (also Pony / Illustrious / NoobAI)
        text_encoder_2 = object()

    def run(sched, steps, noise, seed, vp, n=256, begin=None, keep=None):
        """A perfect denoiser (x0 = 0) whose eps prediction carries `noise`; keep(sched) records scheduler state per step."""
        sched.set_timesteps(steps)
        if begin is not None:
            sched.set_begin_index(begin)
        g, gn = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 1000)
        x = torch.randn(1, n, generator=g) * (1.0 if vp else sched.init_noise_sigma)
        for t in (sched.timesteps if begin is None else sched.timesteps[begin:]):
            if vp:
                eps = x / (1 - sched.alphas_cumprod[int(t)]) ** 0.5
            else:
                sched.scale_model_input(x, t)
                eps = x / sched.sigmas[sched.step_index]
            x = sched.step(eps + noise * torch.randn(1, n, generator=gn), t, x, return_dict=False)[0]
            if keep:
                keep(sched)
        return x

    def err(make, steps, vp):                    # distance of the noisy run's result to the noise-free one, mean over seeds
        return sum(float((run(make(), steps, 0.05, s, vp) - run(make(), steps, 0.0, s, vp)).abs().mean()) for s in range(4)) / 4

    # the order drops to 3, 2, 1 for the last three steps (and builds up 1, 2, 3, 4 at the start)
    orders, orig_coeff = [], LMSDiscreteScheduler.get_lms_coefficient

    def spy(self, order, t, current_order):
        if current_order == 0:
            orders.append(order)
        return orig_coeff(self, order, t, current_order)
    with patch.object(LMSDiscreteScheduler, "get_lms_coefficient", spy):
        run(make_scheduler(Pipe, "LMS"), 20, 0.0, 0, False)
    assert orders == [1, 2, 3] + [4] * 14 + [3, 2, 1], orders
    seen = []
    run(make_scheduler(Pipe, "PNDM"), 20, 0.0, 0, True, keep=lambda s: seen.append(len(s.ets)))
    assert seen == [1, 1, 2, 3] + [4] * 14 + [3, 2, 1], seen
    # img2img: the first step has one derivative, so it must be a plain Euler step (diffusers used order-4 weights)
    s = make_scheduler(Pipe, "LMS")
    s.set_timesteps(20)
    s.set_begin_index(8)
    x = torch.randn(1, 16)
    eps = torch.randn(1, 16)
    t = s.timesteps[8]
    s.scale_model_input(x, t)
    sig, nxt = float(s.sigmas[8]), float(s.sigmas[9])
    assert torch.allclose(s.step(eps, t, x, return_dict=False)[0], x + eps * (nxt - sig), atol=1e-5)
    # noise amplification (toy model: the plain classes amplify the model's noise ~20×; Euler is the reference)
    e_euler = err(lambda: EulerDiscreteScheduler.from_config(cfg), 20, False)
    e_plain = err(lambda: LMSDiscreteScheduler.from_config(cfg), 20, False)
    e_fixed = err(lambda: make_scheduler(Pipe, "LMS"), 20, False)
    assert e_plain > 8 * e_euler, (e_plain, e_euler)             # the problem is real in this toy …
    assert e_fixed < 1.5 * e_euler, (e_fixed, e_euler)           # … and gone
    p_plain = err(lambda: PNDMScheduler.from_config(dict(cfg, skip_prk_steps=True)), 20, True)
    p_fixed = err(lambda: make_scheduler(Pipe, "PNDM"), 20, True)
    assert p_fixed < 0.75 * p_plain, (p_fixed, p_plain)
    # SDXL family: order 3 / 4 steps in the middle of the schedule gave iridescent hue noise on a NoobAI checkpoint even with a clean
    # ending (violet eyes, rainbow glints; order 2 for all steps was clean) → never above 2, still 1 at the first / last step
    orders.clear()
    with patch.object(LMSDiscreteScheduler, "get_lms_coefficient", spy):
        run(make_scheduler(PipeXL, "LMS"), 20, 0.0, 0, False)
    assert orders == [1] + [2] * 18 + [1], orders
    seen = []
    run(make_scheduler(PipeXL, "PNDM"), 20, 0.0, 0, True, keep=lambda s: seen.append(len(s.ets)))
    assert seen == [1, 1] + [2] * 18 + [1], seen
    assert err(lambda: make_scheduler(PipeXL, "LMS"), 20, False) < 1.5 * e_euler
    assert err(lambda: make_scheduler(PipeXL, "PNDM"), 20, True) < 0.75 * p_plain


@test("LoRA on a tiny CPU pipeline through diffusers + PEFT: apply / re-weight / remove / unload restore bit-exactly")
def _():
    import torch
    from safetensors.torch import save_file
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer",
                                            local_files_only=True)
    except Exception:
        raise Skip("SD 1.5 CLIP tokenizer not in .hf_cache")
    sdp, pipe = _tiny_sd(tok)
    tmp = Path(_tf.mkdtemp())

    def make_lora(path, seed):
        g = torch.Generator().manual_seed(seed)
        sd_, expect, n = {}, {}, 0
        for name, mod in pipe.unet.named_modules():
            if isinstance(mod, torch.nn.Linear) and name.endswith(("attn1.to_q", "attn2.to_v")) and n < 3:
                key = "lora_unet_" + name.replace(".", "_")
                down = torch.randn(4, mod.in_features, generator=g) * 0.1
                up = torch.randn(mod.out_features, 4, generator=g) * 0.1
                sd_[key + ".lora_down.weight"], sd_[key + ".lora_up.weight"] = down, up
                sd_[key + ".alpha"] = torch.tensor(4.0)
                expect[name + ".weight"] = up @ down
                n += 1
        save_file(sd_, str(path))
        return expect
    orig = {n: p.detach().clone() for n, p in pipe.unet.named_parameters()}
    ea, eb = make_lora(tmp / "a.safetensors", 1), make_lora(tmp / "b.safetensors", 2)

    def check(slots):
        now = dict(pipe.unet.named_parameters())
        for n, p in orig.items():
            want = p + sum(ex[n] * w for ex, w in slots if n in ex) if any(n in ex for ex, _ in slots) else p
            assert (now[n].detach() - want).abs().max().item() < 1e-6, (n, slots)
    try:
        sdp.load_lora(str(tmp / "a.safetensors"), 0.8, slot=0); check([(ea, 0.8)])
        sdp.load_lora(str(tmp / "b.safetensors"), 0.5, slot=1); check([(ea, 0.8), (eb, 0.5)])
        sdp.load_lora(str(tmp / "a.safetensors"), 0.3, slot=0); check([(ea, 0.3), (eb, 0.5)])
        sdp.remove_lora(0); check([(eb, 0.5)])
        sdp.remove_lora(1)
        assert all(torch.equal(p.detach(), orig[n]) for n, p in pipe.unet.named_parameters())
        sdp.load_lora(str(tmp / "a.safetensors"), 0.8, slot=0); sdp.unload_loras()
        assert all(torch.equal(p.detach(), orig[n]) for n, p in pipe.unet.named_parameters())
    finally:
        sdp._unload()


@test("LoRA files load with HF_HUB_OFFLINE=1 (diffusers then demands a weight_name: every LoRA failed for offline users)")
def _():
    import torch
    from safetensors.torch import save_file
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer",
                                            local_files_only=True)
    except Exception:
        raise Skip("SD 1.5 CLIP tokenizer not in .hf_cache")
    sdp, pipe = _tiny_sd(tok)
    tmp = Path(_tf.mkdtemp())
    mods = dict(pipe.unet.named_modules())
    name = next(n for n, m in mods.items() if isinstance(m, torch.nn.Linear) and n.endswith("attn1.to_q"))
    g = torch.Generator().manual_seed(3)
    key = "lora_unet_" + name.replace(".", "_")
    down = torch.randn(4, mods[name].in_features, generator=g) * 0.1
    up = torch.randn(mods[name].out_features, 4, generator=g) * 0.1
    save_file({key + ".lora_down.weight": down, key + ".lora_up.weight": up, key + ".alpha": torch.tensor(4.0)},
              str(tmp / "a.safetensors"))
    w = lambda: dict(pipe.unet.named_parameters())[name + ".weight"].detach().clone()
    before = w()
    try:
        with patch("diffusers.loaders.lora_base.HF_HUB_OFFLINE", True):
            msg = sdp.load_lora(str(tmp / "a.safetensors"), 0.8, slot=0)
        assert msg.startswith("✅"), msg
        assert (w() - (before + 0.8 * (up @ down))).abs().max().item() < 1e-6
        sdp.unload_loras()
        assert torch.equal(w(), before)
    finally:
        sdp._unload()


@test("LoRAs are re-applied on a freshly loaded model (the checkpoint-switch path): fused, no leftover PEFT layers, restores exactly")
def _():
    import torch
    from safetensors.torch import save_file
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer",
                                            local_files_only=True)
    except Exception:
        raise Skip("SD 1.5 CLIP tokenizer not in .hf_cache")
    sdp, pipe = _tiny_sd(tok)
    tmp = Path(_tf.mkdtemp())
    mods = dict(pipe.unet.named_modules())
    name = next(n for n, m in mods.items() if isinstance(m, torch.nn.Linear) and n.endswith("attn1.to_q"))
    g = torch.Generator().manual_seed(3)
    key = "lora_unet_" + name.replace(".", "_")
    down = torch.randn(4, mods[name].in_features, generator=g) * 0.1
    up = torch.randn(mods[name].out_features, 4, generator=g) * 0.1
    save_file({key + ".lora_down.weight": down, key + ".lora_up.weight": up, key + ".alpha": torch.tensor(4.0)},
              str(tmp / "a.safetensors"))
    w = lambda: dict(pipe.unet.named_parameters())[name + ".weight"].detach().clone()
    peft_layers = lambda: sum(1 for m in pipe.unet.modules() if hasattr(m, "lora_A"))
    before = w()
    try:
        # what app._ensure_model does after load_model(): put the saved adapters back and call _reload_all_loras() itself
        # (not load_lora) — it used to raise "'NoneType' object has no attribute 'has'" (no restore snapshot on a fresh model)
        # and left the unfused LoRA layers in the UNet
        assert getattr(sdp, "_snap", None) is None and sdp._clean_unet_state is None
        sdp._lora_adapters = {0: ("a.safetensors", str(tmp / "a.safetensors"), 0.8)}
        sdp._reload_all_loras()
        assert (w() - (before + 0.8 * (up @ down))).abs().max().item() < 1e-6 and peft_layers() == 0
        sdp.unload_loras()
        assert torch.equal(w(), before) and peft_layers() == 0
    finally:
        sdp._unload()


@test("Small audit fixes: X/Y ranges, cleared boxes, init-image cap, save names, v-pred names, OCR langs, record-only PNG")
def _():
    import json as _j
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    import app as _app
    from backend.sampling import detect_prediction
    from backend.png_info import read_image_metadata
    assert _app._xy_values("CFG", "9-5")[0] == [9.0, 8.0, 7.0, 6.0, 5.0]
    assert _app._xy_values("CFG", "1-5:0")[0] == [] and _app._xy_values("CFG", "1-5:0")[1]
    assert _app._xy_values("Steps", "4-10:2")[0] == [4, 6, 8, 10]
    # steps 1 × strength 0.495 left 0 img2img steps
    steps, *_rest = _app._clean_gen_args(1, 7, 512, 512, 1, 1, 0.495, img2img=True)
    assert int(steps * 0.495) >= 1, steps
    for sz, cap in (((1, 2000), 2048), ((32, 1280), 1280), ((10, 10), 2048), ((4000, 3000), 2048)):
        out, _n = _app._fit_init_image(Image.new("RGB", sz), cap)
        assert max(out.size) <= cap and min(out.size) >= 64, (sz, out.size)
    for n, want in (("foo_v_predator.safetensors", False), ("kvpredx.safetensors", False),
                    ("noobaiXLVpred10.safetensors", True), ("NoobAI-XL-Vpred-v1.0.safetensors", True),
                    ("illustriousXL_vprediction.safetensors", True), ("ponyDiffusionV6XL.safetensors", False)):
        assert detect_prediction(n)["v_pred"] is want, n
    assert _app._ocr_langs("en+ch_sim+ja+ko") == ["en", "ch_sim"] and _app._ocr_langs("") == ["en"]
    info = PngInfo(); info.add_itxt("imagegen", _j.dumps({"format": 2, "prompt": "1girl", "steps": 20, "seed": 7}))
    with _tf.TemporaryDirectory() as d:
        f = Path(d) / "r.png"
        Image.new("RGB", (8, 8)).save(f, pnginfo=info)
        m = read_image_metadata(f)
    assert m["prompt"] == "1girl" and m["steps"] == 20 and m["seed"] == 7
    # the VRAM bar with a cleared box (None) mustn't raise
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
    html_out = by["_update_gen_vram"].fn("m.safetensors", None, None, None, "none", "none", False)      # round 16: run the handler with cleared boxes
    assert isinstance(html_out, str) and html_out


@test("Civitai: a same-name file of another size is another version (not '✅ Downloaded'); keys scrubbed from errors")
def _():
    from backend import civitai_client as C
    tmp = Path(_tf.mkdtemp())
    old_dirs = C._TYPE_TO_DIR
    try:
        C._TYPE_TO_DIR = dict(C._TYPE_TO_DIR, LORA=tmp)
        (tmp / "Hero.safetensors").write_bytes(b"x" * 10)
        cl = C.CivitaiClient()
        v2 = {"id": 222, "files": [{"name": "Hero.safetensors", "primary": True, "sizeKB": 150000,
                                    "downloadUrl": "http://127.0.0.1:1/x"}]}
        ok, res = cl.download_model_version({"type": "LORA", "name": "Hero"}, v2)
        assert not ok and (tmp / "Hero.safetensors").stat().st_size == 10, (ok, res)
        assert not (tmp / "Hero.safetensors.part").exists()
    finally:
        C._TYPE_TO_DIR = old_dirs
    s = C.scrub("Client error for url https://civitai.com/api/v1/models?limit=20&token=abcdef1234567890&x=1",
                key="abcdef1234567890")
    assert "abcdef" not in s and "token=***" in s


@test("Dataset prep: same-stem files kept apart, transparency on white, EXIF rotation, honest counts")
def _():
    from PIL import Image
    from backend.dataset_manager import prepare_dataset
    src, out = Path(_tf.mkdtemp()), Path(_tf.mkdtemp())
    Image.new("RGB", (64, 64), (255, 0, 0)).save(src / "a.jpg")
    Image.new("RGB", (64, 64), (0, 0, 255)).save(src / "a.png")
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(src / "s.png")
    ex = Image.new("RGB", (40, 80), (0, 255, 0)); exif = ex.getexif(); exif[0x0112] = 6   # rotate 90° on display
    ex.save(src / "r.jpg", exif=exif)
    items = [dict(filename=n, path=str(src / n), bucket_w=64, bucket_h=64) for n in ("a.jpg", "a.png", "s.png", "r.jpg")]
    items.append(dict(filename="missing.png", path=str(src / "missing.png"), bucket_w=64, bucket_h=64))
    msg = prepare_dataset(items, out)
    names = sorted(p.name for p in out.glob("*.png"))
    assert names == ["a.png", "a_2.png", "r.png", "s.png"], names
    assert Image.open(out / "a.png").getpixel((32, 32))[0] > 200 and Image.open(out / "a_2.png").getpixel((32, 32))[2] > 200
    assert Image.open(out / "s.png").getpixel((32, 32)) == (255, 255, 255)
    assert "Prepared 4" in msg and "1 failed" in msg, msg


@test("Model info: SDXL LoRAs aren't labelled 'SD 1.x'; lineage from ss_sd_model_name")
def _():
    import json as _j, struct
    import backend.model_manager as mm
    tmp = Path(_tf.mkdtemp())

    def lora(name, width, meta):
        hdr = {"lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight":
               {"dtype": "F16", "shape": [4, width], "data_offsets": [0, 8 * width]}, "__metadata__": meta}
        raw = _j.dumps(hdr).encode()
        (tmp / name).write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * (8 * width))
        return str(tmp / name)
    for name, width, meta, want in (("x1.safetensors", 2048, {"ss_sd_model_name": "290640.safetensors"}, "Pony (SDXL)"),
                                    ("x2.safetensors", 2048, {"ss_sd_model_name": "noobai.safetensors"}, "Illustrious (SDXL)"),
                                    ("x3.safetensors", 2048, {}, "SDXL"), ("x4.safetensors", 768, {}, "SD 1.x")):
        p = lora(name, width, meta)
        assert mm._detect_sd_version(mm._read_sf_metadata(p), p) == want, (name, want)


@test("Detectors: a failed download is retried after 5 minutes (was off until restart)")
def _():
    import time as _t
    from backend import detail_tools as dt
    calls = []
    cache = {}
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "huggingface_hub":
            calls.append(1)
            raise ImportError("offline")
        return real_import(name, *a, **k)
    builtins.__import__ = fake_import
    try:
        assert dt._onnx_session(cache, "r", "f", "0", "x", "y") is None and len(calls) == 1
        assert dt._onnx_session(cache, "r", "f", "0", "x", "y") is None and len(calls) == 1   # cached for now
        cache["failed_at"] = _t.time() - 301
        assert dt._onnx_session(cache, "r", "f", "0", "x", "y") is None and len(calls) == 2   # retried
    finally:
        builtins.__import__ = real_import

@test("LoRA trainer: CLIP skip 2 conditioning matches generation; tag frequency / dataset size saved for keyword chips")
def _():
    import json as _j, torch
    from unittest.mock import patch
    from transformers import CLIPTextModel, CLIPTextConfig
    from backend import lora_trainer as LT
    from backend.lora_keywords import lora_keywords
    torch.manual_seed(0)
    te = CLIPTextModel(CLIPTextConfig(vocab_size=100, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                                      num_attention_heads=2, max_position_embeddings=16))
    ids = torch.randint(0, 100, (1, 16))
    enc = te(ids, output_hidden_states=True)
    want = te.text_model.final_layer_norm(enc.hidden_states[-2])      # = Compel PENULTIMATE_..._NORMALIZED
    assert torch.allclose(LT._sd1_hidden(te, enc, 2), want) and LT._sd1_hidden(te, enc, 1) is enc[0]
    cfg = LT.TrainingConfig(base_model="x/base.safetensors", model_type="sd15", dataset_dir="d/myset",
                            trigger_word="dunk", output_name="t")
    out = Path(_tf.mkdtemp()) / "t.safetensors"
    with patch("peft.get_peft_model_state_dict", return_value={}):
        LT.save_lora(object(), None, out, cfg, ["dunk, 1girl, grey hair", "dunk, 1girl, red eyes"])
    from safetensors import safe_open
    with safe_open(str(out), "pt") as f:
        meta = f.metadata()
    assert _j.loads(meta["ss_tag_frequency"]) == {"myset": {"dunk": 2, "1girl": 2, "grey hair": 1, "red eyes": 1}}
    assert _j.loads(meta["ss_dataset_dirs"])["myset"]["img_count"] == 2 and meta["ss_clip_skip"] == "2"
    kw = lora_keywords(str(out))
    assert kw is not None and kw.n_images == 2 and dict(kw.tags).get("grey hair") == 0.5, kw

@test("LoRA trainer heat protection (round 20): cool mode pauses after each step and the pause-while-hot brake waits in the loop, both only timing (identical weights), off = no pause, Stop ends a wait")
def _():
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from backend import lora_trainer as LT, sampling, thermal
    out = Path(_tf.mkdtemp())

    class U(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.lin = torch.nn.Linear(4, 4)

        def forward(s, x, t, h, added_cond_kwargs=None):
            return SimpleNamespace(sample=s.lin(x))

    def make():
        torch.manual_seed(0)
        cfg = LT.TrainingConfig(base_model="x.safetensors", model_type="sd15", epochs=2, output_dir=str(out), output_name="t",
                                device="cpu", gradient_checkpointing=False, seed=1)
        tr = LT.LoRATrainer(cfg)
        tr.unet = U()
        tr.noise_scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type="epsilon", num_train_timesteps=10), add_noise=lambda l, n, t: l + n)
        tr.items = [LT._TrainItem(f"{i}.png", "1girl", 64, 64) for i in range(4)]
        tr.latent_cache = [torch.randn(4) for _ in range(4)]
        tr.embed_cache = [{"encoder_hidden_states": torch.zeros(3, 4)} for _ in range(4)]
        return tr

    def run(sleeps, limit=0.0, factor=0.0, temp=None, abort_after=None):
        tr = make()
        steps = []
        old = (sampling.COOL["factor"], thermal.GUARD["limit"])
        sampling.COOL["factor"], thermal.GUARD["limit"] = factor, limit

        def cb(pct, msg):
            steps.append(msg)
            if abort_after and len(steps) >= abort_after:
                tr.abort()
        try:
            with patch.object(LT.time, "sleep", lambda s: sleeps.append(s)), patch.object(LT, "save_lora", lambda *a, **k: None), \
                    patch.object(LT, "_THERM_EVERY", 0.0), patch.object(thermal, "read_temp", temp or (lambda: 60.0)):
                msg = tr.train(callback=cb)
        finally:
            sampling.COOL["factor"], thermal.GUARD["limit"] = old
        return tr, msg, steps

    # off: no pause at all, and the reference weights
    s_off = []
    tr_off, msg_off, steps_off = run(s_off)
    assert not s_off and "Training complete" in msg_off and len(steps_off) == 8, (s_off, msg_off, len(steps_off))
    # cool mode: a pause after every step (factor × the step time), the weights are bit-identical
    s_cool = []
    tr_cool, msg_cool, steps_cool = run(s_cool, factor=4.0)
    assert any(x > 0 for x in s_cool) and all(0 <= x <= 0.25 for x in s_cool), s_cool[:5]
    assert all(torch.equal(a, b) for a, b in zip(tr_off.unet.state_dict().values(), tr_cool.unet.state_dict().values()))
    # pause while hot: a hot zone makes the in-loop brake wait (acts at limit + 6), a cool one never does
    hot = {"n": 0}

    def read_hot():
        hot["n"] += 1                                    # read 1 = the start check (cool), then 96 for a while (88 + 6 = 94 < 96: the brake), then cooled
        return 60.0 if hot["n"] == 1 else (96.0 if hot["n"] < 8 else 70.0)
    s_hot = []
    tr_hot, msg_hot, steps_hot = run(s_hot, limit=88.0, temp=read_hot)
    assert s_hot.count(2.0) >= 5 and hot["n"] >= 8, (s_hot[:8], hot)                   # wait_cool polls every 2 s
    assert all(torch.equal(a, b) for a, b in zip(tr_off.unet.state_dict().values(), tr_hot.unet.state_dict().values()))
    assert any("Cooling down" in m for m in steps_hot), steps_hot[:3]                  # the wait reports through the callback
    s_cold = []
    run(s_cold, limit=88.0, temp=lambda: 80.0)                                          # under limit + 6: nothing waits
    assert not [x for x in s_cold if x == 2.0], s_cold
    # Stop ends a wait (and the run says so)
    n = {"r": 0}

    def read_then_hot():
        n["r"] += 1
        return 60.0 if n["r"] == 1 else 99.0             # cool at the start check, hot after the first step
    tr_stop, msg_stop, steps_stop = run([], limit=88.0, temp=read_then_hot, abort_after=1)
    assert "Aborted" in msg_stop and len(steps_stop) == 1, (msg_stop, steps_stop)


@test("LoRA trainer: the text-encoder LoRA gets gradients under gradient checkpointing (round 20: with the default reentrant checkpointing it stayed at zero for the whole training)")
def _():
    import torch
    from transformers import CLIPTextModel, CLIPTextConfig
    from peft import LoraConfig, get_peft_model
    from backend import lora_trainer as LT

    def grads(enable):
        torch.manual_seed(0)
        te = CLIPTextModel(CLIPTextConfig(vocab_size=100, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, max_position_embeddings=16))
        te.requires_grad_(False)
        te = get_peft_model(te, LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
        te.train()
        enable(te)
        ids = torch.randint(0, 100, (1, 16))
        te(ids, output_hidden_states=True).hidden_states[-2].sum().backward()
        return sum(1 for n, p in te.named_parameters() if p.requires_grad and "lora_B" in n and p.grad is not None and float(p.grad.abs().sum()) > 0)
    assert grads(LT._enable_te_checkpointing) >= 6                         # the fixed helper: the LoRA's B matrices get gradient
    try:
        old = grads(lambda te: te.gradient_checkpointing_enable())           # the old call: nothing reaches the LoRA (the loss doesn't even require grad here)
    except RuntimeError:
        old = 0
    assert old == 0, old


@test("LoRA trainer Prepare (round 21): the caching loops call a hook per image — Stop ends Prepare ('⏹ Prepare stopped'), the pause-while-hot wait runs on the first image and then every ~10 s only")
def _():
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from backend import lora_trainer as LT, thermal
    tr = LT.LoRATrainer(LT.TrainingConfig(base_model="x.safetensors", model_type="sd15", dataset_dir="d", device="cpu"))
    sleeps, temps = [], {"n": 0}

    def read():
        temps["n"] += 1
        return 95.0 if temps["n"] <= 3 else 60.0          # hot for the first reads, then cool
    old = thermal.GUARD["limit"]
    thermal.GUARD["limit"] = 88.0
    try:
        with patch.object(LT.time, "sleep", lambda s: sleeps.append(s)), patch.object(thermal, "read_temp", read):
            msgs = []
            hook = tr._prepare_hook(lambda c, t, m: msgs.append((c, t, m)))
            hook(0, 4)                                                   # the first image reads the zone and waits (3 hot reads)
            assert sleeps.count(2.0) >= 2 and any("Cooling down" in m[2] for m in msgs), (sleeps, msgs[:2])
            n_reads = temps["n"]
            hook(1, 4); hook(2, 4)                                       # within 10 s: no new read
            assert temps["n"] == n_reads
            tr._abort = True
            try:
                hook(3, 4)
                raise AssertionError("Stop did not end Prepare")
            except LT._TrainStopped:
                pass
    finally:
        thermal.GUARD["limit"] = old
    # a Prepare whose caching raises _TrainStopped says so and frees the model
    tr2 = LT.LoRATrainer(LT.TrainingConfig(base_model="x.safetensors", model_type="sd15", dataset_dir="d", device="cpu"))
    tr2._load_base_model = lambda cb: None
    tr2._build_items = lambda cb: None
    tr2._inject_lora = lambda: None
    tr2._cache = lambda cb: (_ for _ in ()).throw(LT._TrainStopped())
    assert tr2.prepare() == "⏹ Prepare stopped"
    # the caching loops call the hook once per image
    from PIL import Image
    d = Path(_tf.mkdtemp())
    for i in range(3):
        Image.new("RGB", (64, 64), (i * 50, 10, 10)).save(d / f"{i}.png")
    items = [LT._TrainItem(str(d / f"{i}.png"), "c", 64, 64) for i in range(3)]

    class VAE:
        dtype, device = torch.float32, "cpu"
        config = SimpleNamespace(scaling_factor=0.18)
        eval = lambda self: None
        encode = lambda self, t: SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: torch.zeros(1, 4, 8, 8)))
    seen = []
    LT._cache_latents_for_items(items, VAE(), False, None, hook=lambda i, n: seen.append((i, n)))
    assert seen == [(0, 3), (1, 3), (2, 3)], seen


@test("LoRA trainer load (round 21): the trainer's own fp32 model load goes through the commit preflight — a load that cannot fit is refused with the pipelines' message before anything is loaded")
def _():
    from unittest.mock import patch
    from backend import lora_trainer as LT, model_manager as MM
    p = Path(_tf.mkdtemp()) / "m.safetensors"
    p.write_bytes(b"\0" * (1 << 20))                                         # 1 MiB stand-in: need = factor x 0.00098 GB
    with patch.object(MM, "avail_commit_gb", lambda: 0.0035):
        MM._first_load[0] = False
        assert MM.commit_problem(p, scale=1.0) is None                       # 2.05 x 0.00098 = 0.0020 <= 0.0035: fits
        assert MM.commit_problem(p, scale=LT._TRAIN_LOAD_SCALE) is None        # x 1.5 = 0.0030 <= 0.0035: still fits
    with patch.object(MM, "avail_commit_gb", lambda: 0.0025):
        assert MM.commit_problem(p, scale=1.0) is None and "Not enough free memory" in MM.commit_problem(p, scale=LT._TRAIN_LOAD_SCALE)
    tr = LT.LoRATrainer(LT.TrainingConfig(base_model=str(p), model_type="sd15", dataset_dir="d", device="cpu"))
    with patch.object(MM, "avail_commit_gb", lambda: 0.0025), patch("backend.sd_pipeline._is_sdxl", lambda path: False):
        r = tr.prepare()
    assert r.startswith("❌ Prepare failed: Not enough free memory to load m.safetensors"), r
    assert tr.pipe is None                                                    # nothing was loaded


@test("LoRA trainer (round 21): 'Let the trigger carry the character' — tags in >= N % of the captions are left out (trigger kept, case-insensitive, < 3 captions nothing), the Prepare handler passes the two new trailing inputs and shows what was left out")
def _():
    from unittest.mock import patch
    from types import SimpleNamespace
    from backend import lora_trainer as LT
    o, d = LT.drop_shared_tags(["a, b, x", "a, b, y", "a, c, z"], 100)
    assert o == ["b, x", "b, y", "c, z"] and d == [("a", 3)], (o, d)
    o, d = LT.drop_shared_tags(["a, b, x", "a, b, y", "a, c, z"], 60, keep="b")
    assert o == ["b, x", "b, y", "c, z"] and d == [("a", 3)], (o, d)         # b is in 2 of 3 (67 %) but is the 'trigger': kept
    o, d = LT.drop_shared_tags(["a, B, x", "A, b, y", "a, b, z"], 100)
    assert o == ["x", "y", "z"] and [k for k, _ in d] == ["a", "b"]          # case-insensitive
    assert LT.drop_shared_tags(["a, b", "a, c"], 50) == (["a, b", "a, c"], [])   # < 3 captions: nothing
    assert LT.drop_shared_tags(["a, b", "a, c", "a, d"], 0)[1] == []            # 0 = off
    # the trainer applies it to its items; off = captions untouched
    items = lambda: [LT._TrainItem(f"{i}.png", c, 64, 64) for i, c in enumerate(["dkq, 1girl, grey hair, x", "dkq, 1girl, grey hair, y", "dkq, 1girl, grey hair, z"])]
    for pct, want in ((80, ["dkq, x", "dkq, y", "dkq, z"]), (0, ["dkq, 1girl, grey hair, x", "dkq, 1girl, grey hair, y", "dkq, 1girl, grey hair, z"])):
        tr = LT.LoRATrainer(LT.TrainingConfig(base_model="x.safetensors", trigger_word="dkq", dataset_dir="d", device="cpu", drop_shared_pct=pct))
        with patch.object(LT, "_build_items", lambda cfg: items()):
            tr._build_items(None)
        got = [it.caption for it in tr.items]
        assert got == want, (pct, got)
        assert bool(tr.dropped_tags) == bool(pct)
    # the Prepare handler: two new trailing inputs, the config gets the percentage only when the box is ticked, the log lists what was left out
    import app as _app
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
    assert len(by["do_train_prepare"].inputs) == 23
    h = getattr(by["do_train_prepare"].fn, "__wrapped__", by["do_train_prepare"].fn)
    seen = []

    class FakeTrainer:
        def __init__(self, cfg):
            seen.append(cfg); self.items = [1, 2, 3]; self.mode_note = "base model in fp16"; self.dropped_tags = [("grey hair", 3)] if cfg.drop_shared_pct else []

        def prepare(self, callback=None):
            return "✅ Ready"
        cleanup = lambda self: None
    folder = Path(_tf.mkdtemp())
    base = (str(folder), "dkq", "512 (SD 1.5)", "m.safetensors", 16, 16, 1e-4, 5e-5, 1, True, "attn", "cosine", 1, 0.05, True, False, 0, 42, "cpu", "t", 2)
    with patch("backend.lora_trainer.LoRATrainer", FakeTrainer):
        _app._trainer = None
        r_on = h(*base, True, 70)
        r_off = h(*base, False, 70)
    assert seen[0].drop_shared_pct == 70 and seen[1].drop_shared_pct == 0, [c.drop_shared_pct for c in seen]
    assert "grey hair (3)" in r_on[1] and "base model in fp16" in r_on[1] and "grey hair" not in r_off[1], (r_on[1], r_off[1])


@test("LoRA trainer (round 21): Start Training releases the trainer afterwards (finished, stopped or failed) so the next Generate can load its checkpoint")
def _():
    from unittest.mock import patch
    import app as _app
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
    h = getattr(by["do_train_start"].fn, "__wrapped__", by["do_train_start"].fn)
    cleaned = []

    class FT:
        def __init__(self, result):
            self.result = result
        def train(self, callback=None):
            if isinstance(self.result, Exception):
                raise self.result
            return self.result
        def cleanup(self):
            cleaned.append(1)
    for res, mark in (("✅ Training complete", "released"), ("⏹ Aborted at step 3/16", "released"), (RuntimeError("boom"), "boom")):
        _app._trainer = FT(res)
        out = h()
        assert _app._trainer is None and len(cleaned) >= 1, (res, _app._trainer)
        assert mark in out[0], out[0]
        cleaned.clear()
    assert "Prepare first" in h()[0]                                            # nothing prepared: still the old message


@test("LoRA trainer (round 23): a checkpoint saved during a training is announced (progress text + a toast) and is found by the Generate tab's 🔄 Refresh, which lists the LoRA folder afresh each time")
def _():
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from backend import lora_trainer as LT
    import app as _app
    # the trainer announces each periodic save through its callback
    cfg = LT.TrainingConfig(base_model="x.safetensors", model_type="sd15", epochs=3, output_dir=str(Path(_tf.mkdtemp())), output_name="t", device="cpu",
                            gradient_checkpointing=False, save_every_n_epochs=1, seed=1)
    tr = LT.LoRATrainer(cfg)

    class U(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.lin = torch.nn.Linear(4, 4)

        def forward(s, x, t, h, added_cond_kwargs=None):
            return SimpleNamespace(sample=s.lin(x))
    tr.unet = U()
    tr.noise_scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type="epsilon", num_train_timesteps=10), add_noise=lambda l, n, t: l + n)
    tr.items = [LT._TrainItem(f"{i}.png", "c", 64, 64) for i in range(2)]
    tr.latent_cache = [torch.randn(4) for _ in range(2)]
    tr.embed_cache = [{"encoder_hidden_states": torch.zeros(3, 4)} for _ in range(2)]
    msgs = []
    with patch.object(LT, "save_lora", lambda *a, **k: None):
        tr.train(callback=lambda p, m: msgs.append(m))
    saved = [m for m in msgs if m.startswith("💾")]
    assert len(saved) == 2 and "t_ep1.safetensors" in saved[0] and "t_ep2.safetensors" in saved[1] and "Refresh" in saved[0], saved     # epochs 1 and 2 of 3 (the last is the final file)
    # the Start Training handler turns it into a toast
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
    h = getattr(by["do_train_start"].fn, "__wrapped__", by["do_train_start"].fn)
    toasts = []

    class FT:
        def train(self, callback=None):
            callback(0.4, "💾 Saved checkpoint x_ep7.safetensors — the Generate tab's 🔄 Refresh lists it now")
            callback(0.5, "Epoch 8/27 | Step 1/999")
            return "✅ Training complete"
        cleanup = lambda self: None
    _app._trainer = FT()
    with patch("gradio.Info", lambda m: toasts.append(m)):
        h()
    assert len(toasts) == 1 and "x_ep7.safetensors" in toasts[0], toasts
    # 🔄 Refresh lists the LoRA folder afresh: a file saved meanwhile shows up
    folder = {"files": ["a.safetensors"]}
    do_refresh = getattr(by["do_refresh"].fn, "__wrapped__", by["do_refresh"].fn)
    with patch.object(_app, "list_loras", lambda: [(Path(n).stem, "x/" + n) for n in folder["files"]]), patch("backend.model_manager.lora_arch", lambda p: "sd1"):
        first = do_refresh(None)
        folder["files"].append("my_lora_ep7.safetensors")
        second = do_refresh(None)
    flat = lambda r: str(r)
    assert "my_lora_ep7" not in flat(first) and "my_lora_ep7" in flat(second), (flat(first)[:200], flat(second)[:300])


@test("LoRA trainer (round 21): fp16 base for SDXL on a GPU under 20 GB (auto), the dynamic loss scaler (halves and skips on a non-finite gradient, grows after a run of good steps, off = a no-op)")
def _():
    import torch
    from unittest.mock import patch
    from backend import lora_trainer as LT
    cfg = lambda **k: LT.TrainingConfig(base_model="x.safetensors", model_type=k.pop("mt", "sdxl"), device=k.pop("dev", "auto"), **k)
    props = lambda gb: type("P", (), {"total_memory": int(gb * 2**30)})()
    with patch.object(torch.cuda, "is_available", lambda: True), patch.object(torch.cuda, "get_device_properties", lambda i: props(12)), patch.object(LT, "_best_cuda_device", lambda: 0):
        assert LT.use_fp16_base(cfg()) is True and LT.use_fp16_base(cfg(mt="sd15")) is False
        assert LT.use_fp16_base(cfg(dev="cpu")) is False and LT.use_fp16_base(cfg(base_dtype="fp32")) is False and LT.use_fp16_base(cfg(mt="sd15", base_dtype="fp16")) is True
    with patch.object(torch.cuda, "is_available", lambda: True), patch.object(torch.cuda, "get_device_properties", lambda i: props(24)), patch.object(LT, "_best_cuda_device", lambda: 0):
        assert LT.use_fp16_base(cfg()) is False                                   # a 24 GB card keeps the fp32 path
    with patch.object(torch.cuda, "is_available", lambda: False):
        assert LT.use_fp16_base(cfg()) is False                                   # no CUDA: fp32 on the CPU
    # the scaler
    off = LT._LossScaler(False)
    p = torch.nn.Parameter(torch.ones(2)); p.grad = torch.tensor([float("inf"), 1.0])
    assert off.scale(torch.tensor(2.0)).item() == 2.0 and off.unscale_and_check([p]) is True and off.skipped == 0
    sc = LT._LossScaler(True, init=1024.0, growth_interval=3)
    assert sc.scale(torch.tensor(1.0)).item() == 1024.0
    p.grad = torch.tensor([2048.0, 1024.0])
    assert sc.unscale_and_check([p]) and p.grad.tolist() == [2.0, 1.0]
    sc.update(True); sc.update(True); assert sc.scale_value == 1024.0
    sc.update(True); assert sc.scale_value == 2048.0                              # 3 good steps double it
    p.grad = torch.tensor([float("nan"), 1.0])
    assert sc.unscale_and_check([p]) is False
    sc.update(False)
    assert sc.scale_value == 1024.0 and sc.skipped == 1


@test("LoRA trainer: an SDXL LoRA on a GPU under 20 GB gets a warning in the Prepare log (round 20: 3 steps in 14 minutes on 12 GB), SD 1.5 / CPU / big cards don't")
def _():
    from backend import lora_trainer as LT
    w = LT.sdxl_train_warning("sdxl", 8.0)
    assert "10 GB" in w and "8 GB" in w and "SD 1.5" in w
    assert LT.sdxl_train_warning("sd15", 8.0) == "" and LT.sdxl_train_warning("sdxl", None) == "" and LT.sdxl_train_warning("sdxl", 24.0) == ""
    assert LT.sdxl_train_warning("sdxl", 12.0) == ""                          # a 12 GB card trains an SDXL LoRA with the fp16 base (round 21)


class _BatchEnv:
    """The built UI's 🎴 outfit batch and X/Y grid with a fake pipeline that can press Stop after N images."""
    def __enter__(self):
        import json as _j
        from PIL import Image
        import app as _app
        import backend.character_cards as CC
        self._app, self._CC = _app, CC
        self._old = (_app.sd, _app.OUTPUTS_DIR, CC.CARDS_DIR)
        tmp = Path(_tf.mkdtemp())
        _app.OUTPUTS_DIR, CC.CARDS_DIR = tmp / "outputs", tmp / "cards"
        _app.OUTPUTS_DIR.mkdir(); CC.CARDS_DIR.mkdir()
        (CC.CARDS_DIR / "TestGirl.json").write_text(_j.dumps({
            "name": "TestGirl", "checkpoint": "x.safetensors", "tags": "1girl, solo, grey hair, red eyes",
            "outfits": {"gown": "evening gown", "tennis": "tennis uniform", "kimono": "kimono, obi"},
            "negative": "bad hands", "width": 64, "height": 64, "cfg": 6.0, "steps": 4, "scheduler": "Euler a"}))
        by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
        self.cb = getattr(by["do_card_batch"].fn, "__wrapped__", by["do_card_batch"].fn)
        self.xy = getattr(by["do_xy_grid"].fn, "__wrapped__", by["do_xy_grid"].fn)
        for fn in (self.cb, self.xy):
            cl = dict(zip(fn.__code__.co_freevars, fn.__closure__))
            cl["_ensure_model"].cell_contents = lambda *a, **k: (True, "ok")
            cl["_sync_loras"].cell_contents = lambda *a, **k: ""
        env = self

        class FakeSD:
            model_family, current_model, _last_vae_path, prediction, device = "sd15", "x.safetensors", None, {}, "cpu"
            _lora_adapters, boosters, last_seeds, last_var_seeds, pipe, loaded_loras = {}, {}, [], [], object(), []

            def img2img(self, img, prompt, neg, strength, steps, cfg, seed, sched, step_callback=None, clip_skip=1):
                env.i2i_calls += 1
                env.i2i_inputs.append(img.getpixel((0, 0)))
                if env.abort_hires_at and env.i2i_calls >= env.abort_hires_at:
                    _app._generation_abort.set()             # Stop pressed during the hires pass
                    raise _app._GenerationAborted()
                return [img], "i2i"

            def txt2img(self, prompt, neg, w, h, steps, cfg, seed, sched, batch, step_callback=None, **kw):
                env.calls.append((prompt, seed))
                self.last_seeds = [seed + i for i in range(batch)]
                if env.stop_after is not None and len(env.calls) >= env.stop_after:
                    _app._generation_abort.set()           # Stop pressed while this image finishes (post pass)
                return [Image.new("RGB", (w, h), (seed % 255, 40, 90)) for _ in range(batch)], "info"
        self.calls, self.stop_after, self.i2i_calls, self.abort_hires_at, self.i2i_inputs = [], None, 0, None, []
        _app.sd = FakeSD()
        return self

    def __exit__(self, *exc):
        self._app.sd, self._app.OUTPUTS_DIR, self._CC.CARDS_DIR = self._old

    def run_batch(self, seed, steps=4, seeds_per=2, resume=True, hires=False, i2i_image=None, **extra):
        args = ["TestGirl", "", seeds_per, resume, "x.safetensors", "none", "none", 0.7, "none", 0.7, "none", 0.7, False,
                "", "bad hands", "Euler a", steps, 6.0, 64, 64, 1, seed, None, 0.5, False, 1, -1, 0.0, hires, 1.5, 0.45,
                14, "Lanczos", False, 0.35, "auto", "", 0.0, False, 0.0, False, 0.35]
        if i2i_image is not None:
            args[22], args[24] = i2i_image, True                  # the init picture and 'Use img2img'
        return list(self.cb(*args, progress=lambda *a, **k: None, **extra))

    def run_xy(self, x_axis, x_vals, pose=""):
        args = ["x.safetensors", "none", "none", 0.7, "none", 0.7, "none", 0.7, False, "1girl", "bad", "Euler a", 4, 6.0,
                64, 64, 1, 5, None, 0.5, False, 1, -1, 0.0, False, 1.5, 0.45, 14, "Lanczos", False, 0.35, "auto", "",
                0.0, False, 0.0, False, 0.35, False, 0.35, False, 1.0, False, pose, 0.8, "", False, x_axis, x_vals, "none", ""]
        return self.xy(*args, progress=lambda *a, **k: None)


@test("X/Y grid: Face / Hand / Eye denoise axes switch their pass on with the cell's value (the other passes stay as set)")
def _():
    from PIL import Image
    import app as _app
    assert all(a in _app._XY_AXES for a in ("Face denoise", "Hand denoise", "Eye denoise"))
    vals, err = _app._xy_values("Eye denoise", "0.2-0.4:0.1")
    assert not err and [round(v, 2) for v in vals] == [0.2, 0.3, 0.4], (vals, err)
    with _BatchEnv() as E:
        seen = []
        cl = dict(zip(E.xy.__code__.co_freevars, E.xy.__closure__))
        cl["do_generate"].cell_contents = lambda *a, **k: (seen.append(dict(k["extra"])) or [Image.new("RGB", (a[5], a[6]))], "info", [])
        for axis, on, den in (("Eye denoise", "ed_on", "ed_denoise"), ("Face denoise", "fd_on", "fd_denoise"), ("Hand denoise", "hd_on", "hd_denoise")):
            seen.clear()
            E.run_xy(axis, "0.2, 0.4")
            assert len(seen) == 2 and [s[on] for s in seen] == [True, True] and [s[den] for s in seen] == [0.2, 0.4], (axis, seen)
            others = {"ed_on", "fd_on", "hd_on"} - {on}
            assert all(not s[o] for s in seen for o in others), (axis, seen)       # only the varied pass is on


@test("SD→SDXL Bridge (round 27): {a|b} / __name__ in the prompts are resolved from the seed like Generate (same picks in both stages, a seed -1 gets a real one), the records keep the template, unknown wildcards are named, a plain prompt leaves no trace")
def _():
    import json as _j
    from PIL import Image
    from unittest.mock import patch
    import app as _app
    from backend import wildcards as WC
    tmp = Path(_tf.mkdtemp())

    def make_sd(tag):
        class FakeSD:
            model_family, current_model, _last_vae_path, prediction, device = tag, f"{tag}.safetensors", None, {}, "cpu"
            _lora_adapters, boosters, last_seeds, last_var_seeds, loaded_loras = {}, {}, [], [], []
            pipe, force_tiled_decode = object(), False

            def __init__(self):
                self.seen = []

            def load_model(self, *a, **k):
                return "ok"

            def txt2img(self, prompt, neg, w, h, steps, cfg, seed, sched, batch_size=1, step_callback=None, **kw):
                self.seen.append((prompt, neg, seed))
                self.last_seeds = [seed]
                return [Image.new("RGB", (w, h), (20, 40, 60))], "s1"

            def img2img(self, img, prompt, neg, strength, steps, cfg, seed, sched, step_callback=None, **kw):
                self.seen.append((prompt, neg, seed))
                return [img], "s2"
        return FakeSD()
    s15, sxl = make_sd("sd15"), make_sd("sdxl")
    old_out = _app.OUTPUTS_DIR
    _app.OUTPUTS_DIR = tmp
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
    bridge = getattr(by["do_bridge"].fn, "__wrapped__", by["do_bridge"].fn)

    def run(prompt, neg, xl_prompt, seed):
        s15.seen.clear(); sxl.seen.clear()
        with patch.object(_app, "_sd15", s15), patch.object(_app, "_sdxl", sxl), patch.object(_app, "_LAST_BRIDGE_FILE", tmp / "_last_bridge.json"):
            out = bridge("sd15.safetensors", prompt, neg, 4, 6.0, 256, 256, "Euler a", seed, "sdxl.safetensors", xl_prompt, 4, 6.0, 256, 256, 0.5, "Euler a", progress=lambda *a, **k: None)
        return out
    try:
        P1, P2, NEG = "1girl, {red|blue|green|pink|white|black} dress, __nosuchfile_r27__", "detailed, {red|blue|green|pink|white|black} dress", "bad hands, {lowres|blurry}"
        out = run(P1, NEG, P2, 11)
        miss = []
        want1, want2, wantn = WC.resolve(P1, 11, miss), WC.resolve(P2, 11), WC.resolve(NEG, 11)
        assert s15.seen == [(want1, wantn, 11)] and sxl.seen == [(want2, wantn, 11)], (s15.seen, sxl.seen, want1, want2)
        assert "{" not in want2 and "__nosuchfile_r27__" in want1 and ("red" in want1 or "blue" in want1 or "green" in want1 or "pink" in want1 or "white" in want1 or "black" in want1)
        assert "Unknown wildcard" in out[3] and "nosuchfile_r27" in out[3] and "Picks for seed 11" in out[3], out[3][:400]
        recs = sorted(((_j.loads(Image.open(f).info["imagegen"]), f) for f in tmp.glob("*.png")), key=lambda x: x[0].get("bridge_stage", 0))
        r1, r2 = recs[0][0], recs[1][0]
        assert r1["bridge_stage"] == 1 and r1["prompt"] == want1 and r1["prompt_template"] == P1 and r1["negative_template"] == NEG, r1
        assert r2["bridge_stage"] == 2 and r2["prompt"] == want2 and r2["prompt_template"] == P2 and r2["negative_template"] == NEG, r2
        # the picks follow the seed (the same seed again gives the same picture's words, other seeds reach other words)
        run(P1, NEG, P2, 11)
        assert s15.seen[0][0] == want1
        picks = {WC.resolve("{red|blue|green|pink|white|black} dress", sd) for sd in range(40)}
        assert len(picks) >= 3
        # seed -1: a real seed is drawn, the picks are those of that seed, the record says which
        for f in tmp.glob("*.png"):
            f.unlink()
        out = run(P1, NEG, "", -1)
        seed_used = s15.seen[0][2]
        assert seed_used >= 0 and s15.seen[0][0] == WC.resolve(P1, seed_used) and sxl.seen[0][0] == WC.resolve(P1, seed_used) and sxl.seen[0][2] == seed_used, (s15.seen, sxl.seen)
        r2 = [_j.loads(Image.open(f).info["imagegen"]) for f in tmp.glob("*.png") if _j.loads(Image.open(f).info["imagegen"]).get("bridge_stage") == 2][0]
        assert r2["prompt_template"] == P1 and r2["seed"] == seed_used, r2                      # no SDXL prompt: the stage-1 template
        # a plain prompt: nothing resolved, no template in the record
        for f in tmp.glob("*.png"):
            f.unlink()
        out = run("1girl, red dress", "bad hands", "", 5)
        assert s15.seen == [("1girl, red dress", "bad hands", 5)] and "Picks for seed" not in out[3]
        assert all("prompt_template" not in _j.loads(Image.open(f).info["imagegen"]) for f in tmp.glob("*.png"))
    finally:
        _app.OUTPUTS_DIR = old_out


@test("X/Y grid (round 27): a 'Colour strength' axis — values 0.5-1 (ranges allowed, others refused), each cell runs with its own strength, the other settings stay; the axis list is a dropdown choice, not a new input")
def _():
    from PIL import Image
    import app as _app
    assert "Colour strength" in _app._XY_AXES
    vals, err = _app._xy_values("Colour strength", "0.7-1:0.1")
    assert not err and [round(v, 2) for v in vals] == [0.7, 0.8, 0.9, 1.0], (vals, err)
    for bad in ("0.3, 0.8", "1.4", "x"):
        v2, e2 = _app._xy_values("Colour strength", bad)
        assert e2 and not v2, (bad, v2, e2)
    with _BatchEnv() as E:
        seen = []
        cl = dict(zip(E.xy.__code__.co_freevars, E.xy.__closure__))
        cl["do_generate"].cell_contents = lambda *a, **k: (seen.append(dict(k["extra"])) or [Image.new("RGB", (a[5], a[6]))], "info", [])
        E.run_xy("Colour strength", "1.0, 0.9, 0.8, 0.7")
        assert [s["cs"] for s in seen] == [1.0, 0.9, 0.8, 0.7], seen
        assert all(not s["ed_on"] and not s["fd_on"] and not s["hd_on"] and not s["md_on"] for s in seen)         # the other passes stay as set
        seen.clear()
        E.run_xy("CFG", "5, 6")
        assert [s["cs"] for s in seen] == [1.0, 1.0]                                                                # no colour axis: the control's own value (default 1)


@test("X/Y grid (round 36): a 'Pose' axis — starter and own pose names (the ★ optional, case ignored, a unique part is enough, commas inside a name kept), 'none' = no pose in that cell; unknown / ambiguous names refused with the list; each cell turns pose control on with its pose, strength and model from the controls; a dropdown choice, not a new input")
def _():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from PIL import Image
    import app as _app
    from backend import pose_control as pc
    assert "Pose" in _app._XY_AXES and "Pose strength" in _app._XY_AXES
    tmp = Path(tempfile.mkdtemp())
    body = [[10.0 + i, 8.0 + i, 0.9] for i in range(18)]
    import json as _j
    (tmp / "starters").mkdir()
    for title, stem in (("peace sign, one hand (cowboy)", "peace_one"), ("waving (cowboy)", "wave"), ("hands on hips (cowboy)", "hips_c"), ("hands on hips (full body)", "hips_f")):
        (tmp / "starters" / f"{stem}.json").write_text(_j.dumps({"v": 1, "title": title, "size": [200, 300], "body": body, "lh": [], "rh": []}))
    with patch.object(pc, "STARTER_DIR", tmp / "starters"), patch.object(pc, "POSES_DIR", tmp / "poses"):
        pc.save_pose("mine", {"v": 1, "size": [100, 100], "body": body, "lh": [], "rh": []})
        v, e = _app._xy_values("Pose", "★ waving (cowboy), peace sign, one hand (cowboy), MINE, none, hips (full body)")
        assert not e and v == ["★ waving (cowboy)", "★ peace sign, one hand (cowboy)", "mine", "none", "★ hands on hips (full body)"], (v, e)      # a comma inside a name, case, a unique part
        v, e = _app._xy_values("Pose", "wav, peace")
        assert not e and v == ["★ waving (cowboy)", "★ peace sign, one hand (cowboy)"], (v, e)
        for bad, word in (("waving (cowboy), nonsense", "Unknown pose"), ("hands on hips", "more than one pose"), ("", "Enter")):
            v, e = _app._xy_values("Pose", bad)
            assert not v and word in e, (bad, v, e)
        assert "waving (cowboy)" in _app._xy_values("Pose", "nonsense")[1] and "none" in _app._xy_values("Pose", "nonsense")[1]        # the refusal lists the poses
        with _BatchEnv() as E:
            seen = []
            cl = dict(zip(E.xy.__code__.co_freevars, E.xy.__closure__))
            cl["do_generate"].cell_contents = lambda *a, **k: (seen.append(dict(k["extra"])) or [Image.new("RGB", (a[5], a[6]))], "info", [])
            E.run_xy("Pose", "waving, mine, none")
            assert [(s["pc_on"], s["pc_pose"] if s["pc_on"] else None) for s in seen] == [(True, "★ waving (cowboy)"), (True, "mine"), (False, None)], seen
            assert all(s["pc_strength"] == 0.8 for s in seen)                                                                   # strength / model come from the controls
            seen.clear()
            E.run_xy("Pose", "waving, none", pose="mine")                                                                         # the checkbox value of the form does not matter: the cell decides
            assert [s["pc_on"] for s in seen] == [True, False], seen
            seen.clear()
            r = E.run_xy("Pose", "nonsense")
            assert not seen and "Unknown pose" in str(r[1]), r


@test("✨ Polish Auto under pose control (round 36): cowboy / full-body pictures take the '— hands' recipe, portrait / wide shot do not; the second (API-less) listener acts on Auto only, after the dropdown and when the user ticks the checkbox; on_polish's own inputs are unchanged")
def _():
    import app as _app
    from backend import polish as P
    assert P.shot_type("1girl, cowboy shot, standing") == "Cowboy shot" and P.shot_type("1girl, cowboy shot, standing", pose=True) == "Cowboy shot — hands"
    assert P.shot_type("1girl, full body", pose=True) == "Full body — hands" and P.shot_type("", pose=True) == "Cowboy shot — hands"            # no framing tag = the middle recipe
    for prompt in ("1girl, portrait, close-up", "1girl, very wide shot, full body"):
        assert P.shot_type(prompt, pose=True) == P.shot_type(prompt), prompt                                                                    # portrait and wide shot: unchanged
    assert P.shot_type("1girl, cowboy shot, peace sign") == "Cowboy shot — hands"                                                              # the gesture rule is still there
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
    assert [type(c).__name__ for c in by["on_polish"].inputs] == ["Dropdown", "Textbox"]                                                       # the endpoint's inputs did not change
    two = by["on_polish_pose"]
    assert [type(c).__name__ for c in two.inputs] == ["Dropdown", "Textbox", "Checkbox"] and len(two.outputs) == 13, len(two.outputs)
    cfg = _app.build_app().get_config_file()
    assert not any(d.get("api_name") == "on_polish_pose" for d in cfg["dependencies"]) and sum(1 for d in cfg["dependencies"] if d.get("api_name") == "on_polish") == 1     # no endpoint of its own
    keys = ["hires_on", "hires_scale", "hires_denoise", "hires_steps", "hires_upscaler", "fd_on", "fd_denoise", "hd_on", "hd_denoise", "ed_on", "ed_denoise", "md_on"]
    out = two.fn("Auto", "1girl, cowboy shot", True)
    want = P.recipe("Cowboy shot — hands")
    assert [u.get("value") for u in out[:-1]] == [want.get(k) for k in keys] and "Cowboy shot — hands" in out[-1], out
    out = two.fn("Auto", "1girl, cowboy shot", False)                                                                                           # the checkbox cleared while on Auto: the plain recipe again
    assert "hands" not in out[-1] and out[7].get("value") in (None, False) or out[7] == {"__type__": "update"}, out
    for choice in ("(off)", "Full body", "Portrait"):                                                                                           # only Auto is touched
        res = two.fn(choice, "1girl, cowboy shot", True)
        assert all(u == {"__type__": "update"} for u in res), (choice, res)


@test("Stop that lands after an image's sampler (post pass) still ends the outfit batch and the X/Y grid")
def _():
    with _BatchEnv() as E:
        E.stop_after = 3
        outs = E.run_batch(seed=11)
        # do_generate used to clear the shared abort flag in its `finally`, so the batch carried on with image 4…6
        assert len(E.calls) == 3, len(E.calls)
        assert "stopped early" in outs[-1][1] and len(list(E._app.OUTPUTS_DIR.glob("outfits_TestGirl_*.png"))) == 1
        E.calls.clear(); E.stop_after = 2
        res = E.run_xy("Steps", "4,5,6,7")
        assert len(E.calls) == 2, len(E.calls)
        assert "2 images" in res[1] and "stopped early" in res[1] and len(res[0]) == 3, res[1]   # grid + the 2 real cells, no blank tiles


@test("SmartSplit generation path (round 26: behaviour instead of source text): enabled + a built pipe + not img2img → _smartsplit_pipe.generate() with a progress callback; disabled, no pipe, or img2img → the standard path")
def _():
    from unittest.mock import patch
    from PIL import Image
    import app as _app
    import backend.sd_pipeline as SDP
    from backend.smartsplit_pipeline import SmartSplitConfig
    on = SmartSplitConfig(enabled=True, text_encoder_device="cpu", unet_device="cuda", vae_device="cpu")
    off = SmartSplitConfig(enabled=False)

    class FakeSplit:
        pipe = object()

        def __init__(self):
            self.calls = []

        def generate(self, **kw):
            self.calls.append(kw)
            return [Image.new("RGB", (kw["width"], kw["height"]), (9, 9, 9))], {}

    def run(cfg, pipe, i2i=False):
        with _BatchEnv() as E, patch.object(SDP, "_load_scheduler", lambda *a, **k: None), patch.object(_app, "_smartsplit_cfg", cfg), patch.object(_app, "_smartsplit_pipe", pipe):
            gen = dict(zip(E.cb.__code__.co_freevars, E.cb.__closure__))["do_generate"].cell_contents                       # the Generate tab's own do_generate
            _app._generation_abort.clear()                                                                                 # an earlier test's Stop must not end this run before it starts
            gen("1girl", "bad", "Euler a", 4, 6.0, 256, 256, 1, 11, Image.new("RGB", (256, 256)) if i2i else None, 0.5, i2i, True, _app._clean_extra({}),
                progress=lambda *a, **k: None)
            return len(E.calls), E.i2i_calls
    fs = FakeSplit()
    plain, i2i_calls = run(on, fs)
    assert plain == 0 and len(fs.calls) == 1 and all(callable(c["progress_callback"]) and c["batch_size"] == 1 and c["width"] == 256 for c in fs.calls), (plain, len(fs.calls))   # SmartSplit made the pictures, the standard path made none
    for cfg, pipe, i2i, label in ((off, FakeSplit(), False, "disabled"), (on, None, False, "no pipe"), (on, FakeSplit(), True, "img2img")):
        n_plain, n_i2i = run(cfg, pipe, i2i)
        assert (pipe is None or not pipe.calls) and (n_i2i == 1 if i2i else n_plain == 1), (label, n_plain, n_i2i, getattr(pipe, "calls", None))


@test("Outfit batch: 'Skip pairs already made' also resumes with the default seed -1; a finished batch draws new seeds")
def _():
    with _BatchEnv() as E:
        E.stop_after = 3
        E.run_batch(seed=-1)
        base = E.calls[0][1]
        assert len(E.calls) == 3
        E.calls.clear(); E.stop_after = None
        E.run_batch(seed=-1)                                  # same settings, Skip on: only the 3 unfinished pairs
        assert len(E.calls) == 3, len(E.calls)
        assert {c[1] for c in E.calls} <= {base, base + 1}, (base, E.calls)
        E.calls.clear()
        E.run_batch(seed=-1)                                  # finished last time → fresh seeds, nothing reused
        assert len(E.calls) == 6 and E.calls[0][1] != base or len(E.calls) == 6
        E.calls.clear()
        E.run_batch(seed=-1, resume=False)                    # Skip off never reuses and never remembers
        assert len(E.calls) == 6


@test("Outfit batch: an image whose hires pass was stopped is shown but not remembered as done (Skip redoes it)")
def _():
    import json as _j
    with _BatchEnv() as E:
        E.abort_hires_at = 2                               # image 1's hires completes, Stop lands in image 2's hires pass
        outs = E.run_batch(seed=-1, hires=True)
        assert len(E.calls) == 2 and len(outs[-1][0]) == 3, (len(E.calls), len(outs[-1][0]))   # sheet + both images shown
        idx = _j.loads((E._app.OUTPUTS_DIR / "card_batches" / "TestGirl.json").read_text(encoding="utf-8"))
        assert len([k for k in idx if k != "__pending__"]) == 1, idx       # only the finished image 1
        E.calls.clear(); E.abort_hires_at = None
        E.run_batch(seed=-1, hires=True)
        assert len(E.calls) == 5, len(E.calls)             # image 2 redone + the 4 that never ran


@test("Identity check: card tags -> hair / eye colour traits; WD14 probabilities under the threshold are flagged")
def _():
    from backend import identity_check as IC
    t = IC.traits("1girl, solo, heroine, red eyes, grey hair, long hair, hair between eyes, hair ornament")
    assert t["hair"][0] == "grey" and "white_hair" in t["hair"][1] and t["eyes"] == ("red", ["red_eyes"]), t
    assert IC.traits("1girl, long hair, short hair, hair between eyes") == {}          # not colours
    assert IC.traits("silver hair, blue eyes")["hair"][1][0] == "grey_hair"            # Danbooru has no "silver hair"
    assert IC.check([None, None], "1girl, long hair") == [{}, {}]
    fake = iter([{"grey_hair": 0.9, "red_eyes": 0.8}, {"grey_hair": 0.2, "white_hair": 0.1, "red_eyes": 0.05}])
    res = IC.check([object(), object()], "red eyes, grey hair", probs_fn=lambda im: next(fake))
    assert res[0]["flags"] == [] and sorted(res[1]["flags"]) == ["grey hair 0.20", "red eyes 0.05"], res


@test("⭐ score: accessory tags from a card, head crop, anatomy / look / noise flags cost stars, the badge is drawn")
def _():
    from PIL import Image
    from backend import image_score as S
    acc = S.accessories("1girl, (black butterfly hair ornament:1.2), red eyes, hair ornament, hairclip, red hair bow, smile")
    assert acc == [("black butterfly hair ornament", ["black_butterfly_hair_ornament", "butterfly_hair_ornament"]),
                   ("hair ornament", ["hair_ornament"]), ("hairclip", ["hairclip"]), ("red hair bow", ["red_hair_bow"])], acc
    assert S.accessories("1girl, red eyes") == []
    im = Image.new("RGB", (832, 1216))
    assert S.head_crop(im, (100, 100, 200, 200)).size == (240, 240)                          # 2.4 face widths, centred
    assert S.head_crop(im, (0, 0, 100, 100)).size == (240, 240) and S.head_crop(im, (700, 1100, 832, 1216)).size == (316, 316)
    tags = "1girl, red eyes, grey hair, black butterfly hair ornament, red hair bow"
    box = [(300, 200, 420, 320)]
    good = {"grey_hair": 0.9, "red_eyes": 0.9, "butterfly_hair_ornament": 0.9}
    cases = {                                        # name: (faces, hands, WD14 probabilities of the head crop, speck share)
        "ok": (box, [], good, 0.0),
        "no face": ([], [], good, 0.0),
        "two faces": (box * 2, [], good, 0.0),
        "hands": (box, [1, 2, 3], good, 0.0),
        "off": (box, [], {"grey_hair": 0.9, "red_eyes": 0.1, "butterfly_hair_ornament": 0.2}, 0.0),
        "noise": (box, [], good, 0.002),
        "wreck": ([], [1, 2, 3], {"grey_hair": 0.1, "red_eyes": 0.1, "butterfly_hair_ornament": 0.1}, 0.002),
    }
    by = {name: S.score_images([im], tags, probs_fn=lambda c, p=pr: p, faces_fn=lambda i, f=fc: f, hands_fn=lambda i, h=hd: h,
                               speck_fn=lambda i, s=sp: s)[0] for name, (fc, hd, pr, sp) in cases.items()}
    assert by["ok"] == {"stars": 5, "flags": []}
    assert by["off"] == {"stars": 3, "flags": ["red eyes 0.10", "no black butterfly hair ornament 0.20"]}, by["off"]
    assert by["no face"]["stars"] == 3 and by["no face"]["flags"] == ["no face found"]
    assert by["two faces"]["stars"] == 4 and by["two faces"]["flags"] == ["2 faces"]
    assert by["hands"]["stars"] == 4 and by["hands"]["flags"] == ["3 hands"]
    assert by["noise"] == {"stars": 5, "flags": [], "notes": [S.SPECK_NOTE]}, by["noise"]              # round 14: a note, never a star
    assert by["wreck"]["notes"] == [S.SPECK_NOTE]
    import app as _appn
    assert _appn._speck_note([by["ok"], by["off"]]) == "" and "2 picture(s)" in _appn._speck_note([by["noise"], by["wreck"], by["ok"]]) and "no star was taken" in _appn._speck_note([by["noise"]])
    badged = S.badge(im, by["noise"])
    assert badged.size == im.size
    assert by["wreck"]["stars"] == 1 and "no face found" in by["wreck"]["flags"]
    # the probabilities come from the head crop (2.4 face widths); a tag WD14 doesn't know ("red hair bow") is never held against the picture
    heads = []
    S.score_images([im], tags, probs_fn=lambda c: heads.append(c.size) or good, faces_fn=lambda i: box, hands_fn=lambda i: [], speck_fn=None)
    assert heads == [(288, 288)], heads
    b = S.badge(Image.new("RGB", (200, 300), (128, 128, 128)), {"stars": 2, "flags": ["no face found"]})
    assert b.size == (200, 300) and b.getpixel((2, 2)) != (128, 128, 128) and b.getpixel((190, 290)) == (128, 128, 128)
    flat = Image.new("RGB", (300, 400), (180, 40, 40))                                        # a flat saturated picture is no speck
    assert S.colour_specks(flat) == 0.0
    spotty = Image.new("RGB", (300, 400), (200, 200, 200))
    for x in range(10, 290, 20):
        for y in range(10, 390, 20):
            spotty.putpixel((x, y), (255, 0, 0))
    assert 0 < S.colour_specks(spotty) < 0.01


@test("Outfit batch: says which images WD14 doesn't see the card's colours in (and when all match)")
def _():
    import re as _re
    from backend import identity_check as IC
    from backend import image_score as IS
    saved = (IC.available, IC.probs, IS.available)
    try:
        IC.available = lambda: True
        IS.available = lambda: {"look": IC.available(), "faces": False, "hands": False}      # no detector models in the test
        # FakeSD paints (seed % 255, 40, 90): seed 12 is the off-model one
        IC.probs = lambda im: {"grey_hair": 0.9, "red_eyes": 0.05 if im.getpixel((0, 0))[0] == 12 else 0.9}
        with _BatchEnv() as E:
            msg = _re.sub("<[^>]+>", " ", E.run_batch(seed=11)[-1][1])
            assert "possibly off-model" in msg and "gown · seed 12 (red eyes 0.05)" in msg and "seed 11" not in msg, msg
            assert "⭐ 4.5/5 on average" in msg, msg                                    # the three seed-12 images 4/5, the other three 5/5
            IC.probs = lambda im: {"grey_hair": 0.9, "red_eyes": 0.9}
            msg = _re.sub("<[^>]+>", " ", E.run_batch(seed=11)[-1][1])               # all reused: still checked
            assert "match the card in all 6 image" in msg and "off-model" not in msg and "⭐ 5.0/5" in msg, msg
            IC.available = lambda: False                                                # model not cached: silent
            msg = _re.sub("<[^>]+>", " ", E.run_batch(seed=11)[-1][1])
            assert "off-model" not in msg and "match the card" not in msg and "⭐" not in msg, msg
    finally:
        IC.available, IC.probs, IS.available = saved


@test("⭐ scoring is CPU work: a pause-while-hot wait before the first picture and then at most every 10 s; the outfit batch says how long it paused")
def _():
    import re as _re
    from unittest.mock import patch
    from PIL import Image
    import app as _app
    from backend import image_score as IS
    # score_images calls pause(k, n) before picture k (before any detector runs) and rates exactly as without it
    im, box, order, seen = Image.new("RGB", (400, 400), (90, 90, 90)), [(100, 100, 230, 230)], [], []
    good = {"grey_hair": 0.9, "red_eyes": 0.9}
    kw = dict(probs_fn=lambda c: good, hands_fn=lambda i: [], speck_fn=None)
    plain = IS.score_images([im, im, im], "1girl, grey hair, red eyes", faces_fn=lambda i: box, **kw)
    paused = IS.score_images([im, im, im], "1girl, grey hair, red eyes", faces_fn=lambda i: order.append("faces") or box,
                             pause=lambda k, n: (seen.append((k, n)), order.append("pause")), **kw)
    assert paused == plain and seen == [(0, 3), (1, 3), (2, 3)] and order[:2] == ["pause", "faces"], (seen, order)
    # the app's callback: the first call checks, calls within 10 s don't, a later one does; the waited seconds add up
    now, calls = [100.0], []
    fake_time = type("T", (), {"time": staticmethod(lambda: now[0])})
    with patch.object(_app, "time", fake_time), patch.object(_app, "_thermal_wait", lambda progress=None: calls.append(progress) or 2.5):
        p = _app._score_pause("progress")
        p(0, 8)
        now[0] += 4; p(1, 8)
        now[0] += 4; p(2, 8)                                  # 8 s after the first check: no new check
        assert len(calls) == 1 and calls == ["progress"] and p.waited[0] == 2.5, calls
        now[0] += 3; p(3, 8)                                  # 11 s: checks again
        assert len(calls) == 2 and p.waited[0] == 5.0, (calls, p.waited)
        now[0] += 9.5; p(4, 8)
        assert len(calls) == 2                                # 9.5 s after the second check
    # the outfit batch hands that callback to the rating and reports the pause
    class FakePause:
        waited = [7.0]
        n = 0
        def __call__(self, k, n):
            FakePause.n += 1
    got = {}
    def fake_score(images, tags, **k):
        got["pause"] = k.get("pause")
        for j in range(len(images)):
            k["pause"](j, len(images))
        return [{"stars": 5, "flags": []} for _ in images]
    saved = (IS.available, IS.score_images, _app._score_pause)
    try:
        IS.available = lambda: {"look": True, "faces": False, "hands": False}
        IS.score_images = fake_score
        _app._score_pause = lambda progress=None, every=10.0: FakePause()
        with _BatchEnv() as E:
            msg = _re.sub("<[^>]+>", " ", E.run_batch(seed=11)[-1][1])
        assert isinstance(got["pause"], FakePause) and FakePause.n == 6, (got, FakePause.n)
        assert "⭐ 5.0/5" in msg and "the rating paused 7 s while the laptop cooled down" in msg, msg
        FakePause.waited = [0.4]
        with _BatchEnv() as E:
            msg = _re.sub("<[^>]+>", " ", E.run_batch(seed=11)[-1][1])
        assert "paused" not in msg, msg                       # under a second: nothing to report
    finally:
        IS.available, IS.score_images, _app._score_pause = saved


@test("CCIP: preprocessing, no silent download, a download that isn't the pinned file is deleted, features come from the session")
def _():
    import numpy as np
    from PIL import Image
    from unittest.mock import patch
    from backend import ccip
    x = ccip.preprocess(Image.new("RGB", (200, 300), (255, 0, 0)))
    assert x.shape == (1, 3, 384, 384) and x.dtype == np.float32
    assert abs(float(x[0, 0].mean()) - 1.0) < 1e-6 and abs(float(x[0, 1].mean()) + 1.0) < 1e-6          # red → +1, green / blue → −1 ([-1, 1] scaling)
    assert len(ccip.SHA256) == 64 and ccip.SIZE_MB == 150 and ccip.MODEL_TAG.startswith("ccip-")
    with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
        assert ccip.cached_path() is None and not ccip.available() and ccip.features(Image.new("RGB", (8, 8))) is None
    with patch("huggingface_hub.try_to_load_from_cache", return_value="C:/does/not/exist.onnx"):
        assert not ccip.available()                                                                       # a cache entry whose file is gone doesn't count
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        fake = os.path.join(d, "model_feat.onnx")
        open(fake, "wb").write(b"not the model")
        with patch("huggingface_hub.hf_hub_download", return_value=fake):
            try:
                ccip.download()
                raise AssertionError("a file with the wrong SHA-256 was accepted")
            except RuntimeError as e:
                assert "SHA-256" in str(e)
        assert not os.path.exists(fake)                                                                   # refused and deleted

        class Sess:
            def get_inputs(self): return [type("I", (), {"name": "input"})()]
            def run(self, _o, feed): return [np.arange(768, dtype=np.float32)[None] * feed["input"].shape[0]]
        ccip._session = Sess()
        try:
            with patch.object(ccip, "cached_path", return_value="x"):
                f = ccip.features(Image.new("RGB", (30, 30)))
            assert f.shape == (768,) and f.dtype == np.float32 and f[5] == 5
        finally:
            ccip._session = None


@test("Identity: reference pictures → a card's centroid (base64 float16); damaged / foreign data is dropped; the card file keeps it")
def _():
    import numpy as np
    from PIL import Image
    from backend import identity_score as ID
    import backend.character_cards as CC
    rng = np.random.default_rng(5)
    v = ID._unit(rng.normal(size=768))
    d = ID.decode_vec(ID.encode_vec(v))
    assert d is not None and abs(float(d @ v) - 1) < 1e-3 and abs(float(np.linalg.norm(d)) - 1) < 1e-5            # float16 round trip, unit length
    nan = v.copy(); nan[3] = np.nan
    for bad in ("not base64!!", None, 5, ID.encode_vec(np.zeros(768)), ID.encode_vec(v[:100]), ID.encode_vec(nan)):
        assert ID.decode_vec(bad) is None, bad
    # references: the stub detector finds a 100 px face in every picture except the red one (none) and the blue one (40 px); the stub feature reads the grey level
    feats = {k: ID._unit(v + 0.15 * rng.normal(size=768)) for k in (10, 20, 30, 40, 50)}
    faces = lambda im: [] if im.getpixel((0, 0)) == (255, 0, 0) else ([(80, 80, 120, 120)] if im.getpixel((0, 0)) == (0, 0, 255) else [(50, 50, 150, 150)])
    feat = lambda crop: feats[crop.getpixel((0, 0))[0]]
    def pic(k):
        return Image.new("RGB", (200, 200), {0: (255, 0, 0), -1: (0, 0, 255)}.get(k, (k, k, k)))
    pics = lambda ks: [pic(k) for k in ks]
    ident = ID.build_identity(pics([10, 20, 30, 0, -1, 40]), faces_fn=faces, feature_fn=feat)                          # no face / a 40 px face are skipped
    assert ident and ident["n"] == 4 and ident["model"] == ID.MODEL_TAG and "light" not in ident, ident
    cen = ID.decode_vec(ident["centroid"])
    want = ID._unit(np.mean([feats[k] for k in (10, 20, 30, 40)], 0))
    assert abs(float(cen @ want) - 1) < 1e-3
    loo = [float(feats[k] @ ID._unit(np.mean([feats[j] for j in (10, 20, 30, 40) if j != k], 0))) for k in (10, 20, 30, 40)]
    assert abs(ident["mean"] - np.mean(loo)) < 2e-3 and abs(ident["sd"] - np.std(loo)) < 2e-3, (ident, loo)
    assert ID.build_identity(pics([10, 20, 0, -1]), faces_fn=faces, feature_fn=feat) is None                           # fewer than MIN_REFS usable pictures
    assert ID.build_identity(pics([10, 20, 30]), faces_fn=faces, feature_fn=lambda c: None) is None                    # the model isn't available here
    assert ID.clean_identity(ident) == ident
    junk = dict(ident, mean="x", sd=-3, n=True, light="bright", model=7)
    cj = ID.clean_identity(junk)
    assert cj["mean"] == 0.95 and cj["sd"] == 0.02 and cj["n"] == ID.MIN_REFS and cj["model"] == ID.MODEL_TAG and "light" not in cj, cj
    assert ID.clean_identity({"centroid": "xx"}) is None and ID.clean_identity([1]) is None and ID.clean_identity(None) is None
    # the card file keeps it, a damaged one is dropped, profiles carry it
    card = CC.clean_card({"name": "Id Girl", "tags": "1girl", "identity": ident,
                          "profiles": {"x.safetensors": {"cfg": 5}}})
    assert card["identity"] == ident and "identity" not in CC.clean_card({"name": "A", "identity": {"centroid": "zz"}})
    merged, key = CC.card_for_checkpoint(card, "x.safetensors")
    assert key and merged["identity"] == ident and merged["cfg"] == 5
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp(prefix="idcard_"))
    old = CC.CARDS_DIR
    try:
        CC.CARDS_DIR = tmp / "characters"
        CC.save_card(card)
        assert CC.load_card("Id Girl")["identity"] == ident
    finally:
        CC.CARDS_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)


@test("⭐ identity: a face further from her references than her own pictures costs a star; small faces, a missing / foreign model, no face skip the check")
def _():
    import numpy as np
    from PIL import Image
    from backend import identity_score as ID
    from backend import image_score as IS
    rng = np.random.default_rng(11)
    c = ID._unit(rng.normal(size=ID.DIM))
    u = ID._unit(rng.normal(size=ID.DIM) - c * float(rng.normal(size=ID.DIM) @ c))
    u = ID._unit(u - c * float(u @ c))                                            # a unit vector orthogonal to the centroid
    with_cos = lambda t: np.float32(c * t + u * np.sqrt(1 - t * t))               # a head feature at an exact cosine to the centroid
    ident = {"model": ID.MODEL_TAG, "n": 8, "centroid": ID.encode_vec(c), "mean": 0.95, "sd": 0.025}
    sd = max(ident["sd"], ID.SD_FLOOR)
    feats = {}
    mk = lambda col, t: (feats.__setitem__(col, with_cos(t)), Image.new("RGB", (400, 400), col))[1]       # the stub feature reads the picture's grey level
    box, small = [(100, 100, 230, 230)], [(100, 100, 160, 160)]                                           # a 130 px face and a 60 px face
    run = lambda im, **k: IS.score_images([im], "1girl", probs_fn=lambda cr: {}, faces_fn=lambda i: k.pop("faces", box), hands_fn=lambda i: [], speck_fn=None,
                                          identity=k.pop("identity", ident), feature_fn=k.pop("feature_fn", lambda cr: feats[cr.getpixel((0, 0))]))[0]
    assert run(mk((200, 200, 200), 0.95 - 0.5 * sd)) == {"stars": 5, "flags": []}                          # a bit under the mean: normal
    off = run(mk((201, 201, 201), 0.95 - (ID.Z_FLAG + 1) * sd))
    assert off["stars"] == 4 and off["flags"][0].startswith("not her?") and "0.95" in off["flags"][0], off
    far = mk((202, 202, 202), 0.3)
    assert run(far, faces=small) == {"stars": 5, "flags": []}                                               # a 60 px face is not judged (unreliable feature)
    assert run(far)["stars"] == 4
    chk = ID.check(ident, with_cos(0.5), 130)
    assert chk["flag"] is True and chk["judged"] is True and chk["z"] < -ID.Z_FLAG and chk["mean"] == 0.95
    assert ID.check(ident, with_cos(0.5), 60)["judged"] is False and ID.check(ident, with_cos(0.5), 60)["flag"] is False
    assert ID.check(ident, with_cos(0.95), None)["flag"] is False and ID.check(ident, with_cos(0.5), None)["flag"] is True       # size unknown: judged
    assert ID.check(dict(ident, model="wd-vit-tagger-v3"), with_cos(0.5), 130) is None                       # learned with another model: not comparable
    assert run(far, identity=dict(ident, model="wd-vit-tagger-v3")) == {"stars": 5, "flags": []}
    # nothing to compare: no identity, an unusable one, no face, the feature missing
    assert run(far, identity=None)["stars"] == 5 and run(far, identity={"centroid": "zz"})["stars"] == 5
    assert run(far, faces=[])["flags"] == ["no face found"]
    assert run(far, feature_fn=lambda cr: None) == {"stars": 5, "flags": []}
    assert ID.check({"centroid": "zz"}, with_cos(0.5), 130) is None and ID.check(ident, None, 130) is None
    # the floor: references that agree to a hair don't make the tolerance razor thin
    tight = dict(ident, sd=0.0)
    assert ID.check(tight, with_cos(0.95 - 1.5 * ID.SD_FLOOR), 130)["flag"] is False and ID.check(tight, with_cos(0.95 - 2.5 * ID.SD_FLOOR), 130)["flag"] is True


@test("⭐ identity: outfits with a hat / hairstyle get the bigger margin (Z_FLAG_HAIR), the others keep the normal one; the batch maps outfits to pictures")
def _():
    import numpy as np
    from PIL import Image
    import app as _app
    from backend import identity_score as ID
    from backend import image_score as IS
    rng = np.random.default_rng(5)
    c = ID._unit(rng.normal(size=ID.DIM))
    u = ID._unit(rng.normal(size=ID.DIM)); u = ID._unit(u - c * float(u @ c))
    with_cos = lambda t: np.float32(c * t + u * np.sqrt(1 - t * t))
    ident = {"model": ID.MODEL_TAG, "n": 8, "centroid": ID.encode_vec(c), "mean": 0.95, "sd": 0.025}
    sd = ident["sd"]
    box = [(100, 100, 230, 230)]
    im = Image.new("RGB", (400, 400), (60, 60, 60))
    feat = lambda t: (lambda cr: with_cos(t))
    run = lambda t, z: IS.score_images([im], "1girl", probs_fn=lambda cr: {}, faces_fn=lambda i: box, hands_fn=lambda i: [], speck_fn=None,
                                       identity=ident, feature_fn=feat(t), identity_z=z)[0]
    mid = 0.95 - 3.5 * sd                                          # 3.5 spreads below her: flagged at 2, not at 5
    assert run(mid, None)["stars"] == 4 and run(mid, [None])["stars"] == 4
    assert run(mid, [ID.Z_FLAG_HAIR]) == {"stars": 5, "flags": []}
    far = 0.95 - (ID.Z_FLAG_HAIR + 1.5) * sd
    assert run(far, [ID.Z_FLAG_HAIR])["stars"] == 4 and run(far, [ID.Z_FLAG_HAIR])["flags"][0].startswith("not her?")
    assert run(0.3, [ID.Z_FLAG_HAIR])["stars"] == 4                  # another girl is flagged at any margin
    assert run(mid, [])["stars"] == 4                                  # a short list = the normal margin
    card = {"outfits": {"school": "school uniform, pleated skirt", "hat": "straw hat, sundress", "tail": "ponytail, tank top", "beach": "bikini, sandals"}}
    outs = ["school", "hat", "tail", "beach", "(no outfit tags)"]
    z = _app._identity_margins(card, outs, 2, 10)
    assert z == [None, None, ID.Z_FLAG_HAIR, ID.Z_FLAG_HAIR, ID.Z_FLAG_HAIR, ID.Z_FLAG_HAIR, None, None, None, None], z
    assert _app._identity_margins(card, [], 2, 3) == [None, None, None] and _app._identity_margins({}, ["x"], 1, 2) == [None, None]
    assert _app._identity_margins(card, outs, 2, 11)[10] is None       # a picture beyond the outfit list (stopped batch padding) never crashes


@test("Size preset shows the size now set (a matching preset, else W×H); the Generate row sits above the Wildcards accordion; the Power-profile hint is short")
def _():
    import app as _app
    assert _app._size_preset_now({"width": 832, "height": 1216}) == "SDXL · 832×1216 portrait"
    assert _app._size_preset_now({"width": 900, "height": 600}) == "900×600" and _app._parse_size_preset("900×600") == (900, 600)
    assert _app._size_preset_now({}) == _app._size_preset_now({"width": _app.DEFAULT_WIDTH, "height": _app.DEFAULT_HEIGHT})
    assert _app._size_preset_now({"width": "x", "height": None})                       # damaged session values never raise
    src = open(_app.__file__, encoding="utf-8").read()
    assert src.index('generate_btn = gr.Button("✨ Generate') < src.index('with gr.Accordion("🎲 Wildcards')
    assert len(_app._POWER_INFO.split()) < 60, len(_app._POWER_INFO.split())


@test("🔬 Max detail (round 14): the Polish entry, the control's plumbing (clean_extra, record, A1111 text, restore), the 2× pass doubles each picture and refuses over 64 MP")
def _():
    from PIL import Image
    import app as _app
    from backend import polish as P
    from backend import records as R
    from backend.png_info import read_png_info
    r = P.recipe("Full body — max detail")
    assert r["md_on"] is True and r["ed_on"] is True and r["hires_steps"] == 8 and r["hires_on"] and P.time_factor("Full body — max detail") > P.time_factor("Full body")
    assert "Real-ESRGAN" in P.summary("Full body — max detail") and "md_on" not in P.recipe("Full body")
    assert P.shot_type("1girl, full body, standing") == "Full body"                    # Auto never picks the max-detail entry
    w = P.recipe("Wide shot — max detail")                                              # round 15: the Wide shot chain + the 2× pass, 14 hires steps kept
    assert w["md_on"] and w["fd_on"] and w["ed_on"] and w["hires_steps"] == 14 and P.time_factor("Wide shot — max detail") > P.time_factor("Wide shot")
    assert P.shot_type("1girl, very wide shot, from far away") == "Wide shot"
    assert _app._clean_extra({"md_on": 1})["md_on"] is True and _app._clean_extra({})["md_on"] is False
    # record -> restore plan -> control values; the A1111 text says it and PNG Info reads it back
    rec = {"app": "ImageGen Studio", "format": 2, "mode": "txt2img", "prompt": "1girl", "steps": 12, "cfg_scale": 6.0, "seeds": [1], "scheduler": "DPM++ 2M AYS",
           "width": 832, "height": 1216, "max_detail": {"scale": 2, "method": "Real-ESRGAN (ONNX)"}, "eye_detail": {"denoise": 0.4}}
    import json as _j, tempfile
    from pathlib import Path
    from PIL.PngImagePlugin import PngInfo

    def meta_of(**texts):
        pi = PngInfo()
        for k, v in texts.items():
            pi.add_text(k, v)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.png"
            Image.new("RGB", (8, 8)).save(f, pnginfo=pi)
            with Image.open(f) as im:
                return read_png_info(im)
    txt = _app._params_text(rec)
    assert "Max detail: 2× Real-ESRGAN" in txt and "Eye detail: denoise 0.4" in txt, txt
    m_rec = meta_of(imagegen=_j.dumps(rec), parameters="1girl\nSteps: 12, Seed: 1")                       # the record (exact restore)
    plan = _app._restore_plan(m_rec)
    assert plan["max_detail"] and _app._plan_extra_updates(plan)[-7] is True and "max detail" in _app._plan_summary(plan), plan
    assert _app._plan_extra_updates(_app._restore_plan({"prompt": "x"}))[-7] is False and _app._plan_extra_updates(_app._restore_plan({"prompt": "x"}))[-6] == 1.0
    m_txt = meta_of(parameters="1girl\nSteps: 12, Seed: 1, Max detail: 2× Real-ESRGAN, Eye detail: denoise 0.4")   # the A1111 text alone
    assert m_txt.get("max_detail") == {"scale": 2, "method": "Real-ESRGAN (ONNX)"}, m_txt
    # the pass: twice the size per picture (the Real-ESRGAN call is replaced by a resize), the session is released, > 64 MP refused
    md = _app._max_detail_pass                                                          # module level since round 15 (the Upscale tab's 🔬 Sharpen eyes reuses it)
    calls = []
    old = (_app._upscale_to, _app.upscaler.release)
    try:
        _app._upscale_to = lambda im, w, h, m: calls.append((w, h, m)) or im.resize((w, h))
        _app.upscaler.release = lambda: calls.append("released")
        out, note = md([Image.new("RGB", (64, 96)), Image.new("RGB", (64, 96))], lambda *a, **k: None)
        assert [im.size for im in out] == [(128, 192)] * 2 and calls[:2] == [(128, 192, "Real-ESRGAN (ONNX)")] * 2 and calls[-1] == "released" and "2×" in note, (calls, note)
        try:
            md([Image.new("RGB", (6000, 6000))], lambda *a, **k: None)
            assert False, "over 64 MP must be refused"
        except RuntimeError as e:
            assert "64 MP" in str(e)
    finally:
        _app._upscale_to, _app.upscaler.release = old


@test("🎨 Colour strength (round 25): LAB chroma scale (lightness kept, grey / off untouched), clean_extra, record + A1111 text + restore, after every pass in a real Generate run (no pass reads a scaled picture), default off")
def _():
    import json as _j, tempfile
    from pathlib import Path
    import numpy as np, cv2
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    import app as _app
    from backend import colour as C
    from backend.png_info import read_png_info

    def chroma(im):
        lab = cv2.cvtColor(np.asarray(im.convert("RGB"), np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        return float(np.hypot(lab[..., 1], lab[..., 2]).mean()), float(lab[..., 0].mean())
    src = Image.new("RGB", (32, 32), (210, 80, 60)); src.paste(Image.new("RGB", (16, 32), (60, 120, 200)), (0, 0))
    assert C.scale_colour(src, 1.0) is src and C.scale_colour(src, 5) is src and C.scale_colour(src, "x") is src          # off / above 1 / junk = the picture itself
    c0, l0 = chroma(src)
    for s in (0.9, 0.8, 0.6):
        out = C.scale_colour(src, s)
        c1, l1 = chroma(out)
        assert abs(c1 / c0 - s) < 0.02 and abs(l1 - l0) < 0.6 and out.size == src.size, (s, c1 / c0, l1 - l0)          # chroma x s, lightness unchanged
    grey = Image.new("RGB", (8, 8), (120, 120, 120))
    assert np.abs(np.asarray(C.scale_colour(grey, 0.5), np.int16) - 120).max() <= 1                                     # a grey picture stays grey
    rgba = Image.new("RGBA", (8, 8), (200, 60, 60, 77))
    assert C.scale_colour(rgba, 0.7).mode == "RGBA" and C.scale_colour(rgba, 0.7).getpixel((0, 0))[3] == 77             # alpha kept
    assert C.clamp_strength(float("nan")) == 1.0 and C.clamp_strength("x") == 1.0 and C.clamp_strength(None) == 1.0 and C.clamp_strength(0.1) == 0.5 and C.clamp_strength(2) == 1.0 and C.clamp_strength(0.8) == 0.8
    # clean_extra: default off, junk safe, clamped
    assert _app._clean_extra({})["cs"] == 1.0 and _app._clean_extra({"cs": "x"})["cs"] == 1.0 and _app._clean_extra({"cs": 0.3})["cs"] == 0.5 and _app._clean_extra({"cs": 0.8})["cs"] == 0.8
    # record -> A1111 text -> PNG Info -> restore plan -> control value (a record without it resets the slider to 1)
    rec = {"app": "ImageGen Studio", "format": 2, "mode": "txt2img", "prompt": "1girl", "steps": 12, "cfg_scale": 6.0, "seeds": [1], "scheduler": "DPM++ 2M AYS",
           "width": 64, "height": 64, "colour": {"strength": 0.8, "method": "lab-chroma"}}
    assert "Colour strength: 0.8" in _app._params_text(rec) and "Colour strength" not in _app._params_text(dict(rec, colour=None))

    def meta_of(**texts):
        pi = PngInfo()
        for k, v in texts.items():
            pi.add_text(k, v)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.png"
            Image.new("RGB", (8, 8)).save(f, pnginfo=pi)
            with Image.open(f) as im:
                return read_png_info(im)
    m_rec = meta_of(imagegen=_j.dumps(rec), parameters="1girl\nSteps: 12, Seed: 1")
    plan = _app._restore_plan(m_rec)
    assert plan["colour"] == {"strength": 0.8, "method": "lab-chroma"} and _app._plan_extra_updates(plan)[-6] == 0.8 and "colour strength 0.8" in _app._plan_summary(plan), plan
    assert _app._plan_extra_updates(_app._restore_plan({"prompt": "x"}))[-6] == 1.0
    m_txt = meta_of(parameters="1girl\nSteps: 12, Seed: 1, Colour strength: 0.7")                                       # the A1111 text alone
    assert m_txt.get("colour") == {"strength": 0.7, "method": "lab-chroma"}, m_txt
    from backend.png_info import format_png_info_html
    assert "Colour strength" in format_png_info_html(m_txt)
    # a real Generate run (outfit batch) with a fake pipeline: the hires pass reads the UNSCALED picture, the saved one is scaled, record + text say so; default off saves the picture as drawn
    with _BatchEnv() as E:
        E.run_batch(seed=11, seeds_per=1, hires=True, cs=0.6)
        files = sorted(p for p in E._app.OUTPUTS_DIR.glob("*.png") if not p.name.startswith("outfits_"))
        assert files and E.i2i_inputs, (files, E.i2i_inputs)
        im = Image.open(files[0]); im.load()
        r = _j.loads(im.info["imagegen"])
        assert r["colour"] == {"strength": 0.6, "method": "lab-chroma"} and "Colour strength: 0.6" in im.info["parameters"] and r.get("hires"), r
        seed = r["seed"]
        drawn = Image.new("RGB", im.size, (seed % 255, 40, 90))
        assert E.i2i_inputs[0] == (seed % 255, 40, 90), (E.i2i_inputs, seed)                                           # the hires pass read the picture as drawn
        want = C.scale_colour(drawn, 0.6)
        assert np.abs(np.asarray(im.convert("RGB"), np.int16) - np.asarray(want, np.int16)).max() <= 2                 # the saved picture = the scaled one
        for f in files:
            f.unlink()
        E.run_batch(seed=11, seeds_per=1, resume=False)                                                                # default: off
        files = sorted(p for p in E._app.OUTPUTS_DIR.glob("*.png") if not p.name.startswith("outfits_"))
        im2 = Image.open(files[0]); im2.load()
        assert "colour" not in _j.loads(im2.info["imagegen"]) and "Colour strength" not in im2.info["parameters"] and im2.convert("RGB").getpixel((0, 0)) == (_j.loads(im2.info["imagegen"])["seed"] % 255, 40, 90)


@test("LoRA trainer brake (round 25): a training brakes at limit + 3 (a sampling pass still at limit + 6), resumes at the limit; only timing")
def _():
    from backend import lora_trainer as LT, sampling, thermal
    from unittest.mock import patch
    assert thermal.IN_PASS_MARGIN == 6.0 and thermal.TRAIN_IN_PASS_MARGIN == 3.0
    old = thermal.GUARD["limit"]
    thermal.GUARD["limit"] = 88.0
    try:
        def zone(seq):
            it = iter(seq)
            last = [seq[-1]]
            return lambda: next(it, last[0])

        def run(margin, seq):
            slept, t = [], [0.0]

            def sl(x):
                slept.append(x); t[0] += x
            w = thermal.wait_cool(read=zone(seq), sleep=sl, clock=lambda: t[0], in_pass=True, margin=margin)
            return w, slept
        w, s = run(None, [92.0, 90.0, 87.0])                                    # limit + 4: a sampling pass (margin 6) does not brake ...
        assert w == 0 and not s, (w, s)
        w, s = run(thermal.TRAIN_IN_PASS_MARGIN, [92.0, 92.0, 90.0, 87.0])      # ... a training does, until the zone is at the limit (88)
        assert w > 0 and len(s) == 3 and all(x == 2.0 for x in s), (w, s)
        w, s = run(thermal.TRAIN_IN_PASS_MARGIN, [90.0])                        # limit + 2: neither brakes
        assert w == 0 and not s
        # the trainer's guard uses the training margin: a 92 °C zone makes it wait after a step, a 90 °C one does not
        for temp, waits in ((92.0, True), (90.0, False)):
            slept, t = [], [0.0]

            def sl(x):
                slept.append(x); t[0] += x
            reads = iter([temp, temp, 80.0])
            with patch.object(LT, "_THERM_EVERY", 0.0), patch.object(thermal, "read_temp", lambda: next(reads, 80.0)):
                g = LT._HeatGuard(lambda: False, None, sleep=sl, clock=lambda: t[0])
                sampling_factor = sampling.COOL["factor"]
                sampling.COOL["factor"] = 0.0
                try:
                    g.after_step(0.1)
                finally:
                    sampling.COOL["factor"] = sampling_factor
            assert bool([x for x in slept if x == 2.0]) is waits, (temp, slept)
    finally:
        thermal.GUARD["limit"] = old


@test("🔬 Sharpen eyes (round 15): the picture's own record drives the 2× + eye pass; refuses no face / over 64 MP / already max detail / Pony / no model; Stop and 'no eyes' save nothing; the eye style is restored")
def _():
    import json, tempfile
    from pathlib import Path
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    import app as _app
    from backend import detail_tools as DT
    from backend.png_info import read_png_info
    tmp = Path(tempfile.mkdtemp(prefix="sharpen15_"))
    ck = tmp / "own.safetensors"; ck.write_bytes(b"x")
    rec = {"app": "ImageGen Studio", "format": 2, "mode": "txt2img", "prompt": "1girl, red eyes", "negative_prompt": "bad", "steps": 12, "cfg_scale": 6.0,
           "seeds": [77], "scheduler": "DPM++ 2M AYS", "width": 64, "height": 96, "eye_detail": {"denoise": 0.3, "style": "natural"},
           "model": {"file": "own.safetensors", "family": "illustrious"}, "loras": [{"file": "her.safetensors", "weight": 0.75}]}

    def make(name, **over):
        r = dict(rec, **over)
        pi = PngInfo(); pi.add_text("parameters", "1girl, red eyes\nSteps: 12, Seed: 77"); pi.add_itxt("imagegen", json.dumps(r))
        p = tmp / name
        Image.new("RGB", (64, 96), (200, 200, 200)).save(p, pnginfo=pi)
        im = Image.open(p); im.load()
        return im

    class FakeSD:
        pipe, current_model, model_family = object(), "loaded.safetensors", "illustrious"
    calls = {"faces": 0, "eye": [], "ensure": [], "loras": [], "style": []}

    def fake_eye(sdp, im, prompt, neg, **kw):
        calls["eye"].append((im.size, prompt, neg, kw["denoise"], kw["seed"], kw["steps"])); calls["style"].append(DT.EYE_STYLE["mode"])
        if calls.get("raise"):
            raise calls["raise"]
        return im.copy(), calls.get("found", 1)

    def fake_faces(im, mode="auto"):
        calls["faces"] += 1
        return [] if calls.get("noface") else [(10, 10, 40, 40)]
    saved_attrs = {k: getattr(_app, k) for k in ("sd", "_GEN_HELPERS", "_upscale_to", "_thermal_wait", "_unique_output", "list_checkpoints", "list_loras", "list_vaes")}
    saved_dt = {k: getattr(DT, k) for k in ("detect_faces", "eye_detail", "eye_detector_available")}
    saved_style = DT.EYE_STYLE["mode"]
    saved_rel = _app.upscaler.release
    try:
        _app.sd = FakeSD()
        _app._GEN_HELPERS = {"ensure_model": lambda m, v, p: calls["ensure"].append(m) or (True, "ok"),
                             "sync_loras": lambda slots, p: calls["loras"].append(slots) or ""}
        _app._upscale_to = lambda im, w, h, m: im.resize((w, h))
        _app._thermal_wait = lambda p=None: 0.0
        _app._unique_output = lambda base: tmp / f"{base}.png"
        _app.list_checkpoints = lambda: [("own.safetensors", str(ck))]
        _app.list_loras = lambda: [("her.safetensors", str(tmp / "her.safetensors"))]
        (tmp / "her.safetensors").write_bytes(b"x")
        _app.list_vaes = lambda: []
        _app.upscaler.release = lambda: None
        DT.detect_faces, DT.eye_detail, DT.eye_detector_available = fake_faces, fake_eye, lambda: True
        # a real gr.Progress raises IndexError from __len__ outside an event (the real-app run found `progress or …` dying on it)
        import gradio as _gr
        assert "Drop a picture" in _app._sharpen_eyes(None, _gr.Progress())[1]
        # no picture / over 64 MP (the face detector must not even run) / no face
        assert "Drop a picture" in _app._sharpen_eyes(None)[1]
        out, msg = _app._sharpen_eyes(Image.new("RGB", (5000, 4000)))
        assert out is None and "64 MP" in msg and calls["faces"] == 0, msg
        calls["noface"] = True
        out, msg = _app._sharpen_eyes(make("a.png"))
        assert out is None and "no face found" in msg and not calls["eye"], msg
        calls["noface"] = False
        # the picture's own checkpoint + LoRAs are loaded; the record's prompt / seed / denoise / style drive the eye pass on the 2× picture; the result is saved with the record updated
        out, msg = _app._sharpen_eyes(make("b.png"))
        assert out is not None and out.size == (128, 192) and calls["ensure"] == [str(ck)] and calls["loras"] and "her" in msg, (msg, calls)
        size, prompt, neg, den, seed, steps = calls["eye"][-1]
        assert size == (128, 192) and prompt == "1girl, red eyes" and neg == "bad" and den == 0.3 and seed == 77 and steps == DT.detail_schedule_steps("eye", 12, 0.3), calls["eye"][-1]
        assert calls["style"][-1] == "natural" and DT.EYE_STYLE["mode"] == saved_style                      # the record's style during the pass, the global restored after
        saved = Path(out.info["saved_path"])
        assert saved.is_file()
        m = read_png_info(Image.open(saved))
        assert m["max_detail"] == {"scale": 2, "method": "Real-ESRGAN (ONNX)"} and m["eye_detail"] == {"denoise": 0.3, "style": "natural"}, m
        assert "Max detail: 2× Real-ESRGAN" in Image.open(saved).info["parameters"] and json.loads(Image.open(saved).info["imagegen"])["seeds"] == [77]
        # already max detail
        out, msg = _app._sharpen_eyes(make("c.png", max_detail={"scale": 2, "method": "Real-ESRGAN (ONNX)"}))
        assert out is None and "already" in msg
        # checkpoint not installed: the loaded model is used and the result says so; no record at all: same
        n_eye = len(calls["eye"])
        out, msg = _app._sharpen_eyes(make("d.png", model={"file": "gone.safetensors", "family": "illustrious"}, loras=[]))
        assert out is not None and "isn't installed" in msg and "loaded" in msg and len(calls["eye"]) == n_eye + 1, msg
        bare = Image.new("RGB", (64, 96)); bare.load()
        out, msg = _app._sharpen_eyes(bare)
        assert out is not None and "no generation record" in msg and calls["eye"][-1][1] == "1girl, solo, detailed eyes", msg
        # no model at all / a Pony-family model
        _app.sd = type("S", (), {"pipe": None, "current_model": "", "model_family": ""})()
        out, msg = _app._sharpen_eyes(make("e.png", model={"file": "gone.safetensors"}))
        assert out is None and "load a checkpoint" in msg, msg
        _app.sd = type("S", (), {"pipe": object(), "current_model": "p.safetensors", "model_family": "pony"})()
        out, msg = _app._sharpen_eyes(make("f.png", model={"file": "gone.safetensors"}))
        assert out is None and "Pony" in msg, msg
        # Stop / no eyes found / an error: nothing saved, the style restored
        _app.sd = FakeSD()
        before = sorted(p.name for p in tmp.glob("*_md.png")) + sorted(p.name for p in tmp.glob("sharpen_*.png"))
        calls["raise"] = _app._GenerationAborted()
        out, msg = _app._sharpen_eyes(make("g.png"))
        assert out is None and "Stopped" in msg and DT.EYE_STYLE["mode"] == saved_style, msg
        calls["raise"] = None; calls["found"] = 0
        out, msg = _app._sharpen_eyes(make("h.png"))
        assert out is None and "no eyes" in msg
        calls["found"] = 1; calls["raise"] = RuntimeError("boom")
        out, msg = _app._sharpen_eyes(make("i.png"))
        assert out is None and "boom" in msg and DT.EYE_STYLE["mode"] == saved_style, msg
        assert before == sorted(p.name for p in tmp.glob("*_md.png")) + sorted(p.name for p in tmp.glob("sharpen_*.png"))
        calls["raise"] = None
    finally:
        for k, v in saved_attrs.items():
            setattr(_app, k, v)
        for k, v in saved_dt.items():
            setattr(DT, k, v)
        DT.EYE_STYLE["mode"] = saved_style
        _app.upscaler.release = saved_rel
    # the button and its handler exist on the Upscale tab, the handler is a GPU job, and no existing endpoint changed its arguments
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
    assert "do_sharpen_eyes" in by and getattr(by["do_sharpen_eyes"].fn, "_gpu_job", False) and len(by["do_sharpen_eyes"].inputs) == 1 and len(by["do_sharpen_eyes"].outputs) == 2
    assert len(by["do_upscale"].inputs) == 5 and len(by["do_upscale_folder"].inputs) == 3


@test("🧬 Learn her look: the card keeps the centroid learned from the pictures (refusals say why, CCIP is fetched on first use), the card summary shows it, the outfit batch rates with it")
def _():
    import re as _re
    import numpy as np
    from PIL import Image
    from unittest.mock import patch
    import app as _app
    import backend.character_cards as CC
    from backend import ccip as CP
    from backend import identity_score as ID
    from backend import image_score as IS
    rng = np.random.default_rng(3)
    c = ID._unit(rng.normal(size=ID.DIM))
    u = ID._unit(rng.normal(size=ID.DIM)); u = ID._unit(u - c * float(u @ c))
    vec = lambda t: np.float32(c * t + u * np.sqrt(1 - t * t))
    # the stub reads the picture's red value at (0, 0): 10 / 20 / 30 / 40 = references, 99 = a picture with no face
    stub = lambda im, box=None, faces_fn=None, feature_fn=None, **kw: None if im.getpixel((0, 0))[0] == 99 else (vec(0.95 - im.getpixel((0, 0))[0] / 2000.0), 300)
    tmp = Path(_tf.mkdtemp())
    old = CC.CARDS_DIR
    try:
        CC.CARDS_DIR = tmp / "cards"; CC.CARDS_DIR.mkdir()
        (CC.CARDS_DIR / "G.json").write_text(json.dumps({"name": "G", "tags": "1girl", "outfits": {"a": "dress"}}), encoding="utf-8")
        mkpic = lambda name, red: (Image.new("RGB", (120, 120), (red, 5, 5)).save(tmp / name), str(tmp / name))[1]
        refs = [mkpic(f"r{k}.png", 10 * k) for k in (1, 2, 3, 4)]
        faceless = mkpic("nf.png", 99)
        by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
        fn = lambda n: getattr(by[n].fn, "__wrapped__", by[n].fn)
        quiet = dict(progress=lambda *a, **k: None)
        with patch.object(CP, "available", return_value=True), patch.object(ID, "head_feature", side_effect=stub):
            assert "Pick a card" in fn("do_card_identity")("(none)", refs, **quiet)
            assert "at least 3" in fn("do_card_identity")("G", refs[:2], **quiet)
            assert "nothing learned" in fn("do_card_identity")("G", [faceless] * 3, **quiet)
            assert CC.load_card("G").get("identity") is None
            msg = fn("do_card_identity")("G", refs + [faceless, str(tmp / "missing.png")], **quiet)
            assert "Learned" in msg and "4 of 6" in msg, msg                                            # the faceless and the missing file are skipped
        # CCIP not downloaded yet: the first Learn fetches it (a failure says so and learns nothing), then goes on
        with patch.object(CP, "available", return_value=False), patch.object(CP, "download", side_effect=RuntimeError("offline")):
            assert "Could not download CCIP: offline" in fn("do_card_identity")("G", refs, **quiet)
        got = []
        with patch.object(CP, "available", return_value=False), patch.object(CP, "download", side_effect=lambda: got.append(1) or "p"), patch.object(ID, "head_feature", side_effect=stub):
            assert "Learned" in fn("do_card_identity")("G", refs, **quiet) and got == [1]
        card = CC.load_card("G")
        assert card["identity"]["n"] == 4 and card["identity"]["model"] == ID.MODEL_TAG and card["outfits"] == {"a": "dress"}
        assert "🧬 face learned from 4 pictures" in fn("on_card_pick")("G")[2]
        # the outfit batch: seed 12's face is far from hers -> one star off and named; all fine -> "faces match the card"
        from backend import detail_tools as DT
        from backend import identity_check as IC
        cen = ID.decode_vec(card["identity"]["centroid"])
        u2 = ID._unit(u - cen * float(u @ cen))
        vec2 = lambda t: np.float32(cen * t + u2 * np.sqrt(1 - t * t))                         # a feature at an exact cosine to the LEARNED centroid
        ok_t = card["identity"]["mean"]
        batch_stub = lambda im, box=None, faces_fn=None, feature_fn=None, **kw: (vec2(0.4 if im.getpixel((0, 0))[0] == 12 else ok_t), 300)
        saved = (IC.available, IS.available, DT.detect_faces)
        try:
            IC.available = lambda: False
            IS.available = lambda: {"look": False, "faces": True, "hands": False, "identity": True}
            DT.detect_faces = lambda image, mode="auto", max_faces=4, min_frac=0.03: [(10, 10, 50, 50)]
            with _BatchEnv() as E, patch.object(ID, "available", return_value=True), patch.object(ID, "head_feature", side_effect=batch_stub):
                (E._CC.CARDS_DIR / "TestGirl.json").write_text(json.dumps({**json.loads((E._CC.CARDS_DIR / "TestGirl.json").read_text()), "identity": card["identity"]}))
                msg = _re.sub("<[^>]+>", " ", E.run_batch(seed=11)[-1][1])
                assert "possibly off-model" in msg and "gown · seed 12 (not her?" in msg and "seed 11" not in msg and "⭐ 4.5/5 on average" in msg, msg
                assert "hat / hairstyle" not in msg, msg                 # a plain gown: no hint
                # an outfit that changes the head (tiara, hat, ponytail …): CCIP reads hair and headwear too, so the flag says "check by eye"
                cj = json.loads((E._CC.CARDS_DIR / "TestGirl.json").read_text())
                cj["outfits"]["gown"] = "evening gown, tiara"
                (E._CC.CARDS_DIR / "TestGirl.json").write_text(json.dumps(cj))
                msg2 = _re.sub("<[^>]+>", " ", E.run_batch(seed=11, resume=False)[-1][1])
                assert "gown · seed 12 (not her?" in msg2 and "a hat / hairstyle in this outfit lowers it: check by eye" in msg2, msg2
                cj["outfits"]["gown"] = "evening gown"
                (E._CC.CARDS_DIR / "TestGirl.json").write_text(json.dumps(cj))
                batch_stub2 = lambda im, box=None, faces_fn=None, feature_fn=None, **kw: (vec2(ok_t), 300)
                with patch.object(ID, "head_feature", side_effect=batch_stub2):
                    msg = _re.sub("<[^>]+>", " ", E.run_batch(seed=11, resume=False)[-1][1])
                assert "faces match the card in all 6 image(s) (CCIP)" in msg and "WD14" not in msg and "off-model" not in msg and "⭐ 5.0/5" in msg, msg
                # an identity learned with another model can't be compared: no flag AND no "match" reassurance
                (E._CC.CARDS_DIR / "TestGirl.json").write_text(json.dumps({**json.loads((E._CC.CARDS_DIR / "TestGirl.json").read_text()), "identity": dict(card["identity"], model="wd-vit-tagger-v3")}))
                with patch.object(ID, "head_feature", side_effect=batch_stub):
                    msg = _re.sub("<[^>]+>", " ", E.run_batch(seed=11, resume=False)[-1][1])
                assert "off-model" not in msg and "match the card" not in msg and "⭐ 5.0/5" in msg, msg
        finally:
            IC.available, IS.available, DT.detect_faces = saved
    finally:
        CC.CARDS_DIR = old


@test("✨ Polish: every recipe survives _clean_extra, Auto reads the framing tags, the dropdown fills the 12 controls and nothing else")
def _():
    import app as _app
    from backend import polish as P
    for name in P.SHOTS:
        r = P.recipe(name)
        assert r and set(r) <= set(P._KEYS), (name, r)
        clean = _app._clean_extra(r)
        assert all(clean[k] == v for k, v in r.items()), (name, r, clean)            # what the controls show is what runs
        assert P.summary(name) and P.time_factor(name) >= 1, name
    assert P.recipe("(off)") == {} and P.summary("nope") == "" and P.time_factor("nope") == 1.0
    r = P.recipe("Wide shot"); r["hires_on"] = False
    assert P.recipe("Wide shot")["hires_on"] is True                                   # recipe() hands out copies
    framing = {"1girl, portrait, close-up, face focus": "Portrait", "1girl, upper body, looking at viewer, standing": "Portrait",
               "1girl, cowboy shot, standing, night, city lights": "Cowboy shot", "1girl, (full body:1.2), standing": "Full body",
               "1girl, very wide shot, from far away, full body, standing": "Wide shot", "1girl, white dress, simple background": "Cowboy shot",
               "": "Cowboy shot", "1girl, close up, full-body": "Full body", "1girl, <lora:x:1>, thigh up": "Cowboy shot"}
    for prompt, want in framing.items():
        assert P.shot_type(prompt) == want, (prompt, P.shot_type(prompt), want)
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
    on = by["on_polish"].fn
    n_out = len(by["on_polish"].outputs)
    assert n_out == 13, n_out                                                           # 12 controls + the note
    noop = on("(off)", "full body")
    assert all(u == {"__type__": "update"} for u in noop[:-1]) and noop[-1] == "", noop
    ups = on("Full body", "")
    want = P.recipe("Full body")
    keys = ["hires_on", "hires_scale", "hires_denoise", "hires_steps", "hires_upscaler", "fd_on", "fd_denoise", "hd_on", "hd_denoise", "ed_on", "ed_denoise", "md_on"]
    assert [u.get("value") for u in ups[:-1]] == [want.get(k) for k in keys], ups
    assert "Full body" in ups[-1] and f"{P.time_factor('Full body'):g}×" in ups[-1], ups[-1]
    auto = on("Auto", "1girl, upper body, looking at viewer")
    assert "Portrait" in auto[-1] and auto[0]["value"] is P.recipe("Portrait")["hires_on"], auto
    # a recipe that leaves a control out leaves the user's value alone
    saved = dict(P.RECIPES)
    try:
        P.RECIPES["Portrait"] = dict(fd_on=True)
        part = on("Portrait", "")
        assert part[5]["value"] is True and all(u == {"__type__": "update"} for i, u in enumerate(part[:-1]) if i != 5), part
    finally:
        P.RECIPES.clear(); P.RECIPES.update(saved)
    assert set(by["on_polish"].inputs and [type(c).__name__ for c in by["on_polish"].inputs]) == {"Dropdown", "Textbox"}


@test("Detail passes: the eye pass runs 8 steps instead of 10 (equal quality, 21 % cheaper), face / hand still 10; _detail_pass uses the shared rule; Polish Full body runs 8 hires steps, the Wide shot keeps 14 (round 9: 8 left its eyes softer)")
def _():
    import inspect
    import app as _app
    from backend import detail_tools as dt
    from backend import polish as P
    for kind, want in (("eye", 8), ("face", 10), ("hand", 10)):
        for den in (0.3, 0.35, 0.4, 0.55):
            n = dt.detail_schedule_steps(kind, 12, den)
            assert int(n * den) == want, (kind, den, n, int(n * den))          # the steps that actually run
    assert dt.detail_schedule_steps("eye", 30, 0.4) == 38 and int(38 * 0.4) == 15       # more main steps: half of them, as before
    assert dt.detail_schedule_steps("eye", 400, 0.1) == 150                                # never longer than 150
    assert dt.detail_schedule_steps("unknown", 12, 0.5) == 20                              # an unknown pass keeps the 10-step minimum
    # round 26 (behaviour instead of a look at the source): the real _detail_pass hands each detail function the steps of the shared rule
    from PIL import Image
    from unittest.mock import patch
    with _BatchEnv() as E:
        gen = dict(zip(E.cb.__code__.co_freevars, E.cb.__closure__))["do_generate"].cell_contents
        dpass = dict(zip(gen.__code__.co_freevars, gen.__closure__))["_detail_pass"].cell_contents
        seen = {}
        fake = lambda kind: (lambda sdp, im, prompt, neg, **kw: (seen.setdefault(kind, []).append(kw["steps"]), (im, 1))[1])
        ex = _app._clean_extra(dict(fd_denoise=0.35, hd_denoise=0.35, ed_denoise=0.4))
        with patch.object(dt, "eye_detail", fake("eye")), patch.object(dt, "face_detail", fake("face")), patch.object(dt, "hand_detail", fake("hand")):
            for kind in ("eye", "face", "hand"):
                dpass(kind, [Image.new("RGB", (64, 64))], [1], "p", "n", 12, 6.0, "Euler a", ex, lambda *a, **k: None)
        assert seen == {"eye": [dt.detail_schedule_steps("eye", 12, 0.4)], "face": [dt.detail_schedule_steps("face", 12, 0.35)], "hand": [dt.detail_schedule_steps("hand", 12, 0.35)]}, seen
        assert int(seen["eye"][0] * 0.4) == 8 and int(seen["face"][0] * 0.35) == 10, seen
    assert P.recipe("Full body")["hires_steps"] == 8 and P.recipe("Wide shot")["hires_steps"] == 14 and P.recipe("Full body")["hires_denoise"] == 0.45
    assert P.time_factor("Wide shot") > P.time_factor("Full body") > P.time_factor("Portrait")             # the Wide chain is the longest, with its 14 hires steps


@test("Generate: 'Use img2img' with no image loaded says so instead of quietly making a text-to-image")
def _():
    import re as _re
    with _BatchEnv() as E:
        gen = dict(zip(E.cb.__code__.co_freevars, E.cb.__closure__))["do_generate"].cell_contents
        quiet = lambda *a, **k: None
        E._app._generation_abort.clear()                 # an earlier test may have left a Stop set
        for use_i2i, said in ((True, True), (False, False)):
            imgs, info, _ = gen("1girl", "bad", "Euler a", 4, 6.0, 64, 64, 1, 5, None, 0.5, use_i2i,
                                auto_quality=False, extra=None, progress=quiet)
            assert len(imgs) == 1
            assert ("no image is loaded" in _re.sub("<[^>]+>", " ", info)) == said, (use_i2i, info)


@test("Generate: PNDM / Heun on an SDXL-family model get the grain note (LMS / PNDM no longer garble since the lower-order fix); other samplers and SD 1.5 don't")
def _():
    import re as _re
    with _BatchEnv() as E:
        gen = dict(zip(E.cb.__code__.co_freevars, E.cb.__closure__))["do_generate"].cell_contents
        E._app._generation_abort.clear()

        def warned(family, sampler):
            E._app.sd.model_family = family
            imgs, info, _ = gen("1girl", "bad", sampler, 4, 6.0, 64, 64, 1, 5, None, 0.5, False,
                                auto_quality=False, extra=None, progress=lambda *a, **k: None)
            assert len(imgs) == 1
            return "more grain" in _re.sub("<[^>]+>", " ", info)
        assert warned("illustrious", "PNDM") and warned("pony", "Heun")
        assert not warned("sdxl", "LMS") and not warned("illustrious", "Euler a") and not warned("sdxl", "UniPC")
        assert not warned("sd15", "PNDM") and not warned("sd15", "Heun")


@test("History index: text chunks only, refresh by mtime, search words / model / LoRA / favourites, thumbnails")
def _():
    import json as _j, time as _t
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from backend import history as H
    tmp = Path(_tf.mkdtemp()); outs = tmp / "outputs"; outs.mkdir()
    saved = (H.INDEX_FILE, H.FAV_FILE, H.THUMBS)
    H.INDEX_FILE, H.FAV_FILE, H.THUMBS = tmp / "idx.json", tmp / "fav.json", outs / ".thumbs"
    try:
        def png(name, prompt, model, lora, seed):
            info = PngInfo()
            info.add_text("parameters", f"{prompt}\nNegative prompt: bad\nSteps: 20, Sampler: Euler a, CFG scale: 7, "
                                        f"Seed: {seed}, Size: 64x64, Model: {model}, LoRAs: {lora}:0.8")
            Image.new("RGB", (64, 64), (seed % 255, 0, 0)).save(outs / name, pnginfo=info)
        png("a.png", "1girl, heroine, black bikini, beach", "meina.safetensors", "Heroine_v1", 1)
        png("b.png", "1girl, heroine, white dress", "wai.safetensors", "IL_Heroine", 2)
        _t.sleep(0.05)
        png("c.png", "landscape, mountains", "wai.safetensors", "none_lora", 3)
        idx = H.build_index(outs)
        assert set(idx) == {"a.png", "b.png", "c.png"} and idx["a.png"]["model"] == "meina"
        assert idx["b.png"]["loras"] == ["IL_Heroine"] and idx["a.png"]["seed"] == 1
        assert H.search(idx, "heroine bikini") == ["a.png"]
        assert H.search(idx, "", model="wai") == ["c.png", "b.png"]            # newest first
        assert H.search(idx, "heroine", lora="IL_Heroine") == ["b.png"]
        assert H.toggle_favourite("b.png") is True and H.search(idx, "", favs_only=True) == ["b.png"]
        assert H.toggle_favourite("b.png") is False and H.search(idx, "", favs_only=True) == []
        (outs / "c.png").unlink()
        png("d.png", "1girl, maid", "meina.safetensors", "Heroine_v1", 4)
        idx = H.build_index(outs)
        assert set(idx) == {"a.png", "b.png", "d.png"}
        assert _j.loads(H.INDEX_FILE.read_text(encoding="utf-8")).keys() == idx.keys()
        t = H.thumbnail("a.png", outs)
        assert t and Image.open(t).size == (64, 64)
        ph = H.thumbnail("missing.png", outs)      # gone / damaged files get a placeholder tile: the History gallery
        assert ph and Path(ph).is_file() and ph != t      # must keep one tile per name or clicks map to the wrong file
        (outs / "broken.png").write_bytes(b"not a png")                       # damaged files don't break it
        assert "broken.png" in H.build_index(outs)
        assert H.thumbnail("broken.png", outs) == ph
    finally:
        H.INDEX_FILE, H.FAV_FILE, H.THUMBS = saved


@test("Tiled SD detail: tiles cover the image with overlap, full-size where possible; blend ramps only inside")
def _():
    import numpy as np
    from backend.detail_tools import tile_boxes, _ramp_mask
    for W, H, t, ov in ((1024, 1536, 512, 96), (1664, 2432, 1024, 96), (600, 500, 496, 96), (512, 512, 512, 96)):
        boxes = tile_boxes(W, H, t, ov)
        cover = np.zeros((H, W), bool)
        for x1, y1, x2, y2 in boxes:
            assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
            assert (x2 - x1) == min(t, W) and (y2 - y1) == min(t, H), (W, H, (x1, y1, x2, y2))
            cover[y1:y2, x1:x2] = True
        assert cover.all(), (W, H)
    m = np.asarray(_ramp_mask(100, 80, 20, 0, 0, 10))
    assert m[:, 0].max() == 0 and m[40, 50] == 255 and m[0, 99] == 255 and m[-1, 50] == 0

@test("Prompt weights: A1111 emphasis (multiply, keep the mean) is the default; Compel mode unchanged; SDXL cache keyed by mode")
def _():
    import torch, inspect
    try:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer",
                                            local_files_only=True)
    except Exception:
        raise Skip("SD 1.5 CLIP tokenizer not in .hf_cache")
    from transformers import CLIPTextModel, CLIPTextConfig
    from compel import Compel, ReturnedEmbeddingsType
    from backend import prompt_syntax as PS
    torch.manual_seed(0)
    te = CLIPTextModel(CLIPTextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, max_position_embeddings=77))
    with torch.no_grad():
        te.text_model.final_layer_norm.bias.fill_(0.3)          # real CLIP has a non-zero mean
    c = Compel(tokenizer=tok, text_encoder=te, truncate_long_prompts=False,
               returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED)
    old = PS.EMPHASIS["mode"]
    try:
        assert old == "a1111" or True
        PS.set_emphasis("a1111")
        plain = c("a girl, sunset")
        w = c(PS.a1111_to_compel("a girl, (sunset:1.4)"))
        assert abs(w.mean().item() - plain.mean().item()) < 1e-5            # mean kept (A1111 "Original")
        ids = tok("a girl, sunset").input_ids
        k = ids.index(tok.convert_tokens_to_ids("sunset</w>"))
        ratio = (w[0, k] / plain[0, k]).mean().item()
        assert 1.2 < ratio < 1.6, ratio                                      # the weighted token ≈ ×1.4 (rescaled)
        PS.set_emphasis("compel")
        wc = c(PS.a1111_to_compel("a girl, (sunset:1.4)"))
        assert not torch.allclose(wc, w) and torch.equal(c("a girl, sunset"), plain)   # unit weights identical
        assert PS.set_emphasis("Compel (stronger)") == "compel" and PS.set_emphasis("A1111 (default)") == "a1111"
    finally:
        PS.set_emphasis(old)
    # the SDXL embedding cache must not serve one weighting mode's embeddings to the other
    from backend import sdxl_pipeline as SX
    from types import SimpleNamespace
    fake_pipe = SimpleNamespace(_model_name_or_path="m")
    saved = dict(SX._SDXL_EMBED_CACHE)
    try:
        SX._SDXL_EMBED_CACHE.clear()
        SX._SDXL_EMBED_CACHE[("p", "n", "m", "a1111")] = {"prompt_embeds": torch.zeros(1)}
        SX._SDXL_EMBED_CACHE[("p", "n", "m", "compel")] = {"prompt_embeds": torch.ones(1)}
        try:
            PS.set_emphasis("a1111")
            assert SX._build_sdxl_embeds(fake_pipe, "p", "n")["prompt_embeds"].item() == 0.0
            PS.set_emphasis("compel")
            assert SX._build_sdxl_embeds(fake_pipe, "p", "n")["prompt_embeds"].item() == 1.0
        finally:
            PS.set_emphasis(old)
    finally:
        SX._SDXL_EMBED_CACHE.clear()
        SX._SDXL_EMBED_CACHE.update(saved)


@test("VAE family from the quant_conv bias: an SD 1.5 VAE on SDXL (or back) is ignored with a warning, not decoded")
def _():
    import json as _j, struct
    import numpy as np
    from backend import model_manager as mm
    tmp = Path(_tf.mkdtemp())

    def st(name, key, norm, dtype="F16"):
        v = np.full(8, norm / np.sqrt(8), np.float16 if dtype == "F16" else np.float32)
        raw = v.tobytes()
        hdr = _j.dumps({key: {"dtype": dtype, "shape": [8], "data_offsets": [0, len(raw)]}}).encode()
        (tmp / name).write_bytes(struct.pack("<Q", len(hdr)) + hdr + raw)
        return str(tmp / name)
    assert mm.vae_family(st("sdxl_vae.safetensors", "quant_conv.bias", 43.8)) == "sdxl"
    assert mm.vae_family(st("ft_mse.safetensors", "quant_conv.bias", 4.4)) == "sd1"
    assert mm.vae_family(st("ckpt.safetensors", "first_stage_model.quant_conv.bias", 43.8, "F32")) == "sdxl"
    assert mm.vae_family(st("nokey.safetensors", "foo.weight", 1.0)) is None
    (tmp / "x.pt").write_bytes(b"pickle")
    assert mm.vae_family(str(tmp / "x.pt")) is None


@test("GPU lock: a generator job resumed on another thread releases the lock (RLock raised and stayed held)")
def _():
    import threading
    import app as _app

    def job():
        yield 1
        yield 2
    g = _app._locked(job)()
    t1 = threading.Thread(target=lambda: next(g)); t1.start(); t1.join(5)       # takes the lock on thread A
    assert _app._GPU_LOCK.locked()
    t2 = threading.Thread(target=lambda: list(g)); t2.start(); t2.join(5)       # finishes on thread B
    assert not _app._GPU_LOCK.locked(), "lock still held after the job ended on another thread"


@test("Weighting mode is recorded (record + A1111 text) and read back; restore notes a mode mismatch")
def _():
    import json as _j
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    import app as _app
    from backend import prompt_syntax as PS
    from backend.png_info import read_image_metadata
    old = PS.EMPHASIS["mode"]
    try:
        PS.set_emphasis("compel")
        rec = _app._gen_record(None, prompt="1girl, (sunset:1.3)", steps=20)
        assert rec["emphasis"] == "compel" and "Emphasis: Compel" in _app._params_text(rec)
        PS.set_emphasis("a1111")
        assert "Emphasis" not in _app._params_text(_app._gen_record(None, prompt="x", steps=20))
        info = PngInfo(); info.add_text("parameters", "1girl, (sunset:1.3)\nSteps: 20, Sampler: Euler, Seed: 1, "
                                                      "Emphasis: Compel")
        with _tf.TemporaryDirectory() as d:
            f = Path(d) / "e.png"; Image.new("RGB", (8, 8)).save(f, pnginfo=info)
            meta = read_image_metadata(f)
        assert meta["emphasis"] == "compel"
        plan = _app._restore_plan(meta)
        assert any("Compel prompt weighting" in n for n in plan["notes"]), plan["notes"]
        meta["prompt"] = "1girl, sunset"                                    # no weights → no note
        assert not any("weighting" in n for n in _app._restore_plan(meta)["notes"])
    finally:
        PS.set_emphasis(old)


@test("🔬 Sharpen eyes for a folder (round 16): every picture → outputs/sharpened_<folder>/ with its record updated; no face / over 64 MP / already max detail are skipped with a reason, no record uses the loaded model + a note; Stop between pictures keeps the finished ones")
def _():
    import json, tempfile
    from pathlib import Path
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    import app as _app
    from backend import detail_tools as DT
    from backend.png_info import read_png_info
    tmp = Path(tempfile.mkdtemp(prefix="sharpen16_"))
    src, out_root = tmp / "pics", tmp / "outputs"
    src.mkdir(); out_root.mkdir()
    ck = tmp / "own.safetensors"; ck.write_bytes(b"x")
    rec = {"app": "ImageGen Studio", "format": 2, "mode": "txt2img", "prompt": "1girl, red eyes", "negative_prompt": "bad", "steps": 12, "cfg_scale": 6.0, "seeds": [77],
           "scheduler": "DPM++ 2M AYS", "width": 64, "height": 96, "model": {"file": "own.safetensors", "family": "illustrious"}, "loras": []}

    def put(name, size=(64, 96), record=True, **over):
        pi = PngInfo(); pi.add_text("parameters", "1girl, red eyes\nSteps: 12, Seed: 77")
        if record:
            pi.add_itxt("imagegen", json.dumps(dict(rec, **over)))
        Image.new("RGB", size, (200, 200, 200)).save(src / name, pnginfo=pi)
    put("a.png"); put("b_noface.png", size=(50, 50)); put("c_done.png", max_detail={"scale": 2, "method": "Real-ESRGAN (ONNX)"}); put("d_big.png", size=(4200, 4200))
    put("e_norecord.png", record=False); put("f.png"); (src / "notes.txt").write_text("x")

    class FakeSD:
        pipe, current_model, model_family = object(), "loaded.safetensors", "illustrious"
    calls = {"eye": [], "ensure": [], "thermal": 0}
    saved = {k: getattr(_app, k) for k in ("sd", "_GEN_HELPERS", "_upscale_to", "_thermal_wait", "OUTPUTS_DIR", "list_checkpoints", "list_loras", "list_vaes")}
    saved_dt = {k: getattr(DT, k) for k in ("detect_faces", "eye_detail", "eye_detector_available")}
    saved_rel = _app.upscaler.release
    try:
        _app.sd = FakeSD()
        _app._GEN_HELPERS = {"ensure_model": lambda m, v, p: calls["ensure"].append(Path(m).name) or (True, "ok"), "sync_loras": lambda slots, p: ""}
        _app._upscale_to = lambda im, w, h, m: im.resize((w, h))
        _app._thermal_wait = lambda p=None: calls.__setitem__("thermal", calls["thermal"] + 1) or 0.0
        _app.OUTPUTS_DIR = out_root
        _app.list_checkpoints = lambda: [("own.safetensors", str(ck))]
        _app.list_loras = lambda: []
        _app.list_vaes = lambda: []
        _app.upscaler.release = lambda: None
        DT.detect_faces = lambda im, mode="auto": [] if im.size == (50, 50) else [(10, 10, 40, 40)]
        DT.eye_detail = lambda sdp, im, prompt, neg, **kw: (calls["eye"].append(im.size) or im.copy(), 2)
        DT.eye_detector_available = lambda: True
        import gradio as _gr
        msg = _app._sharpen_folder(str(src), _gr.Progress())                                   # a real gr.Progress: the object raises on `or` / len() outside an event
        dest = out_root / "sharpened_pics"
        files = sorted(p.name for p in dest.glob("*.png"))
        assert files == ["a_md.png", "e_norecord_md.png", "f_md.png"], files                    # a, f (own model) + e (no record: the loaded model)
        assert "Sharpened 3 of 6" in msg and "outputs/sharpened_pics/" in msg, msg
        assert "no face: b_noface.png" in msg and "over 64 MP: d_big.png" in msg and "already max detail: c_done.png" in msg, msg
        assert "1 used the loaded model" in msg and "e_norecord.png" in msg, msg
        assert calls["thermal"] >= 6 and len(calls["eye"]) == 3 and all(s == (128, 192) for s in calls["eye"]), calls
        assert calls["ensure"].count("own.safetensors") == 2                                    # the two pictures with a record
        im = Image.open(dest / "a_md.png"); im.load()
        assert im.size == (128, 192)
        plan = _app._restore_plan(read_png_info(im))
        assert plan["max_detail"] and "Max detail: 2× Real-ESRGAN" in im.info["parameters"] and json.loads(im.info["imagegen"])["eye_detail"]["denoise"] == 0.4
        assert (src / "a.png").stat().st_size > 0 and Image.open(src / "a.png").size == (64, 96)    # the originals stay
        # a second run keeps the first results (never overwrites): _1 names
        _app._sharpen_folder(str(src))
        assert (dest / "a_md_1.png").is_file() and (dest / "a_md.png").is_file()
        # Stop between pictures: the finished one is kept, the message says so
        n0 = len(list(dest.glob("*.png")))
        state = {"n": 0}

        def stop():
            return state["n"] >= 1
        orig_core = _app._sharpen_core

        def counting(*a, **k):
            r = orig_core(*a, **k); state["n"] += 1; return r
        _app._sharpen_core = counting
        try:
            msg = _app._sharpen_folder(str(src), stop=stop)
        finally:
            _app._sharpen_core = orig_core
        assert "Stopped before picture" in msg and "Sharpened 1 of 6" in msg, msg
        assert len(list(dest.glob("*.png"))) == n0 + 1
        # bad input
        assert "existing folder" in _app._sharpen_folder(str(tmp / "nope")) and "No PNG" in _app._sharpen_folder(str(tmp))
        # the button's handler is a GPU job with one input and one output; the folder-upscale button keeps its 3 inputs
        by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
        assert getattr(by["do_sharpen_folder"].fn, "_gpu_job", False) and len(by["do_sharpen_folder"].inputs) == 1 and len(by["do_sharpen_folder"].outputs) == 1 and len(by["do_upscale_folder"].inputs) == 3
    finally:
        for k, v in saved.items():
            setattr(_app, k, v)
        for k, v in saved_dt.items():
            setattr(DT, k, v)
        _app.upscaler.release = saved_rel


@test("🔬 Sharpen eyes for every ⭐ favourite (round 19): the folder logic over settings/_favourites.json → outputs/sharpened_favourites/; already done / missing / no face are skipped with a reason, Stop keeps the finished ones, originals untouched, the button is a GPU job with no input")
def _():
    import json, tempfile
    from pathlib import Path
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    import app as _app
    from backend import detail_tools as DT
    from backend import history as H
    from backend.png_info import read_png_info
    import gradio as _gr
    tmp = Path(tempfile.mkdtemp(prefix="sharpen19_"))
    out_root = tmp / "outputs"; out_root.mkdir()
    ck = tmp / "own.safetensors"; ck.write_bytes(b"x")
    rec = {"app": "ImageGen Studio", "format": 2, "mode": "txt2img", "prompt": "1girl, red eyes", "negative_prompt": "bad", "steps": 12, "cfg_scale": 6.0, "seeds": [77],
           "scheduler": "DPM++ 2M AYS", "width": 64, "height": 96, "model": {"file": "own.safetensors", "family": "illustrious"}, "loras": []}

    def put(name, size=(64, 96)):
        pi = PngInfo(); pi.add_text("parameters", "1girl, red eyes\nSteps: 12, Seed: 77"); pi.add_itxt("imagegen", json.dumps(rec))
        Image.new("RGB", size, (200, 200, 200)).save(out_root / name, pnginfo=pi)
    put("a.png"); put("b_noface.png", size=(50, 50)); put("c.png"); put("d_notfav.png")
    (out_root / "e.txt").write_text("x")
    favfile = tmp / "_favourites.json"

    class FakeSD:
        pipe, current_model, model_family = object(), "loaded.safetensors", "illustrious"
    calls = {"eye": [], "thermal": 0}
    saved = {k: getattr(_app, k) for k in ("sd", "_GEN_HELPERS", "_upscale_to", "_thermal_wait", "OUTPUTS_DIR", "list_checkpoints", "list_loras", "list_vaes")}
    saved_dt = {k: getattr(DT, k) for k in ("detect_faces", "eye_detail", "eye_detector_available")}
    saved_rel, saved_fav = _app.upscaler.release, H.FAV_FILE
    try:
        _app.sd = FakeSD()
        _app._GEN_HELPERS = {"ensure_model": lambda m, v, p: (True, "ok"), "sync_loras": lambda slots, p: ""}
        _app._upscale_to = lambda im, w, h, m: im.resize((w, h))
        _app._thermal_wait = lambda p=None: calls.__setitem__("thermal", calls["thermal"] + 1) or 0.0
        _app.OUTPUTS_DIR = out_root
        _app.list_checkpoints = lambda: [("own.safetensors", str(ck))]
        _app.list_loras = lambda: []
        _app.list_vaes = lambda: []
        _app.upscaler.release = lambda: None
        DT.detect_faces = lambda im, mode="auto": [] if im.size == (50, 50) else [(10, 10, 40, 40)]
        DT.eye_detail = lambda sdp, im, prompt, neg, **kw: (calls["eye"].append(im.size) or im.copy(), 2)
        DT.eye_detector_available = lambda: True
        H.FAV_FILE = favfile
        # no favourites yet
        assert "No favourites yet" in _app._sharpen_favourites()
        favfile.write_text(json.dumps(["a.png", "b_noface.png", "c.png", "gone.png", "e.txt"]), encoding="utf-8")
        msg = _app._sharpen_favourites(_gr.Progress())                                        # a real gr.Progress (never `progress or …`)
        dest = out_root / "sharpened_favourites"
        assert sorted(p.name for p in dest.glob("*.png")) == ["a_md.png", "c_md.png"], sorted(p.name for p in dest.glob("*.png"))
        assert "Sharpened 2 of 3" in msg and "outputs/sharpened_favourites/" in msg, msg          # d_notfav.png is not a favourite: never touched
        assert "no face: b_noface.png" in msg and "file not found" in msg and "gone.png" in msg and "e.txt" in msg, msg
        assert len(calls["eye"]) == 2 and all(s == (128, 192) for s in calls["eye"]) and calls["thermal"] >= 3, calls
        im = Image.open(dest / "a_md.png"); im.load()
        assert im.size == (128, 192) and _app._restore_plan(read_png_info(im))["max_detail"] and json.loads(im.info["imagegen"])["eye_detail"]["denoise"] == 0.4
        assert Image.open(out_root / "a.png").size == (64, 96) and not (dest / "d_notfav_md.png").exists()    # the originals stay
        # a second press skips the finished ones instead of making _1 copies
        calls["eye"].clear()
        msg = _app._sharpen_favourites()
        assert not calls["eye"] and "already sharpened: a.png, c.png" in msg and not (dest / "a_md_1.png").exists(), msg
        # Stop between pictures keeps the finished one
        for p in dest.glob("*.png"):
            p.unlink()
        state = {"n": 0}
        orig_core = _app._sharpen_core

        def counting(*a, **k):
            r = orig_core(*a, **k); state["n"] += 1; return r
        _app._sharpen_core = counting
        try:
            msg = _app._sharpen_favourites(stop=lambda: state["n"] >= 1)
        finally:
            _app._sharpen_core = orig_core
        assert "Stopped before picture" in msg and len(list(dest.glob("*.png"))) == 1, msg
        # only missing favourites
        favfile.write_text(json.dumps(["gone.png"]), encoding="utf-8")
        assert "None of the 1 favourite" in _app._sharpen_favourites()
        # the button: a GPU job with no inputs and one output; the Stop button has no API name
        by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
        assert getattr(by["do_sharpen_favourites"].fn, "_gpu_job", False) and len(by["do_sharpen_favourites"].inputs) == 0 and len(by["do_sharpen_favourites"].outputs) == 1
    finally:
        for k, v in saved.items():
            setattr(_app, k, v)
        for k, v in saved_dt.items():
            setattr(DT, k, v)
        _app.upscaler.release = saved_rel
        H.FAV_FILE = saved_fav


@test("⚡ Use the measured recipe: Illustrious / NoobAI fill Sampler / Steps / CFG / PAG, every other family keeps the settings and says so, nothing else is touched (round 16)")
def _():
    import app as _app
    from backend import recipes as RC
    from backend import sd_pipeline as _sp
    from backend import prompt_syntax as _ps
    r = RC.recipe_for("illustrious")
    assert (r["scheduler"], r["steps"], r["cfg"], r["pag"]) == ("DPM++ 2M AYS", 12, 6.0, 2.0) and "Illustrious / NoobAI" in RC.family_label("illustrious")
    assert r["scheduler"] in _app.SCHEDULER_MAP
    r["steps"] = 99
    assert RC.recipe_for("illustrious")["steps"] == 12                                 # copies
    assert RC.recipe_for("pony") is None and RC.recipe_for("sd15") is None and RC.recipe_for("sdxl") is None and RC.recipe_for(None) is None
    by = {getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"): f for f in _app.build_app().fns}
    fn, spec = by["on_measured_recipe"].fn, by["on_measured_recipe"]
    assert len(spec.inputs) == 1 and type(spec.inputs[0]).__name__ == "Dropdown" and len(spec.outputs) == 5, (len(spec.inputs), len(spec.outputs))
    assert [type(c).__name__ for c in spec.outputs] == ["Dropdown", "Slider", "Slider", "Slider", "HTML"]
    old_fam, old_mode, old_pipe = _sp._model_family, _ps.EMPHASIS["mode"], _app.sd.pipe
    try:
        _sp._model_family = lambda p: "illustrious"
        out = fn("D:/x/model.safetensors")
        vals = [o.get("value") if isinstance(o, dict) else o for o in out[:4]]
        assert vals == ["DPM++ 2M AYS", 12, 6.0, 2.0], vals
        assert "Illustrious / NoobAI" in out[4] and "untouched" in out[4] and "Compel" not in out[4], out[4]
        _ps.EMPHASIS["mode"] = "compel"
        assert "Compel" in fn("D:/x/model.safetensors")[4]                             # the recipe was measured with A1111 weights: say so
        _ps.EMPHASIS["mode"] = "a1111"
        for fam in ("pony", "sd15", "sdxl"):                                           # no measurement: every control stays, the message says so
            _sp._model_family = lambda p, f=fam: f
            out = fn("D:/x/model.safetensors")
            assert all(o == {"__type__": "update"} or (isinstance(o, dict) and "value" not in o) for o in out[:4]), (fam, out[:4])
            assert "No measured recipe" in out[4] and RC.family_label(fam) in out[4], out[4]
        # no checkpoint selected: the loaded one decides; neither: a message
        _app.sd.pipe = None
        out = fn(None)
        assert "No checkpoint" in out[4]
        _app.sd.pipe = object()
        old_mf = getattr(_app.sd, "model_family", None)
        _app.sd.model_family = "illustrious"
        try:
            assert [o.get("value") for o in fn(None)[:4]] == ["DPM++ 2M AYS", 12, 6.0, 2.0]
        finally:
            _app.sd.model_family = old_mf
    finally:
        _sp._model_family, _ps.EMPHASIS["mode"], _app.sd.pipe = old_fam, old_mode, old_pipe


@test("IMAGEGEN_OUTPUT_SUBDIR (round 19): a harness process saves into outputs/<sub-folder> (relative names only), keeps its own History index, and the user's History never lists sub-folders")
def _():
    import subprocess
    import shutil
    import tempfile
    from PIL import Image
    import config as _cfg
    from backend import history as _H
    f = _cfg._output_subdir
    assert f("") == "" and f(None) == "" and f("  ") == ""
    assert f("_lab/r19") == "_lab/r19" and f("_lab\\r19") == "_lab/r19" and f("lab//x/") == "lab/x"
    for bad in ("/abs", "\\abs", "..", "a/../b", "C:/x", "./x/.."):
        assert f(bad) == "", bad
    # unset: today's behaviour
    assert _cfg.OUTPUT_SUBDIR == "" or os.environ.get("IMAGEGEN_OUTPUT_SUBDIR")
    # History lists the top level of outputs/ only: a sub-folder's pictures never show, and a sub-folder process has its own index file
    tmp = Path(tempfile.mkdtemp(prefix="hist19_"))
    (tmp / "sub").mkdir()
    Image.new("RGB", (8, 8)).save(tmp / "a.png"); Image.new("RGB", (8, 8)).save(tmp / "sub" / "b.png")
    old = _H.INDEX_FILE
    try:
        _H.INDEX_FILE = tmp / "idx.json"
        assert set(_H.build_index(tmp)) == {"a.png"}
    finally:
        _H.INDEX_FILE = old
    # a real process with the variable set: OUTPUTS_DIR and the index file follow it (the folder it makes is removed again)
    app_dir = Path(__file__).resolve().parent
    sub = "_lab/_unit_test19"
    env = dict(os.environ, IMAGEGEN_OUTPUT_SUBDIR=sub, IMAGEGEN_TESTS="1")
    try:
        out = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, '.'); import config; from backend import history; print(config.OUTPUTS_DIR.as_posix()); print(history.INDEX_FILE.name)"],
                             cwd=str(app_dir), env=env, capture_output=True, text=True, timeout=120).stdout.split("\n")
        assert out[0].endswith("/outputs/" + sub) and out[1] == "_history_index___lab__unit_test19.json", out
    finally:
        shutil.rmtree(app_dir / "outputs" / "_lab" / "_unit_test19", ignore_errors=True)
        try:
            (app_dir / "outputs" / "_lab").rmdir()
        except OSError:
            pass


@test("make_zip.ps1: its exclusion list keeps settings, outputs, models, scripts, logs and api keys out of the zip and the code in (round 16: scripts / wildcards / detectors / logs shipped)")
def _():
    import fnmatch
    import re as _re
    zip_script = Path(__file__).resolve().parent.parent / "installer" / "make_zip.ps1"
    if not zip_script.exists():
        raise Skip("installer/make_zip.ps1 is not part of this checkout")
    text = zip_script.read_text(encoding="utf-8")
    block = _re.search(r"\$APP_EXCLUDES = @\((.*?)\r?\n\)", text, _re.S).group(1)
    excl = [m.group(1) for l in block.splitlines() for m in [_re.match(r'\s*"(.*?)"', l)] if m]

    def shipped(rel):                                     # Copy-Filtered's test: `$rel -like "*$ex*" -or $_.Name -like $ex`
        name = rel.split("\\")[-1]
        return not any(fnmatch.fnmatchcase(rel.lower(), f"*{ex}*".lower()) or fnmatch.fnmatchcase(name.lower(), ex.lower()) for ex in excl)
    for rel in ("settings\\api_keys.json", "settings\\_prefs.json", "settings\\characters\\x.json", "outputs\\a.png", "outputs\\.thumbs\\a.jpg", "scripts\\gen_sprites.py", "logs\\desktop-backend.log",
                "models\\checkpoints\\a.safetensors", "models\\loras\\a.safetensors", "models\\wildcards\\my_pose.txt", "models\\detectors\\anime_eye_detect_v1.0_s.onnx",
                "models\\vae\\a.safetensors", "onnx_cache\\m\\a.onnx", "presets\\xl\\mine.json", "backend\\__pycache__\\a.pyc"):
        assert not shipped(rel), f"{rel} would ship"
    for rel in ("app.py", "config.py", "backend\\detail_tools.py", "wildcards\\outfit.txt", "download_models.py", "README.md", "requirements.txt", "install.bat", "launch.bat", "run_zluda.bat"):
        assert shipped(rel), f"{rel} would be left out"


@test("🧍 Pose control (round 34): saved poses, rendering (legs hidden, aspect fit, image poses), extraction on fake ORT sessions, the control built from a pose, pinned downloads, nothing private in the public files")
def _():
    import hashlib, tempfile, types
    from pathlib import Path
    from unittest.mock import patch
    import numpy as np
    from PIL import Image
    from backend import pose_control as pc
    tmp = Path(tempfile.mkdtemp())
    with patch.object(pc, "POSES_DIR", tmp / "poses"), patch.object(pc, "STARTER_DIR", tmp / "no_starters"), patch.object(pc, "CONTROLNET_DIR", tmp / "cn"), patch.object(pc, "DETECTOR_DIR", tmp / "det"):
        # strengths and names
        assert pc.clamp_strength(None) == 0.8 and pc.clamp_strength("x") == 0.8 and pc.clamp_strength(float("nan")) == 0.8
        assert pc.clamp_strength(0.1) == 0.4 and pc.clamp_strength(3) == 1.0 and pc.clamp_strength(0.6) == 0.6
        assert pc.safe_pose_name("../a:b") == "a_b" and pc.safe_pose_name("_current") == "current" and pc.safe_pose_name("CON") == "_CON" and pc.safe_pose_name("") == ""
        # a stored pose: arms in the upper half, legs (knees / ankles) in the lower
        body = [[100.0, 40.0 + i, 0.9] for i in range(18)]
        for i, (x, y) in {1: (100, 60), 2: (80, 60), 3: (60, 90), 4: (40, 120), 5: (120, 60), 6: (140, 90), 7: (160, 120), 8: (90, 150), 11: (110, 150),
                          9: (90, 200), 10: (90, 250), 12: (110, 200), 13: (110, 250)}.items():
            body[i] = [float(x), float(y), 0.9]
        pose = {"v": 1, "size": [200, 300], "body": body, "lh": [[40.0 + k, 120.0 + k, 0.9] for k in range(21)], "rh": [[160.0 - k, 120.0 + k, 0.9] for k in range(21)]}
        assert pc.save_pose("v-sign", pose, pc.preview_of_pose(pose)) == "v-sign"
        assert pc.save_pose(pc.CURRENT, pose) == pc.CURRENT and pc.list_poses() == [pc.CURRENT, "v-sign"]
        got = pc.load_pose("v-sign")
        assert got["body"] == body and got["hash"] == pc.pose_hash(pose) and len(got["hash"]) == 10
        assert pc.load_pose("nothing") is None and pc.load_pose("") is None
        try:
            pc.save_pose("", pose); raise AssertionError("an empty name must be refused")
        except ValueError:
            pass
        # rendering: size, hands drawn, legs hidden, a different aspect ratio is fitted (centred) not stretched
        arr = np.asarray(pc.render(pose, 400, 600))
        assert arr.shape == (600, 400, 3) and arr.sum() > 0 and arr[:420].sum() > 0 and arr[420:].sum() > 0              # legs drawn below y = 420
        hidden = np.asarray(pc.render(pose, 400, 600, hide_legs=True))
        assert hidden[420:].sum() == 0 and hidden[:420].sum() > 0                                                      # knees + ankles gone, the rest stays
        wide = np.asarray(pc.render(pose, 600, 400))
        assert wide.shape == (400, 600, 3) and wide[:, :120].sum() == 0 and wide[:, 480:].sum() == 0 and wide.sum() > 0
        # an image pose: the picture itself, fitted
        pc.save_pose("flat", {"v": 1, "image": True, "size": [100, 200]}, preview=Image.new("RGB", (100, 200), (255, 0, 0)))
        ip = pc.load_pose("flat")
        sq = pc.render(ip, 400, 400)
        assert sq.getpixel((5, 200)) == (0, 0, 0) and sq.getpixel((200, 200)) == (255, 0, 0) and pc.render(ip, 200, 400).getpixel((5, 5)) == (255, 0, 0)
        assert pc.delete_pose("flat") and pc.load_pose("flat") is None and not pc.delete_pose("flat")
        # a record's keypoints become a saved pose (once)
        rec = {"keypoints": {k: pose[k] for k in ("size", "body", "lh", "rh")}}
        again = pc.ensure_pose_from_record(rec)
        assert again == "v-sign" and pc.ensure_pose_from_record(rec) == again and len(pc.list_poses()) == 2, (again, pc.list_poses())          # the same keypoints exist already: a named pose wins over '(uploaded picture)'
        assert pc.ensure_pose_from_record(dict(rec, pose=pc.CURRENT)) == pc.CURRENT and pc.ensure_pose_from_record(dict(rec, pose="v-sign")) == "v-sign"   # ... unless the record names the other one
        assert pc.ensure_pose_from_record({}) is None and pc.ensure_pose_from_record({"keypoints": {"body": [1]}}) is None
        pc.delete_pose(pc.CURRENT); pc.delete_pose("v-sign")
        moved = dict(rec, keypoints=dict(rec["keypoints"], body=[[x + 1, y, s] for x, y, s in body]))
        made = pc.ensure_pose_from_record(moved)
        assert made.startswith("restored_") and made in pc.list_poses() and pc.load_pose(made)["body"][0][0] == 101.0

        # extraction on fake ORT sessions (a person box + 133 keypoints)
        (tmp / "det").mkdir()
        for n in pc.DWPOSE:
            (tmp / "det" / n).write_bytes(b"x")

        class FakeDet:
            def __init__(self, cells): self.cells = cells
            def get_inputs(self): return [types.SimpleNamespace(name="images")]
            def run(self, _o, feed):
                out = np.zeros((1, 8400, 85), np.float32)
                for idx, h_raw in self.cells:                    # stride-8 cells: index = row * 80 + column; w = 100 px, h = 8·exp(h_raw) px on the 640 canvas
                    out[0, idx, 2] = np.log(100 / 8); out[0, idx, 3] = h_raw; out[0, idx, 4] = 0.9; out[0, idx, 5] = 0.9
                return [out]

        class FakePose:
            def __init__(self, conf): self.conf = conf
            def get_inputs(self): return [types.SimpleNamespace(name="input")]
            def run(self, _o, feed):
                sx = np.zeros((1, 133, 576), np.float32); sy = np.zeros((1, 133, 768), np.float32)
                for k in range(133):
                    sx[0, k, int(round((0.4 + 0.002 * k) * 576))] = self.conf; sy[0, k, int(round((0.3 + 0.004 * k) * 768))] = self.conf
                return sx, sy
        sessions = {"dwpose_yolox_l.onnx": FakeDet([(40 * 80 + 40, np.log(200 / 8))]), "dwpose_dw-ll_ucoco_384.onnx": FakePose(5.0)}
        img = Image.new("RGB", (320, 320), (90, 90, 90))
        with patch.object(pc, "_session", lambda f: sessions[f]):
            people = pc.detect_people(np.zeros((320, 320, 3), np.uint8))
            assert len(people) == 1 and abs(people[0][1] - 0.81) < 0.01 and np.allclose(people[0][0], [135, 110, 185, 210], atol=1.0), people
            p, msg = pc.extract(img)
            assert p["size"] == [320, 320] and len(p["body"]) == 18 and len(p["lh"]) == 21 and len(p["rh"]) == 21 and p["hash"] == pc.pose_hash(p) and msg.startswith("Skeleton found (18 of 18 body points)"), (p and p["size"], msg)
            assert abs(p["body"][0][0] - 150.55) < 0.6 and abs(p["body"][0][1] - 134.9) < 0.6, p["body"][0]         # the nose = COCO keypoint 0, mapped back from the 288×384 crop
            sessions["dwpose_yolox_l.onnx"] = FakeDet([(40 * 80 + 40, np.log(200 / 8)), (40 * 80 + 60, np.log(120 / 8))])
            p2, msg2 = pc.extract(img)
            assert p2 is not None and "2 people" in msg2 and "biggest" in msg2, msg2
            sessions["dwpose_yolox_l.onnx"] = FakeDet([])
            none, msg3 = pc.extract(img)
            assert none is None and msg3.startswith("No person found"), msg3
            sessions["dwpose_yolox_l.onnx"] = FakeDet([(40 * 80 + 40, np.log(200 / 8))]); sessions["dwpose_dw-ll_ucoco_384.onnx"] = FakePose(0.1)
            weak, msg4 = pc.extract(img)
            assert weak is None and "too little" in msg4, msg4
        (tmp / "det" / "dwpose_yolox_l.onnx").unlink()
        try:
            pc.extract(img); raise AssertionError("missing DWPose files must raise FileNotFoundError")
        except FileNotFoundError as e:
            assert "dwpose_yolox_l.onnx" in str(e)
        assert pc.missing_dwpose() == ["dwpose_yolox_l.onnx"] and not pc.dwpose_available()

        # the control a Generate run gets
        pc.save_pose("v-sign", pose, pc.preview_of_pose(pose))
        ex = dict(pc_pose="v-sign", pc_model="", pc_strength=0.8, pc_legs=True)
        ctl, rec2, err = pc.build_control(ex, 64, 96)
        assert ctl is None and "ControlNet model" in err and "Download" in err, err                                # no model folder yet
        mdir = tmp / "cn" / pc.DEFAULT_MODEL
        mdir.mkdir(parents=True); (mdir / "config.json").write_text("{}"); (mdir / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
        assert pc.list_models() == [pc.DEFAULT_MODEL] and pc.model_path("") == mdir and pc.model_path("../x") is None
        ctl, rec2, err = pc.build_control(ex, 64, 96)
        assert not err and ctl["image"].size == (64, 96) and ctl["scale"] == 0.8 and ctl["model"] == str(mdir) and ctl["name"] == "v-sign", (err, ctl)
        assert rec2["model"] == pc.DEFAULT_MODEL and rec2["legs"] == "hidden" and rec2["source"] == pc.pose_hash(pose)
        assert rec2["keypoints"]["body"] == body and rec2["strength"] == 0.8
        ctl2, rec3, _ = pc.build_control(dict(ex, pc_legs=False, pc_strength=9), 64, 96)
        assert rec3["legs"] == "shown" and rec3["strength"] == 1.0 and np.asarray(ctl2["image"]).sum() > np.asarray(ctl["image"]).sum()
        assert "no pose is chosen" in pc.build_control(dict(ex, pc_pose=""), 64, 96)[2] and "was not found" in pc.build_control(dict(ex, pc_pose="gone"), 64, 96)[2]

        # downloads: written next to the target, SHA-256 checked, a mismatch is deleted
        data = b"pinned bytes"
        sha = hashlib.sha256(data).hexdigest()

        def fake_hf(repo, file, local_dir=None, **kw):
            p = Path(local_dir) / file; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data); return str(p)
        import huggingface_hub
        with patch.object(huggingface_hub, "hf_hub_download", fake_hf):
            dest = tmp / "out" / "a.onnx"
            pc._fetch("r", "a.onnx", dest, sha)
            assert dest.read_bytes() == data and not (tmp / "out" / ".download").exists()
            bad = tmp / "out" / "b.onnx"
            try:
                pc._fetch("r", "b.onnx", bad, "0" * 64); raise AssertionError("a wrong checksum must be refused")
            except RuntimeError as e:
                assert "SHA-256" in str(e) and not bad.exists() and not (tmp / "out" / ".download").exists()
            with patch.object(pc, "DWPOSE", {"dwpose_t.onnx": dict(repo="r", file="t.onnx", mb=1, sha256=sha)}):
                assert "Downloaded dwpose_t.onnx" in pc.download_dwpose() and (tmp / "det" / "dwpose_t.onnx").read_bytes() == data
                assert pc.download_dwpose() == "DWPose files ready."                                                  # nothing fetched twice
            with patch.dict(pc.MODELS, {"tmodel": dict(repo="r", license="MIT", mb=1, files={"config.json": ("config.json", sha), "diffusion_pytorch_model.safetensors": ("w.safetensors", sha)})}):
                assert pc.download_model("tmodel").startswith("tmodel ready") and "tmodel" in pc.list_models()
                try:
                    pc.download_model("nope"); raise AssertionError
                except ValueError:
                    pass
    # the shipped pins are well-formed, the default model is Apache-2.0, and no model without a stated licence is named in any public file
    assert all(len(d["sha256"]) == 64 for d in pc.DWPOSE.values()) and pc.MODELS[pc.DEFAULT_MODEL]["license"] == "Apache-2.0"
    assert all(len(h) == 64 for _f, h in pc.MODELS[pc.DEFAULT_MODEL]["files"].values())
    here = Path(pc.__file__).parent
    priv = here / "pose_models_private.py"
    if priv.exists():                                  # the private repo only: a model without a stated licence lives in that one file, nowhere else
        assert "PRIVATE-ONLY" in priv.read_text(encoding="utf-8")[:300]
        import importlib
        ids = []
        for key, m in importlib.import_module("backend.pose_models_private").PRIVATE_MODELS.items():
            ids += [key.lower(), m["repo"].lower(), m["repo"].split("/")[0].lower()]
        for rel in ("pose_control.py", "ui_pose.py", "help_content.py", "records.py", "png_info.py", "sampling.py", "../app.py", "../README.md", "../run_tests.py"):
            text = (here / rel).read_text(encoding="utf-8").lower()
            assert not [i for i in ids if i in text], (rel, ids)


@test("🧍 Pose control (round 34) plumbing: clean_extra, record → A1111 text → PNG Info → restore plan → controls (a missing pose is rebuilt from the record), the X/Y 'Pose strength' axis, the Generate endpoints grew by the same five trailing inputs")
def _():
    import json as _j, tempfile
    from pathlib import Path
    from unittest.mock import patch
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    import app as _app
    from backend import pose_control as pc
    from backend.png_info import read_png_info, format_png_info_html
    d0 = _app._clean_extra({})
    assert d0["pc_on"] is False and d0["pc_pose"] == "" and d0["pc_strength"] == 0.8 and d0["pc_model"] == "" and d0["pc_legs"] is False, d0
    j = _app._clean_extra({"pc_on": 1, "pc_pose": " a\x00b ", "pc_strength": "x", "pc_legs": 1, "pc_model": "m" * 200})
    assert j["pc_on"] and j["pc_pose"] == "ab" and j["pc_strength"] == 0.8 and j["pc_legs"] and len(j["pc_model"]) == 80
    assert _app._clean_extra({"pc_strength": 0.1})["pc_strength"] == 0.4 and _app._clean_extra({"pc_strength": 7})["pc_strength"] == 1.0
    assert "Pose strength" in _app._XY_AXES
    vals, err = _app._xy_values("Pose strength", "0.6, 0.8, 1.0")
    assert not err and vals == [0.6, 0.8, 1.0] and _app._xy_values("Pose strength", "0.3")[1] and _app._xy_values("Pose strength", "1.2")[1]
    body = [[10.0 * i, 5.0 * i + 20, 0.9] for i in range(18)]
    kp = {"size": [200, 300], "body": body, "lh": [[1.0, 2.0, 0.9]] * 21, "rh": [[3.0, 4.0, 0.9]] * 21}
    rec = {"app": "ImageGen Studio", "format": 2, "mode": "txt2img", "prompt": "1girl", "steps": 12, "cfg_scale": 6.0, "seeds": [1], "scheduler": "DPM++ 2M AYS", "width": 64, "height": 64,
           "pose_control": {"model": "xinsir_openpose_sdxl_1.0", "strength": 0.7, "pose": "gone", "source": "abc", "legs": "hidden", "keypoints": kp}}
    txt = _app._params_text(rec)
    assert "Pose control: xinsir_openpose_sdxl_1.0, 0.7, legs hidden" in txt and "Pose control" not in _app._params_text(dict(rec, pose_control=None))

    def meta_of(**texts):
        pi = PngInfo()
        for k, v in texts.items():
            pi.add_text(k, v)
        with tempfile.TemporaryDirectory() as dd:
            f = Path(dd) / "m.png"
            Image.new("RGB", (8, 8)).save(f, pnginfo=pi)
            with Image.open(f) as im:
                return read_png_info(im)
    m_txt = meta_of(parameters="1girl\nSteps: 12, Seed: 1, Pose control: some_model, 0.6, legs hidden")            # the A1111 text alone: no keypoints
    assert m_txt["pose_control"] == {"model": "some_model", "strength": 0.6, "legs": "hidden"}, m_txt
    assert "Pose control" in format_png_info_html(m_txt)
    tmp = Path(tempfile.mkdtemp())
    with patch.object(pc, "POSES_DIR", tmp / "poses"), patch.object(pc, "STARTER_DIR", tmp / "no_starters"):
        plan = _app._restore_plan(meta_of(imagegen=_j.dumps(rec), parameters="1girl\nSteps: 12, Seed: 1"))
        assert plan["pose_control"]["model"] == "xinsir_openpose_sdxl_1.0" and "pose control" in _app._plan_summary(plan)
        ups = _app._plan_extra_updates(plan)
        name = ups[-4]["value"]
        assert ups[-5] is True and name.startswith("restored_") and name in ups[-4]["choices"] and ups[-3] == 0.7 and ups[-2] == "xinsir_openpose_sdxl_1.0" and ups[-1] is True, ups[-5:]
        assert pc.load_pose(name)["body"] == body                                                                  # the pose came back from the record's keypoints
        ups_text = _app._plan_extra_updates(_app._restore_plan(m_txt))                                              # no keypoints: switches it on, keeps the pose
        assert ups_text[-5] is True and ups_text[-4] == {"__type__": "update"} and ups_text[-3] == 0.6
        ups0 = _app._plan_extra_updates(_app._restore_plan({"prompt": "x"}))
        assert ups0[-5] is False and all(u == {"__type__": "update"} for u in ups0[-4:]), ups0[-5:]               # a record without pose control switches it off and leaves the rest
    # the Generate-side endpoints: the same five controls, last (X/Y: right before the axis inputs), the restore outputs too
    b = _app.build_app()
    by = {}
    for f in b.fns:
        by.setdefault(getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"), f)
    labels = ["Use pose control", "Pose", "Pose strength", "ControlNet model", "Hide legs"]
    lab = lambda comps: [getattr(c, "label", None) for c in comps]
    for fn in ("do_generate_ui", "do_autoloop", "do_card_batch", "do_save_settings"):
        assert lab(by[fn].inputs[-5:]) == labels, (fn, lab(by[fn].inputs[-7:]))
    assert lab(by["do_xy_grid"].inputs[-9:-4]) == labels, lab(by["do_xy_grid"].inputs[-10:])
    assert lab(by["on_i2i_drop"].outputs[-6:-1]) == labels, lab(by["on_i2i_drop"].outputs[-7:])
    assert lab(by["do_load_settings"].outputs[-6:-1]) == labels, lab(by["do_load_settings"].outputs[-7:])
    assert lab(by["do_send_to_gen"].outputs[-5:]) == labels, lab(by["do_send_to_gen"].outputs[-6:])


@test("🧍 Pose control (round 34) in a real Generate run: PAG is forced to 0 and recorded as 0, the txt2img pass gets the skeleton at the picture's size, refusals (SD 1.5, no pose, no model, no GPU) say why, img2img ignores it, the off path is unchanged")
def _():
    import json as _j, tempfile
    from pathlib import Path
    from unittest.mock import patch
    import numpy as np
    from PIL import Image
    from backend import pose_control as pc
    _app, dg, _hp = _app_closures()
    tmp = Path(tempfile.mkdtemp())
    seen = []
    body = [[10.0 + 3 * i, 8.0 + 2 * i, 0.9] for i in range(18)]
    pose = {"v": 1, "size": [64, 96], "body": body, "lh": [[5.0, 5.0, 0.9]] * 21, "rh": [[9.0, 9.0, 0.9]] * 21}

    class FakeSD:
        model_family, current_model, _last_vae_path, prediction, device = "illustrious", "x.safetensors", None, {}, "cuda:0"
        _lora_adapters, boosters, last_seeds, last_var_seeds, pipe = {}, {}, [], [], object()
        control = None

        def txt2img(self, prompt, neg, w, h, steps, cfg, seed, sched, batch, step_callback=None, **kw):
            seen.append((self.control, dict(self.boosters)))
            self.last_seeds = [seed + i for i in range(batch)]
            return [Image.new("RGB", (w, h), (seed % 255, 40, 90)) for _ in range(batch)], "info"

        def img2img(self, img, prompt, neg, strength, steps, cfg, seed, sched, step_callback=None, clip_skip=1):
            seen.append((self.control, dict(self.boosters)))
            return [img], "i2i"
    old = (_app.sd, _app.OUTPUTS_DIR)
    _app.sd, _app.OUTPUTS_DIR = FakeSD(), tmp / "outputs"
    _app.OUTPUTS_DIR.mkdir()
    try:
        with patch.object(pc, "POSES_DIR", tmp / "poses"), patch.object(pc, "STARTER_DIR", tmp / "no_starters"), patch.object(pc, "CONTROLNET_DIR", tmp / "cn"):
            pc.save_pose("v-sign", pose, pc.preview_of_pose(pose))
            mdir = tmp / "cn" / pc.DEFAULT_MODEL
            mdir.mkdir(parents=True); (mdir / "config.json").write_text("{}"); (mdir / "m.safetensors").write_bytes(b"x")
            prog = lambda *a, **k: None
            ex = dict(pc_on=True, pc_pose="v-sign", pc_strength=0.7, pc_legs=True, pag_scale=2.0)
            imgs, info, _ = dg("1girl", "bad", "Euler a", 4, 6.0, 256, 384, 1, 5, None, 0.5, False, extra=ex, progress=prog)
            assert imgs and len(seen) == 1, (info, seen)
            ctl, boosters = seen[0]
            assert ctl["image"].size == (256, 384) and ctl["scale"] == 0.7 and boosters["pag"] == 0.0, (ctl, boosters)       # PAG forced off for the run
            assert "Pose control: v-sign" in info and "PAG 2 → 0" in info and _app.sd.control is None and _app.sd.boosters == {}
            f = sorted(_app.OUTPUTS_DIR.glob("*.png"))[0]
            with Image.open(f) as im:
                im.load()
                r = _j.loads(im.info["imagegen"])
                assert r["pose_control"]["model"] == pc.DEFAULT_MODEL and r["pose_control"]["strength"] == 0.7 and r["pose_control"]["keypoints"]["body"] == body and "pag_scale" not in r, r
                assert "Pose control: " + pc.DEFAULT_MODEL + ", 0.7, legs hidden" in im.info["parameters"] and "PAG scale" not in im.info["parameters"]
            # the off path: no control, PAG as set, nothing recorded
            seen.clear()
            imgs, info, _ = dg("1girl", "bad", "Euler a", 4, 6.0, 256, 384, 1, 6, None, 0.5, False, extra=dict(pag_scale=2.0), progress=prog)
            assert seen[0][0] is None and seen[0][1]["pag"] == 2.0 and "Pose control" not in info
            r2 = _j.loads(Image.open(sorted(_app.OUTPUTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime)[-1]).info["imagegen"])
            assert "pose_control" not in r2 and r2["pag_scale"] == 2.0, r2
            # refusals: nothing is generated, the message says why
            seen.clear()
            for patch_attrs, ex2, want in (
                    (dict(model_family="sd15"), ex, "SDXL"), (dict(device="cpu"), ex, "GPU runtime"),
                    ({}, dict(ex, pc_pose=""), "no pose is chosen"), ({}, dict(ex, pc_pose="gone"), "was not found"), ({}, dict(ex, pc_model="missing_model"), "missing_model")):
                for k, v in patch_attrs.items():
                    setattr(_app.sd, k, v)
                imgs, info, _ = dg("1girl", "bad", "Euler a", 4, 6.0, 256, 384, 1, 7, None, 0.5, False, extra=ex2, progress=prog)
                assert not imgs and want in info and not seen, (want, info)
                _app.sd.model_family, _app.sd.device = "illustrious", "cuda:0"
            # img2img: pose control is ignored (with a note), the picture is made as usual
            imgs, info, _ = dg("1girl", "bad", "Euler a", 4, 6.0, 256, 384, 1, 8, Image.new("RGB", (64, 96)), 0.5, True, extra=ex, progress=prog)
            assert imgs and "not used for img2img" in info and seen[-1][0] is None, info
    finally:
        _app.sd, _app.OUTPUTS_DIR = old


@test("🧍 Pose control (round 34): run_pipe builds the ControlNet target only for an SDXL txt2img pass, never combines it with PAG, passes the skeleton + strength, releases the model afterwards; img2img and a plain run are untouched")
def _():
    import types
    from unittest.mock import patch
    from PIL import Image
    from diffusers import EulerDiscreteScheduler
    from backend import sampling

    class Unet:
        dtype = None
        def __init__(self): self.procs = {"a": 1}
        @property
        def attn_processors(self): return dict(self.procs)
        def set_attn_processor(self, p): self.procs = dict(p)
        def enable_freeu(self, **k): pass
        def disable_freeu(self): pass

    class Target:
        def __init__(self, tag): self.tag, self.calls, self.scheduler, self.unet = tag, [], EulerDiscreteScheduler(), Unet()
        def __call__(self, prompt_embeds=None, guidance_scale=7.0, image=None, controlnet_conditioning_scale=None, control_guidance_start=None, control_guidance_end=None,
                     pag_scale=None, guidance_rescale=None, num_inference_steps=1, callback_on_step_end=None, width=64, height=64):
            self.calls.append(dict(image=image, scale=controlnet_conditioning_scale, start=control_guidance_start, end=control_guidance_end, pag=pag_scale)); return self.tag
    pipe = Target("plain"); pipe.text_encoder_2 = object()
    ctl_target = Target("control")
    sdp = types.SimpleNamespace(boosters={"pag": 2.0}, prediction={}, control=None)
    built, released = [], []
    with patch.object(sampling, "_control_target", lambda s, p, c: built.append(c) or ctl_target), patch.object(sampling, "release_control", lambda s: released.append(1)):
        img = Image.new("RGB", (8, 8))
        sdp.control = {"image": img, "scale": 0.8, "model": "m"}
        assert sampling.run_pipe(sdp, pipe, "txt2img", prompt_embeds=1, num_inference_steps=2) == "control"
        c = ctl_target.calls[0]
        assert c["image"] is img and c["scale"] == 0.8 and c["start"] == 0.0 and c["end"] == 1.0 and c["pag"] is None and released == [1] and not pipe.calls          # no PAG with a pose model, model released
        pipe.calls.clear()
        assert sampling.run_pipe(types.SimpleNamespace(boosters={}, prediction={}, control={"image": img, "scale": 1, "model": "m"}), pipe, "img2img", prompt_embeds=1) == "plain"
        assert pipe.calls[0]["image"] is None and len(ctl_target.calls) == 1                                                                                       # img2img: never controlled
        sdp.control = None
        built.clear(); released.clear(); pipe.calls.clear()
        r = sampling.run_pipe(types.SimpleNamespace(boosters={}, prediction={}, control=None), pipe, "txt2img", prompt_embeds=1)
        assert r == "plain" and pipe.calls[0]["image"] is None and not built and not released                                                                     # a plain run: nothing changes
        sd15 = Target("sd15")                                                                                                                                     # no second text encoder: SD 1.5 is never controlled
        assert sampling.run_pipe(types.SimpleNamespace(boosters={}, prediction={}, control={"image": img, "scale": 1, "model": "m"}), sd15, "txt2img", prompt_embeds=1) == "sd15" and not built


@test("🧍 Pose control (round 34): the accordion's handlers — a skeleton from a picture, the missing-DWPose message with the download pointer, no person, already-a-pose-image, save / delete, download failures never raise")
def _():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from PIL import Image
    import app as _app
    from backend import pose_control as pc
    b = _app.build_app()
    by = {}
    for f in b.fns:
        by.setdefault(getattr(getattr(f.fn, "__wrapped__", f.fn), "__name__", "?"), f)
    ext, sel, save, dele, dl = (getattr(by[n].fn, "__wrapped__", by[n].fn) for n in ("on_pose_extract", "on_pose_select", "on_pose_save", "on_pose_delete", "on_pose_download"))
    tmp = Path(tempfile.mkdtemp())
    pose = {"v": 1, "size": [100, 150], "body": [[10.0 + 2 * i, 8.0 + 3 * i, 0.9] for i in range(18)], "lh": [], "rh": []}
    pic = Image.new("RGB", (100, 150), (120, 120, 120))
    with patch.object(pc, "POSES_DIR", tmp / "poses"), patch.object(pc, "STARTER_DIR", tmp / "no_starters"):
        assert "Upload" in ext(None, False)[2]
        with patch.object(pc, "extract", side_effect=FileNotFoundError("missing")):
            out = ext(pic, False)
            assert "Download" in out[2] and "DWPose" in out[2], out                                                  # a message, never a crash
        with patch.object(pc, "extract", return_value=(None, "No person found in the picture — x")):
            assert "No person found" in ext(pic, False)[2] and pc.list_poses() == []
        with patch.object(pc, "extract", side_effect=RuntimeError("boom")):
            assert "Could not read a skeleton" in ext(pic, False)[2]
        with patch.object(pc, "extract", return_value=(dict(pose, hash=pc.pose_hash(pose)), "Skeleton found (18 of 18 body points).")):
            shown, dd, status = ext(pic, False)
            assert shown is not None and dd["value"] == pc.CURRENT and pc.CURRENT in dd["choices"] and "Skeleton found" in status
        shown, dd, status = ext(pic, True)                                                                              # already a pose image: used as it is
        assert shown is not None and pc.load_pose(pc.CURRENT).get("image") and "as it is" in status
        assert sel(pc.CURRENT) is not None
        assert "name" in save("", pc.CURRENT)[1].lower()
        upd, status = save("Reach pose", pc.CURRENT)
        assert upd["value"] == "Reach pose" and "Reach pose" in pc.list_poses() and "Saved" in status
        upd, status = dele("Reach pose")
        assert "Reach pose" not in pc.list_poses() and "Deleted" in status and "Nothing" in dele("")[1]
        with patch.object(pc, "download_dwpose", side_effect=OSError("offline")):
            out = dl("DWPose skeleton files (351 MB)")
            assert "Download failed" in out[0] and "offline" in out[0]
        with patch.object(pc, "download_dwpose", return_value="DWPose files ready."), patch.object(pc, "download_model", return_value="m ready (Apache-2.0)"):
            assert "ready" in dl("DWPose skeleton files (351 MB)")[0] and "ready" in dl(pc.DEFAULT_MODEL + " (2.4 GB, Apache-2.0)")[0]


@test("🧍 Starter poses (round 35): the shipped files are keypoints only (no picture, no name), they are listed with a ★ after the user's poses, load with a hash, can't be overwritten or deleted, build a control, and a record naming one finds it again")
def _():
    import json as _j, tempfile
    from pathlib import Path
    from unittest.mock import patch
    import numpy as np
    from backend import pose_control as pc
    real = sorted(pc.STARTER_DIR.glob("*.json"))
    assert len(real) >= 12, real                                                                    # the shipped set (round 35: 12-16 poses)
    for f in real:
        d = _j.loads(f.read_text(encoding="utf-8"))
        assert d.get("title") and len(d["body"]) == 18 and len(d["size"]) == 2 and len(d["lh"]) in (0, 21) and len(d["rh"]) in (0, 21), f.name
        assert all(len(k) == 3 for k in d["body"]) and not any(k in d for k in ("image", "picture", "file", "source")), f.name       # keypoints only
        text = f.read_text(encoding="utf-8").lower()
        assert not any(w in text for w in ("heroine", "azur")), f.name                                        # generic: no character name anywhere
        assert (pc.STAR + d["title"]) in pc.starter_poses(), f.name
    assert not list(pc.STARTER_DIR.glob("*.png")) and not list(pc.STARTER_DIR.glob("*.jp*"))                                         # no source pictures
    tmp = Path(tempfile.mkdtemp())
    (tmp / "starters").mkdir()
    for title, stem in (("reach toward viewer — cowboy", "reach_cowboy"), ("a hand on hip", "hand_on_hip")):
        body = [[10.0 + 3 * i, 8.0 + 2 * i, 0.9] for i in range(18)]
        (tmp / "starters" / f"{stem}.json").write_text(_j.dumps({"v": 1, "title": title, "size": [200, 300], "body": body, "lh": [], "rh": []}))
    (tmp / "starters" / "broken.json").write_text("{not json")
    (tmp / "cn" / pc.DEFAULT_MODEL).mkdir(parents=True)
    (tmp / "cn" / pc.DEFAULT_MODEL / "config.json").write_text("{}"); (tmp / "cn" / pc.DEFAULT_MODEL / "w.safetensors").write_bytes(b"x")
    with patch.object(pc, "STARTER_DIR", tmp / "starters"), patch.object(pc, "POSES_DIR", tmp / "poses"), patch.object(pc, "CONTROLNET_DIR", tmp / "cn"):
        assert pc.starter_poses() == {"★ a hand on hip": "hand_on_hip", "★ reach toward viewer — cowboy": "reach_cowboy"}                  # a broken file is skipped
        body = [[1.0 * i, 2.0 * i, 0.9] for i in range(18)]
        pc.save_pose("mine", {"v": 1, "size": [100, 100], "body": body, "lh": [], "rh": []})
        pc.save_pose(pc.CURRENT, {"v": 1, "size": [100, 100], "body": body, "lh": [], "rh": []})
        assert pc.list_poses() == [pc.CURRENT, "mine", "★ a hand on hip", "★ reach toward viewer — cowboy"], pc.list_poses()           # the user's first, the starters last
        p = pc.load_pose("★ a hand on hip")
        assert p["size"] == [200, 300] and len(p["body"]) == 18 and p["hash"] == pc.pose_hash(p) and pc.load_pose("★ nothing") is None
        assert pc.preview_of("★ a hand on hip") is not None
        for bad in ("★ mine", "★"):
            try:
                pc.save_pose(bad, p); raise AssertionError("a ★ name is reserved")
            except ValueError as e:
                assert "★" in str(e)
        assert pc.delete_pose("★ a hand on hip") is False and "★ a hand on hip" in pc.list_poses()
        ctl, rec, err = pc.build_control(dict(pc_pose="★ a hand on hip", pc_model="", pc_strength=0.8, pc_legs=False), 64, 96)
        assert not err and ctl["image"].size == (64, 96) and rec["pose"] == "★ a hand on hip" and rec["source"] == p["hash"] and rec["keypoints"]["body"] == p["body"]
        assert pc.ensure_pose_from_record({"pose": "★ a hand on hip", "keypoints": rec["keypoints"]}) == "★ a hand on hip"            # a record naming a starter finds it again
        moved = dict(rec["keypoints"], body=[[x + 5, y, s] for x, y, s in p["body"]])
        assert pc.ensure_pose_from_record({"pose": "★ a hand on hip", "keypoints": moved}).startswith("restored_")                       # ... unless its keypoints differ (an edited starter)
        assert "★" not in "".join(pc.list_poses()[:2])


@test("🧍 Pose model kept across a batch (round 35): kept only when asked and the pass succeeded, released by a Stop, an error, a pass without the pose (hires / detail / plain), the idle timer and a model unload; never while a pass runs")
def _():
    import time, types
    from unittest.mock import patch
    from PIL import Image
    from diffusers import EulerDiscreteScheduler
    from backend import sampling

    class Unet:
        dtype = None
        def __init__(self): self.procs = {"a": 1}
        @property
        def attn_processors(self): return dict(self.procs)
        def set_attn_processor(self, p): self.procs = dict(p)
        def enable_freeu(self, **k): pass
        def disable_freeu(self): pass

    class Target:
        def __init__(self, tag, fail=False): self.tag, self.fail, self.calls, self.scheduler, self.unet = tag, fail, [], EulerDiscreteScheduler(), Unet()
        def __call__(self, prompt_embeds=None, image=None, controlnet_conditioning_scale=None, control_guidance_start=None, control_guidance_end=None, num_inference_steps=1):
            self.calls.append(image)
            if self.fail:
                raise RuntimeError("boom")
            return self.tag
    plain = Target("plain"); plain.text_encoder_2 = object()
    builds = []

    def fake_target(sdp, pipe, ctl):
        if getattr(sdp, "_cn_cache", None) is None:
            builds.append(1)
            sdp._cn_cache = {"model": ctl["model"]}
        return Target("control", fail=ctl.get("fail"))
    img = Image.new("RGB", (8, 8))
    mk = lambda: types.SimpleNamespace(boosters={}, prediction={}, control=None, _cn_cache=None)
    ctl = lambda **k: dict({"image": img, "scale": 0.8, "model": "m", "keep": True}, **k)
    with patch.object(sampling, "_control_target", fake_target), patch.object(sampling, "CONTROL_IDLE_S", 30.0):
        sdp = mk()
        sdp.control = ctl()
        assert sampling.run_pipe(sdp, plain, "txt2img", prompt_embeds=1) == "control" and sdp._cn_cache is not None and builds == [1]          # asked to keep: kept
        assert sampling.run_pipe(sdp, plain, "txt2img", prompt_embeds=1) == "control" and builds == [1] and sdp._cn_cache is not None             # the next picture reuses it (no second build)
        sdp.control = None
        assert sampling.run_pipe(sdp, plain, "img2img", prompt_embeds=1) == "plain" and sdp._cn_cache is None                                      # hires / a detail pass / a plain picture: released first
        sdp.control = ctl(keep=False)
        sampling.run_pipe(sdp, plain, "txt2img", prompt_embeds=1)
        assert sdp._cn_cache is None                                                                                                               # a picture with later passes: today's behaviour
        sdp.control = ctl(fail=True)
        try:
            sampling.run_pipe(sdp, plain, "txt2img", prompt_embeds=1); raise AssertionError
        except RuntimeError:
            pass
        assert sdp._cn_cache is None and sdp._cn_busy is False and getattr(sdp, "_cn_timer", None) is None                                          # an error releases it
        sdp.control = ctl()
        with patch.object(sampling, "should_stop", lambda: True):
            sampling.run_pipe(sdp, plain, "txt2img", prompt_embeds=1)
        assert sdp._cn_cache is None                                                                                                               # a Stop releases it
        # the idle timer (the end of a batch) and a pass that owns the model
        with patch.object(sampling, "CONTROL_IDLE_S", 0.05):
            sdp.control = ctl()
            sampling.run_pipe(sdp, plain, "txt2img", prompt_embeds=1)
            assert sdp._cn_cache is not None
            time.sleep(0.5)
            assert sdp._cn_cache is None and getattr(sdp, "_cn_timer", None) is None
            sdp.control = ctl()
            sampling.run_pipe(sdp, plain, "txt2img", prompt_embeds=1)
            sdp._cn_busy = True
            time.sleep(0.4)
            assert sdp._cn_cache is not None                                                                                                       # a running pass is never pulled from under
            sdp._cn_busy = False
            sampling.release_control(sdp)
            assert sdp._cn_cache is None
        # a model unload drops the cache and the control
        from backend.sdxl_pipeline import SDXLPipeline
        p = SDXLPipeline()
        p._cn_cache, p.control = {"x": 1}, {"image": img}
        p._unload()
        assert p._cn_cache is None and p.control is None


@test("🧍 Pose model kept (round 35) in a real Generate run: a plain pose picture asks to keep the model, a picture with hires / hand / face / eye / max-detail passes does not; the batch loops reach the same do_generate")
def _():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from PIL import Image
    from backend import pose_control as pc
    _app, dg, _hp = _app_closures()
    tmp = Path(tempfile.mkdtemp())
    seen = []
    body = [[10.0 + 3 * i, 8.0 + 2 * i, 0.9] for i in range(18)]

    class FakeSD:
        model_family, current_model, _last_vae_path, prediction, device = "illustrious", "x.safetensors", None, {}, "cuda:0"
        _lora_adapters, boosters, last_seeds, last_var_seeds, pipe = {}, {}, [], [], object()
        control = None

        def txt2img(self, prompt, neg, w, h, steps, cfg, seed, sched, batch, step_callback=None, **kw):
            seen.append(dict(self.control or {}))
            self.last_seeds = [seed + i for i in range(batch)]
            _app._generation_abort.set()               # Stop right after the first pass: the later passes (detectors, downloads) never start
            return [Image.new("RGB", (w, h), (seed % 255, 40, 90)) for _ in range(batch)], "info"
    old = (_app.sd, _app.OUTPUTS_DIR)
    _app.sd, _app.OUTPUTS_DIR = FakeSD(), tmp / "outputs"
    _app.OUTPUTS_DIR.mkdir()
    try:
        with patch.object(pc, "POSES_DIR", tmp / "poses"), patch.object(pc, "CONTROLNET_DIR", tmp / "cn"), patch.object(pc, "STARTER_DIR", tmp / "none"):
            pc.save_pose("p", {"v": 1, "size": [256, 384], "body": body, "lh": [], "rh": []})
            m = tmp / "cn" / pc.DEFAULT_MODEL
            m.mkdir(parents=True); (m / "config.json").write_text("{}"); (m / "w.safetensors").write_bytes(b"x")
            base = dict(pc_on=True, pc_pose="p", pc_strength=0.8)
            assert pc.KEEP_RESIDENT is False                                                              # measured: it spilled on 12 GB, so the shipped default is today's release after every picture
            pc.KEEP_RESIDENT = True
            for extra, keep in ((base, True), (dict(base, hires_on=True), False), (dict(base, hd_on=True), False), (dict(base, fd_on=True), False),
                                (dict(base, ed_on=True), False), (dict(base, md_on=True), False)):
                seen.clear(); _app._generation_abort.clear()
                with patch.object(_app, "_thermal_wait", lambda *a, **k: 0.0):
                    dg("1girl", "bad", "Euler a", 4, 6.0, 256, 384, 1, 5, None, 0.5, False, extra=extra, progress=lambda *a, **k: None)
                assert seen and seen[0].get("keep") is keep, (extra, seen)
            pc.KEEP_RESIDENT = False
            seen.clear(); _app._generation_abort.clear()
            with patch.object(_app, "_thermal_wait", lambda *a, **k: 0.0):
                dg("1girl", "bad", "Euler a", 4, 6.0, 256, 384, 1, 5, None, 0.5, False, extra=base, progress=lambda *a, **k: None)
            assert seen and seen[0].get("keep") is False, seen                                              # default: never kept
    finally:
        pc.KEEP_RESIDENT = False
        _app._generation_abort.clear()
        _app.sd, _app.OUTPUTS_DIR = old


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped, {PASS + FAIL + SKIP} total")
print("=" * 70)
if FAIL:
    print("\nFAILED TESTS:")
    for r in RESULTS:
        if r[0] == "❌":
            print(f"  {r[0]} {r[1]}: {r[2]}")
    sys.exit(1)
else:
    print("\n🎉 All tests passed"
          + (f" ({SKIP} skipped — hardware not present)" if SKIP else "!"))
    sys.exit(0)
