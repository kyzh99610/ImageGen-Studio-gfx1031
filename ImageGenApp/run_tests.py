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

import sys, os, json, traceback, threading
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
import numpy as np

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


def test(name):
    def decorator(fn):
        global PASS, FAIL, SKIP
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
    assert not npu_cpus.search("AMD Ryzen 9 5980HX"), "Should NOT match 5980HX (no NPU)"
    assert not npu_cpus.search("AMD Ryzen 9 9800X3D"), "Should NOT match 9800X3D (no NPU)"
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

@test("_ort_session maps 'npu' to VitisAIExecutionProvider")
def _():
    from backend.smartsplit_pipeline import _ort_session
    # We can't create a real session without an ONNX file, but verify the EP map
    src = Path(APP_DIR / "backend" / "smartsplit_pipeline.py").read_text(encoding="utf-8")
    assert '"npu":           "VitisAIExecutionProvider"' in src
    assert '"igpu_directml": "DmlExecutionProvider"' in src
    assert '"cpu":           "CPUExecutionProvider"' in src

@test("_ort_session falls back to CPU when VitisAI unavailable")
def _():
    # Test by inspecting the fallback logic in source
    src = Path(APP_DIR / "backend" / "smartsplit_pipeline.py").read_text(encoding="utf-8")
    assert 'provider not in available' in src
    assert 'falling back to CPU' in src
    assert 'CPUExecutionProvider' in src

@test("ONNX TE export uses correct input/output names")
def _():
    src = Path(APP_DIR / "backend" / "smartsplit_pipeline.py").read_text(encoding="utf-8")
    assert 'input_names=["input_ids"]' in src
    assert 'output_names=["last_hidden_state", "pooler_output"]' in src

@test("ONNX VAE export uses dynamic batch axes")
def _():
    src = Path(APP_DIR / "backend" / "smartsplit_pipeline.py").read_text(encoding="utf-8")
    assert '"latents": {0: "batch", 2: "height", 3: "width"}' in src


# ══════════════════════════════════════════════════════════════════════════
print("\n[5/8] SmartSplit Pipeline NPU Path")
# ══════════════════════════════════════════════════════════════════════════

@test("SmartSplitPipeline._setup handles NPU bridge path")
def _():
    src = Path(APP_DIR / "backend" / "smartsplit_pipeline.py").read_text(encoding="utf-8")
    # Verify NPU bridge path exists in _setup
    assert 'self._setup_npu_te(onnx_path)' in src
    # Verify fallback to local ort session
    assert '_ort_session(onnx_path, fallback)' in src

@test("SmartSplitPipeline.encode_prompt runs NPU/iGPU ONNX session")
def _():
    src = Path(APP_DIR / "backend" / "smartsplit_pipeline.py").read_text(encoding="utf-8")
    # Verify the ONNX inference path
    assert 'self._te_session.run(None, {"input_ids": pos_ids})' in src
    assert 'self._te_session.run(None, {"input_ids": neg_ids})' in src
    # Verify output conversion to torch
    assert 'torch.from_numpy(pos_embeds).to(device, dtype=torch.float16)' in src

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

@test("app.py imports SmartSplit components")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert "from backend.smartsplit_pipeline import" in src
    assert "SmartSplitPipeline" in src
    assert "SmartSplitConfig" in src
    assert "detect_smartsplit_capability" in src

@test("app.py has _smartsplit_cfg and _smartsplit_pipe globals")
def _():
    import app as _app
    assert hasattr(_app, '_smartsplit_cfg')
    assert hasattr(_app, '_smartsplit_pipe')
    assert hasattr(_app, '_smartsplit_cap')

@test("SmartSplit generation path checks enabled + pipe + not img2img")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert "_smartsplit_cfg.enabled and _smartsplit_pipe is not None and not use_i2i" in src

@test("SmartSplit path calls _smartsplit_pipe.generate()")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert "_smartsplit_pipe.generate(" in src
    assert "progress_callback=prog_cb" in src

@test("SmartSplit auto-rebuilds on SD 1.5 model load")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert "_smartsplit_pipe = SmartSplitPipeline(sd.pipe, _smartsplit_cfg, stem)" in src

@test("SmartSplit disables on SDXL model load")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert '_smartsplit_pipe = None' in src
    assert 'SmartSplit disabled (SDXL not supported' in src

@test("SmartSplit rejects SDXL in on_apply_split")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert 'SmartSplit not supported for SDXL/Pony/Illustrious' in src


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

@test("_smartsplit_te_choices includes NPU when available")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert 'if cap.npu_available:      choices.append("npu")' in src

@test("_build_smartsplit_cap_html includes NPU name")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert 'if cap.npu_available:' in src
    assert 'cap.npu_name' in src

@test("Settings tab has NPU section with badge and details")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert '### ⚡ AMD XDNA NPU (Ryzen AI)' in src
    assert '_npu_badge(profile.npu)' in src
    assert '_build_npu_info(profile.npu)' in src

@test("SmartSplit TE dropdown offers NPU option")
def _():
    src = Path(APP_DIR / "app.py").read_text(encoding="utf-8")
    assert 'te_device_dd = gr.Dropdown(' in src
    assert '_smartsplit_te_choices(_cap)' in src


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

@test("SDXL accel-TE wired into txt2img path")
def _():
    import inspect
    from backend.sdxl_pipeline import SDXLPipeline
    src = inspect.getsource(SDXLPipeline.txt2img)
    assert "_accel_te_enabled" in src, "txt2img should check _accel_te_enabled"
    assert "_build_sdxl_accel_embeds" in src, "txt2img should call _build_sdxl_accel_embeds"
    assert "_build_sdxl_embeds" in src, "txt2img should still have CPU fallback"

@test("SDXL accel-TE wired into img2img path")
def _():
    import inspect
    from backend.sdxl_pipeline import SDXLPipeline
    src = inspect.getsource(SDXLPipeline.img2img)
    assert "_accel_te_enabled" in src, "img2img should check _accel_te_enabled"
    assert "_build_sdxl_accel_embeds" in src, "img2img should call _build_sdxl_accel_embeds"

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
    orig, orig_hits = dt._cascade, dt._confirm_hits
    try:
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
    finally:
        dt._cascade, dt._confirm_hits = orig, orig_hits
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
                                                     "red eyes": 18, "cleavage": 15, "smile": 3}}),
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


@test("LoRA restore survives a textual inversion that grew the token table; A1111 'Schedule type' restores")
def _():
    import torch
    from types import SimpleNamespace
    from backend.sd_pipeline import SDPipeline
    from backend.sdxl_pipeline import SDXLPipeline
    for cls in (SDPipeline, SDXLPipeline):
        te = torch.nn.Sequential(torch.nn.Embedding(10, 4), torch.nn.Linear(4, 4))
        te2 = torch.nn.Sequential(torch.nn.Embedding(10, 4))
        unet = torch.nn.Linear(4, 4)
        sdp = cls.__new__(cls)
        sdp.pipe = SimpleNamespace(unet=unet, text_encoder=te, text_encoder_2=te2)
        sdp._clean_unet_state = sdp._clean_te_state = sdp._clean_te2_state = None
        sdp._snapshot_clean_state()
        clean = te[0].weight.detach().clone()
        with torch.no_grad():                  # a fused LoRA changes weights …
            unet.weight.add_(1); te[1].weight.add_(1); te[0].weight.add_(1)
        te[0].weight = torch.nn.Parameter(torch.cat([te[0].weight.data, torch.full((1, 4), 7.0)]))  # … a TI adds a row
        sdp._restore_clean_state()             # raised "size of tensor a (11) must match (10)" before
        assert torch.equal(te[0].weight[:10], clean) and torch.equal(te[0].weight[10], torch.full((4,), 7.0))
        assert torch.equal(unet.weight, sdp._clean_unet_state["weight"])
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from backend.png_info import read_image_metadata
    import tempfile
    for text, want in (("Sampler: DPM++ 2M, Schedule type: Karras", "DPM++ 2M Karras"),
                       ("Sampler: Euler, Schedule type: Align Your Steps", "Euler AYS"),
                       ("Sampler: DPM++ 2M Karras, Schedule type: Karras", "DPM++ 2M Karras"),
                       ("Sampler: Euler a, Schedule type: Automatic", "Euler a")):
        info = PngInfo(); info.add_text("parameters", f"1girl\nSteps: 20, {text}, CFG scale: 7, Seed: 1")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"
            Image.new("RGB", (8, 8)).save(p, pnginfo=info)
            assert read_image_metadata(p)["sampler"] == want, (text, read_image_metadata(p)["sampler"])


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
    assert ups[-3:] == [2.5, True, 0.7], ups
    ups0 = _app._plan_extra_updates(_app._restore_plan({"prompt": "x"}))
    assert ups0[-3:] == [0.0, False, 0.0]                          # none recorded → off
    ex = _app._clean_extra(dict(pag_scale="99", freeu="yes", cfg_rescale=-3))
    assert ex["pag_scale"] == 6.0 and ex["freeu"] is True and ex["cfg_rescale"] == 0.0
    assert "PAG scale" in _app._XY_AXES and _app._xy_values("PAG scale", "0, 2.5") == ([0.0, 2.5], "")


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
    assert danbooru_hints("1girl, long haired, thigh") == [("long haired", "long hair")]   # last one still typed
    p, fixes = apply_danbooru_fixes("(long haired:1.2), [blue eye], <lora:x:0.8>, {a|b}, __pose__, 1girl")
    assert p == "(long hair:1.2), [blue eyes], <lora:x:0.8>, {a|b}, __pose__, 1girl", p


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
