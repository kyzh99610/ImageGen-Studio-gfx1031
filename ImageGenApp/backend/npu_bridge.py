"""
backend/npu_bridge.py
Subprocess bridge for AMD XDNA NPU inference via VitisAI EP.

Since the Ryzen AI SDK's onnxruntime-vitisai wheel requires Python 3.12 but
our app runs on Python 3.10, we bridge the gap by calling the SDK's conda
environment (ryzen-ai-1.7.0) via subprocess for NPU-specific ONNX sessions.

Architecture:
  Main process (Python 3.10)          NPU worker (Python 3.12 conda)
  ┌─────────────────────────┐         ┌────────────────────────────┐
  │  SmartSplitPipeline     │  stdin  │  _npu_worker.py            │
  │  .encode_prompt()  ─────┼────────>│  reads ONNX path + tokens  │
  │                         │  stdout │  runs VitisAI EP session    │
  │  receives embeddings <──┼─────────│  returns numpy arrays       │
  └─────────────────────────┘         └────────────────────────────┘

The worker process is long-lived (started once, reused across generations)
to avoid the ~2s Python startup cost on every text encode call.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Conda env discovery ──────────────────────────────────────────────────────

_CONDA_ENV_NAMES = ["ryzen-ai-1.7.0", "ryzen-ai-1.6.1", "ryzen-ai-1.6.0"]
_CONDA_BASES = [
    Path(r"C:\ProgramData\miniconda3"),
    Path(os.environ.get("CONDA_PREFIX", "")) / ".." / "..",
    Path(os.environ.get("USERPROFILE", "")) / "miniconda3",
    Path(os.environ.get("USERPROFILE", "")) / "anaconda3",
]

_SDK_PATH = os.environ.get(
    "RYZEN_AI_INSTALLATION_PATH",
    r"C:\Program Files\RyzenAI\1.7.0",
)
_VAIP_CONFIG = os.path.join(_SDK_PATH, "voe-4.0-win_amd64", "vaip_config.json")


def find_conda_python() -> Optional[Path]:
    """Find the Ryzen AI conda environment's Python executable."""
    for base in _CONDA_BASES:
        base = base.resolve()
        if not base.exists():
            continue
        for env_name in _CONDA_ENV_NAMES:
            python = base / "envs" / env_name / "python.exe"
            if python.exists():
                return python
    return None


def check_npu_bridge_available() -> tuple[bool, str]:
    """
    Check if NPU bridge can be used.
    Returns (available: bool, message: str).
    """
    python = find_conda_python()
    if not python:
        return False, (
            "Ryzen AI conda env not found. "
            "Expected: miniconda3/envs/ryzen-ai-1.7.0/python.exe"
        )

    if not os.path.isfile(_VAIP_CONFIG):
        return False, f"VAIP config not found: {_VAIP_CONFIG}"

    # Quick check: does the conda Python have VitisAI EP?
    try:
        result = subprocess.run(
            [str(python), "-c",
             "import onnxruntime as ort; "
             "print('VitisAIExecutionProvider' in ort.get_available_providers())"],
            capture_output=True, text=True, timeout=15,
        )
        if "True" in result.stdout:
            return True, f"NPU bridge ready via {python}"
        else:
            return False, (
                f"Conda env found ({python}) but VitisAI EP not available. "
                f"Providers: {result.stdout.strip()}"
            )
    except Exception as e:
        return False, f"NPU bridge check failed: {e}"


# ── Worker script (written to temp dir, executed in conda env) ────────────────

_WORKER_SCRIPT = r'''
"""NPU worker — long-lived subprocess for VitisAI EP inference."""
import json, os, struct, sys, time
import numpy as np

SDK = os.environ.get("RYZEN_AI_INSTALLATION_PATH", r"C:\Program Files\RyzenAI\1.7.0")
VAIP = os.path.join(SDK, "voe-4.0-win_amd64", "vaip_config.json")

# Ensure SDK DLLs are findable
deploy = os.path.join(SDK, "deployment")
xrt = os.path.join(SDK, "xrt")
os.environ["PATH"] = deploy + ";" + xrt + ";" + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(deploy)
    os.add_dll_directory(xrt)

import onnxruntime as ort

sessions = {}  # cache: onnx_path -> InferenceSession


def get_session(onnx_path: str) -> ort.InferenceSession:
    if onnx_path not in sessions:
        providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
        options = [{"config_file": VAIP}, {}]
        sess = ort.InferenceSession(onnx_path, providers=providers,
                                    provider_options=options)
        actual = sess.get_providers()
        sessions[onnx_path] = sess
        sys.stderr.write(f"[NPU Worker] Loaded {os.path.basename(onnx_path)} "
                         f"on {actual[0]}\n")
        sys.stderr.flush()
    return sessions[onnx_path]


def send_response(data: dict):
    """Send a JSON response + binary numpy arrays."""
    # Serialize numpy arrays as binary chunks
    arrays = {}
    clean_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
            clean_data[k] = {"__numpy__": True, "dtype": str(v.dtype),
                             "shape": list(v.shape)}
        else:
            clean_data[k] = v

    header = json.dumps(clean_data).encode("utf-8")
    # Protocol: [header_len:4][header][for each array: array_bytes_len:4][array_bytes]
    sys.stdout.buffer.write(struct.pack("<I", len(header)))
    sys.stdout.buffer.write(header)
    for k in sorted(arrays.keys()):
        arr_bytes = arrays[k].tobytes()
        sys.stdout.buffer.write(struct.pack("<I", len(arr_bytes)))
        sys.stdout.buffer.write(arr_bytes)
    sys.stdout.buffer.flush()


def recv_request() -> dict | None:
    """Read a JSON request from stdin."""
    raw = sys.stdin.buffer.read(4)
    if len(raw) < 4:
        return None
    msg_len = struct.unpack("<I", raw)[0]
    data = sys.stdin.buffer.read(msg_len)
    if len(data) < msg_len:
        return None
    req = json.loads(data.decode("utf-8"))
    # Read attached numpy arrays in sorted key order (must match client _send())
    for k in sorted(req.keys()):
        v = req[k]
        if isinstance(v, dict) and v.get("__numpy__"):
            arr_len = struct.unpack("<I", sys.stdin.buffer.read(4))[0]
            arr_bytes = sys.stdin.buffer.read(arr_len)
            req[k] = np.frombuffer(arr_bytes, dtype=v["dtype"]).reshape(v["shape"])
    return req


# Signal ready
sys.stderr.write("[NPU Worker] Ready\n")
sys.stderr.flush()
send_response({"status": "ready",
               "providers": ort.get_available_providers()})

# Main loop
while True:
    try:
        req = recv_request()
        if req is None:
            break

        cmd = req.get("cmd")
        if cmd == "ping":
            send_response({"status": "pong"})

        elif cmd == "run":
            onnx_path = req["onnx_path"]
            inputs = {k: v for k, v in req.items()
                      if isinstance(v, np.ndarray)}
            t0 = time.perf_counter()
            sess = get_session(onnx_path)
            outputs = sess.run(None, inputs)
            elapsed = time.perf_counter() - t0
            resp = {"status": "ok", "elapsed_s": elapsed,
                    "provider": sess.get_providers()[0]}
            for i, out in enumerate(outputs):
                resp[f"output_{i}"] = out
            send_response(resp)

        elif cmd == "quit":
            send_response({"status": "bye"})
            break

        else:
            send_response({"status": "error", "message": f"Unknown cmd: {cmd}"})

    except Exception as e:
        try:
            send_response({"status": "error", "message": str(e)})
        except:
            break

sys.stderr.write("[NPU Worker] Exiting\n")
sys.stderr.flush()
'''


# ── NPU Bridge client ────────────────────────────────────────────────────────

class NPUBridge:
    """
    Client for the NPU worker subprocess. Manages lifecycle and provides
    a simple run() interface for ONNX inference on the NPU.
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._conda_python: Optional[Path] = None
        self._worker_script: Optional[Path] = None
        self._ready = False

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> str:
        """Start the NPU worker subprocess. Returns status message."""
        if self.is_running:
            return "NPU bridge already running"

        self._conda_python = find_conda_python()
        if not self._conda_python:
            return "ERROR: Ryzen AI conda Python not found"

        # Write worker script to temp location
        self._worker_script = Path(os.environ.get("TEMP", "/tmp")) / "_npu_worker.py"
        self._worker_script.write_text(_WORKER_SCRIPT, encoding="utf-8")

        env = os.environ.copy()
        env["RYZEN_AI_INSTALLATION_PATH"] = _SDK_PATH

        self._process = subprocess.Popen(
            [str(self._conda_python), str(self._worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        # Wait for ready signal
        try:
            resp = self._recv(timeout=30)
            if resp and resp.get("status") == "ready":
                providers = resp.get("providers", [])
                self._ready = True
                return (
                    f"NPU bridge started (PID {self._process.pid}). "
                    f"Providers: {', '.join(providers)}"
                )
            else:
                return f"NPU bridge started but unexpected response: {resp}"
        except Exception as e:
            self.stop()
            return f"ERROR: NPU bridge startup failed: {e}"

    def stop(self):
        """Stop the NPU worker subprocess."""
        if self._process:
            try:
                self._send({"cmd": "quit"})
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            self._ready = False
        if self._worker_script and self._worker_script.exists():
            try:
                self._worker_script.unlink()
            except Exception:
                pass

    def run(self, onnx_path: str, inputs: dict[str, np.ndarray],
            ) -> tuple[list[np.ndarray], float, str]:
        """
        Run ONNX inference on the NPU.

        Args:
            onnx_path: Path to the ONNX model file.
            inputs: Dict of input_name -> numpy array.

        Returns:
            (outputs, elapsed_seconds, provider_name)
        """
        if not self.is_running:
            raise RuntimeError("NPU bridge not running — call start() first")

        req = {"cmd": "run", "onnx_path": onnx_path}
        req.update(inputs)
        self._send(req)

        resp = self._recv(timeout=120)
        if resp is None:
            raise RuntimeError("NPU bridge: no response (worker may have crashed)")
        if resp.get("status") == "error":
            raise RuntimeError(f"NPU bridge error: {resp.get('message')}")

        outputs = []
        i = 0
        while f"output_{i}" in resp:
            outputs.append(resp[f"output_{i}"])
            i += 1

        return outputs, resp.get("elapsed_s", 0.0), resp.get("provider", "unknown")

    def _send(self, data: dict):
        """Send a request to the worker."""
        import json as _json
        arrays = {}
        clean = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                arrays[k] = v
                clean[k] = {"__numpy__": True, "dtype": str(v.dtype),
                             "shape": list(v.shape)}
            else:
                clean[k] = v

        header = _json.dumps(clean).encode("utf-8")
        self._process.stdin.write(struct.pack("<I", len(header)))
        self._process.stdin.write(header)
        for k in sorted(arrays.keys()):
            arr_bytes = arrays[k].tobytes()
            self._process.stdin.write(struct.pack("<I", len(arr_bytes)))
            self._process.stdin.write(arr_bytes)
        self._process.stdin.flush()

    def _recv(self, timeout: float = 30) -> Optional[dict]:
        """Receive a response from the worker (blocking; timeout not yet enforced)."""
        import json as _json

        # Read header length
        raw = self._process.stdout.read(4)
        if len(raw) < 4:
            return None
        header_len = struct.unpack("<I", raw)[0]
        header_data = self._process.stdout.read(header_len)
        if len(header_data) < header_len:
            return None

        resp = _json.loads(header_data.decode("utf-8"))

        # Read numpy arrays
        for k in sorted(resp.keys()):
            v = resp[k]
            if isinstance(v, dict) and v.get("__numpy__"):
                arr_len = struct.unpack("<I", self._process.stdout.read(4))[0]
                arr_bytes = self._process.stdout.read(arr_len)
                resp[k] = np.frombuffer(
                    arr_bytes, dtype=v["dtype"]
                ).reshape(v["shape"]).copy()  # copy to own memory

        return resp

    def __del__(self):
        self.stop()


# ── Module-level singleton ────────────────────────────────────────────────────
_bridge: Optional[NPUBridge] = None


def get_npu_bridge() -> NPUBridge:
    """Get or create the global NPU bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = NPUBridge()
    return _bridge
