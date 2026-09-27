# Copilot instructions

All project context for AI assistants lives in [`AGENTS.md`](../AGENTS.md) at the repo root —
read it first and keep it up to date instead of adding content here.

Key rules (details in AGENTS.md):
- Launch/run GPU code via `ImageGenApp\launch.bat` or `ImageGenApp\run_zluda.bat <script.py>`; `import config` first.
- cuDNN/MIOpen stays **disabled** on AMD/ZLUDA; never re-add `AMD_SERIALIZE_KERNEL=3` or CPU-only VAE workarounds.
- Always pass `torch_dtype=self.dtype` to diffusers `from_pipe()`.
- Never hardcode DirectML device indices or API keys.
- `.bat` files are CP1252 — edit byte-wise.
