"""
smoke_gpu.py — quick end-to-end check on the real GPU (the CPU suite can't see ZLUDA / VRAM behaviour).

    ImageGenApp\\run_zluda.bat smoke_gpu.py            SD 1.5 checks (~2–4 min warm)
    ImageGenApp\\run_zluda.bat smoke_gpu.py --sdxl     + a short SDXL round

Close the app first (one process owns the GPU). Uses the first installed SD 1.5 checkpoint and LoRA (or the ones
given with --ckpt / --lora, file names). Exit code 0 = every check passed.
Checks: images aren't NaN/black · same seed reproduces · a LoRA changes the weights and removing it restores them
bit-exactly · img2img · inpaint leaves pixels outside the mask untouched · face / hand detail run · unloading frees
VRAM · (--sdxl) SDXL generate + unload.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402,F401  (cuDNN off, env)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

RESULTS = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn() or ""
        RESULTS.append((True, name))
        print(f"  ✅ {name} ({time.time() - t0:.1f}s) {detail}", flush=True)
    except Exception as e:
        RESULTS.append((False, name))
        print(f"  ❌ {name}: {type(e).__name__}: {e}", flush=True)


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv) else default


def pick(items, arch_fn, arch, wanted=None):
    if wanted:
        hit = [p for n, p in items if n.lower() == wanted.lower() or os.path.splitext(n)[0].lower() == wanted.lower()]
        if hit:
            return hit[0]
    return next((p for n, p in items if arch_fn(p) == arch), None)


def ok_image(im, label):
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    assert np.isfinite(a).all() and a.std() > 8, f"{label}: flat / black image (std {a.std():.1f})"
    return a


def vram_gb():
    return torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def main() -> int:
    from backend.model_manager import list_checkpoints, list_loras, checkpoint_arch, lora_arch
    from backend.sd_pipeline import SDPipeline
    from backend import detail_tools as dt
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    ckpt = pick(list_checkpoints(), checkpoint_arch, "sd1", arg("--ckpt"))
    lora = pick(list_loras(), lora_arch, "sd1", arg("--lora"))
    if not ckpt:
        print("no SD 1.5 checkpoint installed"); return 2
    print(f"checkpoint: {os.path.basename(ckpt)} · LoRA: {os.path.basename(lora) if lora else '(none)'}", flush=True)
    sd = SDPipeline()
    v0 = vram_gb()
    state = {}
    P, N = "1girl, solo, smile, grey hair, red eyes, upper body, white dress", "lowres, bad anatomy"
    gen = lambda **k: sd.txt2img(**{**dict(prompt=P, negative_prompt=N, width=384, height=512, steps=10,
                                             cfg_scale=7, seed=1234, scheduler="DPM++ 2M Karras"), **k})[0]

    check("load model", lambda: sd.load_model(ckpt)[:60])

    def t_gen():
        state["img"] = gen()[0]
        state["a"] = ok_image(state["img"], "txt2img")
    check("txt2img 384×512", t_gen)

    def t_repro():
        b = ok_image(gen()[0], "repeat")
        d = np.abs(b - state["a"])
        assert d.mean() < 3, f"same seed differs: mean {d.mean():.2f}/255"
        return f"mean|Δ| {d.mean():.2f}/255 (native convs aren't bit-exact)"
    check("same seed reproduces", t_repro)

    def t_lora():
        if not lora:
            return "skipped (no SD 1.5 LoRA)"
        names = [n for n, _ in sd.pipe.unet.named_parameters() if "attn2.to_k.weight" in n][:4]
        before = {n: sd.pipe.unet.get_parameter(n).detach().float().cpu().clone() for n in names}
        sd.load_lora(lora, 0.8, slot=0)
        changed = sum(not torch.equal(sd.pipe.unet.get_parameter(n).detach().float().cpu(), before[n]) for n in names)
        ok_image(gen()[0], "with LoRA")
        sd.remove_lora(0)
        same = all(torch.equal(sd.pipe.unet.get_parameter(n).detach().float().cpu(), before[n]) for n in names)
        assert changed > 0, "the LoRA didn't change any sampled weight"
        assert same, "weights not restored bit-exactly after removing the LoRA"
        return f"{changed}/{len(names)} sampled layers changed, restored exactly"
    check("LoRA apply / remove", t_lora)

    def t_i2i():
        ok_image(sd.img2img(state["img"], P, N, strength=0.5, steps=12, cfg_scale=7, seed=5)[0][0], "img2img")
    check("img2img", t_i2i)

    def t_inpaint():
        img = state["img"]
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).rectangle([150, 200, 250, 300], fill=255)
        out, _ = dt.inpaint_region(sd, img, mask, P, N, steps=12, denoise=0.75, seed=3)
        a, b = np.asarray(img, np.int16), np.asarray(out, np.int16)
        far = np.ones(a.shape[:2], bool); far[180:320, 130:270] = False          # well outside the feathered edge
        assert np.abs(a - b)[far].max() == 0, "pixels outside the mask changed"
        assert np.abs(a - b)[200:300, 150:250].mean() > 1, "the masked area didn't change"
    check("inpaint (outside the mask untouched)", t_inpaint)

    def t_face_hand():
        out, nf = dt.face_detail(sd, state["img"], P, N, denoise=0.4, steps=20, seed=7)
        out, nh = dt.hand_detail(sd, out, P, N, denoise=0.35, steps=20, seed=7)
        ok_image(out, "detail")
        return f"{nf} face(s), {nh} hand(s)"
    check("face + hand detail", t_face_hand)

    def t_unload():
        sd._unload()
        import gc
        gc.collect(); torch.cuda.empty_cache()
        held = vram_gb() - v0
        assert held < 0.3, f"{held:.2f} GB still allocated after unload"
        return f"{held:.2f} GB held after unload"
    check("unload frees VRAM", t_unload)

    if "--sdxl" in sys.argv:
        from backend.sdxl_pipeline import SDXLPipeline
        xl_ckpt = pick(list_checkpoints(), checkpoint_arch, "sdxl", arg("--xl-ckpt"))
        if not xl_ckpt:
            print("  (no SDXL checkpoint — SDXL round skipped)")
        else:
            xl = SDXLPipeline()
            check(f"SDXL load {os.path.basename(xl_ckpt)}", lambda: xl.load_model(xl_ckpt)[:60])
            check("SDXL txt2img 768×768", lambda: ok_image(
                xl.txt2img(prompt=P, negative_prompt=N, width=768, height=768, steps=8, cfg_scale=6, seed=1,
                           scheduler="DPM++ 2M AYS")[0][0], "SDXL") is not None and None)

            def t_xl_unload():
                xl._unload()
                import gc
                gc.collect(); torch.cuda.empty_cache()
                held = vram_gb() - v0
                assert held < 0.3, f"{held:.2f} GB still allocated after SDXL unload"
                return f"{held:.2f} GB held"
            check("SDXL unload frees VRAM", t_xl_unload)

    failed = [n for ok, n in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed" + (f" — FAILED: {', '.join(failed)}" if failed else ""),
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)          # ZLUDA can hang in interpreter shutdown
