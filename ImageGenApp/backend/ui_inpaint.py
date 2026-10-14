"""
ui_inpaint.py - the handlers of the Inpaint accordion (🖌 Inpaint, 🦋 Swap, 🖐 Re-draw hand), moved out of app._build_generate_tab in round 16 (no behaviour change).
`build_handlers(ns)` returns (do_inpaint_ui, do_swap_ui, do_hand_redraw_ui); `ns` is the app module, so everything the handlers use — the model (`ns.sd`, rebound on a model switch), the Generate tab's model helpers
(`ns._GEN_HELPERS`, registered when the tab is built), `ns._save_outputs`, the abort flag — is looked up at call time (the tests patch these on the app module) and nothing here imports app.
"""
from __future__ import annotations

import html
import time

import gradio as gr


def build_handlers(ns):
    def do_inpaint_ui(model_path, vae_path, lora1, w1, lora2, w2, lora3, w3, prompt, neg_prompt, scheduler,
                      steps, cfg, seed, clip_skip, editor, denoise, padding, progress=gr.Progress()):
        from backend.detail_tools import inpaint_region
        ns._generation_abort.clear()
        image, mask = ns._editor_parts(editor)
        if image is None:
            return [], '<p style="color:#f38ba8;">❌ Upload an image in the Inpaint box first.</p>', \
                gr.update(), gr.update(), gr.update()
        if mask is None:
            return [], '<p style="color:#f38ba8;">❌ Paint over the part you want redrawn.</p>', \
                gr.update(), gr.update(), gr.update()
        image, fixed = ns._fit_init_image(image, max_side=2048)
        if fixed:
            mask = mask.resize(image.size, ns.Image.NEAREST)
        steps, cfg, _w, _h, _b, seed, denoise, fixes = ns._clean_gen_args(steps, cfg, 512, 512, 1, seed, denoise,
                                                                      img2img=True)
        ok, status = ns._GEN_HELPERS["ensure_model"](model_path, vae_path, progress)
        if not ok:
            return [], (status if isinstance(status, str) else ""), status, gr.update(), gr.update()
        err = ns._GEN_HELPERS["sync_loras"]([(lora1, ns._num(w1, 0.8)), (lora2, ns._num(w2, 0.7)), (lora3, ns._num(w3, 0.7))], progress)
        if err:
            return [], f'<p style="color:#f38ba8;">❌ LoRA problem: {err}</p>', status, gr.update(), gr.update()
        cs = 2 if int(ns._num(clip_skip, 1)) >= 2 else 1

        def cb(step, total):
            if ns._generation_abort.is_set():
                raise ns._GenerationAborted()
            progress(step / total, desc=f"Inpainting: step {step}/{total}")
        t0 = time.time()
        try:
            out, used = inpaint_region(ns.sd, image, mask, prompt or "", neg_prompt or "", steps=ns._inpaint_steps(steps, denoise), cfg=cfg,
                                       denoise=denoise, seed=seed, scheduler=scheduler, clip_skip=cs,
                                       padding=int(ns._num(padding, 48)), step_callback=cb)
        except ns._GenerationAborted:
            return [], '<p style="color:#fab387;">⏹ Inpaint stopped.</p>', status, gr.update(), gr.update()
        except Exception as e:
            return [], f'<p style="color:#f38ba8;">❌ Inpaint failed: {html.escape(str(e)[:300])}</p>', \
                status, gr.update(), gr.update()
        saved = ns._save_outputs([out], dict(mode="inpaint", prompt=prompt or "", negative_prompt=neg_prompt or "",
                                          steps=steps, cfg_scale=cfg, seeds=[used], scheduler=scheduler,
                                          width=out.width, height=out.height, strength=denoise,
                                          inpaint_padding=int(ns._num(padding, 48)),
                                          clip_skip=cs if cs > 1 else None), pipe=ns.sd)
        msg = (f'<p style="color:#a6adc8;font-size:13px;">🖌 Inpainted (denoise {denoise:g}, seed {used}) in '
               f'{time.time() - t0:.1f}s · Saved: {saved[0].name if saved else "-"}'
               + (f'<br><span style="color:#f9e2af;">Adjusted: {"; ".join(fixes)}</span>' if fixes else "")
               + "</p>")
        return [out], msg, status, [out], [used]

    @ns.gpu_job
    def do_swap_ui(model_path, vae_path, lora1, w1, lora2, w2, lora3, w3, prompt, neg_prompt, scheduler,
                   steps, cfg, seed, clip_skip, editor, padding, item, progress=gr.Progress()):
        """🦋 Swap a hair accessory: paint over the old one, name the new one — one inpaint at denoise 1.0 with a hair-only
        prompt (detail_tools.swap_item). At 0.9–0.95 the old shape still shows through as lace."""
        from backend.detail_tools import SWAP_MIN_PADDING, swap_item, swap_prompt_from
        ns._generation_abort.clear()
        item = (item or "").strip()
        image, mask = ns._editor_parts(editor)
        if image is None:
            return [], '<p style="color:#f38ba8;">❌ Upload an image in the Inpaint box first.</p>', \
                gr.update(), gr.update(), gr.update()
        if mask is None:
            return [], '<p style="color:#f38ba8;">❌ Paint over the accessory you want to replace.</p>', \
                gr.update(), gr.update(), gr.update()
        if not item:
            return [], '<p style="color:#f38ba8;">❌ Type what should be there instead (e.g. black butterfly hair ornament).</p>', \
                gr.update(), gr.update(), gr.update()
        image, fixed = ns._fit_init_image(image, max_side=2048)
        if fixed:
            mask = mask.resize(image.size, ns.Image.NEAREST)
        steps, cfg, _w, _h, _b, seed, _d, fixes = ns._clean_gen_args(steps, cfg, 512, 512, 1, seed, 1.0, img2img=True)
        ok, status = ns._GEN_HELPERS["ensure_model"](model_path, vae_path, progress)
        if not ok:
            return [], (status if isinstance(status, str) else ""), status, gr.update(), gr.update()
        err = ns._GEN_HELPERS["sync_loras"]([(lora1, ns._num(w1, 0.8)), (lora2, ns._num(w2, 0.7)), (lora3, ns._num(w3, 0.7))], progress)
        if err:
            return [], f'<p style="color:#f38ba8;">❌ LoRA problem: {err}</p>', status, gr.update(), gr.update()
        cs = 2 if int(ns._num(clip_skip, 1)) >= 2 else 1
        pad = max(int(ns._num(padding, 48)), SWAP_MIN_PADDING)

        def cb(step, total):
            if ns._generation_abort.is_set():
                raise ns._GenerationAborted()
            progress(step / total, desc=f"Swapping: step {step}/{total}")
        t0 = time.time()
        try:
            out, used = swap_item(ns.sd, image, mask, item, prompt or "", neg_prompt or "", steps=steps, cfg=cfg, seed=seed,
                                  scheduler=scheduler, clip_skip=cs, padding=pad, step_callback=cb)
        except ns._GenerationAborted:
            return [], '<p style="color:#fab387;">⏹ Swap stopped.</p>', status, gr.update(), gr.update()
        except Exception as e:
            return [], f'<p style="color:#f38ba8;">❌ Swap failed: {html.escape(str(e)[:300])}</p>', \
                status, gr.update(), gr.update()
        saved = ns._save_outputs([out], dict(mode="inpaint", prompt=swap_prompt_from(prompt or "", item), negative_prompt=neg_prompt or "",
                                          steps=steps, cfg_scale=cfg, seeds=[used], scheduler=scheduler,
                                          width=out.width, height=out.height, strength=1.0, inpaint_padding=pad,
                                          inpaint_swap=item, clip_skip=cs if cs > 1 else None), pipe=ns.sd)
        msg = (f'<p style="color:#a6adc8;font-size:13px;">🦋 Swapped for “{html.escape(item)}” (seed {used}) in '
               f'{time.time() - t0:.1f}s · Saved: {saved[0].name if saved else "-"}'
               + (f'<br><span style="color:#f9e2af;">Adjusted: {"; ".join(fixes)}</span>' if fixes else "")
               + "</p>")
        return [out], msg, status, [out], [used]

    @ns.gpu_job
    def do_hand_redraw_ui(model_path, vae_path, lora1, w1, lora2, w2, lora3, w3, prompt, neg_prompt, scheduler,
                          steps, cfg, seed, clip_skip, editor, tries, progress=gr.Progress()):
        """🖐 Re-draw a hand: several re-draws (seed, seed + 1, …) of the painted or the biggest detected hand at 0.7 with a
        hand-only prompt (detail_tools.redraw_hand); the gallery shows the original first, then every try, each saved."""
        from backend.detail_tools import HAND_REDRAW_DENOISE, detect_hands, hand_mask, hand_prompt_from, redraw_hand
        ns._generation_abort.clear()
        image, mask = ns._editor_parts(editor)
        if image is None:
            return [], '<p style="color:#f38ba8;">❌ Upload an image in the Inpaint box first.</p>',                 gr.update(), gr.update(), gr.update()
        image, fixed = ns._fit_init_image(image, max_side=2048)
        if mask is not None and fixed:
            mask = mask.resize(image.size, ns.Image.NEAREST)
        found = ""
        if mask is None:
            hands = detect_hands(image, 6)
            if not hands:
                return [], ('<p style="color:#f38ba8;">❌ No hand found — paint over the hand you want re-drawn.</p>'),                     gr.update(), gr.update(), gr.update()
            mask = hand_mask(image.size, hands[0])
            found = f" (the biggest of {len(hands)} detected hand(s))" if len(hands) > 1 else " (the detected hand)"
        steps, cfg, _w, _h, _b, seed, _d, fixes = ns._clean_gen_args(steps, cfg, 512, 512, 1, seed, HAND_REDRAW_DENOISE, img2img=True)
        n = int(min(8, max(1, ns._num(tries, 4))))
        ok, status = ns._GEN_HELPERS["ensure_model"](model_path, vae_path, progress)
        if not ok:
            return [], (status if isinstance(status, str) else ""), status, gr.update(), gr.update()
        err = ns._GEN_HELPERS["sync_loras"]([(lora1, ns._num(w1, 0.8)), (lora2, ns._num(w2, 0.7)), (lora3, ns._num(w3, 0.7))], progress)
        if err:
            return [], f'<p style="color:#f38ba8;">❌ LoRA problem: {err}</p>', status, gr.update(), gr.update()
        cs = 2 if int(ns._num(clip_skip, 1)) >= 2 else 1
        cur = [0]

        def on_try(k):
            cur[0] = k
            ns._thermal_wait(progress)
            if ns._generation_abort.is_set():
                raise ns._GenerationAborted()

        def cb(step, total):
            if ns._generation_abort.is_set():
                raise ns._GenerationAborted()
            progress(step / total, desc=f"Hand try {cur[0] + 1}/{n}: step {step}/{total}")
        t0 = time.time()
        done, results = [], []            # redraw_hand appends every try to `done` as it finishes: a Stop or an error at try k keeps the k − 1 finished tries
        try:
            results = redraw_hand(ns.sd, image, mask, prompt or "", neg_prompt or "", tries=n, steps=steps, cfg=cfg, seed=seed,
                                  scheduler=scheduler, clip_skip=cs, step_callback=cb, on_try=on_try, out=done)
            stopped = ""
        except ns._GenerationAborted:
            results, stopped = done, f" · ⏹ stopped after {len(done)} of {n} tries"
        except Exception as e:
            if not done:
                return [], f'<p style="color:#f38ba8;">❌ Hand re-draw failed: {html.escape(str(e)[:300])}</p>',                 status, gr.update(), gr.update()
            results, stopped = done, f" · ⚠ stopped at try {len(done) + 1} of {n}: {html.escape(str(e)[:120])}"
        if not results:
            return [], '<p style="color:#fab387;">⏹ Hand re-draw stopped.</p>', status, gr.update(), gr.update()
        imgs = [r[0] for r in results]
        used = [r[1] for r in results]
        saved = ns._save_outputs(imgs, dict(mode="inpaint", prompt=hand_prompt_from(prompt or ""), negative_prompt=neg_prompt or "",
                                         steps=steps, cfg_scale=cfg, seeds=used, scheduler=scheduler,
                                         width=imgs[0].width, height=imgs[0].height, strength=HAND_REDRAW_DENOISE, inpaint_hand_redraw=True,
                                         clip_skip=cs if cs > 1 else None), pipe=ns.sd)
        shown = [image] + imgs
        msg = (f'<p style="color:#a6adc8;font-size:13px;">🖐 {len(imgs)} re-draw(s) of the hand{found} in {time.time() - t0:.1f}s '
               f'(seeds {used[0]}–{used[-1]}){stopped} — the first picture is the original; keep the try you like '
               f'(all saved: {html.escape(", ".join(p.name for p in saved[:2]))}{" …" if len(saved) > 2 else ""})'
               + (f'<br><span style="color:#f9e2af;">Adjusted: {"; ".join(fixes)}</span>' if fixes else "")
               + "</p>")
        return shown, msg, status, shown, [used[0]] + used

    return do_inpaint_ui, do_swap_ui, do_hand_redraw_ui
