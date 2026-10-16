"""
ui_train.py - the handlers of the Train tab (scan / prepare / captions / Prepare + Start + Stop of a LoRA training), moved out of app._build_train_tab in round 28 (no behaviour change).
`build_handlers(ns, n_lora_dds)` returns (do_scan, do_prepare, do_autocaption, do_wdtag, do_unload_blip, do_refresh_cap_list, do_select_caption, do_save_caption, do_refresh_models,
do_train_prepare, do_train_start, do_train_stop, on_train_model); `ns` is the app module, so the module state the handlers read and rebind (`ns._trainer`, `ns._train_dataset`, `ns._train_assignments`,
the pipelines `ns._sd15` / `ns._sdxl`, `ns._smartsplit_pipe`) and the choice helpers are looked up at call time (the tests set them on the app module) and nothing here imports app.
`n_lora_dds` = how many Generate-tab LoRA dropdowns Start Training refreshes when a new LoRA was saved.
"""
from __future__ import annotations

import html
from pathlib import Path

import gradio as gr


def build_handlers(ns, n_lora_dds):
    def do_scan(folder, res_choice):
        if not folder or not Path(folder).is_dir():
            return '<p style="color:#f38ba8">❌ Folder not found.</p>', ""
        from backend.dataset_manager import scan_images, compute_bucket_assignments
        ns._train_dataset = scan_images(folder)
        if not ns._train_dataset:
            return '<p style="color:#f38ba8">❌ No images found.</p>', ""

        base_res = 512
        if "1024" in res_choice:
            base_res = 1024
        elif "768" in res_choice:
            base_res = 768

        result = compute_bucket_assignments(ns._train_dataset, base_res=base_res)
        ns._train_assignments = result["assignments"]

        # Build stats HTML
        total = len(ns._train_assignments)
        capt_count = sum(1 for a in ns._train_assignments if a.get("caption"))
        max_crop = max(a["crop_pct"] for a in ns._train_assignments)
        avg_crop = sum(a["crop_pct"] for a in ns._train_assignments) / total

        bkt_html = '<div class="tr-h">Bucket Assignments</div>'
        bkt_html += f'<p style="color:#a6e3a1;font-size:13px">✅ {total} images scanned · '
        bkt_html += f'{capt_count} have captions · '
        bkt_html += f'avg crop {avg_crop:.1f}% · max crop {max_crop:.1f}%</p>'
        bkt_html += '<table style="width:100%;font-size:13px;border-collapse:collapse;margin:6px 0">'
        bkt_html += ('<tr style="background:#313244"><th style="padding:4px 8px;text-align:left">Bucket</th>'
                     '<th style="padding:4px 8px;text-align:left">Ratio</th>'
                     '<th style="padding:4px 8px;text-align:left">Images</th></tr>')
        for bkt, cnt in sorted(result["stats"].items(),
                               key=lambda x: -x[1]):
            w, h = bkt.split("×")
            ratio = int(w) / max(int(h), 1)
            bkt_html += (f'<tr><td style="padding:3px 8px;color:#89b4fa">{bkt}</td>'
                         f'<td style="padding:3px 8px;color:#9399b2">{ratio:.2f}</td>'
                         f'<td style="padding:3px 8px">{cnt}</td></tr>')
        bkt_html += '</table>'
        if max_crop > 15:
            bkt_html += (f'<div class="tr-warn">Some images lose >{max_crop:.0f}% to cropping. '
                         'Consider cropping them manually to a cleaner aspect ratio.</div>')
        return f'<p style="color:#a6e3a1">✅ Found {total} images</p>', bkt_html

    def do_prepare(folder, trigger, res_choice, progress=gr.Progress()):
        if not ns._train_assignments:
            return "❌ Scan the folder first."
        from backend.dataset_manager import prepare_dataset
        base_res = 1024 if "1024" in res_choice else (768 if "768" in res_choice else 512)
        out_dir = Path(folder) / f"_prepared_{base_res}"

        def cb(cur, tot, msg):
            progress(cur / max(tot, 1), desc=msg)

        msg = prepare_dataset(ns._train_assignments, out_dir, trigger, callback=cb)
        return f'<p style="color:#a6e3a1">{msg}<br>'  \
               f'<span class="tr-dim">Use this folder for training: <code>{out_dir}</code></span></p>'

    def do_autocaption(folder, trigger, prefix, progress=gr.Progress()):
        if not folder or not Path(folder).is_dir():
            return "❌ Set folder first."
        from backend.auto_tagger import caption_batch
        from backend.dataset_manager import scan_images, write_caption
        imgs = scan_images(folder)
        if not imgs:
            return "❌ No images."

        def cb(cur, tot, msg):
            progress(cur / max(tot, 1), desc=msg)

        results = caption_batch(
            [i["path"] for i in imgs],
            trigger_word=trigger, prefix=prefix, callback=cb)
        for img in imgs:
            fn = Path(img["path"]).name
            if fn in results:
                write_caption(img["path"], results[fn])
        return f'<p style="color:#a6e3a1">✅ Captioned {len(results)} images</p>'

    def do_wdtag(folder, trigger, thr, progress=gr.Progress()):
        if not folder or not Path(folder).is_dir():
            return "❌ Set folder first."
        from backend.dataset_manager import scan_images, write_caption
        from backend.wd_tagger import tag_image, tags_text
        from PIL import Image as _Img
        imgs = scan_images(folder)
        if not imgs:
            return "❌ No images."
        trig = (trigger or "").strip().strip(",")
        done, failed = 0, []
        for i, it in enumerate(imgs):
            try:
                progress((i + 1) / len(imgs), desc=f"Tagging {Path(it['path']).name}")
            except Exception:
                pass
            try:
                with _Img.open(it["path"]) as im:
                    tags = tags_text(tag_image(im, general_threshold=float(thr or 0.35)), exclude=trig)
            except Exception as e:
                failed.append(f"{Path(it['path']).name}: {e}")
                continue
            write_caption(it["path"], f"{trig}, {tags}" if trig and tags else (trig or tags))
            done += 1
        err = (f'<br><span style="color:#f38ba8;">{len(failed)} failed: {html.escape(failed[0])}</span>'
               if failed else "")
        return (f'<p style="color:#a6e3a1">✅ Tagged {done} images (trigger word first, then Danbooru tags)'
                f'{err}</p>')

    def do_unload_blip():
        from backend.auto_tagger import unload_blip
        unload_blip()
        return '<p style="color:#a6e3a1">✅ BLIP model freed</p>'

    def do_refresh_cap_list(folder):
        if not folder or not Path(folder).is_dir():
            return gr.update(choices=[])
        from backend.dataset_manager import scan_images
        imgs = scan_images(folder)
        return gr.update(choices=[i["filename"] for i in imgs])

    def do_select_caption(folder, filename):
        if not folder or not filename:
            return None, ""
        from backend.dataset_manager import read_caption
        p = Path(folder) / filename
        if not p.exists():
            return None, ""
        cap = read_caption(str(p))
        try:
            img = ns.Image.open(p)
            return img, cap
        except Exception:
            return None, cap

    def do_save_caption(folder, filename, caption):
        if not folder or not filename:
            return "❌ No image selected."
        from backend.dataset_manager import write_caption
        p = Path(folder) / filename
        if not p.exists():
            return "❌ Image not found."
        write_caption(str(p), caption)
        return f'<p style="color:#a6e3a1">✅ Caption saved for {filename}</p>'

    def do_refresh_models():
        return gr.update(choices=ns._checkpoint_choices())

    def _parse_res(res_choice):
        if "1024" in res_choice:
            return 1024
        if "768" in res_choice:
            return 768
        return 512

    @ns.gpu_job
    def do_train_prepare(
        folder, trigger, res_choice, model_path,
        rank, alpha, lr, te_lr, epochs, train_te,
        target, sched, grad_accum, warmup, grad_ckpt, flip,
        save_every, seed, device_choice, output_name, train_clip_skip=2,
        drop_shared=False, drop_pct=80,
        progress=gr.Progress(),
    ):
        if not model_path:
            return "❌ Select a base model.", ""
        if not folder or not Path(folder).is_dir():
            return "❌ Set dataset folder.", ""

        from backend.lora_trainer import TrainingConfig, LoRATrainer, safe_output_name, sdxl_train_warning
        from config import LORAS_DIR

        # Training loads its own copy of the base model; a generation model left in
        # VRAM (6.6 GB for SDXL) would push the card into shared memory and crawl.
        freed = [p.current_model for p in (ns._sd15, ns._sdxl) if p.pipe is not None]
        for _p in (ns._sd15, ns._sdxl):
            if _p.pipe is not None:
                _p._unload()
        ns._smartsplit_pipe = None       # it holds its own copy of the model (and its VRAM)
        output_name = safe_output_name(output_name)

        # If a _prepared_ subfolder exists, use it
        base_res = _parse_res(res_choice)
        prep_dir = Path(folder) / f"_prepared_{base_res}"
        ds_dir = str(prep_dir) if prep_dir.is_dir() else folder

        cfg = TrainingConfig(
            base_model=model_path,
            lora_rank=int(rank),
            lora_alpha=int(alpha),
            train_te=train_te,
            target_coverage="full" if "full" in target else "attn",
            dataset_dir=ds_dir,
            trigger_word=trigger,
            resolution=base_res,
            learning_rate=float(lr),
            te_learning_rate=float(te_lr),
            lr_scheduler=sched,
            epochs=int(epochs),
            gradient_accumulation=int(grad_accum),
            warmup_ratio=float(warmup),
            gradient_checkpointing=grad_ckpt,
            flip_augment=flip,
            save_every_n_epochs=int(save_every),
            seed=int(seed),
            device=device_choice,
            output_dir=str(LORAS_DIR),
            output_name=output_name,
            clip_skip=2 if str(train_clip_skip) == "2" else 1,
            drop_shared_pct=(int(drop_pct or 80) if drop_shared else 0),
        )

        if ns._trainer is not None:
            ns._trainer.cleanup()

        ns._trainer = LoRATrainer(cfg)

        def cb(cur, tot, msg):
            progress(cur / max(tot, 1), desc=msg)

        msg = ns._trainer.prepare(callback=cb)
        log = (f'<p style="font-size:13px;color:#a6adc8">'
               f'Model: {Path(model_path).stem} ({cfg.model_type.upper()})<br>'
               f'Dataset: {ds_dir} ({len(ns._trainer.items)} images)<br>'
               f'LoRA rank={rank} alpha={alpha} | LR={lr} | TE={train_te}<br>'
               f'Epochs: {epochs} | Device: {device_choice}'
               + (f'<br>Unloaded the Generate tab model(s) to free VRAM: '
                  f'{", ".join(Path(str(m)).stem for m in freed)} — they reload '
                  f'automatically on your next Generate.' if freed else '') + '</p>')
        color = "#a6e3a1" if "✅" in msg else "#f38ba8"
        try:
            import torch as _t
            _vram = (_t.cuda.get_device_properties(_t.device(cfg.device if cfg.device.startswith("cuda") else "cuda:0")).total_memory / 2**30
                     if cfg.device in ("auto", "cuda") and _t.cuda.is_available() else None)
        except Exception:
            _vram = None
        _warn = sdxl_train_warning(cfg.model_type, _vram)
        if _warn:
            log += f'<p style="color:#f9e2af;font-size:13px;">{html.escape(_warn)}</p>'
        if getattr(ns._trainer, "mode_note", ""):
            log += f'<p style="font-size:13px;color:#a6adc8;">{html.escape(ns._trainer.mode_note)}</p>'
        if getattr(ns._trainer, "dropped_tags", None):
            log += ('<p style="font-size:13px;color:#a6adc8;">Let the trigger carry the character: left out of every caption (tag, captions): '
                    + html.escape(", ".join(f"{t} ({k})" for t, k in ns._trainer.dropped_tags)) + '</p>')
        return f'<p style="color:{color}">{msg}</p>', log

    def do_train_start(progress=gr.Progress()):
        no_upd = [gr.update()] * n_lora_dds
        if ns._trainer is None:
            return ('<p style="color:#f38ba8">❌ Click 📦 Prepare first.</p>', "", *no_upd)
        def cb(pct, msg):
            progress(pct, desc=msg)
            if str(msg).startswith("💾"):                    # a checkpoint was saved: a toast (the progress text is replaced a second later)
                try:
                    gr.Info(str(msg))
                except Exception:
                    pass

        try:
            result = ns._trainer.train(callback=cb)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (f'<p style="color:#f38ba8">❌ Training failed: {e}</p>', "", *no_upd)
        finally:
            # The trainer keeps its base model in RAM (6-14 GB of commit): release it, or the next Generate's checkpoint load is refused / crashes
            # (round 21: a 6.5 GB SDXL load was refused right after an SDXL training). Prepare again to train another LoRA.
            try:
                ns._trainer.cleanup()
            except Exception:
                pass
            ns._trainer = None
        color = "#a6e3a1" if "✅" in result else ("#fab387" if "⏹" in result else "#f38ba8")
        # New LoRA shows up in the Generate tab's slots right away
        lora_upd = gr.update(choices=ns._lora_choices())
        extra = ('<p class="tr-dim">It is now listed in the LoRA slots of the 🎨 Generate tab '
                 '— remember to add your trigger word to the prompt. The training model was released from memory (Prepare again to train another LoRA).</p>' if "✅" in result else
                 '<p class="tr-dim">The training model was released from memory — Prepare again to train.</p>')
        return (f'<p style="color:{color}">{result}</p>{extra}', "", *[lora_upd] * n_lora_dds)

    def do_train_stop():
        if ns._trainer:
            ns._trainer.abort()
        return '<p style="color:#fab387">⏹ Stop signal sent — will finish current step</p>'


    def on_train_model(model_path, res_choice):
        """Keep the target resolution matched to the base model's family."""
        from backend.sd_pipeline import _is_sdxl
        if not model_path:
            return gr.update()
        if _is_sdxl(model_path):
            return "1024 (SDXL / Pony / Illustrious)"
        return res_choice if res_choice and "SD 1.5" in res_choice else "512 (SD 1.5)"

    return (do_scan, do_prepare, do_autocaption, do_wdtag, do_unload_blip, do_refresh_cap_list, do_select_caption, do_save_caption,
            do_refresh_models, do_train_prepare, do_train_start, do_train_stop, on_train_model)
