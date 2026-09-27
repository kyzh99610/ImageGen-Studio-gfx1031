═══════════════════════════════════════════════════════════
  LoRA Training Datasets
═══════════════════════════════════════════════════════════

Each subfolder = one training project.

Structure per project:
  my_character/
    raw/          ← Drop your raw images here (PNG, JPG, WebP)
    captions/     ← (Optional) Put manual .txt caption files here
                     If empty, use auto-caption in the Training tab

The training tab will scan the "raw/" folder.
After you click "Prepare Dataset", a "_prepared_1024/" (or 512/768)
subfolder is created automatically with resized + bucketed images.

Workflow:
  1. Dump images into  raw/
  2. Open app → Train LoRA tab
  3. Set folder path to:  ...\training_datasets\my_character\raw
  4. Scan → Prepare → Auto-Caption → Review → Train!

Tips:
  • 15-50 images for character LoRAs
  • Include varied poses, angles, outfits, backgrounds
  • Solo character images work best
  • Body pillows (16:5) are handled natively — no forced square crop
