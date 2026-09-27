"""Generate assets/icon.ico + icon.png for the desktop window (run once; bundled Python)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

S = 1024
out = Path(__file__).parent / "assets"
out.mkdir(exist_ok=True)

# Diagonal gradient: Catppuccin blue -> mauve (the app's accent colours)
a, b = (137, 180, 250), (203, 166, 247)
grad = Image.new("RGB", (S, S))
px = grad.load()
for y in range(S):
    for x in range(S):
        t = (x + y) / (2 * (S - 1))
        px[x, y] = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle((40, 40, S - 40, S - 40), radius=210, fill=255)
icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
icon.paste(grad, (0, 0), mask)

# A four-point sparkle (image generation) plus a small companion sparkle
d = ImageDraw.Draw(icon)


def sparkle(cx, cy, r, w, fill):
    pts = [(cx, cy - r), (cx + w, cy - w), (cx + r, cy), (cx + w, cy + w),
           (cx, cy + r), (cx - w, cy + w), (cx - r, cy), (cx - w, cy - w)]
    d.polygon(pts, fill=fill)


shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
sd = ImageDraw.Draw(shadow)
for (cx, cy, r, w) in ((470, 560, 330, 70), (760, 280, 140, 30)):
    pts = [(cx, cy - r), (cx + w, cy - w), (cx + r, cy), (cx + w, cy + w),
           (cx, cy + r), (cx - w, cy + w), (cx - r, cy), (cx - w, cy - w)]
    sd.polygon([(x + 14, y + 18) for x, y in pts], fill=(30, 30, 46, 120))
icon = Image.alpha_composite(icon, shadow.filter(ImageFilter.GaussianBlur(14)))
d = ImageDraw.Draw(icon)
sparkle(470, 560, 330, 70, (30, 30, 46, 255))
sparkle(760, 280, 140, 30, (30, 30, 46, 255))
sparkle(470, 560, 150, 32, (249, 226, 175, 255))   # warm core

icon.resize((512, 512), Image.LANCZOS).save(out / "icon.png")
icon.save(out / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("wrote", out / "icon.ico", out / "icon.png")
