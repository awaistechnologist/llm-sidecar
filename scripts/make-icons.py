"""Draw the app icons: the web app's, and the Android app's launcher icons.

    .venv/bin/python scripts/make-icons.py

The mark is a circle with a smaller one beside it — the thing you run, and
the sidecar next to it. Drawn rather than generated so it is identical at
every size and can be changed in one place. Needs Pillow (in the [images]
extra, or `pip install pillow`).
"""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "llm_sidecar" / "ui"
RES = ROOT / "mobile" / "android" / "app" / "src" / "main" / "res"

TOP_LEFT = (79, 70, 229)      # indigo-600, the dashboard accent
BOTTOM_RIGHT = (124, 58, 237)  # violet-600
SS = 4                        # supersampling factor for clean edges


def gradient(n: int) -> Image.Image:
    g = Image.new("RGB", (n, n))
    px = g.load()
    for y in range(n):
        for x in range(n):
            t = (x + y) / (2 * (n - 1))
            px[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(TOP_LEFT, BOTTOM_RIGHT))
    return g


def mark(draw: ImageDraw.ImageDraw, n: int, scale: float) -> None:
    """The two circles, in a 24-unit box centred in n px, at `scale` of it."""
    u = n * scale / 24
    ox = (n - 24 * u) / 2
    oy = (n - 24 * u) / 2

    def circle(cx, cy, r, fill):
        draw.ellipse([ox + (cx - r) * u, oy + (cy - r) * u,
                      ox + (cx + r) * u, oy + (cy + r) * u], fill=fill)

    # Shifted left by 1.6 units so the pair, not the big circle, is centred.
    circle(8.4, 12, 6.2, (255, 255, 255, 255))
    circle(17.9, 12, 3.1, (255, 255, 255, 195))


def icon(size: int, *, rounded: bool, scale: float) -> Image.Image:
    n = size * SS
    base = gradient(n).convert("RGBA")
    over = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    mark(ImageDraw.Draw(over), n, scale)
    out = Image.alpha_composite(base, over)
    if rounded:
        mask = Image.new("L", (n, n), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=int(n * 0.23), fill=255)
        out.putalpha(mask)
    return out.resize((size, size), Image.LANCZOS)


def foreground(size: int) -> Image.Image:
    """Adaptive-icon foreground: the mark alone on transparency, inside the
    66% safe zone Android guarantees will not be cropped by any mask."""
    n = size * SS
    over = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    mark(ImageDraw.Draw(over), n, 0.50)
    return over.resize((size, size), Image.LANCZOS)


def main() -> None:
    UI.mkdir(parents=True, exist_ok=True)
    icon(192, rounded=True, scale=0.62).save(UI / "icon-192.png", optimize=True)
    icon(512, rounded=True, scale=0.62).save(UI / "icon-512.png", optimize=True)
    # Maskable: full bleed, mark inside the central 80% safe zone.
    icon(512, rounded=False, scale=0.50).save(UI / "icon-maskable-512.png", optimize=True)
    # iOS rounds the corners itself and shows transparency as black.
    icon(180, rounded=False, scale=0.58).save(UI / "apple-touch-icon.png", optimize=True)
    print("web icons ->", UI)

    if RES.parent.exists():
        legacy = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}
        fg = {"mdpi": 108, "hdpi": 162, "xhdpi": 216, "xxhdpi": 324, "xxxhdpi": 432}
        for d, px in legacy.items():
            folder = RES / f"mipmap-{d}"
            folder.mkdir(parents=True, exist_ok=True)
            icon(px, rounded=True, scale=0.62).save(folder / "ic_launcher.png", optimize=True)
            foreground(fg[d]).save(folder / "ic_launcher_foreground.png", optimize=True)
        any_dpi = RES / "mipmap-anydpi-v26"
        any_dpi.mkdir(parents=True, exist_ok=True)
        (any_dpi / "ic_launcher.xml").write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">\n'
            '    <background android:drawable="@drawable/ic_launcher_background"/>\n'
            '    <foreground android:drawable="@mipmap/ic_launcher_foreground"/>\n'
            '    <monochrome android:drawable="@mipmap/ic_launcher_foreground"/>\n'
            '</adaptive-icon>\n', encoding="utf-8")
        draw = RES / "drawable"
        draw.mkdir(parents=True, exist_ok=True)
        (draw / "ic_launcher_background.xml").write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<shape xmlns:android="http://schemas.android.com/apk/res/android">\n'
            '    <gradient android:angle="315" android:startColor="#4F46E5" android:endColor="#7C3AED"/>\n'
            '</shape>\n', encoding="utf-8")
        print("android icons ->", RES)


if __name__ == "__main__":
    main()
