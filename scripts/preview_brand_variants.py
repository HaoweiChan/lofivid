"""Re-typeset the existing _raw.png previews with multiple title layouts
and accent colours so we can pick a brand without re-rendering keyframes.

Outputs (in `previews/<backend>/variants/`):

    {scene}_{layout}_{accent}.png

Combinations:
  layout = "centred" | "stacked" | "single_line"
  accent = "rust" | "teal" | "mustard" | "charcoal" | "ink"

Usage (no GPU needed):
    source .venv/bin/activate
    python scripts/preview_brand_variants.py                  # default: flux_klein
    python scripts/preview_brand_variants.py flux_klein
    python scripts/preview_brand_variants.py z_image_turbo
    python scripts/preview_brand_variants.py sdxl             # legacy

If raws live directly under `previews/` (legacy layout) pass `.` as the
backend name to read from there.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PREVIEW_ROOT = ROOT / "previews"

KICKER = "LOFIVID  ・  PLAYLIST"
TITLE_SINGLE = "WORK CAFE JAZZ"
TITLE_STACKED = ("WORK CAFE", "JAZZ")
ZH_TW = "工作日   咖啡   爵士節奏"
TRACKLIST = (
    "01 cafe afternoon   /   02 rainy window   /   03 late night booth   "
    "/   04 vinyl spin   /   05 latte art   /   06 bookshelf hour"
)

FONT_DISPLAY = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_KICKER = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_BODY = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


@dataclass(frozen=True)
class Accent:
    name: str
    colour: tuple[int, int, int]
    body_colour: tuple[int, int, int]   # for kicker / tracklist (calmer than the title)


ACCENTS = [
    Accent("rust",     (168, 52, 30),  (40, 24, 16)),
    Accent("teal",     (24, 80, 78),   (16, 36, 36)),
    Accent("mustard",  (193, 138, 16), (52, 36, 8)),
    Accent("charcoal", (38, 36, 32),   (38, 36, 32)),
    Accent("ink",      (10, 22, 64),   (10, 22, 64)),
]


def _load(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def _draw_kicker_and_tracklist(
    img: Image.Image, kicker_y: int, body_colour: tuple[int, int, int]
) -> tuple[ImageDraw.ImageDraw, int, int]:
    """Common chrome (kicker on top, tracklist on bottom). Returns the draw
    object plus the y-baseline below the kicker so the caller can stack the
    title underneath."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    f_kicker = _load(FONT_KICKER, max(int(h * 0.022), 14))
    f_tracklist = _load(FONT_BODY, max(int(h * 0.022), 14))

    kw, kh = _measure(draw, KICKER, f_kicker)
    draw.text(((w - kw) / 2, kicker_y), KICKER, font=f_kicker, fill=body_colour)
    next_y = kicker_y + kh + int(h * 0.025)

    trw, trh = _measure(draw, TRACKLIST, f_tracklist)
    tr_y = h - int(h * 0.045) - trh
    draw.text(((w - trw) / 2, tr_y), TRACKLIST, font=f_tracklist, fill=body_colour)

    return draw, next_y, kh


def _draw_zh(
    draw: ImageDraw.ImageDraw, w: int, h: int, accent_colour: tuple[int, int, int], y: int
) -> None:
    f_cjk = _load(FONT_CJK, max(int(h * 0.062), 32))
    cw, _ = _measure(draw, ZH_TW, f_cjk)
    draw.text(((w - cw) / 2, y), ZH_TW, font=f_cjk, fill=accent_colour)


def compose_centred(raw: Image.Image, accent: Accent) -> Image.Image:
    """Two-line title, both lines centred (no right-offset)."""
    img = raw.copy().convert("RGB")
    w, h = img.size
    draw, title_top, _ = _draw_kicker_and_tracklist(img, int(h * 0.06), accent.body_colour)
    title_size = max(int(h * 0.115), 60)
    f_title = _load(FONT_DISPLAY, title_size)

    line1, line2 = TITLE_STACKED
    tw1, th1 = _measure(draw, line1, f_title)
    tw2, th2 = _measure(draw, line2, f_title)
    line1_y = title_top
    line1_x = (w - tw1) / 2
    line2_y = line1_y + th1 - int(title_size * 0.08)
    line2_x = (w - tw2) / 2
    draw.text((line1_x, line1_y), line1, font=f_title, fill=accent.colour)
    draw.text((line2_x, line2_y), line2, font=f_title, fill=accent.colour)

    _draw_zh(draw, w, h, accent.colour, line2_y + th2 + int(h * 0.045))
    return img


def compose_stacked(raw: Image.Image, accent: Accent) -> Image.Image:
    """Two-line title, with the second line right-shifted. Same as v2; kept
    so the variant grid shows it next to the cleaner centred option."""
    img = raw.copy().convert("RGB")
    w, h = img.size
    draw, title_top, _ = _draw_kicker_and_tracklist(img, int(h * 0.06), accent.body_colour)
    title_size = max(int(h * 0.115), 60)
    f_title = _load(FONT_DISPLAY, title_size)

    line1, line2 = TITLE_STACKED
    tw1, th1 = _measure(draw, line1, f_title)
    tw2, th2 = _measure(draw, line2, f_title)
    line1_y = title_top
    line1_x = (w - tw1) / 2
    line2_y = line1_y + th1 - int(title_size * 0.08)
    line2_x = (w - tw2) / 2 + int(w * 0.10)
    draw.text((line1_x, line1_y), line1, font=f_title, fill=accent.colour)
    draw.text((line2_x, line2_y), line2, font=f_title, fill=accent.colour)

    _draw_zh(draw, w, h, accent.colour, line2_y + th2 + int(h * 0.045))
    return img


def compose_single_line(raw: Image.Image, accent: Accent) -> Image.Image:
    """One-line title (`WORK CAFE JAZZ`). Smaller font so it fits."""
    img = raw.copy().convert("RGB")
    w, h = img.size
    draw, title_top, _ = _draw_kicker_and_tracklist(img, int(h * 0.06), accent.body_colour)
    title_size = max(int(h * 0.085), 44)
    f_title = _load(FONT_DISPLAY, title_size)

    tw, th = _measure(draw, TITLE_SINGLE, f_title)
    title_x = (w - tw) / 2
    title_y = title_top
    draw.text((title_x, title_y), TITLE_SINGLE, font=f_title, fill=accent.colour)

    _draw_zh(draw, w, h, accent.colour, title_y + th + int(h * 0.045))
    return img


LAYOUTS = {
    "centred": compose_centred,
    "stacked": compose_stacked,
    "single_line": compose_single_line,
}


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "flux_klein"
    src_dir = PREVIEW_ROOT if backend == "." else PREVIEW_ROOT / backend
    out_dir = src_dir / "variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    raws = sorted(src_dir.glob("*_raw.png"))
    if not raws:
        print(f"No *_raw.png in {src_dir} — "
              f"run `python scripts/preview_workday_cafe.py {backend}` first.")
        sys.exit(1)

    for raw_path in raws:
        scene = raw_path.stem.removesuffix("_raw")
        with Image.open(raw_path) as raw:
            for layout, fn in LAYOUTS.items():
                for acc in ACCENTS:
                    out = fn(raw, acc)
                    out_path = out_dir / f"{scene}_{layout}_{acc.name}.png"
                    out.save(out_path)
        print(f"  → {scene}: {len(LAYOUTS) * len(ACCENTS)} variants")
    print(f"\nDone. {len(raws)} scenes × {len(LAYOUTS)} layouts × {len(ACCENTS)} accents = "
          f"{len(raws) * len(LAYOUTS) * len(ACCENTS)} files in {out_dir}/")


if __name__ == "__main__":
    main()
