"""Per-glyph font fallback for mixed-script text rendering.

PIL's ImageFont can't natively swap fonts for codepoints the primary font
doesn't cover. The standard idiom is to lay out the text run-by-run,
picking a font per glyph based on .getbbox() returning a non-None result.
This module hides that bookkeeping behind two helpers:

    pick_font_for_char(ch, primary, fallback)
    draw_text_with_fallback(draw, xy, text, primary, fallback, **kwargs)

For typical lofi titles (~60 chars) the per-glyph cost is trivial.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import ImageDraw, ImageFont

log = logging.getLogger(__name__)


def load_font(path: str | Path, size: int) -> ImageFont.FreeTypeFont:
    """Thin wrapper around ImageFont.truetype with a clearer error.

    Hard-fails on missing files — the env preflight in lofivid/env.py
    is the layer that catches this gracefully; reaching here with a
    missing path is a programming error, not a config error.
    """
    return ImageFont.truetype(str(path), size)


def _font_supports_char(font: ImageFont.FreeTypeFont, ch: str) -> bool:
    """Does this font cover the given codepoint?

    PIL exposes glyph coverage via the underlying freetype face; the
    cheapest portable check is font.getbbox(ch) returning a non-None
    result with non-zero dimensions. Whitespace gets a positive answer
    because it has metrics even without a glyph.
    """
    if ch.isspace():
        return True
    try:
        # In Pillow >= 10, getbbox returns None for unsupported glyphs.
        bbox = font.getbbox(ch)
    except (UnicodeEncodeError, OSError):
        return False
    if bbox is None:
        return False
    # A "tofu" glyph still produces a bbox. We additionally check that
    # the bbox has non-zero width — zero-width means no glyph was found.
    x0, _t, x1, _b = bbox
    return (x1 - x0) > 0


def pick_font_for_char(
    ch: str,
    primary: ImageFont.FreeTypeFont,
    fallback: ImageFont.FreeTypeFont | None,
) -> ImageFont.FreeTypeFont:
    """Return `primary` if it covers ch, else `fallback` (when provided)."""
    if fallback is None:
        return primary
    if _font_supports_char(primary, ch):
        return primary
    return fallback


def runs_by_font(
    text: str,
    primary: ImageFont.FreeTypeFont,
    fallback: ImageFont.FreeTypeFont | None,
) -> list[tuple[str, ImageFont.FreeTypeFont]]:
    """Group consecutive characters that share the same chosen font."""
    if not text:
        return []
    runs: list[tuple[str, ImageFont.FreeTypeFont]] = []
    cur_font = pick_font_for_char(text[0], primary, fallback)
    cur_text = text[0]
    for ch in text[1:]:
        f = pick_font_for_char(ch, primary, fallback)
        if f is cur_font:
            cur_text += ch
        else:
            runs.append((cur_text, cur_font))
            cur_text = ch
            cur_font = f
    runs.append((cur_text, cur_font))
    return runs


def measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    primary: ImageFont.FreeTypeFont,
    fallback: ImageFont.FreeTypeFont | None,
) -> tuple[int, int]:
    """Width × height of `text` rendered with per-glyph font selection."""
    runs = runs_by_font(text, primary, fallback)
    if not runs:
        return 0, 0
    width = 0
    max_h = 0
    for run_text, run_font in runs:
        x0, y0, x1, y1 = draw.textbbox((0, 0), run_text, font=run_font)
        width += x1 - x0
        max_h = max(max_h, y1 - y0)
    return width, max_h


def draw_text_with_fallback(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    primary: ImageFont.FreeTypeFont,
    fallback: ImageFont.FreeTypeFont | None,
    *,
    fill: str | tuple,
    shadow: tuple[str | tuple, tuple[int, int]] | None = None,
) -> None:
    """Draw `text` at `xy`, using `fallback` for codepoints `primary` lacks.

    `shadow`, when set, is `(color, (dx, dy))` — drawn first with offset.
    """
    x, y = xy
    runs = runs_by_font(text, primary, fallback)
    cursor_x = x
    if shadow is not None:
        scol, (dx, dy) = shadow
        sx = cursor_x + dx
        sy = y + dy
        for run_text, run_font in runs:
            draw.text((sx, sy), run_text, font=run_font, fill=scol)
            x0, _y0, x1, _y1 = draw.textbbox((0, 0), run_text, font=run_font)
            sx += x1 - x0
    for run_text, run_font in runs:
        draw.text((cursor_x, y), run_text, font=run_font, fill=fill)
        x0, _y0, x1, _y1 = draw.textbbox((0, 0), run_text, font=run_font)
        cursor_x += x1 - x0


def truncate_to_width(
    text: str,
    max_width_px: int,
    draw: ImageDraw.ImageDraw,
    primary: ImageFont.FreeTypeFont,
    fallback: ImageFont.FreeTypeFont | None,
    ellipsis: str = "…",
) -> str:
    """If `text` exceeds `max_width_px`, drop chars from the end and append `ellipsis`.

    Returns the original text when it already fits.
    """
    w, _ = measure_text(draw, text, primary, fallback)
    if w <= max_width_px:
        return text
    cropped = text
    while cropped:
        candidate = cropped + ellipsis
        cw, _ = measure_text(draw, candidate, primary, fallback)
        if cw <= max_width_px:
            return candidate
        cropped = cropped[:-1]
    return ellipsis
