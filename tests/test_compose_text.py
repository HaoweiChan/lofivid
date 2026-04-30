"""Tests for lofivid/compose/_text.py — per-glyph font fallback helpers."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

DEJAVU_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
DEJAVU_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
WQY_CJK = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def _dejavu_or_skip() -> Path:
    if not DEJAVU_BOLD.exists():
        pytest.skip("DejaVu Bold font not found on this system")
    return DEJAVU_BOLD


def _wqy_or_skip() -> Path:
    if not WQY_CJK.exists():
        pytest.skip("WQY Zenhei CJK font not found on this system")
    return WQY_CJK


def _make_draw() -> ImageDraw.ImageDraw:
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    return ImageDraw.Draw(img)


# ---------------------------------------------------------------------------
# runs_by_font
# ---------------------------------------------------------------------------

def test_runs_by_font_ascii_single_run() -> None:
    from lofivid.compose._text import load_font, runs_by_font

    font_path = _dejavu_or_skip()
    primary = load_font(font_path, 20)
    runs = runs_by_font("Hello World", primary, None)
    assert len(runs) == 1
    assert runs[0][0] == "Hello World"
    assert runs[0][1] is primary


def test_runs_by_font_empty_text() -> None:
    from lofivid.compose._text import load_font, runs_by_font

    font_path = _dejavu_or_skip()
    primary = load_font(font_path, 20)
    runs = runs_by_font("", primary, None)
    assert runs == []


def test_runs_by_font_mixed_cjk_ascii() -> None:
    """runs_by_font groups chars by chosen font; all text must be preserved.

    DejaVu renders tofu for CJK (non-zero bbox) so the primary font "covers"
    those chars from PIL's perspective and all chars stay in one run. The
    important invariant is that no text is lost: reconstructing the runs
    produces the original string.
    """
    from lofivid.compose._text import load_font, runs_by_font

    primary_path = _dejavu_or_skip()
    fallback_path = _wqy_or_skip()

    primary = load_font(primary_path, 20)
    fallback = load_font(fallback_path, 20)

    text = "ABC工作DEF"
    runs = runs_by_font(text, primary, fallback)

    # All text must be accounted for — no characters lost
    reconstructed = "".join(r[0] for r in runs)
    assert reconstructed == text

    # There must be at least one run
    assert len(runs) >= 1


def test_runs_by_font_no_fallback_stays_primary() -> None:
    """When fallback is None, all runs use primary regardless of codepoints."""
    from lofivid.compose._text import load_font, runs_by_font

    font_path = _dejavu_or_skip()
    primary = load_font(font_path, 20)
    runs = runs_by_font("ABC工作", primary, None)
    assert all(r[1] is primary for r in runs)


# ---------------------------------------------------------------------------
# truncate_to_width
# ---------------------------------------------------------------------------

def test_truncate_to_width_fits_unchanged() -> None:
    from lofivid.compose._text import load_font, truncate_to_width

    font_path = _dejavu_or_skip()
    font = load_font(font_path, 16)
    draw = _make_draw()
    result = truncate_to_width("Hi", 10000, draw, font, None)
    assert result == "Hi"


def test_truncate_to_width_truncates_with_ellipsis() -> None:
    from lofivid.compose._text import load_font, measure_text, truncate_to_width

    font_path = _dejavu_or_skip()
    font = load_font(font_path, 20)
    draw = _make_draw()
    # Pick a very small max width so truncation is forced
    long_text = "This is a very long title that should be truncated"
    result = truncate_to_width(long_text, 50, draw, font, None)
    assert result != long_text
    assert result.endswith("…")
    w, _ = measure_text(draw, result, font, None)
    assert w <= 50


def test_truncate_to_width_minimal_budget_returns_ellipsis() -> None:
    from lofivid.compose._text import load_font, truncate_to_width

    font_path = _dejavu_or_skip()
    font = load_font(font_path, 20)
    draw = _make_draw()
    # Budget of 1px — can't fit even one char + ellipsis; should return bare ellipsis
    result = truncate_to_width("Hello", 1, draw, font, None)
    assert result == "…"


# ---------------------------------------------------------------------------
# measure_text
# ---------------------------------------------------------------------------

def test_measure_text_returns_positive_dimensions() -> None:
    from lofivid.compose._text import load_font, measure_text

    font_path = _dejavu_or_skip()
    font = load_font(font_path, 20)
    draw = _make_draw()
    w, h = measure_text(draw, "Hello", font, None)
    assert w > 0
    assert h > 0


def test_measure_text_empty_returns_zero() -> None:
    from lofivid.compose._text import load_font, measure_text

    font_path = _dejavu_or_skip()
    font = load_font(font_path, 20)
    draw = _make_draw()
    w, h = measure_text(draw, "", font, None)
    assert w == 0
    assert h == 0
