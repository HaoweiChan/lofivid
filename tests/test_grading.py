"""Tests for the duotone + paper-border post-process helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image

from lofivid.visuals._grading import duotone, grade, paper_border


def _solid(rgb: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Image.Image:
    return Image.new("RGB", size, rgb)


def test_duotone_pure_black_maps_to_shadow():
    img = _solid((0, 0, 0))
    out = duotone(img, shadow_rgb=(40, 22, 8), highlight_rgb=(244, 222, 184))
    arr = np.asarray(out)
    # Allow ±1 rounding error from float→uint8 conversion
    assert np.all(np.abs(arr - np.array((40, 22, 8))) <= 1)


def test_duotone_pure_white_maps_to_highlight():
    img = _solid((255, 255, 255))
    out = duotone(img, shadow_rgb=(40, 22, 8), highlight_rgb=(244, 222, 184))
    arr = np.asarray(out)
    assert np.all(np.abs(arr - np.array((244, 222, 184))) <= 1)


def test_duotone_midgray_lands_between_endpoints():
    img = _solid((128, 128, 128))
    out = duotone(img, shadow_rgb=(0, 0, 0), highlight_rgb=(200, 200, 200))
    arr = np.asarray(out)
    # 128/255 ≈ 0.502 → ≈ 100. Allow generous tolerance.
    assert np.all(np.abs(arr - 100) <= 2)


def test_paper_border_pads_dimensions():
    img = _solid((128, 128, 128), size=(100, 50))
    out = paper_border(img, border_pct=0.1)
    expected_border = int(min(100, 50) * 0.1)  # 5
    assert out.size == (100 + 2 * expected_border, 50 + 2 * expected_border)


def test_paper_border_grain_is_deterministic_with_rng():
    img = _solid((180, 180, 180))
    rng_a = np.random.default_rng(seed=42)
    rng_b = np.random.default_rng(seed=42)
    out_a = paper_border(img, rng=rng_a)
    out_b = paper_border(img, rng=rng_b)
    assert np.array_equal(np.asarray(out_a), np.asarray(out_b))


def test_grade_combines_duotone_and_border_when_requested():
    img = _solid((128, 128, 128))
    rng = np.random.default_rng(0)
    out = grade(img, (0, 0, 0), (200, 200, 200), with_border=True, rng=rng)
    # Border increases dimensions; duotone changes hue toward grey-200.
    assert out.size[0] > img.size[0]
    assert out.size[1] > img.size[1]


def test_grade_skips_border_when_with_border_false():
    img = _solid((128, 128, 128))
    out = grade(img, (0, 0, 0), (200, 200, 200), with_border=False)
    assert out.size == img.size


def test_legacy_reexport_from_preview_themes():
    """Confirm scripts/preview_themes.py re-exports the canonical helpers.

    `preview_greenhouse_cast.py` imports `duotone, paper_border` from there;
    we keep that working after the refactor.
    """
    from lofivid.visuals._grading import duotone as g_duotone
    from lofivid.visuals._grading import paper_border as g_paper_border
    from scripts.preview_themes import duotone as pt_duotone
    from scripts.preview_themes import paper_border as pt_paper_border

    assert pt_duotone is g_duotone
    assert pt_paper_border is g_paper_border
