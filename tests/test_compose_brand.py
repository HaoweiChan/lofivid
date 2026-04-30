"""Tests for lofivid/compose/brand.py — persistent brand text overlay renderer."""
from __future__ import annotations

from pathlib import Path

import pytest

DEJAVU_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
DEJAVU_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _dejavu_or_skip() -> tuple[Path, Path]:
    if not DEJAVU_BOLD.exists() or not DEJAVU_REGULAR.exists():
        pytest.skip("DejaVu fonts not found on this system")
    return DEJAVU_BOLD, DEJAVU_REGULAR


def _make_layers(bold: Path, regular: Path):
    """Build a representative 3-layer list: kicker, title, tracklist."""
    from lofivid.styles.schema import TextLayerSpec

    return [
        TextLayerSpec(
            text="LOFIVID  ·  PLAYLIST",
            font_path=bold,
            size_pct=0.022,
            color="#281810",
            position="top_centre",
            margin_pct=0.06,
        ),
        TextLayerSpec(
            text="WORK CAFE JAZZ",
            font_path=bold,
            size_pct=0.085,
            color="#A8341E",
            shadow_color="#00000055",
            position="top_centre",
            margin_pct=0.11,
        ),
        TextLayerSpec(
            text="01 cafe afternoon   /   02 rainy window   /   03 late night booth",
            font_path=regular,
            size_pct=0.022,
            color="#3C281E",
            position="bottom_centre",
            margin_pct=0.045,
        ),
    ]


# ---------------------------------------------------------------------------
# render_brand_layer
# ---------------------------------------------------------------------------

def test_render_brand_layer_creates_png(tmp_path: Path) -> None:
    bold, regular = _dejavu_or_skip()
    from lofivid.compose.brand import render_brand_layer

    layers = _make_layers(bold, regular)
    out = render_brand_layer(layers, 1920, 1080, tmp_path)

    assert out is not None
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_brand_layer_correct_mode_and_size(tmp_path: Path) -> None:
    from PIL import Image

    bold, regular = _dejavu_or_skip()
    from lofivid.compose.brand import render_brand_layer

    layers = _make_layers(bold, regular)
    out = render_brand_layer(layers, 1920, 1080, tmp_path)
    assert out is not None

    with Image.open(out) as img:
        assert img.mode == "RGBA"
        assert img.width == 1920
        assert img.height == 1080


def test_render_brand_layer_cache_hit(tmp_path: Path) -> None:
    bold, regular = _dejavu_or_skip()
    from lofivid.compose.brand import render_brand_layer

    layers = _make_layers(bold, regular)
    out1 = render_brand_layer(layers, 1920, 1080, tmp_path)
    mtime1 = out1.stat().st_mtime

    out2 = render_brand_layer(layers, 1920, 1080, tmp_path)
    assert out1 == out2
    assert out2.stat().st_mtime == mtime1  # not re-rendered


def test_render_brand_layer_empty_list_returns_none(tmp_path: Path) -> None:
    from lofivid.compose.brand import render_brand_layer

    result = render_brand_layer([], 1920, 1080, tmp_path)
    assert result is None


def test_render_brand_layer_all_disabled_returns_none(tmp_path: Path) -> None:
    bold, regular = _dejavu_or_skip()
    from lofivid.compose.brand import render_brand_layer
    from lofivid.styles.schema import TextLayerSpec

    layers = [
        TextLayerSpec(
            enabled=False,
            text="Hidden",
            font_path=bold,
            size_pct=0.022,
            color="#FFFFFF",
            position="top_centre",
        ),
        TextLayerSpec(
            enabled=False,
            text="Also hidden",
            font_path=regular,
            size_pct=0.022,
            color="#FFFFFF",
            position="bottom_centre",
        ),
    ]
    result = render_brand_layer(layers, 1920, 1080, tmp_path)
    assert result is None


def test_render_brand_layer_different_frame_size_different_file(tmp_path: Path) -> None:
    bold, regular = _dejavu_or_skip()
    from lofivid.compose.brand import render_brand_layer

    layers = _make_layers(bold, regular)
    out_1080 = render_brand_layer(layers, 1920, 1080, tmp_path)
    out_720 = render_brand_layer(layers, 1280, 720, tmp_path)

    assert out_1080 != out_720
    assert out_1080.exists()
    assert out_720.exists()


def test_render_brand_layer_partial_disabled(tmp_path: Path) -> None:
    """One disabled layer in a list of 3 should still produce a PNG."""
    bold, regular = _dejavu_or_skip()
    from lofivid.compose.brand import render_brand_layer
    from lofivid.styles.schema import TextLayerSpec

    layers = _make_layers(bold, regular)
    # Disable the second layer
    layers[1] = TextLayerSpec(
        enabled=False,
        text=layers[1].text,
        font_path=layers[1].font_path,
        size_pct=layers[1].size_pct,
        color=layers[1].color,
        position=layers[1].position,
    )
    out = render_brand_layer(layers, 1920, 1080, tmp_path)
    assert out is not None
    assert out.exists()
