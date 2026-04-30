"""Tests for lofivid/compose/waveform.py — showwaves filter fragment builder."""
from __future__ import annotations

import pytest


def _make_spec(**overrides):
    from lofivid.styles.schema import WaveformSpec

    defaults = dict(
        enabled=True,
        mode="cline",
        height_px=80,
        position="bottom",
        scale="sqrt",
        color_source="fixed",
        fixed_color="#E0E0C4",
        opacity=0.6,
    )
    defaults.update(overrides)
    return WaveformSpec(**defaults)


# ---------------------------------------------------------------------------
# resolve_color
# ---------------------------------------------------------------------------

def test_resolve_color_fixed_returns_fixed_color() -> None:
    from lofivid.compose.waveform import resolve_color

    spec = _make_spec(color_source="fixed", fixed_color="#ABCDEF")
    result = resolve_color(spec, None)
    assert result == "#ABCDEF"


def test_resolve_color_duotone_highlight() -> None:
    from lofivid.compose.waveform import resolve_color

    spec = _make_spec(color_source="duotone_highlight")
    duotone = ((10, 20, 30), (200, 210, 220))
    result = resolve_color(spec, duotone)
    # highlight = second tuple → (200, 210, 220) → #C8D2DC
    assert result == "#C8D2DC"


def test_resolve_color_duotone_shadow() -> None:
    from lofivid.compose.waveform import resolve_color

    spec = _make_spec(color_source="duotone_shadow")
    duotone = ((10, 20, 30), (200, 210, 220))
    result = resolve_color(spec, duotone)
    # shadow = first tuple → (10, 20, 30) → #0A141E
    assert result == "#0A141E"


def test_resolve_color_duotone_highlight_no_duotone_raises() -> None:
    from lofivid.compose.waveform import resolve_color

    spec = _make_spec(color_source="duotone_highlight")
    with pytest.raises(ValueError, match="duotone"):
        resolve_color(spec, None)


def test_resolve_color_duotone_shadow_no_duotone_raises() -> None:
    from lofivid.compose.waveform import resolve_color

    spec = _make_spec(color_source="duotone_shadow")
    with pytest.raises(ValueError, match="duotone"):
        resolve_color(spec, None)


# ---------------------------------------------------------------------------
# build_waveform_filter
# ---------------------------------------------------------------------------

def test_build_waveform_filter_disabled_returns_none() -> None:
    from lofivid.compose.waveform import build_waveform_filter

    spec = _make_spec(enabled=False)
    result = build_waveform_filter(spec, None, 1920, 24, 1)
    assert result is None


def test_build_waveform_filter_returns_overlay() -> None:
    from lofivid.compose.waveform import WaveformOverlay, build_waveform_filter

    spec = _make_spec(
        enabled=True,
        color_source="fixed",
        fixed_color="#E0E0C4",
        mode="cline",
        height_px=80,
        scale="sqrt",
        opacity=0.6,
        position="bottom",
    )
    result = build_waveform_filter(spec, None, 1920, 24, 1)
    assert isinstance(result, WaveformOverlay)


def test_build_waveform_filter_fragment_contains_showwaves_dimensions() -> None:
    from lofivid.compose.waveform import build_waveform_filter

    spec = _make_spec(
        enabled=True,
        color_source="fixed",
        fixed_color="#E0E0C4",
        height_px=80,
    )
    result = build_waveform_filter(spec, None, 1920, 24, 1)
    assert result is not None
    assert "showwaves=s=1920x80" in result.filter_fragment


def test_build_waveform_filter_fragment_contains_audio_input() -> None:
    from lofivid.compose.waveform import build_waveform_filter

    spec = _make_spec(color_source="fixed")
    result = build_waveform_filter(spec, None, 1920, 24, audio_idx=3)
    assert result is not None
    assert "[3:a]" in result.filter_fragment


def test_build_waveform_filter_output_label() -> None:
    from lofivid.compose.waveform import build_waveform_filter

    spec = _make_spec(color_source="fixed")
    result = build_waveform_filter(spec, None, 1920, 24, 1, output_label="[mywave]")
    assert result is not None
    assert result.output_label == "[mywave]"
    assert "[mywave]" in result.filter_fragment


def test_build_waveform_filter_position_propagated() -> None:
    from lofivid.compose.waveform import build_waveform_filter

    spec = _make_spec(color_source="fixed", position="top")
    result = build_waveform_filter(spec, None, 1920, 24, 1)
    assert result is not None
    assert result.position == "top"


def test_build_waveform_filter_duotone_color() -> None:
    """Filter fragment should embed the resolved duotone color."""
    from lofivid.compose.waveform import build_waveform_filter

    spec = _make_spec(color_source="duotone_highlight", opacity=0.55)
    duotone = ((40, 22, 8), (244, 222, 184))
    result = build_waveform_filter(spec, duotone, 1920, 24, 1)
    assert result is not None
    # highlight = (244, 222, 184) → #F4DEB8
    assert "#F4DEB8" in result.filter_fragment


def test_build_waveform_filter_fragment_has_format_yuva() -> None:
    from lofivid.compose.waveform import build_waveform_filter

    spec = _make_spec(color_source="fixed")
    result = build_waveform_filter(spec, None, 1920, 24, 1)
    assert result is not None
    assert "yuva420p" in result.filter_fragment


# ---------------------------------------------------------------------------
# overlay_y_expr
# ---------------------------------------------------------------------------

def test_overlay_y_expr_top() -> None:
    from lofivid.compose.waveform import overlay_y_expr

    assert overlay_y_expr("top", 1080, 80) == "0"


def test_overlay_y_expr_bottom() -> None:
    from lofivid.compose.waveform import overlay_y_expr

    assert overlay_y_expr("bottom", 1080, 80) == "1000"


def test_overlay_y_expr_unknown_raises() -> None:
    from lofivid.compose.waveform import overlay_y_expr

    with pytest.raises(ValueError, match="unknown waveform position"):
        overlay_y_expr("middle", 1080, 80)
