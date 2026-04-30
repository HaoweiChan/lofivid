"""Tests for assert_fonts_present — hard error on missing fonts."""

from __future__ import annotations

from pathlib import Path

import pytest

from lofivid.env import assert_fonts_present
from lofivid.styles.schema import (
    HUDSpec,
    MusicAnchor,
    MusicVariation,
    StyleSpec,
    TextLayerSpec,
    WaveformSpec,
)


def _style_with_fonts(brand_font: Path, hud_font: Path, brand_cjk: Path | None = None,
                      hud_cjk: Path | None = None) -> StyleSpec:
    return StyleSpec(
        name="t",
        keyframe_prompt_template="x",
        music_anchor=MusicAnchor(bpm_range=(75, 85), key_pool=["F major"]),
        music_variations=[MusicVariation(mood="m", instruments=[])],
        brand_layers=[
            TextLayerSpec(text="hi", font_path=brand_font, cjk_font_path=brand_cjk),
        ],
        hud=HUDSpec(font_path=hud_font, cjk_font_path=hud_cjk),
        waveform=WaveformSpec(color_source="fixed"),
    )


def test_present_fonts_pass(tmp_path: Path):
    f1 = tmp_path / "primary.ttf"
    f2 = tmp_path / "hud.ttf"
    f1.write_bytes(b"fake font")
    f2.write_bytes(b"fake font")
    style = _style_with_fonts(f1, f2)
    # No exception
    assert_fonts_present(style, tmp_path)


def test_missing_brand_font_raises(tmp_path: Path):
    f2 = tmp_path / "hud.ttf"
    f2.write_bytes(b"fake font")
    missing = tmp_path / "missing_brand.ttf"
    style = _style_with_fonts(missing, f2)
    with pytest.raises(RuntimeError, match="missing font"):
        assert_fonts_present(style, tmp_path)


def test_missing_hud_font_raises(tmp_path: Path):
    f1 = tmp_path / "primary.ttf"
    f1.write_bytes(b"fake font")
    missing = tmp_path / "missing_hud.ttf"
    style = _style_with_fonts(f1, missing)
    with pytest.raises(RuntimeError, match="missing font"):
        assert_fonts_present(style, tmp_path)


def test_missing_cjk_font_raises(tmp_path: Path):
    f1 = tmp_path / "primary.ttf"
    f2 = tmp_path / "hud.ttf"
    missing_cjk = tmp_path / "missing_cjk.otf"
    f1.write_bytes(b"x")
    f2.write_bytes(b"x")
    style = _style_with_fonts(f1, f2, brand_cjk=missing_cjk)
    with pytest.raises(RuntimeError, match="missing font"):
        assert_fonts_present(style, tmp_path)


def test_disabled_hud_skips_hud_font_check(tmp_path: Path):
    f1 = tmp_path / "primary.ttf"
    f1.write_bytes(b"x")
    missing = tmp_path / "missing_hud.ttf"
    style = StyleSpec(
        name="t",
        keyframe_prompt_template="x",
        music_anchor=MusicAnchor(bpm_range=(75, 85), key_pool=["F major"]),
        music_variations=[MusicVariation(mood="m", instruments=[])],
        brand_layers=[TextLayerSpec(text="hi", font_path=f1)],
        hud=HUDSpec(font_path=missing, enabled=False),
        waveform=WaveformSpec(color_source="fixed"),
    )
    # Should not raise — disabled HUD font path is not preflighted.
    assert_fonts_present(style, tmp_path)


def test_error_message_lists_each_missing_path(tmp_path: Path):
    missing1 = tmp_path / "a.ttf"
    missing2 = tmp_path / "b.ttf"
    style = _style_with_fonts(missing1, missing2)
    with pytest.raises(RuntimeError) as exc:
        assert_fonts_present(style, tmp_path)
    msg = str(exc.value)
    assert "a.ttf" in msg
    assert "b.ttf" in msg
