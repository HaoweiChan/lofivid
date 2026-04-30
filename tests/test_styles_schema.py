"""Tests for the StyleSpec schema + style_hash determinism."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lofivid.styles.loader import style_hash
from lofivid.styles.schema import (
    HUDSpec,
    MusicAnchor,
    MusicVariation,
    StyleSpec,
    TextLayerSpec,
    WaveformSpec,
)

FONT = Path("assets/fonts/IBMPlexSans-Bold.ttf")
CJK = Path("assets/fonts/NotoSansCJKtc-Bold.otf")


def _minimal_style(**overrides) -> StyleSpec:
    base = dict(
        name="test_style",
        keyframe_prompt_template="cafe interior",
        music_anchor=MusicAnchor(
            bpm_range=(75, 85),
            key_pool=["F major"],
            style_tags=["lo-fi"],
        ),
        music_variations=[MusicVariation(mood="cafe afternoon", instruments=["Rhodes"])],
        hud=HUDSpec(font_path=FONT),
        waveform=WaveformSpec(color_source="fixed", fixed_color="#FFFFFF"),
    )
    base.update(overrides)
    return StyleSpec(**base)


def test_minimal_style_validates():
    style = _minimal_style()
    assert style.name == "test_style"
    assert style.preset == "photo"
    assert style.music_backend == "library"


def test_style_hash_is_deterministic():
    s1 = _minimal_style()
    s2 = _minimal_style()
    assert style_hash(s1) == style_hash(s2)


def test_description_does_not_change_hash():
    s1 = _minimal_style(description="version 1")
    s2 = _minimal_style(description="completely different docstring")
    assert style_hash(s1) == style_hash(s2)


def test_prompt_template_changes_hash():
    s1 = _minimal_style(keyframe_prompt_template="cafe interior")
    s2 = _minimal_style(keyframe_prompt_template="rainy window")
    assert style_hash(s1) != style_hash(s2)


def test_music_backend_change_changes_hash():
    s1 = _minimal_style(music_backend="library")
    s2 = _minimal_style(music_backend="acestep")
    assert style_hash(s1) != style_hash(s2)


def test_waveform_duotone_required_when_color_source_is_duotone():
    with pytest.raises(ValidationError):
        _minimal_style(
            duotone=None,
            waveform=WaveformSpec(color_source="duotone_highlight"),
        )


def test_waveform_duotone_ok_when_set():
    style = _minimal_style(
        duotone=((40, 22, 8), (244, 222, 184)),
        waveform=WaveformSpec(color_source="duotone_highlight"),
    )
    assert style.duotone is not None


def test_waveform_fixed_does_not_require_duotone():
    style = _minimal_style(
        duotone=None,
        waveform=WaveformSpec(color_source="fixed", fixed_color="#ABCDEF"),
    )
    assert style.duotone is None


def test_brand_layers_default_empty():
    style = _minimal_style()
    assert style.brand_layers == []


def test_brand_layer_extra_field_rejected():
    with pytest.raises(ValidationError):
        TextLayerSpec(
            text="hello",
            font_path=FONT,
            unknown_field="oops",  # type: ignore[call-arg]
        )


def test_hud_extra_field_rejected():
    with pytest.raises(ValidationError):
        HUDSpec(font_path=FONT, unknown_field="oops")  # type: ignore[call-arg]


def test_style_extra_field_rejected():
    with pytest.raises(ValidationError):
        _minimal_style(unknown_field="oops")
