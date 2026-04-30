"""Verifies that style-defining fields in a run YAML raise loudly.

The whole point of the style/run split is to force separation. A typo
or copy-paste from an old monolithic config (e.g. `music.backend: acestep`)
must fail-loud against `extra="forbid"`, not silently no-op.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lofivid.config import MusicInstance, VisualsInstance


def test_music_instance_rejects_backend_field():
    with pytest.raises(ValidationError):
        MusicInstance(
            backend="acestep",  # type: ignore[call-arg]
            track_count=6,
            track_seconds_range=(280, 320),
            crossfade_seconds=6,
            target_lufs=-14,
        )


def test_music_instance_rejects_anchor_field():
    with pytest.raises(ValidationError):
        MusicInstance(
            anchor={"bpm_range": (70, 90), "key_pool": ["A minor"]},  # type: ignore[call-arg]
            track_count=6,
            track_seconds_range=(280, 320),
            crossfade_seconds=6,
            target_lufs=-14,
        )


def test_music_instance_rejects_variations_field():
    with pytest.raises(ValidationError):
        MusicInstance(
            variations=[],  # type: ignore[call-arg]
            track_count=6,
            track_seconds_range=(280, 320),
            crossfade_seconds=6,
            target_lufs=-14,
        )


def test_visuals_instance_rejects_preset_field():
    with pytest.raises(ValidationError):
        VisualsInstance(
            preset="photo",  # type: ignore[call-arg]
            scene_count=6,
            scene_seconds=300,
            parallax_loop_seconds=30,
            premium_scenes=0,
        )


def test_visuals_instance_rejects_keyframe_backend_field():
    with pytest.raises(ValidationError):
        VisualsInstance(
            keyframe_backend="sdxl",  # type: ignore[call-arg]
            scene_count=6,
            scene_seconds=300,
            parallax_loop_seconds=30,
            premium_scenes=0,
        )


def test_visuals_instance_rejects_keyframe_prompt_template_field():
    with pytest.raises(ValidationError):
        VisualsInstance(
            keyframe_prompt_template="cafe",  # type: ignore[call-arg]
            scene_count=6,
            scene_seconds=300,
            parallax_loop_seconds=30,
            premium_scenes=0,
        )


def test_clean_music_instance_validates():
    inst = MusicInstance(
        track_count=6,
        track_seconds_range=(280, 320),
        crossfade_seconds=6,
        target_lufs=-14,
    )
    assert inst.track_count == 6


def test_clean_visuals_instance_validates():
    inst = VisualsInstance(
        scene_count=6,
        scene_seconds=300,
        parallax_loop_seconds=30,
        premium_scenes=0,
    )
    assert inst.scene_count == 6
