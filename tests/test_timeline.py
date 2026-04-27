"""Tests for the visual scene scheduler."""

from __future__ import annotations

from pathlib import Path

import pytest

from lofivid.compose.timeline import schedule
from lofivid.visuals.base import GeneratedClip, ParallaxSpec


def _fake_clip(idx: int) -> GeneratedClip:
    spec = ParallaxSpec(
        scene_index=idx,
        image_path=Path(f"/tmp/{idx}.png"),
        duration_seconds=30,
        width=1920,
        height=1080,
        fps=24,
        seed=idx,
    )
    return GeneratedClip(spec=spec, path=Path(f"/tmp/{idx}.mp4"))


def test_schedule_single_clip_spans_full_duration():
    scenes = schedule([_fake_clip(0)], total_seconds=600.0)
    assert len(scenes) == 1
    assert scenes[0].start_seconds == 0.0
    assert scenes[0].end_seconds == 600.0


def test_schedule_multi_clip_lands_on_target():
    clips = [_fake_clip(i) for i in range(4)]
    scenes = schedule(clips, total_seconds=600.0, crossfade_seconds=2.0)
    assert len(scenes) == 4
    assert scenes[-1].end_seconds == 600.0


def test_schedule_first_scene_has_no_crossfade_in():
    clips = [_fake_clip(i) for i in range(3)]
    scenes = schedule(clips, total_seconds=300.0, crossfade_seconds=2.0)
    assert scenes[0].crossfade_in == 0.0
    for s in scenes[1:]:
        assert s.crossfade_in == 2.0


def test_schedule_empty_clips_raises():
    with pytest.raises(ValueError):
        schedule([], total_seconds=600.0)
