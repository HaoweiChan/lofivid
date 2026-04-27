"""Tests for the YAML config schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from lofivid.config import load


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ["smoke_30sec", "anime_rainy_window", "photo_cozy_cafe"])
def test_shipped_configs_are_valid(name: str):
    cfg = load(REPO_ROOT / "configs" / f"{name}.yaml")
    assert cfg.run_id
    assert cfg.duration_minutes > 0
    assert len(cfg.music.variations) >= 1
    assert cfg.visuals.scene_count >= 1


def test_invalid_yaml_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run_id: oops\nduration_minutes: 60\n")  # missing required fields
    with pytest.raises(Exception):
        load(bad)


def test_visual_scene_seconds_must_match_total(tmp_path: Path):
    bad = tmp_path / "mismatched.yaml"
    bad.write_text(
        """
run_id: bad
duration_minutes: 60     # = 3600 seconds
output_resolution: [640, 360]
fps: 24
seed: 1
music:
  backend: acestep
  track_count: 1
  track_seconds_range: [30, 30]
  crossfade_seconds: 0
  target_lufs: -16
  anchor:
    bpm_range: [75, 75]
    key_pool: [A minor]
    style_tags: [lo-fi]
  variations:
    - { mood: x, instruments: [piano] }
visuals:
  preset: anime
  scene_count: 1
  scene_seconds: 30      # 30 != 3600
  parallax_loop_seconds: 15
  premium_scenes: 0
  keyframe_prompt_template: x
  loras: []
overlays:
  rain_video: null
  vinyl_crackle: null
"""
    )
    with pytest.raises(Exception):
        load(bad)
