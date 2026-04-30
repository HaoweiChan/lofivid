"""Tests for the run YAML config schema (post-pivot v3)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from lofivid.config import Config, MusicInstance, VisualsInstance, load


def _write_style(root: Path, name: str = "test_style") -> None:
    styles_dir = root / "styles"
    styles_dir.mkdir(parents=True, exist_ok=True)
    (styles_dir / f"{name}.yaml").write_text(dedent(f"""
        name: {name}
        keyframe_prompt_template: cafe interior
        music_anchor:
          bpm_range: [75, 85]
          key_pool: [F major]
          style_tags: [lo-fi]
        music_variations:
          - mood: cafe afternoon
            instruments: [Rhodes]
        hud:
          font_path: assets/fonts/IBMPlexSans-Bold.ttf
        waveform:
          color_source: fixed
          fixed_color: '#FFFFFF'
    """).lstrip())


def _write_run(path: Path, *, style_ref: str = "test_style", duration_minutes: float = 30,
               scene_count: int = 6, scene_seconds: int = 300) -> None:
    path.write_text(dedent(f"""
        run_id: test
        style_ref: {style_ref}
        duration_minutes: {duration_minutes}
        output_resolution: [640, 360]
        fps: 24
        seed: 42
        music:
          track_count: 6
          track_seconds_range: [280, 320]
          crossfade_seconds: 6
          target_lufs: -14
        visuals:
          scene_count: {scene_count}
          scene_seconds: {scene_seconds}
          parallax_loop_seconds: 30
          premium_scenes: 0
    """).lstrip())


def test_load_valid_run_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_style(tmp_path)
    run = tmp_path / "run.yaml"
    _write_run(run)
    monkeypatch.setenv("LOFIVID_REPO_ROOT", str(tmp_path))
    cfg = load(run)
    assert cfg.run_id == "test"
    assert cfg.style_ref == "test_style"
    assert cfg.resolved_style.name == "test_style"


def test_load_raises_when_missing_required_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from pydantic import ValidationError
    _write_style(tmp_path)
    run = tmp_path / "bad.yaml"
    run.write_text("run_id: oops\nduration_minutes: 60\n")  # missing required fields
    monkeypatch.setenv("LOFIVID_REPO_ROOT", str(tmp_path))
    with pytest.raises(ValidationError):
        load(run)


def test_duration_vs_visuals_validator_fires(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_style(tmp_path)
    run = tmp_path / "mismatched.yaml"
    # 60 min = 3600s; visuals = 1 * 30s = 30s → outside 5% slack → must raise.
    _write_run(run, duration_minutes=60, scene_count=1, scene_seconds=30)
    monkeypatch.setenv("LOFIVID_REPO_ROOT", str(tmp_path))
    with pytest.raises(Exception, match="does not match"):
        load(run)


def test_load_raises_when_style_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Don't write the style file
    run = tmp_path / "run.yaml"
    _write_run(run, style_ref="ghost")
    monkeypatch.setenv("LOFIVID_REPO_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="ghost"):
        load(run)


def test_music_instance_extra_field_rejected_in_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_style(tmp_path)
    run = tmp_path / "bad.yaml"
    run.write_text(dedent("""
        run_id: test
        style_ref: test_style
        duration_minutes: 30
        music:
          backend: acestep   # forbidden — belongs in style
          track_count: 6
          track_seconds_range: [280, 320]
          crossfade_seconds: 6
          target_lufs: -14
        visuals:
          scene_count: 6
          scene_seconds: 300
          parallax_loop_seconds: 30
          premium_scenes: 0
    """).lstrip())
    monkeypatch.setenv("LOFIVID_REPO_ROOT", str(tmp_path))
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        load(run)


def test_minimal_programmatic_construction():
    music = MusicInstance(
        track_count=6,
        track_seconds_range=(280, 320),
        crossfade_seconds=6,
        target_lufs=-14,
    )
    visuals = VisualsInstance(
        scene_count=6,
        scene_seconds=300,
        parallax_loop_seconds=30,
        premium_scenes=0,
    )
    cfg = Config(
        run_id="test",
        style_ref="ghost",  # not resolved unless we touch resolved_style
        duration_minutes=30,
        music=music,
        visuals=visuals,
    )
    assert cfg.run_id == "test"
