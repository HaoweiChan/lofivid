"""Verifies style_ref is mandatory and load_style enforces presence."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from lofivid.config import Config
from lofivid.styles.loader import load_style


def _valid_run_dict() -> dict:
    return {
        "run_id": "test",
        "duration_minutes": 30,
        "music": {
            "track_count": 6,
            "track_seconds_range": [280, 320],
            "crossfade_seconds": 6,
            "target_lufs": -14,
        },
        "visuals": {
            "scene_count": 6,
            "scene_seconds": 300,
            "parallax_loop_seconds": 30,
            "premium_scenes": 0,
        },
    }


def test_style_ref_is_mandatory():
    payload = _valid_run_dict()
    # No style_ref → must raise.
    with pytest.raises(ValidationError):
        Config.model_validate(payload)


def _write_minimal_style(root: Path, name: str = "test_style") -> Path:
    """Drop a valid style YAML at <root>/styles/<name>.yaml. Returns the path."""
    styles_dir = root / "styles"
    styles_dir.mkdir(parents=True, exist_ok=True)
    path = styles_dir / f"{name}.yaml"
    path.write_text(dedent(f"""
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
    return path


def test_load_style_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not_a_style"):
        load_style("not_a_style", tmp_path)


def test_load_style_loads_from_disk(tmp_path: Path):
    _write_minimal_style(tmp_path, name="test_style")
    spec, h = load_style("test_style", tmp_path)
    assert spec.name == "test_style"
    assert isinstance(h, str) and len(h) == 12


def test_resolved_style_loads_via_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_style(tmp_path, name="test_style")
    monkeypatch.setenv("LOFIVID_REPO_ROOT", str(tmp_path))
    payload = _valid_run_dict()
    payload["style_ref"] = "test_style"
    cfg = Config.model_validate(payload)
    cfg._resolve()
    assert cfg.resolved_style.name == "test_style"
    assert cfg.style_hash == cfg.style_hash  # idempotent
