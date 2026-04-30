"""Pivot v3 invariants: style/run split + cross-cutting integration tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from lofivid.config import Config, load
from lofivid.music.base import TrackSpec
from lofivid.styles.loader import load_style, style_hash


def _write_style(root: Path, name: str = "test_style", *, prompt_template: str = "cafe interior") -> None:
    styles_dir = root / "styles"
    styles_dir.mkdir(parents=True, exist_ok=True)
    (styles_dir / f"{name}.yaml").write_text(dedent(f"""
        name: {name}
        keyframe_prompt_template: {prompt_template}
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


def _write_run(path: Path, *, style_ref: str = "test_style") -> None:
    path.write_text(dedent(f"""
        run_id: test
        style_ref: {style_ref}
        duration_minutes: 30
        output_resolution: [640, 360]
        fps: 24
        seed: 42
        music:
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


def test_resolved_style_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_style(tmp_path)
    run = tmp_path / "run.yaml"
    _write_run(run)
    monkeypatch.setenv("LOFIVID_REPO_ROOT", str(tmp_path))
    cfg = load(run)
    a = cfg.resolved_style
    b = cfg.resolved_style
    # Same object both times — resolution is cached.
    assert a is b


def test_style_hash_stable_across_loads(tmp_path: Path):
    _write_style(tmp_path)
    s1, h1 = load_style("test_style", tmp_path)
    s2, h2 = load_style("test_style", tmp_path)
    assert h1 == h2
    assert style_hash(s1) == style_hash(s2)


def test_style_hash_changes_with_prompt_template(tmp_path: Path):
    _write_style(tmp_path, prompt_template="cafe interior")
    _, h1 = load_style("test_style", tmp_path)
    _write_style(tmp_path, prompt_template="rainy window")
    _, h2 = load_style("test_style", tmp_path)
    assert h1 != h2


def test_track_spec_cache_key_includes_mood():
    a = TrackSpec(track_index=0, prompt="p", bpm=80, key="A minor",
                  duration_seconds=60, seed=1, mood=None)
    b = TrackSpec(track_index=0, prompt="p", bpm=80, key="A minor",
                  duration_seconds=60, seed=1, mood="cafe afternoon")
    assert a.cache_key() != b.cache_key()


def test_track_spec_cache_key_includes_lyrics():
    a = TrackSpec(track_index=0, prompt="p", bpm=80, key="A minor",
                  duration_seconds=60, seed=1, lyrics=None)
    b = TrackSpec(track_index=0, prompt="p", bpm=80, key="A minor",
                  duration_seconds=60, seed=1, lyrics="hello")
    assert a.cache_key() != b.cache_key()


def test_keyframe_backend_default_extras_is_empty():
    """Backends not overriding cache_key_extras() return {} so the existing
    cache layout is unchanged."""
    from lofivid.visuals.base import KeyframeBackend, KeyframeSpec

    class Dummy(KeyframeBackend):
        name = "dummy"

        def generate(self, spec, output_dir):
            raise NotImplementedError

    spec = KeyframeSpec(scene_index=0, prompt="x", width=64, height=64, seed=0)
    assert Dummy().cache_key_extras(spec) == {}


def test_full_config_resolves_both_layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_style(tmp_path)
    run = tmp_path / "run.yaml"
    _write_run(run)
    monkeypatch.setenv("LOFIVID_REPO_ROOT", str(tmp_path))
    cfg: Config = load(run)
    # Identity comes from the style:
    assert cfg.resolved_style.music_backend == "library"
    assert cfg.resolved_style.keyframe_backend == "unsplash"
    # Per-run params come from the run config:
    assert cfg.music.track_count == 6
    assert cfg.visuals.scene_count == 6
