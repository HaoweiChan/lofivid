"""Tests for the channel-direction pivot — config schema, backends, plumbing.

We don't exercise the cloud (Suno/Unsplash) or GPU (SDXL/DepthFlow) paths
here; those are smoke-tested live. These tests cover only the local,
deterministic surface that ships in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lofivid.config import (
    Config,
    MusicAnchor,
    MusicConfig,
    MusicVariation,
    VisualsConfig,
    load,
)
from lofivid.music.tracklist import design_tracklist, plans_to_specs
from lofivid.seeds import SeedRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------- new shipped configs validate -----------------------------

@pytest.mark.parametrize(
    "name",
    ["smoke_30sec", "anime_rainy_window", "photo_cozy_cafe",
     "jazz_cafe_unsplash", "minimal_design_lofi"],
)
def test_shipped_configs_validate(name: str):
    cfg = load(REPO_ROOT / "configs" / f"{name}.yaml")
    target = cfg.duration_minutes * 60
    actual = cfg.visuals.scene_count * cfg.visuals.scene_seconds
    # Same 5% slack as the model validator
    assert abs(actual - target) <= 0.05 * target


def test_jazz_cafe_uses_suno_and_unsplash():
    cfg = load(REPO_ROOT / "configs" / "jazz_cafe_unsplash.yaml")
    assert cfg.music.backend == "suno"
    assert cfg.visuals.keyframe_backend == "unsplash"
    assert cfg.visuals.parallax_backend == "overlay_motion"
    assert cfg.visuals.motion_type == "slow_zoom"
    assert cfg.visuals.duotone is not None
    # Every variation in this config has lyrics.
    assert all(v.lyrics for v in cfg.music.variations)


def test_minimal_design_uses_local_stack():
    cfg = load(REPO_ROOT / "configs" / "minimal_design_lofi.yaml")
    assert cfg.music.backend == "acestep"
    assert cfg.visuals.keyframe_backend == "sdxl"
    assert cfg.visuals.parallax_backend == "overlay_motion"
    assert cfg.visuals.motion_type == "dust_motes"


# ---------- schema defaults preserve existing behaviour --------------

def _minimal_visuals(**overrides) -> VisualsConfig:
    base = dict(
        preset="anime",
        scene_count=1,
        scene_seconds=30,
        parallax_loop_seconds=15,
        keyframe_prompt_template="x",
    )
    base.update(overrides)
    return VisualsConfig(**base)


def test_visuals_defaults_preserve_sdxl_depthflow():
    v = _minimal_visuals()
    assert v.keyframe_backend == "sdxl"
    assert v.parallax_backend == "depthflow"
    assert v.motion_type == "slow_zoom"
    assert v.duotone is None


def test_visuals_rejects_unknown_keyframe_backend():
    with pytest.raises(ValidationError):
        _minimal_visuals(keyframe_backend="dalle")


def test_visuals_accepts_2025_plus_keyframe_backends():
    """flux_klein and z_image_turbo are the commercial-OK upgrades from
    SDXL. See MODEL_OPTIONS.md for the shortlist rationale."""
    assert _minimal_visuals(keyframe_backend="flux_klein").keyframe_backend == "flux_klein"
    assert _minimal_visuals(keyframe_backend="z_image_turbo").keyframe_backend == "z_image_turbo"


def test_visuals_rejects_unknown_parallax_backend():
    with pytest.raises(ValidationError):
        _minimal_visuals(parallax_backend="runway")


def test_visuals_rejects_unknown_motion_type():
    with pytest.raises(ValidationError):
        _minimal_visuals(parallax_backend="overlay_motion", motion_type="kaboom")


def test_visuals_accepts_duotone_pair():
    v = _minimal_visuals(duotone=[[40, 22, 8], [244, 222, 184]])
    assert v.duotone == ((40, 22, 8), (244, 222, 184))


# ---------- music schema additions -----------------------------------

def _minimal_music(backend: str = "acestep") -> MusicConfig:
    return MusicConfig(
        backend=backend,
        track_count=1,
        track_seconds_range=(30, 30),
        crossfade_seconds=0,
        target_lufs=-16,
        anchor=MusicAnchor(bpm_range=(75, 75), key_pool=["A minor"], style_tags=["lo-fi"]),
        variations=[MusicVariation(mood="x", instruments=["piano"])],
    )


def test_music_backend_literal_includes_suno():
    cfg = _minimal_music("suno")
    assert cfg.backend == "suno"


def test_music_backend_rejects_unknown():
    with pytest.raises(ValidationError):
        _minimal_music("riffusion")


def test_music_variation_lyrics_default_none():
    v = MusicVariation(mood="x", instruments=["piano"])
    assert v.lyrics is None


def test_music_variation_accepts_lyrics():
    v = MusicVariation(mood="x", instruments=["piano"], lyrics="hello world")
    assert v.lyrics == "hello world"


def test_suno_model_version_default():
    cfg = _minimal_music("suno")
    assert cfg.suno_model_version == "v3.5"


# ---------- tracklist propagates lyrics ------------------------------

def test_tracklist_propagates_lyrics_into_specs():
    cfg = MusicConfig(
        backend="suno",
        track_count=4,
        track_seconds_range=(60, 90),
        crossfade_seconds=4,
        target_lufs=-14,
        anchor=MusicAnchor(bpm_range=(70, 80), key_pool=["A minor"], style_tags=["lo-fi"]),
        variations=[
            MusicVariation(mood="rainy", instruments=["piano"], lyrics="line one"),
            MusicVariation(mood="sunny", instruments=["Rhodes"]),  # lyrics=None
        ],
    )
    seeds = SeedRegistry(42)
    plans = design_tracklist(cfg, seeds)
    specs = plans_to_specs(plans, seeds)
    assert specs[0].lyrics == "line one"
    assert specs[1].lyrics is None
    assert specs[2].lyrics == "line one"   # round-robin → first variation
    assert specs[3].lyrics is None


def test_track_spec_cache_key_includes_lyrics():
    from lofivid.music.base import TrackSpec
    a = TrackSpec(track_index=0, prompt="p", bpm=80, key="A minor",
                  duration_seconds=60, seed=1, lyrics=None)
    b = TrackSpec(track_index=0, prompt="p", bpm=80, key="A minor",
                  duration_seconds=60, seed=1, lyrics="hello")
    assert a.cache_key() != b.cache_key()


# ---------- KeyframeBackend default extras --------------------------

def test_keyframe_backend_default_extras_is_empty():
    """SDXL (and any backend not overriding) returns {} so the existing
    cache layout is unchanged."""
    from lofivid.visuals.base import KeyframeBackend, KeyframeSpec

    class Dummy(KeyframeBackend):
        name = "dummy"

        def generate(self, spec, output_dir):
            raise NotImplementedError

    spec = KeyframeSpec(scene_index=0, prompt="x", width=64, height=64, seed=0)
    assert Dummy().cache_key_extras(spec) == {}


# ---------- end-to-end Config validation -----------------------------

def test_full_config_with_pivot_fields_validates():
    cfg = Config(
        run_id="test",
        duration_minutes=30,
        output_resolution=(1920, 1080),
        fps=24,
        seed=42,
        music=MusicConfig(
            backend="suno",
            suno_model_version="v3.5",
            track_count=6,
            track_seconds_range=(280, 320),
            crossfade_seconds=6,
            target_lufs=-14,
            anchor=MusicAnchor(bpm_range=(70, 90), key_pool=["A minor"], style_tags=["lo-fi"]),
            variations=[MusicVariation(mood="x", instruments=["piano"], lyrics="la la")],
        ),
        visuals=VisualsConfig(
            preset="photo",
            scene_count=6,
            scene_seconds=300,
            parallax_loop_seconds=30,
            keyframe_prompt_template="cafe",
            keyframe_backend="unsplash",
            parallax_backend="overlay_motion",
            motion_type="slow_zoom",
            duotone=((40, 22, 8), (244, 222, 184)),
        ),
    )
    assert cfg.music.backend == "suno"
    assert cfg.visuals.keyframe_backend == "unsplash"
