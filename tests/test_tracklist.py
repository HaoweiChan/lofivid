"""Tests for tracklist design — variation rotation, BPM/key sampling, prompt synthesis."""

from __future__ import annotations

from lofivid.config import MusicAnchor, MusicConfig, MusicVariation
from lofivid.music.tracklist import design_tracklist, plans_to_specs
from lofivid.seeds import SeedRegistry


def _example_music_config(track_count: int = 5) -> MusicConfig:
    return MusicConfig(
        backend="acestep",
        track_count=track_count,
        track_seconds_range=(60, 90),
        crossfade_seconds=4.0,
        target_lufs=-14.0,
        anchor=MusicAnchor(
            bpm_range=(70, 80),
            key_pool=["A minor", "F major"],
            style_tags=["lo-fi", "chillhop"],
        ),
        variations=[
            MusicVariation(mood="rainy night", instruments=["piano", "pad"]),
            MusicVariation(mood="sunset cafe", instruments=["Rhodes", "bass"]),
        ],
    )


def test_design_tracklist_produces_correct_count():
    cfg = _example_music_config(track_count=7)
    plans = design_tracklist(cfg, SeedRegistry(42))
    assert len(plans) == 7


def test_design_tracklist_rotates_variations():
    cfg = _example_music_config(track_count=4)
    plans = design_tracklist(cfg, SeedRegistry(42))
    moods = [p.mood for p in plans]
    # With 2 variations and 4 tracks, each variation appears twice (round-robin).
    assert sorted(moods) == sorted(["rainy night", "sunset cafe"] * 2)


def test_design_tracklist_respects_bpm_bounds():
    cfg = _example_music_config(track_count=20)
    plans = design_tracklist(cfg, SeedRegistry(42))
    for p in plans:
        assert 70 <= p.bpm <= 80


def test_design_tracklist_respects_duration_bounds():
    cfg = _example_music_config(track_count=10)
    plans = design_tracklist(cfg, SeedRegistry(42))
    for p in plans:
        assert 60 <= p.duration_seconds <= 90


def test_design_tracklist_is_deterministic_for_same_seed():
    cfg = _example_music_config(track_count=10)
    a = design_tracklist(cfg, SeedRegistry(123))
    b = design_tracklist(cfg, SeedRegistry(123))
    assert [(p.bpm, p.key, p.mood, p.duration_seconds) for p in a] == \
           [(p.bpm, p.key, p.mood, p.duration_seconds) for p in b]


def test_design_tracklist_changes_with_seed():
    cfg = _example_music_config(track_count=10)
    a = design_tracklist(cfg, SeedRegistry(1))
    b = design_tracklist(cfg, SeedRegistry(2))
    # Highly unlikely the BPM sequences match exactly across seeds
    assert [p.bpm for p in a] != [p.bpm for p in b]


def test_to_prompt_includes_required_tokens():
    cfg = _example_music_config(track_count=1)
    plan = design_tracklist(cfg, SeedRegistry(42))[0]
    prompt = plan.to_prompt()
    assert plan.mood in prompt
    assert f"{plan.bpm} BPM" in prompt
    assert plan.key in prompt
    for tag in cfg.anchor.style_tags:
        assert tag in prompt


def test_plans_to_specs_assigns_unique_seeds():
    cfg = _example_music_config(track_count=5)
    seeds = SeedRegistry(42)
    plans = design_tracklist(cfg, seeds)
    specs = plans_to_specs(plans, seeds)
    assert len({s.seed for s in specs}) == len(specs)
