"""Tests for tracklist design — variation rotation, BPM/key sampling, prompt synthesis."""

from __future__ import annotations

from lofivid.config import MusicInstance
from lofivid.music.tracklist import design_tracklist, plans_to_specs
from lofivid.seeds import SeedRegistry
from lofivid.styles.schema import MusicAnchor, MusicVariation


def _example_inputs(track_count: int = 5) -> tuple[MusicAnchor, list[MusicVariation], MusicInstance]:
    anchor = MusicAnchor(
        bpm_range=(70, 80),
        key_pool=["A minor", "F major"],
        style_tags=["lo-fi", "chillhop"],
    )
    variations = [
        MusicVariation(mood="rainy night", instruments=["piano", "pad"]),
        MusicVariation(mood="sunset cafe", instruments=["Rhodes", "bass"]),
    ]
    instance = MusicInstance(
        track_count=track_count,
        track_seconds_range=(60, 90),
        crossfade_seconds=4.0,
        target_lufs=-14.0,
    )
    return anchor, variations, instance


def test_design_tracklist_produces_correct_count():
    anchor, variations, instance = _example_inputs(track_count=7)
    plans = design_tracklist(anchor, variations, instance, SeedRegistry(42))
    assert len(plans) == 7


def test_design_tracklist_rotates_variations():
    anchor, variations, instance = _example_inputs(track_count=4)
    plans = design_tracklist(anchor, variations, instance, SeedRegistry(42))
    moods = [p.mood for p in plans]
    assert sorted(moods) == sorted(["rainy night", "sunset cafe"] * 2)


def test_design_tracklist_respects_bpm_bounds():
    anchor, variations, instance = _example_inputs(track_count=20)
    plans = design_tracklist(anchor, variations, instance, SeedRegistry(42))
    for p in plans:
        assert 70 <= p.bpm <= 80


def test_design_tracklist_respects_duration_bounds():
    anchor, variations, instance = _example_inputs(track_count=10)
    plans = design_tracklist(anchor, variations, instance, SeedRegistry(42))
    for p in plans:
        assert 60 <= p.duration_seconds <= 90


def test_design_tracklist_is_deterministic_for_same_seed():
    anchor, variations, instance = _example_inputs(track_count=10)
    a = design_tracklist(anchor, variations, instance, SeedRegistry(123))
    b = design_tracklist(anchor, variations, instance, SeedRegistry(123))
    assert [(p.bpm, p.key, p.mood, p.duration_seconds) for p in a] == \
           [(p.bpm, p.key, p.mood, p.duration_seconds) for p in b]


def test_design_tracklist_changes_with_seed():
    anchor, variations, instance = _example_inputs(track_count=10)
    a = design_tracklist(anchor, variations, instance, SeedRegistry(1))
    b = design_tracklist(anchor, variations, instance, SeedRegistry(2))
    assert [p.bpm for p in a] != [p.bpm for p in b]


def test_to_prompt_includes_required_tokens():
    anchor, variations, instance = _example_inputs(track_count=1)
    plan = design_tracklist(anchor, variations, instance, SeedRegistry(42))[0]
    prompt = plan.to_prompt()
    assert plan.mood in prompt
    assert f"{plan.bpm} BPM" in prompt
    assert plan.key in prompt
    for tag in anchor.style_tags:
        assert tag in prompt


def test_plans_to_specs_assigns_unique_seeds():
    anchor, variations, instance = _example_inputs(track_count=5)
    seeds = SeedRegistry(42)
    plans = design_tracklist(anchor, variations, instance, seeds)
    specs = plans_to_specs(plans, seeds)
    assert len({s.seed for s in specs}) == len(specs)


def test_plans_to_specs_propagates_mood():
    anchor, variations, instance = _example_inputs(track_count=4)
    seeds = SeedRegistry(42)
    plans = design_tracklist(anchor, variations, instance, seeds)
    specs = plans_to_specs(plans, seeds)
    # Round-robin: track 0 = rainy night, track 1 = sunset cafe, ...
    assert specs[0].mood == "rainy night"
    assert specs[1].mood == "sunset cafe"
    assert specs[2].mood == "rainy night"
