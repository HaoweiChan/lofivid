"""Designs N distinct lofi track prompts from a shared anchor + variation matrix.

Goal: produce a tracklist that *feels* like a curated lofi mix, not 20 cuts of
the same song. Each track inherits genre tags / BPM range / key pool from the
anchor (cohesion), then samples one of the user-supplied variations for
mood + instrumentation (variety).

Inputs come from two layers:
  - the active style's `music_anchor` and `music_variations` (identity)
  - the run config's `MusicInstance` (per-render counts and durations)
"""

from __future__ import annotations

from dataclasses import dataclass

from lofivid.config import MusicInstance
from lofivid.music.base import TrackSpec
from lofivid.seeds import SeedRegistry
from lofivid.styles.schema import MusicAnchor, MusicVariation


@dataclass
class TrackPlan:
    """One row in the tracklist; rendered to a `TrackSpec` with a seed."""
    index: int
    bpm: int
    key: str
    mood: str
    instruments: list[str]
    style_tags: list[str]
    duration_seconds: int
    lyrics: str | None = None    # only populated for vocal-capable backends (Suno)

    def to_prompt(self) -> str:
        """Compose the natural-language prompt fed to ACE-Step.

        Format chosen to play well with ACE-Step's tag-style conditioning:
        comma-separated tags, with explicit BPM and key tokens.
        """
        parts = list(self.style_tags)
        parts.append(self.mood)
        parts.extend(self.instruments)
        parts.extend([f"{self.bpm} BPM", f"key of {self.key}", "stereo", "vinyl crackle"])
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = [p for p in parts if not (p in seen or seen.add(p))]
        return ", ".join(unique)


def design_tracklist(
    anchor: MusicAnchor,
    variations: list[MusicVariation],
    instance: MusicInstance,
    seeds: SeedRegistry,
) -> list[TrackPlan]:
    """Sample N TrackPlans by rotating through variations + perturbing BPM/key/duration.

    Identity (anchor / variations) comes from the resolved style; the
    `instance` carries only per-run counts and durations.
    """
    rng = seeds.seed_python_rng("music.tracklist")
    plans: list[TrackPlan] = []
    bpm_lo, bpm_hi = anchor.bpm_range
    dur_lo, dur_hi = instance.track_seconds_range

    for i in range(instance.track_count):
        variation = variations[i % len(variations)]
        plans.append(TrackPlan(
            index=i,
            bpm=rng.randint(bpm_lo, bpm_hi),
            key=anchor.key_pool[(i + rng.randint(0, len(anchor.key_pool) - 1)) % len(anchor.key_pool)],
            mood=variation.mood,
            instruments=list(variation.instruments),
            style_tags=list(anchor.style_tags),
            duration_seconds=rng.randint(dur_lo, dur_hi),
            lyrics=variation.lyrics,
        ))

    return plans


def plans_to_specs(plans: list[TrackPlan], seeds: SeedRegistry) -> list[TrackSpec]:
    """Materialise TrackPlans into TrackSpecs with deterministic per-track seeds."""
    return [
        TrackSpec(
            track_index=p.index,
            prompt=p.to_prompt(),
            bpm=p.bpm,
            key=p.key,
            duration_seconds=p.duration_seconds,
            seed=seeds.derive(f"music.track.{p.index}"),
            lyrics=p.lyrics,
            mood=p.mood,
        )
        for p in plans
    ]
