"""Music backend ABC.

The pipeline depends on this interface, not on ACE-Step directly. This is
what lets us swap in MusicGen for non-commercial runs later, or upgrade to
ACE-Step 2 when it ships, without touching the rest of the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackSpec:
    """Inputs to one track generation call.

    `prompt` is the full prompt string; `bpm`, `key`, etc. are duplicated
    in dataclass form so the cache key is structured (not just a free-text hash).

    `lyrics` is optional. When None (the default) we ask the backend for a
    fully-instrumental render — this is the only mode ACE-Step supports
    locally. Backends that accept lyrics (Suno) treat None as "instrumental".
    """
    track_index: int
    prompt: str
    bpm: int
    key: str
    duration_seconds: int
    seed: int
    lyrics: str | None = None
    mood: str | None = None

    def cache_key(self) -> dict:
        return {
            "prompt": self.prompt,
            "bpm": self.bpm,
            "key": self.key,
            "duration_seconds": self.duration_seconds,
            "seed": self.seed,
            "lyrics": self.lyrics,
            "mood": self.mood,
        }


@dataclass(frozen=True)
class GeneratedTrack:
    spec: TrackSpec
    path: Path
    sample_rate: int
    actual_duration_seconds: float
    title: str = ""
    artist: str | None = None


class MusicBackend(ABC):
    """Synchronous, single-track-at-a-time interface."""

    name: str

    @abstractmethod
    def generate(self, spec: TrackSpec, output_dir: Path) -> GeneratedTrack:
        """Render one track to disk and return metadata.

        Implementations should write to `output_dir/<spec.track_index>.wav`
        and load/unload model weights internally so the caller does not need
        to manage VRAM lifecycle.
        """

    def warmup(self) -> None:
        """Optional: pre-load weights so the first generate() call isn't slow."""
        return None

    def shutdown(self) -> None:
        """Optional: release VRAM (called between music and visuals passes)."""
        return None
