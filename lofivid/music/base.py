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
    # Optional source/license/attribution dict — populated by LibraryMusicBackend
    # from the per-track sidecar JSON written by the ingest layer (see
    # lofivid/ingest/base.py). Keys: source, source_id, license,
    # attribution_text, original_url, fetched_at. Surfaced through the
    # manifest's `music_attributions` list. None for backends that don't
    # provide a sidecar (e.g. ACE-Step / Suno) or for legacy library files
    # without a sidecar.
    attribution: dict | None = None


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

    def cache_key_extras(self, spec: TrackSpec) -> dict:
        """Extra contributions to the per-track cache key beyond `spec.cache_key()`.

        Override when the backend resolves external state (e.g. a library
        backend's selected file path + content hash) that should invalidate
        the cache when it changes. Default returns {} so generative backends
        like ACE-Step / Suno stay unchanged.
        """
        return {}

    def warmup(self) -> None:
        """Optional: pre-load weights so the first generate() call isn't slow."""
        return None

    def shutdown(self) -> None:
        """Optional: release VRAM (called between music and visuals passes)."""
        return None
