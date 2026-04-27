"""Timeline scheduler — decides which clip plays from t=X to t=Y.

Visual scenes don't have to align 1:1 with music tracks. The timeline
takes a list of GeneratedClip objects + a target total duration and emits
a list of (clip_path, start_seconds, end_seconds) entries that, when
concatenated with crossfades, fill the target exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lofivid.visuals.base import GeneratedClip


@dataclass(frozen=True)
class ScheduledScene:
    clip_path: Path
    start_seconds: float
    end_seconds: float
    crossfade_in: float

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


def schedule(
    clips: list[GeneratedClip],
    total_seconds: float,
    crossfade_seconds: float = 2.0,
) -> list[ScheduledScene]:
    """Distribute clips across `total_seconds` with `crossfade_seconds` overlap.

    Each scene runs for `total_seconds / len(clips)` ± crossfade adjustment.
    The first scene has no crossfade-in; subsequent scenes overlap the prior
    by `crossfade_seconds`. The last scene is padded/trimmed to land exactly
    on `total_seconds`.
    """
    if not clips:
        raise ValueError("schedule requires at least one clip")
    n = len(clips)
    if n == 1:
        return [ScheduledScene(clips[0].path, 0.0, total_seconds, 0.0)]

    # With (n-1) crossfades subtracting time, raw per-scene length is:
    per_scene = (total_seconds + crossfade_seconds * (n - 1)) / n
    scheduled: list[ScheduledScene] = []
    cursor = 0.0
    for i, clip in enumerate(clips):
        xfade_in = 0.0 if i == 0 else crossfade_seconds
        start = cursor - xfade_in
        if i == n - 1:
            end = total_seconds   # last clip lands on the target
        else:
            end = start + per_scene
        scheduled.append(ScheduledScene(clip.path, start, end, xfade_in))
        cursor = end
    return scheduled
