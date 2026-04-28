"""DJ-mix assembly: per-track loudness normalisation + crossfaded concatenation.

Pure-FFmpeg implementation. We considered pydub but it loads everything to
RAM as int16 numpy arrays (a 2-hour stereo 44.1k mix is ~1.4 GB), and FFmpeg
can stream the same operation. The downside is a long filter_complex string;
we build it programmatically.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lofivid._ffmpeg import ffmpeg_bin
from lofivid.music.base import GeneratedTrack

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MixSettings:
    crossfade_seconds: float = 6.0
    target_lufs: float = -14.0
    sample_rate: int = 44100


def mix_tracks(
    tracks: list[GeneratedTrack],
    output_path: Path,
    settings: MixSettings,
) -> Path:
    """Crossfade-concatenate tracks → loudness-normalise → write to output_path.

    The crossfade math:
      For tracks T0..Tn each of length L_i, with crossfade c:
        total_length = sum(L_i) - c * (n - 1)
      During each transition window of length c, both tracks play at
      complementary gains (equal-power triangular curve via FFmpeg `acrossfade`).
    """
    if not tracks:
        raise ValueError("mix_tracks requires at least one track")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(tracks) == 1:
        # No mixing needed; just normalise loudness and copy.
        return _normalise_and_write(tracks[0].path, output_path, settings)

    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "warning"]
    for t in tracks:
        cmd.extend(["-i", str(t.path)])

    # Build chained acrossfade graph. acrossfade combines two streams; for N
    # streams we chain pairwise, accumulating into [aN-1].
    #   [0][1]acrossfade=d=c:c1=tri:c2=tri[a1];
    #   [a1][2]acrossfade=d=c:c1=tri:c2=tri[a2];
    #   ...
    fade_segments = []
    accum = "[0]"
    for i in range(1, len(tracks)):
        out_label = "[mix]" if i == len(tracks) - 1 else f"[a{i}]"
        fade_segments.append(
            f"{accum}[{i}]acrossfade=d={settings.crossfade_seconds}:c1=tri:c2=tri{out_label}"
        )
        accum = out_label

    # Append loudness normalisation (single-pass dynamic loudnorm).
    filter_complex = ";".join(fade_segments) + (
        f";{accum}loudnorm=I={settings.target_lufs}:TP=-1.5:LRA=11[out]"
    )

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ar", str(settings.sample_rate),
        "-ac", "2",
        "-c:a", "pcm_s16le",   # WAV output; final encode happens in compose stage
        str(output_path),
    ])

    log.info("Mixing %d tracks → %s (xfade=%.1fs, LUFS=%.1f)",
             len(tracks), output_path, settings.crossfade_seconds, settings.target_lufs)
    log.debug("ffmpeg cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def _normalise_and_write(src: Path, dst: Path, settings: MixSettings) -> Path:
    cmd = [
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(src),
        "-af", f"loudnorm=I={settings.target_lufs}:TP=-1.5:LRA=11",
        "-ar", str(settings.sample_rate),
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def expected_total_seconds(track_durations: list[float], crossfade: float) -> float:
    """Predicted final mix length given per-track durations and one crossfade between adjacent tracks."""
    if not track_durations:
        return 0.0
    if len(track_durations) == 1:
        return track_durations[0]
    return sum(track_durations) - crossfade * (len(track_durations) - 1)
