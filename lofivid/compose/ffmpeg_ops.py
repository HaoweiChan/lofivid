"""FFmpeg helpers — concatenation, looping, encoding, muxing.

Implementation notes:
- We avoid MoviePy entirely (10x slowdown in 2.x, RAM leaks in subprocess piping).
- We avoid Remotion (no GPU encode path, browser-rendered).
- ffmpeg-python is used for filter graph composition; raw subprocess.run
  is used for final encode where we want explicit control.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lofivid.compose.timeline import ScheduledScene

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodeSettings:
    """av1_nvenc settings — Blackwell-optimised.

    cq=28 is a lofi-appropriate quality target: visually lossless for
    soft, gradient-heavy content. preset=p4 balances speed and quality.
    Avoid combining hevc_nvenc + tune=uhq + highbitdepth on RTX 50 series
    (known artifact bug).
    """
    encoder: str = "av1_nvenc"
    preset: str = "p4"
    cq: int = 28
    fps: int = 24
    width: int = 1920
    height: int = 1080
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"


def loop_clip_to_duration(src: Path, target_seconds: float, dst: Path, fps: int) -> Path:
    """Loop a short clip to fill `target_seconds`, encoded losslessly to a temp file.

    Used to extend a 30-sec parallax loop into a multi-minute scene.
    `-stream_loop -1` plus `-t` does this in a single pass without RAM.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-stream_loop", "-1",
        "-i", str(src),
        "-t", f"{target_seconds:.3f}",
        "-r", str(fps),
        "-c:v", "copy",          # losslessly extend; encode happens later
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def concat_with_crossfades(
    scenes: list[ScheduledScene],
    audio_path: Path,
    output_path: Path,
    settings: EncodeSettings,
    overlay_video: Path | None = None,
    overlay_opacity: float = 0.15,
    overlay_audio: Path | None = None,
    overlay_audio_gain_db: float = -28.0,
) -> Path:
    """Final pass: chain scene clips with xfade, layer overlays, mux audio, encode.

    For 1-2 hour outputs we let FFmpeg stream rather than holding the timeline
    in RAM. The xfade filter is the load-bearing piece — it requires absolute
    offset times which timeline.schedule() pre-computes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]

    # Inputs: one per scene clip + music + optional overlays.
    for s in scenes:
        cmd.extend(["-i", str(s.clip_path)])
    cmd.extend(["-i", str(audio_path)])
    if overlay_video is not None:
        cmd.extend(["-stream_loop", "-1", "-i", str(overlay_video)])
    if overlay_audio is not None:
        cmd.extend(["-stream_loop", "-1", "-i", str(overlay_audio)])

    n_scenes = len(scenes)
    audio_idx = n_scenes
    overlay_v_idx = n_scenes + 1 if overlay_video is not None else None
    overlay_a_idx = n_scenes + (2 if overlay_video is not None else 1) if overlay_audio is not None else None

    # Build the video filter chain
    vf_parts: list[str] = []
    # Scale + pad each scene to the target resolution
    for i in range(n_scenes):
        vf_parts.append(
            f"[{i}:v]scale={settings.width}:{settings.height}:force_original_aspect_ratio=decrease,"
            f"pad={settings.width}:{settings.height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={settings.fps}[v{i}]"
        )

    # Chain xfades between scenes
    if n_scenes == 1:
        vf_parts.append("[v0]copy[vbase]")
    else:
        prev_label = "[v0]"
        cum_offset = scenes[0].duration  # offset for next xfade = current scene length
        for i in range(1, n_scenes):
            out_label = f"[vc{i}]" if i < n_scenes - 1 else "[vbase]"
            xfade_d = scenes[i].crossfade_in
            offset = cum_offset - xfade_d
            vf_parts.append(
                f"{prev_label}[v{i}]xfade=transition=fade:duration={xfade_d:.3f}:offset={offset:.3f}{out_label}"
            )
            cum_offset += scenes[i].duration - xfade_d
            prev_label = out_label
        # vbase now holds the full chained video stream

    # Optional rain overlay
    if overlay_v_idx is not None:
        vf_parts.append(
            f"[{overlay_v_idx}:v]scale={settings.width}:{settings.height},format=yuva420p,"
            f"colorchannelmixer=aa={overlay_opacity}[ov]"
        )
        vf_parts.append("[vbase][ov]overlay=shortest=1[vfinal]")
        v_out = "[vfinal]"
    else:
        v_out = "[vbase]"

    # Audio: optional vinyl-crackle overlay mixed with music
    if overlay_a_idx is not None:
        vf_parts.append(f"[{overlay_a_idx}:a]volume={overlay_audio_gain_db}dB[oa]")
        vf_parts.append(f"[{audio_idx}:a][oa]amix=inputs=2:duration=first:dropout_transition=0[aout]")
        a_out = "[aout]"
    else:
        a_out = f"[{audio_idx}:a]"

    cmd.extend([
        "-filter_complex", ";".join(vf_parts),
        "-map", v_out, "-map", a_out,
        "-c:v", settings.encoder,
        "-preset", settings.preset,
        "-tune", "hq",
        "-cq", str(settings.cq),
        "-pix_fmt", "yuv420p",
        "-c:a", settings.audio_codec,
        "-b:a", settings.audio_bitrate,
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ])

    log.info("Composing → %s (%dx%d @ %dfps, encoder=%s)",
             output_path, settings.width, settings.height, settings.fps, settings.encoder)
    log.debug("ffmpeg cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def probe_duration_seconds(path: Path) -> float:
    """Lightweight duration probe via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out) if out else 0.0
