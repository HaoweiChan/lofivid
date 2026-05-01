"""FFmpeg helpers — concatenation, looping, encoding, muxing.

Implementation notes:
- We avoid MoviePy entirely (10x slowdown in 2.x, RAM leaks in subprocess piping).
- We avoid Remotion (no GPU encode path, browser-rendered).
- ffmpeg-python is used for filter graph composition; raw subprocess.run
  is used for final encode where we want explicit control.
- Encoder is auto-probed via lofivid._ffmpeg.select_encoder() so Docker
  (NVENC av1) and host-mode (libx264) both work without code changes.

The compose stage's overlay order is fixed (no config knob): rain → brand
→ HUD per-track → waveform. This matches how the reference channels read
visually; reordering it makes the typography fight the music-reactive band.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from lofivid._ffmpeg import EncoderProfile, ffmpeg_bin, ffprobe_bin, select_encoder
from lofivid.compose.hud import HUDOverlay
from lofivid.compose.timeline import ScheduledScene
from lofivid.compose.waveform import build_waveform_filter, overlay_y_expr
from lofivid.styles.schema import WaveformSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodeSettings:
    """Resolution / fps / audio-codec settings; encoder is auto-probed.

    cq=28 is a lofi-appropriate quality target (visually lossless for
    soft, gradient-heavy content). The NVENC vs libx264 differences are
    abstracted away in `_ffmpeg.select_encoder` — see that module for the
    exact flag bundles.
    """
    fps: int = 24
    width: int = 1920
    height: int = 1080
    cq: int = 28
    preset: str = "p4"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    encoder_override: EncoderProfile | None = field(default=None)

    def encoder(self) -> EncoderProfile:
        return self.encoder_override or select_encoder(self.cq, self.preset)


def loop_clip_to_duration(src: Path, target_seconds: float, dst: Path, fps: int) -> Path:
    """Loop a short clip to fill `target_seconds`, using stream copy where possible."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "warning",
        "-stream_loop", "-1",
        "-i", str(src),
        "-t", f"{target_seconds:.3f}",
        "-r", str(fps),
        "-c:v", "copy",
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
    *,
    brand_layer: Path | None = None,
    hud_overlays: list[HUDOverlay] | None = None,
    waveform_spec: WaveformSpec | None = None,
    waveform_duotone: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
) -> Path:
    """Final pass: chain scene clips with xfade, layer overlays, mux audio, encode.

    Overlay order (fixed): rain → brand → HUD per-track → waveform → encode.

    Parameters
    ----------
    brand_layer
        Path to a frame-sized transparent PNG produced by
        `lofivid.compose.brand.render_brand_layer`. None = no brand overlay.
    hud_overlays
        Per-track now-playing badges. Each HUDOverlay has its own visibility
        window expressed via `enable='between(t,start,end)'`. None or [] = no HUD.
    waveform_spec
        Active style's waveform configuration. None = no waveform.
    waveform_duotone
        Required when `waveform_spec.color_source` is duotone-derived; the
        StyleSpec-level validator already enforces this at config-load time.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hud_overlays = hud_overlays or []
    cmd: list[str] = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "warning"]

    # Inputs in fixed order. The stage downstream relies on the indices:
    #   [0..N-1]      scene clips
    #   [N]           music
    #   [N+1]         (optional) rain overlay video
    #   [N+1 or 2]    (optional) vinyl-crackle audio
    #   then          (optional) brand PNG
    #   then          (optional) one HUD PNG per track
    for s in scenes:
        cmd.extend(["-i", str(s.clip_path)])
    cmd.extend(["-i", str(audio_path)])
    if overlay_video is not None:
        cmd.extend(["-stream_loop", "-1", "-i", str(overlay_video)])
    if overlay_audio is not None:
        cmd.extend(["-stream_loop", "-1", "-i", str(overlay_audio)])
    if brand_layer is not None:
        cmd.extend(["-loop", "1", "-i", str(brand_layer)])
    for h in hud_overlays:
        cmd.extend(["-loop", "1", "-i", str(h.png_path)])

    n_scenes = len(scenes)
    audio_idx = n_scenes
    cursor = n_scenes + 1
    overlay_v_idx = cursor if overlay_video is not None else None
    if overlay_video is not None:
        cursor += 1
    overlay_a_idx = cursor if overlay_audio is not None else None
    if overlay_audio is not None:
        cursor += 1
    brand_idx = cursor if brand_layer is not None else None
    if brand_layer is not None:
        cursor += 1
    hud_idx_start = cursor  # contiguous range of length len(hud_overlays)

    # ---- video filter chain ----
    vf_parts: list[str] = []
    for i in range(n_scenes):
        vf_parts.append(
            f"[{i}:v]scale={settings.width}:{settings.height}:force_original_aspect_ratio=decrease,"
            f"pad={settings.width}:{settings.height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={settings.fps}[v{i}]"
        )

    if n_scenes == 1:
        vf_parts.append("[v0]copy[vbase]")
    else:
        prev_label = "[v0]"
        cum_offset = scenes[0].duration
        for i in range(1, n_scenes):
            out_label = f"[vc{i}]" if i < n_scenes - 1 else "[vbase]"
            xfade_d = scenes[i].crossfade_in
            offset = cum_offset - xfade_d
            vf_parts.append(
                f"{prev_label}[v{i}]xfade=transition=fade:duration={xfade_d:.3f}:offset={offset:.3f}{out_label}"
            )
            cum_offset += scenes[i].duration - xfade_d
            prev_label = out_label

    # Each layer reads from the previous label and writes to the next.
    # Track the current label as we layer overlays on.
    current = "[vbase]"

    # 1. rain overlay
    if overlay_v_idx is not None:
        vf_parts.append(
            f"[{overlay_v_idx}:v]scale={settings.width}:{settings.height},format=yuva420p,"
            f"colorchannelmixer=aa={overlay_opacity}[ov]"
        )
        vf_parts.append(f"{current}[ov]overlay=shortest=1[vrain]")
        current = "[vrain]"

    # 2. brand overlay (full-frame transparent PNG, one input)
    if brand_idx is not None:
        vf_parts.append(f"[{brand_idx}:v]format=yuva420p,setpts=PTS-STARTPTS[brand]")
        vf_parts.append(f"{current}[brand]overlay=0:0:shortest=1[vbrand]")
        current = "[vbrand]"

    # 3. HUD overlays — one per track, each with its own enable window
    for i, h in enumerate(hud_overlays):
        in_idx = hud_idx_start + i
        # Format the PNG to yuva so the alpha channel is honoured.
        vf_parts.append(f"[{in_idx}:v]format=yuva420p,setpts=PTS-STARTPTS[hud{i}]")
        next_label = f"[vhud{i}]"
        vf_parts.append(
            f"{current}[hud{i}]overlay={h.x_expr}:{h.y_expr}:"
            f"enable='between(t,{h.start_seconds:.3f},{h.end_seconds:.3f})':shortest=1{next_label}"
        )
        current = next_label

    # 4. waveform overlay — splice in the showwaves fragment + a final overlay step
    if waveform_spec is not None:
        wave = build_waveform_filter(
            waveform_spec, waveform_duotone,
            frame_w=settings.width, fps=settings.fps,
            audio_idx=audio_idx, output_label="[wave]",
        )
        if wave is not None:
            vf_parts.append(wave.filter_fragment)
            y_expr = overlay_y_expr(wave.position, settings.height, waveform_spec.height_px)
            vf_parts.append(f"{current}[wave]overlay=0:{y_expr}:shortest=1[vfinal]")
            current = "[vfinal]"

    v_out = current

    # ---- audio chain (vinyl + music) ----
    if overlay_a_idx is not None:
        vf_parts.append(f"[{overlay_a_idx}:a]volume={overlay_audio_gain_db}dB[oa]")
        vf_parts.append(f"[{audio_idx}:a][oa]amix=inputs=2:duration=first:dropout_transition=0[aout]")
        a_out = "[aout]"
    else:
        a_out = f"{audio_idx}:a"  # direct stream specifier, not a filter label

    profile = settings.encoder()
    cmd.extend([
        "-filter_complex", ";".join(vf_parts),
        "-map", v_out, "-map", a_out,
        "-c:v", profile.name,
        *profile.extra_flags,
        "-pix_fmt", profile.pix_fmt,
        "-c:a", settings.audio_codec,
        "-b:a", settings.audio_bitrate,
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ])

    log.info("Composing → %s (%dx%d @ %dfps, encoder=%s)",
             output_path, settings.width, settings.height, settings.fps, profile.name)
    log.debug("ffmpeg cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def probe_duration_seconds(path: Path) -> float:
    """Lightweight duration probe via ffprobe."""
    try:
        binary = ffprobe_bin()
    except RuntimeError:
        return _probe_duration_via_ffmpeg(path)
    out = subprocess.run(
        [binary, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out) if out else 0.0


def _probe_duration_via_ffmpeg(path: Path) -> float:
    """Parse `Duration: HH:MM:SS.ms` from ffmpeg stderr when ffprobe is unavailable."""
    out = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, check=False,
    ).stderr
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Duration:"):
            try:
                hms = s.split("Duration:", 1)[1].split(",", 1)[0].strip()
                h, m, sec = hms.split(":")
                return int(h) * 3600 + int(m) * 60 + float(sec)
            except (ValueError, IndexError):
                pass
    return 0.0
