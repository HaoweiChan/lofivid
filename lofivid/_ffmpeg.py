"""FFmpeg binary discovery + encoder/decoder capability probing.

Centralised so env.py / mixer.py / ffmpeg_ops.py all agree on:
  - which ffmpeg binary to call (system PATH first, then imageio-ffmpeg fallback)
  - which video encoder is actually available (NVENC > libx264 > libopenh264 > mpeg4)

Override behaviour with env vars:
  LOFIVID_FFMPEG_BIN     absolute path to ffmpeg
  LOFIVID_FFPROBE_BIN    absolute path to ffprobe
  LOFIVID_VIDEO_ENCODER  pin a specific encoder name (skip auto-probe)
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from dataclasses import dataclass

# Encoder preference: best-quality NVENC AV1 first, software libx264 last resort.
# Order matters — first hit on `ffmpeg -encoders` wins.
_ENCODER_PREFERENCE: tuple[str, ...] = (
    "av1_nvenc",
    "hevc_nvenc",
    "h264_nvenc",
    "libx264",
    "libopenh264",
    "mpeg4",
)


@dataclass(frozen=True)
class EncoderProfile:
    """Resolved encoder + the ffmpeg flags appropriate for it."""
    name: str
    extra_flags: tuple[str, ...]   # e.g. ("-preset", "p4", "-tune", "hq", "-cq", "28")
    pix_fmt: str = "yuv420p"


def _resolve_binary(env_var: str, fallback_name: str) -> str:
    """Return path to a binary. Prefer LOFIVID_*_BIN env, then PATH, then imageio-ffmpeg."""
    explicit = os.environ.get(env_var)
    if explicit:
        if not os.path.isfile(explicit):
            raise RuntimeError(f"{env_var}={explicit!r} but file does not exist")
        return explicit

    on_path = shutil.which(fallback_name)
    if on_path:
        return on_path

    # imageio-ffmpeg ships a static ffmpeg binary inside its wheel; ffprobe
    # isn't bundled, so this fallback only helps for the ffmpeg case.
    if fallback_name == "ffmpeg":
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    raise RuntimeError(
        f"Cannot find {fallback_name} on PATH or via imageio-ffmpeg. "
        f"Install it (apt install ffmpeg) or set {env_var} to an absolute path."
    )


@functools.lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    return _resolve_binary("LOFIVID_FFMPEG_BIN", "ffmpeg")


@functools.lru_cache(maxsize=1)
def ffprobe_bin() -> str:
    """Path to ffprobe.

    imageio-ffmpeg does NOT ship ffprobe. If it's missing we fall back to
    using `ffmpeg -i <file>` parsing, but most callers want a real ffprobe.
    """
    try:
        return _resolve_binary("LOFIVID_FFPROBE_BIN", "ffprobe")
    except RuntimeError:
        # If only ffmpeg (from imageio-ffmpeg) is around, callers can use
        # _probe_duration_via_ffmpeg() — but most still prefer a real ffprobe.
        # Re-raise so the missing dep is loud and fixable.
        raise


@functools.lru_cache(maxsize=1)
def list_encoders() -> set[str]:
    """Return the set of video encoder names ffmpeg reports."""
    out = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-encoders"],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout
    encoders: set[str] = set()
    in_table = False
    for line in out.splitlines():
        if line.strip().startswith("------"):
            in_table = True
            continue
        if not in_table or not line.strip():
            continue
        # Lines look like " V..... libx264              libx264 H.264 / ..."
        # Type column is the 4th char position; we skip non-video by checking
        # the leading 'V' in the flag block.
        parts = line.split()
        if len(parts) < 2:
            continue
        flags, name = parts[0], parts[1]
        if flags.startswith("V"):
            encoders.add(name)
    return encoders


def _flags_for(name: str, cq: int = 28, preset: str = "p4") -> tuple[str, ...]:
    """Encoder-specific flag bundle.

    NVENC encoders use -cq + -preset pX + -tune hq.
    libx264 uses -crf + -preset (medium).
    libopenh264 uses -b:v.
    mpeg4 uses -q:v.
    """
    if name in ("av1_nvenc", "hevc_nvenc", "h264_nvenc"):
        return ("-preset", preset, "-tune", "hq", "-cq", str(cq))
    if name == "libx264":
        # crf 20 ≈ visually lossless for soft lofi content; medium preset gives
        # good compression at reasonable speed. Tune for grain to preserve
        # vinyl/film overlay texture instead of smearing it.
        return ("-preset", "medium", "-crf", str(max(18, min(28, cq - 4))))
    if name == "libopenh264":
        return ("-b:v", "8M")
    if name == "mpeg4":
        return ("-q:v", "5")
    return ()


@functools.lru_cache(maxsize=1)
def select_encoder(cq: int = 28, preset: str = "p4") -> EncoderProfile:
    """Pick the best available encoder, honouring LOFIVID_VIDEO_ENCODER.

    Returns a `EncoderProfile` ready to splice into an ffmpeg command:
        ["-c:v", profile.name, *profile.extra_flags, "-pix_fmt", profile.pix_fmt]
    """
    pinned = os.environ.get("LOFIVID_VIDEO_ENCODER")
    if pinned:
        available = list_encoders()
        if pinned not in available:
            raise RuntimeError(
                f"LOFIVID_VIDEO_ENCODER={pinned!r} not available. "
                f"Encoders this ffmpeg knows: {sorted(available) or '(none)'}"
            )
        return EncoderProfile(name=pinned, extra_flags=_flags_for(pinned, cq, preset))

    available = list_encoders()
    for cand in _ENCODER_PREFERENCE:
        if cand in available:
            return EncoderProfile(name=cand, extra_flags=_flags_for(cand, cq, preset))

    raise RuntimeError(
        f"No usable video encoder found. ffmpeg at {ffmpeg_bin()} reports: "
        f"{sorted(available) or '(none)'}"
    )
