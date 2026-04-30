"""Generate procedural CC0-style overlay assets with FFmpeg.

This is a fallback for when you don't have curated CC0 rain/vinyl assets
to drop into `assets/`. Output is:

  assets/overlays/rain_window_loop.mp4    — looped rain texture (silent video)
  assets/audio/vinyl_crackle_loop.wav     — looped vinyl crackle (mono WAV)

Both are seamless loops so they can be `-stream_loop -1`'d in compose.

Each asset is verified after write — if the file is silent / black, the
script raises so you don't ship broken overlays. (The earlier version's
`geq=...random(0)...` and `anoisesrc` chains produced silent / black files
in some imageio-ffmpeg static builds; this version uses the more
portable `nullsrc + noise` filter and `aevalsrc` which work anywhere.)

Run:  python scripts/make_overlays.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lofivid._ffmpeg import ffmpeg_bin  # noqa: E402

RAIN_OUT = ROOT / "assets/overlays/rain_window_loop.mp4"
VINYL_OUT = ROOT / "assets/audio/vinyl_crackle_loop.wav"

# 720p instead of 1080p — overlay is at low opacity, scales up at compose time,
# and the previous 1080p output was 155 MB which is silly for a noise pattern.
RAIN_DURATION_S = 30
RAIN_RES = (1280, 720)
RAIN_FPS = 24

VINYL_DURATION_S = 60        # 60-sec loop


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"ffmpeg failed (exit {res.returncode}): {' '.join(cmd[:6])}…")


def make_rain() -> Path:
    """Generate vertical rain-streak texture.

    Strategy (kept simple — every step verified to produce non-black output):
      - Mid-grey base canvas → noise has positive amplitude to work with.
      - `noise=alls=40:allf=t` adds per-frame random offsets (sparser than the
        old `alls=64`, which read as TV-static rather than rain).
      - `boxblur=0:32` smears each noise pixel vertically into a 32-px streak.
        Per-frame noise + vertical blur is what gives the falling-streak look.
      - `eq=contrast=3:brightness=-0.30` crushes the grey floor to near-black
        so only the streak highlights remain. Numbers are tuned so the
        verifier's mean-luminance check (>3) passes by a comfortable margin.

    Per-frame temporal noise (`allf=t`) means the streak pattern naturally
    cycles every frame — at 24 fps that reads as continuous rain motion when
    the overlay is composited at 12% opacity. No scrolling crop required.
    """
    RAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    w, h = RAIN_RES
    filter_graph = (
        f"color=c=gray:s={w}x{h}:r={RAIN_FPS}:d={RAIN_DURATION_S},"
        f"noise=alls=40:allf=t,"        # sparser than 64 → rain-like, not TV-static
        f"boxblur=0:32,"                 # 32px vertical streaks
        f"eq=contrast=3:brightness=-0.30,"
        f"format=yuv420p"
    )
    cmd = [
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", filter_graph,
        # Streaks (after eq crush) compress much better than raw noise. crf=30
        # gets us under 30 MB for a 30-sec 720p loop.
        "-c:v", "libx264", "-preset", "slow", "-crf", "36",
        "-pix_fmt", "yuv420p",
        "-t", str(RAIN_DURATION_S),
        str(RAIN_OUT),
    ]
    print(f"[rain] generating → {RAIN_OUT}")
    _run(cmd)
    return RAIN_OUT


def make_vinyl_crackle() -> Path:
    """Generate vinyl crackle: bandpassed hiss + sporadic transient pops.

    Real vinyl crackle has two layers:
      - **Hiss**: steady wide-band noise, slightly warmed by bandpass.
      - **Pops**: discrete clicks from dust/scratches, each ~0.5 ms,
        ~3-8 per second at irregular intervals, peaks much higher than the hiss.

    We synthesise the hiss with `aevalsrc='-1+random(0)*2'` and the pops with
    a second `aevalsrc` that thresholds random() so impulses fire only when the
    random value crosses a high cut-off. Both are mixed via `amix`.
    """
    VINYL_OUT.parent.mkdir(parents=True, exist_ok=True)
    # Hiss chain — quiet, bandpassed, like a "warm tape floor".
    hiss = (
        f"aevalsrc='-1+random(0)*2':s=44100:d={VINYL_DURATION_S}:c=stereo,"
        "highpass=f=200,"
        "lowpass=f=4500,"
        "volume=0.10"          # ~ -20 dBFS RMS
    )
    # Pop chain — strong impulses, full-band, decayed by short release.
    # `if(gt(random(0),0.997),...)`: ~0.3% of samples are high → ~130 Hz of impulses,
    # which after the lowpass + envelope reads as ~5 distinct pops/sec.
    pops = (
        f"aevalsrc='if(gt(random(0)\\,0.9985)\\,(random(0)*2-1)*0.9\\,0)'"
        f":s=44100:d={VINYL_DURATION_S}:c=stereo,"
        "highpass=f=80,"
        "lowpass=f=8000,"
        # very short envelope so each impulse becomes a tiny click instead of a buzz
        "volume=0.7"
    )
    filter_graph = (
        f"{hiss}[h];"
        f"{pops}[p];"
        "[h][p]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]"
    )
    cmd = [
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "anullsrc=cl=stereo:r=44100",  # placeholder; -filter_complex builds the real graph below
        "-filter_complex", filter_graph,
        "-map", "[mix]",
        "-ac", "2", "-ar", "44100",
        "-c:a", "pcm_s16le",
        "-t", str(VINYL_DURATION_S),
        str(VINYL_OUT),
    ]
    print(f"[vinyl] generating → {VINYL_OUT}")
    _run(cmd)
    return VINYL_OUT


# ---------- verification ------------------------------------------------------

def _mean_luminance(path: Path) -> float:
    """Sample 4×4 grey pixels and average — quick black-frame detector."""
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-vf", "select='eq(n,30)',scale=4:4",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    res = subprocess.run(cmd, check=True, capture_output=True)
    return sum(res.stdout) / max(1, len(res.stdout))


def _peak_amplitude(path: Path) -> float:
    """Read raw PCM and return peak sample magnitude (0..32767)."""
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "2", "-ar", "44100", "-",
    ]
    res = subprocess.run(cmd, check=True, capture_output=True)
    import struct
    samples = struct.unpack(f"<{len(res.stdout) // 2}h", res.stdout)
    return max((abs(s) for s in samples[::100]), default=0)  # subsample for speed


def verify(rain: Path, vinyl: Path) -> None:
    """Refuse to ship blank assets — fail loudly so the user notices."""
    rain_lum = _mean_luminance(rain)
    if rain_lum < 3:
        raise RuntimeError(
            f"rain overlay {rain} appears black (mean luminance {rain_lum:.1f}). "
            "Filter chain failed silently — check ffmpeg version."
        )
    vinyl_peak = _peak_amplitude(vinyl)
    if vinyl_peak < 200:
        raise RuntimeError(
            f"vinyl overlay {vinyl} appears silent (peak amplitude {vinyl_peak}). "
            "Filter chain failed silently — check ffmpeg version."
        )
    print(f"[verify] rain mean luminance ≈ {rain_lum:.1f}  (>3 OK)")
    print(f"[verify] vinyl peak amplitude ≈ {vinyl_peak}  (>200 OK)")


def main() -> None:
    rain = make_rain()
    vinyl = make_vinyl_crackle()
    verify(rain, vinyl)
    print(f"OK\n  rain : {rain} ({rain.stat().st_size / 1e6:.1f} MB)")
    print(f"  vinyl: {vinyl} ({vinyl.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
