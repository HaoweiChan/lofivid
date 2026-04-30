"""Overlay-motion parallax backend.

Drop-in replacement for `DepthFlowBackend` when the channel direction is
"photographic / minimal-design lofi" rather than anime-keyframe parallax.
DepthFlow's camera orbit reads as "AI music video"; for a lofi feed we
want the still to feel like a poster on the wall and the *air* in front
of it to move (gentle zoom, vignette breath, drifting motes).

Three motion presets, all pure FFmpeg filter graphs (no extra deps):

  • slow_zoom      — 1.000 → 1.0+amp → 1.000 over the loop, cosine-eased
                     so the seam at t=0 == t=duration is invisible.
  • light_flicker  — gentle per-frame vignette intensity oscillation,
                     periodic at t=d.
  • dust_motes     — sparse white dots whose positions cycle exactly once
                     across the canvas in `duration_seconds` (so frame N
                     == frame 0 and the loop is seamless).
  • none           — passthrough still (for testing).

Loop seamlessness is the load-bearing invariant: every preset must have
output frame N == output frame 0 so `loop_clip_to_duration` can stitch
the clip back-to-back without a visible jump. We achieve this by making
every time-varying expression periodic over `duration_seconds`.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path
from typing import Literal

from lofivid._ffmpeg import ffmpeg_bin
from lofivid.visuals.base import GeneratedClip, ParallaxBackend, ParallaxSpec

log = logging.getLogger(__name__)

MotionType = Literal["slow_zoom", "dust_motes", "light_flicker", "none"]


class OverlayMotionBackend(ParallaxBackend):
    """Animate a still keyframe with element-level overlays via FFmpeg filters."""

    name = "overlay_motion"

    def __init__(
        self,
        motion_type: MotionType = "slow_zoom",
        *,
        zoom_amplitude: float = 0.05,        # 0.05 → max 5% in / out
        flicker_amplitude: float = 0.08,     # 0.08 → ±8% vignette intensity
        dust_density: float = 0.00015,       # threshold; lower → fewer motes
        crf: int = 22,
        x264_preset: str = "medium",
    ) -> None:
        self.motion_type: MotionType = motion_type
        self.zoom_amplitude = zoom_amplitude
        self.flicker_amplitude = flicker_amplitude
        self.dust_density = dust_density
        self.crf = crf
        self.x264_preset = x264_preset

    def warmup(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def generate(self, spec: ParallaxSpec, output_dir: Path) -> GeneratedClip:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{spec.scene_index:03d}.mp4"

        log.info(
            "Overlay-motion scene %d (%ds @ %dx%d, motion=%s)",
            spec.scene_index, spec.duration_seconds, spec.width, spec.height,
            self.motion_type,
        )

        cmd = self._build_ffmpeg_cmd(spec, out_path)
        log.debug("overlay_motion cmd: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"OverlayMotion render failed for scene {spec.scene_index} "
                f"(image={spec.image_path}, motion={self.motion_type}). exit={e.returncode}"
            ) from e

        if not out_path.exists():
            raise RuntimeError(
                f"OverlayMotion ran but did not produce {out_path}. Check stderr above."
            )
        return GeneratedClip(spec=spec, path=out_path)

    # ---------- ffmpeg command -------------------------------------------

    def _build_ffmpeg_cmd(self, spec: ParallaxSpec, out_path: Path) -> list[str]:
        if self.motion_type == "dust_motes":
            return self._cmd_with_filter_complex(spec, out_path)
        return self._cmd_simple(spec, out_path)

    def _cmd_simple(self, spec: ParallaxSpec, out_path: Path) -> list[str]:
        """Single-input pipeline used by slow_zoom / light_flicker / none."""
        return [
            ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "warning",
            "-loop", "1",
            "-i", str(spec.image_path),
            "-t", f"{spec.duration_seconds}",
            "-r", str(spec.fps),
            "-vf", self._simple_filter(spec),
            *self._encode_flags(),
            str(out_path),
        ]

    def _cmd_with_filter_complex(self, spec: ParallaxSpec, out_path: Path) -> list[str]:
        """Two-source pipeline (still + procedural dust layer)."""
        w, h, d, fps = spec.width, spec.height, max(spec.duration_seconds, 1), spec.fps
        return [
            ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "warning",
            "-loop", "1",
            "-i", str(spec.image_path),
            "-f", "lavfi",
            "-i", f"color=c=black@0.0:s={w}x{h}:d={d}:r={fps}",
            "-t", f"{d}",
            "-r", str(fps),
            "-filter_complex", self._dust_motes_complex(spec),
            "-map", "[out]",
            *self._encode_flags(),
            str(out_path),
        ]

    @staticmethod
    def _encode_flags() -> list[str]:
        return [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-tune", "film",
            "-an",
        ]

    # ---------- filter graphs --------------------------------------------

    def _simple_filter(self, spec: ParallaxSpec) -> str:
        w, h = spec.width, spec.height
        d = max(spec.duration_seconds, 1)
        cover = self._cover_fit(w, h)

        if self.motion_type == "none":
            return cover
        if self.motion_type == "slow_zoom":
            return self._slow_zoom_graph(w, h, d, cover)
        if self.motion_type == "light_flicker":
            return self._light_flicker_graph(d, cover)
        raise ValueError(f"_simple_filter doesn't handle {self.motion_type!r}")

    @staticmethod
    def _cover_fit(w: int, h: int) -> str:
        """Cover-fit the still: scale-up + centre-crop, no letterboxing."""
        return (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1"
        )

    def _slow_zoom_graph(self, w: int, h: int, d: int, cover: str) -> str:
        """Cosine-eased breath zoom: z(0) = z(d) = 1.0, z(d/2) = 1+amp.

        Implemented via `scale + crop` rather than `zoompan` because
        zoompan's per-frame-input model interacts badly with `-loop 1` on
        a single still image (it iterates `on` once and freezes).
        """
        amp = self.zoom_amplitude
        zoom_expr = f"1+{amp}*(1-cos(2*PI*t/{d}))/2"
        # Compute crop window from zoom; centre-crop for symmetric breath.
        # Output dims fixed at w×h regardless of zoom level.
        return (
            f"{cover},"
            f"scale=w='iw*({zoom_expr})':h='ih*({zoom_expr})':eval=frame,"
            f"crop={w}:{h}:'(iw-{w})/2':'(ih-{h})/2'"
        )

    def _light_flicker_graph(self, d: int, cover: str) -> str:
        """Vignette with a sinusoidal angle; periodic at t=d."""
        amp = self.flicker_amplitude
        base = math.pi / 5
        angle_expr = f"{base}+{amp}*sin(2*PI*t/{d})"
        return f"{cover},vignette=angle='{angle_expr}':eval=frame"

    def _dust_motes_complex(self, spec: ParallaxSpec) -> str:
        """Drifting white motes overlaid on the cover-fit still.

        Mote positions are deterministic from the seed (so re-runs of the
        same scene produce identical dust). Drift is a horizontal modulus
        that completes exactly one full canvas-width loop in `duration`,
        so frame N == frame 0 and the loop is seamless.

        Hash strategy: ffmpeg's `geq.random(N)` is a *stateful* PRNG (N is a
        register index 0–9), not a coordinate hash, so we can't use it for
        per-pixel deterministic dots. Instead we synthesise a coordinate
        hash with `fract(sin(...) * 43758.5453)` — the standard GLSL idiom
        for cheap deterministic noise. Snapping coordinates to a 4-pixel
        grid + a 0.6-sigma gaussian blur turns the speckles into soft
        round motes.
        """
        w, h = spec.width, spec.height
        d = max(spec.duration_seconds, 1)
        density = self.dust_density
        seed = (spec.seed & 0xFFFF) * 1.0
        cover = self._cover_fit(w, h)

        # cell snaps to a 4-px grid; drift wraps once per loop so frame N
        # == frame 0. T is geq's time-in-seconds variable (uppercase).
        cell = 4
        # phase = sin(cell_x * a + cell_y * b + seed) * c
        # hash  = phase - floor(phase)        ∈ [0, 1)
        hash_expr = (
            f"(sin("
            f"floor(mod(X+T*{w}/{d},{w})/{cell})*12.9898"
            f"+floor(Y/{cell})*78.233"
            f"+{seed}"
            f")*43758.5453)"
        )
        mote_a = f"if(lt({hash_expr}-floor({hash_expr}),{density}),180,0)"

        return (
            f"[0]{cover},format=rgba[base];"
            f"[1]format=rgba,"
            f"geq=r=255:g=255:b=255:a='{mote_a}',"
            f"gblur=sigma=0.6[dust];"
            f"[base][dust]overlay=0:0:format=auto[out]"
        )


from lofivid.visuals.registry import register_parallax as _register_parallax  # noqa: E402

_register_parallax("overlay_motion", OverlayMotionBackend)
