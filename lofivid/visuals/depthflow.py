"""DepthFlow parallax loop generation.

DepthFlow takes a still image, estimates depth (Depth Anything v2), and
renders a 2.5D camera-orbit animation. Output is a seamless loop suitable
as a multi-minute background. Near-real-time on a 5070 Ti, so this carries
~80% of the visual runtime in our pipeline.

The DepthFlow CLI (v0.9.x) uses *chained subcommands* rather than flag-only:

    depthflow input -i img.png orbital -i 0.4 main -w 1920 -h 1080 \\
              -f 24 -t 30 -o out.mp4 --ssaa 1.5 h264 -p medium --crf 22

We construct that chain explicitly because the Python API (`DepthScene`)
shifts shape between releases.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from lofivid.visuals.base import GeneratedClip, ParallaxBackend, ParallaxSpec

log = logging.getLogger(__name__)


class DepthFlowBackend(ParallaxBackend):
    """Render a parallax loop via the DepthFlow CLI.

    `motion_intensity` (0..2-ish) scales the orbital animation amplitude.
    Lofi looks best with a slow, gentle motion — 0.3-0.5 is the sweet spot.
    """

    name = "depthflow"

    def __init__(
        self,
        motion_intensity: float = 0.4,
        ssaa: float = 1.5,
        animation: str = "orbital",   # "orbital" | "dolly" | "vertical" | "horizontal" | "circle"
        crf: int = 22,
        x264_preset: str = "medium",
    ) -> None:
        self.motion_intensity = motion_intensity
        self.ssaa = ssaa
        self.animation = animation
        self.crf = crf
        self.x264_preset = x264_preset
        self._scene: Any = None  # currently unused; CLI is the stable surface

    def warmup(self) -> None:
        # CLI-based — nothing to preload host-side. The first generate() call
        # will trigger Depth Anything weight download to the HF cache.
        return None

    def shutdown(self) -> None:
        return None

    def generate(self, spec: ParallaxSpec, output_dir: Path) -> GeneratedClip:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{spec.scene_index:03d}.mp4"

        log.info(
            "Rendering parallax loop scene %d (%ds @ %dx%d, animation=%s, intensity=%.2f)",
            spec.scene_index, spec.duration_seconds, spec.width, spec.height,
            self.animation, self.motion_intensity,
        )

        # CLI structure (chained subcommands):
        #   depthflow
        #     input  -i <image>
        #     <animation_preset>  -i <intensity>
        #     main   -w <w> -h <h> -f <fps> -t <time> -o <out> --ssaa <s>
        #     h264   -p <preset> --crf <crf>  --tune film
        cmd = [
            "depthflow",
            "input", "-i", str(spec.image_path),
            self.animation, "-i", f"{self.motion_intensity:.3f}",
            "main",
            "-w", str(spec.width),
            "-h", str(spec.height),
            "-f", str(spec.fps),
            "-t", str(spec.duration_seconds),
            "-o", str(out_path),
            "--ssaa", str(self.ssaa),
            "h264",
            "--preset", self.x264_preset,   # long form: short -p resolves to --profile (depthflow CLI bug)
            "--crf", str(self.crf),
            "--tune", "film",
        ]
        log.debug("depthflow cmd: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "`depthflow` CLI not found on PATH. Install with `pip install depthflow`."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"DepthFlow render failed for scene {spec.scene_index} (image={spec.image_path}). "
                f"exit={e.returncode}. Most common causes:\n"
                "  - OpenGL context unavailable (WSL2 needs WSLg or `LIBGL_ALWAYS_SOFTWARE=1`)\n"
                "  - Depth Anything v2 weights download failed (check HF_HUB cache + network)\n"
                "  - depthflow CLI flag drift between minor versions"
            ) from e

        if not out_path.exists():
            raise RuntimeError(
                f"DepthFlow ran but did not produce {out_path}. Check stderr above."
            )

        return GeneratedClip(spec=spec, path=out_path)
