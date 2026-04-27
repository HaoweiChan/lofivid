"""DepthFlow parallax loop generation.

DepthFlow takes a still image, estimates depth (Depth Anything v2), and
renders a 2.5D camera-orbit animation. Output is a seamless loop suitable
as a multi-minute background. Near-real-time on a 5070 Ti, so this carries
~80% of the visual runtime in our pipeline.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from lofivid.visuals.base import GeneratedClip, ParallaxBackend, ParallaxSpec

log = logging.getLogger(__name__)


class DepthFlowBackend(ParallaxBackend):
    name = "depthflow"

    def __init__(self, motion_intensity: float = 0.5) -> None:
        self.motion_intensity = motion_intensity
        self._scene: Any = None

    def _ensure_loaded(self) -> Any:
        if self._scene is not None:
            return self._scene
        try:
            from DepthFlow.Scene import DepthScene  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError(
                "DepthFlow not available. Install with `pip install depthflow` "
                "(plus its torch + opengl deps; pre-installed in Docker)."
            ) from e
        log.info("Initialising DepthFlow scene")
        self._scene = DepthScene()
        return self._scene

    def warmup(self) -> None:
        self._ensure_loaded()

    def shutdown(self) -> None:
        # DepthFlow holds GL resources; let it lazily clean up.
        self._scene = None

    def generate(self, spec: ParallaxSpec, output_dir: Path) -> GeneratedClip:
        """Render a parallax loop using DepthFlow's CLI for stability.

        The DepthFlow Python API is powerful but its surface shifts between
        releases. The CLI is more stable and easier to reason about for a
        single canonical orbit pattern.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{spec.scene_index:03d}.mp4"

        log.info("Rendering parallax loop scene %d (%ds @ %dx%d)",
                 spec.scene_index, spec.duration_seconds, spec.width, spec.height)

        # Use the DepthFlow CLI (`depthflow`), which Wraps depth estimation +
        # render + ffmpeg output into a single command.
        cmd = [
            "depthflow",
            "input", "-i", str(spec.image_path),
            "main",
            "--width", str(spec.width),
            "--height", str(spec.height),
            "--fps", str(spec.fps),
            "--time", str(spec.duration_seconds),
            "--output", str(out_path),
            # Subtle, slow camera orbit appropriate for lofi backgrounds.
            "--ssaa", "1.5",
        ]
        log.debug("depthflow cmd: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "`depthflow` CLI not found on PATH. Install with `pip install depthflow`."
            ) from e

        return GeneratedClip(spec=spec, path=out_path)
