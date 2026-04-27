"""Overlay sourcing helpers.

Right now this is a thin module — the actual overlay layering happens
in `ffmpeg_ops.concat_with_crossfades`. This file exists to (a) validate
overlay paths exist before we kick off a long render and (b) house future
asset-discovery / auto-download logic.
"""

from __future__ import annotations

from pathlib import Path

from lofivid.config import OverlaysConfig


class OverlayValidationError(Exception):
    """Raised if a configured overlay path doesn't exist on disk."""


def validate(cfg: OverlaysConfig, project_root: Path) -> None:
    """Resolve relative paths against project_root and ensure they exist."""
    for label, p in [("rain_video", cfg.rain_video), ("vinyl_crackle", cfg.vinyl_crackle)]:
        if p is None:
            continue
        resolved = p if p.is_absolute() else (project_root / p)
        if not resolved.exists():
            raise OverlayValidationError(
                f"Overlay {label} not found at {resolved}. "
                "Drop a CC0 asset there or set the field to null in the YAML."
            )
