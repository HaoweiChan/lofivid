"""Shared audio-probing helpers used by every music backend.

Three backends previously each defined their own near-identical
`_probe_duration_seconds`. Consolidated here so the soundfile fallback
behaviour stays consistent across acestep / suno / library.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def probe_duration_seconds(path: Path) -> float:
    """Cheap duration probe via soundfile; returns 0.0 if it can't read."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception as e:
        log.warning("Could not probe duration of %s: %s", path, e)
        return 0.0
