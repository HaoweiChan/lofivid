"""Style template schema + loader. See lofivid.styles.schema for the spec."""
from __future__ import annotations

from lofivid.styles.loader import load_style, style_hash
from lofivid.styles.schema import (
    HUDSpec,
    LoraSpec,
    MusicAnchor,
    MusicVariation,
    StyleSpec,
    TextLayerSpec,
    WaveformSpec,
)

__all__ = [
    "HUDSpec",
    "LoraSpec",
    "MusicAnchor",
    "MusicVariation",
    "StyleSpec",
    "TextLayerSpec",
    "WaveformSpec",
    "load_style",
    "style_hash",
]
