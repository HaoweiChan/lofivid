"""Visual presets — model + LoRA selection + scene-prompt synthesis."""

from __future__ import annotations

from lofivid.presets.anime import AnimePreset
from lofivid.presets.base import Preset
from lofivid.presets.photo import PhotoPreset

PRESETS: dict[str, type[Preset]] = {
    "anime": AnimePreset,
    "photo": PhotoPreset,
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]()
