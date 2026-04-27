"""Preset ABC: ties together model choice, LoRAs, and prompt template."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PresetSpec:
    name: str
    model_id: str
    loras: list[tuple[str, float]]
    width: int
    height: int
    quality_suffix: str   # appended to every prompt for consistent quality
    negative_prompt: str


class Preset(ABC):
    @abstractmethod
    def spec(self) -> PresetSpec: ...

    def render_prompt(self, base_prompt: str, scene_index: int) -> str:
        s = self.spec()
        return f"{base_prompt.strip()}, {s.quality_suffix}".strip(", ")
