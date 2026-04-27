"""Pydantic schema for the per-video YAML config.

Loaded by `lofivid generate --config <path>`. The schema is intentionally
strict: typos in YAML keys raise immediately rather than being silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MusicAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bpm_range: tuple[int, int] = Field(..., description="Inclusive [low, high] BPM bounds")
    key_pool: list[str] = Field(..., min_length=1)
    style_tags: list[str] = Field(default_factory=list)


class MusicVariation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mood: str
    instruments: list[str] = Field(default_factory=list)


class MusicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["acestep", "musicgen"] = "acestep"
    track_count: int = Field(..., ge=1, le=200)
    track_seconds_range: tuple[int, int] = (300, 420)
    crossfade_seconds: float = Field(6.0, ge=0.0, le=20.0)
    target_lufs: float = Field(-14.0, ge=-30.0, le=-6.0)
    anchor: MusicAnchor
    variations: list[MusicVariation] = Field(..., min_length=1)


class LoraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    weight: float = Field(0.7, ge=0.0, le=2.0)


class VisualsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preset: Literal["anime", "photo"] = "anime"
    scene_count: int = Field(..., ge=1, le=200)
    scene_seconds: int = Field(..., ge=10)
    parallax_loop_seconds: int = Field(30, ge=5, le=120)
    premium_scenes: int = Field(0, ge=0)
    keyframe_prompt_template: str = ""
    loras: list[LoraSpec] = Field(default_factory=list)


class OverlaysConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rain_video: Path | None = None
    rain_opacity: float = Field(0.15, ge=0.0, le=1.0)
    vinyl_crackle: Path | None = None
    vinyl_gain_db: float = Field(-28.0, le=0.0)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    duration_minutes: float = Field(..., gt=0)
    output_resolution: tuple[int, int] = (1920, 1080)
    fps: int = Field(24, ge=12, le=60)
    seed: int = 42

    music: MusicConfig
    visuals: VisualsConfig
    overlays: OverlaysConfig = Field(default_factory=OverlaysConfig)

    @model_validator(mode="after")
    def _check_duration_matches_visuals(self) -> "Config":
        target_seconds = self.duration_minutes * 60
        visuals_seconds = self.visuals.scene_count * self.visuals.scene_seconds
        # Allow 5% slack — the timeline scheduler will pad/trim as needed.
        if abs(visuals_seconds - target_seconds) > 0.05 * target_seconds:
            raise ValueError(
                f"visuals.scene_count * visuals.scene_seconds ({visuals_seconds}s) "
                f"does not match duration_minutes ({target_seconds}s) within 5%."
            )
        return self


def load(path: str | Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)
