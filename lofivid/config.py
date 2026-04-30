"""Pydantic schema for the per-RUN YAML config.

Loaded by `lofivid generate --config <path>`. The schema is intentionally
strict: typos in YAML keys raise immediately rather than being silently
ignored.

Pivot v3: visual + music IDENTITY (prompts, palette, anchor, brand layers,
HUD, waveform) lives in style files at <repo>/styles/<style_ref>.yaml.
This Config holds only per-run instance parameters (duration, seed, counts).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from lofivid.styles.loader import load_style

# Re-exported for backward compatibility with consumers that import these
# from lofivid.config (e.g. tracklist.py).
from lofivid.styles.schema import LoraSpec, MusicAnchor, MusicVariation  # noqa: F401

if TYPE_CHECKING:
    from lofivid.styles.schema import StyleSpec


def _repo_root() -> Path:
    """Repo root for resolving style_ref. Override via LOFIVID_REPO_ROOT."""
    env = os.environ.get("LOFIVID_REPO_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.cwd()


class MusicInstance(BaseModel):
    """Per-run music parameters. Identity (anchor/backend/variations) lives in the style."""
    model_config = ConfigDict(extra="forbid")
    track_count: int = Field(..., ge=1, le=200)
    track_seconds_range: tuple[int, int] = (300, 420)
    crossfade_seconds: float = Field(6.0, ge=0.0, le=20.0)
    target_lufs: float = Field(-14.0, ge=-30.0, le=-6.0)


class VisualsInstance(BaseModel):
    """Per-run visual parameters. Identity (preset/backend/prompts) lives in the style."""
    model_config = ConfigDict(extra="forbid")
    scene_count: int = Field(..., ge=1, le=200)
    scene_seconds: int = Field(..., ge=10)
    parallax_loop_seconds: int = Field(30, ge=5, le=120)
    premium_scenes: int = Field(0, ge=0)


class OverlaysConfig(BaseModel):
    """Per-run CC0 overlay layers (rain video, vinyl crackle audio)."""
    model_config = ConfigDict(extra="forbid")
    rain_video: Path | None = None
    rain_opacity: float = Field(0.15, ge=0.0, le=1.0)
    vinyl_crackle: Path | None = None
    vinyl_gain_db: float = Field(-28.0, le=0.0)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    style_ref: str = Field(..., description="Name of style file at <repo>/styles/<style_ref>.yaml")
    duration_minutes: float = Field(..., gt=0)
    output_resolution: tuple[int, int] = (1920, 1080)
    fps: int = Field(24, ge=12, le=60)
    seed: int = 42

    music: MusicInstance
    visuals: VisualsInstance
    overlays: OverlaysConfig = Field(default_factory=OverlaysConfig)

    # Cached on first resolve. Not part of the model schema.
    _resolved_style: StyleSpec | None = PrivateAttr(default=None)
    _style_hash: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_duration_matches_visuals(self) -> Config:
        target_seconds = self.duration_minutes * 60
        visuals_seconds = self.visuals.scene_count * self.visuals.scene_seconds
        if abs(visuals_seconds - target_seconds) > 0.05 * target_seconds:
            raise ValueError(
                f"visuals.scene_count * visuals.scene_seconds ({visuals_seconds}s) "
                f"does not match duration_minutes ({target_seconds}s) within 5%."
            )
        return self

    def _resolve(self) -> None:
        if self._resolved_style is not None:
            return
        spec, h = load_style(self.style_ref, _repo_root())
        # PrivateAttr setattr requires going through object.__setattr__ in Pydantic v2.
        object.__setattr__(self, "_resolved_style", spec)
        object.__setattr__(self, "_style_hash", h)

    @property
    def resolved_style(self) -> StyleSpec:
        self._resolve()
        assert self._resolved_style is not None
        return self._resolved_style

    @property
    def style_hash(self) -> str:
        self._resolve()
        assert self._style_hash is not None
        return self._style_hash


def load(path: str | Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    cfg = Config.model_validate(data)
    # Eagerly resolve style so missing files / bad style YAMLs raise at load time,
    # not deep inside the pipeline.
    cfg._resolve()
    return cfg
