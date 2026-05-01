"""Style template schema — reusable channel-brand definition.

A *style* is the brand: prompts, music anchor, motion type, palette,
typography, HUD/waveform settings. Hashable; one style = one look across
many videos. A *run* (in lofivid/config.py) references a style by name
and only sets per-render parameters.

This split is what makes "rotate through styles, but every video on a
given style looks like the same channel" work. Editing a style file
invalidates that style's caches but leaves other styles untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Re-exported from config so styles can compose them without circular imports.
# (config.py imports schema.py, not the other way round.)
class MusicAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bpm_range: tuple[int, int] = Field(..., description="Inclusive [low, high] BPM bounds")
    key_pool: list[str] = Field(..., min_length=1)
    style_tags: list[str] = Field(default_factory=list)


class MusicVariation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mood: str
    instruments: list[str] = Field(default_factory=list)
    lyrics: str | None = None


class LoraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    weight: float = Field(0.7, ge=0.0, le=2.0)


class TextLayerSpec(BaseModel):
    """One static text overlay that's baked once and composited on every frame."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    text: str
    font_path: Path
    cjk_font_path: Path | None = None
    size_pct: float = Field(0.022, gt=0, le=0.5)
    color: str = "#FFFFFF"
    shadow_color: str | None = None
    position: Literal[
        "top_centre", "top_left", "top_right",
        "bottom_centre", "bottom_left", "bottom_right",
        "centre",
    ] = "top_centre"
    margin_pct: float = 0.06


class HUDSpec(BaseModel):
    """Per-track 'now playing' badge that updates at track boundaries."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    font_path: Path
    cjk_font_path: Path | None = None
    title_size_pct: float = 0.028
    artist_size_pct: float = 0.020
    counter_size_pct: float = 0.018
    text_color: str = "#FFFFFF"
    panel_color: str = "#000000"
    panel_opacity: float = 0.45
    panel_padding_px: int = 14
    corner: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = "bottom_left"
    margin_pct: float = 0.04
    show_track_counter: bool = True
    show_artist: bool = True
    max_width_pct: float = 0.4


class WaveformSpec(BaseModel):
    """Audio-driven waveform band along the bottom (or top) of the video."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    mode: Literal["line", "cline", "p2p", "point"] = "cline"
    height_px: int = 80
    position: Literal["top", "bottom"] = "bottom"
    scale: Literal["lin", "log", "sqrt", "cbrt"] = "sqrt"
    color_source: Literal["fixed", "duotone_highlight", "duotone_shadow"] = "duotone_highlight"
    fixed_color: str = "#E0E0C4"
    opacity: float = 0.6


class StyleSpec(BaseModel):
    """A reusable channel-brand definition."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Human-readable identifier; matches filename stem")
    description: str = ""

    # Visual identity
    preset: Literal["anime", "photo"] = "photo"
    keyframe_backend: str = "unsplash"
    keyframe_backend_params: dict[str, Any] = Field(default_factory=dict)
    keyframe_prompt_template: str
    parallax_backend: str = "overlay_motion"
    parallax_backend_params: dict[str, Any] = Field(default_factory=dict)
    motion_type: Literal["slow_zoom", "dust_motes", "light_flicker", "none"] = "slow_zoom"
    duotone: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    loras: list[LoraSpec] = Field(default_factory=list)

    # Music identity
    music_backend: str = "library"
    music_backend_params: dict[str, Any] = Field(default_factory=dict)
    music_anchor: MusicAnchor
    music_variations: list[MusicVariation] = Field(..., min_length=1)

    # Brand layers + per-track HUD + waveform
    brand_layers: list[TextLayerSpec] = Field(default_factory=list)
    hud: HUDSpec
    waveform: WaveformSpec

    # Ingest hint: per-mood search tags for `lofivid music-ingest --style <name>`.
    # Maps mood_slug → list of search keywords passed to the ingest source.
    # Empty mapping = ingest CLI must be run with explicit --mood as the only tag.
    # Participates in the style hash (changing it changes which library you'd ingest).
    library_search_tags: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_waveform_color_source(self) -> StyleSpec:
        # If color_source is duotone-derived, duotone must be set.
        if self.waveform.color_source != "fixed" and self.duotone is None:
            raise ValueError(
                f"style {self.name!r}: waveform.color_source={self.waveform.color_source!r} "
                "requires style.duotone to be set; otherwise use color_source='fixed'."
            )
        return self
