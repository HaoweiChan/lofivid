"""Audio-driven waveform overlay (FFmpeg `showwaves` filter).

Compose stage splices the returned fragment into its `-filter_complex`,
then overlays the [wave] stream on the running video at y=0 (top) or
y=H-h (bottom). Width auto-matches the frame; height is fixed by the
WaveformSpec.

Color resolution:
  - `color_source: fixed` → `WaveformSpec.fixed_color`
  - `color_source: duotone_highlight` → second tuple of style.duotone
  - `color_source: duotone_shadow` → first tuple of style.duotone

`build_waveform_filter` validates that the duotone is set when the
source asks for one — the StyleSpec validator already does this, so
reaching here with an inconsistent state is a programming error.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from lofivid.styles.schema import WaveformSpec

log = logging.getLogger(__name__)

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class WaveformOverlay:
    """Compose-stage descriptor for the waveform integration.

    Compose stage adds:
      * one new -filter_complex segment (the `filter_fragment` field)
      * one overlay step `[base][wave]overlay=0:Y[next]` where Y depends
        on `position`.
    The audio input index `audio_idx` is embedded in the fragment as
    `[<audio_idx>:a]`.
    """

    filter_fragment: str
    output_label: str  # e.g. "[wave]"
    position: str      # "top" | "bottom"


def resolve_color(spec: WaveformSpec, duotone: tuple[RGB, RGB] | None) -> str:
    """Return a CSS-style hex color string for the showwaves `colors=` arg."""
    if spec.color_source == "fixed":
        return spec.fixed_color
    if duotone is None:
        raise ValueError(
            f"waveform color_source={spec.color_source!r} requires style.duotone "
            "to be set; otherwise switch to color_source='fixed'."
        )
    shadow_rgb, highlight_rgb = duotone
    rgb = highlight_rgb if spec.color_source == "duotone_highlight" else shadow_rgb
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def build_waveform_filter(
    spec: WaveformSpec,
    duotone: tuple[RGB, RGB] | None,
    frame_w: int,
    fps: int,
    audio_idx: int,
    output_label: str = "[wave]",
) -> WaveformOverlay | None:
    """Construct the showwaves filter fragment.

    Returns None when `spec.enabled is False`.
    """
    if not spec.enabled:
        return None
    color = resolve_color(spec, duotone)
    # showwaves color spec accepts hex with optional `@opacity` suffix;
    # FFmpeg interprets opacity as a 0..1 float.
    color_arg = f"{color}@{spec.opacity:.3f}"
    fragment = (
        f"[{audio_idx}:a]showwaves="
        f"s={frame_w}x{spec.height_px}:"
        f"mode={spec.mode}:"
        f"colors={color_arg}:"
        f"scale={spec.scale}:"
        f"rate={fps},"
        f"format=yuva420p,setpts=PTS-STARTPTS{output_label}"
    )
    return WaveformOverlay(
        filter_fragment=fragment,
        output_label=output_label,
        position=spec.position,
    )


def overlay_y_expr(position: str, frame_h: int, height_px: int) -> str:
    """Y coordinate for the [base][wave] overlay step."""
    if position == "top":
        return "0"
    if position == "bottom":
        return str(frame_h - height_px)
    raise ValueError(f"unknown waveform position: {position!r}")
