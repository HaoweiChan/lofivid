"""Per-track 'now playing' badge HUD overlay.

Two responsibilities:
  1. Render a transparent PNG per track with title/artist/counter on a
     rounded panel (PIL).
  2. Provide the dataclass + corner-offset math the FFmpeg compose stage
     needs to overlay each PNG during its track window.

Caching: PNG paths are content-hashed so re-renders skip already-rendered
HUDs. Inputs to the hash: the HUDSpec dump, the track's title+artist, the
counter text, and the frame size.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw

from lofivid.compose._text import (
    draw_text_with_fallback,
    load_font,
    measure_text,
    truncate_to_width,
)
from lofivid.music.mixer import TrackWindow
from lofivid.styles.schema import HUDSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HUDOverlay:
    """One PNG to overlay during one track window."""

    png_path: Path
    start_seconds: float
    end_seconds: float
    x_expr: str
    y_expr: str


def hud_corner_xy(
    corner: Literal["top_left", "top_right", "bottom_left", "bottom_right"],
    margin_px: int,
    frame_w: int,
    frame_h: int,
    panel_w: int,
    panel_h: int,
) -> tuple[str, str]:
    """Return (x_expr, y_expr) ffmpeg overlay coordinates given the corner.

    Returns literal pixel values as strings; using constants rather than
    FFmpeg W/H expressions keeps the overlay filter evaluation cheap.
    """
    if corner == "top_left":
        return str(margin_px), str(margin_px)
    if corner == "top_right":
        return str(frame_w - panel_w - margin_px), str(margin_px)
    if corner == "bottom_left":
        return str(margin_px), str(frame_h - panel_h - margin_px)
    if corner == "bottom_right":
        return str(frame_w - panel_w - margin_px), str(frame_h - panel_h - margin_px)
    raise ValueError(f"unknown corner: {corner!r}")


def _hud_cache_key(
    spec: HUDSpec,
    title: str,
    artist: str | None,
    counter_text: str,
    frame_size: tuple[int, int],
) -> str:
    payload = {
        "spec": spec.model_dump(mode="json"),
        "title": title,
        "artist": artist or "",
        "counter": counter_text,
        "frame": list(frame_size),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def render_hud_png(
    spec: HUDSpec,
    window: TrackWindow,
    track_index: int,
    track_total: int,
    frame_size: tuple[int, int],
    cache_dir: Path,
) -> tuple[Path, int, int]:
    """Render one HUD PNG; return (path, panel_w, panel_h) for placement.

    Layout: title on top (bold), artist below (smaller), counter in
    the panel's leading corner. Rounded-rect background with configured
    opacity.

    Caching: identical inputs → identical filename → no re-render.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_w, frame_h = frame_size

    title = window.track.title or f"Track {track_index + 1:02d}"
    artist = window.track.artist if spec.show_artist else None
    counter_text = (
        f"{track_index + 1:02d} / {track_total:02d}" if spec.show_track_counter else ""
    )

    key = _hud_cache_key(spec, title, artist, counter_text, frame_size)
    out_path = cache_dir / f"hud_{track_index:03d}_{key}.png"

    title_size = max(int(frame_h * spec.title_size_pct), 14)
    artist_size = max(int(frame_h * spec.artist_size_pct), 12)
    counter_size = max(int(frame_h * spec.counter_size_pct), 10)
    pad = spec.panel_padding_px

    f_title = load_font(spec.font_path, title_size)
    f_artist = load_font(spec.font_path, artist_size)
    f_counter = load_font(spec.font_path, counter_size)
    f_title_cjk = load_font(spec.cjk_font_path, title_size) if spec.cjk_font_path else None
    f_artist_cjk = load_font(spec.cjk_font_path, artist_size) if spec.cjk_font_path else None

    # Measure using a throwaway 1×1 image just for the draw context.
    probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    pd = ImageDraw.Draw(probe)
    max_text_w = int(frame_w * spec.max_width_pct) - 2 * pad
    title = truncate_to_width(title, max_text_w, pd, f_title, f_title_cjk)
    title_w, title_h = measure_text(pd, title, f_title, f_title_cjk)
    artist_w, artist_h = 0, 0
    if artist:
        artist_w, artist_h = measure_text(pd, artist, f_artist, f_artist_cjk)
    counter_w, counter_h = 0, 0
    if counter_text:
        counter_w, counter_h = measure_text(pd, counter_text, f_counter, None)

    inner_w = max(title_w, artist_w, counter_w)
    inner_h = title_h + (4 + artist_h if artist else 0) + (4 + counter_h if counter_text else 0)
    panel_w = inner_w + 2 * pad
    panel_h = inner_h + 2 * pad

    if out_path.exists():
        return out_path, panel_w, panel_h

    img = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    panel_alpha = int(255 * spec.panel_opacity)
    panel_rgba = _hex_with_alpha(spec.panel_color, panel_alpha)
    draw.rounded_rectangle(
        [(0, 0), (panel_w, panel_h)],
        radius=8,
        fill=panel_rgba,
    )

    cy = pad
    draw_text_with_fallback(
        draw, (pad, cy), title, f_title, f_title_cjk, fill=spec.text_color,
    )
    cy += title_h
    if artist:
        cy += 4
        draw_text_with_fallback(
            draw, (pad, cy), artist, f_artist, f_artist_cjk,
            fill=_dim_hex(spec.text_color, 0.85),
        )
        cy += artist_h
    if counter_text:
        cy += 4
        draw.text(
            (pad, cy), counter_text, font=f_counter,
            fill=_dim_hex(spec.text_color, 0.7),
        )

    img.save(str(out_path), format="PNG")
    log.debug("HUD PNG rendered: %s (%dx%d)", out_path, panel_w, panel_h)
    return out_path, panel_w, panel_h


def build_hud_overlays(
    spec: HUDSpec,
    windows: list[TrackWindow],
    frame_size: tuple[int, int],
    cache_dir: Path,
) -> list[HUDOverlay]:
    """Render every track's HUD PNG and return the FFmpeg overlay descriptors.

    When `spec.enabled is False`, returns []. Caller is the compose stage,
    which splices each HUDOverlay into its filter graph as
    `[base][hud_i]overlay=x=X:y=Y:enable='between(t,start,end)'[next]`.
    """
    if not spec.enabled or not windows:
        return []
    margin_px = max(int(frame_size[1] * spec.margin_pct), 4)
    overlays: list[HUDOverlay] = []
    for i, w in enumerate(windows):
        png_path, panel_w, panel_h = render_hud_png(
            spec, w, track_index=i, track_total=len(windows),
            frame_size=frame_size, cache_dir=cache_dir,
        )
        x_expr, y_expr = hud_corner_xy(
            spec.corner, margin_px, frame_size[0], frame_size[1], panel_w, panel_h,
        )
        overlays.append(HUDOverlay(
            png_path=png_path,
            start_seconds=w.start_seconds,
            end_seconds=w.end_seconds,
            x_expr=x_expr,
            y_expr=y_expr,
        ))
    return overlays


# ---- private helpers -------------------------------------------------------

def _hex_with_alpha(color: str, alpha: int) -> tuple[int, int, int, int]:
    """`#RRGGBB` → (r, g, b, alpha)."""
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"expected #RRGGBB, got {color!r}")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return r, g, b, alpha


def _dim_hex(color: str, factor: float) -> str:
    """Multiply RGB channels by `factor` (0..1). Returns a #RRGGBB string."""
    c = color.lstrip("#")
    if len(c) != 6:
        return color
    r = max(0, min(255, int(int(c[0:2], 16) * factor)))
    g = max(0, min(255, int(int(c[2:4], 16) * factor)))
    b = max(0, min(255, int(int(c[4:6], 16) * factor)))
    return f"#{r:02X}{g:02X}{b:02X}"
