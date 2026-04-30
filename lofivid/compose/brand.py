"""Persistent brand-text overlay renderer.

A *brand layer* is a static text overlay rendered once and composited on
every frame of the output (kicker line, display title, CJK subtitle,
tracklist strip — see preview_workday_cafe.py for the reference layout).

Multiple TextLayerSpec entries from the active StyleSpec produce one
composite PNG sized exactly to the frame, so the compose stage adds at
most one extra overlay to its filter graph regardless of how many
layers the style declares.

Cache key includes the layers + frame size so per-style edits invalidate
cleanly.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw

from lofivid.compose._text import draw_text_with_fallback, load_font, measure_text
from lofivid.styles.schema import TextLayerSpec

log = logging.getLogger(__name__)


def render_brand_layer(
    layers: list[TextLayerSpec],
    frame_w: int,
    frame_h: int,
    cache_dir: Path,
) -> Path | None:
    """Render all enabled `layers` into one frame-sized transparent PNG.

    Returns the PNG path, or None if no enabled layers.
    """
    enabled = [layer for layer in layers if layer.enabled]
    if not enabled:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(enabled, frame_w, frame_h)
    out_path = cache_dir / f"brand_{key}.png"
    if out_path.exists():
        return out_path

    img = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for layer in enabled:
        _render_single_layer(draw, layer, frame_w, frame_h)
    img.save(str(out_path), format="PNG")
    log.debug("Brand PNG rendered: %s (%dx%d, %d layers)", out_path, frame_w, frame_h, len(enabled))
    return out_path


def _render_single_layer(
    draw: ImageDraw.ImageDraw,
    layer: TextLayerSpec,
    frame_w: int,
    frame_h: int,
) -> None:
    size = max(int(frame_h * layer.size_pct), 10)
    primary = load_font(layer.font_path, size)
    fallback = load_font(layer.cjk_font_path, size) if layer.cjk_font_path else None

    text_w, text_h = measure_text(draw, layer.text, primary, fallback)
    margin = int(frame_h * layer.margin_pct)

    # Vertical anchor: top / bottom / centre
    pos = layer.position
    if pos.startswith("top_"):
        y = margin
    elif pos.startswith("bottom_"):
        y = frame_h - text_h - margin
    elif pos == "centre":
        y = (frame_h - text_h) // 2
    else:
        raise ValueError(f"unknown position: {pos!r}")

    # Horizontal anchor: centre / left / right
    if pos.endswith("_centre") or pos == "centre":
        x = (frame_w - text_w) // 2
    elif pos.endswith("_left"):
        x = margin
    elif pos.endswith("_right"):
        x = frame_w - text_w - margin
    else:
        raise ValueError(f"unknown position: {pos!r}")

    shadow = None
    if layer.shadow_color:
        shadow = (layer.shadow_color, (1, 1))

    draw_text_with_fallback(
        draw, (x, y), layer.text, primary, fallback,
        fill=layer.color, shadow=shadow,
    )


def _cache_key(layers: list[TextLayerSpec], frame_w: int, frame_h: int) -> str:
    payload = {
        "layers": [layer.model_dump(mode="json") for layer in layers],
        "frame": [frame_w, frame_h],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
