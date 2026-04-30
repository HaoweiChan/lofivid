"""Image post-process helpers (duotone tint + paper border).

Originally lived in `scripts/preview_themes.py`; promoted here so the
Unsplash keyframe backend can apply the same look without duplicating
the implementation. Pure PIL + NumPy — no heavy GPU deps.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RGB = tuple[int, int, int]


def duotone(img: Image.Image, shadow_rgb: RGB, highlight_rgb: RGB) -> Image.Image:
    """Map an image onto a 2-colour gradient (shadow → highlight).

    The grayscale luminance of the input picks where on the gradient each
    pixel lands; pure black → shadow, pure white → highlight.
    """
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    sh = np.array(shadow_rgb, dtype=np.float32)
    hi = np.array(highlight_rgb, dtype=np.float32)
    out = sh[None, None, :] * (1 - gray[..., None]) + hi[None, None, :] * gray[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def paper_border(
    img: Image.Image,
    border_pct: float = 0.06,
    paper_rgb: RGB = (236, 226, 200),
    *,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Wrap the image in a cream paper border with subtle grain.

    Pass `rng` for deterministic grain (we want re-runs to be reproducible
    across the cache). When `rng is None` we fall back to NumPy's global
    state to keep behaviour identical to the original script.
    """
    w, h = img.size
    bw = int(min(w, h) * border_pct)
    new_w, new_h = w + 2 * bw, h + 2 * bw
    canvas = Image.new("RGB", (new_w, new_h), paper_rgb)

    if rng is None:
        grain = np.random.normal(0, 6, (new_h, new_w, 3)).astype(np.int16)
    else:
        grain = rng.normal(0, 6, (new_h, new_w, 3)).astype(np.int16)
    arr = np.array(canvas, dtype=np.int16) + grain
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    canvas.paste(img, (bw, bw))

    # Slight inner shadow on the image edge for tactile feel.
    shadow = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(shadow)
    d.rectangle([(0, 0), img.size], outline=80, width=1)
    shadow = shadow.filter(ImageFilter.GaussianBlur(2))
    canvas.paste((30, 22, 14), (bw, bw), shadow)
    return canvas


def grade(img: Image.Image, shadow_rgb: RGB, highlight_rgb: RGB,
          *, with_border: bool = True, rng: np.random.Generator | None = None) -> Image.Image:
    """Convenience: duotone + optional paper border in one call."""
    out = duotone(img, shadow_rgb, highlight_rgb)
    if with_border:
        out = paper_border(out, rng=rng)
    return out
