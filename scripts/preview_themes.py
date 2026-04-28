"""Render one Loaf-style still per theme so the user can pick visually.

Outputs:
  previews/<theme>_raw.png       — raw SDXL output
  previews/<theme>_graded.png    — after duotone color grade + paper border

Usage:
  python scripts/preview_themes.py [theme1 theme2 ...]   # default: all themes

The first run downloads SDXL base 1.0 (~6.5 GB) into the HF cache. Subsequent
runs hit cache and just re-generate the images. Designed to run on a 16 GB
Blackwell GPU; uses bf16 to fit comfortably with headroom for the post-process.
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Apply Blackwell env defaults BEFORE torch is imported
from lofivid.cli import _apply_blackwell_env_defaults  # noqa: E402

_apply_blackwell_env_defaults()

PREVIEW_DIR = ROOT / "previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Loaf-style universal prompt fragments — appended to every theme.
COMMON_QUALITY_SUFFIX = (
    "single subject still life, centered composition, isolated object, "
    "vintage photograph, riso print aesthetic, soft studio lighting, "
    "shallow depth of field, matte texture, minimalist, "
    "negative space around subject, warm muted tones, masterpiece, hyperdetailed"
)

COMMON_NEGATIVE = (
    "anime, cartoon, illustration, drawing, character, person, hand, face, "
    "multiple objects, cluttered, busy background, text, watermark, signature, "
    "lowres, jpeg artifacts, oversaturated, neon, cyberpunk"
)


@dataclass(frozen=True)
class Theme:
    name: str
    subject: str           # the focal still-life subject
    duotone: tuple[tuple[int, int, int], tuple[int, int, int]]  # (shadow, highlight) RGB


THEMES: list[Theme] = [
    Theme(
        "cafe_hours",
        subject="iced latte in tall clear glass, cream slowly swirling into espresso, "
                "ice cubes catching warm sunlight, beige linen tablecloth",
        duotone=((40, 22, 8), (244, 222, 184)),       # deep espresso → cream
    ),
    Theme(
        "last_call",
        subject="classic martini cocktail in stemmed glass, single olive, "
                "ice cubes glistening, dark wood bar surface, soft amber backlight",
        duotone=((10, 24, 32), (236, 184, 92)),       # midnight teal → amber
    ),
    Theme(
        "bookshelf_hours",
        subject="open antique hardcover book on a desk, fountain pen resting on the page, "
                "tiny brass paperclip, soft daylight, warm wood tabletop",
        duotone=((28, 28, 16), (210, 188, 132)),      # olive shadow → ochre highlight
    ),
    Theme(
        "tape_deck",
        subject="single cassette tape on a polished surface, exposed tape spool, "
                "reflection of a soft warm light, dust motes",
        duotone=((22, 26, 36), (236, 212, 132)),      # slate → warm yellow
    ),
    Theme(
        "greenhouse",
        subject="single monstera leaf in soft directional sunlight, "
                "pale terracotta pot, gentle morning haze, dewdrops",
        duotone=((26, 38, 28), (224, 224, 196)),      # sage → cream
    ),
]


# ---------- post-process ------------------------------------------------------

def duotone(img: Image.Image, shadow_rgb: tuple[int, int, int],
            highlight_rgb: tuple[int, int, int]) -> Image.Image:
    """Map an image onto a 2-colour gradient (shadow → highlight)."""
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    sh = np.array(shadow_rgb, dtype=np.float32)
    hi = np.array(highlight_rgb, dtype=np.float32)
    out = sh[None, None, :] * (1 - gray[..., None]) + hi[None, None, :] * gray[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def paper_border(img: Image.Image, border_pct: float = 0.06,
                 paper_rgb: tuple[int, int, int] = (236, 226, 200)) -> Image.Image:
    """Wrap the image in a cream paper border with subtle grain."""
    w, h = img.size
    bw = int(min(w, h) * border_pct)
    new_w, new_h = w + 2 * bw, h + 2 * bw
    canvas = Image.new("RGB", (new_w, new_h), paper_rgb)
    # Add subtle paper grain (very low-amplitude noise)
    grain = np.random.normal(0, 6, (new_h, new_w, 3)).astype(np.int16)
    arr = np.array(canvas, dtype=np.int16) + grain
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    canvas.paste(img, (bw, bw))
    # Slight inner shadow on the image edge for tactile feel
    shadow = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(shadow)
    d.rectangle([(0, 0), img.size], outline=80, width=1)
    shadow = shadow.filter(ImageFilter.GaussianBlur(2))
    canvas.paste((30, 22, 14), (bw, bw), shadow)
    return canvas


def grade(raw: Image.Image, theme: Theme) -> Image.Image:
    return paper_border(duotone(raw, *theme.duotone))


# ---------- render -----------------------------------------------------------

def render_theme(pipe, theme: Theme, seed: int = 42) -> Image.Image:
    import torch
    prompt = f"{theme.subject}, {COMMON_QUALITY_SUFFIX}"
    print(f"[{theme.name}] prompt: {prompt[:120]}…")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    img = pipe(
        prompt=prompt,
        negative_prompt=COMMON_NEGATIVE,
        width=1024, height=1024,
        num_inference_steps=28,
        guidance_scale=6.0,
        generator=generator,
    ).images[0]
    return img


def main() -> None:
    selected = sys.argv[1:] or [t.name for t in THEMES]
    themes = [t for t in THEMES if t.name in selected]
    if not themes:
        print(f"unknown theme(s); available: {[t.name for t in THEMES]}")
        sys.exit(1)

    import torch
    from diffusers import StableDiffusionXLPipeline

    print(f"Loading SDXL base ({SDXL_MODEL_ID})…")
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        add_watermarker=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"  loaded in {time.time() - t0:.1f}s")

    for theme in themes:
        t1 = time.time()
        raw = render_theme(pipe, theme)
        graded = grade(raw, theme)
        raw_path = PREVIEW_DIR / f"{theme.name}_raw.png"
        graded_path = PREVIEW_DIR / f"{theme.name}_graded.png"
        raw.save(raw_path)
        graded.save(graded_path)
        print(f"  → {graded_path.name}  ({time.time() - t1:.1f}s)")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nDone. Previews at: {PREVIEW_DIR}/")


if __name__ == "__main__":
    main()
