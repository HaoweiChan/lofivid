"""Render 8 greenhouse-style subjects at YouTube 16:9 aspect.

Locks in the sage→cream duotone + paper border from preview_themes.py
and rotates through 8 still-life subjects that will become the cast for
the actual MV.

Usage:  python scripts/preview_greenhouse_cast.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lofivid.cli import _apply_blackwell_env_defaults  # noqa: E402

_apply_blackwell_env_defaults()

# Reuse the post-process from the theme preview script
from scripts.preview_themes import duotone, paper_border  # noqa: E402

PREVIEW_DIR = ROOT / "previews/greenhouse"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# 16:9 SDXL-native resolution. 1344×768 = 1.75 (close to 1.778) and is the
# trained sweet-spot for SDXL landscape. Any closer to true 16:9 (e.g. 1280×720)
# loses fidelity because SDXL was trained at 1024² + a few aspect buckets.
WIDTH, HEIGHT = 1344, 768

# Sage shadow → cream highlight (matches the chosen greenhouse_graded.png)
SHADOW = (26, 38, 28)
HIGHLIGHT = (224, 224, 196)

COMMON_SUFFIX = (
    "single subject still life, centered composition, isolated object, "
    "vintage photograph, riso print aesthetic, soft directional sunlight, "
    "shallow depth of field, matte texture, minimalist, "
    "negative space, warm muted tones, masterpiece, hyperdetailed"
)
COMMON_NEGATIVE = (
    "anime, cartoon, illustration, drawing, character, person, hand, face, "
    "multiple objects, cluttered, busy background, text, watermark, signature, "
    "lowres, jpeg artifacts, oversaturated, neon, cyberpunk"
)

# 8 subjects — varied silhouettes so consecutive scenes feel different even
# under one duotone. Each prompt is prefix-only; the common suffix appends quality.
SUBJECTS: list[tuple[str, str]] = [
    ("01_monstera",      "single monstera leaf in pale terracotta pot, soft directional sunlight, tabletop"),
    ("02_succulent",     "single small succulent in cream ceramic pot, close-up, soft window light, dewdrops"),
    ("03_watering_can",  "vintage galvanised metal watering can on a wooden bench, water droplets clinging to spout, morning light"),
    ("04_hanging_fern",  "hanging boston fern in clay pot, silhouette against frosted greenhouse window"),
    ("05_lemon",         "single lemon hanging from a branch with two leaves, soft greenhouse haze, dappled sun"),
    ("06_fiddle_leaf",   "fiddle leaf fig sapling in pale terracotta pot, sunlit stripes across leaves"),
    ("07_terrarium",     "small glass dome terrarium with moss and tiny fern inside, polished wood pedestal, soft sunlight"),
    ("08_pothos",        "trailing pothos vine spilling from a hanging brass planter, soft warm window backlight"),
]


def render_one(pipe, name: str, subject: str, seed: int = 17) -> Path:
    import torch
    prompt = f"{subject}, {COMMON_SUFFIX}"
    print(f"[{name}] prompt: {prompt[:120]}…")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    img = pipe(
        prompt=prompt,
        negative_prompt=COMMON_NEGATIVE,
        width=WIDTH, height=HEIGHT,
        num_inference_steps=28,
        guidance_scale=6.0,
        generator=gen,
    ).images[0]
    raw_path = PREVIEW_DIR / f"{name}_raw.png"
    graded_path = PREVIEW_DIR / f"{name}_graded.png"
    img.save(raw_path)
    paper_border(duotone(img, SHADOW, HIGHLIGHT)).save(graded_path)
    return graded_path


def main() -> None:
    import torch
    from diffusers import StableDiffusionXLPipeline

    print("Loading SDXL base…")
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        add_watermarker=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"  loaded in {time.time() - t0:.1f}s")

    for name, subject in SUBJECTS:
        t1 = time.time()
        out = render_one(pipe, name, subject)
        print(f"  → {out.name}  ({time.time() - t1:.1f}s)")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nDone. Cast at: {PREVIEW_DIR}/")


if __name__ == "__main__":
    main()
