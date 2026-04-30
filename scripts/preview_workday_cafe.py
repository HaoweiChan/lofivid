"""Cinematic photo + bold display-type lofi previews.

Renders three keyframes that match the visual direction of the
reference channels (think "WORK CAFE JAZZ" / "WORKDAY GROOVE"):

  • cinematic / lifestyle photography (modern cafe, urban aerial,
    work-desk overhead) — no still-life, no riso, no AI illustration cues
  • a constant playlist title overlaid in heavy display serif type, with
    a Traditional-Chinese subtitle line for the Taiwanese vibe, and a
    placeholder tracklist strip at the bottom

Outputs into `previews/<backend>/` (gitignored). Run from the repo root:

    source .venv/bin/activate
    python scripts/preview_workday_cafe.py            # default: SDXL base
    python scripts/preview_workday_cafe.py flux_klein
    python scripts/preview_workday_cafe.py z_image_turbo

See `MODEL_OPTIONS.md` for the model shortlist. Each scene is rendered
at the SDXL-native 1344×768 16:9 bucket so the comparison images are
directly stackable, and saved twice — `*_raw.png` (the bare model
output) and `*_typed.png` (with the typography overlay composed on top).
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lofivid.cli import _apply_blackwell_env_defaults  # noqa: E402

_apply_blackwell_env_defaults()

PREVIEW_ROOT = ROOT / "previews"
PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)

# ---- title block (constant across the preview set) ----------------------
KICKER = "LOFIVID  ・  PLAYLIST"
TITLE_LINE_1 = "WORK CAFE"
TITLE_LINE_2 = "JAZZ"
ZH_TW = "工作日   咖啡   爵士節奏"
TRACKLIST = (
    "01 cafe afternoon   /   02 rainy window   /   03 late night booth   "
    "/   04 vinyl spin   /   05 latte art   /   06 bookshelf hour"
)

# Single warm-rust accent to keep the set cohesive.
ACCENT = (168, 52, 30)
KICKER_COLOUR = (40, 24, 16)
TRACKLIST_COLOUR = (60, 40, 30)

# Font paths — DejaVu and WenQuanYi ship with the typical ubuntu/wsl base
# image. If you swap them for nicer fonts (Playfair Display, Noto Serif TC),
# just point these constants at the new files.
FONT_DISPLAY = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_DISPLAY_ITALIC = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf"
FONT_KICKER = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_BODY = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

# 16:9 SDXL-native landscape bucket
WIDTH, HEIGHT = 1344, 768

QUALITY_SUFFIX = (
    "cinematic photography, soft natural light, shallow depth of field, "
    "neutral colour palette, photorealistic, ultra detailed, kodak film"
)
NEGATIVE = (
    "anime, cartoon, illustration, painting, drawing, riso print, "
    "vintage texture, paper border, duotone, oversaturated, neon, "
    "text, watermark, signature, lowres, blurry, jpeg artifacts"
)


@dataclass(frozen=True)
class Scene:
    name: str
    prompt: str


SCENES: list[Scene] = [
    Scene(
        "01_cafe_interior",
        "open silver MacBook laptop on a light oak wood cafe table, screen "
        "glowing softly, tall clear iced latte glass beside the laptop with "
        "ice cubes and condensation, small ceramic plate with a slice of "
        "matcha cake, blurred white wall background, soft afternoon side "
        "light through a window, lifestyle interior product photography, "
        "tabletop close-up perspective",
    ),
    Scene(
        "02_city_aerial",
        "cinematic aerial drone photograph of a modern city skyline at "
        "golden hour, a long steel cable-stayed bridge crossing a wide "
        "river in the foreground, soft hazy sky with scattered clouds, "
        "neutral natural colour grading, wide cinematic composition, "
        "kodak portra film stock",
    ),
    Scene(
        "03_work_desk",
        "overhead photograph of a clean light oak desk surface, open "
        "hardcover notebook with handwritten notes and a slim brass "
        "fountain pen resting across the page, white ceramic cup of black "
        "coffee, small monstera leaf in a pale terracotta pot, soft natural "
        "side light, minimalist flat-lay composition, lifestyle photography",
    ),
]


# ---- typography pass -----------------------------------------------------

def _load(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def compose_typography(raw: Image.Image) -> Image.Image:
    """Layer kicker / display title / zh-TW / tracklist on top of `raw`.

    Layout sized off image height so the same code works at any resolution.
    Tuned to match the reference channels' proportions: small kicker, a
    medium-weight two-line display title in the upper third (so the photo
    still breathes below), a substantial CJK line as a co-equal element,
    and a thin tracklist strip pinned to the bottom.
    """
    img = raw.copy().convert("RGB")
    w, h = img.size
    base = Image.new("RGB", img.size)
    base.paste(img)
    draw = ImageDraw.Draw(base)

    # Font sizes scale with image height. Title trimmed from 0.18 → 0.115
    # so the photo carries more visual weight, matching ref1/ref2.
    kicker_size = max(int(h * 0.022), 14)
    title_size = max(int(h * 0.115), 60)
    cjk_size = max(int(h * 0.062), 32)
    tracklist_size = max(int(h * 0.022), 14)

    f_kicker = _load(FONT_KICKER, kicker_size)
    f_title = _load(FONT_DISPLAY, title_size)
    f_cjk = _load(FONT_CJK, cjk_size)
    f_tracklist = _load(FONT_BODY, tracklist_size)

    # ---- kicker (centred, top ~6% of canvas) ----
    kw, kh = _measure(draw, KICKER, f_kicker)
    kicker_y = int(h * 0.06)
    draw.text(((w - kw) / 2, kicker_y), KICKER, font=f_kicker, fill=KICKER_COLOUR)

    # ---- main title (two lines, accent colour, modest right-offset on line 2) ----
    tw1, th1 = _measure(draw, TITLE_LINE_1, f_title)
    tw2, th2 = _measure(draw, TITLE_LINE_2, f_title)
    line1_y = kicker_y + kh + int(h * 0.025)
    line1_x = (w - tw1) / 2
    line2_y = line1_y + th1 - int(title_size * 0.08)
    line2_x = (w - tw2) / 2 + int(w * 0.10)   # subtler offset than v1

    # Single soft shadow only — keeps the type flat (no 3D bevel feel).
    soft_shadow = (0, 0, 0, 90)
    for dx, dy in ((1, 1),):
        draw.text((line1_x + dx, line1_y + dy), TITLE_LINE_1, font=f_title, fill=soft_shadow)
        draw.text((line2_x + dx, line2_y + dy), TITLE_LINE_2, font=f_title, fill=soft_shadow)
    draw.text((line1_x, line1_y), TITLE_LINE_1, font=f_title, fill=ACCENT)
    draw.text((line2_x, line2_y), TITLE_LINE_2, font=f_title, fill=ACCENT)

    # ---- zh-TW line (co-equal element, accent-tinted, just below title block) ----
    cw, ch = _measure(draw, ZH_TW, f_cjk)
    cjk_y = line2_y + th2 + int(h * 0.045)
    draw.text(((w - cw) / 2, cjk_y), ZH_TW, font=f_cjk, fill=ACCENT)

    # ---- tracklist strip at the bottom ----
    trw, trh = _measure(draw, TRACKLIST, f_tracklist)
    tr_y = h - int(h * 0.045) - trh
    draw.text(((w - trw) / 2, tr_y), TRACKLIST, font=f_tracklist, fill=TRACKLIST_COLOUR)

    return base


# ---- render ---------------------------------------------------------------

from lofivid.visuals.base import KeyframeBackend, KeyframeSpec  # noqa: E402


def _make_backend(name: str) -> KeyframeBackend:
    """Map CLI shorthand → backend instance. Imports are lazy so e.g. a
    diffusers without FLUX.2 support doesn't block running the SDXL path."""
    if name == "sdxl":
        from lofivid.visuals.keyframes import SDXLKeyframeBackend
        return SDXLKeyframeBackend(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            negative_prompt=NEGATIVE,
        )
    if name == "flux_klein":
        from lofivid.visuals.flux_klein import FluxKleinKeyframeBackend
        return FluxKleinKeyframeBackend()
    if name == "z_image_turbo":
        from lofivid.visuals.z_image import ZImageTurboKeyframeBackend
        return ZImageTurboKeyframeBackend()
    raise SystemExit(f"unknown backend {name!r}; expected one of: sdxl, flux_klein, z_image_turbo")


def render_scene(
    backend: KeyframeBackend, scene: Scene, out_dir: Path, seed: int = 17
) -> tuple[Path, Path]:
    prompt = f"{scene.prompt}, {QUALITY_SUFFIX}"
    print(f"[{scene.name}] {prompt[:120]}…")
    spec = KeyframeSpec(
        scene_index=int(scene.name.split("_", 1)[0]),
        prompt=prompt,
        width=WIDTH, height=HEIGHT,
        seed=seed,
    )
    gen = backend.generate(spec, out_dir)
    raw_path = gen.path.with_name(f"{scene.name}_raw.png")
    typed_path = gen.path.with_name(f"{scene.name}_typed.png")
    gen.path.rename(raw_path)
    with Image.open(raw_path) as raw:
        compose_typography(raw).save(typed_path)
    return raw_path, typed_path


def main() -> None:
    backend_name = sys.argv[1] if len(sys.argv) > 1 else "sdxl"
    out_dir = PREVIEW_ROOT / backend_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {backend_name}")
    t0 = time.time()
    backend = _make_backend(backend_name)
    backend.warmup()
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Between-scene cleanup matters on small-RAM hosts (e.g. WSL2 default 15 GB
    # cap with FLUX.2 Klein in cpu-offload mode — Python's lazy GC otherwise
    # lets activations + matmul intermediates accumulate across scenes and the
    # kernel OOM-kills the process during scene 2's text-encoder pass).
    for scene in SCENES:
        t1 = time.time()
        _, typed = render_scene(backend, scene, out_dir)
        print(f"  → {typed.relative_to(ROOT)}  ({time.time() - t1:.1f}s)")
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    backend.shutdown()
    gc.collect()
    print(f"\nDone. Previews at: {out_dir}/")


if __name__ == "__main__":
    main()
