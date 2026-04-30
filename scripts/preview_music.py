"""Short music samples to align style direction (analogue of preview_workday_cafe.py).

Renders one short clip per mood **variation** so we can compare audio
directions side-by-side without committing to a full 30-min render. The
anchor (BPM range, key pool, style tags) is locked — variety lives in
the variation list, exactly like ``lofivid/music/tracklist.py`` does for
the production pipeline. That keeps the preview honest: what you hear
is what the production tracklist designer will actually generate.

Two backends:

* ``acestep`` (default) — local, instrumental, Apache-2.0. Matches the
  ``minimal_design_lofi.yaml`` flow.
* ``suno`` — cloud, vocals supported. Requires ``SUNO_API_KEY``.
  Matches the ``jazz_cafe_unsplash.yaml`` flow.

Defaults to the **cafe jazz** direction (78–92 BPM, F/Bb/Eb major / G
minor) so the previews align with the FLUX.2 Klein / Z-Image-Turbo
visual previews already in ``previews/{flux_klein,z_image_turbo}/``.
Override with ``--direction <name>`` to try other anchors as we
diversify.

Outputs::

    previews/music/<backend>/<direction>/<NN>_<mood_slug>.wav
    previews/music/<backend>/<direction>/_index.txt    # mood ↔ filename map

Run::

    source .venv/bin/activate
    python scripts/preview_music.py                     # acestep, cafe_jazz
    python scripts/preview_music.py acestep cafe_jazz
    python scripts/preview_music.py acestep greenhouse
    python scripts/preview_music.py suno cafe_jazz      # needs SUNO_API_KEY
"""

from __future__ import annotations

import gc
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lofivid.cli import _apply_blackwell_env_defaults  # noqa: E402

_apply_blackwell_env_defaults()

PREVIEW_ROOT = ROOT / "previews" / "music"
PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)

# ---- preview budget ------------------------------------------------------
# Short enough that 4–6 variations finish in a few minutes on a 5070 Ti
# (ACE-Step is ~1.5× realtime), long enough to hear the groove + a chorus.
CLIP_SECONDS = 45
INFER_STEP = 60       # ACE-Step default; lower sacrifices quality
GUIDANCE_SCALE = 15.0


# ---- direction registry --------------------------------------------------
# Matches the production configs but trimmed to a 4-mood grid. Each direction
# is a (anchor, variations) pair you'd drop into a config's `music:` block.

@dataclass(frozen=True)
class Anchor:
    bpm_range: tuple[int, int]
    key_pool: tuple[str, ...]
    style_tags: tuple[str, ...]


@dataclass(frozen=True)
class Variation:
    mood: str
    instruments: tuple[str, ...]
    lyrics: str | None = None       # only used by Suno; ignored by ACE-Step


@dataclass(frozen=True)
class Direction:
    name: str
    anchor: Anchor
    variations: tuple[Variation, ...]


DIRECTIONS: dict[str, Direction] = {
    # Locked-in style — 4 close cousins of the user-approved
    # "greenhouse morning" recipe (piano + soft pads + light kick at
    # ~73 BPM in D minor, with the subtle funk vibe ACE-Step produced
    # from that exact tag set). Mood words vary, instrument bedrock
    # stays; related keys for harmonic variety without leaving the lane.
    "morning": Direction(
        name="morning",
        anchor=Anchor(
            bpm_range=(70, 76),
            key_pool=("D minor", "F major", "A minor", "C major"),
            style_tags=("lo-fi", "instrumental", "jazzy", "mellow", "vinyl crackle", "soft"),
        ),
        variations=(
            # Carbon copy of the user-approved prompt — different seed will
            # give a different take of the same character.
            Variation(mood="greenhouse morning", instruments=("piano", "soft pads", "light kick")),
            # Same recipe, different mood word.
            Variation(mood="morning haze",       instruments=("piano", "soft pads", "light kick")),
            # Rhodes swap for a slightly warmer electric-piano variant.
            Variation(mood="early morning glow", instruments=("Rhodes", "soft pads", "light kick")),
            # Same recipe again — pure seed variation, third take of the lane.
            Variation(mood="morning calm",       instruments=("piano", "soft pads", "light kick")),
        ),
    ),
    # Aligned with the FLUX cafe/city/desk visual previews. Mirrors
    # configs/jazz_cafe_unsplash.yaml's anchor + a 4-variation slice.
    "cafe_jazz": Direction(
        name="cafe_jazz",
        anchor=Anchor(
            bpm_range=(78, 92),
            key_pool=("F major", "Bb major", "Eb major", "G minor"),
            style_tags=("jazz", "vocal jazz", "smooth jazz", "lo-fi", "vinyl crackle"),
        ),
        variations=(
            Variation(
                mood="cafe afternoon",
                instruments=("Rhodes", "upright bass", "brushed drums"),
                lyrics=(
                    "Soft afternoon light is pouring in / Cup of coffee growing thin"
                ),
            ),
            Variation(
                mood="rainy window",
                instruments=("piano", "double bass", "soft kick"),
                lyrics=(
                    "Watch the raindrops chase each other down / Tiny rivers crossing town"
                ),
            ),
            Variation(
                mood="late night booth",
                instruments=("muted trumpet", "walking bass", "hihat"),
                lyrics=(
                    "Last call but we're still talking / Streetlights through the window walking"
                ),
            ),
            Variation(
                mood="vinyl spin",
                instruments=("vibraphone", "acoustic guitar", "shaker"),
                lyrics=(
                    "Needle dropping on a memory / Crackle, hum, and gently"
                ),
            ),
        ),
    ),
    # Jazzy instrumental lofi — back to the v1 anchor that the user liked
    # (rejected the "modern lofi / chillhop / tape saturation" retune as
    # glitchier and more AI-feeling, with chillhop tags steering ACE-Step
    # toward chopped vocal stabs). Only change vs strict v1: muted trumpet
    # → soft saxophone, since the user wants brass present overall but not
    # the bright trumpet character carrying a whole variation.
    "greenhouse": Direction(
        name="greenhouse",
        anchor=Anchor(
            bpm_range=(70, 82),
            key_pool=("D minor", "F major", "A minor", "C major"),
            style_tags=("lo-fi", "instrumental", "jazzy", "mellow", "vinyl crackle", "soft"),
        ),
        variations=(
            Variation(mood="greenhouse morning", instruments=("piano", "soft pads", "light kick")),
            Variation(mood="terracotta haze",    instruments=("Rhodes", "upright bass", "brushed drums")),
            Variation(mood="paper still life",   instruments=("acoustic guitar", "vibraphone", "shaker")),
            Variation(mood="window dust",        instruments=("Wurlitzer", "soft saxophone", "hihat")),
        ),
    ),
}


# ---- prompt composition --------------------------------------------------
# Mirrors `TrackPlan.to_prompt()` from lofivid/music/tracklist.py so what we
# preview here is the same prompt shape the pipeline will use in production.

def _compose_prompt(anchor: Anchor, var: Variation, bpm: int, key: str) -> str:
    parts: list[str] = list(anchor.style_tags)
    parts.append(var.mood)
    parts.extend(var.instruments)
    parts.extend([f"{bpm} BPM", f"key of {key}", "stereo", "vinyl crackle"])
    seen: set[str] = set()
    return ", ".join(p for p in parts if not (p in seen or seen.add(p)))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# ---- backend wiring ------------------------------------------------------

from lofivid.music.base import MusicBackend, TrackSpec  # noqa: E402


def _make_backend(name: str) -> MusicBackend:
    if name == "acestep":
        from lofivid.music.acestep import ACEStepBackend
        return ACEStepBackend(
            dtype="bfloat16",
            infer_step=INFER_STEP,
            guidance_scale=GUIDANCE_SCALE,
        )
    if name == "suno":
        from lofivid.music.suno import SunoMusicBackend
        return SunoMusicBackend()      # reads SUNO_API_KEY from env
    raise SystemExit(f"unknown backend {name!r}; expected one of: acestep, suno")


# ---- render --------------------------------------------------------------

def render_one(
    backend: MusicBackend,
    var: Variation,
    bpm: int,
    key: str,
    track_index: int,
    seed: int,
    out_dir: Path,
) -> Path:
    prompt = _compose_prompt(var=var, anchor=current_anchor, bpm=bpm, key=key)
    print(f"[{var.mood}] {prompt[:120]}…")
    spec = TrackSpec(
        track_index=track_index,
        prompt=prompt,
        bpm=bpm,
        key=key,
        duration_seconds=CLIP_SECONDS,
        seed=seed,
        # ACE-Step ignores lyrics (instrumental only); Suno consumes them.
        lyrics=var.lyrics if backend.name.startswith("suno") else None,
    )
    track = backend.generate(spec, out_dir)
    final = out_dir / f"{track_index:02d}_{_slug(var.mood)}.wav"
    track.path.rename(final)
    return final


# Module-level handle so `render_one` can compose the prompt without us
# threading the anchor through every helper. Set in `main`.
current_anchor: Anchor


def main() -> None:
    global current_anchor

    backend_name = sys.argv[1] if len(sys.argv) > 1 else "acestep"
    direction_name = sys.argv[2] if len(sys.argv) > 2 else "cafe_jazz"

    if direction_name not in DIRECTIONS:
        raise SystemExit(
            f"unknown direction {direction_name!r}; "
            f"available: {sorted(DIRECTIONS)}"
        )
    direction = DIRECTIONS[direction_name]
    current_anchor = direction.anchor

    out_dir = PREVIEW_ROOT / backend_name / direction_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend:   {backend_name}")
    print(f"Direction: {direction_name}")
    print(f"Anchor:    {direction.anchor}")
    print(f"Out dir:   {out_dir}")

    t0 = time.time()
    backend = _make_backend(backend_name)
    backend.warmup()
    print(f"  loaded in {time.time() - t0:.1f}s\n")

    # Deterministic per-track BPM/key pick — round-robin through the key
    # pool, BPM = midpoint+/-2 to keep the preview audibly varied without
    # going random. Seeds are derived from the track index so re-running
    # is reproducible without dragging in the SeedRegistry machinery.
    bpm_lo, bpm_hi = direction.anchor.bpm_range
    bpm_mid = (bpm_lo + bpm_hi) // 2
    bpm_jitter = (-3, -1, 1, 3)

    index_path = out_dir / "_index.txt"
    with index_path.open("w") as idx:
        idx.write(f"# music previews — backend={backend_name} direction={direction_name}\n")
        idx.write(f"# anchor: bpm_range={direction.anchor.bpm_range} keys={direction.anchor.key_pool}\n")
        idx.write(f"# style:  {', '.join(direction.anchor.style_tags)}\n\n")

        for i, var in enumerate(direction.variations):
            bpm = max(bpm_lo, min(bpm_hi, bpm_mid + bpm_jitter[i % len(bpm_jitter)]))
            key = direction.anchor.key_pool[i % len(direction.anchor.key_pool)]
            seed = 1000 + i

            t1 = time.time()
            try:
                final = render_one(
                    backend=backend,
                    var=var,
                    bpm=bpm,
                    key=key,
                    track_index=i,
                    seed=seed,
                    out_dir=out_dir,
                )
                idx.write(f"{final.name}\t{var.mood}\t{bpm} BPM\t{key}\t{','.join(var.instruments)}\n")
                print(f"  → {final.relative_to(ROOT)}  ({time.time() - t1:.1f}s)")
            except Exception as e:
                idx.write(f"FAILED\t{var.mood}\t{bpm} BPM\t{key}\t{e}\n")
                print(f"  FAILED for {var.mood}: {e}")
            # Free intermediates between tracks — same defensive pattern as
            # preview_workday_cafe.py for the visual side.
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    backend.shutdown()
    gc.collect()
    print(f"\nDone. Music previews at: {out_dir}/")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
