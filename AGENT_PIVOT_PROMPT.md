# lofivid — Channel-Direction Pivot

## Context

The repo currently targets anime keyframes (Animagine XL 4) → DepthFlow parallax → ACE-Step instrumental lofi. Channel-direction research has revised the goal: **photographic / minimal-design visuals + jazz/chill-pop music with optional vocals**, modelled after two reference channels (graphic-design minimalism and Unsplash-photo-backed jazz).

This pivot adds new backends *alongside* the existing ones rather than ripping them out. The existing ACE-Step / SDXL / DepthFlow paths remain working; the new backends become the default for the new configs. **Do not delete any working code.** Phase 1 is still a walking skeleton — finish the pivot, then run a smoke render before refactoring anything else.

Read `CLAUDE.md` first if you have not already. The conventions there (Pydantic `extra="forbid"`, ABC backends with stable `name` and `cache_key()`, per-purpose seed derivation, no MoviePy, no torch.compile / flash-attn) are non-negotiable. Match the existing file style: `from __future__ import annotations` at the top of every module, type hints throughout, comments only when *why* is non-obvious.

## Tasks (do them in this order)

### 1. Add `UnsplashKeyframeBackend` (new file: `lofivid/visuals/unsplash.py`)

Subclass `KeyframeBackend` from `lofivid/visuals/base.py`. The contract is the same as `SDXLKeyframeBackend`: take a `KeyframeSpec`, return a `GeneratedImage` written to `output_dir`.

Implementation details:

- Use the Unsplash API (`https://api.unsplash.com/search/photos`). API key from `UNSPLASH_ACCESS_KEY` env var; raise a clear error if missing, do not silently skip.
- The `KeyframeSpec.prompt` is the search query. Don't append the preset's quality suffix to Unsplash queries — quality suffixes like "ultra detailed, 4k" make Unsplash search worse, not better. Strip them in the backend before querying. Take the top result, optionally jitter by `spec.seed % page_size` to vary across scenes deterministically.
- Download the `urls.regular` (1080px) image, save to `output_dir/<scene_index:03d>.jpg`. Resize/crop to `spec.width × spec.height` with PIL using `ImageOps.fit` (centre crop, no distortion).
- Apply the same duotone + paper-border post-process used in `scripts/preview_themes.py`. Refactor `duotone()` and `paper_border()` out of that script into a new module `lofivid/visuals/_grading.py` and import from both places. Don't duplicate the functions.
- The duotone shadow/highlight pair must come from the preset, not be hard-coded. Add a `duotone: tuple[tuple[int,int,int], tuple[int,int,int]] | None` field to `PresetSpec` (default `None` = skip grading). Existing presets keep `duotone=None` so SDXL output is unchanged.
- Cache key contribution: include the resolved Unsplash photo `id` (not just the prompt), so re-runs with the same prompt that pick a different photo invalidate downstream stages correctly.
- Honour Unsplash's attribution requirement: write a sidecar `<scene_index:03d>.jpg.attribution.txt` next to each image with photographer name + photo URL. The compose stage / channel-description generator will pick these up later.

### 2. Replace DepthFlow with `OverlayMotionBackend` (new file: `lofivid/visuals/overlay_motion.py`)

DepthFlow's parallax orbit reads as "AI music video" rather than "lofi" — wrong motion model for the new direction. Add a new `ParallaxBackend` that animates a static keyframe with element-level overlays instead of camera motion.

- Subclass `ParallaxBackend` from `lofivid/visuals/base.py`. Same `name`, `generate`, `warmup`, `shutdown` contract.
- Inputs: still keyframe + duration + fps + resolution. Outputs: an MP4 loop where the keyframe sits stationary and overlays drift on top.
- Three stock overlay options, configurable per scene: `dust_motes` (slow drifting white dots, low opacity), `light_flicker` (gentle vignette intensity oscillation), `slow_zoom` (1.00 → 1.05 over loop duration with ease-in-out, returns to 1.00 to keep loop seamless). Implement all three as pure FFmpeg filter graphs — no new deps.
- The motion type is a `Literal["dust_motes", "light_flicker", "slow_zoom", "none"]` field on the new visuals config (see task 4). Default `slow_zoom` — least intrusive, works with any subject.
- Loop seamlessness matters. The output MP4 must concat back-to-back without a visible jump. For `slow_zoom`, end frame must equal start frame (zoom 1.0 at both t=0 and t=duration). Test by ffprobe-ing the output and visually checking the seam.
- Keep `lofivid/visuals/depthflow.py` as-is. It stays selectable via config for users who want the parallax look.

### 3. Add `SunoMusicBackend` (new file: `lofivid/music/suno.py`)

Subclass `MusicBackend` from `lofivid/music/base.py`. This one is cloud-based, not local — that's a deliberate departure from the all-local architecture and must be flagged in `lofivid licenses` output and the README.

- Use the Suno HTTP API. API key from `SUNO_API_KEY` env var. Raise a clear error if missing.
- Suno doesn't offer a stable public API; you'll likely use one of the third-party wrappers (`sunoapi.org`, `PiAPI`, AIML API). Pick one, document the choice in the module docstring with a note that the wrapper is a legal grey area and the user must verify their own subscription terms allow it. Don't bundle credentials.
- `TrackSpec.prompt` becomes the Suno style prompt. Add an optional `lyrics: str | None` field to `TrackSpec` (default None = instrumental). Update `tracklist.py` to populate `lyrics` from a new `MusicVariation.lyrics` field — for now, simple per-mood lyric fragments the user can override in YAML. If `lyrics is None`, Suno is called in instrumental mode.
- Poll the Suno job until completion (typical 30–60s). Hard timeout at 5 minutes per track — fail loudly, don't hang the whole pipeline.
- Download the resulting WAV/MP3 to `output_dir/<track_index:03d>.{ext}`. Convert to WAV with FFmpeg if Suno only returns MP3, since downstream expects WAV throughout.
- Cache key contribution: include the Suno model version (e.g. `v5.5`) the user pinned. The user must commit to a model version per run — don't auto-upgrade.
- The 2026 Suno copyright situation is unsettled. The README and `lofivid licenses` output must say: *"Suno-generated audio is not eligible for copyright as a fully-AI work. To claim copyright on the music, write your own lyrics and meaningfully edit the stems. Verify your Suno subscription tier permits commercial use before publishing."*

### 4. Update config schema (`lofivid/config.py`)

Two new fields, both with sensible defaults so existing YAMLs still validate.

- `MusicConfig.backend`: extend the `Literal` to include `"suno"`. Existing `"acestep"` and `"musicgen"` values stay.
- `VisualsConfig.keyframe_backend: Literal["sdxl", "unsplash"] = "sdxl"`. Existing configs with no field set keep the SDXL behaviour.
- `VisualsConfig.parallax_backend: Literal["depthflow", "overlay_motion"] = "depthflow"`. Same back-compat reasoning.
- `VisualsConfig.motion_type: Literal["slow_zoom", "dust_motes", "light_flicker", "none"] = "slow_zoom"`. Only consulted when `parallax_backend == "overlay_motion"`.
- `MusicVariation.lyrics: str | None = None`. Only consulted when `MusicConfig.backend == "suno"`.

Wire these through `lofivid/pipeline.py:_make_music_backend` and `_make_visual_backends` so the config selects the right backend implementation.

### 5. Add two new YAML configs

- `configs/jazz_cafe_unsplash.yaml` — Suno jazz with vocals, Unsplash keyframes, overlay-motion `slow_zoom`. 30 minutes target, 6 scenes × 5 min, 6 tracks × 5 min. Search prompts like "cafe interior warm light", "rainy window cafe", "vinyl record close up", "latte art overhead". Light duotone (warm cream → deep brown).
- `configs/minimal_design_lofi.yaml` — ACE-Step instrumental, SDXL keyframes (reusing the greenhouse cast direction from `preview_greenhouse_cast.py`), overlay-motion `dust_motes`. Sage→cream duotone, paper border. 30 minutes target.

Both must validate against the existing `Config` schema with no errors and pass the duration check (`scene_count * scene_seconds ≈ duration_minutes * 60`).

### 6. Update `lofivid licenses` table (`lofivid/cli.py`)

Add rows for Unsplash, Suno (clearly flagged as cloud-based and copyright-uncertain), and the third-party Suno API wrapper if used. Match the existing table format.

### 7. Update README

Add a third runtime mode row to the table at the top of the README:

| Mode | When to use |
|---|---|
| **Cloud-augmented** | Local visuals + Suno API for music (with vocals). Requires `SUNO_API_KEY` and a paid Suno tier permitting commercial use. |

Add a one-paragraph note that the Unsplash photo backend exists for users who want photographic visuals without GPU keyframe generation.

## Conventions to follow

- All new backends must implement `cache_key()` on their spec inputs so re-runs with identical inputs short-circuit. Include backend `name` in the cache key so swapping backends invalidates correctly.
- Per-purpose seeds via `SeedRegistry`. New purposes: `music.suno.{i}`, `visuals.unsplash.{i}`, `visuals.overlay_motion.{i}`. Never reuse `cfg.seed` directly.
- Pydantic `extra="forbid"` on every new config model. Typos in YAML must raise.
- No new heavy deps in `pyproject.toml` — they live in the Dockerfile if they're GPU-related, but Unsplash/Suno are HTTP-only and `requests` is already a transitive dep of HF libs. Reuse it.
- Logging via the standard `log = logging.getLogger(__name__)` pattern at module top, no print statements except in `scripts/`.
- Cloud calls (Unsplash, Suno) need timeouts and retry-with-backoff. A flaky network shouldn't kill a 2-hour render. 3 retries with exponential backoff on transient failures (5xx, timeouts), fail-fast on auth errors (4xx).

## Verification before claiming done

- `pytest -q` passes (existing 33 tests must stay green; add new tests for the schema additions and for the duotone/paper-border refactor).
- `ruff check .` passes.
- `lofivid generate -c configs/minimal_design_lofi.yaml` completes end-to-end. `ffprobe` confirms output duration is within ±1s of target, resolution and fps match config, audio LUFS is in range.
- `lofivid generate -c configs/jazz_cafe_unsplash.yaml` completes end-to-end with `SUNO_API_KEY` set. Same ffprobe checks. If `SUNO_API_KEY` isn't available in your environment, generate at least the visuals and confirm the music stage's failure mode is clean (clear error, no half-written cache entries).
- `lofivid licenses` prints the updated table without errors.

## Out of scope

- Don't redesign the cache layer, the seed registry, or the Pydantic config style. They work.
- Don't add MoviePy, Remotion, or any other heavy media dep. Stick with FFmpeg.
- Don't touch the Blackwell sm_120 env-var workarounds in `cli.py` — they exist for documented reasons.
- Don't auto-upload to YouTube or write the channel-description / thumbnail generator. Those are Phase 3.
- Don't try to make Suno fully local. It isn't.

## Reference: files that already exist and you should read before touching anything

- `CLAUDE.md` — conventions, Phase 1 status, what NOT to redesign
- `lofivid/visuals/base.py` — the ABCs you're subclassing
- `lofivid/music/base.py` — the ABCs you're subclassing
- `lofivid/visuals/keyframes.py` — pattern for a `KeyframeBackend` impl
- `lofivid/visuals/depthflow.py` — pattern for a `ParallaxBackend` impl
- `lofivid/music/acestep.py` — pattern for a `MusicBackend` impl, including the torchaudio shim style
- `scripts/preview_themes.py` — duotone + paper-border functions to refactor out
- `lofivid/pipeline.py` — where backend selection is wired
- `lofivid/config.py` — where the schema lives
