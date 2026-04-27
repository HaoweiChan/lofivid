# AI Lofi Video Generator — Implementation Plan

## Context

You want a repo that produces 1–2 hour lofi music videos end-to-end on local hardware: AI-generated lofi music + AI-generated visuals composited into a single MP4. The intent is **commercial use** (e.g., monetized YouTube), so every model selected must allow it. Hardware is **WSL2 + RTX 5070 Ti (Blackwell, sm_120, 16 GB GDDR7, CUDA 12.8)**, which carries a known platform constraint: as of April 2026, stable PyTorch does **not** support sm_120 — you must use PyTorch nightly cu128 (we'll isolate this in Docker).

Build target: a **single Python CLI** (`lofivid generate --config configs/anime_rainy_window.yaml`) with clean module boundaries so a job queue + auto-uploader can be added later without rewriting the core. Two visual presets ship by default: **anime/Ghibli "lofi girl"** and **photo-realistic cozy scenes**.

No production-grade open-source equivalent exists today; this is novel infrastructure built by composing best-in-class open models.

---

## Implementation Status

**Last updated:** 2026-04-27. Repo layout, all module skeletons, Pydantic config schema, CLI entrypoints, FFmpeg composition logic, and 33 passing unit tests are landed (commits `e9fe59a`, `85737d7`). The pipeline has **not yet been executed end-to-end on real GPU hardware** — the first GPU run is the next agent's first task.

### ✅ Done in Phase 1 (walking skeleton)

| Area | Status | Notes |
|---|---|---|
| Repo skeleton | ✅ | `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `LICENSE` (Apache-2.0), `.gitignore`, `README.md` |
| Blackwell Docker image definition | ✅ | CUDA 12.8 + PyTorch nightly cu128 + FFmpeg built from source with NVENC + Video Codec SDK 13.0. **Not yet built or run.** |
| `lofivid/env.py` | ✅ | sm_120 + NVENC + Python preflight; gates the CLI before model loads |
| `lofivid/cache.py` | ✅ | SQLite content-addressed cache; full unit tests for hit/miss/invalidate |
| `lofivid/seeds.py` | ✅ | Deterministic per-purpose seed derivation (torch + numpy + python rng) |
| `lofivid/cli.py` | ✅ | Typer CLI: `generate`, `music-only`, `visuals-only`, `verify-env`, `licenses` |
| `lofivid/config.py` | ✅ | Pydantic schema with `extra="forbid"`; loads + validates 3 shipped YAML configs |
| `lofivid/music/base.py` | ✅ | `MusicBackend` ABC + `TrackSpec` / `GeneratedTrack` dataclasses |
| `lofivid/music/tracklist.py` | ✅ | Designs N distinct prompts from anchor + variation matrix; verified producing 20 unique tracks for anime config |
| `lofivid/music/acestep.py` | ✅ | ACE-Step 1.5 backend wrapper. **Not yet exercised against real model weights** — call signature is best-effort against v1.5 docs and may need adjustment after first run |
| `lofivid/music/mixer.py` | ✅ | Pure-FFmpeg DJ-mix with chained `acrossfade` + `loudnorm` to -14 LUFS |
| `lofivid/visuals/base.py` | ✅ | `KeyframeBackend` + `ParallaxBackend` ABCs |
| `lofivid/visuals/keyframes.py` | ✅ | SDXL pipeline via diffusers, with LoRA loading + adapter weighting |
| `lofivid/visuals/depthflow.py` | ✅ | DepthFlow CLI wrapper (Python API too unstable across versions) |
| `lofivid/compose/timeline.py` | ✅ | Scene scheduler: distributes N clips over target duration with crossfades |
| `lofivid/compose/ffmpeg_ops.py` | ✅ | `loop_clip_to_duration`, `concat_with_crossfades` (xfade chain + overlay + mux + AV1 encode), `probe_duration_seconds` |
| `lofivid/compose/overlays.py` | ✅ | Path validation for rain video + vinyl crackle |
| `lofivid/presets/{anime,photo}.py` | ✅ | Animagine XL 4 (anime) + SDXL base (photo); both commercial-safe |
| `lofivid/pipeline.py` | ✅ | End-to-end orchestration with cache short-circuiting per stage |
| `configs/*.yaml` | ✅ | `smoke_30sec.yaml`, `anime_rainy_window.yaml` (120 min), `photo_cozy_cafe.yaml` (60 min) |
| Unit tests | ✅ | 33 tests, all green (env, cache, tracklist, mixer math, timeline, config validation) |

### ⏳ Phase 1 leftovers (do these before first GPU run)

| Task | Notes |
|---|---|
| **Build the Docker image** | `docker compose build` — first-time ~20 min; will surface any apt/pip pin drift |
| **Run `verify-env` in container** | `docker compose run --rm lofivid verify-env` — must show `OK` for python/torch/ffmpeg before anything else |
| **Smoke render** | `docker compose run --rm lofivid generate -c configs/smoke_30sec.yaml` — first end-to-end. Likely failure points: ACE-Step API signature drift in `lofivid/music/acestep.py:80`, DepthFlow CLI flags in `lofivid/visuals/depthflow.py:53`, or FFmpeg filter graph syntax in `lofivid/compose/ffmpeg_ops.py:69`. Fix each iteratively. |
| **Bundle CC0 overlay assets** | Drop a rain-on-window MP4 loop into `assets/overlays/rain_window_loop.mp4` and a vinyl-crackle WAV into `assets/audio/vinyl_crackle_loop.wav`. Sources: Pixabay (CC0), Freesound (`DefySolipsis` LO-FI pack, CC0). The `anime_rainy_window.yaml` config references both paths and `overlays.validate()` will refuse to render until they exist. |

### ⏳ Phase 2 — Production-quality outputs

| Task | Notes |
|---|---|
| **Verify both presets render cleanly** | Re-run `anime_rainy_window` and `photo_cozy_cafe` end-to-end after smoke succeeds |
| **Tune scene scheduling** | Current `timeline.schedule()` evenly divides; consider beat-synced cuts (align scene transitions to musical phrases) |
| **Tune mixer crossfade** | `mixer.py` uses fixed `acrossfade=tri:tri`; consider beat-aware overlap (analyse last bar of track N + first bar of track N+1 with `librosa.beat.beat_track`) |
| **`lofivid prune`** | Cache cleanup CLI — listed in plan §"Things to Revisit at Scale" but not yet implemented |
| **Improve `lofivid licenses`** | Inspect actually-downloaded weights at `/models/huggingface` and report their real model-card licenses (current impl is hardcoded) |

### ⏳ Phase 3 — LTX-Video + Wan 2.2

| Task | Notes |
|---|---|
| **`lofivid/visuals/ltx_video.py`** | New `ParallaxBackend` (or new `MotionBackend`) for short ambient motion clips (rain on glass, steam, candle flicker). ~5-sec clips at 720p. |
| **`lofivid/visuals/wan22.py`** | Wan 2.2 hero-shot backend via Wan2GP wrapper. ~7 min per 5-sec clip; gated by `visuals.premium_scenes > 0` in config |
| **Pipeline integration** | `pipeline._do_visuals` should optionally splice LTX clips into long parallax scenes and Wan hero clips at scene transitions |

### ⏳ Phase 4 — Growth seams

| Task | Notes |
|---|---|
| Job queue (file-based or Celery) | Watch `queue/*.yaml`, render in sequence |
| YouTube auto-uploader | `google-api-python-client` + OAuth; thumbnail generator from a cherry-picked keyframe |
| MusicGen backend | Add `lofivid/music/musicgen.py` for non-commercial runs; the `MusicBackend` ABC already supports the swap |
| LLM-driven prompt expander | Auto-generate `variations` lists from a single seed mood — addresses the "mix repetition over many videos" risk |
| Web UI | Probably FastAPI + a thin React frontend over the existing CLI |

### Known risks / things to watch

- **ACE-Step weights license** — code is Apache 2.0, but verify the released model weights' license at install time (the Dockerfile's `pip install git+https://...` pulls code; weights download lazily via `from_pretrained`). The `lofivid licenses` table has a TODO line for this.
- **Animagine XL 4 model ID** — `cagliostrolab/animagine-xl-4.0` was the stable HF repo as of April 2026; if it's renamed, override via the `model_id` field on `AnimePreset.spec()`.
- **DepthFlow CLI flag drift** — DepthFlow's CLI surface has changed between minor versions. Pin in Dockerfile (`pip install "depthflow>=0.8"`); if the smoke render fails inside `depthflow.generate()`, the fix is local to that one method.
- **PyTorch nightly breakage** — nightly cu128 occasionally ships broken builds. If the Docker build fails at PyTorch install, fall back to a known-good date-pinned nightly or to the source build at `bajegani/pytorch-build-blackwell-sm120`.

### Quick handover commands

```bash
cd /home/willy/lofivid

# 1. Confirm tests still green on whatever Python is around
source .venv/bin/activate && pytest -q

# 2. Build the Docker image (first GPU-side step)
docker compose build

# 3. Validate the container can see the GPU
docker compose run --rm lofivid verify-env

# 4. Smoke render — the first thing that exercises every model
docker compose run --rm lofivid generate -c configs/smoke_30sec.yaml

# 5. Full anime preset (after smoke succeeds)
docker compose run --rm lofivid generate -c configs/anime_rainy_window.yaml
```

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  YAML config (style preset, duration, prompts, seeds, output path)  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────┐   ┌────────────────────┐   ┌──────────────────┐
│  Music pipeline  │   │  Visual pipeline   │   │  Asset pipeline  │
│  (ACE-Step 1.5)  │   │  (SDXL → DepthFlow │   │  (rain MP4,      │
│  20 tracks, DJ-  │   │   → LTX-Video)     │   │   vinyl crackle) │
│  mixed → 120 min │   │                    │   │                  │
└──────────────────┘   └────────────────────┘   └──────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Composition (FFmpeg av1_nvenc, concat demuxer, loudnorm -14 LUFS)  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      output/<run_id>.mp4
```

Stages are **independently runnable and cached**. If the music pass succeeds but composition fails, the next run reuses the music. Cache key = hash(config_subset + seed + model_version).

---

## Environment: Docker-First Blackwell Setup

A single `Dockerfile` based on `nvidia/cuda:12.8.0-devel-ubuntu22.04` is the source of truth. Host requirements: Windows NVIDIA driver **572.xx or later** + WSL2 (do **not** install drivers inside WSL). Verify with `nvidia-smi` in WSL.

Inside the container:
- Python 3.11 (3.12 lacks PyTorch nightly binaries)
- `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`
- FFmpeg built with `--enable-nvenc` + Video Codec SDK 13.0 headers (Blackwell NVENC improvements landed Feb 2025)
- Disable `torch.compile` and `flash-attention` (neither supports sm_120 yet); use PyTorch's native `scaled_dot_product_attention`

Verification gate (must pass before any model loads):
```bash
python -c "import torch; assert torch.cuda.get_device_capability() == (12, 0), 'sm_120 not detected'"
```

Reference for the source-build escape hatch if nightly breaks: [bajegani/pytorch-build-blackwell-sm120](https://github.com/bajegani/pytorch-build-blackwell-sm120).

---

## Tool Selection (all commercial-safe — verify weights licenses at install time)

| Stage | Tool | License | VRAM | Notes |
|---|---|---|---|---|
| Music | **ACE-Step 1.5** ([repo](https://github.com/ace-step/ACE-Step-1.5)) | Apache 2.0 (code; verify weights) | ~5 GB | Native lofi/chillhop tags; 4-min chunks; no flash-attn dep |
| Music continuous mode | **ACE-Step-RADIO** fork ([repo](https://github.com/PasiKoodaa/ACE-Step-RADIO)) | Apache 2.0 | — | Reference impl for chunk overlap scheduling |
| Image (anime preset) | **Animagine XL 4** + Ghibli LoRAs | CreativeML Open RAIL++-M (commercial OK with content rules) | ~6 GB | Avoid Illustrious XL (research-only license) |
| Image (photo preset) | **SDXL base 1.0** + RealVisXL/JuggernautXL LoRAs | CreativeML Open RAIL++-M | ~6 GB | FLUX.1 dev is non-commercial — skip; FLUX.1 schnell (Apache 2.0) is fallback if needed |
| Parallax animation | **DepthFlow** ([repo](https://github.com/BrokenSource/DepthFlow)) + Depth Anything v2 | LGPL / Apache 2.0 | ~4 GB | Real-time 2.5D loops from a still image — primary visual workhorse |
| Img→Video (motion clips) | **LTX-Video** ([repo](https://github.com/Lightricks/LTX-Video)) | Apache 2.0 (verify model card) | ~10 GB | 5-sec clips at 720p in seconds; for rain on glass, steam, etc. |
| Img→Video (premium scenes) | **Wan 2.2 via Wan2GP** ([repo](https://github.com/deepbeepmeep/Wan2GP)) | Apache 2.0 | ~16 GB quantized | Use sparingly; ~7 min per 5-sec clip |
| Composition | **FFmpeg + av1_nvenc** + `ffmpeg-python` wrapper | LGPL/GPL | — | Avoid MoviePy 2.x (10× slower than 1.0.3, RAM leaks); avoid Remotion |
| Overlay assets | Pixabay/Pexels CC0 + Freesound CC0 packs | CC0 | — | Rain loops, vinyl crackle, paper textures |

**Known Blackwell gotcha to encode in FFmpeg defaults**: do **not** combine `hevc_nvenc -tune uhq -highbitdepth 1` (artifact bug on RTX 50 series). Use `av1_nvenc -preset p4 -tune hq -cq 28` instead.

---

## Repository Structure

```
lofivid/
├── Dockerfile                    # CUDA 12.8 + PyTorch nightly cu128
├── docker-compose.yml            # GPU passthrough, volume mounts for cache + output
├── pyproject.toml                # uv/pip-installable; entrypoint: `lofivid`
├── README.md                     # quickstart, license matrix, troubleshooting
├── LICENSE                       # Apache 2.0 (matches model deps)
├── configs/
│   ├── anime_rainy_window.yaml   # default anime preset
│   ├── photo_cozy_cafe.yaml      # default photo preset
│   └── smoke_30sec.yaml          # CI-friendly smoke render config
├── lofivid/
│   ├── __init__.py
│   ├── cli.py                    # Click/Typer CLI: `generate`, `music-only`, `visuals-only`, `verify-env`
│   ├── env.py                    # Blackwell verification gate (sm_120 check, NVENC probe)
│   ├── cache.py                  # Content-hash-keyed disk cache (SQLite manifest)
│   ├── seeds.py                  # Centralised RNG; logs every seed used
│   ├── config.py                 # Pydantic schema for the YAML configs
│   ├── pipeline.py               # End-to-end orchestrator
│   ├── music/
│   │   ├── base.py               # MusicBackend ABC (so MusicGen can be swapped in for non-commercial later)
│   │   ├── acestep.py            # Primary backend: generates one full ~5-7 min track per call
│   │   ├── tracklist.py          # Designs N distinct track prompts from anchor + variation matrix
│   │   └── mixer.py              # DJ-mix assembly: ffmpeg acrossfade between tracks (4-8 s)
│   ├── visuals/
│   │   ├── base.py               # VisualBackend ABC
│   │   ├── keyframes.py          # SDXL/Animagine via diffusers
│   │   ├── depthflow.py          # parallax loop generation (primary motion source)
│   │   ├── ltx_video.py          # short ambient motion clips  ⏳ Phase 3
│   │   └── wan22.py              # premium scenes (optional, gated by config flag)  ⏳ Phase 3
│   ├── compose/
│   │   ├── ffmpeg_ops.py         # ffmpeg-python helpers (concat, overlay, loudnorm, mux)
│   │   ├── timeline.py           # plans which scene plays when, with crossfade durations
│   │   └── overlays.py           # rain + vinyl crackle layering
│   └── presets/
│       ├── base.py
│       ├── anime.py              # prompt templates, LoRA weights, motion settings
│       └── photo.py              # ditto for photo-realistic
├── assets/
│   ├── overlays/                 # CC0 rain loops, paper textures  ⏳ drop CC0 files here
│   └── audio/                    # CC0 vinyl crackle stems         ⏳ drop CC0 files here
├── cache/                        # gitignored; intermediate artifacts
├── output/                       # gitignored; final MP4s
└── tests/
    ├── test_env.py               # sm_120 + NVENC gate
    ├── test_cache.py             # hash stability across runs
    ├── test_tracklist.py         # variation sampling, track count math, no duplicate prompts
    ├── test_mixer.py             # crossfade alignment, per-track loudness balancing
    ├── test_timeline.py          # scene scheduling
    ├── test_config.py            # YAML config validation
    └── fixtures/                 # tiny pre-generated clips for fast tests  ⏳ Phase 2
```

---

## Critical Files To Create / Modify

- `Dockerfile` — pins CUDA 12.8, PyTorch nightly cu128, FFmpeg + NVENC, ACE-Step + DepthFlow + diffusers
- `lofivid/env.py` — single source of truth for Blackwell preflight (fail loudly with remediation hint)
- `lofivid/cache.py` — SQLite-backed content-addressed cache; this is what makes 2-hour iteration tolerable
- `lofivid/music/tracklist.py` — generates N distinct track prompts (each a real "song" with intro/outro feel) from a shared anchor + variation matrix
- `lofivid/music/acestep.py` — wraps ACE-Step to render one full ~5–7 min track per call
- `lofivid/music/mixer.py` — DJ-mix assembly: per-track loudness normalisation, beat-aware crossfades, optional vinyl-crackle bridge between tracks
- `lofivid/visuals/depthflow.py` — the workhorse; ~80% of total video runtime is parallax loops
- `lofivid/compose/timeline.py` — declarative scene scheduler (which clip plays from t=X to t=Y, with what overlay)
- `configs/anime_rainy_window.yaml` and `configs/photo_cozy_cafe.yaml` — prove the dual-preset design works

## Reusable Building Blocks Referenced (don't rewrite these)

- `ace-step/ACE-Step-1.5` — installed as a package; we wrap, don't fork
- `BrokenSource/DepthFlow` — Python API used directly
- `Lightricks/LTX-Video` — diffusers-style API
- `deepbeepmeep/Wan2GP` — wrapper over Wan 2.2 with built-in quantization for 16 GB
- `ffmpeg-python` — lazy filter graph builder; spares us subprocess plumbing

---

## Pipeline Stages (in execution order)

1. **Preflight** (`lofivid verify-env`) — assert sm_120 detected, NVENC available, ffmpeg has `av1_nvenc`, all model weights downloaded.
2. **Plan timeline** — read config, decide counts: e.g., 120 min ≈ 20 distinct tracks × ~6 min (real "songs" in a DJ mix), 12 visual scenes × 10 min, each scene = 1 keyframe + 1 parallax loop + optional 1 LTX motion clip. Music tracks are scheduled independently from visual scenes (they don't have to align 1:1).
3. **Music pass** — `tracklist.py` synthesises N unique prompts (shared anchor: "lo-fi, 70–80 BPM, vinyl crackle"; per-track variation: instrumentation, mood, time-of-day vibe). For each track: ACE-Step generates a full ~5–7 min piece → loudness-normalised per-track. `mixer.py` then DJ-mixes them with 4–8 sec beat-aware crossfades into one continuous 120-min stereo file → `cache/<run_id>/music.wav` (final pass loudnormed to −14 LUFS integrated).
4. **Visual pass (parallel where VRAM permits, sequential by default)**:
   - 4a. Generate keyframe images (SDXL, ~30s each)
   - 4b. Run each through DepthFlow → 30-sec parallax MP4 (near real-time)
   - 4c. Optional: LTX-Video motion details for chosen scenes
   - 4d. Optional: Wan 2.2 hero shot if `config.premium_scenes > 0`
5. **Compose** — FFmpeg concat demuxer chains looped/extended scene clips, overlays rain + vinyl crackle, muxes music, encodes with `av1_nvenc -preset p4 -cq 28`, faststart MP4.
6. **Verify** — probe output: duration matches target ±1s, audio LUFS in range, video resolution + framerate match config.

Each stage writes a manifest line (stage, duration, output path, hash) to `cache/<run_id>/manifest.jsonl` for resumability.

---

## Configuration Example (`configs/anime_rainy_window.yaml`)

```yaml
run_id: anime_rainy_window_v1
duration_minutes: 120
output_resolution: [1920, 1080]
fps: 24
seed: 42

music:
  backend: acestep
  track_count: 20                 # ~20 distinct ~6-min "songs" mixed into one 120-min stream
  track_seconds_range: [300, 420] # ACE-Step generates a full track per call
  crossfade_seconds: 6            # DJ-style overlap between tracks
  target_lufs: -14
  anchor:                         # constraints applied to every track for cohesion
    bpm_range: [70, 82]
    key_pool: [D minor, F major, A minor, C major]
    style_tags: [lo-fi, chillhop, vinyl crackle]
  variations:                     # per-track mood/instrumentation rotated through the mix
    - { mood: rainy night,    instruments: [piano, soft pads, light kick] }
    - { mood: sunset cafe,    instruments: [Rhodes, upright bass, brushed drums] }
    - { mood: late study,     instruments: [acoustic guitar, vinyl noise, vibraphone] }
    - { mood: morning haze,   instruments: [Wurlitzer, muted trumpet, shaker] }
    # ... repo ships ~12 templates; tracklist.py samples + permutes to fill track_count

visuals:
  preset: anime               # or "photo"
  scene_count: 12
  scene_seconds: 600          # 12 × 600s = 7200s = 120 min
  parallax_loop_seconds: 30   # each scene = 20 loops of one parallax clip
  premium_scenes: 0           # set 1-2 to invoke Wan 2.2 (slow)
  keyframe_prompt_template: |
    1girl, sitting at desk, rain on window, cozy bedroom, warm lighting,
    bookshelf, steam from mug, soft focus, ghibli style, lofi aesthetic
  loras:
    - { name: ghibli_ponyxl,  weight: 0.8 }
    - { name: lofi_aesthetic, weight: 0.6 }

overlays:
  rain_video: assets/overlays/rain_window_loop.mp4
  rain_opacity: 0.15
  vinyl_crackle: assets/audio/vinyl_crackle_loop.wav
  vinyl_gain_db: -28
```

---

## Verification (end-to-end test plan)

After implementation, the pipeline must pass these checks before being declared working:

1. **Env gate**: `docker compose run lofivid verify-env` exits 0 on a clean machine.
2. **Smoke render** (CI-friendly, ~3 min): `lofivid generate --config tests/fixtures/smoke_30sec.yaml` produces a 30-second MP4 with audio + video + overlays, validated with `ffprobe`.
3. **Music mix quality**: generate 3 distinct tracks via `tracklist.py` + mix them — verify each track has a real musical structure (intro/middle/outro), per-track loudness is balanced, and crossfades land on or near a downbeat (the beat-aware overlap logic in `mixer.py` is the load-bearing piece).
4. **Cache resumability**: run a 5-min job, kill it after the music pass, re-run — second run should skip music generation and reuse the cached `music.wav`.
5. **Both presets**: `lofivid generate --config configs/anime_rainy_window.yaml` and `--config configs/photo_cozy_cafe.yaml` both produce valid output.
6. **Full 2-hour render**: end-to-end on the 5070 Ti. Target: under 90 minutes wall time (most spent in music generation, since DepthFlow is near real-time and LTX clips are seconds each). Output MP4 should be ~2–3 GB at AV1 CQ 28, audio at −14 LUFS ±1.
7. **License audit**: `lofivid licenses` command prints every model + asset used with its license string — sanity-check it's all commercial-safe before publishing.

---

## Phased Rollout (start simple, leave room to grow)

**Phase 1 — Walking skeleton (one-off CLI)** — ✅ code landed, awaiting first GPU run
Dockerfile, env.py preflight, ACE-Step music pipeline, SDXL keyframes + DepthFlow parallax, FFmpeg composition, anime preset only, smoke test. Goal: produce one watchable 10-min video.

**Phase 2 — Both presets + overlays + caching** — ⏳ partial (caching done; overlays + asset bundling remaining)
Photo preset, rain/vinyl overlay layer, content-addressed cache, full 2-hour render path, license audit command.

**Phase 3 — LTX-Video motion details + Wan 2.2 premium scenes** — ⏳ not started
Optional richer motion. Gated by config flags so basic users don't pay the VRAM/time cost.

**Phase 4 — Growth seams (not built yet, but designed for)** — ⏳ not started
Job queue (file-based or Celery), YouTube auto-uploader, web UI, MusicBackend swap to MusicGen for personal-use jobs, A/B prompt rotation for variety.

---

## Things to Revisit at Scale

- **Mix repetition over many videos** — with ~12 variation templates × 4 keys, the prompt space is large but finite. After ~30 videos some moods will repeat; either expand `variations` in configs or add an LLM-driven prompt expander.
- **Track-edge artifacts** — if ACE-Step outputs occasionally fade out mid-phrase, add a per-track silence-trim step in `mixer.py` before crossfade.
- **Storage** — cached intermediates for many videos add up fast; add a `lofivid prune --older-than 7d` once batch use begins.
- **VRAM contention** — current design loads one model at a time. If parallel music + visuals becomes desirable, will need explicit unload/reload orchestration.
- **PyTorch stable sm_120 support** — once it lands (likely PyTorch 2.9+), drop the nightly pin and simplify Dockerfile.
- **Model upgrades** — ACE-Step 2, Wan 2.3, etc. The backend ABCs (`MusicBackend`, `VisualBackend`) exist precisely so model swaps are local changes.
