# lofivid

Local, fully open-source AI lofi music-video generator. Composes:

- **Music** — pluggable backends: ACE-Step 1.5 (local), Suno API (cloud, vocals), and a **library** backend for pre-licensed audio (e.g. Epidemic Sound). DJ-mixed into a continuous 1–2 hour stream with EBU R128 loudness normalisation.
- **Visuals** — SDXL / Animagine / FLUX.2 Klein / Z-Image-Turbo keyframes → DepthFlow parallax loops or pure-FFmpeg overlay-motion (slow zoom / dust motes / light flicker).
- **Composition** — FFmpeg muxing, with **persistent brand text overlays**, a **per-track now-playing HUD**, and an **audio-driven waveform band** baked into every frame.

Designed for **WSL2 + NVIDIA RTX 5070 Ti** (Blackwell sm_120, 16 GB). PyTorch stable does not yet support sm_120, so we pin a known-working nightly + CUDA 12.8 toolkit. Two runtime modes are supported:

| Mode | When to use |
|---|---|
| **Host-mode** (`.venv`) | Docker Desktop not running; natively installed toolchain. Uses libx264 software encode. |
| **Docker** (`docker compose`) | Canonical setup. Uses `av1_nvenc` hardware encode, Python 3.11, NVENC-tuned FFmpeg. |

The default local components are commercial-use-friendly (Apache 2.0 / MIT / CreativeML Open RAIL++-M). The cloud-augmented Suno path is a *legal grey area* — Suno has no official public API, fully-AI music is not US-copyrightable, and the third-party API wrappers have their own ToS to verify. Run `lofivid licenses` before publishing for a per-asset audit.

The **Unsplash photo backend** (`keyframe_backend: unsplash` in the style file) pulls top search hits from the Unsplash API, applies the configured duotone + paper-border post-process, and writes per-image attribution sidecars. Requires `UNSPLASH_ACCESS_KEY`.

## Quickstart — host-mode (no Docker)

```bash
# 1. Activate the venv (PyTorch nightly cu128 + ACE-Step + DepthFlow)
source .venv/bin/activate

# 2. Sanity-check GPU + encoder
lofivid verify-env

# 3. Smoke render — fastest end-to-end test (no library, no fonts beyond
#    what the smoke style references — and the smoke style references none
#    because brand layers / HUD are disabled).
lofivid generate --config configs/smoke_30sec.yaml

# 4. The default production config — 30 minutes of cafe jazz with full
#    brand overlay, HUD, and waveform. Drop pre-licensed WAVs into
#    assets/music/cafe_jazz/<mood>/ first (see assets/music/README.md).
lofivid generate --config configs/2026-04-30_morning_cafe_30min.yaml
```

## Quickstart — Docker (canonical)

```bash
nvidia-smi
docker compose build
docker compose run --rm lofivid verify-env
docker compose run --rm lofivid generate --config configs/smoke_30sec.yaml
docker compose run --rm lofivid generate --config configs/2026-04-30_morning_cafe_30min.yaml
```

Output lands in `./output/<run_id>.mp4`. Intermediate artefacts (per-track audio, keyframe images, parallax loops, brand / HUD PNGs) live under `./cache/<run_id>/` and are reused on re-runs.

## Style templates

A *style* is a reusable channel-brand definition (visual prompts, music anchor, motion type, palette, typography, HUD/waveform settings). A *run config* is a thin instance that references a style by name and only sets per-render parameters (duration, seed, scene_count). Editing a style file invalidates only that style's caches; two runs that share a style produce videos with the same look.

```yaml
# configs/2026-04-30_morning_cafe_30min.yaml
run_id: morning_cafe_30min_v1
style_ref: morning_cafe          # → loads styles/morning_cafe.yaml
duration_minutes: 30
output_resolution: [1920, 1080]
fps: 24
seed: 42

music:
  track_count: 6
  track_seconds_range: [280, 320]
  crossfade_seconds: 6
  target_lufs: -14

visuals:
  scene_count: 6
  scene_seconds: 300
  parallax_loop_seconds: 30
  premium_scenes: 0
```

The style file holds everything else: backends, prompts, palette, brand layers, HUD, waveform. See `styles/morning_cafe.yaml` for the reference layout.

`style_ref` is **mandatory**. There is no inline-style fallback; missing or typo'd style refs raise loudly at config-load time.

## Project structure

```
lofivid/
├── lofivid/
│   ├── env.py             # Blackwell sm_120 + NVENC + font preflight
│   ├── cache.py           # SQLite-backed content-addressed cache
│   ├── seeds.py           # central RNG; logs every seed used
│   ├── cli.py             # `lofivid generate | music-only | visuals-only | verify-env | licenses`
│   ├── config.py          # per-run YAML schema (style_ref + MusicInstance + VisualsInstance)
│   ├── styles/            # StyleSpec + loader (the channel brand)
│   ├── music/             # ACE-Step / Suno / Library backends + registry + tracklist + mixer
│   ├── visuals/           # SDXL / FLUX.2 / Z-Image / Unsplash + DepthFlow / overlay-motion + registry
│   ├── compose/           # FFmpeg ops, brand layer, HUD, waveform, scene timeline
│   └── presets/           # anime / photo prompt templates
├── styles/                # bundled channel-brand definitions (_smoke, morning_cafe)
├── configs/               # per-video run YAMLs
├── assets/                # CC0 rain loops, vinyl crackle stems, OFL fonts, music library
└── tests/                 # full schema + registry + mixer + compose unit tests
```

See `CLAUDE.md` for current implementation status and development conventions.

## License

This project is Apache 2.0. Bundled and downloaded model weights retain their original licenses; verify with `lofivid licenses` before any commercial use.
