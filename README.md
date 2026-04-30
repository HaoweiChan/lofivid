# lofivid

Local, fully open-source AI lofi music-video generator. Composes:

- **Music** — [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) generating ~20 distinct lofi tracks per video, DJ-mixed into a continuous 1–2 hour stream.
- **Visuals** — SDXL/Animagine keyframes → [DepthFlow](https://github.com/BrokenSource/DepthFlow) parallax loops → optional [LTX-Video](https://github.com/Lightricks/LTX-Video) motion details and [Wan 2.2](https://github.com/Wan-Video/Wan2.2) hero scenes.
- **Composition** — FFmpeg (`av1_nvenc`) muxing music, video, rain overlay, vinyl crackle, and EBU R128 loudness normalisation.

Designed for **WSL2 + NVIDIA RTX 5070 Ti** (Blackwell sm_120, 16 GB). PyTorch stable does not yet support sm_120, so we pin a known-working nightly + CUDA 12.8 toolkit. Three runtime modes are supported:

| Mode | When to use |
|---|---|
| **Host-mode** (`.venv`) | Docker Desktop not running; natively installed toolchain. Uses libx264 software encode. |
| **Docker** (`docker compose`) | Canonical setup. Uses `av1_nvenc` hardware encode, Python 3.11, NVENC-tuned FFmpeg. |
| **Cloud-augmented** | Local visuals + Suno API for music (with vocals). Requires `SUNO_API_KEY` and a paid Suno tier permitting commercial use. See `configs/jazz_cafe_unsplash.yaml`. |

The default local components are commercial-use-friendly (Apache 2.0 / MIT / CreativeML Open RAIL++-M). The cloud-augmented Suno path is a *legal grey area* — Suno has no official public API, fully-AI music is not US-copyrightable, and the third-party API wrappers (`sunoapi.org`, PiAPI, AIML) have their own ToS to verify. Run `lofivid licenses` before publishing for a per-asset audit.

The **Unsplash photo backend** (`keyframe_backend: unsplash` in YAML) is available for users who want photographic visuals without GPU keyframe generation — pulls top hits from the Unsplash search API, applies the configured duotone + paper-border post-process, and writes per-image attribution sidecars. Requires `UNSPLASH_ACCESS_KEY`.

## Quickstart — host-mode (no Docker)

```bash
# 1. Activate the venv (PyTorch nightly cu128 + ACE-Step + DepthFlow)
source .venv/bin/activate

# 2. Sanity-check GPU + encoder
lofivid verify-env

# 3. 5-minute demo render (fastest end-to-end test)
lofivid generate --config configs/demo_5min_anime.yaml

# 4. Scale up as desired
lofivid generate --config configs/medium_30min_anime.yaml
lofivid generate --config configs/anime_rainy_window.yaml
```

## Quickstart — Docker (canonical)

```bash
# 1. Verify host: NVIDIA driver 572+ installed, Docker Desktop running, WSL2 enabled.
nvidia-smi

# 2. Build the image (~20 min the first time)
docker compose build

# 3. Sanity-check GPU + NVENC inside the container
docker compose run --rm lofivid verify-env

# 4. Smoke render
docker compose run --rm lofivid generate --config configs/smoke_30sec.yaml

# 5. Full 2-hour anime lofi video
docker compose run --rm lofivid generate --config configs/anime_rainy_window.yaml
```

Output lands in `./output/<run_id>.mp4`. Intermediate artifacts (per-track audio, keyframe PNGs, parallax MP4s) live under `./cache/<run_id>/` and are reused on re-runs.

## Project structure

```
lofivid/
├── lofivid/
│   ├── env.py             # Blackwell sm_120 + NVENC preflight gate
│   ├── cache.py           # SQLite-backed content-addressed cache
│   ├── seeds.py           # central RNG; logs every seed used
│   ├── cli.py             # `lofivid generate | music-only | visuals-only | verify-env`
│   ├── music/             # ACE-Step backend + tracklist designer + DJ mixer
│   ├── visuals/           # SDXL keyframes + DepthFlow parallax + LTX/Wan extras
│   ├── compose/           # FFmpeg ops, timeline scheduler, overlay layering
│   └── presets/           # anime / photo prompt templates
├── configs/               # per-video YAML configs
├── assets/                # CC0 rain loops, vinyl crackle stems
└── tests/                 # env/cache/tracklist/mixer + 30-sec smoke render
```

See `CLAUDE.md` for current implementation status and development conventions.

## License

This project is Apache 2.0. Bundled and downloaded model weights retain their original licenses; verify with `lofivid licenses` before any commercial use.
