# lofivid

Local, fully open-source AI lofi music-video generator. Composes:

- **Music** — [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) generating ~20 distinct lofi tracks per video, DJ-mixed into a continuous 1–2 hour stream.
- **Visuals** — SDXL/Animagine keyframes → [DepthFlow](https://github.com/BrokenSource/DepthFlow) parallax loops → optional [LTX-Video](https://github.com/Lightricks/LTX-Video) motion details and [Wan 2.2](https://github.com/Wan-Video/Wan2.2) hero scenes.
- **Composition** — FFmpeg (`av1_nvenc`) muxing music, video, rain overlay, vinyl crackle, and EBU R128 loudness normalisation.

Designed for **WSL2 + NVIDIA RTX 5070 Ti** (Blackwell sm_120, 16 GB). Everything runs inside a Docker image because PyTorch stable does not yet support sm_120; we pin a known-working nightly + CUDA 12.8 toolkit.

All shipped components are commercial-use-friendly (Apache 2.0 / MIT / CreativeML Open RAIL++-M). Run `lofivid licenses` before publishing for a per-asset audit.

## Quickstart

```bash
# 1. Verify host (Windows side): NVIDIA driver 572+ installed, WSL2 enabled.
nvidia-smi   # should report your RTX 5070 Ti

# 2. Build the image (~20 min the first time)
docker compose build

# 3. Sanity-check the GPU + NVENC paths inside the container
docker compose run --rm lofivid verify-env

# 4. Generate a 30-second smoke render to prove the pipeline works
docker compose run --rm lofivid generate --config configs/smoke_30sec.yaml

# 5. Generate a full 2-hour anime lofi video
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

See [`PLAN.md`](./PLAN.md) for the full design rationale and a current implementation-status checklist (what's done in Phase 1, what's left for Phases 2–4, and exact handover commands).

## License

This project is Apache 2.0. Bundled and downloaded model weights retain their original licenses; verify with `lofivid licenses` before any commercial use.
