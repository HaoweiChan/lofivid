# CLAUDE.md — lofivid

Project memory for Claude Code. Keep this file authoritative for repo-specific conventions; broader user-level rules live in `~/.claude/`.

## What this repo is

Local AI lofi music-video generator. A single Python CLI (`lofivid`) composes AI-generated lofi music + AI-generated visuals into 1–2 hour MP4s. Designed for **commercial use** (e.g., monetised YouTube), so every shipped model must allow it.

Target hardware: **WSL2 + RTX 5070 Ti** (Blackwell sm_120, 16 GB GDDR7, CUDA 12.8). Stable PyTorch does **not** support sm_120 yet — use PyTorch nightly cu128, either inside Docker or natively in `.venv/`.

Design rationale, phased rollout, and license matrix live in `PLAN.md` (local-only, not in git). **Read it before making non-trivial changes** — it documents *why* specific tools were chosen and lists known gotchas (e.g. NVENC artifact bug on RTX 50 series).

## Current status (as of 2026-04-28)

- Phase 1 walking skeleton **landed**: full module skeleton, Pydantic config, CLI, FFmpeg composition, 33 passing unit tests.
- **Host-mode pivot** (2026-04-28): Docker Desktop wasn't running. Toolchain installed natively into `.venv/` — PyTorch nightly cu128, ACE-Step via `pip install ace-step`, imageio-ffmpeg static binary. GPU work (ACE-Step, SDXL, DepthFlow) unchanged. Encode switches to `libx264 -crf 20` (software; ~2× realtime on 24 cores, visually equivalent). Docker path intact and auto-detected when the daemon is up.
- **First GPU smoke render has not happened yet.** Next task: `source .venv/bin/activate && lofivid verify-env` → `lofivid generate -c configs/demo_5min_anime.yaml`.
- Likely first-failure points: ACE-Step API signature in `lofivid/music/acestep.py`, DepthFlow CLI flags in `lofivid/visuals/depthflow.py`, FFmpeg filter graph in `lofivid/compose/ffmpeg_ops.py`. Fix locally; don't redesign.
- CC0 overlay assets not yet bundled — `assets/overlays/rain_window_loop.mp4` and `assets/audio/vinyl_crackle_loop.wav` must be dropped in before any config that references them will render.

## Repo layout (essentials)

```
lofivid/
├── Dockerfile, docker-compose.yml   # CUDA 12.8 + PyTorch nightly cu128 + FFmpeg w/ NVENC
├── pyproject.toml                   # entrypoint: `lofivid` (typer)
├── PLAN.md                          # design doc — local-only, not in git
├── configs/                         # smoke_30sec, anime_rainy_window, photo_cozy_cafe
├── lofivid/
│   ├── env.py        # Blackwell sm_120 + NVENC preflight (assert_ready before model loads)
│   ├── cache.py      # SQLite content-addressed cache; key = hash(config_subset + seed + model_version)
│   ├── seeds.py      # SeedRegistry — deterministic per-purpose seed derivation
│   ├── cli.py        # `generate`, `music-only`, `visuals-only`, `verify-env`, `licenses`
│   ├── config.py     # Pydantic schema, extra="forbid" (typos in YAML raise loudly)
│   ├── pipeline.py   # End-to-end orchestrator with per-stage cache short-circuit
│   ├── music/        # ACE-Step backend + tracklist designer + DJ mixer (FFmpeg acrossfade)
│   ├── visuals/      # SDXL/Animagine keyframes → DepthFlow parallax (LTX/Wan = Phase 3)
│   ├── compose/      # FFmpeg ops, scene scheduler, overlay layering
│   └── presets/      # anime (Animagine XL 4) + photo (SDXL base) — both commercial-safe
├── assets/                          # CC0 rain loops, vinyl crackle (drop CC0 files here)
├── cache/                           # gitignored; intermediate artifacts
├── output/                          # gitignored; final MP4s
└── tests/                           # 33 tests, all green; run with `pytest -q`
```

## Key conventions

- **`extra="forbid"` everywhere on Pydantic configs.** Silent typos in YAML are unacceptable.
- **Backends are ABC-based** (`MusicBackend`, `KeyframeBackend`, `ParallaxBackend`). Adding ACE-Step 2, MusicGen, Wan 2.2 etc. should be a new file under `lofivid/{music,visuals}/`, not edits to existing backends.
- **Every pipeline stage is cached.** Cache key = `content_hash({backend_name, **spec.cache_key()})`. If you add a backend, give it a stable `name` and `spec.cache_key()` — otherwise re-runs will re-render unnecessarily.
- **Seeds are derived, not raw.** Use `SeedRegistry(cfg.seed).derive("purpose.path.N")` — never reuse `cfg.seed` directly. This is what makes runs reproducible across stages.
- **FFmpeg defaults are NVENC-tuned for Blackwell.** Use `av1_nvenc -preset p4 -tune hq -cq 28`. Do **not** combine `hevc_nvenc -tune uhq -highbitdepth 1` (artifact bug on RTX 50 series — documented in PLAN.md).
- **Disable `torch.compile` and flash-attention.** Neither supports sm_120; `Dockerfile` sets `TORCHDYNAMO_DISABLE=1` and `PYTORCH_DISABLE_FLASH_ATTENTION=1`. Use PyTorch's native `scaled_dot_product_attention`.
- **Commercial-safe weights only.** Avoid Illustrious XL (research-only) and FLUX.1 dev (non-commercial). FLUX.1 schnell is the Apache-2.0 fallback if SDXL output ever becomes a problem.
- **`lofivid licenses` must stay accurate.** If you add or swap a model, update the table in `cli.py`.

## Common commands

```bash
# Tests (host-side, no GPU needed)
source .venv/bin/activate && pytest -q

# Lint
ruff check .

# Build the GPU image (first time ~20 min)
docker compose build

# Preflight inside the container
docker compose run --rm lofivid verify-env

# Smoke render — first thing that exercises every model
docker compose run --rm lofivid generate -c configs/smoke_30sec.yaml

# Full anime preset (after smoke succeeds)
docker compose run --rm lofivid generate -c configs/anime_rainy_window.yaml
```

## Ground rules for code changes

- **Don't redesign Phase 1.** It's a deliberate walking skeleton; finish the GPU validation loop before refactoring.
- **Don't pin heavy GPU deps in `pyproject.toml`.** They live in the Dockerfile so the host stays clean. The host-side install is just CLI + config validation.
- **Don't add MoviePy.** 2.x is 10× slower than 1.0.3 and leaks RAM; we use `ffmpeg-python` for a reason.
- **Don't add new long-lived branches.** Solo-dev workflow: feature branches rebased onto `main`, merged with `--no-ff`. See `~/.claude/rules/git-workflow.md`.
- **Don't claim work is done without verification.** For pipeline changes: smoke render must complete and `ffprobe` must confirm duration ± 1s, audio LUFS in range, video resolution + framerate match config. For non-pipeline changes: `pytest -q` must be green.
- **Match the existing style.** Pydantic models are strict, dataclasses for value objects, `from __future__ import annotations` at the top of every module, type hints throughout, comments only when *why* is non-obvious.

## When in doubt

Re-read `PLAN.md`. Specifically:
- §"Tool Selection" — license + VRAM matrix, what NOT to use and why
- §"Implementation Status" — what's done, what's next, exact handover commands
- §"Known risks / things to watch" — ACE-Step weights license, Animagine model ID drift, DepthFlow CLI drift, PyTorch nightly breakage

For unfamiliar SDK calls (ACE-Step, DepthFlow, diffusers): check the upstream docs/repo first — code in `music/acestep.py` and `visuals/depthflow.py` is described as "best-effort against v1.5/v0.8 docs and may need adjustment after first run." Treat their call signatures as suspect until exercised.
