"""End-to-end pipeline orchestration.

Stages:
  1. Load + validate config
  2. Music: design tracklist → generate per-track WAVs → DJ-mix into one stereo file
  3. Visuals: generate keyframes → DepthFlow parallax loops
  4. Compose: schedule clips → loop+xfade → overlay → mux audio → encode AV1

Each stage uses the cache so re-runs short-circuit unchanged steps.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from lofivid.cache import Cache, content_hash
from lofivid.compose import ffmpeg_ops, overlays
from lofivid.compose.ffmpeg_ops import EncodeSettings
from lofivid.compose.timeline import schedule
from lofivid.config import Config, load
from lofivid.music.acestep import ACEStepBackend
from lofivid.music.base import GeneratedTrack, MusicBackend
from lofivid.music.mixer import MixSettings, mix_tracks
from lofivid.music.tracklist import design_tracklist, plans_to_specs
from lofivid.presets import get_preset
from lofivid.seeds import SeedRegistry
from lofivid.visuals.base import (
    GeneratedClip,
    KeyframeBackend,
    KeyframeSpec,
    ParallaxBackend,
    ParallaxSpec,
)
from lofivid.visuals.depthflow import DepthFlowBackend
from lofivid.visuals.keyframes import SDXLKeyframeBackend

log = logging.getLogger(__name__)


# ---------- public entry points -------------------------------------------

def generate(config_path: Path, cache_dir: Path, output_dir: Path) -> Path:
    cfg = load(config_path)
    run_cache = Cache(cache_dir / cfg.run_id)

    music_path = _do_music(cfg, run_cache, _make_music_backend(cfg))
    clips = _do_visuals(cfg, run_cache, *_make_visual_backends(cfg))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cfg.run_id}.mp4"
    return _do_compose(cfg, run_cache, music_path, clips, output_path, project_root=Path.cwd())


def generate_music_only(config_path: Path, cache_dir: Path) -> Path:
    cfg = load(config_path)
    run_cache = Cache(cache_dir / cfg.run_id)
    return _do_music(cfg, run_cache, _make_music_backend(cfg))


def generate_visuals_only(config_path: Path, cache_dir: Path) -> list[Path]:
    cfg = load(config_path)
    run_cache = Cache(cache_dir / cfg.run_id)
    return [c.path for c in _do_visuals(cfg, run_cache, *_make_visual_backends(cfg))]


# ---------- stage implementations -----------------------------------------

def _do_music(cfg: Config, cache: Cache, backend: MusicBackend) -> Path:
    seeds = SeedRegistry(cfg.seed)
    plans = design_tracklist(cfg.music, seeds)
    specs = plans_to_specs(plans, seeds)

    log.info("Music: rendering %d tracks via %s", len(specs), backend.name)
    backend.warmup()

    tracks_dir = cache.stage_dir("music_tracks")
    rendered: list[GeneratedTrack] = []
    for spec in specs:
        key = content_hash({"backend": backend.name, **spec.cache_key()})
        cached = cache.get("music_track", key)
        if cached is not None:
            log.info("Track %d cache hit (%s)", spec.track_index, cached.name)
            # Reconstruct GeneratedTrack metadata from disk
            from lofivid.music.acestep import _probe_duration_seconds
            rendered.append(GeneratedTrack(
                spec=spec, path=cached, sample_rate=44100,
                actual_duration_seconds=_probe_duration_seconds(cached),
            ))
            continue
        gen = backend.generate(spec, tracks_dir)
        cache.put("music_track", key, gen.path)
        rendered.append(gen)

    backend.shutdown()  # free VRAM before visuals start

    # Mix
    mix_path = cache.stage_dir("music_mix") / "mix.wav"
    mix_key = content_hash({
        "tracks": [t.spec.cache_key() for t in rendered],
        "crossfade": cfg.music.crossfade_seconds,
        "lufs": cfg.music.target_lufs,
    })
    cached_mix = cache.get("music_mix", mix_key)
    if cached_mix is not None:
        log.info("Music mix cache hit")
        return cached_mix

    mix_tracks(
        rendered, mix_path,
        MixSettings(
            crossfade_seconds=cfg.music.crossfade_seconds,
            target_lufs=cfg.music.target_lufs,
        ),
    )
    cache.put("music_mix", mix_key, mix_path)
    return mix_path


def _do_visuals(
    cfg: Config,
    cache: Cache,
    keyframe_backend: KeyframeBackend,
    parallax_backend: ParallaxBackend,
) -> list[GeneratedClip]:
    seeds = SeedRegistry(cfg.seed)
    preset = get_preset(cfg.visuals.preset)

    keyframe_dir = cache.stage_dir("keyframes")
    parallax_dir = cache.stage_dir("parallax")

    log.info("Visuals: %d scenes via %s preset", cfg.visuals.scene_count, cfg.visuals.preset)
    keyframe_backend.warmup()

    images = []
    for i in range(cfg.visuals.scene_count):
        prompt = preset.render_prompt(cfg.visuals.keyframe_prompt_template, i)
        spec = KeyframeSpec(
            scene_index=i,
            prompt=prompt,
            width=preset.spec().width,
            height=preset.spec().height,
            seed=seeds.derive(f"visuals.keyframe.{i}"),
        )
        key = content_hash({"backend": keyframe_backend.name, **spec.cache_key()})
        cached = cache.get("keyframe", key)
        if cached is not None:
            from lofivid.visuals.base import GeneratedImage
            log.info("Keyframe %d cache hit", i)
            images.append(GeneratedImage(spec=spec, path=cached))
            continue
        img = keyframe_backend.generate(spec, keyframe_dir)
        cache.put("keyframe", key, img.path)
        images.append(img)
    keyframe_backend.shutdown()

    parallax_backend.warmup()
    clips: list[GeneratedClip] = []
    for img in images:
        spec = ParallaxSpec(
            scene_index=img.spec.scene_index,
            image_path=img.path,
            duration_seconds=cfg.visuals.parallax_loop_seconds,
            width=cfg.output_resolution[0],
            height=cfg.output_resolution[1],
            fps=cfg.fps,
            seed=seeds.derive(f"visuals.parallax.{img.spec.scene_index}"),
        )
        key = content_hash({"backend": parallax_backend.name, **spec.cache_key()})
        cached = cache.get("parallax", key)
        if cached is not None:
            log.info("Parallax %d cache hit", img.spec.scene_index)
            clips.append(GeneratedClip(spec=spec, path=cached))
            continue
        clip = parallax_backend.generate(spec, parallax_dir)
        cache.put("parallax", key, clip.path)
        clips.append(clip)
    parallax_backend.shutdown()
    return clips


def _do_compose(
    cfg: Config,
    cache: Cache,
    music_path: Path,
    clips: list[GeneratedClip],
    output_path: Path,
    project_root: Path,
) -> Path:
    overlays.validate(cfg.overlays, project_root)

    total_seconds = cfg.duration_minutes * 60

    # Each parallax clip is a short loop; extend each to its scheduled scene length.
    scene_seconds = total_seconds / len(clips)
    extended_dir = cache.stage_dir("scenes_extended")
    extended_clips: list[GeneratedClip] = []
    for clip in clips:
        ext_path = extended_dir / f"{clip.spec.scene_index:03d}.mp4"
        key = content_hash({"src": str(clip.path), "duration": scene_seconds, "fps": cfg.fps})
        cached = cache.get("scene_extended", key)
        if cached is not None:
            extended_clips.append(GeneratedClip(spec=clip.spec, path=cached))
            continue
        ffmpeg_ops.loop_clip_to_duration(clip.path, scene_seconds, ext_path, cfg.fps)
        cache.put("scene_extended", key, ext_path)
        extended_clips.append(GeneratedClip(spec=clip.spec, path=ext_path))

    scenes = schedule(extended_clips, total_seconds=total_seconds, crossfade_seconds=2.0)

    rain = _resolve_overlay(cfg.overlays.rain_video, project_root)
    vinyl = _resolve_overlay(cfg.overlays.vinyl_crackle, project_root)

    settings = EncodeSettings(
        fps=cfg.fps,
        width=cfg.output_resolution[0],
        height=cfg.output_resolution[1],
    )
    final = ffmpeg_ops.concat_with_crossfades(
        scenes=scenes,
        audio_path=music_path,
        output_path=output_path,
        settings=settings,
        overlay_video=rain,
        overlay_opacity=cfg.overlays.rain_opacity,
        overlay_audio=vinyl,
        overlay_audio_gain_db=cfg.overlays.vinyl_gain_db,
    )

    actual = ffmpeg_ops.probe_duration_seconds(final)
    log.info("Final video: %s (%.1fs, target %.1fs)", final, actual, total_seconds)

    _write_manifest(cache, cfg, output_path, actual)
    return final


# ---------- helpers --------------------------------------------------------

def _make_music_backend(cfg: Config) -> MusicBackend:
    if cfg.music.backend == "acestep":
        return ACEStepBackend()
    raise ValueError(f"Unsupported music backend: {cfg.music.backend}")


def _make_visual_backends(cfg: Config) -> tuple[KeyframeBackend, ParallaxBackend]:
    preset = get_preset(cfg.visuals.preset)
    spec = preset.spec()
    # Allow per-config LoRA override
    loras = [(l.name, l.weight) for l in cfg.visuals.loras] or spec.loras
    return (
        SDXLKeyframeBackend(model_id=spec.model_id, loras=loras),
        DepthFlowBackend(),
    )


def _resolve_overlay(p: Path | None, project_root: Path) -> Path | None:
    if p is None:
        return None
    return p if p.is_absolute() else (project_root / p)


def _write_manifest(cache: Cache, cfg: Config, output_path: Path, actual_duration: float) -> None:
    manifest = {
        "run_id": cfg.run_id,
        "created_at": time.time(),
        "output": str(output_path),
        "actual_duration_seconds": actual_duration,
        "config": json.loads(cfg.model_dump_json()),
    }
    with open(cache.root / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
