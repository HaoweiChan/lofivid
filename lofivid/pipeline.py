"""End-to-end pipeline orchestration.

Stages:
  1. Load + validate config (eagerly resolves the referenced style).
  2. Music: design tracklist → generate per-track WAVs → DJ-mix into one stereo file.
  3. Visuals: generate keyframes → DepthFlow / overlay-motion parallax loops.
  4. Compose: schedule clips → loop+xfade → overlay (rain / brand / HUD / waveform)
     → mux audio → encode AV1 / libx264.

Each stage uses the cache so re-runs short-circuit unchanged steps. Every
stage's cache key includes `cfg.style_hash` so editing a style file
invalidates that style's downstream artefacts cleanly without manual nuking.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

# Importing the music + visuals packages triggers backend registration.
import lofivid.music  # noqa: F401
import lofivid.visuals  # noqa: F401
from lofivid._memcap import collect_between_stages
from lofivid.cache import Cache, content_hash
from lofivid.compose import ffmpeg_ops, overlays
from lofivid.compose.brand import render_brand_layer
from lofivid.compose.ffmpeg_ops import EncodeSettings
from lofivid.compose.hud import build_hud_overlays
from lofivid.compose.timeline import schedule
from lofivid.config import Config, load
from lofivid.env import assert_fonts_present
from lofivid.music import registry as music_registry
from lofivid.music.base import GeneratedTrack, MusicBackend
from lofivid.music.mixer import MixSettings, compute_timeline, mix_tracks
from lofivid.music.tracklist import design_tracklist, plans_to_specs
from lofivid.presets import get_preset
from lofivid.seeds import SeedRegistry
from lofivid.styles.schema import StyleSpec
from lofivid.visuals import registry as visuals_registry
from lofivid.visuals.base import (
    GeneratedClip,
    KeyframeBackend,
    KeyframeSpec,
    ParallaxBackend,
    ParallaxSpec,
)

log = logging.getLogger(__name__)

# Backends that introduce non-replayable cloud state. A run with any of these
# is recorded in the manifest as `regenerate_status: cloud_dependent`.
_CLOUD_MUSIC_BACKENDS = {"suno"}
_CLOUD_KEYFRAME_BACKENDS = {"unsplash"}


# ---------- public entry points -------------------------------------------

def generate(config_path: Path, cache_dir: Path, output_dir: Path) -> Path:
    cfg = load(config_path)
    style = cfg.resolved_style
    project_root = Path.cwd()

    # Hard-fail before any GPU work if a font referenced by the style is missing.
    assert_fonts_present(style, project_root)

    run_cache = Cache(cache_dir / cfg.run_id)

    music_path, tracks = _do_music(cfg, style, run_cache)
    collect_between_stages("music")
    clips = _do_visuals(cfg, style, run_cache)
    collect_between_stages("visuals")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cfg.run_id}.mp4"
    return _do_compose(cfg, style, run_cache, music_path, tracks, clips, output_path,
                       project_root=project_root)


def generate_music_only(config_path: Path, cache_dir: Path) -> Path:
    cfg = load(config_path)
    style = cfg.resolved_style
    run_cache = Cache(cache_dir / cfg.run_id)
    music_path, _tracks = _do_music(cfg, style, run_cache)
    return music_path


def generate_visuals_only(config_path: Path, cache_dir: Path) -> list[Path]:
    cfg = load(config_path)
    style = cfg.resolved_style
    run_cache = Cache(cache_dir / cfg.run_id)
    return [c.path for c in _do_visuals(cfg, style, run_cache)]


# ---------- stage implementations -----------------------------------------

def _do_music(cfg: Config, style: StyleSpec, cache: Cache) -> tuple[Path, list[GeneratedTrack]]:
    seeds = SeedRegistry(cfg.seed)
    backend = _make_music_backend(style)
    plans = design_tracklist(style.music_anchor, style.music_variations, cfg.music, seeds)
    specs = plans_to_specs(plans, seeds)

    log.info("Music: rendering %d tracks via %s", len(specs), backend.name)
    backend.warmup()

    tracks_dir = cache.stage_dir("music_tracks")
    rendered: list[GeneratedTrack] = []
    for spec in specs:
        key = content_hash({
            "backend": backend.name,
            "style_hash": cfg.style_hash,
            **spec.cache_key(),
        })
        cached = cache.get("music_track", key)
        if cached is not None:
            log.info("Track %d cache hit (%s)", spec.track_index, cached.name)
            title, artist = _read_track_sidecar(cached)
            from lofivid.music.acestep import _probe_duration_seconds
            rendered.append(GeneratedTrack(
                spec=spec, path=cached, sample_rate=44100,
                actual_duration_seconds=_probe_duration_seconds(cached),
                title=title, artist=artist,
            ))
            continue
        gen = backend.generate(spec, tracks_dir)
        cache.put("music_track", key, gen.path)
        _write_track_sidecar(gen)
        rendered.append(gen)

    backend.shutdown()  # free VRAM before visuals start

    # Mix
    mix_path = cache.stage_dir("music_mix") / "mix.wav"
    mix_key = content_hash({
        "tracks": [t.spec.cache_key() for t in rendered],
        "crossfade": cfg.music.crossfade_seconds,
        "lufs": cfg.music.target_lufs,
        "style_hash": cfg.style_hash,
    })
    cached_mix = cache.get("music_mix", mix_key)
    if cached_mix is not None:
        log.info("Music mix cache hit")
        return cached_mix, rendered

    mix_tracks(
        rendered, mix_path,
        MixSettings(
            crossfade_seconds=cfg.music.crossfade_seconds,
            target_lufs=cfg.music.target_lufs,
        ),
    )
    cache.put("music_mix", mix_key, mix_path)
    return mix_path, rendered


def _do_visuals(cfg: Config, style: StyleSpec, cache: Cache) -> list[GeneratedClip]:
    seeds = SeedRegistry(cfg.seed)
    keyframe_backend, parallax_backend = _make_visual_backends(style)
    preset = get_preset(style.preset)

    keyframe_dir = cache.stage_dir("keyframes")
    parallax_dir = cache.stage_dir("parallax")

    log.info("Visuals: %d scenes via %s preset", cfg.visuals.scene_count, style.preset)
    keyframe_backend.warmup()

    images = []
    for i in range(cfg.visuals.scene_count):
        prompt = preset.render_prompt(style.keyframe_prompt_template, i)
        spec = KeyframeSpec(
            scene_index=i,
            prompt=prompt,
            width=preset.spec().width,
            height=preset.spec().height,
            seed=seeds.derive(f"visuals.keyframe.{i}"),
        )
        extras = keyframe_backend.cache_key_extras(spec)
        key = content_hash({
            "backend": keyframe_backend.name,
            "style_hash": cfg.style_hash,
            **spec.cache_key(),
            **extras,
        })
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
        key = content_hash({
            "backend": parallax_backend.name,
            "style_hash": cfg.style_hash,
            **spec.cache_key(),
        })
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
    style: StyleSpec,
    cache: Cache,
    music_path: Path,
    tracks: list[GeneratedTrack],
    clips: list[GeneratedClip],
    output_path: Path,
    project_root: Path,
) -> Path:
    overlays.validate(cfg.overlays, project_root)

    total_seconds = cfg.duration_minutes * 60
    frame_w, frame_h = cfg.output_resolution

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

    # ---- new compose-stage overlays from the resolved style ----
    brand_dir = cache.stage_dir("brand")
    hud_dir = cache.stage_dir("hud")
    brand_png = render_brand_layer(style.brand_layers, frame_w, frame_h, brand_dir)
    windows = compute_timeline(tracks, cfg.music.crossfade_seconds)
    hud_overlays = build_hud_overlays(style.hud, windows, (frame_w, frame_h), hud_dir)

    settings = EncodeSettings(
        fps=cfg.fps,
        width=frame_w,
        height=frame_h,
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
        brand_layer=brand_png,
        hud_overlays=hud_overlays,
        waveform_spec=style.waveform,
        waveform_duotone=style.duotone,
    )

    actual = ffmpeg_ops.probe_duration_seconds(final)
    log.info("Final video: %s (%.1fs, target %.1fs)", final, actual, total_seconds)

    _write_manifest(cache, cfg, style, output_path, actual)
    return final


# ---------- backend factories ---------------------------------------------

def _make_music_backend(style: StyleSpec) -> MusicBackend:
    """Resolve the music backend through the registry.

    Per-backend init kwargs flow through `style.music_backend_params`. Suno's
    `model_version` lives there too — see styles/morning_cafe.yaml etc.
    """
    return music_registry.make(style.music_backend, **style.music_backend_params)


def _make_visual_backends(style: StyleSpec) -> tuple[KeyframeBackend, ParallaxBackend]:
    preset = get_preset(style.preset)
    spec = preset.spec()

    keyframe_kwargs = dict(style.keyframe_backend_params)
    if style.keyframe_backend == "sdxl":
        # Allow per-style LoRA override. SDXL backend takes (model_id, loras, ...).
        loras = [(lora.name, lora.weight) for lora in style.loras] or spec.loras
        keyframe_kwargs.setdefault("model_id", spec.model_id)
        keyframe_kwargs.setdefault("loras", loras)
        keyframe_kwargs.setdefault("negative_prompt", spec.negative_prompt)
    elif style.keyframe_backend == "unsplash":
        keyframe_kwargs.setdefault("quality_suffix", spec.quality_suffix)
        keyframe_kwargs.setdefault("duotone_pair", style.duotone or spec.duotone)

    keyframe = visuals_registry.make_keyframe(style.keyframe_backend, **keyframe_kwargs)

    parallax_kwargs = dict(style.parallax_backend_params)
    if style.parallax_backend == "overlay_motion":
        parallax_kwargs.setdefault("motion_type", style.motion_type)
    parallax = visuals_registry.make_parallax(style.parallax_backend, **parallax_kwargs)
    return keyframe, parallax


# ---------- helpers --------------------------------------------------------

def _resolve_overlay(p: Path | None, project_root: Path) -> Path | None:
    if p is None:
        return None
    return p if p.is_absolute() else (project_root / p)


def _track_sidecar_path(track_path: Path) -> Path:
    """Return the JSON sidecar location for a cached track."""
    return track_path.with_suffix(track_path.suffix + ".meta.json")


def _write_track_sidecar(track: GeneratedTrack) -> None:
    """Persist title/artist next to the cached WAV so cache hits don't lose metadata."""
    side = _track_sidecar_path(track.path)
    side.write_text(json.dumps({
        "title": track.title,
        "artist": track.artist,
    }, indent=2, default=str))


def _read_track_sidecar(track_path: Path) -> tuple[str, str | None]:
    side = _track_sidecar_path(track_path)
    if not side.exists():
        return "", None
    try:
        data = json.loads(side.read_text())
        return str(data.get("title") or ""), data.get("artist")
    except Exception as e:
        log.warning("Could not read track sidecar %s: %s", side, e)
        return "", None


@dataclass(frozen=True)
class _ManifestProvenance:
    regenerate_status: str       # "deterministic" | "cloud_dependent"
    cloud_sources: list[str]


def _classify_provenance(style: StyleSpec) -> _ManifestProvenance:
    cloud: list[str] = []
    if style.music_backend in _CLOUD_MUSIC_BACKENDS:
        cloud.append(style.music_backend)
    if style.keyframe_backend in _CLOUD_KEYFRAME_BACKENDS:
        cloud.append(style.keyframe_backend)
    return _ManifestProvenance(
        regenerate_status="cloud_dependent" if cloud else "deterministic",
        cloud_sources=cloud,
    )


def _write_manifest(cache: Cache, cfg: Config, style: StyleSpec,
                    output_path: Path, actual_duration: float) -> None:
    provenance = _classify_provenance(style)
    manifest = {
        "run_id": cfg.run_id,
        "created_at": time.time(),
        "output": str(output_path),
        "actual_duration_seconds": actual_duration,
        "config": json.loads(cfg.model_dump_json()),
        "style_ref": cfg.style_ref,
        "style_hash": cfg.style_hash,
        "style_content": json.loads(style.model_dump_json()),
        "regenerate_status": provenance.regenerate_status,
        "cloud_sources": provenance.cloud_sources,
    }
    with open(cache.root / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
