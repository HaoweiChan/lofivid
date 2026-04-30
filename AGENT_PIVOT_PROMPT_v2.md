# lofivid — Architecture Pivot v3

## Context

The v1 pivot landed: `UnsplashKeyframeBackend`, `OverlayMotionBackend`, `SunoMusicBackend`, the `_grading.py` refactor, and two new configs (`jazz_cafe_unsplash.yaml`, `minimal_design_lofi.yaml`) all exist and work. Beyond v1, FLUX.2 Klein and Z-Image-Turbo keyframe backends were added per `MODEL_OPTIONS.md`. Good. **Do not touch any of that.**

This pivot does three things, in priority order:

1. **Split style template from MV instance.** The repo currently conflates them in monolithic YAMLs. After this pivot, a *style* is a reusable, content-hashed brand definition (visual prompts, music anchor, motion type, palette, typography, HUD/waveform settings) and a *run* is a thin instance that references a style by name and sets per-render parameters (duration, seed, scene_count). This is what makes "rotate through styles, but every video on a given style looks like the same channel" work.
2. **Add HUD + waveform overlays + persistent brand text layer.** Every output frame should carry the channel's brand (kicker / title / tracklist strip — the look the user has been iterating in `preview_workday_cafe.py` and `preview_brand_variants.py`) as a static overlay, plus a per-track "now playing" HUD, plus a real audio-driven waveform from FFmpeg `showwaves`.
3. **Add `LibraryMusicBackend` and refactor backend dispatch into a registry.** Library backend ingests pre-licensed Epidemic Sound WAVs from a folder; registry replaces the if/elif chains in `pipeline.py` so adding a new backend is one new file plus a registration line.

Read `CLAUDE.md` and `MODEL_OPTIONS.md` first. Conventions there (Pydantic `extra="forbid"`, ABC backends with stable `name` and `cache_key()`, per-purpose seeds via `SeedRegistry`, no MoviePy / torch.compile / flash-attn, `from __future__ import annotations` at the top of every module, type hints throughout, comments only when *why* is non-obvious) are non-negotiable.

**This is a clean break — no back-compat.** The repo is pre-production; the existing 7 monolithic configs in `configs/` get deleted and replaced with two new files (one smoke, one production). `style_ref` is **mandatory** on every run config; there is no "inline style" fallback path. Any existing tests that validate the old `MusicConfig` / `VisualsConfig` shape get rewritten to match the new schema. Don't preserve the old API surface "just in case"; deleting the dead branches is part of the work.

**Goal of this pivot**: brand consistency across videos sharing a style. Not byte-identical regeneration of old MVs — that's a non-goal; the manifest just records what was used so we have provenance, with cloud-source asymmetry made explicit (`regenerate_status: cloud_dependent` flag).

## Tasks (in this order — earlier tasks unlock later ones)

### 1. Backend registry refactor (small, do this first)

The current `pipeline._make_music_backend()` and `pipeline._make_visual_backends()` are if/elif chains. Replace with a registry pattern so adding a backend is one new file plus a `register()` call.

Create three small registry modules:

- `lofivid/music/registry.py` — `MUSIC_BACKENDS: dict[str, Callable[..., MusicBackend]]` plus `register(name, factory)` and `make(name, **kwargs) -> MusicBackend`. Factories are zero-arg callables that return a fresh backend instance; per-backend init args flow through `**kwargs`.
- `lofivid/visuals/registry.py` — same pattern, but two dicts: `KEYFRAME_BACKENDS` and `PARALLAX_BACKENDS`.
- Each existing backend module gets a `register()` call near the bottom (e.g. `acestep.py` ends with `register("acestep", ACEStepBackend)`).

The `Literal[...]` fields in `lofivid/config.py` shift to `str` validated at runtime against `registry.MUSIC_BACKENDS.keys()` etc. Use a Pydantic `field_validator` so the error message lists the available names. This is a deliberate small loosening — strict literals lose IDE autocomplete on YAML, but the runtime validator preserves the loud-on-typo behaviour `CLAUDE.md` requires.

Imports from the registry trigger registration. To make sure all backends register, add `lofivid/music/__init__.py` that imports each backend module (`from . import acestep, suno`) and same for visuals (`from . import keyframes, unsplash, depthflow, overlay_motion, flux_klein, z_image`). Lazy backends (Suno, FLUX.2, Z-Image) currently use lazy imports in `pipeline.py`; preserve that — the registry registers a *factory* (a small lambda that does the import), not the class itself, so bringing in the registry doesn't drag heavy GPU deps into the host-mode startup path.

After this refactor, `pipeline._make_music_backend` is `return registry.make(cfg.music.backend, **cfg.music.backend_params)`. Same for visuals.

Tests: existing music/visual tests still pass; one new test confirms unknown backend names raise with a list of available names.

### 2. Style template schema + loader (`lofivid/styles/`)

New module. The style is a Pydantic model layered above the existing `Config` schema; a run config can either reference a style by name (`style_ref: morning_cafe`) or inline the same fields (back-compat with existing monolithic configs).

#### `lofivid/styles/schema.py`

```python
class TextLayerSpec(BaseModel):
    """One static text overlay that's baked once and composited on every frame."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    text: str
    font_path: Path                  # absolute or relative to repo root
    cjk_font_path: Path | None = None  # used for codepoints the primary font lacks
    size_pct: float = Field(0.022, gt=0, le=0.5)  # of frame height
    color: str = "#FFFFFF"
    shadow_color: str | None = None  # set to a color hex to enable a 1px drop shadow
    position: Literal["top_centre", "top_left", "top_right",
                      "bottom_centre", "bottom_left", "bottom_right",
                      "centre"] = "top_centre"
    margin_pct: float = 0.06         # distance from the chosen edge, as a fraction of frame height

class HUDSpec(BaseModel):
    """Per-track 'now playing' badge that updates at track boundaries."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    font_path: Path
    cjk_font_path: Path | None = None
    title_size_pct: float = 0.028
    artist_size_pct: float = 0.020
    counter_size_pct: float = 0.018
    text_color: str = "#FFFFFF"
    panel_color: str = "#000000"
    panel_opacity: float = 0.45
    panel_padding_px: int = 14
    corner: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = "bottom_left"
    margin_pct: float = 0.04
    show_track_counter: bool = True
    show_artist: bool = True
    max_width_pct: float = 0.4       # of frame width — wraps/truncates beyond

class WaveformSpec(BaseModel):
    """Audio-driven waveform band along the bottom (or top) of the video."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    mode: Literal["line", "cline", "p2p", "point"] = "cline"
    height_px: int = 80
    position: Literal["top", "bottom"] = "bottom"
    scale: Literal["lin", "log", "sqrt", "cbrt"] = "sqrt"
    # Color sourcing is a discriminator: either a fixed colour or derived
    # from the active style's duotone palette (so the waveform automatically
    # harmonises with the visual grade).
    color_source: Literal["fixed", "duotone_highlight", "duotone_shadow"] = "duotone_highlight"
    fixed_color: str = "#E0E0C4"     # only consulted when color_source == "fixed"
    opacity: float = 0.6

class StyleSpec(BaseModel):
    """A reusable channel-brand definition. Hashable; one style = one look."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Human-readable identifier; matches filename stem")
    description: str = ""

    # Visual identity
    preset: Literal["anime", "photo"] = "photo"
    keyframe_backend: str = "unsplash"
    keyframe_backend_params: dict[str, Any] = Field(default_factory=dict)
    keyframe_prompt_template: str
    parallax_backend: str = "overlay_motion"
    parallax_backend_params: dict[str, Any] = Field(default_factory=dict)
    motion_type: Literal["slow_zoom", "dust_motes", "light_flicker", "none"] = "slow_zoom"
    duotone: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    loras: list[LoraSpec] = Field(default_factory=list)

    # Music identity
    music_backend: str = "library"
    music_backend_params: dict[str, Any] = Field(default_factory=dict)
    music_anchor: MusicAnchor
    music_variations: list[MusicVariation]

    # Brand layers (rendered once, composited on every frame)
    brand_layers: list[TextLayerSpec] = Field(default_factory=list)

    # Per-track HUD + audio-driven waveform
    hud: HUDSpec
    waveform: WaveformSpec

    @model_validator(mode="after")
    def _validate_backends_known(self) -> StyleSpec:
        # Defer to runtime — backends register lazily, so we can't import
        # the registry at schema-definition time without circularity.
        # The pipeline's _make_*_backend will re-validate.
        return self
```

#### `lofivid/styles/loader.py`

- `load_style(name: str, root: Path) -> tuple[StyleSpec, str]` — reads `<root>/styles/<name>.yaml`, validates against `StyleSpec`, returns `(style, hash)`.
- `style_hash(style: StyleSpec) -> str` — canonical-JSON SHA-256, first 12 hex chars. Use `style.model_dump(mode="json", exclude={"description"})` so editing the description doesn't invalidate the hash. Keys are sorted; lists keep order (order matters in `music_variations` because the tracklist designer round-robins through them).

#### Config integration

Refactor `lofivid/config.py` for a clean break:

- Add a required field `Config.style_ref: str` (no default; YAML must specify it).
- Delete the existing `MusicConfig` and `VisualsConfig` classes; the style-defining fields they held now live exclusively in `StyleSpec`.
- Replace them with new `MusicInstance` and `VisualsInstance` classes that hold *only* per-run fields:
  - `MusicInstance`: `track_count`, `track_seconds_range`, `crossfade_seconds`, `target_lufs`. Nothing else.
  - `VisualsInstance`: `scene_count`, `scene_seconds`, `parallax_loop_seconds`, `premium_scenes`. Nothing else.
- `Config.music: MusicInstance` and `Config.visuals: VisualsInstance` replace the old fields.
- `OverlaysConfig` stays as-is (it's per-run: which CC0 rain/vinyl files to layer, at what opacity, depends on the specific video, not the brand).
- `Config.resolved_style: StyleSpec` is a computed property (not a stored field) that calls `load_style(self.style_ref, repo_root)`. `Config.style_hash: str` is the matching computed hash. Both raise loudly if the style file is missing or fails validation.
- The pipeline always works against the resolved style for identity (backends, prompts, palette, brand layers, HUD, waveform) and against the run config for instance parameters (duration, seed, counts, overlays). There is no synthesised fallback style.

Pydantic `extra="forbid"` on the new `MusicInstance` / `VisualsInstance` means any leftover style-defining field in a run YAML (`backend:`, `preset:`, `anchor:`, etc.) raises a clear validation error pointing the user at the style file. That's the desired ergonomic — it forces clean separation.

### 3. `LibraryMusicBackend` (`lofivid/music/library.py`)

Subclass `MusicBackend`. Reads pre-licensed audio files from a directory tree.

- Init args: `library_dir: Path`, `match_by: Literal["mood", "round_robin"] = "mood"`, `prefer_extensions: tuple[str, ...] = (".wav", ".flac", ".mp3")`.
- Folder convention: `<library_dir>/<mood_slug>/<title>.wav`. Mood slug matches `slugify(MusicVariation.mood)`. `slugify` is a small helper (lowercase, non-alnum → `_`, collapse runs).
- `generate(spec, output_dir)`:
  1. Compute mood from `spec.prompt` — either the prompt contains the mood string verbatim (it does; `tracklist.to_prompt()` includes mood as a tag), or it's stashed in a new optional `TrackSpec.mood` field. Add the field; default None for backwards compat.
  2. Build a deterministic shortlist: list files in `<library_dir>/<mood_slug>/` (or in `<library_dir>/` with `match_by="round_robin"`), filter by extension, sorted alphabetically.
  3. Pick `shortlist[spec.seed % len(shortlist)]`. Same seed + same library = same track.
  4. Copy the chosen file to `output_dir/<spec.track_index:03d>.wav`. If source is MP3, transcode to WAV (mirror `_transcode_to_wav` from `suno.py`).
  5. Read ID3 / Vorbis / RIFF metadata for `title` and `artist` via `mutagen` (already a transitive dep of HF libs — verify with `python -c 'import mutagen'`; if not present, add it to `pyproject.toml` `dependencies` since it's pure-Python and tiny).
  6. Return a `GeneratedTrack` with `title` and `artist` populated. (See task 4.)
- Empty-folder failure: raise a clear error pointing at the expected path. Don't silently fall through to round-robin if mood-folder match was requested — that hides config bugs.
- Cache key contribution: include the resolved file path *and* its content hash (SHA-256 of first 1 MB; full hash is overkill for cache invalidation and slow on long files). Library swaps invalidate cleanly.

### 4. Track metadata on `GeneratedTrack`

Extend `lofivid/music/base.py`:

```python
@dataclass(frozen=True)
class GeneratedTrack:
    spec: TrackSpec
    path: Path
    sample_rate: int
    actual_duration_seconds: float
    title: str = ""               # NEW — for HUD display
    artist: str | None = None     # NEW — for HUD display
```

Update each backend to populate sensibly:

- `ACEStepBackend`: synthesise `title` from mood + first instrument (`f"{mood.title()} — {instruments[0].title()}"`); leave `artist=None`.
- `SunoMusicBackend`: pass through Suno's returned track title; `artist = "Suno AI"`.
- `LibraryMusicBackend`: read from file metadata; fall back to filename stem if no metadata.

The `pipeline._do_music()` cache-rehydration path currently reconstructs `GeneratedTrack` from disk. Extend it to read title/artist from a small sidecar JSON written next to the cached WAV at insertion time, so cache hits don't lose metadata.

### 5. Track timeline from the mixer

For HUD windowing the pipeline needs to know "track i plays from t=A to t=B in the final mixed audio." The current `mix_tracks` returns just a path. Add:

```python
@dataclass(frozen=True)
class TrackWindow:
    track: GeneratedTrack
    start_seconds: float          # in the mixed-output timeline
    end_seconds: float            # in the mixed-output timeline

def compute_timeline(tracks: list[GeneratedTrack],
                     crossfade_seconds: float) -> list[TrackWindow]: ...
```

The math is: `acrossfade=d=c` overlaps adjacent tracks by `c` seconds. So:

- Track 0 plays `[0, dur_0]`.
- Track i (i ≥ 1) plays `[cumulative_offset_i, cumulative_offset_i + dur_i]` where `cumulative_offset_i = sum(dur_0..dur_{i-1}) - c * i`.
- For HUD-display purposes (when does the badge switch?), use the *midpoint* of the crossfade as the boundary: track i's HUD window ends, and track i+1's begins, at `cumulative_offset_{i+1} + c/2`.

`mix_tracks` should still return a `Path` (no signature change), but the pipeline calls `compute_timeline` directly with the same inputs and threads the result into the compose stage. Don't change the mixer's existing tests.

### 6. HUD overlay (`lofivid/compose/hud.py`)

Per-track now-playing badge. Two halves:

**6a. PNG renderer.** PIL renders one transparent PNG per track containing the track's title + (optional) artist + (optional) "Track 3 / 12" counter, on a semi-opaque rounded panel. Inputs: `HUDSpec`, `TrackWindow` (for the counter), frame resolution. Outputs: `Path` to a PNG, content-hash-cached so re-renders skip already-rendered HUDs.

Implementation notes that matter:
- Font handling: load `font_path`. If the title contains any codepoint the font doesn't cover and `cjk_font_path` is set, fall back to it for those codepoints. Implementation: `PIL.ImageFont.truetype` for both, then lay out the text run-by-run picking per-glyph fonts. There's no built-in for this; do the manual pass — given the typical title length (< 60 chars) the cost is trivial.
- Wrapping: if the title's rendered width > `max_width_pct * frame_width`, truncate with an ellipsis. Don't multi-line the title — keep the badge compact.
- Panel: rounded rectangle (radius = 8 px), filled with `panel_color` at `panel_opacity`. Padding `panel_padding_px` around the text. Layout: title on top, artist below in smaller font, counter in the corner of the panel.

**6b. ffmpeg overlay integrator.** Given a list of `(TrackWindow, hud_png_path)` and the per-track corner offsets, build the filter-graph fragment that overlays each PNG during its track window. Pattern:

```
[base_v][hud_png_0]overlay=x=20:y=H-h-20:enable='between(t,0,140)'[v0];
[v0][hud_png_1]overlay=x=20:y=H-h-20:enable='between(t,140,280)'[v1];
...
```

This integrates into `compose/ffmpeg_ops.py:concat_with_crossfades` as one new optional kwarg `hud_overlays: list[HUDOverlay] | None = None` where `HUDOverlay = (png_path, start_seconds, end_seconds, x_expr, y_expr)`. When None, behaviour unchanged.

### 7. Brand text layers (`lofivid/compose/brand.py`)

The persistent "WORK CAFE JAZZ / 工作日 咖啡 爵士節奏 / 01 cafe afternoon..." overlays from `preview_workday_cafe.py`. Same renderer pattern as HUD but rendered *once* per video (not per-track) and overlaid on every frame.

- One module-level function `render_brand_layer(spec: TextLayerSpec, frame_w: int, frame_h: int, cache_dir: Path) -> Path` — produces a transparent PNG sized `frame_w × frame_h` with the text at the configured position. Cache key = `content_hash(spec.model_dump() + (frame_w, frame_h))`.
- Multiple layers: each `TextLayerSpec` in `style.brand_layers` becomes one PNG; ffmpeg overlays them in order. Composite once into a single `brand_layer.png` at compose time so the final filter graph has only one extra overlay regardless of how many layers the style declares.
- Same font fallback logic as HUD (use the same helper from `compose/_text.py` — refactor the per-glyph font selector into one place; both HUD and brand use it).

Style-level brand layers are typically: (1) the kicker line at top-centre, (2) the primary title in display serif, (3) optional CJK subtitle, (4) tracklist strip at bottom-centre. The `preview_workday_cafe.py` layout serves as the reference; replicate its proportions in the default `morning_cafe.yaml` style we'll ship in task 11.

### 8. Waveform overlay (`lofivid/compose/waveform.py`)

FFmpeg `showwaves` filter on the *final mixed* audio (after loudness normalisation). Pure filter-graph fragment, no new files written.

Returns a string fragment that the compose stage splices into its `-filter_complex`:

```
[0:a]showwaves=s={frame_w}x{height_px}:mode={mode}:colors={color}@{opacity}:scale={scale}:rate={fps},format=yuva420p,setpts=PTS-STARTPTS[wave]
```

Then the compose stage overlays `[wave]` on the video at `y=0` (top) or `y=H-h` (bottom). The final filter graph in `concat_with_crossfades` becomes:

1. Existing scene xfade chain → `[vbase]`.
2. Optional rain overlay on `[vbase]` → `[vrain]`.
3. Optional brand layer on the result → `[vbrand]`.
4. Optional HUD overlays (one per track window) on the result → `[vhud]`.
5. Optional waveform overlay on the result → `[vfinal]`.
6. Map `[vfinal]` and the final audio.

Order matters: brand under HUD under waveform reads cleanest in our reference channels. Don't make the order configurable — pin it.

**Color resolution**: when the style says `color_source: duotone_highlight`, the compose stage looks up `style.duotone[1]` (the highlight RGB), converts to hex, and substitutes into the filter. Same for `duotone_shadow`. If `style.duotone is None` and `color_source != fixed`, raise a clear error at config-load time.

### 9. Per-stage logging + manifest changes

`lofivid/pipeline.py:_write_manifest` currently records the run config. Extend it to also record:

```python
manifest = {
    "run_id": cfg.run_id,
    "created_at": ...,
    "output": ...,
    "actual_duration_seconds": ...,
    "config": ...,
    # NEW
    "style_ref": cfg.style_ref,
    "style_hash": <hash> if cfg.style_ref else None,
    "style_content": <full effective style as dict>,  # always recorded, hash or no
    "regenerate_status": "deterministic" | "cloud_dependent",
    "cloud_sources": [...],     # list of cloud backends used, e.g. ["unsplash", "suno"]
}
```

`regenerate_status` is computed by inspecting the resolved music + keyframe backend names. Suno and Unsplash mark the run `cloud_dependent`. Library + ACE-Step + SDXL/FLUX/Z-Image keep it `deterministic`. The user knows that `cloud_dependent` runs may not byte-replicate.

### 10. Style-aware caching

The cache key already includes backend name + spec. Extend keyframe + parallax + music cache keys to also include `style_hash` (or a short prefix of it). This means: editing `morning_cafe.yaml` invalidates everything cleanly without manual cache nuking. Two separate styles that happen to share a backend share nothing in the cache, even if their prompts collide.

(Reasoning: today, two styles using `unsplash` + `"cafe interior"` prompt would cache-collide. Including style hash prevents that — small cost on cache space, large win on conceptual cleanliness.)

### 11. Bundled fonts + new style + new run config

#### Fonts

Drop three font files into `assets/fonts/`. Add an exemption in `.gitignore` so they survive (`!assets/fonts/**/*.ttf`, `!assets/fonts/**/*.otf`). Fonts to ship:

- **Playfair Display Bold** (SIL OFL, free for commercial use) — display serif for titles. Source: Google Fonts.
- **IBM Plex Sans Regular + Bold** (SIL OFL) — body / kicker / counter / artist text. Source: Google Fonts.
- **Noto Sans CJK TC Bold** (SIL OFL) — CJK fallback. Source: Google Fonts. Required for `工作日 咖啡 爵士節奏` and any future CJK title text.

The agent **must** obtain these files (download from Google Fonts, package mirrors, or wherever) and place them at the paths the default style references. **There is no fallback path and no graceful degradation.** Font load failure is a hard error — the user has explicitly chosen no compromise on visual quality. Implementation:

- A new preflight check in `lofivid/env.py` (or a dedicated `lofivid/compose/_fonts.py`) iterates every `font_path` and `cjk_font_path` referenced in the resolved style. If any path doesn't exist, raise a clear `RuntimeError` listing the missing files, *before* any GPU work starts.
- The HUD / brand renderers use `PIL.ImageFont.truetype` directly with no try/except around the load. Let the exception propagate; the preflight guarantees we never reach this code with a missing font.
- If the build environment cannot fetch these files at all, the task is incomplete. Don't ship a `assets/fonts/README.md` that documents manual download as a workaround. Don't fall back to system fonts. Don't fall back to DejaVu (the preview scripts use it because it's bundled with Ubuntu, but the production styles must use the licensed display fonts).

Update `lofivid licenses` to add SIL OFL rows for Playfair Display, IBM Plex Sans, and Noto Sans CJK TC.

#### Default style

`styles/morning_cafe.yaml` — the user's already-iterated brand:

```yaml
name: morning_cafe
description: |
  Cafe afternoon jazz with vocals, photographic warm-cream visuals,
  WORK CAFE JAZZ display title with Traditional Chinese subtitle line,
  rust accent. Mirrors preview_workday_cafe.py's locked aesthetic.

preset: photo
keyframe_backend: unsplash
keyframe_prompt_template: cafe interior warm light
parallax_backend: overlay_motion
motion_type: slow_zoom
duotone: [[40, 22, 8], [244, 222, 184]]   # espresso → cream

music_backend: library
music_backend_params:
  library_dir: assets/music/cafe_jazz
music_anchor:
  bpm_range: [78, 92]
  key_pool: [F major, Bb major, Eb major, G minor]
  style_tags: [jazz, vocal jazz, smooth jazz, lo-fi, vinyl crackle]
music_variations:
  - mood: cafe afternoon
    instruments: [Rhodes, upright bass, brushed drums]
  - mood: rainy window
    instruments: [piano, double bass, soft kick]
  - mood: late night booth
    instruments: [muted trumpet, walking bass, hihat]
  - mood: vinyl spin
    instruments: [vibraphone, acoustic guitar, shaker]
  - mood: latte art
    instruments: [Wurlitzer, soft pads, brushes]
  - mood: bookshelf hour
    instruments: [piano, cello, light kick]

brand_layers:
  - text: "LOFIVID  ・  PLAYLIST"
    font_path: assets/fonts/IBMPlexSans-Bold.ttf
    size_pct: 0.022
    color: "#281810"           # KICKER_COLOUR from preview script
    position: top_centre
    margin_pct: 0.06
  - text: "WORK CAFE JAZZ"
    font_path: assets/fonts/PlayfairDisplay-Bold.ttf
    size_pct: 0.085
    color: "#A8341E"           # rust accent
    shadow_color: "#0000005A"
    position: top_centre
    margin_pct: 0.11
  - text: "工作日   咖啡   爵士節奏"
    font_path: assets/fonts/PlayfairDisplay-Bold.ttf
    cjk_font_path: assets/fonts/NotoSansCJKtc-Bold.otf
    size_pct: 0.062
    color: "#A8341E"
    position: top_centre
    margin_pct: 0.22
  - text: "01 cafe afternoon   /   02 rainy window   /   03 late night booth   /   04 vinyl spin   /   05 latte art   /   06 bookshelf hour"
    font_path: assets/fonts/IBMPlexSans-Regular.ttf
    size_pct: 0.022
    color: "#3C281E"
    position: bottom_centre
    margin_pct: 0.045

hud:
  enabled: true
  font_path: assets/fonts/IBMPlexSans-Bold.ttf
  cjk_font_path: assets/fonts/NotoSansCJKtc-Bold.otf
  title_size_pct: 0.028
  artist_size_pct: 0.020
  counter_size_pct: 0.018
  text_color: "#FFFFFF"
  panel_color: "#0A0A0A"
  panel_opacity: 0.5
  panel_padding_px: 14
  corner: bottom_left
  margin_pct: 0.04
  show_track_counter: true
  show_artist: true

waveform:
  enabled: true
  mode: cline
  height_px: 80
  position: bottom
  scale: sqrt
  color_source: duotone_highlight
  opacity: 0.55
```

#### Default run config

`configs/2026-04-30_morning_cafe_30min.yaml` — thin instance:

```yaml
run_id: morning_cafe_30min_v1
style_ref: morning_cafe
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

overlays:
  rain_video: null
  rain_opacity: 0.0
  vinyl_crackle: null
  vinyl_gain_db: -28
```

Note the run config has *no* `music.backend`, *no* `music.anchor`, *no* `visuals.preset`, *no* prompts — those live in the style. Pydantic enforces this via `MusicInstance` / `VisualsInstance` having `extra="forbid"` and only carrying per-run fields.

#### Delete legacy configs and replace with one smoke

Delete all 7 existing files under `configs/`:

- `smoke_30sec.yaml`
- `demo_5min_anime.yaml`
- `medium_30min_anime.yaml`
- `anime_rainy_window.yaml`
- `photo_cozy_cafe.yaml`
- `jazz_cafe_unsplash.yaml`
- `minimal_design_lofi.yaml`

Replace the smoke with `configs/smoke_30sec.yaml` + `styles/_smoke.yaml` (filename starts with `_` so it sorts first and signals "internal / not a real channel brand"). The smoke style:

- Uses `music_backend: acestep` (no library dependency, no API key, fastest path to "is the pipeline alive").
- Single mood variation, single style tag, minimal prompt.
- `keyframe_backend: sdxl` against the bundled SDXL base (already commercial-OK and already cached in HF on dev machines).
- `parallax_backend: overlay_motion`, `motion_type: none` (a static still — fastest).
- `brand_layers: []` — smoke verifies the pipeline, not the brand.
- `hud.enabled: false`, `waveform.enabled: false` — those have their own unit tests; the smoke shouldn't depend on them.
- `duotone: null`.

The matching `configs/smoke_30sec.yaml` is a 30-second 640×360 single-track single-scene run, equivalent to the deleted `smoke_30sec.yaml` but routed through `style_ref: _smoke`.

The user can recreate the deleted demo / medium / anime / photo / jazz / minimal configs as new `(style, run config)` pairs at their own pace; that's not part of this pivot.

### 12. CLI / docs

- `lofivid licenses`: add rows for Playfair Display, IBM Plex Sans, Noto Sans CJK TC (SIL OFL). Add a row for Epidemic Sound (note: requires user subscription; license valid only for the channel registered with Epidemic Sound; see assets/music/README.md).
- `assets/music/README.md` (new) — explains the folder convention `assets/music/<mood_slug>/<title>.wav` and the Epidemic Sound channel-registration requirement (Creator plan covers 1 channel per platform; track licences stay valid for videos uploaded *while subscribed* even after cancellation, but new uploads after cancellation are not licensed).
- `README.md`: add a "Style templates" section (between "Quickstart" and "Project structure") explaining the style/instance split with one example. Update the "Quickstart" command to reference `configs/2026-04-30_morning_cafe_30min.yaml` instead of any of the deleted configs.

## Conventions to follow

- `from __future__ import annotations` at the top of every new module.
- All new backends + style components implement `cache_key()` / `content_hash()` participation. Style hash is part of every downstream stage's cache key.
- Per-purpose seeds: new purposes are `music.library.{i}`, `compose.hud.{i}`, `compose.brand`, `compose.waveform`. Never reuse `cfg.seed` directly.
- Pydantic `extra="forbid"` on every new model. Typos in YAML must raise.
- No new heavy deps. `mutagen` is fine (pure Python, ~150 KB). `requests` already transitive. No PIL feature beyond what's already in `_grading.py` + `preview_*.py`.
- Logging via `log = logging.getLogger(__name__)`. No print statements outside `scripts/`.
- Cloud calls keep their existing retry-with-backoff. New library backend has no network.
- The new `compose/_text.py` (font fallback helper) is the single place per-glyph font selection lives — both HUD and brand import it.

## Verification before claiming done

- `pytest -q` passes. Existing tests that referenced the deleted `MusicConfig` / `VisualsConfig` / legacy YAML configs get **rewritten**, not preserved. Coverage stays equivalent or better; specifically add new tests for:
  - Style schema (load + hash; same input → same hash; description edits don't change hash)
  - `style_ref` is mandatory (Pydantic raises if missing)
  - Run config rejects style-defining fields (e.g. `music.backend` in the run YAML raises)
  - Registry (round-trip register/make, unknown name raises with available list)
  - `LibraryMusicBackend` track selection determinism (same seed → same track)
  - `compute_timeline` math (3 tracks of 10/15/20s with 2s xfade → expected windows)
  - HUD PNG renderer (golden-image diff against a checked-in 256×128 fixture; pin font + size + colour)
  - Per-glyph CJK font fallback path (a title containing one CJK char triggers the secondary font for that glyph only)
  - Waveform filter-graph string generation (smoke test: parses valid `-filter_complex` syntax)
  - `regenerate_status` correctly classifies known backend combinations
  - Font preflight raises clearly when a referenced font path is missing
- `ruff check .` passes.
- `lofivid licenses` prints without errors and includes the new rows.
- `lofivid generate -c configs/smoke_30sec.yaml` completes end-to-end (this is the new smoke; no library, no fonts beyond what the smoke style references — and the smoke style references none because brand layers / HUD are disabled). ffprobe confirms the 30-second 640×360 output.
- `lofivid generate -c configs/2026-04-30_morning_cafe_30min.yaml` completes end-to-end with a small library of test WAVs in `assets/music/cafe_jazz/<mood>/` and the three font files in `assets/fonts/`. ffprobe confirms duration ± 1s, resolution + fps match config. Visual inspection of the first 30s confirms: brand layers visible (kicker / WORK CAFE JAZZ / 工作日 咖啡 爵士節奏 / tracklist strip all rendered correctly), HUD switches at first track boundary, waveform reactive to actual audio.
- `configs/` directory contains exactly two YAML files at the end of the pivot: `smoke_30sec.yaml` and `2026-04-30_morning_cafe_30min.yaml`. No legacy files left.

## Out of scope (do NOT do)

- MCP server. Confirmed not needed for now. Don't add it.
- Channel-description / thumbnail generators. Phase 3.
- `lofivid lock-style` CLI that auto-generates a style from preview output. Manual YAML editing is fine.
- Auto-upload to YouTube.
- Further model swaps (FLUX.2 dev, Qwen-Image, etc.). Stay with what's in `MODEL_OPTIONS.md`.
- Re-creating the deleted demo / medium / anime / photo / jazz / minimal configs as new style+run pairs. Only `_smoke` and `morning_cafe` are required deliverables; the rest is for the user to create later.
- Beat-detection / onset-driven HUD pulsing. The waveform is the music-reactive element. Keep HUD static per-track.
- Per-scene different brand text (e.g. scene 1 says "track 1", scene 2 says "track 2"). HUD already handles per-track text; brand layers are channel-wide.
- A web UI for editing styles. CLI + YAML.
- **Any form of font fallback or graceful degradation.** Missing fonts = hard error. Don't fall back to DejaVu, don't fall back to system fonts, don't ship a manual-install README as a workaround. Obtain the fonts or fail the task.
- **Any form of inline-style fallback in `Config`.** Missing `style_ref` = hard error. The `lofivid/styles/` directory is the only source of truth for visual + music identity.

## Reference: files to read before touching anything

Existing repo state (already done, read for patterns):
- `CLAUDE.md` — conventions, what NOT to redesign
- `MODEL_OPTIONS.md` — commercial-OK model shortlist
- `lofivid/visuals/base.py`, `lofivid/music/base.py` — ABCs
- `lofivid/visuals/unsplash.py`, `lofivid/visuals/overlay_motion.py`, `lofivid/music/suno.py` — recent backend impls; match this style
- `lofivid/visuals/_grading.py` — duotone + paper-border; HUD/brand reuse PIL patterns
- `lofivid/pipeline.py` — backend dispatch (gets refactored to registry)
- `lofivid/compose/ffmpeg_ops.py` — concat + overlay; HUD/waveform integrate here
- `lofivid/compose/timeline.py` — scene scheduling; reuse pattern for track timeline
- `lofivid/cache.py` — content-addressed cache; new style hash participates in keys
- `scripts/preview_workday_cafe.py` — the reference for brand layer typography (kicker / display title / CJK subtitle / tracklist strip — all four need to reproduce in the new `brand_layers` system)
- `scripts/preview_brand_variants.py` — reference for colour palette options the user iterated
- `scripts/preview_music.py` — reference for how the style's `music_anchor` + `music_variations` interact with backends; the "morning" / "cafe_jazz" / "greenhouse" directions there map directly to candidate styles

External docs to skim:
- FFmpeg `showwaves` filter — https://ffmpeg.org/ffmpeg-filters.html#showwaves
- Pillow font handling — https://pillow.readthedocs.io/en/stable/reference/ImageFont.html (specifically the `getlength` / `getbbox` / `chars` interaction for the per-glyph fallback logic)
- mutagen — https://mutagen.readthedocs.io/ (use the format-agnostic `mutagen.File(path, easy=True)` API; titles are at `tags["title"][0]`)
