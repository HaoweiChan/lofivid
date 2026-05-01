# lofivid — Architecture Pivot v4 (Music Ingest Layer)

## Context

This pivot is **v3-blocked**: do not start work here until the v2 pivot (`AGENT_PIVOT_PROMPT_v2.md`) is verified complete — `pytest -q` green, `lofivid licenses` accurate, both required configs (`smoke_30sec.yaml` and `2026-04-30_morning_cafe_30min.yaml`) rendering end-to-end with brand layers + HUD + waveform visible. The two pivots touch different surfaces but share the `StyleSpec` schema, and v3 extends it. Land v2 cleanly first.

This pivot does one thing: **automate music ingestion** so populating `assets/music/<mood_slug>/` is a CLI command instead of 2–3 hours of manual downloads per mood.

**Pixabay Music** is the primary source. Their Content License explicitly permits commercial monetised use ("You may use Pixabay music in commercial video projects, including content you sell or distribute … as long as the music is part of a larger creative work and not distributed as a standalone file"). A lofivid render — multi-track mix + AI visuals + brand layers + HUD + waveform — is unambiguously a "larger creative work." **Free Music Archive** is a possible secondary (deferred behind step 0b verification). **Manual** stays as the third path for pre-licensed audio you've downloaded yourself (your own catalog, CC0 finds, anything you have proof of license for) — the layer just validates folder structure and writes sidecars.

The non-license risk that *does* matter: **YouTube Content ID claims**. Pixabay's own FAQ acknowledges that some uploaders register their tracks with Content ID to monetise on other platforms; this can trigger automated claims even on properly-licensed use. Claims are not copyright strikes (no channel-termination risk) but they redirect monetisation to the claimant until you dispute. The architecture handles this with two new fields:

- Sidecar `license_certificate_url` — populated for Pixabay tracks the uploader registered with Content ID (their dispute-proof PDF). Null otherwise.
- Manifest `at_risk_for_content_id_claim` — boolean derived from `(source == "pixabay" and license_certificate_url is None)`. Surfaces per-track in `music_attributions` so you can scan a render's risk profile before upload.

Paid services that prohibit scripted ingest (Epidemic Sound, Uppbeat, Soundstripe, Artlist, YouTube Audio Library) are **out of scope**. Their ToS either gates their APIs to integrators only or explicitly forbids automated downloads on user tiers; the ban-then-retroactive-license-loss risk is unacceptable for monetised channels. If you have audio from those services, drop the WAVs into the target folder by hand and use `--source manual` to wrap sidecars.

Read `AGENT_PIVOT_PROMPT_v2.md` first, especially the `LibraryMusicBackend` and `StyleSpec` sections — this pivot extends both. Conventions there carry forward unchanged.

**Scope discipline:** this is *not* a chance to add Mubert, Soundstripe, Artlist, Suno-as-library, or anything else. Three sources only. The architecture makes adding more trivial later, but extra implementations now is scope creep.

## Tasks (in this order — earlier tasks unlock later ones)

### 0. Verification gates (do this BEFORE writing any code)

Three external dependencies must be confirmed before the architectural decisions are committed. If any fails, the plan changes — escalate to the user before proceeding.

**0a. Pixabay Music API exists and is usable.** Pixabay's image/video API at `https://pixabay.com/api/` is well-documented and stable; the music endpoint is less so. Sign up for a free key (free tier is 100 req/60s) and hit it with a `q=lofi` query. Confirm: (i) JSON response with track metadata, (ii) downloadable audio URL in the response, (iii) license field clearly indicating Pixabay Content License or equivalent. If the music endpoint 403s, returns no usable URLs, or has license ambiguity, **stop and escalate**.

**0b. FMA API current state.** FMA was sold to Tribe of Noise in 2018 and the API has had stability issues since. Verify: (i) the documented API base URL still resolves, (ii) a tracks search filtered to `license=cc-by` returns results, (iii) returned tracks have direct download URLs. If FMA is dead or unstable, **drop it from this pivot** — Pixabay alone is enough for v3, and FMA can be revisited if/when the catalog becomes a constraint.

**0c. Pixabay Music tag granularity.** Confirm Pixabay's music search returns useful results for the mood slugs you want (`cafe_afternoon`, `late_night_booth`, `vinyl_spin`, etc.). If their tag taxonomy is too coarse (only "lofi" / "jazz" / "ambient" with no subgenre filtering), the mood→tag mapping in step 6 needs to compensate by using their text search rather than tag filters. Document what you find — it shapes the StyleSpec extension in step 6.

Record findings in `notes/v3_ingest_verification.md` (working memory; doesn't need to be PR'd). Once 0a + 0c pass, proceed. 0b's outcome is "include FMA" or "skip FMA"; both are acceptable.

### 1. `IngestSource` ABC + `IngestedTrack` dataclass

New module `lofivid/ingest/`. Mirror the structure of `lofivid/music/` — base ABC, then per-source implementations.

#### `lofivid/ingest/base.py`

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class IngestedTrack:
    """One downloaded track with its provenance."""
    title: str
    artist: str | None
    duration_s: float
    source: str             # "pixabay" | "fma" | "manual"
    source_id: str          # source-specific stable ID for dedup
    original_url: str       # where this track was downloaded from
    license: str            # "pixabay-content-license" | "cc0" | "cc-by-4.0" | "manual-licensed"
    attribution_text: str | None  # required for cc-by; None for cc0 / pixabay
    local_path: Path        # where the WAV was written
    sidecar_path: Path      # path to <title>.attribution.json
    license_certificate_url: str | None = None
    # ^ Non-null when the source provides a downloadable proof-of-license
    # (Pixabay issues these for tracks registered with YouTube's Content ID).
    # Used as primary evidence when disputing automated Content ID claims.

class IngestSource(ABC):
    """Adapter that fetches tracks from one external library."""
    name: str  # stable; used in CLI flag and sidecar `source` field

    @abstractmethod
    def fetch(self,
              mood_tags: list[str],
              count: int,
              target_dir: Path,
              min_duration_s: float = 60.0,
              max_duration_s: float = 600.0,
              already_downloaded: set[str] | None = None) -> list[IngestedTrack]:
        """Search this source for tracks matching mood_tags, download up to `count`,
        write WAVs + sidecars to target_dir, return what was actually fetched.

        `already_downloaded` is a set of `source_id` values to skip (idempotency).
        Implementations must not exceed `count` newly-fetched tracks per call."""
```

Sidecar JSON schema (one per WAV, written as `<title>.attribution.json`):

```json
{
  "source": "pixabay",
  "source_id": "12345",
  "license": "pixabay-content-license",
  "attribution_text": null,
  "original_url": "https://pixabay.com/music/12345/",
  "license_certificate_url": null,
  "fetched_at": "2026-04-30T14:32:00Z"
}
```

`license_certificate_url` is non-null when the source provides a per-track
proof-of-license URL — Pixabay issues these for tracks the uploader has
registered with YouTube's Content ID. Used as primary evidence in dispute
flows. Null doesn't mean unlicensed; it means "no Pixabay-issued
certificate, dispute relies on track URL + license-summary URL."

`title`, `artist`, `duration_s` go into the WAV's ID3 / Vorbis tags via `mutagen` — **not** into the sidecar. Sidecar holds *only* license/attribution/source metadata. The two surfaces have different invalidation rules: tags edit cleanly with audio metadata tools; sidecars are an append-only audit trail.

`already_downloaded` is computed by the CLI before calling `fetch()`: scan `target_dir` for existing sidecars, collect their `source_id` values, pass through. This makes re-running the same ingest command idempotent.

### 2. `PixabayIngestSource` (`lofivid/ingest/pixabay.py`)

Primary source. Read API key from `PIXABAY_API_KEY` env var; fail loudly if missing.

API call shape (verify in step 0a — this is best-effort against current docs):
- `GET https://pixabay.com/api/music/?key=<KEY>&q=<tags-joined>&min_duration=<s>&max_duration=<s>&per_page=<count>`
- Response includes track metadata + `audio_files` array with download URLs (typically MP3 + WAV options).

Prefer WAV download URL when available; fall back to MP3 + ffmpeg-convert-to-WAV. **Don't ship MP3 in the library** — `LibraryMusicBackend` is format-uniform (WAV).

Filename derivation: `{slugified_title}.wav`. If the slug already exists in `target_dir` (different track, name collision), append `_<source_id>` to disambiguate. Don't overwrite.

Write sidecar with `license: "pixabay-content-license"`, `attribution_text: null` (Pixabay license doesn't require attribution but keep `original_url` for traceability). Tag the WAV with title + artist + duration via `mutagen.File(path, easy=True)`.

Rate limiting: Pixabay free tier is 100 req/60s. Add a 0.7s sleep between downloads as defensive padding; configurable via `--rate-limit-s` CLI flag.

Existing retry-with-backoff pattern from `lofivid/music/suno.py` — copy it.

### 3. `FMAIngestSource` (`lofivid/ingest/fma.py`) — only if 0b passes

Secondary source. **Critical:** FMA has mixed licenses. The adapter MUST filter strictly:
- Accept: `cc0` (no attribution, no constraints)
- Accept: `cc-by-4.0` and `cc-by-3.0` (attribution required — capture into sidecar)
- **Reject** anything else, especially `cc-by-nc` (non-commercial — kills monetisation) and `cc-by-sa` (share-alike — viral license terms unclear for video composition)

If FMA's API doesn't expose a license filter at the query level, fetch broadly and reject post-fetch by license string. Log every rejection so the user can see the catalog shrinkage.

Attribution text format (used by manifest → video description): `"<title>" by <artist>, <license url>`. Example: `"Slow Tuesday" by jellyfish, CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)`.

If 0b failed, **skip this task entirely** and document why in `notes/v3_ingest_verification.md`. Pixabay-only is acceptable for v3; revisit FMA in a later pivot if needed.

### 4. `ManualIngestSource` (`lofivid/ingest/manual.py`)

For pre-licensed audio downloaded outside the pipeline (your own catalog, friend's band, CC0 finds, anything you have proof of license for). This adapter does **not** download — it validates that an existing folder is in the expected shape and writes sidecars where missing.

Behaviour: scan `target_dir` for WAVs without sidecars; for each one, accept license + attribution via CLI flags (`--license cc0 --attribution-text 'Track "X" by Y, CC0' [--license-certificate-url https://...]`), write the sidecar. Existing sidecars are not overwritten.

This makes the "I downloaded 30 tracks into the right folder" workflow one command instead of 30 manual JSON files. The `--license-certificate-url` flag is for cases where you have a downloadable proof-of-license (some paid catalogues issue these); leaving it null is fine when the license itself is the proof.

### 5. CLI: `lofivid music-ingest`

New typer subcommand in `lofivid/cli.py`:

```
lofivid music-ingest \
  --source pixabay \
  --mood cafe_afternoon \
  --count 20 \
  --target assets/music/cafe_jazz/cafe_afternoon/ \
  [--style morning_cafe]   # optional: read mood_tags from style's library_search_tags
  [--min-duration 60]
  [--max-duration 600]
  [--rate-limit-s 0.7]
  [--license manual-licensed]            # only with --source manual
  [--attribution-text "..."]             # only with --source manual
```

If `--style <name>` is passed, the CLI reads the style's `library_search_tags[mood]` mapping (see step 6) and uses that as `mood_tags`. If `--style` is omitted, the CLI uses `[mood]` as the only tag — usable but less precise.

Output: write a one-line summary per fetched track, plus a final count. Exit non-zero if zero tracks fetched (likely indicates API failure or filter mismatch — must not silently succeed).

Per-purpose seed for the "which subset of available results" choice when `count < returned`: `seeds.derive("ingest.{source}.{mood}")`. Defensive determinism — even though ingest isn't part of the render-time chain, it helps when re-creating a library after asset loss.

### 6. `library_search_tags` extension to `StyleSpec`

The style YAML already declares mood slugs implicitly (via `music_variations`). Add an explicit mapping so each mood translates to source-specific search keywords:

```python
# lofivid/styles/schema.py — additive change to StyleSpec
class StyleSpec(BaseModel):
    # ... existing fields from v2 ...
    library_search_tags: dict[str, list[str]] = Field(default_factory=dict)
    """Mapping of mood_slug → search tags for ingest sources.
    Example: {"cafe_afternoon": ["cafe", "morning", "jazz", "lofi"]}
    Empty mapping = ingest CLI must be run with explicit --mood as the only tag."""
```

Update `styles/morning_cafe.yaml` with the mapping for its existing moods. Doesn't affect render-time behaviour — pure ingest hint.

**Style hash computation includes this field** (it's part of style identity — changing search tags means the library you'd ingest is different).

### 7. `LibraryMusicBackend` reads sidecars

Minimal change. When `LibraryMusicBackend` selects a track from `assets/music/<mood>/`, after reading the WAV's audio metadata for title/artist, also try to load `<title>.attribution.json` from the same directory. If present, attach to the `GeneratedTrack` as an optional field:

```python
# lofivid/music/base.py — additive change to GeneratedTrack
@dataclass
class GeneratedTrack:
    # ... existing fields from v2 ...
    attribution: dict | None = None  # raw sidecar dict; None if no sidecar exists
```

`attribution` is the raw sidecar dict (unparsed, since the manifest will serialise it back to JSON anyway). If the sidecar is missing, `attribution: None` — render proceeds normally but manifest's `music_attributions` will lack this track's source info. Log a warning at render time but do **not** fail the render — manual library tracks may legitimately predate the sidecar convention.

### 8. Manifest: `music_attributions` list

Pipeline writes one entry per `GeneratedTrack` used in the final mix:

```json
{
  "music_attributions": [
    {
      "track_title": "Morning Brew",
      "track_artist": "Lo-Fi Cafe Project",
      "track_duration_s": 187.3,
      "source": "pixabay",
      "source_id": "12345",
      "license": "pixabay-content-license",
      "attribution_text": null,
      "original_url": "https://pixabay.com/music/12345/",
      "license_certificate_url": null,
      "at_risk_for_content_id_claim": true
    }
  ]
}
```

`at_risk_for_content_id_claim` is a derived flag — three states:

- `true`  : Pixabay track with no `license_certificate_url`. License permits the use, but if YouTube fires an automated Content ID claim you'll dispute without a Pixabay-issued certificate (still works, just slower).
- `false` : everything else — manual / FMA / Pixabay-with-certificate. Either the user holds the proof (manual) or the source provides one inline.
- `null`  : no sidecar — provenance unknown.

Tracks without sidecars produce an entry with `source: null`, `at_risk_for_content_id_claim: null`, and a warning logged at render time.

### 9. `lofivid licenses` table extension

Add two rows:

| Component | Source | License | Attribution required? | Notes |
|-----------|--------|---------|----------------------|-------|
| Library music (Pixabay) | pixabay.com/api/music/ | Pixabay Content License | No | Commercial OK + monetisation OK as part of a larger creative work. Per-track Content ID risk: see manifest's `at_risk_for_content_id_claim`. Disputable, not a copyright strike. |
| Library music (manual) | user-supplied | user-supplied | depends on source | For audio of any provenance you've licensed yourself. The `--source manual` ingest writes a sidecar with whatever `--license` / `--attribution-text` / `--license-certificate-url` you pass. |
| Library music (FMA / CC-BY) | freemusicarchive.org | CC-BY-4.0 | **Yes** | Attribution text auto-included in manifest's `music_attributions` — must be copied into video description. |

If FMA was skipped in step 3, omit its row.

### 10. `assets/music/README.md` rewrite

Document:
- Folder convention (`assets/music/<mood_slug>/<title>.wav` + `<title>.attribution.json`)
- Recommended workflow: `lofivid music-ingest --source pixabay --style morning_cafe --mood cafe_afternoon --count 20 --target assets/music/cafe_jazz/cafe_afternoon/`
- Manual fallback: how to drop pre-licensed WAVs of any provenance into a folder and run `lofivid music-ingest --source manual --license <name> --attribution-text "<text>" [--license-certificate-url <url>]` to generate sidecars
- License notes: which sources require attribution flow into video descriptions; pointer to manifest's `music_attributions` field, including the `at_risk_for_content_id_claim` flag
- Content ID claim workflow: when claims fire on Pixabay tracks, dispute via track URL + license-summary URL + (when present) `license_certificate_url` from the sidecar
- Idempotency note: re-running ingest with the same params skips already-downloaded tracks (by `source_id`)
- Why scripted ingest from paid catalogues (Epidemic, Uppbeat, Soundstripe, Artlist, YouTube Audio Library) is excluded — ToS / retroactive-license-loss risk; one-paragraph summary, link back to this brief's Context section

Keep it under one page. Operator docs, not a tutorial.

### 11. Tests

New test files:
- `tests/ingest/test_base.py` — `IngestedTrack` dataclass, `IngestSource` ABC contract
- `tests/ingest/test_pixabay.py` — mocked `requests` calls (no live API in unit tests); verify URL construction, sidecar shape, idempotency via `already_downloaded` set, MP3→WAV conversion path when only MP3 URL is returned
- `tests/ingest/test_fma.py` — same shape, plus license-filter rejection coverage (only if FMA included)
- `tests/ingest/test_manual.py` — given a folder with WAVs and no sidecars, verify sidecars are created with correct shape; existing sidecars are not overwritten
- `tests/test_library_music_backend.py` — extend existing test: with sidecar present, `GeneratedTrack.attribution` is populated; without sidecar, it's `None` and a warning is logged
- `tests/test_pipeline.py` — extend manifest test: `music_attributions` is correctly populated; tracks without sidecars produce `source: null` entries

One integration test, opt-in via env flag (`LOFIVID_TEST_LIVE_INGEST=1`):
- `tests/integration/test_pixabay_live.py` — actually hits Pixabay API, fetches 1 track, validates response shape. Skipped by default; run manually to detect API drift.

## Conventions to follow

- Carry forward all v2 conventions (`from __future__ import annotations`, Pydantic `extra="forbid"`, ABC pattern, per-purpose seeds, no heavy deps, comments only when *why* is non-obvious).
- New env var: `PIXABAY_API_KEY` in `.env.example`. FMA may not need a key (verify in 0b); document accordingly.
- New deps: **none**. `requests` already transitive; `mutagen` already added in v2; FFmpeg conversion uses existing `ffmpeg-python` wrapper.
- Sidecars are JSON, not YAML. Python's stdlib `json` is enough; no schema validation library needed (frozen dataclass on the backend side is sufficient).
- Idempotency is checked by `source_id`, not filename. Two sources can legitimately use the same title.
- Logging via `log = logging.getLogger(__name__)`. No print statements outside `scripts/`.
- All ingest sources implement retry-with-backoff matching the existing `suno.py` pattern.
- Per-source `name` attribute is stable and forms part of the sidecar's `source` field — don't change it after release.

## Verification before claiming done

- `pytest -q` passes including new ingest tests.
- `ruff check .` passes.
- `lofivid licenses` prints the new rows correctly.
- `lofivid music-ingest --source pixabay --mood cafe_afternoon --count 3 --target /tmp/test_ingest/` completes end-to-end. Three WAVs land in the target dir with three matching `*.attribution.json` files. Audio metadata tags include title/artist/duration.
- Re-running the same command is a no-op: zero new files, zero downloads beyond the initial search call (logged), exit 0.
- (If FMA included) `lofivid music-ingest --source fma --mood cafe_afternoon --count 3 --target /tmp/test_ingest_fma/` completes; sidecars include populated `attribution_text` for any cc-by tracks.
- `lofivid music-ingest --source manual --target tests/fixtures/manual_music/ --license manual-licensed --attribution-text "Test license"` generates sidecars for fixture WAVs without overwriting existing ones.
- A render using a freshly-ingested library produces a manifest with `music_attributions` populated correctly per track.
- `assets/music/README.md` exists, is under one page, and documents the three-source flow.
- `notes/v3_ingest_verification.md` exists and records what was actually confirmed in step 0.

## Out of scope (do NOT do)

- **Browser automation against Epidemic Sound, Uppbeat, YouTube Audio Library, or any other paid / ToS-restricted service.** The retroactive license invalidation risk on a banned account makes this categorically wrong for monetised use. Not "fragile" — wrong.
- **Mubert, Soundstripe, Artlist as ingest sources.** Mubert is generative and per-minute priced — wrong economic model for library mode; if needed later, add it as a parallel `MusicBackend` (like Suno), not as an `IngestSource`. Soundstripe and Artlist gate their APIs to integrators only; individuals can't get keys.
- **Auto-curation / "ML picks the best 20 tracks for this mood".** Out of scope. Ingest fetches what the source returns; user trims by deleting WAVs they don't want before render.
- **Streaming preview during ingest.** No realtime playback; just download and tag.
- **MP3 in the library.** Convert at ingest time. `LibraryMusicBackend` is format-uniform (WAV).
- **Cross-source dedup.** A track with the same title from Pixabay and FMA is fine to have both copies; the user picks. Don't try to identify "same song, different upload".
- **Auto-embedding attribution into the rendered video frame.** Manifest carries it; user composes YouTube description from manifest. Burning attribution into the frame is a later-pivot decision.
- **Replacing or rewriting `LibraryMusicBackend`.** This pivot extends it (sidecar reads); render-time behaviour is unchanged.
- **A "best source per mood" recommender.** User picks `--source` per command. The CLI does not auto-select.
- **Per-track per-style overrides at ingest time.** Tags come from the style; nothing more granular.
- **Retroactive sidecar generation for tracks not present locally.** `manual` source only writes sidecars for files already on disk.

## Reference: files to read before touching anything

Existing repo state (must already be at v2-complete; read for patterns):
- `AGENT_PIVOT_PROMPT_v2.md` — the prerequisite pivot; especially the `LibraryMusicBackend` and `StyleSpec` sections
- `lofivid/music/library.py` — what this pivot extends with sidecar reading
- `lofivid/music/base.py` — `GeneratedTrack` and `MusicBackend` ABCs; `attribution` field gets added here
- `lofivid/music/suno.py` — reference for cloud-source retry-with-backoff patterns; ingest sources should match this style
- `lofivid/styles/schema.py` — `StyleSpec` gets the new `library_search_tags` field; remember it participates in the style hash
- `lofivid/cli.py` — typer subcommand patterns; `music-ingest` adds one
- `lofivid/pipeline.py` — `_write_manifest` gains `music_attributions`
- `lofivid/seeds.py` — `SeedRegistry.derive()` for the `ingest.{source}.{mood}` purpose
- `notes/v3_ingest_verification.md` — your own findings from step 0; consult before implementation diverges from assumptions

External docs to skim:
- Pixabay API — https://pixabay.com/api/docs/ (image/video docs; music endpoint may be on a separate doc page; verify in step 0a)
- Free Music Archive API — https://freemusicarchive.org/api (verify in step 0b; may be deprecated)
- Pixabay Content License — https://pixabay.com/service/license-summary/
- CC-BY 4.0 — https://creativecommons.org/licenses/by/4.0/
- mutagen `easy=True` interface — https://mutagen.readthedocs.io/
