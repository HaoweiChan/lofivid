"""IngestSource ABC + IngestedTrack dataclass + sidecar I/O helpers.

Each ingest source returns a list of `IngestedTrack` records and is responsible
for writing both the audio file (WAV) and the attribution sidecar JSON.

Sidecar contract (one per WAV, stored as `<filename_stem>.attribution.json`):

    {
      "source":                  "pixabay" | "fma" | "manual",
      "source_id":               "<source-stable id, used for idempotency>",
      "license":                 "pixabay-content-license" | "cc0" | "cc-by-4.0" | "manual-licensed",
      "attribution_text":        null | "<text the user copies into video description>",
      "original_url":            "<URL the file came from, or local note for manual>",
      "license_certificate_url": null | "<URL of the downloadable license certificate>",
      "fetched_at":              "<RFC3339 UTC timestamp>"
    }

`license_certificate_url` is set when the source provides a downloadable
proof-of-license (e.g. Pixabay issues these for tracks the uploader has
registered with YouTube's Content ID). Tracks without a certificate are
still legally usable under their license; the URL just gives you faster,
more authoritative proof when disputing automated Content ID claims on
YouTube. Nullable for sources / tracks that don't issue certificates.

The sidecar holds *only* license / attribution / source metadata. Title / artist /
duration go into the WAV's audio metadata tags via mutagen — different surfaces,
different invalidation rules.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ---------- public dataclasses ---------------------------------------------

@dataclass(frozen=True)
class IngestedTrack:
    """One downloaded (or validated) track with its provenance."""
    title: str
    artist: str | None
    duration_s: float
    source: str             # "pixabay" | "fma" | "manual"
    source_id: str          # source-specific stable ID for dedup
    original_url: str       # where this track was downloaded from (or local note)
    license: str            # "pixabay-content-license" | "cc0" | "cc-by-4.0" | "manual-licensed"
    attribution_text: str | None
    local_path: Path
    sidecar_path: Path
    # Optional URL to a downloadable license certificate. Populated for
    # Pixabay tracks the uploader registered with YouTube's Content ID;
    # used as primary proof when disputing automated Content ID claims.
    license_certificate_url: str | None = None


class IngestSource(ABC):
    """Adapter that fetches (or validates) tracks from one external library."""

    name: str  # stable; used in CLI flag and sidecar `source` field

    @abstractmethod
    def fetch(
        self,
        mood_tags: list[str],
        count: int,
        target_dir: Path,
        min_duration_s: float = 60.0,
        max_duration_s: float = 600.0,
        already_downloaded: set[str] | None = None,
    ) -> list[IngestedTrack]:
        """Search this source for tracks matching `mood_tags`, fetch up to `count` new
        ones to `target_dir`, return what was actually fetched.

        `already_downloaded` is a set of `source_id` values to skip — supplied by the
        CLI from existing sidecars in `target_dir`. Implementations must not exceed
        `count` newly-fetched tracks per call.
        """


# ---------- sidecar I/O ----------------------------------------------------

SIDECAR_SUFFIX = ".attribution.json"


def sidecar_path_for(audio_path: Path) -> Path:
    """Return the sidecar path for an audio file. Stem-based — survives rename pairs."""
    return audio_path.with_suffix(SIDECAR_SUFFIX)


def write_sidecar(
    audio_path: Path,
    *,
    source: str,
    source_id: str,
    license: str,
    attribution_text: str | None,
    original_url: str,
    license_certificate_url: str | None = None,
    fetched_at: str | None = None,
) -> Path:
    """Write the attribution sidecar next to `audio_path`. Returns the sidecar path."""
    side = sidecar_path_for(audio_path)
    side.write_text(
        json.dumps(
            {
                "source": source,
                "source_id": source_id,
                "license": license,
                "attribution_text": attribution_text,
                "original_url": original_url,
                "license_certificate_url": license_certificate_url,
                "fetched_at": fetched_at or _utc_now_iso(),
            },
            indent=2,
        )
        + "\n"
    )
    return side


def read_sidecar(audio_path: Path) -> dict | None:
    """Return parsed sidecar dict, or None if missing / malformed."""
    side = sidecar_path_for(audio_path)
    if not side.exists():
        return None
    try:
        return json.loads(side.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Could not read sidecar %s: %s", side, e)
        return None


def existing_source_ids(target_dir: Path) -> set[str]:
    """Scan `target_dir` for sidecars and return the set of `source_id` values."""
    if not target_dir.exists():
        return set()
    out: set[str] = set()
    for p in target_dir.iterdir():
        if not p.name.endswith(SIDECAR_SUFFIX):
            continue
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sid = data.get("source_id")
        if sid:
            out.add(str(sid))
    return out


# ---------- audio metadata tagging -----------------------------------------

def tag_audio(
    path: Path,
    *,
    title: str,
    artist: str | None,
) -> None:
    """Write title + artist into the file's metadata tags via mutagen.

    No-op (with a warning) if mutagen isn't installed. Duration is NOT written —
    audio decoders compute it from the stream; storing it as a tag is redundant
    and decoder-dependent.
    """
    try:
        import mutagen
    except ImportError:
        log.warning("mutagen not installed; cannot tag %s", path)
        return
    try:
        f = mutagen.File(str(path), easy=True)
        if f is None:
            log.warning("mutagen could not open %s for tagging", path)
            return
        f["title"] = title
        if artist is not None:
            f["artist"] = artist
        f.save()
    except Exception as e:  # pragma: no cover — defensive: format-specific quirks
        log.warning("Could not tag %s: %s", path, e)


# ---------- filename helpers -----------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_filename(title: str, fallback: str = "track") -> str:
    """Lower-case, collapse non-alnum runs to '_', strip leading/trailing '_'.

    Mirrors `lofivid.music.library.slugify` — kept local so the ingest layer
    has no dependency on the music backend module beyond the sidecar contract.
    """
    s = _SLUG_RE.sub("_", title.lower()).strip("_")
    return s or fallback


def unique_audio_path(target_dir: Path, slug: str, source_id: str, ext: str = ".wav") -> Path:
    """Return a non-colliding path: <target_dir>/<slug><ext>, or <slug>_<source_id><ext>."""
    candidate = target_dir / f"{slug}{ext}"
    if not candidate.exists():
        return candidate
    return target_dir / f"{slug}_{source_id}{ext}"


# ---------- registry --------------------------------------------------------

_SOURCES: dict[str, type[IngestSource]] = {}


def register(name: str, cls: type[IngestSource]) -> None:
    """Register an ingest source class under a stable name."""
    if name in _SOURCES and _SOURCES[name] is not cls:
        raise RuntimeError(f"ingest source {name!r} is already registered to a different class")
    _SOURCES[name] = cls


def get(name: str) -> type[IngestSource]:
    """Return the registered class for `name`, or raise with the available list."""
    if name not in _SOURCES:
        available = sorted(_SOURCES.keys()) or ["<none registered>"]
        raise ValueError(
            f"unknown ingest source {name!r}; available: {', '.join(available)}"
        )
    return _SOURCES[name]


def available_sources() -> list[str]:
    return sorted(_SOURCES.keys())


# ---------- internals -------------------------------------------------------

def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
