"""Manual ingest source — generates sidecars for pre-licensed local WAVs.

For audio files you've downloaded outside the pipeline (Epidemic Sound,
your own catalog, etc.). This source does NOT download — it scans
`target_dir` for audio files without sidecars and writes one each, using
license + attribution_text passed in via the constructor.

Existing sidecars are not overwritten. Re-running the same command on a
folder is therefore safe: only newly-added WAVs receive sidecars.
"""
from __future__ import annotations

import logging
from pathlib import Path

from lofivid.ingest.base import (
    IngestedTrack,
    IngestSource,
    register,
    sidecar_path_for,
    write_sidecar,
)

log = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


class ManualIngestSource(IngestSource):
    """Validate-and-tag-only source for pre-licensed local audio."""

    name = "manual"

    def __init__(
        self,
        *,
        license: str = "manual-licensed",
        attribution_text: str | None = None,
        original_url: str = "manual: pre-licensed audio (see assets/music/README.md)",
        license_certificate_url: str | None = None,
    ) -> None:
        self.license = license
        self.attribution_text = attribution_text
        self.original_url = original_url
        self.license_certificate_url = license_certificate_url

    def fetch(
        self,
        mood_tags: list[str],  # noqa: ARG002 — manual source ignores tags
        count: int,            # noqa: ARG002 — manual source ignores count
        target_dir: Path,
        min_duration_s: float = 60.0,    # noqa: ARG002
        max_duration_s: float = 600.0,   # noqa: ARG002
        already_downloaded: set[str] | None = None,  # noqa: ARG002
    ) -> list[IngestedTrack]:
        """Scan `target_dir` for audio files without sidecars; write a sidecar for each.

        Returns the list of tracks for which a NEW sidecar was created. Files that
        already have sidecars are skipped silently — this is what makes re-running
        the command idempotent.
        """
        if not target_dir.exists():
            raise FileNotFoundError(
                f"manual ingest target_dir does not exist: {target_dir}. "
                "Create the directory and drop your pre-licensed WAVs into it first."
            )

        created: list[IngestedTrack] = []
        for audio_path in sorted(target_dir.iterdir()):
            if not audio_path.is_file():
                continue
            if audio_path.suffix.lower() not in _AUDIO_EXTENSIONS:
                continue
            if sidecar_path_for(audio_path).exists():
                continue

            title, artist, duration = _read_existing_metadata(audio_path)
            # source_id for manual = the relative filename. Stable across runs and
            # uniquely identifies this file within the folder. The folder itself is
            # the de-facto namespace.
            source_id = audio_path.name
            sidecar = write_sidecar(
                audio_path,
                source=self.name,
                source_id=source_id,
                license=self.license,
                attribution_text=self.attribution_text,
                original_url=self.original_url,
                license_certificate_url=self.license_certificate_url,
            )
            created.append(
                IngestedTrack(
                    title=title or audio_path.stem,
                    artist=artist,
                    duration_s=duration,
                    source=self.name,
                    source_id=source_id,
                    original_url=self.original_url,
                    license=self.license,
                    attribution_text=self.attribution_text,
                    local_path=audio_path,
                    sidecar_path=sidecar,
                    license_certificate_url=self.license_certificate_url,
                )
            )
            log.info(
                "manual: wrote sidecar for %s (license=%s)",
                audio_path.name,
                self.license,
            )

        return created


def _read_existing_metadata(path: Path) -> tuple[str | None, str | None, float]:
    """Best-effort title/artist/duration via mutagen. (None, None, 0.0) if unavailable."""
    title: str | None = None
    artist: str | None = None
    duration: float = 0.0
    try:
        import mutagen
    except ImportError:
        return title, artist, duration
    try:
        f = mutagen.File(str(path), easy=True)
        if f is None:
            return title, artist, duration
        title = (f.get("title") or [None])[0]
        artist = (f.get("artist") or [None])[0]
        if f.info is not None and getattr(f.info, "length", None):
            duration = float(f.info.length)
    except Exception as e:  # pragma: no cover — defensive
        log.debug("Could not read metadata from %s: %s", path, e)
    return title, artist, duration


register(ManualIngestSource.name, ManualIngestSource)
