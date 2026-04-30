"""Library music backend — uses pre-licensed audio files (Epidemic Sound etc.).

Folder convention: <library_dir>/<mood_slug>/<title>.wav

mood_slug = slugify(TrackSpec.mood or inferred from prompt tags).
slugify rule: lowercase, non-alnum -> "_", collapse runs of "_".
Round-robin mode falls back to listing the top-level library_dir
without mood subfolders.

Track selection determinism: sort the shortlist alphabetically, pick
shortlist[spec.seed % len(shortlist)]. Same seed + same library = same track.

Cache key contribution: backend.name plus the resolved path AND a content hash
of the first 1 MB of the file (full hash is overkill on long files).

Title/artist: read via mutagen.File(path, easy=True). Fall back to filename
stem for title and None for artist if metadata is absent.
"""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from lofivid._ffmpeg import ffmpeg_bin
from lofivid.music.base import GeneratedTrack, MusicBackend, TrackSpec

log = logging.getLogger(__name__)


def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    return s.strip("_")


class LibraryMusicBackend(MusicBackend):
    name = "library"

    def __init__(
        self,
        library_dir: Path | str,
        match_by: Literal["mood", "round_robin"] = "mood",
        prefer_extensions: tuple[str, ...] = (".wav", ".flac", ".mp3"),
    ) -> None:
        self.library_dir = Path(library_dir).expanduser()
        self.match_by = match_by
        self.prefer_extensions = tuple(e.lower() for e in prefer_extensions)

    def _shortlist(self, spec: TrackSpec) -> list[Path]:
        if self.match_by == "round_robin":
            search_dir = self.library_dir
        else:
            mood_slug = self._infer_mood_slug(spec)
            search_dir = self.library_dir / mood_slug if mood_slug else self.library_dir
            if not search_dir.exists():
                raise RuntimeError(
                    f"LibraryMusicBackend: mood directory {search_dir} not found. "
                    f"Expected layout: {self.library_dir}/<mood_slug>/<title>.wav. "
                    f"Mood slug derived from prompt: {mood_slug!r}. "
                    "Either create the directory or switch to match_by='round_robin'."
                )
        files = [
            p for p in sorted(search_dir.iterdir())
            if p.is_file() and p.suffix.lower() in self.prefer_extensions
        ]
        if not files:
            raise RuntimeError(
                f"LibraryMusicBackend: no audio files in {search_dir} "
                f"(extensions tried: {self.prefer_extensions})."
            )
        return files

    def _infer_mood_slug(self, spec: TrackSpec) -> str | None:
        # Prefer the explicit mood field on the spec if set.
        if spec.mood:
            slug = slugify(spec.mood)
            if slug and (self.library_dir / slug).is_dir():
                return slug
        # Fall back to scanning comma-separated prompt tokens for a matching subdir.
        candidates = [slugify(t.strip()) for t in spec.prompt.split(",") if t.strip()]
        for cand in candidates:
            if cand and (self.library_dir / cand).is_dir():
                return cand
        return None

    def generate(self, spec: TrackSpec, output_dir: Path) -> GeneratedTrack:
        output_dir.mkdir(parents=True, exist_ok=True)
        shortlist = self._shortlist(spec)
        chosen = shortlist[spec.seed % len(shortlist)]
        out_path = output_dir / f"{spec.track_index:03d}.wav"

        if chosen.suffix.lower() == ".wav":
            shutil.copy2(chosen, out_path)
        else:
            _transcode_to_wav(chosen, out_path, target_sr=44100)

        title, artist = _read_metadata(chosen)

        try:
            import soundfile as sf
            info = sf.info(str(out_path))
            actual = info.frames / info.samplerate
        except Exception as e:
            log.warning("Could not probe duration of %s: %s", out_path, e)
            actual = float(spec.duration_seconds)

        return GeneratedTrack(
            spec=spec,
            path=out_path,
            sample_rate=44100,
            actual_duration_seconds=actual,
            title=title or chosen.stem,
            artist=artist,
        )


def _read_metadata(path: Path) -> tuple[str | None, str | None]:
    """Return (title, artist) from id3/vorbis/riff tags via mutagen.

    Returns (None, None) if mutagen is missing or file has no tags.
    """
    try:
        import mutagen
    except ImportError:
        log.warning("mutagen not installed; library metadata will fall back to filenames")
        return None, None
    try:
        tags = mutagen.File(str(path), easy=True)
        if tags is None:
            return None, None
        title = (tags.get("title") or [None])[0]
        artist = (tags.get("artist") or [None])[0]
        return title, artist
    except Exception as e:
        log.warning("Could not read metadata from %s: %s", path, e)
        return None, None


def _transcode_to_wav(src: Path, dst: Path, *, target_sr: int = 44100) -> None:
    cmd = [
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(src), "-ar", str(target_sr), "-ac", "2",
        "-c:a", "pcm_s16le", str(dst),
    ]
    subprocess.run(cmd, check=True)


def content_hash_first_mb(path: Path) -> str:
    """SHA-256 of the first 1 MB of file content, hex prefix.

    Cache key participant — see pivot section 3.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1 * 1024 * 1024))
    return h.hexdigest()[:16]
