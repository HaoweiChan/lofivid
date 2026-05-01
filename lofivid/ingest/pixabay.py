"""Pixabay Music ingest source.

Pixabay Content License is CC0-equivalent for commercial use — no attribution
required, monetisation explicitly allowed. We still capture `original_url` in
the sidecar for traceability.

API contract (best-effort, see step 0a in AGENT_PIVOT_PROMPT_v3.md)
-------------------------------------------------------------------
Pixabay's image / video APIs are well-documented; the music endpoint is less
so. We call:

    GET <PIXABAY_API_BASE>/?key=<KEY>&q=<tags>&min_duration=<s>&max_duration=<s>&per_page=<n>

PIXABAY_API_BASE defaults to `https://pixabay.com/api/music`. Override with the
env var if Pixabay moves the endpoint or the user wants a wrapper.

Response is expected to look like:

    {
      "total":     <int>,
      "totalHits": <int>,
      "hits": [
        {
          "id":           <int>,
          "title":        "<track title>",
          "artist":       "<optional explicit artist field>",
          "user":         "<uploader handle, fallback when artist absent>",
          "duration":     <int seconds>,
          "audio_files": [
            {"format": "wav", "url": "...", "size": ...},
            {"format": "mp3", "url": "...", "size": ...},
          ],
          "page_url":     "https://pixabay.com/music/<id>/",
          ...
        },
        ...
      ]
    }

Field names will vary if Pixabay redesigns the response. Each unknown field is
defended with `.get(...)` rather than `[...]` so the adapter degrades gracefully
to "skipped: missing required field" instead of crashing.

Authentication
--------------
A free Pixabay key is enough (100 requests/60s). Set `PIXABAY_API_KEY` in `.env`.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from lofivid._ffmpeg import ffmpeg_bin
from lofivid.ingest.base import (
    IngestedTrack,
    IngestSource,
    register,
    slugify_filename,
    tag_audio,
    unique_audio_path,
    write_sidecar,
)

log = logging.getLogger(__name__)

DEFAULT_API_BASE = os.environ.get(
    "PIXABAY_API_BASE", "https://pixabay.com/api/music"
).rstrip("/")
DEFAULT_RATE_LIMIT_S = 0.7
DEFAULT_MAX_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT_S = 60
PIXABAY_LICENSE_NAME = "pixabay-content-license"


class PixabayIngestSource(IngestSource):
    """Fetch lofi / jazz / ambient tracks from Pixabay's music API."""

    name = "pixabay"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str = DEFAULT_API_BASE,
        rate_limit_s: float = DEFAULT_RATE_LIMIT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        request_timeout_s: int = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> None:
        key = api_key or os.environ.get("PIXABAY_API_KEY")
        if not key:
            raise RuntimeError(
                "PixabayIngestSource needs an API key. Set PIXABAY_API_KEY in your "
                "environment (free tier is 100 requests/60s; sign up at "
                "https://pixabay.com/accounts/register/)."
            )
        self.api_key = key
        self.api_base = api_base.rstrip("/")
        self.rate_limit_s = rate_limit_s
        self.max_retries = max_retries
        self.request_timeout_s = request_timeout_s

    def fetch(
        self,
        mood_tags: list[str],
        count: int,
        target_dir: Path,
        min_duration_s: float = 60.0,
        max_duration_s: float = 600.0,
        already_downloaded: set[str] | None = None,
    ) -> list[IngestedTrack]:
        if count <= 0:
            return []
        target_dir.mkdir(parents=True, exist_ok=True)
        skipped: set[str] = set(already_downloaded or set())

        # Over-fetch slightly so post-filter rejects don't starve us.
        per_page = max(count + len(skipped), 10)
        per_page = min(per_page, 200)  # Pixabay caps per_page at 200 for image API

        query = " ".join(t for t in mood_tags if t).strip() or "lofi"
        params = {
            "key": self.api_key,
            "q": query,
            "min_duration": int(min_duration_s),
            "max_duration": int(max_duration_s),
            "per_page": per_page,
        }
        log.info("Pixabay: searching for %r (per_page=%d)", query, per_page)
        data = self._http_get_json(self.api_base, params=params)
        hits = data.get("hits") or []
        if not hits:
            log.warning(
                "Pixabay: zero hits for query %r (total=%s totalHits=%s). "
                "Try broader tags or check your API key.",
                query,
                data.get("total"),
                data.get("totalHits"),
            )
            return []

        out: list[IngestedTrack] = []
        for hit in hits:
            if len(out) >= count:
                break
            track = self._download_one(hit, target_dir, skipped)
            if track is None:
                continue
            out.append(track)
            skipped.add(track.source_id)
            if self.rate_limit_s > 0:
                time.sleep(self.rate_limit_s)

        log.info("Pixabay: fetched %d new tracks (skipped %d already-present)",
                 len(out), len(skipped) - len(out))
        return out

    # ---------- per-hit handling ----------------------------------------

    def _download_one(
        self,
        hit: dict[str, Any],
        target_dir: Path,
        skipped: set[str],
    ) -> IngestedTrack | None:
        source_id = self._coerce_source_id(hit)
        if not source_id:
            log.warning("Pixabay: skipping hit with no id: %s", hit)
            return None
        if source_id in skipped:
            log.debug("Pixabay: skipping %s (already downloaded)", source_id)
            return None

        title = (hit.get("title") or "").strip() or f"track_{source_id}"
        artist = (hit.get("artist") or hit.get("user") or "").strip() or None
        duration = float(hit.get("duration") or 0.0)
        page_url = (
            hit.get("page_url")
            or hit.get("pageURL")
            or f"https://pixabay.com/music/{source_id}/"
        )
        certificate_url = _extract_certificate_url(hit)

        audio_url, audio_format = _pick_audio_url(hit.get("audio_files") or [])
        if not audio_url:
            log.warning("Pixabay: hit %s has no usable audio_files", source_id)
            return None

        slug = slugify_filename(title, fallback=f"pixabay_{source_id}")
        target_audio = unique_audio_path(target_dir, slug, source_id, ext=".wav")

        if audio_format == "wav":
            self._download_to(audio_url, target_audio)
        else:
            tmp = target_audio.with_suffix(f".raw.{audio_format}")
            self._download_to(audio_url, tmp)
            try:
                _transcode_to_wav(tmp, target_audio)
            finally:
                tmp.unlink(missing_ok=True)

        tag_audio(target_audio, title=title, artist=artist)

        sidecar = write_sidecar(
            target_audio,
            source=self.name,
            source_id=source_id,
            license=PIXABAY_LICENSE_NAME,
            attribution_text=None,
            original_url=page_url,
            license_certificate_url=certificate_url,
        )
        log.info(
            "Pixabay: %s → %s (%.1fs, %s, certificate=%s)",
            source_id, target_audio.name, duration, audio_format.upper(),
            "yes" if certificate_url else "no",
        )
        return IngestedTrack(
            title=title,
            artist=artist,
            duration_s=duration,
            source=self.name,
            source_id=source_id,
            original_url=page_url,
            license=PIXABAY_LICENSE_NAME,
            attribution_text=None,
            local_path=target_audio,
            sidecar_path=sidecar,
            license_certificate_url=certificate_url,
        )

    @staticmethod
    def _coerce_source_id(hit: dict[str, Any]) -> str:
        sid = hit.get("id") or hit.get("track_id") or hit.get("hash")
        return str(sid) if sid is not None else ""

    # ---------- HTTP ----------------------------------------------------

    def _http_get_json(self, url: str, *, params: dict[str, Any]) -> dict[str, Any]:
        body = self._http_request("GET", url, params=params)
        return _decode_json(body)

    def _download_to(self, url: str, dest: Path) -> None:
        body = self._http_request("GET", url)
        dest.write_bytes(body)

    def _http_request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        try:
            import requests
        except ImportError as e:
            raise RuntimeError(
                "Pixabay ingest needs the `requests` library. "
                "Install with `pip install requests`."
            ) from e

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.request(
                    method, url,
                    params=params,
                    timeout=self.request_timeout_s,
                    headers={"Accept": "application/json"},
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                self._sleep_backoff(attempt, why=f"transport: {e}")
                continue

            if resp.status_code == 401:
                raise RuntimeError("Pixabay 401 Unauthorized — check PIXABAY_API_KEY.")
            if resp.status_code == 429:
                # Pixabay returns 429 when the per-minute quota is hit.
                last_exc = RuntimeError("Pixabay 429 Rate Limited")
                self._sleep_backoff(attempt, why="HTTP 429")
                continue
            if 400 <= resp.status_code < 500:
                raise RuntimeError(
                    f"Pixabay {resp.status_code}: {resp.text[:200]} "
                    "(fail-fast on 4xx; verify endpoint shape per step 0a "
                    "of AGENT_PIVOT_PROMPT_v3.md)"
                )
            if resp.status_code >= 500:
                last_exc = RuntimeError(f"Pixabay {resp.status_code}: {resp.text[:200]}")
                self._sleep_backoff(attempt, why=f"server: HTTP {resp.status_code}")
                continue
            return resp.content

        raise RuntimeError(
            f"Pixabay {method} {url} failed after {self.max_retries} retries: {last_exc}"
        )

    @staticmethod
    def _sleep_backoff(attempt: int, *, why: str) -> None:
        delay = min(2 ** attempt, 8)
        log.warning("Pixabay retry %d in %ds (%s)", attempt + 1, delay, why)
        time.sleep(delay)


# ---------- helpers ---------------------------------------------------------

def _extract_certificate_url(hit: dict[str, Any]) -> str | None:
    """Pull the license-certificate URL out of a Pixabay /music response hit.

    Pixabay flags Content ID-registered tracks with a downloadable certificate.
    The exact field name in the music API isn't publicly documented, so we
    look for several common shapes and degrade to None when absent.

    None means: legally fine to use, but if YouTube fires an automated
    Content ID claim against you, you'll be disputing it without
    Pixabay-issued proof — the dispute usually still wins on the strength
    of the track URL + license summary, but it's slower.
    """
    direct = (
        hit.get("license_certificate_url")
        or hit.get("certificate_url")
        or hit.get("certificateUrl")
        or hit.get("content_id_certificate")
    )
    if direct:
        return str(direct).strip() or None
    nested = hit.get("license") or hit.get("certificate")
    if isinstance(nested, dict):
        for k in ("certificate_url", "url", "download_url"):
            v = nested.get(k)
            if v:
                return str(v).strip() or None
    return None


def _pick_audio_url(audio_files: list[dict[str, Any]]) -> tuple[str | None, str]:
    """Return (url, format) for the best download option. WAV preferred, MP3 fallback.

    Pixabay's `audio_files` list element shape is uncertain — defend with .get()."""
    wav: tuple[str, str] | None = None
    mp3: tuple[str, str] | None = None
    other: tuple[str, str] | None = None
    for af in audio_files:
        url = af.get("url") or af.get("audio_url")
        fmt = (af.get("format") or "").lower()
        if not url:
            continue
        if fmt == "wav":
            wav = (url, fmt)
        elif fmt in {"mp3", "mpeg"}:
            mp3 = (url, "mp3")
        elif other is None:
            other = (url, fmt or "unknown")
    if wav is not None:
        return wav
    if mp3 is not None:
        return mp3
    if other is not None:
        return other
    return None, ""


def _decode_json(body: bytes) -> dict[str, Any]:
    import json
    return json.loads(body.decode("utf-8"))


def _transcode_to_wav(src: Path, dst: Path, *, target_sr: int = 44100) -> None:
    cmd = [
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(src),
        "-ar", str(target_sr),
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


register(PixabayIngestSource.name, PixabayIngestSource)
