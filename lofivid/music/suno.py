"""Suno cloud music backend.

This is the only **non-local** component in the pipeline. It calls
out to the Suno HTTP API (via a third-party wrapper) to generate music
with optional vocals — something neither ACE-Step nor MusicGen can do
in 2026.

Wrapper choice & legal status
-----------------------------
Suno does not publish a stable public API for music generation; access
is gated by the official iOS/web client. Several third-party wrappers
expose the same endpoints by reverse-engineering the protocol:

  • `sunoapi.org` (default here) — requires you to provide your own
    Suno session cookie or API key obtained from a paid plan.
  • `PiAPI`           — multi-vendor aggregator, paid tier.
  • `AIML API`        — multi-vendor aggregator, paid tier.

Using any of these wrappers is a **legal grey area**. Suno's official
ToS prohibits scraping, automation, and unofficial API access. We do
not bundle credentials or wrapper implementations; the user must:

  1. Acquire their own paid Suno subscription.
  2. Verify their subscription tier permits commercial output.
  3. Verify the wrapper's terms with their use case.
  4. Accept that the wrapper's endpoints can change without warning.

Copyright caveat
----------------
As of 2026, the U.S. Copyright Office position is that **fully
AI-generated music is not eligible for copyright**. To claim copyright
on output from this backend, write your own lyrics (see `lyrics:` in
the YAML config) and meaningfully edit the stems before publishing.
This is reflected in `lofivid licenses` output and the README.

API contract (sunoapi.org-shaped)
---------------------------------
We talk to a generic generate/poll/download flow:

  POST  /api/v1/generate       → returns task_id
  GET   /api/v1/get?ids=...    → returns status + audio_url(s) when done
  GET   <audio_url>            → returns MP3 / WAV bytes

The exact endpoint paths and field names drift between wrapper providers
and revisions. Treat the constants below as starting points; if your
chosen provider's shape differs, override `SUNO_API_BASE` via the env
var or pass a different one in the constructor.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from lofivid._ffmpeg import ffmpeg_bin
from lofivid.music.base import GeneratedTrack, MusicBackend, TrackSpec

log = logging.getLogger(__name__)

DEFAULT_API_BASE = os.environ.get("SUNO_API_BASE", "https://api.sunoapi.org").rstrip("/")
DEFAULT_GENERATE_PATH = "/api/v1/generate"
DEFAULT_POLL_PATH = "/api/v1/get"
DEFAULT_MODEL_VERSION = "v3.5"


class SunoMusicBackend(MusicBackend):
    """Cloud Suno music generation. See module docstring for legal caveats."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str = DEFAULT_API_BASE,
        model_version: str = DEFAULT_MODEL_VERSION,
        poll_interval_s: float = 4.0,
        hard_timeout_s: float = 300.0,           # 5 min — Suno typically 30-60s
        max_retries: int = 3,
    ) -> None:
        key = api_key or os.environ.get("SUNO_API_KEY")
        if not key:
            raise RuntimeError(
                "SunoMusicBackend needs an API key. "
                "Set SUNO_API_KEY in your environment. "
                "Suno does not have an official public API; you must use a "
                "third-party wrapper (sunoapi.org, PiAPI, AIML API) and verify "
                "your Suno subscription tier permits commercial use."
            )
        self.api_key = key
        self.api_base = api_base.rstrip("/")
        self.model_version = model_version
        self.poll_interval_s = poll_interval_s
        self.hard_timeout_s = hard_timeout_s
        self.max_retries = max_retries
        # Bake model_version into `name` so swapping versions invalidates the
        # cache (the backend name participates in every track's cache key).
        self.name = f"suno-{model_version}"

    def warmup(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def generate(self, spec: TrackSpec, output_dir: Path) -> GeneratedTrack:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{spec.track_index:03d}.wav"
        instrumental = spec.lyrics is None

        log.info(
            "Suno track %d (%s, model=%s, instrumental=%s): %r",
            spec.track_index,
            f"{spec.duration_seconds}s",
            self.model_version,
            instrumental,
            spec.prompt[:80],
        )

        task_id = self._submit_generation(spec, instrumental=instrumental)
        log.info("Suno task %s submitted; polling for completion", task_id)
        audio_url = self._await_completion(task_id)

        # Suno typically returns MP3; we transcode to WAV in-place. Even if
        # it returns WAV the re-encode is a no-op-ish copy and normalises
        # sample rate to 44.1 kHz which the downstream mixer expects.
        raw_bytes = self._http_get_bytes(audio_url)
        tmp_path = out_path.with_suffix(".raw")
        tmp_path.write_bytes(raw_bytes)
        try:
            _transcode_to_wav(tmp_path, out_path, target_sr=44100)
        finally:
            tmp_path.unlink(missing_ok=True)

        actual = _probe_duration_seconds(out_path)
        title = f"Track {spec.track_index + 1:02d}"
        return GeneratedTrack(
            spec=spec, path=out_path, sample_rate=44100,
            actual_duration_seconds=actual,
            title=title, artist="Suno AI",
        )

    # ---------- API plumbing --------------------------------------------

    def _submit_generation(self, spec: TrackSpec, *, instrumental: bool) -> str:
        payload: dict[str, Any] = {
            "prompt": spec.prompt,
            "make_instrumental": instrumental,
            "model": self.model_version,
            # Hint duration; many Suno wrappers ignore this and return
            # ~120s clips. We rely on the downstream mixer to handle real
            # duration via probe_duration_seconds.
            "duration": spec.duration_seconds,
            "seed": spec.seed,
            "title": f"lofivid_{spec.track_index:03d}",
        }
        if not instrumental and spec.lyrics:
            payload["lyrics"] = spec.lyrics
            payload["custom_lyrics"] = spec.lyrics  # alt key some wrappers use

        data = self._http_post_json(
            self.api_base + DEFAULT_GENERATE_PATH,
            payload=payload,
        )
        # Wrappers vary: some return {data: {task_id}}, some {id}, some
        # {task_id}. Try the common shapes.
        task_id = (
            data.get("task_id")
            or data.get("id")
            or (data.get("data") or {}).get("task_id")
            or (data.get("data") or {}).get("id")
        )
        if not task_id:
            raise RuntimeError(
                f"Suno generate response missing task id: {data}. "
                "Check the wrapper's response shape and update suno.py."
            )
        return str(task_id)

    def _await_completion(self, task_id: str) -> str:
        deadline = time.time() + self.hard_timeout_s
        while time.time() < deadline:
            data = self._http_get_json(
                self.api_base + DEFAULT_POLL_PATH,
                params={"ids": task_id, "taskId": task_id},
            )
            entries = self._extract_entries(data)
            if not entries:
                time.sleep(self.poll_interval_s)
                continue
            entry = entries[0]
            status = (entry.get("status") or "").lower()
            audio_url = (
                entry.get("audio_url")
                or entry.get("audioUrl")
                or entry.get("audio")
                or (entry.get("data") or {}).get("audio_url")
            )
            if status in {"failed", "error"}:
                raise RuntimeError(
                    f"Suno task {task_id} failed: {entry.get('error') or entry}"
                )
            if audio_url and status in {"complete", "completed", "success", "succeeded", ""}:
                return audio_url
            log.debug("Suno task %s status=%s", task_id, status or "<unknown>")
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            f"Suno task {task_id} did not finish within {self.hard_timeout_s:.0f}s. "
            "Suno latency is typically 30-60s; >5min usually means a stuck job."
        )

    @staticmethod
    def _extract_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Wrappers return either a list, a {data: [...]}, or a single object."""
        if isinstance(data, list):
            return [e for e in data if isinstance(e, dict)]
        inner = data.get("data") or data.get("clips") or data.get("results") or data
        if isinstance(inner, list):
            return [e for e in inner if isinstance(e, dict)]
        if isinstance(inner, dict):
            return [inner]
        return []

    # ---------- HTTP -----------------------------------------------------

    def _http_post_json(self, url: str, *, payload: dict[str, Any]) -> dict[str, Any]:
        body = self._http_request("POST", url, json=payload)
        return _decode_json(body)

    def _http_get_json(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        body = self._http_request("GET", url, params=params)
        return _decode_json(body)

    def _http_get_bytes(self, url: str) -> bytes:
        return self._http_request("GET", url)

    def _http_request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        try:
            import requests
        except ImportError as e:
            raise RuntimeError(
                "The Suno backend needs the `requests` library. "
                "Install with `pip install requests`."
            ) from e

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.request(
                    method, url,
                    headers=headers,
                    json=json,
                    params=params,
                    timeout=60,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                self._sleep_backoff(attempt, why=f"transport: {e}")
                continue

            if resp.status_code == 401:
                raise RuntimeError("Suno 401 Unauthorized — check SUNO_API_KEY.")
            if resp.status_code == 402:
                raise RuntimeError("Suno 402 Payment Required — wrapper credits exhausted.")
            if resp.status_code == 403:
                raise RuntimeError(
                    "Suno 403 Forbidden — your subscription tier may not permit "
                    "this model or this API access. Verify with the wrapper."
                )
            if 400 <= resp.status_code < 500:
                raise RuntimeError(
                    f"Suno {resp.status_code}: {resp.text[:200]} (fail-fast on 4xx)"
                )
            if resp.status_code >= 500:
                last_exc = RuntimeError(f"Suno {resp.status_code}: {resp.text[:200]}")
                self._sleep_backoff(attempt, why=f"server: HTTP {resp.status_code}")
                continue
            return resp.content
        raise RuntimeError(
            f"Suno {method} {url} failed after {self.max_retries} retries: {last_exc}"
        )

    @staticmethod
    def _sleep_backoff(attempt: int, *, why: str) -> None:
        delay = min(2 ** attempt, 8)
        log.warning("Suno retry %d in %ds (%s)", attempt + 1, delay, why)
        time.sleep(delay)


# ---------- helpers (module-level so tests can patch independently) -------

def _decode_json(body: bytes) -> dict[str, Any]:
    import json
    return json.loads(body.decode("utf-8"))


def _transcode_to_wav(src: Path, dst: Path, *, target_sr: int = 44100) -> None:
    """Force-convert any Suno audio response to 44.1 kHz stereo WAV."""
    cmd = [
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(src),
        "-ar", str(target_sr),
        "-ac", "2",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _probe_duration_seconds(path: Path) -> float:
    """Best-effort duration probe, mirrors the ACE-Step helper."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception as e:
        log.warning("Could not probe duration of %s: %s", path, e)
        return 0.0
