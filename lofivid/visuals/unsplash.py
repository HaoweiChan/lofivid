"""Unsplash photo-search keyframe backend.

Fetches a still per scene from the Unsplash API instead of generating one
with SDXL. The look is locked in by the preset's `duotone` pair plus the
shared paper-border post-process from `_grading.py`, so a video composed
from Unsplash photos still has a coherent aesthetic.

Auth + attribution
------------------
The API key is read from `UNSPLASH_ACCESS_KEY` at backend construction
time (raise loudly if missing — silent fallbacks lose runs). For each
selected photo we write `<scene_index:03d>.jpg.attribution.txt` next to
the image; downstream stages (channel description, video credits) pick
this up automatically.

Determinism + cache invalidation
--------------------------------
The Unsplash search index drifts over time, so even a fixed prompt+seed
can resolve to a different photo on a re-run. We surface the resolved
photo id via `cache_key_extras`, which means:

  • If the index is unchanged, cache hits, no re-download.
  • If the index has shifted, the photo id changes, the keyframe cache
    key changes, the keyframe is re-fetched, and the parallax stage
    invalidates because the file mtime moves forward.

We memoise the resolved photo per scene_index so we hit the API once
per scene per run (not once per cache check + once per generate).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from lofivid.visuals._grading import RGB, duotone, paper_border
from lofivid.visuals.base import GeneratedImage, KeyframeBackend, KeyframeSpec

log = logging.getLogger(__name__)

API_SEARCH_URL = "https://api.unsplash.com/search/photos"


@dataclass(frozen=True)
class _ResolvedPhoto:
    photo_id: str
    regular_url: str
    photographer: str
    photographer_url: str
    photo_html_url: str


class UnsplashKeyframeBackend(KeyframeBackend):
    name = "unsplash"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        quality_suffix: str = "",
        duotone_pair: tuple[RGB, RGB] | None = None,
        page_size: int = 10,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        key = api_key or os.environ.get("UNSPLASH_ACCESS_KEY")
        if not key:
            raise RuntimeError(
                "UnsplashKeyframeBackend needs an API key. "
                "Set UNSPLASH_ACCESS_KEY in your environment "
                "(register at https://unsplash.com/developers)."
            )
        self.api_key = key
        self.quality_suffix = quality_suffix.strip().strip(",").strip()
        self.duotone_pair = duotone_pair
        self.page_size = page_size
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._resolved: dict[int, _ResolvedPhoto] = {}

    # ---------- KeyframeBackend hooks ------------------------------------

    def cache_key_extras(self, spec: KeyframeSpec) -> dict:
        info = self._resolve(spec)
        return {"unsplash_photo_id": info.photo_id}

    def generate(self, spec: KeyframeSpec, output_dir: Path) -> GeneratedImage:
        output_dir.mkdir(parents=True, exist_ok=True)
        info = self._resolve(spec)
        out_path = output_dir / f"{spec.scene_index:03d}.jpg"

        log.info(
            "Unsplash scene %d → photo %s by %s",
            spec.scene_index, info.photo_id, info.photographer,
        )

        raw_bytes = self._http_get_bytes(info.regular_url)
        with Image.open(BytesIO(raw_bytes)) as img:
            img = img.convert("RGB")
            img = ImageOps.fit(img, (spec.width, spec.height), method=Image.Resampling.LANCZOS)
            if self.duotone_pair is not None:
                shadow, highlight = self.duotone_pair
                img = duotone(img, shadow, highlight)
                img = paper_border(img)
                # paper_border re-pads — re-fit to target so the framed photo
                # fills the keyframe canvas exactly.
                img = ImageOps.fit(img, (spec.width, spec.height), method=Image.Resampling.LANCZOS)
            img.save(out_path, format="JPEG", quality=92)

        # Sidecar attribution file (Unsplash API guideline §"Hotlinking" /
        # photographer credit). Plain text — both the human credit line and
        # a structured JSON for downstream tooling.
        attribution_path = out_path.with_suffix(out_path.suffix + ".attribution.txt")
        with open(attribution_path, "w") as f:
            f.write(
                f"Photo by {info.photographer} ({info.photographer_url}) on Unsplash\n"
                f"{info.photo_html_url}\n"
                f"\n"
                f"# machine-readable\n"
                + json.dumps(
                    {
                        "photo_id": info.photo_id,
                        "photographer": info.photographer,
                        "photographer_url": info.photographer_url,
                        "photo_url": info.photo_html_url,
                        "regular_url": info.regular_url,
                        "source": "unsplash",
                    },
                    indent=2,
                ),
            )

        return GeneratedImage(spec=spec, path=out_path)

    # ---------- internals ------------------------------------------------

    def _strip_quality_suffix(self, prompt: str) -> str:
        """Quality tags like 'ultra detailed, 4k' hurt Unsplash relevance.

        We append them in the preset for SDXL; here we strip them off so
        the search query is just the scene description."""
        q = prompt
        if self.quality_suffix:
            # Drop the suffix wherever it appears (anchored at end is the
            # common case after preset.render_prompt).
            q = q.replace(self.quality_suffix, "")
        # Tidy up trailing whitespace + commas after the chop.
        q = ", ".join(s.strip() for s in q.split(",") if s.strip())
        return q

    def _resolve(self, spec: KeyframeSpec) -> _ResolvedPhoto:
        if spec.scene_index in self._resolved:
            return self._resolved[spec.scene_index]
        query = self._strip_quality_suffix(spec.prompt)
        if not query:
            raise RuntimeError(
                f"Empty Unsplash query for scene {spec.scene_index} after "
                f"stripping the preset quality suffix; check your config."
            )
        photos = self._search(query)
        if not photos:
            raise RuntimeError(
                f"Unsplash returned 0 photos for query {query!r}. "
                "Try a less specific scene prompt."
            )
        idx = spec.seed % min(len(photos), self.page_size)
        photo = photos[idx]
        info = _ResolvedPhoto(
            photo_id=photo["id"],
            regular_url=photo["urls"]["regular"],
            photographer=photo.get("user", {}).get("name", "Unknown"),
            photographer_url=photo.get("user", {}).get("links", {}).get("html", ""),
            photo_html_url=photo.get("links", {}).get("html", ""),
        )
        self._resolved[spec.scene_index] = info
        return info

    # ---------- HTTP -----------------------------------------------------

    def _search(self, query: str) -> list[dict[str, Any]]:
        """Single-page Unsplash search with retries."""
        params = {"query": query, "per_page": self.page_size, "orientation": "landscape"}
        data = self._http_get_json(API_SEARCH_URL, params=params)
        return list(data.get("results", []))

    def _http_get_json(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        body = self._http_get_bytes(url, params=params)
        return json.loads(body.decode("utf-8"))

    def _http_get_bytes(self, url: str, *, params: dict[str, Any] | None = None) -> bytes:
        # Lazy import: requests is a transitive dep of huggingface_hub, but
        # we don't want to fail at module-import time if it's missing.
        try:
            import requests
        except ImportError as e:
            raise RuntimeError(
                "The Unsplash backend needs the `requests` library. "
                "Install with `pip install requests`."
            ) from e

        headers = {"Authorization": f"Client-ID {self.api_key}", "Accept-Version": "v1"}

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=self.timeout_s)
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                self._sleep_backoff(attempt, why=f"transport: {e}")
                continue

            if resp.status_code >= 500:
                last_exc = RuntimeError(f"Unsplash {resp.status_code}: {resp.text[:200]}")
                self._sleep_backoff(attempt, why=f"server: HTTP {resp.status_code}")
                continue
            if resp.status_code == 401:
                raise RuntimeError("Unsplash 401 Unauthorized — check UNSPLASH_ACCESS_KEY.")
            if resp.status_code == 403:
                raise RuntimeError(
                    "Unsplash 403 Forbidden — likely rate-limited (50/h on demo apps). "
                    "Apply for production access or wait an hour."
                )
            if 400 <= resp.status_code < 500:
                raise RuntimeError(
                    f"Unsplash {resp.status_code}: {resp.text[:200]} (fail-fast on 4xx)"
                )
            return resp.content
        raise RuntimeError(
            f"Unsplash request to {url} failed after {self.max_retries} retries: {last_exc}"
        )

    @staticmethod
    def _sleep_backoff(attempt: int, *, why: str) -> None:
        delay = min(2 ** attempt, 8)  # 1s, 2s, 4s
        log.warning("Unsplash retry %d in %ds (%s)", attempt + 1, delay, why)
        time.sleep(delay)
