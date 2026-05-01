"""Tests for PixabayIngestSource. All HTTP is mocked; no live API calls."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf

from lofivid.ingest.pixabay import (
    PIXABAY_LICENSE_NAME,
    PixabayIngestSource,
    _extract_certificate_url,
    _pick_audio_url,
)


def _silent_wav_bytes(duration_s: float = 1.0, sr: int = 44100) -> bytes:
    """Return real WAV-encoded bytes so mutagen tagging round-trips cleanly."""
    data = np.zeros(int(duration_s * sr), dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def _make_source(monkeypatch) -> PixabayIngestSource:
    monkeypatch.setenv("PIXABAY_API_KEY", "TESTKEY")
    return PixabayIngestSource(rate_limit_s=0.0)


# ---------- init / config ---------------------------------------------------

def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PIXABAY_API_KEY"):
        PixabayIngestSource()


def test_init_accepts_explicit_api_key(monkeypatch):
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    src = PixabayIngestSource(api_key="EXPLICIT")
    assert src.api_key == "EXPLICIT"


# ---------- _pick_audio_url -------------------------------------------------

def test_pick_audio_url_prefers_wav():
    files = [
        {"format": "mp3", "url": "mp3url"},
        {"format": "wav", "url": "wavurl"},
    ]
    url, fmt = _pick_audio_url(files)
    assert url == "wavurl"
    assert fmt == "wav"


def test_pick_audio_url_falls_back_to_mp3():
    url, fmt = _pick_audio_url([{"format": "mp3", "url": "mp3url"}])
    assert url == "mp3url"
    assert fmt == "mp3"


def test_pick_audio_url_returns_none_for_empty_list():
    url, fmt = _pick_audio_url([])
    assert url is None
    assert fmt == ""


def test_pick_audio_url_skips_entries_with_no_url():
    url, fmt = _pick_audio_url([{"format": "wav"}])
    assert url is None


# ---------- fetch happy path ------------------------------------------------

def test_fetch_writes_wav_and_sidecar(tmp_path, monkeypatch):
    src = _make_source(monkeypatch)
    target = tmp_path / "lib"
    target.mkdir()

    search_response = {
        "total": 1, "totalHits": 1,
        "hits": [{
            "id": 12345,
            "title": "Morning Brew",
            "artist": "Lo-Fi Project",
            "duration": 187,
            "audio_files": [
                {"format": "wav", "url": "https://example.com/track.wav"},
            ],
            "page_url": "https://pixabay.com/music/12345/",
        }],
    }

    def fake_request(method, url, *, params=None):
        if "/api/music" in url:
            return json.dumps(search_response).encode()
        if url == "https://example.com/track.wav":
            return _silent_wav_bytes(1.0)
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(src, "_http_request", fake_request)

    fetched = src.fetch(mood_tags=["lofi"], count=1, target_dir=target)

    assert len(fetched) == 1
    track = fetched[0]
    assert track.local_path.exists()
    assert track.sidecar_path.exists()
    assert track.source == "pixabay"
    assert track.source_id == "12345"
    assert track.license == PIXABAY_LICENSE_NAME
    assert track.attribution_text is None
    assert track.original_url == "https://pixabay.com/music/12345/"

    sidecar = json.loads(track.sidecar_path.read_text())
    assert sidecar["source"] == "pixabay"
    assert sidecar["source_id"] == "12345"
    assert sidecar["license"] == PIXABAY_LICENSE_NAME
    assert sidecar["attribution_text"] is None
    assert sidecar["original_url"] == "https://pixabay.com/music/12345/"


def test_fetch_skips_already_downloaded(tmp_path, monkeypatch):
    src = _make_source(monkeypatch)
    target = tmp_path / "lib"
    target.mkdir()

    search_response = {
        "hits": [{
            "id": 12345, "title": "X", "duration": 60,
            "audio_files": [{"format": "wav", "url": "https://example.com/x.wav"}],
            "page_url": "https://pixabay.com/music/12345/",
        }],
    }

    download_count = 0

    def fake_request(method, url, *, params=None):
        nonlocal download_count
        if "/api/music" in url:
            return json.dumps(search_response).encode()
        if url == "https://example.com/x.wav":
            download_count += 1
            return _silent_wav_bytes(1.0)
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(src, "_http_request", fake_request)

    fetched = src.fetch(
        mood_tags=["x"], count=1, target_dir=target,
        already_downloaded={"12345"},
    )

    assert fetched == []
    assert download_count == 0


def test_fetch_returns_empty_when_zero_hits(tmp_path, monkeypatch):
    src = _make_source(monkeypatch)
    target = tmp_path / "lib"
    target.mkdir()

    monkeypatch.setattr(
        src, "_http_request",
        lambda *a, **kw: json.dumps({"hits": [], "total": 0, "totalHits": 0}).encode(),
    )

    fetched = src.fetch(mood_tags=["nothing_matches"], count=10, target_dir=target)
    assert fetched == []


def test_fetch_returns_empty_for_zero_count(tmp_path, monkeypatch):
    src = _make_source(monkeypatch)
    target = tmp_path / "lib"
    target.mkdir()

    # Should not even hit the network when count=0.
    def explode(*a, **kw):
        raise AssertionError("fetch should not call HTTP when count<=0")

    monkeypatch.setattr(src, "_http_request", explode)
    assert src.fetch(mood_tags=["x"], count=0, target_dir=target) == []


def test_fetch_skips_hit_without_audio_files(tmp_path, monkeypatch):
    src = _make_source(monkeypatch)
    target = tmp_path / "lib"
    target.mkdir()

    search_response = {
        "hits": [
            {"id": 1, "title": "no audio", "duration": 60, "audio_files": []},
            {
                "id": 2, "title": "yes audio", "duration": 60,
                "audio_files": [{"format": "wav", "url": "https://example.com/2.wav"}],
                "page_url": "https://pixabay.com/music/2/",
            },
        ],
    }

    def fake_request(method, url, *, params=None):
        if "/api/music" in url:
            return json.dumps(search_response).encode()
        if url.endswith("/2.wav"):
            return _silent_wav_bytes(1.0)
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(src, "_http_request", fake_request)

    fetched = src.fetch(mood_tags=["x"], count=2, target_dir=target)
    assert len(fetched) == 1
    assert fetched[0].source_id == "2"


def test_extract_certificate_url_top_level_field():
    assert _extract_certificate_url(
        {"license_certificate_url": "https://x/cert.pdf"}
    ) == "https://x/cert.pdf"


def test_extract_certificate_url_alternate_top_level_keys():
    assert _extract_certificate_url(
        {"certificateUrl": "https://x/cert.pdf"}
    ) == "https://x/cert.pdf"
    assert _extract_certificate_url(
        {"content_id_certificate": "https://x/c.pdf"}
    ) == "https://x/c.pdf"


def test_extract_certificate_url_nested_field():
    assert _extract_certificate_url(
        {"license": {"certificate_url": "https://x/cert.pdf"}}
    ) == "https://x/cert.pdf"


def test_extract_certificate_url_returns_none_when_missing():
    assert _extract_certificate_url({}) is None
    assert _extract_certificate_url({"license": "string-not-dict"}) is None


def test_extract_certificate_url_strips_empty_strings():
    assert _extract_certificate_url({"license_certificate_url": "  "}) is None


def test_fetch_writes_certificate_url_when_present(tmp_path, monkeypatch):
    src = _make_source(monkeypatch)
    target = tmp_path / "lib"
    target.mkdir()

    cert_url = "https://cdn.pixabay.com/license_certificate/abc.pdf"
    search_response = {
        "hits": [{
            "id": 999,
            "title": "Certified Track",
            "duration": 90,
            "audio_files": [{"format": "wav", "url": "https://example.com/999.wav"}],
            "page_url": "https://pixabay.com/music/999/",
            "license_certificate_url": cert_url,
        }],
    }

    def fake_request(method, url, *, params=None):
        if "/api/music" in url:
            return json.dumps(search_response).encode()
        if url.endswith("/999.wav"):
            return _silent_wav_bytes(1.0)
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(src, "_http_request", fake_request)

    fetched = src.fetch(mood_tags=["x"], count=1, target_dir=target)
    assert len(fetched) == 1
    assert fetched[0].license_certificate_url == cert_url

    sidecar = json.loads(fetched[0].sidecar_path.read_text())
    assert sidecar["license_certificate_url"] == cert_url


def test_fetch_certificate_url_null_when_absent(tmp_path, monkeypatch):
    src = _make_source(monkeypatch)
    target = tmp_path / "lib"
    target.mkdir()

    search_response = {
        "hits": [{
            "id": 1, "title": "Plain Track", "duration": 90,
            "audio_files": [{"format": "wav", "url": "https://example.com/1.wav"}],
            "page_url": "https://pixabay.com/music/1/",
        }],
    }

    def fake_request(method, url, *, params=None):
        if "/api/music" in url:
            return json.dumps(search_response).encode()
        if url.endswith("/1.wav"):
            return _silent_wav_bytes(1.0)
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(src, "_http_request", fake_request)

    fetched = src.fetch(mood_tags=["x"], count=1, target_dir=target)
    assert fetched[0].license_certificate_url is None
    sidecar = json.loads(fetched[0].sidecar_path.read_text())
    assert sidecar["license_certificate_url"] is None


def test_fetch_uses_artist_else_user_field(tmp_path, monkeypatch):
    src = _make_source(monkeypatch)
    target = tmp_path / "lib"
    target.mkdir()

    search_response = {
        "hits": [{
            "id": 7, "title": "Anonymous Track", "user": "uploader42",
            "duration": 60,
            "audio_files": [{"format": "wav", "url": "https://example.com/7.wav"}],
            "page_url": "https://pixabay.com/music/7/",
        }],
    }

    def fake_request(method, url, *, params=None):
        if "/api/music" in url:
            return json.dumps(search_response).encode()
        if url.endswith("/7.wav"):
            return _silent_wav_bytes(1.0)
        raise AssertionError(f"unexpected request: {method} {url}")

    monkeypatch.setattr(src, "_http_request", fake_request)

    fetched = src.fetch(mood_tags=["x"], count=1, target_dir=target)
    assert len(fetched) == 1
    assert fetched[0].artist == "uploader42"
