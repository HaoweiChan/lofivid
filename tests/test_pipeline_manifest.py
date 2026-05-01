"""Targeted unit tests for pipeline manifest helpers (no full-pipeline runs).

The full _write_manifest path is exercised by the smoke render described in
AGENT_PIVOT_PROMPT_v2.md verification; these tests cover the helper logic that
shapes `music_attributions` from per-track attribution dicts.
"""
from __future__ import annotations

from pathlib import Path

from lofivid.music.base import GeneratedTrack, TrackSpec
from lofivid.pipeline import _build_music_attributions, _classify_content_id_risk


def _spec(track_index: int = 0) -> TrackSpec:
    return TrackSpec(
        track_index=track_index,
        prompt="x",
        bpm=80,
        key="C major",
        duration_seconds=10,
        seed=0,
    )


def test_attribution_included_when_sidecar_present():
    t = GeneratedTrack(
        spec=_spec(0),
        path=Path("/tmp/x.wav"),
        sample_rate=44100,
        actual_duration_seconds=180.0,
        title="Morning Brew",
        artist="Lo-Fi Project",
        attribution={
            "source": "pixabay",
            "source_id": "12345",
            "license": "pixabay-content-license",
            "attribution_text": None,
            "original_url": "https://pixabay.com/music/12345/",
        },
    )
    out = _build_music_attributions([t])
    assert len(out) == 1
    e = out[0]
    assert e["track_title"] == "Morning Brew"
    assert e["track_artist"] == "Lo-Fi Project"
    assert e["track_duration_s"] == 180.0
    assert e["source"] == "pixabay"
    assert e["source_id"] == "12345"
    assert e["license"] == "pixabay-content-license"
    assert e["attribution_text"] is None
    assert e["original_url"] == "https://pixabay.com/music/12345/"


def test_attribution_null_fields_when_sidecar_absent():
    t = GeneratedTrack(
        spec=_spec(0),
        path=Path("/tmp/x.wav"),
        sample_rate=44100,
        actual_duration_seconds=200.0,
        title="Anon",
        artist=None,
        attribution=None,
    )
    out = _build_music_attributions([t])
    assert len(out) == 1
    e = out[0]
    assert e["track_title"] == "Anon"
    assert e["track_artist"] is None
    assert e["source"] is None
    assert e["source_id"] is None
    assert e["license"] is None


def test_attribution_handles_cc_by_with_text():
    t = GeneratedTrack(
        spec=_spec(0),
        path=Path("/tmp/x.wav"),
        sample_rate=44100,
        actual_duration_seconds=120.0,
        title="Slow Tuesday",
        artist="jellyfish",
        attribution={
            "source": "fma",
            "source_id": "999",
            "license": "cc-by-4.0",
            "attribution_text": '"Slow Tuesday" by jellyfish, CC-BY-4.0',
            "original_url": "https://freemusicarchive.org/track/999/",
        },
    )
    out = _build_music_attributions([t])
    assert out[0]["license"] == "cc-by-4.0"
    assert out[0]["attribution_text"].startswith('"Slow Tuesday"')


def test_multiple_tracks_preserve_order():
    tracks = [
        GeneratedTrack(
            spec=_spec(i), path=Path(f"/tmp/{i}.wav"),
            sample_rate=44100, actual_duration_seconds=60.0 + i,
            title=f"t{i}", artist=None, attribution=None,
        )
        for i in range(3)
    ]
    out = _build_music_attributions(tracks)
    assert [e["track_title"] for e in out] == ["t0", "t1", "t2"]
    assert [e["track_duration_s"] for e in out] == [60.0, 61.0, 62.0]


# ---------- _classify_content_id_risk ----------------------------------------

def test_classify_content_id_risk_none_when_no_attribution():
    assert _classify_content_id_risk({}) is None


def test_classify_content_id_risk_true_for_pixabay_no_certificate():
    assert _classify_content_id_risk({
        "source": "pixabay",
        "license_certificate_url": None,
    }) is True


def test_classify_content_id_risk_false_for_pixabay_with_certificate():
    assert _classify_content_id_risk({
        "source": "pixabay",
        "license_certificate_url": "https://cdn.pixabay.com/cert/abc.pdf",
    }) is False


def test_classify_content_id_risk_false_for_manual():
    assert _classify_content_id_risk({"source": "manual"}) is False
    assert _classify_content_id_risk({
        "source": "manual",
        "license_certificate_url": None,  # manual without cert is still user-vouched
    }) is False


def test_classify_content_id_risk_false_for_fma():
    """Future FMA source: CC0/CC-BY tracks aren't typically Content ID-claimed."""
    assert _classify_content_id_risk({"source": "fma"}) is False


# ---------- license_certificate_url + at_risk plumbing in _build_music_attributions

def test_attribution_includes_certificate_url_and_risk_flag():
    t = GeneratedTrack(
        spec=_spec(0),
        path=Path("/tmp/x.wav"),
        sample_rate=44100,
        actual_duration_seconds=120.0,
        title="Cert Track",
        artist="Artist",
        attribution={
            "source": "pixabay",
            "source_id": "999",
            "license": "pixabay-content-license",
            "attribution_text": None,
            "original_url": "https://pixabay.com/music/999/",
            "license_certificate_url": "https://cdn.pixabay.com/cert/999.pdf",
        },
    )
    out = _build_music_attributions([t])[0]
    assert out["license_certificate_url"] == "https://cdn.pixabay.com/cert/999.pdf"
    assert out["at_risk_for_content_id_claim"] is False


def test_attribution_pixabay_without_cert_marked_at_risk():
    t = GeneratedTrack(
        spec=_spec(0),
        path=Path("/tmp/x.wav"),
        sample_rate=44100,
        actual_duration_seconds=120.0,
        title="Risky Track",
        artist="Anon",
        attribution={
            "source": "pixabay",
            "source_id": "888",
            "license": "pixabay-content-license",
            "attribution_text": None,
            "original_url": "https://pixabay.com/music/888/",
            "license_certificate_url": None,
        },
    )
    out = _build_music_attributions([t])[0]
    assert out["license_certificate_url"] is None
    assert out["at_risk_for_content_id_claim"] is True


def test_attribution_no_sidecar_yields_null_risk():
    t = GeneratedTrack(
        spec=_spec(0),
        path=Path("/tmp/x.wav"),
        sample_rate=44100,
        actual_duration_seconds=120.0,
        title="Anon",
        artist=None,
        attribution=None,
    )
    out = _build_music_attributions([t])[0]
    assert out["source"] is None
    assert out["license_certificate_url"] is None
    assert out["at_risk_for_content_id_claim"] is None
