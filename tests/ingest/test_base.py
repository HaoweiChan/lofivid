"""Tests for the IngestSource ABC, IngestedTrack, sidecar I/O, and registry."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from lofivid.ingest.base import (
    IngestedTrack,
    IngestSource,
    available_sources,
    existing_source_ids,
    get,
    read_sidecar,
    register,
    sidecar_path_for,
    slugify_filename,
    unique_audio_path,
    write_sidecar,
)

# ---------- IngestedTrack dataclass ----------------------------------------

def test_ingested_track_is_frozen(tmp_path):
    t = IngestedTrack(
        title="x", artist=None, duration_s=1.0,
        source="manual", source_id="x.wav", original_url="local",
        license="cc0", attribution_text=None,
        local_path=tmp_path / "x.wav", sidecar_path=tmp_path / "x.attribution.json",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.title = "y"  # type: ignore[misc]


# ---------- slugify_filename ------------------------------------------------

def test_slugify_filename_basic():
    assert slugify_filename("Morning Brew") == "morning_brew"


def test_slugify_filename_special_chars():
    assert slugify_filename("Track #3 — Dawn!") == "track_3_dawn"


def test_slugify_filename_empty_returns_fallback():
    assert slugify_filename("", fallback="anon") == "anon"
    assert slugify_filename("!!!@@@", fallback="anon") == "anon"


def test_slugify_filename_collapses_runs():
    assert slugify_filename("a    b___c   d") == "a_b_c_d"


# ---------- unique_audio_path ----------------------------------------------

def test_unique_audio_path_no_collision(tmp_path):
    assert unique_audio_path(tmp_path, "track", "abc123") == tmp_path / "track.wav"


def test_unique_audio_path_appends_source_id_on_collision(tmp_path):
    (tmp_path / "track.wav").touch()
    assert unique_audio_path(tmp_path, "track", "abc123") == tmp_path / "track_abc123.wav"


# ---------- sidecar I/O ----------------------------------------------------

def test_sidecar_path_for_uses_attribution_json_suffix():
    assert sidecar_path_for(Path("/x/y/song.wav")) == Path("/x/y/song.attribution.json")
    assert sidecar_path_for(Path("/x/y/song.flac")) == Path("/x/y/song.attribution.json")


def test_write_sidecar_roundtrip(tmp_path):
    audio = tmp_path / "song.wav"
    audio.touch()
    side = write_sidecar(
        audio,
        source="pixabay", source_id="12345",
        license="pixabay-content-license",
        attribution_text=None,
        original_url="https://pixabay.com/music/12345/",
    )
    assert side.exists()
    data = json.loads(side.read_text())
    assert data["source"] == "pixabay"
    assert data["source_id"] == "12345"
    assert data["license"] == "pixabay-content-license"
    assert data["attribution_text"] is None
    assert data["original_url"] == "https://pixabay.com/music/12345/"
    # license_certificate_url defaults to None when not passed.
    assert data["license_certificate_url"] is None
    assert data["fetched_at"]  # ISO timestamp


def test_write_sidecar_persists_license_certificate_url(tmp_path):
    audio = tmp_path / "song.wav"
    audio.touch()
    cert = "https://cdn.pixabay.com/license_certificate/abc.pdf"
    write_sidecar(
        audio,
        source="pixabay", source_id="42",
        license="pixabay-content-license",
        attribution_text=None,
        original_url="https://pixabay.com/music/42/",
        license_certificate_url=cert,
    )
    data = json.loads((tmp_path / "song.attribution.json").read_text())
    assert data["license_certificate_url"] == cert


def test_read_sidecar_returns_none_when_missing(tmp_path):
    audio = tmp_path / "no_sidecar.wav"
    audio.touch()
    assert read_sidecar(audio) is None


def test_read_sidecar_returns_none_on_malformed_json(tmp_path):
    audio = tmp_path / "song.wav"
    audio.touch()
    sidecar_path_for(audio).write_text("{not valid json")
    assert read_sidecar(audio) is None


def test_existing_source_ids_collects_from_dir(tmp_path):
    a = tmp_path / "a.wav"
    a.touch()
    b = tmp_path / "b.wav"
    b.touch()
    write_sidecar(a, source="x", source_id="111", license="cc0",
                  attribution_text=None, original_url="http://a")
    write_sidecar(b, source="x", source_id="222", license="cc0",
                  attribution_text=None, original_url="http://b")
    assert existing_source_ids(tmp_path) == {"111", "222"}


def test_existing_source_ids_empty_dir(tmp_path):
    assert existing_source_ids(tmp_path) == set()


def test_existing_source_ids_skips_non_sidecar_files(tmp_path):
    (tmp_path / "audio.wav").touch()
    (tmp_path / "random.json").write_text('{"source_id": "ignored"}')
    assert existing_source_ids(tmp_path) == set()


# ---------- registry --------------------------------------------------------

class _DummySource(IngestSource):
    name = "_dummy_test_source"

    def fetch(self, mood_tags, count, target_dir,
              min_duration_s=60.0, max_duration_s=600.0,
              already_downloaded=None):
        return []


def test_register_and_get_roundtrip():
    register(_DummySource.name, _DummySource)
    assert get(_DummySource.name) is _DummySource


def test_register_idempotent_when_same_class():
    register(_DummySource.name, _DummySource)
    register(_DummySource.name, _DummySource)  # no error
    assert get(_DummySource.name) is _DummySource


def test_register_rejects_different_class_under_same_name():
    class _Other(IngestSource):
        name = "_dummy_dup"

        def fetch(self, mood_tags, count, target_dir,
                  min_duration_s=60.0, max_duration_s=600.0,
                  already_downloaded=None):
            return []

    class _Other2(IngestSource):
        name = "_dummy_dup"

        def fetch(self, mood_tags, count, target_dir,
                  min_duration_s=60.0, max_duration_s=600.0,
                  already_downloaded=None):
            return []

    register("_dummy_dup", _Other)
    with pytest.raises(RuntimeError, match="already registered"):
        register("_dummy_dup", _Other2)


def test_get_unknown_raises_with_available_list():
    with pytest.raises(ValueError, match="unknown ingest source"):
        get("nonexistent_source_xyz123")


def test_available_sources_includes_default_registrations():
    # Importing lofivid.ingest registers manual + pixabay.
    import lofivid.ingest  # noqa: F401
    names = available_sources()
    assert "manual" in names
    assert "pixabay" in names
