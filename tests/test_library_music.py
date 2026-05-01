"""Tests for LibraryMusicBackend."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lofivid.ingest.base import write_sidecar
from lofivid.music.base import GeneratedTrack, TrackSpec
from lofivid.music.library import LibraryMusicBackend, slugify

# ---------- helpers ----------------------------------------------------------

def _make_spec(seed: int, mood: str = "cafe afternoon") -> TrackSpec:
    return TrackSpec(
        track_index=0,
        prompt="cafe_afternoon, jazz, 80 BPM, key of A minor",
        bpm=80,
        key="A minor",
        duration_seconds=10,
        seed=seed,
        mood=mood,
    )


def _write_silent_wav(path: Path, duration_s: float = 1.0, sr: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(int(duration_s * sr), dtype=np.int16)
    sf.write(str(path), data, sr)


# ---------- slugify ----------------------------------------------------------

def test_slugify_basic():
    assert slugify("Cafe Afternoon") == "cafe_afternoon"


def test_slugify_special_chars():
    assert slugify("lo-fi & jazz") == "lo_fi_jazz"


# ---------- mood slug directory match ----------------------------------------

def test_generate_picks_correct_track_by_seed(tmp_path):
    lib = tmp_path / "lib"
    mood_dir = lib / "cafe_afternoon"
    _write_silent_wav(mood_dir / "aaa.wav")
    _write_silent_wav(mood_dir / "bbb.wav")
    _write_silent_wav(mood_dir / "ccc.wav")

    backend = LibraryMusicBackend(library_dir=lib)
    out = tmp_path / "out"

    # seed=0 → 0 % 3 = 0 → aaa
    spec0 = _make_spec(seed=0)
    track0 = backend.generate(spec0, out)
    assert isinstance(track0, GeneratedTrack)
    assert track0.path.exists()
    assert track0.title == "aaa"  # filename stem as fallback (no mutagen tags in test file)

    # seed=1 → 1 % 3 = 1 → bbb
    spec1 = _make_spec(seed=1)
    track1 = backend.generate(spec1, out)
    assert track1.title == "bbb"

    # seed=2 → 2 % 3 = 2 → ccc
    spec2 = _make_spec(seed=2)
    track2 = backend.generate(spec2, out)
    assert track2.title == "ccc"


def test_generate_is_deterministic(tmp_path):
    lib = tmp_path / "lib"
    mood_dir = lib / "cafe_afternoon"
    _write_silent_wav(mood_dir / "aaa.wav")
    _write_silent_wav(mood_dir / "bbb.wav")

    backend = LibraryMusicBackend(library_dir=lib)
    out = tmp_path / "out"
    spec = _make_spec(seed=0)

    t1 = backend.generate(spec, out)
    t2 = backend.generate(spec, out)
    assert t1.title == t2.title


def test_generate_mood_dir_missing_falls_back_to_library_dir_empty_raises(tmp_path):
    # When mood slug matches no subdir, _infer_mood_slug returns None and the
    # backend searches library_dir directly. If that is also empty of audio
    # files, it raises a "no audio files" error.
    lib = tmp_path / "lib"
    lib.mkdir()
    backend = LibraryMusicBackend(library_dir=lib)
    out = tmp_path / "out"
    spec = _make_spec(seed=0, mood="nonexistent_mood")
    with pytest.raises(RuntimeError, match="no audio files"):
        backend.generate(spec, out)


def test_generate_mood_dir_present_but_no_files_raises(tmp_path):
    # Mood slug does match a subdir (so the subdir IS chosen), but the dir
    # has no audio files — should raise clearly.
    lib = tmp_path / "lib"
    mood_dir = lib / "no_audio_mood"
    mood_dir.mkdir(parents=True)
    backend = LibraryMusicBackend(library_dir=lib)
    out = tmp_path / "out"
    spec = TrackSpec(
        track_index=0,
        prompt="no_audio_mood, jazz",
        bpm=80,
        key="C major",
        duration_seconds=5,
        seed=0,
        mood="no audio mood",
    )
    with pytest.raises(RuntimeError, match="no audio files"):
        backend.generate(spec, out)


def test_generate_empty_dir_raises(tmp_path):
    lib = tmp_path / "lib"
    mood_dir = lib / "cafe_afternoon"
    mood_dir.mkdir(parents=True)  # exists but empty
    backend = LibraryMusicBackend(library_dir=lib)
    out = tmp_path / "out"
    spec = _make_spec(seed=0)
    with pytest.raises(RuntimeError, match="no audio files"):
        backend.generate(spec, out)


def test_round_robin_mode(tmp_path):
    lib = tmp_path / "lib"
    _write_silent_wav(lib / "track_a.wav")
    _write_silent_wav(lib / "track_b.wav")

    backend = LibraryMusicBackend(library_dir=lib, match_by="round_robin")
    out = tmp_path / "out"
    spec = TrackSpec(
        track_index=0,
        prompt="any prompt",
        bpm=80,
        key="C major",
        duration_seconds=5,
        seed=0,
    )
    track = backend.generate(spec, out)
    assert isinstance(track, GeneratedTrack)
    assert track.path.exists()


def test_actual_duration_populated(tmp_path):
    lib = tmp_path / "lib"
    mood_dir = lib / "cafe_afternoon"
    _write_silent_wav(mood_dir / "song.wav", duration_s=2.0)

    backend = LibraryMusicBackend(library_dir=lib)
    out = tmp_path / "out"
    spec = _make_spec(seed=0)
    track = backend.generate(spec, out)
    assert abs(track.actual_duration_seconds - 2.0) < 0.1


# ---------- cache_key_extras (path + content hash) --------------------------

def test_cache_key_extras_returns_path_and_content_hash(tmp_path):
    lib = tmp_path / "lib"
    mood_dir = lib / "cafe_afternoon"
    _write_silent_wav(mood_dir / "song.wav", duration_s=1.0)

    backend = LibraryMusicBackend(library_dir=lib)
    extras = backend.cache_key_extras(_make_spec(seed=0))

    assert "resolved_path" in extras
    assert "content_hash" in extras
    assert extras["resolved_path"] == str(mood_dir / "song.wav")
    assert extras["content_hash"] is not None and len(extras["content_hash"]) == 16


def test_cache_key_extras_changes_when_content_changes(tmp_path):
    """Same filename + same seed but different bytes ⇒ different cache key."""
    lib = tmp_path / "lib"
    mood_dir = lib / "cafe_afternoon"
    target = mood_dir / "song.wav"
    _write_silent_wav(target, duration_s=1.0)

    backend = LibraryMusicBackend(library_dir=lib)
    spec = _make_spec(seed=0)
    extras_before = backend.cache_key_extras(spec)

    # Overwrite with different content (longer file, different samples).
    target.unlink()
    data = (np.ones(int(2.0 * 44100), dtype=np.int16) * 100)
    sf.write(str(target), data, 44100)

    extras_after = backend.cache_key_extras(spec)

    assert extras_before["resolved_path"] == extras_after["resolved_path"]
    assert extras_before["content_hash"] != extras_after["content_hash"]


def test_cache_key_extras_stable_under_unchanged_content(tmp_path):
    lib = tmp_path / "lib"
    mood_dir = lib / "cafe_afternoon"
    _write_silent_wav(mood_dir / "song.wav", duration_s=1.0)

    backend = LibraryMusicBackend(library_dir=lib)
    spec = _make_spec(seed=0)

    a = backend.cache_key_extras(spec)
    b = backend.cache_key_extras(spec)
    assert a == b


def test_cache_key_extras_handles_missing_library_gracefully(tmp_path):
    """Empty library: extras shouldn't raise; let generate() be the loud one."""
    lib = tmp_path / "lib"
    lib.mkdir()
    backend = LibraryMusicBackend(library_dir=lib)
    extras = backend.cache_key_extras(_make_spec(seed=0))
    assert extras == {"resolved_path": None, "content_hash": None}


# ---------- attribution sidecar (v3 ingest layer) ---------------------------

def test_generated_track_attribution_none_when_no_sidecar(tmp_path):
    lib = tmp_path / "lib"
    mood_dir = lib / "cafe_afternoon"
    _write_silent_wav(mood_dir / "song.wav")
    backend = LibraryMusicBackend(library_dir=lib)
    track = backend.generate(_make_spec(seed=0), tmp_path / "out")
    assert track.attribution is None


def test_generated_track_picks_up_sidecar_attribution(tmp_path):
    lib = tmp_path / "lib"
    mood_dir = lib / "cafe_afternoon"
    audio = mood_dir / "song.wav"
    _write_silent_wav(audio)
    write_sidecar(
        audio,
        source="pixabay",
        source_id="99",
        license="pixabay-content-license",
        attribution_text=None,
        original_url="https://pixabay.com/music/99/",
    )
    backend = LibraryMusicBackend(library_dir=lib)
    track = backend.generate(_make_spec(seed=0), tmp_path / "out")
    assert track.attribution is not None
    assert track.attribution["source"] == "pixabay"
    assert track.attribution["source_id"] == "99"
    assert track.attribution["license"] == "pixabay-content-license"


def test_sidecar_carries_cc_by_attribution_text(tmp_path):
    lib = tmp_path / "lib"
    mood_dir = lib / "cafe_afternoon"
    audio = mood_dir / "song.wav"
    _write_silent_wav(audio)
    write_sidecar(
        audio,
        source="fma",
        source_id="abc",
        license="cc-by-4.0",
        attribution_text='"Slow Tuesday" by jellyfish, CC-BY-4.0',
        original_url="https://freemusicarchive.org/track/abc/",
    )
    backend = LibraryMusicBackend(library_dir=lib)
    track = backend.generate(_make_spec(seed=0), tmp_path / "out")
    assert track.attribution["attribution_text"].startswith('"Slow Tuesday"')
    # Sidecar must NOT have replaced the audio metadata fields.
    sidecar_data = json.loads((mood_dir / "song.attribution.json").read_text())
    assert "title" not in sidecar_data
    assert "artist" not in sidecar_data
