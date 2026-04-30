"""Tests for LibraryMusicBackend."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

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
