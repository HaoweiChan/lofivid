"""Tests for compute_timeline() in lofivid.music.mixer."""
from __future__ import annotations

from pathlib import Path

import pytest

from lofivid.music.base import GeneratedTrack, TrackSpec
from lofivid.music.mixer import TrackWindow, compute_timeline

# ---------- helpers ----------------------------------------------------------

def _spec(index: int = 0) -> TrackSpec:
    return TrackSpec(
        track_index=index,
        prompt="test",
        bpm=80,
        key="C major",
        duration_seconds=10,
        seed=0,
    )


def _track(duration: float, index: int = 0) -> GeneratedTrack:
    return GeneratedTrack(
        spec=_spec(index),
        path=Path("/tmp/dummy.wav"),
        sample_rate=44100,
        actual_duration_seconds=duration,
    )


# ---------- edge cases -------------------------------------------------------

def test_empty_list_returns_empty():
    assert compute_timeline([], crossfade_seconds=2.0) == []


def test_single_track_window():
    t = _track(10.0)
    windows = compute_timeline([t], crossfade_seconds=2.0)
    assert len(windows) == 1
    assert windows[0].start_seconds == 0.0
    assert windows[0].end_seconds == 10.0
    assert windows[0].track is t


# ---------- three-track math -------------------------------------------------
#
# Tracks: 10s, 15s, 20s; crossfade=2s
#
# abs_starts: [0, 8, 21]   (8 = 0+10-2; 21 = 8+15-2)
# abs_ends:   [10, 23, 41]
# half = 1s
# HUD switch 0→1: abs_starts[1] + 0.5*c = 8 + 1 = 9
# HUD switch 1→2: abs_starts[2] + 0.5*c = 21 + 1 = 22
#
# windows: [0, 9], [9, 22], [22, 41]

def test_three_track_timeline():
    t0 = _track(10.0, 0)
    t1 = _track(15.0, 1)
    t2 = _track(20.0, 2)
    windows = compute_timeline([t0, t1, t2], crossfade_seconds=2.0)

    assert len(windows) == 3

    w0, w1, w2 = windows
    assert w0.start_seconds == 0.0
    assert w0.end_seconds == pytest.approx(9.0)

    assert w1.start_seconds == pytest.approx(9.0)
    assert w1.end_seconds == pytest.approx(22.0)

    assert w2.start_seconds == pytest.approx(22.0)
    assert w2.end_seconds == pytest.approx(41.0)


def test_three_track_timeline_track_refs():
    t0 = _track(10.0, 0)
    t1 = _track(15.0, 1)
    t2 = _track(20.0, 2)
    windows = compute_timeline([t0, t1, t2], crossfade_seconds=2.0)
    assert windows[0].track is t0
    assert windows[1].track is t1
    assert windows[2].track is t2


def test_two_track_hud_switch():
    # Two tracks of 10s each, crossfade=4s
    # abs_starts: [0, 6]  (6 = 0+10-4)
    # abs_ends:   [10, 16]
    # HUD switch: 6 + 2 = 8
    t0 = _track(10.0, 0)
    t1 = _track(10.0, 1)
    windows = compute_timeline([t0, t1], crossfade_seconds=4.0)
    assert windows[0].start_seconds == 0.0
    assert windows[0].end_seconds == pytest.approx(8.0)
    assert windows[1].start_seconds == pytest.approx(8.0)
    assert windows[1].end_seconds == pytest.approx(16.0)


def test_zero_crossfade():
    # No crossfade → windows are simply concatenated track durations.
    t0 = _track(5.0, 0)
    t1 = _track(7.0, 1)
    windows = compute_timeline([t0, t1], crossfade_seconds=0.0)
    assert windows[0].start_seconds == 0.0
    assert windows[0].end_seconds == pytest.approx(5.0)
    assert windows[1].start_seconds == pytest.approx(5.0)
    assert windows[1].end_seconds == pytest.approx(12.0)


def test_window_is_frozen_dataclass():
    t = _track(10.0)
    w = TrackWindow(track=t, start_seconds=0.0, end_seconds=10.0)
    import dataclasses
    assert dataclasses.is_dataclass(w)
    try:
        w.start_seconds = 99.0  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except AssertionError:
        raise
    except Exception:
        pass
