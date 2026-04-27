"""Tests for the DJ-mix assembler.

These tests cover the pure-Python crossfade math; the actual ffmpeg invocation
is exercised in the smoke render and not unit-tested here (would require a
live ffmpeg + audio fixtures).
"""

from __future__ import annotations

from lofivid.music.mixer import expected_total_seconds


def test_expected_total_single_track():
    assert expected_total_seconds([300.0], crossfade=6.0) == 300.0


def test_expected_total_no_tracks():
    assert expected_total_seconds([], crossfade=6.0) == 0.0


def test_expected_total_subtracts_crossfade_per_join():
    # 3 tracks @ 300s with 6s crossfade between each: 900 - 12 = 888s
    assert expected_total_seconds([300.0, 300.0, 300.0], crossfade=6.0) == 888.0


def test_expected_total_handles_uneven_track_lengths():
    # 200 + 300 + 400 = 900, minus 2 * 5 = 890
    assert expected_total_seconds([200.0, 300.0, 400.0], crossfade=5.0) == 890.0


def test_expected_total_with_zero_crossfade_is_just_sum():
    assert expected_total_seconds([100.0, 200.0, 300.0], crossfade=0.0) == 600.0
