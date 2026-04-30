"""Tests for lofivid/compose/hud.py — HUD PNG renderer and overlay builder."""
from __future__ import annotations

from pathlib import Path

import pytest

DEJAVU_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _dejavu_or_skip() -> Path:
    if not DEJAVU_BOLD.exists():
        pytest.skip("DejaVu Bold font not found on this system")
    return DEJAVU_BOLD


def _make_spec(font_path: Path, **overrides):
    """Build a minimal HUDSpec with DejaVu paths."""
    from lofivid.styles.schema import HUDSpec

    defaults = dict(
        font_path=font_path,
        cjk_font_path=None,
        title_size_pct=0.028,
        artist_size_pct=0.020,
        counter_size_pct=0.018,
        text_color="#FFFFFF",
        panel_color="#000000",
        panel_opacity=0.45,
        panel_padding_px=14,
        corner="bottom_left",
        margin_pct=0.04,
        show_track_counter=True,
        show_artist=True,
        max_width_pct=0.4,
    )
    defaults.update(overrides)
    return HUDSpec(**defaults)


def _make_track(title: str = "Test Title", artist: str | None = "Test Artist"):
    """Build a minimal GeneratedTrack + TrackWindow."""
    from lofivid.music.base import GeneratedTrack, TrackSpec
    from lofivid.music.mixer import TrackWindow

    spec = TrackSpec(
        track_index=0,
        prompt="test prompt",
        bpm=90,
        key="C major",
        duration_seconds=120,
        seed=1,
    )
    track = GeneratedTrack(
        spec=spec,
        path=Path("/tmp/test_track.wav"),
        sample_rate=44100,
        actual_duration_seconds=120.0,
        title=title,
        artist=artist,
    )
    return TrackWindow(track=track, start_seconds=0.0, end_seconds=120.0)


# ---------------------------------------------------------------------------
# render_hud_png
# ---------------------------------------------------------------------------

def test_render_hud_png_creates_file(tmp_path: Path) -> None:
    font_path = _dejavu_or_skip()
    from lofivid.compose.hud import render_hud_png

    spec = _make_spec(font_path)
    window = _make_track()
    out, pw, ph = render_hud_png(spec, window, 0, 3, (1920, 1080), tmp_path)

    assert out.exists()
    assert out.stat().st_size > 0
    assert pw > 0
    assert ph > 0


def test_render_hud_png_cache_hit_reuses_file(tmp_path: Path) -> None:
    font_path = _dejavu_or_skip()
    from lofivid.compose.hud import render_hud_png

    spec = _make_spec(font_path)
    window = _make_track()
    out1, pw1, ph1 = render_hud_png(spec, window, 0, 3, (1920, 1080), tmp_path)
    mtime1 = out1.stat().st_mtime

    out2, pw2, ph2 = render_hud_png(spec, window, 0, 3, (1920, 1080), tmp_path)
    assert out1 == out2
    assert out2.stat().st_mtime == mtime1  # not re-rendered
    assert pw1 == pw2
    assert ph1 == ph2


def test_render_hud_png_different_track_index_different_file(tmp_path: Path) -> None:
    font_path = _dejavu_or_skip()
    from lofivid.compose.hud import render_hud_png

    spec = _make_spec(font_path)
    window = _make_track()
    out0, _, _ = render_hud_png(spec, window, 0, 3, (1920, 1080), tmp_path)
    out1, _, _ = render_hud_png(spec, window, 1, 3, (1920, 1080), tmp_path)

    # Different track index → different filename (counter text differs)
    assert out0 != out1


def test_render_hud_png_produces_valid_png(tmp_path: Path) -> None:
    """Rendered file must be a valid RGBA PNG with non-zero dimensions."""
    from PIL import Image

    font_path = _dejavu_or_skip()
    from lofivid.compose.hud import render_hud_png

    spec = _make_spec(font_path)
    window = _make_track()
    out, pw, ph = render_hud_png(spec, window, 0, 1, (1920, 1080), tmp_path)

    with Image.open(out) as img:
        assert img.mode == "RGBA"
        assert img.width == pw
        assert img.height == ph


def test_render_hud_png_no_artist(tmp_path: Path) -> None:
    """show_artist=False should still produce a file."""
    font_path = _dejavu_or_skip()
    from lofivid.compose.hud import render_hud_png

    spec = _make_spec(font_path, show_artist=False)
    window = _make_track()
    out, pw, ph = render_hud_png(spec, window, 0, 1, (1920, 1080), tmp_path)
    assert out.exists()
    assert pw > 0 and ph > 0


# ---------------------------------------------------------------------------
# build_hud_overlays
# ---------------------------------------------------------------------------

def test_build_hud_overlays_disabled_returns_empty(tmp_path: Path) -> None:
    font_path = _dejavu_or_skip()
    from lofivid.compose.hud import build_hud_overlays

    spec = _make_spec(font_path, enabled=False)
    windows = [_make_track()]
    result = build_hud_overlays(spec, windows, (1920, 1080), tmp_path)
    assert result == []


def test_build_hud_overlays_empty_windows_returns_empty(tmp_path: Path) -> None:
    font_path = _dejavu_or_skip()
    from lofivid.compose.hud import build_hud_overlays

    spec = _make_spec(font_path)
    result = build_hud_overlays(spec, [], (1920, 1080), tmp_path)
    assert result == []


def test_build_hud_overlays_count_matches_windows(tmp_path: Path) -> None:
    font_path = _dejavu_or_skip()
    from lofivid.compose.hud import build_hud_overlays
    from lofivid.music.base import GeneratedTrack, TrackSpec
    from lofivid.music.mixer import TrackWindow

    spec = _make_spec(font_path)
    windows = []
    for i in range(3):
        ts = TrackSpec(track_index=i, prompt="p", bpm=90, key="C", duration_seconds=60, seed=i)
        gt = GeneratedTrack(spec=ts, path=Path("/tmp/t.wav"), sample_rate=44100,
                            actual_duration_seconds=60.0, title=f"Track {i}", artist=None)
        windows.append(TrackWindow(track=gt, start_seconds=float(i * 60), end_seconds=float((i + 1) * 60)))

    result = build_hud_overlays(spec, windows, (1920, 1080), tmp_path)
    assert len(result) == 3
    for overlay in result:
        assert overlay.png_path.exists()
        assert overlay.x_expr.isdigit()
        assert overlay.y_expr.isdigit()


# ---------------------------------------------------------------------------
# hud_corner_xy
# ---------------------------------------------------------------------------

def test_hud_corner_xy_all_corners() -> None:
    from lofivid.compose.hud import hud_corner_xy

    margin = 20
    fw, fh = 1920, 1080
    pw, ph = 200, 80

    x, y = hud_corner_xy("top_left", margin, fw, fh, pw, ph)
    assert x == str(margin)
    assert y == str(margin)

    x, y = hud_corner_xy("top_right", margin, fw, fh, pw, ph)
    assert x == str(fw - pw - margin)
    assert y == str(margin)

    x, y = hud_corner_xy("bottom_left", margin, fw, fh, pw, ph)
    assert x == str(margin)
    assert y == str(fh - ph - margin)

    x, y = hud_corner_xy("bottom_right", margin, fw, fh, pw, ph)
    assert x == str(fw - pw - margin)
    assert y == str(fh - ph - margin)
