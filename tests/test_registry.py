"""Tests for music and visuals backend registries."""
from __future__ import annotations

import pytest

# Side-effect imports trigger all registrations.
import lofivid.music  # noqa: F401
import lofivid.visuals  # noqa: F401
from lofivid.music.base import MusicBackend
from lofivid.music.registry import MUSIC_BACKENDS
from lofivid.music.registry import make as music_make
from lofivid.visuals.base import ParallaxBackend
from lofivid.visuals.registry import (
    KEYFRAME_BACKENDS,
    PARALLAX_BACKENDS,
    make_keyframe,
    make_parallax,
)

# ---------- music registry --------------------------------------------------

def test_music_registry_has_expected_backends():
    assert "acestep" in MUSIC_BACKENDS
    assert "suno" in MUSIC_BACKENDS
    assert "library" in MUSIC_BACKENDS


def test_music_make_acestep_returns_backend():
    backend = music_make("acestep")
    assert isinstance(backend, MusicBackend)
    assert backend.name == "acestep"


def test_music_make_unknown_raises_with_available():
    with pytest.raises(ValueError, match="Available:"):
        music_make("nonsense_backend_xyz")


# ---------- keyframe registry -----------------------------------------------

def test_keyframe_registry_has_expected_backends():
    assert "sdxl" in KEYFRAME_BACKENDS
    assert "unsplash" in KEYFRAME_BACKENDS
    assert "flux_klein" in KEYFRAME_BACKENDS
    assert "z_image_turbo" in KEYFRAME_BACKENDS


def test_keyframe_make_unknown_raises_with_available():
    with pytest.raises(ValueError, match="Available:"):
        make_keyframe("nonexistent_keyframe_abc")


# ---------- parallax registry -----------------------------------------------

def test_parallax_registry_has_expected_backends():
    assert "depthflow" in PARALLAX_BACKENDS
    assert "overlay_motion" in PARALLAX_BACKENDS


def test_parallax_make_overlay_motion_returns_backend():
    backend = make_parallax("overlay_motion")
    assert isinstance(backend, ParallaxBackend)
    assert backend.name == "overlay_motion"


def test_parallax_make_unknown_raises_with_available():
    with pytest.raises(ValueError, match="Available:"):
        make_parallax("nonexistent_parallax_abc")
