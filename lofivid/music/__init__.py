"""Importing this package triggers backend registration."""
from __future__ import annotations

# Eager import: ACEStepBackend registers itself at the bottom of acestep.py
# and has no heavy deps at import time (GPU loads lazily via _ensure_loaded).
from lofivid.music import acestep  # noqa: F401  (side-effect: registers "acestep")

# Lazy backends: registered via factory lambdas so heavy/optional deps
# (requests for Suno, mutagen for Library) are not imported at startup.
from lofivid.music import registry as _registry


def _make_suno(**kwargs):
    from lofivid.music.suno import SunoMusicBackend
    return SunoMusicBackend(**kwargs)


_registry.register("suno", _make_suno)


def _make_library(**kwargs):
    from lofivid.music.library import LibraryMusicBackend
    return LibraryMusicBackend(**kwargs)


_registry.register("library", _make_library)
