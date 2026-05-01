"""Importing this package registers every music backend via factory lambdas.

All backends use the same pattern so heavy imports (acestep weights,
requests, mutagen) stay lazy until a run actually instantiates that
backend. The lambdas themselves are registered eagerly here so the
registry is fully populated by the time the pipeline asks for a backend
by name.
"""
from __future__ import annotations

from lofivid.music import registry as _registry


def _make_acestep(**kwargs):
    from lofivid.music.acestep import ACEStepBackend
    return ACEStepBackend(**kwargs)


def _make_suno(**kwargs):
    from lofivid.music.suno import SunoMusicBackend
    return SunoMusicBackend(**kwargs)


def _make_library(**kwargs):
    from lofivid.music.library import LibraryMusicBackend
    return LibraryMusicBackend(**kwargs)


_registry.register("acestep", _make_acestep)
_registry.register("suno", _make_suno)
_registry.register("library", _make_library)
