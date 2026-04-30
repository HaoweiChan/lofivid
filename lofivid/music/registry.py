"""Registry for music backends. Adding a new backend is one new file plus
one register() call at the bottom of that file. The Pipeline calls
make(name, **kwargs) instead of an if/elif chain.

Factories are zero-arg callables that return a fresh backend instance;
per-backend init args flow through **kwargs to make().
"""
from __future__ import annotations

from collections.abc import Callable

from lofivid.music.base import MusicBackend

MusicBackendFactory = Callable[..., MusicBackend]
MUSIC_BACKENDS: dict[str, MusicBackendFactory] = {}


def register(name: str, factory: MusicBackendFactory) -> None:
    MUSIC_BACKENDS[name] = factory


def make(name: str, **kwargs) -> MusicBackend:
    if name not in MUSIC_BACKENDS:
        available = ", ".join(sorted(MUSIC_BACKENDS.keys()))
        raise ValueError(
            f"Unknown music backend {name!r}. Available: {available or '(none registered)'}"
        )
    return MUSIC_BACKENDS[name](**kwargs)


def available() -> list[str]:
    return sorted(MUSIC_BACKENDS.keys())
