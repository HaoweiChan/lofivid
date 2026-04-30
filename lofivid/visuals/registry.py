"""Registry for visual backends (keyframe + parallax).

Two independent dicts: KEYFRAME_BACKENDS and PARALLAX_BACKENDS.
Each has its own register/make pair so the call sites are explicit
about which tier they're targeting.
"""
from __future__ import annotations

from collections.abc import Callable

from lofivid.visuals.base import KeyframeBackend, ParallaxBackend

KeyframeBackendFactory = Callable[..., KeyframeBackend]
ParallaxBackendFactory = Callable[..., ParallaxBackend]

KEYFRAME_BACKENDS: dict[str, KeyframeBackendFactory] = {}
PARALLAX_BACKENDS: dict[str, ParallaxBackendFactory] = {}


def register_keyframe(name: str, factory: KeyframeBackendFactory) -> None:
    KEYFRAME_BACKENDS[name] = factory


def register_parallax(name: str, factory: ParallaxBackendFactory) -> None:
    PARALLAX_BACKENDS[name] = factory


def make_keyframe(name: str, **kwargs) -> KeyframeBackend:
    if name not in KEYFRAME_BACKENDS:
        available = ", ".join(sorted(KEYFRAME_BACKENDS.keys()))
        raise ValueError(
            f"Unknown keyframe backend {name!r}. Available: {available or '(none registered)'}"
        )
    return KEYFRAME_BACKENDS[name](**kwargs)


def make_parallax(name: str, **kwargs) -> ParallaxBackend:
    if name not in PARALLAX_BACKENDS:
        available = ", ".join(sorted(PARALLAX_BACKENDS.keys()))
        raise ValueError(
            f"Unknown parallax backend {name!r}. Available: {available or '(none registered)'}"
        )
    return PARALLAX_BACKENDS[name](**kwargs)


def available_keyframes() -> list[str]:
    return sorted(KEYFRAME_BACKENDS.keys())


def available_parallax() -> list[str]:
    return sorted(PARALLAX_BACKENDS.keys())
