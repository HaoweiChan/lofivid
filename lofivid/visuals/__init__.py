"""Importing this package registers every visual backend via factory lambdas.

All backends use the same pattern (factory lambda → lazy import) so heavy
GPU / network deps are only imported when a run actually selects that
backend. The lambdas themselves are registered eagerly here so the
registry is fully populated by the time the pipeline asks by name.
"""
from __future__ import annotations

from lofivid.visuals import registry as _registry

# ---------- keyframe backends ----------------------------------------------

def _make_sdxl(**kwargs):
    from lofivid.visuals.keyframes import SDXLKeyframeBackend
    return SDXLKeyframeBackend(**kwargs)


def _make_unsplash(**kwargs):
    from lofivid.visuals.unsplash import UnsplashKeyframeBackend
    return UnsplashKeyframeBackend(**kwargs)


def _make_flux_klein(**kwargs):
    from lofivid.visuals.flux_klein import FluxKleinKeyframeBackend
    return FluxKleinKeyframeBackend(**kwargs)


def _make_z_image(**kwargs):
    from lofivid.visuals.z_image import ZImageTurboKeyframeBackend
    return ZImageTurboKeyframeBackend(**kwargs)


_registry.register_keyframe("sdxl", _make_sdxl)
_registry.register_keyframe("unsplash", _make_unsplash)
_registry.register_keyframe("flux_klein", _make_flux_klein)
_registry.register_keyframe("z_image_turbo", _make_z_image)


# ---------- parallax backends ----------------------------------------------

def _make_overlay_motion(**kwargs):
    from lofivid.visuals.overlay_motion import OverlayMotionBackend
    return OverlayMotionBackend(**kwargs)


def _make_depthflow(**kwargs):
    from lofivid.visuals.depthflow import DepthFlowBackend
    return DepthFlowBackend(**kwargs)


_registry.register_parallax("overlay_motion", _make_overlay_motion)
_registry.register_parallax("depthflow", _make_depthflow)
