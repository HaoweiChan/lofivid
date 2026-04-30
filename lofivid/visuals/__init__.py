"""Importing this package triggers visual backend registration."""
from __future__ import annotations

# Eager: cheap parallax backend with no heavy GPU deps.
# OverlayMotionBackend registers itself at the bottom of overlay_motion.py.
from lofivid.visuals import overlay_motion  # noqa: F401  (side-effect: registers "overlay_motion")

# Lazy keyframe/parallax backends: factory lambdas so GPU/network deps are
# not imported until the backend is actually instantiated.
from lofivid.visuals import registry as _registry


def _make_sdxl(**kwargs):
    from lofivid.visuals.keyframes import SDXLKeyframeBackend
    return SDXLKeyframeBackend(**kwargs)


_registry.register_keyframe("sdxl", _make_sdxl)


def _make_unsplash(**kwargs):
    from lofivid.visuals.unsplash import UnsplashKeyframeBackend
    return UnsplashKeyframeBackend(**kwargs)


_registry.register_keyframe("unsplash", _make_unsplash)


def _make_flux_klein(**kwargs):
    from lofivid.visuals.flux_klein import FluxKleinKeyframeBackend
    return FluxKleinKeyframeBackend(**kwargs)


_registry.register_keyframe("flux_klein", _make_flux_klein)


def _make_z_image(**kwargs):
    from lofivid.visuals.z_image import ZImageTurboKeyframeBackend
    return ZImageTurboKeyframeBackend(**kwargs)


_registry.register_keyframe("z_image_turbo", _make_z_image)


def _make_depthflow(**kwargs):
    from lofivid.visuals.depthflow import DepthFlowBackend
    return DepthFlowBackend(**kwargs)


_registry.register_parallax("depthflow", _make_depthflow)
