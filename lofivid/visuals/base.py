"""Visual backend ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KeyframeSpec:
    scene_index: int
    prompt: str
    width: int
    height: int
    seed: int

    def cache_key(self) -> dict:
        return {
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class GeneratedImage:
    spec: KeyframeSpec
    path: Path


@dataclass(frozen=True)
class ParallaxSpec:
    """Inputs to a parallax-loop generation call."""
    scene_index: int
    image_path: Path
    duration_seconds: int
    width: int
    height: int
    fps: int
    seed: int

    def cache_key(self) -> dict:
        return {
            "image_path": str(self.image_path),
            "image_mtime": self.image_path.stat().st_mtime if self.image_path.exists() else None,
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class GeneratedClip:
    spec: ParallaxSpec
    path: Path


class KeyframeBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, spec: KeyframeSpec, output_dir: Path) -> GeneratedImage: ...

    def warmup(self) -> None: return None
    def shutdown(self) -> None: return None


class ParallaxBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, spec: ParallaxSpec, output_dir: Path) -> GeneratedClip: ...

    def warmup(self) -> None: return None
    def shutdown(self) -> None: return None
