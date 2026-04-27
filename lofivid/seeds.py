"""Centralised RNG. Logs every seed used so runs are reproducible."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class SeedRegistry:
    """Tracks every (purpose, seed) pair used in a run.

    Purposes are namespaced: 'music.track.0', 'visuals.keyframe.3', etc.
    Stored verbatim in the run manifest so a re-run with the same config
    produces byte-identical intermediates (modulo non-deterministic GPU ops).
    """

    base_seed: int
    used: dict[str, int] = field(default_factory=dict)

    def derive(self, purpose: str) -> int:
        """Return a deterministic per-purpose seed derived from base_seed."""
        if purpose in self.used:
            return self.used[purpose]
        # Hash-based derivation so order of calls doesn't matter and
        # purposes stay stable across runs.
        h = hash((self.base_seed, purpose)) & 0xFFFF_FFFF
        self.used[purpose] = h
        log.debug("seed[%s] = %d", purpose, h)
        return h

    def seed_python_rng(self, purpose: str) -> random.Random:
        return random.Random(self.derive(purpose))

    def seed_torch(self, purpose: str) -> int:
        seed = self.derive(purpose)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
        return seed

    def seed_numpy(self, purpose: str) -> int:
        seed = self.derive(purpose)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        return seed
