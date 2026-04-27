"""ACE-Step 1.5 backend.

We don't pin the ACE-Step Python API here because it's installed from a
git tag and the import path may shift between versions. The backend lazily
imports inside `generate` and degrades to a clear error if the package
isn't available, so unit tests for the rest of the pipeline can run on
a CPU-only machine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lofivid.music.base import GeneratedTrack, MusicBackend, TrackSpec

log = logging.getLogger(__name__)


class ACEStepBackend(MusicBackend):
    name = "acestep"

    def __init__(self, model_id: str = "ace-step/ACE-Step-1.5", dtype: str = "bfloat16") -> None:
        self.model_id = model_id
        self.dtype = dtype
        self._pipe: Any = None  # ACE-Step pipeline handle, lazy

    def _ensure_loaded(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        try:
            # ACE-Step ships a `pipeline_ace_step.ACEStepPipeline` class as of v1.5.
            # Real import path may vary between releases; adjust here if upstream renames.
            from acestep.pipeline_ace_step import ACEStepPipeline  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError(
                "ACE-Step is not installed. Inside the Docker image this should be "
                "pre-installed; outside, run: "
                "pip install 'git+https://github.com/ace-step/ACE-Step-1.5'"
            ) from e

        log.info("Loading ACE-Step pipeline %s (dtype=%s)", self.model_id, self.dtype)
        self._pipe = ACEStepPipeline.from_pretrained(self.model_id, torch_dtype=self.dtype)
        try:
            self._pipe.to("cuda")
        except Exception as e:
            log.warning("Could not move ACE-Step to CUDA: %s — falling back to CPU (very slow)", e)
        return self._pipe

    def warmup(self) -> None:
        self._ensure_loaded()

    def shutdown(self) -> None:
        if self._pipe is None:
            return
        try:
            import torch
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()
        except ImportError:
            self._pipe = None

    def generate(self, spec: TrackSpec, output_dir: Path) -> GeneratedTrack:
        pipe = self._ensure_loaded()
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{spec.track_index:03d}.wav"

        log.info(
            "Generating track %d (%ds, %d BPM, %s): '%s'",
            spec.track_index, spec.duration_seconds, spec.bpm, spec.key, spec.prompt,
        )

        # ACE-Step's call signature evolves; this is the v1.5-era contract.
        # Wrap in try/except so signature drift gives a clear error rather
        # than a stack trace inside upstream code.
        try:
            result = pipe(
                prompt=spec.prompt,
                duration=spec.duration_seconds,
                seed=spec.seed,
                output_path=str(out_path),
            )
        except TypeError as e:
            raise RuntimeError(
                f"ACE-Step pipeline call failed (signature drift?). Original error: {e}\n"
                "Check the installed acestep version's pipeline_ace_step.ACEStepPipeline.__call__."
            ) from e

        sr = getattr(result, "sample_rate", 44100) if result is not None else 44100
        actual = _probe_duration_seconds(out_path)
        return GeneratedTrack(spec=spec, path=out_path, sample_rate=sr, actual_duration_seconds=actual)


def _probe_duration_seconds(path: Path) -> float:
    """Cheap duration probe via soundfile; falls back to 0.0 if unavailable."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception as e:
        log.warning("Could not probe duration of %s: %s", path, e)
        return 0.0
