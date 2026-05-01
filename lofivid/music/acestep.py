"""ACE-Step 1.5 backend.

Verified against the upstream `acestep.pipeline_ace_step` API (May 2025
release; PyPI 0.1.0 / `pip install ace-step`):

  pipe = ACEStepPipeline(
      checkpoint_dir=None,    # None → auto-download to HF cache
      device_id=0,
      dtype="bfloat16",
      torch_compile=False,
      cpu_offload=False,
  )
  pipe(
      format="wav",
      audio_duration=60.0,
      prompt="lo-fi, piano, 75 BPM",
      lyrics="",                 # empty for instrumental — None breaks len() check
      infer_step=60,
      guidance_scale=15.0,
      manual_seeds=[42],
      save_path="track.wav",
  )

The pipeline writes the WAV directly to `save_path`; nothing useful is
returned. We keep a `_probe_duration_seconds()` post-step so the caller
gets a verified duration in case ACE-Step trimmed/padded.

WAV-save monkey-patch
---------------------
torchaudio 2.11+ rewrote `torchaudio.save` as a thin wrapper over
`save_with_torchcodec`, which in turn requires libavutil/libavcodec
shared libraries on the host. Our host-mode setup ships only a static
ffmpeg binary (imageio-ffmpeg) — no shared libs. To avoid that hard
dep we install a soundfile-based shim for `torchaudio.save` at import
time. The shim is only active when used for WAV/FLAC; other formats
fall through to the torchcodec path so MP3/OGG would still need it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lofivid.music._audio_probe import probe_duration_seconds as _probe_duration_seconds
from lofivid.music.base import GeneratedTrack, MusicBackend, TrackSpec

log = logging.getLogger(__name__)


def _install_torchaudio_save_shim() -> None:
    """Replace torchaudio.save with a soundfile-based implementation for WAV/FLAC.

    Idempotent; safe to call multiple times. No-op if torchaudio isn't installed.
    """
    try:
        import soundfile as sf
        import torch
        import torchaudio
    except ImportError:
        return

    if getattr(torchaudio.save, "_lofivid_shim", False):
        return

    _original_save = torchaudio.save

    def save(uri, src, sample_rate, channels_first=True, format=None, **kwargs):
        # Determine format from extension if not explicit
        fmt = (format or Path(str(uri)).suffix.lstrip(".")).lower()
        if fmt not in {"wav", "flac"}:
            return _original_save(
                uri, src, sample_rate,
                channels_first=channels_first, format=format, **kwargs,
            )

        # torchaudio passes (channels, samples) when channels_first=True;
        # soundfile wants (samples, channels).
        if hasattr(src, "detach"):
            src = src.detach().cpu().to(torch.float32).numpy()
        if channels_first and src.ndim == 2:
            src = src.T
        sf.write(str(uri), src, sample_rate, format=fmt.upper())

    save._lofivid_shim = True            # type: ignore[attr-defined]
    torchaudio.save = save                # type: ignore[assignment]
    log.debug("Installed torchaudio.save → soundfile shim")


class ACEStepBackend(MusicBackend):
    name = "acestep"

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        device_id: int = 0,
        dtype: str = "bfloat16",
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        torch_compile: bool = False,
        cpu_offload: bool = False,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device_id = device_id
        self.dtype = dtype
        self.infer_step = infer_step
        self.guidance_scale = guidance_scale
        self.torch_compile = torch_compile
        self.cpu_offload = cpu_offload
        self._pipe: Any = None

    def _ensure_loaded(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        # Install the WAV save shim BEFORE acestep imports torchaudio internally.
        _install_torchaudio_save_shim()

        try:
            from acestep.pipeline_ace_step import ACEStepPipeline  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError(
                "ACE-Step is not installed. Run: pip install ace-step"
            ) from e

        log.info(
            "Loading ACE-Step pipeline (dtype=%s, device_id=%d, compile=%s, cpu_offload=%s)",
            self.dtype, self.device_id, self.torch_compile, self.cpu_offload,
        )
        self._pipe = ACEStepPipeline(
            checkpoint_dir=self.checkpoint_dir,
            device_id=self.device_id,
            dtype=self.dtype,
            torch_compile=self.torch_compile,
            cpu_offload=self.cpu_offload,
        )
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
            spec.track_index, spec.duration_seconds, spec.bpm, spec.key,
            (spec.prompt[:80] + "…") if len(spec.prompt) > 80 else spec.prompt,
        )

        try:
            pipe(
                format="wav",
                audio_duration=float(spec.duration_seconds),
                prompt=spec.prompt,
                lyrics="",                           # empty = instrumental
                infer_step=self.infer_step,
                guidance_scale=self.guidance_scale,
                manual_seeds=[spec.seed],
                save_path=str(out_path),
            )
        except TypeError as e:
            raise RuntimeError(
                f"ACE-Step __call__ signature drift: {e}\n"
                "Verify acestep.pipeline_ace_step.ACEStepPipeline against the installed version."
            ) from e

        if not out_path.exists():
            raise RuntimeError(
                f"ACE-Step ran but did not produce {out_path}. "
                "Check your HF cache permissions and free disk space."
            )

        actual = _probe_duration_seconds(out_path)

        # Best-effort title from mood + first non-mood prompt tag.
        mood = (spec.mood or "").strip() if hasattr(spec, "mood") else ""
        title_parts: list[str] = []
        if mood:
            title_parts.append(mood.title())
        parts = [p.strip() for p in spec.prompt.split(",") if p.strip()]
        if len(parts) > 1 and parts[1] != mood:
            title_parts.append(parts[1].title())
        title = " — ".join(title_parts) if title_parts else f"Track {spec.track_index + 1:02d}"

        return GeneratedTrack(
            spec=spec, path=out_path, sample_rate=44100,
            actual_duration_seconds=actual,
            title=title, artist=None,
        )


