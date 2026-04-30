"""FLUX.2 Klein keyframe backend.

FLUX.2 Klein (Black Forest Labs, 25 Nov 2025) is a 4B-parameter distilled
model from the FLUX.2 family, released under **Apache 2.0** — the same
permissive licence as the original FLUX.1 Schnell. We use it for the
photographic / lifestyle direction where SDXL base 1.0 is too obviously
"AI" and FLUX.1 Krea is licence-blocked for monetisation.

Loading
-------
Tries `diffusers.AutoPipelineForText2Image` first so the class resolves
from the HF model card (FLUX.2 may register as `Flux2Pipeline` or as an
extended `FluxPipeline` depending on the diffusers version). Falls back
to an explicit `FluxPipeline` if AutoPipeline can't resolve the model.

Blackwell sm_120 notes
----------------------
- bfloat16 is the safe default (same as our SDXL path).
- No xformers, no flash-attn — PyTorch's native SDPA is fine on Blackwell.
- No `torch.compile` — `TORCHDYNAMO_DISABLE=1` is set globally in cli.py.
- Inference steps default to 4 (FLUX-distilled sweet spot); guidance 3.5
  matches the model card recommendation for Klein.

This backend is best-effort against the FLUX.2 release notes and HF model
card as of 2026-04-28; signature drift after a diffusers update may need
small tweaks here.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from lofivid.visuals.base import GeneratedImage, KeyframeBackend, KeyframeSpec

log = logging.getLogger(__name__)


class FluxKleinKeyframeBackend(KeyframeBackend):
    name = "flux_klein"

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-4B",
        *,
        dtype: str = "bfloat16",
        steps: int = 4,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 512,
        cpu_offload: bool = True,
        offload_strategy: str = "sequential",
    ) -> None:
        # cpu_offload defaults ON: FLUX.2 Klein's transformer (~8 GB bf16)
        # plus the Qwen3 text encoder (~4–6 GB) won't fit together in 16 GB
        # VRAM.
        #
        # offload_strategy:
        #   "sequential" (default) — enable_sequential_cpu_offload():
        #       streams submodules to GPU one at a time. Peak GPU ~1–2 GB.
        #       Required on a 16 GB card at 1344×768 — model_cpu_offload
        #       OOMs because the whole transformer pins ~8 GB on GPU and
        #       activations push past 16 GB during the denoising loop.
        #   "model" — enable_model_cpu_offload(): whole-component swap,
        #       faster per step, only safe at lower res or on ≥24 GB cards.
        self.model_id = model_id
        self.dtype = dtype
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.max_sequence_length = max_sequence_length
        self.cpu_offload = cpu_offload
        self.offload_strategy = offload_strategy
        self._pipe: Any = None

    def _ensure_loaded(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "torch not available. Inside Docker this should be pre-installed."
            ) from e

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]

        log.info("Loading FLUX.2 Klein pipeline %s (dtype=%s, cpu_offload=%s, strategy=%s)",
                 self.model_id, self.dtype, self.cpu_offload, self.offload_strategy)
        pipe = self._load_pipeline(torch_dtype)
        if self.cpu_offload:
            # The offload helpers manage their own device placement; calling
            # .to("cuda") first would undo the offload.
            if self.offload_strategy == "sequential":
                pipe.enable_sequential_cpu_offload()
            elif self.offload_strategy == "model":
                pipe.enable_model_cpu_offload()
            else:
                raise ValueError(
                    f"unknown offload_strategy {self.offload_strategy!r}; "
                    "expected 'sequential' or 'model'"
                )
        else:
            pipe.to("cuda")
        # Belt-and-suspenders: VAE decode at 1344×768 also peaks GPU memory.
        # Tiling + slicing are no-ops on pipelines that don't expose them.
        for fn in ("enable_vae_tiling", "enable_vae_slicing"):
            with contextlib.suppress(AttributeError, Exception):
                getattr(pipe, fn)()
        with contextlib.suppress(Exception):
            pipe.set_progress_bar_config(disable=True)

        self._pipe = pipe
        return pipe

    def _load_pipeline(self, torch_dtype) -> Any:  # noqa: ANN001 (torch.dtype)
        # Prefer AutoPipeline so the right FLUX class is picked up from the
        # model card automatically (FluxPipeline / Flux2Pipeline / etc.).
        try:
            from diffusers import AutoPipelineForText2Image
            return AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        except (ImportError, Exception) as e_auto:
            log.warning("AutoPipelineForText2Image failed for %s (%s); trying FluxPipeline",
                        self.model_id, e_auto)

        try:
            from diffusers import FluxPipeline
            return FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        except ImportError as e:
            raise RuntimeError(
                "diffusers is missing FluxPipeline. Upgrade with "
                "`pip install -U diffusers transformers` (FLUX.2 requires "
                "diffusers ≥ 0.32 typically)."
            ) from e

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

    def generate(self, spec: KeyframeSpec, output_dir: Path) -> GeneratedImage:
        pipe = self._ensure_loaded()
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{spec.scene_index:03d}.png"

        log.info("FLUX.2 Klein scene %d (%dx%d): %r",
                 spec.scene_index, spec.width, spec.height, spec.prompt[:80])

        import torch
        generator = torch.Generator(device="cuda").manual_seed(spec.seed)

        kwargs: dict[str, Any] = dict(
            prompt=spec.prompt,
            width=spec.width,
            height=spec.height,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )
        # FluxPipeline accepts max_sequence_length; AutoPipeline-resolved
        # variants may not, so pass it conditionally to avoid a TypeError.
        try:
            image = pipe(**kwargs, max_sequence_length=self.max_sequence_length).images[0]
        except TypeError:
            image = pipe(**kwargs).images[0]

        image.save(out_path)
        return GeneratedImage(spec=spec, path=out_path)
