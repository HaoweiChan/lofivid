"""Z-Image-Turbo keyframe backend.

Z-Image-Turbo (Tongyi-MAI / Alibaba, 4 Dec 2025) is a 6B-parameter
distilled image model under **Apache 2.0**. Designed for very fast
inference (~8 NFE for full-quality output) — it's the fastest of the
three commercial-OK shortlist members in `MODEL_OPTIONS.md`, which
matters once we're rendering 6+ keyframes per video.

Loading
-------
Same defensive strategy as the FLUX.2 Klein backend: try
`diffusers.AutoPipelineForText2Image` so the class resolves from the HF
model card. If diffusers doesn't yet have a registered pipeline for
Z-Image we surface a clear actionable error rather than silently
falling back to a wrong loader.

Blackwell sm_120 notes
----------------------
- bf16 default; ~12 GB peak VRAM expected on a 16 GB card.
- No flash-attn / xformers (sm_120 unsupported as of 2026-04).
- 8 inference steps is the documented Turbo sweet spot. Going lower
  trades quality for speed; going higher gives diminishing returns.

Best-effort against the Z-Image-Turbo HF model card as of 2026-04-28;
expect to tweak the loader path once diffusers ships an official
Z-Image pipeline class.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from lofivid.visuals.base import GeneratedImage, KeyframeBackend, KeyframeSpec

log = logging.getLogger(__name__)


class ZImageTurboKeyframeBackend(KeyframeBackend):
    name = "z_image_turbo"

    def __init__(
        self,
        model_id: str = "Tongyi-MAI/Z-Image-Turbo",
        *,
        dtype: str = "bfloat16",
        steps: int = 8,
        guidance_scale: float = 1.0,   # turbo distillation: classifier-free guidance baked in
        cpu_offload: bool = True,
        offload_strategy: str = "sequential",
    ) -> None:
        # cpu_offload defaults ON for the same reason as the FLUX.2 Klein
        # backend: 6B DiT + a Qwen-family text encoder won't fit alongside
        # together in 16 GB VRAM at bf16 without offloading. We default to
        # sequential offload because model_cpu_offload pins the whole DiT
        # on GPU during denoising and activations push past 16 GB at
        # 1344×768 — see flux_klein.py for the same finding.
        self.model_id = model_id
        self.dtype = dtype
        self.steps = steps
        self.guidance_scale = guidance_scale
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

        log.info("Loading Z-Image-Turbo pipeline %s (dtype=%s, cpu_offload=%s, strategy=%s)",
                 self.model_id, self.dtype, self.cpu_offload, self.offload_strategy)
        pipe = self._load_pipeline(torch_dtype)
        if self.cpu_offload:
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
        for fn in ("enable_vae_tiling", "enable_vae_slicing"):
            with contextlib.suppress(AttributeError, Exception):
                getattr(pipe, fn)()
        with contextlib.suppress(Exception):
            pipe.set_progress_bar_config(disable=True)

        self._pipe = pipe
        return pipe

    def _load_pipeline(self, torch_dtype) -> Any:  # noqa: ANN001 (torch.dtype)
        try:
            from diffusers import AutoPipelineForText2Image
            return AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                trust_remote_code=True,   # Z-Image's HF repo ships a custom pipeline
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not load Z-Image-Turbo from {self.model_id}: {e}\n"
                "Workarounds:\n"
                "  • Upgrade diffusers: `pip install -U diffusers transformers`\n"
                "  • Use the FP8 / GGUF mirrors if VRAM is tight: "
                "`drbaph/Z-Image-Turbo-FP8`, `lightx2v/Z-Image-Turbo-Quantized`\n"
                "  • Pin the official inference repo if a diffusers class is "
                "not yet registered."
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

        log.info("Z-Image-Turbo scene %d (%dx%d): %r",
                 spec.scene_index, spec.width, spec.height, spec.prompt[:80])

        import torch
        generator = torch.Generator(device="cuda").manual_seed(spec.seed)
        image = pipe(
            prompt=spec.prompt,
            width=spec.width,
            height=spec.height,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        ).images[0]

        image.save(out_path)
        return GeneratedImage(spec=spec, path=out_path)
