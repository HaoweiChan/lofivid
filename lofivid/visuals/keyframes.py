"""SDXL / Animagine keyframe generation via diffusers.

Single-file backend; switching between Animagine (anime preset) and
SDXL+RealVis (photo preset) is just a matter of changing `model_id` and
the LoRA weights, both of which come from the active preset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lofivid.visuals.base import GeneratedImage, KeyframeBackend, KeyframeSpec

log = logging.getLogger(__name__)


class SDXLKeyframeBackend(KeyframeBackend):
    name = "sdxl"

    def __init__(
        self,
        model_id: str,
        loras: list[tuple[str, float]] | None = None,
        dtype: str = "bfloat16",
        steps: int = 28,
        guidance: float = 6.5,
    ) -> None:
        self.model_id = model_id
        self.loras = loras or []
        self.dtype = dtype
        self.steps = steps
        self.guidance = guidance
        self._pipe: Any = None

    def _ensure_loaded(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as e:
            raise RuntimeError(
                "diffusers/torch not available. Inside Docker this should be pre-installed."
            ) from e

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]

        log.info("Loading SDXL pipeline %s (dtype=%s)", self.model_id, self.dtype)
        pipe = StableDiffusionXLPipeline.from_pretrained(self.model_id, torch_dtype=torch_dtype)
        pipe.to("cuda")

        # Apply LoRAs (if any). Each entry is (huggingface_repo_or_path, weight).
        for lora_id, weight in self.loras:
            log.info("Loading LoRA %s @ %.2f", lora_id, weight)
            pipe.load_lora_weights(lora_id, adapter_name=Path(lora_id).name)
        if self.loras:
            pipe.set_adapters([Path(l).name for l, _ in self.loras],
                              adapter_weights=[w for _, w in self.loras])

        self._pipe = pipe
        return pipe

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

        log.info("Generating keyframe %d (%dx%d): '%s'",
                 spec.scene_index, spec.width, spec.height, spec.prompt[:80])

        import torch
        generator = torch.Generator(device="cuda").manual_seed(spec.seed)
        image = pipe(
            prompt=spec.prompt,
            width=spec.width,
            height=spec.height,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=generator,
        ).images[0]
        image.save(out_path)
        return GeneratedImage(spec=spec, path=out_path)
