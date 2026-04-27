"""Photo-realistic cozy-scenes preset.

Defaults to SDXL base + a realistic LoRA. FLUX.1 dev is non-commercial
so we avoid it; FLUX.1 schnell (Apache 2.0) is the fallback if you want
to experiment.
"""

from __future__ import annotations

from lofivid.presets.base import Preset, PresetSpec


class PhotoPreset(Preset):
    def spec(self) -> PresetSpec:
        return PresetSpec(
            name="photo",
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            loras=[],  # add RealVisXL or JuggernautXL via config if desired
            width=1024,
            height=1024,
            quality_suffix=(
                "photorealistic, cinematic lighting, shallow depth of field, "
                "warm colour grading, cozy atmosphere, ultra detailed, 4k"
            ),
            negative_prompt=(
                "cartoon, anime, illustration, painting, drawing, lowres, "
                "blurry, oversaturated, text, watermark"
            ),
        )
