"""Anime / Ghibli "lofi girl" preset.

Defaults to Animagine XL 4 + a Ghibli LoRA + a lofi-aesthetic LoRA.
The exact LoRA repos and weights here are reasonable starting points;
override per-config via the `loras` field in the YAML if you fine-tune.
"""

from __future__ import annotations

from lofivid.presets.base import Preset, PresetSpec


class AnimePreset(Preset):
    def spec(self) -> PresetSpec:
        return PresetSpec(
            name="anime",
            # Animagine XL 4 ships under CreativeML Open RAIL++-M (commercial OK with content rules).
            # Avoid Illustrious XL — it's research-license only.
            model_id="cagliostrolab/animagine-xl-4.0",
            loras=[
                # These are example IDs; the real Civitai repos can be mirrored to HF
                # or downloaded into a local /models directory.
                # Format: (huggingface_id_or_local_path, weight)
                # Override in the YAML config's `visuals.loras` if you want different LoRAs.
            ],
            width=1024,
            height=1024,
            quality_suffix=(
                "soft focus, warm lighting, cozy interior, lofi aesthetic, "
                "studio ghibli inspired, masterpiece, best quality"
            ),
            negative_prompt=(
                "lowres, bad anatomy, bad hands, text, error, extra digits, "
                "fewer digits, jpeg artifacts, signature, watermark, blurry"
            ),
        )
