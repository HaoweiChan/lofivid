# Open-weight image-generation models — commercial-OK shortlist

Last researched: **2026-04-28**.
Constraint matrix: must run on **NVIDIA RTX 5070 Ti (Blackwell sm_120, 16 GB)**,
must allow **commercial monetisation** of generated outputs, weights must be
**publicly downloadable** (not API-only), released **2025 or later**.

## TL;DR

| Rank | Model | Released | License | Params | Fits 16 GB | Best for |
|---|---|---|---|---|---|---|
| 1 | **FLUX.2 Klein** | 2025-11-25 | Apache 2.0 ✅ | 4B | yes (~8 GB bf16) | Production photography; best balance of quality/speed/license |
| 2 | **Z-Image-Turbo** | 2025-12-04 | Apache 2.0 ✅ | 6B | yes | Fast iteration (8-step distilled), latest tech, Alibaba/Tongyi-MAI lineage |
| 3 | **Qwen-Image** (original) | 2025-08-04 | Apache 2.0 ✅ | 20B | only via GGUF Q5/Q6 quant | Highest capability when 16 GB allows; specifically tuned to avoid the "waxy AI look" |

Anything outside this shortlist either fails the date filter (SDXL/SD 1.5/SDXL
base 1.0/FLUX.1 family) or fails the licence filter (FLUX.1 Krea, FLUX.1 dev,
FLUX.2 dev — all non-commercial).

## Detailed scratchpad

### Disqualified — pre-2025

| Model | Released | License | Why it's out |
|---|---|---|---|
| SDXL base 1.0 | 2023-07 | OpenRAIL-M ✅ | Date filter; produces visibly "AI" cafe/lifestyle output |
| Animagine XL 4 | 2024-12 | OpenRAIL-M ✅ | Date filter (anime-only anyway) |
| FLUX.1 schnell | 2024-08 | Apache 2.0 ✅ | Date filter — predates 2025 cutoff. Otherwise excellent. |
| FLUX.1 dev | 2024-08 | non-commercial | Date + license |
| Stable Diffusion 3.5 (Large/Medium) | 2024-10 | Stability Community (free <$1M rev) | Date filter; conditional licence anyway |

### Disqualified — non-commercial license

| Model | Released | License | Why it's out |
|---|---|---|---|
| FLUX.1 Krea dev | 2025-07 | flux-1-dev-non-commercial | License blocks monetised use; **otherwise the strongest "no AI look" photo model in open weights** — note as the upgrade path if licence ever changes |
| FLUX.2 dev | 2025-11-25 | non-commercial | License blocks monetised use; 32B is also too heavy for 16 GB |
| FLUX.2 pro | 2025-11-25 | closed (BFL API only) | API-only, no weights |

### Disqualified — weights not yet public

| Model | Released | Status |
|---|---|---|
| Qwen-Image-2.0 | 2026-02-10 | API-only on Alibaba Cloud BaiLian platform; **monitor — weights expected later** |

### Other candidates considered

| Model | Released | License | Why it didn't make the shortlist |
|---|---|---|---|
| HunyuanDiT | 2024 | Tencent license w/ commercial-use opt-in | Date filter |
| Lumina-Next | 2024 | Apache 2.0 | Date filter |
| AuraFlow | 2024 | Apache 2.0 | Date filter; quality below FLUX |
| Pixart-α / Pixart-Σ | 2024 | OpenRAIL | Date filter |
| Cosmos-2 / SD3.5-Turbo / etc. | various 2024 | various | Date filter |

## Per-model notes

### FLUX.2 Klein (Black Forest Labs)

- 4B params — distilled from the FLUX.2 family
- Apache 2.0 — full commercial use of weights and outputs
- Released 25 Nov 2025 alongside FLUX.2 Pro / Dev / Flex
- Inference: 4–8 steps typical; flow-matching transformer architecture
- VRAM: ~8 GB at bf16, fits comfortably in 16 GB with headroom for the encoder + activations
- HF: `black-forest-labs/FLUX.2-klein-4B`
- Repo: <https://github.com/black-forest-labs/flux2>

### Z-Image-Turbo (Tongyi-MAI / Alibaba)

- 6B params — distilled, 8 NFE for full-quality generation
- Apache 2.0 — fully open commercial use
- Released 4 Dec 2025
- Sub-second per image on enterprise GPUs; ~2–4 s on a 5070 Ti expected
- HF: `Tongyi-MAI/Z-Image-Turbo`
- Quantised forks already exist: `lightx2v/Z-Image-Turbo-Quantized`, `drbaph/Z-Image-Turbo-FP8` if VRAM gets tight

### Qwen-Image (Alibaba)

- 20B MMDiT — *largest* commercial-OK open-weight image model right now
- Apache 2.0 — fully open commercial use
- Released 4 Aug 2025
- Specifically called out for avoiding the "waxy AI look" — generates micro-textures, fabric weave, skin imperfections
- Native bf16 needs 30–40 GB → **does not fit in 16 GB at full precision**
- Workable on 16 GB only via GGUF Q5_K_M / Q6_K quantisation; expect slower inference vs Klein/Z-Image
- HF: `Qwen/Qwen-Image`
- Quantised forks: see `Qwen-Image-2512-GGUF` family

## Decision log

- **2026-04-28** — pivot away from SDXL base 1.0. Visible AI tells are killing the lifestyle direction. SDXL fine-tunes (Juggernaut XL, RealVis XL) considered but rejected because they share the SDXL-era ceiling.
- **Strict filter applied**: ≥2025 only — user position is "models prior to 2025 are trash".
- **Primary**: try FLUX.2 Klein first; expect best quality:license:VRAM trade-off.
- **Secondary**: A/B against Z-Image-Turbo on identical prompts; faster, slightly less detail expected.
- **Tertiary**: Qwen-Image (GGUF) if Klein's output needs more texture detail; accept the speed hit.
- **Watchlist**: Qwen-Image-2.0 weights — drop in here when public.

## Sources

- <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B>
- <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>
- <https://huggingface.co/Qwen/Qwen-Image>
- <https://www.marktechpost.com/2025/11/25/black-forest-labs-releases-flux-2-a-32b-flow-matching-transformer-for-production-image-pipelines/>
- <https://en.wikipedia.org/wiki/Flux_(text-to-image_model)>
- <https://qwenimages.com/blog/qwen-image-2-release>
- <https://www.bentoml.com/blog/a-guide-to-open-source-image-generation-models>
- <https://www.siliconflow.com/articles/en/best-open-source-models-for-photorealism>
