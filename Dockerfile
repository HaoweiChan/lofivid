# lofivid base image — CUDA 12.8 + PyTorch nightly cu128 for RTX 5070 Ti (Blackwell sm_120)
#
# Why nvidia/cuda:12.8.0-devel and not runtime: we may need to compile
# extensions (e.g. xformers community build, ACE-Step ops) against CUDA headers.
#
# Why PyTorch nightly: as of April 2026 the stable PyTorch wheel does not
# expose sm_120 in TORCH_CUDA_ARCH_LIST. Nightly cu128 does. Once stable
# torch>=2.9 ships sm_120 we can swap to a slimmer base.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    # Blackwell workarounds: flash-attn and xformers don't support sm_120 yet.
    # Force PyTorch's native SDPA path and skip torch.compile graph breaks.
    PYTORCH_DISABLE_FLASH_ATTENTION=1 \
    TORCHDYNAMO_DISABLE=1

# ---- System dependencies --------------------------------------------------
# - python3.11: required (3.12 lacks PyTorch nightly binaries as of writing)
# - ffmpeg deps: built from source below to guarantee NVENC + AV1 support
# - libsndfile1, libsox: for audio I/O via torchaudio / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl wget git pkg-config \
        build-essential cmake yasm nasm \
        libsndfile1 libsox-fmt-all sox \
        libgl1 libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip setuptools wheel

# ---- FFmpeg with NVENC + AV1 ----------------------------------------------
# Build FFmpeg with Video Codec SDK 13.0 headers so av1_nvenc is available
# (Blackwell improvements landed Feb 2025). System apt ffmpeg is too old.
RUN git clone --depth=1 -b sdk/13.0 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nvch \
    && cd /tmp/nvch && make install && rm -rf /tmp/nvch

RUN git clone --depth=1 -b release/7.1 https://git.ffmpeg.org/ffmpeg.git /tmp/ffmpeg \
    && cd /tmp/ffmpeg && ./configure \
        --prefix=/usr/local \
        --enable-nonfree \
        --enable-gpl \
        --enable-libx264 --enable-libx265 \
        --enable-nvenc --enable-cuda-nvcc \
        --extra-cflags=-I/usr/local/cuda/include \
        --extra-ldflags=-L/usr/local/cuda/lib64 \
        --enable-shared --disable-static \
    && make -j"$(nproc)" && make install && ldconfig \
    && rm -rf /tmp/ffmpeg

# ---- PyTorch nightly cu128 (Blackwell sm_120) -----------------------------
# Pinned to the cu128 channel; the exact build can drift but the channel
# is stable. If a nightly breaks, fall back to the source-build documented
# at github.com/bajegani/pytorch-build-blackwell-sm120.
RUN pip install --pre \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128

# ---- Python package deps --------------------------------------------------
# Core: diffusers (image gen), transformers, accelerate, soundfile, librosa
RUN pip install \
        "diffusers>=0.30" \
        "transformers>=4.44" \
        "accelerate>=0.33" \
        "safetensors>=0.4" \
        "huggingface_hub>=0.24" \
        "soundfile>=0.12" \
        "librosa>=0.10" \
        "scipy>=1.13" \
        "pyloudnorm>=0.1.1" \
        "depthflow>=0.8" \
        "av>=12.3"

# ACE-Step is installed from GitHub (not yet on PyPI as of April 2026).
# Pinned to a tag rather than main for reproducibility.
RUN pip install "git+https://github.com/ace-step/ACE-Step-1.5@main#egg=acestep" \
    || echo "ACE-Step install failed; will retry at runtime"

# ---- App ------------------------------------------------------------------
WORKDIR /app
COPY pyproject.toml README.md ./
COPY lofivid/ ./lofivid/
COPY configs/ ./configs/
COPY assets/ ./assets/

RUN pip install -e .

# Cache + output mount points (also in docker-compose volumes)
VOLUME ["/app/cache", "/app/output", "/models"]

ENTRYPOINT ["lofivid"]
CMD ["--help"]
