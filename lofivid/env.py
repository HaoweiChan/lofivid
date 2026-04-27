"""Blackwell sm_120 + NVENC preflight gate.

Failing loudly here saves hours of confusing errors later in the pipeline.
The checks are defensive against the specific WSL2 + RTX 50 series quirks
documented in the project plan.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

EXPECTED_COMPUTE_CAPABILITY = (12, 0)  # sm_120 = Blackwell consumer (RTX 50 series)
REQUIRED_NVENC_ENCODER = "av1_nvenc"


@dataclass
class CheckResult:
    name: str
    status: Literal["ok", "warn", "fail"]
    detail: str


def check_torch_cuda() -> CheckResult:
    """Verify PyTorch sees a Blackwell GPU."""
    try:
        import torch
    except ImportError:
        return CheckResult(
            "torch", "fail",
            "PyTorch not installed. Inside the Docker image this should never happen; "
            "outside it, install nightly cu128: "
            "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128",
        )

    if not torch.cuda.is_available():
        return CheckResult(
            "torch.cuda", "fail",
            "CUDA not available. On WSL2: ensure Windows NVIDIA driver >= 572.xx and "
            "do NOT install nvidia drivers inside WSL.",
        )

    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    if cap != EXPECTED_COMPUTE_CAPABILITY:
        return CheckResult(
            "torch.cuda", "warn",
            f"Detected {name} with compute capability {cap}, expected sm_120 ({EXPECTED_COMPUTE_CAPABILITY}). "
            "Pipeline may still work but was tuned for RTX 5070 Ti.",
        )
    return CheckResult("torch.cuda", "ok", f"{name} (sm_{cap[0]}{cap[1]}), torch={torch.__version__}")


def check_ffmpeg() -> CheckResult:
    """Verify ffmpeg binary exists and reports av1_nvenc."""
    binary = shutil.which("ffmpeg")
    if binary is None:
        return CheckResult("ffmpeg", "fail", "ffmpeg not on PATH")

    try:
        out = subprocess.run(
            [binary, "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return CheckResult("ffmpeg", "fail", f"ffmpeg -encoders failed: {e}")

    if REQUIRED_NVENC_ENCODER not in out:
        # Fall back to hevc_nvenc, but warn — av1_nvenc is the recommended path.
        if "hevc_nvenc" in out:
            return CheckResult(
                "ffmpeg", "warn",
                f"{REQUIRED_NVENC_ENCODER} missing; only hevc_nvenc available. "
                "Quality/efficiency will be lower. Rebuild ffmpeg with Video Codec SDK 13.0.",
            )
        return CheckResult(
            "ffmpeg", "fail",
            "No NVENC encoders detected. Rebuild ffmpeg with --enable-nvenc and the SDK 13.0 headers.",
        )
    return CheckResult("ffmpeg", "ok", f"ffmpeg with {REQUIRED_NVENC_ENCODER} at {binary}")


def check_python() -> CheckResult:
    """3.11 only — 3.12 lacks PyTorch nightly binaries as of writing."""
    import sys
    major, minor = sys.version_info[:2]
    if (major, minor) != (3, 11):
        return CheckResult(
            "python", "warn",
            f"Python {major}.{minor} detected; project is pinned to 3.11 because "
            "PyTorch nightly cu128 ships 3.11 wheels.",
        )
    return CheckResult("python", "ok", f"Python {major}.{minor}")


def run_all_checks() -> list[CheckResult]:
    return [check_python(), check_torch_cuda(), check_ffmpeg()]


def assert_ready() -> None:
    """Raise RuntimeError if any check fails. Call before loading models."""
    failures = [c for c in run_all_checks() if c.status == "fail"]
    if failures:
        details = "\n".join(f"  - {c.name}: {c.detail}" for c in failures)
        raise RuntimeError(f"Environment preflight failed:\n{details}")
