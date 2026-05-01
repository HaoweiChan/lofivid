"""Blackwell sm_120 + ffmpeg encoder preflight gate.

Failing loudly here saves hours of confusing errors later in the pipeline.
The checks are defensive against the specific WSL2 + RTX 50 series quirks
documented in the project plan.

Originally Docker-only. Since the host-mode pivot (see HOST_MODE.md), the
checks also accept libx264 software encode as a working configuration —
NVENC is preferred when available but not required.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from lofivid._ffmpeg import ffmpeg_bin, select_encoder

if TYPE_CHECKING:
    from lofivid.styles.schema import StyleSpec

EXPECTED_COMPUTE_CAPABILITY = (12, 0)  # sm_120 = Blackwell consumer (RTX 50 series)


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
    return CheckResult(
        "torch.cuda", "ok",
        f"{name} (sm_{cap[0]}{cap[1]}), torch={torch.__version__}",
    )


def check_ffmpeg() -> CheckResult:
    """Verify ffmpeg is callable and reports at least one usable video encoder."""
    try:
        binary = ffmpeg_bin()
    except RuntimeError as e:
        return CheckResult("ffmpeg", "fail", str(e))

    try:
        subprocess.run(
            [binary, "-hide_banner", "-version"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return CheckResult("ffmpeg", "fail", f"ffmpeg -version failed: {e}")

    try:
        profile = select_encoder()
    except RuntimeError as e:
        return CheckResult("ffmpeg", "fail", str(e))

    if profile.name in ("av1_nvenc", "hevc_nvenc", "h264_nvenc"):
        return CheckResult("ffmpeg", "ok", f"{binary} → encoder={profile.name} (NVENC)")
    if profile.name == "libx264":
        return CheckResult(
            "ffmpeg", "warn",
            f"{binary} → encoder={profile.name} (CPU). Software encode works but is "
            "slower than NVENC. To enable NVENC, run inside the Docker image or "
            "install a system ffmpeg built with --enable-nvenc.",
        )
    return CheckResult(
        "ffmpeg", "warn",
        f"{binary} → encoder={profile.name} (suboptimal — install ffmpeg with libx264 or NVENC).",
    )


def check_python() -> CheckResult:
    """Project supports 3.11 and 3.12 (per pyproject)."""
    import sys
    major, minor = sys.version_info[:2]
    if (major, minor) not in {(3, 11), (3, 12)}:
        return CheckResult(
            "python", "warn",
            f"Python {major}.{minor} detected; project is tested on 3.11 and 3.12.",
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


def assert_fonts_present(style: StyleSpec, project_root: Path) -> None:
    """Verify every font referenced by `style` exists on disk before any
    GPU work starts. Hard error on missing fonts — there is no fallback.

    Disabled layers / disabled HUD don't contribute paths; their fonts are
    never opened so requiring them on disk would be a footgun.
    """
    paths: list[Path] = []
    for layer in style.brand_layers:
        if not layer.enabled:
            continue
        paths.append(layer.font_path)
        if layer.cjk_font_path:
            paths.append(layer.cjk_font_path)
    if style.hud.enabled:
        paths.append(style.hud.font_path)
        if style.hud.cjk_font_path:
            paths.append(style.hud.cjk_font_path)

    missing: list[Path] = []
    for p in paths:
        resolved = p if p.is_absolute() else (project_root / p)
        if not resolved.exists():
            missing.append(resolved)

    if missing:
        bullet = "\n  - ".join(str(p) for p in missing)
        raise RuntimeError(
            f"Style references {len(missing)} missing font file(s):\n  - {bullet}\n"
            "Place the OFL-licensed fonts at the paths above. There is no "
            "fallback path — see assets/fonts/README.md for the bundled set."
        )
