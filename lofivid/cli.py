"""Typer CLI: `lofivid generate | music-only | visuals-only | verify-env | licenses`.

The CLI is deliberately thin — orchestration lives in `lofivid.pipeline`,
which gives us a place to write end-to-end tests without going through Typer.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

def _apply_blackwell_env_defaults() -> None:
    """Apply env-var workarounds for the WSL2 + RTX 50 series + PyTorch nightly stack.

    These are no-ops on other configurations (CUDA driver ignores unknown vars).
    Set BEFORE torch is first imported — that's why this runs at module load.
    """
    import os
    os.environ.setdefault(
        # Default-on: PyTorch's expandable allocator handles the bogus
        # "non-PyTorch memory = 17 EiB" reading we see on WSL2 + sm_120.
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True",
    )
    os.environ.setdefault(
        # flash-attn / xformers don't support sm_120 yet (April 2026).
        "PYTORCH_DISABLE_FLASH_ATTENTION", "1",
    )
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    # HF download speed-up (requires hf_transfer pkg, included in pyproject).
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


_apply_blackwell_env_defaults()

app = typer.Typer(
    name="lofivid",
    help="Generate AI lofi music videos locally on RTX 5070 Ti (Blackwell sm_120).",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, markup=True)],
    )


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug-level logs"),
    memory_cap_gb: float = typer.Option(
        12.0, "--memory-cap-gb",
        help="Soft cap on Python heap (RLIMIT_DATA) + RSS watchdog. "
        "Set 0 to disable. WSL2 freezes when the pipeline exceeds the WSL "
        "allocation; this gives an early WARN before the system locks up.",
    ),
) -> None:
    _setup_logging(verbose)
    from lofivid._memcap import apply_memory_cap
    apply_memory_cap(memory_cap_gb if memory_cap_gb > 0 else None)


@app.command("verify-env")
def verify_env() -> None:
    """Run all preflight checks (sm_120 GPU, FFmpeg with NVENC, Python 3.11).

    Exits 0 on full success, 1 on any failure, 2 if only warnings.
    """
    from lofivid.env import run_all_checks

    table = Table(title="lofivid environment preflight")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")

    results = run_all_checks()
    has_fail = False
    has_warn = False
    for r in results:
        colour = {"ok": "green", "warn": "yellow", "fail": "red"}[r.status]
        table.add_row(r.name, f"[{colour}]{r.status.upper()}[/]", r.detail)
        has_fail |= r.status == "fail"
        has_warn |= r.status == "warn"

    console.print(table)
    sys.exit(1 if has_fail else (2 if has_warn else 0))


@app.command()
def generate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True),
    cache_dir: Path = typer.Option(Path("cache"), "--cache-dir"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir"),
    skip_preflight: bool = typer.Option(False, help="Skip Blackwell env checks (debugging only)"),
) -> None:
    """End-to-end generate: music + visuals + composition."""
    from lofivid import pipeline
    from lofivid.env import assert_ready

    if not skip_preflight:
        assert_ready()

    pipeline.generate(config_path=config, cache_dir=cache_dir, output_dir=output_dir)


@app.command("music-only")
def music_only(
    config: Path = typer.Option(..., "--config", "-c", exists=True),
    cache_dir: Path = typer.Option(Path("cache"), "--cache-dir"),
) -> None:
    """Generate just the audio track (useful for iterating on tracklist)."""
    from lofivid import pipeline

    pipeline.generate_music_only(config_path=config, cache_dir=cache_dir)


@app.command("visuals-only")
def visuals_only(
    config: Path = typer.Option(..., "--config", "-c", exists=True),
    cache_dir: Path = typer.Option(Path("cache"), "--cache-dir"),
) -> None:
    """Generate just the visual scenes (no music or composition)."""
    from lofivid import pipeline

    pipeline.generate_visuals_only(config_path=config, cache_dir=cache_dir)


@app.command()
def licenses() -> None:
    """Print every model + asset license used by the pipeline.

    Run this before publishing any output commercially.
    """
    rows = [
        ("ACE-Step 1.5 (code)", "Apache-2.0", "Music generation"),
        ("ACE-Step 1.5 (weights)", "verify per-release", "Check repo at install time"),
        ("Animagine XL 4 (anime preset)", "CreativeML Open RAIL++-M", "Commercial OK with content rules"),
        ("SDXL base 1.0 (photo preset)", "CreativeML Open RAIL++-M", "Commercial OK with content rules"),
        ("DepthFlow", "LGPL", "Parallax animation"),
        ("Depth Anything v2", "Apache-2.0 (weights vary)", "Depth maps"),
        ("LTX-Video (optional)", "Apache-2.0 (verify model card)", "Img→video motion clips"),
        ("Wan 2.2 (optional)", "Apache-2.0", "Premium animated scenes"),
        ("FFmpeg", "LGPL/GPL (build-dependent)", "Video composition"),
        ("Diffusers / Transformers", "Apache-2.0", "Inference libraries"),
        ("Bundled CC0 assets", "CC0", "See assets/ directory"),
    ]
    table = Table(title="lofivid third-party licenses")
    table.add_column("component"); table.add_column("license"); table.add_column("notes")
    for r in rows:
        table.add_row(*r)
    console.print(table)
    console.print(
        "[yellow]Always verify model-weight licenses at the actual download URL "
        "before commercial use; weights and code can be licensed differently.[/]"
    )


if __name__ == "__main__":
    app()
