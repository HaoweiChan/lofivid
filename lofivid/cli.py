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


@app.command("music-ingest")
def music_ingest(
    source: str = typer.Option(..., "--source", help="Ingest source: 'pixabay' | 'manual'."),
    mood: str = typer.Option(..., "--mood", help="Mood slug — destination subfolder name "
                                                 "and (without --style) the search tag."),
    target: Path = typer.Option(..., "--target", help="Destination directory for WAVs + sidecars."),
    count: int = typer.Option(20, "--count", min=1, max=200,
                              help="Max NEW tracks to fetch this run (already-downloaded tracks "
                                   "are skipped via sidecar source_id)."),
    style: str | None = typer.Option(None, "--style",
                                     help="Optional: read library_search_tags[<mood>] from this style."),
    min_duration: float = typer.Option(60.0, "--min-duration", min=0.0),
    max_duration: float = typer.Option(600.0, "--max-duration", min=0.0),
    rate_limit_s: float = typer.Option(0.7, "--rate-limit-s", min=0.0,
                                       help="Per-download throttle for cloud sources."),
    license_str: str = typer.Option(
        "manual-licensed", "--license",
        help="Only used by --source manual. Sidecar `license` field.",
    ),
    attribution_text: str | None = typer.Option(
        None, "--attribution-text",
        help="Only used by --source manual. Sidecar `attribution_text` field.",
    ),
    license_certificate_url: str | None = typer.Option(
        None, "--license-certificate-url",
        help="Only used by --source manual. URL of a downloadable license "
             "certificate (e.g. proof you can attach when disputing a Content "
             "ID claim). Sidecar `license_certificate_url` field.",
    ),
) -> None:
    """Populate a music library folder from an external source.

    Examples:

      \b
      # Pixabay (Pixabay Content License — commercial OK, no attribution required;
      # individual tracks may still trigger Content ID claims, see assets/music/README.md):
      lofivid music-ingest --source pixabay --style morning_cafe --mood cafe_afternoon \\
          --count 20 --target assets/music/cafe_jazz/cafe_afternoon/

      \b
      # Manual: validate pre-licensed local WAVs and write sidecars:
      lofivid music-ingest --source manual --mood cafe_afternoon \\
          --target assets/music/cafe_jazz/cafe_afternoon/ \\
          --license cc0 --attribution-text "Track \\"X\\" by Y, CC0"

    Re-running with the same arguments is idempotent: existing sidecars are
    not overwritten, and tracks with a matching `source_id` are skipped.
    """
    import lofivid.ingest  # noqa: F401  — trigger registration
    from lofivid.ingest.base import (
        available_sources,
        existing_source_ids,
    )
    from lofivid.ingest.base import (
        get as get_source,
    )

    target.mkdir(parents=True, exist_ok=True)

    if source not in available_sources():
        console.print(
            f"[red]Unknown source {source!r}. Available: "
            f"{', '.join(available_sources())}[/]"
        )
        sys.exit(2)

    mood_tags = _resolve_mood_tags(mood=mood, style_name=style)

    src_cls = get_source(source)
    if source == "manual":
        instance = src_cls(
            license=license_str,
            attribution_text=attribution_text,
            license_certificate_url=license_certificate_url,
        )
    elif source == "pixabay":
        instance = src_cls(rate_limit_s=rate_limit_s)
    else:
        instance = src_cls()  # future-proof for sources with default ctors

    already = existing_source_ids(target)
    if already:
        console.print(f"[dim]Skipping {len(already)} already-downloaded source_id(s) in {target}.[/]")

    fetched = instance.fetch(
        mood_tags=mood_tags,
        count=count,
        target_dir=target,
        min_duration_s=min_duration,
        max_duration_s=max_duration,
        already_downloaded=already,
    )

    for t in fetched:
        line = f"  [green]+[/] {t.local_path.name}  ({t.duration_s:.0f}s, license={t.license}"
        if t.attribution_text:
            line += f", attrib={t.attribution_text!r}"
        line += ")"
        console.print(line)

    if not fetched:
        if already:
            # Idempotent rerun: every source_id we'd consider was already present.
            console.print(
                f"[bold]Nothing to do — {len(already)} track(s) already present in {target}.[/]"
            )
            return
        console.print(
            f"[red]No tracks fetched (source={source!r}, mood={mood!r}, "
            f"tags={mood_tags}). Likely causes: API failure, filter too strict, "
            f"or empty target directory for --source manual.[/]"
        )
        sys.exit(1)
    console.print(f"[bold]Fetched {len(fetched)} new track(s) into {target}.[/]")


def _resolve_mood_tags(mood: str, style_name: str | None) -> list[str]:
    """Resolve the search-tag list for `mood`. Falls back to [mood] when no style given."""
    if style_name is None:
        return [mood]
    from lofivid.config import _repo_root
    from lofivid.styles.loader import load_style

    spec, _ = load_style(style_name, _repo_root())
    tags = spec.library_search_tags.get(mood)
    if not tags:
        console.print(
            f"[yellow]Style {style_name!r} has no library_search_tags[{mood!r}]; "
            f"falling back to [{mood!r}] as the only search tag.[/]"
        )
        return [mood]
    return list(tags)


@app.command()
def licenses() -> None:
    """Print every model + asset license used by the pipeline.

    Run this before publishing any output commercially.
    """
    rows = [
        ("ACE-Step 1.5 (code)", "Apache-2.0", "Music generation (local)"),
        ("ACE-Step 1.5 (weights)", "verify per-release", "Check repo at install time"),
        ("Suno (cloud)", "see notes", "CLOUD service. AI-only output is NOT copyrightable in the US."),
        ("Suno API wrapper (3rd-party)", "wrapper-dependent", "sunoapi.org/PiAPI/AIML — verify ToS yourself"),
        ("Library music (Pixabay, ingest)", "Pixabay Content License", "Commercial OK + monetisation OK as part of a larger creative work. Per-track Content ID risk: see manifest's `at_risk_for_content_id_claim` flag — disputable, not a copyright strike."),
        ("Library music (manual)", "user-supplied", "For pre-licensed audio of any provenance. The `--source manual` ingest writes a sidecar with whatever `--license` / `--attribution-text` / `--license-certificate-url` you pass. You hold the proof."),
        ("Animagine XL 4 (anime preset)", "CreativeML Open RAIL++-M", "Commercial OK with content rules"),
        ("SDXL base 1.0 (photo preset)", "CreativeML Open RAIL++-M", "Commercial OK with content rules"),
        ("FLUX.2 Klein 4B", "Apache-2.0", "2025-11-25 release; commercial-safe upgrade from SDXL"),
        ("Z-Image-Turbo 6B", "Apache-2.0", "2025-12-04 Tongyi-MAI release; fast distilled commercial-safe"),
        ("Unsplash photos", "Unsplash License", "Free commercial use; attribution required"),
        ("DepthFlow", "LGPL", "Parallax animation (local)"),
        ("Depth Anything v2", "Apache-2.0 (weights vary)", "Depth maps"),
        ("Overlay-motion (FFmpeg filters)", "n/a", "Pure FFmpeg filter graphs, no model weights"),
        ("LTX-Video (optional)", "Apache-2.0 (verify model card)", "Img→video motion clips"),
        ("Wan 2.2 (optional)", "Apache-2.0", "Premium animated scenes"),
        ("FFmpeg", "LGPL/GPL (build-dependent)", "Video composition"),
        ("Diffusers / Transformers", "Apache-2.0", "Inference libraries"),
        ("Playfair Display Bold (font)", "SIL OFL 1.1", "Display serif for titles. Bundled at assets/fonts/."),
        ("IBM Plex Sans Regular + Bold (font)", "SIL OFL 1.1", "Body / kicker / counter / artist text."),
        ("Noto Sans CJK TC Bold (font)", "SIL OFL 1.1", "Traditional-Chinese fallback for the CJK subtitle line."),
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
    console.print(
        "[bold yellow]Suno-generated audio is not eligible for copyright as a "
        "fully-AI work. To claim copyright on the music, write your own lyrics "
        "and meaningfully edit the stems. Verify your Suno subscription tier "
        "permits commercial use BEFORE publishing.[/]"
    )
    console.print(
        "[bold yellow]Unsplash images require photographer attribution. The "
        "backend writes a `<scene>.jpg.attribution.txt` sidecar per image — "
        "include the credits in your channel description / video credits.[/]"
    )
    console.print(
        "[bold yellow]Library music tracks carry per-track sidecars at "
        "`<wav>.attribution.json`. The pipeline surfaces them through manifest's "
        "`music_attributions` list — copy non-null `attribution_text` entries "
        "into your video description, and watch entries with "
        "`at_risk_for_content_id_claim: true` (Pixabay tracks without a license "
        "certificate may trigger automated YouTube Content ID claims; dispute "
        "with the track URL + license-summary URL — see assets/music/README.md).[/]"
    )


if __name__ == "__main__":
    app()
