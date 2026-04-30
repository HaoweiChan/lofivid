"""Render a 10-second loop per OverlayMotion preset using one keyframe.

Outputs:
  previews/motion/<keyframe_stem>_<motion_type>.mp4

Useful for sanity-checking the new minimal_design_lofi.yaml direction —
play the resulting MP4s and confirm:
  • slow_zoom is gentle and seamless at the seam (frame N == frame 0)
  • light_flicker breath is subtle, not strobing
  • dust_motes drift slowly, not snowing
  • none is a literal still

Usage:
  python scripts/preview_overlay_motion.py [keyframe.png ...]
  python scripts/preview_overlay_motion.py     # uses the first greenhouse still

Each loop is 10 seconds at 24 fps (240 frames). The full pipeline uses
30-second loops; 10s here keeps the previews quick to scan.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lofivid.visuals.base import ParallaxSpec
from lofivid.visuals.overlay_motion import MotionType, OverlayMotionBackend

OUT_DIR = ROOT / "previews" / "motion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_KEYFRAME = ROOT / "previews" / "greenhouse" / "01_monstera_graded.png"

MOTIONS: list[MotionType] = ["slow_zoom", "light_flicker", "dust_motes", "none"]
DURATION_SECONDS = 10
WIDTH, HEIGHT = 1920, 1080
FPS = 24


def render_one(keyframe: Path, motion: MotionType) -> Path:
    backend = OverlayMotionBackend(motion_type=motion)
    spec = ParallaxSpec(
        scene_index=0,
        image_path=keyframe,
        duration_seconds=DURATION_SECONDS,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        seed=42,
    )
    out_dir = OUT_DIR / keyframe.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = backend.generate(spec, out_dir)
    # Rename from "000.mp4" to "<motion>.mp4" for human-readable previews.
    final = out_dir / f"{motion}.mp4"
    clip.path.rename(final)
    return final


def main() -> None:
    keyframes = [Path(a) for a in sys.argv[1:]] or [DEFAULT_KEYFRAME]
    for kf in keyframes:
        if not kf.exists():
            print(f"skip: {kf} (not found)")
            continue
        print(f"=== {kf.name} ===")
        for motion in MOTIONS:
            try:
                out = render_one(kf, motion)
                size_mb = out.stat().st_size / 1e6
                print(f"  {motion:14s} → {out.relative_to(ROOT)}  ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  {motion:14s} FAILED: {e}")


if __name__ == "__main__":
    main()
