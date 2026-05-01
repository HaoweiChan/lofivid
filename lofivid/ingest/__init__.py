"""Music ingest layer — adapters that populate `assets/music/<mood>/` from external sources.

Three sources, behind a common `IngestSource` ABC:
  - `pixabay`   : public API, Pixabay Content License (CC0-equivalent for commercial use).
  - `fma`       : Free Music Archive (currently disabled pending step-0b verification).
  - `manual`    : no download — validates pre-licensed local WAVs (Epidemic Sound etc.)
                  and writes sidecar JSON so attribution flows through the manifest.

Each adapter writes the WAV's title/artist/duration into audio metadata tags via
mutagen, and writes a `<filename>.attribution.json` sidecar with source + license
+ attribution_text. `LibraryMusicBackend` reads the sidecar at render time and
the pipeline surfaces it through manifest's `music_attributions`.

Importing this package triggers source registration (manual + pixabay).
"""
from __future__ import annotations

from lofivid.ingest import manual, pixabay  # noqa: F401  (registration side-effect)
