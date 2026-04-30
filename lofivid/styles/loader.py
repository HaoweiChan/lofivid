"""Style loading and content hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from lofivid.styles.schema import StyleSpec


def load_style(name: str, root: Path) -> tuple[StyleSpec, str]:
    """Load <root>/styles/<name>.yaml and return (spec, hash_hex_prefix).

    `hash_hex_prefix` is canonical-JSON SHA-256, first 12 hex chars,
    excluding the description field (so doc edits don't invalidate caches).
    """
    path = root / "styles" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Style {name!r} not found at {path}. "
            f"Place a YAML file at <repo>/styles/{name}.yaml."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    spec = StyleSpec.model_validate(data)
    return spec, style_hash(spec)


def style_hash(style: StyleSpec) -> str:
    """Stable 12-char hex SHA-256 of the style, excluding `description`."""
    payload = style.model_dump(mode="json", exclude={"description"})
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
