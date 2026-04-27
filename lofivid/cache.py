"""Content-addressed disk cache.

Why: a 2-hour render takes ~90 minutes of GPU time. If composition fails
because of a typo in the FFmpeg overlay path, regenerating 20 ACE-Step
tracks is unacceptable. Each pipeline stage writes a manifest entry
keyed by hash(stage_inputs); subsequent runs short-circuit hits.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def content_hash(payload: Any) -> str:
    """Stable JSON-based hash of any plain-data payload."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class CacheEntry:
    stage: str
    key: str
    path: Path
    created_at: float


class Cache:
    """SQLite-backed manifest of (stage, key) -> file path.

    The cache directory layout:
      cache/
        manifest.sqlite       # this DB
        <stage>/<key>.<ext>   # actual artifacts (audio, images, video)
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS entries (
        stage TEXT NOT NULL,
        key TEXT NOT NULL,
        path TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (stage, key)
    );
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._db_path = self.root / "manifest.sqlite"
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, stage: str, key: str) -> Path | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT path FROM entries WHERE stage = ? AND key = ?",
                (stage, key),
            ).fetchone()
        if row is None:
            return None
        path = Path(row[0])
        if not path.exists():
            # Manifest is stale — caller should regenerate.
            self.invalidate(stage, key)
            return None
        return path

    def put(self, stage: str, key: str, path: Path) -> CacheEntry:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"cache.put given missing file: {path}")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO entries (stage, key, path, created_at) VALUES (?, ?, ?, ?)",
                (stage, key, str(path), time.time()),
            )
        return CacheEntry(stage=stage, key=key, path=path, created_at=time.time())

    def invalidate(self, stage: str, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM entries WHERE stage = ? AND key = ?", (stage, key))

    def stage_dir(self, stage: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def all_entries(self, stage: str | None = None) -> list[CacheEntry]:
        with self._connect() as conn:
            if stage is None:
                rows = conn.execute("SELECT stage, key, path, created_at FROM entries").fetchall()
            else:
                rows = conn.execute(
                    "SELECT stage, key, path, created_at FROM entries WHERE stage = ?",
                    (stage,),
                ).fetchall()
        return [CacheEntry(stage=s, key=k, path=Path(p), created_at=t) for s, k, p, t in rows]
