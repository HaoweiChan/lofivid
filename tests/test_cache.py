"""Tests for the SQLite content-addressed cache."""

from __future__ import annotations

from pathlib import Path

from lofivid.cache import Cache, content_hash


def test_content_hash_is_stable_for_equivalent_payloads():
    a = {"prompt": "hello", "seed": 42, "tags": ["x", "y"]}
    b = {"tags": ["x", "y"], "seed": 42, "prompt": "hello"}  # key order differs
    assert content_hash(a) == content_hash(b)


def test_content_hash_changes_when_payload_changes():
    a = {"prompt": "hello", "seed": 42}
    b = {"prompt": "hello", "seed": 43}
    assert content_hash(a) != content_hash(b)


def test_cache_put_and_get_roundtrip(tmp_cache_dir: Path):
    cache = Cache(tmp_cache_dir)
    f = tmp_cache_dir / "artifact.bin"
    f.write_bytes(b"x")

    cache.put("music_track", "abc123", f)
    got = cache.get("music_track", "abc123")
    assert got == f.resolve()


def test_cache_miss_returns_none(tmp_cache_dir: Path):
    cache = Cache(tmp_cache_dir)
    assert cache.get("music_track", "nonexistent") is None


def test_cache_invalidates_when_underlying_file_disappears(tmp_cache_dir: Path):
    cache = Cache(tmp_cache_dir)
    f = tmp_cache_dir / "artifact.bin"
    f.write_bytes(b"x")
    cache.put("music_track", "abc123", f)

    f.unlink()  # simulate the artifact being deleted
    assert cache.get("music_track", "abc123") is None
    # And the manifest entry should have been removed too
    assert not cache.all_entries("music_track")


def test_cache_put_overwrites_existing_key(tmp_cache_dir: Path):
    cache = Cache(tmp_cache_dir)
    f1 = tmp_cache_dir / "v1.bin"
    f2 = tmp_cache_dir / "v2.bin"
    f1.write_bytes(b"v1")
    f2.write_bytes(b"v2")
    cache.put("k", "x", f1)
    cache.put("k", "x", f2)
    assert cache.get("k", "x") == f2.resolve()


def test_cache_stage_dir_is_created(tmp_cache_dir: Path):
    cache = Cache(tmp_cache_dir)
    d = cache.stage_dir("music_tracks")
    assert d.exists() and d.is_dir()
