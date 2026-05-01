"""Tests for ManualIngestSource (writes sidecars for pre-existing local audio)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lofivid.ingest.base import sidecar_path_for
from lofivid.ingest.manual import ManualIngestSource


def _silent_wav(path: Path, duration_s: float = 1.0, sr: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(int(duration_s * sr), dtype=np.int16)
    sf.write(str(path), data, sr)


def test_manual_writes_sidecars_for_unsidecared_wavs(tmp_path):
    target = tmp_path / "lib"
    target.mkdir()
    _silent_wav(target / "song_a.wav")
    _silent_wav(target / "song_b.wav")

    src = ManualIngestSource(license="manual-licensed", attribution_text="Test attrib")
    fetched = src.fetch(
        mood_tags=[], count=99, target_dir=target,
        already_downloaded=set(),
    )

    assert len(fetched) == 2
    for name in ("song_a", "song_b"):
        side = target / f"{name}.attribution.json"
        assert side.exists()
        data = json.loads(side.read_text())
        assert data["source"] == "manual"
        assert data["license"] == "manual-licensed"
        assert data["attribution_text"] == "Test attrib"
        assert data["source_id"] == f"{name}.wav"


def test_manual_does_not_overwrite_existing_sidecars(tmp_path):
    target = tmp_path / "lib"
    target.mkdir()
    audio = target / "song.wav"
    _silent_wav(audio)
    side = sidecar_path_for(audio)
    side.write_text(
        json.dumps(
            {
                "source": "manual",
                "source_id": "song.wav",
                "license": "preexisting-license",
                "attribution_text": "preexisting",
                "original_url": "local",
                "fetched_at": "2024-01-01T00:00:00Z",
            }
        )
    )

    src = ManualIngestSource(license="should-not-replace")
    fetched = src.fetch(mood_tags=[], count=99, target_dir=target)

    assert fetched == []
    data = json.loads(side.read_text())
    assert data["license"] == "preexisting-license"


def test_manual_raises_on_missing_target_dir(tmp_path):
    target = tmp_path / "does_not_exist"
    src = ManualIngestSource()
    with pytest.raises(FileNotFoundError, match="target_dir does not exist"):
        src.fetch(mood_tags=[], count=1, target_dir=target)


def test_manual_skips_non_audio_files(tmp_path):
    target = tmp_path / "lib"
    target.mkdir()
    _silent_wav(target / "song.wav")
    (target / "notes.txt").write_text("not audio")
    (target / "cover.jpg").write_bytes(b"\x00\x01\x02")

    src = ManualIngestSource()
    fetched = src.fetch(mood_tags=[], count=99, target_dir=target)

    assert len(fetched) == 1
    assert fetched[0].local_path.name == "song.wav"
    assert not (target / "notes.attribution.json").exists()
    assert not (target / "cover.attribution.json").exists()


def test_manual_idempotent_on_rerun(tmp_path):
    target = tmp_path / "lib"
    target.mkdir()
    _silent_wav(target / "song.wav")

    src = ManualIngestSource()
    first = src.fetch(mood_tags=[], count=99, target_dir=target)
    assert len(first) == 1

    second = src.fetch(mood_tags=[], count=99, target_dir=target)
    assert second == []


def test_manual_propagates_license_certificate_url(tmp_path):
    target = tmp_path / "lib"
    target.mkdir()
    _silent_wav(target / "song.wav")
    cert = "https://example.com/proof.pdf"

    src = ManualIngestSource(
        license="cc0",
        attribution_text=None,
        license_certificate_url=cert,
    )
    fetched = src.fetch(mood_tags=[], count=99, target_dir=target)
    assert len(fetched) == 1
    assert fetched[0].license_certificate_url == cert

    sidecar = json.loads((target / "song.attribution.json").read_text())
    assert sidecar["license_certificate_url"] == cert


def test_manual_certificate_url_defaults_to_none(tmp_path):
    target = tmp_path / "lib"
    target.mkdir()
    _silent_wav(target / "song.wav")

    src = ManualIngestSource(license="cc0")
    fetched = src.fetch(mood_tags=[], count=99, target_dir=target)
    assert fetched[0].license_certificate_url is None
    sidecar = json.loads((target / "song.attribution.json").read_text())
    assert sidecar["license_certificate_url"] is None
