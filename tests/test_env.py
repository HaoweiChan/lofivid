"""Smoke tests for env preflight.

These tests don't require a GPU. They verify that the check functions
return well-formed CheckResult objects and don't crash when prereqs are missing.
"""

from __future__ import annotations

from lofivid.env import CheckResult, check_ffmpeg, check_python, check_torch_cuda, run_all_checks


def test_check_python_returns_result():
    r = check_python()
    assert isinstance(r, CheckResult)
    assert r.name == "python"
    assert r.status in {"ok", "warn", "fail"}


def test_check_torch_cuda_returns_result_even_without_torch():
    r = check_torch_cuda()
    assert isinstance(r, CheckResult)
    assert r.name in {"torch", "torch.cuda"}


def test_check_ffmpeg_returns_result_even_without_ffmpeg():
    r = check_ffmpeg()
    assert isinstance(r, CheckResult)
    assert r.name == "ffmpeg"


def test_run_all_checks_returns_three():
    results = run_all_checks()
    assert len(results) == 3
    assert {r.name for r in results} >= {"python", "ffmpeg"}
