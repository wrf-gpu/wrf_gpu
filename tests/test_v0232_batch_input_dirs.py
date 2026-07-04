"""v0.23.2 — GPUWRF_BATCH_INPUT_DIRS separator ergonomics (ALISIOS operational fix).

The distinct-init batched-ensemble contract (GPUWRF_BATCH_ENSEMBLE=B +
GPUWRF_BATCH_INPUT_DIRS) originally split only on the OS path separator (":"), so a
natural comma-separated list failed with a confusing "requires N input dirs, got 1".
These tests pin the forgiving separator (comma / newline / ":") and the clearer error.
CPU-only, no JAX forecast, no GPU.
"""
from pathlib import Path

import pytest

from gpuwrf.integration.nested_pipeline import (
    NestedPipelineConfig,
    _resolve_batch_input_dirs,
)


def _cfg(input_dir="/case/day1"):
    return NestedPipelineConfig(
        input_dir=Path(input_dir),
        output_dir=Path("/out"),
        proof_dir=Path("/proofs"),
        hours=1,
        max_dom=2,
    )


@pytest.mark.parametrize(
    "raw",
    [
        "/case/day1:/case/day2",          # original OS-pathsep contract (backward-compat)
        "/case/day1,/case/day2",          # comma — the natural operator choice (was broken)
        "/case/day1, /case/day2",         # comma + whitespace
        "/case/day1\n/case/day2",         # newline-separated (e.g. a file list)
        " /case/day1 , /case/day2 ",      # leading/trailing whitespace
    ],
)
def test_batch_input_dirs_accepts_common_separators(monkeypatch, raw):
    monkeypatch.setenv("GPUWRF_BATCH_INPUT_DIRS", raw)
    dirs = _resolve_batch_input_dirs(_cfg(), 2)
    assert dirs == (Path("/case/day1"), Path("/case/day2"))


def test_count_mismatch_error_is_actionable(monkeypatch):
    monkeypatch.setenv("GPUWRF_BATCH_INPUT_DIRS", "/case/day1")  # 1 dir but B=2
    with pytest.raises(ValueError) as exc:
        _resolve_batch_input_dirs(_cfg(), 2)
    msg = str(exc.value)
    assert "requires exactly 2 input dirs" in msg
    assert "','" in msg and "':'" in msg          # names the accepted separators
    assert "Example:" in msg                        # shows a copy-paste example


def test_unset_replicates_input_dir(monkeypatch):
    monkeypatch.delenv("GPUWRF_BATCH_INPUT_DIRS", raising=False)
    assert _resolve_batch_input_dirs(_cfg(), 3) == (
        Path("/case/day1"),
        Path("/case/day1"),
        Path("/case/day1"),
    )


def test_batch_size_one_is_single_dir(monkeypatch):
    monkeypatch.setenv("GPUWRF_BATCH_INPUT_DIRS", "/case/day1,/case/day2")
    assert _resolve_batch_input_dirs(_cfg(), 1) == (Path("/case/day1"),)


def test_config_batch_input_dirs_used_when_env_unset(monkeypatch):
    monkeypatch.delenv("GPUWRF_BATCH_INPUT_DIRS", raising=False)
    cfg = NestedPipelineConfig(
        input_dir=Path("/case/day1"),
        output_dir=Path("/out"),
        proof_dir=Path("/proofs"),
        hours=1,
        max_dom=2,
        batch_input_dirs=(Path("/case/day1"), Path("/case/day2")),
    )
    assert _resolve_batch_input_dirs(cfg, 2) == (Path("/case/day1"), Path("/case/day2"))
