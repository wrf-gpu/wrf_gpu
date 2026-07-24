"""Focused CPU-only guards for the terminal dycore ladder merge."""

from __future__ import annotations

import ast
from pathlib import Path

from scripts import v0234_dycore_suboperator_gpt_gpu_merge as merge


def _metrics(rmse: float) -> dict[str, float]:
    return {"rmse": rmse}


def _envelope(low: float, high: float) -> dict[str, float]:
    return {
        "wrf_envelope_min_rmse": low,
        "wrf_envelope_max_rmse": high,
    }


def test_systematic_requires_literal_envelope_and_two_x_member_max() -> None:
    inside = merge.classification(_metrics(1.5), _envelope(1.0, 2.0))
    outside_but_below_factor = merge.classification(
        _metrics(3.0), _envelope(1.0, 2.0),
    )
    systematic = merge.classification(_metrics(4.01), _envelope(1.0, 2.0))
    assert not inside["outside_literal_wrf_envelope"]
    assert not inside["systematic"]
    assert outside_but_below_factor["outside_literal_wrf_envelope"]
    assert not outside_but_below_factor["systematic"]
    assert systematic["systematic"]


def test_zero_member_envelope_is_explicit_without_nonfinite_json() -> None:
    row = merge.classification(_metrics(0.1), _envelope(0.0, 0.0))
    assert row["systematic"]
    assert row["gpu_to_wrf_member_max_ratio"] is None
    assert row["wrf_member_max_is_exact_zero"]


def test_merge_source_has_no_jax_gpuwrf_or_gpu_query_import() -> None:
    source = Path(merge.__file__).read_text()
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.split(".")[0] in {"jax", "gpuwrf"} for name in imports)
    assert "nvidia-smi" not in source
    assert "rocm-smi" not in source


def test_frozen_authority_constants() -> None:
    assert merge.TERMINAL_CPU_COMMIT == (
        "453c9fa5d9461dece7ba50de722c1f35c7976931"
    )
    assert merge.SYSTEMATIC_FACTOR == 2.0
    assert merge.RUNTIME_SCHEMA == (
        "gpuwrf.v0234.dycore-suboperator-ladder-short-arm.v1"
    )
