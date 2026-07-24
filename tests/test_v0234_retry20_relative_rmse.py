"""Focused Retry20 relative-RMSE and finite-JSON regressions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v0234_corrected_fullbuffer_gate as gate  # noqa: E402


RETRY19_GPU_DIR = (
    gate.CPU_CASE_ROOT / "gpu_validation_retry19_release_6ef5c13e" / "gpu-output"
)
RETRY19_TIMES = gate.EXPECTED_TIMES[:2]
RETRY19_SOURCE_SHA256 = {
    ("cpu", "2025-03-01_00:00:00"): "37db046dad2540340ce8847e76c7f32b1808d6012f1b14c84e24431280677202",
    ("cpu", "2025-03-01_00:20:00"): "0a1157771f8b00f2c2c4fb66ec1cb63e4534cf3c305981ca81e1cfaad0d8d7f1",
    ("gpu", "2025-03-01_00:00:00"): "95fa068b4712090254366b16ca01367c9b4cbab61b09ec621a5d1c431757484d",
    ("gpu", "2025-03-01_00:20:00"): "70e09cf3ca22711c66ef529e716ead53fb998d2f548e9939c35d7767ac3f9e62",
}
RETRY19_PRIOR_NONFINITE_FIELDS = (
    "CROPCAT", "GLW", "HFX", "LH", "LWDNB", "LWUPB", "LWUPT", "OLR",
    "PBLH", "SSTSK", "SST_INPUT", "TGB", "TGV",
)
RETRY19_ADDITIONAL_ZERO_REFERENCE_FIELDS = (
    "COSZEN", "GOT_VAR_SSO", "MAX_MSFTX", "MAX_MSFTY", "Q2B", "Q2V", "QKE",
)


def _row(metric: dict[str, object], *, field: str = "DIAGNOSTIC") -> dict[str, object]:
    return {
        "valid_time": "2025-03-01T00:00:00+00:00",
        "metrics": {field: metric},
        "field_policy": {field: "UNTOLERANCED_REPORT_ONLY"},
        "inventory": {
            "cpu_fields": [field],
            "gpu_fields": [field],
            "common_incompatible_fields": [],
        },
    }


def _raw_frame_metrics(cpu: np.ndarray, gpu: np.ndarray) -> dict[str, object]:
    """The pre-Retry20 arithmetic, excluding informational relative RMSE."""
    delta = gpu.astype(np.float64) - cpu.astype(np.float64)
    flat_cpu = cpu.astype(np.float64).ravel()
    flat_gpu = gpu.astype(np.float64).ravel()
    return {
        "n": int(delta.size),
        "bias": float(delta.mean()),
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.abs(delta).max(initial=0.0)),
        "correlation": (
            float(np.corrcoef(flat_cpu, flat_gpu)[0, 1])
            if flat_cpu.size >= 2 and np.std(flat_cpu) > 0.0 and np.std(flat_gpu) > 0.0
            else None
        ),
        "sufficient": {
            "n": int(delta.size),
            "sum_delta": float(delta.sum()),
            "sum_abs_delta": float(np.abs(delta).sum()),
            "sum_sq_delta": float(np.sum(delta * delta)),
            "sum_sq_cpu": float(np.sum(flat_cpu * flat_cpu)),
            "sum_cpu": float(flat_cpu.sum()),
            "sum_gpu": float(flat_gpu.sum()),
            "sum_sq_gpu": float(np.sum(flat_gpu * flat_gpu)),
            "sum_cpu_gpu": float(np.sum(flat_cpu * flat_gpu)),
        },
    }


def test_zero_reference_zero_rmse_is_exact_finite() -> None:
    result = gate.relative_rmse_fields(0.0, 0.0)
    assert result == {
        "reference_rms": 0.0,
        "relative_rmse": 0.0,
        "relative_rmse_status": "EXACT_ZERO_REFERENCE",
        "relative_rmse_inputs": {
            "rmse": {"finite": True, "value": 0.0},
            "reference_rms": {"finite": True, "value": 0.0},
        },
    }
    assert json.loads(gate.canonical_json(result))["relative_rmse"] == 0.0


def test_zero_reference_nonzero_rmse_is_explicitly_undefined() -> None:
    result = gate.relative_rmse_fields(2.5, 0.0)
    assert result["reference_rms"] == 0.0
    assert result["relative_rmse"] is None
    assert result["relative_rmse_status"] == "UNDEFINED_ZERO_REFERENCE"
    assert result["relative_rmse_inputs"]["rmse"] == {"finite": True, "value": 2.5}
    gate.canonical_json(result)


def test_normal_and_tiny_finite_denominators_are_not_clamped() -> None:
    normal = gate.relative_rmse_fields(3.0, 4.0)
    assert normal["relative_rmse"] == 0.75
    assert normal["relative_rmse_status"] == "FINITE"

    tiny = float(np.nextafter(np.float64(0.0), np.float64(1.0)))
    representable = gate.relative_rmse_fields(tiny, tiny)
    assert representable["reference_rms"] == tiny
    assert representable["relative_rmse"] == 1.0
    assert representable["relative_rmse_status"] == "FINITE"


def test_tiny_denominator_overflow_is_null_with_inputs_retained() -> None:
    tiny = float(np.nextafter(np.float64(0.0), np.float64(1.0)))
    result = gate.relative_rmse_fields(1.0, tiny)
    assert result["reference_rms"] == tiny
    assert result["relative_rmse"] is None
    assert result["relative_rmse_status"] == "OVERFLOW_OR_UNBOUNDED"
    assert result["relative_rmse_inputs"] == {
        "rmse": {"finite": True, "value": 1.0},
        "reference_rms": {"finite": True, "value": tiny},
    }
    gate.canonical_json(result)


@pytest.mark.parametrize(
    ("rmse", "reference", "identity"),
    (
        (math.inf, 1.0, {"finite": False, "value": "POSITIVE_INFINITY"}),
        (-math.inf, 1.0, {"finite": False, "value": "NEGATIVE_INFINITY"}),
        (math.nan, 1.0, {"finite": False, "value": "NAN"}),
        (-1.0, 1.0, {"finite": True, "value": -1.0}),
    ),
)
def test_nonfinite_or_invalid_input_is_json_safe_unbounded(
    rmse: float, reference: float, identity: dict[str, object],
) -> None:
    result = gate.relative_rmse_fields(rmse, reference)
    assert result["relative_rmse"] is None
    assert result["relative_rmse_status"] == "OVERFLOW_OR_UNBOUNDED"
    assert result["relative_rmse_inputs"]["rmse"] == identity
    gate.require_json_finite(result)
    gate.canonical_json(result)


def test_shared_helper_is_used_by_frame_and_pooled_metrics() -> None:
    frame = gate._metric(np.zeros((2,), dtype=np.float64), np.ones((2,), dtype=np.float64))
    assert frame["rmse"] == 1.0
    assert frame["max_abs"] == 1.0
    assert frame["reference_rms"] == 0.0
    assert frame["relative_rmse"] is None
    assert frame["relative_rmse_status"] == "UNDEFINED_ZERO_REFERENCE"

    pooled, _inventory = gate.pooled_metrics([_row(frame)])
    metric = pooled["DIAGNOSTIC"]
    assert metric["pooled_rmse"] == 1.0
    assert metric["max_abs"] == 1.0
    assert metric["reference_rms"] == 0.0
    assert metric["relative_rmse"] is None
    assert metric["relative_rmse_status"] == "UNDEFINED_ZERO_REFERENCE"
    assert metric["pass"] is None
    gate.canonical_json(pooled)


def test_exact_zero_frame_and_pooled_metrics_remain_exact() -> None:
    frame = gate._metric(np.zeros((3,), dtype=np.float64), np.zeros((3,), dtype=np.float64))
    assert frame["rmse"] == 0.0
    assert frame["relative_rmse"] == 0.0
    assert frame["relative_rmse_status"] == "EXACT_ZERO_REFERENCE"
    pooled, _inventory = gate.pooled_metrics([_row(frame)])
    assert pooled["DIAGNOSTIC"]["relative_rmse"] == 0.0
    assert pooled["DIAGNOSTIC"]["relative_rmse_status"] == "EXACT_ZERO_REFERENCE"


@pytest.mark.parametrize("bad", (math.nan, math.inf, -math.inf, np.float64(math.inf)))
def test_recursive_json_finite_validation_rejects_nested_nonfinite(
    bad: float, tmp_path: Path,
) -> None:
    payload = {"outer": ({"metric": [bad]},)}
    with pytest.raises(gate.GateError, match=r"JSON_NONFINITE: \$\.outer\[0\]\.metric\[0\]"):
        gate.require_json_finite(payload)
    with pytest.raises(gate.GateError, match="JSON_NONFINITE"):
        gate.authority_digest(payload)
    target = tmp_path / "uncreated" / "state.json"
    with pytest.raises(gate.GateError, match="JSON_NONFINITE"):
        gate.atomic_write_json(target, payload)
    assert not target.parent.exists()


def test_recursive_validator_rejects_nonfinite_mapping_key() -> None:
    with pytest.raises(gate.GateError, match=r"JSON_NONFINITE: \$\.<key>"):
        gate.canonical_json({math.inf: "invalid"})


def test_recursive_validator_accepts_null_status_payload() -> None:
    payload = {
        "metrics": {
            "CROPCAT": gate.relative_rmse_fields(1.0, 0.0),
            "STATIC": gate.relative_rmse_fields(0.0, 0.0),
        }
    }
    gate.require_json_finite(payload)
    assert gate.authority_digest(payload) == gate.authority_digest(
        json.loads(gate.canonical_json(payload))
    )


def test_absolute_frame_metrics_remain_value_and_canonical_json_exact() -> None:
    cpu = np.asarray([0.0, -2.0, 4.0, 8.0], dtype=np.float64)
    gpu = np.asarray([1.0, -3.0, 7.0, 6.0], dtype=np.float64)
    expected = _raw_frame_metrics(cpu, gpu)
    observed = gate._metric(cpu, gpu)
    raw_observed = {key: observed[key] for key in expected}
    assert raw_observed == expected
    assert gate.canonical_json(raw_observed) == gate.canonical_json(expected)


def test_strict_pooled_raw_metrics_threshold_and_pass_decision_are_unchanged() -> None:
    cpu = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    gpu = np.asarray([1.1, 1.9, 3.2], dtype=np.float64)
    frame = gate._metric(cpu, gpu)
    frame["max_abs"] = float(frame["max_abs"])
    row = _row(frame, field="T")
    row["field_policy"] = {"T": "STRICT_POOLED_RMSE"}
    start = datetime(2025, 3, 1, tzinfo=timezone.utc)
    rows = [
        dict(row, valid_time=(start + timedelta(minutes=20 * index)).isoformat())
        for index in range(55)
    ]

    pooled, _inventory = gate.pooled_metrics(rows)
    observed = pooled["T"]
    sufficient = frame["sufficient"]
    expected_rmse = math.sqrt(
        (55 * sufficient["sum_sq_delta"]) / (55 * sufficient["n"])
    )
    assert observed["pooled_rmse"] == expected_rmse
    assert observed["max_abs"] == frame["max_abs"]
    assert observed["threshold"] == gate.STRICT_RMSE_LIMITS["T"]
    assert observed["pass"] is (expected_rmse <= gate.STRICT_RMSE_LIMITS["T"])


def test_retained_retry19_0000_pair_regression_on_current_direct_gate(tmp_path: Path) -> None:
    roots = {"cpu": gate.CPU_INPUT_DIR, "gpu": RETRY19_GPU_DIR}
    source_paths = {
        (side, valid): root / f"wrfout_d03_{valid.strftime('%Y-%m-%d_%H:%M:%S')}"
        for side, root in roots.items()
        for valid in RETRY19_TIMES
    }
    before = {}
    for (side, valid), path in source_paths.items():
        stamp = valid.strftime("%Y-%m-%d_%H:%M:%S")
        assert gate.sha256_file(path) == RETRY19_SOURCE_SHA256[(side, stamp)]
        before[(side, valid)] = gate.file_authority(path)
    successor = RETRY19_GPU_DIR / "wrfout_d03_2025-03-01_00:40:00"
    assert not successor.exists()

    state_path = tmp_path / "incremental-pairs.json"
    states = [
        gate.incremental_pair(
            gate.CPU_INPUT_DIR, RETRY19_GPU_DIR, state_path, terminal_mode=True,
        )
        for _ in range(3)
    ]
    assert [state["matched_count"] for state in states] == [0, 1, 1]
    final = states[-1]
    assert [row["valid_time"] for row in final["pairs"]] == [
        "2025-03-01T00:00:00+00:00"
    ]
    assert "2025-03-01T00:20:00+00:00" in final["closure_observations"]
    assert all(
        row["valid_time"] != "2025-03-01T00:20:00+00:00" for row in final["pairs"]
    )
    gate.canonical_json(final)
    gate.validate_pair_state_integrity(final, recompute=True)

    row = final["pairs"][0]
    all_zero_reference = tuple(sorted(
        field for field, metric in row["metrics"].items()
        if metric["reference_rms"] == 0.0 and metric["rmse"] > 0.0
    ))
    with np.errstate(over="ignore", invalid="ignore"):
        prior_nonfinite = tuple(sorted(
            field for field in all_zero_reference
            if not np.isfinite(
                np.float64(row["metrics"][field]["rmse"]) / np.finfo(np.float64).tiny
            )
        ))
    additional = tuple(sorted(set(all_zero_reference) - set(prior_nonfinite)))
    assert prior_nonfinite == RETRY19_PRIOR_NONFINITE_FIELDS
    assert additional == RETRY19_ADDITIONAL_ZERO_REFERENCE_FIELDS
    for field in all_zero_reference:
        frame = row["metrics"][field]
        pooled = final["pooled_metrics"][field]
        assert row["field_policy"][field] == "UNTOLERANCED_REPORT_ONLY"
        assert frame["relative_rmse"] is None
        assert frame["relative_rmse_status"] == "UNDEFINED_ZERO_REFERENCE"
        assert pooled["relative_rmse"] is None
        assert pooled["relative_rmse_status"] == "UNDEFINED_ZERO_REFERENCE"
        assert pooled["pooled_rmse"] == frame["rmse"]
        assert pooled["max_abs"] == frame["max_abs"]
        assert pooled["pass"] is None

    cpu_path = source_paths[("cpu", RETRY19_TIMES[0])]
    gpu_path = source_paths[("gpu", RETRY19_TIMES[0])]
    with Dataset(cpu_path) as cpu, Dataset(gpu_path) as gpu:
        for field in gate.STRICT_FIELDS:
            cpu_values = gate._variable_array(cpu, field).astype(np.float64)
            gpu_values = gate._variable_array(gpu, field).astype(np.float64)
            delta = gpu_values - cpu_values
            direct_rmse = float(np.sqrt(np.mean(delta * delta)))
            direct_max_abs = float(np.abs(delta).max(initial=0.0))
            frame = row["metrics"][field]
            pooled = final["pooled_metrics"][field]
            assert frame["rmse"] == direct_rmse
            assert frame["max_abs"] == direct_max_abs
            assert frame["limit"] == gate.STRICT_RMSE_LIMITS[field]
            assert frame["per_frame_pass_diagnostic"] is (
                direct_rmse <= gate.STRICT_RMSE_LIMITS[field]
            )
            assert frame["pass"] is None
            assert pooled["pooled_rmse"] == direct_rmse
            assert pooled["max_abs"] == direct_max_abs
            assert pooled["threshold"] == gate.STRICT_RMSE_LIMITS[field]
            assert pooled["pass"] is False

    for key, path in source_paths.items():
        assert gate.file_authority(path) == before[key]


def test_current_lineage_does_not_revive_retry20_control_plane() -> None:
    source = (ROOT / "scripts" / "v0234_corrected_fullbuffer_gate.py").read_text(encoding="utf-8")
    for obsolete in (
        "v0234_retry20_launch",
        "v0234_retry20_runtime_bootstrap",
        "v0234_retry20_cache_manifest",
        "v0234_retry20_release_proof",
    ):
        assert obsolete not in source
