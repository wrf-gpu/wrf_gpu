"""Gen2 wrfout data-quality checks and M7 RMSE adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from netCDF4 import Dataset
import zarr

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover - non-JAX utility contexts.
    jnp = None

from gpuwrf.io.data_inventory import iter_complete_runs, utc_now_iso
from gpuwrf.io.gen2_wrfout_loader import DEFAULT_SURFACE_FIELDS, Gen2WrfoutLoader, normalize_valid_time, read_wrfout_file


QUALITY_FIELDS = DEFAULT_SURFACE_FIELDS
BOUNDARY_REPLAY_VARIABLES = ("U", "V", "T", "QVAPOR", "PH")
BOUNDARY_SIDES = ("W", "E", "S", "N")


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        batch_count = int(finite.size)
        batch_mean = float(np.mean(finite))
        batch_m2 = float(np.sum((finite - batch_mean) ** 2))
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        self.m2 = self.m2 + batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    @property
    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        return float(np.sqrt(self.m2 / (self.count - 1)))


def _sample_values(values: np.ndarray, *, max_samples: int) -> np.ndarray:
    flat = np.asarray(values).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size <= max_samples:
        return finite.astype(np.float64, copy=False)
    stride = int(np.ceil(finite.size / max_samples))
    return finite[::stride].astype(np.float64, copy=False)


def _field_template() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "nan_count": 0,
        "inf_count": 0,
        "finite_count": 0,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "p01": None,
        "p99": None,
        "histogram": {"bin_edges": [], "counts": []},
        "spike_zscore_threshold": 5.0,
        "spike_count": 0,
        "spike_fraction": 0.0,
        "spike_flag": False,
    }


def _update_minmax(record: dict[str, Any], finite: np.ndarray) -> None:
    if finite.size == 0:
        return
    current_min = float(np.min(finite))
    current_max = float(np.max(finite))
    record["min"] = current_min if record["min"] is None else min(float(record["min"]), current_min)
    record["max"] = current_max if record["max"] is None else max(float(record["max"]), current_max)


def audit_run_quality(
    run_record: dict[str, Any],
    *,
    fields: Iterable[str] = QUALITY_FIELDS,
    max_histogram_samples_per_field: int = 100_000,
    histogram_bins: int = 20,
) -> dict[str, Any]:
    """Audit one run with chunked per-file reads."""

    field_names = tuple(fields)
    if not run_record.get("complete"):
        return {
            "run_id": run_record["run_id"],
            "run_path": run_record["run_path"],
            "status": "PARTIAL",
            "sampled": False,
            "reason": "run is partial; quality sampling is defined for complete runs",
            "missing_time_step_count": int(run_record.get("missing_time_step_count", 0)),
            "fields": {},
        }

    loader = Gen2WrfoutLoader(run_record["run_path"])
    records = {field: _field_template() for field in field_names}
    stats = {field: RunningStats() for field in field_names}
    samples: dict[str, list[np.ndarray]] = {field: [] for field in field_names}
    missing_fields: list[str] = []

    for file_path in loader.files:
        try:
            chunk = read_wrfout_file(file_path, fields=field_names)["fields"]
        except KeyError as exc:
            missing_fields.append(str(exc))
            continue
        for field, array in chunk.items():
            data = np.asarray(array)
            record = records[field]
            record["sample_count"] += int(data.size)
            record["nan_count"] += int(np.isnan(data).sum())
            record["inf_count"] += int(np.isinf(data).sum())
            finite = data[np.isfinite(data)]
            record["finite_count"] += int(finite.size)
            _update_minmax(record, finite)
            stats[field].update(finite)
            if finite.size:
                samples[field].append(_sample_values(finite, max_samples=max(1, max_histogram_samples_per_field // max(1, len(loader.files)))))

    for field in field_names:
        record = records[field]
        record["mean"] = stats[field].mean if stats[field].count else None
        record["std"] = stats[field].std if stats[field].count else None
        merged_sample = np.concatenate(samples[field]) if samples[field] else np.asarray([], dtype=np.float64)
        if merged_sample.size:
            p01 = float(np.percentile(merged_sample, 1.0))
            p99 = float(np.percentile(merged_sample, 99.0))
            record["p01"] = p01
            record["p99"] = p99
            if np.isclose(p01, p99):
                edges = np.linspace(float(np.min(merged_sample)), float(np.max(merged_sample)) + 1.0e-12, histogram_bins + 1)
            else:
                edges = np.linspace(p01, p99, histogram_bins + 1)
            counts, edges = np.histogram(merged_sample, bins=edges)
            record["histogram"] = {"bin_edges": [float(value) for value in edges], "counts": [int(value) for value in counts]}

    # Second pass: flag z-score spikes after mean/std are known.
    for file_path in loader.files:
        try:
            chunk = read_wrfout_file(file_path, fields=field_names)["fields"]
        except KeyError:
            continue
        for field, array in chunk.items():
            record = records[field]
            std = record["std"] or 0.0
            if std <= 0.0 or record["mean"] is None:
                continue
            data = np.asarray(array, dtype=np.float64)
            finite = data[np.isfinite(data)]
            spike_count = int((np.abs(finite - float(record["mean"])) > 5.0 * std).sum())
            record["spike_count"] += spike_count

    status = "GREEN"
    reasons: list[str] = []
    if missing_fields:
        status = "FAIL"
        reasons.append("missing required fields")
    for field, record in records.items():
        if record["nan_count"] > 0 or record["inf_count"] > 0:
            status = "FAIL"
            reasons.append(f"{field} contains NaN or Inf values")
        if record["finite_count"]:
            record["spike_fraction"] = float(record["spike_count"] / record["finite_count"])
        record["spike_flag"] = bool(record["spike_count"] > 0)
        if status == "GREEN" and record["spike_flag"]:
            status = "PARTIAL"
            reasons.append(f"{field} has z-score spikes above threshold")
    if status == "GREEN" and int(run_record.get("missing_time_step_count", 0)) > 0:
        status = "PARTIAL"
        reasons.append("missing valid times")

    return {
        "run_id": run_record["run_id"],
        "run_path": run_record["run_path"],
        "status": status,
        "sampled": True,
        "reasons": sorted(set(reasons)),
        "missing_fields": missing_fields,
        "missing_time_step_count": int(run_record.get("missing_time_step_count", 0)),
        "field_order": list(field_names),
        "fields": records,
    }


def build_quality_audit(
    inventory: dict[str, Any],
    *,
    fields: Iterable[str] = QUALITY_FIELDS,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    runs = [audit_run_quality(run, fields=fields) for run in inventory["runs"]]
    complete_sampled = [run for run in runs if run.get("sampled")]
    audit = {
        "schema": "Gen2D02QualityAudit",
        "schema_version": 1,
        "generated_utc": generated_utc or utc_now_iso(),
        "source_inventory_schema": inventory.get("schema"),
        "source_inventory_root": inventory.get("root"),
        "fields": list(fields),
        "run_count": int(len(runs)),
        "complete_run_count": int(sum(1 for _ in iter_complete_runs(inventory))),
        "sampled_run_count": int(len(complete_sampled)),
        "status_counts": {
            "GREEN": int(sum(1 for run in runs if run["status"] == "GREEN")),
            "PARTIAL": int(sum(1 for run in runs if run["status"] == "PARTIAL")),
            "FAIL": int(sum(1 for run in runs if run["status"] == "FAIL")),
        },
        "runs": runs,
    }
    validate_quality_audit(audit)
    return audit


def validate_quality_audit(audit: dict[str, Any]) -> dict[str, Any]:
    for field in ("schema", "schema_version", "generated_utc", "fields", "run_count", "sampled_run_count", "status_counts", "runs"):
        if field not in audit:
            raise ValueError(f"Gen2D02QualityAudit missing required field {field!r}")
    if audit["schema"] != "Gen2D02QualityAudit":
        raise ValueError("audit schema must be 'Gen2D02QualityAudit'")
    if audit["run_count"] != len(audit["runs"]):
        raise ValueError("audit run_count does not match runs length")
    return audit


def _boundary_strip(field: np.ndarray, side: str) -> np.ndarray:
    data = np.asarray(field, dtype=np.float64)
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    if data.ndim == 3:
        if side == "W":
            return data[:, :, 0]
        if side == "E":
            return data[:, :, -1]
        if side == "S":
            return data[:, 0, :]
        if side == "N":
            return data[:, -1, :]
    if data.ndim == 2:
        if side == "W":
            return data[:, 0]
        if side == "E":
            return data[:, -1]
        if side == "S":
            return data[0, :]
        if side == "N":
            return data[-1, :]
    raise ValueError(f"cannot extract boundary side {side!r} from shape {data.shape}")


def _common_arrays(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple(min(int(a), int(b)) for a, b in zip(left.shape, right.shape, strict=False))
    index = tuple(slice(0, size) for size in shape)
    return left[index], right[index]


def compare_boundary_replay_to_wrfout(
    replay_zarr_path: str | Path,
    gen2_run_path: str | Path,
    *,
    valid_time: str | None = None,
    variables: Iterable[str] = BOUNDARY_REPLAY_VARIABLES,
    rel_mae_threshold: float = 0.01,
) -> dict[str, Any]:
    """Compare d02 replay-zarr boundary strips to d02 wrfout at lead 0."""

    root = zarr.open_group(str(replay_zarr_path), mode="r")
    times = list(root.attrs.get("times_utc", []))
    if not times:
        raise ValueError(f"boundary replay zarr has no times_utc attr: {replay_zarr_path}")
    target = normalize_valid_time(valid_time or times[0])
    lead_index = None
    for index, item in enumerate(times):
        if normalize_valid_time(item) == target:
            lead_index = index
            break
    if lead_index is None:
        raise FileNotFoundError(f"valid time {target.isoformat()} not present in boundary replay {replay_zarr_path}")
    loader = Gen2WrfoutLoader(gen2_run_path, target)
    payload = loader.load(fields=tuple(variables), squeeze_time=True)
    per_variable: dict[str, Any] = {}
    failures: list[str] = []
    for var in variables:
        side_records: dict[str, Any] = {}
        truth_field = payload["fields"][var]
        for side in BOUNDARY_SIDES:
            replay = np.asarray(root[var][side][lead_index], dtype=np.float64)
            truth = _boundary_strip(truth_field, side)
            replay_common, truth_common = _common_arrays(replay, truth)
            diff = replay_common - truth_common
            mae = float(np.mean(np.abs(diff)))
            denom = float(np.mean(np.abs(truth_common))) + 1.0e-12
            rel_mae = float(mae / denom)
            passed = bool(rel_mae <= rel_mae_threshold)
            if not passed:
                failures.append(f"{var}/{side} rel_mae={rel_mae:.6g} > {rel_mae_threshold:.6g}")
            side_records[side] = {
                "mae": mae,
                "rel_mae": rel_mae,
                "max_abs": float(np.max(np.abs(diff))),
                "compared_shape": list(replay_common.shape),
                "passed": passed,
            }
        per_variable[var] = {
            "aggregate_rel_mae_max": max(side_records[side]["rel_mae"] for side in BOUNDARY_SIDES),
            "sides": side_records,
        }
    return {
        "schema": "Gen2BoundaryReplayWrfoutCrossCheck",
        "schema_version": 1,
        "status": "GREEN" if not failures else "FAIL",
        "replay_zarr_path": str(replay_zarr_path),
        "gen2_run_path": str(gen2_run_path),
        "valid_time_utc": target.isoformat(),
        "lead_index": int(lead_index),
        "rel_mae_threshold": float(rel_mae_threshold),
        "failures": failures,
        "variables": per_variable,
    }


def _state_field(state: Any, field: str) -> Any:
    if isinstance(state, dict):
        return state[field]
    return getattr(state, field)


def compute_rmse_against_gen2(
    gpu_forecast_state: Any,
    gen2_wrfout_path: str | Path,
    valid_time: str,
    fields: Iterable[str] = ("U10", "V10", "T2"),
) -> dict[str, dict[str, Any]]:
    """Return per-field RMSE and error maps against a Gen2 d02 wrfout file."""

    if jnp is None:  # pragma: no cover - only when JAX is unavailable.
        raise RuntimeError("JAX is required for compute_rmse_against_gen2")
    source = Path(gen2_wrfout_path)
    field_names = tuple(fields)
    if source.is_dir():
        truth_payload = Gen2WrfoutLoader(source, valid_time).load(field_names, as_jax=True)
    else:
        truth_payload = read_wrfout_file(source, fields=field_names, as_jax=True)
        target = normalize_valid_time(valid_time)
        actual = normalize_valid_time(truth_payload["valid_time_utc"])
        if actual != target:
            raise ValueError(f"truth file valid time {actual.isoformat()} does not match requested {target.isoformat()}")
    results: dict[str, dict[str, Any]] = {}
    for field in field_names:
        predicted = jnp.asarray(_state_field(gpu_forecast_state, field))
        truth = jnp.asarray(truth_payload["fields"][field])
        if predicted.shape != truth.shape:
            raise ValueError(f"{field} shape mismatch: forecast {predicted.shape} vs Gen2 {truth.shape}")
        error_map = predicted - truth
        rmse = jnp.sqrt(jnp.mean(error_map * error_map))
        results[field] = {
            "rmse": float(rmse),
            "error_map": error_map,
            "valid_time_utc": truth_payload["valid_time_utc"],
            "gen2_source_file": truth_payload["source_file"],
        }
    return results


__all__ = [
    "BOUNDARY_REPLAY_VARIABLES",
    "BOUNDARY_SIDES",
    "QUALITY_FIELDS",
    "audit_run_quality",
    "build_quality_audit",
    "compare_boundary_replay_to_wrfout",
    "compute_rmse_against_gen2",
    "validate_quality_audit",
]
