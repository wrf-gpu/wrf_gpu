"""Helpers for M6 Tier-3 timestep-convergence envelopes.

This module is intentionally independent of the dycore operator.  Runners pass
checkpoint arrays in, and these helpers compute norms and verdicts only.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


ALLOWED_VERDICTS = (
    "PASS_TIER3",
    "FAIL_DRIFT",
    "FAIL_NONFINITE",
    "FAIL_INSUFFICIENT_DT_PAIRS",
)


def checkpoint_key(seconds: float) -> str:
    """Returns the stable JSON key used for checkpoint-indexed tables."""

    return f"{float(seconds):g}s"


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _finite_float(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def norm_triplet(candidate: Any, reference: Any) -> dict[str, float | None]:
    """Compute L2, Linf, and RMSE for two same-shaped arrays.

    The returned values are JSON-safe: non-finite results become ``None`` so a
    failed run can still emit a parseable proof object.
    """

    left = _as_array(candidate)
    right = _as_array(reference)
    if left.shape != right.shape:
        raise ValueError(f"norm shape mismatch: {left.shape} vs {right.shape}")
    diff = left - right
    return {
        "l2": _finite_float(np.linalg.norm(diff.ravel(), ord=2)),
        "linf": _finite_float(np.max(np.abs(diff)) if diff.size else 0.0),
        "rmse": _finite_float(np.sqrt(np.mean(diff * diff)) if diff.size else 0.0),
    }


def build_norm_table(
    snapshots_by_dt: Mapping[float, Mapping[float, Mapping[str, Any]]],
    *,
    dt_pairs: Sequence[Mapping[str, float]],
    checkpoints_s: Sequence[float],
    variables: Sequence[str],
) -> dict[str, Any]:
    """Build the contract's per-variable/per-checkpoint dt-pair norm table."""

    norms: dict[str, Any] = {}
    for variable in variables:
        variable_table: dict[str, Any] = {}
        for pair in dt_pairs:
            pair_key = f"pair_{int(pair['pair_index'])}"
            coarse_dt = float(pair["dt_coarse"])
            fine_dt = float(pair["dt_fine"])
            checkpoint_table: dict[str, Any] = {}
            for checkpoint_s in checkpoints_s:
                key = checkpoint_key(checkpoint_s)
                checkpoint_table[key] = {
                    **norm_triplet(
                        snapshots_by_dt[coarse_dt][float(checkpoint_s)][variable],
                        snapshots_by_dt[fine_dt][float(checkpoint_s)][variable],
                    ),
                    "dt_coarse": coarse_dt,
                    "dt_fine": fine_dt,
                }
            variable_table[pair_key] = checkpoint_table
        norms[variable] = variable_table
    return norms


def _iter_norm_records(norms: Mapping[str, Any]):
    for variable, variable_table in norms.items():
        for pair_key, pair_table in variable_table.items():
            for checkpoint, record in pair_table.items():
                yield variable, pair_key, checkpoint, record


def _record_is_finite(record: Mapping[str, Any]) -> bool:
    return all(isinstance(record.get(name), (int, float)) and math.isfinite(float(record[name])) for name in ("l2", "linf", "rmse"))


def first_nonfinite_metadata(per_dt_run_metadata: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the first run metadata record that observed a non-finite state."""

    for metadata in per_dt_run_metadata:
        if metadata.get("first_nonfinite_step") is not None:
            return metadata
    return None


def classify_convergence(
    norms: Mapping[str, Any],
    *,
    dt_pairs: Sequence[Mapping[str, float]],
    per_dt_run_metadata: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, float] | None = None,
) -> tuple[str, str]:
    """Classify a dt-refinement envelope into one of the contract verdicts."""

    if len(dt_pairs) < 1:
        return "FAIL_INSUFFICIENT_DT_PAIRS", "No dt/refined-dt pair was available."

    bad_run = first_nonfinite_metadata(per_dt_run_metadata)
    if bad_run is not None:
        return (
            "FAIL_NONFINITE",
            f"dt={bad_run.get('dt_s')} first became non-finite at step {bad_run.get('first_nonfinite_step')}.",
        )

    nonfinite_norms = [
        (variable, pair_key, checkpoint)
        for variable, pair_key, checkpoint, record in _iter_norm_records(norms)
        if not _record_is_finite(record)
    ]
    if nonfinite_norms:
        variable, pair_key, checkpoint = nonfinite_norms[0]
        return "FAIL_NONFINITE", f"Non-finite norm for {variable} {pair_key} checkpoint {checkpoint}."

    if len(dt_pairs) < 2:
        return (
            "FAIL_INSUFFICIENT_DT_PAIRS",
            "Only one dt pair was available; Tier-3 needs dt, dt/2, and dt/4 to check refinement trend.",
        )

    thresholds = dict(criteria or {})
    rmse_growth = float(thresholds.get("rmse_refined_pair_max_growth_factor", 1.25))
    linf_growth = float(thresholds.get("linf_refined_pair_max_growth_factor", 1.50))
    absolute_floor = float(thresholds.get("absolute_floor", 1.0e-12))

    failures: list[str] = []
    pair0 = "pair_0"
    pair1 = "pair_1"
    for variable, variable_table in norms.items():
        if pair0 not in variable_table or pair1 not in variable_table:
            failures.append(f"{variable} missing pair_0 or pair_1")
            continue
        for checkpoint, coarse_record in variable_table[pair0].items():
            refined_record = variable_table[pair1].get(checkpoint)
            if refined_record is None:
                failures.append(f"{variable} missing pair_1 checkpoint {checkpoint}")
                continue
            coarse_rmse = float(coarse_record["rmse"])
            refined_rmse = float(refined_record["rmse"])
            coarse_linf = float(coarse_record["linf"])
            refined_linf = float(refined_record["linf"])
            rmse_limit = max(absolute_floor, rmse_growth * coarse_rmse)
            linf_limit = max(absolute_floor, linf_growth * coarse_linf)
            if refined_rmse > rmse_limit or refined_linf > linf_limit:
                failures.append(
                    f"{variable}@{checkpoint}: pair_1 rmse={refined_rmse:.6e} limit={rmse_limit:.6e}, "
                    f"linf={refined_linf:.6e} limit={linf_limit:.6e}"
                )

    if failures:
        head = "; ".join(failures[:3])
        suffix = f"; +{len(failures) - 3} more" if len(failures) > 3 else ""
        return "FAIL_DRIFT", head + suffix
    return "PASS_TIER3", "All refined dt-pair RMSE/Linf norms stayed within the configured growth bounds."


def validate_tsc_payload(payload: Mapping[str, Any]) -> None:
    """Validate the stable top-level schema required by the sprint contract."""

    required = {
        "artifact_type",
        "case",
        "config",
        "dt_pairs",
        "checkpoints_s",
        "per_dt_run_metadata",
        "norms",
        "convergence_verdict",
        "rationale",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"missing tsc payload keys: {sorted(missing)}")
    if payload["artifact_type"] != "m6_tier3_tsc_envelope":
        raise ValueError("artifact_type must be m6_tier3_tsc_envelope")
    if payload["convergence_verdict"] not in ALLOWED_VERDICTS:
        raise ValueError(f"invalid convergence_verdict: {payload['convergence_verdict']}")
