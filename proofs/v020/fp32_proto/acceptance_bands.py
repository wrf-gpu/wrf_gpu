"""Importable v0.20 fp32 acceptance-band specification and checker.

Sources:
- proofs/v020/fp32_analysis/FINAL_FP32_SPRINT_PLAN.md section 1
- proofs/v020/fp32_analysis/GPT_FP32_PLAN.md section 7

The bands are deliberately forecast-skill style rather than bit-tight. This
module provides a small checker for synthetic/unit proof objects and later
forecast metrics; callers may provide the subset of metrics available for a
given field and lead.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


LEAD_TIERS: tuple[tuple[float, str], ...] = (
    (24.0, "24h"),
    (72.0, "48_72h"),
    (120.0, "120h"),
)


ACCEPTANCE_BANDS: dict[str, dict[str, dict[str, float]]] = {
    "wind": {
        "24h": {
            "rmse_increase": 0.25,
            "bias_abs": 0.20,
            "p95_abs_diff": 2.0,
            "domain_mean_speed_drift": 0.20,
        },
        "48_72h": {
            "rmse_increase": 0.50,
            "bias_abs": 0.35,
            "p95_abs_diff": 3.0,
            "domain_mean_speed_drift": 0.35,
        },
        "120h": {
            "rmse_increase": 0.80,
            "bias_abs": 0.60,
            "p95_abs_diff": 5.0,
            "domain_mean_speed_drift": 0.60,
        },
    },
    "temperature": {
        "24h": {
            "rmse_increase": 0.6,
            "bias_abs": 0.3,
            "p95_abs_diff": 1.5,
            "domain_mean_theta_drift": 0.3,
        },
        "48_72h": {
            "rmse_increase": 1.0,
            "bias_abs": 0.6,
            "p95_abs_diff": 2.5,
            "domain_mean_theta_drift": 0.6,
        },
        "120h": {
            "rmse_increase": 1.5,
            "bias_abs": 1.0,
            "p95_abs_diff": 4.0,
            "domain_mean_theta_drift": 1.0,
        },
    },
    "cloud_radiation": {
        "24h": {
            "cloud_fraction_abs_diff": 0.10,
            "column_condensate_nrmse": 0.25,
            "swdown_lwdown_bias_abs": 30.0,
        },
        "48_72h": {
            "cloud_fraction_abs_diff": 0.15,
            "column_condensate_nrmse": 0.40,
            "swdown_lwdown_bias_abs": 50.0,
        },
        "120h": {
            "cloud_fraction_abs_diff": 0.20,
            "column_condensate_nrmse": 0.60,
            "swdown_lwdown_bias_abs": 70.0,
        },
    },
    "pressure_mass_geopotential": {
        "24h": {
            "psfc_rmse_increase": 150.0,
            "psfc_bias_abs": 75.0,
            "dry_mass_relative_drift": 1.0e-4,
            "geopotential_boundary_ring_growth": 0.0,
        },
        "48_72h": {
            "psfc_rmse_increase": 300.0,
            "psfc_bias_abs": 150.0,
            "dry_mass_relative_drift": 3.0e-4,
            "geopotential_boundary_ring_growth": 0.0,
        },
        "120h": {
            "psfc_rmse_increase": 500.0,
            "psfc_bias_abs": 250.0,
            "dry_mass_relative_drift": 7.0e-4,
            "geopotential_boundary_ring_growth": 0.0,
        },
    },
    "qvapor": {
        "24h": {
            "domain_mean_water_vapor_rel_diff": 0.20,
            "qvapor_p999": 0.04,
        },
        "48_72h": {
            "domain_mean_water_vapor_rel_diff": 0.35,
            "qvapor_p999": 0.04,
        },
        "120h": {
            "domain_mean_water_vapor_rel_diff": 0.50,
            "qvapor_p999": 0.04,
        },
    },
    "cumulative": {
        "24h": {"domain_total_ratio": 3.0},
        "48_72h": {"domain_total_ratio": 4.0},
        "120h": {"domain_total_ratio": 5.0},
    },
}


FIELD_GROUPS: dict[str, str] = {
    "U": "wind",
    "V": "wind",
    "W": "wind",
    "U10": "wind",
    "V10": "wind",
    "T": "temperature",
    "THETA": "temperature",
    "T2": "temperature",
    "QCLOUD": "cloud_radiation",
    "QICE": "cloud_radiation",
    "QSNOW": "cloud_radiation",
    "QGRAUP": "cloud_radiation",
    "CLOUD_FRACTION": "cloud_radiation",
    "SWDOWN": "cloud_radiation",
    "LWDOWN": "cloud_radiation",
    "RTHRATEN": "cloud_radiation",
    "P": "pressure_mass_geopotential",
    "PB": "pressure_mass_geopotential",
    "PH": "pressure_mass_geopotential",
    "PHB": "pressure_mass_geopotential",
    "MU": "pressure_mass_geopotential",
    "MUB": "pressure_mass_geopotential",
    "PSFC": "pressure_mass_geopotential",
    "QVAPOR": "qvapor",
    "RAINC": "cumulative",
    "RAINNC": "cumulative",
    "SNOWNC": "cumulative",
    "GRAUPELNC": "cumulative",
    "ICENC": "cumulative",
}


PROTECTED_FINITE_FIELDS: frozenset[str] = frozenset(
    {
        "U",
        "V",
        "W",
        "T",
        "THETA",
        "T2",
        "P",
        "PB",
        "PH",
        "PHB",
        "MU",
        "MUB",
        "PSFC",
        "QKE",
        "QCLOUD",
        "QICE",
        "QSNOW",
        "QGRAUP",
    }
)


UNIVERSAL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "PSFC": (50_000.0, 110_000.0),
    "T2": (180.0, 330.0),
    "THETA": (150.0, 1000.0),
    "QKE": (0.0, 150.0),
}


@dataclass(frozen=True)
class BandCheck:
    """Result of checking a field against a lead-tier band."""

    field: str
    group: str
    tier: str
    passes: bool
    checked: dict[str, float]
    failures: tuple[str, ...]


def lead_tier(lead_hours: float) -> str:
    """Return the canonical lead tier for a forecast lead in hours."""

    for upper, tier in LEAD_TIERS:
        if lead_hours <= upper:
            return tier
    return "120h"


def field_group(field_name: str) -> str:
    """Return the configured acceptance group for a WRF-style field name."""

    key = field_name.upper()
    if key not in FIELD_GROUPS:
        raise KeyError(f"No fp32 acceptance group configured for field {field_name!r}")
    return FIELD_GROUPS[key]


def check_metric_band(
    field_name: str,
    lead_hours: float,
    metrics: Mapping[str, float],
) -> BandCheck:
    """Check provided metrics against the configured field/tier thresholds.

    Unknown metric names are ignored. Missing metric names are not failures; the
    caller is responsible for requiring the metrics available in that validation
    rung. This keeps the checker usable for synthetic oracles, savepoints, and
    full forecast reports.
    """

    group = field_group(field_name)
    tier = lead_tier(lead_hours)
    limits = ACCEPTANCE_BANDS[group][tier]
    checked: dict[str, float] = {}
    failures: list[str] = []

    for metric_name, value in metrics.items():
        if metric_name not in limits:
            continue
        value_f = float(value)
        checked[metric_name] = value_f
        limit = limits[metric_name]
        if not isfinite(value_f):
            failures.append(f"{metric_name}=nonfinite (limit {limit:g})")
        elif value_f > limit:
            failures.append(f"{metric_name}={value_f:g} > {limit:g}")

    return BandCheck(
        field=field_name.upper(),
        group=group,
        tier=tier,
        passes=not failures,
        checked=checked,
        failures=tuple(failures),
    )


def check_universal_hard_gates(
    samples: Mapping[str, Sequence[float] | np.ndarray],
    baseline: Mapping[str, Sequence[float] | np.ndarray] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Check universal hard stability gates for finite values and bounds."""

    failures: list[str] = []
    baseline = baseline or {}

    for field_name, values in samples.items():
        field = field_name.upper()
        arr = np.asarray(values, dtype=np.float64)
        if field in PROTECTED_FINITE_FIELDS and not np.all(np.isfinite(arr)):
            failures.append(f"{field}: nonfinite protected value")
            continue

        if field in UNIVERSAL_BOUNDS and arr.size:
            lower, upper = UNIVERSAL_BOUNDS[field]
            base_arr = np.asarray(baseline.get(field, []), dtype=np.float64)
            baseline_outside = False
            if base_arr.size:
                if lower is not None and np.any(base_arr < lower):
                    baseline_outside = True
                if upper is not None and np.any(base_arr > upper):
                    baseline_outside = True
            if not baseline_outside:
                if lower is not None and np.any(arr < lower):
                    failures.append(f"{field}: below hard lower bound {lower:g}")
                if upper is not None and np.any(arr > upper):
                    failures.append(f"{field}: above hard upper bound {upper:g}")

    return not failures, tuple(failures)


def export_spec() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the frozen band spec."""

    return {
        "sources": [
            "proofs/v020/fp32_analysis/FINAL_FP32_SPRINT_PLAN.md#section-1",
            "proofs/v020/fp32_analysis/GPT_FP32_PLAN.md#section-7",
        ],
        "lead_tiers": list(LEAD_TIERS),
        "bands": ACCEPTANCE_BANDS,
        "field_groups": FIELD_GROUPS,
        "universal_bounds": UNIVERSAL_BOUNDS,
        "protected_finite_fields": sorted(PROTECTED_FINITE_FIELDS),
    }

