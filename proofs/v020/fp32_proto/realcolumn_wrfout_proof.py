"""CPU-only real-WRF-column proof for the fp32 perturbation prototype.

The source is a real CPU-reference wrfout file committed under .agent/sprints.
This script extracts actual WRF P/PB/PH/PHB/MU/MUB/W columns and runs the same
NumPy acoustic+vertical prototype used by proof_results.json:

- fp64 reference mode;
- naive fp32-total mode;
- perturbation-authoritative fp32 mode.

It writes proof_results_realcolumn.json in this directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from fp32_column_proto import (
    IdealizedColumnCase,
    cancellation_metrics,
    run_sequence,
)


OUT_DIR = Path(__file__).resolve().parent
DEFAULT_WRFOUT = Path(
    ".agent/sprints/2026-05-25-m6-perf-design-acceptance/"
    "artifacts/wrfout_d02_1h_cpu_reference.nc"
)
FALLBACK_NPZ = Path("proofs/v013/_twoway_vram_bitident_ref.npz")

N_STEPS = 1_000
DT = 0.05
CANCELLATION_GATE_RATIO = 27.0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _mass_geopotential(ph_stag: np.ndarray) -> np.ndarray:
    return 0.5 * (ph_stag[:-1] + ph_stag[1:])


def _case_from_wrfout_column(
    name: str,
    arrays: dict[str, np.ndarray],
    y: int,
    x: int,
) -> IdealizedColumnCase:
    p = arrays["P"]
    pb = arrays["PB"]
    ph = _mass_geopotential(arrays["PH"])
    phb = _mass_geopotential(arrays["PHB"])
    w = _mass_geopotential(arrays["W"])
    mu = arrays["MU"]
    mub = arrays["MUB"]
    hgt = arrays["HGT"]

    p0 = p[:, y, x]
    mu0 = np.full_like(p0, mu[y, x])
    mu_delta = np.full_like(p0, mu[y, x + 1] - mu[y, x - 1])

    return IdealizedColumnCase(
        name=name,
        terrain_m=float(hgt[y, x]),
        dt=DT,
        diffusion=0.025,
        damping=0.042,
        p_base_left=pb[:, y, x - 1],
        p_base_right=pb[:, y, x + 1],
        p_base_center=pb[:, y, x],
        p_delta_pert=p[:, y, x + 1] - p[:, y, x - 1],
        ph_base_left=phb[:, y, x - 1],
        ph_base_right=phb[:, y, x + 1],
        ph_base_center=phb[:, y, x],
        ph_delta_pert=ph[:, y, x + 1] - ph[:, y, x - 1],
        mu_base_left=np.full_like(p0, mub[y, x - 1]),
        mu_base_right=np.full_like(p0, mub[y, x + 1]),
        mu_base_center=np.full_like(p0, mub[y, x]),
        mu_delta_pert=mu_delta,
        p0=p0,
        ph0=ph[:, y, x],
        mu0=mu0,
        w0=w[:, y, x],
        accum_increment=0.004,
    )


def _load_wrfout(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = ("P", "PB", "PH", "PHB", "MU", "MUB", "W", "HGT", "XLAT", "XLONG")
    with Dataset(path) as ds:
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(f"{path} missing required variables: {missing}")
        arrays = {name: ds.variables[name][0].astype(np.float64) for name in required}
        variable_dtypes = {name: str(ds.variables[name].dtype) for name in required}
        meta = {
            "source_kind": "wrfout_cpu_reference",
            "source_path": str(path),
            "file_format": ds.file_format,
            "dimensions": {name: len(dim) for name, dim in ds.dimensions.items()},
            "variable_dtypes": variable_dtypes,
        }
    return arrays, meta


def _load_fallback_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    z = np.load(path)
    required = ("p_perturbation", "p_total", "ph_perturbation", "ph_total", "mu_perturbation", "mu_total", "w")
    missing = [name for name in required if name not in z.files]
    if missing:
        raise ValueError(f"{path} missing required arrays: {missing}")
    p = z["p_perturbation"].astype(np.float64)
    ph = z["ph_perturbation"].astype(np.float64)
    mu = z["mu_perturbation"].astype(np.float64)
    arrays = {
        "P": p,
        "PB": z["p_total"].astype(np.float64) - p,
        "PH": ph,
        "PHB": z["ph_total"].astype(np.float64) - ph,
        "MU": mu,
        "MUB": z["mu_total"].astype(np.float64) - mu,
        "W": z["w"].astype(np.float64),
        "HGT": np.zeros_like(mu, dtype=np.float64),
        "XLAT": np.zeros_like(mu, dtype=np.float64),
        "XLONG": np.zeros_like(mu, dtype=np.float64),
    }
    meta = {
        "source_kind": "fallback_npz_perturbation_state",
        "source_path": str(path),
        "note": "Used only if no full wrfout with P/PB/PH/PHB/MU/MUB/W exists.",
        "arrays": list(z.files),
    }
    return arrays, meta


def _select_columns(arrays: dict[str, np.ndarray]) -> list[tuple[str, int, int]]:
    hgt = arrays["HGT"]
    ny, nx = hgt.shape
    interior = (slice(1, ny - 1), slice(1, nx - 1))
    hgt_i = hgt[interior]
    iy, ix = np.unravel_index(np.argmax(hgt_i), hgt_i.shape)
    high = (int(iy + 1), int(ix + 1))

    east_west_slope = np.abs(hgt[:, 2:] - hgt[:, :-2])[:, 1:-1]
    iy, ix = np.unravel_index(np.argmax(east_west_slope), east_west_slope.shape)
    steep = (int(iy), int(ix + 1))

    iy, ix = np.unravel_index(np.argmin(hgt_i), hgt_i.shape)
    low = (int(iy + 1), int(ix + 1))
    return [("real_high_terrain", *high), ("real_steep_gradient", *steep), ("real_low_terrain", *low)]


def _summarize_case(result: dict[str, Any]) -> dict[str, Any]:
    ratios = {
        field: metrics["naive_over_perturbation"]
        for field, metrics in result["initial_cancellation"].items()
    }
    field_gate_failures = {
        field: ratio > CANCELLATION_GATE_RATIO for field, ratio in ratios.items()
    }
    return {
        "max_naive_over_perturbation": float(max(ratios.values())),
        "min_naive_over_perturbation": float(min(ratios.values())),
        "naive_fields_failing_local_gate": [
            field for field, failed in field_gate_failures.items() if failed
        ],
        "naive_case_fails_local_gate": any(field_gate_failures.values()),
        "perturbation_passes_bands": result["modes"]["perturbation"]["all_band_checks_pass"],
        "perturbation_all_finite": result["stability"]["all_finite"]["perturbation"],
        "naive_passes_bands": result["modes"]["naive_totals"]["all_band_checks_pass"],
    }


def run_realcolumn_proof() -> dict[str, Any]:
    if DEFAULT_WRFOUT.exists():
        arrays, source_meta = _load_wrfout(DEFAULT_WRFOUT)
    elif FALLBACK_NPZ.exists():
        arrays, source_meta = _load_fallback_npz(FALLBACK_NPZ)
    else:
        raise FileNotFoundError(
            f"No suitable real column source found: {DEFAULT_WRFOUT} or {FALLBACK_NPZ}"
        )

    cases: list[dict[str, Any]] = []
    for name, y, x in _select_columns(arrays):
        case = _case_from_wrfout_column(name, arrays, y, x)
        result = run_sequence(case, nsteps=N_STEPS, sample_every=100)
        result["column"] = {
            "y": y,
            "x": x,
            "lat": float(arrays["XLAT"][y, x]),
            "lon": float(arrays["XLONG"][y, x]),
            "terrain_m": float(arrays["HGT"][y, x]),
        }
        result["real_case_config"] = asdict(case)
        result["summary"] = _summarize_case(result)
        cases.append(result)

    overall = {
        "case_count": len(cases),
        "nsteps": N_STEPS,
        "dt": DT,
        "cancellation_gate_ratio_threshold": CANCELLATION_GATE_RATIO,
        "naive_cases_failing_local_gate": sum(
            bool(case["summary"]["naive_case_fails_local_gate"]) for case in cases
        ),
        "perturbation_cases_passing_bands": sum(
            bool(case["summary"]["perturbation_passes_bands"]) for case in cases
        ),
        "perturbation_cases_all_finite": sum(
            bool(case["summary"]["perturbation_all_finite"]) for case in cases
        ),
        "max_naive_over_perturbation": float(
            max(case["summary"]["max_naive_over_perturbation"] for case in cases)
        ),
        "min_naive_over_perturbation": float(
            min(case["summary"]["min_naive_over_perturbation"] for case in cases)
        ),
    }

    return {
        "schema": "v020_fp32_realcolumn_proof_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "command_context": {
            "cwd": str(Path.cwd()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "python_module": __file__,
        },
        "source": source_meta,
        "method": {
            "description": "Real WRF columns drive the existing NumPy fp64 reference, naive fp32-total, and perturbation-fp32 prototype paths.",
            "nsteps": N_STEPS,
            "dt": DT,
            "acceptance_bands": "proofs/v020/fp32_proto/acceptance_bands.py",
            "truth": "fp64 reference mode of the same prototype initialized from real WRF profiles",
        },
        "summary": overall,
        "cases": cases,
    }


def main() -> int:
    result = run_realcolumn_proof()
    out = OUT_DIR / "proof_results_realcolumn.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

