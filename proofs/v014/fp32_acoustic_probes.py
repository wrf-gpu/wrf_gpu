#!/usr/bin/env python3
"""CPU-only numerical probes for the v0.14 mixed-fp32 acoustic lane.

The probes are intentionally small and source-independent.  They do not claim
production forecast equivalence; they isolate the numerical mechanism that makes
absolute-total fp32 storage unsafe for acoustic residuals while perturbation
storage can preserve the same increments.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


BASE_PRESSURE_PA = 90_000.0
PRESSURE_PERTURBATION_PA = 100.0
MILLIPASCAL_PA = 1.0e-3
PRESSURE_INCREMENTS_PA = (1.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2, 1.0e-1, 1.0)


def _json_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _l2(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def _linf(value: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(value, dtype=np.float64))))


def absolute_total_cancellation_probe() -> dict[str, Any]:
    """Show that fp32 absolute totals quantize or drop small pressure updates."""

    base32 = np.float32(BASE_PRESSURE_PA)
    old_total32 = np.float32(BASE_PRESSURE_PA + PRESSURE_PERTURBATION_PA)
    old_recovered32 = np.float32(old_total32 - base32)
    total_ulp_pa = float(np.spacing(old_total32))

    rows = []
    for increment in PRESSURE_INCREMENTS_PA:
        inc32 = np.float32(increment)

        # Recurrent storage path: add the acoustic residual to the large absolute
        # pressure value.  Updates below the total-field ULP cannot accumulate.
        recurrent_total32 = np.float32(old_total32 + inc32)
        recurrent_recovered32 = np.float32(recurrent_total32 - base32)
        recurrent_delta32 = np.float32(recurrent_recovered32 - old_recovered32)

        # One-shot cast path: even if the exact new total is formed before the fp32
        # cast, the recovered perturbation is still quantized at the total ULP.
        fresh_total32 = np.float32(BASE_PRESSURE_PA + PRESSURE_PERTURBATION_PA + increment)
        fresh_recovered32 = np.float32(fresh_total32 - base32)
        fresh_delta32 = np.float32(fresh_recovered32 - old_recovered32)

        rows.append(
            {
                "increment_pa": float(increment),
                "increment_to_total_ulp": float(increment / total_ulp_pa),
                "recurrent_abs_total_recovered_delta_pa": float(recurrent_delta32),
                "fresh_abs_total_recovered_delta_pa": float(fresh_delta32),
                "recurrent_abs_error_pa": float(float(recurrent_delta32) - increment),
                "fresh_abs_error_pa": float(float(fresh_delta32) - increment),
            }
        )

    millipascal_row = next(row for row in rows if math.isclose(row["increment_pa"], MILLIPASCAL_PA))
    return {
        "base_pressure_pa": BASE_PRESSURE_PA,
        "pressure_perturbation_pa": PRESSURE_PERTURBATION_PA,
        "old_total_fp32_pa": float(old_total32),
        "old_recovered_perturbation_fp32_pa": float(old_recovered32),
        "fp32_ulp_at_total_pa": total_ulp_pa,
        "rows": rows,
        "millipascal_recurrent_recovered_delta_pa": millipascal_row[
            "recurrent_abs_total_recovered_delta_pa"
        ],
        "millipascal_fresh_recovered_delta_pa": millipascal_row["fresh_abs_total_recovered_delta_pa"],
        "mechanism": (
            "The pressure update is quantized at the ULP of the absolute total "
            "pressure, not at the scale of the acoustic residual."
        ),
    }


def perturbation_preservation_probe() -> dict[str, Any]:
    """Show that the same fp32 values preserve millipascal perturbation updates."""

    old_p32 = np.float32(PRESSURE_PERTURBATION_PA)
    perturb_ulp_pa = float(np.spacing(old_p32))

    rows = []
    for increment in PRESSURE_INCREMENTS_PA:
        inc32 = np.float32(increment)
        new_p32 = np.float32(old_p32 + inc32)
        recovered_delta32 = np.float32(new_p32 - old_p32)
        abs_error = float(float(recovered_delta32) - increment)
        rows.append(
            {
                "increment_pa": float(increment),
                "increment_to_perturbation_ulp": float(increment / perturb_ulp_pa),
                "perturbation_recovered_delta_pa": float(recovered_delta32),
                "perturbation_abs_error_pa": abs_error,
                "perturbation_relative_error": float(abs_error / increment),
            }
        )

    millipascal_row = next(row for row in rows if math.isclose(row["increment_pa"], MILLIPASCAL_PA))
    return {
        "pressure_perturbation_pa": PRESSURE_PERTURBATION_PA,
        "fp32_ulp_at_perturbation_pa": perturb_ulp_pa,
        "rows": rows,
        "millipascal_recovered_delta_pa": millipascal_row["perturbation_recovered_delta_pa"],
        "millipascal_relative_error": millipascal_row["perturbation_relative_error"],
        "mechanism": (
            "The pressure update is quantized at the ULP of the perturbation "
            "field, so millipascal residuals are many fp32 ULPs at p'=100 Pa."
        ),
    }


def _run_column_reference(
    *,
    nz: int,
    steps: int,
    dt_s: float,
    dz_m: float,
    bulk_modulus_pa: float,
    rho_kg_m3: float,
    initial_w_amp_m_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
    p = np.zeros(nz, dtype=np.float64)
    z = np.linspace(0.0, math.pi, nz + 1, dtype=np.float64)
    w = initial_w_amp_m_s * np.sin(z)
    w[0] = 0.0
    w[-1] = 0.0
    ph = np.zeros(nz + 1, dtype=np.float64)
    max_abs_dp = []
    max_abs_dph = []

    for _ in range(steps):
        pressure_gradient = (p[1:] - p[:-1]) / dz_m
        w[1:-1] = w[1:-1] - (dt_s / rho_kg_m3) * pressure_gradient
        w[0] = 0.0
        w[-1] = 0.0
        divergence = (w[1:] - w[:-1]) / dz_m
        dp = -bulk_modulus_pa * dt_s * divergence
        dph = 9.81 * dt_s * w
        max_abs_dp.append(_linf(dp))
        max_abs_dph.append(_linf(dph))
        p = p + dp
        ph = ph + dph
    return p, w, ph, max_abs_dp, max_abs_dph


def _run_column_perturbation32(
    *,
    nz: int,
    steps: int,
    dt_s: float,
    dz_m: float,
    bulk_modulus_pa: float,
    rho_kg_m3: float,
    initial_w_amp_m_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.zeros(nz, dtype=np.float32)
    z = np.linspace(0.0, math.pi, nz + 1, dtype=np.float64).astype(np.float32)
    w = (np.float32(initial_w_amp_m_s) * np.sin(z)).astype(np.float32)
    w[0] = np.float32(0.0)
    w[-1] = np.float32(0.0)
    ph = np.zeros(nz + 1, dtype=np.float32)
    dt32 = np.float32(dt_s)
    dz32 = np.float32(dz_m)
    bulk32 = np.float32(bulk_modulus_pa)
    rho32 = np.float32(rho_kg_m3)
    gravity32 = np.float32(9.81)

    for _ in range(steps):
        pressure_gradient = np.float32((p[1:] - p[:-1]) / dz32)
        w[1:-1] = np.float32(w[1:-1] - np.float32(dt32 / rho32) * pressure_gradient)
        w[0] = np.float32(0.0)
        w[-1] = np.float32(0.0)
        divergence = np.float32((w[1:] - w[:-1]) / dz32)
        dp = np.float32(-bulk32 * dt32 * divergence)
        dph = np.float32(gravity32 * dt32 * w)
        p = np.float32(p + dp)
        ph = np.float32(ph + dph)
    return p.astype(np.float64), w.astype(np.float64), ph.astype(np.float64)


def _run_column_absolute32(
    *,
    nz: int,
    steps: int,
    dt_s: float,
    dz_m: float,
    bulk_modulus_pa: float,
    rho_kg_m3: float,
    initial_w_amp_m_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_p = np.linspace(BASE_PRESSURE_PA, BASE_PRESSURE_PA - 500.0, nz, dtype=np.float64)
    base_ph = 50_000.0 + 9.81 * dz_m * np.arange(nz + 1, dtype=np.float64)
    base_p32 = base_p.astype(np.float32)
    base_ph32 = base_ph.astype(np.float32)
    p_total32 = base_p32.copy()
    ph_total32 = base_ph32.copy()

    z = np.linspace(0.0, math.pi, nz + 1, dtype=np.float64).astype(np.float32)
    w = (np.float32(initial_w_amp_m_s) * np.sin(z)).astype(np.float32)
    w[0] = np.float32(0.0)
    w[-1] = np.float32(0.0)
    dt32 = np.float32(dt_s)
    dz32 = np.float32(dz_m)
    bulk32 = np.float32(bulk_modulus_pa)
    rho32 = np.float32(rho_kg_m3)
    gravity32 = np.float32(9.81)

    for _ in range(steps):
        p_work = np.float32(p_total32 - base_p32)
        pressure_gradient = np.float32((p_work[1:] - p_work[:-1]) / dz32)
        w[1:-1] = np.float32(w[1:-1] - np.float32(dt32 / rho32) * pressure_gradient)
        w[0] = np.float32(0.0)
        w[-1] = np.float32(0.0)
        divergence = np.float32((w[1:] - w[:-1]) / dz32)
        dp = np.float32(-bulk32 * dt32 * divergence)
        dph = np.float32(gravity32 * dt32 * w)
        p_total32 = np.float32(p_total32 + dp)
        ph_total32 = np.float32(ph_total32 + dph)

    return (
        np.float32(p_total32 - base_p32).astype(np.float64),
        w.astype(np.float64),
        np.float32(ph_total32 - base_ph32).astype(np.float64),
    )


def one_column_recurrence_probe() -> dict[str, Any]:
    """Small linear-acoustic recurrence showing accumulation sensitivity."""

    nz = 16
    steps = 600
    dt_s = 0.5
    dz_m = 1000.0
    bulk_modulus_pa = 200_000.0
    rho_kg_m3 = 1.0
    initial_w_amp_m_s = 1.0e-5

    ref_p, ref_w, ref_ph, ref_max_dp, ref_max_dph = _run_column_reference(
        nz=nz,
        steps=steps,
        dt_s=dt_s,
        dz_m=dz_m,
        bulk_modulus_pa=bulk_modulus_pa,
        rho_kg_m3=rho_kg_m3,
        initial_w_amp_m_s=initial_w_amp_m_s,
    )
    pert_p, pert_w, pert_ph = _run_column_perturbation32(
        nz=nz,
        steps=steps,
        dt_s=dt_s,
        dz_m=dz_m,
        bulk_modulus_pa=bulk_modulus_pa,
        rho_kg_m3=rho_kg_m3,
        initial_w_amp_m_s=initial_w_amp_m_s,
    )
    abs_p, abs_w, abs_ph = _run_column_absolute32(
        nz=nz,
        steps=steps,
        dt_s=dt_s,
        dz_m=dz_m,
        bulk_modulus_pa=bulk_modulus_pa,
        rho_kg_m3=rho_kg_m3,
        initial_w_amp_m_s=initial_w_amp_m_s,
    )

    abs_p_err = abs_p - ref_p
    pert_p_err = pert_p - ref_p
    abs_w_err = abs_w - ref_w
    pert_w_err = pert_w - ref_w
    abs_ph_err = abs_ph - ref_ph
    pert_ph_err = pert_ph - ref_ph

    base_p_top = np.float32(BASE_PRESSURE_PA)
    pressure_ulp = float(np.spacing(base_p_top))
    base_ph_min = np.float32(50_000.0)
    ph_ulp = float(np.spacing(base_ph_min))
    max_ref_dp = max(ref_max_dp)
    max_ref_dph = max(ref_max_dph)

    return {
        "model": "1D linear acoustic pressure/w/geopotential residual recurrence",
        "grid": {"nz_mass_levels": nz, "vertical_faces": nz + 1, "dz_m": dz_m},
        "parameters": {
            "steps": steps,
            "dt_s": dt_s,
            "bulk_modulus_pa": bulk_modulus_pa,
            "rho_kg_m3": rho_kg_m3,
            "initial_w_amp_m_s": initial_w_amp_m_s,
            "cfl": float(math.sqrt(bulk_modulus_pa / rho_kg_m3) * dt_s / dz_m),
        },
        "per_step_increment_scale": {
            "max_reference_abs_dp_pa": float(max_ref_dp),
            "fp32_ulp_at_90000_pa": pressure_ulp,
            "max_dp_to_pressure_total_ulp": float(max_ref_dp / pressure_ulp),
            "max_reference_abs_dph_m2_s2": float(max_ref_dph),
            "fp32_ulp_at_50000_ph_m2_s2": ph_ulp,
            "max_dph_to_ph_total_ulp": float(max_ref_dph / ph_ulp),
        },
        "final_linf": {
            "fp64_reference_p_pa": _linf(ref_p),
            "absolute_total32_p_pa": _linf(abs_p),
            "perturbation32_p_pa": _linf(pert_p),
            "fp64_reference_ph_m2_s2": _linf(ref_ph),
            "absolute_total32_ph_m2_s2": _linf(abs_ph),
            "perturbation32_ph_m2_s2": _linf(pert_ph),
        },
        "errors_vs_fp64_reference": {
            "absolute_total32_p_l2_pa": _l2(abs_p_err),
            "perturbation32_p_l2_pa": _l2(pert_p_err),
            "absolute_total32_w_l2_m_s": _l2(abs_w_err),
            "perturbation32_w_l2_m_s": _l2(pert_w_err),
            "absolute_total32_ph_l2_m2_s2": _l2(abs_ph_err),
            "perturbation32_ph_l2_m2_s2": _l2(pert_ph_err),
            "p_error_ratio_absolute_over_perturbation": float(_l2(abs_p_err) / max(_l2(pert_p_err), 1.0e-30)),
            "ph_error_ratio_absolute_over_perturbation": float(_l2(abs_ph_err) / max(_l2(pert_ph_err), 1.0e-30)),
        },
        "mechanism": (
            "Every pressure residual update is far below one fp32 ULP of the "
            "absolute total. Adding those residuals to p_total32 prevents the "
            "recurrence from accumulating them; adding them to p_prime32 keeps "
            "the column close to the fp64 residual reference."
        ),
    }


FIELD_SHAPES = {
    "mass3d": "nz * ny * nx",
    "face3d": "(nz + 1) * ny * nx",
    "xstag3d": "nz * ny * (nx + 1)",
    "ystag3d": "nz * (ny + 1) * nx",
    "mass2d": "ny * nx",
    "xstag2d": "ny * (nx + 1)",
    "ystag2d": "(ny + 1) * nx",
}

CORE_CANDIDATE_FIELDS = (
    ("p", "mass3d"),
    ("pm1", "mass3d"),
    ("theta", "mass3d"),
    ("theta_coupled_work", "mass3d"),
    ("theta_ave", "mass3d"),
    ("theta_tend", "mass3d"),
    ("t_2ave", "mass3d"),
    ("u", "xstag3d"),
    ("u_1", "xstag3d"),
    ("ru_m", "xstag3d"),
    ("v", "ystag3d"),
    ("v_1", "ystag3d"),
    ("rv_m", "ystag3d"),
    ("w", "face3d"),
    ("ww", "face3d"),
    ("ww_1", "face3d"),
    ("ww_m", "face3d"),
    ("ph", "face3d"),
    ("ph_tend", "face3d"),
    ("mu", "mass2d"),
    ("muts", "mass2d"),
    ("muave", "mass2d"),
    ("mudf", "mass2d"),
)

PREP_CARRY_CANDIDATE_FIELDS = (
    ("u_save", "xstag3d"),
    ("u_work", "xstag3d"),
    ("v_save", "ystag3d"),
    ("v_work", "ystag3d"),
    ("w_save", "face3d"),
    ("w_work", "face3d"),
    ("ww_save", "face3d"),
    ("t_save", "mass3d"),
    ("theta_work", "mass3d"),
    ("ph_save", "face3d"),
    ("ph_work", "face3d"),
    ("mu_save", "mass2d"),
    ("mu_work", "mass2d"),
)


def _shape_elements(shape: str, *, nx: int, ny: int, nz: int) -> int:
    if shape == "mass3d":
        return nz * ny * nx
    if shape == "face3d":
        return (nz + 1) * ny * nx
    if shape == "xstag3d":
        return nz * ny * (nx + 1)
    if shape == "ystag3d":
        return nz * (ny + 1) * nx
    if shape == "mass2d":
        return ny * nx
    if shape == "xstag2d":
        return ny * (nx + 1)
    if shape == "ystag2d":
        return (ny + 1) * nx
    raise ValueError(f"unknown shape: {shape}")


def _memory_for_fields(fields: tuple[tuple[str, str], ...], *, nx: int, ny: int, nz: int) -> dict[str, Any]:
    rows = []
    total_elements = 0
    for name, shape in fields:
        elements = _shape_elements(shape, nx=nx, ny=ny, nz=nz)
        total_elements += elements
        rows.append(
            {
                "name": name,
                "shape": shape,
                "formula": FIELD_SHAPES[shape],
                "elements": elements,
                "fp64_mib": elements * 8 / 1024**2,
                "fp32_mib": elements * 4 / 1024**2,
                "saving_mib": elements * 4 / 1024**2,
            }
        )
    return {
        "field_count": len(fields),
        "total_elements": total_elements,
        "fp64_mib": total_elements * 8 / 1024**2,
        "fp32_mib": total_elements * 4 / 1024**2,
        "saving_mib": total_elements * 4 / 1024**2,
        "fields": rows,
    }


def memory_savings_probe() -> dict[str, Any]:
    grid = {"nx": 641, "ny": 321, "nz": 50}
    core = _memory_for_fields(CORE_CANDIDATE_FIELDS, **grid)
    prep = _memory_for_fields(PREP_CARRY_CANDIDATE_FIELDS, **grid)
    combined_fields = CORE_CANDIDATE_FIELDS + PREP_CARRY_CANDIDATE_FIELDS
    combined = _memory_for_fields(combined_fields, **grid)
    return {
        "formula": {
            "per_array_saving_bytes": "4 * element_count when demoting fp64 storage to fp32",
            "mass3d": FIELD_SHAPES["mass3d"],
            "face3d": FIELD_SHAPES["face3d"],
            "xstag3d": FIELD_SHAPES["xstag3d"],
            "ystag3d": FIELD_SHAPES["ystag3d"],
            "mass2d": FIELD_SHAPES["mass2d"],
            "note": (
                "This is resident-buffer arithmetic only. Real VRAM savings depend "
                "on liveness, aliasing, buffer donation, compile choices, and which "
                "fp64 islands remain local rather than resident."
            ),
        },
        "example_grid": grid,
        "core_candidate_set": core,
        "prep_carry_candidate_set": prep,
        "core_plus_prep_candidate_set": combined,
    }


def precision_lane_lists() -> dict[str, Any]:
    return {
        "first_fp64_islands_to_keep": [
            "Explicit base/reference state: pb/p_base, phb/ph_base, mub, php_stage, and final total reconstruction.",
            "calc_p_rho local bracket: mass_h/safe_mass, hydrostatic al, EOS pressure bracket, and smdiv pm1 update.",
            "advance_uv pressure-gradient accumulation: p/ph/base-pressure/al/alt/php/dpn/bracket terms.",
            "advance_w coefficient build and Thomas solve: a, alpha, gamma, RHS, t_2ave, and ph_next local arithmetic.",
            "Terrain lower-boundary and terrain PGF terms, especially ht-gradient surface-w coupling.",
            "Boundary/nesting forcing leaves: u_work_bdy, v_work_bdy, ph_bdy_target, ph_save_for_spec, and rw_tend_pg_buoy.",
            "Diagnostics/restart/history interfaces that reconstruct or export absolute totals.",
        ],
        "plausible_fp32_resident_candidates_after_r1_r2": [
            "Perturbation pressure storage p and pressure memory pm1, with fp64 local calc_p_rho first.",
            "Perturbation geopotential/work storage ph and ph_work, with fp64 local PGF/implicit-w arithmetic first.",
            "Perturbation dry-mass/work storage mu, muts, muave, mudf, mu_work, and mass flux carry arrays.",
            "Coupled acoustic work arrays u, v, w, ww, theta_coupled_work, theta_work, and t_2ave storage.",
            "Substep sumflux/carry arrays ru_m, rv_m, ww_m and same-shape save arrays where not used as fp64 boundary references.",
        ],
    }


def build_results() -> dict[str, Any]:
    np.seterr(all="raise")
    results = {
        "metadata": {
            "date": "2026-06-08",
            "worker": "GPT-5.5 xhigh",
            "lane": "worker/gpt/v014-fp32-probes",
            "cpu_only": True,
            "numpy_version": np.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "production_forecast_claim": False,
            "jax_used": False,
            "gpu_used": False,
            "mode_labels": ["fp64_reference", "absolute_total32", "perturbation32"],
        },
        "absolute_total_cancellation": absolute_total_cancellation_probe(),
        "perturbation_form_preservation": perturbation_preservation_probe(),
        "one_column_recurrence_sensitivity": one_column_recurrence_probe(),
        "precision_lane_lists": precision_lane_lists(),
        "memory_savings": memory_savings_probe(),
        "recommendation": {
            "supports_v013_pull_in": False,
            "recommended_scope": "v0.14-only",
            "reason": (
                "The probes support the numerical mechanism for a mixed "
                "perturbation-authoritative lane, but they are not source "
                "integration, WRF fixture parity, transfer audit, or production "
                "forecast evidence. v0.13 should remain fp64."
            ),
        },
    }
    _validate_results(results)
    return results


def _validate_results(results: dict[str, Any]) -> None:
    abs_probe = results["absolute_total_cancellation"]
    pert_probe = results["perturbation_form_preservation"]
    recurrence = results["one_column_recurrence_sensitivity"]["errors_vs_fp64_reference"]

    if abs_probe["millipascal_recurrent_recovered_delta_pa"] != 0.0:
        raise AssertionError("expected recurrent absolute-total fp32 to drop the 1 mPa pressure update")
    if abs(pert_probe["millipascal_relative_error"]) > 0.01:
        raise AssertionError("expected perturbation-form fp32 to preserve the 1 mPa pressure update within 1%")
    if recurrence["p_error_ratio_absolute_over_perturbation"] < 1.0e4:
        raise AssertionError("expected column pressure error to be at least 1e4x worse for absolute totals")
    if recurrence["ph_error_ratio_absolute_over_perturbation"] < 100.0:
        raise AssertionError("expected column geopotential error to be at least 100x worse for absolute totals")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
        help="JSON proof output path",
    )
    args = parser.parse_args()
    results = build_results()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
