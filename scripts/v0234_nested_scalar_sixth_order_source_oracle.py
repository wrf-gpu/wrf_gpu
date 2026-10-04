"""Offline source/carry oracle for WRF RK1-frozen moist scalar diff6.

This script never dispatches an operational timestep.  It authenticates the
retained Step0 carry and pristine WRF sources, compares the production scalar
operator to an independent direct-loop transcription, closes its flux balance,
and ranks the diagnostic-temperature direction against the immutable CPU WRF
and Retry20 00:20 frames.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-scalar-sixth-order-source-oracle.json"
CASE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
RUN = (
    CASE
    / "corrected_ni_rca_max_22c2bd7a/"
    "nested_theta_sixth_order_04478354_full18h_discriminator1"
)
STEP0 = RUN / "failure/last-healthy-d03-step-0.pkl"
STEP0_SHA256 = "acd0d7ad147a3cf41f8302ae827d59294d5d249c8a007002646b878028ebe69d"
WRFINPUT = CASE / "run/wrf/wrfinput_d03"
WRFINPUT_SHA256 = "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a"
NAMELIST = CASE / "run/wrf/namelist.input"
NAMELIST_SHA256 = "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"
CPU_0020 = CASE / "run/wrf/wrfout_d03_2025-03-01_00:20:00"
CPU_0020_SHA256 = "0a1157771f8b00f2c2c4fb66ec1cb63e4534cf3c305981ca81e1cfaad0d8d7f1"
RETRY_0020 = (
    CASE
    / "gpu_validation_retry20_relative_rmse_3ee02c19/gpu-output/"
    "wrfout_d03_2025-03-01_00:20:00"
)
RETRY_0020_SHA256 = "70e09cf3ca22711c66ef529e716ead53fb998d2f548e9939c35d7767ac3f9e62"
CURRENT_0020 = RUN / "output/wrfout_d03_2025-03-01_00:20:00"
CURRENT_0020_SHA256 = "a19d793c0fb88d08bd07d1ef55daf8ba51f5d53428461d804338c30a24aff871"
WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRF_SOURCES = {
    "dyn_em/module_em.F": "11105cbf8255f30ca6a44cd7429a92cedce1fb91db6ce90fd7217002a72fb7fa",
    "dyn_em/module_big_step_utilities_em.F": "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815",
    "dyn_em/solve_em.F": "de58116cf306e40e60f9c9bb1aa74ed5090fbe8631e122ec73e77487ef55e271",
    "dyn_em/module_first_rk_step_part1.F": "8c666fe88c46b04e297fe7b7289f55ec74fa133287b234a02f10e05cbbd11841",
    "Registry/Registry.EM_COMMON": "6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a",
}
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr")
REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rms(np, value) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _numpy_wrf_scalar(np, field, mu, c1, c2, msftx, msfty):
    """Literal scalar branch of sixth_order_diffusion plus flux balance."""

    nz, ny, nx = field.shape
    mass = c1[:, None, None] * mu[None, :, :] + c2[:, None, None]
    out = np.zeros_like(field, dtype=np.float64)
    coefficient = 0.12 * 0.015625 / (2.0 * 2.0)
    boundary_flux = 0.0
    for k in range(nz):
        for j in range(3, ny - 3):
            for i in range(3, nx - 3):
                x0 = 10 * (field[k, j, i] - field[k, j, i - 1]) - 5 * (
                    field[k, j, i + 1] - field[k, j, i - 2]
                ) + field[k, j, i + 2] - field[k, j, i - 3]
                x1 = 10 * (field[k, j, i + 1] - field[k, j, i]) - 5 * (
                    field[k, j, i + 2] - field[k, j, i - 1]
                ) + field[k, j, i + 3] - field[k, j, i - 2]
                y0 = 10 * (field[k, j, i] - field[k, j - 1, i]) - 5 * (
                    field[k, j + 1, i] - field[k, j - 2, i]
                ) + field[k, j + 2, i] - field[k, j - 3, i]
                y1 = 10 * (field[k, j + 1, i] - field[k, j, i]) - 5 * (
                    field[k, j + 2, i] - field[k, j - 1, i]
                ) + field[k, j + 3, i] - field[k, j - 2, i]
                if x0 * (field[k, j, i] - field[k, j, i - 1]) <= 0:
                    x0 = 0.0
                if x1 * (field[k, j, i + 1] - field[k, j, i]) <= 0:
                    x1 = 0.0
                if y0 * (field[k, j, i] - field[k, j - 1, i]) <= 0:
                    y0 = 0.0
                if y1 * (field[k, j + 1, i] - field[k, j, i]) <= 0:
                    y1 = 0.0
                fx0 = 0.5 * (mass[k, j, i - 1] + mass[k, j, i]) * x0
                fx1 = 0.5 * (mass[k, j, i] + mass[k, j, i + 1]) * x1
                fy0 = 0.5 * (mass[k, j - 1, i] + mass[k, j, i]) * y0
                fy1 = 0.5 * (mass[k, j, i] + mass[k, j + 1, i]) * y1
                out[k, j, i] = coefficient * (
                    msftx[j, i] * (fx1 - fx0)
                    + msfty[j, i] * (fy1 - fy0)
                )
                if i == 3:
                    boundary_flux -= coefficient * fx0
                if i == nx - 4:
                    boundary_flux += coefficient * fx1
                if j == 3:
                    boundary_flux -= coefficient * fy0
                if j == ny - 4:
                    boundary_flux += coefficient * fy1
    weighted_sum = float(np.sum(out / msfty[None, :, :]))
    return out, weighted_sum, float(boundary_flux)


def _read_field(Dataset, np, path: Path, name: str):
    with Dataset(path) as dataset:
        return np.asarray(dataset.variables[name][0], dtype=np.float64)


def main() -> int:
    actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    for path, expected in (
        (STEP0, STEP0_SHA256),
        (WRFINPUT, WRFINPUT_SHA256),
        (NAMELIST, NAMELIST_SHA256),
        (CPU_0020, CPU_0020_SHA256),
        (RETRY_0020, RETRY_0020_SHA256),
        (CURRENT_0020, CURRENT_0020_SHA256),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"authenticated input mismatch: {path}")
    for relative, expected in WRF_SOURCES.items():
        if _sha256(WRF / relative) != expected:
            raise RuntimeError(f"pristine source mismatch: {relative}")

    import jax.numpy as jnp
    import numpy as np
    from netCDF4 import Dataset

    from gpuwrf.dynamics.explicit_diffusion import wrf_sixth_order_scalar_tendf
    from gpuwrf.dynamics.metrics import load_wrfinput_metrics

    namelist = NAMELIST.read_text()
    registry = (WRF / "Registry/Registry.EM_COMMON").read_text()
    module_em = (WRF / "dyn_em/module_em.F").read_text()
    solve_em = (WRF / "dyn_em/solve_em.F").read_text()
    first_rk = (WRF / "dyn_em/module_first_rk_step_part1.F").read_text()
    source_checks = {
        "canonical_diff6": "diff_6th_opt = 2, 2, 2," in namelist
        and "diff_6th_factor = 0.12, 0.12, 0.12," in namelist,
        "canonical_advection": "moist_adv_opt = 1, 1, 1," in namelist
        and "scalar_adv_opt = 1, 1, 1," in namelist,
        "canonical_mp8": "mp_physics = 8, 8, 8," in namelist,
        "mix6_defaults_false": "moist_mix6_off" in registry
        and "scalar_mix6_off" in registry
        and registry.count(".false.") >= 2
        and "moist_mix6_off" not in namelist
        and "scalar_mix6_off" not in namelist,
        "rk1_guard": "rk_step_1: IF( rk_step == 1 ) THEN" in module_em,
        "moist_diff6_call": "config_flags%moist_mix6_off" in solve_em,
        "scalar_diff6_call": "config_flags%scalar_mix6_off" in solve_em,
        "scalar_update_unscaled_sc_tend": "tendency(i,k,j) = tendency(i,k,j) + sc_tend(i,k,j,im)" in module_em,
        "tendencies_zero_once": "CALL init_zero_tendency" in first_rk
        and "moist_tend,chem_tend,scalar_tend" in first_rk,
    }

    with STEP0.open("rb") as stream:
        carry = pickle.load(stream)
    metrics = load_wrfinput_metrics(WRFINPUT)
    mu = np.asarray(carry.state.mu_total, dtype=np.float64)
    qv = np.asarray(carry.state.qv, dtype=np.float64)
    theta_m = np.asarray(carry.state.theta, dtype=np.float64)
    c1 = np.asarray(metrics.c1h, dtype=np.float64)
    c2 = np.asarray(metrics.c2h, dtype=np.float64)
    msftx = np.asarray(metrics.msftx, dtype=np.float64)
    msfty = np.asarray(metrics.msfty, dtype=np.float64)
    mass = c1[:, None, None] * mu[None, :, :] + c2[:, None, None]
    exact = np.asarray(
        wrf_sixth_order_scalar_tendf(
            jnp.asarray(qv),
            jnp.asarray(mu),
            c1=jnp.asarray(c1),
            c2=jnp.asarray(c2),
            msftx=jnp.asarray(msftx),
            msfty=jnp.asarray(msfty),
            dt=2.0,
            diff_6th_factor=0.12,
            monotonic=True,
            specified_or_nested=True,
        )
    )
    oracle, weighted_sum, boundary_flux = _numpy_wrf_scalar(
        np, qv, mu, c1, c2, msftx, msfty
    )
    delta = exact - oracle
    final_q_increment = 6.0 * exact / mass
    rvovrd = 461.6 / 287.0
    dry0 = theta_m / (1.0 + rvovrd * np.maximum(qv, 0.0))
    dry1 = theta_m / (1.0 + rvovrd * np.maximum(qv + final_q_increment, 0.0))
    diagnostic_t_delta = dry1 - dry0
    ny, nx = mu.shape
    yy, xx = np.indices((ny, nx))
    rings = np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx))
    species_nonzero = {
        name: int(np.count_nonzero(np.asarray(getattr(carry.state, name))))
        for name in SPECIES
    }

    current_qv = _read_field(Dataset, np, CURRENT_0020, "QVAPOR")
    current_thm = _read_field(Dataset, np, CURRENT_0020, "THM") + 300.0
    current_mu = _read_field(Dataset, np, CURRENT_0020, "MU") + _read_field(
        Dataset, np, CURRENT_0020, "MUB"
    )
    current_t = _read_field(Dataset, np, CURRENT_0020, "T")
    current_mass = c1[:, None, None] * current_mu[None, :, :] + c2[:, None, None]
    current_tend = np.asarray(
        wrf_sixth_order_scalar_tendf(
            jnp.asarray(current_qv),
            jnp.asarray(current_mu),
            c1=jnp.asarray(c1),
            c2=jnp.asarray(c2),
            msftx=jnp.asarray(msftx),
            msfty=jnp.asarray(msfty),
            dt=2.0,
            diff_6th_factor=0.12,
            monotonic=True,
            specified_or_nested=True,
        )
    )
    current_dq = 6.0 * current_tend / current_mass
    current_dry0 = current_thm / (1.0 + rvovrd * np.maximum(current_qv, 0.0)) - 300.0
    current_dry1 = current_thm / (
        1.0 + rvovrd * np.maximum(current_qv + current_dq, 0.0)
    ) - 300.0
    current_dt = current_dry1 - current_dry0
    directional = {}
    for anchor, path in (("cpu_wrf", CPU_0020), ("retry20", RETRY_0020)):
        anchor_t = _read_field(Dataset, np, path, "T")
        before = current_t - anchor_t
        after = before + current_dt
        directional[anchor] = {
            "before_T_full_rms_K": _rms(np, before),
            "after_one_exact_qv_source_T_full_rms_K": _rms(np, after),
            "error_dot_delta": float(np.mean(before * current_dt)),
        }

    checks = {
        **source_checks,
        "authenticated_inputs": True,
        "represented_species_present": all(hasattr(carry.state, name) for name in SPECIES),
        "only_qv_nonzero_at_step0": species_nonzero["qv"] > 0
        and all(species_nonzero[name] == 0 for name in SPECIES[1:]),
        "production_matches_numpy": _rms(np, delta) < 1.0e-15
        and float(np.max(np.abs(delta))) < 1.0e-13,
        "canonical_mass_map_factors_equal": bool(np.array_equal(msftx, msfty)),
        "flux_balance": abs(weighted_sum - boundary_flux) < 1.0e-12,
        "rings_0_to_2_zero": int(np.count_nonzero(exact[:, rings < 3])) == 0,
        "qv_increment_finite_nonnegative": bool(np.isfinite(final_q_increment).all())
        and float(np.min(qv + final_q_increment)) >= 0.0,
        "diagnostic_T_direction_improves_both_anchors": all(
            row["after_one_exact_qv_source_T_full_rms_K"]
            < row["before_T_full_rms_K"]
            for row in directional.values()
        ),
    }
    proof = {
        "schema": "gpuwrf.v0234.nested-scalar-sixth-order-source-oracle.v1",
        "environment": actual_env,
        "inputs": {
            "step0": {"path": str(STEP0), "sha256": STEP0_SHA256},
            "wrfinput": {"path": str(WRFINPUT), "sha256": WRFINPUT_SHA256},
            "namelist": {"path": str(NAMELIST), "sha256": NAMELIST_SHA256},
            "current_0020": {"path": str(CURRENT_0020), "sha256": CURRENT_0020_SHA256},
            "cpu_0020": {"path": str(CPU_0020), "sha256": CPU_0020_SHA256},
            "retry_0020": {"path": str(RETRY_0020), "sha256": RETRY_0020_SHA256},
        },
        "pristine_sources": WRF_SOURCES,
        "configuration": {
            "dt_s": 6.0,
            "rk1_dt_s": 2.0,
            "diff_6th_opt": 2,
            "diff_6th_factor": 0.12,
            "moist_mix6_off": False,
            "scalar_mix6_off": False,
            "species": list(SPECIES),
            "step0_nonzero_counts": species_nonzero,
        },
        "operator": {
            "production_minus_numpy_rms": _rms(np, delta),
            "production_minus_numpy_max_abs": float(np.max(np.abs(delta))),
            "qv_tendf_rms": _rms(np, exact),
            "qv_tendf_max_abs": float(np.max(np.abs(exact))),
            "final_RK_qv_increment_rms": _rms(np, final_q_increment),
            "final_RK_qv_increment_max_abs": float(np.max(np.abs(final_q_increment))),
            "diagnostic_T_increment_rms_K": _rms(np, diagnostic_t_delta),
            "diagnostic_T_increment_max_abs_K": float(np.max(np.abs(diagnostic_t_delta))),
            "diagnostic_T_ring1_rms_K": _rms(np, diagnostic_t_delta[:, rings == 1]),
            "diagnostic_T_ring3_rms_K": _rms(np, diagnostic_t_delta[:, rings == 3]),
            "weighted_tendency_sum": weighted_sum,
            "boundary_flux": boundary_flux,
            "flux_balance_abs": abs(weighted_sum - boundary_flux),
        },
        "directional_rank": directional,
        "causal_limit": "The endpoint source direction ranks the candidate only; it does not replace the complete one-step A/B or Step200 trajectory gate.",
        "checks": checks,
        "verdict": (
            "NESTED_SCALAR_SIXTH_ORDER_SOURCE_ORACLE_GREEN"
            if all(checks.values())
            else "NESTED_SCALAR_SIXTH_ORDER_SOURCE_ORACLE_RED"
        ),
    }
    proof["proof_sha256"] = _canonical(proof)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUT)
    print(
        json.dumps(
            {
                "verdict": proof["verdict"],
                "proof_sha256": proof["proof_sha256"],
                "operator": proof["operator"],
                "directional_rank": directional,
                "failed_checks": [name for name, value in checks.items() if not value],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
