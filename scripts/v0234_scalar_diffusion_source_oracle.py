"""Retained-carry oracle for the v0.23.4 nested WRF scalar-diffusion RCA.

This is deliberately CPU-only and dispatches no operational model step.  It
compares the active flat/periodic reduction against the equations WRF executes
for ``diff_opt=1, km_opt=4``: live diffusion metrics, map-aware deformation and
Smagorinsky length, nested-domain ownership, and the map-aware
``horizontal_diffusion_3dmp`` scalar flux divergence.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any


REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
}
ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-scalar-diffusion-source-oracle.json"
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_diffopt1_rk1_forward_11c2a085_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
WRFINPUT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf/wrfinput_d03"
)
WRFINPUT_SHA256 = "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a"
WRF_ROOT = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRF_SOURCES = {
    "dyn_em/start_em.F": "4b6c98a06fb2b2c4472af73415099e311502747dc5543d9e9b1323497d83ddb4",
    "dyn_em/module_first_rk_step_part2.F": "9c8c06b246c1cb1bb1632c25e5683a0cfd8545e6832df6eeaa1c744524325738",
    "dyn_em/module_diffusion_em.F": "a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54",
    "dyn_em/module_big_step_utilities_em.F": "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rms(np, value, mask=None) -> float:
    array = np.asarray(value, dtype=np.float64)
    if mask is not None:
        array = array[:, mask]
    return float(np.sqrt(np.mean(array * array)))


def _stats(np, value, ring) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "rms": _rms(np, array),
        "max_abs": float(np.max(np.abs(array))),
        "mean": float(np.mean(array)),
        "nonzero": int(np.count_nonzero(array)),
        "ring0_rms": _rms(np, array, ring == 0),
        "ring1_rms": _rms(np, array, ring == 1),
        "interior_ge5_rms": _rms(np, array, ring >= 5),
        "finite": bool(np.all(np.isfinite(array))),
    }


def _wrf_scalar_diffusion(np, field, xkhh, mass, metrics, dx: float, dy: float):
    """Literal interior of WRF ``horizontal_diffusion_3dmp`` for a nest."""

    msftx = np.asarray(metrics.msftx, dtype=np.float64)
    msfty = np.asarray(metrics.msfty, dtype=np.float64)
    msfux = np.asarray(metrics.msfux, dtype=np.float64)
    msfuy = np.asarray(metrics.msfuy, dtype=np.float64)
    msfvx = np.asarray(metrics.msfvx, dtype=np.float64)
    msfvy = np.asarray(metrics.msfvy, dtype=np.float64)
    q = np.asarray(field, dtype=np.float64)
    k = np.asarray(xkhh, dtype=np.float64)
    mu = np.asarray(mass, dtype=np.float64)
    tendency = np.zeros_like(q)

    k_e = 0.5 * (k[:, 1:-1, 2:] + k[:, 1:-1, 1:-1])
    k_w = 0.5 * (k[:, 1:-1, 1:-1] + k[:, 1:-1, :-2])
    m_e = 0.5 * (mu[:, 1:-1, 2:] + mu[:, 1:-1, 1:-1])
    m_w = 0.5 * (mu[:, 1:-1, 1:-1] + mu[:, 1:-1, :-2])
    k_n = 0.5 * (k[:, 2:, 1:-1] + k[:, 1:-1, 1:-1])
    k_s = 0.5 * (k[:, 1:-1, 1:-1] + k[:, :-2, 1:-1])
    m_n = 0.5 * (mu[:, 2:, 1:-1] + mu[:, 1:-1, 1:-1])
    m_s = 0.5 * (mu[:, 1:-1, 1:-1] + mu[:, :-2, 1:-1])

    mrdx = (msftx[1:-1, 1:-1] * msfty[1:-1, 1:-1])[None, :, :] / dx
    mrdy = (msftx[1:-1, 1:-1] * msfty[1:-1, 1:-1])[None, :, :] / dy
    east_ratio = (msfux[1:-1, 2:-1] / msfuy[1:-1, 2:-1])[None, :, :]
    west_ratio = (msfux[1:-1, 1:-2] / msfuy[1:-1, 1:-2])[None, :, :]
    north_ratio = (msfvy[2:-1, 1:-1] / msfvx[2:-1, 1:-1])[None, :, :]
    south_ratio = (msfvy[1:-2, 1:-1] / msfvx[1:-2, 1:-1])[None, :, :]
    x_flux = mrdx * (
        east_ratio * k_e * m_e / dx * (q[:, 1:-1, 2:] - q[:, 1:-1, 1:-1])
        - west_ratio
        * k_w
        * m_w
        / dx
        * (q[:, 1:-1, 1:-1] - q[:, 1:-1, :-2])
    )
    y_flux = mrdy * (
        north_ratio * k_n * m_n / dy * (q[:, 2:, 1:-1] - q[:, 1:-1, 1:-1])
        - south_ratio
        * k_s
        * m_s
        / dy
        * (q[:, 1:-1, 1:-1] - q[:, :-2, 1:-1])
    )
    tendency[:, 1:-1, 1:-1] = x_flux + y_flux
    return tendency


def main() -> int:
    actual = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual != REQUIRED_ENV:
        raise RuntimeError(f"CPU oracle environment mismatch: {actual!r}")
    if _sha256(STEP0) != STEP0_SHA256 or _sha256(WRFINPUT) != WRFINPUT_SHA256:
        raise RuntimeError("retained input hash mismatch")
    for relative, expected in WRF_SOURCES.items():
        if _sha256(WRF_ROOT / relative) != expected:
            raise RuntimeError(f"pristine WRF source hash mismatch: {relative}")

    print("SCALAR_ORACLE load", flush=True)
    import jax.numpy as jnp
    import numpy as np
    from netCDF4 import Dataset

    from gpuwrf.dynamics.acoustic_wrf import CVPM, P0_PA, R_D
    from gpuwrf.dynamics.explicit_diffusion import (
        horizontal_deformation_2d,
        horizontal_diffusion_coord_scalar_tendency,
        smag2d_horizontal_km,
        wrf_nonperiodic_diffusion_metrics,
    )
    from gpuwrf.dynamics.metrics import load_wrfinput_metrics

    with STEP0.open("rb") as stream:
        carry = pickle.load(stream)
    metrics = load_wrfinput_metrics(WRFINPUT)
    with Dataset(WRFINPUT) as dataset:
        attrs = {
            "HYPSOMETRIC_OPT": int(dataset.getncattr("HYPSOMETRIC_OPT")),
            "USE_THETA_M": int(dataset.getncattr("USE_THETA_M")),
            "DX": float(dataset.getncattr("DX")),
            "DY": float(dataset.getncattr("DY")),
        }
    if attrs != {"HYPSOMETRIC_OPT": 2, "USE_THETA_M": 1, "DX": 1000.0, "DY": 1000.0}:
        raise RuntimeError(f"canonical d03 attributes changed: {attrs!r}")

    state = carry.state
    dx = attrs["DX"]
    dy = attrs["DY"]
    mu_total = np.asarray(state.mu_total, dtype=np.float64)
    mub = np.asarray(state.mu_total - state.mu_perturbation, dtype=np.float64)
    mass = (
        np.asarray(metrics.c1h)[:, None, None] * mu_total[None, :, :]
        + np.asarray(metrics.c2h)[:, None, None]
    )
    pb = np.asarray(state.p_total - state.p_perturbation, dtype=np.float64)
    phb = np.asarray(state.ph_total - state.ph_perturbation, dtype=np.float64)
    dphb = phb[1:, :, :] - phb[:-1, :, :]
    p_top = float(np.asarray(metrics.p_top).reshape(()))
    pfu = (
        np.asarray(metrics.c3f)[1:, None, None] * mub[None, :, :]
        + np.asarray(metrics.c4f)[1:, None, None]
        + p_top
    )
    pfd = (
        np.asarray(metrics.c3f)[:-1, None, None] * mub[None, :, :]
        + np.asarray(metrics.c4f)[:-1, None, None]
        + p_top
    )
    phm = (
        np.asarray(metrics.c3h)[:, None, None] * mub[None, :, :]
        + np.asarray(metrics.c4h)[:, None, None]
        + p_top
    )
    alb_opt2 = dphb / (phm * np.log(pfd / pfu))
    theta_base = alb_opt2 * (P0_PA / R_D) / ((pb / P0_PA) ** CVPM)

    z_at_w = np.asarray(state.ph_total, dtype=np.float64) / 9.81
    rdzw = 1.0 / (z_at_w[1:, :, :] - z_at_w[:-1, :, :])
    zx = np.zeros_like(z_at_w)
    zy = np.zeros_like(z_at_w)
    zx[:, :, 1:] = (z_at_w[:, :, 1:] - z_at_w[:, :, :-1]) / dx
    zy[:, 1:, :] = (z_at_w[:, 1:, :] - z_at_w[:, :-1, :]) / dy

    print("SCALAR_ORACLE deformation", flush=True)
    u = jnp.asarray(state.u)
    v = jnp.asarray(state.v)
    flat_deformation = horizontal_deformation_2d(u, v, dx_m=dx, dy_m=dy)
    map_deformation = horizontal_deformation_2d(
        u,
        v,
        dx_m=dx,
        dy_m=dy,
        msftx=metrics.msftx,
        msfty=metrics.msfty,
        msfux=metrics.msfux,
        msfuy=metrics.msfuy,
        msfvx=metrics.msfvx,
        msfvy=metrics.msfvy,
    )
    full_deformation = horizontal_deformation_2d(
        u,
        v,
        dx_m=dx,
        dy_m=dy,
        msftx=metrics.msftx,
        msfty=metrics.msfty,
        msfux=metrics.msfux,
        msfuy=metrics.msfuy,
        msfvx=metrics.msfvx,
        msfvy=metrics.msfvy,
        zx=jnp.asarray(zx),
        zy=jnp.asarray(zy),
        rdzw=jnp.asarray(rdzw),
        fnm=metrics.fnm,
        fnp=metrics.fnp,
        cf1=metrics.cf1,
        cf2=metrics.cf2,
        cf3=metrics.cf3,
        dn=metrics.dn,
        dnw=metrics.dnw,
    )
    k_flat = np.array(
        smag2d_horizontal_km(
            *flat_deformation, dx_m=dx, dy_m=dy, c_s=0.25
        )[1]
    )
    k_map = np.array(
        smag2d_horizontal_km(
            *map_deformation,
            dx_m=dx,
            dy_m=dy,
            c_s=0.25,
            msftx=metrics.msftx,
            msfty=metrics.msfty,
        )[1]
    )
    k_full = np.array(
        smag2d_horizontal_km(
            *full_deformation,
            dx_m=dx,
            dy_m=dy,
            c_s=0.25,
            msftx=metrics.msftx,
            msfty=metrics.msfty,
        )[1]
    )
    for coefficient in (k_map, k_full):
        coefficient[:, 0, :] = 0.0
        coefficient[:, -1, :] = 0.0
        coefficient[:, :, 0] = 0.0
        coefficient[:, :, -1] = 0.0

    ny, nx = mu_total.shape
    yy, xx = np.indices((ny, nx))
    ring = np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx))
    theta = np.asarray(state.theta, dtype=np.float64)
    perturbation = theta - theta_base
    print("SCALAR_ORACLE flux", flush=True)
    current_tendency = np.asarray(
        horizontal_diffusion_coord_scalar_tendency(
            jnp.asarray(theta),
            jnp.asarray(k_flat),
            jnp.asarray(mass),
            dx_m=dx,
            dy_m=dy,
            base_3d=jnp.asarray(theta_base),
        )
    )
    map_tendency = _wrf_scalar_diffusion(
        np, perturbation, k_map, mass, metrics, dx, dy
    )
    full_tendency = _wrf_scalar_diffusion(
        np, perturbation, k_full, mass, metrics, dx, dy
    )
    production_zx, production_zy, production_rdzw = (
        wrf_nonperiodic_diffusion_metrics(
            jnp.asarray(state.ph_total),
            dx_m=dx,
            dy_m=dy,
        )
    )
    production_tendency = np.asarray(
        horizontal_diffusion_coord_scalar_tendency(
            jnp.asarray(theta),
            jnp.asarray(k_full),
            jnp.asarray(mass),
            dx_m=dx,
            dy_m=dy,
            base_3d=jnp.asarray(theta_base),
            msftx=metrics.msftx,
            msfty=metrics.msfty,
            msfux=metrics.msfux,
            msfuy=metrics.msfuy,
            msfvx=metrics.msfvx,
            msfvy=metrics.msfvy,
            nonperiodic_owned=True,
        )
    )
    production_minus_numpy = production_tendency - full_tendency
    map_delta = map_tendency - current_tendency
    full_delta = full_tendency - current_tendency
    full_vs_map = full_tendency - map_tendency
    map_increment = 6.0 * map_delta / mass
    full_increment = 6.0 * full_delta / mass

    map_ranges = {}
    for name in ("msftx", "msfty", "msfux", "msfuy", "msfvx", "msfvy"):
        value = np.asarray(getattr(metrics, name), dtype=np.float64)
        map_ranges[name] = {"min": float(np.min(value)), "max": float(np.max(value))}
    arrays = {
        "xkhh_current_flat_periodic": k_flat,
        "xkhh_map_nested": k_map,
        "xkhh_full_metrics_nested": k_full,
        "tendency_current": current_tendency,
        "tendency_map_nested": map_tendency,
        "tendency_full_metrics_nested": full_tendency,
        "tendency_candidate_production": production_tendency,
        "candidate_production_minus_numpy_oracle": production_minus_numpy,
        "delta_map_minus_current": map_delta,
        "delta_full_minus_current": full_delta,
        "delta_full_minus_map": full_vs_map,
        "six_second_increment_map_minus_current": map_increment,
        "six_second_increment_full_minus_current": full_increment,
    }
    stats = {name: _stats(np, value, ring) for name, value in arrays.items()}
    checks = {
        "authenticated_step0": True,
        "authenticated_wrfinput": True,
        "pristine_sources_authenticated": True,
        "active_hypsometric_opt2": attrs["HYPSOMETRIC_OPT"] == 2,
        "all_inputs_positive_for_log_inverse": bool(
            np.min(pfu) > 0.0 and np.min(pfd) > 0.0 and np.min(phm) > 0.0
        ),
        "opt2_hydrostatic_reclosure": bool(
            np.max(np.abs(phm * np.log(pfd / pfu) * alb_opt2 - dphb)) < 1.0e-9
        ),
        "full_operator_finite": stats["tendency_full_metrics_nested"]["finite"],
        "wrf_nested_ring0_zero": stats["tendency_full_metrics_nested"]["ring0_rms"] == 0.0,
        "production_metrics_match_numpy": bool(
            np.max(np.abs(np.asarray(production_zx) - zx)) < 1.0e-12
            and np.max(np.abs(np.asarray(production_zy) - zy)) < 1.0e-12
            and np.max(np.abs(np.asarray(production_rdzw) - rdzw)) < 1.0e-12
        ),
        "production_operator_matches_independent_numpy": bool(
            np.max(np.abs(production_minus_numpy)) < 1.0e-9
        ),
        "source_discrepancy_nonzero": stats["delta_full_minus_current"]["rms"] > 0.0,
    }
    proof = {
        "schema": "gpuwrf.v0234.nested-scalar-diffusion-source-oracle.v1",
        "environment": actual,
        "inputs": {
            "step0": {"path": str(STEP0), "sha256": STEP0_SHA256},
            "wrfinput": {"path": str(WRFINPUT), "sha256": WRFINPUT_SHA256},
            "attributes": attrs,
        },
        "pristine_sources": WRF_SOURCES,
        "map_factor_ranges": map_ranges,
        "algebra": {
            "opt2_reclosure_max_abs": float(
                np.max(np.abs(phm * np.log(pfd / pfu) * alb_opt2 - dphb))
            ),
            "theta_base_min": float(np.min(theta_base)),
            "theta_base_max": float(np.max(theta_base)),
            "rdzw_min": float(np.min(rdzw)),
            "rdzw_max": float(np.max(rdzw)),
            "zx_max_abs": float(np.max(np.abs(zx))),
            "zy_max_abs": float(np.max(np.abs(zy))),
        },
        "statistics": stats,
        "checks": checks,
        "causal_interpretation": (
            "The full-metric/nested operator is the first source-backed scalar "
            "candidate only if its retained-carry delta is materially larger than "
            "the already-falsified hypsometric-only delta and focused parity tests "
            "close the exact WRF indexing and conservation semantics."
        ),
        "verdict": (
            "NESTED_SCALAR_DIFFUSION_SOURCE_ORACLE_GREEN"
            if all(checks.values())
            else "NESTED_SCALAR_DIFFUSION_SOURCE_ORACLE_RED"
        ),
    }
    proof["proof_sha256"] = _canonical_hash(proof)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUT)
    print(
        json.dumps(
            {
                "verdict": proof["verdict"],
                "proof_sha256": proof["proof_sha256"],
                "full_delta": stats["delta_full_minus_current"],
                "full_increment": stats["six_second_increment_full_minus_current"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
