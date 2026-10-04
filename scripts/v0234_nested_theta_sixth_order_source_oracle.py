"""CPU-only source/carry oracle for WRF's nested theta sixth-order lane.

No operational model step is dispatched here.  The script authenticates the
canonical case and pristine source, closes the JAX operator against an
independent direct-loop transcription, and quantifies the removed periodic
ring contribution on the retained failed Step200 carry.
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
OUT = SPRINT / "nested-theta-sixth-order-source-oracle.json"
CASE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
RUN = CASE / "corrected_ni_rca_max_22c2bd7a/nested_t_source_cb46ef1b_full18h_discriminator1"
STEP200 = RUN / "failure/first-failed-d03-step-200.pkl"
STEP200_SHA256 = "d0d5769ee25ca98866bd9f305e3502feae7dbb1f8f423745bb2b2ad63317ba5e"
WRFINPUT = CASE / "run/wrf/wrfinput_d03"
WRFINPUT_SHA256 = "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a"
NAMELIST = CASE / "run/wrf/namelist.input"
NAMELIST_SHA256 = "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"
WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRF_SOURCES = {
    "dyn_em/module_em.F": "11105cbf8255f30ca6a44cd7429a92cedce1fb91db6ce90fd7217002a72fb7fa",
    "dyn_em/module_big_step_utilities_em.F": "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815",
    "Registry/Registry.EM_COMMON": "6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a",
}
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
    nz, ny, nx = field.shape
    mass = c1[:, None, None] * mu[None, :, :] + c2[:, None, None]
    out = np.zeros_like(field, dtype=np.float64)
    coefficient = 0.12 * 0.015625 / (2.0 * 6.0)
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
                mx0 = 0.5 * (mass[k, j, i - 1] + mass[k, j, i])
                mx1 = 0.5 * (mass[k, j, i] + mass[k, j, i + 1])
                my0 = 0.5 * (mass[k, j - 1, i] + mass[k, j, i])
                my1 = 0.5 * (mass[k, j, i] + mass[k, j + 1, i])
                out[k, j, i] = coefficient * (
                    msftx[j, i] * (mx1 * x1 - mx0 * x0)
                    + msfty[j, i] * (my1 * y1 - my0 * y0)
                )
    return out


def main() -> int:
    actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    for path, expected in (
        (STEP200, STEP200_SHA256),
        (WRFINPUT, WRFINPUT_SHA256),
        (NAMELIST, NAMELIST_SHA256),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"authenticated input mismatch: {path}")
    for relative, expected in WRF_SOURCES.items():
        if _sha256(WRF / relative) != expected:
            raise RuntimeError(f"pristine source mismatch: {relative}")

    import jax.numpy as jnp
    import numpy as np

    from gpuwrf.dynamics.explicit_diffusion import (
        sixth_order_diffusion_tendency,
        wrf_sixth_order_scalar_tendf,
    )
    from gpuwrf.dynamics.metrics import load_wrfinput_metrics

    namelist_text = NAMELIST.read_text()
    if "diff_6th_opt = 2, 2, 2," not in namelist_text:
        raise RuntimeError("canonical monotonic option changed")
    if "diff_6th_factor = 0.12, 0.12, 0.12," not in namelist_text:
        raise RuntimeError("canonical factor changed")
    if "diff_6th_slopeopt" in namelist_text:
        raise RuntimeError("canonical case no longer uses the WRF default slope option")

    with STEP200.open("rb") as stream:
        carry = pickle.load(stream)
    metrics = load_wrfinput_metrics(WRFINPUT)
    theta = np.asarray(carry.state.theta, dtype=np.float64)
    mu = np.asarray(carry.state.mu_total, dtype=np.float64)
    c1 = np.asarray(metrics.c1h, dtype=np.float64)
    c2 = np.asarray(metrics.c2h, dtype=np.float64)
    msftx = np.asarray(metrics.msftx, dtype=np.float64)
    msfty = np.asarray(metrics.msfty, dtype=np.float64)
    mass = c1[:, None, None] * mu[None, :, :] + c2[:, None, None]

    exact = np.asarray(
        wrf_sixth_order_scalar_tendf(
            jnp.asarray(theta - 300.0),
            jnp.asarray(mu),
            c1=jnp.asarray(c1),
            c2=jnp.asarray(c2),
            msftx=jnp.asarray(msftx),
            msfty=jnp.asarray(msfty),
            dt=6.0,
            diff_6th_factor=0.12,
            monotonic=True,
            specified_or_nested=True,
        )
    )
    oracle = _numpy_wrf_scalar(
        np, theta - 300.0, mu, c1, c2, msftx, msfty
    )
    oracle_delta = exact - oracle
    legacy = np.asarray(
        sixth_order_diffusion_tendency(
            jnp.asarray(theta),
            dt=6.0,
            diff_6th_factor=0.12,
            monotonic=True,
        )
    )
    legacy_increment = 6.0 * msfty[None, :, :] * legacy
    exact_increment = 6.0 * exact / mass

    ny, nx = mu.shape
    yy, xx = np.indices((ny, nx))
    ring = np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx))
    ring1 = ring == 1
    level_rms = np.sqrt(np.mean(legacy_increment[:, ring1] ** 2, axis=1))
    ranked = np.argsort(level_rms)[::-1][:10]
    statistics = {
        "oracle_max_abs": float(np.max(np.abs(oracle_delta))),
        "oracle_rms": _rms(np, oracle_delta),
        "exact_ring0_to_2_nonzero": int(np.count_nonzero(exact[:, ring <= 2])),
        "legacy_full_step_ring1_rms_K": _rms(np, legacy_increment[:, ring1]),
        "legacy_full_step_east_ring1_rms_K": _rms(
            np, legacy_increment[:, 1:-1, -2]
        ),
        "exact_full_step_ring1_rms_K": _rms(np, exact_increment[:, ring1]),
        "legacy_full_step_all_rms_K": _rms(np, legacy_increment),
        "exact_full_step_all_rms_K": _rms(np, exact_increment),
        "candidate_minus_legacy_full_step_all_rms_K": _rms(
            np, exact_increment - legacy_increment
        ),
        "legacy_ring1_ranked_levels": [
            {"k": int(k), "rms_K": float(level_rms[k])} for k in ranked
        ],
    }
    checks = {
        "authenticated_inputs": True,
        "authenticated_pristine_sources": True,
        "canonical_diff_6th_opt_2": True,
        "canonical_factor_0p12": True,
        "canonical_slopeopt_default_zero": True,
        "operator_matches_independent_loop": statistics["oracle_max_abs"] <= 1.0e-9,
        "wrf_nested_rings_zero_to_two": statistics["exact_ring0_to_2_nonzero"] == 0,
        "legacy_ring1_effect_material": statistics["legacy_full_step_ring1_rms_K"] > 1.0e-4,
        "candidate_effect_reaches_t_gate_levels": any(
            26 <= row["k"] <= 33 for row in statistics["legacy_ring1_ranked_levels"]
        ),
    }
    proof = {
        "schema": "gpuwrf.v0234.nested-theta-sixth-order-source-oracle.v1",
        "environment": actual_env,
        "inputs": {
            "step200": {"path": str(STEP200), "sha256": STEP200_SHA256},
            "wrfinput": {"path": str(WRFINPUT), "sha256": WRFINPUT_SHA256},
            "namelist": {"path": str(NAMELIST), "sha256": NAMELIST_SHA256},
        },
        "pristine_sources": WRF_SOURCES,
        "configuration": {
            "diff_6th_opt": 2,
            "diff_6th_factor": 0.12,
            "diff_6th_slopeopt": 0,
            "dt_s": 6.0,
        },
        "statistics": statistics,
        "checks": checks,
        "causal_scope": "theta sixth-order only; U/V/W and retained 2c13b731 wind mechanism are unchanged",
        "verdict": (
            "NESTED_THETA_SIXTH_ORDER_SOURCE_ORACLE_GREEN"
            if all(checks.values())
            else "NESTED_THETA_SIXTH_ORDER_SOURCE_ORACLE_RED"
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
                "statistics": statistics,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
