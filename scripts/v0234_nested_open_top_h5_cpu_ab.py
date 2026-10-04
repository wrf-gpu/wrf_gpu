"""Authenticated complete-carry CPU A/B for nested rigid-vs-WRF-open top.

Pristine WRF defaults ``top_lid`` false and the canonical operational namelist
does not override it.  The live-nest loader still hard-codes true even though
the single-domain production loader and native acoustic oracles use the WRF
open top.  This offline A/B ranks that exact static discrepancy on the sealed
h_sca=5 parent; Step200 remains the only trajectory verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import traceback
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-open-top-h5-complete-cpu-ab-proof.json"
FAILURE = SPRINT / "nested-open-top-h5-complete-cpu-ab-blocker.json"
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_scalar_sixth_order_18d97595_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
NAMELIST_CACHE = Path("/tmp/v0234-scalar-controls-d03-namelist-dc3fecdd.pkl")
NAMELIST_CACHE_SHA256 = "87cd017eb7a8e1194e5d1702b979f004e7a2533d549106e2d44bfb5508365e69"
CANONICAL_NAMELIST = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/config/namelist.input"
)
CANONICAL_NAMELIST_SHA256 = "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"
WRF_REGISTRY = Path("<USER_HOME>/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON")
WRF_REGISTRY_SHA256 = "6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a"
WRF_SMALL_STEP = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F")
CURRENT_0020 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_h_sca_order_395fb800_full18h_discriminator1/output/"
    "wrfout_d03_2025-03-01_00:20:00"
)
CPU_0020 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf/"
    "wrfout_d03_2025-03-01_00:20:00"
)
RETRY_0020 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "gpu_validation_retry20_relative_rmse_3ee02c19/gpu-output/"
    "wrfout_d03_2025-03-01_00:20:00"
)
REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
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


def _ring(np, ny: int, nx: int, distance: int):
    yy, xx = np.indices((ny, nx))
    return np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx)) == distance


def _rmse(np, left, right, mask) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(delta[..., mask] ** 2)))


def _read_fields(path: Path) -> dict[str, Any]:
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        return {name: ds.variables[name][0, ...] for name in ("T", "U")}


def _projection(np, current, candidate_delta, reference) -> dict[str, float]:
    ring1 = _ring(np, current.shape[-2], current.shape[-1], 1)
    projected = current + candidate_delta
    return {
        "before_ring1_rmse": _rmse(np, current, reference, ring1),
        "after_one_step_delta_ring1_rmse": _rmse(np, projected, reference, ring1),
        "error_dot_delta": float(
            np.mean(
                (np.asarray(current)[..., ring1] - np.asarray(reference)[..., ring1])
                * np.asarray(candidate_delta)[..., ring1]
            )
        ),
    }


def main() -> int:
    actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    for path, expected in (
        (STEP0, STEP0_SHA256),
        (NAMELIST_CACHE, NAMELIST_CACHE_SHA256),
        (CANONICAL_NAMELIST, CANONICAL_NAMELIST_SHA256),
        (WRF_REGISTRY, WRF_REGISTRY_SHA256),
    ):
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"authenticated input mismatch: {path}: {actual}")

    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
    import gpuwrf.contracts.state as state_contract

    state_contract._gpu_device = lambda: jax.devices("cpu")[0]
    import gpuwrf.runtime.operational_mode as runtime
    from scripts.v0234_nested_scalar_controls_cpu_ranking import _delta_summary
    from scripts.v0234_rk1_frozen_theta_diffusion_cpu_ab import _manifest, _run_arm

    with NAMELIST_CACHE.open("rb") as stream:
        base, load_authority = pickle.load(stream)
    with STEP0.open("rb") as stream:
        carry = pickle.load(stream)
    if load_authority["dt_by_domain"] != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
        raise RuntimeError(f"timestep hierarchy changed: {load_authority['dt_by_domain']!r}")
    if bool(base.top_lid) is not True:
        raise RuntimeError("cached live-nest parent no longer exposes top_lid=True")

    canonical_text = CANONICAL_NAMELIST.read_text().lower()
    registry_text = WRF_REGISTRY.read_text()
    small_step_text = WRF_SMALL_STEP.read_text()
    small_step_compact = "".join(small_step_text.split()).lower()
    nested_text = (ROOT / "src/gpuwrf/integration/nested_pipeline.py").read_text()
    daily_text = (ROOT / "src/gpuwrf/integration/daily_pipeline.py").read_text()
    source_checks = {
        "canonical_namelist_omits_top_lid": "top_lid" not in canonical_text,
        "wrf_registry_default_false": (
            "rconfig   logical top_lid" in registry_text
            and "max_domains    .false." in next(
                line for line in registry_text.splitlines() if "rconfig   logical top_lid" in line
            )
        ),
        "nested_loader_hardcodes_true": "top_lid=True," in nested_text,
        "daily_loader_defaults_open": (
            'top_lid=os.environ.get("GPUWRF_TOP_LID", "0") == "1"' in daily_text
        ),
        "wrf_acoustic_lid_flag_conditional": (
            "if(top_lid)lid_flag=0" in small_step_compact
        ),
    }

    clock = runtime.build_clock_base(base)
    configs = {
        "A_h5_rigid_top": dataclass_replace(
            base, h_sca_adv_order=5, moist_adv_opt=0, scalar_adv_opt=0, top_lid=True
        ),
        "B_h5_wrf_open_top": dataclass_replace(
            base, h_sca_adv_order=5, moist_adv_opt=0, scalar_adv_opt=0, top_lid=False
        ),
    }
    arms: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for name, namelist in configs.items():
        print(f"OPEN_TOP_H5_CPU_AB {name} lower/compile/dispatch", flush=True)
        result, audit = _run_arm(runtime, jax, jnp, carry, namelist, clock)
        arms[name] = result
        audits[name] = audit
        manifests[name] = _manifest(jax, np, result)

    parent = arms["A_h5_rigid_top"]
    candidate = arms["B_h5_wrf_open_top"]
    delta = _delta_summary(np, parent, candidate)
    theta_a = np.asarray(parent.state.theta, dtype=np.float64)
    theta_b = np.asarray(candidate.state.theta, dtype=np.float64)
    qv_a = np.asarray(parent.state.qv, dtype=np.float64)
    qv_b = np.asarray(candidate.state.qv, dtype=np.float64)
    rvovrd = 461.6 / 287.0
    t_delta = (
        theta_b / (1.0 + rvovrd * np.maximum(qv_b, 0.0))
        - theta_a / (1.0 + rvovrd * np.maximum(qv_a, 0.0))
    )
    u_delta = np.asarray(candidate.state.u, dtype=np.float64) - np.asarray(
        parent.state.u, dtype=np.float64
    )
    current = _read_fields(CURRENT_0020)
    cpu = _read_fields(CPU_0020)
    retry = _read_fields(RETRY_0020)
    if current["T"].shape != t_delta.shape or current["U"].shape != u_delta.shape:
        raise RuntimeError(
            "projection shape mismatch: "
            f"T {current['T'].shape}/{t_delta.shape}, U {current['U'].shape}/{u_delta.shape}"
        )
    projection = {
        field: {
            anchor: _projection(
                np,
                np.asarray(current[field], dtype=np.float64),
                field_delta,
                np.asarray(reference[field], dtype=np.float64),
            )
            for anchor, reference in (("cpu_wrf", cpu), ("retry20", retry))
        }
        for field, field_delta in (("T", t_delta), ("U", u_delta))
    }
    t_direction_green = all(
        row["after_one_step_delta_ring1_rmse"] < row["before_ring1_rmse"]
        and row["error_dot_delta"] < 0.0
        for row in projection["T"].values()
    )
    # This is deliberately weaker than claiming local U improvement: the frozen
    # causal requirement is to retain the 2c wind improvement against immutable
    # 4484.  The projection may move either way but must stay below both baselines.
    u_retains_partial_margin = (
        projection["U"]["cpu_wrf"]["after_one_step_delta_ring1_rmse"]
        < 0.2857256114890511
        and projection["U"]["retry20"]["after_one_step_delta_ring1_rmse"]
        < 0.27772695352886917
    )

    checks = {
        "authenticated_inputs": True,
        "source_contract_green": all(source_checks.values()),
        "complete_output_changed": (
            manifests["A_h5_rigid_top"]["sha256"]
            != manifests["B_h5_wrf_open_top"]["sha256"]
        ),
        "both_interface_106_identity": all(
            row["interface_identity"] for row in audits.values()
        ),
        "both_callback_free": all(
            not row["forbidden_targets"] for row in audits.values()
        ),
        "both_complete_outputs_finite": all(
            all(leaf["finite"] for leaf in manifest["leaves"])
            for manifest in manifests.values()
        ),
        "candidate_reaches_diagnostic_T_ring1": (
            delta["diagnostic_T"]["ring1_rms_K"] > 0.0
        ),
        "T_directional_projection_improves_both_anchors": t_direction_green,
        "U_projection_retains_partial_wind_margin": u_retains_partial_margin,
    }
    proof = {
        "schema": "gpuwrf.v0234.nested-open-top-h5-complete-cpu-ab.v1",
        "environment": actual_env,
        "inputs": {
            "step0": {"path": str(STEP0), "sha256": STEP0_SHA256, "leaf_count": 106},
            "namelist_cache": {"path": str(NAMELIST_CACHE), "sha256": NAMELIST_CACHE_SHA256},
            "canonical_namelist": {
                "path": str(CANONICAL_NAMELIST),
                "sha256": CANONICAL_NAMELIST_SHA256,
            },
            "current_0020": {"path": str(CURRENT_0020), "sha256": _sha256(CURRENT_0020)},
            "cpu_0020": {"path": str(CPU_0020), "sha256": _sha256(CPU_0020)},
            "retry_0020": {"path": str(RETRY_0020), "sha256": _sha256(RETRY_0020)},
        },
        "source_checks": source_checks,
        "arms": {
            name: {
                "configuration": {
                    "top_lid": bool(config.top_lid),
                    "h_sca_adv_order": int(config.h_sca_adv_order),
                    "moist_adv_opt": int(config.moist_adv_opt),
                    "scalar_adv_opt": int(config.scalar_adv_opt),
                },
                "audit": audits[name],
                "manifest_sha256": manifests[name]["sha256"],
            }
            for name, config in configs.items()
        },
        "complete_output_delta": delta,
        "directional_projection": projection,
        "checks": checks,
        "causal_limit": (
            "One complete authenticated CPU step plus local 00:20 projection admits "
            "only the bounded Step200 trajectory discriminator."
        ),
        "prediction": (
            "If nested rigid-top divergence drives the remaining THM incident, the "
            "WRF-open-top B arm must reduce Step200 T ring1 RMSE against CPU WRF and "
            "Retry20 while retaining the source-backed partial-wind improvement."
        ),
        "verdict": (
            "NESTED_OPEN_TOP_H5_CPU_AB_GREEN"
            if all(checks.values())
            else "NESTED_OPEN_TOP_H5_CPU_AB_RED"
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
                "diagnostic_T": delta["diagnostic_T"],
                "directional_projection": projection,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - retain exact offline blocker.
        payload = {
            "schema": "gpuwrf.v0234.nested-open-top-h5-complete-cpu-ab-blocker.v1",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "gpu_commands": 0,
            "model_arms_started": 0,
        }
        payload["proof_sha256"] = _canonical(payload)
        temporary = FAILURE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, FAILURE)
        print(json.dumps(payload, sort_keys=True), flush=True)
        raise
