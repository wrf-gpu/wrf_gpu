"""Authenticated complete-carry CPU A/B for nested WRF theta-source cadence."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-t-source-cadence-cpu-ab-proof.json"
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_scalar_diffusion_aca6b55b_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
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


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(ROOT), *args), text=True
    ).strip()


def _ring_mask(np, ny: int, nx: int, distance: int):
    yy, xx = np.indices((ny, nx))
    return np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx)) == int(distance)


def _rms(np, value) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _state_delta(np, parent, candidate) -> dict[str, Any]:
    rows = []
    for name in sorted(set(dir(parent.state)) & set(dir(candidate.state))):
        if name.startswith("_"):
            continue
        left = getattr(parent.state, name, None)
        right = getattr(candidate.state, name, None)
        if left is None or callable(left) or not hasattr(left, "shape"):
            continue
        a = np.asarray(left)
        b = np.asarray(right)
        if a.shape != b.shape or not np.issubdtype(a.dtype, np.floating):
            continue
        delta = b.astype(np.float64) - a.astype(np.float64)
        if not np.any(delta != 0.0):
            continue
        row = {
            "field": name,
            "changed_values": int(np.count_nonzero(delta)),
            "rms": _rms(np, delta),
            "max_abs": float(np.max(np.abs(delta))),
        }
        if delta.ndim >= 2 and delta.shape[-2:] == candidate.state.theta.shape[-2:]:
            mask = _ring_mask(np, delta.shape[-2], delta.shape[-1], 1)
            row["ring1_rms"] = _rms(np, delta[..., mask])
        rows.append(row)
    return {"changed_fields": [row["field"] for row in rows], "fields": rows}


def main() -> int:
    actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    if _sha256(STEP0) != STEP0_SHA256:
        raise RuntimeError("authenticated Step0 carry mismatch")
    if _git("status", "--porcelain"):
        raise RuntimeError("CPU A/B requires a clean committed candidate")

    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
    import gpuwrf.contracts.state as state_contract

    state_contract._gpu_device = lambda: jax.devices("cpu")[0]

    import gpuwrf.runtime.operational_mode as runtime
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary
    from scripts.v0234_rk1_frozen_theta_diffusion_cpu_ab import (
        _manifest,
        _run_arm,
    )

    head = _git("rev-parse", "HEAD")
    model_delta = _git("diff", "--name-only", "HEAD^", "HEAD", "--", "src/gpuwrf")
    if model_delta.splitlines() != ["src/gpuwrf/integration/nested_pipeline.py"]:
        raise RuntimeError(f"candidate model delta is not loader-only: {model_delta!r}")

    scratch = Path(tempfile.mkdtemp(prefix="v0234-nested-t-source-cpu-ab-"))
    try:
        load_dir = scratch / "load"
        load_dir.mkdir()
        tree, names, _initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(
            load_dir
        )
        if names != ("d01", "d02", "d03"):
            raise RuntimeError(f"domain order changed: {names!r}")
        if dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
            raise RuntimeError(f"timestep hierarchy changed: {dt_by_domain!r}")
        with STEP0.open("rb") as stream:
            carry = pickle.load(stream)
        candidate_namelist = tree.domains["d03"].namelist
        if int(candidate_namelist.rad_rk_tendf) != 1:
            raise RuntimeError("nested loader did not select source-leaf cadence")
        if (
            int(candidate_namelist.ra_sw_physics),
            int(candidate_namelist.ra_lw_physics),
            int(candidate_namelist.bl_pbl_physics),
            int(candidate_namelist.cu_physics),
        ) != (4, 4, 5, 0):
            raise RuntimeError("canonical d03 T-source suite changed")
        parent_namelist = dataclass_replace(candidate_namelist, rad_rk_tendf=0)
        clock = runtime.build_clock_base(candidate_namelist)

        print("NESTED_T_SOURCE_AB legacy lower/compile/dispatch", flush=True)
        parent, parent_audit = _run_arm(
            runtime, jax, jnp, carry, parent_namelist, clock
        )
        print("NESTED_T_SOURCE_AB wrf lower/compile/dispatch", flush=True)
        candidate, candidate_audit = _run_arm(
            runtime, jax, jnp, carry, candidate_namelist, clock
        )

        parent_manifest = _manifest(jax, np, parent)
        candidate_manifest = _manifest(jax, np, candidate)
        deltas = _state_delta(np, parent, candidate)
        theta_delta = np.asarray(
            candidate.state.theta, dtype=np.float64
        ) - np.asarray(parent.state.theta, dtype=np.float64)
        u_delta = np.asarray(candidate.state.u, dtype=np.float64) - np.asarray(
            parent.state.u, dtype=np.float64
        )
        ring1 = _ring_mask(np, theta_delta.shape[-2], theta_delta.shape[-1], 1)
        east = theta_delta[:, 1:-1, -2]
        level_rms = np.sqrt(np.mean(theta_delta[:, ring1] ** 2, axis=1))
        ranked_levels = np.argsort(level_rms)[::-1][:10]
        spatial = {
            "theta_rms": _rms(np, theta_delta),
            "theta_ring1_rms": _rms(np, theta_delta[:, ring1]),
            "theta_east_ring1_rms": _rms(np, east),
            "theta_levels_ranked": [
                {"k": int(k), "ring1_rms": float(level_rms[k])}
                for k in ranked_levels
            ],
            "u_rms": _rms(np, u_delta),
            "u_max_abs": float(np.max(np.abs(u_delta))),
        }
        structure_identity = all(
            a["path"] == b["path"]
            and a["shape"] == b["shape"]
            and a["dtype"] == b["dtype"]
            for a, b in zip(
                parent_manifest["leaves"],
                candidate_manifest["leaves"],
                strict=True,
            )
        )
        checks = {
            "authenticated_step0": True,
            "loader_selected_wrf_source_cadence": True,
            "canonical_d03_suite": True,
            "legacy_interface_106_identity": parent_audit["interface_identity"],
            "candidate_interface_106_identity": candidate_audit["interface_identity"],
            "complete_leaf_structure_identity": structure_identity
            and parent_manifest["leaf_count"] == candidate_manifest["leaf_count"] == 106,
            "legacy_all_106_finite": all(
                row["finite"] for row in parent_manifest["leaves"]
            ),
            "candidate_all_106_finite": all(
                row["finite"] for row in candidate_manifest["leaves"]
            ),
            "legacy_callback_free": not parent_audit["forbidden_targets"],
            "candidate_callback_free": not candidate_audit["forbidden_targets"],
            "hlo_changed": parent_audit["stablehlo_sha256"]
            != candidate_audit["stablehlo_sha256"],
            "complete_output_changed": parent_manifest["sha256"]
            != candidate_manifest["sha256"],
            "theta_changed": "theta" in deltas["changed_fields"]
            and spatial["theta_ring1_rms"] > 0.0,
            "source_effect_reaches_east_upper_gate": spatial["theta_east_ring1_rms"]
            > 0.0
            and any(26 <= row["k"] <= 33 for row in spatial["theta_levels_ranked"]),
        }
        proof = {
            "schema": "gpuwrf.v0234.nested-t-source-cadence-cpu-ab.v1",
            "candidate_commit": head,
            "environment": actual_env,
            "input": {
                "path": str(STEP0),
                "sha256": STEP0_SHA256,
                "start_step": 0,
                "leaf_count": 106,
            },
            "load_authority": load_authority,
            "configuration": {
                "legacy_rad_rk_tendf": 0,
                "candidate_rad_rk_tendf": 1,
                "dt_s": 6.0,
                "use_theta_m": 1,
                "ra_sw_physics": 4,
                "ra_lw_physics": 4,
                "bl_pbl_physics": 5,
                "cu_physics": 0,
            },
            "legacy_A": {**parent_audit, "manifest": parent_manifest},
            "candidate_B": {**candidate_audit, "manifest": candidate_manifest},
            "complete_output_delta": deltas,
            "spatial_projection": spatial,
            "causal_binding": {
                "model_delta": model_delta.splitlines(),
                "new_carry_leaves": 0,
                "new_loop_transfers": 0,
                "mechanism": "existing WRF RTHRATEN/RTHBLTEN/RQVBLTEN moist-theta source-leaf cadence selected by the nested loader",
                "partial_wind_mechanism": "2c13b731 nested edge stencils retained unchanged",
            },
            "checks": checks,
            "verdict": (
                "NESTED_T_SOURCE_CADENCE_CPU_AB_GREEN"
                if all(checks.values())
                else "NESTED_T_SOURCE_CADENCE_CPU_AB_RED"
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
                    "theta_ring1_rms": spatial["theta_ring1_rms"],
                    "theta_east_ring1_rms": spatial["theta_east_ring1_rms"],
                    "u_rms": spatial["u_rms"],
                    "changed_fields": deltas["changed_fields"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if all(checks.values()) else 3
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
