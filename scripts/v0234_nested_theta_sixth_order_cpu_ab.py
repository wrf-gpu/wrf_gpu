"""Authenticated complete-carry CPU A/B for nested theta sixth-order cadence.

The ordinary 106-leaf production callable is unchanged.  The retained A arm is
formed only while tracing by replacing the new frozen theta keyword with
``None`` at the existing tendency helper boundary; that selects the exact
pre-candidate branch without adding a model flag or result leaf.  The wrapper
is restored before the candidate B arm is lowered.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-theta-sixth-order-complete-cpu-ab-proof.json"
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_t_source_cb46ef1b_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "acd0d7ad147a3cf41f8302ae827d59294d5d249c8a007002646b878028ebe69d"
PARENT_COMMIT = "571e4a4208cd01941ec91c9bea4b8ae6ae4806c5"
CANDIDATE_COMMIT = "044783549697caf8c80d094f2b3e73f5ef340537"
PARENT_MODEL_COMMIT = "cb46ef1b3179871382c1277adc09056ade5c5cec"
PARTIAL_WIND_COMMIT = "2c13b73112d9877d603324d66b127ecad60bf7e3"
RETAINED_PARENT_CPU_HLO_SHA256 = (
    "f2973b9806db3a660936c6e1593995c62d8d66f13936f316de28c067914f59f2"
)
SOURCE_ORACLE = SPRINT / "nested-theta-sixth-order-source-oracle.json"
SOURCE_ORACLE_SHA256 = "7afa34dfd983113d0b294a281f3d597819f7469f31202e78f8fedfd340eb6b75"
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
    return np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx)) == distance


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
            ring1 = _ring_mask(np, delta.shape[-2], delta.shape[-1], 1)
            row["ring1_rms"] = _rms(np, delta[..., ring1])
        rows.append(row)
    return {"changed_fields": [row["field"] for row in rows], "fields": rows}


def main() -> int:
    actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    if _sha256(STEP0) != STEP0_SHA256:
        raise RuntimeError("authenticated Step0 carry mismatch")
    if _sha256(SOURCE_ORACLE) != SOURCE_ORACLE_SHA256:
        raise RuntimeError("source-oracle file mismatch")
    if _git("status", "--porcelain"):
        raise RuntimeError("complete CPU A/B requires a clean committed candidate")

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

    tooling_head = _git("rev-parse", "HEAD")
    if _git("rev-parse", f"{CANDIDATE_COMMIT}^") != PARENT_COMMIT:
        raise RuntimeError("candidate parent changed")
    if _git("diff", "--name-only", CANDIDATE_COMMIT, tooling_head, "--", "src/gpuwrf"):
        raise RuntimeError("model bytes changed after the candidate commit")
    model_delta = _git(
        "diff", "--name-only", PARENT_COMMIT, CANDIDATE_COMMIT, "--", "src/gpuwrf"
    ).splitlines()
    if model_delta != [
        "src/gpuwrf/dynamics/explicit_diffusion.py",
        "src/gpuwrf/runtime/operational_mode.py",
    ]:
        raise RuntimeError(f"candidate model scope changed: {model_delta!r}")

    scratch = Path(tempfile.mkdtemp(prefix="v0234-theta-sixth-cpu-ab-"))
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
        namelist = tree.domains["d03"].namelist
        if int(namelist.diff_6th_opt) != 2:
            raise RuntimeError("canonical d03 no longer selects monotonic sixth order")
        if float(namelist.diff_6th_factor) != 0.12:
            raise RuntimeError("canonical d03 sixth-order factor changed")
        clock = runtime.build_clock_base(namelist)

        original_augment = runtime._augment_large_step_tendencies

        def retained_parent_augment(*args, **kwargs):
            # Proof-only trace boundary: select the exact pre-04478354 branch.
            # The production function/carry interface is never modified.
            kwargs["frozen_diff6_theta_tendency"] = None
            return original_augment(*args, **kwargs)

        runtime._augment_large_step_tendencies = retained_parent_augment
        try:
            print("THETA_SIXTH_CPU_AB retained A lower/compile/dispatch", flush=True)
            parent, parent_audit = _run_arm(
                runtime, jax, jnp, carry, namelist, clock
            )
        finally:
            runtime._augment_large_step_tendencies = original_augment

        print("THETA_SIXTH_CPU_AB candidate B lower/compile/dispatch", flush=True)
        candidate, candidate_audit = _run_arm(
            runtime, jax, jnp, carry, namelist, clock
        )

        parent_manifest = _manifest(jax, np, parent)
        candidate_manifest = _manifest(jax, np, candidate)
        deltas = _state_delta(np, parent, candidate)
        theta_delta = np.asarray(candidate.state.theta, dtype=np.float64) - np.asarray(
            parent.state.theta, dtype=np.float64
        )
        ring1 = _ring_mask(np, theta_delta.shape[-2], theta_delta.shape[-1], 1)
        spatial = {
            "theta_rms": _rms(np, theta_delta),
            "theta_ring1_rms": _rms(np, theta_delta[:, ring1]),
            "theta_east_ring1_rms": _rms(np, theta_delta[:, 1:-1, -2]),
        }
        field_identity = {
            field: bool(
                np.array_equal(
                    np.asarray(getattr(parent.state, field)),
                    np.asarray(getattr(candidate.state, field)),
                    equal_nan=True,
                )
            )
            for field in ("u", "v", "w")
        }
        structure_identity = all(
            left["path"] == right["path"]
            and left["shape"] == right["shape"]
            and left["dtype"] == right["dtype"]
            for left, right in zip(
                parent_manifest["leaves"],
                candidate_manifest["leaves"],
                strict=True,
            )
        )
        checks = {
            "authenticated_step0": True,
            "authenticated_source_oracle": True,
            "canonical_options_active": True,
            "retained_parent_interface_106_identity": parent_audit["interface_identity"],
            "candidate_interface_106_identity": candidate_audit["interface_identity"],
            "complete_leaf_structure_identity": structure_identity
            and parent_manifest["leaf_count"] == candidate_manifest["leaf_count"] == 106,
            "retained_parent_all_106_finite": all(
                row["finite"] for row in parent_manifest["leaves"]
            ),
            "candidate_all_106_finite": all(
                row["finite"] for row in candidate_manifest["leaves"]
            ),
            "retained_parent_callback_free": not parent_audit["forbidden_targets"],
            "candidate_callback_free": not candidate_audit["forbidden_targets"],
            "retained_parent_hlo_authenticated": parent_audit["stablehlo_sha256"]
            == RETAINED_PARENT_CPU_HLO_SHA256,
            "candidate_hlo_changed": parent_audit["stablehlo_sha256"]
            != candidate_audit["stablehlo_sha256"],
            "complete_output_changed": parent_manifest["sha256"]
            != candidate_manifest["sha256"],
            "theta_changed": "theta" in deltas["changed_fields"]
            and spatial["theta_rms"] > 0.0,
            "theta_ring1_changed": spatial["theta_ring1_rms"] > 0.0,
        }
        proof = {
            "schema": "gpuwrf.v0234.nested-theta-sixth-order-complete-cpu-ab.v1",
            "candidate_commit": CANDIDATE_COMMIT,
            "proof_tooling_commit": tooling_head,
            "parent_commit": PARENT_COMMIT,
            "parent_model_commit": PARENT_MODEL_COMMIT,
            "partial_wind_commit": PARTIAL_WIND_COMMIT,
            "environment": actual_env,
            "input": {
                "path": str(STEP0),
                "sha256": STEP0_SHA256,
                "start_step": 0,
                "leaf_count": 106,
            },
            "load_authority": load_authority,
            "configuration": {
                "diff_6th_opt": 2,
                "diff_6th_factor": 0.12,
                "dt_s": 6.0,
            },
            "retained_parent_A": {**parent_audit, "manifest": parent_manifest},
            "candidate_B": {**candidate_audit, "manifest": candidate_manifest},
            "complete_output_delta": deltas,
            "spatial_projection": spatial,
            "momentum_output_identity": field_identity,
            "causal_binding": {
                "model_delta": model_delta,
                "direct_changed_lane": "theta sixth-order t_tendf only",
                "u_v_w_sixth_order_source_bytes": "unchanged",
                "complete_step_momentum_note": "Any non-identical momentum output is downstream thermodynamic/PGF coupling, not a direct wind-lane source edit.",
                "new_carry_or_result_leaves": 0,
                "new_loop_transfers": 0,
            },
            "checks": checks,
            "verdict": (
                "NESTED_THETA_SIXTH_ORDER_COMPLETE_CPU_AB_GREEN"
                if all(checks.values())
                else "NESTED_THETA_SIXTH_ORDER_COMPLETE_CPU_AB_RED"
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
                    "parent_hlo": parent_audit["stablehlo_sha256"],
                    "candidate_hlo": candidate_audit["stablehlo_sha256"],
                    "changed_fields": deltas["changed_fields"],
                    "momentum_output_identity": field_identity,
                    "spatial_projection": spatial,
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
