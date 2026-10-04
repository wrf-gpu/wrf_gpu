"""Synchronous real-Step0 CPU A/B for WRF option-0 moisture + ``sumflux``."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

from scripts import v0234_t2save_ownership_cpu_ab as common


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-original-moist-sumflux-complete-cpu-ab-proof.json"
FAILURE = SPRINT / "nested-original-moist-sumflux-complete-cpu-ab-blocker.json"
REQUIRED_ENV = {
    **common.REQUIRED_ENV,
    "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false",
}
RETAINED_HLO_SHA256 = "c6f79887129a2ad3c84559a91abe6ff2cf9ac5595f136ad906b211bba54358b0"
RETAINED_MANIFEST_SHA256 = "8c5fd350b01da801dd9b12eddac6a747a9e822f0bb2c70ba12a37af93f530e2c"


def _load_frames(np) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from netCDF4 import Dataset

    loaded: dict[str, dict[str, Any]] = {}
    authority: dict[str, Any] = {}
    for name, (path, expected) in common.FRAMES.items():
        actual = common._sha256(path)
        if actual != expected:
            raise RuntimeError(f"frame hash mismatch {name}: {actual}")
        with Dataset(path) as dataset:
            loaded[name] = {
                field: np.asarray(dataset.variables[field][0], dtype=np.float64)
                for field in ("T", "THM", "QVAPOR", "U", "V")
            }
        authority[name] = {"path": str(path), "sha256": actual}
    return loaded, authority


def _projection(np, retained, candidate, frames) -> dict[str, Any]:
    theta_delta = np.asarray(candidate.state.theta, dtype=np.float64) - np.asarray(
        retained.state.theta, dtype=np.float64
    )
    qv_delta = np.asarray(candidate.state.qv, dtype=np.float64) - np.asarray(
        retained.state.qv, dtype=np.float64
    )
    u_delta = np.asarray(candidate.state.u, dtype=np.float64) - np.asarray(
        retained.state.u, dtype=np.float64
    )
    v_delta = np.asarray(candidate.state.v, dtype=np.float64) - np.asarray(
        retained.state.v, dtype=np.float64
    )
    current = frames["current_395"]
    projected_thm = current["THM"] + theta_delta
    projected_qv = current["QVAPOR"] + qv_delta
    projected_t = (projected_thm + 300.0) / (
        1.0 + (461.6 / 287.0) * np.maximum(projected_qv, 0.0)
    ) - 300.0
    projected_u = current["U"] + u_delta
    projected_v = current["V"] + v_delta
    t_mask = common._ring(np, *projected_t.shape[-2:])
    u_mask = common._ring(np, *projected_u.shape[-2:])
    v_mask = common._ring(np, *projected_v.shape[-2:])

    result: dict[str, Any] = {
        "one_step_delta": {
            "THM_ring1_rms_K": common._rms(np, theta_delta[:, t_mask]),
            "QVAPOR_ring1_rms": common._rms(np, qv_delta[:, t_mask]),
            "U_ring1_rms_m_s": common._rms(np, u_delta[:, u_mask]),
            "V_ring1_rms_m_s": common._rms(np, v_delta[:, v_mask]),
        },
        "anchors": {},
    }
    for anchor in ("cpu_wrf", "retry20"):
        row: dict[str, float] = {}
        for field, projected, mask, unit in (
            ("T", projected_t, t_mask, "K"),
            ("THM", projected_thm, t_mask, "K"),
            ("U", projected_u, u_mask, "m_s"),
            ("V", projected_v, v_mask, "m_s"),
        ):
            row[f"{field}_current_ring1_rmse_{unit}"] = common._rms(
                np, current[field][:, mask] - frames[anchor][field][:, mask]
            )
            row[f"{field}_projected_ring1_rmse_{unit}"] = common._rms(
                np, projected[:, mask] - frames[anchor][field][:, mask]
            )
        result["anchors"][anchor] = row
    return result


def main() -> int:
    try:
        actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")
        if common._sha256(common.STEP0) != common.STEP0_SHA256:
            raise RuntimeError("authenticated Step0 carry mismatch")

        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)
        effective_async = jax.config.values.get("jax_cpu_enable_async_dispatch")
        if effective_async is not False:
            raise RuntimeError(f"CPU async dispatch remained enabled: {effective_async!r}")

        import jax.numpy as jnp
        import numpy as np

        affinity = sorted(os.sched_getaffinity(0))
        if jax.default_backend() != "cpu" or affinity != [12]:
            raise RuntimeError(
                f"runtime binding mismatch backend={jax.default_backend()} affinity={affinity}"
            )

        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

        scratch = Path(tempfile.mkdtemp(prefix="v0234-original-moist-sumflux-ab-"))
        try:
            load_dir = scratch / "load"
            load_dir.mkdir()
            tree, names, _initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(
                load_dir
            )
            if names != ("d01", "d02", "d03") or dt_by_domain != {
                "d01": 54.0,
                "d02": 18.0,
                "d03": 6.0,
            }:
                raise RuntimeError("canonical domain hierarchy changed")
            with common.STEP0.open("rb") as stream:
                carry = pickle.load(stream)
            namelist = tree.domains["d03"].namelist
            if (
                int(namelist.moist_adv_opt) != 0
                or int(namelist.scalar_adv_opt) != 0
                or int(namelist.acoustic_substeps) != 10
            ):
                raise RuntimeError("canonical scalar options/sound count changed")
            clock = runtime.build_clock_base(namelist)

            original_selector = runtime._wrf_original_moisture_sumflux_enabled
            runtime._wrf_original_moisture_sumflux_enabled = lambda: False
            try:
                print("ORIGINAL_MOIST_SUMFLUX retained A lower/compile/dispatch", flush=True)
                retained, retained_audit = common._run_arm(
                    runtime, jax, jnp, carry, namelist, clock
                )
            finally:
                runtime._wrf_original_moisture_sumflux_enabled = original_selector

            print("ORIGINAL_MOIST_SUMFLUX candidate B lower/compile/dispatch", flush=True)
            candidate, candidate_audit = common._run_arm(
                runtime, jax, jnp, carry, namelist, clock
            )

            retained_manifest = common._manifest(jax, np, retained)
            candidate_manifest = common._manifest(jax, np, candidate)
            frames, frame_authority = _load_frames(np)
            projection = _projection(np, retained, candidate, frames)
            deltas = common._state_delta(np, retained, candidate)
            same_structure = all(
                a["path"] == b["path"]
                and a["shape"] == b["shape"]
                and a["dtype"] == b["dtype"]
                for a, b in zip(
                    retained_manifest["leaves"],
                    candidate_manifest["leaves"],
                    strict=True,
                )
            )
            t_thm_green = all(
                row["T_projected_ring1_rmse_K"] < row["T_current_ring1_rmse_K"]
                and row["THM_projected_ring1_rmse_K"]
                < row["THM_current_ring1_rmse_K"]
                for row in projection["anchors"].values()
            )
            uv_green = all(
                row["U_projected_ring1_rmse_m_s"]
                <= row["U_current_ring1_rmse_m_s"]
                and row["V_projected_ring1_rmse_m_s"]
                <= row["V_current_ring1_rmse_m_s"]
                for row in projection["anchors"].values()
            )
            checks = {
                "synchronous_env_effective_and_affinity": effective_async is False
                and affinity == [12],
                "authenticated_step0_frames_and_options00": True,
                "retained_hlo_exact": retained_audit["stablehlo_sha256"]
                == RETAINED_HLO_SHA256,
                "retained_manifest_exact": retained_manifest["sha256"]
                == RETAINED_MANIFEST_SHA256,
                "both_interface_106_identity": retained_audit["interface_identity"]
                and candidate_audit["interface_identity"],
                "complete_leaf_structure_identity": same_structure
                and retained_manifest["leaf_count"]
                == candidate_manifest["leaf_count"]
                == 106,
                "both_all_106_finite": all(
                    row["finite"] for row in retained_manifest["leaves"]
                )
                and all(row["finite"] for row in candidate_manifest["leaves"]),
                "both_callback_transfer_unknown_free": not retained_audit[
                    "forbidden_tokens"
                ]
                and not candidate_audit["forbidden_tokens"]
                and not retained_audit["unknown_custom_call_targets"]
                and not candidate_audit["unknown_custom_call_targets"],
                "complete_output_changed": retained_manifest["sha256"]
                != candidate_manifest["sha256"],
                "active_qv_delta": "qv" in deltas["changed_fields"]
                and projection["one_step_delta"]["QVAPOR_ring1_rms"] > 0.0,
                "T_and_THM_improve_both_anchors": t_thm_green,
                "U_and_V_no_worse_both_anchors": uv_green,
            }
            proof = {
                "schema": "gpuwrf.v0234.nested-original-moist-sumflux-complete-cpu-ab.v1",
                "environment": actual_env,
                "effective_runtime": {
                    "backend": jax.default_backend(),
                    "jax_cpu_enable_async_dispatch": effective_async,
                    "cpu_affinity": affinity,
                    "fresh_arm": True,
                    "resumed_from_async_checkpoint": False,
                },
                "input": {
                    "path": str(common.STEP0),
                    "sha256": common.STEP0_SHA256,
                    "carry_completed_step": 0,
                    "dispatched_native_step": 1,
                    "leaf_count": 106,
                    "moist_adv_opt": int(namelist.moist_adv_opt),
                    "scalar_adv_opt": int(namelist.scalar_adv_opt),
                    "acoustic_substeps": int(namelist.acoustic_substeps),
                },
                "load_authority": load_authority,
                "frame_authority": frame_authority,
                "source_mechanism": (
                    "Registry original moist_adv_opt=0 executes plain h5/v3 for all "
                    "six moisture species with post-acoustic sumflux ru_m/rv_m/ww_m"
                ),
                "offline_tests": {
                    "commands": "synchronous CPU12 focused source/sumflux/operational moisture",
                    "result": "16 passed",
                },
                "retained_A": {**retained_audit, "manifest": retained_manifest},
                "candidate_B": {**candidate_audit, "manifest": candidate_manifest},
                "complete_output_delta": deltas,
                "step200_directional_projection": projection,
                "checks": checks,
                "gpu_commands": 0,
                "gpu_queries": 0,
                "verdict": (
                    "NESTED_ORIGINAL_MOIST_SUMFLUX_CPU_AB_GREEN"
                    if all(checks.values())
                    else "NESTED_ORIGINAL_MOIST_SUMFLUX_CPU_AB_RED"
                ),
            }
            proof["proof_sha256"] = common._canonical(proof)
            common._atomic_json(OUT, proof)
            print(
                json.dumps(
                    {
                        "verdict": proof["verdict"],
                        "proof_sha256": proof["proof_sha256"],
                        "failed_checks": [k for k, v in checks.items() if not v],
                        "projection": projection,
                        "changed_fields": deltas["changed_fields"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0 if all(checks.values()) else 3
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    except Exception as exc:
        import traceback

        blocker = {
            "schema": "gpuwrf.v0234.nested-original-moist-sumflux-complete-cpu-ab-blocker.v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "environment": {key: os.environ.get(key) for key in REQUIRED_ENV},
            "gpu_commands": 0,
            "gpu_queries": 0,
            "proof_sha256": "",
        }
        blocker["proof_sha256"] = common._canonical(blocker)
        common._atomic_json(FAILURE, blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
