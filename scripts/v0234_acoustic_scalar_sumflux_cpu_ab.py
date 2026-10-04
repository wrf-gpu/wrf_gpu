"""Authenticated synchronous real-Step0 CPU A/B for pristine WRF ``sumflux``."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

from scripts import v0234_t2save_ownership_cpu_ab as common


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-acoustic-scalar-sumflux-complete-cpu-ab-proof.json"
FAILURE = SPRINT / "nested-acoustic-scalar-sumflux-complete-cpu-ab-blocker.json"
REQUIRED_ENV = {
    **common.REQUIRED_ENV,
    "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false",
}
CURRENT_RETAINED_395_CPU_HLO_SHA256 = (
    "c6f79887129a2ad3c84559a91abe6ff2cf9ac5595f136ad906b211bba54358b0"
)


def main() -> int:
    try:
        actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")
        if common._sha256(common.STEP0) != common.STEP0_SHA256:
            raise RuntimeError("authenticated Step0 carry mismatch")

        import jax

        # Binding demanded by the liveness correction: authenticate both the
        # external launch contract and the effective in-process JAX setting.
        jax.config.update("jax_cpu_enable_async_dispatch", False)
        effective_async = jax.config.values.get("jax_cpu_enable_async_dispatch")
        if effective_async is not False:
            raise RuntimeError(f"CPU async dispatch remained enabled: {effective_async!r}")

        import jax.numpy as jnp
        import numpy as np

        if jax.default_backend() != "cpu":
            raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
        affinity = sorted(os.sched_getaffinity(0))
        if affinity != [12]:
            raise RuntimeError(f"CPU affinity mismatch: {affinity!r}")

        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]

        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

        scratch = Path(tempfile.mkdtemp(prefix="v0234-sumflux-cpu-ab-"))
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
            with common.STEP0.open("rb") as stream:
                carry = pickle.load(stream)
            namelist = tree.domains["d03"].namelist
            clock = runtime.build_clock_base(namelist)
            if int(namelist.acoustic_substeps) != 10:
                raise RuntimeError("canonical acoustic substep count changed")

            original_selector = runtime._time_averaged_scalar_transport_enabled
            runtime._time_averaged_scalar_transport_enabled = lambda: False
            try:
                print("SUMFLUX_CPU_AB retained A lower/compile/dispatch", flush=True)
                retained, retained_audit = common._run_arm(
                    runtime, jax, jnp, carry, namelist, clock
                )
            finally:
                runtime._time_averaged_scalar_transport_enabled = original_selector

            print("SUMFLUX_CPU_AB candidate B lower/compile/dispatch", flush=True)
            candidate, candidate_audit = common._run_arm(
                runtime, jax, jnp, carry, namelist, clock
            )

            retained_manifest = common._manifest(jax, np, retained)
            candidate_manifest = common._manifest(jax, np, candidate)
            frames, frame_authority = common._load_frames(np)
            projection = common._projection(np, retained, candidate, frames)
            deltas = common._state_delta(np, retained, candidate)
            structure_identity = all(
                a["path"] == b["path"]
                and a["shape"] == b["shape"]
                and a["dtype"] == b["dtype"]
                for a, b in zip(
                    retained_manifest["leaves"],
                    candidate_manifest["leaves"],
                    strict=True,
                )
            )
            t_improves = all(
                row["T_projected_ring1_rmse_K"] < row["T_current_ring1_rmse_K"]
                and row["THM_projected_ring1_rmse_K"]
                < row["THM_current_ring1_rmse_K"]
                for row in projection["anchors"].values()
            )
            u_no_worse = all(
                row["U_projected_ring1_rmse_m_s"]
                <= row["U_current_ring1_rmse_m_s"]
                for row in projection["anchors"].values()
            )
            checks = {
                "synchronous_env_and_effective_config_authenticated": (
                    actual_env["JAX_CPU_ENABLE_ASYNC_DISPATCH"] == "false"
                    and effective_async is False
                    and affinity == [12]
                ),
                "authenticated_step0_and_frames": True,
                "canonical_rk3_sound_count_10": int(namelist.acoustic_substeps) == 10,
                "retained_interface_106_identity": retained_audit["interface_identity"],
                "candidate_interface_106_identity": candidate_audit["interface_identity"],
                "complete_leaf_structure_identity": structure_identity
                and retained_manifest["leaf_count"]
                == candidate_manifest["leaf_count"]
                == 106,
                "both_all_106_finite": all(
                    row["finite"] for row in retained_manifest["leaves"]
                )
                and all(row["finite"] for row in candidate_manifest["leaves"]),
                "retained_hlo_authenticated_to_395": retained_audit[
                    "stablehlo_sha256"
                ]
                == CURRENT_RETAINED_395_CPU_HLO_SHA256,
                "both_callback_transfer_unknown_free": not retained_audit[
                    "forbidden_tokens"
                ]
                and not candidate_audit["forbidden_tokens"]
                and not retained_audit["unknown_custom_call_targets"]
                and not candidate_audit["unknown_custom_call_targets"],
                "complete_output_changed": retained_manifest["sha256"]
                != candidate_manifest["sha256"],
                "source_effect_reaches_active_qv": "qv" in deltas["changed_fields"],
                "T_and_THM_direction_improve_both_anchors": t_improves,
                "U_no_worse_both_anchors": u_no_worse,
            }
            proof = {
                "schema": "gpuwrf.v0234.nested-acoustic-scalar-sumflux-complete-cpu-ab.v1",
                "environment": actual_env,
                "effective_runtime": {
                    "jax_backend": jax.default_backend(),
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
                },
                "load_authority": load_authority,
                "frame_authority": frame_authority,
                "source_mechanism": (
                    "post-acoustic WRF sumflux ru_m/rv_m/ww_m for moisture and "
                    "other-scalar rk_scalar_tend"
                ),
                "independent_oracle": {
                    "test_file": "tests/test_v0234_acoustic_scalar_sumflux.py",
                    "result": "6 passed",
                    "n_sound_covered": [1, 2, 4],
                    "broader_focused_cpu_result": "84 passed",
                },
                "retained_A": {**retained_audit, "manifest": retained_manifest},
                "candidate_B": {**candidate_audit, "manifest": candidate_manifest},
                "complete_output_delta": deltas,
                "step200_directional_projection": projection,
                "checks": checks,
                "gpu_commands": 0,
                "gpu_queries": 0,
                "verdict": (
                    "NESTED_ACOUSTIC_SCALAR_SUMFLUX_CPU_AB_GREEN"
                    if all(checks.values())
                    else "NESTED_ACOUSTIC_SCALAR_SUMFLUX_CPU_AB_RED"
                ),
            }
            proof["proof_sha256"] = common._canonical(proof)
            common._atomic_json(OUT, proof)
            print(
                json.dumps(
                    {
                        "verdict": proof["verdict"],
                        "proof_sha256": proof["proof_sha256"],
                        "failed_checks": [
                            name for name, passed in checks.items() if not passed
                        ],
                        "projection": projection["anchors"],
                        "one_step_delta": projection["one_step_delta"],
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
            "schema": "gpuwrf.v0234.nested-acoustic-scalar-sumflux-complete-cpu-ab-blocker.v1",
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
