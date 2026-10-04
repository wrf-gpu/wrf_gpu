"""Synchronous real-Step0 A/B for actual WRF scalar opts 1/1 plus ``sumflux``."""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

from scripts import v0234_original_moist_sumflux_cpu_ab as support
from scripts import v0234_t2save_ownership_cpu_ab as common


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-actual-opts11-sumflux-complete-cpu-ab-proof.json"
FAILURE = SPRINT / "nested-actual-opts11-sumflux-complete-cpu-ab-blocker.json"
REQUIRED_ENV = {**common.REQUIRED_ENV, "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false"}


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
        import jax.numpy as jnp
        import numpy as np

        affinity = sorted(os.sched_getaffinity(0))
        if effective_async is not False or jax.default_backend() != "cpu" or affinity != [12]:
            raise RuntimeError(
                f"runtime binding mismatch async={effective_async} "
                f"backend={jax.default_backend()} affinity={affinity}"
            )
        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

        scratch = Path(tempfile.mkdtemp(prefix="v0234-actual-opts11-sumflux-ab-"))
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
                raise RuntimeError("canonical hierarchy changed")
            loaded = tree.domains["d03"].namelist
            if (
                int(loaded.moist_adv_opt),
                int(loaded.scalar_adv_opt),
                int(loaded.acoustic_substeps),
            ) != (1, 1, 10):
                raise RuntimeError(
                    "candidate loader did not reproduce actual WRF opts/sounds: "
                    f"{loaded.moist_adv_opt}/{loaded.scalar_adv_opt}/"
                    f"{loaded.acoustic_substeps}"
                )
            retained_namelist = dataclasses.replace(
                loaded, moist_adv_opt=0, scalar_adv_opt=0
            )
            with common.STEP0.open("rb") as stream:
                carry = pickle.load(stream)
            clock = runtime.build_clock_base(loaded)

            selector = runtime._time_averaged_scalar_transport_enabled
            runtime._time_averaged_scalar_transport_enabled = lambda: False
            try:
                print("ACTUAL_OPTS11_SUMFLUX retained A lower/compile/dispatch", flush=True)
                retained, retained_audit = common._run_arm(
                    runtime, jax, jnp, carry, retained_namelist, clock
                )
            finally:
                runtime._time_averaged_scalar_transport_enabled = selector

            print("ACTUAL_OPTS11_SUMFLUX candidate B lower/compile/dispatch", flush=True)
            candidate, candidate_audit = common._run_arm(
                runtime, jax, jnp, carry, loaded, clock
            )

            retained_manifest = common._manifest(jax, np, retained)
            candidate_manifest = common._manifest(jax, np, candidate)
            frames, frame_authority = support._load_frames(np)
            projection = support._projection(np, retained, candidate, frames)
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
                row["U_projected_ring1_rmse_m_s"] <= row["U_current_ring1_rmse_m_s"]
                and row["V_projected_ring1_rmse_m_s"] <= row["V_current_ring1_rmse_m_s"]
                for row in projection["anchors"].values()
            )
            checks = {
                "synchronous_cpu12_fresh": effective_async is False and affinity == [12],
                "authenticated_input_and_loader_opts11": True,
                "retained_hlo_exact": retained_audit["stablehlo_sha256"]
                == support.RETAINED_HLO_SHA256,
                "retained_manifest_exact": retained_manifest["sha256"]
                == support.RETAINED_MANIFEST_SHA256,
                "both_interface_106_identity": retained_audit["interface_identity"]
                and candidate_audit["interface_identity"],
                "complete_structure_identity": same_structure
                and retained_manifest["leaf_count"]
                == candidate_manifest["leaf_count"]
                == 106,
                "both_106_finite": all(x["finite"] for x in retained_manifest["leaves"])
                and all(x["finite"] for x in candidate_manifest["leaves"]),
                "callback_transfer_unknown_free": not retained_audit["forbidden_tokens"]
                and not candidate_audit["forbidden_tokens"]
                and not retained_audit["unknown_custom_call_targets"]
                and not candidate_audit["unknown_custom_call_targets"],
                "active_qv_and_complete_delta": "qv" in deltas["changed_fields"]
                and retained_manifest["sha256"] != candidate_manifest["sha256"],
                "T_and_THM_improve_both_anchors": t_thm_green,
                "U_and_V_no_worse_both_anchors": uv_green,
            }
            proof = {
                "schema": "gpuwrf.v0234.nested-actual-opts11-sumflux-complete-cpu-ab.v1",
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
                    "retained_options": [0, 0],
                    "candidate_loaded_options": [
                        int(loaded.moist_adv_opt),
                        int(loaded.scalar_adv_opt),
                    ],
                    "acoustic_substeps": int(loaded.acoustic_substeps),
                },
                "load_authority": load_authority,
                "frame_authority": frame_authority,
                "source_mechanism": (
                    "actual parsed WRF moist/scalar opts 1/1 plus post-acoustic "
                    "time-averaged sumflux scalar transport"
                ),
                "offline_tests": "focused synchronous source/loader/sumflux tests",
                "retained_A": {**retained_audit, "manifest": retained_manifest},
                "candidate_B": {**candidate_audit, "manifest": candidate_manifest},
                "complete_output_delta": deltas,
                "step200_directional_projection": projection,
                "checks": checks,
                "gpu_commands": 0,
                "gpu_queries": 0,
                "verdict": (
                    "NESTED_ACTUAL_OPTS11_SUMFLUX_CPU_AB_GREEN"
                    if all(checks.values())
                    else "NESTED_ACTUAL_OPTS11_SUMFLUX_CPU_AB_RED"
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
            "schema": "gpuwrf.v0234.nested-actual-opts11-sumflux-complete-cpu-ab-blocker.v1",
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
