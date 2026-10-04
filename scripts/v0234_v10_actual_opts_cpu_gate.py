"""Current-tree CPU structure gate for the V10 opts-1/1 ``sumflux`` candidate."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

from scripts import v0234_t2save_ownership_cpu_ab as common


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
OUT = SPRINT / "ACTUAL_OPTS11_CURRENT_TREE_CPU_GATE.json"
FAILURE = SPRINT / "ACTUAL_OPTS11_CURRENT_TREE_CPU_GATE_BLOCKER.json"
EXPECTED_AFFINITY = [13, 14, 15, 29, 30, 31]
REQUIRED_ENV = {
    **common.REQUIRED_ENV,
    "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false",
}


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    try:
        actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")
        if common._sha256(common.STEP0) != common.STEP0_SHA256:
            raise RuntimeError("authenticated Step0 carry mismatch")

        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)
        import jax.numpy as jnp
        import numpy as np

        affinity = sorted(os.sched_getaffinity(0))
        if (
            jax.default_backend() != "cpu"
            or jax.config.values.get("jax_cpu_enable_async_dispatch") is not False
            or affinity != EXPECTED_AFFINITY
        ):
            raise RuntimeError(
                "runtime binding mismatch: "
                f"backend={jax.default_backend()} async="
                f"{jax.config.values.get('jax_cpu_enable_async_dispatch')} "
                f"affinity={affinity}"
            )

        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

        scratch = Path(tempfile.mkdtemp(prefix="v0234-v10-opts11-cpu-gate-"))
        try:
            load_dir = scratch / "load"
            load_dir.mkdir()
            tree, names, _initial, dt_by_domain, load_authority = (
                ordinary.load_corrected_tree(load_dir)
            )
            if names != ("d01", "d02", "d03") or dt_by_domain != {
                "d01": 54.0,
                "d02": 18.0,
                "d03": 6.0,
            }:
                raise RuntimeError("canonical domain hierarchy changed")
            namelist = tree.domains["d03"].namelist
            loaded_options = (
                int(namelist.moist_adv_opt),
                int(namelist.scalar_adv_opt),
                int(namelist.acoustic_substeps),
            )
            if loaded_options != (1, 1, 10):
                raise RuntimeError(f"actual opts/sounds mismatch: {loaded_options!r}")

            with common.STEP0.open("rb") as stream:
                carry = pickle.load(stream)
            input_manifest = common._manifest(jax, np, carry)
            clock = runtime.build_clock_base(namelist)
            result, audit = common._run_arm(
                runtime, jax, jnp, carry, namelist, clock
            )
            output_manifest = common._manifest(jax, np, result)
            same_structure = all(
                left["path"] == right["path"]
                and left["shape"] == right["shape"]
                and left["dtype"] == right["dtype"]
                for left, right in zip(
                    input_manifest["leaves"],
                    output_manifest["leaves"],
                    strict=True,
                )
            )
            qv_delta = np.asarray(result.state.qv, dtype=np.float64) - np.asarray(
                carry.state.qv, dtype=np.float64
            )
            checks = {
                "authenticated_real_step0": True,
                "exact_domain_hierarchy": True,
                "loader_resolves_actual_opts11_sounds10": loaded_options
                == (1, 1, 10),
                "input_output_interface_106_identity": audit["interface_identity"]
                and same_structure
                and input_manifest["leaf_count"]
                == output_manifest["leaf_count"]
                == 106,
                "all_output_leaves_finite": all(
                    row["finite"] for row in output_manifest["leaves"]
                ),
                "no_host_callback_or_transfer_tokens": not audit["forbidden_tokens"],
                "no_unknown_custom_call_targets": not audit[
                    "unknown_custom_call_targets"
                ],
                "moisture_transport_is_active": int(np.count_nonzero(qv_delta)) > 0,
            }
            proof = {
                "schema": "gpuwrf.v0234.v10-actual-opts11-current-tree-cpu-gate.v1",
                "candidate_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "candidate_worktree_dirty": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], cwd=ROOT, text=True
                    ).strip()
                ),
                "environment": actual_env,
                "effective_runtime": {
                    "backend": jax.default_backend(),
                    "jax_cpu_enable_async_dispatch": False,
                    "cpu_affinity": affinity,
                },
                "input": {
                    "path": str(common.STEP0),
                    "sha256": common.STEP0_SHA256,
                    "carry_completed_step": 0,
                    "dispatched_native_steps": 1,
                    "leaf_count": input_manifest["leaf_count"],
                },
                "loaded_options": {
                    "moist_adv_opt": loaded_options[0],
                    "scalar_adv_opt": loaded_options[1],
                    "acoustic_substeps": loaded_options[2],
                },
                "source_sha256": {
                    "nested_pipeline.py": _source_sha256(
                        ROOT / "src/gpuwrf/integration/nested_pipeline.py"
                    ),
                    "operational_mode.py": _source_sha256(
                        ROOT / "src/gpuwrf/runtime/operational_mode.py"
                    ),
                },
                "load_authority_sha256": common._canonical(load_authority),
                "lower_compile_dispatch_audit": audit,
                "input_manifest_sha256": input_manifest["sha256"],
                "output_manifest_sha256": output_manifest["sha256"],
                "output_leaf_count": output_manifest["leaf_count"],
                "qv_changed_values": int(np.count_nonzero(qv_delta)),
                "qv_delta_rms": common._rms(np, qv_delta),
                "checks": checks,
                "gpu_commands": 0,
                "gpu_queries": 0,
                "verdict": (
                    "ACTUAL_OPTS11_CURRENT_TREE_CPU_GATE_GREEN"
                    if all(checks.values())
                    else "ACTUAL_OPTS11_CURRENT_TREE_CPU_GATE_RED"
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
                        "audit": audit,
                        "qv_changed_values": proof["qv_changed_values"],
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
            "schema": "gpuwrf.v0234.v10-actual-opts11-current-tree-cpu-gate-blocker.v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "environment": {key: os.environ.get(key) for key in REQUIRED_ENV},
            "gpu_commands": 0,
            "gpu_queries": 0,
        }
        blocker["proof_sha256"] = common._canonical(blocker)
        common._atomic_json(FAILURE, blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
