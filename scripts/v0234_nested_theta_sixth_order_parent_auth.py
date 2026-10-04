"""Authenticate the CPU A trace against the actual pre-candidate source tree.

Run this script with ``PYTHONPATH=<git-archive>/src:<current-root>`` and
``GPUWRF_PARENT_EXPORT=<git-archive>``.  It compiles and dispatches exactly one
ordinary production step from the retained 106-leaf carry, then compares the
complete output manifest to the already retained proof-only A trace.  No model
switch, callback, observer, or extra carry leaf is introduced.
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
OUT = SPRINT / "nested-theta-sixth-order-actual-parent-auth.json"
RED_AB = SPRINT / "nested-theta-sixth-order-complete-cpu-ab-proof.json"
RED_AB_SHA256 = "643595907584e845efdf34c1443b1486078e6bdebd3f39261e7c2396684538d3"
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_t_source_cb46ef1b_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "acd0d7ad147a3cf41f8302ae827d59294d5d249c8a007002646b878028ebe69d"
PARENT_COMMIT = "571e4a4208cd01941ec91c9bea4b8ae6ae4806c5"
RETAINED_PARENT_CPU_HLO_SHA256 = (
    "f2973b9806db3a660936c6e1593995c62d8d66f13936f316de28c067914f59f2"
)
MODEL_FILES = (
    "src/gpuwrf/dynamics/explicit_diffusion.py",
    "src/gpuwrf/runtime/operational_mode.py",
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    export = Path(os.environ["GPUWRF_PARENT_EXPORT"]).resolve()
    if not export.is_dir():
        raise RuntimeError(f"parent export missing: {export}")
    if _sha256(STEP0) != STEP0_SHA256 or _sha256(RED_AB) != RED_AB_SHA256:
        raise RuntimeError("retained proof/input hash mismatch")

    source_rows = {}
    for relative in MODEL_FILES:
        exported = export / relative
        expected = subprocess.check_output(
            ("git", "-C", str(ROOT), "show", f"{PARENT_COMMIT}:{relative}")
        )
        row = {
            "exported_sha256": _sha256(exported),
            "git_object_sha256": _sha256_bytes(expected),
        }
        row["identity"] = row["exported_sha256"] == row["git_object_sha256"]
        source_rows[relative] = row
    if not all(row["identity"] for row in source_rows.values()):
        raise RuntimeError("parent export source identity failed")

    import jax
    import jax.numpy as jnp

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
    import gpuwrf.contracts.state as state_contract

    state_contract._gpu_device = lambda: jax.devices("cpu")[0]

    import gpuwrf.runtime.operational_mode as runtime
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary
    from scripts.v0234_rk1_frozen_theta_diffusion_cpu_ab import _manifest, _run_arm

    runtime_path = Path(runtime.__file__).resolve()
    if export not in runtime_path.parents:
        raise RuntimeError(f"runtime did not import from parent export: {runtime_path}")

    scratch = Path(tempfile.mkdtemp(prefix="v0234-theta-sixth-parent-auth-"))
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
        clock = runtime.build_clock_base(namelist)
        print("THETA_SIXTH_PARENT_AUTH lower/compile/dispatch", flush=True)
        result, audit = _run_arm(runtime, jax, jnp, carry, namelist, clock)

        import numpy as np

        manifest = _manifest(jax, np, result)
        red = json.loads(RED_AB.read_text())
        traced_a = red["retained_parent_A"]
        checks = {
            "actual_parent_sources_authenticated": all(
                row["identity"] for row in source_rows.values()
            ),
            "runtime_imported_from_actual_parent": True,
            "authenticated_step0": True,
            "ordinary_interface_106_identity": audit["interface_identity"],
            "all_106_finite": all(row["finite"] for row in manifest["leaves"]),
            "callback_free": not audit["forbidden_targets"],
            "actual_parent_hlo_matches_retained_parent": audit["stablehlo_sha256"]
            == RETAINED_PARENT_CPU_HLO_SHA256,
            "proof_wrapper_A_complete_output_identity": manifest["sha256"]
            == traced_a["manifest"]["sha256"],
            "proof_wrapper_A_structure_identity": all(
                left["path"] == right["path"]
                and left["shape"] == right["shape"]
                and left["dtype"] == right["dtype"]
                for left, right in zip(
                    manifest["leaves"], traced_a["manifest"]["leaves"], strict=True
                )
            ),
        }
        proof = {
            "schema": "gpuwrf.v0234.nested-theta-sixth-order-actual-parent-auth.v1",
            "parent_commit": PARENT_COMMIT,
            "parent_export": str(export),
            "runtime_module": str(runtime_path),
            "environment": actual_env,
            "source_identity": source_rows,
            "input": {"path": str(STEP0), "sha256": STEP0_SHA256},
            "load_authority": load_authority,
            "actual_parent": {**audit, "manifest": manifest},
            "retained_wrapper_A": {
                "proof_path": str(RED_AB),
                "proof_file_sha256": RED_AB_SHA256,
                "stablehlo_sha256": traced_a["stablehlo_sha256"],
                "manifest_sha256": traced_a["manifest"]["sha256"],
            },
            "checks": checks,
            "verdict": (
                "NESTED_THETA_SIXTH_ORDER_ACTUAL_PARENT_AUTH_GREEN"
                if all(checks.values())
                else "NESTED_THETA_SIXTH_ORDER_ACTUAL_PARENT_AUTH_RED"
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
                    "actual_parent_hlo": audit["stablehlo_sha256"],
                    "actual_parent_manifest": manifest["sha256"],
                    "wrapper_A_manifest": traced_a["manifest"]["sha256"],
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
