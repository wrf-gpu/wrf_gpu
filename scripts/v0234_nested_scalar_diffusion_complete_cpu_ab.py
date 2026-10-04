"""Authenticated complete 106-leaf CPU A/B for the nested scalar repair.

The A output is the retained, authenticated 11c2a085 ordinary one-step
manifest.  This script executes only the new B arm on the identical carry and
compares every output leaf, preserving the earlier A dispatch as evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-scalar-diffusion-complete-cpu-ab-proof.json"
BASELINE = SPRINT / "nested-diffopt1-rk1-forward-bundle-cpu-ab-proof.json"
BASELINE_FILE_SHA256 = "f54724d0a27dec1cc025bf64065b34dd50c9d92adbbdc54b2b6ab41dd7039e61"
BASELINE_PAYLOAD_SHA256 = "96f3ceda7b41ec86bc007f5ef5ae9c705303d73687f0ecd1dcd6f73f53426322"
BASELINE_CANDIDATE_COMMIT = "11c2a08528b72c9d3ca2e7ac9dffd8b37dd6fb8d"
BASELINE_HLO_SHA256 = "3b5a2e36ac9a99a7dd7311fd51569cf119aad4d70afcb2c9b715c3b1f8003ab7"
BASELINE_MANIFEST_SHA256 = "6f9d36bfac86f1c9988b9eda0ea2c38081247ee29d2a3d3892b79ebcea25dc80"
CANDIDATE_COMMIT = "aca6b55bdf0c5feac749f5f94358c73fb284507e"
SOURCE_HASHES = {
    "src/gpuwrf/dynamics/explicit_diffusion.py": "9caabc9e38df44c670bc4a10b9a4c2c25497e0060744219a19aec284154514f1",
    "src/gpuwrf/runtime/operational_mode.py": "d73973967bfb8253ac47f900c713c7e21fb4e476839eaf04a840fa669d474ae2",
}
ORACLE = SPRINT / "nested-scalar-diffusion-source-oracle.json"
ORACLE_FILE_SHA256 = "108fbaa27f25ad758a99296792221622f2d57257a09b24646bb172efe4b2131e"
ORACLE_PAYLOAD_SHA256 = "2c0f76140517b5cad5af95d262a75e5ac858eb70a413f189cd3f5baea9e0a8e9"
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "theta_unlimited_e06583f0_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "acd0d7ad147a3cf41f8302ae827d59294d5d249c8a007002646b878028ebe69d"
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


def _canonical(payload: dict) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    if _sha256(STEP0) != STEP0_SHA256:
        raise RuntimeError("authenticated step-0 carry hash mismatch")
    if _sha256(BASELINE) != BASELINE_FILE_SHA256:
        raise RuntimeError("retained A proof file hash mismatch")
    baseline = json.loads(BASELINE.read_text())
    if _canonical(baseline) != BASELINE_PAYLOAD_SHA256:
        raise RuntimeError("retained A proof payload hash mismatch")
    if baseline.get("proof_sha256") != BASELINE_PAYLOAD_SHA256:
        raise RuntimeError("retained A proof embedded hash mismatch")
    if baseline.get("verdict") != "NESTED_DIFFOPT1_RK1_FORWARD_CPU_AB_GREEN":
        raise RuntimeError("retained A proof is not green")
    if _sha256(ORACLE) != ORACLE_FILE_SHA256:
        raise RuntimeError("scalar oracle file hash mismatch")
    oracle = json.loads(ORACLE.read_text())
    if (
        _canonical(oracle) != ORACLE_PAYLOAD_SHA256
        or oracle.get("proof_sha256") != ORACLE_PAYLOAD_SHA256
        or oracle.get("verdict") != "NESTED_SCALAR_DIFFUSION_SOURCE_ORACLE_GREEN"
    ):
        raise RuntimeError("scalar oracle authority mismatch")
    for relative, expected in SOURCE_HASHES.items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"candidate source hash mismatch: {relative}")

    import jax
    import jax.numpy as jnp

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

    scratch = Path(tempfile.mkdtemp(prefix="v0234-nested-scalar-complete-ab-"))
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
        _periodic, specified, nested = runtime._acoustic_lateral_bc_flags(namelist)
        if (
            (int(namelist.diff_opt), int(namelist.km_opt)) != (1, 4)
            or int(namelist.hypsometric_opt) != 2
            or specified
            or not nested
        ):
            raise RuntimeError("canonical d03 scalar candidate path is not active")
        clock = runtime.build_clock_base(namelist)
        print("SCALAR_COMPLETE_AB candidate lower/compile/dispatch", flush=True)
        candidate, candidate_audit = _run_arm(
            runtime, jax, jnp, carry, namelist, clock
        )
        candidate_manifest = _manifest(jax, __import__("numpy"), candidate)
        baseline_arm = baseline["candidate"]
        baseline_manifest = baseline_arm["manifest"]
        if baseline_arm["stablehlo_sha256"] != BASELINE_HLO_SHA256:
            raise RuntimeError("retained A HLO hash mismatch")
        if baseline_manifest["sha256"] != BASELINE_MANIFEST_SHA256:
            raise RuntimeError("retained A manifest hash mismatch")
        old_rows = baseline_manifest["leaves"]
        new_rows = candidate_manifest["leaves"]
        structure_match = all(
            left["path"] == right["path"]
            and left["shape"] == right["shape"]
            and left["dtype"] == right["dtype"]
            for left, right in zip(old_rows, new_rows, strict=True)
        )
        changed = [
            {
                "path": old["path"],
                "baseline_sha256": old["sha256"],
                "candidate_sha256": new["sha256"],
            }
            for old, new in zip(old_rows, new_rows, strict=True)
            if old["sha256"] != new["sha256"]
        ]
        checks = {
            "authenticated_retained_A": True,
            "authenticated_identical_input": baseline["input"]["sha256"]
            == STEP0_SHA256,
            "canonical_nested_scalar_path_active": True,
            "candidate_interface_106_identity": candidate_audit["interface_identity"],
            "candidate_all_106_leaves_finite": candidate_manifest["leaf_count"] == 106
            and all(row["finite"] for row in new_rows),
            "candidate_callback_free": not candidate_audit["forbidden_targets"],
            "complete_leaf_structure_identity": structure_match
            and len(old_rows) == len(new_rows) == 106,
            "candidate_hlo_differs_from_11c": candidate_audit["stablehlo_sha256"]
            != BASELINE_HLO_SHA256,
            "candidate_output_differs_from_11c": bool(changed)
            and candidate_manifest["sha256"] != BASELINE_MANIFEST_SHA256,
            "source_oracle_green": all(oracle["checks"].values()),
        }
        proof = {
            "schema": "gpuwrf.v0234.nested-scalar-diffusion-complete-cpu-ab.v1",
            "environment": actual_env,
            "candidate_commit": CANDIDATE_COMMIT,
            "candidate_source_hashes": SOURCE_HASHES,
            "input": {
                "path": str(STEP0),
                "sha256": STEP0_SHA256,
                "start_step": 0,
                "leaf_count": 106,
            },
            "retained_A": {
                "candidate_commit": BASELINE_CANDIDATE_COMMIT,
                "proof_path": str(BASELINE),
                "proof_file_sha256": BASELINE_FILE_SHA256,
                "proof_sha256": BASELINE_PAYLOAD_SHA256,
                "stablehlo_sha256": BASELINE_HLO_SHA256,
                "manifest_sha256": BASELINE_MANIFEST_SHA256,
            },
            "candidate_B": {
                **candidate_audit,
                "manifest": candidate_manifest,
            },
            "load_authority": load_authority,
            "complete_output_comparison": {
                "leaf_count": len(new_rows),
                "changed_leaf_count": len(changed),
                "unchanged_leaf_count": len(new_rows) - len(changed),
                "changed_leaves": changed,
            },
            "causal_binding": {
                "direct_momentum_bundle": "bit-identical in focused CPU A/B",
                "scalar_source_oracle_path": str(ORACLE),
                "scalar_source_oracle_sha256": ORACLE_PAYLOAD_SHA256,
                "new_carry_leaves": 0,
                "new_loop_transfers": 0,
                "observer_or_callback_surface": 0,
            },
            "checks": checks,
            "verdict": (
                "NESTED_SCALAR_DIFFUSION_COMPLETE_CPU_AB_GREEN"
                if all(checks.values())
                else "NESTED_SCALAR_DIFFUSION_COMPLETE_CPU_AB_RED"
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
                    "candidate_hlo_sha256": candidate_audit["stablehlo_sha256"],
                    "candidate_manifest_sha256": candidate_manifest["sha256"],
                    "changed_leaf_count": len(changed),
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
