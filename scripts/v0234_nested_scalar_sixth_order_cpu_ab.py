"""Authenticated complete-carry CPU A/B for moist/scalar sixth-order cadence.

The ordinary 04478354 output is reused from its authenticated complete Step0
dispatch; it is not rerun.  This tool lowers, compiles, and dispatches the
candidate ordinary 106-leaf callable once, then compares the complete manifest,
interface, HLO policy, and direct source bundle to that retained A arm.
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
OUT = SPRINT / "nested-scalar-sixth-order-complete-cpu-ab-proof.json"
STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_theta_sixth_order_04478354_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
STEP0_SHA256 = "acd0d7ad147a3cf41f8302ae827d59294d5d249c8a007002646b878028ebe69d"
CANDIDATE_COMMIT = "18d97595c59ca01840081f11109780c291dfcae8"
PARENT_COMMIT = "835d5b6f07c357d5296716df7f2c99bcbb66a10a"
PARENT_MODEL_COMMIT = "044783549697caf8c80d094f2b3e73f5ef340537"
PARTIAL_WIND_COMMIT = "2c13b73112d9877d603324d66b127ecad60bf7e3"
SOURCE_ORACLE = SPRINT / "nested-scalar-sixth-order-source-oracle.json"
SOURCE_ORACLE_SHA256 = "1d0b0c792c1a2be40f477ce60a17835729e0799444ba55dfa57085d0013d556a"
SOURCE_ORACLE_PAYLOAD_SHA256 = "eb263213d60df012f5e4d000ba1a790a477c9a7dc8041abe82cd477c021a0d5d"
CONTRACT = SPRINT / "contract-amendment-16.json"
CONTRACT_SHA256 = "4275c7dea50a5903b4b6f6274fcd878b5ebdfdf1f7401a5b8f81e6131271b110"
CONTRACT_PAYLOAD_SHA256 = "9ef825d9fefaf10e40c5ea5acfe05517f3cb57432639de79dbf42cafbc1ab748"
RETAINED_A_PROOF = SPRINT / "nested-theta-sixth-order-complete-cpu-ab-proof.json"
RETAINED_A_PROOF_SHA256 = "643595907584e845efdf34c1443b1486078e6bdebd3f39261e7c2396684538d3"
RETAINED_A_PROOF_PAYLOAD_SHA256 = "f89469af26eba6f96655d35401ce81db4493b023d7893db3cb9f1bd6bcd602b6"
RETAINED_A_HLO_SHA256 = "47f92470edb808a35582c4763268d025f8994361f19fff909c803be7d2c0c8cf"
RETAINED_A_MANIFEST_SHA256 = "9605755f0217244ed51afdd6ac98fdb76c26d18b283dbb015d079c1011e76ece"
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
    return subprocess.check_output(("git", "-C", str(ROOT), *args), text=True).strip()


def _rms(np, value) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def main() -> int:
    actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"environment mismatch: {actual_env!r}")
    for path, expected, payload_hash in (
        (SOURCE_ORACLE, SOURCE_ORACLE_SHA256, SOURCE_ORACLE_PAYLOAD_SHA256),
        (CONTRACT, CONTRACT_SHA256, CONTRACT_PAYLOAD_SHA256),
        (RETAINED_A_PROOF, RETAINED_A_PROOF_SHA256, RETAINED_A_PROOF_PAYLOAD_SHA256),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"authenticated proof mismatch: {path}")
        if _canonical(json.loads(path.read_text())) != payload_hash:
            raise RuntimeError(f"canonical proof mismatch: {path}")
    if _sha256(STEP0) != STEP0_SHA256:
        raise RuntimeError("authenticated Step0 carry mismatch")
    if _git("status", "--porcelain"):
        raise RuntimeError("complete CPU A/B requires a clean committed candidate/tool")
    if _git("rev-parse", CANDIDATE_COMMIT) != CANDIDATE_COMMIT:
        raise RuntimeError("candidate commit mismatch")
    if _git("rev-parse", f"{CANDIDATE_COMMIT}^") != PARENT_COMMIT:
        raise RuntimeError("candidate parent mismatch")
    tooling_head = _git("rev-parse", "HEAD")
    if _git("diff", "--name-only", CANDIDATE_COMMIT, tooling_head, "--", "src/gpuwrf"):
        raise RuntimeError("model bytes changed after the candidate commit")
    model_delta = _git(
        "diff", "--name-only", PARENT_COMMIT, CANDIDATE_COMMIT, "--", "src/gpuwrf"
    ).splitlines()
    if model_delta != ["src/gpuwrf/runtime/operational_mode.py"]:
        raise RuntimeError(f"candidate model scope changed: {model_delta!r}")
    model_patch = _git(
        "diff", "--unified=0", PARENT_COMMIT, CANDIDATE_COMMIT,
        "--", "src/gpuwrf/runtime/operational_mode.py",
    )
    added_model_lines = "\n".join(
        line[1:]
        for line in model_patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )

    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
    import gpuwrf.contracts.state as state_contract

    state_contract._gpu_device = lambda: jax.devices("cpu")[0]

    import gpuwrf.runtime.operational_mode as runtime
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary
    from scripts.v0234_rk1_frozen_theta_diffusion_cpu_ab import _manifest, _run_arm

    retained_proof = json.loads(RETAINED_A_PROOF.read_text())
    retained_a = retained_proof["candidate_B"]
    retained_manifest = retained_a["manifest"]
    if retained_a["stablehlo_sha256"] != RETAINED_A_HLO_SHA256:
        raise RuntimeError("retained A HLO mismatch")
    if retained_manifest["sha256"] != RETAINED_A_MANIFEST_SHA256:
        raise RuntimeError("retained A manifest mismatch")
    if retained_proof["candidate_commit"] != PARENT_MODEL_COMMIT:
        raise RuntimeError("retained A source commit mismatch")

    scratch = Path(tempfile.mkdtemp(prefix="v0234-scalar-sixth-cpu-ab-"))
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
        if int(namelist.diff_6th_opt) != 2 or float(namelist.diff_6th_factor) != 0.12:
            raise RuntimeError("canonical diff6 options changed")
        if int(namelist.rk_order) != 3 or float(namelist.dt_s) != 6.0:
            raise RuntimeError("canonical RK cadence changed")
        clock = runtime.build_clock_base(namelist)

        direct_bundle = jax.device_get(
            runtime._nested_scalar_sixth_order_tendencies(carry.state, namelist)
        )
        direct_rows = []
        for name, value in zip(
            runtime.NESTED_BOUNDARY_SCALAR_SPECIES,
            direct_bundle,
            strict=True,
        ):
            array = np.asarray(value)
            direct_rows.append(
                {
                    "field": name,
                    "finite": bool(np.isfinite(array).all()),
                    "nonzero": int(np.count_nonzero(array)),
                    "rms": _rms(np, array),
                    "max_abs": float(np.max(np.abs(array))),
                }
            )

        print("SCALAR_SIXTH_CPU_AB candidate B lower/compile/dispatch", flush=True)
        candidate, candidate_audit = _run_arm(
            runtime, jax, jnp, carry, namelist, clock
        )
        candidate_manifest = _manifest(jax, np, candidate)
        structure_identity = all(
            left["path"] == right["path"]
            and left["shape"] == right["shape"]
            and left["dtype"] == right["dtype"]
            for left, right in zip(
                retained_manifest["leaves"],
                candidate_manifest["leaves"],
                strict=True,
            )
        )
        changed_paths = [
            left["path"]
            for left, right in zip(
                retained_manifest["leaves"],
                candidate_manifest["leaves"],
                strict=True,
            )
            if left["sha256"] != right["sha256"]
        ]
        forbidden_added_tokens = [
            token
            for token in (
                "device_get",
                "np.asarray",
                "pure_callback",
                "io_callback",
                "debug.callback",
                "jax.debug",
                "host_callback",
            )
            if token in added_model_lines
        ]
        forbidden_source_lane_edits = [
            token
            for token in (
                "u_t =",
                "v_t =",
                "w_t =",
                "th_t =",
                "frozen_diff6_theta_tendency=",
            )
            if token in added_model_lines
        ]
        checks = {
            "authenticated_step0": True,
            "authenticated_source_oracle": True,
            "authenticated_contract": True,
            "retained_A_complete_output_authenticated": retained_a["interface_identity"]
            and not retained_a["forbidden_targets"]
            and retained_manifest["leaf_count"] == 106
            and all(row["finite"] for row in retained_manifest["leaves"]),
            "candidate_interface_106_identity": candidate_audit["interface_identity"],
            "complete_leaf_structure_identity": structure_identity
            and candidate_manifest["leaf_count"] == retained_manifest["leaf_count"] == 106,
            "candidate_all_106_finite": all(
                row["finite"] for row in candidate_manifest["leaves"]
            ),
            "candidate_callback_free": not candidate_audit["forbidden_targets"],
            "candidate_hlo_changed": candidate_audit["stablehlo_sha256"]
            != RETAINED_A_HLO_SHA256,
            "candidate_complete_output_changed": candidate_manifest["sha256"]
            != RETAINED_A_MANIFEST_SHA256
            and bool(changed_paths),
            "model_scope_exactly_runtime": model_delta
            == ["src/gpuwrf/runtime/operational_mode.py"],
            "no_direct_dry_or_theta_source_edit": not forbidden_source_lane_edits,
            "no_observer_or_loop_transfer_source": not forbidden_added_tokens,
            "direct_bundle_all_finite": all(row["finite"] for row in direct_rows),
            "direct_bundle_only_qv_active_at_step0": direct_rows[0]["nonzero"] > 0
            and all(row["nonzero"] == 0 for row in direct_rows[1:]),
            "direct_qv_tendf_matches_oracle": abs(
                direct_rows[0]["rms"] - 0.02680830331493724
            ) < 1.0e-15
            and abs(direct_rows[0]["max_abs"] - 1.5460017035136657) < 1.0e-13,
        }
        proof = {
            "schema": "gpuwrf.v0234.nested-scalar-sixth-order-complete-cpu-ab.v1",
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
                "rk1_dt_s": 2.0,
                "species": list(runtime.NESTED_BOUNDARY_SCALAR_SPECIES),
            },
            "retained_parent_A": {
                "source_proof": str(RETAINED_A_PROOF.relative_to(ROOT)),
                "source_proof_file_sha256": RETAINED_A_PROOF_SHA256,
                "source_proof_payload_sha256": RETAINED_A_PROOF_PAYLOAD_SHA256,
                "stablehlo_sha256": RETAINED_A_HLO_SHA256,
                "manifest_sha256": RETAINED_A_MANIFEST_SHA256,
                "manifest": retained_manifest,
                "note": "The globally red retained proof failed only its pre-044 wrapper-HLO authentication; its 044 candidate-B arm is the authenticated ordinary parent reused here and was separately admitted by nested-theta-sixth-order-candidate-proof.json.",
            },
            "candidate_B": {**candidate_audit, "manifest": candidate_manifest},
            "complete_output": {
                "changed_leaf_count": len(changed_paths),
                "unchanged_leaf_count": 106 - len(changed_paths),
                "changed_paths": changed_paths,
            },
            "direct_source_bundle": direct_rows,
            "causal_binding": {
                "model_delta": model_delta,
                "direct_changed_lane": "RK1-frozen qv/qc/qr/qi/qs/qg/Ni/Nr sixth-order sc_tend only",
                "u_v_w_theta_source_expressions": "unchanged",
                "new_carry_or_result_leaves": 0,
                "new_loop_transfers": 0,
                "forbidden_added_tokens": forbidden_added_tokens,
                "forbidden_source_lane_edits": forbidden_source_lane_edits,
            },
            "checks": checks,
            "verdict": (
                "NESTED_SCALAR_SIXTH_ORDER_COMPLETE_CPU_AB_GREEN"
                if all(checks.values())
                else "NESTED_SCALAR_SIXTH_ORDER_COMPLETE_CPU_AB_RED"
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
                    "retained_A_hlo": RETAINED_A_HLO_SHA256,
                    "candidate_B_hlo": candidate_audit["stablehlo_sha256"],
                    "changed_leaf_count": len(changed_paths),
                    "failed_checks": [name for name, value in checks.items() if not value],
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
