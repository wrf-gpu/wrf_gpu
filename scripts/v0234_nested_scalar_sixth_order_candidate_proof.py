"""Seal source/CPU admission for the nested moist/scalar diff6 candidate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-scalar-sixth-order-candidate-proof.json"
CANDIDATE = "18d97595c59ca01840081f11109780c291dfcae8"
CANDIDATE_TREE = "ac570ee89be7d18dcc6eccfc5bcfa45c32d30dd6"
PARENT = "835d5b6f07c357d5296716df7f2c99bcbb66a10a"
PARENT_MODEL = "044783549697caf8c80d094f2b3e73f5ef340537"
PARTIAL_WIND = "2c13b73112d9877d603324d66b127ecad60bf7e3"
FILES = {
    "amendment16": (
        SPRINT / "contract-amendment-16.json",
        "4275c7dea50a5903b4b6f6274fcd878b5ebdfdf1f7401a5b8f81e6131271b110",
        "9ef825d9fefaf10e40c5ea5acfe05517f3cb57432639de79dbf42cafbc1ab748",
        False,
    ),
    "source_oracle": (
        SPRINT / "nested-scalar-sixth-order-source-oracle.json",
        "1d0b0c792c1a2be40f477ce60a17835729e0799444ba55dfa57085d0013d556a",
        "eb263213d60df012f5e4d000ba1a790a477c9a7dc8041abe82cd477c021a0d5d",
        True,
    ),
    "complete_cpu_ab": (
        SPRINT / "nested-scalar-sixth-order-complete-cpu-ab-proof.json",
        "6307cf3d303055886568cf4edb6910dc71eb0f5502922a92aaebc0f5bc6275d6",
        "a21248ad8d1e6aa139db05b3028ee27a4069da4312619d70b5586a62b7d0b1d1",
        True,
    ),
}
MODEL_HASHES = {
    "src/gpuwrf/runtime/operational_mode.py": (
        "fdda05e0ceb977c9a7faafd026beb3f83afea734844dd448d8546cccbf8c2b5e"
    ),
}
TEST_COMMAND = (
    "env GPUWRF_JAX_CACHE=0 GPUWRF_JAX_CACHE_LOCK=0 JAX_PLATFORMS=cpu "
    "JAX_ENABLE_X64=true PYTHONPATH=src:. pytest -q "
    "tests/test_v0234_nested_scalar_sixth_order.py "
    "tests/test_v0234_nested_theta_sixth_order.py "
    "tests/test_v0234_nested_advection_degrade.py "
    "tests/test_v0234_nested_frozen_wrf_boundary_bundle.py "
    "tests/test_v0234_nested_boundary_critic_repair.py "
    "tests/dynamics/test_pd_rk3_operational.py "
    "tests/test_namelist_check.py tests/test_namelist_recognition_breadth.py"
)


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


def _read_auth(
    path: Path, expected_file: str, expected_payload: str, embedded: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    file_hash = _sha256(path)
    payload = json.loads(path.read_text())
    canonical = _canonical(payload) if embedded else hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if file_hash != expected_file or canonical != expected_payload:
        raise RuntimeError(
            f"proof mismatch {path}: file={file_hash} canonical={canonical}"
        )
    if embedded and payload.get("proof_sha256") != expected_payload:
        raise RuntimeError(f"embedded proof mismatch: {path}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": file_hash,
        "canonical_payload_sha256": canonical,
    }


def main() -> int:
    if _git("rev-parse", f"{CANDIDATE}^{{tree}}") != CANDIDATE_TREE:
        raise RuntimeError("candidate tree mismatch")
    if _git("rev-parse", f"{CANDIDATE}^") != PARENT:
        raise RuntimeError("candidate parent mismatch")
    if _git("diff", "--name-only", CANDIDATE, "--", "src/gpuwrf"):
        raise RuntimeError("model bytes changed after candidate commit")
    model_delta = _git(
        "diff", "--name-only", PARENT, CANDIDATE, "--", "src/gpuwrf"
    ).splitlines()
    if model_delta != list(MODEL_HASHES):
        raise RuntimeError(f"candidate model scope changed: {model_delta!r}")
    for relative, expected in MODEL_HASHES.items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"model source hash mismatch: {relative}")

    proofs = {}
    rows = {}
    for name, (path, file_hash, payload_hash, embedded) in FILES.items():
        proofs[name], rows[name] = _read_auth(
            path, file_hash, payload_hash, embedded
        )
    amendment = proofs["amendment16"]
    source = proofs["source_oracle"]
    ab = proofs["complete_cpu_ab"]
    source_checks = source.get("checks") or {}
    ab_checks = ab.get("checks") or {}

    parent_runtime = _git("show", f"{PARENT}:src/gpuwrf/runtime/operational_mode.py")
    candidate_runtime = _git(
        "show", f"{CANDIDATE}:src/gpuwrf/runtime/operational_mode.py"
    )
    dry_source_statements = (
        "u_t = u_t + mass_u * sixth_order_diffusion_tendency",
        "v_t = v_t + mass_v * sixth_order_diffusion_tendency",
        "w_t = w_t + mass_f * sixth_order_diffusion_tendency",
        "frozen_diff6_theta_tendency=rk1_forward_diff6_theta",
    )
    dry_theta_source_unchanged = all(
        parent_runtime.count(statement) == candidate_runtime.count(statement) == 1
        for statement in dry_source_statements
    )

    test = subprocess.run(
        TEST_COMMAND,
        cwd=ROOT,
        shell=True,
        executable="/bin/bash",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    test_output = test.stdout.strip()
    tests_green = test.returncode == 0 and "141 passed" in test_output
    candidate_b = ab["candidate_B"]
    direct = ab["direct_source_bundle"]
    checks = {
        "candidate_tree_parent_authenticated": True,
        "model_unchanged_after_candidate_commit": True,
        "model_scope_exactly_runtime": model_delta
        == ["src/gpuwrf/runtime/operational_mode.py"],
        "source_oracle_green": source.get("verdict")
        == "NESTED_SCALAR_SIXTH_ORDER_SOURCE_ORACLE_GREEN"
        and bool(source_checks)
        and all(source_checks.values()),
        "complete_cpu_ab_green": ab.get("verdict")
        == "NESTED_SCALAR_SIXTH_ORDER_COMPLETE_CPU_AB_GREEN"
        and bool(ab_checks)
        and all(ab_checks.values()),
        "candidate_interface_106_identity": candidate_b["interface_identity"]
        and candidate_b["manifest"]["leaf_count"] == 106,
        "candidate_all_106_finite": all(
            leaf["finite"] for leaf in candidate_b["manifest"]["leaves"]
        ),
        "candidate_callback_free": not candidate_b["forbidden_targets"],
        "candidate_complete_output_changed": ab["complete_output"][
            "changed_leaf_count"
        ]
        > 0,
        "direct_bundle_only_qv_active_step0": direct[0]["nonzero"] > 0
        and all(row["nonzero"] == 0 for row in direct[1:]),
        "dry_momentum_theta_sources_unchanged": dry_theta_source_unchanged,
        "no_carry_or_result_leaf_added": ab["causal_binding"][
            "new_carry_or_result_leaves"
        ]
        == 0,
        "no_loop_transfer_or_observer_added": ab["causal_binding"][
            "new_loop_transfers"
        ]
        == 0
        and not ab["causal_binding"]["forbidden_added_tokens"],
        "contract_fail_closed": amendment.get("verdict")
        == "NESTED_RK1_MOIST_SCALAR_SIXTH_ORDER_AMENDED_FAIL_CLOSED",
        "focused_cpu_static_suite_green": tests_green,
    }
    proof = {
        "schema": "gpuwrf.v0234.nested-scalar-sixth-order-candidate.v1",
        "candidate": {
            "commit": CANDIDATE,
            "tree": CANDIDATE_TREE,
            "parent_commit": PARENT,
            "parent_model_commit": PARENT_MODEL,
            "partial_wind_commit": PARTIAL_WIND,
            "model_files": MODEL_HASHES,
        },
        "proof_objects": rows,
        "complete_cpu_ab": {
            "retained_parent_hlo_sha256": ab["retained_parent_A"][
                "stablehlo_sha256"
            ],
            "retained_parent_manifest_sha256": ab["retained_parent_A"][
                "manifest_sha256"
            ],
            "candidate_hlo_sha256": candidate_b["stablehlo_sha256"],
            "candidate_manifest_sha256": candidate_b["manifest"]["sha256"],
            "leaf_count": candidate_b["manifest"]["leaf_count"],
            "changed_leaf_count": ab["complete_output"]["changed_leaf_count"],
            "all_finite": all(
                leaf["finite"] for leaf in candidate_b["manifest"]["leaves"]
            ),
            "forbidden_callback_targets": candidate_b["forbidden_targets"],
        },
        "causal_scope": {
            "direct_changed_lane": "RK1-frozen qv/qc/qr/qi/qs/qg/Ni/Nr sixth-order sc_tend only",
            "direct_dry_momentum_or_theta_source_changed": False,
            "partial_wind_mechanism_retained": True,
            "carry_or_result_leaf_change": False,
            "observer_or_callback_added": False,
            "new_loop_host_device_transfer": False,
            "coefficient_sweep_or_speculative_toggle": False,
            "v10_link_claimed": False,
        },
        "prediction": {
            "source_direction_cpu_T_full_rmse_K": [
                0.10352821662387812,
                0.10342749315073793,
            ],
            "source_direction_retry20_T_full_rmse_K": [
                0.061554676362825626,
                0.06154580790550593,
            ],
            "trajectory_claim": "none; exact d03 Step200 ring1 gate is decisive",
        },
        "focused_tests": {
            "command": TEST_COMMAND,
            "returncode": test.returncode,
            "output": test_output,
        },
        "checks": checks,
        "gpu_discriminator": {
            "gpu_queries_run_for_candidate": 0,
            "gpu_commands_run_for_candidate": 0,
            "full_forecasts_run_for_candidate": 0,
            "first_gate": "fresh canonical full-history process through d03 step 200 / 00:20",
            "green_action": "continue the identical compiled same process directly to exact 19/19/55 full18h identity and JPG",
            "red_action": "stop at Step200, preserve first-red evidence, and continue source-backed scalar RCA",
        },
        "verdict": (
            "NESTED_SCALAR_SIXTH_ORDER_GPU_STEP200_ADMITTED"
            if all(checks.values())
            else "NESTED_SCALAR_SIXTH_ORDER_GPU_STEP200_BLOCKED"
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
                "failed_checks": [name for name, value in checks.items() if not value],
                "test_output": test_output,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
