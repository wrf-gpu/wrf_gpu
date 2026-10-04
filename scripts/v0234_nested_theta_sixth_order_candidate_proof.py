"""Seal CPU/source admission for the nested theta sixth-order candidate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "nested-theta-sixth-order-candidate-proof.json"
CANDIDATE = "044783549697caf8c80d094f2b3e73f5ef340537"
CANDIDATE_TREE = "87b0242e0e7af3b0c6cdcf49bd25c18fe7c42f2a"
PARENT = "571e4a4208cd01941ec91c9bea4b8ae6ae4806c5"
PARTIAL_WIND = "2c13b73112d9877d603324d66b127ecad60bf7e3"
FILES = {
    "amendment14": (
        SPRINT / "contract-amendment-14.json",
        "1d354afa65b63ba17d7e1055edc0bc6c88c32bd53aae44ef4182c3eabed2f0ad",
        "67479e5faf493d496fa32340d02e783b420b14e84ae0ca97caf18c3b120cda98",
        False,
    ),
    "amendment15": (
        SPRINT / "contract-amendment-15.json",
        "0daaa4514a68cf8222d616f11a1ed561d25c4af1e146ab56559f3cf6f5826621",
        "36ed6b56097ed57392b2fbe45b9d63a68656f55015c9083be3f98f95e90f2519",
        False,
    ),
    "source_oracle": (
        SPRINT / "nested-theta-sixth-order-source-oracle.json",
        "7afa34dfd983113d0b294a281f3d597819f7469f31202e78f8fedfd340eb6b75",
        "150d38a525e2e35fc425abb527bbfddd27df02da922b9f7b96baafd06b1c09a6",
        True,
    ),
    "complete_cpu_ab": (
        SPRINT / "nested-theta-sixth-order-complete-cpu-ab-proof.json",
        "643595907584e845efdf34c1443b1486078e6bdebd3f39261e7c2396684538d3",
        "f89469af26eba6f96655d35401ce81db4493b023d7893db3cb9f1bd6bcd602b6",
        True,
    ),
    "actual_parent_auth": (
        SPRINT / "nested-theta-sixth-order-actual-parent-auth.json",
        "2d6d3701065e76f53720a08cccb91fe645faabfcef52e3a8aefddbd3936e86e6",
        "b3d6f06d9543bc7a9c027246aa9d30de54d381392ed8cc952c28e78e57cfc549",
        True,
    ),
}
MODEL_HASHES = {
    "src/gpuwrf/dynamics/explicit_diffusion.py": "8fc7726b2de017cf5ff4896aa85eed67e419ad9fd011b00addb50568279d2f61",
    "src/gpuwrf/runtime/operational_mode.py": "db43339801830ed4196d301239445ef8f4352b9cc3041de2e1d4dc618873afcc",
}
TEST_COMMAND = (
    "env GPUWRF_JAX_CACHE=0 GPUWRF_JAX_CACHE_LOCK=0 JAX_PLATFORMS=cpu "
    "JAX_ENABLE_X64=true PYTHONPATH=src:. pytest -q "
    "tests/test_v0234_nested_theta_sixth_order.py "
    "tests/test_v0234_nested_advection_degrade.py "
    "tests/test_v0234_nested_frozen_wrf_boundary_bundle.py "
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
    if expected_file == "PENDING" or expected_payload == "PENDING":
        raise RuntimeError(f"unsealed proof constant: {path}")
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

    amendment15 = proofs["amendment15"]
    source = proofs["source_oracle"]
    ab = proofs["complete_cpu_ab"]
    parent = proofs["actual_parent_auth"]
    false_checks = sorted(name for name, value in ab["checks"].items() if not value)
    parent_checks = parent.get("checks") or {}
    source_checks = source.get("checks") or {}
    actual_parent = parent["actual_parent"]
    wrapper_a = ab["retained_parent_A"]
    candidate_b = ab["candidate_B"]

    parent_runtime = _git("show", f"{PARENT}:src/gpuwrf/runtime/operational_mode.py")
    candidate_runtime = _git(
        "show", f"{CANDIDATE}:src/gpuwrf/runtime/operational_mode.py"
    )
    momentum_statements = (
        "u_t = u_t + mass_u * sixth_order_diffusion_tendency",
        "v_t = v_t + mass_v * sixth_order_diffusion_tendency",
        "w_t = w_t + mass_f * sixth_order_diffusion_tendency",
    )
    direct_momentum_unchanged = all(
        parent_runtime.count(statement) == candidate_runtime.count(statement) == 1
        for statement in momentum_statements
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
    tests_green = test.returncode == 0 and "98 passed" in test_output

    checks = {
        "candidate_tree_and_parent_authenticated": True,
        "model_unchanged_after_candidate_commit": True,
        "model_scope_exactly_two_source_files": True,
        "source_oracle_green": source.get("verdict")
        == "NESTED_THETA_SIXTH_ORDER_SOURCE_ORACLE_GREEN"
        and bool(source_checks)
        and all(source_checks.values()),
        "initial_cpu_ab_failed_closed_only_on_wrapper_hlo": false_checks
        == ["retained_parent_hlo_authenticated"],
        "actual_parent_auth_green": parent.get("verdict")
        == "NESTED_THETA_SIXTH_ORDER_ACTUAL_PARENT_AUTH_GREEN"
        and bool(parent_checks)
        and all(parent_checks.values()),
        "actual_parent_hlo_authenticated": actual_parent["stablehlo_sha256"]
        == "f2973b9806db3a660936c6e1593995c62d8d66f13936f316de28c067914f59f2",
        "actual_parent_output_equals_wrapper_A": actual_parent["manifest"]["sha256"]
        == wrapper_a["manifest"]["sha256"]
        == "3dffeb6e1fec12cdcafd6696e79906da6f18d2f23b35f5aa038f9412ea2683c6",
        "candidate_interface_106_identity": candidate_b["interface_identity"]
        and candidate_b["manifest"]["leaf_count"] == 106,
        "candidate_all_106_finite": all(
            leaf["finite"] for leaf in candidate_b["manifest"]["leaves"]
        ),
        "candidate_callback_free": not candidate_b["forbidden_targets"],
        "candidate_hlo_changed": candidate_b["stablehlo_sha256"]
        != actual_parent["stablehlo_sha256"],
        "candidate_complete_output_changed": candidate_b["manifest"]["sha256"]
        != actual_parent["manifest"]["sha256"],
        "direct_u_v_w_sixth_order_statements_unchanged": direct_momentum_unchanged,
        "no_carry_or_result_leaf_added": ab["causal_binding"][
            "new_carry_or_result_leaves"
        ]
        == 0,
        "no_loop_transfer_added": ab["causal_binding"]["new_loop_transfers"] == 0,
        "gate_semantics_amendment_fail_closed": amendment15.get("verdict")
        == "NESTED_THETA_SIXTH_ORDER_GATE_SEMANTICS_AMENDED_FAIL_CLOSED",
        "focused_cpu_static_suite_green": tests_green,
    }
    proof = {
        "schema": "gpuwrf.v0234.nested-theta-sixth-order-candidate.v1",
        "candidate": {
            "commit": CANDIDATE,
            "tree": CANDIDATE_TREE,
            "parent_falsified_commit": PARENT,
            "partial_wind_commit": PARTIAL_WIND,
            "model_files": MODEL_HASHES,
        },
        "proof_objects": rows,
        "complete_cpu_ab": {
            "actual_parent_hlo_sha256": actual_parent["stablehlo_sha256"],
            "actual_parent_manifest_sha256": actual_parent["manifest"]["sha256"],
            "candidate_hlo_sha256": candidate_b["stablehlo_sha256"],
            "candidate_manifest_sha256": candidate_b["manifest"]["sha256"],
            "leaf_count": candidate_b["manifest"]["leaf_count"],
            "all_finite": all(
                leaf["finite"] for leaf in candidate_b["manifest"]["leaves"]
            ),
            "forbidden_callback_targets": candidate_b["forbidden_targets"],
            "changed_fields": ab["complete_output_delta"]["changed_fields"],
            "theta_ring1_rms_delta_K": ab["spatial_projection"][
                "theta_ring1_rms"
            ],
            "momentum_output_identity": ab["momentum_output_identity"],
            "momentum_classification": "downstream theta/EOS/PGF coupling; direct U/V/W sixth-order source expressions unchanged",
        },
        "causal_scope": {
            "direct_changed_lane": "nested theta sixth-order t_tendf only",
            "direct_momentum_source_changed": False,
            "partial_wind_mechanism_retained": True,
            "carry_or_result_leaf_change": False,
            "observer_or_callback_added": False,
            "new_loop_host_device_transfer": False,
            "coefficient_sweep_or_speculative_toggle": False,
            "v10_link_claimed": False,
        },
        "focused_tests": {
            "command": TEST_COMMAND,
            "returncode": test.returncode,
            "output": test_output,
        },
        "checks": checks,
        "gpu_discriminator": {
            "gpu_commands_run_for_candidate": 0,
            "gpu_queries_run_for_candidate": 0,
            "full_forecasts_run_for_candidate": 0,
            "first_gate": "fresh canonical full-history process through d03 step 200 / 00:20",
            "green_action": "continue the identical compiled same-process run directly to exact 19/19/55 full18h identity and JPG",
            "red_action": "stop at Step200, preserve first-red evidence, and continue source-backed scalar RCA",
        },
        "verdict": (
            "NESTED_THETA_SIXTH_ORDER_GPU_STEP200_ADMITTED"
            if all(checks.values())
            else "NESTED_THETA_SIXTH_ORDER_GPU_STEP200_BLOCKED"
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
                "checks": checks,
                "test_output": test_output,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
