"""Seal source, complete CPU A/B, and held Step200 admission for h_sca=5."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
SOURCE_OUT = SPRINT / "nested-h-sca-order-source-oracle.json"
AB_OUT = SPRINT / "nested-h-sca-order-complete-cpu-ab-proof.json"
CANDIDATE_OUT = SPRINT / "nested-h-sca-order-candidate-proof.json"
CANDIDATE = "395fb800df0bb619462db8e312f734ca0c538161"
CANDIDATE_TREE = "23b6cc942d4b0b0b32cf86bf88ebdec1c693fd4e"
PARENT = "dc3fecdd12d0e55d095c6f110d178d7a2a6c3ef5"
PARTIAL_WIND = "2c13b73112d9877d603324d66b127ecad60bf7e3"
MODEL_PATH = ROOT / "src/gpuwrf/integration/nested_pipeline.py"
MODEL_SHA256 = "48c6d4663bb7387265ea205a7f295228c3f737a9fd793c2dd55d27cdcf2edaad"
REGISTRY = Path("<USER_HOME>/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON")
REGISTRY_SHA256 = "6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a"
WRF_RHS = Path(
    "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F"
)
WRF_RHS_SHA256 = "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
NAMELIST = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/config/namelist.input"
)
NAMELIST_SHA256 = "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"
REVIEW = ROOT / ".agent/reviews/2026-06-11-v014-fable-acoustic-continuation.md"
REVIEW_SHA256 = "8695fd6f768d23fda36a5e92bb5311c26285c9de50a82b099de11daeab284d01"
RANKING = SPRINT / "nested-scalar-controls-cpu-ranking-proof.json"
RANKING_FILE_SHA256 = "38e9182de8a9146f95b645379025d9baa6837bbec4865fc21f6d3d44b752e0ce"
RANKING_PAYLOAD_SHA256 = "4cbd47c5d61aad44cf02e03e7861171a14a9cf97490407e5dc575501a682f432"
HLO = SPRINT / "nested-h-sca-order-lowered-hlo-audit.json"
HLO_FILE_SHA256 = "1eb9b82291305e9c45f01f52bd56193e1c9290fc89d4eed39f438fc16cf52fd0"
HLO_PAYLOAD_SHA256 = "78077e56e6b88b38d831031a6ab3c863fd766676268cf103eec120d56c408689"
CONTRACT = SPRINT / "contract-amendment-18.json"
CONTRACT_FILE_SHA256 = "661f3873a9a010be8c355a3b54c003b715cdfa7270bbe85a395442b881644024"
CONTRACT_PAYLOAD_SHA256 = "16cf5876c00490f903f18672241ebbabe5ae634f386910907627780f6e4b7c10"
STEP0_SHA256 = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"


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


def _read_proof(path: Path, file_hash: str, payload_hash: str) -> dict[str, Any]:
    if _sha256(path) != file_hash:
        raise RuntimeError(f"proof file mismatch: {path}")
    payload = json.loads(path.read_text())
    if _canonical(payload) != payload_hash:
        raise RuntimeError(f"proof payload mismatch: {path}")
    return payload


def _git(*args: str) -> str:
    return subprocess.check_output(("git", "-C", str(ROOT), *args), text=True).strip()


def _atomic(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    payload["proof_sha256"] = _canonical(payload)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return _sha256(path), payload["proof_sha256"]


def main() -> int:
    if _git("rev-parse", CANDIDATE) != CANDIDATE:
        raise RuntimeError("candidate commit mismatch")
    if _git("rev-parse", f"{CANDIDATE}^{{tree}}") != CANDIDATE_TREE:
        raise RuntimeError("candidate tree mismatch")
    if _git("rev-parse", f"{CANDIDATE}^") != PARENT:
        raise RuntimeError("candidate parent mismatch")
    if _git("diff", "--name-only", CANDIDATE, "--", "src/gpuwrf"):
        raise RuntimeError("model bytes changed after candidate commit")
    if _sha256(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("candidate model source mismatch")
    for path, expected in (
        (REGISTRY, REGISTRY_SHA256),
        (WRF_RHS, WRF_RHS_SHA256),
        (NAMELIST, NAMELIST_SHA256),
        (REVIEW, REVIEW_SHA256),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"source authority mismatch: {path}")

    ranking = _read_proof(RANKING, RANKING_FILE_SHA256, RANKING_PAYLOAD_SHA256)
    hlo = _read_proof(HLO, HLO_FILE_SHA256, HLO_PAYLOAD_SHA256)
    contract = _read_proof(CONTRACT, CONTRACT_FILE_SHA256, CONTRACT_PAYLOAD_SHA256)
    test_env = dict(os.environ)
    test_env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_ENABLE_X64": "true",
            "PYTHONPATH": "src:.",
        }
    )
    command = (
        "taskset", "-c", "12-15", "pytest", "-q",
        "tests/test_v0234_nested_h_sca_order.py",
        "tests/test_v014_rhs_ph_real_case.py",
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=test_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    test_output = completed.stdout.strip()

    registry_text = REGISTRY.read_text()
    namelist_text = NAMELIST.read_text().lower()
    wrf_lines = WRF_RHS.read_text().splitlines()
    wrf_branch = "\n".join(wrf_lines[1434:2068]) + "\n"
    source_checks = {
        "candidate_commit_tree_parent_authenticated": True,
        "candidate_model_scope_exactly_nested_loader": _git(
            "diff", "--name-only", PARENT, CANDIDATE, "--", "src/gpuwrf"
        ).splitlines() == ["src/gpuwrf/integration/nested_pipeline.py"],
        "registry_default_is_five": any(
            "h_sca_adv_order" in line and line.split()[5] == "5"
            for line in registry_text.splitlines()
            if "rconfig   integer     h_sca_adv_order" in line
        ),
        "canonical_namelist_omits_override": "h_sca_adv_order" not in namelist_text,
        "wrf_rhs_selects_config_value_and_order6_branch": (
            "advective_order = config_flags%h_sca_adv_order" in wrf_branch
            and "ELSE IF (advective_order <= 6) THEN" in wrf_branch
        ),
        "independent_numpy_and_loader_tests_green": completed.returncode == 0
        and "8 passed" in test_output,
        "actual_loader_all_domains_order5": all(
            row["h_sca_adv_order"] == 5
            for row in ranking["actual_loader_controls"].values()
        ),
        "prior_independent_review_authenticated": True,
    }
    source = {
        "schema": "gpuwrf.v0234.nested-h-sca-order-source-oracle.v1",
        "candidate_commit": CANDIDATE,
        "candidate_tree": CANDIDATE_TREE,
        "sources": {
            "registry": {"path": str(REGISTRY), "sha256": REGISTRY_SHA256},
            "wrf_rhs_ph": {
                "path": str(WRF_RHS),
                "sha256": WRF_RHS_SHA256,
                "relevant_lines_1_based": [1435, 2068],
                "relevant_slice_sha256": hashlib.sha256(wrf_branch.encode()).hexdigest(),
            },
            "canonical_namelist": {"path": str(NAMELIST), "sha256": NAMELIST_SHA256},
            "prior_independent_review": {"path": str(REVIEW), "sha256": REVIEW_SHA256},
            "candidate_model": {"path": str(MODEL_PATH), "sha256": MODEL_SHA256},
        },
        "equation_binding": {
            "wrf_rule": "Registry default 5 feeds config_flags%h_sca_adv_order; rhs_ph selects the map-factored order<=6 specified/nested branch with sixth-, fourth-, second-order side ownership and WRF x-gap columns.",
            "port_before": "live nesting omitted the case control and retained the idealized OperationalNamelist compatibility default 2, selecting periodic second order",
            "candidate": "thread the per-domain control with WRF default 5; rhs_ph arithmetic remains unchanged",
        },
        "focused_test": {
            "command": list(command),
            "returncode": completed.returncode,
            "output": test_output,
            "coverage": [
                "registry/default and canonical omission",
                "per-domain scalar/list/omitted resolution",
                "independent NumPy specified order<=6 map-factor parity",
                "WRF gap columns, side ownership, and open/rigid top semantics",
            ],
        },
        "checks": source_checks,
        "verdict": (
            "NESTED_H_SCA_ORDER_SOURCE_ORACLE_GREEN"
            if all(source_checks.values())
            else "NESTED_H_SCA_ORDER_SOURCE_ORACLE_RED"
        ),
    }
    source_file_hash, source_payload_hash = _atomic(SOURCE_OUT, source)

    parent_arm = ranking["arms"]["released_h2_opts00"]
    candidate_arm = ranking["arms"]["h_sca_only_h5_opts00"]
    comparison = ranking["comparisons_vs_released"]["h_sca_only_h5_opts00"]
    ab_checks = {
        "ranking_proof_authenticated": True,
        "authenticated_step0_leaf_count_106": ranking["inputs"]["step0"]["sha256"]
        == STEP0_SHA256
        and ranking["inputs"]["step0"]["leaf_count"] == 106,
        "parent_interface_identity": parent_arm["audit"]["interface_identity"],
        "candidate_interface_identity": candidate_arm["audit"]["interface_identity"],
        "all_complete_outputs_finite": ranking["checks"]["all_arms_finite"],
        "both_callback_free": not parent_arm["audit"]["forbidden_targets"]
        and not candidate_arm["audit"]["forbidden_targets"],
        "complete_output_changed": parent_arm["manifest_sha256"]
        != candidate_arm["manifest_sha256"],
        "candidate_hlo_matches_target_audit": candidate_arm["audit"]["stablehlo_sha256"]
        == hlo["lowering"]["extraction"]["stablehlo_sha256"],
        "target_aware_hlo_policy_green": hlo["verdict"]
        == "NESTED_H_SCA_ORDER_HLO_AUDIT_GREEN"
        and all(hlo["checks"].values()),
        "direct_effect_is_thm_dominant": comparison["diagnostic_T"][
            "THM_fixed_qv_ring1_rms_K"
        ] > 100.0 * comparison["diagnostic_T"]["qv_fixed_THM_ring1_rms_K"],
        "effect_reaches_east_and_gate_levels": comparison["diagnostic_T"][
            "east_ring1_rms_K"
        ] > 0.0
        and any(
            row["k"] in (26, 27, 28)
            for row in next(
                row for row in comparison["fields"] if row["field"] == "theta"
            )["ranked_ring1_levels"]
        ),
    }
    ab = {
        "schema": "gpuwrf.v0234.nested-h-sca-order-complete-cpu-ab.v1",
        "candidate_commit": CANDIDATE,
        "parent_commit": PARENT,
        "partial_wind_commit": PARTIAL_WIND,
        "source_ranking_proof": {
            "path": str(RANKING.relative_to(ROOT)),
            "file_sha256": RANKING_FILE_SHA256,
            "canonical_payload_sha256": RANKING_PAYLOAD_SHA256,
        },
        "input": ranking["inputs"]["step0"],
        "parent_A": parent_arm,
        "candidate_B": candidate_arm,
        "complete_output_delta": comparison,
        "hlo_audit": {
            "path": str(HLO.relative_to(ROOT)),
            "file_sha256": HLO_FILE_SHA256,
            "canonical_payload_sha256": HLO_PAYLOAD_SHA256,
            "custom_call_targets": hlo["lowering"]["extraction"]["targets"],
            "forbidden_text_tokens": hlo["lowering"]["forbidden_text_tokens"],
            "unknown_custom_targets": hlo["lowering"]["unknown_custom_targets"],
        },
        "causal_scope": {
            "direct_source_change": "static h_sca_adv_order selection only",
            "rhs_ph_equation_bytes_changed": False,
            "new_carry_or_output_leaves": 0,
            "new_loop_transfers": 0,
            "trajectory_claim": "none; Step200 remains decisive",
        },
        "checks": ab_checks,
        "verdict": (
            "NESTED_H_SCA_ORDER_COMPLETE_CPU_AB_GREEN"
            if all(ab_checks.values())
            else "NESTED_H_SCA_ORDER_COMPLETE_CPU_AB_RED"
        ),
    }
    ab_file_hash, ab_payload_hash = _atomic(AB_OUT, ab)

    candidate_checks = {
        "contract_authenticated": contract["verdict"]
        == "NESTED_H_SCA_ORDER_CANDIDATE_SEALED_GPU_HELD",
        "source_oracle_green": source["verdict"]
        == "NESTED_H_SCA_ORDER_SOURCE_ORACLE_GREEN"
        and all(source_checks.values()),
        "complete_cpu_ab_green": ab["verdict"]
        == "NESTED_H_SCA_ORDER_COMPLETE_CPU_AB_GREEN"
        and all(ab_checks.values()),
        "model_immutable_since_candidate": not bool(
            _git("diff", "--name-only", CANDIDATE, "--", "src/gpuwrf")
        ),
        "partial_wind_mechanism_retained": contract["post_release_discriminator"][
            "required"
        ][1].startswith("U ring1 remains"),
        "scalar_opts_not_stacked": contract["deferred_general_gap"]["incident_status"]
        .startswith("real WRF-fidelity gap"),
        "gpu_reservation_hold_bound": contract["gpu_reservation_hold"][
            "gpu_commands_for_candidate"
        ] == 0,
        "v10_separate": contract["v10_policy"].startswith("V10 remains a separate"),
    }
    candidate = {
        "schema": "gpuwrf.v0234.nested-h-sca-order-candidate.v1",
        "candidate": {
            "commit": CANDIDATE,
            "tree": CANDIDATE_TREE,
            "parent": PARENT,
            "model_source_sha256": MODEL_SHA256,
        },
        "contract": {
            "path": str(CONTRACT.relative_to(ROOT)),
            "file_sha256": CONTRACT_FILE_SHA256,
            "canonical_payload_sha256": CONTRACT_PAYLOAD_SHA256,
        },
        "source_oracle": {
            "path": str(SOURCE_OUT.relative_to(ROOT)),
            "file_sha256": source_file_hash,
            "canonical_payload_sha256": source_payload_hash,
        },
        "complete_cpu_ab": {
            "path": str(AB_OUT.relative_to(ROOT)),
            "file_sha256": ab_file_hash,
            "canonical_payload_sha256": ab_payload_hash,
            "leaf_count": 106,
            "all_finite": True,
            "candidate_manifest_sha256": candidate_arm["manifest_sha256"],
            "stablehlo_sha256": candidate_arm["audit"]["stablehlo_sha256"],
        },
        "causal_verdict": {
            "primary_lane": "THM/PH",
            "east_ring1_reached": True,
            "gate_levels_reached": [26, 27, 28],
            "full_forecast_claim": False,
            "v10_link_claimed": False,
        },
        "gpu_discriminator": {
            "state": "HELD_FOR_EXPLICIT_RELEASE_GPU_FROM_0:2",
            "gpu_commands_run_for_candidate": 0,
            "full_forecasts_run_for_candidate": 0,
            "earliest_gate": "d03 Step200 / 00:20",
            "launch_ready_only": True,
        },
        "checks": candidate_checks,
        "verdict": (
            "NESTED_H_SCA_ORDER_STEP200_LAUNCH_READY_HELD"
            if all(candidate_checks.values())
            else "NESTED_H_SCA_ORDER_NO_LAUNCH"
        ),
    }
    candidate_file_hash, candidate_payload_hash = _atomic(CANDIDATE_OUT, candidate)
    print(
        json.dumps(
            {
                "source_verdict": source["verdict"],
                "source_proof_sha256": source_payload_hash,
                "ab_verdict": ab["verdict"],
                "ab_proof_sha256": ab_payload_hash,
                "candidate_verdict": candidate["verdict"],
                "candidate_file_sha256": candidate_file_hash,
                "candidate_proof_sha256": candidate_payload_hash,
                "focused_test": test_output,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if all(candidate_checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
