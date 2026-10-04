"""Seal the CPU-only independent GPT review proof for v0234 S1 diff6."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REVIEW = REPO / ".agent/sprints/2026-07-18-v0234-s1-diff6-gpt-review-arm-prep"
OUTPUT = REVIEW / "proof.json"
ZERO_SELF = "0" * 64

CANDIDATE = "2e5fa008563cfe56c3c766b21352effc7179c4bf"
FIX = "e65ce784bec85ea4f000e94ba572420c595ecb68"
PARENT = "5193db05cd67cfc79292b0206ca43b1d3c68b2e8"
PRODUCTION_HASHES = {
    "src/gpuwrf/dynamics/explicit_diffusion.py": (
        "58e677d4e7748057f4b3546380296f01e99a58fb161b76971343a46aa9792b8c"
    ),
    "src/gpuwrf/runtime/operational_mode.py": (
        "9480c992b46df6a76f0f492baf8098d7945507d1c9e6c44c002bfbd0a4eceaff"
    ),
}
WRF_ROOT = Path("<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF")
WRF_HASHES = {
    "Registry/Registry.EM_COMMON": (
        "6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a"
    ),
    "dyn_em/module_big_step_utilities_em.F": (
        "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
    ),
    "dyn_em/module_em.F": (
        "11105cbf8255f30ca6a44cd7429a92cedce1fb91db6ce90fd7217002a72fb7fa"
    ),
    "dyn_em/solve_em.F": (
        "680719162a2b9745b4bd12683f512d936a3da02072833cc88e62ae1366f723a9"
    ),
}

RUNTIME_PROOF = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_dycore_suboperator_ladder1/"
    "dycore-suboperator-ladder-terminal-proof.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(REPO), *args), check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def authenticated_json(
    path: Path, expected_self: str, *, field: str = "proof_sha256",
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text())
    embedded = payload.get(field)
    observed = canonical({key: value for key, value in payload.items() if key != field})
    if embedded != expected_self or observed != expected_self:
        raise RuntimeError(f"self-hash mismatch: {path}")
    return payload, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "self_hash_field": field,
        "canonical_self_hash": observed,
        "authenticated": True,
    }


def plain_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }


def main() -> None:
    candidate_sprint = REPO / ".agent/sprints/2026-07-18-v0234-s1-dyn-attribution-fable5"
    candidate_objects = {
        path.name: plain_file(path)
        for path in sorted(candidate_sprint.iterdir())
        if path.is_file()
    }
    if set(candidate_objects) != {
        "CONTRACT.md", "amendment-01-partial-gate-disposition.md", "proof.json",
        "s1-attribution-analysis.json", "worker-report.md",
    }:
        raise RuntimeError("candidate sprint inventory changed")
    candidate_analysis, candidate_analysis_row = authenticated_json(
        candidate_sprint / "s1-attribution-analysis.json",
        "12025ab254f560d4746d25211a5f7b3db45f65c8b8af0bd72b1c4d3cc4066190",
    )
    _candidate_proof, candidate_proof_row = authenticated_json(
        candidate_sprint / "proof.json",
        "e33c78d8c0127e7ea1af40d127d233d4be3ea233100dd31eda5bda05d70194a1",
    )

    preceding_specs = {
        "kimi_internal_split_analysis": (
            REPO / ".agent/sprints/2026-07-18-v0234-dycore-internal-split-kimi/"
            "internal-split-analysis.json",
            "65421e6c7ef823c821e43e3d28a460df56656e4b30e8d638dd1ec88895788996",
        ),
        "kimi_internal_split_proof": (
            REPO / ".agent/sprints/2026-07-18-v0234-dycore-internal-split-kimi/proof.json",
            "c4f1d99074360be501ff898199c316da65fce84fdb78d140c7a9d71b75080226",
        ),
        "gpt_terminal_cpu_proof": (
            REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-continuation/"
            "proof.json",
            "c7c05d55d09fdd095f0a3538b097ba527c3b6228a19cb32880db0d1b339f637a",
        ),
        "gpt_cpu_ladder_analysis": (
            REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-continuation/"
            "cpu-ladder-analysis.json",
            "79d0e5998271923aaec20a9ea11d8e23dd1c02fca504acf8963569ac4eb9814e",
        ),
        "gpt_gpu_arm_proof": (
            REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-gpu-arm/"
            "proof.json",
            "f26bdd65099d722d44644d033cf0734ec64a76ee8f5b6ff116ed17724ee55b4c",
        ),
        "gpt_gpu_arm_analysis": (
            REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-gpu-arm/"
            "gpu-cpu-ladder-analysis.json",
            "7c5e2df4d5a4d9181047c0d24d688e46ddc2b0c768ac46951d0a0b875a2772da",
        ),
        "gpt_runtime_proof": (
            RUNTIME_PROOF,
            "f0d0b526ed4769f83b8dd2403343186197c5f439f8ad838ee9733e5bfd170e4d",
        ),
        "conditioning_ensemble_analysis": (
            REPO / ".agent/sprints/2026-07-18-v0234-conditioning-ensemble-gpt/"
            "ensemble-analysis.json",
            "b0c65ce2451bc3ee6cc393e4d783c91dde3b140dbba08a70a159c67186a5a7f7",
        ),
        "conditioning_proof": (
            REPO / ".agent/sprints/2026-07-18-v0234-conditioning-ensemble-gpt/proof.json",
            "4ed61e303734b8f220f255b35b2ff4348b1388e2e225135c1601ec7fc2b67309",
        ),
        "conditioning_retained_manifest": (
            REPO / ".agent/sprints/2026-07-18-v0234-conditioning-ensemble-gpt/"
            "retained-evidence-manifest.json",
            "a66455da98f7279d793c09fddcdbc5d59546fb84a3b897ab2b4cc8d115e5bbaf",
        ),
    }
    preceding: dict[str, Any] = {}
    runtime_payload: dict[str, Any] | None = None
    for name, (path, expected) in preceding_specs.items():
        payload, row = authenticated_json(path, expected)
        preceding[name] = row
        if name == "gpt_runtime_proof":
            runtime_payload = payload
    assert runtime_payload is not None
    savepoint_rows = runtime_payload["savepoint_manifest"]["rows"]
    if canonical(savepoint_rows) != (
        "bc974c1848cb625a1dc3bc4edd879dbf6fb4a6a9a3cbc195b25af2865f5e9503"
    ):
        raise RuntimeError("frozen savepoint row manifest changed")
    for row in savepoint_rows:
        if sha256_file(Path(row["path"])) != row["file_sha256"]:
            raise RuntimeError(f"frozen savepoint changed: {row['path']}")

    wrf_sources = {}
    for relative, expected in WRF_HASHES.items():
        path = WRF_ROOT / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"pristine WRF source changed: {relative}")
        wrf_sources[relative] = {"path": str(path), "sha256": observed}

    production_sources = {}
    for relative, expected in PRODUCTION_HASHES.items():
        observed = sha256_file(REPO / relative)
        if observed != expected:
            raise RuntimeError(f"candidate production source changed: {relative}")
        production_sources[relative] = observed
    model_delta = git("diff", "--name-only", PARENT, "HEAD", "--", "src/gpuwrf").splitlines()
    if model_delta != sorted(PRODUCTION_HASHES):
        raise RuntimeError(f"production delta changed: {model_delta}")
    reviewer_model_delta = git(
        "diff", "--name-only", CANDIDATE, "HEAD", "--", "src/gpuwrf",
    ).splitlines()
    if reviewer_model_delta:
        raise RuntimeError(f"reviewer changed production: {reviewer_model_delta}")

    arm = REPO / ".agent/sprints/2026-07-18-v0234-s1-diff6-gpt-gpu-arm"
    arm_cpu, arm_cpu_row = authenticated_json(
        arm / "s1-diff6-runner-cpu-proof.json",
        "671bdf8b690c3e5155b371524584e17a4c826b76f09e1c2acb469062744c9ca5",
    )
    if not (
        arm_cpu["static_audit"]["passed"]
        and arm_cpu["exact_launch_audit"]["passed"]
        and arm_cpu["gpu_commands_run"] == 0
        and arm_cpu["gpu_queries_run"] == 0
        and arm_cpu["jax_imported"] is False
    ):
        raise RuntimeError("arm CPU audit is not green/CPU-only")

    step1 = candidate_analysis["steps"]["1"]["attribution"]
    payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.s1-diff6-gpt-independent-review.v1",
        "terminal_verdict": "GPT_REVIEW_ACCEPT_READY_FOR_AUTH",
        "subdisposition": None,
        "review_owner": "GPT-5.6-sol-xhigh",
        "review_head": git("rev-parse", "HEAD"),
        "candidate": {
            "terminal_head": CANDIDATE,
            "fix_commit": FIX,
            "parent": PARENT,
            "candidate_is_ancestor": subprocess.run(
                ("git", "-C", str(REPO), "merge-base", "--is-ancestor",
                 CANDIDATE, "HEAD"), check=False,
            ).returncode == 0,
        },
        "review_contract": plain_file(REVIEW / "CONTRACT.md") | {
            "commit": "896a81b1",
            "committed_before_review_edits": True,
        },
        "candidate_sprint_inventory": candidate_objects,
        "candidate_json_authentication": {
            "analysis": candidate_analysis_row,
            "proof": candidate_proof_row,
        },
        "preceding_proof_authentication": preceding,
        "retained_savepoints": {
            "manifest_rows": len(savepoint_rows),
            "rows_canonical_sha256": canonical(savepoint_rows),
            "files_rehashed": len(savepoint_rows),
            "all_match": True,
        },
        "attribution_reproduction": {
            "rerun_payload_exact_match": True,
            "step1_nonspec_explained_fraction": step1["nonspec"]["explained_fraction"],
            "step1_nonspec_correlation": step1["nonspec"]["corr_R_P"],
            "predicted_postfix_rmse": {
                "nonspec": step1["nonspec"]["rmse_R_minus_P"],
                "relax_rows_1_4": step1["relax_rows_1_4"]["rmse_R_minus_P"],
                "interior_ge_5": step1["interior_ge_5"]["rmse_R_minus_P"],
            },
            "terminal_class": candidate_analysis["terminal_class"],
            "compound_confirmed_gate_fired": False,
            "scope_supported": "exactly one causal 9-step GPU validation arm",
            "scope_not_supported": "S1 closure, second arm, V10, Ni, 18h, performance, release",
        },
        "wrf_v471_trace": {
            "root": str(WRF_ROOT),
            "sources": wrf_sources,
            "production_delta_files": model_delta,
            "production_source_sha256": production_sources,
            "reviewer_production_delta_files": reviewer_model_delta,
            "checks": {
                "u_v_w_ownership_rectangles_and_staggering": True,
                "adjacent_mass_placement": True,
                "c1h_c2h_c1f_c2f_level_placement": True,
                "direction_specific_map_factors_and_dry_fold": True,
                "monotonic_limiter_product_le_zero": True,
                "coefficient_factor_times_0_015625_over_2dt": True,
                "slopeopt_default_zero": True,
                "rk1_time_t_freeze_and_three_stage_reuse": True,
                "w_vertical_stagger_top_bottom_excluded": True,
            },
        },
        "runtime_integration": {
            "one_uvw_bundle_hoisted_before_rk_stage_scan": True,
            "pytree_shapes_match_u_v_w": True,
            "diff6_off_static_path_bit_exact_and_nan_bundle_ignored": True,
            "capture_observer_free_and_separate_static_program": True,
            "host_device_transfer_inside_production_timestep_loop": False,
            "candidate_profile_repin_complete": False,
            "candidate_profile_issue": (
                "legacy source-authority/static audit still required a one-file delta and "
                "the obsolete pre-fix HLO identity"
            ),
            "superseded_by_new_arm_profile": True,
        },
        "review_gates": {
            "R1_evidence_integrity": "PASS",
            "R2_wrf_source_fidelity": "PASS",
            "R3_runtime_integration": "PASS",
            "R4_cpu_validation": "PASS",
            "R5_one_arm_necessity_and_sufficiency": "PASS",
        },
        "arm_preparation": {
            "state": "HOLD_PENDING_CHIEF_0:3_AUTHORIZATION",
            "contract": plain_file(arm / "CONTRACT.md") | {"commit": "116db7fb"},
            "namespace": (
                "nested_stage_omega_transport_470e6111_s1_diff6_fix_validation_gpt1"
            ),
            "launcher": plain_file(arm / "s1-diff6-exact-launch-command.sh"),
            "cpu_audit": arm_cpu_row,
            "thresholds_le": {
                "nonspec": 4.0,
                "relax_rows_1_4": 9.5,
                "interior_ge_5": 1.4,
            },
            "fresh_ordinary_hlo_before_policy_compile_dispatch": True,
            "stop_first_red_and_withhold_step2": True,
            "canonical_lock_v2": {
                "commit": "8152309aff1e85e1052d44d549a5a5409e710bdd",
                "wrapper_sha256": (
                    "c75b3a4eda17e94df921986e1c15fcc71e182e01d51b6fe877e4c52533077e1a"
                ),
                "verifier_sha256": (
                    "ec911f565ee9d2c58dee81d8d1e17411ed677be500a57e73ded5f793e2c4187d"
                ),
                "timeout": 0,
                "label": "v0234-s1-diff6-validation",
                "intent": "production-preemptible",
                "immediate_release_receipt": True,
            },
            "gpu_arm_launched": False,
        },
        "tests": [
            {
                "scope": "arm dry audit + candidate diff6 + Kimi ladder",
                "result": "43 passed",
                "command": (
                    "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu taskset -c 12-15 "
                    "python -m pytest -q tests/test_v0234_s1_diff6_gpu_arm.py "
                    "tests/test_v0234_s1_dyn_attribution_fable5.py "
                    "tests/test_v0234_dycore_suboperator_kimi.py"
                ),
            },
            {
                "scope": "boundary/source/deformation broad regression",
                "result": "85 passed",
                "command": (
                    "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu taskset -c 12-15 "
                    "python -m pytest -q tests/test_v0234_nested_frozen_wrf_boundary_bundle.py "
                    "tests/test_v0234_nested_advection_degrade.py "
                    "tests/test_v0234_nested_boundary_critic_repair.py "
                    "tests/test_v0234_nested_boundary_science_repair.py "
                    "tests/test_v0234_nested_scalar_diffusion_source_repair.py "
                    "tests/test_v0234_nested_t_source_cadence.py "
                    "tests/test_v0234_theta_unlimited_source_repair.py "
                    "tests/dynamics/test_deformation_momentum_diffusion.py"
                ),
            },
            {
                "scope": "preceding ladder/stage/conditioning regression",
                "result": "34 passed",
                "command": (
                    "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu taskset -c 12-15 "
                    "python -m pytest -q tests/test_v0234_first_interval_momentum.py "
                    "tests/test_v0234_dycore_suboperator_gpt_gpu_merge.py "
                    "tests/test_v0234_stage_omega_transport_ownership.py "
                    "tests/test_v0234_nested_h_sca_order.py "
                    "tests/test_v0234_conditioning_ensemble.py"
                ),
            },
            {
                "scope": "Kimi CPU-only import-hygiene discriminator",
                "result": "11 passed in isolated process",
                "command": (
                    "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu taskset -c 12-15 "
                    "python -m pytest -q tests/test_v0234_dycore_internal_split_kimi.py"
                ),
            },
            {
                "scope": "accepted theta/scalar sixth-order regression",
                "result": "22 passed",
                "command": (
                    "CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu taskset -c 12-15 "
                    "python -m pytest -q tests/test_v0234_rk1_frozen_theta_diffusion.py "
                    "tests/test_v0234_nested_theta_sixth_order.py "
                    "tests/test_v0234_nested_scalar_sixth_order.py"
                ),
            },
        ],
        "test_harness_note": {
            "co_collected_import_hygiene_failure_observed": True,
            "reason": (
                "pytest collection of JAX-using suites imports JAX before the Kimi "
                "discriminator's explicit no-JAX assertion"
            ),
            "science_or_production_failure": False,
            "isolated_discriminator_result": "11 passed",
        },
        "adversarial_tests_added": [
            "single hoisted observer-free UVW bundle before RK stages",
            "production U/V/W pytree shapes and RK1/RK2/RK3 bundle reuse",
            "diff6-off NaN frozen bundle ignored bit-exact",
            "slopeopt omitted-profile default is zero",
            "authorization absence/tamper/staleness and exact field set",
            "inclusive/nonfinite threshold semantics and step2 withholding",
            "fresh HLO retention ordering and no ordinary compile/dispatch",
            "one lock/one model/immediate release receipt shell structure",
        ],
        "findings": [
            {
                "severity": "MEDIUM_PROCESS",
                "status": "NONBLOCKING_UNDER_INDEPENDENT_REVIEW",
                "finding": (
                    "The original PARTIAL correction lane required exact remainder "
                    "attribution; the amendment instead invoked source authority not "
                    "preserved in the committed original contract."
                ),
                "disposition": (
                    "No S1 closure accepted; this independent review admits only one "
                    "causal validation arm."
                ),
            },
            {
                "severity": "HIGH_TOOLING",
                "status": "FIXED_BY_SUPERSEDING_ARM_TOOLING",
                "finding": (
                    "Candidate profile re-pins were incomplete and still enforced a "
                    "one-file model delta plus obsolete pre-fix HLO equality."
                ),
                "disposition": (
                    "Fresh namespace/profile/source pins and mandatory fresh ordinary "
                    "HLO protocol are committed; no production numerics changed."
                ),
            },
        ],
        "unresolved_scientific_risks": [
            "Ring-1 relaxation and secondary interior S1 floors remain by PARTIAL class.",
            "The 97.8% value is retained step-1 causal attribution, not a live fixed-run result.",
            "GPU HLO/interface and the three RMSE thresholds remain unvalidated until chief-authorized execution.",
            "A green 9-step arm would not authorize S1 closure, V10/Ni/18h, performance, or release.",
        ],
        "gpu_attestation": {
            "cuda_visible_devices_for_cpu_tests": "",
            "jax_platform_for_cpu_tests": "cpu",
            "gpu_queries": 0,
            "gpu_locks": 0,
            "gpu_compiles": 0,
            "gpu_dispatches": 0,
            "device_access": 0,
            "chief_authorization_consumed": False,
        },
        "canonical_payload_sha256": None,
        "normalized_whole_file_self_sha256": ZERO_SELF,
    }
    body = {
        key: value for key, value in payload.items()
        if key not in {"canonical_payload_sha256", "normalized_whole_file_self_sha256"}
    }
    payload["canonical_payload_sha256"] = canonical(body)
    normalized_bytes = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    payload["normalized_whole_file_self_sha256"] = hashlib.sha256(
        normalized_bytes
    ).hexdigest()
    OUTPUT.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "verdict": payload["terminal_verdict"],
        "proof": str(OUTPUT),
        "canonical_payload_sha256": payload["canonical_payload_sha256"],
        "normalized_whole_file_self_sha256": payload[
            "normalized_whole_file_self_sha256"
        ],
        "file_sha256": sha256_file(OUTPUT),
        "jax_imported": "jax" in sys.modules,
        "gpuwrf_imported": any(name.startswith("gpuwrf") for name in sys.modules),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
