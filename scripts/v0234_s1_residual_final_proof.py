#!/usr/bin/env python3
"""Authenticate and seal the terminal CPU-only S1 residual-closure proof."""

from __future__ import annotations

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-s1-residual-closure-gpt"
PROOF = SPRINT / "proof.json"
REPORT = SPRINT / "worker-report.md"
MANIFEST = SPRINT / "retained-evidence-manifest.json"
ZERO = "0" * 64

EXPECTED_SELF = {
    "residual-field-analysis.json": "0a68c80fe0fcf0d1ff0f05a04ca62bda0eef388ceeda9bdf192cc598ebb68846",
    "operator-ledger-proof.json": "3d8c7c3c775429ebd629a192caa4192c73bc24bd2a572eb21a6f047e3ea3bad1",
    "pgf-component-ledger.json": "c49881d06fcf581505891efaac7b88a033a11327d63cd293b1bcc76d4de3d20a",
    "candidate-cpu-proof.json": "500db72270e5bd0d5eceb5c37a807cc530d41d38bc716f1c71c754906ea6e8b9",
    "s1-residual-runner-cpu-proof.json": "2b3655624f75dff31188612a95d32c645b1f391ed9b244e6b018b72760101f7e",
}
EXPECTED_FILES = {
    "CONTRACT.md": "f4c9368e08c446b81fb32545699a19d7f51012c10df7e01e94ed145d356f4f03",
    "PREREGISTRATION.md": "ec9b80a5fc6496914dcc8595e9f26fef455c30c54027422f495886694cd296bc",
    "source-operator-map.md": "ebbbec7110c0a1121bc35aac1883a9bfa97e9a87e7e7e459e55339fb69813423",
    "CONTRACT-AMENDMENT-01.md": "becb71215ae4b32e5da5cc1f66657bf32103f2e9bcb4e7f5c91d230303f74796",
    "CONTRACT-AMENDMENT-02.md": "99c71a832b7e903fa10cb11ca6f7e6bd129a6b11fd228f6c61f663c28c2a1d86",
    "CONTRACT-AMENDMENT-03.md": "18977b77b343bb53fe90aa9739e9c20f1330dadb53a004dde5c444b46583ac1d",
    "GPU-ARM-CONTRACT.md": "9f2a04b914a95295d1e7ae690a4fc96a5af019ab315f9c1c6d93a3b403517cf2",
    "s1-residual-exact-launch-command.sh": "f72f2c4de63de4ebf6dd7d3934fed370606676b1ca44e14393836d28c9d2126e",
}
MODEL_HASHES = {
    "src/gpuwrf/dynamics/core/rk_addtend_dry.py": "15a9eccc3f66e25176f6b23fca205b7d997c04c82d15aad2b5ae52cc183f2c34",
    "src/gpuwrf/dynamics/explicit_diffusion.py": "f6eaa78d08ee25dd7d2977721521bf30a21fae72a751d9184d2a408459c4b11e",
    "src/gpuwrf/runtime/operational_mode.py": "68efe76b9d1f91860e9a49a6e6c9ab9e573986dd11a47677fa161badb3a79282",
}
PREDICTIONS = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_s1_residual_closure_gpt/operator_ledger1/"
    "current-source-predictions-step1.npz"
)
PREDICTIONS_SHA = "73087c007e556be03a5a1cb40ab975d31d52a6bdb8e08d789ea78fa9714dca3a"
FROZEN_TERMINAL = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_s1_diff6_fix_validation_gpt1/"
    "s1-diff6-validation-terminal-proof.json"
)
FROZEN_TERMINAL_SELF = "6e8ed620932459a6d663feafb8a64cdd3bf5cbd3aaa65a6e878dcdb1e70c677e"
FROZEN_ROWS = "550178b158a849c5da55cee31b82e6c90ebee733620d141ac278c064764cf33a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }


def self_hashed(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = SPRINT / name
    payload = json.loads(path.read_text())
    observed = canonical({key: value for key, value in payload.items() if key != "proof_sha256"})
    require(payload.get("proof_sha256") == observed == EXPECTED_SELF[name], f"self hash: {name}")
    return payload, file_row(path) | {"canonical_self_hash": observed, "authenticated": True}


def junit(name: str) -> dict[str, Any]:
    path = SPRINT / name
    suite = ET.parse(path).getroot().find("testsuite")
    require(suite is not None, f"missing JUnit suite: {name}")
    row = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
    require(row["failures"] == row["errors"] == 0, f"red JUnit: {name}")
    return file_row(path) | row | {"passed": row["tests"] - row["skipped"]}


def seal(payload: dict[str, Any], path: Path) -> dict[str, str]:
    payload["canonical_payload_sha256"] = canonical(payload)
    payload["normalized_whole_file_self_sha256"] = ZERO
    normalized = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    payload["normalized_whole_file_self_sha256"] = hashlib.sha256(normalized).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return {
        "canonical_payload_sha256": payload["canonical_payload_sha256"],
        "normalized_whole_file_self_sha256": payload["normalized_whole_file_self_sha256"],
        "file_sha256": sha256_file(path),
    }


def main() -> None:
    require(not git("status", "--porcelain"), "worktree must be clean before closeout generation")
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    require(branch == "worker/gpt/v0234-s1-residual-closure", "wrong branch")
    require(
        subprocess.run(("git", "-C", str(REPO), "merge-base", "--is-ancestor", "d4b03077f2e8858163d1fbdf99c090fab902e0ac", head), check=False).returncode == 0,
        "source candidate is not ancestor",
    )
    for name, expected in EXPECTED_FILES.items():
        require(sha256_file(SPRINT / name) == expected, f"file identity: {name}")
    for relative, expected in MODEL_HASHES.items():
        require(sha256_file(REPO / relative) == expected, f"model identity: {relative}")
    require(sha256_file(PREDICTIONS) == PREDICTIONS_SHA, "prediction archive identity")

    residual, residual_row = self_hashed("residual-field-analysis.json")
    operator, operator_row = self_hashed("operator-ledger-proof.json")
    pgf, pgf_row = self_hashed("pgf-component-ledger.json")
    candidate, candidate_row = self_hashed("candidate-cpu-proof.json")
    audit, audit_row = self_hashed("s1-residual-runner-cpu-proof.json")
    terminal = json.loads(FROZEN_TERMINAL.read_text())
    require(
        terminal.get("proof_sha256") == canonical({key: value for key, value in terminal.items() if key != "proof_sha256"}) == FROZEN_TERMINAL_SELF,
        "frozen terminal identity",
    )
    require(terminal["window"]["model_dispatches"] == 9, "frozen dispatch count")
    require(len(terminal["savepoint_manifest"]["rows"]) == 180, "frozen savepoint count")
    require(canonical(terminal["savepoint_manifest"]["rows"]) == FROZEN_ROWS, "frozen savepoint rows")
    require(residual["reproduction"]["exact"] is True, "frozen residual reproduction")
    require(operator["scientific_disposition"]["conditioning_claim_rejected"] is True, "conditioning claim not rejected")
    require(pgf["component_sum_score_against_sealed_pgf_prediction"]["explained_sse"] > 0.99998, "PGF input closure")
    require(candidate["scientific_disposition"]["all_complete_zones_no_worse"] is True, "candidate no-worse gate")
    require(audit["verdict"] == "READY_FOR_AUTHORIZED_SINGLE_S1_RESIDUAL_GPU_ARM", "arm audit verdict")
    require(audit["gpu_commands_run"] == audit["gpu_queries_run"] == 0, "GPU use in audit")

    broad = junit("broad-regression-junit.xml")
    historical = junit("historical-regression-junit.xml")
    frozen_metrics = residual["reproduction"]["observed"]
    zone_rows = candidate["zones"]
    cpu_prediction = {
        name: {
            "frozen_raw_rmse": row["before"]["rmse"],
            "candidate_raw_rmse": row["after_hdiff_curvature"]["rmse"],
            "intrinsic_rmse": row["intrinsic_after_subtracting_bound_pgf_adv_cor"]["rmse"],
            "hdiff_curvature_explained_sse": row["explained_sse_hdiff_curvature"],
            "no_worse": row["no_worse"],
        }
        for name, row in zone_rows.items()
    }

    payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.s1-residual-closure-gpt.terminal.v1",
        "terminal_verdict": "S1_RESIDUAL_FIX_READY_FOR_GPU_AUTH",
        "owner": "GPT-5.6-xhigh",
        "branch": branch,
        "closeout_head": head,
        "objective": "authenticate, source-attribute, and close the authentic d03 step-1 S1 residual without GPU use",
        "authority": {
            "terminal": file_row(FROZEN_TERMINAL) | {
                "canonical_self_hash": FROZEN_TERMINAL_SELF,
                "dispatches": 9,
                "savepoints_rehashed_by_terminal": 180,
                "savepoint_rows_sha256": FROZEN_ROWS,
                "lock_released": terminal["gpu_released_at_exit"],
            },
            "residual_analysis": residual_row,
            "frozen_metrics_exact": frozen_metrics,
            "manager_conditioning_assumption": "rejected: ensemble GPU-minus-WRF exceeded 2x WRF-minus-WRF at every retained step",
        },
        "preregistration": {
            "file": file_row(SPRINT / "PREREGISTRATION.md"),
            "commit": "3ee7f0259c8e2983d1724a90c1c1dd1f52faec7b",
            "decisive_results_not_viewed_before_commit": True,
            "selected_hypothesis": "H1 nested horizontal diffusion plus H2 normal-map curvature; H4 PGF incoming-state asymmetry",
        },
        "source_attribution": {
            "operator_ledger": operator_row,
            "source_map": file_row(SPRINT / "source-operator-map.md"),
            "hdiff_explained_sse": {
                zone: operator["unscaled_prediction_scores"]["hdiff"][zone]["explained_sse"]
                for zone in ("nonspec", "relax_rows_1_4", "interior_ge_5", "ring_1")
            },
            "pgf_component_ledger": pgf_row,
            "pgf_component_explained_sse": pgf["component_sum_score_against_sealed_pgf_prediction"]["explained_sse"],
            "pgf_component_closure_rmse": pgf["reconstruction"]["component_sum_prediction_minus_sealed_prediction_rmse"],
            "pgf_disposition": "incoming diagnostic-state/capture asymmetry; explicitly not counted as a source fix",
            "sealed_prediction_archive": file_row(PREDICTIONS),
        },
        "correction": {
            "amendments": [
                file_row(SPRINT / name) | {"commit": commit}
                for name, commit in (
                    ("CONTRACT-AMENDMENT-01.md", "f7a63fd3ed06e32c41b750da939a3de323f41777"),
                    ("CONTRACT-AMENDMENT-02.md", "74c686fcb6f13141b24676426c6378d1ce2df7f6"),
                    ("CONTRACT-AMENDMENT-03.md", "93132eaca7293cc105e4c25cf8ee5b6b66d8f31b"),
                )
            ],
            "source_candidate_commit": "d4b03077f2e8858163d1fbdf99c090fab902e0ac",
            "source_parent": "e65ce784bec85ea4f000e94ba572420c595ecb68",
            "model_source_sha256": MODEL_HASHES,
            "changed_mechanisms": [
                "literal WRF specified/nested diffopt1 U/V/W horizontal diffusion",
                "literal WRF normal-map U/V horizontal curvature after Coriolis",
            ],
            "excluded_edits": ["PGF", "carried diagnostics", "acoustic cadence"],
            "candidate_cpu_proof": candidate_row,
            "cpu_prediction_by_zone": cpu_prediction,
            "oracle": {"literal_numpy_absolute_error_le": 1e-12, "jaxpr_host_callbacks": 0},
            "new_carry_leaves": 0,
            "timestep_host_device_transfers": 0,
            "clamps_masks_nan_replacements": 0,
            "tolerance_or_release_gate_changes": 0,
        },
        "tests": {
            "arm_cpu_audit": audit_row | {
                "verdict": audit["verdict"],
                "focused_result": "13 passed",
                "runner_head_at_audit": audit["runner_head_at_audit"],
            },
            "broad_regression": broad,
            "historical_regression": historical,
            "broad_result": "69 passed, 3 skipped",
            "historical_result": "32 passed",
            "no_tolerance_change": True,
        },
        "gpu_arm": {
            "state": "HOLD; not executed",
            "contract": file_row(SPRINT / "GPU-ARM-CONTRACT.md"),
            "launcher": file_row(SPRINT / "s1-residual-exact-launch-command.sh"),
            "namespace": "nested_stage_omega_transport_470e6111_s1_residual_closure_gpt1",
            "lock_label": "v0234-s1-residual-closure",
            "authorization_path": "/tmp/v0234-s1-residual-gpu-authorization.json",
            "authorization_issuer": "manager-0:1",
            "stop_first_decisive": True,
            "raw_absolute_band": 0.01,
            "intrinsic_rmse_limit": 0.01,
            "dispatches_if_green": 9,
            "savepoints_if_green": 180,
            "cpu_closeout_after_lock_release": True,
            "kimi_critic_required_before_integration": True,
        },
        "commands": [
            "python scripts/v0234_s1_residual_structure.py",
            "taskset -c 13-15,29-31 mpirun -np 12 <fresh instrumented pristine-WRF-v4.7.1 operator run>",
            "CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python scripts/v0234_s1_operator_ledger.py",
            "taskset -c 13-15,29-31 mpirun -np 12 <fresh output-neutral pristine-WRF-v4.7.1 PGF-input run>",
            "CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python scripts/v0234_s1_candidate_cpu_proof.py",
            "env -i ... taskset -c 13-15,29-31 python -m scripts.v0234_nested_frozen_wrf_boundary_window --cpu-dry-run --bootstrap-audit --run-focused-tests",
            "CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python -m pytest <72-test broad gate>",
            "CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python -m pytest tests/test_v0234_s1_diff6_gpu_arm.py tests/test_v0234_dycore_internal_split_kimi.py",
        ],
        "resource_attestation": {
            "cpu_physical_cores_for_new_wrf": "13-15",
            "cpu_allowed_list_observed_for_12_ranks": "13-15,29-31",
            "gpu_queries": 0,
            "gpu_locks": 0,
            "gpu_compiles": 0,
            "gpu_dispatches": 0,
            "gpu_profilers": 0,
            "historical_namespaces_mutated": 0,
            "new_wrf_namespaces": [
                "<DATA_ROOT>/wrf_gpu2/v0234_s1_residual_closure_gpt/operator_ledger1/dumps_step_fixed",
                "<DATA_ROOT>/wrf_gpu2/v0234_s1_residual_closure_gpt/operator_ledger1/dumps_pgf_inputs1",
            ],
        },
        "scope": {
            "gpu_authorized": False,
            "integration_authorized": False,
            "v10_ni_18h_performance_release_authorized": False,
            "unresolved_risks": [
                "GPU realization of the CPU-predicted source correction remains unmeasured by explicit contract.",
                "The raw order-0.8 PGF input-state floor remains visible; only its source provenance, not a prognostic-state closure, is established.",
                "Kimi independent criticism remains mandatory before integration.",
            ],
            "next_decision": "manager pane 0:1 decides whether to authorize the one fresh, minimal, stop-first S1 residual GPU arm",
        },
    }
    proof_hashes = seal(payload, PROOF)

    report = f"""# GPT-5.6 xhigh handoff — v0234 S1 residual closure

Terminal verdict: **S1_RESIDUAL_FIX_READY_FOR_GPU_AUTH**  
Source candidate: `d4b03077f2e8858163d1fbdf99c090fab902e0ac`  
CPU arm audit: **READY_FOR_AUTHORIZED_SINGLE_S1_RESIDUAL_GPU_ARM**

## Outcome

The authentic frozen residual and its authority reproduced exactly: nine GPU
dispatches, 180 authenticated savepoints, released lock, and d03 step-1 RMSE
`{frozen_metrics['nonspec']}` nonspecified, `{frozen_metrics['relax_rows_1_4']}`
relaxation rows 1–4, and `{frozen_metrics['interior_ge_5']}` interior rows >=5.
The inherited “conditioning” framing is rejected: the retained ensemble did
not bound the GPU residual as WRF numerical variability.

The dominant source defect is WRF nested horizontal momentum diffusion:
it explains `{operator['unscaled_prediction_scores']['hdiff']['relax_rows_1_4']['explained_sse']:.10f}`
of relaxation SSE and `{operator['unscaled_prediction_scores']['hdiff']['ring_1']['explained_sse']:.10f}`
of ring-1 SSE. The retained correction implements literal WRF U/V/W
specified-boundary diffusion plus the missing normal-map U/V curvature after
Coriolis. Independent NumPy loop oracles agree within `1e-12`; no callback,
carry leaf, timestep transfer, clamp, mask, tolerance, or release-gate change
was introduced.

| Zone | Frozen raw | Corrected CPU prediction | Source-normalized intrinsic |
|---|---:|---:|---:|
"""
    for name in ("nonspec", "relax_rows_1_4", "interior_ge_5", "ring_1"):
        row = cpu_prediction[name]
        report += f"| {name} | `{row['frozen_raw_rmse']:.10f}` | `{row['candidate_raw_rmse']:.10f}` | `{row['intrinsic_rmse']:.10f}` |\n"
    report += f"""

The remaining raw ~0.8 floor is not relabeled noise. Fresh WRF PGF-input
dumps show that signed pressure/nonhydrostatic/base-pressure/geopotential input
deltas reconstruct `{pgf['component_sum_score_against_sealed_pgf_prediction']['explained_sse']:.10f}`
of the sealed PGF-prediction SSE. This is an incoming diagnostic-state/capture
asymmetry and is explicitly excluded from the model fix.

## Files changed

- Production: `src/gpuwrf/dynamics/explicit_diffusion.py`,
  `src/gpuwrf/dynamics/core/rk_addtend_dry.py`, and
  `src/gpuwrf/runtime/operational_mode.py`.
- Sprint evidence: preregistration, residual structure, complete source map,
  operator and PGF ledgers, three amendments, CPU candidate proof, held arm
  contract/launcher/profile/closeout, CPU arm audit, test XML, terminal proof,
  retained-evidence manifest, and this report.
- Arm infrastructure: the residual profile is selected before legacy profiles;
  the inherited ladder admits the three-file tree only under the exact
  residual flag and hashes.

## Commands and proof objects

- Two fresh pristine-WRF-v4.7.1 CPU runs used 12 ranks restricted to
  `13-15,29-31`; both completed successfully in new evidence namespaces.
- Literal source-operator and arm audit: `13 passed` inside the canonical
  CPU bootstrap proof.
- Broad diffopt/scalar, PGF/hypsometric, theta/diff6, finite-state, source, and
  arm regression: `{broad['passed']} passed, {broad['skipped']} skipped`.
- Historical diff6-arm/internal-split regression: `{historical['passed']} passed`.
- GPU activity in this sprint: zero queries, locks, compiles, dispatches, and
  profilers. The arm launcher remains HOLD and was not executed.
- Terminal proof canonical SHA-256: `{proof_hashes['canonical_payload_sha256']}`.
- Terminal proof normalized whole-file self SHA-256:
  `{proof_hashes['normalized_whole_file_self_sha256']}`.
- Terminal proof actual file SHA-256: `{proof_hashes['file_sha256']}`.

## Unresolved risks and next decision

The CPU source correction still needs one authentic GPU realization. The raw
PGF input-state floor remains visible even though its provenance is closed,
and Kimi review remains mandatory before integration. No V10, Ni, 18-hour,
performance, release, merge, or integration scope is authorized.

Exact next decision: manager pane `0:1` decides whether to authorize the
committed fresh one-run arm (`GPU-ARM-CONTRACT.md`, launcher SHA-256
`{EXPECTED_FILES['s1-residual-exact-launch-command.sh']}`).
"""
    REPORT.write_text(report)

    retained_paths = [
        SPRINT / name for name in (
            "CONTRACT.md", "PREREGISTRATION.md", "source-operator-map.md",
            "CONTRACT-AMENDMENT-01.md", "CONTRACT-AMENDMENT-02.md",
            "CONTRACT-AMENDMENT-03.md", "GPU-ARM-CONTRACT.md",
            "residual-field-analysis.json", "operator-ledger-proof.json",
            "pgf-component-ledger.json", "candidate-cpu-proof.json",
            "s1-residual-exact-launch-command.sh", "s1-residual-runner-cpu-proof.json",
            "broad-regression-junit.xml", "historical-regression-junit.xml",
            "proof.json", "worker-report.md",
        )
    ] + [REPO / name for name in MODEL_HASHES] + [
        REPO / "scripts/v0234_s1_residual_gpu_arm_profile.py",
        REPO / "scripts/v0234_s1_residual_gpu_arm_closeout.py",
        REPO / "scripts/v0234_s1_residual_final_proof.py",
        PREDICTIONS,
        FROZEN_TERMINAL,
    ]
    manifest_payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.s1-residual-closure.retained-evidence.v1",
        "terminal_verdict": payload["terminal_verdict"],
        "closeout_head": head,
        "rows": [file_row(path) for path in retained_paths],
        "row_count": len(retained_paths),
        "all_files_rehashed": True,
        "proof_canonical_payload_sha256": proof_hashes["canonical_payload_sha256"],
        "proof_normalized_whole_file_self_sha256": proof_hashes["normalized_whole_file_self_sha256"],
    }
    manifest_hashes = seal(manifest_payload, MANIFEST)
    print(json.dumps({
        "verdict": payload["terminal_verdict"],
        "proof": proof_hashes,
        "manifest": manifest_hashes,
        "closeout_head": head,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
