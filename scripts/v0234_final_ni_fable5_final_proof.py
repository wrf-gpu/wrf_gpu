#!/usr/bin/env python3
"""Canonical terminal aggregate for the final v0.23.4 Ni candidate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
SPRINT_DIR = REPO / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
GPU_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_final_ni_fable5_terminal3"
)
RAW_GPU_PROOF = GPU_ROOT / "bounded-gpu-proof.json"
GPU_COMMIT = "66a5fbde1a1591cfadbcce74e481bbbc28f5057b"
CORRECTED_CPU_COMMIT = "8fde9c941ba31f292d9921455dde9ea6e8387fc5"
MODEL_FILES = (
    "src/gpuwrf/coupling/boundary_apply.py",
    "src/gpuwrf/dynamics/core/acoustic.py",
)
SCHEMA = "gpuwrf.v0234.final-ni-fable5-terminal-candidate.v1"


def canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_canonical(path: Path, expected_verdict: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("proof_sha256") != canonical_hash(payload):
        raise RuntimeError(f"canonical proof mismatch: {path}")
    if payload.get("verdict") != expected_verdict:
        raise RuntimeError(
            f"{path.name}: verdict {payload.get('verdict')!r} != {expected_verdict!r}"
        )
    return payload


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _git_file_sha256(commit: str, path: str) -> str:
    data = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_proof() -> dict[str, Any]:
    cpu = _load_canonical(
        SPRINT_DIR / "candidate-cpu-proof.json",
        "CPU_SOURCE_CANDIDATE_GREEN_BOUNDED_GPU_PENDING",
    )
    gpu = _load_canonical(
        SPRINT_DIR / "bounded-gpu-policy-proof.json",
        "GPU_DISCRIMINATOR_POLICY_GREEN_RAW_GATE_MISAPPLIED",
    )
    amendment = _load_canonical(
        SPRINT_DIR / "contract-amendment-02.json",
        "GPTMAX_CONTINUATION_OWNERSHIP_AMENDED_FAIL_CLOSED",
    )
    control_amendment = _load_canonical(
        SPRINT_DIR / "contract-amendment-03.json",
        "CPU_CONTROL_STOP_AFTER_ALARM_REPAIR_AMENDED_FAIL_CLOSED",
    )
    raw_gpu = _load_canonical(RAW_GPU_PROOF, "GPU_DISCRIMINATOR_RED")

    if _git("status", "--porcelain", "--", *MODEL_FILES):
        raise RuntimeError("candidate model files are dirty")
    head = _git("rev-parse", "HEAD")
    model_lineage: dict[str, Any] = {}
    for path in MODEL_FILES:
        cpu_sha = _git_file_sha256(CORRECTED_CPU_COMMIT, path)
        gpu_sha = _git_file_sha256(GPU_COMMIT, path)
        head_sha = _git_file_sha256(head, path)
        worktree_sha = _file_sha256(REPO / path)
        if len({cpu_sha, gpu_sha, head_sha, worktree_sha}) != 1:
            raise RuntimeError(f"candidate source lineage differs for {path}")
        model_lineage[path] = {
            "corrected_cpu_commit_sha256": cpu_sha,
            "gpu_execution_commit_sha256": gpu_sha,
            "aggregate_head_sha256": head_sha,
            "worktree_sha256": worktree_sha,
            "identical": True,
        }

    cpu_gates = cpu["cpu_ab_discriminator"]["gates"]
    if not cpu_gates or not all(cpu_gates.values()):
        raise RuntimeError(f"CPU A/B aggregate is not all-green: {cpu_gates!r}")
    if not cpu["correction"]["model_blob_identity_across_cpu_gpu_and_aggregate"]:
        raise RuntimeError("CPU proof model lineage is not green")
    operator = cpu["operator_gates"]
    if (
        not operator["candidate_off_exact"]
        or not operator["ordinary_candidate_on_finite"]
        or any(operator["forbidden_hlo_tokens"].values())
        or operator["production_carry_leaf_count"] != 106
        or operator["new_loop_host_device_transfer"]
        or operator["observer_or_callback_added"]
    ):
        raise RuntimeError(f"operator/interface gate red: {operator!r}")
    gpu_gates = gpu["gates"]
    bool_gpu_gates = {
        name: value for name, value in gpu_gates.items() if isinstance(value, bool)
    }
    if not bool_gpu_gates or not all(bool_gpu_gates.values()):
        raise RuntimeError(f"bounded GPU policy gate red: {gpu_gates!r}")
    if raw_gpu["model"]["commit"] != GPU_COMMIT:
        raise RuntimeError("raw GPU proof commit changed")
    if gpu["raw_proof"]["proof_sha256"] != raw_gpu["proof_sha256"]:
        raise RuntimeError("GPU policy proof does not bind the raw proof")

    proof: dict[str, Any] = {
        "schema": SCHEMA,
        "verdict": "READY_FOR_GPT_CRITIC",
        "candidate": {
            "fix_commit": "41015ce7193fdc3db6362ce8e1e6dcab07fc5f41",
            "corrected_cpu_execution_commit": CORRECTED_CPU_COMMIT,
            "bounded_gpu_execution_commit": GPU_COMMIT,
            "aggregate_head": head,
            "mechanism": (
                "preserve advance_w's advanced PH outside the nested specified ring "
                "while walking the ring from the pre-advance loop-carried PH, matching "
                "WRF's in-place advance_w then spec_bdyupdate_ph order"
            ),
            "model_source_lineage": model_lineage,
        },
        "components": {
            "authority": cpu["frozen_inputs"]["authority"],
            "carry_localization": cpu["frozen_inputs"]["carry_localization"],
            "source_invariant": cpu["frozen_inputs"]["source_invariant"],
            "interface_transfer": cpu["frozen_inputs"]["interface_transfer"],
            "cpu_candidate": {
                "path": str(SPRINT_DIR / "candidate-cpu-proof.json"),
                "proof_sha256": cpu["proof_sha256"],
                "verdict": cpu["verdict"],
            },
            "raw_bounded_gpu": {
                "path": str(RAW_GPU_PROOF),
                "proof_sha256": raw_gpu["proof_sha256"],
                "verdict": raw_gpu["verdict"],
                "red_gates": raw_gpu["red_gates"],
            },
            "bounded_gpu_policy": {
                "path": str(SPRINT_DIR / "bounded-gpu-policy-proof.json"),
                "proof_sha256": gpu["proof_sha256"],
                "verdict": gpu["verdict"],
            },
            "continuation_amendment": {
                "path": str(SPRINT_DIR / "contract-amendment-02.json"),
                "proof_sha256": amendment["proof_sha256"],
                "verdict": amendment["verdict"],
            },
            "cpu_control_repair_amendment": {
                "path": str(SPRINT_DIR / "contract-amendment-03.json"),
                "proof_sha256": control_amendment["proof_sha256"],
                "verdict": control_amendment["verdict"],
            },
        },
        "terminal_gates": {
            "source_discrepancy_proven": True,
            "falsified_cpu_arm_reproduced_step200_ni_and_ph_freeze": True,
            "corrected_cpu_arm_finite_through_step200_and_ph_advances": True,
            "focused_and_proportional_cpu_suites_green": True,
            "candidate_off_e0d0_exact": True,
            "candidate_on_interface_and_hlo_clean": True,
            "production_carry_interface_106_leaves": True,
            "no_new_model_loop_transfer_or_callback": True,
            "bounded_gpu_finite_through_step400": True,
            "bounded_gpu_post_dispatch_frames_each_strict_green": True,
            "bounded_gpu_three_frame_pooled_identity_green": True,
            "no_second_gpu_arm_or_full_history": True,
        },
        "bounded_gpu_metrics": {
            "frames": gpu["frames"],
            "pooled_strict_identity": gpu["pooled_strict_identity"],
        },
        "causal_separation": {
            "immediate_terminal2_failure": (
                "boundary-driven interior PH freeze and acoustic detonation by d03 step 200"
            ),
            "late_ni": (
                "separate unresolved step-9400/9405 interior offshore mechanism at "
                "y48,x78 (28.297913,-16.303406); this bounded window does not close it"
            ),
            "v10": (
                "separate unresolved broad 15:00 drift; bounded V10 is an identity metric "
                "only and no temporal/operator causal link to Ni is inferred"
            ),
        },
        "procedural_admission": {
            **cpu["procedural_admission"],
            "raw_gpu_per_frame_policy_bug": (
                "00:00 T2 false red only; exact known terminal1 bytes; raw proof preserved"
            ),
            "policy_repair": (
                "unchanged released limits applied to pooled exact matched samples, with "
                "the 00:20 and 00:40 post-dispatch frames additionally required per-frame green"
            ),
            "independent_critic_must_accept_admission": True,
        },
        "unresolved_release_gates": [
            "fresh corrected history through 15:00 and steps 9313/9314/9315/9405",
            "late interior/offshore Ni mechanism through step 9405",
            "15:00 frozen no-worse field gates, especially V10 and PSFC",
            "full 18-hour 19/19/55 identity and numbers-first JPG",
        ],
        "commands": [
            "PYTHONPATH=.:src python scripts/v0234_final_ni_fable5_candidate_proof.py",
            "PYTHONPATH=.:src python scripts/v0234_final_ni_fable5_gpu_policy_proof.py",
            "PYTHONPATH=.:src python scripts/v0234_final_ni_fable5_final_proof.py",
        ],
    }
    proof["proof_sha256"] = canonical_hash(proof)
    return proof


def main() -> int:
    proof = build_proof()
    out_path = SPRINT_DIR / "final-candidate-proof.json"
    out_path.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
    print(f"verdict={proof['verdict']}")
    print(f"proof_sha256={proof['proof_sha256']}")
    print(f"written={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
