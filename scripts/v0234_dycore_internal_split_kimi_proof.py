"""Seal the self-hashed terminal proof for the v0234 dycore internal split.

CPU-only closeout: binds the committed contract, the discriminator script
and tests, the sealed analysis, the frozen authorities, the command log,
the report, the falsifier outcomes, and the single exact next action.  No
JAX, no gpuwrf, no GPU operation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.v0234_dycore_internal_split_kimi import (  # noqa: E402
    OUTPUT as ANALYSIS,
    SPRINT,
    canonical_digest,
    read_self_hashed,
    sha256_file,
    write_self_hashed,
)

PROOF = SPRINT / "proof.json"
CONTRACT = SPRINT / "CONTRACT.md"
REPORT = SPRINT / "worker-report.md"
COMMAND_LOG = SPRINT / "command-log.txt"
PROCESS_CORRECTIONS = SPRINT / "process-corrections.md"
SCRIPT = REPO / "scripts/v0234_dycore_internal_split_kimi.py"
TEST = REPO / "tests/test_v0234_dycore_internal_split_kimi.py"


def file_row(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    analysis, analysis_row = read_self_hashed(ANALYSIS)
    if analysis["schema"] != "gpuwrf.v0234.dycore-internal-split-kimi.analysis.v1":
        raise RuntimeError("analysis schema mismatch")

    test_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(TEST)],
        capture_output=True,
        text=True,
        check=False,
    )
    tail = (test_run.stdout + test_run.stderr).strip().splitlines()[-1]
    if test_run.returncode != 0 or "11 passed" not in tail:
        raise RuntimeError(f"focused tests not green: {tail}")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    contract_commit = subprocess.run(
        ["git", "log", "--oneline", "-1", "--", str(CONTRACT)],
        capture_output=True, text=True, check=True,
    ).stdout.split()[0]
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/gpuwrf"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    payload = {
        "schema": "gpuwrf.v0234.dycore-internal-split-kimi.terminal.v1",
        "verdict": (
            "TERMINAL_CPU_INTERNAL_SPLIT_COMPLETE_"
            "DECISIVE_S1_RK_TENDENCY_DYNAMICS_RESIDUAL"
        ),
        "objective": (
            "CPU/offline audit of the terminal single GPU arm and "
            "source-bound internal split of its two terminal systematic "
            "facts into S0 (SP3 provenance), S1 (rk_tendency residual), "
            "S2 (relax_bdy_dry), S3 (spec_bdy_dry), S4 (rk_addtend_dry "
            "accounted), treating them as potentially independent defects"
        ),
        "authority": {
            "contract": file_row(CONTRACT),
            "contract_commit": contract_commit,
            "head_before_terminal_commit": head,
            "frozen_authorities": analysis["authority"]["frozen_proofs"],
            "savepoints_revalidated": analysis["authority"]["savepoints_revalidated"],
            "wrf_source_sha256": analysis["authority"]["wrf_source_sha256"],
            "gpu_source_sha256": analysis["authority"]["gpu_source_sha256"],
            "oracle_inputs": analysis["authority"]["oracle_inputs"],
            "member_step1_dump_files": analysis["authority"]["member_step1_dump_files"],
        },
        "committed_evidence": {
            "analysis": analysis_row,
            "discriminator_script": file_row(SCRIPT),
            "proof_script": file_row(Path(__file__).resolve()),
            "focused_tests": {
                "file": file_row(TEST),
                "command": f"{sys.executable} -m pytest -q {TEST}",
                "result": tail,
            },
            "worker_report": file_row(REPORT),
            "command_log": file_row(COMMAND_LOG),
            "process_corrections": file_row(PROCESS_CORRECTIONS),
        },
        "scientific_result": {
            "terminal_decision": analysis["terminal_decision"],
            "falsifier_outcomes": analysis["falsifier_outcomes"],
            "frozen_envelope_citations_preserved": True,
            "derived_envelopes_labelled": True,
        },
        "execution": {
            "cpu_affinity": "13-15,29-31",
            "gpu_commands": 0,
            "gpu_queries": 0,
            "gpu_locks": 0,
            "jax_imported": analysis["authority"]["jax_imported"],
            "gpuwrf_imported": analysis["authority"]["gpuwrf_imported"],
            "new_model_runs": 0,
            "agents_launched": 0,
            "production_numerics_modified": False,
            "src_gpuwrf_worktree_dirty": bool(dirty),
        },
        "scope_closure": {
            "step2_replication": "removed per manager scope correction (unregistered)",
            "s1b_diff6_numeric_attribution": (
                "removed per manager scope correction (post-result, not "
                "preregistered); hypothesis preserved in unresolved risks "
                "and the exact next action"
            ),
            "correction": False,
            "gpu_arm": False,
            "gate_or_envelope_modified": False,
        },
        "unresolved_risks": [
            "S1 residual bundles all rk_tendency terms; per-term split not possible from retained rungs",
            "source-read diff6 hypothesis (periodic rolls, no nested trim, no msf/adjacent-mass coupling in GPU sixth_order_diffusion_tendency) is untested in this sprint",
            "S2 target-side provenance (boundary package vs leaf interpolation) not separable offline",
            "exact-zero relax envelope is degenerate; S2 verdict rests on structural E-decomposition",
            "step-1-only verdict by manager scope ruling",
        ],
        "handoff": {
            "next_decision": (
                "manager may contract a new preregistered CPU-only step-1 "
                "sixth-order-diffusion lane discriminator against this "
                "retained S1 residual, with the S2 relax target-side "
                "provenance as companion; this proof authorizes neither "
                "that contract's execution, a GPU arm, nor a correction"
            ),
            "callback_state": "TERMINAL_NARROWING_COMPLETE",
        },
    }
    write_self_hashed(PROOF, payload)
    written = json.loads(PROOF.read_text())
    print(json.dumps({
        "wrote": str(PROOF),
        "proof_sha256": written["proof_sha256"],
        "verdict": written["verdict"],
        "analysis_self_hash": analysis_row["canonical_self_hash"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
