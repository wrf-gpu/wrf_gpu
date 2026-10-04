#!/usr/bin/env python3
"""Terminal proof sealer for the v0234 all-remaining-gaps Fable 5 sprint.

Standard-library only; no JAX, no gpuwrf, no GPU action. Binds the contract,
commits, discriminator artifacts, audit facts, retained-evidence manifest,
and the terminal verdict passed on the command line.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPRINT = ROOT / ".agent/sprints/2026-07-19-v0234-all-remaining-gaps-fable5"

ALLOWED_VERDICTS = {
    "FABLE5_ALL_REMAINING_GAPS_FIX_READY_FOR_INTEGRATION",
    "FABLE5_ALL_REMAINING_GAPS_READY_FOR_GPU_AUTH",
    "FABLE5_ALL_REMAINING_GAPS_PARTIAL_LOCALIZED",
    "FABLE5_ALL_REMAINING_GAPS_AUTHORITY_BLOCKED",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_self_hash(path: Path) -> str:
    data = json.loads(path.read_text())
    body = {k: v for k, v in data.items() if k not in ("self_sha256", "proof_sha256")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", "-C", str(ROOT), *args), text=True).strip()


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ALLOWED_VERDICTS:
        raise SystemExit(f"usage: {sys.argv[0]} <verdict in {sorted(ALLOWED_VERDICTS)}>")
    verdict = sys.argv[1]

    tracked = {}
    for name in (
        "CONTRACT.md",
        "LIVE_GPT_SNAPSHOT.md",
        "EVIDENCE-MAP-AND-CAUSAL-GRAPH.md",
        "PREREGISTRATION-D1-D2.md",
        "PREREGISTRATION-AMENDMENT-01.md",
        "POST-INTEGRATION-VALIDATION-DESIGN.md",
        "d1-pgf-representation-floor.json",
        "d2-mynn-sp2-classification.json",
        "worker-report.md",
    ):
        path = SPRINT / name
        entry = {"file_sha256": sha256_file(path)}
        if name.endswith(".json"):
            entry["canonical_self_hash"] = canonical_self_hash(path)
        tracked[name] = entry

    scripts = {
        rel: sha256_file(ROOT / rel)
        for rel in (
            "scripts/v0234_all_gaps_d1_pgf_representation_floor.py",
            "scripts/v0234_all_gaps_d2_mynn_sp2_classification.py",
            "scripts/v0234_all_gaps_fable5_terminal_proof.py",
        )
    }

    d1 = json.loads((SPRINT / "d1-pgf-representation-floor.json").read_text())
    d2 = json.loads((SPRINT / "d2-mynn-sp2-classification.json").read_text())

    payload = {
        "schema": "gpuwrf.v0234.all-remaining-gaps-fable5-terminal.v1",
        "sprint": "2026-07-19-v0234-all-remaining-gaps-fable5",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "base": "74c686fc",
        "src_gpuwrf_tree": git("rev-parse", "HEAD:src/gpuwrf"),
        "production_model_bytes_changed": False,
        "commit_chain": {
            "contract": "8b0770e6",
            "evidence_map_and_preregistration": "690d32b4",
            "preregistration_amendment": "cca4a428",
            "d1_result_and_d2_runner": "ff29f708",
        },
        "gpt_live_audit": {
            "worktree": "<USER_HOME>/src/wrf_gpu2_wt/v0234-s1-residual-closure-gpt",
            "observed_commits": ["93132eac", "d4b03077"],
            "artifacts_reauthenticated": {
                "candidate-cpu-proof.json": "500db72270e5bd0d5eceb5c37a807cc530d41d38bc716f1c71c754906ea6e8b9",
                "pgf-component-ledger.json": "c49881d06fcf581505891efaac7b88a033a11327d63cd293b1bcc76d4de3d20a",
                "operator-ledger-proof.json": "3d8c7c3c775429ebd629a192caa4192c73bc24bd2a572eb21a6f047e3ea3bad1",
                "residual-field-analysis.json": "0a68c80fe0fcf0d1ff0f05a04ca62bda0eef388ceeda9bdf192cc598ebb68846",
            },
            "gpt_files_edited_by_fable": 0,
            "verdict": "S1_CLOSURE_VALID_AND_COMPLETE_AT_CPU_LEVEL_PENDING_ARM_AND_KIMI_CRITIC",
        },
        "discriminators": {
            "d1": {"verdict": d1["verdict"], "gates": d1["gates"], "self_sha256": d1["self_sha256"]},
            "d2": {
                "verdict": d2["verdict"],
                "self_sha256": d2["self_sha256"],
                "r_model_rms": d2.get("r_model_rms"),
                "r_backend_rms": (d2.get("b0_harness_gate") or {}).get("r_backend_rms"),
                "ratio_model_over_comparator": d2.get("ratio_model_over_comparator"),
            },
        },
        "sprint_files": tracked,
        "script_hashes": scripts,
        "retained_evidence": {
            "step0_carry": "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d",
            "wrfinput_d03": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
            "fixed_savepoint_rows": "550178b158a849c5da55cee31b82e6c90ebee733620d141ac278c064764cf33a",
            "wrf_sp2_double_anchor": "kimi momsp cache bit-equal to GPT operator-ledger f64 dumps (rublten, rvblten)",
            "pgf_dump_rows": "94fada4c6cb43564b633446e47fc64e00722c8f3e9ec8a60440a857215dd2bbe",
        },
        "resource_attestation": {
            "gpu_queries": 0,
            "gpu_locks": 0,
            "gpu_compiles": 0,
            "gpu_dispatches": 0,
            "gpu_profiler_actions": 0,
            "agents_spawned": 0,
            "models_spawned": 0,
            "cpu_lane": "13-15,29-31 (one 1-core read-only check on 12)",
            "live_gpt_interference": "none; read-only inspection only",
        },
        "verdict": verdict,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["proof_sha256"] = hashlib.sha256(body).hexdigest()
    out = SPRINT / "proof.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "proof_sha256": payload["proof_sha256"], "file": str(out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
