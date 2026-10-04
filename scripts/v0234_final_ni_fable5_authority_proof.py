"""Authority proof for the v0.23.4 final Ni fix sprint.

Authenticates every frozen input this sprint used: the accepted model/tooling/
review commits, the terminal1/terminal2 namespaces and their canonical proofs,
the retained carries, the pristine WRF checkout, the accepted lock-v2, and the
CPU-WRF oracle frames consumed by the bounded discriminators.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "gpuwrf.v0234.final-ni-fable5-authority.v1"
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
RCA_ROOT = CASE_ROOT / "corrected_ni_rca_max_22c2bd7a"
TERMINAL1 = RCA_ROOT / "v0234_1500_science_60659a2e_terminal1"
TERMINAL2 = RCA_ROOT / "v0234_1500_science_60659a2e_terminal2"

EXPECTED = {
    "model_commit": "60659a2e02e33153d87256ee78bf1380f5eabe76",
    "model_tree": "708b2ad4afea1f03bd16d61c0fe6391994dcc331",
    "tooling_commit": "e2a45a9bd740a629292b4e06727fc14ba87057a4",
    "review_commit": "04d7bbfbc2051fb9f69dd3e502423ca50f853778",
    "lock_commit": "8152309aff1e85e1052d44d549a5a5409e710bdd",
    "lock_wrapper_sha256": "c75b3a4eda17e94df921986e1c15fcc71e182e01d51b6fe877e4c52533077e1a",
    "blocker_proof_sha256": "18e7936c95588d98f5100b35edf61591b9bd1835673de3a214590d36b6fa7584",
    "failure_proof_sha256": "f1abd1c6e2c99af4bc2e74003c09d30898cf84c7efa73834d8ef9705f194fadf",
    "step0_carry_sha256": "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d",
    "step200_carry_sha256": "3f439a40d6467d00d2acc5f719696da6a4cf5392572a704f864366cdbfcc0f55",
}

FROZEN_1500_MAXIMA = {
    "T": 0.5421680888949643,
    "U": 1.1323567330094144,
    "V": 1.1318203205639872,
    "W": 0.18633111790197587,
    "T2": 1.352612988238872,
    "U10": 1.9737860008971808,
    "V10": 2.1128268857679338,
    "PSFC": 20.22747532736003,
}
STRICT_RMSE_LIMITS = {
    "T": 1.5, "U": 1.8, "V": 1.8, "W": 0.3,
    "T2": 1.5, "U10": 1.5, "V10": 1.5, "PSFC": 120.0,
}


def canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    checks: dict[str, Any] = {}
    checks["model_commit_tree"] = git(REPO, "rev-parse", f"{EXPECTED['model_commit']}^{{tree}}")
    if checks["model_commit_tree"] != EXPECTED["model_tree"]:
        raise RuntimeError("model tree mismatch")
    checks["tooling_head"] = git(
        Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-1500-science-terminal-tooling"),
        "rev-parse", "HEAD",
    )
    checks["review_head"] = git(
        Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-1500-science-critic"),
        "rev-parse", "HEAD",
    )
    checks["lock_head"] = git(
        Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2"), "rev-parse", "HEAD"
    )
    checks["lock_wrapper_sha256"] = sha256_file(
        Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2/scripts/with_gpu_lock.sh")
    )
    for key, expected_key in (
        ("tooling_head", "tooling_commit"),
        ("review_head", "review_commit"),
        ("lock_head", "lock_commit"),
        ("lock_wrapper_sha256", "lock_wrapper_sha256"),
    ):
        if checks[key] != EXPECTED[expected_key]:
            raise RuntimeError(f"{key} mismatch: {checks[key]}")

    blocker = json.loads((TERMINAL2 / "science-runtime/full-run-blocker.json").read_text())
    if canonical_hash(blocker) != EXPECTED["blocker_proof_sha256"]:
        raise RuntimeError("blocker canonical hash mismatch")
    failure = json.loads(
        (TERMINAL2 / "science-runtime/failure/failure-proof.json").read_text()
    )
    if canonical_hash(failure) != EXPECTED["failure_proof_sha256"]:
        raise RuntimeError("failure canonical hash mismatch")
    carries = {
        "step0": sha256_file(
            TERMINAL2 / "science-runtime/failure/last-healthy-d03-step-0.pkl"
        ),
        "step200": sha256_file(
            TERMINAL2 / "science-runtime/failure/first-failed-d03-step-200.pkl"
        ),
    }
    if carries["step0"] != EXPECTED["step0_carry_sha256"]:
        raise RuntimeError("step0 carry mismatch")
    if carries["step200"] != EXPECTED["step200_carry_sha256"]:
        raise RuntimeError("step200 carry mismatch")

    wrf = Path("<USER_HOME>/src/wrf_pristine/WRF")
    pristine = {
        "head": git(wrf, "rev-parse", "HEAD"),
        "version_line": next(
            line
            for line in (wrf / "README").read_text(errors="replace").splitlines()
            if "Version" in line
        ).strip(),
    }
    if "4.7.1" not in pristine["version_line"]:
        raise RuntimeError("pristine WRF is not 4.7.1")

    cpu_frames = {}
    for label in ("00:00:00", "00:20:00", "00:40:00"):
        path = CASE_ROOT / "run/wrf" / f"wrfout_d03_2025-03-01_{label}"
        cpu_frames[label] = {"path": str(path), "sha256": sha256_file(path)}

    terminal1_preserved = {
        "namespace": str(TERMINAL1),
        "status": "harness-only (stale 1800s watchdog kill after identity-green 00:00), preserved read-only",
        "frame_pairs": sorted(
            p.name for p in (TERMINAL1 / "science-runtime/frame-pairs").iterdir()
        ),
    }

    proof = {
        "schema": SCHEMA,
        "expected": EXPECTED,
        "checks": checks,
        "terminal2": {
            "namespace": str(TERMINAL2),
            "blocker_verdict": blocker["verdict"],
            "failure_detail": failure["failure"]["detail"],
            "carries": carries,
        },
        "terminal1": terminal1_preserved,
        "pristine_wrf": pristine,
        "cpu_oracle_frames_d03": cpu_frames,
        "frozen_1500_rmse_maxima": FROZEN_1500_MAXIMA,
        "strict_identity_policy_limits": STRICT_RMSE_LIMITS,
        "sprint_contract": {
            "path": ".agent/sprints/2026-07-14-v0234-1500-scientific-rca/sprint-contract.md",
            "sha256": sha256_file(
                REPO / ".agent/sprints/2026-07-14-v0234-1500-scientific-rca/sprint-contract.md"
            ),
        },
        "verdict": "AUTHORITY_GREEN",
        "commands": [
            "PYTHONPATH=.:src python scripts/v0234_final_ni_fable5_authority_proof.py"
        ],
    }
    proof["proof_sha256"] = canonical_hash(proof)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "authority-proof.json"
    out_path.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
    print(f"verdict={proof['verdict']}")
    print(f"proof_sha256={proof['proof_sha256']}")
    print(f"written={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
