#!/usr/bin/env python3
"""Assemble the canonical terminal proof for the final holistic Fable run.

Reads the committed amendments and the on-/mnt arm proofs, re-verifies their
hashes, and writes the sprint's canonical proof.json (self-hash = SHA-256 of
compact sorted-key allow_nan=false JSON minus the proof_sha256 member).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-17-v0234-final-holistic-fable5-xhigh2"
ARMS = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/final_holistic_fable5_xhigh2_8d99421f"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: dict) -> str:
    clean = {k: v for k, v in payload.items() if k != "proof_sha256"}
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def prefix_summary(log_path: Path) -> dict:
    rows = 0
    last_step = None
    corner_lo, corner_hi = None, None
    for line in log_path.read_text().splitlines():
        if not line.startswith("ARM_S step="):
            continue
        rows += 1
        parts = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
        last_step = int(parts["step"])
        if int(parts["nonfinite"]) != 0:
            raise SystemExit(f"green prefix claim violated in {log_path}")
        c = float(parts["corner_mu_maxabs"])
        corner_lo = c if corner_lo is None else min(corner_lo, c)
        corner_hi = c if corner_hi is None else max(corner_hi, c)
    return {
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
        "green_rows": rows,
        "last_green_step": last_step,
        "all_nonfinite_zero": True,
        "corner_mu_maxabs_range": [corner_lo, corner_hi],
    }


def arm_blocker_summary() -> dict:
    return {
        "verdict": "ARM_S_HARNESS_BLOCKER_SCIENCE_INCOMPLETE",
        "closure_amendment": "amendment-05-armS-harness-blocker-closure.json",
        "attempts": [
            prefix_summary(ARMS / "armS_stalled_superseded" / "armS_first42.log"),
            prefix_summary(ARMS / "armS_stall2_superseded" / "armS_at9075.log"),
            prefix_summary(ARMS / "armS_stall3_superseded" / "armS_at9016.log"),
        ],
        "stall_snapshots": [
            str(ARMS / "armS_stall2_superseded" / "armS-stall2-pid1364053-snapshot.txt"),
            str(ARMS / "armS_stall3_superseded" / "armS-stall3-python-snapshot.txt"),
        ],
        "science_read": "bounded-but-incomplete: no corner-pump expression in any completed released-tree dispatch (max prefix 75 steps); no green-window verdict claimed",
    }


def arm_summary(name: str) -> dict:
    path = ARMS / f"arm{name}" / f"arm-{name}-proof.json"
    payload = json.loads(path.read_text())
    if canonical(payload) != payload["proof_sha256"]:
        raise SystemExit(f"arm {name} proof self-hash mismatch")
    rows = payload["rows"]
    last = rows[-1] if rows else {}
    return {
        "verdict": payload["verdict"],
        "first_red_step": payload["first_red_step"],
        "head_commit": payload["head_commit"],
        "head_src_tree": payload["head_src_tree"],
        "input_sha256": payload["input_sha256"],
        "start_step": payload["start_step"],
        "completed_steps": payload["completed_steps"],
        "last_step": last.get("step"),
        "last_nonfinite_total": last.get("nonfinite_total"),
        "last_corner_mu": last.get("corner_mu"),
        "last_mu_min": last.get("mu_min"),
        "last_w_maxabs": last.get("w_maxabs"),
        "max_corner_mu_maxabs_over_window": max(
            (r["corner_mu_maxabs"] for r in rows if r.get("corner_mu_maxabs") is not None),
            default=None,
        ),
        "proof_path": str(path),
        "proof_file_sha256": sha256_file(path),
        "proof_canonical_sha256": payload["proof_sha256"],
        "compile_seconds": payload["compile_seconds"],
    }


def main() -> int:
    verdict = sys.argv[1] if len(sys.argv) > 1 else None
    if verdict not in (
        "FINAL_FABLE5_XHIGH_NO_FIX_LOCALIZED",
        "FINAL_FABLE5_XHIGH_FIX_CPU_GREEN_GPU_PENDING",
        "FINAL_FABLE5_XHIGH_NO_GO",
    ):
        raise SystemExit("usage: v0234_final_holistic_terminal_proof.py <verdict>")

    payload = {
        "schema": "gpuwrf.v0234.final-holistic-fable5-xhigh2-terminal-proof.v1",
        "verdict": verdict,
        "date_utc": "2026-07-17",
        "contract_commit": "9258a35f4fbdc77d2f84247e83cc32c146bde358",
        "head_commit": git("rev-parse", "HEAD"),
        "src_gpuwrf_tree": git("rev-parse", "HEAD:src/gpuwrf"),
        "released_tree_expected": "835dcc29bf316c0715b41a72e064985e9cf099df",
        "model_bytes_changed": False,
        "authentication": {
            "audit_rows_verified": 66,
            "audit_artifacts_byte_verified": 69,
            "post_audit_delta_verified": 11,
            "moist_theta_proof_selfhash_anomaly_recorded": True,
            "kimi_commit": "abeeb112c0cfc6186f4f2c745f4657301f3b3372",
            "kimi_proof_sha256": "992d637085d6c6e811a8123591c07a46be000ff53f82df127395b28663f17d67",
        },
        "late_ni_localization": {
            "mechanism": "nested boundary relax-frame corner mass-pump detonation (old effe1f43-era tree); ring-1 corner mu' -10.6k..-12.3k Pa at 9313; advance_uv work overflow -> dvdxi/dmdt Inf -> Inf*0 NaN in _limit_mu_drain_scale; theta blow-up masked by the positive-definite theta increment limiter",
            "current_tree_windup_at_1500": "zero (ring |mudf| mean 0.018 vs interior 0.014; old lineage 0.11..76)",
            "stage_omega_step9000_carry_sha256": "1b1521090539cc38a5660712ff1fe57a4251c3b790f7d619a176a8f32037d055",
        },
        "v10_localization": {
            "mechanism": "common ~6 km eastward island-wake displacement shared by Retry20 and Stage-Omega; excess is deep-interior SW offshore (97.3% at >=20 cells from boundary), boundary band improved under Stage-Omega (-84% of excess)",
            "wake_shift_km": {"stage_omega": -6, "retry20": -6, "mutual": 1},
            "separate_from_ni": True,
        },
        "discriminator": {
            "amendments": [
                "amendment-01-late-window-corner-discriminator.json",
                "amendment-02-full-window-continuation.json",
                "amendment-03-contention-restart-merged-window.json",
                "amendment-04-second-stall-single-thread-final-attempt.json",
                "amendment-05-armS-harness-blocker-closure.json",
            ],
            "arm_O": arm_summary("O"),
            "arm_S": arm_blocker_summary(),
            "arm_X": (
                arm_summary("X")
                if (ARMS / "armX" / "arm-X-proof.json").exists()
                else {"verdict": "ARM_X_HARNESS_BLOCKER_SCIENCE_UNREAD"}
            ),
        },
        "gpt_continuation": {
            "next_discriminator": "complete the committed arm-S protocol on an idle box: scripts/v0234_final_holistic_late_window_corner_discriminator.py --arm S --steps 405 (input hash 1b152109..., ~30 min at the observed 3.8 s/dispatch); predicted outcome: zero nonfinite and corner mu' in the clean 1.4e3 band through 9405; falsifier: any nonfinite or corner |mu'| > 5e4 inside 9001..9405",
            "after_green": "one fresh full-tree GPU replay under canonical lock-v2/PREEMPT with manager authorization to record-and-continue past the red 15:00 V10 gate to the 9405 finite-Ni endpoint (gates unchanged, nothing reclassified)",
            "step15_savepoint": "the sealed unchanged command remains queued for CPU admission (Nightly was pipeline_running/live WRF throughout this run)",
        },
        "scope": {
            "gpu_queries": 0,
            "gpu_locks": 0,
            "gpu_compiles": 0,
            "gpu_dispatches": 0,
            "second_model_calls": 0,
            "step15_placeholder_consumed": False,
            "production_untouched": True,
            "cpu_lane": "taskset 12-15 nice 5 ionice c2 n5 (yielded to live production WRF)",
        },
    }
    payload["proof_sha256"] = canonical(payload)
    out = SPRINT / "proof.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"verdict": verdict, "proof_sha256": payload["proof_sha256"],
                      "file": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
