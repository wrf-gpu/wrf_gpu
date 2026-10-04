"""Closeout for the v0234 step-9000 autotune discriminator (Kimi turn).

CPU-only, read-only on the discriminator namespace. Compares the three arm
results and autotune dumps, emits the canonical self-hashed discriminator
proof, and prints the verdict line.

Verdict logic (fail-closed, exactly one row true):
- NONDETERMINISM_REPRODUCED__AUTOTUNE_MECHANISM__PIN_VALIDATED:
    arm A and arm B reached step 9000 with different terminal manifests,
    their autotune dumps differ, and arm C (pinned to A's dump) reproduces
    arm A's manifest at every milestone.
- NONDETERMINISM_REPRODUCED__MECHANISM_NOT_AUTOTUNE_DUMP:
    A != B but dumps identical (or C != A): per-process variation exists but
    is not captured by the autotune results file.
- INCONCLUSIVE_PICKS_COINCIDED: A == B at the terminal milestone.
- ARMS_INCOMPLETE: any arm failed or is missing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import v0234_nested_frozen_wrf_boundary_window as runner  # noqa: E402

NS = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_step9000_autotune_discriminator_kimi2"
)
OUT = (
    REPO_ROOT
    / ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi"
    / "autotune-discriminator-proof.json"
)


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> int:
    arms: dict[str, dict] = {}
    envelopes: dict[str, dict] = {}
    dumps: dict[str, str | None] = {}
    incomplete: list[str] = []
    for arm in ("A", "B", "C"):
        arm_dir = NS / f"arm{arm}"
        result = _load(arm_dir / f"arm-{arm}-result.json")
        envelope = _load(arm_dir / f"arm-{arm}-envelope.json")
        if result is None or result.get("failed") or envelope is None:
            incomplete.append(arm)
            if result is not None:
                arms[arm] = result
            if envelope is not None:
                envelopes[arm] = envelope
            continue
        arms[arm] = result
        envelopes[arm] = envelope
        dumps[arm] = _sha(arm_dir / "autotune-results.pb")

    captures_by_arm = {
        arm: (arms[arm].get("dispatch") or {}).get("captured") or {}
        for arm in arms
        if not arms[arm].get("failed")
    }

    def terminal(arm: str) -> str | None:
        return (captures_by_arm.get(arm) or {}).get("d03-200", {}).get(
            "manifest_sha256"
        )

    def all_milestones_equal(left: str, right: str) -> bool:
        lm, rm = captures_by_arm.get(left), captures_by_arm.get(right)
        if not lm or not rm:
            return False
        common = set(lm) & set(rm)
        return bool(common) and all(
            lm[key].get("manifest_sha256") == rm[key].get("manifest_sha256")
            for key in common
        )

    dump_ab_differ = (
        dumps.get("A") is not None
        and dumps.get("B") is not None
        and dumps["A"] != dumps["B"]
    )
    if incomplete:
        verdict = "ARMS_INCOMPLETE"
    elif terminal("A") != terminal("B"):
        if dump_ab_differ and all_milestones_equal("A", "C"):
            verdict = "NONDETERMINISM_REPRODUCED__AUTOTUNE_MECHANISM__PIN_VALIDATED"
        else:
            verdict = "NONDETERMINISM_REPRODUCED__MECHANISM_NOT_AUTOTUNE_DUMP"
    else:
        verdict = "INCONCLUSIVE_PICKS_COINCIDED"

    proof: dict[str, object] = {
        "schema": "gpuwrf.v0234.step9000-autotune-discriminator-proof.v1",
        "namespace": str(NS),
        "verdict": verdict,
        "incomplete_arms": incomplete,
        "arms": {
            arm: {
                "arm_result_file_sha256": envelopes.get(arm, {}).get(
                    "arm_result_file_sha256"
                ),
                "arm_exit_code": envelopes.get(arm, {}).get("arm_exit_code"),
                "compile_wall_seconds": (arms.get(arm, {}).get("compile") or {}).get(
                    "compile_wall_seconds"
                ),
                "lower_wall_seconds": (arms.get(arm, {}).get("compile") or {}).get(
                    "lower_wall_seconds"
                ),
                "stablehlo_sha256": (arms.get(arm, {}).get("compile") or {}).get(
                    "stablehlo_sha256"
                ),
                "lowered_hlo_artifact_file_sha256": (
                    arms.get(arm, {}).get("compile") or {}
                ).get("lowered_hlo_artifact_file_sha256"),
                "captured": captures_by_arm.get(arm),
                "events": (arms.get(arm, {}).get("dispatch") or {}).get("events"),
                "integrate_wall_seconds": (
                    arms.get(arm, {}).get("dispatch") or {}
                ).get("wall_seconds"),
                "autotune_results_file_sha256": dumps.get(arm),
                "wall_seconds": arms.get(arm, {}).get("wall_seconds"),
            }
            for arm in ("A", "B", "C")
        },
        "comparisons": {
            "d03_200_A_vs_B_differ": (
                terminal("A") is not None
                and terminal("B") is not None
                and terminal("A") != terminal("B")
            ),
            "autotune_dump_A_vs_B_differ": dump_ab_differ,
            "arm_C_equals_arm_A_all_captured": all_milestones_equal("A", "C"),
            "d03_200_manifests": {arm: terminal(arm) for arm in ("A", "B", "C")},
            "production_d03_200_candidate_sha256": {
                "discriminator1": "6113fbbe6853 (prefix of full sha in retained frame pair)",
                "toolingrepair1": "900ceefd8a4f (prefix)",
                "toolingrepair2": "5a7b06c3c966 (prefix)",
                "resource_retry1": "e16674b898c0 (prefix)",
            },
            "protocol_note": (
                "Arms replay the exact production startup (load, D03_LOWER_COMPILE "
                "audit gated to the production StableHLO byte hash, live tree "
                "integration along the production output schedule) and capture the "
                "d03 step-200 state payload -- the first frame at which all four "
                "production runs diverged. The payload manifest uses the production "
                "host_tree_manifest scheme; comparison is across arms only."
            ),
        },
        "model_tree": "835dcc29bf316c0715b41a72e064985e9cf099df",
        "model_or_numerical_edit": False,
    }
    unsigned = dict(proof)
    proof["proof_sha256"] = runner.canonical_digest(unsigned)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
    tmp.replace(OUT)
    print("STEP9000_DISCRIMINATOR_PROOF", OUT)
    print("verdict", verdict)
    print("canonical_sha256", proof["proof_sha256"])
    print("file_sha256", _sha(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
