"""Terminal proof sealer for sprint 2026-07-18-v0234-s1-dyn-attribution-fable5.

Binds the committed contract, amendment, attribution analysis, corrected model
files, capture-profile re-pins, and test evidence into one self-hashed
proof.json.  CPU-only; imports neither JAX nor gpuwrf.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import v0234_dycore_internal_split_kimi as split  # noqa: E402

SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-s1-dyn-attribution-fable5"
ANALYSIS = SPRINT / "s1-attribution-analysis.json"


def main() -> None:
    analysis, analysis_row = split.read_self_hashed(ANALYSIS)
    frozen_split, frozen_row = split.read_self_hashed(
        REPO
        / ".agent/sprints/2026-07-18-v0234-dycore-internal-split-kimi/internal-split-analysis.json"
    )
    head = subprocess.check_output(
        ("git", "-C", str(REPO), "rev-parse", "HEAD"), text=True
    ).strip()

    step1 = analysis["steps"]["1"]["attribution"]
    proof = {
        "schema": "gpuwrf.v0234.s1-dyn-attribution-terminal.v1",
        "sprint": SPRINT.name,
        "owner": "Fable 5 max (one-run principal authorization)",
        "git_head": head,
        "contract_commit": "c814aa6f",
        "evidence_commit": "c6a32335",
        "fix_commit": "e65ce784",
        "terminal_decision": "FIX_READY_FOR_GPU_VALIDATION",
        "scientific_class_per_frozen_gates": analysis["terminal_class"],
        "authority": {
            "attribution_analysis": analysis_row,
            "frozen_internal_split": frozen_row,
            "frozen_s1_nonspec_rmse": frozen_split["s1_rk_tendency_residual"][
                "nonspec"
            ]["gpu_rmse"],
        },
        "headline": {
            "f_a1_reproduction_relative_error": analysis["F_A1_reproduction"][
                "relative_error"
            ],
            "step1_nonspec_explained_fraction": step1["nonspec"][
                "explained_fraction"
            ],
            "step1_nonspec_corr_R_P": step1["nonspec"]["corr_R_P"],
            "step1_relax_explained_fraction": step1["relax_rows_1_4"][
                "explained_fraction"
            ],
            "step1_relax_rmse_before_after": [
                step1["relax_rows_1_4"]["rmse_R"],
                step1["relax_rows_1_4"]["rmse_R_minus_P"],
            ],
            "step1_interior_rmse_before_after": [
                step1["interior_ge_5"]["rmse_R"],
                step1["interior_ge_5"]["rmse_R_minus_P"],
            ],
            "step2_member_dyn_envelope_note": (
                "six 1-ULP WRF members show 15.41-16.72 rmse step-2 DYN "
                "residuals (inherited-state amplification); GPU/member ratio "
                "falls 8105x->21.7x; no new step-2 operator signal"
            ),
        },
        "corrected_files_sha256": {
            "src/gpuwrf/runtime/operational_mode.py": (
                "9480c992b46df6a76f0f492baf8098d7945507d1c9e6c44c002bfbd0a4eceaff"
            ),
            "src/gpuwrf/dynamics/explicit_diffusion.py": (
                "58e677d4e7748057f4b3546380296f01e99a58fb161b76971343a46aa9792b8c"
            ),
        },
        "tests": {
            "focused": "tests/test_v0234_s1_dyn_attribution_fable5.py (6 passed; "
            "includes F-B retained-control equality <=1e-10 and the no-wrap "
            "adversarial fixture)",
            "regression": (
                "first_interval 9, stage_omega+h_sca+conditioning+suboperator "
                "batteries 78 passed (one pre-existing order-sensitive "
                "sys.modules hygiene test documented), corner+boundary 42, "
                "diff6-adjacent suites 73"
            ),
        },
        "open_defects_not_claimed_fixed": {
            "ring_1_floor_rmse": 15.237,
            "rings_2_4_floor_rmse": 0.79,
            "interior_floor_rmse": 1.1402,
            "interior_floor_structure": (
                "terrain-correlated (0.51), surface-concentrated k-profile; "
                "Smagorinsky/degraded-advection lanes shortlisted; convention "
                "hypothesis measured dead (corr -0.022)"
            ),
            "sp2_mynn_input_inheritance": "separate upstream lane, k0-4, land-weighted",
            "s2_relax_target_side": "inherited stop (boundary leaves not retained)",
            "spec_bdy_final_absence": (
                "measured immaterial: WRF final overwrite max_abs 5.7e-6, "
                "GPU L5==SP4 bitwise steps 1-9"
            ),
        },
        "gpu_validation_need_frozen": {
            "arm": "9-step d03 ladder rerun on the fixed tree",
            "launcher_lineage": (
                ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi/"
                "dycore-suboperator-ladder-exact-launch-command.sh (new arm "
                "contract required; profile re-pinned to the fixed tree; "
                "lowered-HLO reference MUST be re-dumped - live nested-child "
                "program intentionally differs)"
            ),
            "falsifier": (
                "step-1 nonspec rmse(R) > 4.0 or relax rmse(R) > 9.5 or "
                "interior rmse(R) > 1.4 falsifies the fix (expectations "
                "3.224/7.766/1.140); no V10/18h/late-Ni claim from this arm"
            ),
            "resources": "single canonical lock-v2 production-preemptible arm",
        },
        "gpu_attestation": {
            "gpu_commands_queries_locks_compiles_dispatches": 0,
            "jax_used_only_via_cpu_pinned_tests": True,
            "analysis_imports_jax_or_gpuwrf": False,
        },
    }
    split.write_self_hashed(SPRINT / "proof.json", proof)
    print("sealed", SPRINT / "proof.json")


if __name__ == "__main__":
    main()
