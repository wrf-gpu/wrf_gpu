"""Terminal proof for the v0234 step-9000 V/V10 Kimi turn (canonical, self-hashed)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import v0234_nested_frozen_wrf_boundary_window as runner  # noqa: E402

SPRINT = REPO_ROOT / ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi"
OUT = SPRINT / "proof.json"


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def main() -> int:
    discriminator = json.loads(
        (SPRINT / "autotune-discriminator-proof.json").read_text()
    )
    retained = json.loads((SPRINT / "retained-evidence-manifest.json").read_text())
    proof: dict[str, object] = {
        "schema": "gpuwrf.v0234.step9000-v-v10-kimi-terminal-proof.v1",
        "verdict": "KIMI_STEP9000_V_V10_NO_FIX_LOCALIZED",
        "branch": _git("branch", "--show-current"),
        "head_at_proof": _git("rev-parse", "HEAD"),
        "recorded_utc": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True,
        ).stdout.strip(),
        "contract": {
            "path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/CONTRACT.md",
            "file_sha256": _sha(SPRINT / "CONTRACT.md"),
            "gpt_terminal_commit": "6b863c8c73656a0342fb2027438628004133bb13",
            "gpt_commit_is_ancestor": subprocess.run(
                ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor",
                 "6b863c8c73656a0342fb2027438628004133bb13", "HEAD"],
            ).returncode == 0,
        },
        "authentication": {
            "proof_json_file_sha256": "642d9295f42680125c9c8833fe63272098578e39f8451fbce924cafc8eb4c0f4",
            "proof_json_canonical_sha256": "e0e2fe4edf7629a0428bb00bedf790d34a34375f281c1912afa480a5a880bdf2",
            "blocker_file_sha256": "1c69d9eb6c7a3f73f496e29007500e49558882eeb3e7b7fd78ab9b8848180eef",
            "blocker_canonical_sha256": "e4d54208c193d3d4935640d5d5be08b07ecd4a9ba2feeb11d86bd76896faf36e",
            "failure_proof_file_sha256": "9f963d86a5aff9b3dfa675c49069c5749f7ac72da077e423a6a52c38c9d9f12a",
            "failure_proof_canonical_sha256": "95baf0919e023904430efda19fa99f252997e64db8a606fa2b7c7c610ab7235d",
            "carry_8800_file_sha256": "492cd961c4d4b8a9e47386ae92e2a60167c01eb53e70f3eaf360367d880b1c2a",
            "carry_8800_manifest_sha256": "270f6e0f0f58f670bf4c0c98b9d161c110a7c140cc7c8052d253409f0b234c28",
            "carry_9000_file_sha256": "735196a052be91716c8b90cbdff210af1cb1469cd6f1802a83b86a820cfc3fc9",
            "carry_9000_manifest_sha256": "e88d065ce6dd785aa04aa0ae7c541de32eb2e565bde4ca6d57715cc989b5ca91",
            "carry_9000_toolingrepair2_manifest_sha256": "b92f333ffb22e7c7c586f98a7be0326e224e6afe328109129dfd34bb2ae24dc4",
            "all_canonical_hashes_recomputed_match": True,
        },
        "immutable_code": {
            "src_gpuwrf_tree_before": "835dcc29bf316c0715b41a72e064985e9cf099df",
            "src_gpuwrf_tree_after": _git("rev-parse", "HEAD:src/gpuwrf"),
            "src_gpuwrf_diff_empty": _git("diff", "--name-only", "--", "src/gpuwrf") == "",
            "model_or_numerical_edit": False,
            "model_edit_justified": False,
            "runner_source_sha256": "27f5c675b69599a5112951c4f12e4a7fa11918c7494f95ca6583429a692930e0",
            "production_stablehlo_sha256": "b12b3d64a262326516d706d138e4ff6fe43e831380bad4e3f6651e9d117149cf",
            "production_hlo_artifact_file_sha256": "15575931876ca3126e91e5b1c7cbcd85360e7cc2754354f262c5c82556d3a308",
        },
        "reconstruction": {
            "first_causal_divergence": (
                "cross-process XLA:GPU autotune kernel-selection variation on the "
                "full-tree GPU path, amplified chaotically from <1e-4 (step ~1200) "
                "to gate scale by step 9000"
            ),
            "evidence_summary": {
                "four_runs_four_distinct_d03_200_frames": retained["frame_bytes"][
                    "d03_step200_candidate_sha256"
                ],
                "lowered_hlo_identical_all_runs": retained["frame_bytes"][
                    "lowered_hlo_identical_all_runs"
                ],
                "same_software_env_divergence_toolingrepair1_vs_2": retained[
                    "frame_bytes"
                ]["toolingrepair1_vs_toolingrepair2_same_software_env"],
                "identical_forcing_d03_divergence": {
                    "parent_frames_identical": retained["frame_bytes"][
                        "toolingrepair1_vs_resource_retry1"
                    ]["parent_frames_identical"],
                    "parent_frames_total": retained["frame_bytes"][
                        "toolingrepair1_vs_resource_retry1"
                    ]["parent_frames_total"],
                    "d03_first_divergence": retained["frame_bytes"][
                        "toolingrepair1_vs_resource_retry1"
                    ]["d03_first_divergence"],
                },
                "launcher_parity_proven": retained[
                    "toolingrepair1_vs_2_launcher_parity"
                ]["software_and_numeric_env_identical"],
            },
            "v_v10_mechanism_verdict": {
                "one_systematic_mechanism": (
                    "the documented Tenerife wake placement/depth deficit drives "
                    "both the 3D V error (peak at the wake cell y39,x19 level 2; "
                    "lowest-3-level RMS 2.40 vs 0.35 aloft; not ring-concentrated) "
                    "and the V10 surface diagnostic (max error same cell; both "
                    "realizations best-match CPU-WRF after the same ~6-cell shift; "
                    "mutual realization shift (0,0); wake-box error correlation "
                    "0.986)"
                ),
                "gate_outcome_difference": (
                    "V10's systematic alone (2.2044/2.2340) exceeds the frozen "
                    "ceiling 2.11283 in every retained realization (3-4x the "
                    "0.0296 realization spread) -> robustly red; V's systematic "
                    "(1.1300/1.1351) sits within the 0.0051 realization noise band "
                    "of the 1.13182 ceiling -> realization coin-flip (one green, "
                    "one red)"
                ),
                "v10_is_known_admitted_blocker": True,
                "v_red_is_noise_level": True,
            },
            "late_ni_linkage_verdict": {
                "shared_mechanism_with_v_v10": False,
                "spatial": (
                    "wake lobe deep-interior SW ocean (y39,x19) vs ring-1 corner "
                    "detonation cells; current tree shows zero relax-frame wind-up "
                    "at 15:00"
                ),
                "caveat": (
                    "the nondeterminism is global: every fresh full-tree replay "
                    "is a new realization draw, so the late-Ni gate always "
                    "evaluates one realization and the byte-exact 15:00 waiver "
                    "binding is unfireable"
                ),
            },
            "hypotheses_challenged": {
                "wake_displacement_as_pure_shift": (
                    "falsified as the dominant form: whole-field displacement fit "
                    "explains 8.6% of V10 error variance; regional ~6-cell "
                    "placement/depth deficit documented"
                ),
                "boundary_ring_defect": (
                    "falsified for this red: V ring0/1 RMS 0.80 < interior 1.19"
                ),
                "new_transport_or_coupling_regression": (
                    "falsified: no new localized structure; both realizations "
                    "share the known error pattern at corr 0.986"
                ),
                "pressure_gradient_new_cause": (
                    "falsified: PSFC green in both realizations (16.91/17.01 vs "
                    "20.23 ceiling); PSFC A-B max diff at (27,78), not the wake"
                ),
                "harness_runner_lock_preempt_failure": (
                    "falsified: fail-closed stop worked as designed; carries "
                    "authenticated; lock released"
                ),
            },
        },
        "discriminators_run": {
            "retained_data": {
                "manifest_path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/retained-evidence-manifest.json",
                "file_sha256": _sha(SPRINT / "retained-evidence-manifest.json"),
                "canonical_sha256": retained["proof_sha256"],
                "step9000_gate_statement": retained["trajectory"]["step9000"],
            },
            "aval_probe": {
                "namespace": "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/corrected_ni_rca_max_22c2bd7a/v0234_step9000_autotune_discriminator_kimi1",
                "record": "aval-probe.json",
                "finding": (
                    "retained 8800 carry holds 2-time-level boundary leaves where "
                    "the step-0 audit carry holds 1; initial-carry lowering "
                    "reproduces production b12b3d64 byte-exact in this worktree"
                ),
            },
            "locked_gpu_discriminator": {
                "proof_path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/autotune-discriminator-proof.json",
                "file_sha256": _sha(SPRINT / "autotune-discriminator-proof.json"),
                "canonical_sha256": discriminator["proof_sha256"],
                "verdict": discriminator["verdict"],
                "arms": discriminator["arms"],
                "comparisons": discriminator["comparisons"],
            },
            "full_18h_replay_admitted": False,
            "full_18h_replay_reason": (
                "a blind fresh replay is not the next decisive test: it would "
                "draw another realization that still carries the systematic V10 "
                "red and a coin-flip V, and the byte-exact waiver cannot fire; "
                "the next decisive test is GPT's deterministic-replay pair with "
                "the manager's 15:00 admission decision"
            ),
        },
        "verification": {
            "focused_tests": {
                "path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/discriminator-focused-tests.xml",
                "file_sha256": _sha(SPRINT / "discriminator-focused-tests.xml"),
                "tests": 9,
                "failures": 0,
                "errors": 0,
            },
            "carry_8800_reauthentication_in_focused_suite": True,
            "hlo_artifact_byte_match_all_arms": all(
                (row.get("lowered_hlo_artifact_file_sha256") or "")
                == "15575931876ca3126e91e5b1c7cbcd85360e7cc2754354f262c5c82556d3a308"
                for row in discriminator["arms"].values()
            ),
        },
        "gpu_use": {
            "jobs": [
                {
                    "label": "v0234-step9000-autotune-discriminator-kimi1",
                    "purpose": "v1 carry-resume arm A (design superseded)",
                    "exit": "stopped before compile (rc=143); clean release; no science",
                },
                {
                    "label": "v0234-step9000-aval-probe-kimi1",
                    "purpose": "lower-only aval probe (no compile, no dispatch)",
                    "exit": "rc=0",
                },
                {
                    "label": "v0234-step9000-autotune-discriminator-kimi2",
                    "purpose": "v2 attempt 1 (pre-science tooling KeyError)",
                    "exit": "rc=3; clean release; no dispatch ran",
                },
                {
                    "label": "v0234-step9000-autotune-discriminator-kimi2",
                    "purpose": "v2 retry arms A/B/C",
                    "exit": "all arms rc=0; lease released rc=0",
                },
            ],
            "lock_intent": "production-preemptible",
            "one_job_at_a_time": True,
            "preempt_or_hold_observed": False,
            "terminal_lock_message": "[with_gpu_lock] v0234-step9000-autotune-discriminator-kimi2 released GPU lock (rc=0)",
            "gpu_free_after": True,
        },
        "first_red_or_terminal_status": (
            "no new scientific red produced this turn; the production step-9000 "
            "red is explained (harness nondeterminism + known systematic wake "
            "blocker), not re-run"
        ),
        "next_action_exact": {
            "owner": "GPT-5.6 max (on manager dispatch)",
            "steps": [
                "produce one complete reference full-tree replay (model tree 835dcc29, production env) with XLA_FLAGS=--xla_gpu_dump_autotune_results_to=<pin.pb>; retain carries and the 15:00 payload; expect stop at 15:00 (V10 systematic red; V coin-flip) and retain without interpreting as new science",
                "re-run the identical replay with --xla_gpu_load_autotune_results_from=<pin.pb> and prove bit-identical green frames through 8800 and a bit-identical 15:00 frame (full-path A/C equivalence); if not identical, the pin is incomplete: stop and localize the uncovered compile",
                "present the deterministic 15:00 metric payload to the manager for the admission decision (V10 ~2.20-2.23 vs 2.11283 known wake blocker; V within the proven noise band of 1.13182); the byte-exact waiver binding must be replaced by an explicit metric-grounded admission (manager/principal decision)",
                "only on that admission: continue the same pinned process through 9313/9314/9405 with the exact parent/domain output contract and prove the late-Ni window on the deterministic realization",
            ],
            "alternative_if_admission_rejected": (
                "close the V10 wake-placement whole-history phase first (out of "
                "scope of this blocker)"
            ),
        },
        "unresolved_risks": [
            "pin file validated only for the shallow startup window; a full 18h replay compiles more programs (radiation variants, 9199+ live-shaped one-step, health executable) and must regenerate the pin from one complete deterministic reference replay",
            "full-path A/C bit-equivalence is not yet proven; only the shallow window is",
            "V at 15:00 is a realization coin-flip around the frozen ceiling; V10 is systematically red in every retained realization; no determinism fix makes the current strict 15:00 gate green",
            "late-Ni 9313/9314/9405 remains unproven in the same-process full-tree path until the deterministic replay is admitted through 15:00",
        ],
        "deliverables": {
            "worker_report": {
                "path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/worker-report.md",
                "file_sha256": _sha(SPRINT / "worker-report.md"),
            },
            "retained_evidence_manifest": {
                "path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/retained-evidence-manifest.json",
                "file_sha256": _sha(SPRINT / "retained-evidence-manifest.json"),
            },
            "autotune_discriminator_proof": {
                "path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/autotune-discriminator-proof.json",
                "file_sha256": _sha(SPRINT / "autotune-discriminator-proof.json"),
            },
            "focused_tests_xml": {
                "path": ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/discriminator-focused-tests.xml",
                "file_sha256": _sha(SPRINT / "discriminator-focused-tests.xml"),
            },
            "scripts": [
                "scripts/v0234_step9000_autotune_discriminator.py",
                "scripts/v0234_step9000_autotune_discriminator_closeout.py",
                "scripts/v0234_step9000_aval_probe.py",
                "scripts/v0234_step9000_retained_evidence_analysis.py",
                "tests/test_v0234_step9000_autotune_discriminator.py",
            ],
        },
    }
    unsigned = dict(proof)
    proof["proof_sha256"] = runner.canonical_digest(unsigned)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
    tmp.replace(OUT)
    print("TERMINAL_PROOF", OUT)
    print("canonical_sha256", proof["proof_sha256"])
    print("file_sha256", _sha(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
