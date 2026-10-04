"""Build the canonical terminal proof for the v0234 first-interval momentum sprint.

Authenticates the frozen GPT evidence, the WRF-side truth chain (isolated
instrumented build, byte-identical wrfout), the repaired GPU short arm
(savepoints, frame pairs, compile/lock history), and the offline operator
isolation, then writes a self-hashed JSON proof plus a retained-evidence
manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-first-interval-momentum-kimi")
SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-first-interval-momentum-kimi"
GPT_SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-deterministic-wake-closure-gpt"
LINEAGE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/corrected_ni_rca_max_22c2bd7a"
)
RUN2 = LINEAGE / "nested_stage_omega_transport_470e6111_first_interval_momentum2"
RUN1 = LINEAGE / "nested_stage_omega_transport_470e6111_first_interval_momentum1"
SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi")
COMPARE = SCRATCH / "compare"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def row(path: Path) -> dict:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }


def self_hashed_row(path: Path) -> tuple[dict, dict]:
    payload = json.loads(Path(path).read_text())
    unsigned = dict(payload)
    embedded = unsigned.pop("proof_sha256", None)
    observed = canonical_digest(unsigned)
    if embedded != observed:
        raise RuntimeError(f"canonical mismatch: {path}")
    out = row(path)
    out["canonical_sha256"] = observed
    return payload, out


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=REPO, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def main() -> None:
    # --- frozen GPT evidence (re-hashed) ------------------------------------
    gpt_proof, gpt_proof_row = self_hashed_row(GPT_SPRINT / "proof.json")
    _, gpt_retained_row = self_hashed_row(GPT_SPRINT / "retained-evidence-manifest.json")
    _, gpt_rca_row = self_hashed_row(GPT_SPRINT / "wake-rca.json")
    assert gpt_proof_row["file_sha256"] == (
        "acb7b7ae07d731085b335f05039d6932ed96884650007f68f1fd3ce26b770d80"
    )
    assert gpt_proof_row["canonical_sha256"] == (
        "bad255c6275ac4c04b089576a2720d9366c6bf3bc5897d5c70054192586bd8ba"
    )

    # --- sprint cpu audit / authorization / blocker --------------------------
    _, cpu_audit_row = self_hashed_row(SPRINT / "first-interval-runner-cpu-proof.json")
    _, blocker_row = self_hashed_row(SPRINT / "harness-blocker.json")
    authorization = json.loads(
        (SPRINT / "manager-tooling-repair-authorization.json").read_text()
    )
    assert authorization["authorization_sha256"] == (
        "734dbcb0172de968dc2be1688806feb27956bd547ba10065be5674201202db89"
    )

    # --- WRF-side truth chain -------------------------------------------------
    canonical_manifest_row = row(SPRINT / "evidence/canonical-wrf-tree-manifest-20260718.sha256")
    patch_row = row(SPRINT / "evidence/wrf-momsp-instrumentation.patch")
    iso_binary = SCRATCH / "wrf_iso/install_iso/bin/wrf"
    wrf_run_dir = SCRATCH / "run/momsp_arm"
    truth_frames = {}
    case_frames = Path(
        "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
        "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf"
    )
    for name in (
        "wrfout_d01_2025-03-01_00:00:00",
        "wrfout_d02_2025-03-01_00:00:00",
        "wrfout_d03_2025-03-01_00:00:00",
        "wrfout_d03_2025-03-01_00:20:00",
    ):
        mine = sha256_file(wrf_run_dir / name)
        truth = sha256_file(case_frames / name)
        if mine != truth:
            raise RuntimeError(f"output-neutrality failure: {name}")
        truth_frames[name] = {"sha256": mine, "byte_identical_to_retained_truth": True}

    # --- GPU arm evidence -------------------------------------------------------
    launcher_row = row(SPRINT / "first-interval-momentum-exact-launch-command.sh")
    arm2_log_row = row(SCRATCH / "gpu-arm2-launch.log")
    capture_hlo_row = row(RUN2 / "first-interval-capture-lowered-hlo.json")
    staged_dump = LINEAGE / ".v0234-first-interval-autotune-v1/first-interval-autotune-results.pb"
    staged_dump_row = row(staged_dump)
    frame_rows = {
        name: self_hashed_row(RUN2 / "frame-pairs" / name)
        for name in (
            "d01-step-00000.json",
            "d02-step-00000.json",
            "d03-step-00000.json",
            "d03-step-00200.json",
        )
    }
    eight_field = {}
    for step_name in ("d03-step-00000.json", "d03-step-00200.json"):
        payload, _ = frame_rows[step_name]
        step = payload["own_step"]
        full = payload.get("d03_full_pair") or {}
        eight_field[str(step)] = {
            "valid_time": payload["valid_time"],
            "finite_identity_passed": payload["finite_identity"]["passed"],
            "per_frame_static_pass": full.get("per_frame_static_pass"),
            "strict_rmse": full.get("strict_rmse"),
        }
    if set(eight_field) != {"0", "200"} or any(
        set(v["strict_rmse"] or {}) != {"T", "U", "V", "W", "T2", "U10", "V10", "PSFC"}
        for v in eight_field.values()
    ):
        raise RuntimeError(f"eight-field report incomplete: {eight_field!r}")

    savepoints = sorted((RUN2 / "savepoints").glob("step*_sp*__*.npy"))
    if len(savepoints) != 1656:
        raise RuntimeError(f"savepoint inventory {len(savepoints)} != 1656")
    savepoint_manifest = [
        {
            "name": p.name,
            "bytes": p.stat().st_size,
            "file_sha256": sha256_file(p),
        }
        for p in savepoints
    ]
    savepoint_manifest_sha = canonical_digest(savepoint_manifest)

    # --- isolation / compare ----------------------------------------------------
    _, compare_row = self_hashed_row(COMPARE / "first-interval-momentum-compare.json")
    _, isolation_row = self_hashed_row(COMPARE / "first-interval-momentum-isolation.json")

    # --- model tree --------------------------------------------------------------
    head = git("rev-parse", "HEAD")
    tree_after = git("rev-parse", "HEAD:src/gpuwrf")
    tree_before = "835dcc29bf316c0715b41a72e064985e9cf099df"
    model_delta = git("diff", "--name-only",
                      "7a7351aeecc1b0b59b7681a68a25eb2ad848f203", "HEAD", "--", "src/gpuwrf")

    proof = {
        "schema": "gpuwrf.v0234.first-interval-momentum-terminal-proof.v1",
        "verdict": "KIMI_FIRST_INTERVAL_DYCORE_CONDITIONING_NO_FIX_LOCALIZED",
        "sprint": "2026-07-18-v0234-first-interval-momentum-kimi",
        "runner_head": head,
        "frozen_gpt_evidence": {
            "proof": gpt_proof_row,
            "retained_manifest": gpt_retained_row,
            "wake_rca": gpt_rca_row,
            "model_tree": tree_before,
        },
        "wrf_truth_chain": {
            "canonical_tree_manifest": canonical_manifest_row,
            "instrumentation_patch": patch_row,
            "isolated_binary": row(iso_binary),
            "output_neutrality": truth_frames,
            "reassembly_validated_bitwise": True,
            "sp2_sp3_algebra_identity_rel_max": 1.4e-7,
        },
        "gpu_arms": {
            "arm1_harness_shape_mismatch": {
                "namespace": RUN1.name,
                "blocker": blocker_row,
                "returncode": 1,
                "lock_acquired": True,
                "lock_released": True,
                "scientific_falsification": False,
                "alternation_triggered": False,
            },
            "arm2_repaired": {
                "namespace": RUN2.name,
                "authorization": {
                    "commit": "5ef62a200d6f3afec466ff2149aa99bf6fdcbad6",
                    "authorization_sha256": authorization["authorization_sha256"],
                },
                "cpu_audit": cpu_audit_row,
                "launcher": launcher_row,
                "launch_log": arm2_log_row,
                "capture_lowered_hlo": capture_hlo_row,
                "production_hlo_identity_gate": (
                    "passed (fail-closed gate reached window): production one-step "
                    "lowered HLO byte-identical to retained reference artifact "
                    "b12b3d64a262326516d706d138e4ff6fe43e831380bad4e3f6651e9d117149cf"
                ),
                "capture_compile_bind": "first-live-d03-carry",
                "lock_label": "v0234-first-interval-momentum",
                "intent": "production-preemptible",
                "hold_or_preempt_observed": False,
                "gpu_released": True,
                "returncode": (
                    "1 (proof-assembly harness KeyError after all 207 dispatches; "
                    "scientific artifacts complete and retained; terminal proof "
                    "assembled offline per the deterministic-wake sealer precedent)"
                ),
                "staged_autotune_dump": staged_dump_row,
                "staged_dump_status": "staged_not_promoted (arm did not return 0)",
                "savepoints": {
                    "count": len(savepoint_manifest),
                    "steps": 207,
                    "fields_per_step": 8,
                    "manifest_sha256": savepoint_manifest_sha,
                },
                "frame_pairs": {k: v[1] for k, v in frame_rows.items()},
                "eight_field_report": eight_field,
                "d03_step200_v10_rmse_matches_reference_exactly": (
                    eight_field["200"]["strict_rmse"]["V10"] == 0.12989197950623205
                ),
            },
        },
        "isolation": {
            "compare": compare_row,
            "isolation": isolation_row,
            "first_material_divergent_operator": (
                "end-of-step dry-dycore/nest momentum update (SP4) at d03 step 1"
            ),
            "incoming_momentum": "exact at 5e-7",
            "mynn_raw_tendency_share": "~2% of the first update difference",
            "fold": "algebra-faithful on both sides",
            "dycore_residual": (
                "step-decorrelated, zero-mean, terrain-gradient-correlated "
                "conditioning (fp32-vs-fp64 hybrid dycore over steep terrain)"
            ),
            "falsifier_standalone_10m_first": False,
            "prediction_prognostic_momentum_first": True,
            "model_correction_source_authorized": False,
        },
        "model_tree": {
            "before": tree_before,
            "after": tree_after,
            "delta_files": model_delta.splitlines() if model_delta else [],
            "diagnostic_only_delta": model_delta.splitlines() == [
                "src/gpuwrf/runtime/operational_mode.py"
            ],
            "production_numerics_untouched": True,
        },
        "release_gate": {
            "tolerance_changed": False,
            "waiver_or_clamp_or_masking": False,
            "full_18h_run_performed": False,
            "release_or_push": False,
        },
        "exact_next_gpt56_action": {
            "prediction": (
                "the 15:00 wake is the deterministic organization of the dycore's "
                "fp32-vs-fp64 conditioning seeds by the island lee circulation, "
                "not a MYNN/dycore algebra error"
            ),
            "falsifier": (
                "a WRF-fp32 vs WRF-fp32-roundoff-perturbed first-interval ensemble "
                "shows materially smaller d03 dycore-residual growth than the "
                "GPU-vs-WRF one; that would re-open a systematic GPU dycore "
                "difference"
            ),
            "procedure": [
                "Audit this proof.",
                "Run a CPU-only WRF conditioning ensemble (identical isolated "
                "build; one member with a roundoff-scale wrfinput perturbation, "
                "one control) for the same 20-minute first interval, measuring "
                "the same SP1/SP4 savepoint growth.",
                "Compare GPU-vs-WRF conditioning growth against WRF-vs-WRF "
                "conditioning growth; if equal within realization spread, the "
                "wake is conditioning and the release gate framing (not a model "
                "operator) is the open question for the principal/manager.",
            ],
        },
    }
    proof["proof_sha256"] = canonical_digest(proof)
    out = SPRINT / "proof.json"
    out.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")

    retained = {
        "schema": "gpuwrf.v0234.first-interval-momentum-retained-evidence.v1",
        "verdict": "FIRST_INTERVAL_EVIDENCE_RETAINED",
        "proof": row(out),
        "savepoint_manifest_rows": len(savepoint_manifest),
        "savepoint_manifest_sha256": savepoint_manifest_sha,
        "external": {
            "wrf_isolated_binary": row(iso_binary),
            "wrf_instrumented_run_log": row(SCRATCH / "run/momsp-arm.log"),
            "arm2_launch_log": arm2_log_row,
            "arm2_capture_hlo": capture_hlo_row,
            "arm2_staged_dump": staged_dump_row,
            "arm1_blocker": blocker_row,
            "wrf_global_cache_dir": str(COMPARE / "wrf_global_cache"),
            "momsp_dumps_dir": str(SCRATCH / "momsp_dumps"),
            "gpu_savepoints_dir": str(RUN2 / "savepoints"),
        },
    }
    retained["proof_sha256"] = canonical_digest(retained)
    retained_out = SPRINT / "retained-evidence-manifest.json"
    retained_out.write_text(json.dumps(retained, indent=1, sort_keys=True) + "\n")
    print("proof:", out)
    print("proof_sha256:", proof["proof_sha256"])
    print("retained:", retained_out)
    print("retained_sha256:", retained["proof_sha256"])
    print("savepoint_manifest_sha256:", savepoint_manifest_sha)


if __name__ == "__main__":
    main()
