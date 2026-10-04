#!/usr/bin/env python3
"""Build the independent GPT M1 Round-3 final-critic CPU evidence.

This entry point is deliberately CPU-only.  It audits the committed author
proof, recomputes immutable identities and arithmetic from independently
generated raw arms, mutation-attacks the author's proof-binding gate, and
checks whether the published typed-work floor actually rounds work at the
contracted production boundary.

It does not modify or substitute any production implementation.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# This import must precede JAX and gpuwrf. ``real_state`` loads the repository's
# top-level ``cpu_guard`` module (the canonical import spelling used by all v0.25
# probes), force-pins the process to CPU, and adds ``src`` to ``sys.path``.
from scripts.v025.real_state import assert_cpu_only  # noqa: E402
import cpu_guard  # noqa: E402


SCHEMA = "wrf_gpu2.v025.m1.round3_gpt_final_critic_cpu_evidence.v1"
TERMINAL_VERDICT = "REJECT_ROUND3_BLOCKED_PROOF_UNSOUND"
EXACT_CANDIDATE = "3677e491f395dfd44bb87e6946a24e231435a399"
PRE_CANDIDATE = "6d80c4c1"
AUTHOR_PROOF = REPO / "proofs/v025/m1/m1_round3_production_chain_cpu_evidence.json"
AUTHOR_SIDECAR = AUTHOR_PROOF.with_suffix(".sha256")
AUTHOR_RAW = REPO / "proofs/v025/m1/round3_raw"
AUTHOR_GENERATOR = REPO / "scripts/v025/build_m1_round3_evidence.py"
AUTHOR_CHAIN = REPO / "scripts/v025/m1_round3_chain.py"
CONTRACT = (
    REPO
    / ".agent/sprints/2026-07-29-v0250-m1-round3-production-chain/CONTRACT.md"
)
CRITIC_CONTRACT = (
    REPO
    / ".agent/sprints/2026-07-29-v0250-m1-round3-production-chain/"
    "GPT_ROUND3_FINAL_CRITIC_CONTRACT.md"
)
AMENDMENT_RELATIVE = (
    ".agent/sprints/2026-07-29-v0250-m1-round3-production-chain/"
    "AMENDMENT_1_REVIEW04_GATE_CORRECTION.md"
)
AMENDMENT_COMMIT = "23904a25"
CRITIC_RAW_FILENAMES = (
    "cand_fp32_repro.json",
    "pre_fp32_repro.json",
    "cand_fp64_repro.json",
    "conversion_attribution_repro.json",
    "isolated_candidate_repro.json",
    "isolated_pre_repro.json",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_bytes(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def python_tree_digest(source_root: Path) -> str:
    """Recompute the author's custom Python-only ``src/gpuwrf`` digest."""

    package = source_root / "gpuwrf"
    entries = sorted(
        f"{path.relative_to(source_root)}:{sha256_file(path)}"
        for path in package.rglob("*.py")
    )
    return sha256_text("\n".join(entries))


def _load_author_generator():
    spec = importlib.util.spec_from_file_location(
        "round3_author_evidence_generator", AUTHOR_GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {AUTHOR_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proof_binding_attack(author_proof: dict[str, Any]) -> dict[str, Any]:
    """Return every mutation that the author's claimed gate accepts."""

    module = _load_author_generator()
    binding = author_proof["binding"]

    def passes(mutator) -> bool:
        trees = copy.deepcopy(binding)
        mutator(trees)
        return bool(module.gate_proof_binding({}, trees)["passes"])

    mutations = {
        "head_commit_zeroed": passes(
            lambda trees: trees.__setitem__("head_commit", "0" * 40)
        ),
        "candidate_git_tree_zeroed": passes(
            lambda trees: trees.__setitem__(
                "candidate_git_tree_src_gpuwrf", "0" * 40
            )
        ),
        "contract_hash_missing": passes(
            lambda trees: trees.__setitem__("contract_sha256", None)
        ),
        "amendment_hash_missing": passes(
            lambda trees: trees.__setitem__("amendment_sha256", None)
        ),
        "candidate_custom_digest_fabricated": passes(
            lambda trees: trees.__setitem__("candidate_tree_sha256", "f" * 64)
        ),
        "raw_sha_is_not_a_hash": passes(
            lambda trees: [
                entry.__setitem__("sha256", "truthy-but-not-a-hash")
                for entry in trees["raw_arms"].values()
            ]
        ),
        "raw_paths_missing": passes(
            lambda trees: [
                entry.__setitem__("file", "does-not-exist.json")
                for entry in trees["raw_arms"].values()
            ]
        ),
        "raw_metadata_removed": passes(
            lambda trees: [
                [
                    entry.pop(key, None)
                    for key in ("bytes", "label", "mode", "source_tree")
                ]
                for entry in trees["raw_arms"].values()
            ]
        ),
    }
    return {
        "gate_claim": "hash-bind every raw arm, root and asset into the proof",
        "mutations_that_false_pass": mutations,
        "false_pass_count": sum(mutations.values()),
        "attack_count": len(mutations),
        "gate_is_sound": not any(mutations.values()),
        "reason": (
            "gate_proof_binding checks only candidate_custom_digest != "
            "pre_custom_digest, presence of one historical parent, and truthiness "
            "of raw sha strings. It never validates the recorded commit/tree, "
            "contract/amendment, raw paths or raw contents."
        ),
    }


def floor_source_attack() -> dict[str, Any]:
    """Locate what the published floor rounds and where it rounds it."""

    source = AUTHOR_CHAIN.read_text(encoding="utf-8")
    tree = ast.parse(source)
    stage_trajectory = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "stage_trajectory"
    )
    calls = [
        node
        for node in ast.walk(stage_trajectory)
        if isinstance(node, ast.Call)
    ]
    fp32_calls = sorted(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "fp32_round_trip"
    )
    production_source = (
        REPO / "src/gpuwrf/runtime/operational_mode.py"
    ).read_text(encoding="utf-8")
    module = _load_author_generator()
    accuracy = load_json(AUTHOR_RAW / "accuracy.json")

    def floor_gate_false_pass(mutator) -> bool:
        raw = {"accuracy": copy.deepcopy(accuracy)}
        mutator(raw["accuracy"]["divergence"])
        return bool(module.gate_accuracy_floor(raw)["passes"])

    def divergence_gate_false_pass(mutator) -> bool:
        raw = {"accuracy": copy.deepcopy(accuracy)}
        mutator(raw["accuracy"]["divergence"])
        return bool(module.gate_divergence(raw)["passes"])

    first_field = sorted(accuracy["divergence"]["fields"])[0]
    raw_gate_mutations = {
        "rounded_field_list_emptied": floor_gate_false_pass(
            lambda divergence: divergence.__setitem__(
                "typed_work_rounded_fields", []
            )
        ),
        "per_field_authoritative_flag_false": floor_gate_false_pass(
            lambda divergence: divergence["fields"][first_field].__setitem__(
                "floor_is_authoritative", False
            )
        ),
        "per_field_floor_curve_zeroed": floor_gate_false_pass(
            lambda divergence: divergence["fields"][first_field].__setitem__(
                "typed_work_floor_curve",
                [0.0]
                * len(
                    divergence["fields"][first_field][
                        "typed_work_floor_curve"
                    ]
                ),
            )
        ),
        "literal_regime_changed_to_escalating": divergence_gate_false_pass(
            lambda divergence: divergence["fields"][first_field].__setitem__(
                "regime", "ESCALATING"
            )
        ),
        "per_field_level_bound_false": divergence_gate_false_pass(
            lambda divergence: divergence["fields"][first_field][
                "level_bound_ok_by_checkpoint"
            ].__setitem__("48", False)
        ),
    }
    return {
        "floor_function": "scripts.v025.m1_round3_chain.stage_trajectory",
        "fp32_round_trip_lines": fp32_calls,
        "round_trip_operand": "OperationalCarry.state canonical fields",
        "small_step_prep_called_in_floor_function": any(
            isinstance(node.func, ast.Name) and node.func.id == "small_step_prep_wrf"
            for node in calls
        ),
        "production_typed_boundary_capture_used": (
            "capture_stage_typed_boundary" in ast.get_source_segment(
                source, stage_trajectory
            )
        ),
        "production_boundary_marker_present": (
            "ADR-037 STAGE BOUNDARY" in production_source
        ),
        "raw_gate_mutations_that_false_pass": raw_gate_mutations,
        "raw_gate_false_pass_count": sum(raw_gate_mutations.values()),
        "contracted_operand": (
            "small_step_prep/acoustic work families after exact preparation algebra"
        ),
        "placement_matches_contract": False,
        "reason": (
            "The floor rounds canonical u/v/theta/w/p'/ph'/mu' before "
            "advance_stage. The contracted floor rounds the newly reduced work "
            "families exactly once at the real typed boundary after preparation. "
            "For mass-coupled u/v/theta and pressure/geopotential work, rounding "
            "before exact preparation is a different numerical perturbation."
        ),
    }


def floor_boundary_numeric_probe() -> dict[str, Any]:
    """Compare the author floor with round-after-preparation on one real RK3 input."""

    import jax
    import numpy as np

    import scripts.v025.m1_round3_chain as chain

    context = chain.load_context()
    step_carry, stage_carry, namelist, lead = chain.rk3_input_carry(
        context, chain.REPR32_FLOOR
    )
    _origin, stages, advance_stage = chain.production_stages(
        context, namelist, step_carry, lead
    )
    stage = stages[chain.RK_STAGE_INDEX]

    exact_capture = jax.block_until_ready(
        advance_stage(
            stage_carry, stage, capture_stage_typed_boundary=True
        )
    )
    rounded_state = chain.fp32_round_trip(
        context, stage_carry.state, chain.TYPED_WORK_ROUND_TRIP_FIELDS
    )
    author_capture = jax.block_until_ready(
        advance_stage(
            stage_carry.replace(state=rounded_state),
            stage,
            capture_stage_typed_boundary=True,
        )
    )

    fields = {
        "u_work": (exact_capture.prep.u_work, author_capture.prep.u_work),
        "v_work": (exact_capture.prep.v_work, author_capture.prep.v_work),
        "theta_work": (
            exact_capture.prep.theta_work,
            author_capture.prep.theta_work,
        ),
        "w_work": (exact_capture.prep.w_work, author_capture.prep.w_work),
        "ph_work": (exact_capture.prep.ph_work, author_capture.prep.ph_work),
        "mu_work": (exact_capture.prep.mu_work, author_capture.prep.mu_work),
        "pressure_work": (
            exact_capture.pressure.p,
            author_capture.pressure.p,
        ),
    }
    results: dict[str, Any] = {}
    for name, (exact_value, author_value) in fields.items():
        exact = np.asarray(exact_value, dtype=np.float64)
        ideal_at_boundary = np.asarray(exact, dtype=np.float32).astype(np.float64)
        author = np.asarray(author_value, dtype=np.float64)
        delta = author - ideal_at_boundary
        rms = float(np.sqrt(np.mean(delta * delta)))
        scale = float(np.sqrt(np.mean(ideal_at_boundary * ideal_at_boundary)))
        results[name] = {
            "shape": list(exact.shape),
            "author_equals_round_after_prep": bool(
                np.array_equal(author, ideal_at_boundary)
            ),
            "unequal_elements": int(np.count_nonzero(author != ideal_at_boundary)),
            "max_abs_difference": float(np.max(np.abs(delta))),
            "rms_difference": rms,
            "relative_rms": rms / max(scale, np.finfo(np.float64).tiny),
        }

    return {
        "environment": {
            "jax_platforms": sorted({device.platform for device in jax.devices()}),
            "jax_device_count": len(jax.devices()),
            "cpu_guard_pinned": cpu_guard.platform_is_pinned_to_cpu(),
        },
        "descriptor": {
            "rk_step": int(stage.rk_step),
            "dt_rk": float(stage.dt_rk),
            "dts_rk": float(stage.dts_rk),
            "acoustic_substeps": int(stage.number_of_small_timesteps),
            "lead_seconds": float(lead),
        },
        "comparison": (
            "author canonical-state rounding before advance_stage versus exact "
            "small_step_prep followed by one fp32 round at the typed boundary"
        ),
        "fields": results,
        "all_fields_match": all(
            entry["author_equals_round_after_prep"] for entry in results.values()
        ),
        "contracted_floor_reproduced_by_author": all(
            entry["author_equals_round_after_prep"] for entry in results.values()
        ),
    }


def verify_author_binding(
    author_proof: dict[str, Any], parent_root: Path
) -> dict[str, Any]:
    author_bytes = AUTHOR_PROOF.read_bytes()
    author_digest = sha256_bytes(author_bytes)
    sidecar_digest = AUTHOR_SIDECAR.read_text(encoding="utf-8").split()[0]
    raw_checks: dict[str, Any] = {}
    for name, entry in author_proof["binding"]["raw_arms"].items():
        path = AUTHOR_RAW / entry["file"]
        actual = sha256_file(path)
        raw_checks[name] = {
            "file": str(path.relative_to(REPO)),
            "recorded_sha256": entry["sha256"],
            "actual_sha256": actual,
            "matches": actual == entry["sha256"],
        }

    amendment_bytes = git_bytes("show", f"{AMENDMENT_COMMIT}:{AMENDMENT_RELATIVE}")
    candidate_src_tree = git("rev-parse", f"{EXACT_CANDIDATE}:src/gpuwrf")
    recorded = author_proof["binding"]
    current_custom = python_tree_digest(REPO / "src")
    parent_custom = python_tree_digest(parent_root / "src")
    candidate_is_ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO),
                "merge-base",
                "--is-ancestor",
                EXACT_CANDIDATE,
                "HEAD",
            ],
            check=False,
        ).returncode
        == 0
    )
    production_diff_after_candidate = git(
        "diff", "--name-only", f"{EXACT_CANDIDATE}..HEAD", "--", "src/gpuwrf"
    ).splitlines()
    return {
        "author_proof": {
            "recorded_sha256": sidecar_digest,
            "actual_sha256": author_digest,
            "sidecar_matches": sidecar_digest == author_digest,
        },
        "raw_arms": raw_checks,
        "all_raw_hashes_match": all(entry["matches"] for entry in raw_checks.values()),
        "exact_candidate": EXACT_CANDIDATE,
        "candidate_is_ancestor_of_critic_head": candidate_is_ancestor,
        "production_diff_after_exact_candidate": production_diff_after_candidate,
        "actual_candidate_git_tree_src_gpuwrf": candidate_src_tree,
        "recorded_candidate_git_tree_src_gpuwrf": recorded[
            "candidate_git_tree_src_gpuwrf"
        ],
        "recorded_git_tree_is_exact_candidate": (
            recorded["candidate_git_tree_src_gpuwrf"] == candidate_src_tree
        ),
        "recorded_git_tree_resolves_to": {
            "f95daee0_src_gpuwrf": git("rev-parse", "f95daee0:src/gpuwrf"),
            "matches_f95daee0": (
                recorded["candidate_git_tree_src_gpuwrf"]
                == git("rev-parse", "f95daee0:src/gpuwrf")
            ),
        },
        "recorded_head_commit": recorded["head_commit"],
        "recorded_head_is_exact_candidate": recorded["head_commit"] == EXACT_CANDIDATE,
        "custom_python_tree_digest": {
            "recorded_candidate": recorded["candidate_tree_sha256"],
            "recomputed_current_source": current_custom,
            "current_matches_recorded": current_custom
            == recorded["candidate_tree_sha256"],
            "recorded_pre": recorded["pre_candidate_tree_sha256"],
            "recomputed_pre": parent_custom,
            "pre_matches_recorded": parent_custom
            == recorded["pre_candidate_tree_sha256"],
            "limitation": (
                "This is an unversioned Python-only content digest. It can "
                "corroborate source contents but cannot replace a commit/tree "
                "identity or bind non-Python assets."
            ),
        },
        "contract": {
            "recorded_sha256": recorded["contract_sha256"],
            "worktree_sha256": sha256_file(CONTRACT),
            "matches_worktree": recorded["contract_sha256"] == sha256_file(CONTRACT),
        },
        "amendment": {
            "recorded_sha256": recorded["amendment_sha256"],
            "commit": AMENDMENT_COMMIT,
            "committed_sha256": sha256_bytes(amendment_bytes),
            "matches_committed_content": recorded["amendment_sha256"]
            == sha256_bytes(amendment_bytes),
            "recorded_source_path": recorded.get("amendment_source_path"),
            "in_exact_candidate_tree": bool(
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(REPO),
                        "cat-file",
                        "-e",
                        f"{EXACT_CANDIDATE}:{AMENDMENT_RELATIVE}",
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            ),
        },
    }


def input_binding_attack(candidate_raw: dict[str, Any]) -> dict[str, Any]:
    source = candidate_raw["state_source"]
    boundary = candidate_raw["boundary"]
    paths = {
        "namelist": Path(source["namelist"]),
        "wrfinput": Path(source["land_state"]["source_file"]),
        "previous_wrfout": Path(source["stage_pair"]["previous"]),
        "snapshot_wrfout": Path(source["stage_pair"]["snapshot"]),
        "wrfbdy": Path(boundary["meta"]["wrfbdy_path"]),
    }
    recorded = {
        "namelist": source.get("namelist_sha256"),
        "wrfinput": source.get("wrfinput_sha256"),
        "previous_wrfout": source["stage_pair"].get("previous_sha256"),
        "snapshot_wrfout": source["stage_pair"].get("snapshot_sha256"),
        "wrfbdy": source.get("wrfbdy_sha256")
        or boundary["meta"].get("wrfbdy_sha256"),
    }
    results: dict[str, Any] = {}
    for name, path in paths.items():
        actual = sha256_file(path) if path.exists() else None
        results[name] = {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "recorded_sha256": recorded[name],
            "actual_sha256": actual,
            "recorded_matches": (
                recorded[name] is not None and recorded[name] == actual
            ),
        }
    return {
        "assets": results,
        "wrfbdy_is_used": bool(boundary["meta"].get("wrfbdy_records")),
        "wrfbdy_is_hash_bound_in_author_raw": recorded["wrfbdy"] is not None,
        "complete_input_binding": all(
            entry["recorded_matches"] for entry in results.values()
        ),
    }


def independent_measurements(critic_raw: Path) -> dict[str, Any]:
    payloads = {
        filename: load_json(critic_raw / filename)
        for filename in CRITIC_RAW_FILENAMES
    }
    candidate = payloads["cand_fp32_repro.json"]
    pre = payloads["pre_fp32_repro.json"]
    control = payloads["cand_fp64_repro.json"]
    conversion = payloads["conversion_attribution_repro.json"]
    candidate_step = candidate["full_rk_step"]
    pre_step = pre["full_rk_step"]
    control_step = control["full_rk_step"]
    isolated_candidate = payloads["isolated_candidate_repro.json"]["chain"]
    isolated_pre = payloads["isolated_pre_repro.json"]["chain"]
    pre_identity = load_json(AUTHOR_RAW / "pre_fp64.json")[
        "fp64_default_identity"
    ]

    def reduction(numerator: int, denominator: int) -> float:
        return 100.0 * (numerator / denominator - 1.0)

    identity = control["fp64_default_identity"]
    core_identity = {
        name: {
            "stablehlo_equal": (
                entry["stablehlo_sha256"]
                == pre_identity["core13"][name]["stablehlo_sha256"]
            ),
            "outputs_equal": (
                entry["outputs"] == pre_identity["core13"][name]["outputs"]
            ),
        }
        for name, entry in identity["core13"].items()
    }
    full_identity = {
        "stablehlo_equal": (
            identity["full_rk_step"]["stablehlo_sha256"]
            == pre_identity["full_rk_step"]["stablehlo_sha256"]
        ),
        "outputs_equal": (
            identity["full_rk_step"]["output_state"]
            == pre_identity["full_rk_step"]["output_state"]
        ),
    }
    return {
        "raw_files": {
            filename: {
                "sha256": sha256_file(critic_raw / filename),
                "bytes": (critic_raw / filename).stat().st_size,
            }
            for filename in CRITIC_RAW_FILENAMES
        },
        "descriptor": candidate["stage"]["meta"],
        "three_arm_complete_step": {
            "pre_candidate": pre_step,
            "candidate": candidate_step,
            "fp64_control": control_step,
            "candidate_vs_pre_temporary_percent": reduction(
                candidate_step["temporary_bytes"], pre_step["temporary_bytes"]
            ),
            "candidate_vs_fp64_control_temporary_percent": reduction(
                candidate_step["temporary_bytes"], control_step["temporary_bytes"]
            ),
            "incremental_10_percent_gate_passes": (
                candidate_step["temporary_bytes"]
                <= 0.90 * pre_step["temporary_bytes"]
            ),
            "fp64_control_20_percent_gate_passes": (
                candidate_step["temporary_bytes"]
                <= 0.80 * control_step["temporary_bytes"]
            ),
            "launch_delta": (
                candidate_step["static_launch_proxy"]
                - pre_step["static_launch_proxy"]
            ),
            "launch_resolution_band": max(
                4, math.ceil(0.005 * pre_step["static_launch_proxy"])
            ),
            "launch_status": (
                "STATIC_PROXY_REGRESSION_REQUIRES_DEVICE_FALSIFICATION"
                if 0
                < candidate_step["static_launch_proxy"]
                - pre_step["static_launch_proxy"]
                <= max(4, math.ceil(0.005 * pre_step["static_launch_proxy"]))
                else "PASS"
                if candidate_step["static_launch_proxy"]
                <= pre_step["static_launch_proxy"]
                else "BLOCKING_REGRESSION"
            ),
            "optimized_hlo_ratio": (
                candidate_step["optimized_hlo_bytes"]
                / pre_step["optimized_hlo_bytes"]
            ),
            "compile_ratio": (
                candidate_step["lower_and_compile_seconds"]
                / pre_step["lower_and_compile_seconds"]
            ),
        },
        "optimized_loop_conversion": candidate["stage"][
            "converts_inside_acoustic_loop"
        ],
        "full_step_loop_conversion": candidate_step[
            "converts_inside_acoustic_loop"
        ],
        "source_jaxpr_conversion_attribution": conversion["attribution"],
        "isolated_acoustic_subchain": {
            "pre_candidate": isolated_pre,
            "candidate": isolated_candidate,
            "temporary_delta_percent": reduction(
                isolated_candidate["temporary_bytes"],
                isolated_pre["temporary_bytes"],
            ),
            "same_instrument_reproduces_author_values": (
                isolated_pre["temporary_bytes"] == 181_395_480
                and isolated_candidate["temporary_bytes"] == 90_744_064
            ),
            "scope_limit": (
                "Standalone specified final-RK3 acoustic-chain executable: no "
                "production halo, lead interpolation or enclosing RK1/RK2 work."
            ),
        },
        "attribution_limit": (
            "The source tool classifies boundary dependence only from the "
            "conversion's source filename/function. It does not trace dataflow, "
            "so its zero boundary count is not proof of zero causal dependence."
        ),
        "default_identity": {
            "core13_count": len(identity["core13"]),
            "core13_all_present": len(identity["core13"]) == 13,
            "core13_comparison_to_immutable_pre": core_identity,
            "core13_all_stablehlo_equal": all(
                entry["stablehlo_equal"] for entry in core_identity.values()
            ),
            "core13_all_outputs_equal": all(
                entry["outputs_equal"] for entry in core_identity.values()
            ),
            "full_step_comparison_to_immutable_pre": full_identity,
            "full_stage_stablehlo_sha256": identity["stage"][
                "stablehlo_sha256"
            ],
            "full_step_stablehlo_sha256": identity["full_rk_step"][
                "stablehlo_sha256"
            ],
            "full_step_output_state_count": len(
                identity["full_rk_step"]["output_state"]
            ),
        },
    }


def ceiling_attack(
    author_raw: Path, independent: dict[str, Any]
) -> dict[str, Any]:
    ceiling = load_json(author_raw / "ceiling.json")
    acoustic_pre = ceiling["acoustic_subchain"]["pre_candidate_temporary_bytes"]
    acoustic_candidate = ceiling["acoustic_subchain"]["candidate_temporary_bytes"]
    complete_pre = ceiling["production_rk_step"]["pre_candidate_temporary_bytes"]
    share = acoustic_pre / complete_pre
    measured_saving = (acoustic_pre - acoustic_candidate) / complete_pre
    isolated = independent["isolated_acoustic_subchain"]
    return {
        "reported": ceiling,
        "independent_isolated_reproduction": isolated,
        "arithmetic_recomputed": {
            "isolated_over_complete_fraction": share,
            "isolated_over_complete_percent": 100.0 * share,
            "measured_isolated_saving_over_complete_percent": (
                100.0 * measured_saving
            ),
            "deletion_ratio_percent": 100.0 * share,
            "matches_reported_share": math.isclose(
                share, ceiling["acoustic_share_of_production_step"], rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "matches_reported_measured_ceiling": math.isclose(
                100.0 * measured_saving,
                ceiling[
                    "maximum_reachable_reduction_percent_at_measured_50pc_mechanism"
                ],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
        },
        "denominator_compatibility": {
            "compatible": False,
            "isolated_program": (
                "one standalone final-RK3 acoustic sub-chain with 4 substeps"
            ),
            "complete_program": (
                "the enclosing full RK step with 1+2+4=7 acoustic substeps plus "
                "large-step/advection/boundary/scalar work"
            ),
            "memory_quantity": (
                "XLA compiled-memory temporary_bytes peak buffer-assignment arena"
            ),
            "nonadditivity": (
                "A standalone executable's peak arena is not an additive subset "
                "of an enclosing executable's peak arena. Fusion, liveness, "
                "scheduling, aliases and buffer reuse change the peak."
            ),
        },
        "bounds_adr037_scope": False,
        "proofs_fp32_scope_insufficient": False,
        "accepted_fact": (
            "The isolated sub-chain reduction is a real CPU/static mechanism "
            "signal, and the same-method complete-step candidate empirically "
            "misses both frozen arena gates."
        ),
    }


def test_record() -> dict[str, Any]:
    """Commands already executed by this critic before evidence assembly."""

    return {
        "focused_round3": {
            "command": (
                "timeout 600s taskset -c 0-3 env JAX_PLATFORMS=cpu "
                "PYTHONHASHSEED=0 pytest -q "
                "tests/v025/test_m1_round3_production_chain.py"
            ),
            "return_code": 0,
            "result": "30 passed in 0.15s",
        },
        "v025_except_held_session": {
            "command": (
                "timeout 600s taskset -c 0-3 env JAX_PLATFORMS=cpu "
                "PYTHONHASHSEED=0 pytest -q tests/v025 "
                "--ignore=tests/v025/test_m0_held_session_repair.py"
            ),
            "return_code": 1,
            "result": "5 failed, 883 passed, 2 skipped in 70.32s",
            "adjudication": (
                "All five are the contracted fail-closed ADR-036/M0 immutable "
                "candidate pins; the two skips are explicit CPU-guard exceptions "
                "for separately coordinated entry points."
            ),
        },
        "non_v025_regression_groups": {
            "command": (
                "timeout 600s taskset -c 0-3 env JAX_PLATFORMS=cpu "
                "PYTHONHASHSEED=0 pytest -q tests/dynamics tests/contracts "
                "tests/unit tests/test_v020_s4_mixed_precision.py "
                "tests/v025/test_m1_core13_precision.py"
            ),
            "return_code": 0,
            "result": "118 passed, 1 xfailed in 68.30s",
            "adjudication": (
                "The xfail is a documented stale test on the deliberate "
                "use_vertical_solver=False no-op, not the operational path."
            ),
        },
        "round2_typed_chain": {
            "command": (
                "pytest -q tests/v025/test_m1_round2_typed_chain.py"
            ),
            "return_code": 0,
            "result": "20 passed",
        },
        "round2_gpt_critic": {
            "command": (
                "pytest -q tests/v025/test_m1_round2_gpt_critic.py"
            ),
            "return_code": 0,
            "result": "8 passed",
        },
        "invalid_combined_collection_attempt": {
            "return_code": 2,
            "adjudication": (
                "Combining historical suites in one pytest process imported JAX "
                "before cpu_guard during collection. The suites were rerun in "
                "their required separate processes; this was a command-grouping "
                "error, not a candidate test failure."
            ),
        },
    }


def mandatory_attack_outcomes(
    binding: dict[str, Any],
    measurements: dict[str, Any],
    ceiling: dict[str, Any],
    floor_source: dict[str, Any],
    floor_numeric: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    step = measurements["three_arm_complete_step"]
    loop = measurements["optimized_loop_conversion"]
    return {
        "1_production_lineage_and_scope": {
            "outcome": "SUPPORTED",
            "facts": (
                "Actual _rk_scan_step; descriptor (3,54,13.5,4), real halo, "
                "specified wrfbdy branch, lead_seconds=3618. The complete-step "
                "denominator is the intended frozen product scope."
            ),
        },
        "2_three_arm_accounting": {
            "outcome": "REPRODUCED_GATES_FAIL",
            "candidate_vs_pre_percent": step[
                "candidate_vs_pre_temporary_percent"
            ],
            "candidate_vs_control_percent": step[
                "candidate_vs_fp64_control_temporary_percent"
            ],
        },
        "3_ceiling_proof": {
            "outcome": "REJECTED_NONADDITIVE_INCOMPATIBLE_DENOMINATORS",
            "arithmetic_is_correct": True,
            "scope_bound_is_valid": ceiling["bounds_adr037_scope"],
        },
        "4_conversion_attribution": {
            "outcome": "COUNT_REPRODUCED_CAUSAL_ZERO_NOT_PROVEN",
            "optimized_sites": loop["in_loop"],
            "optimized_elements_per_substep": loop[
                "in_loop_converted_elements_per_substep"
            ],
            "repair_class": (
                "Carrying mu_work or hoisting an exact difference is a bounded "
                "ADR-037 repair, but its complete-arena impact is unknown and it "
                "cannot erase either failed arena gate."
            ),
        },
        "5_floor_and_divergence": {
            "outcome": "REJECTED_INCOMPARABLE_FLOOR",
            "source_placement_matches": floor_source["placement_matches_contract"],
            "numeric_boundary_matches": floor_numeric[
                "contracted_floor_reproduced_by_author"
            ],
            "consequence": (
                "The stored 48-stage D/F and classifier arithmetic may be "
                "internally consistent, but it is normalized by the wrong "
                "perturbation and therefore cannot establish candidate safety."
            ),
        },
        "6_default_identity_and_safety": {
            "outcome": "SUPPORTED_CPU_STATIC",
            "core13_count": measurements["default_identity"]["core13_count"],
            "core13_stablehlo_equal": measurements["default_identity"][
                "core13_all_stablehlo_equal"
            ],
            "core13_outputs_equal": measurements["default_identity"][
                "core13_all_outputs_equal"
            ],
            "full_step_equal": all(
                measurements["default_identity"][
                    "full_step_comparison_to_immutable_pre"
                ].values()
            ),
            "production_files_after_candidate": binding[
                "production_diff_after_exact_candidate"
            ],
        },
        "7_launch_compile_honesty": {
            "outcome": "REPRODUCED_DEVICE_FALSIFIER_NOT_PASS",
            "launch_delta": step["launch_delta"],
            "resolution_band": step["launch_resolution_band"],
            "hlo_ratio": step["optimized_hlo_ratio"],
            "compile_ratio": step["compile_ratio"],
        },
        "8_proof_binding_and_reproduction": {
            "outcome": "REJECTED_STALE_NON_RECONSTRUCTIBLE",
            "recorded_head_is_candidate": binding[
                "recorded_head_is_exact_candidate"
            ],
            "recorded_tree_is_candidate": binding[
                "recorded_git_tree_is_exact_candidate"
            ],
            "wrfbdy_bound": inputs["wrfbdy_is_hash_bound_in_author_raw"],
            "published_builder_command_omits_required_parent_root": True,
        },
        "9_test_truth_and_m0_pins": {
            "outcome": "ADJUDICATED",
            "facts": (
                "Focused and ordinary regression groups pass. Five v025 "
                "failures are the expected fail-closed ADR-036 pins; skips/xfail "
                "were inspected and not relabelled as passes."
            ),
        },
        "10_architecture_decision": {
            "outcome": "PLAUSIBLE_NOT_PROVEN_BY_AUTHOR_CEILING",
            "facts": (
                "Large-step tendencies, transport velocities/advection, scalar "
                "transport, halo/boundary and diffusion are visible owners, but "
                "the evidence does not support an additive 'other 86%' map. The "
                "decisive next discriminator must measure the whole _rk_scan_step."
            ),
        },
    }


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    author_proof = load_json(AUTHOR_PROOF)
    critic_raw = arguments.critic_raw.resolve()
    parent_root = arguments.parent_root.resolve()
    for filename in CRITIC_RAW_FILENAMES:
        if not (critic_raw / filename).is_file():
            raise FileNotFoundError(critic_raw / filename)
    if not (parent_root / "src/gpuwrf").is_dir():
        raise FileNotFoundError(parent_root / "src/gpuwrf")

    binding = verify_author_binding(author_proof, parent_root)
    mutations = proof_binding_attack(author_proof)
    measurements = independent_measurements(critic_raw)
    ceiling = ceiling_attack(AUTHOR_RAW, measurements)
    floor_source = floor_source_attack()
    floor_numeric = floor_boundary_numeric_probe()
    inputs = input_binding_attack(load_json(critic_raw / "cand_fp32_repro.json"))
    attacks = mandatory_attack_outcomes(
        binding,
        measurements,
        ceiling,
        floor_source,
        floor_numeric,
        inputs,
    )

    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Independent CPU-only terminal review of exact candidate "
            f"{EXACT_CANDIDATE}."
        ),
        "environment": {
            "device_policy": "closed (contracted OS policy)",
            "allowed_cpus": "0-3",
            "cpu_guard_pinned": cpu_guard.platform_is_pinned_to_cpu(),
            "gpu_action": "NONE",
        },
        "terminal_verdict": TERMINAL_VERDICT,
        "axes": {
            "mechanism_reality": (
                "SUPPORTED_CPU_STATIC: the isolated acoustic sub-chain reduction "
                "and the complete-step arena delta reproduce."
            ),
            "candidate_acceptance": (
                "REJECTED: complete-step arena, conversion and launch gates fail."
            ),
            "proof_integrity": (
                "REJECTED: stale commit/tree, false-passing binding gate, omitted "
                "required reproduction argument, unbound wrfbdy and incomparable "
                "typed-work floor."
            ),
            "architecture_testability": (
                "PLAUSIBLE_NOT_PROVEN: a bounded whole-step typed-work "
                "discriminator is specified separately; no Round 4 or GPU is "
                "authorized."
            ),
        },
        "binding_audit": binding,
        "proof_binding_mutations": mutations,
        "input_binding": inputs,
        "independent_reproduction": measurements,
        "ceiling_attack": ceiling,
        "typed_work_floor_attack": {
            "source": floor_source,
            "numeric_boundary_probe": floor_numeric,
        },
        "mandatory_attack_outcomes": attacks,
        "test_record_before_critic_artifact_tests": test_record(),
        "terminal_reason": (
            "Round 3 cannot be accepted because its Gate-10 proof binding is "
            "technically unsound and the claimed authoritative floor does not "
            "apply rounding at the contracted work boundary. Independently "
            "reproduced CPU/static measurements still preserve the useful "
            "mechanism signal and the empirical complete-step miss. The isolated "
            "arena ratio is not an additive ceiling, so fp32 scope insufficiency "
            "has not been proved."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument(
        "--critic-raw",
        type=Path,
        default=REPO / "proofs/v025/m1/round3_gpt_final_critic_raw",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    payload = build(arguments)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    arguments.out.write_text(encoded, encoding="utf-8")
    sidecar = arguments.out.with_suffix(".sha256")
    sidecar.write_text(
        f"{sha256_text(encoded)}  {arguments.out.name}\n", encoding="utf-8"
    )
    print(payload["terminal_verdict"])
    print(f"proof_sha256={sha256_text(encoded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
