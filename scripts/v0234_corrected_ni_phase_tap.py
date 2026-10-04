#!/usr/bin/env python3
"""One-shot same-input option-2 phase tap for corrected d03 step 9314.

The retained ordinary output is the A arm.  This runner compiles and executes
only the tapped B arm.  It does not transfer or decode the tap summary until the
complete B carry has matched the retained A carry byte-for-byte.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Mapping


REQUIRED_PREIMPORT_ENV = {
    "JAX_PLATFORMS": "cuda",
    "JAX_ENABLE_X64": "true",
    "CUDA_VISIBLE_DEVICES": "0",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "GPUWRF_ALLOCATOR": "cuda_async",
    "GPUWRF_FINITE_CHECK": "1",
    "GPUWRF_NESTED_FUSE": "0",
    "GPUWRF_NESTED_DEFUSE_COMPILE": "0",
    "GPUWRF_NESTED_PARALLEL_COMPILE": "0",
    "GPUWRF_NESTED_AOT": "0",
    "GPUWRF_AOT_VERIFY": "0",
    "GPUWRF_NESTED_ASYNC_OUTPUT": "0",
    "GPUWRF_NEST_OUTPUT_PIPELINE": "0",
    "GPUWRF_BATCH_ENSEMBLE": "1",
    "GPUWRF_NESTED_SYNC_MODE": "root",
    "GPUWRF_BITWISE": "1",
    "GPUWRF_ADVANCE_CHUNK_LOOP": "fori",
    "GPUWRF_JAX_CACHE": "1",
    "GPUWRF_JAX_CACHE_LOCK": "1",
}


def validate_preimport_environment(environment: Mapping[str, str]) -> dict[str, Any]:
    actual = {name: environment.get(name) for name in REQUIRED_PREIMPORT_ENV}
    if actual != REQUIRED_PREIMPORT_ENV:
        raise RuntimeError(
            "phase-tap environment differs from the frozen production lane: "
            f"expected={REQUIRED_PREIMPORT_ENV!r} actual={actual!r}"
        )
    cache = environment.get("GPUWRF_JAX_CACHE_DIR")
    if not cache or cache != environment.get("JAX_COMPILATION_CACHE_DIR"):
        raise RuntimeError("JAX cache paths must be equal, explicit, and non-empty")
    if not environment.get("GPUWRF_WRF_ROOT"):
        raise RuntimeError("GPUWRF_WRF_ROOT must bind the Retry20 WRF authority")
    return {
        "validated_before_jax_import": True,
        "required_environment": actual,
        "cache_dir": cache,
        "wrf_root": environment["GPUWRF_WRF_ROOT"],
    }


if __name__ == "__main__":
    _PREIMPORT_AUTHORITY = validate_preimport_environment(os.environ)


import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.core.acoustic import acoustic_substep_core
from gpuwrf.runtime.operational_mode import (
    PHASE_TAP_SCRATCH_FIELDS,
    PHASE_TAP_SUMMARY_METRICS,
    _rk_scan_step,
    advance_one_step_with_corrected_ni_phase_tap,
    build_clock_base,
)
from scripts import v0234_corrected_ni_ordinary_bisection as ordinary


SCHEMA = "gpuwrf.v0234.corrected-ni-phase-tap-option2.v1"
CONTINUATION_PARENT_SHA = "5baf399708fa318ca66f0a758302957e87007573"
NUMERICAL_BASE_SHA = "bb2ffe33dbff5e04246dc6903bf6aa26e5827163"
RUNTIME_PARENT_SHA = "16774bed70b26d465e3aa87e91060e6a17de7f92"
ORDINARY_PROOF = Path(
    ".agent/sprints/2026-07-13-v0234-corrected-ni-ordinary-bisection/"
    "ordinary-bisection-proof.json"
)
ORDINARY_PROOF_SHA256 = "237e6e59e3f080a28434b72e8e149eb85f2d032139c7f8d4524caee04ff5087b"
CHECKPOINT_DIR = (
    ordinary.LINEAGE_WORK_DIR / "ordinary_bisection_effe1f43/checkpoints"
)
INPUT_CHECKPOINT = CHECKPOINT_DIR / "last-finite-input-to-first-bad-d03-step-9313.pkl"
REFERENCE_CHECKPOINT = CHECKPOINT_DIR / "first-bad-output-d03-step-9314.pkl"
INPUT_FILE_SHA256 = "f82a25c35e6bd738cd248d759c291d825ca0cfc4e08753dc5082741a78b23fc7"
REFERENCE_FILE_SHA256 = "633d9b09b8d19084ab69b3d260abab7f5bffed1a22fb5fba4f51da583e47298f"
INPUT_MANIFEST_SHA256 = "bef61cfa3e91b9e1a4c50b21975f57e1cf08bf6678738a5bdd48397e5f1d24e8"
REFERENCE_MANIFEST_SHA256 = "60bba36797c09100bd86d09f885930cdb0f72d5be587d05692a998ef9080bfc7"
STEP_INDEX = 9314
EXPECTED_AFFINITY = [12, 13, 14, 15]


def git_output(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def assert_candidate_authority() -> dict[str, Any]:
    head = git_output("rev-parse", "HEAD")
    approved = os.environ.get("GPUWRF_CORRECTED_NI_PHASE_TAP_APPROVED_SHA")
    if approved != head:
        raise RuntimeError(
            "GPUWRF_CORRECTED_NI_PHASE_TAP_APPROVED_SHA must equal committed HEAD"
        )
    if git_output("status", "--porcelain"):
        raise RuntimeError("phase tap requires a clean committed launch worktree")
    for ancestor in (CONTINUATION_PARENT_SHA, NUMERICAL_BASE_SHA, RUNTIME_PARENT_SHA):
        subprocess.run(("git", "merge-base", "--is-ancestor", ancestor, head), check=True)
    source_rows = [
        (relative, ordinary.sha256_file(Path(relative)))
        for relative in git_output("ls-files", "src/gpuwrf").splitlines()
    ]
    return {
        "head": head,
        "approved_sha": approved,
        "continuation_parent_sha": CONTINUATION_PARENT_SHA,
        "numerical_base_sha": NUMERICAL_BASE_SHA,
        "runtime_parent_sha": RUNTIME_PARENT_SHA,
        "tracked_source_manifest_sha256": ordinary.canonical_digest(source_rows),
        "worktree_clean": True,
    }


def _checkpoint_object(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_manifest_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    digest = ordinary.sha256_file(path)
    if digest != expected_file_sha256:
        raise RuntimeError(f"retained checkpoint file changed: {path}")
    with path.open("rb") as handle:
        value = pickle.load(handle)
    manifest = ordinary.host_tree_manifest(value)
    if manifest["manifest_sha256"] != expected_manifest_sha256:
        raise RuntimeError(f"retained checkpoint carry manifest changed: {path}")
    return value, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": digest,
        "manifest_sha256": manifest["manifest_sha256"],
        "leaf_count": manifest["leaf_count"],
        "total_bytes": manifest["total_bytes"],
        "floating_nonfinite_count": manifest["floating_nonfinite_count"],
    }


def assert_checkpoint_authority() -> tuple[Any, Any, dict[str, Any]]:
    if ordinary.sha256_file(ORDINARY_PROOF) != ORDINARY_PROOF_SHA256:
        raise RuntimeError("ordinary localization proof changed")
    proof = json.loads(ORDINARY_PROOF.read_text())
    if (
        proof.get("verdict") != "ORDINARY_BISECTION_LOCALIZED"
        or proof["ordinary_replay"]["first_bad"]["native_step"] != STEP_INDEX
        or proof["retained_carries"]["last_finite"]["file_sha256"] != INPUT_FILE_SHA256
        or proof["retained_carries"]["first_bad"]["file_sha256"] != REFERENCE_FILE_SHA256
    ):
        raise RuntimeError("ordinary proof no longer authenticates step-9314 A/B objects")
    input_carry, input_row = _checkpoint_object(
        INPUT_CHECKPOINT,
        expected_file_sha256=INPUT_FILE_SHA256,
        expected_manifest_sha256=INPUT_MANIFEST_SHA256,
    )
    reference_carry, reference_row = _checkpoint_object(
        REFERENCE_CHECKPOINT,
        expected_file_sha256=REFERENCE_FILE_SHA256,
        expected_manifest_sha256=REFERENCE_MANIFEST_SHA256,
    )
    if input_row["floating_nonfinite_count"] != 0:
        raise RuntimeError("step-9313 input is no longer finite")
    if reference_row["floating_nonfinite_count"] != 9900:
        raise RuntimeError("ordinary step-9314 reference nonfinite count changed")
    if jax.tree_util.tree_structure(input_carry) != jax.tree_util.tree_structure(reference_carry):
        raise RuntimeError("retained input/reference carry structures differ")
    return input_carry, reference_carry, {
        "ordinary_proof": {
            "path": str(ORDINARY_PROOF.resolve()),
            "sha256": ORDINARY_PROOF_SHA256,
            "payload_sha256": proof["proof_sha256"],
        },
        "input": input_row,
        "ordinary_reference": reference_row,
    }


def audit_tap_source() -> dict[str, Any]:
    core = inspect.getsource(acoustic_substep_core)
    rk = inspect.getsource(_rk_scan_step)
    public = inspect.getsource(advance_one_step_with_corrected_ni_phase_tap)
    mu_halo = core.find("state_for_w = _maybe_exchange_sharded_acoustic_halos(state_for_w)")
    mu_summary = core.find("post_advance_mu_t_summary =")
    ph_boundary = core.find("ph_next = spec_bdyupdate_ph_inloop(")
    w_boundary = core.find("w_solved = _specified_w_zero_grad_work(")
    pressure_halo = core.find(
        "state_for_pressure = _maybe_exchange_sharded_acoustic_halos(state_for_pressure)"
    )
    w_summary = core.find("post_advance_w_summary =")
    pressure = core.find("p_rho = calc_p_rho_step(")
    ordered = bool(
        0 <= mu_halo < mu_summary < ph_boundary < w_boundary < pressure_halo < w_summary < pressure
    )
    rk_anchor = (
        "carry = advance_stage(carry, stages[0])\n"
        "        carry = advance_stage(carry, stages[1])\n"
        "        return advance_stage(carry, stages[2], capture_stage_phase_tap=True)"
    )
    ordinary_operators = "observe_mass_primitive=True" not in core[mu_summary:w_summary]
    public_single_step = public.count("_physics_boundary_step_with_phase_tap(") == 1
    result = {
        "acoustic_source_sha256": sha256_text(core),
        "rk_source_sha256": sha256_text(rk),
        "public_source_sha256": sha256_text(public),
        "boundary_order_exact": ordered,
        "rk3_substep1_only_anchor": rk_anchor in rk,
        "ordinary_operator_interval": ordinary_operators,
        "public_single_step_call": public_single_step,
        "scratch_fields": list(PHASE_TAP_SCRATCH_FIELDS),
        "summary_metrics": list(PHASE_TAP_SUMMARY_METRICS),
        "bounded_scalar_count": 2 * len(PHASE_TAP_SCRATCH_FIELDS) * len(PHASE_TAP_SUMMARY_METRICS),
    }
    if not all(
        result[name]
        for name in (
            "boundary_order_exact",
            "rk3_substep1_only_anchor",
            "ordinary_operator_interval",
            "public_single_step_call",
        )
    ) or result["bounded_scalar_count"] != 40:
        raise RuntimeError(f"phase-tap source structure audit failed: {result}")
    return result


def _avals(value: Any) -> list[dict[str, Any]]:
    return [
        {"shape": list(leaf.shape), "dtype": str(leaf.dtype)}
        for leaf in jax.tree_util.tree_leaves(value)
    ]


def lower_and_compile_once(
    input_carry: Any,
    namelist: Any,
    clock_base: Any,
) -> tuple[Any, dict[str, Any]]:
    cadence = int(namelist.radiation_cadence_steps)
    lowered = advance_one_step_with_corrected_ni_phase_tap.lower(
        input_carry,
        namelist,
        jnp.asarray(STEP_INDEX, dtype=jnp.int32),
        clock_base,
        cadence=cadence,
    )
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    lower_calls = 1
    forbidden = [
        token
        for token in (
            "xla_python_cpu_callback",
            "host_callback",
            "io_callback",
            "pure_callback",
            "outside_compilation",
            "observe_mass_primitive",
            "advance_mu_t_wrf_observed",
            "corrected_ni_rca",
            "rca_acoustic",
        )
        if token in stablehlo.lower()
    ]
    input_structure = jax.tree_util.tree_structure(input_carry)
    carry_out_structure = jax.tree_util.tree_structure(lowered.out_info.carry)
    input_avals = _avals(input_carry)
    carry_out_avals = _avals(lowered.out_info.carry)
    summary_avals = _avals(lowered.out_info.summary)
    carry_structure_equal = bool(
        input_structure == carry_out_structure and input_avals == carry_out_avals
    )
    bounded_summary = summary_avals == [
        {"shape": [5, 4], "dtype": "float64"},
        {"shape": [5, 4], "dtype": "float64"},
    ]
    if forbidden or not carry_structure_equal or not bounded_summary:
        raise RuntimeError(
            "phase-tap StableHLO/output audit failed: "
            f"forbidden={forbidden} carry_equal={carry_structure_equal} "
            f"summary={summary_avals}"
        )
    executable = lowered.compile()
    compile_calls = 1
    if executable.out_tree != jax.tree_util.tree_structure(lowered.out_info):
        raise RuntimeError("compiled tap output tree changed after lowering")
    return executable, {
        "callable": (
            "gpuwrf.runtime.operational_mode."
            "advance_one_step_with_corrected_ni_phase_tap"
        ),
        "step_index": STEP_INDEX,
        "cadence": cadence,
        "radiation_step": STEP_INDEX % cadence == 0,
        "lower_calls": lower_calls,
        "compile_calls": compile_calls,
        "stablehlo_sha256": sha256_text(stablehlo),
        "stablehlo_bytes": len(stablehlo.encode()),
        "forbidden_tokens_found": forbidden,
        "callback_free": not forbidden,
        "carry_input_output_structure_and_avals_identical": carry_structure_equal,
        "carry_leaf_count": len(input_avals),
        "summary_avals": summary_avals,
        "bounded_summary": bounded_summary,
        "bounded_scalar_count": 40,
    }


def _json_float(value: float) -> float | str:
    value = float(value)
    if np.isfinite(value):
        return value
    if np.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def decode_admissible_summary(summary: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for boundary_name in ("post_advance_mu_t", "post_advance_w"):
        values = np.asarray(getattr(summary, boundary_name), dtype=np.float64)
        if values.shape != (5, 4):
            raise RuntimeError(f"invalid tap summary shape at {boundary_name}: {values.shape}")
        fields = []
        for field_index, field_name in enumerate(PHASE_TAP_SCRATCH_FIELDS):
            row = values[field_index]
            fields.append({
                "field": field_name,
                "nonfinite_count": int(row[0]),
                "first_nonfinite_flat_index": int(row[1]),
                "max_abs_finite": _json_float(row[2]),
                "max_abs_finite_flat_index": int(row[3]),
            })
        rows[boundary_name] = fields
    post_mu_bad = {
        row["field"]: row["nonfinite_count"]
        for row in rows["post_advance_mu_t"]
        if row["nonfinite_count"]
    }
    post_w_bad = {
        row["field"]: row["nonfinite_count"]
        for row in rows["post_advance_w"]
        if row["nonfinite_count"]
    }
    if post_mu_bad:
        interval = "advance_uv_through_post_advance_mu_t_ring_halo"
    elif post_w_bad:
        interval = "advance_w_through_ph_w_boundary_and_halo_before_calc_p_rho"
    else:
        interval = "authorized_two-boundary_hypothesis_falsified"
    return {
        "status": "ADMISSIBLE_IDENTITY_PASSED",
        "boundaries": rows,
        "post_advance_mu_t_nonfinite": post_mu_bad,
        "post_advance_w_nonfinite": post_w_bad,
        "earliest_interval": interval,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--proof-output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    proof_output = args.proof_output.resolve()
    if run_dir == ordinary.LINEAGE_WORK_DIR.resolve() or ordinary.LINEAGE_WORK_DIR.resolve() not in run_dir.parents:
        raise RuntimeError("--run-dir must be a fresh child of the existing RCA workdir")
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(run_dir)
    if proof_output.exists() or proof_output.is_symlink():
        raise FileExistsError(proof_output)
    if proof_output.parent != run_dir:
        raise RuntimeError("--proof-output must be a direct child of --run-dir")
    if Path(os.environ["GPUWRF_WRF_ROOT"]).resolve() != (
        ordinary.RETRY20_ROOT / "authority/wrf_root"
    ).resolve():
        raise RuntimeError("GPUWRF_WRF_ROOT is not the retained Retry20 WRF authority")
    if sorted(os.sched_getaffinity(0)) != EXPECTED_AFFINITY:
        raise RuntimeError(
            f"phase tap must be taskset to CPUs 12-15: {sorted(os.sched_getaffinity(0))}"
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    authority: dict[str, Any] = {
        "preimport": _PREIMPORT_AUTHORITY,
        "candidate": assert_candidate_authority(),
        "inputs_pre": ordinary.assert_input_authority(),
        "prior_pre": ordinary.assert_prior_authority(),
        "lock_pre": ordinary.assert_lock_authority(),
        "preemption_pre": ordinary.assert_preemption_clear("phase-tap-startup"),
        "cuda": ordinary.assert_cuda_runtime(),
        "process": {
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "run_dir": str(run_dir),
            "cache_dir": str(ordinary.LINEAGE_CACHE.resolve()),
        },
    }
    input_carry, reference_carry, checkpoint_authority = assert_checkpoint_authority()
    source_audit = audit_tap_source()

    tree, names, initial_carries, dt_by_domain, load_authority = ordinary.load_corrected_tree(run_dir)
    if names != ("d01", "d02", "d03") or dt_by_domain["d03"] != 6.0:
        raise RuntimeError("corrected d03 domain authority changed")
    namelist = tree.domains["d03"].namelist
    clock_base = build_clock_base(namelist)
    cadence = int(namelist.radiation_cadence_steps)
    if cadence != 300 or STEP_INDEX % cadence != 14:
        raise RuntimeError("step-9314 radiation cadence authority changed")
    del initial_carries, tree
    gc.collect()

    ordinary.assert_preemption_clear("phase-tap-pre-lower")
    executable, program_audit = lower_and_compile_once(input_carry, namelist, clock_base)
    ordinary.assert_preemption_clear("phase-tap-pre-call")
    tap_result = executable(
        input_carry,
        namelist,
        jnp.asarray(STEP_INDEX, dtype=jnp.int32),
        clock_base,
        cadence=cadence,
    )
    model_calls = 1

    # Materialize the complete carry only.  The summary remains device-resident
    # and scientifically unread until the identity comparison below is complete.
    tapped_host_carry = jax.device_get(tap_result.carry)
    tapped_manifest = ordinary.host_tree_manifest(tapped_host_carry)
    reference_manifest = ordinary.host_tree_manifest(reference_carry)
    identity = ordinary.compare_manifests(reference_manifest, tapped_manifest)
    exact_identity = bool(
        identity["all_leaf_bytes_equal"]
        and tapped_manifest["manifest_sha256"] == REFERENCE_MANIFEST_SHA256
    )

    if exact_identity:
        summary_host = jax.device_get(tap_result.summary)
        tap_science: dict[str, Any] = decode_admissible_summary(summary_host)
        interim_verdict = "PHASE_TAP_ADMISSIBLE"
    else:
        tap_science = {
            "status": "DISCARDED_UNREAD",
            "reason": (
                "complete tapped output carry differs from retained ordinary output; "
                "the device summary was not transferred or decoded"
            ),
        }
        interim_verdict = "NO_FIX_PHASE_TAP_IDENTITY_BLOCKED"

    checkpoint_post = {
        "input_file_sha256": ordinary.sha256_file(INPUT_CHECKPOINT),
        "reference_file_sha256": ordinary.sha256_file(REFERENCE_CHECKPOINT),
    }
    if checkpoint_post != {
        "input_file_sha256": INPUT_FILE_SHA256,
        "reference_file_sha256": REFERENCE_FILE_SHA256,
    }:
        raise RuntimeError("retained A/B checkpoint changed during phase tap")
    inputs_post = ordinary.assert_input_authority()
    if inputs_post["authority_sha256"] != authority["inputs_pre"]["authority_sha256"]:
        raise RuntimeError("corrected input authority changed during phase tap")
    authority.update({
        "inputs_post": inputs_post,
        "checkpoint_files_post": checkpoint_post,
        "lock_post": ordinary.assert_lock_authority(),
        "preemption_post": ordinary.assert_preemption_clear("phase-tap-complete"),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
    })

    proof: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "interim_verdict": interim_verdict,
        "authority": authority,
        "checkpoint_authority": checkpoint_authority,
        "load_authority": load_authority,
        "source_audit": source_audit,
        "program_audit": program_audit,
        "execution_audit": {
            "ordinary_reference_model_calls": 0,
            "tap_model_calls": model_calls,
            "tap_lower_calls": program_audit["lower_calls"],
            "tap_compile_calls": program_audit["compile_calls"],
            "model_call_scope": "exactly native d03 step 9314 from retained step-9313 carry",
            "summary_transferred_before_identity_gate": False,
        },
        "complete_carry_identity_gate": {
            "required_manifest_sha256": REFERENCE_MANIFEST_SHA256,
            "actual_manifest_sha256": tapped_manifest["manifest_sha256"],
            "passed": exact_identity,
            "comparison": identity,
            "ordinary_reference_manifest": reference_manifest,
            "tapped_output_manifest": tapped_manifest,
        },
        "tap_science": tap_science,
        "mechanism_separation": {
            "ni_track": "bounded step-9314 acoustic scratch phase tap",
            "v10_track": "OPEN_SEPARATE_NO_CAUSAL_LINK_FROM_THIS_EXPERIMENT",
            "common_root_proved": False,
        },
        "scope_gates": {
            "full_18h_run": False,
            "ordinary_reference_reexecuted": False,
            "recorder_on_reused": False,
            "model_or_numerical_fix_in_launch_commit": False,
            "tolerance_change": False,
            "clamp_or_sanitizer": False,
        },
    }
    proof["proof_sha256"] = ordinary.canonical_digest(proof)
    ordinary.atomic_write_json(proof_output, proof)
    print(
        json.dumps({
            "proof": str(proof_output),
            "proof_sha256": proof["proof_sha256"],
            "interim_verdict": interim_verdict,
            "required_manifest": REFERENCE_MANIFEST_SHA256,
            "actual_manifest": tapped_manifest["manifest_sha256"],
            "different_leaf_count": identity["different_leaf_count"],
            "tap_science_status": tap_science["status"],
        }, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return 0 if exact_identity else 3


if __name__ == "__main__":
    raise SystemExit(main())
