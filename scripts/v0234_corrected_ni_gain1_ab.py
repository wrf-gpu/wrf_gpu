#!/usr/bin/env python3
"""Non-observing same-carry strength-1 discriminator for corrected d03.

The retained ordinary strength-20 step-9314 carry is the A arm.  This runner
compiles only the unchanged ordinary B callable, executes step 9314 from the
authenticated step-9313 carry, evaluates host-side health after completion,
and executes step 9315 only when step 9314 is finite and physically bounded.
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


REQUIRED_ORDINARY_ENV = {
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
GAIN_ENV_NAME = "GPUWRF_NORMAL_BDY_RELAX_STRENGTH"
GAIN_ENV_VALUE = "1.0"
INFRASTRUCTURE_GPUWRF_ENV = {
    "GPUWRF_JAX_CACHE_DIR",
    "GPUWRF_WRF_ROOT",
    "GPUWRF_CORRECTED_NI_GAIN1_APPROVED_SHA",
    "GPUWRF_GPU_LOCK_HELD",
    "GPUWRF_GPU_LOCK_FD",
    "GPUWRF_GPU_LOCK_FILE",
    "GPUWRF_GPU_LOCK_HOLDER_FILE",
    "GPUWRF_GPU_LOCK_LABEL",
    "GPUWRF_GPU_LOCK_TOKEN",
}


def validate_preimport_environment(environment: Mapping[str, str]) -> dict[str, Any]:
    ordinary = {name: environment.get(name) for name in REQUIRED_ORDINARY_ENV}
    if ordinary != REQUIRED_ORDINARY_ENV:
        raise RuntimeError(
            "gain-1 environment differs from the frozen ordinary lane: "
            f"expected={REQUIRED_ORDINARY_ENV!r} actual={ordinary!r}"
        )
    if environment.get(GAIN_ENV_NAME) != GAIN_ENV_VALUE:
        raise RuntimeError(f"{GAIN_ENV_NAME} must be the exact string {GAIN_ENV_VALUE!r}")
    cache = environment.get("GPUWRF_JAX_CACHE_DIR")
    if not cache or cache != environment.get("JAX_COMPILATION_CACHE_DIR"):
        raise RuntimeError("JAX cache paths must be explicit, equal, and non-empty")
    if not environment.get("GPUWRF_WRF_ROOT"):
        raise RuntimeError("GPUWRF_WRF_ROOT must bind the Retry20 WRF authority")
    allowed = set(REQUIRED_ORDINARY_ENV) | INFRASTRUCTURE_GPUWRF_ENV | {GAIN_ENV_NAME}
    unexpected = sorted(
        name for name in environment if name.startswith("GPUWRF_") and name not in allowed
    )
    if unexpected:
        raise RuntimeError(f"unapproved GPUWRF environment overrides: {unexpected}")
    return {
        "validated_before_jax_or_gpuwrf_import": True,
        "ordinary_environment": ordinary,
        "only_numerical_environment_difference": {GAIN_ENV_NAME: GAIN_ENV_VALUE},
        "unexpected_gpuwrf_environment": unexpected,
        "cache_dir": cache,
        "wrf_root": environment["GPUWRF_WRF_ROOT"],
    }


_PREIMPORT_AUTHORITY: dict[str, Any] | None = None
if __name__ == "__main__":
    _PREIMPORT_AUTHORITY = validate_preimport_environment(os.environ)


import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling import boundary_apply
from gpuwrf.runtime.operational_mode import _advance_chunk_fori, build_clock_base
from scripts import v0234_corrected_ni_ordinary_bisection as ordinary


SCHEMA = "gpuwrf.v0234.corrected-ni-gain1-same-carry-ab.v1"
REPORT_COMMIT = "d9dad2d2c06caa266eb53ab73f74cba38a03038e"
NUMERICAL_BASE_SHA = "bb2ffe33dbff5e04246dc6903bf6aa26e5827163"
RUNTIME_PARENT_SHA = "16774bed70b26d465e3aa87e91060e6a17de7f92"
SPRINT_DIR = Path(".agent/sprints/2026-07-13-v0234-corrected-ni-gain1-ab")
REDESIGN_PROOF = Path(
    ".agent/sprints/2026-07-13-v0234-corrected-ni-phase-tap/"
    "identity-redesign-proof.json"
)
REDESIGN_PROOF_SHA256 = "49a7bbe94b71e4bacdf4904d14c6f47cf9a3120ed5494fcde99d394bdeae32a1"
REDESIGN_PAYLOAD_SHA256 = "d6c46caa07ef564bf4f485cc35ee322bedd01e778e093793a0d2195cda2e3369"
PHASE_FAILURE_PROOF = Path(
    ".agent/sprints/2026-07-13-v0234-corrected-ni-phase-tap/"
    "phase-tap-identity-failure-proof.json"
)
PHASE_FAILURE_PROOF_SHA256 = "5ea4ed1faf0f795ab72c0396a12bcf85bc37ee60f1b23af57f204fa2f18faa95"
ORDINARY_PROOF = Path(
    ".agent/sprints/2026-07-13-v0234-corrected-ni-ordinary-bisection/"
    "ordinary-bisection-proof.json"
)
ORDINARY_PROOF_SHA256 = "237e6e59e3f080a28434b72e8e149eb85f2d032139c7f8d4524caee04ff5087b"
ORDINARY_HLO_SHA256 = "24035858ab6c6f555d670a09491ac2737367741916f9edc51a00e5ec427dbfe3"
CHECKPOINT_DIR = ordinary.LINEAGE_WORK_DIR / "ordinary_bisection_effe1f43/checkpoints"
INPUT_CHECKPOINT = CHECKPOINT_DIR / "last-finite-input-to-first-bad-d03-step-9313.pkl"
REFERENCE_CHECKPOINT = CHECKPOINT_DIR / "first-bad-output-d03-step-9314.pkl"
INPUT_FILE_SHA256 = "f82a25c35e6bd738cd248d759c291d825ca0cfc4e08753dc5082741a78b23fc7"
REFERENCE_FILE_SHA256 = "633d9b09b8d19084ab69b3d260abab7f5bffed1a22fb5fba4f51da583e47298f"
INPUT_MANIFEST_SHA256 = "bef61cfa3e91b9e1a4c50b21975f57e1cf08bf6678738a5bdd48397e5f1d24e8"
REFERENCE_MANIFEST_SHA256 = "60bba36797c09100bd86d09f885930cdb0f72d5be587d05692a998ef9080bfc7"
EXPECTED_TERMINAL_MANIFEST = "2dcfc195baaf3e59ba539f6701fdb82b9a134a0461cfc68dec3a429e38433565"
EXPECTED_AFFINITY = [12, 13, 14, 15]
STEP_9314 = 9314
STEP_9315 = 9315
ABSOLUTE_SCALE_CEILING = 1.0e6
AMPLIFICATION_CEILING = 1000.0
STATE_HEALTH_FIELDS = (
    "u",
    "v",
    "w",
    "theta",
    "p_total",
    "p_perturbation",
    "ph_total",
    "ph_perturbation",
    "mu_total",
    "mu_perturbation",
)
SAVE_HEALTH_FIELDS = (
    "u_save",
    "v_save",
    "w_save",
    "t_save",
    "ph_save",
    "mu_save",
    "ww_save",
)
SCRATCH_HEALTH_FIELDS = ("t_2ave", "ww", "mudf", "muave", "muts")
ALL_SCALE_FIELDS = (
    *(f"state.{name}" for name in STATE_HEALTH_FIELDS),
    *SAVE_HEALTH_FIELDS,
    *SCRATCH_HEALTH_FIELDS,
)


def git_output(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def assert_candidate_authority() -> dict[str, Any]:
    head = git_output("rev-parse", "HEAD")
    approved = os.environ.get("GPUWRF_CORRECTED_NI_GAIN1_APPROVED_SHA")
    if approved != head:
        raise RuntimeError("GPUWRF_CORRECTED_NI_GAIN1_APPROVED_SHA must equal committed HEAD")
    if git_output("status", "--porcelain"):
        raise RuntimeError("gain-1 discriminator requires a clean committed launch worktree")
    for ancestor in (REPORT_COMMIT, NUMERICAL_BASE_SHA, RUNTIME_PARENT_SHA):
        subprocess.run(("git", "merge-base", "--is-ancestor", ancestor, head), check=True)
    source_rows = [
        (relative, ordinary.sha256_file(Path(relative)))
        for relative in git_output("ls-files", "src/gpuwrf").splitlines()
    ]
    return {
        "head": head,
        "approved_sha": approved,
        "report_commit": REPORT_COMMIT,
        "report_is_ancestor": True,
        "tree": git_output("show", "-s", "--format=%T", head),
        "parents": git_output("show", "-s", "--format=%P", head).split(),
        "tracked_source_manifest_sha256": ordinary.canonical_digest(source_rows),
        "worktree_clean": True,
    }


def assert_report_authority() -> dict[str, Any]:
    if ordinary.sha256_file(REDESIGN_PROOF) != REDESIGN_PROOF_SHA256:
        raise RuntimeError("identity-redesign proof changed")
    redesign = json.loads(REDESIGN_PROOF.read_text())
    embedded = redesign.pop("proof_sha256")
    if embedded != REDESIGN_PAYLOAD_SHA256 or ordinary.canonical_digest(redesign) != embedded:
        raise RuntimeError("identity-redesign canonical payload changed")
    if redesign.get("verdict") != "NO_FIX_LOCALIZED":
        raise RuntimeError("identity redesign no longer authorizes the discriminator")
    shortest = redesign["shortest_decisive_experiment"]
    if (
        shortest.get("name") != "NESTED_NORMAL_GAIN_1_NONOBSERVING_SAME_CARRY_AB"
        or shortest["A"]["input"] != INPUT_FILE_SHA256
        or shortest["A"]["output_reference_manifest"] != REFERENCE_MANIFEST_SHA256
        or shortest["B"]["only_static_difference"]
        != "GPUWRF_NORMAL_BDY_RELAX_STRENGTH=1.0 before import"
    ):
        raise RuntimeError("identity redesign experiment authority changed")
    if ordinary.sha256_file(PHASE_FAILURE_PROOF) != PHASE_FAILURE_PROOF_SHA256:
        raise RuntimeError("preserved phase-tap identity failure changed")
    if ordinary.sha256_file(ORDINARY_PROOF) != ORDINARY_PROOF_SHA256:
        raise RuntimeError("ordinary-bisection proof changed")
    ordinary_proof = json.loads(ORDINARY_PROOF.read_text())
    if (
        ordinary_proof.get("verdict") != "ORDINARY_BISECTION_LOCALIZED"
        or ordinary_proof["ordinary_program"]["stablehlo_sha256"] != ORDINARY_HLO_SHA256
        or ordinary_proof["ordinary_program"]["output_leaf_count"] != 106
        or ordinary_proof["terminal_identity"]["actual_manifest_sha256"]
        != EXPECTED_TERMINAL_MANIFEST
    ):
        raise RuntimeError("ordinary callable/terminal authority changed")
    return {
        "report_commit": REPORT_COMMIT,
        "identity_redesign": {
            "path": str(REDESIGN_PROOF.resolve()),
            "file_sha256": REDESIGN_PROOF_SHA256,
            "payload_sha256": REDESIGN_PAYLOAD_SHA256,
        },
        "phase_tap_failure": {
            "path": str(PHASE_FAILURE_PROOF.resolve()),
            "file_sha256": PHASE_FAILURE_PROOF_SHA256,
            "science": "DISCARDED_UNREAD",
        },
        "ordinary_bisection": {
            "path": str(ORDINARY_PROOF.resolve()),
            "file_sha256": ORDINARY_PROOF_SHA256,
            "ordinary_hlo_sha256": ORDINARY_HLO_SHA256,
            "carry_leaf_count": 106,
        },
        "frozen_terminal_manifest": EXPECTED_TERMINAL_MANIFEST,
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
        raise RuntimeError(f"retained checkpoint manifest changed: {path}")
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
    if input_row["leaf_count"] != 106 or reference_row["leaf_count"] != 106:
        raise RuntimeError("retained carry leaf count changed")
    if input_row["floating_nonfinite_count"] != 0:
        raise RuntimeError("authenticated step-9313 carry is no longer finite")
    if reference_row["floating_nonfinite_count"] != 9900:
        raise RuntimeError("retained ordinary step-9314 A signature changed")
    if jax.tree_util.tree_structure(input_carry) != jax.tree_util.tree_structure(reference_carry):
        raise RuntimeError("retained A and input carry structures differ")
    return input_carry, reference_carry, {
        "input_step_9313": input_row,
        "retained_ordinary_A_step_9314": reference_row,
        "ordinary_A_model_calls": 0,
    }


def boundary_response_oracle() -> dict[str, Any]:
    dt = 6.0
    dts = 0.6
    substeps = 10
    wrf = substeps * dts * (0.1 / dt)
    weight_20 = 20.0 * (dts / dt) * 0.1
    response_20 = 1.0 - (1.0 - weight_20) ** substeps
    weight_1 = 1.0 * (dts / dt) * 0.1
    response_1 = 1.0 - (1.0 - weight_1) ** substeps
    if not (
        abs(wrf - 0.1) < 1.0e-15
        and abs(response_20 - 0.8926258176) < 1.0e-15
        and abs(response_1 - 0.09561792499119559) < 1.0e-15
    ):
        raise RuntimeError("boundary response oracle changed")
    return {
        "dt_s": dt,
        "dts_s": dts,
        "substeps": substeps,
        "pristine_wrf_frozen_tendency_response": wrf,
        "released_moving_residual_strength20_response": response_20,
        "candidate_moving_residual_strength1_response": response_1,
        "candidate_to_wrf_ratio": response_1 / wrf,
    }


def audit_source_before_dispatch() -> dict[str, Any]:
    boundary_source = inspect.getsource(boundary_apply)
    ordinary_source = inspect.getsource(_advance_chunk_fori)
    if boundary_apply.NORMAL_BDY_RELAX_STRENGTH != 1.0:
        raise RuntimeError("gain environment was not resolved before boundary import")
    required_boundary = (
        'GPUWRF_NORMAL_BDY_RELAX_STRENGTH", "20.0"',
        "u = u_work + wu * (u_target - u_work)",
        "v = v_work + wv * (v_target - v_work)",
    )
    if any(token not in boundary_source for token in required_boundary):
        raise RuntimeError("normal-boundary source anchors changed")
    forbidden = (
        "phase_tap",
        "corrected_ni_rca",
        "recorder",
        "callback",
        "summary",
    )
    found = [token for token in forbidden if token in ordinary_source.lower()]
    if found:
        raise RuntimeError(f"ordinary production entry contains observer tokens: {found}")
    return {
        "effective_import_time_normal_strength": boundary_apply.NORMAL_BDY_RELAX_STRENGTH,
        "boundary_module_source_sha256": sha256_text(boundary_source),
        "ordinary_callable_source_sha256": sha256_text(ordinary_source),
        "ordinary_callable_forbidden_tokens": found,
        "ordinary_callable": "gpuwrf.runtime.operational_mode._advance_chunk_fori",
        "algebra_oracle": boundary_response_oracle(),
    }


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
    lowered = _advance_chunk_fori.lower(
        input_carry,
        namelist,
        jnp.asarray(STEP_9314, dtype=jnp.int32),
        clock_base,
        n_steps=1,
        cadence=cadence,
    )
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    lowered_text = stablehlo.lower()
    forbidden = [
        token
        for token in (
            "xla_python_cpu_callback",
            "host_callback",
            "io_callback",
            "pure_callback",
            "outside_compilation",
            "corrected_ni_rca",
            "rca_acoustic",
            "phase_tap",
            "recorder",
        )
        if token in lowered_text
    ]
    input_treedef = jax.tree_util.tree_structure(input_carry)
    output_treedef = jax.tree_util.tree_structure(lowered.out_info)
    input_avals = _avals(input_carry)
    output_avals = _avals(lowered.out_info)
    interface_equal = bool(
        input_treedef == output_treedef
        and input_avals == output_avals
        and len(input_avals) == 106
    )
    if forbidden or not interface_equal:
        raise RuntimeError(
            "gain-1 ordinary HLO/interface audit failed: "
            f"forbidden={forbidden} interface_equal={interface_equal}"
        )
    hlo_sha256 = sha256_text(stablehlo)
    hlo_bytes = len(stablehlo.encode())
    del lowered_text, stablehlo
    executable = lowered.compile()
    if executable.out_tree != output_treedef:
        raise RuntimeError("compiled ordinary output tree changed after lowering")
    return executable, {
        "callable": "gpuwrf.runtime.operational_mode._advance_chunk_fori",
        "n_steps": 1,
        "cadence": cadence,
        "lower_calls": 1,
        "compile_calls": 1,
        "stablehlo_sha256": hlo_sha256,
        "stablehlo_bytes": hlo_bytes,
        "retained_strength20_ordinary_hlo_sha256": ORDINARY_HLO_SHA256,
        "hlo_expected_to_differ_only_because_coefficient_changed": hlo_sha256 != ORDINARY_HLO_SHA256,
        "forbidden_tokens_found": forbidden,
        "callback_recorder_tap_free": not forbidden,
        "input_output_structure_and_avals_identical": interface_equal,
        "input_leaf_count": len(input_avals),
        "output_leaf_count": len(output_avals),
        "input_avals": input_avals,
        "output_avals": output_avals,
    }


def named_health_arrays(carry: Any) -> dict[str, np.ndarray]:
    arrays = {
        f"state.{name}": np.asarray(getattr(carry.state, name))
        for name in STATE_HEALTH_FIELDS
    }
    arrays.update({name: np.asarray(getattr(carry, name)) for name in SAVE_HEALTH_FIELDS})
    arrays.update({name: np.asarray(getattr(carry, name)) for name in SCRATCH_HEALTH_FIELDS})
    if tuple(arrays) != ALL_SCALE_FIELDS:
        raise RuntimeError("named physical-health field order changed")
    return arrays


def _array_health_stat(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    nonfinite = int(array.size - np.count_nonzero(finite))
    if np.any(finite):
        finite_values = array[finite]
        max_abs = float(np.max(np.abs(finite_values)))
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
    else:
        max_abs = None
        minimum = None
        maximum = None
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nonfinite_count": nonfinite,
        "max_abs_finite": max_abs,
        "min_finite": minimum,
        "max_finite": maximum,
    }


def baseline_health(named_arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    rows = {name: _array_health_stat(value) for name, value in named_arrays.items()}
    if any(row["nonfinite_count"] for row in rows.values()):
        raise RuntimeError("step-9313 named health baseline is nonfinite")
    if any(row["max_abs_finite"] is None for row in rows.values()):
        raise RuntimeError("step-9313 named health baseline has an empty finite field")
    return rows


def evaluate_physical_health(
    named_arrays: Mapping[str, np.ndarray],
    baseline: Mapping[str, Mapping[str, Any]],
    *,
    all_carry_nonfinite_count: int,
) -> dict[str, Any]:
    if tuple(named_arrays) != ALL_SCALE_FIELDS or set(baseline) != set(ALL_SCALE_FIELDS):
        raise RuntimeError("physical-health field set changed")
    violations: list[dict[str, Any]] = []
    rows: dict[str, Any] = {}
    for name, value in named_arrays.items():
        row = _array_health_stat(value)
        base_max = float(baseline[name]["max_abs_finite"])
        denominator = max(base_max, 1.0)
        amplification = (
            None if row["max_abs_finite"] is None else row["max_abs_finite"] / denominator
        )
        row["baseline_max_abs_finite"] = base_max
        row["amplification_vs_step9313"] = amplification
        if row["nonfinite_count"]:
            violations.append({"field": name, "gate": "finite", "actual": row["nonfinite_count"]})
        if row["max_abs_finite"] is None or row["max_abs_finite"] > ABSOLUTE_SCALE_CEILING:
            violations.append({
                "field": name,
                "gate": "absolute_scale_le_1e6",
                "actual": row["max_abs_finite"],
            })
        if amplification is None or amplification > AMPLIFICATION_CEILING:
            violations.append({
                "field": name,
                "gate": "amplification_vs_step9313_le_1000",
                "actual": amplification,
            })
        rows[name] = row
    for name in ("state.p_total", "state.mu_total"):
        positive = bool(np.all(np.isfinite(named_arrays[name]) & (named_arrays[name] > 0.0)))
        rows[name]["strictly_positive_everywhere"] = positive
        if not positive:
            violations.append({"field": name, "gate": "strictly_positive_everywhere", "actual": False})
    if all_carry_nonfinite_count:
        violations.insert(0, {
            "field": "complete_106_leaf_carry",
            "gate": "zero_nonfinite_values",
            "actual": int(all_carry_nonfinite_count),
        })
    return {
        "passed": not violations,
        "complete_carry_nonfinite_count": int(all_carry_nonfinite_count),
        "absolute_scale_ceiling": ABSOLUTE_SCALE_CEILING,
        "amplification_ceiling": AMPLIFICATION_CEILING,
        "named_fields": rows,
        "violations": violations,
        "ni_onset_gate": "covered separately by complete-carry zero-nonfinite gate",
    }


def materialize_completed_health(
    device_carry: Any,
    baseline: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    host_carry = jax.device_get(device_carry)
    manifest = ordinary.host_tree_manifest(host_carry)
    health = evaluate_physical_health(
        named_health_arrays(host_carry),
        baseline,
        all_carry_nonfinite_count=manifest["floating_nonfinite_count"],
    )
    return host_carry, manifest, health


def retain_completed_carry(run_dir: Path, *, step: int, carry: Any) -> dict[str, Any]:
    path = run_dir / f"gain1-output-d03-step-{step}.pkl"
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    before = ordinary.host_tree_manifest(carry)
    ordinary.atomic_write_pickle(path, carry)
    with path.open("rb") as handle:
        reread = pickle.load(handle)
    after = ordinary.host_tree_manifest(reread)
    comparison = ordinary.compare_manifests(before, after)
    if not comparison["all_leaf_bytes_equal"]:
        raise RuntimeError(f"retained gain-1 step-{step} carry failed reread identity")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": ordinary.sha256_file(path),
        "manifest_sha256": before["manifest_sha256"],
        "leaf_count": before["leaf_count"],
        "floating_nonfinite_count": before["floating_nonfinite_count"],
        "reread_identity": comparison,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--proof-output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    proof_output = args.proof_output.resolve()
    if run_dir == ordinary.LINEAGE_WORK_DIR.resolve() or ordinary.LINEAGE_WORK_DIR.resolve() not in run_dir.parents:
        raise RuntimeError("--run-dir must be a fresh child of the existing RCA workdir lineage")
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(run_dir)
    if proof_output.exists() or proof_output.is_symlink():
        raise FileExistsError(proof_output)
    if Path(os.environ["GPUWRF_WRF_ROOT"]).resolve() != (
        ordinary.RETRY20_ROOT / "authority/wrf_root"
    ).resolve():
        raise RuntimeError("GPUWRF_WRF_ROOT is not the retained Retry20 WRF authority")
    if sorted(os.sched_getaffinity(0)) != EXPECTED_AFFINITY:
        raise RuntimeError(
            f"gain-1 discriminator must be taskset to CPUs 12-15: {sorted(os.sched_getaffinity(0))}"
        )
    if _PREIMPORT_AUTHORITY is None:
        raise RuntimeError("pre-import environment authority was not established")

    started = datetime.now(timezone.utc)
    candidate = assert_candidate_authority()
    run_dir.mkdir(parents=True, exist_ok=False)
    authority: dict[str, Any] = {
        "preimport": _PREIMPORT_AUTHORITY,
        "candidate": candidate,
        "report": assert_report_authority(),
        "inputs_pre": ordinary.assert_input_authority(),
        "prior_pre": ordinary.assert_prior_authority(),
        "lock_pre": ordinary.assert_lock_authority(),
        "preemption_pre": ordinary.assert_preemption_clear("gain1-startup"),
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
    reference_manifest = ordinary.host_tree_manifest(reference_carry)
    baseline = baseline_health(named_health_arrays(input_carry))
    source_audit = audit_source_before_dispatch()

    tree, names, initial_carries, dt_by_domain, load_authority = ordinary.load_corrected_tree(run_dir)
    if names != ("d01", "d02", "d03") or dt_by_domain["d03"] != 6.0:
        raise RuntimeError("corrected d03 domain authority changed")
    namelist = tree.domains["d03"].namelist
    if getattr(namelist.boundary_config, "normal_bdy_relax_strength", None) is not None:
        raise RuntimeError("corrected nested d03 unexpectedly overrides the import-time gain")
    clock_base = build_clock_base(namelist)
    cadence = int(namelist.radiation_cadence_steps)
    if cadence != 300 or STEP_9314 % cadence != 14 or STEP_9315 % cadence != 15:
        raise RuntimeError("step-9314/9315 radiation cadence authority changed")
    del initial_carries, tree
    gc.collect()

    ordinary.assert_preemption_clear("gain1-pre-lower")
    executable, program_audit = lower_and_compile_once(input_carry, namelist, clock_base)
    ordinary.assert_preemption_clear("gain1-pre-step-9314")
    b9314 = executable(
        input_carry,
        namelist,
        jnp.asarray(STEP_9314, dtype=jnp.int32),
        clock_base,
        n_steps=1,
        cadence=cadence,
    )
    model_calls = 1
    host9314, manifest9314, health9314 = materialize_completed_health(b9314, baseline)
    retained_outputs = {
        "step_9314": retain_completed_carry(run_dir, step=STEP_9314, carry=host9314)
    }
    comparisons = {
        "retained_strength20_A_vs_gain1_B_step9314": ordinary.compare_manifests(
            reference_manifest, manifest9314
        )
    }
    print(
        "GAIN1_HEALTH "
        f"step={STEP_9314} passed={health9314['passed']} "
        f"nonfinite={manifest9314['floating_nonfinite_count']} "
        f"violations={len(health9314['violations'])}",
        flush=True,
    )

    step_rows: list[dict[str, Any]] = [{
        "native_step": STEP_9314,
        "radiation_step": False,
        "manifest": manifest9314,
        "physical_health": health9314,
    }]
    if health9314["passed"]:
        ordinary.assert_preemption_clear("gain1-pre-step-9315")
        b9315 = executable(
            b9314,
            namelist,
            jnp.asarray(STEP_9315, dtype=jnp.int32),
            clock_base,
            n_steps=1,
            cadence=cadence,
        )
        model_calls += 1
        host9315, manifest9315, health9315 = materialize_completed_health(b9315, baseline)
        retained_outputs["step_9315"] = retain_completed_carry(
            run_dir, step=STEP_9315, carry=host9315
        )
        step_rows.append({
            "native_step": STEP_9315,
            "radiation_step": False,
            "manifest": manifest9315,
            "physical_health": health9315,
        })
        print(
            "GAIN1_HEALTH "
            f"step={STEP_9315} passed={health9315['passed']} "
            f"nonfinite={manifest9315['floating_nonfinite_count']} "
            f"violations={len(health9315['violations'])}",
            flush=True,
        )
    else:
        health9315 = None

    both_healthy = bool(
        health9314["passed"] and health9315 is not None and health9315["passed"]
    )
    verdict = (
        "GAIN20_INCIDENT_TRIGGER_LOCALIZED"
        if both_healthy
        else "GAIN1_CANDIDATE_FALSIFIED"
    )
    checkpoint_post = {
        "input_file_sha256": ordinary.sha256_file(INPUT_CHECKPOINT),
        "reference_file_sha256": ordinary.sha256_file(REFERENCE_CHECKPOINT),
    }
    if checkpoint_post != {
        "input_file_sha256": INPUT_FILE_SHA256,
        "reference_file_sha256": REFERENCE_FILE_SHA256,
    }:
        raise RuntimeError("retained A/B checkpoint changed during gain-1 experiment")
    inputs_post = ordinary.assert_input_authority()
    if inputs_post["authority_sha256"] != authority["inputs_pre"]["authority_sha256"]:
        raise RuntimeError("corrected input authority changed during gain-1 experiment")
    finished = datetime.now(timezone.utc)
    authority.update({
        "inputs_post": inputs_post,
        "checkpoint_files_post": checkpoint_post,
        "lock_post": ordinary.assert_lock_authority(),
        "preemption_post": ordinary.assert_preemption_clear("gain1-complete"),
        "finished_utc": finished.isoformat(),
        "wall_seconds": (finished - started).total_seconds(),
    })

    proof: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "verdict": verdict,
        "authority": authority,
        "checkpoint_authority": checkpoint_authority,
        "load_authority": load_authority,
        "source_audit": source_audit,
        "program_audit": program_audit,
        "health_contract": {
            "defined_before_dispatch": True,
            "complete_carry_zero_nonfinite": True,
            "absolute_scale_ceiling": ABSOLUTE_SCALE_CEILING,
            "amplification_ceiling": AMPLIFICATION_CEILING,
            "strictly_positive_fields": ["state.p_total", "state.mu_total"],
            "scale_fields": list(ALL_SCALE_FIELDS),
            "ni": "complete-carry finite at both steps; excluded from 1e6 scale because healthy input max exceeds it",
            "step9313_baseline": baseline,
        },
        "execution": {
            "ordinary_A_model_calls": 0,
            "gain1_B_model_calls": model_calls,
            "lower_calls": 1,
            "compile_calls": 1,
            "step_9315_dispatched_only_after_step_9314_health": True,
            "steps": step_rows,
            "retained_outputs": retained_outputs,
            "comparisons": comparisons,
        },
        "causal_decision": {
            "both_steps_healthy": both_healthy,
            "gain20_incident_trigger_localized": both_healthy,
            "strength_only_fix_authorized": False,
            "full_pristine_wrf_frozen_rk1_bundle_next": both_healthy,
            "gain1_candidate_falsified": not both_healthy,
        },
        "mechanism_separation": {
            "ni_track": "step-9315 complete-carry onset gate",
            "v10_track": "OPEN_SEPARATE_NO_CAUSAL_LINK_FROM_THIS_EXPERIMENT",
            "common_root_proved": False,
        },
        "scope_gates": {
            "full_18h_run": False,
            "prefix_run": False,
            "ordinary_A_reexecuted": False,
            "recorder_or_tap_used": False,
            "summary_or_callback_used": False,
            "model_or_numerical_source_edit": False,
            "carry_interface_changed": False,
            "additional_gpu_hypothesis_arm": False,
        },
    }
    proof["proof_sha256"] = ordinary.canonical_digest(proof)
    ordinary.atomic_write_json(proof_output, proof)
    print(
        json.dumps({
            "proof": str(proof_output),
            "proof_sha256": proof["proof_sha256"],
            "verdict": verdict,
            "model_calls": model_calls,
            "step9314_healthy": health9314["passed"],
            "step9315_healthy": None if health9315 is None else health9315["passed"],
        }, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return 0 if both_healthy else 4


if __name__ == "__main__":
    raise SystemExit(main())
