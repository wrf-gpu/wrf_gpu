"""Restart-safe CPU-only first-activation search and local two-step A/B.

The released worktree remains unchanged.  ``gpuwrf`` must be imported from a
detached worktree at the preserved complete-interface commit.  History scans
each completed d03 step, so no monotonicity assumption is used.  The bounded
step-200 endpoint is the first authenticated 00:20 CPU-WRF/Retry20 frame.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-17-v0234-moist-theta-activation-bisection"
HISTORY_PATH = SPRINT / "activation-history.json"
AB_PROOF_PATH = SPRINT / "activation-ab-proof.json"
BLOCKER_PATH = SPRINT / "activation-bisection-blocker.json"
RUN_DIR = Path(
    os.environ.get(
        "GPUWRF_ACTIVATION_RUN_DIR",
        "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
        "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
        "corrected_ni_rca_max_22c2bd7a/"
        "moist_theta_activation_bisection_fa71e2ec",
    )
)

CANDIDATE_COMMIT = "454c348fc73805c3c9531103619fb843f6717e71"
CANDIDATE_SRC_TREE = "70b5b5b393eb4af693fae780e78f073b277f9023"
RELEASED_SRC_TREE = "835dcc29bf316c0715b41a72e064985e9cf099df"
CONTRACT_COMMIT = "fa71e2ecd1647ba7142eb7255bbcfb88f3eac22c"
PRIOR_PROOF_SHA256 = "336ad297f6dcd7eefacfba2c4519866f3ed5eef4396aa0b7148510ac894be6ae"
MAX_STEP = 200
CHECKPOINT_EVERY = 25
DT_SECONDS = 6.0
H_ROUNDOFF_ULP_BUDGET = 128
HYDROMETEORS = ("qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng")
HEALTH_FIELDS = (
    "theta",
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "Ni",
    "Nr",
    "Ns",
    "Ng",
    "p_perturbation",
    "ph_perturbation",
    "mu_perturbation",
    "u",
    "v",
    "w",
)
REQUIRED_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
    "GPUWRF_MOIST_THETA_INTERFACE": "1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        pickle.dump(value, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(("git", "-C", str(cwd), *args), text=True).strip()


def _activation_stats(np, carry: Any) -> dict[str, Any]:
    h = np.asarray(carry.h_diabatic, dtype=np.float64)
    theta = np.asarray(carry.state.theta, dtype=np.float64)
    theta_scale = max(1.0, float(np.max(np.abs(theta))))
    # Independent pre-run IEEE-754 budget: the literal prep/finish path has
    # fewer than 128 elementary operations.  This ceiling is intentionally
    # conservative and scale-aware; it prevents arithmetic residue from being
    # promoted to physics and is not fitted to the history being evaluated.
    h_roundoff_ceiling = (
        H_ROUNDOFF_ULP_BUDGET * float(np.finfo(np.float64).eps) * theta_scale / DT_SECONDS
    )
    species: dict[str, Any] = {}
    positive_total = 0
    all_species_valid = True
    for name in HYDROMETEORS:
        values = np.asarray(getattr(carry.state, name), dtype=np.float64)
        finite = bool(np.all(np.isfinite(values)))
        minimum = float(np.min(values))
        positive = int(np.count_nonzero(values > 0.0))
        positive_total += positive
        all_species_valid = all_species_valid and finite and minimum >= 0.0
        species[name] = {
            "finite": finite,
            "min": minimum,
            "max": float(np.max(values)),
            "positive_count": positive,
        }
    health = {}
    for name in HEALTH_FIELDS:
        values = np.asarray(getattr(carry.state, name))
        health[name] = bool(
            not np.issubdtype(values.dtype, np.inexact) or np.all(np.isfinite(values))
        )
    h_max_abs = float(np.max(np.abs(h)))
    nonroundoff_h = h_max_abs > h_roundoff_ceiling
    real_activation = positive_total > 0 or nonroundoff_h
    return {
        "h_diabatic": {
            "finite": bool(np.all(np.isfinite(h))),
            "min": float(np.min(h)),
            "max": float(np.max(h)),
            "max_abs": h_max_abs,
            "nonzero_count": int(np.count_nonzero(h)),
            "roundoff_ceiling": h_roundoff_ceiling,
            "roundoff_ulp_budget": H_ROUNDOFF_ULP_BUDGET,
            "nonroundoff": nonroundoff_h,
            "classification": (
                "PHYSICAL_PHASE_COUPLED"
                if positive_total > 0
                else "PHYSICAL_NONROUNDOFF_H"
                if nonroundoff_h
                else "ROUNDING_ONLY_WITH_ZERO_PHASE_RESERVOIRS"
            ),
        },
        "hydrometeors": species,
        "hydrometeor_positive_total": positive_total,
        "all_hydrometeors_finite_nonnegative": all_species_valid,
        "selected_state_finite": bool(all(health.values())),
        "selected_state_finite_by_field": health,
        "real_activation": real_activation,
    }


def _manifest_authority(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _persist_carry(common, jax, np, carry: Any, *, role: str, step: int) -> dict[str, Any]:
    checkpoint_dir = RUN_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{role}-d03-step-{step}.pkl"
    manifest_path = checkpoint_dir / f"{role}-d03-step-{step}.manifest.json"
    host = jax.device_get(carry)
    expected = common._manifest(jax, np, host)
    if path.exists() or manifest_path.exists():
        if not path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"partial checkpoint collision for {role} step {step}")
        with path.open("rb") as stream:
            reread = pickle.load(stream)
        actual = common._manifest(jax, np, reread)
        recorded = json.loads(manifest_path.read_text())
        if actual != expected or recorded != expected:
            raise RuntimeError(f"existing checkpoint identity mismatch for {role} step {step}")
    else:
        _atomic_pickle(path, host)
        with path.open("rb") as stream:
            reread = pickle.load(stream)
        actual = common._manifest(jax, np, reread)
        if actual != expected:
            raise RuntimeError(f"checkpoint reread identity mismatch for {role} step {step}")
        _atomic_json(manifest_path, expected)
    return {
        "role": role,
        "step": step,
        "sim_time_seconds": step * DT_SECONDS,
        "carry_path": str(path.resolve()),
        "carry_bytes": path.stat().st_size,
        "carry_file_sha256": _sha256(path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": _sha256(manifest_path),
        "manifest_authority_sha256": _manifest_authority(expected),
        "carry_manifest_sha256": expected["sha256"],
        "leaf_count": expected["leaf_count"],
        "all_leaves_finite": bool(all(row["finite"] for row in expected["leaves"])),
        "reread_identity": True,
    }


def _load_persisted(common, jax, np, row: dict[str, Any]) -> Any:
    carry_path = Path(row["carry_path"])
    manifest_path = Path(row["manifest_path"])
    if _sha256(carry_path) != row["carry_file_sha256"]:
        raise RuntimeError(f"checkpoint file hash mismatch: {carry_path}")
    if _sha256(manifest_path) != row["manifest_file_sha256"]:
        raise RuntimeError(f"checkpoint manifest file hash mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if _manifest_authority(manifest) != row["manifest_authority_sha256"]:
        raise RuntimeError(f"checkpoint manifest authority mismatch: {manifest_path}")
    with carry_path.open("rb") as stream:
        carry = pickle.load(stream)
    actual = common._manifest(jax, np, carry)
    if actual != manifest or actual["sha256"] != row["carry_manifest_sha256"]:
        raise RuntimeError(f"checkpoint leaf identity mismatch: {carry_path}")
    return carry


def _history_template(environment: dict[str, str], candidate_root: Path) -> dict[str, Any]:
    return {
        "schema": "gpuwrf.v0234.moist-theta-activation-history.v1",
        "status": "RUNNING",
        "authority": {
            "contract_commit": CONTRACT_COMMIT,
            "prior_closeout_proof_sha256": PRIOR_PROOF_SHA256,
            "candidate_commit": CANDIDATE_COMMIT,
            "candidate_src_tree": CANDIDATE_SRC_TREE,
            "released_src_tree": RELEASED_SRC_TREE,
            "candidate_root": str(candidate_root),
        },
        "environment": environment,
        "bound": {
            "first_step": 1,
            "last_step": MAX_STEP,
            "last_sim_time_seconds": MAX_STEP * DT_SECONDS,
            "checkpoint_every_steps": CHECKPOINT_EVERY,
            "rationale": "Step 200 is d03 00:20 and has authenticated CPU-WRF and Retry20 anchors.",
        },
        "activation_predicate": {
            "expression": "hydrometeor_positive_total > 0 or h_max_abs > 128*float64_eps*max(1,max_abs(theta))/dt",
            "threshold_fitted": False,
            "reason": "Initialized phase reservoirs are exactly zero, so first exact positivity is a source-literal phase-change event. The independent 128-operation IEEE-754 ceiling is fixed before history and conservatively bounds literal prep/finish rounding; it is not fitted to results.",
            "monotonicity_assumed": False,
            "scan": "every completed step sequentially",
        },
        "input": {},
        "compile": {},
        "checkpoints": [],
        "observations": [],
        "activation": None,
        "scope": {
            "backend": "cpu",
            "gpu_queries": 0,
            "gpu_commands": 0,
            "gpu_compiles": 0,
            "gpu_dispatches": 0,
            "fable_or_claude": 0,
            "model_changes": 0,
        },
    }


def _observation(step: int, stats: dict[str, Any]) -> dict[str, Any]:
    return {"step": step, "sim_time_seconds": step * DT_SECONDS, "stats": stats}


def _source_effect_reaches_thermo_pressure_momentum(changed: list[str]) -> bool:
    required = {
        "theta",
        "qv",
        "p_perturbation",
        "ph_perturbation",
        "u",
        "v",
        "w",
    }
    phase_mass = {"qc", "qr", "qi", "qs", "qg"}
    return required.issubset(changed) and bool(phase_mass.intersection(changed))


def _compile(runtime, jax, carry, namelist, clock, *, label: str):
    started = time.perf_counter()
    lowered = runtime._advance_chunk_fori.lower(
        carry,
        namelist,
        1,
        clock,
        n_steps=1,
        cadence=int(namelist.radiation_cadence_steps),
    )
    lower_seconds = time.perf_counter() - started
    input_tree = jax.tree_util.tree_structure(carry)
    output_tree = jax.tree_util.tree_structure(lowered.out_info)
    input_avals = [(tuple(x.shape), str(x.dtype)) for x in jax.tree_util.tree_leaves(carry)]
    output_avals = [
        (tuple(x.shape), str(x.dtype)) for x in jax.tree_util.tree_leaves(lowered.out_info)
    ]
    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    audit = {
        "label": label,
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "interface_identity": bool(input_tree == output_tree and input_avals == output_avals),
        "leaf_count": len(input_avals),
        "compiled_n_steps_argument_is_dynamic": True,
    }
    return executable, audit


def _run(executable, carry, namelist, start_step: int, clock, n_steps: int, cadence: int):
    return executable(
        carry,
        namelist,
        start_step,
        clock,
        n_steps=n_steps,
        cadence=cadence,
    )


def _run_activation_ab(
    common,
    runtime,
    jax,
    jnp,
    np,
    history: dict[str, Any],
    candidate_executable,
    candidate_compile: dict[str, Any],
    candidate_namelist,
    clock,
) -> tuple[dict[str, Any], int]:
    activation = history["activation"]
    preceding_row = activation["preceding_carry"]
    active_row = activation["first_active_carry"]
    preceding = _load_persisted(common, jax, np, preceding_row)
    retained_active = _load_persisted(common, jax, np, active_row)
    first_step = int(activation["first_active_step"])
    cadence = int(candidate_namelist.radiation_cadence_steps)
    baseline_namelist = dataclasses.replace(candidate_namelist, moist_theta_interface=False)

    runtime._advance_chunk_fori.clear_cache()
    baseline_executable, baseline_compile = _compile(
        runtime, jax, preceding, baseline_namelist, clock, label="baseline_A"
    )
    started = time.perf_counter()
    baseline = _run(
        baseline_executable, preceding, baseline_namelist, first_step, clock, 2, cadence
    )
    baseline = jax.device_get(baseline)
    baseline_dispatch_seconds = time.perf_counter() - started

    started = time.perf_counter()
    candidate = _run(
        candidate_executable,
        preceding,
        candidate_namelist,
        first_step,
        clock,
        2,
        cadence,
    )
    candidate = jax.device_get(candidate)
    candidate_dispatch_seconds = time.perf_counter() - started

    reproduced_active = _run(
        candidate_executable,
        preceding,
        candidate_namelist,
        first_step,
        clock,
        1,
        cadence,
    )
    reproduced_active = jax.device_get(reproduced_active)
    retained_active_manifest = common._manifest(jax, np, retained_active)
    reproduced_active_manifest = common._manifest(jax, np, reproduced_active)
    baseline_manifest = common._manifest(jax, np, baseline)
    candidate_manifest = common._manifest(jax, np, candidate)
    frames, frame_authority = common._load_frames(np)
    projection = common._projection(
        np, runtime, baseline, candidate, candidate_namelist, frames
    )

    species = {}
    for name in ("qv", *HYDROMETEORS):
        values = np.asarray(getattr(candidate.state, name), dtype=np.float64)
        species[name] = {
            "finite": bool(np.all(np.isfinite(values))),
            "min": float(np.min(values)),
            "nonnegative": bool(np.min(values) >= 0.0),
        }
    changed = sorted(
        name
        for name in common.STATE_TO_HISTORY
        if getattr(baseline.state, name) is not None
        and getattr(candidate.state, name) is not None
        and not np.array_equal(
            np.asarray(getattr(baseline.state, name)),
            np.asarray(getattr(candidate.state, name)),
        )
    )
    prior_observations = [
        row for row in history["observations"] if int(row["step"]) < first_step
    ]
    active_stats = next(
        row["stats"] for row in history["observations"] if int(row["step"]) == first_step
    )
    checks = {
        "sequential_first_activation_proved": bool(
            active_stats["real_activation"]
            and all(not row["stats"]["real_activation"] for row in prior_observations)
        ),
        "activation_carries_authenticated": bool(
            preceding_row["reread_identity"] and active_row["reread_identity"]
        ),
        "candidate_one_step_reproduces_retained_active": bool(
            retained_active_manifest["sha256"] == reproduced_active_manifest["sha256"]
        ),
        "both_interfaces_identical": bool(
            candidate_compile["interface_identity"] and baseline_compile["interface_identity"]
        ),
        "complete_structure_identity": bool(
            baseline_manifest["leaf_count"] == candidate_manifest["leaf_count"]
            and [
                (row["path"], row["shape"], row["dtype"])
                for row in baseline_manifest["leaves"]
            ]
            == [
                (row["path"], row["shape"], row["dtype"])
                for row in candidate_manifest["leaves"]
            ]
        ),
        "both_complete_carries_finite": bool(
            all(row["finite"] for row in baseline_manifest["leaves"])
            and all(row["finite"] for row in candidate_manifest["leaves"])
        ),
        "all_species_finite_nonnegative": bool(
            all(row["finite"] and row["nonnegative"] for row in species.values())
        ),
        "complete_output_changed": bool(
            baseline_manifest["sha256"] != candidate_manifest["sha256"]
        ),
        "source_effect_reaches_thermo_pressure_momentum": (
            _source_effect_reaches_thermo_pressure_momentum(changed)
        ),
        "all_affected_available_field_masks_no_worse": bool(projection["green"]),
    }
    verdict = "ACTIVATION_AB_GREEN" if all(checks.values()) else "ACTIVATION_AB_RED"
    proof = {
        "schema": "gpuwrf.v0234.moist-theta-activation-ab.v1",
        "verdict": verdict,
        "authority": history["authority"],
        "activation": activation,
        "candidate_compile": candidate_compile,
        "baseline_compile": baseline_compile,
        "dispatch_seconds": {
            "baseline_two_step": baseline_dispatch_seconds,
            "candidate_two_step": candidate_dispatch_seconds,
        },
        "frame_authority": frame_authority,
        "baseline_A": {"manifest": baseline_manifest},
        "candidate_B": {
            "manifest": candidate_manifest,
            "species": species,
            "changed_state_fields": changed,
        },
        "retained_active_manifest_sha256": retained_active_manifest["sha256"],
        "reproduced_active_manifest_sha256": reproduced_active_manifest["sha256"],
        "affected_field_projection": projection,
        "checks": checks,
        "scope": history["scope"],
        "incident_separation": {
            "V10": "Separate unless the dual-anchor affected-field gate supplies linkage.",
            "Ni": "Separate late finite gate; this CPU discriminator does not run the late window.",
        },
    }
    proof["proof_sha256"] = _canonical(proof)
    _atomic_json(AB_PROOF_PATH, proof)
    return proof, 0 if verdict == "ACTIVATION_AB_GREEN" else 3


def main() -> int:
    loader_scratch: Path | None = None
    try:
        environment = {name: os.environ.get(name) for name in REQUIRED_ENV}
        if environment != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {environment!r}")
        candidate_root_raw = os.environ.get("GPUWRF_CANDIDATE_ROOT")
        if not candidate_root_raw:
            raise RuntimeError("GPUWRF_CANDIDATE_ROOT is required")
        candidate_root = Path(candidate_root_raw).resolve()
        if _git("rev-parse", "HEAD", cwd=candidate_root) != CANDIDATE_COMMIT:
            raise RuntimeError("candidate worktree commit mismatch")
        if _git("rev-parse", "HEAD:src/gpuwrf", cwd=candidate_root) != CANDIDATE_SRC_TREE:
            raise RuntimeError("candidate src/gpuwrf tree mismatch")
        if _git("rev-parse", "HEAD:src/gpuwrf") != RELEASED_SRC_TREE:
            raise RuntimeError("released src/gpuwrf tree changed")
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", CONTRACT_COMMIT, "HEAD"),
            cwd=ROOT,
            check=True,
        )

        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)
        import jax.numpy as jnp
        import numpy as np
        import gpuwrf

        imported_root = Path(gpuwrf.__file__).resolve()
        if candidate_root / "src" not in imported_root.parents:
            raise RuntimeError(f"gpuwrf imported outside candidate tree: {imported_root}")
        if jax.default_backend() != "cpu":
            raise RuntimeError(f"unexpected backend {jax.default_backend()}")
        if jax.config.values.get("jax_cpu_enable_async_dispatch") is not False:
            raise RuntimeError("CPU async dispatch remained enabled")
        affinity = sorted(os.sched_getaffinity(0))
        if affinity != [12]:
            raise RuntimeError(f"CPU affinity changed: {affinity}")

        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary
        from scripts import v0234_moist_theta_interface_cpu_ab as common

        if common._sha256(common.STEP0) != common.STEP0_SHA256:
            raise RuntimeError("authenticated Step0 carry mismatch")
        loader_scratch = Path(tempfile.mkdtemp(prefix="v0234-moist-theta-activation-load-"))
        tree, names, _initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(
            loader_scratch
        )
        if names != ("d01", "d02", "d03") or dt_by_domain != {
            "d01": 54.0,
            "d02": 18.0,
            "d03": 6.0,
        }:
            raise RuntimeError("canonical hierarchy changed")
        candidate_namelist = tree.domains["d03"].namelist
        if not runtime._moist_theta_interface_active(candidate_namelist):
            raise RuntimeError("preserved candidate inactive")
        if int(candidate_namelist.mp_physics) != 8 or float(
            candidate_namelist.mp_tend_lim
        ) != 10.0:
            raise RuntimeError("mp=8/mp_tend_lim authority changed")
        clock = runtime.build_clock_base(candidate_namelist)

        RUN_DIR.mkdir(parents=True, exist_ok=True)
        if HISTORY_PATH.exists():
            history = json.loads(HISTORY_PATH.read_text())
            if history["authority"]["candidate_commit"] != CANDIDATE_COMMIT:
                raise RuntimeError("history candidate authority mismatch")
            if history["bound"]["last_step"] != MAX_STEP:
                raise RuntimeError("history bound mismatch")
        else:
            history = _history_template(environment, candidate_root)

        if history["checkpoints"]:
            latest = history["checkpoints"][-1]
            carry = _load_persisted(common, jax, np, latest)
            current_step = int(latest["step"])
            print(f"ACTIVATION_RESUME step={current_step}", flush=True)
        else:
            with common.STEP0.open("rb") as stream:
                carry = pickle.load(stream)
            if hasattr(carry, "h_diabatic"):
                raise RuntimeError("pre-candidate Step0 unexpectedly owns h_diabatic")
            object.__setattr__(carry, "h_diabatic", jnp.zeros_like(carry.state.theta))
            seed_stats = _activation_stats(np, carry)
            if seed_stats["hydrometeor_positive_total"] != 0:
                raise RuntimeError("initialized Step0 hydrometeor reservoir is not zero")
            seed = _persist_carry(common, jax, np, carry, role="seed", step=0)
            history["input"] = {
                "path": str(common.STEP0),
                "sha256": common.STEP0_SHA256,
                "completed_step": 0,
                "h_diabatic_seed": "source-authorized exact zero",
                "seed_stats": seed_stats,
                "candidate_aware_seed": seed,
                "load_authority": load_authority,
            }
            history["checkpoints"].append(seed)
            _atomic_json(HISTORY_PATH, history)
            current_step = 0

        candidate_executable, candidate_compile = _compile(
            runtime, jax, carry, candidate_namelist, clock, label="candidate_history_B"
        )
        if not candidate_compile["interface_identity"]:
            raise RuntimeError("candidate history compile changed carry interface")
        history["compile"] = candidate_compile
        _atomic_json(HISTORY_PATH, history)
        observations = {int(row["step"]): row for row in history["observations"]}
        cadence = int(candidate_namelist.radiation_cadence_steps)

        if history["activation"] is None:
            for step in range(current_step + 1, MAX_STEP + 1):
                preceding = carry
                started = time.perf_counter()
                carry = _run(
                    candidate_executable,
                    carry,
                    candidate_namelist,
                    step,
                    clock,
                    1,
                    cadence,
                )
                carry = jax.device_get(carry)
                dispatch_seconds = time.perf_counter() - started
                stats = _activation_stats(np, carry)
                if not stats["selected_state_finite"] or not stats[
                    "all_hydrometeors_finite_nonnegative"
                ] or not stats["h_diabatic"]["finite"]:
                    raise RuntimeError(f"nonfinite or invalid physical state at step {step}")
                row = _observation(step, stats)
                row["dispatch_seconds"] = dispatch_seconds
                if step in observations:
                    prior = dict(observations[step])
                    prior.pop("dispatch_seconds", None)
                    comparison = dict(row)
                    comparison.pop("dispatch_seconds", None)
                    if prior != comparison:
                        raise RuntimeError(f"restart replay observation mismatch at step {step}")
                else:
                    history["observations"].append(row)
                    observations[step] = row

                if stats["real_activation"]:
                    preceding_saved = _persist_carry(
                        common, jax, np, preceding, role="preceding", step=step - 1
                    )
                    active_saved = _persist_carry(
                        common, jax, np, carry, role="first-active", step=step
                    )
                    history["activation"] = {
                        "first_active_step": step,
                        "first_active_sim_time_seconds": step * DT_SECONDS,
                        "stats": stats,
                        "preceding_carry": preceding_saved,
                        "first_active_carry": active_saved,
                    }
                    history["status"] = "ACTIVATION_LOCALIZED"
                    history["proof_sha256"] = _canonical(history)
                    _atomic_json(HISTORY_PATH, history)
                    print(
                        f"ACTIVATION_FOUND step={step} sim_s={step * DT_SECONDS:.0f} "
                        f"hydro_positive={stats['hydrometeor_positive_total']} "
                        f"h_maxabs={stats['h_diabatic']['max_abs']:.17g}",
                        flush=True,
                    )
                    break

                if step % CHECKPOINT_EVERY == 0:
                    saved = _persist_carry(
                        common, jax, np, carry, role="checkpoint", step=step
                    )
                    history["checkpoints"].append(saved)
                history.pop("proof_sha256", None)
                _atomic_json(HISTORY_PATH, history)
                if step % 5 == 0:
                    print(
                        f"ACTIVATION_SCAN step={step}/{MAX_STEP} "
                        f"hydro_positive=0 h_maxabs={stats['h_diabatic']['max_abs']:.17g} "
                        f"dispatch_s={dispatch_seconds:.3f}",
                        flush=True,
                    )

        if history["activation"] is None:
            history["status"] = "NO_ACTIVATION_WITHIN_BOUND"
            history["next_action"] = (
                "Extend the same restart-safe CPU scan from authenticated step 200 to "
                "the next corrected WRF output boundary; do not use GPU or fabricate h."
            )
            history["proof_sha256"] = _canonical(history)
            _atomic_json(HISTORY_PATH, history)
            print(
                json.dumps(
                    {
                        "verdict": history["status"],
                        "bound_step": MAX_STEP,
                        "proof_sha256": history["proof_sha256"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 5

        proof, code = _run_activation_ab(
            common,
            runtime,
            jax,
            jnp,
            np,
            history,
            candidate_executable,
            candidate_compile,
            candidate_namelist,
            clock,
        )
        history["status"] = proof["verdict"]
        history["activation_ab_proof"] = {
            "path": str(AB_PROOF_PATH.resolve()),
            "file_sha256": _sha256(AB_PROOF_PATH),
            "proof_sha256": proof["proof_sha256"],
        }
        history.pop("proof_sha256", None)
        history["proof_sha256"] = _canonical(history)
        _atomic_json(HISTORY_PATH, history)
        print(
            json.dumps(
                {
                    "verdict": proof["verdict"],
                    "proof_sha256": proof["proof_sha256"],
                    "failed_checks": [
                        name for name, green in proof["checks"].items() if not green
                    ],
                    "no_worse_failure_count": len(
                        proof["affected_field_projection"]["no_worse_failures"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return code
    except Exception as exc:
        blocker = {
            "schema": "gpuwrf.v0234.moist-theta-activation-bisection-blocker.v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "environment": {name: os.environ.get(name) for name in REQUIRED_ENV},
            "candidate_root": os.environ.get("GPUWRF_CANDIDATE_ROOT"),
            "run_dir": str(RUN_DIR),
            "gpu_queries": 0,
            "gpu_commands": 0,
        }
        blocker["proof_sha256"] = _canonical(blocker)
        _atomic_json(BLOCKER_PATH, blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 4
    finally:
        if loader_scratch is not None:
            shutil.rmtree(loader_scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
