"""Bounded CPU falsification/validation arm for the v0.23.4 step-200 blocker.

Runs the full corrected three-domain tree ON CPU from the authenticated cold
start for exactly 23 root steps (d03 own step 207 > 200), materializing the
d03 carry at own step 200 (sim 1200 s = the 00:20 output that killed the
`60659a2e` GPU run).  The probe never raises out of the output callback: it
records health and lets both arms complete, so the falsified and corrected
models face identical control flow.

Arm selection is by PYTHONPATH: the launcher points ``gpuwrf`` at either the
retained falsified model worktree (`60659a2e`) or the corrected worktree.
The probe records the arm's commit/tree and refuses a dirty arm unless the
launcher passes the expected dirty-diff SHA (used only before the corrected
commit exists).

Gates evaluated by the launcher from the emitted proof JSON:

- falsified arm reproduces the blocker on CPU: non-finite Ni at step 200 and
  ``ph`` beyond ring 0 bit-frozen versus step 0;
- corrected arm: every floating leaf finite at step 200 and interior ``ph``
  advancing.

CPU-only; no GPU lock, query, compile, or dispatch.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
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
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
}

SCHEMA = "gpuwrf.v0234.final-ni-fable5-cpu-ab-arm.v1"
INPUT_DIR = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf"
)
RETAINED_STEP0 = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_1500_science_60659a2e_terminal2/"
    "science-runtime/failure/last-healthy-d03-step-0.pkl"
)
RETAINED_STEP0_SHA = (
    "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
)
ROOT_STEPS = 23
D03_ALARM_STEP = 200
D03_DT_S = 6.0


def canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_env() -> dict[str, str]:
    actual = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual != REQUIRED_ENV:
        raise RuntimeError(f"pre-import env mismatch: {actual!r}")
    if "jax" in sys.modules or "gpuwrf" in sys.modules:
        raise RuntimeError("jax/gpuwrf imported before environment validation")
    return dict(REQUIRED_ENV)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("falsified", "corrected"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty-diff-sha256")
    parser.add_argument(
        "--stop-after-alarm",
        action="store_true",
        help=(
            "CPU-probe repair: after materializing the completed d03 step-200 "
            "observation, raise a private host sentinel to avoid executing the "
            "seven post-failure child steps in root step 23"
        ),
    )
    args = parser.parse_args()

    env_authority = _validate_env()

    model_root = args.model_root.resolve()
    head = _git(model_root, "rev-parse", "HEAD")
    tree_sha = _git(model_root, "rev-parse", "HEAD^{tree}")
    dirty = _git(model_root, "status", "--porcelain")
    if head != args.expected_commit:
        raise RuntimeError(f"arm HEAD {head} != expected {args.expected_commit}")
    if dirty and not args.allow_dirty_diff_sha256:
        raise RuntimeError(f"arm worktree dirty: {dirty!r}")
    if dirty:
        diff = subprocess.run(
            ["git", "-C", str(model_root), "diff"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff_sha = hashlib.sha256(diff.encode()).hexdigest()
        if diff_sha != args.allow_dirty_diff_sha256:
            raise RuntimeError(
                f"arm dirty diff sha {diff_sha} != allowed {args.allow_dirty_diff_sha256}"
            )

    import numpy as np

    import gpuwrf

    gpuwrf_file = Path(gpuwrf.__file__).resolve()
    if model_root not in gpuwrf_file.parents:
        raise RuntimeError(f"gpuwrf resolved outside arm root: {gpuwrf_file}")

    import jax

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"backend {jax.default_backend()} != cpu")

    # The State constructors pin a device through a single seam; the reviewed
    # terminal tooling rebinds the same seam to select its device
    # (v0234_nested_frozen_wrf_boundary_window.py:3503/3775).  Bind it to the
    # CPU backend for this CPU-only probe.  No model numerics change.
    import gpuwrf.contracts.state as state_contract

    state_contract._gpu_device = lambda: jax.devices("cpu")[0]

    from gpuwrf.integration.nested_pipeline import (
        NestedPipelineConfig,
        _load_domains,
        _nested_sync_mode_from_env,
        domain_names_for,
    )
    from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree
    from gpuwrf.runtime.finite_state_guard import NonFiniteStateError, assert_state_finite_at_boundary

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = out_dir / "runtime"

    names = domain_names_for(3)
    config = NestedPipelineConfig(
        input_dir=INPUT_DIR,
        output_dir=runtime_root / "unused-output",
        proof_dir=runtime_root / "unused-pipeline-proof",
        hours=0,
        max_dom=3,
        feedback=False,
    )
    started = time.time()
    hierarchy, bundles, _meta, run_start, dt_by_domain, carries = _load_domains(
        config, names
    )
    if dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
        raise RuntimeError(f"timestep hierarchy changed: {dt_by_domain!r}")
    if run_start.isoformat() != "2025-03-01T00:00:00+00:00":
        raise RuntimeError(f"run start changed: {run_start.isoformat()}")
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)

    # Identity anchor: the cold-start d03 prognostic state versus the retained
    # terminal2 step-0 carry (report-only; the A/B compares like against like).
    retained_sha = sha256_file(RETAINED_STEP0)
    if retained_sha != RETAINED_STEP0_SHA:
        raise RuntimeError("retained step-0 carry hash changed")
    import pickle

    with RETAINED_STEP0.open("rb") as handle:
        retained0 = pickle.load(handle)

    def _state_arrays(state: Any) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for attr in sorted(dir(state)):
            if attr.startswith("_"):
                continue
            value = getattr(state, attr, None)
            if value is None or callable(value):
                continue
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                arr = np.asarray(value)
                if np.issubdtype(arr.dtype, np.floating):
                    out[attr] = arr
        return out

    d03_initial = carries["d03"]
    init_arrays = _state_arrays(d03_initial.state)
    retained_arrays = _state_arrays(retained0.state)
    common = sorted(set(init_arrays) & set(retained_arrays))
    identity_mismatches = [
        name
        for name in common
        if not np.array_equal(init_arrays[name], retained_arrays[name])
    ]
    ph_initial = np.asarray(d03_initial.state.ph).copy()

    ny, nx = ph_initial.shape[1:]
    ring0 = np.zeros((ny, nx), dtype=bool)
    ring0[0, :] = True
    ring0[-1, :] = True
    ring0[:, 0] = True
    ring0[:, -1] = True

    observations: list[dict[str, Any]] = []

    class _AlarmCaptured(RuntimeError):
        """Private host-side stop after the completed alarm dispatch."""

    class Capture:
        wants_carry = True

        def __call__(self, name: str, step: int, carry: Any) -> tuple[str, int]:
            if name != "d03" or int(step) != D03_ALARM_STEP:
                return (name, int(step))
            host = jax.device_get(carry)
            state = host.state
            arrays = _state_arrays(state)
            nonfinite = {
                field: int(np.sum(~np.isfinite(arr)))
                for field, arr in arrays.items()
                if int(np.sum(~np.isfinite(arr))) > 0
            }
            ni = arrays.get("Ni")
            ni_first: list[int] | None = None
            if ni is not None and not np.all(np.isfinite(ni)):
                bad = np.argwhere(~np.isfinite(ni))
                ni_first = [
                    int(v) for v in bad[np.lexsort((bad[:, 2], bad[:, 1], bad[:, 0]))][0]
                ]
            ph_now = arrays["ph"]
            beyond_ring0_frozen = bool(
                np.array_equal(ph_now[:, ~ring0], ph_initial[:, ~ring0])
            )
            interior_max_dph = float(
                np.max(np.abs(ph_now[:, ~ring0] - ph_initial[:, ~ring0]))
            )
            guard = {"raised": False, "detail": None}
            try:
                assert_state_finite_at_boundary(
                    state,
                    domain=name,
                    step=int(step),
                    sim_time_s=float(step) * D03_DT_S,
                )
            except NonFiniteStateError as exc:
                guard = {"raised": True, "detail": str(exc)}
            observations.append(
                {
                    "domain": name,
                    "own_step": int(step),
                    "sim_time_s": float(step) * D03_DT_S,
                    "nonfinite_fields": nonfinite,
                    "ni_first_nonfinite_scan_index": ni_first,
                    "ph_beyond_ring0_bit_frozen": beyond_ring0_frozen,
                    "ph_interior_max_abs_change": interior_max_dph,
                    "max_abs_w": float(np.nanmax(np.abs(arrays["w"]))),
                    "max_abs_u": float(np.nanmax(np.abs(arrays["u"]))),
                    "finite_guard": guard,
                }
            )
            if args.stop_after_alarm:
                raise _AlarmCaptured("completed d03 step-200 observation retained")
            return (name, int(step))

    block_between, root_sync_cadence = _nested_sync_mode_from_env()
    stopped_at_alarm = False
    try:
        result = run_operational_domain_tree(
            tree,
            root_steps=ROOT_STEPS,
            feedback_enabled=False,
            output=Capture(),
            output_alarm_steps={"d03": (D03_ALARM_STEP,)},
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            carries=carries,
        )
    except _AlarmCaptured:
        if not args.stop_after_alarm:
            raise
        stopped_at_alarm = True
        result = None
    if args.stop_after_alarm and not stopped_at_alarm:
        raise RuntimeError("stop-after-alarm requested but private sentinel did not fire")
    if result is not None:
        jax.block_until_ready(result.carries["d03"].state.theta)
    wall_s = time.time() - started

    if len(observations) != 1:
        raise RuntimeError(f"expected one step-200 observation, got {len(observations)}")

    proof = {
        "schema": SCHEMA,
        "arm": args.arm,
        "model": {
            "root": str(model_root),
            "commit": head,
            "tree": tree_sha,
            "dirty_diff_sha256": args.allow_dirty_diff_sha256 or None,
        },
        "environment": env_authority,
        "backend": "cpu",
        "input_dir": str(INPUT_DIR),
        "root_steps": ROOT_STEPS,
        "d03_alarm_step": D03_ALARM_STEP,
        "runner_termination": {
            "stop_after_alarm_requested": bool(args.stop_after_alarm),
            "stopped_after_completed_alarm": stopped_at_alarm,
            "last_completed_d03_step": D03_ALARM_STEP if stopped_at_alarm else ROOT_STEPS * 9,
            "model_or_numerical_change": False,
        },
        "cold_start_vs_retained_step0": {
            "retained_sha256": retained_sha,
            "compared_leaves": len(common),
            "mismatched_leaves": identity_mismatches,
        },
        "observation": observations[0],
        "wall_seconds": wall_s,
        "gpu_commands": 0,
    }
    proof["proof_sha256"] = canonical_hash(proof)
    out_path = out_dir / f"cpu-ab-{args.arm}-proof.json"
    out_path.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
    print(f"arm={args.arm}")
    print(f"observation={json.dumps(observations[0], sort_keys=True)}")
    print(f"proof_sha256={proof['proof_sha256']}")
    print(f"written={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
