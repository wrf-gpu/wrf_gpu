"""V0234 step-9000 cross-process nondeterminism discriminator v2 (Kimi turn).

PURPOSE
-------
The post-Fable full-tree replay (resource_retry1) stopped fail-closed at d03
step 9000 because its candidate bytes differed from the admitted known-V10-only
record (OBSERVATION_DRIFT) while V/V10 exceeded the frozen Retry20 ceilings.
Retained evidence shows four same-model full-tree runs (model tree 835dcc29,
byte-identical lowered d03 audit StableHLO b12b3d64...) produced four
byte-distinct d03-step-200 frames, and that toolingrepair1 and toolingrepair2
diverge at d01 step 67 under byte-identical software and environment: the
full-tree GPU path is not bit-reproducible across processes. The leading
mechanism is XLA:GPU autotune kernel selection, which is per-process and
timing-dependent when no autotune cache or results file pins the picks
(production launches set neither).

This discriminator binds the question to the EXACT production tree path and
the EXACT first-divergence frame. Each arm replays the production startup:

1. load the corrected 3-domain tree (production loader, production env);
2. run the production d03 audit lower+compile exactly as production's
   D03_LOWER_COMPILE stage and gate the lowered StableHLO byte-hash to
   production's b12b3d64... (fail-closed);
3. integrate the live tree from step 0 along the production output schedule
   and capture the d03 step-200 state payload (00:20) -- the first frame at
   which all four production runs diverged -- hashing every state leaf with
   the production manifest scheme.

Arms:
- A: fresh autotune (production-like), results dumped to file A;
- B: fresh autotune (production-like), results dumped to file B;
- C: autotune results LOADED from arm A's dump (pinned picks).

Interpretation:
- manifest(A) != manifest(B): per-process nondeterminism reproduced on the
  production tree path from byte-identical inputs;
- dump(A) != dump(B): autotune picks differ -- mechanism identified;
- manifest(C) == manifest(A): pinned autotune restores bit-reproducibility
  on the production path -- harness repair validated;
- manifest(C) != manifest(A) with equal dumps: in-process nondeterminism
  instead (different repair).

No model bytes change. No tolerance changes. GPU access only via the
canonical lock-v2 wrapper, intent production-preemptible, one job at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle  # noqa: F401  (kept for symmetry with retained tooling)
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import v0234_nested_frozen_wrf_boundary_window as runner  # noqa: E402


EXPECTED_MODEL_TREE = "835dcc29bf316c0715b41a72e064985e9cf099df"
EXPECTED_RUNNER_SHA256 = "27f5c675b69599a5112951c4f12e4a7fa11918c7494f95ca6583429a692930e0"
EXPECTED_STABLEHLO_SHA256 = (
    "b12b3d64a262326516d706d138e4ff6fe43e831380bad4e3f6651e9d117149cf"
)
EXPECTED_HLO_ARTIFACT_FILE_SHA256 = (
    "15575931876ca3126e91e5b1c7cbcd85360e7cc2754354f262c5c82556d3a308"
)
PRODUCTION_CANDIDATE_COMMIT = "470e6111d516479bed4bc0c3b2be1007bb082afd"
PRODUCTION_CANDIDATE_TREE = "bb0d7c9a4fe7befdf3120bdfdabde11db985682a"
CAPTURE_DOMAIN = "d03"
CAPTURE_STEP = 200
ROOT_STEPS = 23  # 23 d01 steps -> d03 207 >= 200 (first d03 alarm at 00:20)
TERMINAL_OWN_STEPS = {"d01": 1200, "d02": 3600, "d03": 10800}

# Module constants the production profile binds for the HLO audit artifact.
runner.CANDIDATE_COMMIT = PRODUCTION_CANDIDATE_COMMIT
runner.CANDIDATE_TREE = PRODUCTION_CANDIDATE_TREE


class _ShallowStop(RuntimeError):
    """Internal unwind: the d03-200 payload was captured; stop integrating."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_arm(arm: str, arm_dir: Path) -> dict:
    """Replay the production startup and capture the d03-200 state payload."""

    result: dict = {
        "schema": "gpuwrf.v0234.step9000-autotune-discriminator-arm.v2",
        "arm": arm,
        "started_utc": _utcnow(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "JAX_ENABLE_COMPILATION_CACHE",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "GPUWRF_ALLOCATOR",
                "GPUWRF_FINITE_CHECK",
                "GPUWRF_BITWISE",
                "GPUWRF_BATCH_ENSEMBLE",
                "GPUWRF_ADVANCE_CHUNK_LOOP",
                "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE",
                "GPUWRF_WRF_ROOT",
            )
        },
    }
    result["preemption_start"] = runner.assert_preemption_clear(f"arm-{arm}-start")

    model_tree = runner._git(REPO_ROOT, "rev-parse", "HEAD:src/gpuwrf")
    if model_tree != EXPECTED_MODEL_TREE:
        raise runner.RunnerGateError("MODEL_TREE", model_tree)
    runner_sha = hashlib.sha256(
        (REPO_ROOT / "scripts/v0234_nested_frozen_wrf_boundary_window.py")
        .read_bytes()
    ).hexdigest()
    if runner_sha != EXPECTED_RUNNER_SHA256:
        raise runner.RunnerGateError("RUNNER_SOURCE", runner_sha)
    result["authority"] = {
        "model_tree": model_tree,
        "runner_source_sha256": runner_sha,
    }

    runtime = runner._import_runtime()
    devices = runtime.jax.devices()
    if not any(getattr(d, "platform", None) == "gpu" for d in devices):
        raise runner.RunnerGateError("CUDA_RUNTIME", repr(devices))
    result["devices"] = [str(d) for d in devices]

    load_dir = arm_dir / "load"
    load_dir.mkdir(parents=True, exist_ok=False)
    tree, names, initial_carries, dt_by_domain, load_authority = (
        runtime.ordinary.load_corrected_tree(load_dir)
    )
    if names != ("d01", "d02", "d03") or dt_by_domain != {
        "d01": 54.0,
        "d02": 18.0,
        "d03": 6.0,
    }:
        raise runner.RunnerGateError("DOMAIN_AUTHORITY", repr((names, dt_by_domain)))
    result["load_authority"] = load_authority
    runner.assert_preemption_clear(f"arm-{arm}-post-load")

    # Production D03_LOWER_COMPILE stage, on the initial carry exactly like
    # production; gate the lowered program byte-hash and the artifact file.
    _executable, _namelist, _clock, _cadence, audit = (
        runner._lower_compile_d03_once(
            tree,
            initial_carries["d03"],
            runtime,
            artifact_path=arm_dir / "d03-one-step-lowered-hlo.json",
        )
    )
    stablehlo_sha = audit["stablehlo_sha256"]
    artifact_file_sha = audit["lowered_hlo_artifact"]["file_sha256"]
    if stablehlo_sha != EXPECTED_STABLEHLO_SHA256:
        raise runner.RunnerGateError("STABLEHLO_DRIFT", stablehlo_sha)
    result["compile"] = {
        "stablehlo_sha256": stablehlo_sha,
        "lowered_hlo_artifact_file_sha256": artifact_file_sha,
        "lowered_hlo_artifact_matches_production_byte_file": (
            artifact_file_sha == EXPECTED_HLO_ARTIFACT_FILE_SHA256
        ),
        "lower_wall_seconds": audit["lower_wall_seconds"],
        "compile_wall_seconds": audit["compile_wall_seconds"],
        "hlo_policy_passed": audit["hlo_policy"]["passed"],
    }
    runner.assert_preemption_clear(f"arm-{arm}-post-compile")

    cadence, nonintegral, schedule = runtime.ordinary.scheduler_contract(
        names, dt_by_domain, total_steps=TERMINAL_OWN_STEPS,
    )
    result["schedule_alarm_d03_first"] = schedule["output_alarm_schedule"][
        "d03"
    ][0]

    captured: dict[str, dict] = {}
    events: list[dict] = []

    def capture_output(name, current_step, payload):
        events.append({"domain": name, "own_step": int(current_step)})
        host = runtime.jax.device_get(payload)
        manifest = runtime.ordinary.host_tree_manifest(host)
        nonfinite = int(manifest.get("floating_nonfinite_count", 0))
        captured[f"{name}-{current_step}"] = {
            "manifest_sha256": str(manifest["manifest_sha256"]),
            "leaf_count": int(manifest["leaf_count"]),
            "floating_nonfinite_count": nonfinite,
        }
        if nonfinite != 0:
            raise runner.RunnerGateError(
                "CAPTURE_NONFINITE", f"{name} step {current_step}"
            )
        if name == CAPTURE_DOMAIN and int(current_step) == CAPTURE_STEP:
            raise _ShallowStop(f"captured {name} step {current_step}")
        return {"domain": name, "own_step": int(current_step)}

    block_between, root_sync_cadence = runtime._nested_sync_mode_from_env()
    integrate_started = time.perf_counter()
    try:
        runtime.run_operational_domain_tree(
            tree,
            root_steps=ROOT_STEPS,
            feedback_enabled=False,
            output=capture_output,
            output_cadence_steps=cadence,
            output_alarm_steps=nonintegral,
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            carries=initial_carries,
            initial_own_steps={name: 0 for name in names},
        )
    except _ShallowStop:
        pass
    result["dispatch"] = {
        "root_steps_requested": ROOT_STEPS,
        "wall_seconds": time.perf_counter() - integrate_started,
        "events": events,
        "captured": captured,
    }
    key = f"{CAPTURE_DOMAIN}-{CAPTURE_STEP}"
    if key not in captured:
        raise runner.RunnerGateError(
            "CAPTURE_MISSED", f"events={events!r} captured={list(captured)!r}"
        )
    runner.assert_preemption_clear(f"arm-{arm}-post-integrate")
    result["finished_utc"] = _utcnow()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("A", "B", "C"))
    parser.add_argument("--arm-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    arm_dir: Path = args.arm_dir
    arm_dir.mkdir(parents=True, exist_ok=True)
    result_path = arm_dir / f"arm-{args.arm}-result.json"
    started = time.perf_counter()
    try:
        result = run_arm(args.arm, arm_dir)
    except BaseException as exc:  # noqa: BLE001 - fail-closed retention
        result = {
            "schema": "gpuwrf.v0234.step9000-autotune-discriminator-arm.v2",
            "arm": args.arm,
            "failed": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "recorded_utc": _utcnow(),
        }
        rc = 3
    else:
        result["wall_seconds"] = time.perf_counter() - started
        rc = 0
    unsigned = dict(result)
    result["proof_sha256"] = runner.canonical_digest(unsigned)
    runner.atomic_write_json(result_path, result)
    print(
        f"STEP9000_DISCRIMINATOR_ARM arm={args.arm} rc={rc} "
        f"proof={result['proof_sha256']}",
        flush=True,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
