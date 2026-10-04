#!/usr/bin/env python3
"""Matched production-path timing probe for the v0234 drift bisection.

This is test infrastructure only.  It loads the real two-domain WRF fixture
through ``nested_pipeline._load_domains`` and advances it through the same
public ``run_operational_domain_tree`` entry used by the production segment
loop.  The original ``paired`` protocol runs a one-root-step initialization
call followed by two matched 1/K pairs from identical post-initialization
carries.  The paired difference

    (wall(K) - wall(1)) / (K - 1)

cancels per-call tree/AOT setup and measures the steady non-radiation root-step
cost.  The additive ``segmented`` protocol instead resumes a carry through a
small fixed number of output-like segments and can select either the released
per-call runtime lifetime or one prepared runtime reused across all segments.
It is the bounded mechanism probe for ``GPUWRF_PREPARED_RUNTIME_REUSE``.
Every protocol must remain below the next radiation refresh on every domain.

The command fails closed unless it runs inside a canonical lock-v2
``production-preemptible`` lease and the prescribed low-priority CPU envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTHORIZED_CPUS = frozenset({13, 14, 15, 29, 30, 31})
REQUIRED_CPU_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
TIMING_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "GPUWRF_ALLOCATOR",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "GPUWRF_NESTED_AOT",
    "GPUWRF_AOT_VERIFY",
    "GPUWRF_NESTED_FUSE",
    "GPUWRF_NESTED_DEFUSE_COMPILE",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE",
    "GPUWRF_NESTED_PARALLEL_COMPILE",
    "GPUWRF_NESTED_SYNC_MODE",
    "GPUWRF_BITWISE",
    "GPUWRF_GWD_NESTED",
    "GPUWRF_PREPARED_RUNTIME_REUSE",
    "GPUWRF_JAX_CACHE_DIR",
    "GPUWRF_WRF_ROOT",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_ENABLE_X64",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_FLAGS",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    os.replace(temporary, path)


def _cpu_policy() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    problems: list[str] = []
    if not affinity or not set(affinity).issubset(AUTHORIZED_CPUS):
        problems.append(
            f"affinity {affinity} is not a nonempty subset of {sorted(AUTHORIZED_CPUS)}"
        )
    observed = {key: os.environ.get(key) for key in REQUIRED_CPU_ENV}
    for key, expected in REQUIRED_CPU_ENV.items():
        if observed[key] != expected:
            problems.append(f"{key}={observed[key]!r}, expected {expected!r}")
    return {
        "ok": not problems,
        "authorized_cpus": sorted(AUTHORIZED_CPUS),
        "observed_affinity": affinity,
        "required_environment": REQUIRED_CPU_ENV,
        "observed_environment": observed,
        "problems": problems,
    }


def _source_identity(source_repo: Path, expected_sha: str | None) -> dict[str, Any]:
    source_repo = source_repo.resolve()
    head = _git(source_repo, "rev-parse", "HEAD")
    model_dirty = bool(_git(source_repo, "status", "--short", "--", "src/gpuwrf"))
    imported = Path(__import__("gpuwrf").__file__).resolve()
    expected_root = (source_repo / "src" / "gpuwrf").resolve()
    try:
        imported.relative_to(expected_root)
        import_matches = True
    except ValueError:
        import_matches = False
    problems = []
    if expected_sha and head != expected_sha:
        problems.append(f"HEAD {head} does not match expected {expected_sha}")
    if model_dirty:
        problems.append("src/gpuwrf has uncommitted changes")
    if not import_matches:
        problems.append(f"imported gpuwrf from {imported}, not {expected_root}")
    return {
        "ok": not problems,
        "repo": str(source_repo),
        "head": head,
        "expected_head": expected_sha,
        "model_tree_dirty": model_dirty,
        "imported_gpuwrf": str(imported),
        "import_matches_source_repo": import_matches,
        "problems": problems,
    }


def _lock_attestation(preflight: dict[str, Any]) -> dict[str, Any]:
    lock = preflight.get("lock") or {}
    raw_holder = lock.get("holder") if isinstance(lock, dict) else None
    problems: list[str] = []
    holder: dict[str, Any] = {}
    try:
        holder = json.loads(str(raw_holder))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        problems.append(f"holder sidecar is not valid JSON: {exc}")
    if preflight.get("status") != "PASS":
        problems.append(f"nested GPU preflight status={preflight.get('status')!r}")
    if not bool(lock.get("ok")):
        problems.append(f"lock status failed: {lock.get('reason')}")
    if holder.get("schema") != "wrf_gpu2.gpu_lock_holder" or holder.get("version") != 2:
        problems.append("holder is not lock-v2 schema version 2")
    if holder.get("intent") != "production-preemptible":
        problems.append(f"holder intent={holder.get('intent')!r}, expected production-preemptible")
    return {
        "ok": not problems,
        "preflight": preflight,
        "holder": holder,
        "problems": problems,
    }


def _paired_estimate(one_s: float, k_s: float, k_steps: int) -> float:
    if k_steps <= 1:
        raise ValueError("k_steps must exceed 1")
    value = (float(k_s) - float(one_s)) / float(k_steps - 1)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(
            f"invalid paired estimate from one={one_s:.6f}s k={k_s:.6f}s K={k_steps}"
        )
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--namespace", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--max-dom", type=int, default=2)
    parser.add_argument("--k-steps", type=int, default=4)
    parser.add_argument(
        "--protocol", choices=("paired", "segmented"), default="paired"
    )
    parser.add_argument(
        "--runtime-lifetime",
        choices=("released", "prepared"),
        default="released",
    )
    parser.add_argument("--segment-count", type=int, default=3)
    parser.add_argument("--segment-root-steps", type=int, default=1)
    parser.add_argument("--expected-load-count", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_dom != 2:
        raise SystemExit("this matched bisection is frozen to max_dom=2")
    if args.protocol == "paired" and args.k_steps < 3:
        raise SystemExit("--k-steps must be at least 3")
    if args.segment_count < 1:
        raise SystemExit("--segment-count must be positive")
    if args.segment_root_steps < 1:
        raise SystemExit("--segment-root-steps must be positive")
    if args.namespace.exists():
        raise SystemExit(f"fresh namespace already exists: {args.namespace}")
    if not args.input_dir.is_dir():
        raise SystemExit(f"input fixture is absent: {args.input_dir}")

    cpu = _cpu_policy()
    if not cpu["ok"]:
        raise SystemExit("CPU policy refusal: " + "; ".join(cpu["problems"]))

    # Import JAX/model code only after cheap filesystem and CPU-policy checks.
    import jax
    import numpy as np

    from gpuwrf.integration.nested_pipeline import (
        NestedPipelineConfig,
        _aggregate_nested_aot_reports,
        _load_domains,
        _nested_sync_mode_from_env,
        _output_cadence_steps_by_domain,
        domain_names_for,
    )
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.runtime.domain_tree import (
        DomainTree,
        _prepare_operational_domain_tree_runtime,
        maybe_prewarm_defused_nest,
        nested_aot_report,
        nested_defuse_report,
        run_operational_domain_tree,
    )
    from gpuwrf.runtime.gpu_preflight import run_nested_gpu_preflight

    source = _source_identity(args.source_repo, args.expected_source_sha)
    if not source["ok"]:
        raise SystemExit("source identity refusal: " + "; ".join(source["problems"]))

    preflight = run_nested_gpu_preflight()
    lock = _lock_attestation(preflight)
    if not lock["ok"]:
        raise SystemExit("lock-v2 refusal: " + "; ".join(lock["problems"]))

    args.namespace.mkdir(parents=True, exist_ok=False)
    output_dir = args.namespace / "unused_output"
    proof_dir = args.namespace / "pipeline_proofs"
    scratch_dir = args.namespace / "scratch"
    output_dir.mkdir()
    proof_dir.mkdir()
    scratch_dir.mkdir()

    payload: dict[str, Any] = {
        "schema": "wrfgpu2-v0234-drift-perf-probe-v1",
        "schema_version": 1,
        "status": "STARTED",
        "started_utc": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "probe_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "source": source,
        "fixture": str(args.input_dir.resolve()),
        "namespace": str(args.namespace.resolve()),
        "k_steps": args.k_steps,
        "protocol": args.protocol,
        "runtime_lifetime": args.runtime_lifetime,
        "segment_count": args.segment_count,
        "segment_root_steps": args.segment_root_steps,
        "expected_load_count": args.expected_load_count,
        "cpu_policy": cpu,
        "lock_v2": lock,
        "environment": {key: os.environ.get(key) for key in TIMING_ENV_KEYS},
        "profiler": {
            "used": False,
            "reason": "bounded matched timing discriminator; profiler arm not yet authorized",
        },
        "calls": [],
    }

    def checkpoint(status: str) -> None:
        payload["status"] = status
        payload["updated_utc"] = _utc_now()
        _write_json(args.output_json, payload)
        _write_json(args.namespace / "PERF_PROBE.json", payload)

    checkpoint("ATTESTED")
    print(
        "MARKER:ATTESTED "
        f"source={source['head']} lease={lock['holder'].get('lease_id')} "
        f"intent={lock['holder'].get('intent')}",
        flush=True,
    )

    names = domain_names_for(args.max_dom)
    cfg = NestedPipelineConfig(
        input_dir=args.input_dir,
        output_dir=output_dir,
        proof_dir=proof_dir,
        hours=1,
        max_dom=args.max_dom,
        scratch_dir=scratch_dir,
    )
    load_t0 = time.perf_counter()
    hierarchy, bundles, load_meta, run_start, dt_by_domain, initial_carries = _load_domains(
        cfg, names
    )
    load_s = time.perf_counter() - load_t0
    cadence_run = Gen2Run(args.input_dir)
    output_cadence, history_interval = _output_cadence_steps_by_domain(
        cadence_run, names, dt_by_domain
    )
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=bool(cfg.feedback))
    block_between, root_sync_cadence = _nested_sync_mode_from_env()
    radiation_cadence = {
        name: int(bundles[name].namelist.radiation_cadence_steps) for name in names
    }
    first_next_radiation_root = min(
        radiation_cadence[name]
        * float(dt_by_domain[name])
        / float(dt_by_domain[names[0]])
        for name in names
    )
    maximum_protocol_steps = (
        args.k_steps
        if args.protocol == "paired"
        else args.segment_count * args.segment_root_steps
    )
    if maximum_protocol_steps >= first_next_radiation_root:
        raise RuntimeError(
            f"protocol reaches root step {maximum_protocol_steps}, at or beyond "
            "the next radiation refresh at root step "
            f"{first_next_radiation_root:g}"
        )
    payload["setup"] = {
        "load_wall_s": load_s,
        "run_start_utc": run_start.isoformat(),
        "domains": list(names),
        "dt_s": {name: float(dt_by_domain[name]) for name in names},
        "grid_theta_shapes": {
            name: list(initial_carries[name].state.theta.shape) for name in names
        },
        "radiation_cadence_steps": radiation_cadence,
        "next_radiation_refresh_root_step": first_next_radiation_root,
        "history_interval_minutes": history_interval,
        "output_cadence_steps": output_cadence,
        "feedback": bool(cfg.feedback),
        "block_between": block_between,
        "root_sync_cadence": root_sync_cadence,
        "protocol_maximum_root_steps": maximum_protocol_steps,
        "runtime_lifetime": args.runtime_lifetime,
        "persistent_state_bytes": tree.persistent_state_bytes(),
        "load_meta_domain_keys": sorted((load_meta.get("domains") or {}).keys()),
        "nested_frozen_wrf_boundary_bundle": {
            name: bool(
                bundles[name].namelist.boundary_config.nested_frozen_wrf_boundary_bundle
            )
            for name in names
        },
        "jax": {
            "version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "enable_x64": bool(jax.config.jax_enable_x64),
        },
    }
    prewarm_t0 = time.perf_counter()
    prewarm = maybe_prewarm_defused_nest(tree, carries=initial_carries)
    payload["setup"]["prewarm"] = {
        "wall_s": time.perf_counter() - prewarm_t0,
        "report": prewarm,
    }
    checkpoint("LOADED")
    print(
        f"MARKER:LOADED wall_s={load_s:.3f} dt={dt_by_domain[names[0]]} "
        f"rad_refresh_root={first_next_radiation_root:g} prewarm={prewarm.get('source')}",
        flush=True,
    )

    prepared_runtime = (
        _prepare_operational_domain_tree_runtime(
            tree, feedback_enabled=bool(cfg.feedback)
        )
        if args.runtime_lifetime == "prepared"
        else None
    )
    payload["setup"]["prepared_runtime_constructed"] = prepared_runtime is not None
    call_aot_reports: list[dict[str, Any]] = []

    def one_call(
        *, label: str, root_steps: int, carries: dict[str, Any], own_steps: dict[str, int]
    ) -> Any:
        started = _utc_now()
        t0 = time.perf_counter()
        result = run_operational_domain_tree(
            tree,
            root_steps=root_steps,
            feedback_enabled=bool(cfg.feedback),
            output=None,
            output_cadence_steps=output_cadence,
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            carries=carries,
            initial_own_steps=own_steps,
            max_event_tail=64,
            prepared_runtime=prepared_runtime,
        )
        jax.block_until_ready(tuple(state.theta for state in result.states.values()))
        wall_s = time.perf_counter() - t0
        finite = {
            name: bool(np.isfinite(np.asarray(result.states[name].theta)).all()) for name in names
        }
        aot_report = nested_aot_report()
        record = {
            "label": label,
            "started_utc": started,
            "finished_utc": _utc_now(),
            "root_steps": root_steps,
            "wall_s": wall_s,
            "finite_theta": finite,
            "own_steps": {name: int(result.own_steps[name]) for name in names},
            "aot": aot_report,
            "defuse": nested_defuse_report(),
            "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
        call_aot_reports.append(aot_report)
        payload["calls"].append(record)
        checkpoint(f"CALL_{label}_DONE")
        print(
            f"CALL {label} root_steps={root_steps} wall_s={wall_s:.6f} "
            f"finite={all(finite.values())} own_steps={record['own_steps']}",
            flush=True,
        )
        if not all(finite.values()):
            raise RuntimeError(f"{label} produced nonfinite theta")
        return result

    def pytree_digest(value: Any) -> dict[str, Any]:
        """Hash every terminal numeric pytree leaf after the timed region."""

        entries: list[dict[str, Any]] = []
        problems: list[str] = []
        flattened, _treedef = jax.tree_util.tree_flatten_with_path(value)
        for path, leaf in flattened:
            path_text = jax.tree_util.keystr(path)
            try:
                host = np.asarray(jax.device_get(leaf))
                if host.dtype.hasobject:
                    raise TypeError(f"object dtype {host.dtype}")
                contiguous = np.ascontiguousarray(host)
                digest = hashlib.sha256()
                digest.update(str(contiguous.dtype).encode("utf-8"))
                digest.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
                digest.update(contiguous.tobytes(order="C"))
                entries.append(
                    {
                        "path": path_text,
                        "shape": list(contiguous.shape),
                        "dtype": str(contiguous.dtype),
                        "sha256": digest.hexdigest(),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - proof must fail closed
                problems.append(f"{path_text}: {type(exc).__name__}: {exc}")
        combined = hashlib.sha256()
        for entry in entries:
            combined.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
        return {
            "ok": not problems and bool(entries),
            "leaf_count": len(entries),
            "sha256": combined.hexdigest(),
            "entries": entries,
            "problems": problems,
        }

    root_dt = float(dt_by_domain[names[0]])
    initial_own = {name: 0 for name in names}
    if args.protocol == "paired":
        initialized = one_call(
            label="INITIAL_1",
            root_steps=1,
            carries=initial_carries,
            own_steps=initial_own,
        )
        steady_carries = initialized.carries
        steady_own = {name: int(initialized.own_steps[name]) for name in names}

        # Each timed call starts from the exact same immutable carry/clock state.
        one_call(
            label="PAIR_A_1",
            root_steps=1,
            carries=steady_carries,
            own_steps=steady_own,
        )
        one_call(
            label="PAIR_A_K",
            root_steps=args.k_steps,
            carries=steady_carries,
            own_steps=steady_own,
        )
        one_call(
            label="PAIR_B_1",
            root_steps=1,
            carries=steady_carries,
            own_steps=steady_own,
        )
        terminal = one_call(
            label="PAIR_B_K",
            root_steps=args.k_steps,
            carries=steady_carries,
            own_steps=steady_own,
        )

        by_label = {item["label"]: item for item in payload["calls"]}
        estimates = [
            _paired_estimate(
                by_label[f"PAIR_{pair}_1"]["wall_s"],
                by_label[f"PAIR_{pair}_K"]["wall_s"],
                args.k_steps,
            )
            for pair in ("A", "B")
        ]
        median_step = statistics.median(estimates)
        payload["result"] = {
            "method": "paired difference (K-call minus 1-call)/(K-1), repeated twice",
            "timing_boundary": (
                "public run_operational_domain_tree entry through "
                "block_until_ready(theta); output disabled; identical "
                "post-initialization carries and own_steps"
            ),
            "radiation_excluded": True,
            "paired_s_per_root_step": estimates,
            "median_s_per_root_step": median_step,
            "spread_fraction": abs(estimates[0] - estimates[1]) / median_step,
            "root_dt_s": root_dt,
            "projected_s_per_fc_hour": median_step * 3600.0 / root_dt,
            "projected_wall_hours_per_fc_hour": median_step / root_dt,
            "all_theta_finite": all(
                all(item["finite_theta"].values()) for item in payload["calls"]
            ),
            "aot_last": nested_aot_report(),
            "defuse_last": nested_defuse_report(),
            "terminal_states": pytree_digest(terminal.states),
            "terminal_carries": pytree_digest(terminal.carries),
        }
    else:
        carries = initial_carries
        own_steps = initial_own
        terminal = None
        for index in range(args.segment_count):
            terminal = one_call(
                label=f"SEGMENT_{index + 1}",
                root_steps=args.segment_root_steps,
                carries=carries,
                own_steps=own_steps,
            )
            carries = terminal.carries
            own_steps = {
                name: int(terminal.own_steps[name]) for name in names
            }
        assert terminal is not None
        if args.runtime_lifetime == "prepared":
            process_aot = nested_aot_report()
        else:
            process_aot = _aggregate_nested_aot_reports(call_aot_reports)
        load_count = int(process_aot.get("load_count", 0))
        total_wall = sum(float(item["wall_s"]) for item in payload["calls"])
        total_steps = args.segment_count * args.segment_root_steps
        post_first = [float(item["wall_s"]) for item in payload["calls"][1:]]
        state_digest = pytree_digest(terminal.states)
        carry_digest = pytree_digest(terminal.carries)
        expected_load_ok = (
            args.expected_load_count is None
            or load_count == args.expected_load_count
        )
        payload["result"] = {
            "method": "contiguous resumed fixed-size segments",
            "timing_boundary": (
                "sum of public run_operational_domain_tree segment entries "
                "through block_until_ready(theta); output disabled"
            ),
            "radiation_excluded": True,
            "runtime_lifetime": args.runtime_lifetime,
            "segment_count": args.segment_count,
            "segment_root_steps": args.segment_root_steps,
            "segment_wall_s": [
                float(item["wall_s"]) for item in payload["calls"]
            ],
            "total_segment_wall_s": total_wall,
            "post_first_segment_median_wall_s": (
                statistics.median(post_first) if post_first else None
            ),
            "mean_s_per_root_step_including_segment_overhead": total_wall
            / total_steps,
            "projected_s_per_fc_hour_including_segment_overhead": (
                total_wall / total_steps * 3600.0 / root_dt
            ),
            "root_dt_s": root_dt,
            "all_theta_finite": all(
                all(item["finite_theta"].values()) for item in payload["calls"]
            ),
            "aot_process_lifetime": process_aot,
            "observed_load_count": load_count,
            "expected_load_count": args.expected_load_count,
            "expected_load_count_ok": expected_load_ok,
            "defuse_last": nested_defuse_report(),
            "terminal_states": state_digest,
            "terminal_carries": carry_digest,
        }
        if not state_digest["ok"] or not carry_digest["ok"]:
            checkpoint("FAIL_TERMINAL_HASH")
            raise RuntimeError("terminal pytree digest failed closed")
        if not expected_load_ok:
            checkpoint("FAIL_LOAD_COUNT")
            raise RuntimeError(
                f"observed {load_count} real AOT loads; expected "
                f"{args.expected_load_count}"
            )
    payload["finished_utc"] = _utc_now()
    checkpoint("PASS")
    if args.protocol == "paired":
        print(
            "SUMMARY status=PASS "
            f"s_per_root_step={payload['result']['median_s_per_root_step']:.6f} "
            f"s_per_fc_hour={payload['result']['projected_s_per_fc_hour']:.3f} "
            "pair_estimates="
            + ",".join(
                f"{value:.6f}" for value in payload["result"]["paired_s_per_root_step"]
            ),
            flush=True,
        )
    else:
        print(
            "SUMMARY status=PASS "
            f"runtime_lifetime={args.runtime_lifetime} "
            f"segment_wall_s={payload['result']['total_segment_wall_s']:.6f} "
            f"loads={payload['result']['observed_load_count']} "
            f"state_sha256={payload['result']['terminal_states']['sha256']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
