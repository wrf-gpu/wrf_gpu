#!/usr/bin/env python3
"""
Run ONE v0.12.0 d02 24h GPU forecast for the powered-TOST campaign.

v0.13 KI-5 wrfbdy fix (root cause + routing change documented in
``proofs/v013/tost_wrfbdy_fix.md``):

The L2 corpus cases are a max_dom=2 ONE-WAY nest (namelist.input: ``nested =
.false., .true.``; ``specified = .true., .false.``).  real.exe writes lateral
boundary forcing ONLY for the outermost specified domain, so each case dir has
``wrfinput_d01`` + ``wrfinput_d02`` + ``wrfbdy_d01`` but NO ``wrfbdy_d02`` (a
nest never gets its own wrfbdy -- its LBC comes from the live parent d01).

The OLD per-case path forced a d02-ONLY single-domain standalone forecast
(``build_l2_daily_case`` -> ``build_replay_case`` standalone branch), which then
demanded ``wrfbdy_d02`` and failed rc=2 with
``FileNotFoundError: standalone native-init requires wrfbdy_d02``.

The FIX routes the per-case forecast through the SAME standalone live-nested
driver the production nested CLI uses (``gpuwrf.integration.nested_pipeline.
execute_nested_pipeline``, max_dom=2 / the ``python -m gpuwrf.cli run --max-dom 2``
path): d01 runs standalone (IC ``wrfinput_d01`` + LBC ``wrfbdy_d01``) and feeds
d02's lateral boundary LIVE each parent step -- NO ``wrfbdy_d02`` needed.  The
nested driver writes one ``wrfout_<domain>_<valid_time>`` per domain into the
output dir; the d02 wrfouts are the ones the TOST scorer compares against the
CPU-WRF d02 truth + AEMET.  The scoring config (d02 T2/U10/V10) is unchanged.

This reuses the existing, validated native-init/live-nest runtime; it does NOT
reimplement any init/LBC code.

Usage (called by run_powered_tost_n15_v0120.py via the repo GPU lock wrapper):
    scripts/run_gpu_lowprio.sh --cores 0-3 -- \\
        python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \\
        --run-root /tmp/v0120_merged_run_root \\
        --run-id <RUN_ID> \\
        --hours 24 \\
        --output-root /tmp/v0120_powered_tost_runs \\
        --proof-dir /path/to/proof/subdir

    # CPU setup-only verification (no forecast; proves d01+d02 build cleanly):
    JAX_PLATFORMS=cpu python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \\
        --run-id <RUN_ID> --setup-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# Root relative to __file__ → .../worktrees/<this worktree>
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for _p in [str(SRC), str(ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpuwrf.integration.daily_pipeline import resolve_run_dir, write_json   # noqa: E402
from gpuwrf.integration.nested_pipeline import (                            # noqa: E402
    NestedPipelineConfig,
    domain_names_for,
    execute_nested_pipeline,
)
from scripts.m7_l2_d02_replay import (                                      # noqa: E402
    _pin_orchestration_cpus,
    _write_blocked_proofs,
    _write_json,
    write_tier4_rmse,
    write_bounds_check,
    write_wall_clock,
    L2_RUN_ROOT,
    OUTPUT_ROOT,
)

# The L2 cases are max_dom=2 one-way nests; the TOST forecast scores d02.
SCORE_DOMAIN = "d02"
MAX_DOM = 2


def setup_only_check(run_id: str, run_root: Path, hours: int) -> dict:
    """Verify the per-case LBC routing CPU-side WITHOUT a GPU or a forecast.

    The device-resident ``State.zeros`` constructor mandates a JAX GPU backend
    (``src/gpuwrf/contracts/state.py:_gpu_device``), so the full
    ``_load_domains`` device build cannot run CPU-only.  This check instead
    exercises everything that decides WHICH lateral-boundary source each domain
    reads from disk -- the exact logic that raised the wrfbdy_d02
    FileNotFoundError on the old single-domain path -- with no device op:

      * the case dir resolves and is a max_dom=2 ONE-WAY nest (so the
        live-nested d01->d02 path is the correct one);
      * d01's LBC source ``wrfbdy_d01`` RESOLVES via the same
        ``wrfbdy_path_for_run`` the d01 root case uses;
      * d02 is a nest with NO ``wrfbdy_d02`` on disk (none ever exists for a
        nest) -- and the live-nested d02 child path
        (``build_replay_case(domain='d02', load_lateral_boundaries=False)``)
        takes the ``not load_lateral_boundaries`` branch, which leaves the
        ``*_bdy`` leaves empty and NEVER calls ``load_wrfbdy_boundary_leaves``,
        so ``wrfbdy_d02`` is never read.

    Returns a JSON-serialisable report; raises on any setup inconsistency.
    """
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.boundary_replay import wrfbdy_path_for_run

    run_dir = resolve_run_dir(run_id, run_root)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing run directory: {run_dir}")
    names = domain_names_for(MAX_DOM)

    run = Gen2Run(run_dir)
    namelist = run.namelist

    # max_dom=2 one-way nest sanity (namelist: nested=.false.,.true. /
    # specified=.true.,.false. / feedback=0).
    def _list1(group, key, default=None):
        raw = namelist.get(group, {}).get(key, default)
        return raw if isinstance(raw, (list, tuple)) else [raw]

    max_dom = int(namelist.get("domains", {}).get("max_dom", 1))
    nested = _list1("bdy_control", "nested")
    specified = _list1("bdy_control", "specified")
    feedback = int(namelist.get("domains", {}).get("feedback", 0))

    # d01 LBC source must resolve (this is what the d01 root case reads).
    d01_bdy = wrfbdy_path_for_run(run, "d01")  # raises if missing

    # d02 has no wrfbdy (correct for a nest); the live child never reads one.
    wrfbdy_d02 = run_dir / "wrfbdy_d02"
    wrfbdy_d02_present = wrfbdy_d02.exists()

    # Per-domain grids load (Gen2GridSpec carries the nest metadata the
    # live-nested driver reads: parent_id / parent_grid_ratio).
    gen2_grids = {n: run.grid(n) for n in names}
    grids = {n: g.as_grid_spec() for n, g in gen2_grids.items()}
    child = names[1]
    nest_edge = {
        "child": child,
        "parent": f"d{int(gen2_grids[child].parent_id):02d}",
        "parent_grid_ratio": int(gen2_grids[child].parent_grid_ratio),
    }

    present = {f"wrfinput_{n}": (run_dir / f"wrfinput_{n}").exists() for n in names}
    present["wrfbdy_d01"] = d01_bdy.exists()
    present["wrfbdy_d02"] = wrfbdy_d02_present

    problems = []
    if max_dom < 2:
        problems.append(f"max_dom={max_dom} (<2): not a nest")
    if not all(present[f"wrfinput_{n}"] for n in names):
        problems.append("missing a wrfinput file")
    if not present["wrfbdy_d01"]:
        problems.append("wrfbdy_d01 missing (d01 LBC source)")
    if wrfbdy_d02_present:
        problems.append("wrfbdy_d02 unexpectedly present (a nest should have none)")
    if nest_edge["parent_grid_ratio"] <= 1:
        problems.append(f"d02 parent_grid_ratio={nest_edge['parent_grid_ratio']} (must be >1 for a child)")
    if problems:
        raise RuntimeError("setup-only check failed: " + "; ".join(problems))

    return {
        "schema": "PoweredTOSTSetupOnlyCheck",
        "schema_version": 1,
        "verdict": "SETUP_OK",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "init_mode": "standalone_native_init_nested",
        "score_domain": SCORE_DOMAIN,
        "max_dom": int(max_dom),
        "domains": list(names),
        "nest_config": {
            "nested": [bool(x) for x in nested],
            "specified": [bool(x) for x in specified],
            "feedback": int(feedback),
            "one_way": feedback == 0,
        },
        "grids_loaded": {
            n: {"nz": int(g.nz), "ny": int(g.ny), "nx": int(g.nx)}
            for n, g in grids.items()
        },
        "nest_edge": nest_edge,
        "disk_inputs": present,
        "d01_lbc_source": str(d01_bdy),
        "wrfbdy_d02_required": False,
        "wrfbdy_d02_present": wrfbdy_d02_present,
        "old_path_would_fail_here": (
            "the OLD single-domain d02 path called load_wrfbdy_boundary_leaves -> "
            "wrfbdy_path_for_run(run,'d02') -> FileNotFoundError(wrfbdy_d02)"
        ),
        "new_path": (
            "live-nested: d01 LBC from wrfbdy_d01 (resolved above); d02 child uses "
            "build_replay_case(load_lateral_boundaries=False) -> empty *_bdy leaves, "
            "parent feeds d02 LBC live; load_wrfbdy_boundary_leaves NEVER called for d02"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",       type=Path, default=L2_RUN_ROOT)
    parser.add_argument("--cpu-truth-root", type=Path,
                        default=Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output"),
                        help="Root for CPU-WRF reference wrfouts")
    parser.add_argument("--run-id",    required=True)
    parser.add_argument("--hours",     type=int, default=24)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--proof-dir", type=Path)
    parser.add_argument("--setup-only", action="store_true",
                        help="CPU-safe: build d01+d02 cases (LBC wiring) and exit; "
                             "no forecast. Proves the wrfbdy_d02 error is gone.")
    args = parser.parse_args(argv)

    proof_dir = args.proof_dir or (ROOT / ".agent/sprints/powered_tost_n15_v0120" / args.run_id)
    proof_dir.mkdir(parents=True, exist_ok=True)

    # ── CPU setup-only verification (no GPU, no forecast) ──────────────────────
    if args.setup_only:
        report = setup_only_check(args.run_id, Path(args.run_root), int(args.hours))
        _write_json(proof_dir / "setup_only_check.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    affinity  = _pin_orchestration_cpus()
    # The nested driver writes wrfout_d01_* AND wrfout_d02_* into one output dir;
    # keep the historical l2_d02_<RID> name so the orchestrator scorer (which
    # globs wrfout_d02_* there) finds the d02 history unchanged.
    output_dir = args.output_root / f"l2_d02_{args.run_id}"
    run_dir = resolve_run_dir(args.run_id, args.run_root)
    cpu_truth_dir = args.cpu_truth_root / args.run_id
    if not cpu_truth_dir.is_dir():
        cpu_truth_dir = run_dir  # fallback

    config = NestedPipelineConfig(
        input_dir=run_dir,
        output_dir=output_dir,
        proof_dir=proof_dir,
        hours=int(args.hours),
        max_dom=MAX_DOM,
        feedback=False,  # one-way nest (namelist feedback=0)
    )

    blocked_reason: str | None = None
    pipeline_payload: dict = {}
    try:
        pipeline_payload = execute_nested_pipeline(config)
    except Exception as exc:  # noqa: BLE001 - surface cleanly as a blocked case
        reason = f"{type(exc).__name__}: {exc}"
        blocked_reason = reason
        _write_blocked_proofs(proof_dir, reason=reason, detail={"run_dir": str(run_dir)})
        verdict = "L2_D02_BLOCKED"
        rmse = bounds = wall = {"status": "BLOCKED", "reason": reason}
        summary = _summary(args, run_dir, output_dir, verdict, blocked_reason,
                           proof_dir, pipeline_payload, rmse, bounds, wall)
        _write_json(proof_dir / "l2_d02_validation_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2

    if affinity is not None:
        pipeline_payload["orchestration_cpu_affinity"] = affinity
    write_json(proof_dir / "pipeline_run_l2_d02.json", pipeline_payload)

    # d02 wrfout history written by the nested driver (the scored domain).
    d02_files = sorted(output_dir.glob(f"wrfout_{SCORE_DOMAIN}_*"))
    try:
        if pipeline_payload.get("verdict") not in ("PIPELINE_GREEN", "PIPELINE_PARTIAL") \
                or not d02_files:
            reason = pipeline_payload.get("reason", "nested pipeline produced no d02 wrfout")
            blocked_reason = str(reason)
            _write_blocked_proofs(proof_dir, reason=str(reason), detail=pipeline_payload)
            verdict = "L2_D02_BLOCKED"
            rmse = bounds = wall = {"status": "BLOCKED"}
        else:
            final_wrfout = d02_files[-1]
            rmse = write_tier4_rmse(
                final_wrfout=final_wrfout,
                reference_run_dir=cpu_truth_dir,
                proof_path=proof_dir / "tier4_rmse_l2_d02.json",
            )
            bounds = write_bounds_check(
                wrfout_files=[str(p) for p in d02_files],
                proof_path=proof_dir / "bounds_check_l2_d02.json",
            )
            wall = write_wall_clock(
                pipeline_payload=pipeline_payload,
                proof_path=proof_dir / "wall_clock_l2_d02.json",
                run_dir=run_dir,
                affinity=affinity,
            )
            verdict = (
                "L2_D02_GREEN"
                if rmse["status"] == "PASS" and bounds["status"] == "PASS"
                else "L2_D02_BOUNDED_FAIL"
            )
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        blocked_reason = reason
        detail = {"pipeline_payload": dict(pipeline_payload), "run_dir": str(run_dir)}
        _write_blocked_proofs(proof_dir, reason=reason, detail=detail)
        verdict = "L2_D02_BLOCKED"
        rmse = bounds = wall = {"status": "BLOCKED", "reason": reason}

    summary = _summary(args, run_dir, output_dir, verdict, blocked_reason,
                       proof_dir, pipeline_payload, rmse, bounds, wall)
    _write_json(proof_dir / "l2_d02_validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if verdict in ("L2_D02_GREEN", "L2_D02_BOUNDED_FAIL") else 2


def _summary(args, run_dir, output_dir, verdict, blocked_reason, proof_dir,
             pipeline_payload, rmse, bounds, wall) -> dict:
    return {
        "schema": "M7L2D02ReplayValidationSummary",
        "schema_version": 1,
        "verdict": verdict,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "init_mode": "standalone_native_init_nested",
        "score_domain": SCORE_DOMAIN,
        # Upstream blocked reason (None on GREEN / BOUNDED_FAIL).
        "blocked_reason": blocked_reason,
        "proofs": {
            "tier4_rmse":  str(proof_dir / "tier4_rmse_l2_d02.json"),
            "bounds":      str(proof_dir / "bounds_check_l2_d02.json"),
            "wall_clock":  str(proof_dir / "wall_clock_l2_d02.json"),
        },
        "statuses": {
            "pipeline":  pipeline_payload.get("verdict"),
            "rmse":      rmse.get("status"),
            "bounds":    bounds.get("status"),
            "wall_clock": wall.get("status"),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
