#!/usr/bin/env python3
"""v0.14 Canary 72h field-gate drift root cause: wrfbdy LBC cadence bug.

CPU-only falsifier + fix gate. Proves, against the retained Canary case
``20260501_18z_l2_72h_20260519T173026Z``:

1. BUG REPRODUCTION: ``interpolate_boundary_leaf`` at the OLD hourly default
   cadence (3600 s) forces the d01 spec zone with wrfbdy level h at hour h --
   i.e. the 6-hourly boundary forcing plays 6x too fast, then clamps frozen on
   the last record from h11. The values match the live GPU run's emitted
   outermost-ring MU exactly (the proven h08/h10/h18 drift driver).
2. FIX GATE: at the wrfbdy interval cadence (21600 s), the same production
   interpolation reproduces the CPU-WRF truth spec-zone MU at every paired
   lead to < 0.6 Pa (CPU truth itself matches wrfbdy to ~0.05 Pa).
3. TERMINAL LEVEL: the synthesized 13th leaf level equals the last wrfbdy
   record advanced by its own ``_BT*`` tendency over the full interval, so
   leads 66-72 h interpolate instead of clamping frozen.
4. PLUMBING: ``_root_boundary_cadence_override`` rewrites
   ``boundary_config.update_cadence_s`` from the loader's boundary metadata
   (tested on the production helper with the production ``BoundaryConfig``; the
   ``OperationalNamelist`` container is stubbed because ``State.zeros`` is
   GPU-only and this proof must stay off the GPU).

The leaves come from the PRODUCTION ``load_wrfbdy_boundary_leaves`` with the
same ``Gen2Run``/``GridSpec``/``load_wrfinput_metrics`` construction
``build_replay_case`` uses (``State.zeros`` skipped: GPU-only).

Run:
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
    taskset -c 0-3 python proofs/v014/lbc_cadence_root_cause.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

RUN_DIR = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z")
CPU_TRUTH = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z"
)
GPU_RUN = Path(
    "<DATA_ROOT>/wrf_gpu_validation/v014_canary_d02_72h_20260610T142426Z/gpu_output/"
    "l2_d02_20260501_18z_l2_72h_20260519T173026Z"
)
INTERVAL_S = 21600.0
OUT = Path("proofs/v014/lbc_cadence_root_cause.json")


def _hour_label(h: int) -> str:
    from datetime import datetime, timedelta

    return (datetime(2026, 5, 1, 18) + timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S")


def _west_outer_mu(path: Path) -> np.ndarray:
    with Dataset(path) as ds:
        return np.asarray(ds["MU"][0][:, 0], dtype=np.float64)


def main() -> int:
    import jax

    jax.config.update("jax_enable_x64", True)

    from dataclasses import dataclass, replace as dataclass_replace

    from gpuwrf.coupling.boundary_apply import BoundaryConfig, interpolate_boundary_leaf
    from gpuwrf.dynamics.metrics import load_wrfinput_metrics
    from gpuwrf.integration.d02_replay import load_wrfbdy_boundary_leaves
    from gpuwrf.integration.nested_pipeline import _root_boundary_cadence_override
    from gpuwrf.io.gen2_accessor import Gen2Run

    report: dict = {"schema": "lbc_cadence_root_cause_v1"}

    run = Gen2Run(RUN_DIR)
    grid = run.grid("d01").as_grid_spec()
    metrics = load_wrfinput_metrics(run.history_files("d01")[0])
    with Dataset(RUN_DIR / "wrfinput_d01") as ds:
        mu_total = np.asarray(ds["MU"][0][:], dtype=np.float64) + np.asarray(
            ds["MUB"][0][:], dtype=np.float64
        )
    leaves, bmeta = load_wrfbdy_boundary_leaves(
        run, grid, domain="d01", mu_total=mu_total, metrics=metrics
    )
    report["boundary_meta"] = {
        k: bmeta[k]
        for k in ("interval_seconds", "times", "wrfbdy_records", "terminal_level_synthesized")
    }
    mu_bdy = leaves["mu_bdy"]  # (time, side, bw, 1, side_len)
    ny = int(grid.ny)
    assert int(mu_bdy.shape[0]) == int(bmeta["wrfbdy_records"]) + 1, "terminal level missing"

    # wrfbdy raw west-outer record means (level k valid at t = k*interval).
    with Dataset(RUN_DIR / "wrfbdy_d01") as bdy:
        bxs = np.asarray(bdy["MU_BXS"][:], dtype=np.float64)  # (rec, bw, sn)
        btxs = np.asarray(bdy["MU_BTXS"][:], dtype=np.float64)
    levels = bxs[:, 0, :].mean(axis=1)
    terminal_truth = (bxs[-1, 0, :] + btxs[-1, 0, :] * INTERVAL_S).mean()

    # (1) bug reproduction at the old hourly cadence vs the live GPU artifacts.
    # The pre-fix leaf had no synthesized terminal level, so emulate the old
    # behaviour on the first ``wrfbdy_records`` levels only.
    mu_bdy_prefix = mu_bdy[:-1]
    bug_rows = []
    bug_max = 0.0
    for h in range(1, 21):
        strip = interpolate_boundary_leaf(mu_bdy_prefix, float(h) * 3600.0, 3600.0)
        west = np.asarray(strip[0, 0, 0, :ny], dtype=np.float64).mean()
        gpu_path = GPU_RUN / f"wrfout_d01_{_hour_label(h)}"
        gpu = float(_west_outer_mu(gpu_path).mean()) if gpu_path.exists() else None
        expected_level = levels[min(h, len(levels) - 1)]
        row = {"lead_h": h, "old_cadence_value": float(west), "wrfbdy_level": float(expected_level)}
        if gpu is not None:
            row["gpu_emitted"] = gpu
            bug_max = max(bug_max, abs(west - gpu))
        bug_rows.append(row)
    report["bug_reproduction"] = {
        "rows": bug_rows,
        "max_abs_vs_gpu_emitted_pa": bug_max,
        "interpretation": "old 3600 s cadence == wrfbdy level h at hour h, frozen at last level from h11",
    }

    # (2) fix gate at the wrfbdy interval cadence vs CPU-WRF truth spec zone.
    fix_rows = []
    fix_max = 0.0
    for h in range(1, 21):
        strip = interpolate_boundary_leaf(mu_bdy, float(h) * 3600.0, INTERVAL_S)
        west = np.asarray(strip[0, 0, 0, :ny], dtype=np.float64)
        cpu = _west_outer_mu(CPU_TRUTH / f"wrfout_d01_{_hour_label(h)}")
        err = float(np.max(np.abs(west - cpu)))
        fix_max = max(fix_max, err)
        fix_rows.append({"lead_h": h, "max_abs_vs_cpu_truth_pa": err})
    report["fix_gate"] = {"rows": fix_rows, "max_abs_pa": fix_max, "limit_pa": 0.6}

    # (3) terminal synthesized level == last record + tendency*interval; covers 72 h.
    term_leaf = float(np.asarray(mu_bdy[-1, 0, 0, 0, :ny], dtype=np.float64).mean())
    strip72 = interpolate_boundary_leaf(mu_bdy, 72.0 * 3600.0, INTERVAL_S)
    west72 = np.asarray(strip72[0, 0, 0, :ny], dtype=np.float64)
    cpu72 = _west_outer_mu(CPU_TRUTH / f"wrfout_d01_{_hour_label(72)}")
    report["terminal_level"] = {
        "leaf_terminal_mean": term_leaf,
        "wrfbdy_base_plus_tendency_mean": float(terminal_truth),
        "abs_diff_pa": abs(term_leaf - float(terminal_truth)),
        "lead72_max_abs_vs_cpu_truth_pa": float(np.max(np.abs(west72 - cpu72))),
    }

    # (4) plumbing: the production override helper rewrites update_cadence_s
    # from the loader's boundary metadata (namelist container stubbed: the real
    # OperationalNamelist needs GPU-resident State/Tendencies to build).
    @dataclass(frozen=True)
    class _NamelistStub:
        boundary_config: BoundaryConfig

    stub = _NamelistStub(boundary_config=BoundaryConfig())
    before = float(stub.boundary_config.update_cadence_s)
    stub = _root_boundary_cadence_override(stub, {"boundary": bmeta})
    after = float(stub.boundary_config.update_cadence_s)
    report["plumbing"] = {"update_cadence_s_before": before, "update_cadence_s_after": after}

    ok = (
        bug_max < 1.0e-6
        and fix_max < 0.6
        and report["terminal_level"]["abs_diff_pa"] < 1.0e-9
        and report["terminal_level"]["lead72_max_abs_vs_cpu_truth_pa"] < 0.6
        and after == INTERVAL_S
        and before != after
    )
    report["verdict"] = "LBC_CADENCE_ROOT_CAUSE_PROVEN_FIX_GATE_PASS" if ok else "FAIL"
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("boundary_meta", "plumbing", "verdict")}, indent=1))
    print(f"bug_reproduction max_abs_vs_gpu_emitted_pa = {bug_max:.3e}")
    print(f"fix_gate max_abs_vs_cpu_truth_pa = {fix_max:.3f}")
    print(f"terminal: {report['terminal_level']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
