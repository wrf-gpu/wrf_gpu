#!/usr/bin/env python3
"""V0.14 nested-pipeline Noah-MP activation proof (CPU-ONLY, no GPU touched).

Proves, against the REAL Canary L2 case the running 72h gate consumes, every
CPU-provable element of the v0.14 frozen-land fix
(`src/gpuwrf/integration/nested_pipeline.py`):

  1. per-domain namelist resolution: both domains select sf_surface_physics=4
     through the production `_domain_sf_surface_physics` (fail-closed reader);
  2. the Noah-MP bundles the production loader would attach to each domain's
     namelist actually BUILD from this case's wrfinput on a CPU backend:
     `build_noahmp_land_state` (land carry + static) and `build_noahmp_params`
     (energy/rad parameter bundles, concrete nroot) are non-null;
  3. land coverage + a physical initial land-mean TSK from the seeded carry;
  4. the WRF Noah-MP clock (0-based fractional julian, leap-aware yearlen);
  5. structural acceptance: OperationalNamelist carries every field the fix
     replaces; OperationalCarry carries noahmp_land/noahmp_rad; the domain-tree
     output path honours `wants_carry`; the nested writer opts in;
  6. fail-closed: an unsupported land option raises instead of silently running
     the frozen prescribed bulk path.

LIMITATION (documented per the sprint contract): the full `_load_domains`
device build (State + initial carry + `noahmp_initial_rad` RRTMG seed) cannot
run CPU-only because `contracts/state.py::State.zeros` mandates a JAX GPU
backend. Those parts ride the manager's GPU gates (memory preflight + Canary
h1-h4) after merge. This proof never initializes a GPU backend
(JAX_PLATFORMS=cpu + CUDA_VISIBLE_DEVICES='') so the running Canary 72h job is
untouched.

Usage:
  python proofs/v014/noahmp_nested_pipeline_activation.py \
      [--run-dir <DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z]
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# HARD CPU lock BEFORE any jax import: the Canary 72h GPU run must not be touched.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "4")
try:  # repo CPU budget: Claude work pinned to cores 0-3
    os.sched_setaffinity(0, {0, 1, 2, 3})
except (AttributeError, OSError):
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

DEFAULT_RUN_DIR = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z"
)
DOMAINS = ("d01", "d02")


def _run_start_from_namelist(namelist: dict) -> datetime:
    tc = namelist.get("time_control", {})

    def first(key):
        raw = tc.get(key, 0)
        return int(raw[0]) if isinstance(raw, (list, tuple)) else int(raw)

    return datetime(
        first("start_year"), first("start_month"), first("start_day"),
        first("start_hour"), first("start_minute"), first("start_second"),
        tzinfo=timezone.utc,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-json", type=Path, default=HERE / "noahmp_nested_pipeline_activation.json")
    parser.add_argument("--out-md", type=Path, default=HERE / "noahmp_nested_pipeline_activation.md")
    args = parser.parse_args()

    import jax

    from gpuwrf.integration.nested_pipeline import (
        _PerDomainWrfoutWriter,
        _SUPPORTED_NESTED_LAND_OPTIONS,
        _domain_sf_surface_physics,
        _wrf_julian_yearlen,
    )
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
    from gpuwrf.runtime.domain_tree import run_domain_tree_callbacks
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry

    run_dir = args.run_dir
    if not (run_dir / "namelist.input").is_file():
        raise FileNotFoundError(f"case namelist missing: {run_dir}/namelist.input")
    run = Gen2Run(run_dir)
    run_start = _run_start_from_namelist(run.namelist)
    julian, yearlen = _wrf_julian_yearlen(run_start)

    problems: list[str] = []
    per_domain: dict[str, dict] = {}
    for name in DOMAINS:
        sf = _domain_sf_surface_physics(run, name)
        row: dict = {"sf_surface_physics": int(sf), "use_noahmp": sf == 4}
        if sf == 4:
            land, static, init_meta = build_noahmp_land_state(run_dir, name)
            energy_params, rad_params, nroot = build_noahmp_params(static)
            xland = np.asarray(static.xland, dtype=np.float64)
            is_land = xland < 1.5
            tsk0 = np.asarray(land.t_skin, dtype=np.float64)
            land_mean_tsk = float(np.mean(tsk0[is_land])) if is_land.any() else None
            row.update(
                {
                    "noahmp_land_nonnull": land is not None,
                    "noahmp_static_nonnull": static is not None,
                    "noahmp_energy_params_nonnull": energy_params is not None,
                    "noahmp_rad_params_nonnull": rad_params is not None,
                    "noahmp_nroot": int(nroot),
                    "n_land_cells": int(init_meta["n_land_cells"]),
                    "grid_shape_yx": init_meta["grid_shape_yx"],
                    "initial_land_mean_tsk_K": land_mean_tsk,
                    "provenance_wrfinput": init_meta["wrfinput_file"],
                    "prognostic_state_real_from_corpus": init_meta[
                        "prognostic_state_real_from_corpus"
                    ],
                }
            )
            if init_meta["n_land_cells"] <= 0:
                problems.append(f"{name}: no land cells")
            if land_mean_tsk is None or not (240.0 <= land_mean_tsk <= 320.0):
                problems.append(f"{name}: unphysical initial land-mean TSK {land_mean_tsk}")
        else:
            problems.append(f"{name}: case did not select Noah-MP (sf={sf})")
        per_domain[name] = row

    # --- structural acceptance: the fields/paths the fix relies on exist -------
    namelist_fields = set(OperationalNamelist.__dataclass_fields__)
    required_namelist_fields = {
        "use_noahmp", "sf_surface_physics", "noahmp_static", "noahmp_energy_params",
        "noahmp_rad_params", "noahmp_nroot", "noahmp_julian", "noahmp_yearlen",
    }
    missing_namelist = sorted(required_namelist_fields - namelist_fields)
    if missing_namelist:
        problems.append(f"OperationalNamelist missing fields: {missing_namelist}")

    carry_fields = set(OperationalCarry.__dataclass_fields__)
    if not {"noahmp_land", "noahmp_rad"} <= carry_fields:
        problems.append("OperationalCarry lacks noahmp_land/noahmp_rad")
    carry_kwargs = set(inspect.signature(initial_operational_carry).parameters)
    if not {"noahmp_land", "noahmp_rad"} <= carry_kwargs:
        problems.append("initial_operational_carry lacks noahmp seeding kwargs")

    writer_opts_in = bool(getattr(_PerDomainWrfoutWriter, "wants_carry", False))
    if not writer_opts_in:
        problems.append("_PerDomainWrfoutWriter does not opt into the carry payload")
    runner_honours = "wants_carry" in inspect.getsource(run_domain_tree_callbacks)
    if not runner_honours:
        problems.append("run_domain_tree_callbacks does not honour wants_carry")

    # --- fail-closed probe: unsupported land option must raise -----------------
    class _StubRun:
        namelist = {"physics": {"sf_surface_physics": [2, 2]}}

    fail_closed = {"raised": False, "message": None}
    try:
        _domain_sf_surface_physics(_StubRun(), "d01")
    except ValueError as exc:
        fail_closed = {"raised": True, "message": str(exc)}
    if not fail_closed["raised"]:
        problems.append("unsupported land option did not fail closed")

    backend = jax.default_backend()
    if backend != "cpu":
        problems.append(f"proof unexpectedly ran on backend {backend}")

    verdict = "NOAHMP_NESTED_ACTIVATION_CPU_PROVEN" if not problems else "FAIL"
    payload = {
        "proof": "v0.14 nested-pipeline Noah-MP activation (CPU-provable subset)",
        "verdict": verdict,
        "problems": problems,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "run_start_utc": run_start.isoformat(),
        "noahmp_clock": {"julian": julian, "yearlen": yearlen,
                         "convention": "WRF grid%julian = 0-based fractional day-of-year"},
        "supported_nested_land_options": list(_SUPPORTED_NESTED_LAND_OPTIONS),
        "per_domain": per_domain,
        "structural": {
            "operational_namelist_has_all_replaced_fields": not missing_namelist,
            "operational_carry_has_noahmp_slots": {"noahmp_land", "noahmp_rad"} <= carry_fields,
            "initial_operational_carry_accepts_noahmp_seeds": {"noahmp_land", "noahmp_rad"} <= carry_kwargs,
            "writer_wants_carry": writer_opts_in,
            "domain_tree_output_honours_wants_carry": runner_honours,
        },
        "fail_closed_probe_sf2": fail_closed,
        "gpu_used": False,
        "jax_backend": backend,
        "gpu_only_remainder": (
            "State/initial-carry device build + noahmp_initial_rad RRTMG t=0 seed "
            "require a visible GPU (contracts/state.py State.zeros); covered by the "
            "manager GPU gates (memory preflight, Canary h1-h4, 72h rerun)."
        ),
    }
    args.out_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    lines = [
        "# V0.14 Nested-Pipeline Noah-MP Activation Proof (CPU-only)",
        "",
        f"Date: 2026-06-10 · Case: `{run_dir.name}` · Verdict: **{verdict}**",
        "",
        "Fix under proof: `nested_pipeline._load_domains` now reads per-domain",
        "`sf_surface_physics`, wires Noah-MP (namelist bundle + seeded initial",
        "carry) when it is 4, fails closed on unsupported options, and the wrfout",
        "writer reads the EVOLVED land carry (`wants_carry`).",
        "",
        f"Run start: {run_start.isoformat()} → WRF clock julian={julian} yearlen={yearlen}",
        "",
        "| domain | sf_surface_physics | use_noahmp | static/params/land non-null | n_land_cells | init land-mean TSK [K] |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in per_domain.items():
        nonnull = all(
            row.get(k) for k in (
                "noahmp_land_nonnull", "noahmp_static_nonnull",
                "noahmp_energy_params_nonnull", "noahmp_rad_params_nonnull",
            )
        )
        tsk = row.get("initial_land_mean_tsk_K")
        lines.append(
            f"| {name} | {row['sf_surface_physics']} | {row['use_noahmp']} | "
            f"{nonnull} | {row.get('n_land_cells')} | "
            f"{f'{tsk:.2f}' if isinstance(tsk, float) else tsk} |"
        )
    lines += [
        "",
        f"Fail-closed (sf=2 stub): raised={fail_closed['raised']}",
        f"Structural: {json.dumps(payload['structural'])}",
        "",
        f"GPU used: NO (backend={backend}; JAX_PLATFORMS=cpu, CUDA_VISIBLE_DEVICES='').",
        "",
        "GPU-only remainder (manager gates): " + payload["gpu_only_remainder"],
        "",
        "Problems: " + (json.dumps(problems) if problems else "none"),
    ]
    args.out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": verdict, "problems": problems,
                      "json": str(args.out_json), "md": str(args.out_md)}, indent=2))
    return 0 if verdict.endswith("PROVEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
