#!/usr/bin/env python
"""DIAGNOSIS: does the d02 +2.6 kPa pressure inflation come from the per-hour
`run_forecast_operational` restart path (pipeline/d03_replay) vs the continuous
`_advance_chunk` validation-harness path?

Advances the SAME d02 case3 IC 2 forecast hours TWO ways and compares state.p[0]
(the surface pressure that feeds T2's Exner) against each other and the corpus.

CPU-pinned orchestration; ONE short GPU job. No production src/ edits.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except OSError:
        pass

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
from netCDF4 import Dataset

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import (
    run_forecast_operational,
    _advance_chunk,
    _enforce_operational_precision,
)
from gpuwrf.runtime.operational_state import initial_operational_carry

RUN_ID = "20260521_18z_l3_24h_20260522T133443Z"
RUN_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3")
CORPUS = RUN_ROOT / RUN_ID
HOURS = 2
DT = 10.0
CADENCE = 180


def corpus_psfc(hour: int) -> np.ndarray:
    from datetime import datetime, timedelta
    t0 = datetime(2026, 5, 21, 18, 0, 0)
    valid = t0 + timedelta(hours=hour)
    p = CORPUS / f"wrfout_d02_{valid:%Y-%m-%d_%H:%M:%S}"
    with Dataset(p) as d:
        P = np.asarray(d.variables["P"][0], float)
        PB = np.asarray(d.variables["PB"][0], float)
    return (P + PB)[0]


def main() -> int:
    cfg = DailyPipelineConfig(run_id=RUN_ID, run_root=RUN_ROOT, domain="d02",
                              dt_s=DT, acoustic_substeps=10, radiation_cadence_steps=CADENCE)
    case, _ = _build_real_case(cfg)
    nl = case.namelist
    state0 = _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))

    # run_forecast_operational has donate_argnums=(0,) and DELETES its input buffer.
    # Materialise two INDEPENDENT device copies of the IC so each path starts clean
    # (otherwise Path B's initial_operational_carry(state0) hits a deleted buffer).
    state_A = jax.tree_util.tree_map(lambda x: jnp.array(np.asarray(jax.device_get(x))), state0)
    state_B = jax.tree_util.tree_map(lambda x: jnp.array(np.asarray(jax.device_get(x))), state0)

    # ---- Path A: per-hour run_forecast_operational (PIPELINE path) ----
    sA = state_A
    for _ in range(HOURS):
        sA = run_forecast_operational(sA, nl, 1.0)
    jax.block_until_ready(sA.theta)
    psfcA = np.asarray(jax.device_get(sA.p[0]), float)

    # ---- Path B: continuous _advance_chunk (HARNESS path) ----
    carry = initial_operational_carry(state_B)
    steps = int(round(HOURS * 3600.0 / DT))
    seg = 180
    start = 1
    while start <= steps:
        n = min(seg, steps - start + 1)
        carry = _advance_chunk(carry, nl, jnp.asarray(start, dtype=jnp.int32),
                               n_steps=int(n), cadence=CADENCE)
        jax.block_until_ready(carry.state.theta)
        start += n
    psfcB = np.asarray(jax.device_get(carry.state.p[0]), float)

    cps = corpus_psfc(HOURS)
    print("=== d02 case3 +2h surface-pressure (state.p[0]) by path ===")
    print(f"corpus psfc mean                         = {np.nanmean(cps):.1f} Pa")
    print(f"Path A per-hour run_forecast_operational = {np.nanmean(psfcA):.1f} Pa  bias vs corpus = {np.nanmean(psfcA-cps):+.1f} Pa")
    print(f"Path B continuous _advance_chunk         = {np.nanmean(psfcB):.1f} Pa  bias vs corpus = {np.nanmean(psfcB-cps):+.1f} Pa")
    print(f"A - B (path divergence)                  = {np.nanmean(psfcA-psfcB):+.1f} Pa  max|A-B| = {np.nanmax(np.abs(psfcA-psfcB)):.1f} Pa")
    # also theta[0]
    thA = np.asarray(jax.device_get(sA.theta[0]), float)
    thB = np.asarray(jax.device_get(carry.state.theta[0]), float)
    print(f"theta[0] A - B                           = {np.nanmean(thA-thB):+.3f} K  max|A-B| = {np.nanmax(np.abs(thA-thB)):.3f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
