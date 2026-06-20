#!/usr/bin/env python
"""DESIGN CHECK (CPU): verify the hydrostatic ph' target + residual structure.

Builds the d03 1km Tenerife case at t=0, computes:
  (a) the VERIFIED hydrostatic perturbation geopotential ``_hydrostatic_ph_perturbation``
      from the forced mu'/theta'/qv', and round-trips it back through the dycore al/EOS
      to confirm it reproduces the corpus hydrostatic pressure (machine-exact inverse);
  (b) the residual ``ph'_state - ph'_hydro`` by level in the boundary ring vs the deep
      interior, to confirm the direction/magnitude of the forcing the in-loop relax
      tendency must apply.

No GPU model run; this is a static analysis of the t=0 case state. Pins CPU 0-3.

Usage: d03_phfix_target_check.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except OSError:
        pass

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gpuwrf.coupling.boundary_apply import _hydrostatic_ph_perturbation
from gpuwrf.dynamics.acoustic_wrf import diagnose_pressure_al_alt
from gpuwrf.contracts.state import BaseState

import d03_replay
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig


def main() -> int:
    run_id = "20260521_18z_l3_24h_20260522T133443Z"
    cfg = DailyPipelineConfig(
        run_id=run_id,
        domain="d03",
        run_root=Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3"),
        dt_s=3.0,
        acoustic_substeps=10,
    )
    case, _ = d03_replay.build_l3_d03_daily_case(cfg)
    state = case.state
    metrics = case.namelist.metrics

    mub = np.asarray(state.mu_total - state.mu_perturbation)
    mu_p = np.asarray(state.mu_perturbation)
    pb = np.asarray(state.p_total - state.p_perturbation)
    theta_full = np.asarray(state.theta)
    qv = np.asarray(state.qv)
    ph_state = np.asarray(state.ph_perturbation)  # (nz+1, ny, nx)
    print("grid (nz+1,ny,nx) =", ph_state.shape)

    # (a) hydrostatic target ph'
    ph_hydro = np.asarray(
        _hydrostatic_ph_perturbation(
            jnp.asarray(theta_full),
            jnp.asarray(qv),
            jnp.asarray(mu_p),
            jnp.asarray(mub),
            jnp.asarray(pb),
            metrics,
        )
    )
    print("ph_hydro[0] max|.| =", float(np.max(np.abs(ph_hydro[0]))), "(should be ~0, anchored)")

    # round-trip: feed ph_hydro back through diagnose_pressure_al_alt and compare p to corpus
    base = BaseState(
        pb=jnp.asarray(pb),
        phb=jnp.asarray(np.asarray(state.ph_total - state.ph_perturbation)),
        mub=jnp.asarray(mub),
        t0=jnp.asarray(300.0),
        theta_base=jnp.full_like(jnp.asarray(theta_full), 300.0),
    )
    state_hydro = state.replace(
        ph_perturbation=jnp.asarray(ph_hydro),
        ph_total=jnp.asarray(np.asarray(state.ph_total - state.ph_perturbation) + ph_hydro),
    )
    p_pert_rt, al_rt, alt_rt = diagnose_pressure_al_alt(state_hydro, base, metrics)
    p_state_pert = np.asarray(state.p_perturbation)
    p_rt = np.asarray(p_pert_rt)
    # surface pressure check: pb+p' at level 0
    psfc_state = (pb + p_state_pert)[0]
    psfc_hydro = (pb + p_rt)[0]
    print("psfc(state) mean =", float(np.mean(psfc_state)))
    print("psfc(hydro) mean =", float(np.mean(psfc_hydro)))
    print("psfc(state-hydro) mean =", float(np.mean(psfc_state - psfc_hydro)),
          "max|.| =", float(np.max(np.abs(psfc_state - psfc_hydro))))

    # (b) residual ph_state - ph_hydro by level, ring vs interior
    resid = ph_state - ph_hydro  # (nz+1, ny, nx)
    ring_w = 5  # spec_zone+relax_zone
    ny, nx = resid.shape[1], resid.shape[2]
    ring_mask = np.zeros((ny, nx), dtype=bool)
    ring_mask[:ring_w, :] = True
    ring_mask[-ring_w:, :] = True
    ring_mask[:, :ring_w] = True
    ring_mask[:, -ring_w:] = True
    interior_mask = ~ring_mask
    print("\n[A] target = _hydrostatic_ph_perturbation")
    print("level  ring_mean   interior_mean  (ph_state - ph_hydro), m^2/s^2")
    for k in [0, 1, 5, 10, 20, 30, resid.shape[0] - 1]:
        rm = float(np.mean(resid[k][ring_mask]))
        im = float(np.mean(resid[k][interior_mask]))
        print(f"{k:5d}  {rm:+11.2f}  {im:+13.2f}")

    # (c) residual ph_state - ph_PARENT_LEAF (the WRF-faithful relaxation target).
    # Reconstruct the parent perturbation-geopotential leaf interpolated to the
    # child grid: ph_bdy is parent PH (perturbation), phb_bdy parent PHB.
    # Build the full-field parent ph' by taking the OUTER row of ph_bdy at the
    # W/E/S/N boundary; here we just compare the spec-adjacent (b_dist=0..4) ring
    # cells of the state ph' vs the leaf target, since only the ring is forced.
    from gpuwrf.coupling.boundary_apply import (
        interpolate_boundary_leaf,
        _strip,
        SIDES,
    )

    ph_bdy = state.ph_bdy  # (time, side, bdy_width, z, side_len)
    leaf0 = interpolate_boundary_leaf(ph_bdy, 0.0, 3600.0)  # (side, bdy_width, z, side_len) at lead 0
    nzp1 = ph_state.shape[0]
    print("\n[B] target = parent ph_bdy leaf (WRF-faithful relaxation target)")
    print("    residual ph_state - ph_parent_leaf at the spec/relax ring (b_dist=0,1,3)")
    print("    side  b_dist  level  mean(ph_state-leaf)   max|.|")
    for side in ("W", "S"):
        for b_dist in (0, 1, 3):
            for k in (1, 10, 20, 30, nzp1 - 1):
                leaf_strip = np.asarray(_strip(leaf0, side, b_dist, nzp1, max(ny, nx)))[k]  # (side_len,)
                if side == "W":
                    st = np.asarray(ph_state[k, :, b_dist])
                elif side == "S":
                    st = np.asarray(ph_state[k, b_dist, :])
                n = min(len(st), len(leaf_strip))
                d = st[:n] - leaf_strip[:n]
                print(f"    {side:>4}  {b_dist:6d}  {k:5d}  {float(np.mean(d)):+18.2f}   {float(np.max(np.abs(d))):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
