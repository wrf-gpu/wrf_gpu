#!/usr/bin/env python3
"""CPU coupled smoke: real Swiss 42x42 d01 with RUC LSM + MYNN + Thompson + RRTMG.

HRRR/RAP-like physics on the operational scan (sf_surface_physics=3 through
``coupling.ruc_surface_hook``, sf_sfclay=5 MYNN-SL, bl=5 MYNN, mp=8 Thompson,
ra=4 RRTMG), a few root steps on CPU; checks every carry leaf finite and (with
GPUWRF_CENSUS=1) zero guard events.

Smoke-only simplifications (disclosed, not a WRF real.exe RUC initialisation):
* the Swiss wrfinput is a Noah-MP (4-layer) case; TSLB/SMOIS are linearly
  interpolated in depth onto the RUC soil levels (TSK at the z=0 surface level, TMN at 3 m);
* CPU harness: ``jax.lax.linalg.tridiagonal_solve`` is replaced by a pure-JAX Thomas
  scan (jaxlib LAPACK dgtsv deadlocks the affinity-sized XLA:CPU pool, E105; not the
  product solver).

Usage::

    JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=0 GPUWRF_CENSUS=1 \\
      PYTHONPATH=src taskset -c 9 python proofs/v034/ruclsm_swiss_smoke.py --steps 5 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

t_start = time.perf_counter()
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[2] / "examples" / "switzerland_d01")
ap.add_argument("--levels", type=int, choices=(6, 9), default=9)
args = ap.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"

import gpuwrf  # noqa: F401,E402  (fast-default resolution first)
import jax  # noqa: E402

jax.config.update("jax_cpu_enable_async_dispatch", False)
assert jax.devices()[0].platform == "cpu"
from gpuwrf.contracts import state as state_contract  # noqa: E402

state_contract._gpu_device = lambda: jax.devices("cpu")[0]

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402


def _thomas_tridiagonal_solve(dl, d, du, b):
    """Pure-JAX Thomas solve (CPU harness only; see module doc)."""

    dl_t, d_t, du_t = (jnp.moveaxis(x, -1, 0)[..., None] for x in (dl, d, du))
    b_t = jnp.moveaxis(b, -2, 0)

    def forward(carry, row):
        cp_prev, dp_prev = carry
        a_i, b_i, c_i, r_i = row
        denom = b_i - a_i * cp_prev
        cp = c_i / denom
        dp = (r_i - a_i * dp_prev) / denom
        return (cp, dp), (cp, dp)

    init = (jnp.zeros_like(d_t[0]), jnp.zeros_like(b_t[0]))
    _, (cp, dp) = lax.scan(forward, init, (dl_t, d_t, du_t, b_t))

    def backward(x_next, row):
        cp_i, dp_i = row
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, xs = lax.scan(backward, jnp.zeros_like(b_t[0]), (cp, dp), reverse=True)
    return jnp.moveaxis(xs, 0, -2)


jax.lax.linalg.tridiagonal_solve = _thomas_tridiagonal_solve

import netCDF4  # noqa: E402

from gpuwrf.coupling.ruc_surface_hook import build_ruc_bundles  # noqa: E402
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains  # noqa: E402
from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    _commit_to_operational_device,
    _initial_carry_for_run,
    _resolve_operational_suite,
)

ZS9 = (0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00)
ZS6 = (0.0, 0.05, 0.20, 0.40, 1.60, 3.00)  # WRF share/module_soil_pre.F RUC 6-level grid

args.out.mkdir(parents=True, exist_ok=True)
config = NestedPipelineConfig(args.input, args.out / "stream", args.out / "proof", hours=1, max_dom=1)
t0 = time.perf_counter()
hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, ("d01",))
load_s = time.perf_counter() - t0
bundle = bundles["d01"]
nml = bundle.namelist
state = carries["d01"].state

ds = netCDF4.Dataset(args.input / "wrfinput_d01")
f = {name: np.asarray(ds[name][0]) for name in (
    "IVGTYP", "ISLTYP", "XLAND", "SEAICE", "TMN", "SHDMIN", "SHDMAX", "ALBBCK", "VEGFRA",
    "TSK", "TSLB", "SMOIS", "SNOW", "SNOWH", "CANWAT", "LAI", "SNOALB")}
zs_noah = np.asarray(ds["ZS"][0], np.float64)
mminlu, iswater, isice = str(ds.MMINLU), int(ds.ISWATER), int(ds.ISICE)
ds.close()
zs_ruc = ZS9 if args.levels == 9 else ZS6


def _interp_profile(noah, deep, top=None):
    # smoke-only: linear in depth; RUC level 1 is the surface (z=0): `top` there (TSK for
    # temperature), else the top Noah layer; `deep` at 3 m (TMN for temperature)
    zz = np.concatenate([[0.0] if top is not None else [], zs_noah, [3.0]])
    out = np.empty((len(zs_ruc),) + noah.shape[1:])
    for j in range(noah.shape[1]):
        for i in range(noah.shape[2]):
            yy = np.concatenate([[top[j, i]] if top is not None else [], noah[:, j, i], [deep[j, i]]])
            out[:, j, i] = np.interp(zs_ruc, zz, yy)
    return out


f["TSLB"] = _interp_profile(f["TSLB"], f["TMN"], top=f["TSK"])
f["SMOIS"] = _interp_profile(f["SMOIS"], f["SMOIS"][-1])
f["SNOWC"] = (f["SNOW"] > 0.0).astype(np.float32)
p1 = np.asarray(state.p_total[0], np.float64)
static, land = build_ruc_bundles(f, mminlu=mminlu, iswater=iswater, isice=isice, dt=float(nml.dt_s),
                                 zs=zs_ruc, p1=p1, frpcpn=True)
nml_ruc = replace(nml, use_noahmp=False, sf_surface_physics=3, noahmp_static=None,
                  noahmp_energy_params=None, noahmp_rad_params=None, ruc_static=static, ruc_land=land)
_resolve_operational_suite(nml_ruc)
carry = _commit_to_operational_device(_initial_carry_for_run(bundle.state, nml_ruc))
bundles["d01"] = replace(bundle, namelist=nml_ruc)
tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
carries = {"d01": carry}
own = {"d01": 0}
times = []
land0 = carry.ruc_land
for _ in range(args.steps):
    t0 = time.perf_counter()
    result = run_operational_domain_tree(tree, root_steps=1, carries=carries, initial_own_steps=own,
                                         block_between=False, root_sync_cadence=0)
    jax.block_until_ready(result.carries)
    carries, own = result.carries, dict(result.own_steps)
    times.append(time.perf_counter() - t0)
    print("step", own, round(times[-1], 2), flush=True)

c = carries["d01"]
nonfinite = []
for path, leaf in jax.tree_util.tree_leaves_with_path(c):
    a = np.asarray(leaf)
    if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
        nonfinite.append(jax.tree_util.keystr(path))
land_mask = np.asarray(static.xland) < 1.5
rl = c.ruc_land


def _rng(x, mask=land_mask):
    a = np.asarray(x)
    a = a[mask] if a.ndim == 2 else a[mask, :]
    return [float(np.min(a)), float(np.max(a))]


census = None
if c.census is not None:
    census = {"guards": np.asarray(c.census.guards).tolist(), "guards_total": int(np.sum(np.asarray(c.census.guards)))}
record = {
    "case": str(args.input), "levels": list(zs_ruc), "steps": args.steps, "dt_s": float(nml.dt_s),
    "physics": {"sf_surface_physics": 3, "sf_sfclay_physics": int(nml_ruc.sf_sfclay_physics),
                "bl_pbl_physics": int(nml_ruc.bl_pbl_physics), "mp_physics": int(nml_ruc.mp_physics),
                "ra_lw_physics": int(nml_ruc.ra_lw_physics), "ra_sw_physics": int(nml_ruc.ra_sw_physics)},
    "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS"),
    "load_s": load_s, "step_s": times, "process_s": time.perf_counter() - t_start,
    "nonfinite_leaves": nonfinite, "census": census,
    "land_columns": int(land_mask.sum()),
    "ruc_land_ranges": {k: _rng(getattr(rl, k)) for k in (
        "soilt", "hfx", "qfx", "lh", "grdflx", "qsfc", "mavail", "snow", "snowh", "tso", "soilmois")},
    "tsk_change_land": _rng(np.asarray(rl.soilt) - np.asarray(land0.soilt)),
    "state_t_skin_range": [float(np.min(np.asarray(c.state.t_skin))), float(np.max(np.asarray(c.state.t_skin)))],
    "tridiagonal_solve": "pure-JAX Thomas scan (CPU harness, E105)",
    "simplifications": "Noah 4-layer TSLB/SMOIS interpolated to RUC levels (smoke-only init)",
}
(args.out / "smoke.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps({k: record[k] for k in ("steps", "step_s", "nonfinite_leaves", "census", "tsk_change_land")}))
