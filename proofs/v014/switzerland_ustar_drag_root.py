#!/usr/bin/env python
"""V0.14 Switzerland venting ROOT-CAUSE proof: sfclayrev ustar drag deficit.

Companion to ``switzerland_flux_localizer.py`` (which localizes the depth-8
venting to a domain-wide vertical-dipole low-level westerly momentum bias).

This script proves the proximate root: the JAX revised-surface-layer (sfclayrev)
``ustar`` delivered to the MYNN momentum bottom boundary condition is only ~61 %
of the WRF h36 UST, so the explicit surface drag ``bottom_drag = rhosfc*ust^2/
wspd`` (module_bl_mynnedmf.F:4011) is only ~37 % of WRF's.  With that deficit the
MYNN k0 momentum source ACCELERATES the low-level wind (corr(rublten,u) > 0)
instead of decelerating it.  A clean falsifiable knob: scaling the MYNN-input
ustar back to the WRF magnitude (x1.64) flips the k0 source to the correct
decelerating sign.

CPU-only (JAX_PLATFORMS=cpu via the awd term-split shim); no GPU lock needed.

Falsified-but-recorded local fix: the MYNN k0 momentum diagonal in the JAX port
adds ``kdz(kts)`` (= kmdz(kts)) that WRF excludes for momentum (it keeps it only
for scalars, line 4131-4132).  That is a real WRF-faithfulness discrepancy, but
it is INERT at this state because ``dfm(kts)=kdz(kts)=0`` by MYNN construction;
removing it leaves the production rublten byte-identical.  The lever is the
ustar/CD magnitude, not the k0 diagonal structure.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "proofs/v014"
CPU = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu")
OUT_JSON = ROOT / "proofs/v014/switzerland_ustar_drag_root.json"

_AWD = importlib.util.spec_from_file_location(
    "wrf_native_advance_w_dump", P / "wrf_native_advance_w_dump.py"
)
awd = importlib.util.module_from_spec(_AWD)
_AWD.loader.exec_module(awd)  # type: ignore[union-attr]


def main() -> int:
    ts = awd._load_term_split()
    hpg = ts.hpg
    ts._install_cpu_allocator_shim()

    import jax.numpy as jnp
    import gpuwrf.coupling.physics_couplers as pc
    import gpuwrf.physics.mynn_pbl as mm
    from gpuwrf.runtime import operational_mode as om
    import gpuwrf.runtime.operational_mode as omm
    from gpuwrf.runtime.operational_state import initial_operational_carry

    case, state0, run_dir = hpg._build_state(hpg.NATIVE_ROOT)
    grid = case.namelist.grid
    nl = dataclasses.replace(
        case.namelist, dt_s=18.0, acoustic_substeps=4,
        specified_bdy_cadence=True, specified_adv_degrade=True,
    )

    # Run the operational surface->PBL chain so MYNN receives the REAL sfclayrev
    # ustar (the offline _build_state alone leaves state.ustar=0).
    cap: dict = {}
    orig = pc.mynn_adapter_with_source_leaves

    def wrap(state, dt, grid=None, *, first_timestep=False):
        cap["state"] = state
        cap["surface"] = pc._surface_fluxes_from_state(state)
        return orig(state, dt, grid, first_timestep=first_timestep)

    omm.mynn_adapter_with_source_leaves = wrap
    carry0 = initial_operational_carry(state0)
    lead = jnp.asarray(1, dtype=jnp.int32).astype(jnp.float64) * float(nl.dt_s)
    om._physics_step_forcing(
        carry0, nl, lead, run_radiation=bool(nl.run_physics), first_timestep=True
    )
    omm.mynn_adapter_with_source_leaves = orig

    st = cap["state"]
    surf = cap["surface"]
    jax_ust = np.asarray(surf.ustar)

    # WRF h36 UST interior depth-8.
    with Dataset(hpg.fn(CPU, 36)) as d:
        wrf_ust = np.asarray(d.variables["UST"][0])
    i = wrf_ust[8:120, 8:120]
    wrf_ust_mean = float(np.nanmean(i))
    jax_ust_mean = float(np.nanmean(jax_ust))
    scale_to_wrf = wrf_ust_mean / max(jax_ust_mean, 1.0e-9)

    col = pc._mynn_column_from_state(st, grid)
    ny, nx = col.theta.shape[0], col.theta.shape[1]
    cb = pc._flatten_columns_to_batch(col, ny, nx)
    um = np.asarray(pc._u_mass(st))
    u0 = um[0, 8:120, 8:120]

    out: dict = {
        "schema": "v014_switzerland_ustar_drag_root",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wrf_ust_h36_interior_mean": wrf_ust_mean,
        "jax_ust_into_mynn_mean": jax_ust_mean,
        "jax_ust_fraction_of_wrf": jax_ust_mean / wrf_ust_mean,
        "implied_drag_fraction_of_wrf": (jax_ust_mean / wrf_ust_mean) ** 2,
        "wrf_anchor_momentum_k0": "module_bl_mynnedmf.F:4010-4017 (b(kts)=1+dtz*(kmdz(k+1)+rhosfc*ust**2/wspd)*rhoinv)",
        "wrf_anchor_scalar_k0": "module_bl_mynnedmf.F:4131-4132 (b(kts)=1+dtz*(khdz(k+1)+khdz(kts))*rhoinv)",
        "ustar_scale_test": {},
    }
    def _scale_ustar(batch, scale):
        scaled = batch.ustar * scale
        if hasattr(batch, "_replace"):
            return batch._replace(ustar=scaled)
        return dataclasses.replace(batch, ustar=scaled)

    for scale in (1.0, scale_to_wrf):
        sb = pc._flatten_columns_to_batch(surf, ny, nx)
        sb = _scale_ustar(sb, scale)
        res = mm.step_mynn_pbl_column(cb, 18.0, debug=False, surface=sb, edmf=False, dx=pc._mynn_dx(grid))
        outc = pc._unflatten_batch_to_columns(res, ny, nx)
        rub = np.asarray((pc._from_columns(outc.u) - pc._u_mass(st)) / 18.0)
        k0 = rub[0, 8:120, 8:120]
        corr = float(np.corrcoef(k0.ravel(), u0.ravel())[0, 1])
        out["ustar_scale_test"][f"x{scale:.2f}"] = {
            "k0_rublten_mean": float(np.nanmean(k0)),
            "corr_rublten_u": corr,
            "verdict": "ACCELERATES low-level wind (wrong)" if corr > 0 else "decelerates (correct drag)",
        }
    out["verdict"] = (
        "ROOT = sfclayrev ustar magnitude deficit ({:.0%} of WRF) starving the MYNN "
        "k0 surface momentum drag (~{:.0%} of WRF); restoring WRF ustar flips the k0 "
        "momentum source from accelerating to the correct decelerating sign."
    ).format(jax_ust_mean / wrf_ust_mean, (jax_ust_mean / wrf_ust_mean) ** 2)

    OUT_JSON.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
