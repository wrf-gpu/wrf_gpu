#!/usr/bin/env python
"""V0.14 Switzerland sfclayrev/MYNN-surface ustar TERM-BY-TERM oracle at h36.

drag_root proved JAX ustar into MYNN = 61% of WRF UST (0.380 vs 0.624). This
oracle goes a layer deeper: it runs the REAL operational surface->PBL chain at
the h36 reinit (so the surface layer receives the genuine warm-step inputs:
the carried prior-step UST, the phy_prep dry t_air/psfc/rho, the real znt) and
captures EVERY intermediate term of the friction-velocity closure
(module_sf_mynn.F:945-962):

    PSIX  = GZ1OZ0 - PSIM
    UST   = 0.5*UST_in + 0.5*KARMAN*WSPD/PSIX

per land/water, to localize WHICH term (wspd, znt->gz1oz0, psim, or the carried
ust_in) makes the JAX ustar 61% of WRF. It also reports the column z (za),
roughness (znt), bulk-Ri (br), z/L (zol) and the wspd convective augmentation,
all interior depth-8 means, against the WRF wrfout UST truth.

CPU-only. No source change. The instrumentation captures the live intermediate
arrays by importing the module and recomputing the closure terms from the
returned diagnostics (znt/psim are returned) plus the column view fed in.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

P = ROOT / "proofs/v014"
_AWD = importlib.util.spec_from_file_location(
    "wrf_native_advance_w_dump", P / "wrf_native_advance_w_dump.py"
)
awd = importlib.util.module_from_spec(_AWD)
_AWD.loader.exec_module(awd)  # type: ignore[union-attr]

_HPG = importlib.util.spec_from_file_location(
    "hpg_native_face_proof", P / "switzerland_hpg_native_face_fix.py"
)
hpg = importlib.util.module_from_spec(_HPG)
_HPG.loader.exec_module(hpg)  # type: ignore[union-attr]

CPU = hpg.CPU
OUT_JSON = ROOT / "proofs/v014/switzerland_sfclay_ustar_oracle.json"

# Constant from surface_constants (mirror to avoid re-import ordering issues).
KARMAN = 0.4


def _imean(a, depth=8):
    a = np.asarray(a)
    return float(np.nanmean(a[depth:-depth, depth:-depth]))


def main() -> int:
    import jax.numpy as jnp
    import numpy as _np
    from netCDF4 import Dataset

    ts = awd._load_term_split()
    hpg_mod = ts.hpg
    ts._install_cpu_allocator_shim()

    import gpuwrf.coupling.physics_couplers as pc
    import gpuwrf.physics.surface_layer as sl
    from gpuwrf.runtime import operational_mode as om
    import gpuwrf.runtime.operational_state as ostate
    from gpuwrf.runtime.operational_state import initial_operational_carry

    case, state0, run_dir = hpg_mod._build_state(hpg_mod.NATIVE_ROOT)
    grid = case.namelist.grid
    nl = dataclasses.replace(
        case.namelist, dt_s=18.0, acoustic_substeps=4,
        specified_bdy_cadence=True, specified_adv_degrade=True,
    )

    # The LIVE Switzerland surface path is use_noahmp=False + sf_sfclay_physics=5
    # -> physics_couplers.surface_adapter -> physics_couplers.surface_layer. Wrap
    # surface_layer to capture the EXACT column view fed in and run the full
    # diagnostics on it (same view), so all intermediate terms are captured.
    cap: dict = {}
    orig = pc.surface_layer

    def wrap(view, *, first_timestep=False):
        cap["view"] = view
        cap["diag"] = sl.surface_layer_with_diagnostics(view, first_timestep=first_timestep)
        return orig(view, first_timestep=first_timestep)

    pc.surface_layer = wrap
    carry0 = initial_operational_carry(state0)
    lead = jnp.asarray(1, dtype=jnp.int32).astype(jnp.float64) * float(nl.dt_s)
    try:
        om._physics_step_forcing(
            carry0, nl, lead, run_radiation=bool(nl.run_physics), first_timestep=False
        )
    finally:
        pc.surface_layer = orig

    view = cap["view"]
    diag = cap["diag"]
    sf = diag.fluxes

    # ---- recompute the closure terms from the SAME view (independent path) ----
    u0 = sl._surface(jnp.asarray(view.u, jnp.float64))
    v0 = sl._surface(jnp.asarray(view.v, jnp.float64))
    shape = u0.shape
    xland = sl._as_surface(sl._field(view, "xland", 1.0), shape)
    is_land = (xland - 1.5) < 0.0
    ust_in = jnp.maximum(sl._as_surface(sl._field(view, "ustar", 0.1), shape), 0.0)
    dz = jnp.maximum(sl._as_surface(sl._field(view, "dz", 100.0), shape), 1.0)
    za = 0.5 * dz
    wspd_raw = jnp.sqrt(u0 * u0 + v0 * v0)

    znt = jnp.asarray(diag.znt, jnp.float64)
    psim = jnp.asarray(diag.psim, jnp.float64)
    gz1oz0 = jnp.log((za + znt) / znt)
    psix = gz1oz0 - psim
    ustar_out = jnp.asarray(sf.ustar, jnp.float64)
    # K*wspd/psix implied by the OUTPUT ust and the carried ust_in:
    # ust = 0.5*ust_in + 0.5*K*wspd/psix  ->  K*wspd/psix = 2*ust - ust_in
    kwspd_over_psix = 2.0 * ustar_out - ust_in
    # implied wspd (incl convective vconv) = (K*wspd/psix)*psix/K
    wspd_implied = kwspd_over_psix * psix / KARMAN

    # WRF UST truth (interior depth-8) from wrfout h36.
    with Dataset(hpg_mod.fn(CPU, 36)) as d:
        wrf_ust = _np.asarray(d.variables["UST"][0])
        wrf_u10 = _np.asarray(d.variables["U10"][0])
        wrf_v10 = _np.asarray(d.variables["V10"][0])
        wrf_pblh = _np.asarray(d.variables["PBLH"][0])
        wrf_xland = _np.asarray(d.variables["XLAND"][0])

    land_mask = _np.asarray(is_land)
    wrf_ust_land = float(_np.nanmean(wrf_ust[8:-8, 8:-8][land_mask[8:-8, 8:-8]]))
    jax_ust_land = float(_np.nanmean(_np.asarray(ustar_out)[8:-8, 8:-8][land_mask[8:-8, 8:-8]]))

    out = {
        "schema": "v014_switzerland_sfclay_ustar_oracle",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": "warm-step (first_timestep=False) operational surface chain at h36 reinit",
        "interior_depth8_means": {
            "wrf_ust": _imean(wrf_ust),
            "jax_ustar_out": _imean(ustar_out),
            "jax_ust_in_carried": _imean(ust_in),
            "jax_wspd_raw": _imean(wspd_raw),
            "jax_wspd_implied_with_vconv": _imean(wspd_implied),
            "jax_znt": _imean(znt),
            "jax_za": _imean(za),
            "jax_gz1oz0": _imean(gz1oz0),
            "jax_psim": _imean(psim),
            "jax_psix": _imean(psix),
            "jax_K_wspd_over_psix": _imean(kwspd_over_psix),
            "jax_br": _imean(diag.br),
            "jax_zol": _imean(diag.zol),
            "wrf_pblh": _imean(wrf_pblh),
            "wrf_wspd10": _imean(_np.sqrt(wrf_u10**2 + wrf_v10**2)),
        },
        "land_only_depth8": {
            "wrf_ust_land": wrf_ust_land,
            "jax_ust_land": jax_ust_land,
            "jax_fraction_of_wrf_land": jax_ust_land / wrf_ust_land,
            "land_fraction": float(_np.mean(land_mask)),
        },
        "fractions": {
            "jax_ust_out_over_wrf": _imean(ustar_out) / _imean(wrf_ust),
            "jax_ust_in_over_wrf": _imean(ust_in) / _imean(wrf_ust),
        },
        "diagnosis_hint": (
            "If K*wspd/psix (the half WRF drives ust toward) is itself ~WRF UST, the "
            "carried ust_in is dragging the blend down (carry/seed defect). If "
            "K*wspd/psix is itself low, the deficit is in wspd (missing vconv) or psix "
            "(znt too small -> gz1oz0 too big, or psim wrong)."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
