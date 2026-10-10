"""Operational ``State -> State`` adapter for NSSL 2-moment microphysics (``mp_physics=18``).

Mirrors WRF ``microphysics_driver`` CASE (NSSL_2MOM) (module_microphysics_driver.F:2237-2303) on the
port's State, exactly like the Thompson adapter does for mp=8:

* inputs: dry ``th_phy`` from theta_m (``phy_prep`` use_theta_m=1), ``pi_phy = (p/p0)**rcp``,
  ``rho`` = WRF phy_prep density ``rho_d*(1+qv)`` (EOS dry density times (1+qv)), ``dz8w`` (g=9.81), ``w`` at the
  bottom w-face (WRF ``w(i,k,j)``), hydrometeors/numbers/volumes in WRF NSSL Registry order:
  qg -> NSSL graupel, qh -> NSSL hail, Nc -> qndrop, Nn -> qnn (activated CCN), Ni/Nr/Ns/Ng/Nh,
  qvolg/qvolh;
* outputs: theta_m recoupled with ``moist_physics_finish_em`` (``_theta_m_from_microphysics``),
  every prognostic replaced, precipitation added to the port's DISJOINT accumulators
  (rain_acc = liquid ``dtp*dn*xfall(lr)``, snow_acc = SNOWNCV, graupel_acc = GRPLNCV,
  hail_acc = HAILNCV; the wrfout writer folds them into WRF's all-phase RAINNC).

``first_step`` (traced bool) reproduces WRF's ``itimestep == 1`` cold start (CN := 0 and calcnfromq).
REAL precision follows the carry: fp32 State -> WRF REAL build constants, fp64 State -> the
``-fdefault-real-8`` constants (both oracle-qualified).
"""

from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.contracts.state import State

from .constants import get_constants
from .driver import nssl2mom_column
from .indices import FP32, FP64

# State leaf -> WRF NSSL field name used by driver.nssl2mom_column
_STATE_TO_WRF = {
    "qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs", "qg": "qg", "qh": "qh",
    "Nc": "qndrop", "Nr": "qnr", "Ni": "qni", "Ns": "qns", "Ng": "qng", "Nh": "qnh", "Nn": "qnn",
    "qvolg": "qvolg", "qvolh": "qvolh",
}


def nssl2mom_adapter(state: State, dt: float, grid=None, *, first_step=False, return_precipitation=False):
    """mp=18 NSSL 2-moment scan adapter (one WRF microphysics call)."""
    from gpuwrf.coupling.physics_couplers import (
        P0_PA, R_D_OVER_CP, _dry_theta_view, _from_columns, _output_dtype, _surface_dz_from_state,
        _temperature_from_theta, _theta_m_from_microphysics, _to_columns, density_from_pressure_temperature)

    del grid
    state = state.ensure_conditional_leaves(mp_physics=18)
    R = jnp.asarray(state.theta).dtype
    prec = FP32 if R == jnp.float32 else FP64
    C = get_constants(prec.name)
    theta_dry = _dry_theta_view(state)
    T = _temperature_from_theta(theta_dry, state.p)
    # WRF phy_prep passes rho = (1+qv)/alt = rho_d*(1+qv) as NSSL's DN (module_big_step_utilities_em.F:4856);
    # density_from_pressure_temperature is the DRY density rho_d (Thompson convention) -> times (1+qv).
    rho = density_from_pressure_temperature(state.p, T, state.qv) * (1.0 + state.qv)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    fields = {wrf: _to_columns(getattr(state, leaf)) for leaf, wrf in _STATE_TO_WRF.items()}
    fields.update(th=_to_columns(theta_dry), pii=_to_columns(pii), p=_to_columns(state.p),
                  rho=_to_columns(rho), w=_to_columns(state.w[:-1]), dz=_surface_dz_from_state(state))
    out, precip, _diag = nssl2mom_column(fields, dt, C, prec, first_step=first_step)

    qv_new = _from_columns(out["qv"]).astype(_output_dtype(state, "qv"))
    theta_new_dry = _from_columns(out["th"]).astype(_output_dtype(state, "theta"))
    updates = {"theta": _theta_m_from_microphysics(theta_dry, state.qv, theta_new_dry, qv_new,
                                                   _output_dtype(state, "theta")),
               "qv": qv_new}
    for leaf, wrf in _STATE_TO_WRF.items():
        if leaf != "qv":
            updates[leaf] = _from_columns(out[wrf]).astype(_output_dtype(state, leaf))
    for acc, key in (("rain_acc", "rain_liquid"), ("snow_acc", "snowncv"), ("graupel_acc", "grplncv"),
                     ("hail_acc", "hailncv")):
        prev = getattr(state, acc)
        if prev is not None:
            updates[acc] = (jnp.asarray(prev, jnp.float64) + jnp.asarray(precip[key], jnp.float64)).astype(
                _output_dtype(state, acc))
    new_state = state.replace(**updates)
    if return_precipitation:
        return new_state, precip
    return new_state
