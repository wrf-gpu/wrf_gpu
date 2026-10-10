"""Operational coupling of WRF CAM radiation (``ra_lw_physics = 3`` / ``ra_sw_physics = 3``).

WRF's radiation_driver passes CAM the same operands as RRTMG (first_rk_step_part1: hydrostatic P/P8W, phy_prep T,
PI, the icloud_bl-merged CLDFRA/QC/QI, ALBEDO/EMISS, COSZEN at xtime + radt/2, SOLCON, JULIAN, CLWRF gases), so the
operands come from the RRTMG radiation-operand builder (``physics_couplers._rrtmg_column_inputs``, MP radii off).
The CLWRF gases are evaluated in double like ``read_CAMgases`` (the REAL RRTMG preparation would round them).

Held-rate contract (as RRTMG/Dudhia/RRTM in the operational scan): the theta tendency RTHRATEN (K/s) is refreshed
at the radiation cadence and held in between; the SW scheme's GSW gives WRF's ``SWDOWN = GSW/(1 - ALBEDO)``
(module_radiation_driver.F:3002) and the LW scheme's GLW feeds the land surface.

Held absorptivities (WRF ``cam_abs_freq_s``, default 21600 s): the LW absorptivity/emissivity arrays are recomputed
only on WRF's STEPABS cadence (module_radiation_driver.F:1601, :func:`cam_doabsems`) and otherwise reused from the
REAL ``OperationalCarry.cam_abs`` (WRF state abstot/absnxt/emstot; restart v2 serializes it like every carry field).

Deviations (disclosed, CPU-qualified wave): (1) XICE = 0 (no sea-ice field in the operational State); (2) SNOW =
Noah-MP SNEQV over land, 0 elsewhere; (3) adaptive time stepping (WRF's adapt_step_flag doabsems branch) is not
supported by the port.
"""

from __future__ import annotations

from datetime import timedelta
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.physics import ra_cam as cam
from gpuwrf.physics.column_tiling import env_int, tiled_column_apply
from gpuwrf.physics.ra_cam_common import cam_ozmixm, lit, load_cam_abs_tables

# CAM LW absorptivities are O(nz^2) per column ((ncol, nz+1, nz+1) float64 temporaries): the coupler runs the column
# kernels over fixed-size column tiles (pure execution shape; columns are independent -> identical values).
CAM_TILE_COLS = env_int("GPUWRF_CAM_TILE_COLS", 2048)
_PER_COLUMN = ("t_phy", "p_phy", "p8w", "pi_phy", "qv", "qc", "qi", "qs", "cldfra", "xland", "xice", "snow", "emiss",
               "tsk", "xlat", "coszen", "albedo")


class CamOperands(NamedTuple):
    t_phy: jnp.ndarray
    p_phy: jnp.ndarray
    p8w: jnp.ndarray
    pi_phy: jnp.ndarray
    qv: jnp.ndarray
    qc: jnp.ndarray
    qi: jnp.ndarray
    qs: jnp.ndarray
    cldfra: jnp.ndarray
    xland: jnp.ndarray
    xice: jnp.ndarray
    snow: jnp.ndarray
    emiss: jnp.ndarray
    tsk: jnp.ndarray
    xlat: jnp.ndarray
    coszen: jnp.ndarray
    albedo: jnp.ndarray
    julian: jnp.ndarray
    solcon: jnp.ndarray
    gases: cam.CamGases


# module_ra_clWRF_support.F orig_val("CAM"): used when no dated CLWRF clock exists (idealized / bare calls).
_CAM_ORIG_GASES = (lit(3.55e-4), lit(0.311e-6), lit(1.714e-6), lit(0.280e-9), lit(0.503e-9))


def _cam_gases(grid, time_utc, lead_seconds, clock_base) -> cam.CamGases:
    gas_clock = getattr(clock_base, "ghg", None)
    if getattr(grid, "metrics", None) is not None and gas_clock is not None:
        with jax.enable_x64(True):
            g = pc.clwrf_gases_at_lead(gas_clock, lead_seconds)
        values = (g.co2_vmr, g.n2o_vmr, g.ch4_vmr, g.cfc11_vmr, g.cfc12_vmr)
    elif getattr(grid, "metrics", None) is not None and time_utc is not None:
        when = pc._coerce_datetime_utc(time_utc)
        if isinstance(lead_seconds, (int, float)):
            when += timedelta(seconds=float(lead_seconds))
        g = pc.clwrf_ssp245_gases_for_time(when)
        values = (g.co2_vmr, g.n2o_vmr, g.ch4_vmr, g.cfc11_vmr, g.cfc12_vmr)
    else:
        values = _CAM_ORIG_GASES
    return cam.CamGases(*(jnp.asarray(v, jnp.float64) for v in values))


def cam_operands(state, grid, *, time_utc=None, lead_seconds=0.0, clock_base=None, radiation_static=None,
                 land_state=None) -> CamOperands:
    """Flat ``(ncol, nz)`` WRF-order REAL operands of WRF's camrad call."""

    _sw, lw, albedo, emiss, geometry, _topo = pc._rrtmg_column_inputs(
        state, grid, time_utc=time_utc, lead_seconds=lead_seconds, clock_base=clock_base,
        radiation_static=radiation_static, land_state=land_state, use_mp_re=0)
    nz, ny, nx = state.theta.shape
    ncol = ny * nx
    flat = lambda a: jnp.reshape(jnp.asarray(a, jnp.float32), (ncol, -1))
    surf = lambda a: jnp.reshape(jnp.asarray(a, jnp.float32), (ncol,))
    p8w = lw.pressure_interfaces if lw.pressure_interfaces is not None else pc._interface_pressure_from_state(state)
    # phy_prep pi_phy = (p_phy/p1000mb)**rcp in REAL, p_phy the full (nonhydrostatic) pressure.
    p_full = jnp.asarray(pc._to_columns(state.p), jnp.float32)
    rcp = np.float32(np.float32(287.0) / (np.float32(7.0) * np.float32(287.0) / np.float32(2.0)))
    pi_phy = jnp.power(p_full / jnp.float32(1.0e5), rcp)
    static = pc._radiation_static_for_grid(state.t_skin.shape, grid, radiation_static, state.t_skin.dtype)
    lat = (static.xlat_deg if static is not None
           else pc._grid_lat_lon(state.t_skin.shape, grid, state.t_skin.dtype)[0])
    julian, utc_minute = pc._resolve_clock_parts(time_utc, clock_base)
    julian_real = pc._wrf_solar_julian_real(julian, utc_minute, lead_seconds)
    solcon = pc._solcon_for_time(time_utc, lead_seconds, clock_base=clock_base)
    xland = jnp.asarray(state.xland, jnp.float32)
    snow = jnp.zeros_like(xland)
    if land_state is not None and hasattr(land_state, "sneqv"):
        snow = jnp.where(xland < 1.5, jnp.asarray(land_state.sneqv, jnp.float32), 0.0)
    return CamOperands(
        t_phy=flat(lw.T), p_phy=flat(lw.p), p8w=flat(p8w), pi_phy=flat(pi_phy), qv=flat(lw.qv), qc=flat(lw.qc),
        qi=flat(lw.qi), qs=flat(lw.qs), cldfra=flat(lw.cloud_fraction), xland=surf(xland),
        xice=jnp.zeros((ncol,), jnp.float32), snow=surf(snow), emiss=surf(emiss), tsk=surf(lw.surface_temperature),
        xlat=surf(lat), coszen=surf(geometry.coszen), albedo=surf(albedo),
        julian=jnp.asarray(julian_real, jnp.float32), solcon=jnp.asarray(solcon, jnp.float32),
        gases=_cam_gases(grid, time_utc, lead_seconds, clock_base))


def _tiled(fn, op: CamOperands, extra=None):
    cols = {name: getattr(op, name) for name in _PER_COLUMN}
    if extra:
        cols.update(extra)
    ncol = op.t_phy.shape[0]
    if ncol <= CAM_TILE_COLS:
        return fn(cols)
    return tiled_column_apply(fn, cols, ncol=ncol, tile_cols=CAM_TILE_COLS)


def _lw_call(op: CamOperands, doabsems=True, held=None):
    tables = load_cam_abs_tables()
    if doabsems:
        def fn(c):
            ozm, pin = cam_ozmixm(c["xlat"])
            return cam.camrad_lw(**c, julian=op.julian, gases=op.gases, ozmixm=ozm, pin=pin, tables=tables,
                                 doabsems=True)

        return _tiled(fn, op)

    def fn_held(c):
        held_tile = cam.CamHeld(c.pop("abstot"), c.pop("absnxt"), c.pop("emstot"))
        ozm, pin = cam_ozmixm(c["xlat"])
        return cam.camrad_lw(**c, julian=op.julian, gases=op.gases, ozmixm=ozm, pin=pin, tables=tables,
                             doabsems=False, held=held_tile)

    return _tiled(fn_held, op, extra=held._asdict())


def cam_doabsems(lead_seconds, dt_s, radiation_cadence_steps, cam_abs_freq_s, held=None):
    """WRF radiation_driver doabsems (module_radiation_driver.F:1601) at a radiation call of the operational scan.

    STEPABS = nint(cam_abs_freq_s/(dt*STEPRA))*STEPRA; doabsems = itimestep == 1 .or. mod(itimestep, STEPABS) == 1
    .or. STEPABS == 1, with itimestep = 1 + lead/dt (radiation calls at mod(itimestep - 1, STEPRA) == 0).  camrad's
    restart guard (module_ra_cam.F: abstot_3d(its,kts,kts,jts) == 0 forces doabsems) is kept for all-zero held arrays.
    STEPABS <= 1 (cam_abs_freq_s <= radt/2 rounding to 0, where WRF's MOD by 0 is undefined) recomputes every call.
    """

    stepra = int(radiation_cadence_steps)
    stepabs = int(np.floor(float(cam_abs_freq_s) / (float(dt_s) * stepra) + 0.5)) * stepra
    itimestep = 1 + jnp.rint(jnp.asarray(lead_seconds, jnp.float64) / float(dt_s)).astype(jnp.int32)
    if stepabs <= 1:
        due = jnp.asarray(True)
    else:
        due = (itimestep == 1) | (jnp.mod(itimestep, stepabs) == 1)
    if held is not None:
        due = due | (jnp.asarray(held.abstot)[0, 0, 0] == 0.0)
    return due


def initial_cam_held(state) -> cam.CamHeld:
    """WRF abstot_3d/absnxt_3d/emstot_3d start as zero REAL state arrays (CAM order, flat columns)."""

    nz, ny, nx = state.theta.shape
    ncol = ny * nx
    z = lambda *shape: jnp.zeros((ncol,) + shape, jnp.float32)
    return cam.CamHeld(z(nz + 1, nz + 1), z(nz, 4), z(nz + 1))


def _lw_held_call(op: CamOperands, held, doabsems):
    """camrad LW with WRF's doabsems switch: fresh absorptivities (stored as REAL) or the held REAL arrays."""

    if held is None:
        return _lw_call(op)
    return jax.lax.cond(doabsems, lambda _: _lw_call(op), lambda _: _lw_call(op, doabsems=False, held=held), None)


def _mxaerl(grid) -> int:
    """aerosol_init background-aerosol depth from the grid's ZNU (= mid-points of ZNW) and p_top."""

    if grid is None:
        raise ValueError("CAM shortwave needs a GridSpec (aerosol_init mxaerl from ZNU and p_top)")
    znu = cam.znu_from_znw(np.asarray(grid.vertical.eta_levels))
    return cam.cam_mxaerl(znu, float(grid.vertical.top_pressure_pa))


def _sw_call(op: CamOperands, grid):
    mxaerl = _mxaerl(grid)

    def fn(c):
        ozm, pin = cam_ozmixm(c["xlat"])
        return cam.camrad_sw(**c, julian=op.julian, solcon=op.solcon, co2vmr=op.gases.co2vmr, ozmixm=ozm, pin=pin,
                             mxaerl=mxaerl)

    return _tiled(fn, op)


def _to_grid(field_cols, state):
    nz, ny, nx = state.theta.shape
    return jnp.moveaxis(jnp.reshape(field_cols, (ny, nx, nz)), -1, 0)


def cam_lw_theta_tendency(state, grid=None, *, time_utc=None, lead_seconds=0.0, clock_base=None,
                          radiation_static=None, land_state=None, held=None, doabsems=True):
    """CAM longwave RTHRATEN (K/s, theta tendency) on the ``(nz, ny, nx)`` mass grid.

    With ``held`` (the carry's :class:`~gpuwrf.physics.ra_cam.CamHeld`) returns ``(rthraten, new_held)`` and switches
    between fresh and held absorptivities on the traced ``doabsems`` (see :func:`cam_doabsems`)."""

    op = cam_operands(state, grid, time_utc=time_utc, lead_seconds=lead_seconds, clock_base=clock_base,
                      radiation_static=radiation_static, land_state=land_state)
    out = _lw_held_call(op, held, doabsems)
    rth = _to_grid(out.rthratenlw, state).astype(pc._output_dtype(state, "theta"))
    return rth if held is None else (rth, out.held)


def cam_sw_theta_tendency(state, grid=None, *, time_utc=None, lead_seconds=0.0, clock_base=None,
                          radiation_static=None, land_state=None):
    """CAM shortwave RTHRATEN (K/s, theta tendency) on the ``(nz, ny, nx)`` mass grid."""

    op = cam_operands(state, grid, time_utc=time_utc, lead_seconds=lead_seconds, clock_base=clock_base,
                      radiation_static=radiation_static, land_state=land_state)
    out = _sw_call(op, grid)
    return _to_grid(out.rthratensw, state).astype(pc._output_dtype(state, "theta"))


def cam_surface_lw(state, grid=None, *, time_utc=None, lead_seconds=0.0, clock_base=None, radiation_static=None,
                   land_state=None):
    """CAM GLW (W/m2, ``(ny, nx)``) for the land surface forcing."""

    op = cam_operands(state, grid, time_utc=time_utc, lead_seconds=lead_seconds, clock_base=clock_base,
                      radiation_static=radiation_static, land_state=land_state)
    return jnp.reshape(_lw_call(op).glw, state.t_skin.shape)


def cam_surface_forcing(state, grid, soldn, lwdn, *, ra_lw_physics, ra_sw_physics, time_utc=None, lead_seconds=0.0,
                        clock_base=None, radiation_static=None, land_state=None, held=None, doabsems=True):
    """Replace the surface forcing pair by CAM's GLW (ra_lw = 3) / SWDOWN = GSW/(1-ALBEDO) (ra_sw = 3)."""

    op = cam_operands(state, grid, time_utc=time_utc, lead_seconds=lead_seconds, clock_base=clock_base,
                      radiation_static=radiation_static, land_state=land_state)
    if ra_lw_physics == 3:
        glw = _lw_held_call(op, held, doabsems).glw
        lwdn = jnp.reshape(glw, state.t_skin.shape).astype(jnp.asarray(lwdn).dtype)
    if ra_sw_physics == 3:
        soldn = jnp.reshape(_sw_call(op, grid).swdown, state.t_skin.shape).astype(jnp.asarray(soldn).dtype)
    return soldn, lwdn


__all__ = ["CamOperands", "cam_doabsems", "cam_lw_theta_tendency", "initial_cam_held", "cam_operands", "cam_surface_forcing", "cam_surface_lw",
           "cam_sw_theta_tendency"]
