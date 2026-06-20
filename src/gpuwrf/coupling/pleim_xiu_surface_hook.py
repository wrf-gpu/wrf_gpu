"""Operational surface hook for the Pleim-Xiu LSM (sf_surface_physics=7).

State<->land-carry adapter for the JAX Pleim-Xiu (PX) LSM column kernel
(``physics.lsm_pleim_xiu.pxlsm_columns``, a faithful fp64-oracle-validated port
of WRF ``phys/module_sf_pxlsm.F`` SURFPX+QFLUX). It mirrors the Noah-classic /
slab operational pattern:

1. the selected surface-layer path (the PX surface layer ``sf_sfclay_physics=7``
   is the WRF-paired choice) has already populated the resident surface exchange
   handles (``ustar``, ``rhosfc``, ``theta_flux``, ``qv_flux``);
2. the PX LSM advances its 2-layer ISBA land state (TG/T2 soil temperatures,
   WG/W2 soil moisture, WR canopy water) from the down-radiation (GSW/GLW) and a
   WRF-derived vegetation/soil static bundle (the Noilhan-Mahfouf ISBA constants
   ``SOILPROP`` produces + the ``VEGELAND`` vegetation fields);
3. land ``TSK``/``HFX``/``QFX`` overwrite the same State handles the PBL reads;
   water columns keep the surface-layer path via one land/water ``where``.

The Monin-Obukhov inverse length ``RMOL`` SURFPX consumes is NOT carried on the
resident State, so -- exactly as the Noah-classic hook back-derives ``CH`` and
the slab hook back-derives ``FLHC``/``FLQC`` from the resident kinematic flux
handles -- this hook reconstructs ``RMOL`` from the resident ``ustar`` and
``theta_flux`` via Monin-Obukhov similarity, which keeps the LSM SOLVE consistent
with the surface-layer exchange that produced the handles.

The ``PleimXiuStatic`` bundle (ISBA soil constants + vegetation/surface fields)
is a per-run read-only input, like the Noah-classic REDPRM block. The PX sub-step
count ``ntsps`` is a Python static int (depends only on the step), so the kernel
stays fully ``jax.jit``/``jax.vmap``-traceable.

Cited to ``<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_pxlsm.F`` (SURFPX,
QFLUX, and the ``pxlsm`` driver per-column input prep, lines 600-672).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from gpuwrf.coupling.physics_couplers import (
    GRAVITY_M_S2,
    _output_dtype,
    _temperature_from_theta,
)
from gpuwrf.physics.lsm_pleim_xiu import (
    PleimXiuStatic,
    ntsps_substeps,
    pxlsm_columns,
)

CPD = 1004.67
KARMAN = 0.4
XLV = 2.5e6


class PleimXiuStaticBundle(NamedTuple):
    """Read-only PX-LSM static inputs for one operational tile.

    ``params`` is the per-column :class:`PleimXiuStatic` (ISBA soil constants +
    vegetation/surface fields, each a ``(ny, nx)`` array). ``ifsnow`` toggles the
    snow path (the resident State does not carry a PX snow field, so it is a
    bundle option, default off).
    """

    params: PleimXiuStatic
    ifsnow: int = 0


class PleimXiuLandState(NamedTuple):
    """2-layer Pleim-Xiu ISBA land carry plus last land flux diagnostics."""

    tg: jax.Array            # (ny, nx) soil temperature, layer 1 (= TSLB(1)) (K)
    t2: jax.Array            # (ny, nx) soil temperature, root zone (= TSLB(2)) (K)
    wg: jax.Array            # (ny, nx) soil moisture, layer 1 (m3/m3)
    w2: jax.Array            # (ny, nx) soil moisture, root zone (m3/m3)
    wr: jax.Array            # (ny, nx) canopy water content (m)
    hfx: jax.Array           # (ny, nx) last land sensible heat flux (W/m^2)
    qfx: jax.Array           # (ny, nx) last land moisture flux (kg/m^2/s)
    lh: jax.Array            # (ny, nx) last land latent heat flux (W/m^2)
    grdflx: jax.Array        # (ny, nx) ground heat flux (W/m^2)
    canwat: jax.Array        # (ny, nx) canopy water (mm) diagnostic
    capg: jax.Array          # (ny, nx) ground heat capacity diagnostic
    ta2: jax.Array           # (ny, nx) diagnostic 2-m temperature (K)
    qa2: jax.Array           # (ny, nx) diagnostic 2-m mixing ratio (kg/kg)

    def replace(self, **updates) -> "PleimXiuLandState":
        return self._replace(**updates)


class PleimXiuRadiation(NamedTuple):
    """Held surface down-radiation forcing into the PX LSM (GSW/GLW)."""

    gsw: jax.Array           # downward shortwave at ground (W/m^2)
    glw: jax.Array           # downward longwave at ground (W/m^2)


def _surface_2d(field):
    a = jnp.asarray(field, dtype=jnp.float64)
    return a[..., 0] if a.ndim >= 3 else a


def _held_rad(radiation: Any, shape) -> PleimXiuRadiation:
    if radiation is None:
        zero = jnp.zeros(shape, dtype=jnp.float64)
        return PleimXiuRadiation(zero, zero)
    if isinstance(radiation, PleimXiuRadiation):
        return PleimXiuRadiation(_surface_2d(radiation.gsw), _surface_2d(radiation.glw))
    if isinstance(radiation, tuple):
        return PleimXiuRadiation(_surface_2d(radiation[0]), _surface_2d(radiation[1]))
    return PleimXiuRadiation(
        _surface_2d(getattr(radiation, "gsw")),
        _surface_2d(getattr(radiation, "glw")),
    )


def _reference_height(state) -> jax.Array:
    """Lowest half-level height Z1 = 0.5*dz (m), from the geopotential interfaces."""

    interface_z = jnp.asarray(state.ph, dtype=jnp.float64) / GRAVITY_M_S2
    return 0.5 * jnp.maximum(interface_z[1] - interface_z[0], 1.0)


def _rmol_from_handles(state, theta1, qv1) -> jax.Array:
    """Reconstruct RMOL (=1/MOL) from the resident ustar + theta_flux handles.

    The surface-layer path wrote ``theta_flux = HFX/(rho*cp)`` (kinematic, K m/s).
    The Monin-Obukhov length is ``MOL = -ustar^3 * theta_v / (karman * g * w'th')``
    where ``w'th'`` is the kinematic virtual heat flux. Using the resident handle
    keeps RMOL consistent with the surface-layer exchange that produced it (the
    same seam the slab hook uses for FLHC/FLQC). Where the flux is ~0 (neutral),
    RMOL -> 0 (MOL -> +-inf), which the kernel handles (molx clamp).
    """

    ust = jnp.maximum(jnp.asarray(state.ustar, dtype=jnp.float64), 0.005)
    theta_flux = jnp.asarray(state.theta_flux, dtype=jnp.float64)
    qv_flux = jnp.asarray(state.qv_flux, dtype=jnp.float64)
    # Virtual kinematic heat flux (K m/s): w'thv' = w'th' + 0.61*theta*w'qv'.
    wthv = theta_flux + 0.61 * theta1 * qv_flux
    theta_v = theta1 * (1.0 + 0.61 * jnp.maximum(qv1, 0.0))
    # MOL = -ustar^3 * theta_v / (karman * g * wthv); rmol = 1/MOL.
    denom = ust * ust * ust * theta_v
    denom_safe = jnp.where(jnp.abs(denom) > 1.0e-30, denom, 1.0e-30)
    rmol = -KARMAN * GRAVITY_M_S2 * wthv / denom_safe
    # Clamp |rmol| to the WRF MOLX-equivalent range (|MOL| >= 1e-3 -> |rmol|<=1000).
    return jnp.clip(rmol, -1000.0, 1000.0)


def pleim_xiu_surface_step(
    state: Any,
    land_state: PleimXiuLandState,
    static: PleimXiuStaticBundle,
    dt: float,
    *,
    radiation: Any = None,
) -> tuple[Any, PleimXiuLandState]:
    """Run the PX LSM over land and write land flux handles back into State.

    Returns ``(state_out, next_land)``. Water columns (XLAND >= 1.5) keep the
    surface-layer path; land columns advance TSK and the 2-layer ISBA carry.
    """

    if land_state is None or static is None:
        raise ValueError(
            "Pleim-Xiu scan coupling requires px_land (PleimXiuLandState) and "
            "px_static (PleimXiuStaticBundle)"
        )

    ntsps = ntsps_substeps(float(dt))
    grid_shape = jnp.asarray(state.t_skin, dtype=jnp.float64).shape  # (ny, nx)
    rad = _held_rad(radiation, grid_shape)

    # --- per-column forcing from the resident State (mirrors the WRF driver) ---
    ta1_2d = _temperature_from_theta(state.theta, state.p)[0]
    theta1_2d = jnp.asarray(state.theta[0], dtype=jnp.float64)
    qv1_2d = jnp.maximum(jnp.asarray(state.qv[0], dtype=jnp.float64), 0.0)
    # Surface pressure proxy = lowest model level, converted Pa -> cb (PSFC/1000).
    psurf_cb_2d = jnp.maximum(jnp.asarray(state.p[0], dtype=jnp.float64), 1.0) / 1000.0
    dens1_2d = jnp.maximum(jnp.asarray(state.rhosfc, dtype=jnp.float64), 1.0e-6)
    ust_2d = jnp.asarray(state.ustar, dtype=jnp.float64)
    z1_2d = _reference_height(state)
    cpair_2d = CPD * (1.0 + 0.84 * qv1_2d)
    rmol_2d = _rmol_from_handles(state, theta1_2d, qv1_2d)
    xland_2d = _surface_2d(getattr(state, "xland", jnp.ones_like(ta1_2d)))
    # ifland: 1 = land, 2 = water (WRF convention). isnow from snow fraction.
    ifland_2d = jnp.where(xland_2d < 1.5, 1.0, 2.0)
    isnow_2d = jnp.where(jnp.asarray(static.ifsnow) > 0, _surface_2d(static.params.snow_fra), 0.0)
    # No precip coupling in this seam (like Noah-classic prcp=0); qst12 ~ 0 (the
    # 2-level moisture-gradient diagnostic only feeds QA2, a diagnostic output).
    zero2d = jnp.zeros_like(ta1_2d)
    precip_2d = zero2d
    qst12_2d = zero2d

    p = static.params

    def _flat(a):
        return jnp.asarray(a, dtype=jnp.float64).reshape(-1)

    out_flat = pxlsm_columns(
        _flat(rad.gsw),          # soldn (SOLDN = GSW for this seam)
        _flat(rad.gsw),          # gsw (net SW)
        _flat(rad.glw),          # lwdn
        _flat(z1_2d), _flat(rmol_2d), _flat(ust_2d), _flat(psurf_cb_2d),
        _flat(dens1_2d), _flat(qv1_2d), _flat(ta1_2d), _flat(theta1_2d),
        _flat(precip_2d), _flat(cpair_2d), _flat(qst12_2d),
        _flat(ifland_2d), _flat(isnow_2d),
        _flat(land_state.tg), _flat(land_state.t2), _flat(land_state.wg),
        _flat(land_state.w2), _flat(land_state.wr),
        vegfrc=_flat(p.vegfrc), lai=_flat(p.lai), imperv=_flat(p.imperv),
        canfra=_flat(p.canfra), rstmin=_flat(p.rstmin), emissi=_flat(p.emissi),
        znt=_flat(p.znt), wetfra=_flat(p.wetfra), hc_snow=_flat(p.hc_snow),
        snow_fra=_flat(p.snow_fra), wwlt=_flat(p.wwlt), wfc=_flat(p.wfc),
        wres=_flat(p.wres), cgsat=_flat(p.cgsat), wsat=_flat(p.wsat),
        b=_flat(p.b), c1sat=_flat(p.c1sat), c2r=_flat(p.c2r), asoil=_flat(p.asoil),
        jp=_flat(p.jp), c3=_flat(p.c3), ds1=_flat(p.ds1), ds2=_flat(p.ds2),
        dt=float(dt), ntsps=ntsps,
    )
    out = {k: v.reshape(grid_shape) for k, v in out_flat.items() if v.ndim == 1}

    is_land = (xland_2d - 1.5) < 0.0
    rho = dens1_2d
    theta_flux = out["hfx"] / (rho * CPD)
    qv_flux = out["qfx"] / rho

    def _blend(old, new):
        return jnp.where(is_land, jnp.asarray(new, dtype=jnp.float64), jnp.asarray(old, dtype=jnp.float64))

    state_out = state.replace(
        t_skin=_blend(state.t_skin, out["tsk"]).astype(_output_dtype(state, "t_skin")),
        theta_flux=_blend(state.theta_flux, theta_flux).astype(_output_dtype(state, "theta_flux")),
        qv_flux=_blend(state.qv_flux, qv_flux).astype(_output_dtype(state, "qv_flux")),
    )
    next_land = land_state.replace(
        tg=_blend(land_state.tg, out["tg"]),
        t2=_blend(land_state.t2, out["t2"]),
        wg=_blend(land_state.wg, out["wg"]),
        w2=_blend(land_state.w2, out["w2"]),
        wr=_blend(land_state.wr, out["wr"]),
        hfx=_blend(land_state.hfx, out["hfx"]),
        qfx=_blend(land_state.qfx, out["qfx"]),
        lh=_blend(land_state.lh, out["lh"]),
        grdflx=_blend(land_state.grdflx, out["grdflx"]),
        canwat=_blend(land_state.canwat, out["canwat"]),
        capg=_blend(land_state.capg, out["capg"]),
        ta2=_blend(land_state.ta2, out["ta2"]),
        qa2=_blend(land_state.qa2, out["qa2"]),
    )
    return state_out, next_land


def initial_pleim_xiu_land(state: Any, static: PleimXiuStaticBundle) -> PleimXiuLandState:
    """Seed the PX 2-layer ISBA land carry from the resident State + static bundle.

    TG (surface soil temp) is seeded from TSK; T2 (root-zone) from a TSK->field
    deep value; WG/W2 from State.soil_moisture clipped to [WRES, WSAT]; WR=0
    (dry canopy cold-start). This is the standard PX cold-start when a wrfinput
    TSLB/SMOIS is unavailable.
    """

    p = static.params
    tsk = jnp.asarray(state.t_skin, dtype=jnp.float64)
    sm = jnp.asarray(_surface_2d(state.soil_moisture), dtype=jnp.float64)
    wg0 = jnp.clip(sm, jnp.asarray(p.wres, dtype=jnp.float64), jnp.asarray(p.wsat, dtype=jnp.float64))
    zero = jnp.zeros_like(tsk)
    return PleimXiuLandState(
        tg=tsk,
        t2=tsk,
        wg=wg0,
        w2=wg0,
        wr=zero,
        hfx=zero,
        qfx=zero,
        lh=zero,
        grdflx=zero,
        canwat=zero,
        capg=zero,
        ta2=jnp.asarray(_temperature_from_theta(state.theta, state.p)[0], dtype=jnp.float64),
        qa2=jnp.asarray(state.qv[0], dtype=jnp.float64),
    )


__all__ = [
    "PleimXiuStaticBundle",
    "PleimXiuLandState",
    "PleimXiuRadiation",
    "pleim_xiu_surface_step",
    "initial_pleim_xiu_land",
]
