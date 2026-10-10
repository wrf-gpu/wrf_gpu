"""Operational surface hook for the RUC land-surface model (``sf_surface_physics=3``).

State <-> land-carry adapter for the faithful JAX port ``physics.ruclsm`` of WRF
``phys/module_sf_ruclsm.F`` (LSMRUC).  Follows the WRF call order in
``phys/module_surface_driver.F`` (surface layer, then ``CASE (RUCLSMSCHEME)`` at
:3541-3640):

1. the MYNN surface layer (``SFCLAY_mynn``, the port's ``surface_layer``) runs once
   on the resident State (TSK/QSFC/MAVAIL/ZNT/SNOWH are the values RUC wrote on the
   previous step) and writes the usual flux handles; its exchange coefficients
   ``FLHC``/``FLQC`` are taken directly from the surface layer (no flux inversion);
2. LSMRUC advances every column (``myj=.false.``: ``QKMS=FLQC/RHO/MAVAIL``,
   ``TKMS=FLHC/RHO/(CP*(1+0.84*QV))``), forced by the level-1 atmosphere exactly as
   the driver passes it (``t_phy``, ``qv_curr``, ``qc_curr``, hydrostatic ``p_phy``,
   ``rho``, ``dz8w``), the held radiation and the previous-step precipitation
   (``RAINBL=RAINCV+RAINNCV``, ``SR`` = frozen fraction -> ``frpcpn=.true.`` when the
   microphysics provides it, as with Thompson);
3. land ``TSK``/``HFX``/``QFX``/``QSFC``/``MAVAIL``/``ZNT`` overwrite the State handles
   the PBL and the next surface-layer call read; the kinematic flux handles are
   rebuilt from the land HFX/QFX (same seam as the Noah-MP coupler).  Water columns
   keep the surface-layer values.

Known coupling simplifications (documented, not hidden):

* GSW: with the native RRTMG driver the held WRF ``GSW = SWDOWN - SWUP`` (net, radiation
  albedo) is used as in WRF; on the legacy surface-only refresh the held DOWNWARD
  shortwave is converted with the step-start RUC albedo (``GSWIN`` then equals SOLDN).
* RUC ALBEDO/EMISS are carried and written to history but not yet fed back into the
  port's radiation surface albedo/emissivity.
* sea ice (``XICE >= xice_threshold`` on land) and lake-model points fail closed
  (``unsupported``), as does ``sf_sfclay_physics`` other than MYNN-SL, an MP other than Thompson
  (the precipitation carry) and the MYJ PBL (LSMRUC myj path).
* WRF calls ``SFCDIAGS_RUCLSM`` after LSMRUC (module_surface_driver.F:3608-3620: RUC 2-m T2/Q2/TH2,
  and it recomputes CHS2/CQS2); that routine is NOT ported -- the 2-m history fields of a RUC run come
  from the generic surface-layer diagnostics path, not WRF's RUC formulas (station T2 is affected).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from gpuwrf.physics.ruclsm import (
    STATE_PROFILES,
    STATE_SCALARS,
    RucConfig,
    RucTables,
    _c,
    lsmruc_step,
    qsn,
    ruclsminit,
    tbq_table,
)

RucLandState = NamedTuple("RucLandState", [(name, Any) for name in STATE_SCALARS + STATE_PROFILES])
RucLandState.__module__ = __name__
RucLandState.__doc__ = "RUC land carry: WRF LSMRUC INOUT grid fields, (ny, nx) and (ny, nx, nzs)."


class RucRadiation(NamedTuple):
    """Surface radiation into RUC: ``gsw`` = WRF GSW (NET shortwave absorbed at the
    ground, W m-2) when passed to :func:`ruc_surface_step`; the legacy held carry
    stores the DOWNWARD shortwave instead (see :func:`ruc_net_radiation`)."""

    gsw: jax.Array
    glw: jax.Array  # downward longwave at the ground (W m-2)


def ruc_net_radiation(soldn, glw, alb):
    """Legacy held SOLDN -> WRF GSW with the step-start RUC albedo (module doc)."""

    return RucRadiation(jnp.asarray(soldn) * (1.0 - jnp.asarray(alb, jnp.asarray(soldn).dtype)), glw)


class RucStaticBundle(NamedTuple):
    """Read-only per-run RUC inputs (rides the namelist static aux)."""

    cfg: RucConfig
    tables: RucTables
    ivgtyp: Any
    isltyp: Any
    xland: Any
    xice: Any
    tbot: Any       # TMN
    shdmin: Any
    shdmax: Any
    albbck: Any
    landusef: Any = None   # (ny, nx, nlcat) for mosaic_lu=1
    soilctop: Any = None   # (ny, nx, nscat) for mosaic_soil=1


def _flat(x):
    a = jnp.asarray(x)
    return a.reshape((-1,) + a.shape[2:])


def _unflat(x, shape):
    return x.reshape(tuple(shape) + x.shape[1:])


def ruc_precipitation_forcing(precipitation, dt: float, shape, dtype):
    """Previous-step precipitation for LSMRUC from the held MP/cumulus rates.

    ``precipitation`` is the operational ``NoahMPPrecipitation`` carry (rates in
    mm/s: convective, total grid-scale, snow+ice, graupel).  WRF surface_driver
    accumulates ``RAINBL += RAINCV + RAINNCV`` (:1568) and Thompson sets
    ``SR = (snow+graupel+ice)/(RAINNCV+1e-12)``.
    """

    if precipitation is None:
        z = jnp.zeros(shape, dtype)
        return dict(rainbl=z, rainncv=z, snowncv=z, graupelncv=z, frzfrac=z)
    d = jnp.asarray(dt, dtype)
    conv = jnp.asarray(precipitation.prcpconv, dtype) * d
    nonc = jnp.asarray(precipitation.prcpnonc, dtype) * d
    snow = jnp.asarray(precipitation.prcpsnow, dtype) * d
    grpl = jnp.asarray(precipitation.prcpgrpl, dtype) * d
    rainbl = jnp.maximum(conv + nonc, 0.0)
    frz = (snow + grpl) / (nonc + 1.0e-12)
    return dict(rainbl=rainbl, rainncv=nonc, snowncv=snow, graupelncv=grpl, frzfrac=frz)


def _state_surface(state, name, value):
    from gpuwrf.coupling.physics_couplers import _output_dtype

    return jnp.asarray(value).astype(_output_dtype(state, name))


def ruc_surface_step(state: Any, land: RucLandState, static: RucStaticBundle, dt: float,
                     grid: Any = None, *, radiation: Any = None, precipitation: Any = None,
                     first_timestep: Any = False):
    """MYNN surface layer + LSMRUC over all columns; returns ``(state', land')``."""

    from gpuwrf.coupling.physics_couplers import _surface_column_view, _temperature_from_theta
    from gpuwrf.physics.surface_constants import CP_D, EP1
    from gpuwrf.physics.surface_layer import (
        _mynn_surface_exner_kwargs,
        mynn_driver_surface_fluxes,
        mynn_fltv_wrf_enabled,
        surface_layer_with_exchange,
    )

    if land is None or static is None:
        raise ValueError("RUC scan coupling requires ruc_land (RucLandState) and ruc_static (RucStaticBundle)")
    cfg = static.cfg
    dt_ = cfg.np_dtype
    if abs(float(cfg.dt) - float(dt)) > 1.0e-9:
        raise ValueError(f"ruc_static.cfg.dt={cfg.dt} does not match the scan dt={dt}")

    # ---- 1. MYNN surface layer on the RUC-written TSK/QSFC/MAVAIL/ZNT/SNOWH
    view = _surface_column_view(state, grid)
    diag, flhc, flqc = surface_layer_with_exchange(view, first_timestep=first_timestep, snowh=land.snowh)
    flux = diag.fluxes
    shape = jnp.shape(state.t_skin)

    # ---- 2. LSMRUC forcing (surface_driver :3603 argument mapping)
    if view.t_air is not None:
        t1 = view.t_air[..., 0]
    else:
        t1 = _temperature_from_theta(state.theta, state.p)[0]
    p1 = jnp.asarray(view.p)[..., 0]
    rho1 = (jnp.asarray(view.rho)[..., 0] if view.rho is not None else jnp.asarray(flux.rhosfc))
    dz1 = jnp.asarray(view.dz)[..., 0]
    qv1 = jnp.asarray(state.qv[0])
    qc1 = jnp.asarray(state.qc[0])
    rad = radiation if radiation is not None else RucRadiation(jnp.zeros(shape), jnp.zeros(shape))
    gsw = jnp.asarray(rad.gsw, dt_)  # NET shortwave absorbed at the ground (WRF GSW)
    precip = ruc_precipitation_forcing(precipitation, float(dt), shape, dt_)
    forcing = dict(
        t3d=t1, qv3d=qv1, qc3d=qc1, p8w=p1, rho3d=rho1, z3d=dz1, glw=rad.glw, gsw=gsw,
        chs=jnp.zeros(shape), flqc=flqc, flhc=flhc, **precip,
    )
    forcing = {k: _flat(jnp.asarray(v, dt_)) for k, v in forcing.items()}
    st = {k: _flat(jnp.asarray(getattr(land, k), dt_)) for k in STATE_SCALARS + STATE_PROFILES}
    static_f = {
        "ivgtyp": _flat(static.ivgtyp).astype(jnp.int32), "isltyp": _flat(static.isltyp).astype(jnp.int32),
        "xland": _flat(static.xland), "xice": _flat(static.xice), "tbot": _flat(static.tbot),
        "shdmin": _flat(static.shdmin), "shdmax": _flat(static.shdmax), "albbck": _flat(static.albbck),
    }
    extra = {}
    if cfg.mosaic_lu == 1:
        extra["landusef"] = _flat(jnp.asarray(static.landusef, dt_))
    if cfg.mosaic_soil == 1:
        extra["soilctop"] = _flat(jnp.asarray(static.soilctop, dt_))
    # The ktau=1 block is applied once at carry seeding (initial_ruc_land), so every
    # scan step is a ktau>1 call (restart-safe, no traced first-step flag).
    out = lsmruc_step(st, forcing, static_f, cfg, static.tables, 2, **extra)
    new = {k: _unflat(out[k], shape) for k in STATE_SCALARS + STATE_PROFILES}
    next_land = RucLandState(**new)

    # ---- 3. land write-back (water keeps the surface layer)
    xland = jnp.asarray(static.xland)
    is_land = (xland - 1.5) < 0.0
    rhosfc = jnp.asarray(flux.rhosfc)
    qx = jnp.maximum(jnp.asarray(view.qv)[..., 0], 0.0)
    hfx_l = jnp.asarray(new["hfx"], rhosfc.dtype)
    qfx_l = jnp.asarray(new["qfx"], rhosfc.dtype)
    theta_flux_l = hfx_l / jnp.maximum(rhosfc * (CP_D * (1.0 + 0.84 * qx)), 1.0e-12)
    qv_flux_l = qfx_l / jnp.maximum(rhosfc, 1.0e-12)
    thx = jnp.asarray(view.theta)[..., 0]
    fltv_l = (1.0 + EP1 * qx) * theta_flux_l + EP1 * thx * qv_flux_l
    if mynn_fltv_wrf_enabled():
        mynn = mynn_driver_surface_fluxes(hfx_l, qfx_l, rhosfc, qx,
                                          jnp.asarray(new["soilt"], rhosfc.dtype), p1,
                                          **_mynn_surface_exner_kwargs(state, rhosfc.dtype))
        theta_flux_l, qv_flux_l, fltv_l = mynn.flt, mynn.flqv, mynn.fltv

    def blend(name, water_value, land_value):
        return _state_surface(state, name, jnp.where(is_land, jnp.asarray(land_value, jnp.asarray(water_value).dtype), water_value))

    updates = {
        "ustar": _state_surface(state, "ustar", flux.ustar),
        "tau_u": _state_surface(state, "tau_u", flux.tau_u),
        "tau_v": _state_surface(state, "tau_v", flux.tau_v),
        "rhosfc": _state_surface(state, "rhosfc", flux.rhosfc),
        "theta_flux": blend("theta_flux", flux.theta_flux, theta_flux_l),
        "qv_flux": blend("qv_flux", flux.qv_flux, qv_flux_l),
        "fltv": blend("fltv", flux.fltv, fltv_l),
        "t_skin": blend("t_skin", state.t_skin, new["soilt"]),
        "roughness_m": blend("roughness_m", state.roughness_m, new["znt"]),
        "mavail": blend("mavail", state.mavail, new["mavail"]),
        "soil_moisture": blend("soil_moisture", state.soil_moisture, new["soilmois"][..., 0]),
    }
    if flux.wspd is not None and getattr(state, "sfc_wspd", None) is not None:
        updates["sfc_wspd"] = _state_surface(state, "sfc_wspd", flux.wspd)
    carried = {"mol": diag.mol, "hfx": (diag.hfx, new["hfx"]), "qfx": (flux.qv_flux * flux.rhosfc, new["qfx"]),
               "qsfc": (diag.qsfc, new["qsfc"])}
    for name, value in carried.items():
        if getattr(state, name, None) is None:
            continue
        if isinstance(value, tuple):
            updates[name] = blend(name, value[0], value[1])
        else:
            updates[name] = _state_surface(state, name, value)
    return state.replace(**updates), next_land


#: WRF Registry.EM_COMMON history ('h') fields RUC owns -> RucLandState attribute
#: (TSK/HFX/QFX/LH/QSFC ride the generic State/surface history path).
RUC_HISTORY_FIELDS: dict[str, str] = {
    "TSLB": "tso", "SMOIS": "soilmois", "SH2O": "sh2o", "SOILT1": "soilt1", "SNOW": "snow",
    "SNOWH": "snowh", "SNOWC": "snowc", "CANWAT": "canwat", "SFROFF": "sfcrunoff",
    "UDROFF": "udrunoff", "ACRUNOFF": "acrunoff", "ACSNOM": "snom", "SNOWFALLAC": "snowfallac",
    "RHOSNF": "rhosnf", "GRDFLX": "grdflx", "LAI": "lai", "ALBEDO": "alb", "EMISS": "emiss",
    "SNOALB": "snoalb",
}
#: restart-only ('r') RUC fields (also carried exactly by the RucLandState pytree group)
RUC_RESTART_ONLY_FIELDS: dict[str, str] = {
    "SMFR3D": "smfr3d", "KEEPFR3DFLAG": "keepfr3dflag", "TSNAV": "tsnav", "QVG": "qvg",
    "QSG": "qsg", "QCG": "qcg", "DEW": "dew", "MAVAIL": "mavail", "SMSTAV": "smavail",
    "SMSTOT": "smmax", "SFCEVP": "sfcevp", "SFCEXC": "sfcexc", "ACSNOW": "acsnow",
    "ZNT": "znt", "QSFC": "qsfc",
}


def ruc_history_fields(land: RucLandState) -> dict:
    """WRF-named history fields of the RUC carry; soil fields as ``(nzs, ny, nx)``.

    Units follow WRF LSMRUC: SNOW/CANWAT kg m-2, SNOWH m, SFROFF/UDROFF/ACRUNOFF/ACSNOM/
    SNOWFALLAC mm (accumulated since ktau=1), RHOSNF kg m-3.  Not yet consumed by the
    nested-pipeline writer (sf=3 runs through the explicit-bundle scan API, like sf=1/7).
    """

    out = {}
    for wrf_name, attr in RUC_HISTORY_FIELDS.items():
        a = np.asarray(getattr(land, attr))
        out[wrf_name] = np.moveaxis(a, -1, 0) if a.ndim == 3 else a
    return out


def ruc_unsupported_mask(land: RucLandState, static: RucStaticBundle) -> np.ndarray:
    """Host-side admission: columns the RUC port cannot run (fail closed)."""

    cfg = static.cfg
    xland = np.asarray(static.xland)
    xice = np.asarray(static.xice)
    iforest = np.asarray(static.tables.ifortbl)[np.clip(np.asarray(static.ivgtyp), 1, len(static.tables.ifortbl)) - 1]
    zs = cfg.zs

    def first(th):
        for k in range(2, cfg.nzs + 1):
            if zs[k - 1] >= th:
                return k
        return 4

    nroot = np.where(iforest > 2, first(0.4), first(1.1))
    land_pts = xland < 1.5
    bad = land_pts & (xice >= cfg.xice_threshold)
    if cfg.oob_root_policy == "fail":
        bad |= land_pts & (nroot + 1 > cfg.nzs)
    return bad


def initial_ruc_land(*, cfg: RucConfig, tables: RucTables, ivgtyp, isltyp, xland, xice,
                     tsk, tslb, smois, tmn, vegfra, albbck, snoalb, p1, qc1, snow=None,
                     snowh=None, snowc=None, canwat=None, lai=None, emiss=None, sh2o=None,
                     smfr3d=None, restart: bool = False) -> RucLandState:
    """Seed the RUC carry like WRF RUCLSMINIT + the LSMRUC ``ktau=1`` block.

    ``tslb``/``smois`` are on the RUC soil levels ``cfg.zs`` (``(ny, nx, nzs)``).
    ``p1``/``qc1`` are the level-1 hydrostatic pressure and cloud water of the first
    LSM call (the ``ktau=1`` initialisation of QSG/QVG/QCG).
    """

    dt_ = cfg.np_dtype
    shape = np.shape(tsk)
    n = int(np.prod(shape))
    nzs = cfg.nzs

    def f(x, default=0.0):
        if x is None:
            return np.full(n, default, dtype=dt_)
        return np.asarray(x, dtype=dt_).reshape(n)

    def p(x):
        return np.asarray(x, dtype=dt_).reshape(n, nzs)

    ivg = np.asarray(ivgtyp, np.int32).reshape(n)
    isl = np.asarray(isltyp, np.int32).reshape(n)
    ini = ruclsminit(tables, tslb=p(tslb), smois=p(smois), isltyp=isl, ivgtyp=ivg,
                     xice=f(xice), dtype=dt_)
    tso = p(tslb)
    soilt = f(tsk)
    snowc_v = f(snowc)
    if sh2o is not None and restart:
        sh2o_v, smfr_v = p(sh2o), p(smfr3d)
    else:
        sh2o_v, smfr_v = np.asarray(ini["sh2o"]), np.asarray(ini["smfr3d"])
    mavail = np.asarray(ini["mavail"])
    tbq = tbq_table(dt_)
    patmb = f(p1) * dt_.type(1.0e-2)
    qsg = np.asarray(qsn(jnp.asarray(soilt), tbq)) / patmb
    qvg = qsg * mavail
    soilt1 = np.where(snowc_v > 0.0, dt_.type(0.5) * (soilt + tso[:, 0]), tso[:, 0])
    tsnav = dt_.type(0.5) * (soilt + tso[:, 0]) - dt_.type(273.15)
    z = np.zeros(n, dt_)
    zero_p = np.zeros((n, nzs), dt_)
    lemi = np.asarray(tables.lemitbl, dt_)[np.clip(ivg, 1, len(tables.lemitbl)) - 1]
    fields = dict(
        soilt=soilt, soilt1=soilt1, tsnav=tsnav, snow=f(snow), snowh=f(snowh), snowc=snowc_v,
        canwat=f(canwat), alb=f(albbck, 0.18), emiss=f(emiss) if emiss is not None else lemi,
        znt=np.asarray(ini["znt"]), z0=np.asarray(ini["znt"]), lai=f(lai), mavail=mavail,
        vegfra=f(vegfra), snoalb=f(snoalb, 0.75), qvg=qvg, qsg=qsg, qcg=z, dew=z,
        qsfc=qvg / (dt_.type(1.0) + qvg), chklowq=np.ones(n, dt_), hfx=z, qfx=z, lh=z,
        grdflx=z, sfcrunoff=z, udrunoff=z, acrunoff=z, sfcexc=z, sfcevp=z, smavail=z,
        smmax=z, snowfallac=z, acsnow=z, snom=z, rhosnf=np.full(n, -1.0e3, dt_),
        precipfr=z, tso=tso, soilmois=p(smois), sh2o=sh2o_v, smfr3d=smfr_v,
        keepfr3dflag=zero_p,
    )
    del qc1  # wrfinput QCG=0 lies inside [0, 0.1], so LSMRUC keeps it (ktau=1 block :505)
    fields["soilt1"] = np.where((soilt1 < 170.0) | (soilt1 > 400.0), tso[:, 0], soilt1)
    out = {}
    for k, v in fields.items():
        v = np.asarray(v, dt_)
        out[k] = jnp.asarray(v.reshape(shape + v.shape[1:]))
    return RucLandState(**out)


def build_ruc_bundles(fields, *, mminlu: str, iswater: int, isice: int, dt: float, zs,
                      p1, frpcpn: bool = True, dtype: str = "float32", mosaic_lu: int = 0,
                      mosaic_soil: int = 0, run_dir=None):
    """RucStaticBundle + seeded RucLandState from wrfinput-style 2-D/3-D fields.

    ``fields`` maps WRF names to arrays: IVGTYP, ISLTYP, XLAND, SEAICE (XICE), TMN,
    SHDMIN, SHDMAX, ALBBCK, VEGFRA, TSK, TSLB/SMOIS ``(nzs, ny, nx)`` on the RUC levels
    ``zs`` (WRF ZS), optional SNOW, SNOWH, SNOWC, CANWAT, LAI, SNOALB, EMISS, LANDUSEF,
    SOILCTOP.  ``p1`` is the level-1 pressure of the first LSM call (QSG/QVG init).
    """

    from gpuwrf.physics.ruclsm import load_ruc_tables, mminlu_to_ruc

    tables = load_ruc_tables(mminlu_to_ruc(mminlu), run_dir)
    cfg = RucConfig(dt=float(dt), zs=tuple(float(z) for z in zs), dtype=dtype, frpcpn=bool(frpcpn),
                    iswater=int(iswater), isice=int(isice), mosaic_lu=int(mosaic_lu),
                    mosaic_soil=int(mosaic_soil))
    dt_ = cfg.np_dtype

    def g(name, default=None):
        v = fields.get(name)
        if v is None:
            return default
        return np.asarray(v)

    shape = np.shape(g("TSK"))
    zero = np.zeros(shape, dt_)
    xice = g("SEAICE", g("XICE", zero))

    def k_last(a):
        return np.moveaxis(np.asarray(a, dt_), 0, -1)

    static = RucStaticBundle(
        cfg=cfg, tables=tables,
        ivgtyp=jnp.asarray(g("IVGTYP"), jnp.int32), isltyp=jnp.asarray(g("ISLTYP"), jnp.int32),
        xland=jnp.asarray(g("XLAND"), dt_), xice=jnp.asarray(xice, dt_), tbot=jnp.asarray(g("TMN"), dt_),
        shdmin=jnp.asarray(g("SHDMIN"), dt_), shdmax=jnp.asarray(g("SHDMAX"), dt_),
        albbck=jnp.asarray(g("ALBBCK"), dt_),
        landusef=None if mosaic_lu != 1 else jnp.asarray(k_last(g("LANDUSEF"))),
        soilctop=None if mosaic_soil != 1 else jnp.asarray(k_last(g("SOILCTOP"))),
    )
    land = initial_ruc_land(
        cfg=cfg, tables=tables, ivgtyp=g("IVGTYP"), isltyp=g("ISLTYP"), xland=g("XLAND"),
        xice=xice, tsk=g("TSK"), tslb=k_last(g("TSLB")), smois=k_last(g("SMOIS")),
        tmn=g("TMN"), vegfra=g("VEGFRA"), albbck=g("ALBBCK"), snoalb=g("SNOALB"), p1=p1,
        qc1=None, snow=g("SNOW"), snowh=g("SNOWH"), snowc=g("SNOWC"), canwat=g("CANWAT"),
        lai=g("LAI"), emiss=None,
    )
    return static, land


__all__ = [
    "RUC_HISTORY_FIELDS",
    "RUC_RESTART_ONLY_FIELDS",
    "build_ruc_bundles",
    "ruc_history_fields",
    "RucLandState",
    "RucRadiation",
    "RucStaticBundle",
    "initial_ruc_land",
    "ruc_net_radiation",
    "ruc_precipitation_forcing",
    "ruc_surface_step",
    "ruc_unsupported_mask",
]
