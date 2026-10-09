"""Native REAL32 whole-MYNN entry behind the retained column interface.

Bulk equations and coefficient builders reuse mynn_pbl. Plume recurrences
and BouLac parcel searches remain within Pallas kernels; the fp32 tridiagonal
primitive is already a device-resident vendor PCR solver. No dense BouLac
matrices remain, so the native path processes the full batch without tiling.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from gpuwrf.physics import mynn_pbl as P

def _native_edmf_arrays(state, flux, fltv, pblh, dt, dx, *, qc_bl=None, cldfra_bl=None):
    """Build the MYNN-EDMF mass-flux solver arrays from the column state.

    Calls the WRF-faithful :func:`mynn_edmf.dmp_mf_columns` (verified against the
    pristine WRF ``DMP_mf`` to <0.5% rel error, see ``proofs/mynn_edmf``). The
    standalone MYNN column can carry ``qv``/``qc``/``qi``/``theta``; the liquid
    potential temperature and total water arrays passed here mirror WRF's
    moisture-aware MYNN path. The surface flux struct supplies the kinematic
    fluxes WRF's main MYNN derives as
    ``flqv=qfx/rho`` (=qv_flux), ``flt=hfx/(rho*cpm)`` (=theta_flux), ``flq=flqv``.

    WRF computes ``th_sfc = ts/ex1(kts)`` in the MYNN driver and passes that
    potential temperature as DMP_mf's misleadingly named ``ts`` argument.  The
    operational surface contract supplies real ``State.t_skin``; standalone
    analytic fixtures retain the ``ts<=0`` fallback implemented by callers that
    do not provide a skin temperature. ``dx`` is the grid spacing (m).
    """
    from gpuwrf.kernels.phys_mynn_plume import dmp_mf_columns_native

    sqv, sqc, sqi, sqw = P._specific_moisture_components(state)
    theta = state.theta
    thl = P._liquid_potential_temperature(state)
    thv = theta * (1.0 + P.mynn_constant("P608", state.theta.dtype) * sqv)
    exner = (state.p / 100000.0) ** (287.0 / (3.5 * 287.0)) if state.exner is None else state.exner

    # kinematic surface fluxes (already in flux struct)
    flqv = flux.qv_flux
    flq = flqv
    flt = flux.theta_flux
    ts = jnp.asarray(flux.t_skin, dtype=fltv.dtype)
    ts = jnp.broadcast_to(ts, fltv.shape) / exner[..., 0]

    zw = jnp.concatenate(
        (P._zero_edge_like(state.dz), jnp.cumsum(state.dz, axis=-1)), axis=-1
    )
    xland = jnp.broadcast_to(
        jnp.asarray(flux.xland, dtype=state.theta.dtype), fltv.shape
    )

    return dmp_mf_columns_native(
        sqw, sqv, sqc, state.u, state.v, state.w, theta, thl, thv, state.theta * 0.0,
        2.0 * state.tke, state.p, exner, state.rho, state.dz, zw,
        ust=flux.ustar, flt=flt, fltv=fltv, flq=flq, flqv=flqv,
        pblh=pblh, ts=ts, dx=dx, xland=xland, dt=dt,
        psig_shcu=P._edmf_scale_aware_psig(dx, pblh),
        **P._edmf_cloud_base_kwargs(state, pblh, qc_bl, cldfra_bl),
        **P._ni_plume_kwargs(state),
        interpret=jax.default_backend() == "cpu",
    )

def _advance_native(state: P.MynnPBLColumnState, dt: float, debug: bool,
                                  surface=None, edmf: bool = False, dx: float = 1000.0):
    """Unjitted implementation; returns the advanced state and the MYNN PBLH.

    ``edmf`` activates the MYNN-EDMF mass-flux nonlocal scalar transport (the
    ``s_awqv``/``s_awthl`` updraft flux) in the theta/qv solves. ``dx`` is the
    horizontal grid spacing (m) the mass-flux plume sizing needs.

    v0.15 SGS-cloud chain (WRF driver order, module_bl_mynnedmf.F:900-1100):
    GET_PBLH -> mym_condensation(CASE 2, sigma from the prognosed ``qsq``) ->
    DMP_mf (+ shallow-cu cldfra/qc_bl overwrite) -> SGS-aware ``thlv`` ->
    mym_turbulence -> mym_predict (qke + closure-2.6 ``qsq``) -> tendencies.
    ``GPUWRF_MYNN_SGS_CLOUD=0`` rolls back to the pre-v0.15 dry-buoyancy path.
    """

    state = P._clip_state(state)
    flux, wind, fltv, rhosfc = P._surface_terms(state, surface)
    qke = 2.0 * state.tke
    sgs_cloud = P._MYNN_SGS_CLOUD
    mf = None
    if sgs_cloud:
        # WRF GET_PBLH runs before mym_condensation; the same value feeds
        # DMP_mf (bitwise-identical to the previous turb["pblh"] EDMF input).
        pblh0 = P._get_pblh(state, qke, flux.xland)
        sqv, sqc, sqi, sqw = P._specific_moisture_components(state)
        exner = P._column_exner(state)
        qc_bl, qi_bl, cldfra_bl = P.mym_condensation_cloudpdf2(
            theta=state.theta,
            p=state.p,
            exner=exner,
            dz=state.dz,
            qw=sqw,
            qc=sqc,
            qi=sqi,
            qs=P._specific_snow_content(state),
            qsq=state.qsq,
            pblh=pblh0,
        )
        if edmf:
            mf = _native_edmf_arrays(state, flux, fltv, pblh0, dt, dx,
                **P._edmf_fresh_cloud_kwargs(qc_bl, cldfra_bl))
            qc_bl, cldfra_bl = P.dmp_shallow_cu_overwrite(
                qc_bl=qc_bl,
                cldfra_bl=cldfra_bl,
                edmf_a=mf["edmf_a"],
                edmf_qc=mf["edmf_qc"],
                edmf_qt=mf["edmf_qt"],
                theta=state.theta,
                thl=P._liquid_potential_temperature(state),
                qw=sqw,
                p=state.p,
                exner=exner,
                dz=state.dz,
                xland=jnp.broadcast_to(
                    jnp.asarray(flux.xland, dtype=state.theta.dtype), fltv.shape
                ),
            )
        # The freshly diagnosed SGS cloud enters the buoyancy (thlv) used by
        # mym_level2/mym_turbulence below (the WRF thlv1 rebuild).
        state = state.replace(qc_bl=qc_bl, qi_bl=qi_bl, cldfra_bl=cldfra_bl)
    if P._MYNN_ELB_MF and edmf and mf is None:
        mf = _native_edmf_arrays(
            state, flux, fltv, P._get_pblh(state, qke, flux.xland), dt, dx
        )
    turb = P._mym_turbulence(state, qke, fltv, flux.ustar, dx, flux.xland,
                             **P._mynn_length_mf_kwargs(state, mf),
                             **P._mynn_wrf_turbulence_kwargs(state, mf))
    qke_new, _qwt, qdiss, _pdk = P._mym_predict_qke(state, qke, turb, dt, flux.ustar, flux,
                                               **P._mym_predict_mf_kwargs(mf))
    if sgs_cloud:
        qsq_new = P._mym_predict_qsq(
            state, state.qsq, turb, dt, s_aw=mf["s_aw"] if mf is not None else None
        )
    else:
        qsq_new = state.qsq
    if edmf and mf is None:
        mf = _native_edmf_arrays(state, flux, fltv, turb["pblh"], dt, dx)
    means = P._apply_mean_tendencies_with_clouds(state, turb, dt, flux, wind, rhosfc, mf=mf,
        **P._mynn_dheat_kwargs(qke_new, turb["el"], state.p))
    u, v, theta, qv, qc, qi = means[:6]
    ni_update = {} if state.ni is None else {'ni':means[6]}
    km, kh = P._retrieve_exchange_coeffs(state, turb)
    tke = 0.5 * qke_new

    u = P.assert_finite(u, "mynn_u", enabled=debug)
    v = P.assert_finite(v, "mynn_v", enabled=debug)
    theta = P.assert_finite(theta, "mynn_theta", enabled=debug)
    qv = P.assert_physical_bounds(qv, 0.0, 0.1, "mynn_qv", enabled=debug)
    tke = P.assert_physical_bounds(tke, P.TKE_EPS, 200.0, "mynn_tke", enabled=debug)
    km = P.assert_finite(km, "mynn_km", enabled=debug)
    kh = P.assert_finite(kh, "mynn_kh", enabled=debug)
    el = P.assert_finite(turb["el"], "mynn_el", enabled=debug)
    del qdiss
    return (
        state.replace(u=u, v=v, theta=theta, qv=qv, tke=tke, km=km, kh=kh, el=el, qsq=qsq_new, qc=qc, qi=qi,
                      **ni_update,
                      **P._mynn_plume_diagnostics(state, mf)),
        turb["pblh"],
    )

def step_mynn_columns_native(state, dt, *, debug=False, surface=None, edmf=False, dx=1000.0):
    """Return (state, pblh) with native fp32 leaves and the retained signature."""
    with jax.enable_x64(False):
        state = jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float32), state)
        surface = jax.tree_util.tree_map(lambda v: jnp.asarray(v, jnp.float32), surface)
        return _advance_native(state, dt, debug, surface, edmf, dx)
