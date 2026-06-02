"""v0.4.0 S5 — GPU body for the native-init -> forecast gate (the standalone proof).

This is the GPU-bound executor that ``comparator.run_forecast_gate(execute=True)``
delegates to. It is the SINGLE-GPU serialization point the S4 scaffold reserves
for S5/the manager. It does NOT rebuild the dycore: it glues the assembled
native real-init product into the SAME operational forecast entry the validated
d02/d03 replay path uses (``run_forecast_operational_segmented``), changing ONLY
the IC + LBC source from a CPU-WRF replay to the NATIVE init product.

Pipeline (per case, d01):
  1. ``driver.build_real_init`` (via the S5 integrated factory) -> RealInitProduct
     (native dynamics + base state + surface + soil + the native LateralBC).
  2. Pack the IC ``State`` + ``BaseState`` purely from the native product.
  3. Build the operational ``*_bdy`` lateral-boundary leaves by DECOUPLING the
     native LateralBC (wrfbdy-equiv) back to the decoupled raw fields the
     operational ``apply_lateral_boundaries`` adapter consumes. The NATIVE wrfbdy
     is the ONLY LBC source -- NO CPU-WRF replay.
  4. Reuse the VALIDATED static grid geometry (map factors / hybrid-eta / Coriolis
     metrics) via ``load_wrfinput_metrics`` on the reference t0 wrfout. These are
     STATIC geometry the native-init parity gate already proves native reproduces
     within the frozen WRFINPUT_TOLS (MAPFAC/C1*/F/E ... all PASS); sourcing them
     from the file is the exact, non-dynamical path and keeps the dycore contract
     byte-identical to the replay path. The IC dynamics + base state + LBC -- the
     things the standalone-init claim is ABOUT -- are 100% native.
  5. Run the operational forecast for the lead (segmented entry), write per-lead
     wrfout, and score vs the CPU-WRF wrfout reference using the SAME
     continuous-gate metric set (T2/U10/V10 core blocking; PSFC/PBLH/Q2 diag).

Honest scope: this is a foundation STABILITY + early-lead MATCH smoke. It reports
the per-lead finite/physical check (the classic native-init hour-0 imbalance ->
blowup signature shows in the first hours) and the core-field deltas vs CPU-WRF.
NO masking / clamping of a blowup or a bias -- if it blows up or mismatches, that
is reported as the verdict.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Decouple the native LateralBC -> operational *_bdy leaves.
# ---------------------------------------------------------------------------
# The native LateralBC stores WRF wrfbdy COUPLED values (mass-weighted) per side:
#   t  = thm * (c1h*total_mass + c2h)
#   u  = u   * (c1h*mass_u + c2h) / msfuy
#   v  = v   * (c1h*mass_v + c2h) / msfvx
#   ph = ph  * (c1f*total_mass + c2f)
#   qv = qv  * (c1h*total_mass + c2h)
#   mu = mu  (uncoupled MU_2)
# The operational State.*_bdy leaves are DECOUPLED raw fields in layout
#   (time, side, bdy_width, z, side_len).
# We decouple by dividing the coupled side strip by the SAME mass term sampled on
# the matching boundary strip, then pack into the (time, side, bdy_width, z,
# side_len) leaf. theta is offset to FULL theta (+300) to match the operational
# convention (build_replay_case loads T+300). The two leaf time levels are
# t0 = native specified VALUE (decoupled) and t1 = t0 + interval*tendency
# (decoupled with the t1 mass = t0 mass + interval*0; we hold mass at t0 since the
# decoupled tendency over the short smoke lead is dominated by the field change).
_SIDE_ORDER = ("W", "E", "S", "N")  # matches boundary_apply.SIDES / SIDE_INDEX
_VAL_KEYS = {"W": "bxs", "E": "bxe", "S": "bys", "N": "bye"}
_TEND_KEYS = {"W": "btxs", "E": "btxe", "S": "btys", "N": "btye"}

T0_K = 300.0
RVOVRD = 461.6 / 287.0


def _strip_mass_h(mass_h: np.ndarray, side: str, width: int) -> np.ndarray:
    """Slice the (z, ny, nx) mass term onto a side strip (bdy_width, z, side_len).

    Mirrors lateral_bc._stuff_bdy so the decoupling samples the SAME mass cells the
    coupling used. W/E -> tangential y; S/N -> tangential x.
    """
    a = np.asarray(mass_h, dtype=np.float64)
    if side == "W":
        return np.moveaxis(a[:, :, :width], 2, 0)
    if side == "E":
        return np.moveaxis(a[:, :, -width:][:, :, ::-1], 2, 0)
    if side == "S":
        return np.moveaxis(a[:, :width, :], 1, 0)
    return np.moveaxis(a[:, -width:, :][:, ::-1, :], 1, 0)


def _strip_mass_2d(mass_2d: np.ndarray, side: str, width: int) -> np.ndarray:
    a = np.asarray(mass_2d, dtype=np.float64)
    if side == "W":
        return a[:, :width].T
    if side == "E":
        return a[:, -width:][:, ::-1].T
    if side == "S":
        return a[:width, :]
    return a[-width:, :][::-1, :]


def _pack_leaf(per_side: dict[str, np.ndarray], n_time: int, n_side: int,
               bdy_width: int, z_len: int, side_len: int,
               dtype) -> np.ndarray:
    """Pack per-side (time, bdy_width, z, tan) strips into (time, 4, bdy_width, z, side_len)."""
    leaf = np.zeros((n_time, n_side, bdy_width, z_len, side_len), dtype=dtype)
    for s, idx in ((s, i) for i, s in enumerate(_SIDE_ORDER)):
        strip = per_side[s]  # (time, bdy_width, z, tan)
        t, bw, zl, tan = strip.shape
        leaf[:t, idx, :bw, :zl, :tan] = strip
    return leaf


def build_native_boundary_leaves(product, metrics, *, n_lead_time: int) -> dict[str, Any]:
    """Decouple the native LateralBC -> operational State *_bdy leaves.

    Returns a dict of numpy leaves keyed by the State boundary-leaf names. The
    native wrfbdy is the ONLY LBC source. ``n_lead_time`` boundary time levels are
    emitted (value at t0, then t0 + k*interval*tendency for k=1..n_lead_time-1) so
    the operational interpolate_boundary_leaf can advance the forcing across the
    lead from purely-native data.
    """
    lbc = product.lateral_bc
    if lbc is None:
        raise ValueError("native product has no LateralBC; cannot drive a standalone forecast")
    dyn = product.dynamics
    base = product.base
    surf = product.surface
    vc = product.vcoord

    nz = int(dyn.theta.shape[0])
    ny = int(dyn.theta.shape[1])
    nx = int(dyn.theta.shape[2])
    width = int(lbc.spec_bdy_width)
    side_len = max(nx + 1, ny + 1)
    interval = float(lbc.bdyfrq_seconds)

    total_mass = (np.asarray(dyn.mu, dtype=np.float64) + np.asarray(base.mub, dtype=np.float64))
    c1h = np.asarray(vc.c1h, dtype=np.float64)
    c2h = np.asarray(vc.c2h, dtype=np.float64)
    c1f = np.asarray(vc.c1f, dtype=np.float64)
    c2f = np.asarray(vc.c2f, dtype=np.float64)
    mass_h = c1h[:, None, None] * total_mass[None, :, :] + c2h[:, None, None]   # (nz,ny,nx)
    mass_f = c1f[:, None, None] * total_mass[None, :, :] + c2f[:, None, None]   # (nz+1,ny,nx)
    # staggered U/V mass (lateral_bc._staggered_total_mass)
    mass_u_2d = np.empty((ny, nx + 1), dtype=np.float64)
    mass_u_2d[:, 0] = total_mass[:, 0]
    mass_u_2d[:, 1:nx] = 0.5 * (total_mass[:, 1:] + total_mass[:, :-1])
    mass_u_2d[:, nx] = total_mass[:, nx - 1]
    mass_v_2d = np.empty((ny + 1, nx), dtype=np.float64)
    mass_v_2d[0, :] = total_mass[0, :]
    mass_v_2d[1:ny, :] = 0.5 * (total_mass[1:, :] + total_mass[:-1, :])
    mass_v_2d[ny, :] = total_mass[ny - 1, :]
    msfuy = np.asarray(surf.mapfac_uy, dtype=np.float64)
    msfvx = np.asarray(surf.mapfac_vx, dtype=np.float64)
    mass_u = (c1h[:, None, None] * mass_u_2d[None, :, :] + c2h[:, None, None]) / msfuy[None, :, :]
    mass_v = (c1h[:, None, None] * mass_v_2d[None, :, :] + c2h[:, None, None]) / msfvx[None, :, :]

    def decouple_side(name: str, side: str, mass_strip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (value, tendency) decoupled side strips (bdy_width, z, tan)."""
        v = np.asarray(lbc.values[name][_VAL_KEYS[side]], dtype=np.float64)  # (n_int, bw, z, tan)
        t = np.asarray(lbc.tendencies[name][_TEND_KEYS[side]], dtype=np.float64)
        v0 = v[0]  # first interval = specified value (coupled)
        t0 = t[0]
        safe = np.where(np.abs(mass_strip) > 1e-12, mass_strip, 1e-12)
        return v0 / safe, t0 / safe

    def decouple_mu_side(side: str) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(lbc.values["mu"][_VAL_KEYS[side]], dtype=np.float64)  # (n_int, bw, tan)
        t = np.asarray(lbc.tendencies["mu"][_TEND_KEYS[side]], dtype=np.float64)
        return v[0], t[0]  # mu is uncoupled

    # mass strips per side for each coupled field stagger
    leaves: dict[str, np.ndarray] = {}

    def assemble(name_state: str, lbc_name: str, mass_field: np.ndarray, z_dim: int) -> None:
        per_side_v: dict[str, np.ndarray] = {}
        for side in _SIDE_ORDER:
            mstrip = _strip_mass_h(mass_field, side, width) if mass_field.ndim == 3 else None
            v0, t0 = decouple_side(lbc_name, side, mstrip)
            # time levels: t0=value, then value + k*interval*tend
            stack = [v0]
            for k in range(1, n_lead_time):
                stack.append(v0 + k * interval * t0)
            per_side_v[side] = np.stack(stack, axis=0)  # (time, bw, z, tan)
        z_len = per_side_v["W"].shape[2]
        leaves[name_state] = _pack_leaf(per_side_v, n_lead_time, 4, width, z_len, side_len,
                                        np.float64)

    assemble("u_bdy", "u", mass_u, nz)
    assemble("v_bdy", "v", mass_v, nz)
    assemble("qv_bdy", "qv", mass_h, nz)
    assemble("ph_bdy", "ph", mass_f, nz + 1)

    # theta_bdy: the operational State.theta is the FULL DRY theta (T+300; the
    # replay path loads wrfout T and adds 300). The native wrfbdy ``t`` couples
    # WRF's THM (moist theta, use_theta_m=1); decoupling gives the THM
    # perturbation. Convert back to FULL DRY theta consistent with the interior:
    #   theta_full = (thm_pert + 300) / (1 + Rv/Rd * qv) ,
    # using the matching decoupled qv strip. This keeps the forced boundary theta
    # in the SAME convention as state.theta so apply_lateral_boundaries does not
    # inject a moist/dry or perturbation/full mismatch into the ring.
    per_side_th: dict[str, np.ndarray] = {}
    for side in _SIDE_ORDER:
        mh = _strip_mass_h(mass_h, side, width)
        thm_v0, thm_t0 = decouple_side("t", side, mh)
        qv_v0, qv_t0 = decouple_side("qv", side, mh)
        stack = []
        for k in range(n_lead_time):
            thm = thm_v0 + k * interval * thm_t0
            qv_k = np.maximum(qv_v0 + k * interval * qv_t0, 0.0)
            theta_full = (thm + T0_K) / (1.0 + RVOVRD * qv_k)
            stack.append(theta_full)
        per_side_th[side] = np.stack(stack, axis=0)
    leaves["theta_bdy"] = _pack_leaf(per_side_th, n_lead_time, 4, width, nz, side_len,
                                     np.float64)

    # mu (uncoupled) -> (time, 4, bdy_width, 1, side_len)
    per_side_mu: dict[str, np.ndarray] = {}
    for side in _SIDE_ORDER:
        v0, t0 = decouple_mu_side(side)  # (bw, tan)
        stack = [v0]
        for k in range(1, n_lead_time):
            stack.append(v0 + k * interval * t0)
        arr = np.stack(stack, axis=0)  # (time, bw, tan)
        per_side_mu[side] = arr[:, :, None, :]  # (time, bw, 1, tan)
    leaves["mu_bdy"] = _pack_leaf(per_side_mu, n_lead_time, 4, width, 1, side_len, np.float64)

    # w_bdy / p_bdy / pb_bdy / phb_bdy / mub_bdy: native init has no specified-W LBC
    # and the operational apply path only consumes these in the nested
    # force_geopotential=False branch (NOT used here -- d01 is the parent and
    # force_geopotential stays True). Seed them from the native base/IC strips so
    # the leaf shapes are valid; with force_geopotential=True only u/v/w/theta/qv/
    # p/mu rings are forced (ph/phb via the ph_bdy/phb_bdy strips).
    def base_leaf_h(field: np.ndarray, z_len: int) -> np.ndarray:
        per: dict[str, np.ndarray] = {}
        for side in _SIDE_ORDER:
            s = _strip_mass_h(field, side, width)  # (bw, z, tan)
            per[side] = np.broadcast_to(s[None], (n_lead_time,) + s.shape).copy()
        return _pack_leaf(per, n_lead_time, 4, width, z_len, side_len, np.float64)

    def base_leaf_2d(field: np.ndarray) -> np.ndarray:
        per: dict[str, np.ndarray] = {}
        for side in _SIDE_ORDER:
            s = _strip_mass_2d(field, side, width)  # (bw, tan)
            per[side] = np.broadcast_to(s[None, :, None, :],
                                        (n_lead_time, width, 1, s.shape[-1])).copy()
        return _pack_leaf(per, n_lead_time, 4, width, 1, side_len, np.float64)

    leaves["w_bdy"] = base_leaf_h(np.asarray(dyn.w, dtype=np.float64), nz + 1)
    leaves["p_bdy"] = base_leaf_h(np.asarray(dyn.p, dtype=np.float64), nz)
    leaves["pb_bdy"] = base_leaf_h(np.asarray(base.pb, dtype=np.float64), nz)
    leaves["phb_bdy"] = base_leaf_h(np.asarray(base.phb, dtype=np.float64), nz + 1)
    leaves["mub_bdy"] = base_leaf_2d(np.asarray(base.mub, dtype=np.float64))
    return leaves


# ---------------------------------------------------------------------------
# Build the operational State + BaseState + GridSpec from the native product.
# ---------------------------------------------------------------------------
def build_native_forecast_case(product, *, reference_wrfout: Path, run_start: datetime,
                               n_lead_time: int):
    """Assemble (State, GridSpec, BaseState, DycoreMetrics) from the native product.

    The IC dynamics/base/surface/soil and the LBC come 100% from ``product``; the
    static grid geometry (map factors / hybrid-eta / Coriolis) is sourced via the
    exact ``load_wrfinput_metrics`` loader on the reference t0 wrfout (the parity
    gate proves native reproduces these within the frozen tols).
    """
    import jax
    import jax.numpy as jnp
    from gpuwrf.contracts.state import BaseState, State, Tendencies
    from gpuwrf.dynamics.metrics import load_wrfinput_metrics
    from gpuwrf.io.gen2_accessor import Gen2Run

    run_dir = reference_wrfout.parent
    run = Gen2Run(run_dir)
    grid = run.grid("d01").as_grid_spec()
    metrics = load_wrfinput_metrics(reference_wrfout)

    dyn = product.dynamics
    base = product.base
    surf = product.surface
    soil = product.soil

    def dp(a):
        return jax.device_put(jnp.asarray(np.asarray(a, dtype=np.float64)))

    state = State.zeros(grid)
    # WRF base potential temperature t0+t_init: native base columns carry t_init
    # (perturbation theta minus t0). theta_base = t0 + t_init so the dycore's
    # recomputed base inverse density alb matches the native discrete base state.
    theta_base = T0_K + np.asarray(base.t_init, dtype=np.float64)
    pb = np.asarray(base.pb, dtype=np.float64)
    phb = np.asarray(base.phb, dtype=np.float64)
    mub = np.asarray(base.mub, dtype=np.float64)
    p_pert = np.asarray(dyn.p, dtype=np.float64)
    ph_pert = np.asarray(dyn.ph, dtype=np.float64)
    mu_pert = np.asarray(dyn.mu, dtype=np.float64)
    theta_full = np.asarray(dyn.theta, dtype=np.float64) + T0_K  # operational convention (T+300)

    leaves_np = build_native_boundary_leaves(product, metrics, n_lead_time=n_lead_time)
    bdy = {k: dp(v) for k, v in leaves_np.items()}

    zeros2d = jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64)
    # soil_moisture top layer; t_skin from surface TSK; xland/landmask from surface.
    smois_top = np.asarray(soil.smois, dtype=np.float64)
    smois_top = smois_top[0] if smois_top.ndim == 3 else smois_top
    state = state.replace(
        u=dp(dyn.u),
        v=dp(dyn.v),
        w=dp(dyn.w),
        theta=dp(theta_full),
        qv=dp(dyn.qv),
        p_total=dp(pb + p_pert),
        p_perturbation=dp(p_pert),
        ph_total=dp(phb + ph_pert),
        ph_perturbation=dp(ph_pert),
        mu_total=dp(mub + mu_pert),
        mu_perturbation=dp(mu_pert),
        qc=dp(dyn.qc) if dyn.qc is not None else jnp.zeros_like(state.qc),
        qr=dp(dyn.qr) if dyn.qr is not None else jnp.zeros_like(state.qr),
        qi=dp(dyn.qi) if dyn.qi is not None else jnp.zeros_like(state.qi),
        qs=dp(dyn.qs) if dyn.qs is not None else jnp.zeros_like(state.qs),
        qg=dp(dyn.qg) if dyn.qg is not None else jnp.zeros_like(state.qg),
        t_skin=dp(surf.tsk),
        soil_moisture=dp(smois_top),
        xland=dp(surf.xland),
        lakemask=zeros2d,
        mavail=jnp.ones_like(zeros2d),
        roughness_m=jnp.full_like(zeros2d, 0.1),
        lu_index=jnp.asarray(np.asarray(soil.lu_index), dtype=jnp.int32),
        **bdy,
    )
    base_state = BaseState(
        pb=dp(pb),
        phb=dp(phb),
        mub=dp(mub),
        t0=dp(np.full_like(theta_base, T0_K)),
        theta_base=dp(theta_base),
    )
    tendencies = Tendencies.zeros(grid)
    return state, grid, base_state, metrics, tendencies


# ---------------------------------------------------------------------------
# Forecast + scoring orchestration (the GPU-bound gate body).
# ---------------------------------------------------------------------------
def _finite_physical_check(state) -> dict[str, Any]:
    """Per-field finite + gross-physical-range check on a forecast State."""
    import jax
    import numpy as _np
    out: dict[str, Any] = {"all_finite": True, "physical": True, "fields": {}}
    checks = {
        "theta": (150.0, 600.0),     # full theta K
        "qv": (-1e-6, 0.06),         # kg/kg
        "w": (-120.0, 120.0),        # m/s
        "u": (-200.0, 200.0),
        "v": (-200.0, 200.0),
        "mu_total": (1.0, 2.0e5),    # Pa column dry mass
    }
    for name, (lo, hi) in checks.items():
        arr = _np.asarray(jax.device_get(getattr(state, name)), dtype=_np.float64)
        finite = bool(_np.isfinite(arr).all())
        amin = float(_np.nanmin(arr)) if arr.size else None
        amax = float(_np.nanmax(arr)) if arr.size else None
        in_range = bool(finite and amin is not None and amin >= lo and amax <= hi)
        out["fields"][name] = {"finite": finite, "min": amin, "max": amax,
                               "range": [lo, hi], "in_range": in_range}
        out["all_finite"] = out["all_finite"] and finite
        out["physical"] = out["physical"] and in_range
    return out


def run_one_case_forecast_gate(
    product,
    *,
    case_id: str,
    reference_run_dir: Path,
    run_start: datetime,
    init_vt_label: str,
    forecast_hours: int,
    dt_s: float,
    acoustic_substeps: int,
    radiation_cadence_steps: int,
    output_dir: Path,
    core_fields: tuple[str, ...],
    diag_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Run ONE native-init forecast and score it per-lead vs the CPU-WRF wrfout.

    Returns a per-case record: stability (per-lead finite/physical), and the
    per-lead + worst-over-leads core/diag deltas vs CPU-WRF.
    """
    import jax
    from netCDF4 import Dataset

    from gpuwrf.runtime.operational_mode import (
        OperationalNamelist,
        run_forecast_operational_segmented,
    )
    from gpuwrf.coupling.boundary_apply import BoundaryConfig
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf

    # boundary time levels: cover the whole lead from native data (one per lead hr +1)
    n_lead_time = max(2, forecast_hours + 1)
    state, grid, base_state, metrics, tendencies = build_native_forecast_case(
        product, reference_wrfout=reference_run_dir / f"wrfout_d01_{init_vt_label}",
        run_start=run_start, n_lead_time=n_lead_time,
    )

    # Operational d01 namelist with the SAME Sprint-U numerics the replay path uses.
    namelist = OperationalNamelist.from_grid(
        grid,
        tendencies=tendencies,
        metrics=metrics,
        dt_s=float(dt_s),
        acoustic_substeps=int(acoustic_substeps),
        radiation_cadence_steps=int(radiation_cadence_steps),
        boundary_config=BoundaryConfig(update_cadence_s=3600.0, force_geopotential=True),
        use_vertical_solver=True,
        use_flux_advection=True,
        force_fp64=True,
        diff_6th_opt=2,
        diff_6th_factor=0.12,
        w_damping=1,
        damp_opt=3,
        zdamp=5000.0,
        dampcoef=0.2,
        epssm=0.5,
        top_lid=True,
        time_utc=run_start,
    )

    from proofs.m20 import continuous_gate as CG  # the frozen metric + margins

    output_dir.mkdir(parents=True, exist_ok=True)
    surface_fields = tuple(core_fields) + tuple(diag_fields)

    per_lead: list[dict[str, Any]] = []
    stability_records: list[dict[str, Any]] = []
    field_series: dict[str, list[dict[str, Any]]] = {f: [] for f in surface_fields}
    blew_up_at_hour: int | None = None

    cur = state
    import time as _time
    for hour in range(1, int(forecast_hours) + 1):
        t0 = _time.perf_counter()
        cur = run_forecast_operational_segmented(cur, namelist, 1.0)
        jax.block_until_ready(cur.theta)
        wall = _time.perf_counter() - t0
        chk = _finite_physical_check(cur)
        chk["hour"] = hour
        chk["wall_s"] = float(wall)
        stability_records.append(chk)
        if not chk["all_finite"]:
            blew_up_at_hour = hour
            break

        valid_time = run_start + timedelta(hours=hour)
        wrfout = output_dir / f"wrfout_d01_{valid_time:%Y-%m-%d_%H:%M:%S}"
        diagnostics = _surface_diag(cur, namelist, run_start, hour)
        write_wrfout_netcdf(cur, grid, namelist, wrfout, valid_time=valid_time,
                            lead_hours=float(hour), run_start=run_start,
                            diagnostics=diagnostics)

        # score this lead vs CPU-WRF reference
        ref = reference_run_dir / f"wrfout_d01_{valid_time:%Y-%m-%d_%H:%M:%S}"
        lead_fields: dict[str, Any] = {}
        if ref.is_file():
            with Dataset(wrfout) as gd, Dataset(ref) as cd:
                gvars = set(gd.variables)
                for f in surface_fields:
                    if f not in gvars:
                        lead_fields[f] = {"status": "not_in_gpu_artifact"}
                        continue
                    if f not in cd.variables:
                        lead_fields[f] = {"status": "not_in_cpu_corpus"}
                        continue
                    g = CG.read_field(gd, f)
                    c = CG.read_field(cd, f)
                    res = CG.score_field_pair(g, c)
                    lead_fields[f] = res
                    if res.get("status") == "OK":
                        field_series[f].append({"lead": hour, "bias": res["bias"],
                                                "rmse": res["rmse"], "n": res["n"]})
        else:
            lead_fields = {"status": "no_cpu_reference_for_lead"}
        per_lead.append({"hour": hour, "valid_time": valid_time.isoformat(),
                         "fields": lead_fields,
                         "stable": bool(chk["all_finite"] and chk["physical"])})

    # per-field envelope summary (continuous_gate margins)
    field_summary: dict[str, Any] = {}
    core_blocking_pass = True
    for f in surface_fields:
        series = field_series[f]
        margin = CG.REGRESSION_MARGINS.get(f)
        if not series:
            field_summary[f] = {"status": "not_scored", "regression_margin": margin}
            if f in core_fields:
                core_blocking_pass = False
            continue
        biases = np.array([s["bias"] for s in series])
        rmses = np.array([s["rmse"] for s in series])
        worst_abs_bias = float(np.max(np.abs(biases)))
        worst_rmse = float(np.max(rmses))
        within = (worst_abs_bias <= margin) if margin is not None else None
        field_summary[f] = {
            "status": "scored",
            "n_leads": len(series),
            "mean_bias": float(np.mean(biases)),
            "worst_abs_bias": worst_abs_bias,
            "mean_rmse": float(np.mean(rmses)),
            "worst_rmse": worst_rmse,
            "regression_margin": margin,
            "within_margin": within,
            "blocking": f in core_fields,
        }
        if f in core_fields and within is not True:
            core_blocking_pass = False

    stable = blew_up_at_hour is None and all(
        r["all_finite"] for r in stability_records)
    physical = blew_up_at_hour is None and all(
        r["physical"] for r in stability_records)

    return {
        "case_id": case_id,
        "domain": "d01",
        "forecast_hours_attempted": int(forecast_hours),
        "n_leads_run": len(per_lead),
        "dt_s": float(dt_s),
        "acoustic_substeps": int(acoustic_substeps),
        "init_utc": run_start.isoformat(),
        "reference_run_dir": str(reference_run_dir),
        "stability": {
            "stable_finite": bool(stable),
            "physical_range_ok": bool(physical),
            "blew_up_at_hour": blew_up_at_hour,
            "per_hour": stability_records,
        },
        "per_field_summary": field_summary,
        "core_within_margin": bool(core_blocking_pass) if stable else False,
        "per_lead": per_lead,
        "gpu_output_dir": str(output_dir),
    }


def _surface_diag(state, namelist, run_start, hour):
    """Best-effort operational surface diagnostics for the wrfout writer."""
    try:
        import jax
        import numpy as _np
        from dataclasses import replace as _replace
        from gpuwrf.runtime.operational_mode import (
            compute_m9_diagnostics, surface_layer_diagnostics)
        clock = namelist
        if getattr(namelist, "time_utc", None) is None:
            clock = _replace(namelist, time_utc=run_start)
        m9 = compute_m9_diagnostics(state, clock, float(hour) * 3600.0)
        out: dict[str, Any] = {}
        for wrf_name, attr in (("T2", "t2"), ("U10", "u10"), ("V10", "v10"),
                               ("PSFC", "psfc"), ("PBLH", "pblh"), ("TSK", "tsk"),
                               ("SWDOWN", "swdown"), ("GLW", "glw")):
            val = getattr(m9, attr, None)
            if val is not None:
                out[wrf_name] = _np.asarray(jax.device_get(val))
        try:
            q2 = getattr(surface_layer_diagnostics(state, clock.grid), "q2", None)
            if q2 is not None:
                out["Q2"] = _np.asarray(jax.device_get(q2))
        except Exception:
            pass
        return out or None
    except Exception:
        return None

