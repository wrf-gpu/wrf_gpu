"""F7K: definitive WRF-vs-JAX center-column per-substep diff (em_quarter_ss).

Builds a JAX initial condition aligned to the pristine WRF v4.7.1 em_quarter_ss
warm-bubble case (same stratified base sounding + the WRF `delt=3 K` bubble,
dx=2000, ztop=20000, 41-level exponential-stretch eta, dt=12 s, sound_steps=6),
runs ONE operational timestep with per-(rk,iteration) center-column capture, and
diffs against the WRF savepoint records.

The WRF dump (solve_em.F:1638-1674) records, AFTER calc_p_rho at the end of each
acoustic substep, the bubble-center column (i=20,j=20):
  w_2 (coupled w), ph_2, p, rw_tend (frozen/stage), ph_tend (frozen/stage),
  t_2save, muave, muts, mut, a, alpha, gamma, cqw, c2a.

This script captures the JAX analogs from the operational acoustic core and
writes proofs/f7k/wrf_vs_jax_center_diff.json.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _enforce_operational_precision,
    _acoustic_core_state_from_prep,
    _RKStageDescriptor,
)
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry
from gpuwrf.dynamics.core.acoustic import (
    AcousticCoreConfig,
    acoustic_substep_core,
)
from gpuwrf.dynamics.core.advance_w import dry_cqw
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.runtime.operational_mode import _augment_large_step_tendencies

GRAVITY = 9.81
R_D = 287.0
CP = 1004.0
CV = CP - R_D
P0 = 1.0e5
T0 = 300.0
CENTER_I = 20  # 0-indexed center column matches WRF i=20 (1-indexed nxc)

WRF_JSON = "proofs/m9/wrf_em_quarter_ss_savepoints.json"


def _alpha_dry(theta, p):
    theta = np.asarray(theta, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return (R_D / P0) * theta * (p / P0) ** (-CV / CP)


def build_wrf_matched_numpy(*, periodic_3col: bool = True):
    """Build the em_quarter_ss-aligned IC for a JAX periodic slab.

    Uses the WRF base sounding (reconstructed from the savepoint center column by
    subtracting the analytic bubble) broadcast over a small x-slab, with the WRF
    `delt=3 K` bubble re-imposed analytically so a center column (i=CENTER_I)
    matches WRF i=20.  The geopotential/pressure are hydrostatically rebalanced
    at fixed dry mass exactly as WRF module_initialize_ideal (quarter_ss).
    """

    d = json.load(open(WRF_JSON))
    ic = d["initial_condition"]
    PHB = np.array(ic["PHB"], dtype=np.float64)  # (41,) base geopotential face
    PH = np.array(ic["PH"], dtype=np.float64)    # (41,) perturbation ph' face
    PB = np.array(ic["PB"], dtype=np.float64)    # (40,) base pressure mass
    Pp = np.array(ic["P"], dtype=np.float64)     # (40,) pert pressure mass
    Tic = np.array(ic["T"], dtype=np.float64)    # (40,) theta-300 (base+bubble)
    MUB = float(ic["MUB"])
    MU = float(ic["MU"])  # perturbation dry mass (~ -449, the bubble rebalance)

    dx = 2000.0
    dy = 2000.0
    ztop = 20000.0
    kde = 41
    nz = kde - 1  # 40
    z_scale = 8000.0 / ztop

    # WRF exponential-stretch eta (module_initialize_ideal.F:619-622).
    k = np.arange(1, kde + 1)
    znw = (np.exp(-(k - 1) / float(kde - 1) / z_scale) - np.exp(-1.0 / z_scale)) / (
        1.0 - np.exp(-1.0 / z_scale)
    )
    znw[0] = 1.0
    znw[-1] = 0.0
    znu = 0.5 * (znw[:-1] + znw[1:])

    # Base geopotential faces (no bubble) directly from WRF PHB; base pressure
    # mass from WRF PB; p_top from the sigma fit pb = MUB*znu + p_top.
    p_top = float(np.mean(PB - MUB * znu))

    # Base height faces from PHB; mass heights for the bubble formula.
    zface = PHB / GRAVITY
    zmass = 0.5 * (zface[:-1] + zface[1:])

    # Reconstruct the bubble (xrad=yrad=0 at center) and base theta sounding.
    zrad = (zmass - 1500.0) / 1500.0
    bubble_center = np.where(np.abs(zrad) <= 1.0, 3.0 * np.cos(0.5 * np.pi * zrad) ** 2, 0.0)
    theta_base_1d = (Tic + T0) - bubble_center  # (40,) stratified base theta (no bubble)

    # x-slab: smallest domain that keeps the center column away from the periodic
    # seam at low cost.  Use 3 columns so neighbour stencils see the base state;
    # the bubble is centered on the middle column (CENTER_I via tiling later).
    nx = 5
    nxc = nx // 2  # center column index in the slab
    x_m = (np.arange(nx, dtype=np.float64) - nxc) * dx  # xrad measured from center

    # Bubble over the slab (horizontal cosine).  WRF: xrad=dx*(i-nxc)/10000.
    xx, zz = np.meshgrid(x_m, zmass)  # (nz, nx)
    xrad = xx / 10000.0
    zrad2 = (zz - 1500.0) / 1500.0
    rad = np.sqrt(xrad * xrad + zrad2 * zrad2)
    theta_prime = np.where(rad <= 1.0, 3.0 * np.cos(0.5 * np.pi * rad) ** 2, 0.0)  # (nz, nx)
    theta_full = theta_base_1d[:, None] + theta_prime  # (nz, nx) full theta - 0 (absolute)

    base_state = {
        "znw": znw,
        "znu": znu,
        "p_top": p_top,
        "mub": MUB,
        "pb_1d": PB.copy(),
        "phb_face_1d": PHB.copy(),
        "theta_base_1d": theta_base_1d,
        "zmass": zmass,
        "dx": dx,
        "dy": dy,
        "nz": nz,
        "nx": nx,
        "nxc": nxc,
        "ztop": ztop,
    }
    return theta_full, theta_prime, base_state


def _make_grid(base_state, device):
    nz = base_state["nz"]
    nx = base_state["nx"]
    eta_levels = jax.device_put(jnp.asarray(base_state["znw"], dtype=jnp.float64), device)
    top_pressure_pa = float(base_state["p_top"])
    terrain_height = jax.device_put(jnp.zeros((1, nx), dtype=jnp.float64), device)
    projection = Projection("lambert", 0.0, 0.0, base_state["dx"], base_state["dx"], nx, 1)
    terrain = TerrainProvenance(
        source_path="idealized:wrf-em-quarter-ss-aligned",
        sha256="f7k-wrf-match",
        shape=(1, nx),
        units="m",
        projection_transform="flat-xz-slab",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, base_state["ztop"], eta_levels)
    bc = BCMetadata(
        source="ideal",
        fields=("u", "v", "w", "theta", "p", "ph", "mu"),
        update_cadence_h=999,
        interpolation="linear",
        restart_compatible=False,
    )
    metrics = DycoreMetrics.flat(
        ny=1, nx=nx, nz=nz, eta_levels=eta_levels,
        top_pressure_pa=top_pressure_pa, provenance="f7k-wrf-match",
    )
    # Pure sigma (hybrid_opt=0 in the WRF namelist): c1=1, c2=0.
    one_h = jnp.ones((nz,), dtype=jnp.float64)
    zero_h = jnp.zeros((nz,), dtype=jnp.float64)
    one_f = jnp.ones((nz + 1,), dtype=jnp.float64)
    zero_f = jnp.zeros((nz + 1,), dtype=jnp.float64)
    metrics = DycoreMetrics(
        msftx=metrics.msftx, msfty=metrics.msfty, msfux=metrics.msfux, msfuy=metrics.msfuy,
        msfvx=metrics.msfvx, msfvy=metrics.msfvy,
        c1h=one_h, c2h=zero_h, c3h=one_h, c4h=zero_h,
        c1f=one_f, c2f=zero_f, c3f=one_f, c4f=zero_f,
        dn=metrics.dn, dnw=metrics.dnw, rdn=metrics.rdn, rdnw=metrics.rdnw,
        cf1=metrics.cf1, cf2=metrics.cf2, cf3=metrics.cf3, fnm=metrics.fnm, fnp=metrics.fnp,
        dzdx=metrics.dzdx, dzdy=metrics.dzdy, dzdx_u=metrics.dzdx_u, dzdy_v=metrics.dzdy_v,
        p_top=metrics.p_top, provenance="f7k-wrf-match-pure-sigma",
    )
    return GridSpec(
        projection=projection, terrain=terrain, vertical=vertical, bc=bc,
        eta_levels=eta_levels, terrain_height=terrain_height, metrics=metrics,
        halo_width=2, staggering="c-grid",
    )


def _make_state(theta_full, base_state, grid, device):
    nz = base_state["nz"]
    nx = base_state["nx"]
    znw = base_state["znw"]
    mub = base_state["mub"]
    p_top = base_state["p_top"]
    pb_1d = base_state["pb_1d"]
    phb_face_1d = base_state["phb_face_1d"]

    shapes = _state_field_shapes(grid)
    fields = {name: jax.device_put(jnp.zeros(shape, dtype=jnp.float64), device) for name, shape in shapes.items()}

    # Pure-sigma fixed-mass hydrostatic rebalance (module_initialize_ideal quarter_ss).
    wrf_dnw = znw[1:] - znw[:-1]  # (nz,) signed (negative)
    pb_col = np.broadcast_to(pb_1d[:, None], (nz, nx)).copy()
    # alt_full = EOS(theta_full, pb); alb = EOS(theta_base) per column? WRF uses
    # base alb from the BASE column (theta_base) at pb.
    theta_base_col = np.broadcast_to(base_state["theta_base_1d"][:, None], (nz, nx)).copy()
    alt_full = _alpha_dry(theta_full, pb_col)
    alb = _alpha_dry(theta_base_col, pb_col)
    al = alt_full - alb  # grid%al

    # Base geopotential phb from WRF PHB directly (broadcast).
    phb_col = np.broadcast_to(phb_face_1d[:, None], (nz + 1, nx)).copy()
    # ph' integrated at fixed mass: ph'(k+1)=ph'(k) - dnw(k)*mub*al(k); ph'(1)=0.
    ph_pert_col = np.zeros((nz + 1, nx), dtype=np.float64)
    for k in range(nz):
        ph_pert_col[k + 1, :] = ph_pert_col[k, :] - wrf_dnw[k] * mub * al[k, :]
    ph_total = (phb_col + ph_pert_col)[:, None, :]
    ph_pert = ph_pert_col[:, None, :]

    # Diagnostic p' from rebalanced ph' (start_em recompute), like idealized.py.
    wrf_rdnw = 1.0 / wrf_dnw
    al_diag = -(wrf_rdnw[:, None] * (ph_pert_col[1:, :] - ph_pert_col[:-1, :])) / mub
    alt_diag = al_diag + alb
    cpovcv = CP / CV
    p_full = P0 * ((R_D * theta_full) / (P0 * alt_diag)) ** cpovcv
    p_pert = (p_full - pb_col)[:, None, :]
    p_total_field = pb_col[:, None, :] + p_pert

    mu_total = np.ones((1, nx), dtype=np.float64) * mub  # fixed mass: mu'=0
    theta_field = theta_full[:, None, :]

    fields.update({
        "theta": jax.device_put(jnp.asarray(theta_field), device),
        "p": jax.device_put(jnp.asarray(p_total_field), device),
        "p_total": jax.device_put(jnp.asarray(p_total_field), device),
        "p_perturbation": jax.device_put(jnp.asarray(p_pert), device),
        "ph": jax.device_put(jnp.asarray(ph_total), device),
        "ph_total": jax.device_put(jnp.asarray(ph_total), device),
        "ph_perturbation": jax.device_put(jnp.asarray(ph_pert), device),
        "mu": jax.device_put(jnp.asarray(mu_total), device),
        "mu_total": jax.device_put(jnp.asarray(mu_total), device),
        "mu_perturbation": jax.device_put(jnp.zeros((1, nx), dtype=jnp.float64), device),
        "Ni": jax.device_put(jnp.ones((nz, 1, nx), dtype=jnp.float64) * 1.0e5, device),
        "Nr": jax.device_put(jnp.ones((nz, 1, nx), dtype=jnp.float64) * 1.0e5, device),
        "xland": jax.device_put(jnp.ones((1, nx), dtype=jnp.float64), device),
        "mavail": jax.device_put(jnp.ones((1, nx), dtype=jnp.float64) * 0.2, device),
        "roughness_m": jax.device_put(jnp.ones((1, nx), dtype=jnp.float64) * 0.05, device),
        "t_skin": jax.device_put(jnp.ones((1, nx), dtype=jnp.float64) * T0, device),
        "rhosfc": jax.device_put(jnp.ones((1, nx), dtype=jnp.float64), device),
    })
    return State(**fields)


def _make_namelist(grid, device, *, dt_s, sound_steps, epssm):
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    tend = Tendencies(
        u=jnp.zeros((nz, ny, nx + 1)), v=jnp.zeros((nz, ny + 1, nx)),
        w=jnp.zeros((nz + 1, ny, nx)), theta=jnp.zeros((nz, ny, nx)),
        qv=jnp.zeros((nz, ny, nx)), p=jnp.zeros((nz, ny, nx)),
        ph=jnp.zeros((nz + 1, ny, nx)), mu=jnp.zeros((ny, nx)),
    )
    namelist = OperationalNamelist.from_grid(
        grid, tendencies=tend, metrics=grid.metrics, dt_s=float(dt_s),
        acoustic_substeps=int(sound_steps), radiation_cadence_steps=999999,
        use_vertical_solver=True, disable_guards=True, force_fp64=True,
        use_flux_advection=True, const_nu_m2_s=0.0,
    )
    # Match the WRF em_quarter_ss namelist exactly: damp_opt=2 (diffusive), no
    # rigid lid forcing beyond WRF's; epssm=0.1.  We keep top_lid as WRF (the
    # quarter_ss case is NOT top_lid; w(top) is the open upper BC) -> top_lid False.
    namelist = dataclass_replace(
        namelist, run_physics=False, run_boundary=False,
        top_lid=False, w_damping=0, damp_opt=2, dampcoef=0.003, zdamp=5000.0,
        epssm=float(epssm),
    )
    return namelist


def _capture_stage(carry, namelist, stage, wrf_records, records_out, *, rk_label):
    """Run one RK stage substep-by-substep, capturing center-column fields."""
    origin_haloed = apply_halo(carry.state, halo_spec(namelist.grid))
    rk1_reference = origin_haloed

    haloed = apply_halo(carry.state, halo_spec(namelist.grid))
    tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
    tendencies = _augment_large_step_tendencies(haloed, tendencies, namelist, rk_step=int(stage.rk_step))
    candidate = apply_halo(carry.state, halo_spec(namelist.grid))
    prep = small_step_prep_wrf(
        candidate, int(stage.rk_step), float(stage.dt_rk),
        metrics=namelist.metrics, reference_state=rk1_reference, ww=carry.ww,
    )
    pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
    acoustic = _acoustic_core_state_from_prep(
        carry.replace(state=candidate), prep, pressure, namelist, tendencies
    )

    cqw_field = dry_cqw(int(prep.theta_work.shape[0]), int(prep.theta_work.shape[1]),
                        int(prep.theta_work.shape[2]), dtype=prep.theta_work.dtype)
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        prep.mut, namelist.metrics, dt=float(stage.dts_rk), epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid), cqw=cqw_field, c2a=prep.c2a,
    )
    cfg = AcousticCoreConfig(
        dt=float(stage.dts_rk), dx=float(namelist.grid.projection.dx_m),
        dy=float(namelist.grid.projection.dy_m), epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid), w_damping=int(namelist.w_damping),
        damp_opt=int(namelist.damp_opt), dampcoef=float(namelist.dampcoef),
        zdamp=float(namelist.zdamp),
    )

    cc = CENTER_I if CENTER_I < int(prep.theta_work.shape[2]) else int(prep.theta_work.shape[2]) // 2
    nsmall = int(stage.number_of_small_timesteps)
    cur = acoustic
    for it in range(1, nsmall + 1):
        cur = acoustic_substep_core(cur, a=a, alpha=alpha, gamma=gamma, cfg=cfg, cqw=cqw_field)
        rec = {
            "rk_label": rk_label, "rk_step": int(stage.rk_step), "iteration": it,
            "w_2": np.asarray(jax.device_get(cur.w[:, 0, cc])).tolist(),
            "ph_2": np.asarray(jax.device_get(cur.ph[:, 0, cc])).tolist(),
            "p": np.asarray(jax.device_get(cur.p[:, 0, cc])).tolist(),
            "rw_tend": np.asarray(jax.device_get(cur.rw_tend_pg_buoy[:, 0, cc])).tolist()
                if cur.rw_tend_pg_buoy is not None else None,
            "ph_tend": np.asarray(jax.device_get(cur.ph_tend[:, 0, cc])).tolist(),
            "t_2ave": np.asarray(jax.device_get(cur.t_2ave[:, 0, cc])).tolist(),
            "muave": float(jax.device_get(cur.muave[0, cc])),
            "muts": float(jax.device_get(cur.muts[0, cc])),
            "mut": float(jax.device_get(cur.mut[0, cc])),
        }
        records_out.append(rec)
    # Build the next carry from the finished stage to continue the RK sequence.
    from gpuwrf.runtime.operational_mode import _carry_from_finished_stage
    next_carry = _carry_from_finished_stage(carry, prep, cur, namelist)
    return next_carry.replace(state=apply_halo(next_carry.state, halo_spec(namelist.grid)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="proofs/f7k/wrf_vs_jax_center_diff.json")
    args = ap.parse_args()

    device = jax.devices("gpu")[0]
    theta_full, theta_prime, base_state = build_wrf_matched_numpy()
    grid = _make_grid(base_state, device)
    state = _make_state(theta_full, base_state, grid, device)
    # WRF namelist: time_step=12, time_step_sound=6, epssm=0.1.
    namelist = _make_namelist(grid, device, dt_s=12.0, sound_steps=6, epssm=0.1)

    state = _enforce_operational_precision(state, force_fp64=True)
    carry = initial_operational_carry(state)

    dt = 12.0
    sound = 6
    stages = (
        _RKStageDescriptor(1, dt / 3.0, dt / float(sound), 1),
        _RKStageDescriptor(2, 0.5 * dt, dt / float(sound), max(1, sound // 2)),
        _RKStageDescriptor(3, dt, dt / float(sound), sound),
    )

    d_wrf = json.load(open(WRF_JSON))
    wrf_records = [r for r in d_wrf["records"] if r["itimestep"] == 1]

    jax_records: list[dict[str, Any]] = []
    rk_labels = {1: "rk1", 2: "rk2", 3: "rk3"}
    for st in stages:
        carry = _capture_stage(carry, namelist, st, wrf_records, jax_records, rk_label=rk_labels[st.rk_step])

    # Diff: align WRF records (itimestep=1) to JAX records by (rk_step, iteration).
    def key(r):
        return (int(r["rk_step"]), int(r["iteration"]))

    wrf_by_key = {key(r): r for r in wrf_records}
    jax_by_key = {key(r): r for r in jax_records}

    diff_rows = []
    fields_phys = ["w_2", "ph_2", "p", "rw_tend", "ph_tend", "muave", "muts", "mut"]
    for k in sorted(set(wrf_by_key) & set(jax_by_key)):
        wr = wrf_by_key[k]
        jr = jax_by_key[k]
        row = {"rk_step": k[0], "iteration": k[1]}
        for f in fields_phys:
            wv = wr.get(f)
            jv = jr.get(f)
            if wv is None or jv is None:
                row[f] = None
                continue
            wa = np.atleast_1d(np.array(wv, dtype=np.float64))
            ja = np.atleast_1d(np.array(jv, dtype=np.float64))
            n = min(wa.size, ja.size)
            wa, ja = wa[:n], ja[:n]
            denom = max(np.max(np.abs(wa)), 1e-30)
            row[f] = {
                "wrf_max_abs": float(np.max(np.abs(wa))),
                "jax_max_abs": float(np.max(np.abs(ja))),
                "max_abs_diff": float(np.max(np.abs(wa - ja))),
                "rel_max_diff": float(np.max(np.abs(wa - ja)) / denom),
                "corr": float(np.corrcoef(wa, ja)[0, 1]) if n > 1 and np.std(wa) > 0 and np.std(ja) > 0 else None,
            }
        diff_rows.append(row)

    out = {
        "schema": "f7k_wrf_vs_jax_center_diff",
        "wrf_source": d_wrf.get("source"),
        "wrf_git_commit": d_wrf.get("git_commit"),
        "note": "JAX em_quarter_ss-aligned IC (WRF stratified base + delt=3K bubble, "
                "dx=2000, ztop=20000, 41-lvl stretch eta, dt=12, sound=6, epssm=0.1). "
                "Per-(rk,iteration) center-column diff vs WRF savepoint itimestep=1. "
                "NOTE: WRF case is 3D open-BC; JAX slab is periodic 5-col -- expect "
                "exact match on stage forcings, drift on advection-coupled fields.",
        "center_column": {"jax_index": CENTER_I if CENTER_I < base_state["nx"] else base_state["nx"] // 2,
                          "wrf_index": "i=20,j=20"},
        "jax_records": jax_records,
        "diff": diff_rows,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {outp}")
    # Console summary.
    for row in diff_rows:
        parts = [f"rk{row['rk_step']}/it{row['iteration']}"]
        for f in ["w_2", "ph_2", "p", "rw_tend", "ph_tend", "muts"]:
            c = row.get(f)
            if isinstance(c, dict):
                parts.append(f"{f}:rel={c['rel_max_diff']:.2e}(W{c['wrf_max_abs']:.2e}/J{c['jax_max_abs']:.2e})")
        print("  ".join(parts))


if __name__ == "__main__":
    main()
