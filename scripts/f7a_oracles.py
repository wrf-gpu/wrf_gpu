#!/usr/bin/env python
"""F7.A oracles: flat-rest (AC3), nonzero analytic acoustic (AC4), conservation (AC5).

Drives the production acoustic core (``acoustic_substep_core``) directly on a
constructed hydrostatically-balanced rest state, so the proofs exercise the real
operators with no JAX-vs-JAX self-compare and no operational replay machinery.

Run: ``taskset -c 0-3 python scripts/f7a_oracles.py --output-dir proofs/f7a2``
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.dynamics.acoustic_wrf import CPOVCV, calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, AcousticCoreState, acoustic_substep_core
from gpuwrf.dynamics.core.advance_w import dry_cqw

config.update("jax_enable_x64", True)

R_D = 287.0
CP_D = 1004.0
P0 = 100000.0
G = 9.81
T0 = 300.0


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def build_hydrostatic_column(
    *, nz: int, ny: int, nx: int, p_top: float = 1.0e4, p_surface: float = 1.0e5, pure_sigma: bool = True
):
    """Build a flat, dry, constant-theta hydrostatic rest state and its metrics.

    Returns ``(metrics, fields)`` where ``fields`` carries the resident base and
    perturbation arrays needed to assemble an ``AcousticCoreState`` at rest.

    Constant ``theta = T0``.  Mass-coordinate hydrostatic column: dry mass
    ``mu = p_surface - p_top``; pressure on mass levels from the eta coordinate;
    geopotential on faces from the hydrostatic relation
    ``dphi/deta = -mu*alpha`` with ``alpha = (R_d theta / p0) (p/p0)^(cv/cp)``.
    """

    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    metrics = DycoreMetrics.flat(ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=float(p_top))
    # WRF idealized cases default to a PURE SIGMA coordinate (hybrid_opt=0):
    # c1h=c1f=1, c2h=c2f=0, so the dry-mass-weighted column factor
    # (c1*mu+c2) == mu and the top face mass (c1f(kde)*mut+c2f(kde)) is never
    # zero.  DycoreMetrics.flat uses a HYBRID c1f=eta (zero at the lid) with
    # c2f=0, which makes the top face mass vanish; override to pure sigma so the
    # idealized acoustic oracle matches a WRF hybrid_opt=0 column.
    one_h = jnp.ones((nz,), dtype=jnp.float64)
    zero_h = jnp.zeros((nz,), dtype=jnp.float64)
    one_f = jnp.ones((nz + 1,), dtype=jnp.float64)
    zero_f = jnp.zeros((nz + 1,), dtype=jnp.float64)
    if pure_sigma:
        c1h_n, c2h_n, c1f_n, c2f_n = one_h, zero_h, one_f, zero_f
        prov = "analytic-pure-sigma"
    else:
        # WRF hybrid sigma-pressure (hybrid_opt>0): c1 tapers to a constant at
        # the top while c2 carries the pressure offset, so (c1f*mut+c2f) stays
        # nonzero at the lid.  Build a representative hybrid: c1 = eta-blended,
        # c2 = (eta-coord - c1)*(p_surface-p_top).  This exercises the nonzero
        # c2h/c2f terms that the pure-sigma column cannot.
        eta_np2 = np.asarray(eta)
        eta_mass2 = 0.5 * (eta_np2[:-1] + eta_np2[1:])
        # simple hybrid blend B(eta): pure sigma below eta_c, tapering above
        def bcoef(e):
            ec = 0.2
            return np.where(e >= ec, e, (e / ec) ** 2 * ec)
        c1h_np = bcoef(eta_mass2)
        c1f_np = bcoef(eta_np2)
        # c2 = (eta - c1)*(p_surf - p_top) so total mass column reproduces p.
        psp = float(p_surface) - float(p_top)
        c2h_np = (eta_mass2 - c1h_np) * psp
        c2f_np = (eta_np2 - c1f_np) * psp
        c1h_n = jnp.asarray(c1h_np); c2h_n = jnp.asarray(c2h_np)
        c1f_n = jnp.asarray(c1f_np); c2f_n = jnp.asarray(c2f_np)
        prov = "analytic-hybrid-sigma-pressure"
    metrics = DycoreMetrics(
        msftx=metrics.msftx, msfty=metrics.msfty, msfux=metrics.msfux, msfuy=metrics.msfuy,
        msfvx=metrics.msfvx, msfvy=metrics.msfvy,
        c1h=c1h_n, c2h=c2h_n, c3h=c1h_n, c4h=c2h_n,
        c1f=c1f_n, c2f=c2f_n, c3f=c1f_n, c4f=c2f_n,
        dn=metrics.dn, dnw=metrics.dnw, rdn=metrics.rdn, rdnw=metrics.rdnw,
        cf1=metrics.cf1, cf2=metrics.cf2, cf3=metrics.cf3, fnm=metrics.fnm, fnp=metrics.fnp,
        dzdx=metrics.dzdx, dzdy=metrics.dzdy, dzdx_u=metrics.dzdx_u, dzdy_v=metrics.dzdy_v,
        p_top=metrics.p_top, provenance=prov,
    )

    mu = float(p_surface) - float(p_top)  # full dry column perturbation+base mass
    eta_np = np.asarray(eta)
    eta_mass = 0.5 * (eta_np[:-1] + eta_np[1:])  # eta midpoints (mass levels)
    dnw = np.asarray(metrics.dnw)
    c1h_np = np.asarray(metrics.c1h)
    c2h_np = np.asarray(metrics.c2h)
    # Dry hydrostatic full pressure on mass levels: p_full(eta) = eta*mu + p_top
    # holds for both pure-sigma and the hybrid blend used here (c2 carries the
    # offset so the column total still integrates to p).
    p_full_mass = eta_mass * mu + float(p_top)  # (nz,)
    # Inverse density (alpha) from dry EOS: alpha = (R_d theta / p0)*(p/p0)^(cv/cp).
    cvocp = (CP_D - R_D) / CP_D
    alpha_mass = (R_D * T0 / P0) * (p_full_mass / P0) ** (-cvocp)  # alt on mass levels (nz,)
    # Geopotential on faces from the WRF hydrostatic relation (signed metric):
    # phb(k+1) = phb(k) - dnw(k)*(c1h*mut+c2h)*alt  (module_initialize_ideal.F:982),
    # with WRF-signed dnw<0 so -dnw*(...)>0 integrates the column upward.  F7G uses
    # the signed ``dnw`` directly (was abs(dnw)); numerically identical column.
    mass_h = c1h_np * mu + c2h_np
    phi_faces = np.zeros(nz + 1)
    for k in range(nz):
        phi_faces[k + 1] = phi_faces[k] - dnw[k] * mass_h[k] * alpha_mass[k]

    def b3(arr1d):
        return jnp.broadcast_to(jnp.asarray(arr1d, dtype=jnp.float64)[:, None, None], (arr1d.shape[0], ny, nx))

    fields = {
        "mu": mu,
        "p_total_mass": b3(p_full_mass),  # total pressure on mass levels
        "pb_mass": b3(p_full_mass),  # base == total at rest -> perturbation 0
        "alt_mass": b3(alpha_mass),  # base inverse density
        "ph_total_faces": b3(phi_faces),
        "phb_faces": b3(phi_faces),
        "theta_mass": jnp.full((nz, ny, nx), T0, dtype=jnp.float64),
        "eta_mass": eta_mass,
        "p_full_mass_1d": p_full_mass,
        "alpha_mass_1d": alpha_mass,
        "phi_faces_1d": phi_faces,
    }
    return metrics, fields


def rest_acoustic_state(metrics, fields, *, theta_pert_mass=None) -> AcousticCoreState:
    """Assemble an ``AcousticCoreState`` for the rest column (RK1 work arrays).

    All coupled perturbation work arrays are zero at rest.  ``theta_pert_mass``
    optionally injects a perturbation-theta field (mass levels) to drive a
    hydrostatic-adjustment response (AC4); the coupled theta work then becomes
    ``mass_h_cur*(-theta_pert)`` because the RK reference (``theta_1``) stays at
    the unperturbed base while the current ``t_2`` carries the perturbation.
    """

    nz = int(metrics.c1h.shape[0])
    ny = int(metrics.msftx.shape[0])
    nx = int(metrics.msftx.shape[1])
    mu = float(fields["mu"])
    mut = jnp.full((ny, nx), mu, dtype=jnp.float64)  # base dry mass
    zeros_face = jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64)
    zeros_mass = jnp.zeros((nz, ny, nx), dtype=jnp.float64)
    zeros_u = jnp.zeros((nz, ny, nx + 1), dtype=jnp.float64)
    zeros_v = jnp.zeros((nz, ny + 1, nx), dtype=jnp.float64)

    c1h = metrics.c1h
    c2h = metrics.c2h
    mass_h_cur = c1h[:, None, None] * mut[None, :, :] + c2h[:, None, None]

    if theta_pert_mass is None:
        theta_pert = zeros_mass
    else:
        theta_pert = jnp.asarray(theta_pert_mass, dtype=jnp.float64)

    # WRF small_step_prep theta work: t_2 = (c1h*muts+c2h)*t_1 - (c1h*mut+c2h)*t_2.
    # At RK1 with mu_work=0 -> muts=mut; t_1 = reference perturbation theta (0 here),
    # t_2 = current perturbation theta = theta_pert.  So theta_work = -mass_h_cur*theta_pert.
    theta_work = -mass_h_cur * theta_pert
    theta_1 = zeros_mass  # reference perturbation theta (rest)

    # c2a from small_step_prep: cpovcv*(pb+p)/alt with p'=0 at rest.
    c2a = CPOVCV * fields["pb_mass"] / jnp.maximum(jnp.abs(fields["alt_mass"]), 1.0e-12)

    cqw = dry_cqw(nz, ny, nx)

    return AcousticCoreState(
        ww=zeros_face,
        ww_1=zeros_face,
        u=zeros_u,
        u_1=zeros_u,
        v=zeros_v,
        v_1=zeros_v,
        w=zeros_face,
        mu=zeros_mass[0] * 0.0,  # placeholder, replaced below
        mut=mut,
        muave=jnp.zeros((ny, nx), dtype=jnp.float64),
        muts=mut,  # muts = mut + mu_work, mu_work=0 at entry
        muu=jnp.zeros((ny, nx + 1), dtype=jnp.float64) + mu,
        muv=jnp.zeros((ny + 1, nx), dtype=jnp.float64) + mu,
        mudf=jnp.zeros((ny, nx), dtype=jnp.float64),
        theta=theta_pert,  # perturbation theta (decoupled view feeding mass-couple step)
        theta_1=theta_1,
        theta_ave=theta_pert,
        theta_tend=zeros_mass,
        mu_tend=jnp.zeros((ny, nx), dtype=jnp.float64),
        ph_tend=zeros_face,
        ph=zeros_face,  # perturbation geopotential work (0 at rest)
        p=zeros_mass,  # perturbation pressure (0 at rest)
        t_2ave=theta_pert,
        dnw=metrics.dnw,
        fnm=metrics.fnm,
        fnp=metrics.fnp,
        rdnw=metrics.rdnw,
        c1h=c1h,
        c2h=c2h,
        msfuy=metrics.msfuy,
        msfvx_inv=1.0 / metrics.msfvx,
        msftx=metrics.msftx,
        msfty=metrics.msfty,
        coef_mut=mut,
        u_tend=zeros_u,
        v_tend=zeros_v,
        p_base=fields["pb_mass"],
        ph_base=fields["phb_faces"],
        al=zeros_mass,
        alt=fields["alt_mass"],
        cqu=jnp.ones_like(zeros_u),
        cqv=jnp.ones_like(zeros_v),
        msfux=metrics.msfux,
        msfvx=metrics.msfvx,
        msfvy=metrics.msfvy,
        cf1=metrics.cf1,
        cf2=metrics.cf2,
        cf3=metrics.cf3,
        theta_work_reference=theta_1,
        c2a=c2a,
        cqw=cqw,
        c1f=metrics.c1f,
        c2f=metrics.c2f,
        rdn=metrics.rdn,
        phb=fields["phb_faces"],
        ph_1=zeros_face,
        ht=jnp.zeros((ny, nx), dtype=jnp.float64),
        pm1=zeros_mass,
        ru_m=zeros_u,
        rv_m=zeros_v,
        ww_m=zeros_face,
    ).replace(mu=jnp.zeros((ny, nx), dtype=jnp.float64))


def _max_abs(x) -> float:
    return float(np.asarray(jax.device_get(jnp.max(jnp.abs(jnp.asarray(x, dtype=jnp.float64))))))


def run_flat_rest(metrics, fields, *, dts: float, dx: float, dy: float, epssm: float = 0.5) -> dict:
    state = rest_acoustic_state(metrics, fields)
    cfg = AcousticCoreConfig(dt=dts, dx=dx, dy=dy, epssm=float(epssm), top_lid=False)
    cqw = state.cqw
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        state.mut, metrics, dt=dts, epssm=float(epssm), top_lid=False, cqw=cqw, c2a=state.c2a
    )
    nxt = acoustic_substep_core(state, a=a, alpha=alpha, gamma=gamma, cfg=cfg, cqw=cqw)
    deltas = {
        "u": _max_abs(nxt.u - state.u),
        "v": _max_abs(nxt.v - state.v),
        "w": _max_abs(nxt.w - state.w),
        "ph": _max_abs(nxt.ph - state.ph),
        "theta": _max_abs(nxt.theta - state.theta),
        "p": _max_abs(nxt.p - state.p),
        "mu": _max_abs(nxt.mu - state.mu),
    }
    # Scale eps to the rest-state magnitudes (pressure ~1e5, ph ~1e5).
    pscale = _max_abs(fields["p_total_mass"])
    phscale = _max_abs(fields["ph_total_faces"])
    tol = {
        "u": 1e-9, "v": 1e-9, "w": 1e-9,
        "ph": 1e-6 * max(phscale, 1.0),
        "theta": 1e-9,
        "p": 1e-6 * max(pscale, 1.0),
        "mu": 1e-9 * max(float(fields["mu"]), 1.0),
    }
    passed = all(deltas[k] <= tol[k] for k in deltas)
    return {"deltas": deltas, "tolerances": tol, "passed": bool(passed),
            "pressure_scale": pscale, "ph_scale": phscale}


def run_analytic_acoustic(metrics, fields, *, dts: float, dx: float, dy: float, epssm: float = 0.5) -> dict:
    """Hydrostatic-adjustment oracle (AC4): a warm mid-column theta bubble.

    Analytic expectation (sign + order of magnitude): a positive potential-
    temperature perturbation theta' creates a buoyancy force b = g*theta'/theta0
    on the air column.  In the WRF small-step w equation the perturbation theta
    enters the implicit-w RHS through c2a*alt*t_2ave (the buoyancy term), so a
    warm bubble must accelerate the overlying faces UPWARD (w>0 above the
    perturbation), raise the geopotential (ph'>0), and produce a positive
    pressure perturbation in/below the warm layer.  The leading-order
    acceleration after one acoustic substep is dw ~ b*dts = g*(theta'/theta0)*dts.
    """

    nz = int(metrics.c1h.shape[0])
    ny = int(metrics.msftx.shape[0])
    nx = int(metrics.msftx.shape[1])
    theta_amp = 1.0  # K warm perturbation
    kc = nz // 2
    theta_pert = np.zeros((nz, ny, nx))
    theta_pert[kc, :, :] = theta_amp  # one mass level warmed by +1 K
    state = rest_acoustic_state(metrics, fields, theta_pert_mass=theta_pert)
    cfg = AcousticCoreConfig(dt=dts, dx=dx, dy=dy, epssm=float(epssm), top_lid=False)
    cqw = state.cqw
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        state.mut, metrics, dt=dts, epssm=float(epssm), top_lid=False, cqw=cqw, c2a=state.c2a
    )
    nxt = acoustic_substep_core(state, a=a, alpha=alpha, gamma=gamma, cfg=cfg, cqw=cqw)

    # ``advance_w`` returns the COUPLED small-step w work array
    # w_coupled = (c1f*mut+c2f)*w_phys/msfty (WRF small_step_prep coupling).
    # Decouple before the physical sign/magnitude checks.
    c1f = np.asarray(metrics.c1f)
    c2f = np.asarray(metrics.c2f)
    mut = float(fields["mu"])
    mass_f = c1f * mut + c2f  # (nz+1,)
    w_coupled = np.asarray(jax.device_get(nxt.w))[:, 0, 0]
    w_resp = w_coupled * 1.0 / np.maximum(np.abs(mass_f), 1.0e-12)  # msfty=1 -> w_phys
    ph_resp = np.asarray(jax.device_get(nxt.ph))[:, 0, 0]
    p_resp = np.asarray(jax.device_get(nxt.p))[:, 0, 0]

    b = G * theta_amp / T0  # buoyancy accel magnitude (m/s^2)
    dw_expected_scale = b * dts  # leading-order physical w scale (m/s)

    # The face just ABOVE the warm level should be pushed up (positive coupled w),
    # the face just BELOW pushed down (negative) -- a buoyant dipole around kc.
    w_above = float(w_resp[kc + 1])
    w_below = float(w_resp[kc])
    sign_ok = (w_above > 0.0) and (w_below < 0.0)
    # geopotential above the warm layer rises (ph' increases upward through the bubble)
    ph_top = float(ph_resp[-1])
    ph_rise_ok = ph_top > 0.0
    # magnitude sanity: decoupled |w| within ~2 orders of magnitude of the
    # analytic single-substep buoyancy scale dw ~ b*dts.
    wmax = float(np.max(np.abs(w_resp)))
    mag_ok = (0.01 * dw_expected_scale <= wmax <= 100.0 * dw_expected_scale)

    return {
        "theta_perturbation_K": theta_amp,
        "warm_level_index": kc,
        "buoyancy_accel_m_s2": b,
        "dw_expected_order_m_s": dw_expected_scale,
        "w_face_above_warm": w_above,
        "w_face_below_warm": w_below,
        "w_abs_max": wmax,
        "ph_top_response": ph_top,
        "p_at_warm_level": float(p_resp[kc]),
        "w_column": w_resp.tolist(),
        "ph_column": ph_resp.tolist(),
        "p_column": p_resp.tolist(),
        "sign_dipole_ok": bool(sign_ok),
        "ph_rise_ok": bool(ph_rise_ok),
        "magnitude_ok": bool(mag_ok),
        "passed": bool(sign_ok and ph_rise_ok and mag_ok),
    }


def run_conservation(metrics, fields, *, dts: float, dx: float, dy: float, steps: int, epssm: float = 0.5) -> dict:
    """AC5: long pure-acoustic run; dry-mass + theta-mass drift, finiteness."""

    nz = int(metrics.c1h.shape[0])
    ny = int(metrics.msftx.shape[0])
    nx = int(metrics.msftx.shape[1])
    # small smooth theta perturbation to keep the run nontrivially dynamic
    theta_pert = np.zeros((nz, ny, nx))
    kc = nz // 2
    theta_pert[kc, :, :] = 0.5
    state = rest_acoustic_state(metrics, fields, theta_pert_mass=theta_pert)
    cfg = AcousticCoreConfig(dt=dts, dx=dx, dy=dy, epssm=float(epssm), top_lid=False)
    cqw = state.cqw
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        state.mut, metrics, dt=dts, epssm=float(epssm), top_lid=False, cqw=cqw, c2a=state.c2a
    )

    def dry_mass(s):
        return float(np.asarray(jax.device_get(jnp.sum(s.mut + s.mu))))

    def theta_mass(s):
        mass = s.c1h[:, None, None] * s.muts[None, :, :] + s.c2h[:, None, None]
        return float(np.asarray(jax.device_get(jnp.sum(jnp.asarray(s.theta) * mass))))

    dry0 = dry_mass(state)
    th0 = theta_mass(state)

    @jax.jit
    def body(s):
        return acoustic_substep_core(s, a=a, alpha=alpha, gamma=gamma, cfg=cfg, cqw=cqw)

    cur = state
    # Early-transient w amplitude (first 10% of the run) is the reference scale;
    # a stable scheme keeps later w within a small factor, an unstable one grows.
    settle = max(1, int(steps) // 10)
    w_early_max = 0.0
    w_late_max = 0.0
    for i in range(int(steps)):
        cur = body(cur)
        wmax = _max_abs(cur.w)
        if i < settle:
            w_early_max = max(w_early_max, wmax)
        else:
            w_late_max = max(w_late_max, wmax)
    jax.block_until_ready(cur)

    dryN = dry_mass(cur)
    thN = theta_mass(cur)
    finite = bool(
        np.isfinite(np.asarray(jax.device_get(jnp.sum(cur.w))))
        and np.isfinite(np.asarray(jax.device_get(jnp.sum(cur.theta))))
        and np.isfinite(np.asarray(jax.device_get(jnp.sum(cur.p))))
    )
    dry_drift = abs(dryN - dry0) / max(abs(dry0), 1.0)
    th_drift = abs(thN - th0) / max(abs(th0), 1.0)
    w_bounded = bool(w_late_max <= 5.0 * max(w_early_max, 1.0))
    return {
        "steps": int(steps),
        "epssm": float(epssm),
        "dry_mass_initial": dry0,
        "dry_mass_final": dryN,
        "dry_mass_relative_drift": dry_drift,
        "dry_mass_tolerance": 1.0e-6,
        "theta_mass_initial": th0,
        "theta_mass_final": thN,
        "theta_mass_relative_drift": th_drift,
        "finite": finite,
        "w_early_transient_max": w_early_max,
        "w_late_max": w_late_max,
        "w_abs_max_final": _max_abs(cur.w),
        "w_bounded": w_bounded,
        "theta_abs_max_final": _max_abs(cur.theta),
        "passed": bool(dry_drift <= 1.0e-6 and finite and np.isfinite(th_drift) and w_bounded),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "proofs/f7a2")
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--dts", type=float, default=2.0)
    parser.add_argument("--dx", type=float, default=1000.0)
    parser.add_argument("--dy", type=float, default=1000.0)
    parser.add_argument("--conservation-steps", type=int, default=300)
    parser.add_argument("--epssm", type=float, default=0.5, help="off-centering (Gen2 d02 namelist uses 0.5)")
    parser.add_argument("--hybrid", action="store_true", help="use hybrid sigma-pressure metrics (nonzero c2)")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, fields = build_hydrostatic_column(
        nz=args.nz, ny=args.ny, nx=args.nx, pure_sigma=not args.hybrid
    )

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "grid": {"nz": args.nz, "ny": args.ny, "nx": args.nx},
        "dts_s": args.dts, "dx_m": args.dx, "dy_m": args.dy, "epssm": args.epssm,
        "device": [str(d) for d in jax.devices()],
    }

    flat = run_flat_rest(metrics, fields, dts=args.dts, dx=args.dx, dy=args.dy, epssm=args.epssm)
    flat_payload = {**meta, "oracle": "flat_rest_AC3", **flat}
    (args.output_dir / "flat_rest_oracle.json").write_text(json.dumps(flat_payload, indent=2) + "\n")
    print("[AC3 flat-rest] passed =", flat["passed"], "deltas =", flat["deltas"])

    ana = run_analytic_acoustic(metrics, fields, dts=args.dts, dx=args.dx, dy=args.dy, epssm=args.epssm)
    ana_payload = {**meta, "oracle": "analytic_hydrostatic_adjustment_AC4", **ana}
    (args.output_dir / "analytic_acoustic_oracle.json").write_text(json.dumps(ana_payload, indent=2) + "\n")
    print("[AC4 analytic] passed =", ana["passed"], "sign_dipole_ok =", ana["sign_dipole_ok"],
          "ph_rise_ok =", ana["ph_rise_ok"], "w_abs_max =", ana["w_abs_max"])

    cons = run_conservation(
        metrics, fields, dts=args.dts, dx=args.dx, dy=args.dy, steps=args.conservation_steps, epssm=args.epssm
    )
    cons_payload = {**meta, "oracle": "conservation_AC5", **cons}
    (args.output_dir / "conservation_long_run.json").write_text(json.dumps(cons_payload, indent=2) + "\n")
    print("[AC5 conservation] passed =", cons["passed"], "dry_drift =", cons["dry_mass_relative_drift"],
          "theta_drift =", cons["theta_mass_relative_drift"], "w_bounded =", cons["w_bounded"],
          "w_final =", cons["w_abs_max_final"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
