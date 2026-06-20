"""Opus v040 round — savepoint parity for the WRF-faithful PGF (al + dpn) fixes
and the momentum-advection interior/boundary residual decomposition.

Predeclared tolerances (machine-precision fp64, NO loosening):
  * PGF interior (ring-5-excluded) JAX-vs-WRF-faithful: rmse < 1e-10 of the WRF
    term rmse -- the al+dpn fixes are now WRF-faithful so the interior PGF must
    match a correct WRF PGF oracle to roundoff.
  * Momentum advection interior (ring-5-excluded) JAX-periodic vs WRF-specified:
    rmse < 1e-12 absolute -- WRF does NOT degrade the flux order in the interior,
    so the periodic flux5 path is bit-identical to WRF there (this is the KEY
    adjudication finding: the GPT "5x divergence" is 100% in the boundary ring).
  * Momentum advection BOUNDARY ring (width 5): reported HONESTLY -- JAX-periodic
    differs from WRF-specified there; that gap is overwritten by the lateral
    boundary spec/relax zone every step (it is NOT an interior bias driver).

The WRF-faithful PGF oracle here uses the CORRECT al (alb/muts) and dpn (cfn/cfn1
top) -- unlike the GPT round-4 oracle which shared the JAX al/dpn transcription
and therefore could not detect either bug (a self-compare for those terms).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for c in (ROOT / "src", ROOT / "proofs" / "v040", ROOT):
    if str(c) not in sys.path:
        sys.path.insert(0, str(c))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import config as jax_config  # noqa: E402

jax_config.update("jax_enable_x64", True)

import netCDF4  # noqa: E402

from gpuwrf.dynamics.acoustic_wrf import _inverse_density_from_theta_pressure, moisture_coupling_factors  # noqa: E402
from gpuwrf.dynamics.core.rk_addtend_dry import large_step_horizontal_pgf  # noqa: E402
from gpuwrf.dynamics.flux_advection import advect_u_flux, advect_v_flux, couple_velocities_periodic  # noqa: E402
from gpuwrf.dynamics.metrics import load_wrfinput_metrics  # noqa: E402

from lbc_budget_trace import _state_from_wrfout  # noqa: E402
from pgf_inloop_isolation import _metrics_np, _state_np, wrf_advect_u_components_np, wrf_advect_v_components_np, wrf_transport_np  # noqa: E402

CASES = [
    ("20260429_d01_l2",
     "<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260429_18z_l2_72h_20260524T204451Z/wrfinput_d01",
     "<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260429_18z_l2_72h_20260524T204451Z/wrfout_d01_2026-04-29_18:00:00"),
]

# Gen2 WRF dyn_em source tree used for provenance hashing only (sha256 -> "missing"
# if absent, so this is safe for outsiders). Override with WRF_GEN2_SRC_ROOT.
WRF_GEN2_SRC_ROOT = Path(os.environ.get("WRF_GEN2_SRC_ROOT", str(ROOT.parent / "canairy_meteo" / "Gen2" / "artifacts" / "wrf_gpu_src" / "WRF")))
WRF_PGF_SRC = WRF_GEN2_SRC_ROOT / "dyn_em/module_big_step_utilities_em.F"
WRF_ADV_SRC = WRF_GEN2_SRC_ROOT / "dyn_em/module_advect_em.F"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else "missing"


def _np(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _interior(shape2d, ring):
    m = np.ones(shape2d, dtype=bool)
    m[:ring] = m[-ring:] = m[:, :ring] = m[:, -ring:] = False
    return m


def _cmp(jax_arr, wrf_arr, ring=5):
    jax_arr = _np(jax_arr); wrf_arr = _np(wrf_arr)
    d = jax_arr - wrf_arr
    ny, nx = d.shape[1], d.shape[2]
    intr = np.broadcast_to(_interior((ny, nx), ring)[None, :, :], d.shape)
    def st(a):
        a = np.asarray(a)
        return {"rmse": float(np.sqrt(np.mean(a*a))) if a.size else 0.0,
                "mean": float(np.mean(a)) if a.size else 0.0,
                "max_abs": float(np.max(np.abs(a))) if a.size else 0.0, "n": int(a.size)}
    return {
        "full": st(d),
        "interior_ring%d_excluded" % ring: st(d[intr]),
        "boundary_ring%d" % ring: st(d[~intr]),
        "wrf_term_rmse": float(np.sqrt(np.mean(wrf_arr*wrf_arr))),
    }


def wrf_faithful_pgf(state, metrics, dx_m, dy_m):
    """PGF with WRF-correct al (alb/muts) and dpn (cfn/cfn1 top), top_lid=True."""
    c1h = _np(metrics.c1h)[:, None, None]; c2h = _np(metrics.c2h)[:, None, None]
    rdnw = _np(metrics.rdnw)[:, None, None]
    rdx = 1.0/dx_m; rdy = 1.0/dy_m
    ph = _np(state.ph_perturbation); p_abs = _np(state.p_perturbation)
    mu_total = _np(state.mu_total); mu_pert = _np(state.mu_perturbation); mub = mu_total - mu_pert
    pb = _np(state.p_total) - p_abs
    phb = _np(state.ph_total) - ph
    alt = _np(_inverse_density_from_theta_pressure(jnp.asarray(state.theta, jnp.float64), jnp.asarray(state.p_total, jnp.float64)))
    alb = -rdnw*(phb[1:]-phb[:-1])/(c1h*mub[None]+c2h)
    muts = mu_total + mu_pert
    al = -(alb*c1h*mu_pert[None] + rdnw*(ph[1:]-ph[:-1]))/(c1h*muts[None]+c2h)
    ph_total = phb + ph; php = 0.5*(ph_total[:-1]+ph_total[1:])
    msf_u = _np(metrics.msfux/metrics.msfuy)[None]; msf_v = _np(metrics.msfvy/metrics.msfvx)[None]
    cqu, cqv = (_np(c) for c in moisture_coupling_factors(state))
    dnw_v = _np(metrics.dnw); dn_v = _np(metrics.dn)
    cfn = (0.5*dnw_v[-1]+dn_v[-1])/dn_v[-1]; cfn1 = -0.5*dnw_v[-1]/dn_v[-1]

    def xp3(f): p=np.pad(f,((0,0),(0,0),(1,1)),mode="edge"); return p[:,:,:-1],p[:,:,1:]
    def yp3(f): p=np.pad(f,((0,0),(1,1),(0,0)),mode="edge"); return p[:,:-1,:],p[:,1:,:]
    def xp2(f): p=np.pad(f,((0,0),(1,1)),mode="edge"); return p[:,:-1],p[:,1:]
    def yp2(f): p=np.pad(f,((1,1),(0,0)),mode="edge"); return p[:-1,:],p[1:,:]

    def dpn(ps):
        nz=ps.shape[0]; out=np.zeros((nz+1,)+ps.shape[1:])
        out[0]=0.5*(_np(metrics.cf1)*ps[0]+_np(metrics.cf2)*ps[1]+_np(metrics.cf3)*ps[2])
        out[1:nz]=0.5*(_np(metrics.fnm)[1:,None,None]*ps[1:]+_np(metrics.fnp)[1:,None,None]*ps[:-1])
        out[nz]=0.5*(cfn*ps[-1]+cfn1*ps[-2])  # top_lid cfn/cfn1
        return out

    muu=0.5*sum(xp2(mu_total)); mass_u=c1h*muu[None]+c2h
    muv=0.5*sum(yp2(mu_total)); mass_v=c1h*muv[None]+c2h
    ph_l,ph_r=xp3(ph); p_l,p_r=xp3(p_abs); pb_l,pb_r=xp3(pb); al_l,al_r=xp3(al); alt_l,alt_r=xp3(alt)
    dpx=msf_u*0.5*rdx*mass_u*(((ph_r[1:]-ph_l[1:])+(ph_r[:-1]-ph_l[:-1]))+(alt_l+alt_r)*(p_r-p_l)+(al_l+al_r)*(pb_r-pb_l))
    php_l,php_r=xp3(php); ps_l,ps_r=xp3(p_abs); dpn_x=dpn(ps_l+ps_r); mu_l,mu_r=xp2(mu_pert)
    dpx=dpx+msf_u*rdx*(php_r-php_l)*(rdnw*(dpn_x[1:]-dpn_x[:-1])-0.5*(c1h*(mu_l+mu_r)[None]))
    ph_s,ph_n=yp3(ph); p_s,p_n=yp3(p_abs); pb_s,pb_n=yp3(pb); al_s,al_n=yp3(al); alt_s,alt_n=yp3(alt)
    dpy=msf_v*0.5*rdy*mass_v*(((ph_n[1:]-ph_s[1:])+(ph_n[:-1]-ph_s[:-1]))+(alt_s+alt_n)*(p_n-p_s)+(al_s+al_n)*(pb_n-pb_s))
    php_s,php_n=yp3(php); ps_s,ps_n=yp3(p_abs); dpn_y=dpn(ps_s+ps_n); mu_s,mu_n=yp2(mu_pert)
    dpy=dpy+msf_v*rdy*(php_n-php_s)*(rdnw*(dpn_y[1:]-dpn_y[:-1])-0.5*(c1h*(mu_s+mu_n)[None]))
    return -cqu*dpx, -cqv*dpy


def main() -> int:
    report: dict[str, Any] = {
        "schema": "v040_boundary_transport_savepoint_parity.v1",
        "created_by": "Opus 4.8 MAX",
        "fixes_under_test": [
            "rk_addtend_dry._absolute_diagnostics: al uses alb (base inv density) + muts (total column mass) — WRF calc_p_rho_phi:1029",
            "rk_addtend_dry.large_step_horizontal_pgf._dpn_faces: top_lid face uses cfn/cfn1 (WRF :2357-2362) not cf1/cf2/cf3",
        ],
        "tolerances": {
            "pgf_interior_rmse_over_wrf_rmse_max": 1.0e-10,
            "advection_interior_rmse_abs_max": 1.0e-12,
            "advection_boundary_ring": "reported honestly; overwritten by spec/relax LBC zone each step",
        },
        "wrf_source": {"pgf": {"path": str(WRF_PGF_SRC), "sha256": _sha(WRF_PGF_SRC)},
                       "advection": {"path": str(WRF_ADV_SRC), "sha256": _sha(WRF_ADV_SRC)}},
        "cases": {},
    }
    overall_pass = True
    for cid, md, wo in CASES:
        metrics = load_wrfinput_metrics(md); state = _state_from_wrfout(wo)
        metrics_n = _metrics_np(metrics); state_n = _state_np(state)
        with netCDF4.Dataset(md) as ds:
            dx_m = float(ds.DX); dy_m = float(ds.DY)
        rdx, rdy = 1.0/dx_m, 1.0/dy_m

        # --- PGF: live JAX (fixed) vs WRF-faithful oracle ---
        jru, jrv = large_step_horizontal_pgf(state, metrics, dx_m=dx_m, dy_m=dy_m, top_lid=True)
        wru, wrv = wrf_faithful_pgf(state, metrics, dx_m, dy_m)
        pgf_u = _cmp(jru, wru); pgf_v = _cmp(jrv, wrv)

        # --- momentum advection: JAX-periodic vs WRF-specified oracle ---
        vel = couple_velocities_periodic(state.u, state.v, state.mu_total,
            c1h=metrics.c1h, c2h=metrics.c2h, dnw=metrics.dnw, rdx=rdx, rdy=rdy,
            msfuy=metrics.msfuy, msfvx=metrics.msfvx, msftx=metrics.msftx, msfux=metrics.msfux, msfvy=metrics.msfvy)
        wrf_tr = wrf_transport_np(state_n, metrics_n, dx_m=dx_m, dy_m=dy_m)
        wu = wrf_advect_u_components_np(state_n, wrf_tr, metrics_n, dx_m=dx_m, dy_m=dy_m)
        wv = wrf_advect_v_components_np(state_n, wrf_tr, metrics_n, dx_m=dx_m, dy_m=dy_m)
        ju = advect_u_flux(state.u, vel, rdx=rdx, rdy=rdy, rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp)
        jv = advect_v_flux(state.v, vel, rdx=rdx, rdy=rdy, rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp)
        adv_u = _cmp(ju, wu["total"]); adv_v = _cmp(jv, wv["total"])

        pgf_int_u = pgf_u["interior_ring5_excluded"]["rmse"] / max(pgf_u["wrf_term_rmse"], 1e-30)
        pgf_int_v = pgf_v["interior_ring5_excluded"]["rmse"] / max(pgf_v["wrf_term_rmse"], 1e-30)
        adv_int_u = adv_u["interior_ring5_excluded"]["rmse"]
        adv_int_v = adv_v["interior_ring5_excluded"]["rmse"]
        case_pass = (pgf_int_u < 1e-10 and pgf_int_v < 1e-10 and adv_int_u < 1e-12 and adv_int_v < 1e-12)
        overall_pass = overall_pass and case_pass
        report["cases"][cid] = {
            "wrfout": wo, "dx_m": dx_m,
            "pgf_u_jax_vs_wrffaithful": pgf_u, "pgf_v_jax_vs_wrffaithful": pgf_v,
            "advection_u_jaxperiodic_vs_wrfspecified": adv_u,
            "advection_v_jaxperiodic_vs_wrfspecified": adv_v,
            "gate": {
                "pgf_interior_rmse_over_wrf_u": pgf_int_u, "pgf_interior_rmse_over_wrf_v": pgf_int_v,
                "advection_interior_rmse_u": adv_int_u, "advection_interior_rmse_v": adv_int_v,
                "pass": bool(case_pass),
            },
        }
    report["overall_savepoint_parity_pass"] = bool(overall_pass)
    out = Path(__file__).resolve().parent / "boundary_transport_savepoint_parity.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["cases"], indent=2, sort_keys=True)[:2500])
    print("OVERALL_SAVEPOINT_PARITY_PASS =", overall_pass)
    print("WROTE", out)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
