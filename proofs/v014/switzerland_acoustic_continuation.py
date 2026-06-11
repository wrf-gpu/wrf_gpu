#!/usr/bin/env python
"""V0.14 Switzerland acoustic continuation: term-level rhs_ph/ww/rw_tend parity.

GPT consensus (proofs/v014/gpt_acoustic_wrf_source_equation_audit.md) ranked the
real-case geopotential-tendency lane (stage ``ww`` from calc_ww_cp + ``rhs_ph``)
as the most likely remaining p/ph-first divergence.  This script settles that
WITHOUT a new WRF run or GPU: at WRF call 21601 (RK1 of step 7201) the stage
inputs are bit-identical to the h36 wrfout, and ``calc_ww_cp``, ``rhs_ph`` and
``pg_buoy_w`` are pure functions of those inputs.  We therefore port the WRF
real-case (specified-boundary, h_sca_adv_order=5 -> order<=6 branch,
map-factored) reference in fp64 numpy as the ORACLE and evaluate the production
JAX kernels on the same arrays on the CPU backend.

Discriminators (GPT acceptance list):
  D1  JAX ``couple_velocities_periodic().rom``  vs  oracle calc_ww_cp ``ww``.
  D2  JAX ``rhs_ph_wrf``                         vs  oracle rhs_ph
      (both fed the SAME omega, isolating the operator), plus the production
      combination rom+rhs_ph_wrf vs the oracle pair.
  D3  JAX ``pg_buoy_w_moist``+stage w_damp       vs  oracle pg_buoy_w + w_damp.
Each error is also converted to an expected one-stage Delta-ph via the
advance_w fold (msfty*dts*ph_tend/(c1f*mut+c2f)) and compared against the
measured stage-1 ph increment error (0.876 m2/s2 rmse interior, tag
sub4_dt18_omfix) to decide whether this lane explains the first divergence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_HPG_SPEC = importlib.util.spec_from_file_location(
    "hpg_native_face_proof", Path(__file__).with_name("switzerland_hpg_native_face_fix.py")
)
hpg = importlib.util.module_from_spec(_HPG_SPEC)
_HPG_SPEC.loader.exec_module(hpg)  # type: ignore[union-attr]

OUT_JSON = ROOT / "proofs/v014/switzerland_acoustic_continuation.json"
G = 9.81
INTERIOR_DEPTH = 8


def _stats(arr: np.ndarray) -> dict[str, float]:
    v = np.asarray(arr, dtype=np.float64)
    if v.size == 0:
        return {"count": 0}
    return {
        "count": int(v.size),
        "mean": float(np.nanmean(v)),
        "rmse": float(np.sqrt(np.nanmean(v * v))),
        "max_abs": float(np.nanmax(np.abs(v))),
    }


def _interior_mask(ny: int, nx: int, depth: int = INTERIOR_DEPTH) -> np.ndarray:
    jj, ii = np.mgrid[0:ny, 0:nx]
    return (ii >= depth) & (ii < nx - depth) & (jj >= depth) & (jj < ny - depth)


def _split(diff: np.ndarray) -> dict[str, Any]:
    if diff.ndim == 2:
        ny, nx = diff.shape
        m2 = _interior_mask(ny, nx)
        return {"full": _stats(diff), "interior": _stats(diff[m2]), "band": _stats(diff[~m2])}
    _, ny, nx = diff.shape
    m2 = _interior_mask(ny, nx)
    m3 = np.broadcast_to(m2[None], diff.shape)
    return {"full": _stats(diff), "interior": _stats(diff[m3]), "band": _stats(diff[~m3])}


def load_h36() -> dict[str, np.ndarray | float]:
    from netCDF4 import Dataset

    with Dataset(hpg.fn(hpg.CPU, 36)) as d:
        g = lambda name: np.asarray(d.variables[name][0], dtype=np.float64)
        out: dict[str, Any] = {
            "u": g("U"), "v": g("V"), "w": g("W"),
            "ph": g("PH"), "phb": g("PHB"),
            "mu": g("MU"), "mub": g("MUB"),
            "p": g("P"), "pb": g("PB"),
            "qv": g("QVAPOR"), "qc": g("QCLOUD"), "qr": g("QRAIN"),
            "qi": g("QICE"), "qs": g("QSNOW"), "qg": g("QGRAUP"),
            "c1h": g("C1H"), "c2h": g("C2H"), "c1f": g("C1F"), "c2f": g("C2F"),
            "fnm": g("FNM"), "fnp": g("FNP"), "rdnw": g("RDNW"), "dnw": g("DNW"),
            "rdn": g("RDN"),
            "msfux": g("MAPFAC_UX"), "msfuy": g("MAPFAC_UY"),
            "msfvx": g("MAPFAC_VX"), "msfvy": g("MAPFAC_VY"),
            "msftx": g("MAPFAC_MX"), "msfty": g("MAPFAC_MY"),
            "cf1": float(np.asarray(d.variables["CF1"][0])),
            "cf2": float(np.asarray(d.variables["CF2"][0])),
            "cf3": float(np.asarray(d.variables["CF3"][0])),
            "cfn": float(np.asarray(d.variables["CFN"][0])),
            "cfn1": float(np.asarray(d.variables["CFN1"][0])),
            "rdx": 1.0 / float(d.DX),
            "rdy": 1.0 / float(d.DY),
        }
    out["mu_total"] = out["mu"] + out["mub"]
    return out


# --------------------------------------------------------------------------
# Oracle 1: WRF calc_ww_cp (module_big_step_utilities_em.F:640-782), full tile.
# Edge faces need the mu halo outside the domain; WRF fills it via the bdy/halo
# machinery.  We edge-pad (replicate), and flag ring<=1 as halo-ambiguous.
# --------------------------------------------------------------------------

def oracle_calc_ww_cp(f: dict[str, Any]) -> np.ndarray:
    mu_t = f["mu_total"]  # (ny, nx)
    ny, nx = mu_t.shape
    nz = f["c1h"].shape[0]
    mu_pad_x = np.pad(mu_t, ((0, 0), (1, 0)), mode="edge")
    mu_pad_y = np.pad(mu_t, ((1, 0), (0, 0)), mode="edge")
    muu = 0.5 * (mu_pad_x[:, 1:] + mu_pad_x[:, :-1])  # faces i=1..nx -> (ny, nx)
    muu = np.concatenate([muu, mu_t[:, -1:]], axis=1)  # face nx+1 (edge, pad)
    muv = 0.5 * (mu_pad_y[1:, :] + mu_pad_y[:-1, :])
    muv = np.concatenate([muv, mu_t[-1:, :]], axis=0)
    # WRF muu(i,j) = 0.5*(mu(i)+mu(i-1)); face arrays here are (ny, nx+1)/(ny+1, nx)
    c1h = f["c1h"][:, None, None]
    c2h = f["c2h"][:, None, None]
    ru = (c1h * muu[None] + c2h) * f["u"] / f["msfuy"][None]  # (nz, ny, nx+1)
    rv = (c1h * muv[None] + c2h) * f["v"] / f["msfvx"][None]  # (nz, ny+1, nx)
    divv = f["msftx"][None] * f["dnw"][:, None, None] * (
        f["rdx"] * (ru[:, :, 1:] - ru[:, :, :-1]) + f["rdy"] * (rv[:, 1:, :] - rv[:, :-1, :])
    )  # (nz, ny, nx)
    dmdt = divv.sum(axis=0)  # (ny, nx)
    ww = np.zeros((nz + 1, ny, nx), dtype=np.float64)
    # ww(k) = ww(k-1) - dnw(k-1)*c1h(k-1)*dmdt - divv(k-1), faces 1..nz-1 (0-based)
    increments = -(f["dnw"][:, None, None] * f["c1h"][:, None, None] * dmdt[None]) - divv
    ww[1:nz, :, :] = np.cumsum(increments, axis=0)[: nz - 1]
    return ww


# --------------------------------------------------------------------------
# Oracle 2: WRF rhs_ph real case (module_big_step_utilities_em.F:1365-2178):
# phi_adv_z=1 term 3, non-hydrostatic gw on faces 2..kde (top face zeroed then
# gw-added), advective_order<=6 horizontal advection with map factors, the
# specified-boundary degradation rows, and the top-face cfn/cfn1 row.
# --------------------------------------------------------------------------

def oracle_rhs_ph(f: dict[str, Any], ww: np.ndarray, advective_order: int = 5) -> np.ndarray:
    ph, phb, w = f["ph"], f["phb"], f["w"]
    nzp1, ny, nx = ph.shape
    nz = nzp1 - 1
    ph_tot = ph + phb
    msfty = f["msfty"]
    ph_tend = np.zeros_like(ph)

    # term 3 (phi_adv_z=1): faces 1..nz-1
    dphi = ph_tot[1:] - ph_tot[:-1]  # (nz, ny, nx)
    wdwn = 0.5 * (ww[1:] + ww[:-1]) * f["rdnw"][:, None, None] * dphi
    ph_tend[1:nz] -= f["fnm"][1:nz, None, None] * wdwn[1:nz] + f["fnp"][1:nz, None, None] * wdwn[: nz - 1]

    # term 4 gw: zero top face, then add on faces 1..nz (WRF k=2..kte, kte=kde)
    mass_f = f["c1f"][:, None, None] * f["mu_total"][None] + f["c2f"][:, None, None]
    ph_tend[nz] = 0.0
    ph_tend[1:] += mass_f[1:] * G * w[1:] / msfty[None]

    # horizontal advection, advective_order<=6 branch (5 and 6 share it).
    muuf = 0.5 * (np.pad(f["mu_total"], ((0, 0), (1, 0)), mode="edge")[:, 1:]
                  + np.pad(f["mu_total"], ((0, 0), (1, 0)), mode="edge")[:, :-1])
    muuf = np.concatenate([muuf, f["mu_total"][:, -1:]], axis=1)  # (ny, nx+1)
    muvf = 0.5 * (np.pad(f["mu_total"], ((1, 0), (0, 0)), mode="edge")[1:, :]
                  + np.pad(f["mu_total"], ((1, 0), (0, 0)), mode="edge")[:-1, :])
    muvf = np.concatenate([muvf, f["mu_total"][-1:, :]], axis=0)  # (ny+1, nx)

    c1f = f["c1f"]
    c2f = f["c2f"]
    cfn, cfn1 = f["cfn"], f["cfn1"]
    u, v = f["u"], f["v"]

    def v_pair(k_face: int) -> np.ndarray:
        # interior faces: v(k)+v(k-1) at mass levels k_face, k_face-1
        return v[k_face] + v[k_face - 1]

    def u_pair(k_face: int) -> np.ndarray:
        return u[k_face] + u[k_face - 1]

    # stencil helpers on ph_tot at fixed face k: derivative sums about mass cell j/i
    def d6_y(k: int, j: np.ndarray) -> np.ndarray:
        a = ph_tot[k]
        return (1.0 / 60.0) * (
            45.0 * (a[j + 1, :] - a[j - 1, :]) - 9.0 * (a[j + 2, :] - a[j - 2, :]) + (a[j + 3, :] - a[j - 3, :])
        )

    def d4_y(k: int, j: np.ndarray) -> np.ndarray:
        a = ph_tot[k]
        return (1.0 / 12.0) * (8.0 * (a[j + 1, :] - a[j - 1, :]) - (a[j + 2, :] - a[j - 2, :]))

    def d6_x(k: int, i: np.ndarray) -> np.ndarray:
        a = ph_tot[k]
        return (1.0 / 60.0) * (
            45.0 * (a[:, i + 1] - a[:, i - 1]) - 9.0 * (a[:, i + 2] - a[:, i - 2]) + (a[:, i + 3] - a[:, i - 3])
        )

    def d4_x(k: int, i: np.ndarray) -> np.ndarray:
        a = ph_tot[k]
        return (1.0 / 12.0) * (8.0 * (a[:, i + 1] - a[:, i - 1]) - (a[:, i + 2] - a[:, i - 2]))

    assert advective_order in (5, 6)

    # ---- y advection ----
    # interior rows j in [3, ny-4] (WRF jds+3..jde-4, 0-based mass), 6th-order sym.
    for k in range(1, nzp1):  # faces 1..nz; top face uses cfn/cfn1 winds & 0.5 weight
        if k < nz:
            vp = v_pair(k)  # (ny+1, nx)
            wgt = 0.25
        else:
            vp = cfn * v[nz - 1] + cfn1 * v[nz - 2]
            wgt = 0.5
        mass_vf = c1f[k] * muvf + c2f[k]  # (ny+1, nx)
        flow = mass_vf * vp * f["msfvy"]  # (ny+1, nx)

        j_int = np.arange(3, ny - 3)  # 6th-order needs j+-3
        j6 = j_int[(j_int >= 3) & (j_int <= ny - 4)]
        if j6.size:
            adv = (flow[j6 + 1, :] + flow[j6, :]) * d6_y(k, j6)
            ph_tend[k][j6, :] -= wgt * f["rdy"] / msfty[j6, :] * adv
        # 4th-order rows j=2 and j=ny-3 (specified)
        for j4 in (2, ny - 3):
            adv = (flow[j4 + 1, :] + flow[j4, :]) * d4_y(k, np.asarray([j4]))[0]
            ph_tend[k][j4, :] -= wgt * f["rdy"] / msfty[j4, :] * adv
        # 2nd-order rows j=1 and j=ny-2 (specified, one-in form)
        for j2 in (1, ny - 2):
            a = ph_tot[k]
            adv = flow[j2 + 1, :] * (a[j2 + 1, :] - a[j2, :]) + flow[j2, :] * (a[j2, :] - a[j2 - 1, :])
            ph_tend[k][j2, :] -= wgt * f["rdy"] / msfty[j2, :] * adv
        # rows j=0 and j=ny-1: no y-advection under specified.

    # ---- x advection ----
    for k in range(1, nzp1):
        if k < nz:
            up = u_pair(k)  # (ny, nx+1)
            wgt = 0.25
        else:
            up = cfn * u[nz - 1] + cfn1 * u[nz - 2]
            wgt = 0.5
        mass_uf = c1f[k] * muuf + c2f[k]
        flow = mass_uf * up * f["msfux"]  # (ny, nx+1)

        # WRF re-initialises j_start=jts / jtf=jde-1 for the x section: the
        # x-advection applies on ALL mass rows j=0..ny-1 (only i is trimmed).
        i6 = np.arange(3, nx - 3)
        i6 = i6[(i6 >= 3) & (i6 <= nx - 4)]
        if i6.size:
            adv = (flow[:, i6 + 1] + flow[:, i6]) * d6_x(k, i6)
            ph_tend[k][:, i6] -= wgt * f["rdx"] / msfty[:, i6] * adv
        # NOTE WRF quirk: 4th-order x rows i=2 / i=nx-3 are OPEN-only; under
        # SPECIFIED they get NO x-advection at all.
        for i2 in (1, nx - 2):
            a = ph_tot[k]
            adv = flow[:, i2 + 1] * (a[:, i2 + 1] - a[:, i2]) + flow[:, i2] * (a[:, i2] - a[:, i2 - 1])
            ph_tend[k][:, i2] -= wgt * f["rdx"] / msfty[:, i2] * adv
        # rows i=0, nx-1 and the specified-gap rows i=2, nx-3: no x-advection.

    return ph_tend


def _wrf_x_trims_note() -> str:
    return (
        "WRF order<=6 rhs_ph x-advection under specified BCs: interior 6th-order "
        "i in [ids+3, ide-4]; 2nd-order rows at ids+1 and ide-2; the 4th-order "
        "rows ids+2/ide-3 are gated on open_x* ONLY, so specified domains have "
        "NO x-advection on those two columns (module_big_step_utilities_em.F:"
        "1973-2021 open-only blocks vs 2022-2071 specified 2nd-order blocks)."
    )


# --------------------------------------------------------------------------
# Oracle 3: pg_buoy_w (moist) + stage-level w_damp on the large-step rw_tend.
# --------------------------------------------------------------------------

def oracle_rw_tend(f: dict[str, Any], ww_stage: np.ndarray, dt: float = 18.0) -> dict[str, np.ndarray]:
    p = f["p"]  # grid%p at RK1 == file P (bit-exact)
    nz, ny, nx = p.shape
    mu_prime = f["mu"]
    qtot = f["qv"] + f["qc"] + f["qr"] + f["qi"] + f["qs"] + f["qg"]
    cqw = np.zeros((nz + 1, ny, nx))
    cqw[1:nz] = 0.5 * (qtot[1:nz] + qtot[: nz - 1])
    cq1 = 1.0 / (1.0 + cqw)
    cq2 = cqw * cq1
    msft_inv = 1.0 / f["msfty"]
    rw = np.zeros((nz + 1, ny, nx))
    rw[1:nz] = msft_inv[None] * G * (
        cq1[1:nz] * f["rdn"][1:nz, None, None] * (p[1:nz] - p[: nz - 1])
        - f["c1f"][1:nz, None, None] * mu_prime[None]
        - cq2[1:nz] * (f["c1f"][1:nz, None, None] * f["mub"][None] + f["c2f"][1:nz, None, None])
    )
    cqw_top = cqw[nz - 1]
    cq1_t = 1.0 / (1.0 + cqw_top)
    cq2_t = cqw_top * cq1_t
    rw[nz] = msft_inv * G * (
        cq1_t * 2.0 * f["rdnw"][nz - 1] * (-p[nz - 1])
        - f["c1f"][nz] * mu_prime
        - cq2_t * (f["c1f"][nz] * f["mub"] + f["c2f"][nz])
    )
    # WRF w_damp (non-IEVA): activation vert_cfl > w_beta=1.0; magnitude
    # -sign(w)*0.3*(vert_cfl-2.0)*(c1f*mut+c2f), on rw_tend, full dt, stage ww/w.
    mass_f = f["c1f"][:, None, None] * f["mu_total"][None] + f["c2f"][:, None, None]
    vert_cfl = np.zeros_like(rw)
    vert_cfl[1:nz] = np.abs(ww_stage[1:nz] / mass_f[1:nz] * f["rdnw"][1:nz, None, None] * dt)
    damp = -np.sign(f["w"]) * 0.3 * (vert_cfl - 2.0) * mass_f
    rw_damped = np.where((vert_cfl > 1.0) & (np.arange(nz + 1)[:, None, None] >= 1)
                         & (np.arange(nz + 1)[:, None, None] < nz), rw + damp, rw)
    return {"rw_pg_buoy": rw, "rw_with_wdamp": rw_damped, "vert_cfl_max": vert_cfl.max(axis=0)}


# --------------------------------------------------------------------------
# JAX-side production kernels on the same arrays (CPU backend).
# --------------------------------------------------------------------------

def jax_terms(f: dict[str, Any]) -> dict[str, np.ndarray]:
    import jax.numpy as jnp
    from gpuwrf.dynamics.flux_advection import couple_velocities_periodic
    from gpuwrf.dynamics.core.rhs_ph import rhs_ph_wrf

    vel = couple_velocities_periodic(
        jnp.asarray(f["u"]),
        jnp.asarray(f["v"]),
        jnp.asarray(f["mu_total"]),
        c1h=jnp.asarray(f["c1h"]),
        c2h=jnp.asarray(f["c2h"]),
        dnw=jnp.asarray(f["dnw"]),
        rdx=float(f["rdx"]),
        rdy=float(f["rdy"]),
        msfuy=jnp.asarray(f["msfuy"]),
        msfvx=jnp.asarray(f["msfvx"]),
        msftx=jnp.asarray(f["msftx"]),
        msfux=jnp.asarray(f["msfux"]),
        msfvy=jnp.asarray(f["msfvy"]),
    )
    rom = np.asarray(vel.rom)

    mu_pad_x = np.pad(f["mu_total"], ((0, 0), (1, 0)), mode="edge")
    muu = 0.5 * (mu_pad_x[:, 1:] + mu_pad_x[:, :-1])
    muu = np.concatenate([muu, f["mu_total"][:, -1:]], axis=1)
    mu_pad_y = np.pad(f["mu_total"], ((1, 0), (0, 0)), mode="edge")
    muv = 0.5 * (mu_pad_y[1:, :] + mu_pad_y[:-1, :])
    muv = np.concatenate([muv, f["mu_total"][-1:, :]], axis=0)

    def run_rhs(ww_in: np.ndarray) -> np.ndarray:
        return np.asarray(
            rhs_ph_wrf(
                u=jnp.asarray(f["u"]),
                v=jnp.asarray(f["v"]),
                ww=jnp.asarray(ww_in),
                ph=jnp.asarray(f["ph"]),
                phb=jnp.asarray(f["phb"]),
                w=jnp.asarray(f["w"]),
                mut=jnp.asarray(f["mu_total"]),
                muu=jnp.asarray(muu),
                muv=jnp.asarray(muv),
                c1f=jnp.asarray(f["c1f"]),
                c2f=jnp.asarray(f["c2f"]),
                fnm=jnp.asarray(f["fnm"]),
                fnp=jnp.asarray(f["fnp"]),
                rdnw=jnp.asarray(f["rdnw"]),
                rdx=float(f["rdx"]),
                rdy=float(f["rdy"]),
                msfty=jnp.asarray(f["msfty"]),
                non_hydrostatic=True,
            )
        )

    return {"rom": rom, "run_rhs": run_rhs}  # type: ignore[return-value]


def gate(args: argparse.Namespace) -> dict[str, Any]:
    """h36->h37 depth-8 budget gate for the committed fix set."""

    RHSPH = hpg.PROBE_ROOT / "gpu_output_acoustic_rhsph2"
    CPU = hpg.CPU
    cpu = hpg.budget_between(CPU, 36, CPU, 37, depth=8)
    runs = {
        "old_ec4d6769": hpg.BASELINE_GPU,
        "hypso_3d0b439c": hpg.FIX_GPU,
        "rejected_candidate_for_record": hpg.PROBE_ROOT / "gpu_output_acoustic_substep_fix",
        # rhsph = real-case rhs_ph + constants + WRAPPED-edge stage omega (the
        # intermediate run that exposed the band poisoning):
        "rhsph_wrapped_omega_for_record": hpg.PROBE_ROOT / "gpu_output_acoustic_rhsph",
        # rhsph2 = the committed set: real-case rhs_ph + constants +
        # edge-faithful specified stage omega:
        "rhsph_fix": RHSPH,
    }
    out: dict[str, Any] = {"cpu_budget": cpu}
    excess: dict[str, float] = {}
    residual: dict[str, float] = {}
    for name, base in runs.items():
        if not hpg.fn(base, 37).exists():
            out[name] = {"available": False, "path": str(base)}
            continue
        budget = hpg.budget_between(CPU, 36, base, 37, depth=8)
        out[name] = budget
        excess[name] = float(budget["net_influx_pa_per_cell_h"] - cpu["net_influx_pa_per_cell_h"])
        residual[name] = float(budget["residual_pa_per_cell_h"])
    out["excess_outflux_pa_per_cell_h"] = excess
    out["residual_pa_per_cell_h"] = residual
    if "rhsph_fix" in excess:
        out["collapse_vs_old"] = float(1.0 - abs(excess["rhsph_fix"]) / max(abs(excess["old_ec4d6769"]), 1e-12))
        out["collapse_vs_hypso"] = float(1.0 - abs(excess["rhsph_fix"]) / max(abs(excess["hypso_3d0b439c"]), 1e-12))
        out["metrics_h37"] = hpg.field_metrics(RHSPH, 37)
        if hpg.fn(RHSPH, 38).exists():
            cpu38 = hpg.budget_between(CPU, 36, CPU, 38, depth=8)
            fix38 = hpg.budget_between(CPU, 36, RHSPH, 38, depth=8)
            out["h36_h38"] = {
                "cpu": cpu38,
                "rhsph_fix": fix38,
                "excess": float(fix38["net_influx_pa_per_cell_h"] - cpu38["net_influx_pa_per_cell_h"]),
            }
    existing = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    existing["hourly_gate"] = out
    hpg.write_json(Path(args.out), existing)
    print(json.dumps({
        "excess": excess, "residual": residual,
        "collapse_vs_old": out.get("collapse_vs_old"),
        "collapse_vs_hypso": out.get("collapse_vs_hypso"),
    }, indent=2))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_JSON))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    if args.gate:
        gate(args)
        return 0

    f = load_h36()
    nzp1, ny, nx = f["ph"].shape
    nz = nzp1 - 1

    ww_oracle = oracle_calc_ww_cp(f)
    terms = jax_terms(f)
    rom = terms["rom"]
    run_rhs = terms["run_rhs"]

    d1 = _split(rom - ww_oracle)
    scale1 = _split(ww_oracle)

    ph_tend_oracle = oracle_rhs_ph(f, ww_oracle, advective_order=5)
    ph_tend_jax_same_ww = run_rhs(ww_oracle)
    ph_tend_jax_prod = run_rhs(rom)
    # Faces 1..nz-1 only: the top face (nz) is gw+advection in open-top WRF but
    # identically zero under the JAX production top_lid (advance_w forces
    # rhs(nz)=0 and w(nz)=0), so it never enters the lid solve -- excluding it
    # isolates the dynamically active mismatch.
    sl = slice(1, nz)
    d2_op = _split((ph_tend_jax_same_ww - ph_tend_oracle)[sl])
    d2_lane = _split((ph_tend_jax_prod - ph_tend_oracle)[sl])
    scale2 = _split(ph_tend_oracle[sl])
    d2_top = _split((ph_tend_jax_same_ww - ph_tend_oracle)[nz][None])

    rw = oracle_rw_tend(f, ww_oracle, dt=18.0)

    # convert ph_tend operator error to an expected one-substep Delta-ph
    mass_f = f["c1f"][:, None, None] * f["mu_total"][None] + f["c2f"][:, None, None]
    dts_stage1 = 6.0  # dt/3 at dt=18
    dph_est = f["msfty"][None] * dts_stage1 * (ph_tend_jax_prod - ph_tend_oracle) / np.where(
        np.abs(mass_f) > 1e-12, mass_f, 1e-12
    )
    d2_dph = _split(dph_est[sl])

    # Port validation: the new production rhs_ph_wrf real-case branch vs the
    # oracle (open-top to compare ALL faces; lid semantics checked separately).
    import jax.numpy as jnp
    from gpuwrf.dynamics.core.rhs_ph import rhs_ph_wrf

    mu_pad_x = np.pad(f["mu_total"], ((0, 0), (1, 0)), mode="edge")
    muu_f = 0.5 * (mu_pad_x[:, 1:] + mu_pad_x[:, :-1])
    muu_f = np.concatenate([muu_f, f["mu_total"][:, -1:]], axis=1)
    mu_pad_y = np.pad(f["mu_total"], ((1, 0), (0, 0)), mode="edge")
    muv_f = 0.5 * (mu_pad_y[1:, :] + mu_pad_y[:-1, :])
    muv_f = np.concatenate([muv_f, f["mu_total"][-1:, :]], axis=0)

    def _run_port(top_lid: bool) -> np.ndarray:
        return np.asarray(
            rhs_ph_wrf(
                u=jnp.asarray(f["u"]), v=jnp.asarray(f["v"]), ww=jnp.asarray(ww_oracle),
                ph=jnp.asarray(f["ph"]), phb=jnp.asarray(f["phb"]), w=jnp.asarray(f["w"]),
                mut=jnp.asarray(f["mu_total"]), muu=jnp.asarray(muu_f), muv=jnp.asarray(muv_f),
                c1f=jnp.asarray(f["c1f"]), c2f=jnp.asarray(f["c2f"]),
                fnm=jnp.asarray(f["fnm"]), fnp=jnp.asarray(f["fnp"]), rdnw=jnp.asarray(f["rdnw"]),
                rdx=float(f["rdx"]), rdy=float(f["rdy"]), msfty=jnp.asarray(f["msfty"]),
                non_hydrostatic=True, advective_order=5, specified=True,
                msfux=jnp.asarray(f["msfux"]), msfvy=jnp.asarray(f["msfvy"]),
                cfn=float(f["cfn"]), cfn1=float(f["cfn1"]), top_lid=top_lid,
            )
        )

    port_open = _run_port(False)
    port_lid = _run_port(True)
    port_validation = {
        "open_top_vs_oracle_all_faces": _split(port_open - ph_tend_oracle),
        "oracle_scale_all_faces": _split(ph_tend_oracle),
        "lid_top_face_all_zero": bool(np.all(port_lid[nz] == 0.0)),
        "lid_interior_identical_to_open": float(np.abs((port_lid - port_open)[1:nz]).max()),
    }

    payload: dict[str, Any] = {
        "schema": "v014_switzerland_acoustic_continuation_terms",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": "WRF call 21601 == h36 wrfout (bit-identical state, prior native-face proof)",
        "advective_order": 5,
        "wrf_x_trims_note": _wrf_x_trims_note(),
        "D1_rom_vs_oracle_ww": {"err": d1, "oracle_scale": scale1},
        "D2_rhs_ph_operator_err_same_ww": {"err": d2_op, "oracle_scale": scale2},
        "D2_rhs_ph_lane_err_prod": {"err": d2_lane},
        "D2_top_face_err_excluded_from_above": d2_top,
        "D2_expected_stage1_dph_from_err": d2_dph,
        "measured_stage1_ph_incr_err_rmse_interior": 0.8758,
        "D3_rw_tend": {
            "vert_cfl_gt1_cells": int((rw["vert_cfl_max"] > 1.0).sum()),
            "wdamp_minus_pgbuoy": _split(rw["rw_with_wdamp"] - rw["rw_pg_buoy"]),
        },
        "port_validation_new_rhs_ph_real_case": port_validation,
    }
    hpg.write_json(Path(args.out), payload)
    print(json.dumps({
        "D1_rom_interior_rmse": d1["interior"].get("rmse"),
        "D1_oracle_ww_interior_rmse": scale1["interior"].get("rmse"),
        "D2_operator_interior_rmse": d2_op["interior"].get("rmse"),
        "D2_lane_interior_rmse": d2_lane["interior"].get("rmse"),
        "D2_oracle_ph_tend_interior_rmse": scale2["interior"].get("rmse"),
        "D2_expected_stage1_dph_interior_rmse": d2_dph["interior"].get("rmse"),
        "measured_stage1_ph_incr_err_rmse": 0.8758,
        "D3_vert_cfl_gt1_cells": int((rw["vert_cfl_max"] > 1.0).sum()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
