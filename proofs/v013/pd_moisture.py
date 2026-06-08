"""Proof: positive-definite / monotonic limiter extended to MOISTURE species.

v0.13 Tier-2 sprint -- extend the v0.12.0 theta-only PD (``scalar_adv_opt``=1) /
monotonic (=2) flux-renormalization limiter to ALL advected moisture species
(qv, qc, qr, qi, qs, qg), selected by ``moist_adv_opt``.  WRF runs the SAME
``advect_scalar`` / ``advect_scalar_pd`` / ``advect_scalar_mono`` routines per
moist species (``solve_em.F:2282-2408`` ``moist_variable_loop`` ->
``rk_scalar_tend(..., config_flags%moist_adv_opt)``).

This script regenerates ``proofs/v013/pd_moisture.json`` with three mandatory
gates plus supporting checks, all on CPU / fp64:

  (1) DEFAULT UNCHANGED -- moist_adv_opt=0 per-species coupled tendency is
      BYTE-IDENTICAL to the plain ``advect_scalar_flux`` path (opt-in only); the
      limiter is also byte-identical to plain on non-final RK stages.
  (2) POSITIVITY -- moist_adv_opt=1 keeps every moisture species >= 0 through a
      multi-step forward integration on a sharp-gradient blob, where the
      UNLIMITED h5/v3 scheme drives unphysical NEGATIVE mixing ratios.
  (2b) MONOTONICITY -- moist_adv_opt=2 introduces NO new per-species extrema.
  (3) WRF-PARITY -- (a) the multi-species loop reduces EXACTLY (bit) to the
      per-field ``advect_scalar_flux_limited`` already validated vs the
      independent WRF-Fortran advect_scalar_pd/_mono transcription to round-off;
      and (b) a direct WRF-Fortran advect_scalar_pd x-row transcription matches
      the loop output on a moisture mixing-ratio profile to round-off.
  (4) CONSERVATION -- each species' single-step coupled-mass tendency telescopes
      to ~0 (the limiter only redistributes mass).

CPU-jax dev path: JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 \
    python proofs/v013/pd_moisture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from gpuwrf.dynamics.flux_advection import (
    CoupledVelocities,
    advect_moisture_scalars,
    advect_scalar_flux,
    advect_scalar_flux_limited,
)

PROOF_DIR = Path(__file__).resolve().parent
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")


def _setup(nz, ny, nx, *, u_const, v_const, dx, dt, rom_const=0.0):
    mu = jnp.ones((ny, nx))
    c1 = jnp.ones((nz,))
    c2 = jnp.zeros((nz,))
    ru = jnp.full((nz, ny, nx), float(u_const))
    rv = jnp.full((nz, ny, nx), float(v_const))
    rom = jnp.zeros((nz + 1, ny, nx)).at[1:nz, :, :].set(float(rom_const))
    vel = CoupledVelocities(ru=ru, rv=rv, rom=rom)
    return dict(
        mu=mu, c1=c1, c2=c2, vel=vel, rdx=1.0 / dx, rdy=1.0 / dx,
        rdzw=jnp.ones((nz,)), fzm=jnp.full((nz,), 0.5), fzp=jnp.full((nz,), 0.5), dt=dt,
    )


def _sharp_blob(nz, ny, nx, amp):
    f = np.zeros((nz, ny, nx))
    f[:, ny // 2 - 2 : ny // 2 + 2, nx // 2 - 2 : nx // 2 + 2] = amp
    return jnp.asarray(f)


def _moisture(nz, ny, nx):
    xs = np.arange(nx)
    qv = 0.012 + 0.004 * np.where(np.abs(xs - nx / 2) <= 3.0, 1.0, 0.0)[None, None, :] * np.ones((nz, ny, nx))
    amps = (None, 1.0e-3, 5.0e-4, 2.0e-4, 3.0e-4, 1.0e-4)  # qv handled above
    fields = [jnp.asarray(qv)]
    for amp in amps[1:]:
        fields.append(_sharp_blob(nz, ny, nx, amp))
    return tuple(fields)


def _loop(fields, fields_old, s, *, opt, final):
    return advect_moisture_scalars(
        fields, fields_old, s["vel"], moist_adv_opt=opt, is_final_rk_stage=final,
        mut=s["mu"], mu_old=s["mu"], c1=s["c1"], c2=s["c2"], rdx=s["rdx"], rdy=s["rdy"],
        rdzw=s["rdzw"], fzm=s["fzm"], fzp=s["fzp"], dt=s["dt"],
    )


def _plain(q, s):
    return advect_scalar_flux(
        q, s["vel"], mut=s["mu"], c1=s["c1"], rdx=s["rdx"], rdy=s["rdy"],
        rdzw=s["rdzw"], fzm=s["fzm"], fzp=s["fzp"],
    )


# --- WRF-Fortran advect_scalar_pd x-row transcription (independent oracle) ----
def _np_flux_upwind(qm1, q, cr):
    return 0.5 * np.minimum(1.0, cr + np.abs(cr)) * qm1 + 0.5 * np.maximum(-1.0, cr - np.abs(cr)) * q


def _np_flux5(qm3, qm2, qm1, q, qp1, qp2, ua):
    f6 = (37.0 / 60.0) * (q + qm1) - (2.0 / 15.0) * (qp1 + qm2) + (1.0 / 60.0) * (qp2 + qm3)
    return f6 - np.sign(ua) * (1.0 / 60.0) * ((qp2 - qm3) - 5.0 * (qp1 - qm2) + 10.0 * (q - qm1))


def _wrf_pd_x_1d(field, field_old, ru, mu, dx, dt, eps=1e-20):
    R = lambda a, sh: np.roll(a, sh)
    cr = ru * dt / dx / mu
    fqxl = mu * (dx / dt) * _np_flux_upwind(R(field_old, 1), field_old, cr)
    hi = ru * _np_flux5(R(field, 3), R(field, 2), R(field, 1), field, R(field, -1), R(field, -2), ru)
    fqx = hi - fqxl
    rdx = 1.0 / dx
    ph_low = mu * field_old - dt * (rdx * (R(fqxl, -1) - fqxl))
    flux_out = dt * (rdx * (np.maximum(0.0, R(fqx, -1)) - np.minimum(0.0, fqx)))
    scale = np.where(flux_out > ph_low, np.maximum(0.0, ph_low / (flux_out + eps)), 1.0)
    fqx_lim = np.where(fqx > 0.0, R(scale, 1) * fqx, np.where(fqx < 0.0, scale * fqx, fqx))
    tot = fqx_lim + fqxl
    return -rdx * (R(tot, -1) - tot)


def main() -> int:
    results: dict = {}
    nz, ny, nx = 4, 32, 32
    dx, dt = 1000.0, 5.0
    s = _setup(nz, ny, nx, u_const=20.0, v_const=10.0, dx=dx, dt=dt, rom_const=0.0)
    fields = _moisture(nz, ny, nx)

    # (1) DEFAULT UNCHANGED: opt=0 byte-identical to plain; limiter inactive on
    # non-final stages.
    opt0_bytes, nonfinal_bytes = True, True
    for final in (False, True):
        out0 = _loop(fields, fields, s, opt=0, final=final)
        for q, tend in zip(fields, out0):
            opt0_bytes = opt0_bytes and bool(np.array_equal(np.asarray(tend), np.asarray(_plain(q, s))))
    base_final = _loop(fields, fields, s, opt=0, final=True)
    for opt in (1, 2):
        out = _loop(fields, fields, s, opt=opt, final=False)
        for a, b in zip(out, base_final):
            nonfinal_bytes = nonfinal_bytes and bool(np.array_equal(np.asarray(a), np.asarray(b)))
    results["default_unchanged"] = {
        "opt0_byte_identical_to_plain_all_species": opt0_bytes,
        "limiter_inactive_on_nonfinal_rk_stage": nonfinal_bytes,
        "passed": bool(opt0_bytes and nonfinal_bytes),
    }

    # (2) POSITIVITY (opt=1) vs plain undershoot.
    fpd = [np.asarray(q, dtype=np.float64) for q in fields]
    min_pd = [float("inf")] * len(fields)
    for _ in range(40):
        tends = _loop(tuple(jnp.asarray(q) for q in fpd), tuple(jnp.asarray(q) for q in fpd), s, opt=1, final=True)
        fpd = [q + dt * np.asarray(t) for q, t in zip(fpd, tends)]
        for i, q in enumerate(fpd):
            min_pd[i] = min(min_pd[i], float(q.min()))
    # plain undershoot on a condensate species (qc).
    qc_plain = np.asarray(fields[1], dtype=np.float64)
    min_plain = float("inf")
    for _ in range(40):
        qc_plain = qc_plain + dt * np.asarray(_plain(jnp.asarray(qc_plain), s))
        min_plain = min(min_plain, float(qc_plain.min()))
    results["positivity_opt1"] = {
        "per_species_min": {sp: m for sp, m in zip(SPECIES, min_pd)},
        "plain_qc_min": min_plain,
        "passed": bool(all(m >= -1e-12 for m in min_pd) and min_plain < -1e-9),
    }

    # (2b) MONOTONICITY (opt=2): no new per-species extrema.
    fmo = [np.asarray(q, dtype=np.float64) for q in fields]
    init_min = [float(q.min()) for q in fmo]
    init_max = [float(q.max()) for q in fmo]
    min_mo, max_mo = list(init_min), list(init_max)
    for _ in range(40):
        tends = _loop(tuple(jnp.asarray(q) for q in fmo), tuple(jnp.asarray(q) for q in fmo), s, opt=2, final=True)
        fmo = [q + dt * np.asarray(t) for q, t in zip(fmo, tends)]
        for i, q in enumerate(fmo):
            min_mo[i] = min(min_mo[i], float(q.min()))
            max_mo[i] = max(max_mo[i], float(q.max()))
    mono_ok = all(min_mo[i] >= init_min[i] - 1e-12 and max_mo[i] <= init_max[i] + 1e-12 for i in range(len(fmo)))
    results["monotonicity_opt2"] = {
        "per_species_min": {sp: m for sp, m in zip(SPECIES, min_mo)},
        "per_species_max": {sp: m for sp, m in zip(SPECIES, max_mo)},
        "passed": bool(mono_ok),
    }

    # (3a) WRF-PARITY via exact reduction to the single-scalar limiter (the theta
    # test pins advect_scalar_flux_limited to the WRF-Fortran transcription to
    # round-off; here we prove the loop is BIT-EXACT to that per-species call,
    # including a non-zero vertical flux).
    s3 = _setup(6, 12, 24, u_const=15.0, v_const=8.0, dx=1000.0, dt=4.0, rom_const=0.03)
    f3 = _moisture(6, 12, 24)
    loop_reduction_max = 0.0
    for opt in (1, 2):
        out = _loop(f3, f3, s3, opt=opt, final=True)
        for q, tend in zip(f3, out):
            ref = advect_scalar_flux_limited(
                q, q, s3["vel"], scalar_adv_opt=opt, mut=s3["mu"], mu_old=s3["mu"],
                c1=s3["c1"], c2=s3["c2"], rdx=s3["rdx"], rdy=s3["rdy"], rdzw=s3["rdzw"],
                fzm=s3["fzm"], fzp=s3["fzp"], dt=s3["dt"],
            )
            loop_reduction_max = max(loop_reduction_max, float(jnp.max(jnp.abs(tend - ref))))

    # (3b) Direct WRF-Fortran PD x-row transcription parity on a moisture profile.
    nxp = 24
    dxp, dtp = 1.0, 0.2
    q1 = np.zeros(nxp)
    q1[10:13] = 1.5e-3
    ru = np.full(nxp, 1.5)
    tw = _wrf_pd_x_1d(q1, q1, ru, 1.0, dxp, dtp)
    velp = CoupledVelocities(ru=jnp.asarray(ru)[None, None, :], rv=jnp.zeros((1, 1, nxp)), rom=jnp.zeros((2, 1, nxp)))
    outp = advect_moisture_scalars(
        (jnp.asarray(q1)[None, None, :],), (jnp.asarray(q1)[None, None, :],), velp,
        moist_adv_opt=1, is_final_rk_stage=True, mut=jnp.ones((1, nxp)), mu_old=jnp.ones((1, nxp)),
        c1=jnp.ones(1), c2=jnp.zeros(1), rdx=1.0 / dxp, rdy=1.0, rdzw=jnp.ones(1),
        fzm=jnp.full(1, 0.5), fzp=jnp.full(1, 0.5), dt=dtp,
    )
    transcription_max = float(jnp.max(jnp.abs(jnp.asarray(outp[0])[0, 0, :] - tw)))
    results["wrf_parity"] = {
        "loop_vs_single_scalar_limiter_max_abs_diff": loop_reduction_max,
        "wrf_fortran_pd_xrow_transcription_max_abs_diff": transcription_max,
        "passed": bool(loop_reduction_max == 0.0 and transcription_max < 1e-13),
    }

    # (4) per-species single-step conservation.
    cons = {}
    for opt in (1, 2):
        tends = _loop(fields, fields, s, opt=opt, final=True)
        per = {}
        for sp, tend in zip(SPECIES, tends):
            total = float(jnp.sum(tend))
            scale = float(jnp.sum(jnp.abs(tend))) + 1e-30
            per[sp] = abs(total) / scale
        cons[f"opt{opt}"] = {"per_species_rel_total_tend": per, "passed": all(v < 1e-12 for v in per.values())}
    results["conservation"] = cons

    all_passed = bool(
        results["default_unchanged"]["passed"]
        and results["positivity_opt1"]["passed"]
        and results["monotonicity_opt2"]["passed"]
        and results["wrf_parity"]["passed"]
        and all(c["passed"] for c in cons.values())
    )
    results["species"] = list(SPECIES)
    results["verdict"] = "PASS" if all_passed else "FAIL"

    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROOF_DIR / "pd_moisture.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nWROTE {out_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
