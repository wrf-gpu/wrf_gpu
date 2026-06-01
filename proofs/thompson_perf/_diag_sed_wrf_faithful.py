"""DIAGNOSTIC (not shipped): reproduce WRF's exact rain-sedimentation surface
accumulation on the oracle column using numpy, to isolate the two diagnosed
causes of the JAX +13% surface-precip excess:
  (1) FIXED NSED=64 substeps vs WRF adaptive nstep = MAX_k INT(DT/(dz/vt)+1)
  (2) NO surface threshold vs WRF gate rr(kts) > R1*1000 = 1e-9 kg/m3

This drives the SAME state the JAX kernel sees at the sedimentation operator
(after rain-evap) so the only variable is the sedimentation integration itself.
Run: PYTHONPATH=src JAX_PLATFORMS=cpu taskset -c 0-3 python3 this.py
"""
from __future__ import annotations
import sys
import numpy as np
import jax

sys.path.insert(0, "proofs/thompson_perf")
import gpuwrf.physics.thompson_column as tc
from precip_oracle_validate import build_state, _meta

R1 = 1.0e-12
R2 = 1.0e-6


def wrf_sed_rain(qr, Nr, vt_mass, vt_num, dz, rho, dt, nsed_max, *, use_threshold, use_adaptive):
    nz = qr.shape[0]
    rr = np.maximum(qr * rho, R1)
    nr = np.maximum(Nr * rho, R2)
    if use_adaptive:
        nstep = 0
        for k in range(nz):
            if max(vt_mass[k], vt_num[k]) > 1.0e-3:
                delta_tp = dz[k] / max(vt_mass[k], vt_num[k])
                nstep = max(nstep, int(dt / delta_tp + 1.0))
        nstep = max(nstep, 1)
    else:
        nstep = nsed_max
    onstep = 1.0 / nstep
    ppt = 0.0
    for _ in range(nstep):
        sed_r = vt_mass * rr
        k = nz - 1
        rr[k] = max(R1, rr[k] - sed_r[k] / dz[k] * dt * onstep)
        for k in range(nz - 2, -1, -1):
            rr[k] = max(R1, rr[k] + (sed_r[k + 1] - sed_r[k]) / dz[k] * dt * onstep)
        if (not use_threshold) or (rr[0] > R1 * 1000.0):
            ppt += sed_r[0] * dt * onstep
    return ppt, nstep


def main():
    jax.config.update("jax_enable_x64", True)
    ni, nk, nj = _meta()
    state = build_state("in", ni, nk, nj)
    captured = {}
    orig = tc._sedimentation

    def capture(s, dt):
        captured["state"] = s
        return orig(s, dt)

    tc._sedimentation = capture
    try:
        _out, _precip = tc._step_thompson_column_full_impl(state, 18.0, False)
    finally:
        tc._sedimentation = orig
    pre = captured["state"]

    vts = tc._fall_speeds(pre)
    vt_r_mass = np.asarray(vts[0], dtype=np.float64)
    vt_r_num = np.asarray(vts[1], dtype=np.float64)
    qr = np.asarray(pre.qr, dtype=np.float64)
    Nr = np.asarray(pre.Nr, dtype=np.float64)
    dz = np.maximum(np.asarray(pre.dz, dtype=np.float64), 1.0)
    rho = np.maximum(np.asarray(pre.rho, dtype=np.float64), R1)
    dt = 18.0
    ncol = qr.shape[0]

    wrf_total = 0.34667097590863705
    variants = {
        "fixed64_nothresh (current JAX)": dict(nsed_max=64, use_threshold=False, use_adaptive=False),
        "fixed64_thresh": dict(nsed_max=64, use_threshold=True, use_adaptive=False),
        "adaptive_nothresh": dict(nsed_max=64, use_threshold=False, use_adaptive=True),
        "adaptive_thresh (WRF-faithful)": dict(nsed_max=64, use_threshold=True, use_adaptive=True),
    }
    for name, kw in variants.items():
        tot = 0.0
        nsteps = []
        for c in range(ncol):
            ppt, nstep = wrf_sed_rain(qr[c].copy(), Nr[c].copy(), vt_r_mass[c], vt_r_num[c],
                                      dz[c], rho[c], dt, **kw)
            tot += ppt
            nsteps.append(nstep)
        print(f"{name:38s} total={tot:.6f} mm  ratio={tot/wrf_total:.4f}  nstep={sorted(set(nsteps))}")
    print(f"{'WRF oracle RAINNCV':38s} total={wrf_total:.6f} mm  ratio=1.0000")


if __name__ == "__main__":
    main()
