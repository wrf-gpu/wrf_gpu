"""Validate the faithful NumPy ref against the byte-identical Fortran oracle.

Drives both with the SAME column inputs, prints per-field max-abs and max-rel
residual. The fp64 ref vs the fp64 (-fdefault-real-8) oracle should agree to
near fp64 roundoff IF the transcription is faithful; vs the fp32 oracle the floor
is fp32 roundoff (~1e-4 rel). This is a transcription check, NOT the production
parity test.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mynn_faithful_ref import sfclay1d_mynn  # noqa: E402

# Oracle output column order (must match driver.f90 write list):
OUT_COLS = [
    "ust", "mol", "rmol", "zol", "regime", "psim", "psih", "br",
    "flhc", "flqc", "hfx", "qfx", "lh", "qsfc", "qgh",
    "chs", "chs2", "cqs2", "ch", "wspd", "gz1oz0",
    "u10", "v10", "th2", "t2", "q2", "cpm", "wstar", "qstar", "znt",
]

# Input column order (must match driver.f90 read list):
IN_COLS = [
    "u", "v", "t1d", "qv", "p1d", "dz8w", "rho", "u1d2", "v1d2", "dz2w",
    "mavail", "pblh", "xland", "tsk", "psfcpa", "qcg", "snowh",
    "znt", "ust", "mol", "qsfc", "hfx", "qfx",
]


def run_oracle(cases, exe, itimestep=2, isfflx=1, isftcflx=0, iz0tlnd=0, spp_pbl=0, dx=3000.0):
    n = len(cases)
    lines = [f"{n} {itimestep} {isfflx} {isftcflx} {iz0tlnd} {spp_pbl} {dx}"]
    for c in cases:
        lines.append(" ".join(repr(float(c[k])) for k in IN_COLS))
    proc = subprocess.run(
        ["taskset", "-c", "0-3", os.path.join(HERE, exe)],
        input="\n".join(lines) + "\n", capture_output=True, text=True, check=True,
    )
    rows = [ln for ln in proc.stdout.splitlines() if ln and not ln.startswith("#")]
    out = {k: np.zeros(n) for k in OUT_COLS}
    for r in rows:
        parts = r.split()
        i = int(parts[0]) - 1
        for j, k in enumerate(OUT_COLS):
            out[k][i] = float(parts[1 + j])
    return out


def ref_inputs(cases):
    inp = {k: np.array([float(c[k]) for c in cases]) for k in IN_COLS}
    inp["t1d"] = inp.pop("t1d")
    inp["dx"] = np.full(len(cases), 3000.0)
    return inp


CASES = [
    # daytime unstable land (the crux): warm skin, light wind
    dict(u=3.0, v=2.0, t1d=298.0, qv=0.010, p1d=95000.0, dz8w=51.4, rho=1.10, u1d2=4.0, v1d2=3.0, dz2w=100.0,
         mavail=0.30, pblh=1200.0, xland=1.0, tsk=312.0, psfcpa=95200.0, qcg=0.0, snowh=0.0,
         znt=0.10, ust=0.30, mol=-0.20, qsfc=-1.0, hfx=200.0, qfx=0.0001),
    # stable night land
    dict(u=1.5, v=0.8, t1d=288.0, qv=0.008, p1d=95000.0, dz8w=51.4, rho=1.15, u1d2=2.0, v1d2=1.0, dz2w=100.0,
         mavail=0.30, pblh=200.0, xland=1.0, tsk=285.0, psfcpa=95200.0, qcg=0.0, snowh=0.0,
         znt=0.10, ust=0.15, mol=0.05, qsfc=-1.0, hfx=-20.0, qfx=0.00001),
    # near-neutral land
    dict(u=5.0, v=5.0, t1d=295.0, qv=0.009, p1d=95000.0, dz8w=51.4, rho=1.12, u1d2=6.0, v1d2=6.0, dz2w=100.0,
         mavail=0.30, pblh=800.0, xland=1.0, tsk=295.05, psfcpa=95200.0, qcg=0.0, snowh=0.0,
         znt=0.10, ust=0.25, mol=0.0, qsfc=-1.0, hfx=0.0, qfx=0.0),
    # daytime water
    dict(u=8.0, v=6.0, t1d=293.0, qv=0.013, p1d=101000.0, dz8w=51.4, rho=1.18, u1d2=9.0, v1d2=7.0, dz2w=100.0,
         mavail=1.0, pblh=600.0, xland=2.0, tsk=294.0, psfcpa=101100.0, qcg=0.0, snowh=0.0,
         znt=0.0028, ust=0.26, mol=-0.05, qsfc=-1.0, hfx=50.0, qfx=0.0001),
]


def main():
    for exe, label, rtol in [("mynn_oracle_r8", "fp64-oracle", 1e-6), ("mynn_oracle", "fp32-oracle", 3e-3)]:
        orc = run_oracle(CASES, exe)
        ref = sfclay1d_mynn(ref_inputs(CASES))
        print(f"\n=== faithful NumPy ref vs {label} ===")
        worst = 0.0
        for k in OUT_COLS:
            a = np.asarray(ref[k]); b = orc[k]
            denom = np.maximum(np.abs(b), 1e-12)
            rel = np.max(np.abs(a - b) / denom)
            absd = np.max(np.abs(a - b))
            flag = "" if rel <= rtol or absd < 1e-9 else "  <-- DIVERGE"
            worst = max(worst, rel if absd >= 1e-9 else 0.0)
            print(f"  {k:8s} absmax={absd:12.4e} relmax={rel:12.4e}{flag}")
        print(f"  WORST rel (excl. <1e-9 abs): {worst:.4e}")


if __name__ == "__main__":
    main()
