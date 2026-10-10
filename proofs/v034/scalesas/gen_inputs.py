"""Generate CU_SCALESAS oracle input columns (synthetic regimes + real PROD CPU-WRF columns).

Every value is float32-representable, written with ``repr`` of the float32 value, so the r4
(WRF REAL) and r8 oracle builds and the JAX port all see bit-identical inputs.

Usage: python gen_inputs.py <out_dir> [--real <wrfout>] [--seed N]
Writes <out_dir>/<name>.txt (oracle stdin) and <out_dir>/<name>_inputs.npz per configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

KX = 44
G = 9.81
RD = 287.0
CP = 1004.5
KAPPA = RD / CP
P0 = 1.0e5

CONFIGS = {
    # name: (DT, STEPCU, ITIMESTEP, DY, PGCON)
    "cfg_dt54_stepcu5": (54.0, 5, 10, 9000.0, 0.55),
    "cfg_dt18_stepcu1": (18.0, 1, 2, 3000.0, 0.55),
}
DX2D_CHOICES = np.array([2000.0, 3000.0, 4000.0, 6000.0, 9000.0, 12000.0, 15000.0, 27000.0])


def f32(x):
    return np.asarray(x, dtype=np.float32)


def qsat_tetens(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    es = np.minimum(es, 0.5 * p)
    return 0.622 * es / (p - es)


def synthetic_column(rng, kind):
    """One hydrostatic sounding (bottom-up mass levels) for a named regime."""

    ztop = rng.uniform(18000.0, 22000.0)
    expo = rng.uniform(1.6, 2.4) if kind == "lowbase" else rng.uniform(1.1, 1.35)
    zint = ztop * (np.arange(KX + 1) / KX) ** expo
    zmid = 0.5 * (zint[1:] + zint[:-1])
    dz = np.diff(zint)

    if kind == "cold":
        tsfc = rng.uniform(258.0, 276.0)
    elif kind == "tropical":
        tsfc = rng.uniform(298.0, 306.0)
    else:
        tsfc = rng.uniform(283.0, 304.0)
    psfc = rng.uniform(96000.0, 102500.0)
    zml = rng.uniform(30.0, 250.0) if kind == "lowbase" else rng.uniform(250.0, 2400.0)
    lapse = rng.uniform(1.5, 8.0) * 1e-3          # theta lapse K/m above ML
    ztp = rng.uniform(10500.0, 16500.0)
    rh_ml = rng.uniform(0.45, 0.99)
    rh_tr = rng.uniform(0.03, 0.85)
    inv = rng.uniform(0.0, 7.0) if rng.random() < 0.35 else 0.0
    if kind == "capped":
        inv = rng.uniform(1.0, 6.0)
    dry_layer = rng.random() < (0.9 if kind == "dry_mid" else 0.3)
    theta_sfc = tsfc * (P0 / psfc) ** KAPPA

    theta = np.where(zmid <= zml, theta_sfc, theta_sfc + inv + lapse * (zmid - zml))
    theta = np.where(zmid > ztp, theta_sfc + inv + lapse * (ztp - zml) + 0.02 * (zmid - ztp), theta)
    rh = np.where(zmid <= zml, rh_ml, rh_tr + (rh_ml - rh_tr) * np.exp(-(zmid - zml) / rng.uniform(1500.0, 6000.0)))
    if dry_layer:
        zd = rng.uniform(1500.0, 6000.0)
        rh = np.where(np.abs(zmid - zd) < rng.uniform(400.0, 1500.0), rh * rng.uniform(0.1, 0.6), rh)
    if kind in ("moist_neutral", "lowbase"):
        rh = np.maximum(rh, 0.9 if kind == "moist_neutral" else rng.uniform(0.6, 0.97))
    if kind == "lowbase":
        rh = np.where(zmid <= zml + 300.0, np.maximum(rh, rng.uniform(0.95, 1.0)), rh)
    rh = np.clip(rh, 0.01, 1.0)

    # Hydrostatic integration in Exner form with virtual theta (two passes for qv).
    qv = np.full(KX, 0.01)
    for _ in range(3):
        thv = theta * (1.0 + 0.608 * qv)
        pi_int = np.empty(KX + 1)
        pi_int[0] = (psfc / P0) ** KAPPA
        for k in range(KX):
            pi_int[k + 1] = pi_int[k] - G / (CP * thv[k]) * dz[k]
        pi_mid = 0.5 * (pi_int[1:] + pi_int[:-1])
        p_mid = P0 * pi_mid ** (1.0 / KAPPA)
        t = theta * pi_mid
        qv = np.maximum(rh * qsat_tetens(t, p_mid), 1.0e-8)
    p_int = P0 * pi_int ** (1.0 / KAPPA)

    qc = np.zeros(KX)
    qi = np.zeros(KX)
    if rng.random() < 0.4:
        cloudy = rh > 0.9
        qc = np.where(cloudy & (t > 253.0), rng.uniform(1e-5, 4e-4), 0.0)
        qi = np.where(cloudy & (t < 268.0), rng.uniform(1e-6, 8e-5), 0.0)
    rho = p_mid / (RD * t * (1.0 + 0.608 * qv))
    shear = rng.uniform(0.0, 35.0)
    u = rng.uniform(-8.0, 8.0) + shear * np.minimum(zmid, 12000.0) / 12000.0
    v = rng.uniform(-8.0, 8.0) + rng.uniform(-0.5, 0.5) * shear * np.minimum(zmid, 12000.0) / 12000.0
    wmax = rng.choice([rng.uniform(-0.12, 0.0), rng.uniform(0.0, 0.08), rng.uniform(0.05, 0.6)])
    w = wmax * np.exp(-(((zint - rng.uniform(800.0, 5000.0)) / rng.uniform(1500.0, 5000.0)) ** 2))
    w[0] = 0.0
    return dict(T=t, QV=qv, QC=qc, QI=qi, P=p_mid, PI=pi_mid, RHO=rho, DZ=dz, U=u, V=v, P8W=p_int, W=w)


def real_columns(path, rng, ncol):
    """Columns from an original CPU-WRF wrfout (PROD D5), WRF-consistent derivation."""

    from netCDF4 import Dataset

    with Dataset(path) as ds:
        p = ds["P"][0] + ds["PB"][0]
        th = ds["T"][0] + 300.0
        ph = (ds["PH"][0] + ds["PHB"][0]) / G
        qv = ds["QVAPOR"][0]
        qc = ds["QCLOUD"][0]
        qi = ds["QICE"][0] if "QICE" in ds.variables else np.zeros_like(qv)
        u = ds["U"][0]
        v = ds["V"][0]
        w = ds["W"][0]
        psfc = ds["PSFC"][0]
        xland = ds["XLAND"][0]
    nz, ny, nx = p.shape
    nz = min(nz, KX)
    cols = []
    # Bias the sample toward the most unstable/moist low levels (where SAS can trigger).
    score = (qv[0] * 2.5e6 + 1004.5 * th[0] * (p[0] / P0) ** KAPPA).ravel()
    order = np.argsort(score)[::-1]
    pick = np.concatenate([order[: ncol // 2], rng.choice(order[ncol // 2 :], ncol - ncol // 2, replace=False)])
    for flat in pick:
        j, i = divmod(int(flat), nx)
        pm = np.asarray(p[:nz, j, i], float)
        pim = (pm / P0) ** KAPPA
        t = np.asarray(th[:nz, j, i], float) * pim
        q = np.maximum(np.asarray(qv[:nz, j, i], float), 1e-10)
        zi = np.asarray(ph[: nz + 1, j, i], float)
        dz = np.diff(zi)
        um = 0.5 * (np.asarray(u[:nz, j, i], float) + np.asarray(u[:nz, j, i + 1], float))
        vm = 0.5 * (np.asarray(v[:nz, j, i], float) + np.asarray(v[:nz, j + 1, i], float))
        p8w = np.empty(nz + 1)
        p8w[0] = float(psfc[j, i])
        p8w[1:nz] = 0.5 * (pm[1:] + pm[:-1])
        p8w[nz] = max(pm[-1] - 0.5 * (pm[-2] - pm[-1]), 1000.0)
        cols.append(
            dict(T=t, QV=q, QC=np.maximum(np.asarray(qc[:nz, j, i], float), 0.0),
                 QI=np.maximum(np.asarray(qi[:nz, j, i], float), 0.0), P=pm, PI=pim,
                 RHO=pm / (RD * t * (1.0 + 0.608 * q)), DZ=dz, U=um, V=vm, P8W=p8w,
                 W=np.asarray(w[: nz + 1, j, i], float), XLAND=float(xland[j, i]))
        )
    return cols


FIELDS = ("T", "QV", "QC", "QI", "P", "PI", "RHO", "DZ", "U", "V")


def write_case(out_dir: Path, name: str, cols, xland, dx2d, cfg, kinds):
    dt, stepcu, itimestep, dy, pgcon = cfg
    n = len(cols)
    arr = {f: np.stack([f32(c[f]) for c in cols]) for f in FIELDS + ("P8W", "W")}
    arr["XLAND"] = f32(xland)
    arr["DX2D"] = f32(dx2d)
    lines = [f"{n} {KX}", f"{float(f32(dt))!r} {stepcu} {itimestep} {float(f32(dy))!r} {float(f32(pgcon))!r}"]
    fmt = lambda a: " ".join(repr(float(x)) for x in a)  # noqa: E731
    for i in range(n):
        lines.append(f"{float(arr['XLAND'][i])!r} {float(arr['DX2D'][i])!r}")
        for f in FIELDS + ("P8W", "W"):
            lines.append(fmt(arr[f][i]))
    (out_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
    np.savez_compressed(out_dir / f"{name}_inputs.npz", DT=np.float32(dt), STEPCU=np.int32(stepcu),
                        ITIMESTEP=np.int32(itimestep), DY=np.float32(dy), PGCON=np.float32(pgcon),
                        KIND=np.array(kinds), **arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--real", default=None)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--nsyn", type=int, default=320)
    ap.add_argument("--nreal", type=int, default=64)
    ap.add_argument("--kinds", default="generic,tropical,moist_neutral,cold")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    kinds_cycle = args.kinds.split(",")
    cols, kinds, xland, dx2d = [], [], [], []
    for n in range(args.nsyn):
        kind = kinds_cycle[n % len(kinds_cycle)] if (n % 5 or args.prefix) else "tropical"
        cols.append(synthetic_column(rng, kind))
        kinds.append(kind)
        xland.append(1.0 if rng.random() < 0.55 else 2.0)
        dx2d.append(float(rng.choice(DX2D_CHOICES)))
    if args.real:
        for c in real_columns(args.real, rng, args.nreal):
            cols.append(c)
            kinds.append("real_prod")
            xland.append(c["XLAND"])
            dx2d.append(float(rng.choice(DX2D_CHOICES)))
    for name, cfg in CONFIGS.items():
        write_case(out, args.prefix + name, cols, xland, dx2d, cfg, kinds)
    print(f"wrote {len(cols)} columns x {len(CONFIGS)} configs to {out}")


if __name__ == "__main__":
    main()
