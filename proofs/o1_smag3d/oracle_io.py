"""o1-smag3d oracle harness: real-structured WRF crop -> pristine driver -> arrays.

``build_case`` crops a CPU-WRF wrfout (Swiss Alps 42x42x44) into a stand-alone
specified (or periodic) domain and derives the phy_prep-style fields the WRF
diff_opt=2 routines read.  ``run_oracle`` writes the Fortran stream input, runs
``smag3d_driver.exe`` (pristine libwrflib.a) and returns the full memory arrays
``(k, j, i)`` with the driver halo (5).  Everything is REAL4.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

SWISS_DIR = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
ORACLE_EXE = Path(os.environ.get("O1_SMAG3D_ORACLE", "<USER_HOME>/wrf_gpu2_lanes/o1-smag3d/oracle/smag3d_driver.exe"))
DRIVER_HALO = 5
OUT_FIELDS = ("z", "rdz", "rdzw", "zx", "zy", "div", "defor11", "defor22", "defor33", "defor12", "defor13",
              "defor23", "bn2", "xkmh", "xkmv", "xkhh", "xkhv", "ru_tendf", "rv_tendf", "rw_tendf", "t_tendf")
MOIST_NAMES = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW")  # WRF slots 2..6 (p_qv,p_qc,p_qr,p_qi,p_qs)
R_D, CP = np.float32(287.0), np.float32(np.float32(7.0) * np.float32(287.0) / np.float32(2.0))


def _f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def build_case(wrfout: str | Path, i0: int, j0: int, ncx: int, ncy: int, *, wind_scale: float = 1.0,
               theta_pert: float = 0.0, seed: int = 0) -> dict:
    """Crop ``[j0:j0+ncy, i0:i0+ncx]`` mass columns (all levels) of a wrfout.

    ``wind_scale`` multiplies u/v/w (real structure, stronger deformation: drives
    the unstable/cap branches); ``theta_pert`` adds seeded noise (K) to theta.
    """
    import netCDF4

    ds = netCDF4.Dataset(str(wrfout))
    g = lambda n: np.asarray(ds.variables[n][0], dtype=np.float32)  # noqa: E731
    js, is_ = slice(j0, j0 + ncy), slice(i0, i0 + ncx)
    jsv, isu = slice(j0, j0 + ncy + 1), slice(i0, i0 + ncx + 1)
    rng = np.random.default_rng(seed)
    u = g("U")[:, js, isu] * np.float32(wind_scale)
    v = g("V")[:, jsv, is_] * np.float32(wind_scale)
    w = g("W")[:, js, is_] * np.float32(wind_scale)
    thp = g("THM")[:, js, is_]
    th_dry = g("T")[:, js, is_] + np.float32(300.0)
    if theta_pert:
        noise = rng.standard_normal(thp.shape).astype(np.float32) * np.float32(theta_pert)
        thp = thp + noise
        th_dry = th_dry + noise
    p = g("P")[:, js, is_] + g("PB")[:, js, is_]
    t = th_dry * (p / np.float32(1.0e5)) ** (R_D / CP)
    moist = [g(n)[:, js, is_] for n in MOIST_NAMES]
    qv = moist[0]
    ph, phb = g("PH")[:, js, is_], g("PHB")[:, js, is_]
    fnm, fnp = g("FNM"), g("FNP")
    cf1, cf2, cf3 = (float(ds.variables[n][0]) for n in ("CF1", "CF2", "CF3"))
    nz = u.shape[0]
    # phy_prep-style face pressure/temperature (interior fnm/fnp, cf surface, linear top)
    def faces(a):
        out = np.empty((nz + 1,) + a.shape[1:], dtype=np.float32)
        out[1:nz] = fnm[1:nz, None, None] * a[1:nz] + fnp[1:nz, None, None] * a[0:nz - 1]
        out[0] = np.float32(cf1) * a[0] + np.float32(cf2) * a[1] + np.float32(cf3) * a[2]
        out[nz] = a[nz - 1] + (a[nz - 1] - a[nz - 2]) * np.float32(0.5)
        return out
    rho = p / (R_D * t * (np.float32(1.0) + np.float32(0.608) * qv))
    case = dict(
        u=u, v=v, w=w, thp=thp, th_phy=th_dry, t_phy=t, p_phy=p, p8w=faces(p), t8w=faces(t), ph=ph, phb=phb,
        rho=rho.astype(np.float32), moist=moist,
        msftx=g("MAPFAC_MX")[js, is_], msfty=g("MAPFAC_MY")[js, is_],
        msfux=g("MAPFAC_UX")[js, isu], msfuy=g("MAPFAC_UY")[js, isu],
        msfvx=g("MAPFAC_VX")[jsv, is_], msfvy=g("MAPFAC_VY")[jsv, is_],
        ust=g("UST")[js, is_], hfx=g("HFX")[js, is_], qfx=g("QFX")[js, is_],
        fnm=np.append(fnm, 0.0).astype(np.float32), fnp=np.append(fnp, 0.0).astype(np.float32),
        dn=np.append(g("DN"), 0.0).astype(np.float32), dnw=np.append(g("DNW"), 0.0).astype(np.float32),
        cf1=cf1, cf2=cf2, cf3=cf3, rdx=float(ds.variables["RDX"][0]), rdy=float(ds.variables["RDY"][0]),
        dx=float(ds.DX), dy=float(ds.DY), dt=float(ds.DT),
        tke=np.maximum(np.float32(0.5) * g("QKE")[:, js, is_], np.float32(0.0)) if "QKE" in ds.variables
        else np.full((nz,) + ph.shape[1:], np.float32(0.1)),
        mut=g("MU")[js, is_] + g("MUB")[js, is_],
        c1h=np.append(g("C1H"), 0.0).astype(np.float32), c2h=np.append(g("C2H"), 0.0).astype(np.float32),
        u_base=np.zeros(nz + 1, np.float32), v_base=np.zeros(nz + 1, np.float32),
        t_base=np.zeros(nz + 1, np.float32), qv_base=np.zeros(nz + 1, np.float32),
    )
    ds.close()
    return {k: (_f32(v) if isinstance(v, np.ndarray) else ([_f32(x) for x in v] if k == "moist" else v))
            for k, v in case.items()}


def _fo(a):
    """Port (k, j, i) -> Fortran (i, k, j) column-major bytes."""
    return np.asarray(a, dtype=np.float32).transpose(2, 0, 1).tobytes(order="F")


def write_input(case: dict, path: str | Path, cfg: dict) -> None:
    nz, ny, nx1 = case["u"].shape
    hdr = np.zeros(16, np.int32)
    hdr[:15] = [nx1 - 1, ny, nz, 1 + len(case["moist"]), 2 if cfg.get("nested") else (0 if cfg.get("periodic") else 1),
                int(bool(cfg.get("periodic"))), int(bool(cfg.get("periodic"))), int(cfg.get("isotropic", 0)),
                int(cfg.get("isfflx", 1)), 0 if cfg.get("vertical") else 5, int(cfg.get("use_theta_m", 1)),
                int(bool(cfg.get("mix_full_fields", False))), int(cfg.get("km_opt", 3)),
                int(bool(cfg.get("moist_mix2_off", False))), int(bool(cfg.get("tke_mix2_off", False)))]
    rh = np.zeros(16, np.float32)
    rh[:13] = [case["dx"], case["dy"], case["dt"], case["rdx"], case["rdy"], case["cf1"], case["cf2"], case["cf3"],
               cfg.get("c_s", 0.25), cfg.get("mix_upper_bound", 0.1), cfg.get("tke_drag_coefficient", 0.0),
               cfg.get("tke_heat_flux", 0.0), cfg.get("c_k", 0.15)]
    with open(path, "wb") as f:
        f.write(hdr.tobytes()); f.write(rh.tobytes())
        for name in ("u", "v", "w", "thp", "th_phy", "t_phy", "p_phy", "p8w", "t8w", "ph", "phb", "rho"):
            f.write(_fo(case[name]))
        for q in case["moist"]:
            f.write(_fo(q))
        for name in ("msftx", "msfty", "msfux", "msfuy", "msfvx", "msfvy", "ust", "hfx", "qfx"):
            f.write(np.asarray(case[name], np.float32).T.tobytes(order="F"))
        for name in ("fnm", "fnp", "dn", "dnw", "u_base", "v_base", "t_base", "qv_base"):
            f.write(np.asarray(case[name], np.float32).tobytes())
        if int(cfg.get("km_opt", 3)) == 2:
            f.write(_fo(case["tke"]))
            f.write(np.asarray(case["mut"], np.float32).T.tobytes(order="F"))
            f.write(np.asarray(case["c1h"], np.float32).tobytes())
            f.write(np.asarray(case["c2h"], np.float32).tobytes())


def read_output(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    ims, ime, jms, jme, kms, kme, n_moist = np.frombuffer(raw[:28], np.int32)
    ni, nj, nk = ime - ims + 1, jme - jms + 1, kme - kms + 1
    n3 = ni * nj * nk
    data = np.frombuffer(raw[28:], np.float32)
    out = {}
    for idx, name in enumerate(OUT_FIELDS):
        out[name] = data[idx * n3:(idx + 1) * n3].reshape((ni, nk, nj), order="F").transpose(1, 2, 0).copy()
    base = len(OUT_FIELDS) * n3
    moist = data[base:base + n3 * n_moist].reshape((ni, nk, nj, n_moist), order="F")
    out["moist_tendf"] = [moist[..., m].transpose(1, 2, 0).copy() for m in range(1, n_moist)]
    tke = data[base + n3 * n_moist:base + n3 * (n_moist + 1)]
    out["tke_tendf"] = tke.reshape((ni, nk, nj), order="F").transpose(1, 2, 0).copy()
    out["_bounds"] = (int(ims), int(ime), int(jms), int(jme))
    return out


def run_oracle(case: dict, cfg: dict, workdir: str | Path, tag: str) -> dict:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    fin, fout = workdir / f"{tag}.in.bin", workdir / f"{tag}.out.bin"
    write_input(case, fin, cfg)
    res = subprocess.run([str(ORACLE_EXE), str(fin), str(fout)], capture_output=True, text=True, timeout=300)
    if res.returncode != 0 or "ORACLE OK" not in res.stdout:
        raise RuntimeError(f"oracle failed rc={res.returncode}: {res.stdout[-2000:]} {res.stderr[-2000:]}")
    out = read_output(fout)
    fin.unlink(); fout.unlink()
    return out


def crop_to_halo(arr: np.ndarray, halo: int) -> np.ndarray:
    """Drop the driver's extra halo so the array matches a ``halo``-padded layout."""
    d = DRIVER_HALO - halo
    return arr[:, d:arr.shape[1] - d, d:arr.shape[2] - d]
