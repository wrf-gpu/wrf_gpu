"""RE01: build the pristine-RRTMG caller inputs for the census picks (WRF phy_prep + radiation_driver wiring).

Per column (CPU-WRF WN3 0227 history at tau, read-only):
  phy_prep: p_phy=P+PB, pi=(p/p1000)**rcp, th_phy=(THM+t0)/(1+rvovrd*qv), t_phy=th_phy*pi,
            alt = EOS(THM, p) [calc_p_rho_phi], rho=(1+qv)/alt, z_w=(PH+PHB)/g, dz8w=dz (top 0),
            p8w/t8w: fzm/fzp (FNM/FNP) interior, z-weighted bottom, log-p top (module_big_step_utilities_em phy_prep).
  MP state for calc_effectRad: th_phy, p_phy, qv, QCLOUD, QICE, QNICE, QSNOW (Nc = Nt_c/rho inside the driver).
  radiation_driver (icloud=1, icloud_bl=1, itimestep>1): CLDFRA = history CLDFRA (= CLDFRA_BL of the last call);
            qc_rad = qc + QC_BL where qc<1e-6 and CLDFRA_BL>0.001 (QC_BL/CLDFRA_BL from the GPU W9 frame at the same
            tau, real-structured operand, only tau<=24; QI_BL is not in history -> qi_rad = qi).
  radiation_driver operands (first_rk_step_part1:264-330): P = p_hyd, P8W = p_hyd_w (hydrostatic, phy_prep REAL
            recursion), PI = pi_phy, T = t_phy, T8W = t8w, RHO = grid%rho, DZ8W = dz8w.  MP (calc_effectRad) gets P = p_phy.
  O3: o3rad is rdf=(p2c) and is computed on d01 only (EM_CORE) -> the driver interpolates CAM ozone on the d01
            PARENT column (its own p_hyd) and copies it level-wise to the nest column.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

CPU = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
GPU = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/W9/finalb_r3/20260227_18z_a1/gpu_24h")
NML = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/cases/20260227_18z_a1/namelist.input")
R_D, CP, R_V, G, P1000, T0 = 287.0, 7.0 * 287.0 / 2.0, 461.6, 9.81, 1.0e5, 300.0
RCP, RVOVRD, CVPM = R_D / CP, R_V / R_D, -(CP - R_D) / CP
f4 = np.float32


def namelist_ints(key):
    for line in NML.read_text().splitlines():
        if line.strip().startswith(key):
            return [int(x) for x in line.split("=")[1].replace(",", " ").split()]
    raise KeyError(key)


def parent_cell(dom, j, i):
    """Child mass cell (0-based) -> d01 mass cell (0-based) through the nest chain (p2c: containing parent cell)."""
    ips, jps, ratio = namelist_ints("i_parent_start"), namelist_ints("j_parent_start"), namelist_ints("parent_grid_ratio")
    d = int(dom[1:])
    while d > 1:
        i = ips[d - 1] - 1 + i // ratio[d - 1]
        j = jps[d - 1] - 1 + j // ratio[d - 1]
        d -= 1
    return j, i


def hydrostatic(ds, j, i):
    """phy_prep (module_big_step_utilities_em.F:4943-4970) in REAL: p_hyd_w(kte)=p_top, downward
    p_hyd_w(k) = p_hyd_w(k+1) - (1+qtot)*(c1(k)*MUT+c2(k))*dnw(k), qtot = qv+qc+qr+qi+qs+qg (moist order),
    p_hyd(k) = 0.5*(p_hyd_w(k)+p_hyd_w(k+1)).  These are radiation_driver's P / P8W (first_rk_step_part1:280)."""
    v = lambda n: np.asarray(ds[n][0, :, j, i], f4)
    qtot = np.zeros_like(v("QVAPOR"))
    for name in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
        qtot = (qtot + v(name)).astype(f4)
    c1, c2, dnw = (np.asarray(ds[n][0], f4) for n in ("C1H", "C2H", "DNW"))
    mut = f4(np.asarray(ds["MU"][0, j, i], f4) + np.asarray(ds["MUB"][0, j, i], f4))
    nz = qtot.size
    pw = np.zeros(nz + 1, f4)
    pw[nz] = f4(np.asarray(ds["P_TOP"][0]))
    for k in range(nz - 1, -1, -1):
        pw[k] = f4(pw[k + 1] - f4(f4(f4(f4(1) + qtot[k]) * f4(f4(c1[k] * mut) + c2[k])) * dnw[k]))
    ph = (f4(0.5) * (pw[:-1] + pw[1:])).astype(f4)
    return pw, ph


def column(ds, j, i):
    v = lambda n: np.asarray(ds[n][0, :, j, i], np.float64)
    p = v("P") + v("PB")
    qv = v("QVAPOR")
    thm = v("THM") + T0
    th = thm / (1.0 + RVOVRD * qv)
    pi = (p / P1000) ** RCP
    t = th * pi
    alt = (R_D / P1000) * thm * (p / P1000) ** CVPM
    rho = (1.0 + qv) / alt
    zw = (v("PH") + v("PHB")) / G
    nz = p.size
    dz8w = np.zeros(nz + 1)
    dz8w[:nz] = zw[1:] - zw[:-1]
    fzm = np.asarray(ds["FNM"][0], np.float64)
    fzp = np.asarray(ds["FNP"][0], np.float64)
    p8w = np.zeros(nz + 1); t8w = np.zeros(nz + 1)
    for k in range(1, nz):
        p8w[k] = fzm[k] * p[k] + fzp[k] * p[k - 1]
        t8w[k] = fzm[k] * t[k] + fzp[k] * t[k - 1]
    z = 0.5 * (zw[1:] + zw[:-1])
    w1 = (zw[0] - z[1]) / (z[0] - z[1]); w2 = 1.0 - w1
    p8w[0] = w1 * p[0] + w2 * p[1]; t8w[0] = w1 * t[0] + w2 * t[1]
    w1 = (zw[nz] - z[nz - 2]) / (z[nz - 1] - z[nz - 2]); w2 = 1.0 - w1
    p8w[nz] = np.exp(w1 * np.log(p[nz - 1]) + w2 * np.log(p[nz - 2])); t8w[nz] = w1 * t[nz - 1] + w2 * t[nz - 2]
    s = lambda n: float(np.asarray(ds[n][0, j, i]))
    p_hyd_w, p_hyd = hydrostatic(ds, j, i)
    return dict(t=t, p=p, pi=pi, th=th, rho=rho, dz8w=dz8w, p8w=p8w, t8w=t8w, qv=qv, p_hyd=p_hyd, p_hyd_w=p_hyd_w,
                qc=v("QCLOUD"), qi=v("QICE"), ni=v("QNICE"), qs=v("QSNOW"), qr=v("QRAIN"), qg=v("QGRAUP"),
                cldfra=v("CLDFRA"), tsk=s("TSK"), emiss=s("EMISS"), albedo=s("ALBEDO"), coszen=s("COSZEN"),
                xland=s("XLAND"), xlat=s("XLAT"), xlong=s("XLONG"), snow=s("SNOW"),
                xice=s("SEAICE") if "SEAICE" in ds.variables else 0.0)


def main(out_bin, out_json):
    picks = json.loads(Path("census.json").read_text())["picks"]
    cols, meta = [], []
    for pk in picks:
        with Dataset(CPU / f"wrfout_{pk['domain']}_{pk['stamp']}") as c:
            col = column(c, pk["j"], pk["i"])
            julday, gmt = int(c.getncattr("JULDAY")), float(c.getncattr("GMT"))
            p_top = float(np.asarray(c["P_TOP"][0]))
            c_start = c.getncattr("START_DATE")
        qc_rad, qi_rad, sgs_src = col["qc"].copy(), col["qi"].copy(), "none"
        g = GPU / f"wrfout_{pk['domain']}_{pk['stamp']}"
        if g.is_file():
            with Dataset(g) as gd:
                qcbl = np.asarray(gd["QC_BL"][0, :, pk["j"], pk["i"]], np.float64)
                cfbl = np.asarray(gd["CLDFRA_BL"][0, :, pk["j"], pk["i"]], np.float64)
            add = (col["qc"] < 1e-6) & (cfbl > 0.001)
            qc_rad = np.where(add, col["qc"] + qcbl, col["qc"])
            sgs_src = "gpu_W9_QC_BL_CLDFRA_BL"
        pj, pi_ = parent_cell(pk["domain"], pk["j"], pk["i"])
        with Dataset(sorted(CPU.glob(f"wrfout_d01_{pk['stamp'][:13]}*"))[0]) as d1:
            p_d01 = hydrostatic(d1, pj, pi_)[1]  # radiation_driver P = p_hyd of d01 (ozn_p_int on id 1)
            lat_d01 = float(np.asarray(d1["XLAT"][0, pj, pi_]))
        start = datetime.strptime(c_start, "%Y-%m-%d_%H:%M:%S")
        hours = (datetime.strptime(pk["stamp"], "%Y-%m-%d_%H:%M:%S") - start).total_seconds() / 3600.0
        assert hours == float(pk["tau"]), (pk, hours)  # census label == true lead from START_DATE
        julian = (julday - 1) + (gmt + hours) / 24.0  # WRF currentDayOfYearReal at the frame time
        col.update(qc_rad=qc_rad, qi_rad=qi_rad, p_d01=p_d01, lat_d01=lat_d01)
        cols.append(col)
        meta.append({**pk, "parent_d01": [pj, pi_], "sgs_source": sgs_src, "julday": julday, "gmt": gmt, "julian": julian,
                     "p_top": p_top, "coszen": col["coszen"], "xland": col["xland"],
                     "n_sgs_merged_levels": int((qc_rad != col["qc"]).sum())})
    nz = cols[0]["p"].size
    assert all(c["p"].size == nz for c in cols)
    ncol = len(cols)
    with open(out_bin, "wb") as f:
        np.asarray([ncol, nz], np.int32).tofile(f)
        assert len({(m["p_top"], m["gmt"]) for m in meta}) == 1
        np.asarray([meta[0]["p_top"], meta[0]["gmt"]], np.float32).tofile(f)
        np.asarray([m["julian"] for m in meta], np.float32).tofile(f)
        np.asarray([m["julday"] for m in meta], np.int32).tofile(f)
        for name in ("tsk", "emiss", "albedo", "coszen", "xland", "xlat", "xlong", "snow", "xice", "lat_d01"):
            np.asarray([c[name] for c in cols], np.float32).tofile(f)
        for name in ("t", "p", "pi", "th", "rho", "qv", "qc", "qi", "ni", "qs", "qr", "qg", "qc_rad", "qi_rad", "cldfra",
                     "p_hyd", "p_d01"):
            np.asarray([c[name] for c in cols], np.float32).ravel(order="F").tofile(f)
        for name in ("dz8w", "p_hyd_w", "t8w"):
            np.asarray([c[name] for c in cols], np.float32).ravel(order="F").tofile(f)
        np.asarray([m["tau"] for m in meta], np.float32).tofile(f)
    Path(out_json).write_text(json.dumps({"ncol": ncol, "nz": nz, "columns": meta}, indent=1))
    print(ncol, "columns", nz, "levels; sgs-merged columns:", sum(m["n_sgs_merged_levels"] > 0 for m in meta))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
