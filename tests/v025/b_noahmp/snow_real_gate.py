"""CPU: native REAL Noah-MP snow vs pristine WRF snow routines at REAL kind 4 (registered rule NS01/REGISTRATION.json)."""
import importlib.util, json, os, sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
repo = Path(__file__).resolve().parents[3]; D = Path(__file__).resolve().parent / "snow_real_oracle"
spec = importlib.util.spec_from_file_location("snow_parity", repo / "proofs/noahmp/snow_parity.py")
sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
snowmod, NSNOW = sp.snowmod, sp.NSNOW
r4, _ = sp.parse_savepoints(str(D / "snow_r4_savepoints.txt"))
r8, _ = sp.parse_savepoints(str(repo / "proofs/noahmp/fixtures/snow_oracle_savepoints.txt"))
static = sp.build_static()

def internal_real(land, f, imelt):
    """sp._run_internal_snowwater with REAL work (WRF kind 4)."""
    land = jax.tree.map(lambda v: v.astype(jnp.float32) if jnp.issubdtype(v.dtype, jnp.floating) else v, land)
    dt32 = jnp.float32
    sfctmp = jnp.asarray([[f["sfctmp"]]], dt32)
    bdfall = jnp.minimum(120.0, 67.92 + 51.25 * jnp.exp((sfctmp - snowmod.TFRZ) / 2.59)).astype(dt32)
    qsnow = jnp.asarray([[f["qsnow"]]], dt32)
    snowhin = jnp.where(qsnow > 0.0, qsnow / bdfall, 0.0).astype(dt32)
    qrain = jnp.asarray([[f["qrain"]]], dt32)
    qsnsub = jnp.asarray([[f.get("qsnsub", 0.0)]], dt32); qsnfro = jnp.asarray([[f.get("qsnfro", 0.0)]], dt32)
    isnow = land.isnow.astype(jnp.int32)
    wx_old = land.snice + land.snliq
    ficeold = jnp.where(wx_old > 0.0, land.snice / jnp.where(wx_old > 0.0, wx_old, 1.0), 0.0)
    zss = land.zsnso[:NSNOW]
    prev = jnp.concatenate([jnp.zeros_like(zss[:1]), zss[:-1]], axis=0)
    dzsnso_snow = jnp.where(snowmod._active_mask(isnow), prev - zss, 0.0)
    out = snowmod._snowwater_column(isnow, land.snowh, land.sneqv, land.snice, land.snliq, land.sh2o[0],
        land.smois[0] - land.sh2o[0], land.tsno, static.zsoil.astype(dt32), qsnow, snowhin, qsnfro, qsnsub,
        qrain, sfctmp, ficeold, imelt[:NSNOW].astype(jnp.int32), dzsnso_snow, sp.DT)
    isnow_n, snowh_n, sneqv_n, snice_n, snliq_n, sh2o_n, _sice, tsno_n, _dz, zsnso_full = out[:10]
    tauss_n, albold_n = land.tauss, land.albold   # SNOWWATER passes them through (aged in ENERGY, :2925-2945)
    g = lambda v: np.asarray(v)
    dtypes = sorted({str(v.dtype) for v in (snowh_n, sneqv_n, snice_n, snliq_n, tsno_n, zsnso_full, tauss_n, albold_n)})
    return {"isnow": int(g(isnow_n)[0, 0]), "snowh": float(g(snowh_n)[0, 0]), "sneqv": float(g(sneqv_n)[0, 0]),
            "tauss": float(g(tauss_n)[0, 0]), "albold": float(g(albold_n)[0, 0]), "snice": g(snice_n)[:, 0, 0],
            "snliq": g(snliq_n)[:, 0, 0], "tsno": g(tsno_n)[:, 0, 0], "zsnso": g(zsnso_full)[:, 0, 0],
            "sh2o0": float(g(sh2o_n)[0, 0]), "_dtypes": dtypes}

def ref(pre, post):
    # TAUSS/ALBOLD are ENERGY outputs (ALBEDO :2925-2945); the oracle ages them after SNOWWATER (not NOAHMP_SFLX's
    # order), so SNOWWATER's reference is the pass-through PRE value (composed step: test_albedo_hold_oracle.py).
    return {"isnow": post["isnow"], "snowh": post["snowh"], "sneqv": post["sneqv"], "tauss": pre["tauss"],
            "albold": pre["albold"], "snice": np.asarray(post["snice"]), "snliq": np.asarray(post["snliq"]),
            "tsno": np.asarray(post["stc"][:NSNOW]), "zsnso": np.asarray(post["zsnso"]), "sh2o0": post["sh2o"][0]}

def ulp(x):
    return np.spacing(np.abs(np.asarray(x, np.float32))).astype(np.float64)

FIELDS = ("snowh", "sneqv", "tauss", "albold", "sh2o0", "snice", "snliq", "tsno", "zsnso")


def run_gate():
    """Return the NS01 summary dict (registered rule in snow_real_oracle/REGISTRATION.json)."""
    results, all_pass, worst_ratio = [], True, {}
    for s in sorted(r4):
        pre, f = r4[s]["PRE"], sp.SCEN_FORCING[s]
        sp.SCEN_TG = f["tg"]
        land = sp.build_land_state(pre)
        imelt = np.zeros((NSNOW + sp.NSOIL, 1, 1), np.int32)
        if s == 6: imelt[1, 0, 0] = imelt[2, 0, 0] = 1
        imelt = jnp.asarray(imelt)
        if s in sp.SUBLIM_SCEN:
            got = internal_real(land, f, imelt); path = "internal_snowwater_real"
        else:
            out = snowmod.noahmp_snow(land, sp.build_forcing(f), static, jnp.asarray([[f["qsnow"]]]), imelt,
                                      jnp.asarray([[f["qrain"]]]), sp.DT)
            g = lambda v: np.asarray(v)
            got = {"isnow": int(g(out.isnow)[0, 0]), "snowh": float(g(out.snowh)[0, 0]), "sneqv": float(g(out.sneqv)[0, 0]),
                   "tauss": float(g(out.tauss)[0, 0]), "albold": float(g(out.albold)[0, 0]), "snice": g(out.snice)[:, 0, 0],
                   "snliq": g(out.snliq)[:, 0, 0], "tsno": g(out.tsno)[:, 0, 0], "zsnso": g(out.zsnso)[:, 0, 0],
                   "sh2o0": float(g(out.sh2o)[0, 0, 0]), "_dtypes": sorted({str(v.dtype) for v in (out.snowh, out.snice, out.tsno, out.zsnso)})}
            path = "public_noahmp_snow"
        a, b = ref(r4[s]["PRE"], r4[s]["POST"]), ref(r8[s]["PRE"], r8[s]["POST"])
        checks = {"isnow": {"port": got["isnow"], "r4": a["isnow"], "pass": got["isnow"] == a["isnow"]}}
        for k in FIELDS:
            port, x4, x8 = (np.atleast_1d(np.asarray(v, np.float64)) for v in (got[k], a[k], b[k]))
            err = np.abs(port - x4); bound = np.maximum(1.5 * np.abs(x4 - x8), 4 * ulp(np.maximum(np.abs(x4), np.abs(x8))))
            ok = bool(np.all(err <= bound))
            ratio = float(np.max(np.where(bound > 0, err / bound, np.where(err > 0, np.inf, 0.0))))
            worst_ratio[k] = max(worst_ratio.get(k, 0.0), ratio)
            checks[k] = {"max_err": float(err.max()), "max_bound": float(bound.max()), "max_err_over_bound": ratio, "pass": ok}
        if got["isnow"] < 0:
            port_res = abs(float(np.sum(got["snice"] + got["snliq"])) - got["sneqv"])
            r4_res = abs(float(np.sum(a["snice"] + a["snliq"])) - a["sneqv"])
            checks["swe_conservation"] = {"port": port_res, "r4": r4_res, "pass": port_res <= 1.5 * r4_res + 4 * float(ulp(got["sneqv"]))}
        sp_ok = all(c["pass"] for c in checks.values()); all_pass &= sp_ok
        results.append({"scenario": s, "path": path, "pass": sp_ok, "dtypes": got["_dtypes"], "checks": checks})
    summary = {"gate": "NS01 native REAL snow vs pristine WRF R4 (registered rule)", "registration": str(D / "REGISTRATION.json"),
               "all_pass": bool(all_pass), "n_pass": sum(r["pass"] for r in results), "n": len(results),
               "worst_err_over_bound": worst_ratio, "scenarios": results}
    return summary


if __name__ == "__main__":
    os.environ["GPUWRF_NOAHMP_NATIVE_REAL"] = "1"
    summary = run_gate()
    print(json.dumps({k: summary[k] for k in ("all_pass", "n_pass", "n", "worst_err_over_bound")}))
