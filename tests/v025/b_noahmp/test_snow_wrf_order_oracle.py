"""Noah-MP snow/water in WRF order vs the WIRED pristine NOAHMP_SFLX on real Swiss columns (b-noahmp NF05).

WRF (module_sf_noahmplsm.F): ENERGY's TSNOSOI updates STC(ISNOW+1:NSOIL) incl. the active snow
layers before PHASECHANGE (:2352, :5258-5371) and fixes FROZEN_CANOPY from the TV entering ENERGY
(:2210-2216); WATER (:6095-6160) runs CANWATER with that flag, splits QSNSUB/QSNFRO off the ground
vapour/dew flux on the pre-SNOWWATER SNEQV, runs SNOWWATER (ground rain QRAIN, FICEOLD from the
start-of-step pack, module_sf_noahmpdrv.F:1027-1028) and then SOILWATER with QINSUR =
(PONDING+PONDING1+PONDING2)/DT + QSNBOT + QSDEW [+ QRAIN as is]. The port dropped the TSNO
write-back, took FICEOLD after melt, fed QMELT as QRAIN, set QSNSUB/QSNFRO = 0, ran SOILWATER (QINSUR
from QMELT, QRAIN minus the frozen channels) before SNOWWATER and took FROZEN_CANOPY after ENERGY.

Oracle: proofs/noahmp/noahmp_offline_driver.F90 with NOAHMP_WIRE_SOIL=1 NOAHMP_WIRE_FICEOLD=1
NOAHMP_NSTEPS=N (calculate_soil=.true., soil_update_steps=1 = the WRF driver wiring; the frozen
savepoints never ran TSNOSOI/SOILWATER). Records: every step 1-200, then every 100th.
* swiss: 12 real Swiss columns (CPU-WRF t0 state, h1 forcing held fixed; 6 melting / 4 cold packs,
  2 snow-free), 2400 steps (12 h). Measured worst: TSNO 3.1e-5 K, TSLB 3.1e-5 K, TG 1.5e-4 K,
  SWE 2.1e-5 mm, SNLIQ/SNICE per layer <= 7.8e-5 mm, SNOWH 6.6e-7 m, SH2O 4.5e-5.
* precip: 6 real Swiss states with synthetic precipitation (rain+graupel, mixed rain/snow on bare
  soil, snowfall on a cold pack, rain-on-snow, mixed on a melting pack), 1200 steps. Measured worst:
  SWE 5.2e-5 mm, TG 4.3e-3 K, per-layer SNICE 1.0e-2 mm (one DIVIDE-timing record), SH2O 2.3e-5; bounds
  for this set come from pristine WRF's own 1-ulp SFCTMP spread (TOL_SET).
* rain: 9 REAL rainy WN3 land columns (wg_20260225_18z_a2 d02, CPU-WRF state at 20Z, forcing + hourly precip
  accumulation deltas 20->21Z: heavy rain, graupel cells, canopies at TV = 273.16 K), 600 steps. Measured worst:
  TG 5.2e-4 K, TSLB 2.6e-3 K (one decaying transient, see TOL_SET), SH2O 1.8e-5; both new-fix mutants fail here.
Release path (GPUWRF_NOAHMP_NATIVE_REAL=1). Every coupling term has a deletion mutant that must fail (E39).
"""
import inspect
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).with_name("fixtures")
TOL = {"tsno": 1e-3, "tslb": 1e-3, "tg": 2e-3, "swe": 1e-3, "snliq": 1e-3, "snice": 1e-3, "snowh": 1e-5,
       "sh2o": 1e-3, "smc": 1e-3}
# precip: COMBINE/DIVIDE/FWET timing makes pristine WRF itself move per-layer SNICE by up to 0.64 mm, SWE by
# 1.2e-2 mm and TG by 4.3e-3 K under a 1-ulp SFCTMP perturbation (E100) -> floor-based bounds there; port measured
# per-layer 1.0e-2 (release defaults) / <1e-3 (pytest defaults), SWE 5.2e-5, TG 4.3e-3.
TOL_SET = {"swiss": TOL, "precip": {**TOL, "tg": 2e-2, "swe": 2e-2, "snice": 5e-2, "snliq": 5e-2},
           # rain: 8/9 columns at round-off (TG <= 3.1e-5 K, SH2O <= 2.4e-6); one TV=273.16 canopy column carries a decaying
           # TSLB(1) transient (peak 2.6e-3 K at step 100, 1.3e-3 K at 600; TG 5.2e-4 K, SH2O 1.8e-5) -> TSLB 5e-3, SH2O 1e-4.
           "rain": {**TOL, "tslb": 5e-3, "sh2o": 1e-4, "smc": 1e-4}}
NSTEPS = {"swiss": 2400, "precip": 1200, "rain": 600}
# record columns: isnow, snowh, sneqv, stc[7], snice[3], snliq[3], sh2o[4], smc[4], qmelt, ponding, fsh, ssoil, runsrf, runsub, tg
SNOWH, SNEQV, STC, SNICE, SNLIQ, SH2O, SMC, TG = 1, 2, slice(3, 10), slice(10, 13), slice(13, 16), slice(16, 20), slice(20, 24), 30


@pytest.fixture(scope="module")
def native():
    import jax
    mp = pytest.MonkeyPatch()
    mp.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    jax.clear_caches()
    yield
    mp.undo()
    jax.clear_caches()


def _build(cols):
    import jax.numpy as jnp
    from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
    from gpuwrf.io.noahmp_land_init import build_noahmp_params
    from gpuwrf.physics.noahmp.tables import load_noahmp_parameters
    from gpuwrf.physics.noahmp.types import NoahMPForcing

    n = len(cols); f64 = jnp.float64
    row = lambda vals: jnp.asarray(np.asarray(vals, float).reshape(1, n), f64)  # noqa: E731
    lay = lambda key, nl: jnp.asarray(np.stack([np.asarray(c["state_in"][key], float) for c in cols], -1).reshape(nl, 1, n), f64)  # noqa: E731
    si = lambda k: row([c["state_in"][k] for c in cols])  # noqa: E731
    stc = np.stack([np.asarray(c["state_in"]["stc"], float) for c in cols], -1)
    land = NoahMPLandState(
        tslb=jnp.asarray(stc[NSNOW:].reshape(NSOIL, 1, n), f64), smois=lay("smc", NSOIL), sh2o=lay("sh2o", NSOIL), smcwtd=si("smcwtd"),
        isnow=jnp.asarray(np.asarray([int(c["state_in"]["isnow"]) for c in cols]).reshape(1, n), jnp.int32),
        tsno=jnp.asarray(stc[:NSNOW].reshape(NSNOW, 1, n), f64), snice=lay("snice", NSNOW), snliq=lay("snliq", NSNOW),
        zsnso=lay("zsnso", NSNOW + NSOIL), snowh=si("snowh"), sneqv=si("sneqv"), sneqvo=si("sneqvo"), tauss=si("tauss"),
        albold=si("albold"), tv=si("tv"), tg=si("tg"), tah=si("tah"), eah=si("eah"), canliq=si("canliq"), canice=si("canice"),
        fwet=si("fwet"), lai=row([c["lai0"] for c in cols]), sai=row([c["sai0"] for c in cols]), cm=si("cm"), ch=si("ch"),
        t_skin=si("tg"), qsfc=si("qsfc"), znt=row([0.05] * n), emiss=row([0.97] * n), albedo=si("albold"),
        sfcrunoff=row([0.0] * n), udrunoff=row([0.0] * n))
    f = lambda k: row([c["forcing"][k] for c in cols])  # noqa: E731
    forcing = NoahMPForcing(
        sfctmp=f("sfctmp"), sfcprs=f("sfcprs"), psfc=f("psfc"), uu=f("uu"), vv=f("vv"), qair=f("q2"), qc=f("qc"),
        soldn=f("soldn"), lwdn=f("lwdn"), prcpconv=f("prcpconv"), prcpnonc=f("prcpnonc"), prcpsnow=f("prcpsnow"),
        prcpgrpl=f("prcpgrpl"), prcphail=f("prcphail"), cosz=f("cosz"), zlvl=f("zlvl"),
        julian=jnp.asarray(float(cols[0]["julian"])), yearlen=jnp.asarray(float(cols[0]["yearlen"])))
    ivg = jnp.asarray(np.asarray([c["vegtyp"] for c in cols]).reshape(1, n), jnp.int32)
    static = NoahMPStatic(
        ivgtyp=ivg, isltyp=jnp.asarray(np.asarray([c["isltyp"] for c in cols]).reshape(1, n), jnp.int32),
        xland=row([1.0] * n), landmask=row([1.0] * n), lakemask=row([0.0] * n), lu_index=ivg, tbot=row([c["tbot"] for c in cols]),
        dzs=jnp.asarray(cols[0]["dzs"], f64), zsoil=jnp.asarray(cols[0]["zsoil"], f64), lat=row([np.degrees(c["lat_rad"]) for c in cols]),
        dx_m=float(cols[0]["dx"]), parameters=load_noahmp_parameters(str(ROOT / "data" / "wrf_pristine" / "WRF" / "run")),
        shdmax=row([c["shdmax"] for c in cols]), shdfac=row([c["shdfac"] for c in cols]))
    ep, rp, _nroot = build_noahmp_params(static)
    return {"land": land, "forcing": forcing, "static": static, "ep": ep, "rp": rp, "dt": float(cols[0]["dt"])}


@pytest.fixture(scope="module")
def cases(native):
    fx = json.loads((FIXTURES / "swiss_snow_wrf_order.json").read_text())
    rec = np.load(FIXTURES / "swiss_snow_wrf_order_steps.npz")
    out = {}
    for name, key in (("swiss", "columns"), ("precip", "precip_columns"), ("rain", "rain_columns")):
        c = _build(fx[key])
        c["records"] = dict(zip(rec[f"{name}_steps"].tolist(), rec[name]))
        out[name] = c
    return out


def _worst(c, nsteps):
    """Run noah_mp_step nsteps (jitted, current module globals); score every WRF record <= nsteps per layer."""
    import jax
    from gpuwrf.physics.noahmp import noahmp_driver as drv
    step = jax.jit(lambda ls: drv.noah_mp_step(ls, c["forcing"], c["static"], c["dt"], energy_params=c["ep"], rad_params=c["rp"])[0])
    land, worst = c["land"], {k: 0.0 for k in list(TOL) + ["isnow", "nonfinite"]}
    for it in range(1, nsteps + 1):
        land = step(land)
        if it not in c["records"]:
            continue
        w = c["records"][it]
        host = jax.device_get(land)
        worst["nonfinite"] += sum(int((~np.isfinite(np.asarray(x, float))).sum()) for x in jax.tree_util.tree_leaves(host)
                                  if np.asarray(x).dtype.kind == "f")
        g = lambda k: np.asarray(getattr(host, k), float)[..., 0, :] if np.asarray(getattr(host, k)).ndim == 3 else np.asarray(getattr(host, k), float)[0]  # noqa: E731
        isn = np.asarray(host.isnow).reshape(-1)
        wi = w[:, 0].astype(int)
        worst["isnow"] = max(worst["isnow"], float((isn != wi).any()))
        m = (np.arange(3)[:, None] >= (3 + wi)[None, :]) & (isn == wi)[None, :]   # active snow layers (WRF ISNOW)
        for key, port, ref in (("tsno", g("tsno"), w[:, STC][:, :3].T), ("snice", g("snice"), w[:, SNICE].T),
                               ("snliq", g("snliq"), w[:, SNLIQ].T)):
            if m.any():
                worst[key] = max(worst[key], float(np.abs(port - ref)[m].max()))
        for key, port, ref in (("tslb", g("tslb"), w[:, STC][:, 3:].T), ("sh2o", g("sh2o"), w[:, SH2O].T),
                               ("smc", g("smois"), w[:, SMC].T)):
            worst[key] = max(worst[key], float(np.abs(port - ref).max()))
        for key, port, ref in (("tg", g("tg"), w[:, TG]), ("swe", g("sneqv"), w[:, SNEQV]), ("snowh", g("snowh"), w[:, SNOWH])):
            worst[key] = max(worst[key], float(np.abs(port - ref).max()))
    return worst


def _fails(worst, tol):
    return worst["isnow"] > 0 or worst["nonfinite"] > 0 or any(worst[k] > t for k, t in tol.items())


@pytest.mark.parametrize("name", ["swiss", "precip", "rain"])
def test_noahmp_matches_wired_pristine_wrf(cases, name):
    worst = _worst(cases[name], NSTEPS[name])
    assert worst["nonfinite"] == 0, worst
    assert worst["isnow"] == 0, worst
    for key, tol in TOL_SET[name].items():
        assert worst[key] <= tol, (name, key, worst)


def _swap_water_before_snow(src):
    """The pre-fix ORDER: SOILWATER (QINSUR from the PHASECHANGE melt) before SNOWWATER; all other fixes kept."""
    snow = src[src.index("    forcing_s = forcing._replace(prcpsnow=precip.qsnow)\n"):src.index("    # SOILWATER's snow-to-soil input")]
    et_new = src[src.index("    # SOILWATER's snow-to-soil input"):src.index("    water_result = noahmp_water_hydro(")]
    water = src[src.index("    water_result = noahmp_water_hydro("):src.index("    # ----- coupler-facing fluxes")]
    assert src.count(snow + et_new + water) == 1
    order = "    et_w = et._replace(qsnow=precip.qsnow, qmelt=qmelt, imelt=imelt)\n" + water + snow + "\n"
    return src.replace(snow + et_new + water, order)


def _mutate(monkeypatch, which):
    from gpuwrf.physics.noahmp import energy, noahmp_driver as drv
    if which in ("tsno", "order"):
        mod, fn = (energy, "noahmp_energy_canopy") if which == "tsno" else (drv, "noah_mp_step")
        src = textwrap.dedent(inspect.getsource(getattr(mod, fn)))
        if which == "tsno":   # producer without the TSNO write-back
            assert src.count("tslb=tslb_new, tsno=tsno_new,") == 1
            src = src.replace("tslb=tslb_new, tsno=tsno_new,", "tslb=tslb_new,")
        else:
            src = _swap_water_before_snow(src)
        ns = dict(vars(mod))
        exec(compile(src, mod.__file__, "exec"), ns)
        monkeypatch.setattr(drv, fn, ns[fn])
        return
    snow, water = drv.noahmp_snow, drv.noahmp_water_hydro
    if which == "qrain":           # SNOWWATER QRAIN falls back to QMELT (the old placeholder)
        monkeypatch.setattr(drv, "noahmp_snow", lambda *a, qrain=None, **k: snow(*a, **k))
    elif which == "subl":          # QSNSUB/QSNFRO = 0 into SNOWWATER
        monkeypatch.setattr(drv, "noahmp_snow", lambda *a, qsnsub=None, qsnfro=None, **k: snow(*a, **k))
    elif which == "ficeold":       # FICEOLD from the post-PHASECHANGE pack
        monkeypatch.setattr(drv, "noahmp_snow", lambda *a, ficeold=None, **k: snow(*a, **k))
    elif which == "qinsur":        # SOILWATER without the snow-to-soil flux
        monkeypatch.setattr(drv, "noahmp_water_hydro", lambda ls, fo, st, et, dt, **k: water(
            ls, fo, st, et._replace(qmelt=et.qmelt * 0.0), dt, **k))
    elif which == "qrain_ground":  # SOILWATER QRAIN = rain minus the frozen channels (old)
        monkeypatch.setattr(drv, "noahmp_water_hydro", lambda *a, qrain_ground=None, **k: water(*a, **k))
    elif which == "frozen_canopy":  # CANWATER FROZEN_CANOPY from the post-ENERGY TV (old)
        monkeypatch.setattr(drv, "noahmp_water_hydro", lambda *a, frozen_canopy=None, **k: water(*a, **k))


@pytest.mark.parametrize("which,name,nsteps", [
    ("tsno", "swiss", 200), ("qrain", "swiss", 200), ("subl", "swiss", 200), ("ficeold", "swiss", 200),
    ("qinsur", "swiss", 2400), ("order", "precip", 1200), ("qrain_ground", "precip", 1200),
    ("frozen_canopy", "precip", 1200), ("qrain_ground", "rain", 600), ("frozen_canopy", "rain", 600)])
def test_each_coupling_term_is_deletion_sensitive(cases, monkeypatch, which, name, nsteps):
    _mutate(monkeypatch, which)
    worst = _worst(cases[name], nsteps)
    assert _fails(worst, TOL_SET[name]), (which, worst)


def _water_inputs(c):
    """The real SOILWATER call of one eager step (spy on the driver seam)."""
    from gpuwrf.physics.noahmp import noahmp_driver as drv
    seen = {}
    water = drv.noahmp_water_hydro

    def spy(*a, **k):
        seen["a"], seen["k"] = a, k
        return water(*a, **k)

    mp = pytest.MonkeyPatch()
    mp.setattr(drv, "noahmp_water_hydro", spy)
    try:
        drv.noah_mp_step(c["land"], c["forcing"], c["static"], c["dt"], energy_params=c["ep"], rad_params=c["rp"])
    finally:
        mp.undo()
    return seen["a"], seen["k"]


def test_soilwater_splits_ground_vapour_on_the_pre_snowwater_sneqv(cases):
    """QSEVA = QVAP - MIN(QVAP, SNEQV_pre/DT), QSDEW = QDEW - QSNFRO (:6116-6126), not on the post-SNOWWATER pack.

    Vanishing pack (pre > 0, post 0): soil evaporation must equal the call with only the remaining
    vapour flux. First snowfall (pre 0, post > 0): dew still enters the soil. Values are exact binary.
    """
    import jax
    import jax.numpy as jnp
    from gpuwrf.physics.noahmp.water_hydro import noahmp_water_hydro
    (land, forcing, static, et, dt), kw = _water_inputs(cases["swiss"])
    kw = {k: v for k, v in kw.items() if k != "sneqv_before_snow"}
    warm = land.replace(tg=jnp.full_like(land.tg, 280.0))    # unfrozen ground (else QSEVA/QSDEW go to SICE)
    z = jnp.zeros_like(land.sneqv)
    e = 2.0 ** -13                                           # mm/s
    s = jnp.full_like(z, dt * 2.0 ** -14)                    # SNEQV_pre/DT = e/2
    full = lambda x: jnp.full_like(et.edir, x)               # noqa: E731
    run = lambda ls, edir, pre: noahmp_water_hydro(           # noqa: E731
        ls, forcing, static, et._replace(edir=full(edir), qmelt=et.qmelt * 0.0), dt, sneqv_before_snow=pre, **kw)
    leaves = lambda r: np.concatenate([np.asarray(x, float).ravel()  # noqa: E731  (SNEQV is an input here)
                                       for x in jax.tree_util.tree_leaves(r.replace(sneqv=z)) if np.asarray(x).dtype.kind == "f"])
    # vanishing pack: half of the vapour flux sublimated the last snow, the rest evaporates from the soil
    got = leaves(run(warm.replace(sneqv=z), e, s))
    ref = leaves(run(warm.replace(sneqv=z), e / 2, None))
    np.testing.assert_array_equal(got, ref)
    assert not np.array_equal(leaves(run(warm.replace(sneqv=z), e, None)), ref, equal_nan=True)  # mutant: post-SNOWWATER SNEQV
    # first snowfall: no pack before SNOWWATER -> QSNFRO = 0, the dew enters the soil
    got = leaves(run(warm.replace(sneqv=s), -e, z))
    ref = leaves(run(warm.replace(sneqv=z), -e, None))
    np.testing.assert_array_equal(got, ref)
    assert not np.array_equal(leaves(run(warm.replace(sneqv=s), -e, None)), ref, equal_nan=True)
