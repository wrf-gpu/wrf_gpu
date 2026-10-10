"""v0.3.4 RUC LSM (sf_surface_physics=3): faithful JAX port vs the pristine-WRF oracle.

Oracle: unmodified WRF ``LSMRUC`` (``proofs/v034/oracle/ruclsm``), full state after
each of 12 steps, fp64 and REAL=4 builds.  The kernel is ``gpuwrf.physics.ruclsm``;
the operational seam is ``gpuwrf.coupling.ruc_surface_hook`` (scan-wired opt-in).

* fp64: machine precision on every compared field and step (rtol 1e-9; observed
  <=1.3e-10, nonlinear VILKA/energy-budget rounding growth);
* fp32: the port's deviation from WRF REAL must stay inside WRF's own REAL rounding
  noise (|JAX32-WRF32| <= 3 * max(|WRF64-WRF32|, 4 ulp of the field scale); cancelling
  surface fluxes additionally 1e-4 of their scale);
* deletion-sensitive: the regimes cover thin/one/two-layer snow, the melt second
  iteration, frpcpn snowfall, fresh snow, frozen soil, mosaic, the 9-level grid.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GPUWRF_JAX_CACHE", "0")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
# mutant runs (lane o1-ruc): prove which kernel source is imported (E93/E202)
if os.environ.get("RUC_EXPECT_SRC"):
    import gpuwrf.physics.ruclsm as _imported

    assert _imported.__file__.startswith(os.environ["RUC_EXPECT_SRC"]), _imported.__file__
    print("RUC_SRC_OK", _imported.__file__)
_spec = importlib.util.spec_from_file_location("ruclsm_parity", ROOT / "proofs" / "v034" / "ruclsm_parity.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

# 1-based oracle regime indices exercising every LSMRUC branch family
FP64_REGIMES = (1, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 21, 22, 23, 24, 25, 28, 32, 33, 34, 35,
                # 36-51: critic rv-ruc E154 coverage extension (convective/liquid frpcpn split, bottom
                # melt, drip into snow, snow fully gone, snow roughness, fresh-snow albedo, runoff, dew)
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
# fp32: regime 7 (thin snow) is excluded -- with XLA:CPU FMA contraction a
# KEEPFR3DFLAG knife-edge (tso>told .and. soilmois>smold decided by 1 ulp) flips at
# step 9; with --xla_cpu_max_isa=AVX (no FMA, like the gfortran build) every field of
# regime 7 is within WRF's own REAL noise (lane o1-ruc notes, RUC04).
FP32_REGIMES = (1, 6, 10, 11, 12, 24, 25, 26, 28, 32, 35, 37, 41, 43)
# derived surface fluxes are differences of nearly equal REAL temperatures/humidities:
# one ulp of TSK moves them by hundreds of their own ulps (regime 11 step 1: TSK -1 ulp
# -> GRDFLX +347 ulp, while WRF64-vs-WRF32 moves HFX 1685 ulp, LH 45863 ulp).
_CANCELLING_FLUXES = {"hfx", "lh", "qfx", "grdflx", "dew", "sfcevp", "qcg"}
# Surface runoff is the residual RUNOFF = -TOTLIQ - INFMAX (SOILMOIST) of two near-equal water
# rates; under bottom melt the operand is the melt rate (SMELTG capped at 5.8e-9 m/s), so its
# rounding scale is that of the accumulated melt SNOM, not of the tiny runoff itself (regime 43:
# runoff 1.55e-12 m/s per step = 1/3700 of its operands; JAX32 vs WRF32 differ by 1 ulp of SMELT).
_RUNOFF_ACCUMULATORS = {"sfcrunoff", "udrunoff", "acrunoff"}


@pytest.fixture(scope="module")
def oracle64():
    return rp.load("fp64")


@pytest.fixture(scope="module")
def oracle32():
    return rp.load("fp32")


def _regime(data, idx):
    reg = data["regimes"][idx - 1]
    assert reg["status"] == "ok", reg["name"]
    return reg


@pytest.mark.parametrize("idx", FP64_REGIMES)
def test_fp64_machine_precision_every_step(oracle64, idx):
    reg = _regime(oracle64, idx)
    out = rp.run_regime(reg, oracle64["scalars"], "fp64")
    assert not bool(np.any(out[0]["unsupported"]))
    assert all(bool(np.all(o["vilka_ok"])) for o in out)
    bad = []
    for n, ref in enumerate(reg["steps"]):
        for k in rp.COMPARED:
            g = np.asarray(out[n][k], np.float64).reshape(-1)
            o = np.asarray(ref[k.upper()], np.float64).reshape(-1)
            assert np.all(np.isfinite(g)), (reg["name"], k, n + 1)
            tol = 1.0e-9 * float(np.max(np.abs(o))) + 1.0e-15
            err = float(np.max(np.abs(g - o)))
            if err > tol:
                bad.append(f"step {n + 1} {k}: |d|={err:.3e} > {tol:.3e}")
    assert not bad, f"{reg['name']}: " + "; ".join(bad[:10])


@pytest.mark.parametrize("idx, steps", [(10, 12), (25, 12), (32, 12), (33, 12), (34, 7), (35, 12)])
def test_surface_melt_iteration_is_exercised(oracle64, idx, steps):
    """Branch census: these regimes run SNOWTEMP's nmelt=1 second energy iteration."""

    out = rp.run_regime(_regime(oracle64, idx), oracle64["scalars"], "fp64")
    assert sum(bool(np.any(o["surface_melt"])) for o in out) == steps


@pytest.mark.parametrize("idx", (42, 43, 44, 45))
def test_bottom_melt_branch_is_exercised(oracle64, idx):
    """Branch census: SNOWTEMP bottom melt (tso(1) > 273.15 under snow) runs in these regimes."""

    out = rp.run_regime(_regime(oracle64, idx), oracle64["scalars"], "fp64")
    assert any(bool(np.any(o["bottom_melt"])) for o in out)


@pytest.mark.parametrize("idx", (37, 38, 39, 40, 41))
def test_convective_frpcpn_split_is_exercised(oracle64, idx):
    """LSMRUC :623-634 runs only with frpcpn and RAINBL > RAINNCV (the HRRR/RAP path)."""

    inp = _regime(oracle64, idx)["inputs"]
    assert int(inp["FRPCPN"]) == 1 and float(inp["RAINBL"]) > float(inp["RAINNCV"])


@pytest.mark.parametrize("idx", FP32_REGIMES)
def test_fp32_within_wrf_real_rounding_noise(oracle32, oracle64, idx):
    reg32 = _regime(oracle32, idx)
    reg64 = _regime(oracle64, idx)
    out = rp.run_regime(reg32, oracle32["scalars"], "fp32")
    bad = []
    for k in rp.COMPARED:
        jx = np.stack([np.asarray(o[k], np.float64).reshape(-1) for o in out])
        w32 = np.stack([np.asarray(s[k.upper()], np.float64).reshape(-1) for s in reg32["steps"]])
        w64 = np.stack([np.asarray(s[k.upper()], np.float64).reshape(-1) for s in reg64["steps"]])
        assert np.all(np.isfinite(jx)), (reg32["name"], k)
        if k == "keepfr3dflag":
            # rounding-knife-edge flag: no more flips than WRF's own REAL-vs-DOUBLE build
            assert int(np.sum(jx != w32)) <= max(1, int(np.sum(w64 != w32))), (reg32["name"], k)
            continue
        scale = float(np.max(np.abs(w32)))
        noise = max(float(np.max(np.abs(w64 - w32))), 4.0 * 2.0**-23 * scale)
        tol = 3.0 * noise
        if k in _CANCELLING_FLUXES:
            tol = max(tol, 1.0e-4 * scale)
        if k in _RUNOFF_ACCUMULATORS:
            snom = np.stack([np.asarray(s["SNOM"], np.float64).reshape(-1) for s in reg32["steps"]])
            tol = max(tol, 4.0 * 2.0**-23 * float(np.max(np.abs(snom))))
        err = float(np.max(np.abs(jx - w32)))
        if err > tol:
            bad.append(f"{k}: |jax32-wrf32|={err:.3e} > tol {tol:.3e} (WRF REAL noise {noise:.3e})")
    assert not bad, f"{reg32['name']}: " + "; ".join(bad)


def _first_step(reg, scalars):
    from gpuwrf.physics import ruclsm as r

    cfg = rp.regime_config(reg, scalars, "float64")
    inp = reg["inputs"]
    prof = {"tso", "soilmois", "sh2o", "smfr3d", "keepfr3dflag"}
    state = {k: rp._arr(inp[n], "float64", k in prof) for k, n in rp.STATE_IN.items()}
    forcing = {k: rp._arr(inp[n], "float64") for k, n in rp.FORCING_IN.items()}
    static = {k: rp._arr(inp[n], "float64") for k, n in rp.STATIC_IN.items()}
    return r.lsmruc_step(state, forcing, static, cfg, r.load_ruc_tables(), 1)


def test_sea_ice_columns_fail_closed(oracle64):
    for idx in (29, 30):
        reg = _regime(oracle64, idx)
        assert bool(np.all(_first_step(reg, oracle64["scalars"])["unsupported"])), reg["name"]


def test_wrf_root_depth_out_of_bounds_fails_closed(oracle64):
    """USGS 14 on the v017 6-level grid makes WRF read zshalf(7) (the oracle regime
    crashes inside WRF); the port flags the column instead of inventing a value."""

    reg = oracle64["regimes"][14]
    assert reg["status"] == "wrf_fatal"
    assert bool(np.all(_first_step(reg, oracle64["scalars"])["unsupported"]))
    ok = oracle64["regimes"][30]  # same column on WRF's real 6-level grid (zs .. 1.6, 3.0)
    assert not bool(np.any(_first_step(ok, oracle64["scalars"])["unsupported"]))


def test_tables_parsed_like_ruclsm_soilvegparm():
    from gpuwrf.physics.ruclsm import load_ruc_tables

    t = load_ruc_tables("USGS-RUC")
    assert (t.lucats, t.slcats, t.urban, t.crop, t.natural, t.bare) == (28, 19, 1, 3, 5, 19)
    assert t.cfactr_data == 0.5 and t.rsmax_data == 5000.0
    assert t.z0tbl[6] == 0.075  # USGS 7 grassland row lacks a comma after Z0 in the TBL
    assert int(t.ifortbl[13]) == 1
    m = load_ruc_tables("MODI-RUC")
    assert (m.lucats, m.urban, m.crop, m.natural, m.bare) == (21, 13, 12, 10, 16)


def test_v017_front_end_matches_v017_oracle_to_machine_precision():
    """The v0.18 API (physics.lsm_ruc) now runs the faithful kernel."""

    import json

    from gpuwrf.physics.lsm_ruc import ruc_columns

    data = json.loads((ROOT / "proofs/v017/savepoints/ruclsm/fp64/ruclsm_fp64.json").read_text())
    c = data["columns"]
    zs = np.asarray([0.0, 0.05, 0.20, 0.40, 1.0, 2.0])
    s0 = np.asarray(c["SOILT_IN"])
    tb = np.asarray(c["TBOT"])
    sm1 = np.asarray(c["SOILMOIS_IN_1"])
    sm2 = np.asarray([0.25, 0.25, 0.18, 0.36, 0.20])
    tso = s0[:, None] + (tb - s0)[:, None] * (zs / zs[-1])
    sm = sm1[:, None] + (sm2 - sm1)[:, None] * (zs / zs[-1])
    out = ruc_columns(gsw=c["GSW"], glw=c["GLW"], emiss=c["EMISS_IN"], tabs=c["TABS"], qv=c["QV"],
                      qc=c["QC"], rho=c["RHO"], p8w=c["P8W"], z3d=c["Z3D"], rainbl=c["RAINBL"],
                      vegfra=c["VEGFRA"], flhc=c["FLHC"], flqc=c["FLQC"], tbot=c["TBOT"],
                      xland=c["XLAND"], mavail=c["MAVAIL_IN"], ivgtyp=c["IVGTYP"], isltyp=c["ISLTYP"],
                      soilt=s0, tso=tso, soilmois=sm, sh2o=sm)
    for k in ("SOILT", "HFX", "LH", "QFX", "GRDFLX", "SFCEVP", "QVG", "MAVAIL"):
        ref = np.asarray(c[k])
        assert np.max(np.abs(np.asarray(out[k.lower()]) - ref)) <= 1.0e-10 * np.max(np.abs(ref)) + 1e-15, k
    for k in ("TSO", "SOILMOIS", "SH2O"):
        ref = np.asarray(data["profiles"][k])
        assert np.max(np.abs(np.asarray(out[k.lower()]) - ref)) <= 1.0e-12 * np.max(np.abs(ref)), k


# --------------------------------------------------------------------------- wiring


def _synthetic_fields(seaice_col=False):
    ny, nx, nzs = 2, 3, 9
    zs = (0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00)
    f = {
        "IVGTYP": np.array([[10, 12, 17], [1, 5, 10]]), "ISLTYP": np.array([[6, 3, 14], [6, 6, 12]]),
        "XLAND": np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 1.0]]),
        "SEAICE": np.zeros((ny, nx)), "TMN": np.full((ny, nx), 285.0), "SHDMIN": np.full((ny, nx), 5.0),
        "SHDMAX": np.full((ny, nx), 80.0), "ALBBCK": np.full((ny, nx), 0.18), "VEGFRA": np.full((ny, nx), 50.0),
        "TSK": np.full((ny, nx), 288.0), "TSLB": np.full((nzs, ny, nx), 286.0),
        "SMOIS": np.full((nzs, ny, nx), 0.25), "SNOALB": np.full((ny, nx), 0.7),
    }
    if seaice_col:
        f["SEAICE"][1, 2] = 0.9
    return f, zs


def _admission_namelist(**overrides):
    from gpuwrf.runtime.operational_mode import OperationalNamelist

    base = OperationalNamelist.__new__(OperationalNamelist)
    defaults = dict(mp_physics=8, bl_pbl_physics=5, sf_sfclay_physics=5, cu_physics=0,
                    sf_surface_physics=3, use_noahmp=False, ra_sw_physics=4, ra_lw_physics=4,
                    noahclassic_static=None, noahclassic_land=None, noahclassic_rad=None,
                    slab_static=None, slab_land=None, slab_rad=None, px_static=None, px_land=None,
                    px_rad=None, ruc_static=None, ruc_land=None, ruc_rad=None)
    defaults.update(overrides)
    for k, v in defaults.items():
        object.__setattr__(base, k, v)
    return base


def _bundles(seaice_col=False):
    from gpuwrf.coupling.ruc_surface_hook import build_ruc_bundles

    f, zs = _synthetic_fields(seaice_col)
    return build_ruc_bundles(f, mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, isice=15, dt=18.0, zs=zs,
                             p1=np.full((2, 3), 95000.0))


def test_operational_admission_is_explicit_and_fail_closed():
    from gpuwrf.coupling.physics_dispatch import UnsupportedSchemeSelection
    from gpuwrf.runtime.operational_mode import _resolve_operational_suite

    static, land = _bundles()
    _resolve_operational_suite(_admission_namelist(ruc_static=static, ruc_land=land))
    for bad in (dict(), dict(ruc_static=static), dict(ruc_static=static, ruc_land=land, sf_sfclay_physics=1),
                dict(ruc_static=static, ruc_land=land, use_noahmp=True)):
        with pytest.raises(UnsupportedSchemeSelection, match="sf_surface_physics=3"):
            _resolve_operational_suite(_admission_namelist(**bad))
    # rv-ruc F4: non-Thompson MP has no previous-step precipitation carry -> RUC would see zero rain/snow
    with pytest.raises(UnsupportedSchemeSelection, match="precipitation carry"):
        _resolve_operational_suite(_admission_namelist(ruc_static=static, ruc_land=land, mp_physics=6))
    # rv-ruc A5: MYJ PBL (LSMRUC myj chs path, not wired) is unreachable: the MYJ pairing rule needs
    # sf_sfclay=2, which RUC admission rejects (needs MYNN-SL 5); the explicit MYJ guard is defence in depth
    for myj in (dict(bl_pbl_physics=2), dict(bl_pbl_physics=2, sf_sfclay_physics=2)):
        with pytest.raises(UnsupportedSchemeSelection):
            _resolve_operational_suite(_admission_namelist(ruc_static=static, ruc_land=land, **myj))
    static_i, land_i = _bundles(seaice_col=True)
    with pytest.raises(UnsupportedSchemeSelection, match="sea-ice"):
        _resolve_operational_suite(_admission_namelist(ruc_static=static_i, ruc_land=land_i))


def test_seeded_carry_follows_ruclsminit_and_ktau1_block():
    from gpuwrf.physics.ruclsm import qsn, tbq_table

    static, land = _bundles()
    assert static.cfg.isice == 15 and static.cfg.iswater == 17 and static.tables.mminluruc == "MODI-RUC"
    tbq = tbq_table("float32")
    patmb = np.float32(95000.0) * np.float32(1.0e-2)  # LSMRUC :503 patmb=p8w*1.e-2 in REAL
    qsg = np.asarray(qsn(jnp_asarray(np.full(6, 288.0, np.float32)), tbq)) / patmb
    np.testing.assert_array_equal(np.asarray(land.qsg).ravel(), qsg)
    np.testing.assert_array_equal(np.asarray(land.qvg).ravel(), qsg * np.asarray(land.mavail).ravel())
    assert np.all(np.asarray(land.rhosnf) == -1.0e3) and np.all(np.asarray(land.qcg) == 0.0)
    assert np.asarray(land.tso).shape == (2, 3, 9)


def jnp_asarray(x):
    import jax.numpy as jnp

    return jnp.asarray(x)


def test_precipitation_forcing_matches_surface_driver_accumulation():
    from gpuwrf.coupling.ruc_surface_hook import ruc_precipitation_forcing
    from gpuwrf.runtime.noahmp_precipitation import NoahMPPrecipitation

    one = np.ones((1, 2), np.float32)
    p = NoahMPPrecipitation(prcpconv=0.001 * one, prcpnonc=0.002 * one, prcpsnow=0.0015 * one,
                            prcpgrpl=0.0002 * one, prcphail=0 * one)
    out = ruc_precipitation_forcing(p, 18.0, (1, 2), np.float32)
    np.testing.assert_allclose(np.asarray(out["rainbl"]), 0.054, rtol=1e-6)      # RAINCV+RAINNCV mm
    np.testing.assert_allclose(np.asarray(out["rainncv"]), 0.036, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out["snowncv"]), 0.027, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out["frzfrac"]), (0.027 + 0.0036) / 0.036, rtol=1e-5)


def test_history_and_restart_schema_cover_the_ruc_carry():
    from gpuwrf.coupling.ruc_surface_hook import RUC_HISTORY_FIELDS, RUC_RESTART_ONLY_FIELDS, ruc_history_fields
    from gpuwrf.io import wrfrst_netcdf as rst
    from gpuwrf.physics.ruclsm import STATE_PROFILES, STATE_SCALARS

    _, land = _bundles()
    h = ruc_history_fields(land)
    assert h["TSLB"].shape == (9, 2, 3) and h["SNOW"].shape == (2, 3)
    mapped = set(RUC_HISTORY_FIELDS.values()) | set(RUC_RESTART_ONLY_FIELDS.values())
    assert set(STATE_PROFILES) <= mapped
    assert {"ruc_land", "ruc_rad"} <= set(rst.PYTREE_CARRY_FIELDS) & set(rst._LATE_PYTREE_CARRY_FIELDS)
    assert "ruc_land" not in rst.UNSUPPORTED_CARRY_FIELDS
    assert len(set(STATE_SCALARS + STATE_PROFILES)) == len(land._fields)
