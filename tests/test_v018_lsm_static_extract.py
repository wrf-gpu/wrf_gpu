"""v0.18 LSM "L2 statics" — falsifiable tests for the slab(1)/PX(7) static extract.

A wrong land-surface static is a silently-wrong forecast (the project forbids it),
so the decisive gate here is NOT a happy-path shape check: it is the SOILPROP-vs-real-
WRF faithfulness gate. The PX oracle savepoints (proofs/v017/savepoints/pxlsm/fp64)
carry the WRF-faithful Noilhan & Mahfouf ISBA constants the oracle driver's
``soil_consts`` produced from chosen (AVS, AVC) regimes; we feed the SAME (AVS, AVC)
to our ``_soilprop`` and require EVERY ISBA constant to match within rtol 1e-5.

Tests:
  1. SOILPROP-vs-oracle (LOAM avs=43/avc=18 and SAND avs=80/avc=5) -- bit-faithful.
  2. Real-case extraction of both bundles -- shapes, finiteness, physical ranges.
  3. Operational scan accepts sf=1 / sf=7 once the extracted bundle is present
     (and still fails closed without it).
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pytest

PROBE_RUN = Path(
    "/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/run_h36"
)
SAVEPOINT_DIR = Path(__file__).resolve().parents[1] / "proofs/v017/savepoints/pxlsm/fp64"

# Oracle regime (AVS, AVC) percentages (pxlsm_oracle_driver.f90:390-391); col1=LOAM,
# col2=SAND in every land case (the AVS_LOAM/AVS_SAND PARAMETERs feed soil_consts).
REGIME_AVS_AVC = {"LOAM": (43.0, 18.0), "SAND": (80.0, 5.0)}
# Savepoint ISBA column -> our _soilprop key.
SAVEPOINT_TO_KEY = {
    "WWLT": "wwlt", "WFC": "wfc", "WSAT": "wsat", "BCH": "b", "CGSAT": "cgsat",
    "C1SAT": "c1sat", "C2R": "c2r", "C3": "c3", "ASOIL": "asoil", "JP": "jp",
    "WRES": "wres",
}


def _have_probe() -> bool:
    try:
        return (PROBE_RUN / "wrfinput_d01").exists()
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# 1. SOILPROP-vs-real-WRF-oracle (the key faithfulness gate)                   #
# --------------------------------------------------------------------------- #
def test_soilprop_matches_pxlsm_oracle_loam_and_sand() -> None:
    """Every ISBA constant from _soilprop(AVS,AVC) matches the PX oracle savepoint.

    Maps each savepoint LAND column to LOAM (col 0) / SAND (col 1) and asserts our
    formula reproduces the WRF-faithful constant to rtol 1e-5. This is falsifiable:
    a single wrong coefficient (a typo in a power / scale) trips it.
    """
    from gpuwrf.io.lsm_static_extract import _soilprop

    if not SAVEPOINT_DIR.exists():
        pytest.skip(f"PX oracle savepoints absent: {SAVEPOINT_DIR}")

    # Collect the worst per-constant rtol across every land savepoint column.
    worst: dict[str, float] = {k: 0.0 for k in SAVEPOINT_TO_KEY}
    checked_columns = 0
    for case_path in sorted(SAVEPOINT_DIR.glob("pxlsm_case_*.json")):
        payload = json.loads(case_path.read_text())
        regime = payload["scalars"].get("REGIME_NAME", "")
        cols = payload["columns"]
        ifland = np.asarray(cols["IFLAND"], dtype=np.float64)
        # In every LAND case the driver lays col0=LOAM, col1=SAND; water/ice cases
        # set WSAT=1 (no soil) -> skip (the soil constants there are the water
        # passthrough, not SOILPROP output).
        if "water" in regime or "ice" in regime:
            continue
        for ci, regime_name in ((0, "LOAM"), (1, "SAND")):
            if ifland[ci] >= 1.5:  # this column is water -> skip
                continue
            avs, avc = REGIME_AVS_AVC[regime_name]
            got = _soilprop(np.array([avs]), np.array([avc]))
            for sp_col, key in SAVEPOINT_TO_KEY.items():
                ref = float(np.asarray(cols[sp_col])[ci])
                mine = float(got[key][0])
                rel = abs(mine - ref) / max(abs(ref), 1e-30)
                worst[sp_col] = max(worst[sp_col], rel)
                assert rel < 1e-5, (
                    f"{case_path.name} regime={regime} col{ci}({regime_name}) {sp_col}: "
                    f"mine={mine:.10g} oracle={ref:.10g} rtol={rel:.3e} > 1e-5"
                )
            checked_columns += 1
    assert checked_columns > 0, "no land savepoint columns were checked"
    # Surface the worst rtol per constant (visible with -s); all already < 1e-5.
    print("\nSOILPROP worst rtol vs PX oracle:", {k: f"{v:.2e}" for k, v in worst.items()})


def test_soil_avs_avc_weighting_and_no_soil_fallback() -> None:
    """The 16-category SOILCBOT weighting + no-soil ISTI=9 fallback are exact."""
    from gpuwrf.io.lsm_static_extract import _soil_avs_avc

    # Pure clay-loam (category 9, CSAND=0 FMSAND=32 CLAY=34) -> AVS=32, AVC=34.
    fr = np.zeros((16, 1, 1), dtype=np.float64)
    fr[8, 0, 0] = 1.0
    avs, avc = _soil_avs_avc(fr)
    assert avs[0, 0] == pytest.approx(32.0)
    assert avc[0, 0] == pytest.approx(34.0)

    # No soil at all (TFRAC<=0.001) -> WRF falls back to ISTI=9 clay loam (32, 34).
    avs0, avc0 = _soil_avs_avc(np.zeros((16, 1, 1), dtype=np.float64))
    assert avs0[0, 0] == pytest.approx(32.0)
    assert avc0[0, 0] == pytest.approx(34.0)

    # 50/50 sand(cat1: CSAND=46,FMSAND=46,CLAY=3) + clay(cat12: 0,22,58) average.
    fr2 = np.zeros((16, 1, 1), dtype=np.float64)
    fr2[0, 0, 0] = 0.5
    fr2[11, 0, 0] = 0.5
    avs2, avc2 = _soil_avs_avc(fr2)
    assert avs2[0, 0] == pytest.approx((46 + 46 + 0 + 22) / 2.0)  # (92 + 22)/2 = 57
    assert avc2[0, 0] == pytest.approx((3 + 58) / 2.0)            # 30.5


def test_landuse_season_matches_wrf() -> None:
    """ISN season index matches phys/module_physics_init.F:1833-1835."""
    from gpuwrf.io.lsm_static_extract import landuse_season

    assert landuse_season(15, 46.5) == 2    # NH January -> winter
    assert landuse_season(160, 46.5) == 1   # NH June -> summer
    assert landuse_season(15, -46.5) == 1   # SH January -> summer (flipped)
    assert landuse_season(160, -46.5) == 2  # SH June -> winter (flipped)
    assert landuse_season(104, 46.5) == 2   # boundary: day 104 < 105 -> winter
    assert landuse_season(105, 46.5) == 1   # boundary: day 105 -> summer
    assert landuse_season(288, 46.5) == 1   # boundary: day 288 -> summer
    assert landuse_season(289, 46.5) == 2   # boundary: day 289 > 288 -> winter


# --------------------------------------------------------------------------- #
# 2. Real-case extraction: shapes / finiteness / physical ranges              #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_probe(), reason="real wrfinput PROBE absent")
def test_real_case_slab_extraction() -> None:
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.lsm_static_extract import extract_slab_static

    run = Gen2Run(PROBE_RUN)
    slab = extract_slab_static(run, "d01")
    thc = np.asarray(slab.thc)
    emiss = np.asarray(slab.emiss)
    ny, nx = thc.shape
    for name, arr in (("thc", thc), ("emiss", emiss), ("tmn", np.asarray(slab.tmn)),
                      ("snowc", np.asarray(slab.snowc))):
        assert arr.shape == (ny, nx), f"{name} shape {arr.shape} != {(ny, nx)}"
        assert np.isfinite(arr).all(), f"{name} non-finite"
    assert np.asarray(slab.zs).shape == np.asarray(slab.dzs).shape
    assert np.asarray(slab.zs).ndim == 1
    # WRF THC = THERIN/100 -> O(0.01..0.06); EMISS = SFEM in [0.88, 1.0].
    assert (thc > 0.0).all() and (thc < 0.1).all(), f"thc out of (0,0.1): {thc.min()},{thc.max()}"
    assert (emiss >= 0.8).all() and (emiss <= 1.0).all(), f"emiss out of [0.8,1]: {emiss.min()},{emiss.max()}"
    assert (np.asarray(slab.tmn) > 200.0).all() and (np.asarray(slab.tmn) < 350.0).all()
    assert slab.ifsnow == 0


@pytest.mark.skipif(not _have_probe(), reason="real wrfinput PROBE absent")
def test_real_case_pleim_xiu_extraction() -> None:
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.lsm_static_extract import _load, extract_pleim_xiu_static

    run = Gen2Run(PROBE_RUN)
    px = extract_pleim_xiu_static(run, "d01")
    p = px.params
    ny, nx = np.asarray(p.wwlt).shape
    fields = [
        "vegfrc", "lai", "imperv", "canfra", "rstmin", "emissi", "znt", "wetfra",
        "hc_snow", "snow_fra", "wwlt", "wfc", "wres", "cgsat", "wsat", "b",
        "c1sat", "c2r", "asoil", "jp", "c3", "ds1", "ds2",
    ]
    for name in fields:
        arr = np.asarray(getattr(p, name))
        assert arr.shape == (ny, nx), f"PX {name} shape {arr.shape} != {(ny, nx)}"
        assert np.isfinite(arr).all(), f"PX {name} non-finite"

    # Physical ranges. Soil ordering 0<wwlt<wfc<wsat<1 must hold on LAND columns
    # (water columns are the WRF passthrough wsat=wfc=1, wwlt=0.1).
    xland = _load(run, "d01", "XLAND")
    land = xland < 1.5
    wwlt = np.asarray(p.wwlt)[land]
    wfc = np.asarray(p.wfc)[land]
    wsat = np.asarray(p.wsat)[land]
    assert (wwlt > 0.0).all(), "wwlt<=0 on land"
    assert (wwlt < wfc).all(), "wwlt>=wfc on land"
    assert (wfc < wsat).all(), "wfc>=wsat on land"
    assert (wsat < 1.0).all(), "wsat>=1 on land"
    assert (np.asarray(p.b) > 0.0).all(), "Clapp-Hornberger B<=0"
    assert (np.asarray(p.rstmin) > 0.0).all(), "rstmin<=0"
    assert (np.asarray(p.emissi) >= 0.8).all() and (np.asarray(p.emissi) <= 1.0).all()
    # vegfrc converted to a 0..1 fraction (wrfinput VEGFRA is percent).
    assert (np.asarray(p.vegfrc) >= 0.0).all() and (np.asarray(p.vegfrc) <= 1.0).all()
    # PX-only absent input fields default to 0 (LANDUSE.TBL "no impact").
    assert np.asarray(p.imperv).max() == 0.0
    assert np.asarray(p.canfra).max() == 0.0
    assert np.asarray(p.wetfra).max() == 0.0
    assert px.ifsnow == 0


@pytest.mark.skipif(not _have_probe(), reason="real wrfinput PROBE absent")
def test_real_case_water_columns_use_wrf_passthrough() -> None:
    """Water columns (XLAND>=1.5) get WRF's FWWLT=0.1/FWFC=1/FWSAT=1 (pxlsm:447-457)."""
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.lsm_static_extract import _load, extract_pleim_xiu_static

    run = Gen2Run(PROBE_RUN)
    p = extract_pleim_xiu_static(run, "d01").params
    xland = _load(run, "d01", "XLAND")
    water = xland >= 1.5
    if not water.any():
        pytest.skip("no water columns in this case")
    assert np.allclose(np.asarray(p.wwlt)[water], 0.1)
    assert np.allclose(np.asarray(p.wfc)[water], 1.0)
    assert np.allclose(np.asarray(p.wsat)[water], 1.0)


# --------------------------------------------------------------------------- #
# 3. Operational scan accepts sf=1 / sf=7 with the extracted bundle           #
# --------------------------------------------------------------------------- #
def _operational_namelist(**overrides):
    """Minimal OperationalNamelist for _resolve_operational_suite (T3 pattern)."""
    from gpuwrf.runtime.operational_mode import OperationalNamelist

    base = OperationalNamelist.__new__(OperationalNamelist)
    defaults = dict(
        mp_physics=8, bl_pbl_physics=5, sf_sfclay_physics=5, cu_physics=0,
        sf_surface_physics=None, use_noahmp=False,
        ra_sw_physics=4, ra_lw_physics=4,
        use_flux_advection=False, moist_adv_opt=0,
        noahclassic_static=None, noahclassic_land=None, noahclassic_rad=None,
        slab_static=None, slab_land=None, slab_rad=None,
        px_static=None, px_land=None, px_rad=None,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        object.__setattr__(base, k, v)
    return base


@pytest.mark.skipif(not _have_probe(), reason="real wrfinput PROBE absent")
def test_operational_scan_accepts_slab_with_extracted_bundle() -> None:
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.lsm_static_extract import extract_slab_static
    from gpuwrf.runtime.operational_mode import (
        UnsupportedSchemeSelection,
        _resolve_operational_suite,
    )

    run = Gen2Run(PROBE_RUN)
    slab = extract_slab_static(run, "d01")
    # Fails closed without the bundle...
    with pytest.raises(UnsupportedSchemeSelection):
        _resolve_operational_suite(_operational_namelist(sf_surface_physics=1))
    # ...and resolves (sf=1 NOT in not_wired) with the extracted bundle.
    suite = _resolve_operational_suite(
        _operational_namelist(sf_surface_physics=1, slab_static=slab)
    )
    assert suite.land_surface.option == 1


@pytest.mark.skipif(not _have_probe(), reason="real wrfinput PROBE absent")
def test_operational_scan_accepts_pleim_xiu_with_extracted_bundle() -> None:
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.lsm_static_extract import extract_pleim_xiu_static
    from gpuwrf.runtime.operational_mode import (
        UnsupportedSchemeSelection,
        _resolve_operational_suite,
    )

    run = Gen2Run(PROBE_RUN)
    px = extract_pleim_xiu_static(run, "d01")
    with pytest.raises(UnsupportedSchemeSelection):
        _resolve_operational_suite(_operational_namelist(sf_surface_physics=7))
    suite = _resolve_operational_suite(
        _operational_namelist(sf_surface_physics=7, px_static=px)
    )
    assert suite.land_surface.option == 7


# --------------------------------------------------------------------------- #
# Scheme-specific soil geometry (regression: the slab is a FIXED 5-layer model #
# and PX a 2-layer ISBA -- their bundles must NOT inherit the wrfinput Noah    #
# 4-layer ZS/DZS, else the operational slab solve fails the num_soil_layers=5  #
# contract and PX runs the wrong force-restore layer thicknesses).             #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _have_probe(), reason="real wrfinput PROBE absent")
def test_slab_bundle_uses_wrf_5layer_geometry_not_wrfinput() -> None:
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.lsm_static_extract import SLAB_DZS, SLAB_ZS, extract_slab_static
    from gpuwrf.physics.lsm_slab import NUM_SOIL_LAYERS

    slab = extract_slab_static(Gen2Run(PROBE_RUN), "d01")
    zs, dzs = np.asarray(slab.zs), np.asarray(slab.dzs)
    # The slab is ALWAYS the WRF 5-layer thermal-diffusion model.
    assert zs.shape == (NUM_SOIL_LAYERS,) == (5,)
    assert dzs.shape == (5,)
    np.testing.assert_allclose(dzs, [0.01, 0.02, 0.04, 0.08, 0.16])
    np.testing.assert_allclose(zs, [0.005, 0.02, 0.05, 0.11, 0.23])
    np.testing.assert_allclose(zs, SLAB_ZS)
    np.testing.assert_allclose(dzs, SLAB_DZS)


@pytest.mark.skipif(not _have_probe(), reason="real wrfinput PROBE absent")
def test_pleim_xiu_bundle_uses_wrf_isba_layer_thicknesses() -> None:
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.lsm_static_extract import PX_DS1, PX_DS2, extract_pleim_xiu_static

    px = extract_pleim_xiu_static(Gen2Run(PROBE_RUN), "d01").params
    ds1, ds2 = np.asarray(px.ds1), np.asarray(px.ds2)
    # WRF PX 2-layer force-restore ISBA: ~1 cm surface over ~1 m root zone.
    assert np.allclose(ds1, PX_DS1) and PX_DS1 == 0.01
    assert np.allclose(ds2, PX_DS2) and PX_DS2 == 0.99
    # NOT the wrfinput Noah first-two DZS (which are O(0.1 m)).
    assert float(ds1.flat[0]) < 0.05
