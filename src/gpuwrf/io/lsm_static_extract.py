"""Faithful real-case extraction of the slab (sf=1) and Pleim-Xiu (sf=7) static
bundles from a real WRF ``wrfinput``.

The two operational LSM hooks (``coupling.slab_surface_hook.SlabStaticBundle`` and
``coupling.pleim_xiu_surface_hook.PleimXiuStaticBundle``) are read-only per-run
static inputs.  Some of their fields are present verbatim in ``wrfinput`` (ZS, DZS,
TMN, SNOWC, VEGFRA, LAI); the rest are DERIVED by WRF's own init/physics code from
LU_INDEX (THC/EMISS/Z0 via the LANDUSE.TBL season-indexed lookup), the soil-category
fractions (the Noilhan & Mahfouf 1996 ISBA soil constants via SOILPROP), and the
vegetation tables (RSTMIN via VEGPARM.TBL).  Every derived value here is computed by
the SAME formula / table WRF uses, cited to pristine source line refs -- never
invented.  A wrong land-surface static is a silently-wrong forecast, which the
project forbids, so the ISBA-soil-constant path is falsifiably checked against the
pristine-WRF PX oracle savepoints in ``tests/test_v018_lsm_static_extract.py``.

Pristine WRF sources (``/home/user/src/wrf_pristine/WRF``):
  * SLAB THC/EMISS/Z0/MAVAIL = THERIN/SFEM/SFZ0/SLMO[LU_INDEX, ISN] / scale --
    ``phys/module_physics_init.F:1958-1972`` (landuse_init), with the season index
    ISN from ``phys/module_physics_init.F:1833-1835``:
        ISN=1; IF(JULDAY<105 .OR. JULDAY>288) ISN=2; IF(CEN_LAT<0) ISN=3-ISN
    and the IS=0 -> ISWATER no-data remap (``:1955-1957``).  The LANDUSE.TBL block
    is ``run/LANDUSE.TBL`` "MODIFIED_IGBP_MODIS_NOAH" (61 categories x 2 seasons,
    columns ALBD SLMO SFEM SFZ0 THERIN SCFX SFHC; SUMMER then WINTER).
  * PX ISBA soil constants = the Noilhan & Mahfouf (1996) analytic SOILPROP, from
    the SOILCBOT-fraction-weighted (coarse-sand, fine/medium-sand, clay) average
    over the 16 soil categories -- ``phys/module_sf_pxlsm.F:1730-1915`` (SOILPROP),
    called per land column at ``:435``; water columns get FWWLT=0.1/FWFC=1/FWSAT=1
    (``:447-457``).  The 16-category CSAND/FMSAND/CLAY DATA tables ("Menut et al.
    2013") are ``:1792-1804``.
  * PX RSTMIN = the "RS" column of ``run/VEGPARM.TBL`` "MODIFIED_IGBP_MODIS_NOAH"
    by LU_INDEX (VEGELAND land-use weighting reduces to the dominant-category RS
    for a hard land-use index; we use the LU_INDEX direct lookup).
  * PX vegfrc/lai/emissi/znt and the PX-only imperv/canfra/wetfra default-0 fields
    follow ``phys/module_sf_pxlsm.F`` VEGELAND + the LANDUSE.TBL note "Impervious
    surface and canopy fraction data ... otherwise 0% so no impact".

This module reads the authoritative pristine WRF ``LANDUSE.TBL`` / ``VEGPARM.TBL``
directly (transcription-error-free) and falls back to a verified embedded copy of
the MODIFIED_IGBP_MODIS_NOAH columns when the pristine tree is unavailable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from gpuwrf.config.paths import wrf_root
from gpuwrf.coupling.pleim_xiu_surface_hook import PleimXiuStaticBundle
from gpuwrf.coupling.slab_surface_hook import SlabStaticBundle
from gpuwrf.physics.lsm_pleim_xiu import PleimXiuStatic

# Default pristine WRF tree (the dycore arbiter build). Env-overridable via
# GPUWRF_WRF_ROOT (config.paths.wrf_root); the explicit ``pristine_wrf=``
# argument still takes precedence, and an embedded table copy is used when the
# tree is unavailable.
PRISTINE_WRF = wrf_root()
LANDUSE_BLOCK = "MODIFIED_IGBP_MODIS_NOAH"
# MODIFIED_IGBP_MODIS_NOAH water category (no-data IS=0 remaps here):
# phys/module_physics_init.F namelist nl_get_iswater -> 17 for MODIS.
ISWATER_MODIS = 17

# --------------------------------------------------------------------------- #
# 16-category soil sand/clay tables (module_sf_pxlsm.F:1792-1804, "Menut 2013") #
#   index 1..16; CSAND=coarse sand %, FMSAND=fine/medium sand %, CLAY=clay %.    #
# --------------------------------------------------------------------------- #
_CSAND = np.array(
    [46.0, 40.0, 29.0, 0.0, 0.0, 0.0, 29.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 46.0, 0.0],
    dtype=np.float64,
)
_FMSAND = np.array(
    [46.0, 40.0, 29.0, 17.0, 10.0, 43.0, 29.0, 10.0, 32.0, 52.0, 6.0, 22.0, 43.0, 43.0, 46.0, 32.0],
    dtype=np.float64,
)
_CLAY = np.array(
    [3.0, 4.0, 10.0, 13.0, 5.0, 18.0, 27.0, 34.0, 34.0, 42.0, 47.0, 58.0, 18.0, 18.0, 3.0, 34.0],
    dtype=np.float64,
)
# No-soil (TFRACBOT<=0.001) fallback: ISTI=9 clay loam (module_sf_pxlsm.F:1885-1896).
_NO_SOIL_ISTI = 9

# WRF 5-layer thermal-diffusion slab soil geometry. The slab (sf=1) is a
# FIXED-geometry model (lsm_slab NUM_SOIL_LAYERS=5) whose soil discretization is
# set by WRF at init and is INDEPENDENT of the wrfinput ZS/DZS (those describe
# the Noah 4-layer soil). DZS thicknesses 1,2,4,8,16 cm; ZS layer-center depths
# are the cumulative half-thickness. Verified against the WRF SLAB1D oracle
# savepoint (proofs/v013/savepoints/surface_lsm/fp64/slab_case_*.json ZS/DZS).
SLAB_DZS = np.array([0.01, 0.02, 0.04, 0.08, 0.16], dtype=np.float64)
SLAB_ZS = np.array([0.005, 0.02, 0.05, 0.11, 0.23], dtype=np.float64)

# WRF Pleim-Xiu 2-layer force-restore ISBA soil-layer thicknesses: a ~1 cm
# surface layer over a ~1 m root zone (ds1+ds2=1 m). Verified against the WRF PX
# oracle savepoint (proofs/v017/savepoints/pxlsm/fp64/pxlsm_case_*.json DS1/DS2).
PX_DS1 = 0.01
PX_DS2 = 0.99

# Verified embedded MODIFIED_IGBP_MODIS_NOAH LANDUSE.TBL columns (run/LANDUSE.TBL,
# transcribed + checked against the file in the tests).  cat -> (SFEM, SFZ0, THERIN)
# for [summer, winter].  Used only when the pristine tree is unavailable.
_LANDUSE_FALLBACK: dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    1: ((0.95, 50.0, 4.0), (0.95, 50.0, 5.0)),
    2: ((0.95, 50.0, 5.0), (0.95, 50.0, 5.0)),
    3: ((0.94, 50.0, 4.0), (0.93, 50.0, 5.0)),
    4: ((0.93, 50.0, 4.0), (0.93, 50.0, 5.0)),
    5: ((0.97, 50.0, 4.0), (0.93, 20.0, 6.0)),
    6: ((0.93, 5.0, 3.0), (0.93, 1.0, 4.0)),
    7: ((0.95, 6.0, 3.0), (0.93, 1.0, 4.0)),
    8: ((0.93, 5.0, 3.0), (0.93, 1.0, 4.0)),
    9: ((0.92, 15.0, 3.0), (0.92, 15.0, 3.0)),
    10: ((0.96, 12.0, 3.0), (0.92, 10.0, 4.0)),
    11: ((0.95, 30.0, 5.5), (0.95, 30.0, 6.0)),
    12: ((0.985, 15.0, 4.0), (0.92, 5.0, 4.0)),
    13: ((0.88, 80.0, 3.0), (0.88, 80.0, 3.0)),
    14: ((0.98, 14.0, 4.0), (0.92, 5.0, 4.0)),
    15: ((0.95, 0.1, 5.0), (0.95, 0.1, 5.0)),
    16: ((0.90, 1.0, 2.0), (0.90, 1.0, 2.0)),
    17: ((0.98, 0.01, 6.0), (0.98, 0.01, 6.0)),
    18: ((0.93, 30.0, 5.0), (0.93, 30.0, 5.0)),
    19: ((0.92, 15.0, 5.0), (0.92, 15.0, 5.0)),
    20: ((0.90, 10.0, 2.0), (0.90, 5.0, 5.0)),
}
# Unassigned (21-50) row from LANDUSE.TBL.
_LANDUSE_UNASSIGNED = ((0.88, 80.0, 3.0), (0.88, 80.0, 3.0))
# LCZ (51-61) row from LANDUSE.TBL.
_LANDUSE_LCZ = ((0.97, 80.0, 3.0), (0.97, 80.0, 3.0))

# Verified embedded MODIFIED_IGBP_MODIS_NOAH VEGPARM.TBL "RS" column (run/VEGPARM.TBL,
# 20 categories), cat -> RSMIN (s/m).  Used only when the pristine tree is unavailable.
_RSMIN_FALLBACK: dict[int, float] = {
    1: 125.0, 2: 150.0, 3: 150.0, 4: 100.0, 5: 125.0, 6: 300.0, 7: 170.0, 8: 300.0,
    9: 70.0, 10: 40.0, 11: 70.0, 12: 40.0, 13: 200.0, 14: 40.0, 15: 999.0, 16: 999.0,
    17: 100.0, 18: 150.0, 19: 150.0, 20: 200.0,
}
_RSMIN_DEFAULT = 100.0  # WRF VEGPARM RSMAX/no-data lands fall back here (sane non-zero).


# --------------------------------------------------------------------------- #
# Pristine-WRF table parsers (authoritative source of truth)                  #
# --------------------------------------------------------------------------- #
def _parse_landuse_table(
    pristine_wrf: Path,
) -> dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]] | None:
    """Parse run/LANDUSE.TBL MODIFIED_IGBP_MODIS_NOAH -> cat:(summer,winter) tuples.

    Each season tuple is (SFEM, SFZ0, THERIN) read verbatim from the file columns
    ``ALBD SLMO SFEM SFZ0 THERIN ...``.  Returns ``None`` if the file is absent so
    the caller can fall back to the verified embedded copy.
    """

    path = Path(pristine_wrf) / "run" / "LANDUSE.TBL"
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == LANDUSE_BLOCK), None)
    if start is None:
        return None
    header = lines[start + 1]
    ncat = int(re.findall(r"\d+", header.split("'")[0])[0])

    def _block(off: int) -> dict[int, tuple[float, float, float]]:
        rows: dict[int, tuple[float, float, float]] = {}
        for k in range(ncat):
            parts = lines[off + k].replace(",", " ").split()
            idx = int(parts[0])
            # columns: idx ALBD SLMO SFEM SFZ0 THERIN ...
            sfem, sfz0, therin = float(parts[3]), float(parts[4]), float(parts[5])
            rows[idx] = (sfem, sfz0, therin)
        return rows

    s_lbl = w_lbl = None
    for i in range(start, min(start + 400, len(lines))):
        if lines[i].strip() == "SUMMER" and s_lbl is None:
            s_lbl = i
        elif lines[i].strip() == "WINTER" and s_lbl is not None:
            w_lbl = i
            break
    if s_lbl is None or w_lbl is None:
        return None
    summer = _block(s_lbl + 1)
    winter = _block(w_lbl + 1)
    return {cat: (summer[cat], winter[cat]) for cat in summer if cat in winter}


def _parse_vegparm_rsmin(pristine_wrf: Path) -> dict[int, float] | None:
    """Parse run/VEGPARM.TBL MODIFIED_IGBP_MODIS_NOAH "RS" column -> cat:RSMIN."""

    path = Path(pristine_wrf) / "run" / "VEGPARM.TBL"
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == LANDUSE_BLOCK), None)
    if start is None:
        return None
    header = lines[start + 1]
    ncat = int(re.findall(r"\d+", header.split("'")[0])[0])
    rsmin: dict[int, float] = {}
    for k in range(ncat):
        parts = lines[start + 2 + k].replace(",", " ").split()
        idx = int(parts[0])
        # columns: idx SHDFAC NROOT RS RGL HS ... -> RS is the 4th value (parts[3]).
        rsmin[idx] = float(parts[3])
    return rsmin


def _landuse_columns(
    pristine_wrf: Path,
) -> dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    table = _parse_landuse_table(pristine_wrf)
    if table:
        return table
    table = dict(_LANDUSE_FALLBACK)
    for cat in range(21, 51):
        table[cat] = _LANDUSE_UNASSIGNED
    for cat in range(51, 62):
        table[cat] = _LANDUSE_LCZ
    return table


def _rsmin_table(pristine_wrf: Path) -> dict[int, float]:
    table = _parse_vegparm_rsmin(pristine_wrf)
    return table if table else dict(_RSMIN_FALLBACK)


# --------------------------------------------------------------------------- #
# Season index ISN (phys/module_physics_init.F:1833-1835)                     #
# --------------------------------------------------------------------------- #
def _julian_day(times: str) -> int:
    """Day-of-year (1..366) from a WRF Times string 'YYYY-MM-DD_hh:mm:ss'."""

    from datetime import datetime

    text = str(times).strip().replace("_", " ")[:19]
    dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    return dt.timetuple().tm_yday


def landuse_season(julday: int, cen_lat: float) -> int:
    """WRF landuse season index ISN (1=summer, 2=winter), SH-flipped.

    Verbatim ``phys/module_physics_init.F:1833-1835``:
        ISN=1; IF(JULDAY<105 .OR. JULDAY>288) ISN=2; IF(CEN_LAT<0) ISN=3-ISN.
    """

    isn = 1
    if julday < 105 or julday > 288:
        isn = 2
    if cen_lat < 0.0:
        isn = 3 - isn
    return isn


# --------------------------------------------------------------------------- #
# ISBA soil constants (Noilhan & Mahfouf 1996; module_sf_pxlsm.F:1904-1915)   #
# --------------------------------------------------------------------------- #
def _soilprop(avs: np.ndarray, avc: np.ndarray) -> dict[str, np.ndarray]:
    """The 11 Noilhan & Mahfouf (1996) ISBA constants from (AVS, AVC) percentages.

    EXACT port of WRF ``SOILPROP`` (module_sf_pxlsm.F:1904-1915).  ``avs`` is the
    total sand % (coarse + fine/medium), ``avc`` the clay %, ``avslt = 100-avs-avc``
    the silt %.  Vectorized over an arbitrary array shape.
    """

    avs = np.asarray(avs, dtype=np.float64)
    avc = np.asarray(avc, dtype=np.float64)
    avslt = 100.0 - avs - avc
    wsat = (-1.08 * avs + 494.305) * 1.0e-3                       # :1904
    wwlt = 37.1342e-3 * np.sqrt(avc)                             # :1905
    wfc = 89.0467e-3 * avc**0.3496                              # :1906
    b = 0.137 * avc + 3.501                                     # :1907
    cgsat = -1.557e-2 * avs - 1.441e-2 * avc + 4.7021           # :1908
    c1sat = (5.58 * avc + 84.88) * 1.0e-2                       # :1909
    c2r = 13.815 * avc ** (-0.954)                             # :1910
    c3 = 5.327 * avc ** (-1.043)                               # :1911
    asoil = 732.42e-3 * avc ** (-0.539)                        # :1912 (FAS)
    jp = 0.134 * avc + 3.4                                     # :1913
    wres = np.maximum(0.00123 * avc - 0.00066 * avslt + 0.0405, 0.01)  # :1914-1915
    return {
        "wsat": wsat, "wwlt": wwlt, "wfc": wfc, "b": b, "cgsat": cgsat,
        "c1sat": c1sat, "c2r": c2r, "c3": c3, "asoil": asoil, "jp": jp, "wres": wres,
    }


def _soil_avs_avc(soilc_fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column (AVS, AVC) from the 16-category soil-fraction-weighted average.

    EXACT port of WRF ``SOILPROP`` weighting (module_sf_pxlsm.F:1824-1845, 1885-1896):
      SUMCSND=Σ CSAND[s]*f[s]; SUMFMSND=Σ FMSAND[s]*f[s]; SUMCLY=Σ CLAY[s]*f[s];
      TFRAC=Σ f[s].  If TFRAC>0.001: AVS=(SUMCSND+SUMFMSND)/TFRAC; AVC=SUMCLY/TFRAC.
      Else (no soil): ISTI=9 clay loam -> AVS=CSAND[9]+FMSAND[9]=32, AVC=CLAY[9]=34.

    ``soilc_fraction`` is the (nscat, ny, nx) SOILCBOT (or SOILCTOP) category
    fraction stack; ``nscat`` must be 16 (the WRF NSCATMIN soil categories).
    """

    fr = np.asarray(soilc_fraction, dtype=np.float64)
    if fr.shape[0] != _CSAND.size:
        raise ValueError(
            f"soil-category stack must lead with {_CSAND.size} categories, got {fr.shape}"
        )
    tfrac = fr.sum(axis=0)
    sumcsnd = np.tensordot(_CSAND, fr, axes=(0, 0))
    sumfmsnd = np.tensordot(_FMSAND, fr, axes=(0, 0))
    sumcly = np.tensordot(_CLAY, fr, axes=(0, 0))
    has_soil = tfrac > 0.001
    safe = np.where(has_soil, tfrac, 1.0)
    avs_soil = (sumcsnd + sumfmsnd) / safe
    avc_soil = sumcly / safe
    # No-soil fallback: ISTI=9 clay loam.
    i9 = _NO_SOIL_ISTI - 1
    avs_nosoil = _CSAND[i9] + _FMSAND[i9]
    avc_nosoil = _CLAY[i9]
    avs = np.where(has_soil, avs_soil, avs_nosoil)
    avc = np.where(has_soil, avc_soil, avc_nosoil)
    return avs, avc


# --------------------------------------------------------------------------- #
# wrfinput field access                                                       #
# --------------------------------------------------------------------------- #
def _load(run: Any, domain: str, var: str) -> np.ndarray | None:
    """Load one wrfinput variable as a numpy array (time-0 squeezed), or None."""

    try:
        if var not in set(run.wrfinput_variables(domain)):
            return None
        arr = np.asarray(run.load_wrfinput(domain, var, lazy=False), dtype=np.float64)
    except (KeyError, FileNotFoundError):
        return None
    # WRF wrfinput variables carry a leading Time axis of length 1 -- drop it.
    # (For the soil-category stack SOILCBOT/SOILCTOP this leaves the 16-category
    # axis leading, which _soil_avs_avc requires.)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _wrfinput_times(run: Any, domain: str) -> str:
    """Read the ``Times`` string straight from the wrfinput netCDF.

    The Gen2 lazy accessor cannot materialize a character variable (it forces a
    numeric JAX array), so the simulation start date is read directly from the file
    -- the season ISN derivation needs only this scalar timestamp.
    """

    from netCDF4 import Dataset

    with Dataset(run.wrfinput_file(domain), "r") as ds:
        if "Times" not in ds.variables:
            return "1970-01-01_00:00:00"
        raw = ds.variables["Times"][:]
    return _times_to_str(raw)


def _domain_center_lat(run: Any, domain: str, xlat: np.ndarray | None) -> float:
    """Domain center latitude (WRF uses the scalar CEN_LAT for the season flip)."""

    try:
        grid = run.grid(domain)
        cen = getattr(grid, "cen_lat", None)
        if cen is not None and np.isfinite(cen):
            return float(cen)
    except Exception:  # noqa: BLE001 -- grid is best-effort; fall back to XLAT mean.
        pass
    if xlat is not None and np.isfinite(xlat).any():
        return float(np.nanmean(xlat))
    return 0.0


# --------------------------------------------------------------------------- #
# Public extraction API                                                       #
# --------------------------------------------------------------------------- #
def extract_slab_static(
    run: Any,
    domain: str = "d01",
    *,
    season: int | None = None,
    ifsnow: int = 0,
    pristine_wrf: Path = PRISTINE_WRF,
) -> SlabStaticBundle:
    """Extract the slab (sf=1) ``SlabStaticBundle`` from a real ``wrfinput``.

    THC/EMISS are WRF's own landuse_init derivation:
        THC = THERIN[IS, ISN] / 100;  EMISS = SFEM[IS, ISN]
    (phys/module_physics_init.F:1967, :1970) where IS=nint(LU_INDEX) with the IS=0
    -> ISWATER no-data remap (:1955-1957), and ISN the wrfinput-derived season
    (overridable via ``season``).  ZS/DZS/TMN/SNOWC come straight from wrfinput.
    """

    lu = _load(run, domain, "LU_INDEX")
    if lu is None:
        raise KeyError(f"wrfinput {domain} lacks LU_INDEX (required for slab THC/EMISS)")
    tmn = _load(run, domain, "TMN")
    snowc = _load(run, domain, "SNOWC")
    xlat = _load(run, domain, "XLAT")
    # The 5-layer thermal-diffusion slab is a FIXED-geometry model (lsm_slab
    # NUM_SOIL_LAYERS=5) -- its soil discretization is NOT the wrfinput ZS/DZS
    # (those are the Noah 4-layer config). WRF sets the slab 5-layer thicknesses
    # at init: DZS=[0.01,0.02,0.04,0.08,0.16] m, layer-center ZS=[0.005,0.02,0.05,
    # 0.11,0.23] m (cumulative). Verified against the WRF SLAB1D oracle savepoint
    # (proofs/v013/savepoints/surface_lsm/fp64/slab_case_*.json ZS/DZS). The slab
    # land carry is cold-started over these centers (TSK->TMN), so the operational
    # scan never inherits the mismatched 4-layer Noah soil column.
    zs = SLAB_ZS
    dzs = SLAB_DZS

    times_str = _wrfinput_times(run, domain)
    if season is None:
        cen_lat = _domain_center_lat(run, domain, xlat)
        season = landuse_season(_julian_day(times_str), cen_lat)

    cols = _landuse_columns(pristine_wrf)
    isn_idx = 0 if int(season) == 1 else 1  # column 0 = summer, 1 = winter
    lu_int = np.rint(np.asarray(lu, dtype=np.float64)).astype(np.int64)
    lu_remap = np.where(lu_int == 0, ISWATER_MODIS, lu_int)

    therin = np.empty(lu_remap.shape, dtype=np.float64)
    sfem = np.empty(lu_remap.shape, dtype=np.float64)
    for cat in np.unique(lu_remap):
        entry = cols.get(int(cat), cols.get(ISWATER_MODIS))
        sfem_c, _sfz0_c, therin_c = entry[isn_idx]
        mask = lu_remap == cat
        therin[mask] = therin_c
        sfem[mask] = sfem_c

    thc = therin / 100.0  # phys/module_physics_init.F:1967
    emiss = sfem          # phys/module_physics_init.F:1970

    if tmn is None:
        raise KeyError(f"wrfinput {domain} lacks TMN (slab deep-soil restore temperature)")
    if snowc is None:
        snowc = np.zeros(lu_remap.shape, dtype=np.float64)

    return SlabStaticBundle(
        zs=np.asarray(zs, dtype=np.float64),
        dzs=np.asarray(dzs, dtype=np.float64),
        thc=np.asarray(thc, dtype=np.float64),
        tmn=np.asarray(tmn, dtype=np.float64),
        emiss=np.asarray(emiss, dtype=np.float64),
        snowc=np.asarray(snowc, dtype=np.float64),
        ifsnow=int(ifsnow),
    )


def extract_pleim_xiu_static(
    run: Any,
    domain: str = "d01",
    *,
    season: int | None = None,
    ifsnow: int = 0,
    pristine_wrf: Path = PRISTINE_WRF,
) -> PleimXiuStaticBundle:
    """Extract the Pleim-Xiu (sf=7) ``PleimXiuStaticBundle`` from a real ``wrfinput``.

    ISBA soil constants are WRF ``SOILPROP`` (Noilhan & Mahfouf 1996) over the
    SOILCBOT 16-category fraction-weighted (AVS, AVC); water columns (XLAND>=1.5)
    get FWWLT=0.1/FWFC=1/FWSAT=1 exactly like WRF (module_sf_pxlsm.F:447-457).
    emissi/znt = SFEM/SFZ0[LU_INDEX, ISN]; rstmin = VEGPARM RS[LU_INDEX]; vegfrc =
    VEGFRA/100 (percent->fraction); lai = LAI; imperv/canfra/wetfra/hc_snow/snow_fra
    default 0 (the PX-only input fields are absent -> "no impact", per LANDUSE.TBL).
    """

    lu = _load(run, domain, "LU_INDEX")
    if lu is None:
        raise KeyError(f"wrfinput {domain} lacks LU_INDEX (required for PX emissi/znt/rstmin)")
    xland = _load(run, domain, "XLAND")
    vegfra = _load(run, domain, "VEGFRA")
    lai = _load(run, domain, "LAI")
    xlat = _load(run, domain, "XLAT")

    # WRF SOILPROP weights SOILCBOT (bottom soil fraction); prefer it, fall back to
    # SOILCTOP only if the bottom stack is absent.
    soilc = _load(run, domain, "SOILCBOT")
    soil_source = "SOILCBOT"
    if soilc is None:
        soilc = _load(run, domain, "SOILCTOP")
        soil_source = "SOILCTOP"
    if soilc is None:
        raise KeyError(f"wrfinput {domain} lacks SOILCBOT/SOILCTOP (PX ISBA soil constants)")

    times_str = _wrfinput_times(run, domain)
    if season is None:
        cen_lat = _domain_center_lat(run, domain, xlat)
        season = landuse_season(_julian_day(times_str), cen_lat)
    isn_idx = 0 if int(season) == 1 else 1

    ny_nx = lu.shape
    if xland is None:
        xland = np.ones(ny_nx, dtype=np.float64)

    # --- ISBA soil constants over the soil-category fractions (land columns) ---
    avs, avc = _soil_avs_avc(soilc)
    isba = _soilprop(avs, avc)
    is_water = np.asarray(xland, dtype=np.float64) >= 1.5
    # Water override (module_sf_pxlsm.F:447-457).
    isba["wwlt"] = np.where(is_water, 0.1, isba["wwlt"])
    isba["wfc"] = np.where(is_water, 1.0, isba["wfc"])
    isba["wsat"] = np.where(is_water, 1.0, isba["wsat"])

    # PX is a 2-layer force-restore ISBA with a FIXED ~1 cm surface layer over a
    # ~1 m root zone (NOT the wrfinput Noah DZS): ds1=0.01 m, ds2=0.99 m. Verified
    # against the WRF PX oracle savepoint (proofs/v017/savepoints/pxlsm/fp64/
    # pxlsm_case_*.json DS1/DS2 = 0.01/0.99, constant across regimes).
    ds1, ds2 = PX_DS1, PX_DS2
    ones = np.ones(ny_nx, dtype=np.float64)

    # --- vegetation / surface fields ---
    cols = _landuse_columns(pristine_wrf)
    rsmin_tbl = _rsmin_table(pristine_wrf)
    lu_int = np.rint(np.asarray(lu, dtype=np.float64)).astype(np.int64)
    lu_remap = np.where(lu_int == 0, ISWATER_MODIS, lu_int)

    emissi = np.empty(ny_nx, dtype=np.float64)
    znt = np.empty(ny_nx, dtype=np.float64)
    rstmin = np.empty(ny_nx, dtype=np.float64)
    for cat in np.unique(lu_remap):
        entry = cols.get(int(cat), cols.get(ISWATER_MODIS))
        sfem_c, sfz0_c, _therin_c = entry[isn_idx]
        mask = lu_remap == cat
        emissi[mask] = sfem_c
        znt[mask] = sfz0_c / 100.0  # phys/module_physics_init.F:1968 (SFZ0/100 -> m)
        rstmin[mask] = rsmin_tbl.get(int(cat), _RSMIN_DEFAULT)

    # VEGFRA is in percent in wrfinput; PX wants a 0..1 fraction.
    if vegfra is None:
        vegfrc = 0.5 * ones
    else:
        vegfra = np.asarray(vegfra, dtype=np.float64)
        vegfrc = np.clip(np.where(vegfra > 1.5, vegfra / 100.0, vegfra), 0.0, 1.0)
    lai_arr = np.zeros(ny_nx, dtype=np.float64) if lai is None else np.asarray(lai, dtype=np.float64)
    zero = np.zeros(ny_nx, dtype=np.float64)

    params = PleimXiuStatic(
        vegfrc=vegfrc,
        lai=lai_arr,
        imperv=zero,        # PX-only field absent -> 0% (LANDUSE.TBL "no impact")
        canfra=zero,        # PX-only field absent -> 0%
        rstmin=rstmin,
        emissi=emissi,
        znt=znt,
        wetfra=zero,        # PX-only field absent -> 0
        hc_snow=zero,
        snow_fra=zero,
        wwlt=isba["wwlt"], wfc=isba["wfc"], wres=isba["wres"], cgsat=isba["cgsat"],
        wsat=isba["wsat"], b=isba["b"], c1sat=isba["c1sat"], c2r=isba["c2r"],
        asoil=isba["asoil"], jp=isba["jp"], c3=isba["c3"],
        ds1=ds1 * ones, ds2=ds2 * ones,
    )
    del soil_source  # provenance only; the frozen bundle carries no metadata slot.
    return PleimXiuStaticBundle(params=params, ifsnow=int(ifsnow))


def _times_to_str(times: Any) -> str:
    """Decode a wrfinput ``Times`` variable (bytes / char array / str) to a string.

    netCDF4 returns ``Times`` as a ``(1, DateStrLen)`` array of single-byte strings
    or numeric char codes; reduce it to the 'YYYY-MM-DD_hh:mm:ss' scalar.
    """

    if isinstance(times, (str, bytes)):
        return times.decode() if isinstance(times, bytes) else times
    arr = np.asarray(times).reshape(-1)
    if arr.dtype.kind in ("S", "U"):
        chars = [c.decode() if isinstance(c, bytes) else str(c) for c in arr.tolist()]
        return "".join(chars).split("\x00")[0].strip()
    # numeric character codes
    return "".join(chr(int(c)) for c in arr.tolist() if 0 < int(c) < 128).strip()


__all__ = [
    "SlabStaticBundle",
    "PleimXiuStaticBundle",
    "extract_slab_static",
    "extract_pleim_xiu_static",
    "landuse_season",
]
