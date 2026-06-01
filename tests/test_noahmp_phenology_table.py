"""Oracle parity test for Noah-MP table phenology (Sprint S5, dveg=4).

Gate (ADR-NOAHMP-INTERFACES.md §6.3): pristine-WRF savepoint parity for
PHENOLOGY. No S0b savepoint fixtures exist under ``proofs/noahmp/`` yet, so this
test uses the contract's fallback: a NumPy oracle that replicates the WRF Fortran
PHENOLOGY arithmetic *line-for-line* (``module_sf_noahmplsm.F:1255-1358``, dveg=4
+ croptype==0 branch) plus the caller's FVEG block (``:863-875``), driven by the
MODIS (``MODIFIED_IGBP_MODIS_NOAH``) parameter table taken verbatim from
``run/MPTABLE.TBL``. LAI/SAI/ELAI/ESAI/FVEG/IGS must match WRF across all 20
MODIS land categories over the full annual cycle, both hemispheres, and a snow-
burial sweep that exercises the short-canopy override.

When real WRF Noah-MP savepoints land under ``proofs/noahmp/``, add a second
parity test against the dumped fields; the JAX port is the same.
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic  # noqa: E402
from gpuwrf.physics.noahmp.phenology import (  # noqa: E402
    ISBARREN_MODIS,
    ISICE_MODIS,
    ISURBAN_MODIS,
    ISWATER_MODIS,
    noahmp_phenology_table,
)
from gpuwrf.physics.noahmp.tables import NoahMPParameters  # noqa: E402
from gpuwrf.physics.noahmp.types import NoahMPForcing  # noqa: E402

# --- MODIS parameter table (verbatim from run/MPTABLE.TBL &noahmp_modis_parameters) ---
# 20 categories; rows are the 12 monthly values JAN..DEC.
_LAIM = np.array(
    [
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],  # 1 ENF
        [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5],  # 2 EBF
        [0.0, 0.0, 0.0, 0.6, 1.2, 2.0, 2.6, 1.7, 1.0, 0.5, 0.2, 0.0],  # 3 DNF
        [0.0, 0.0, 0.3, 1.2, 3.0, 4.7, 4.5, 3.4, 1.2, 0.3, 0.0, 0.0],  # 4 DBF
        [2.0, 2.0, 2.2, 2.6, 3.5, 4.3, 4.3, 3.7, 2.6, 2.2, 2.0, 2.0],  # 5 MF
        [0.0, 0.0, 0.3, 0.9, 2.2, 3.5, 3.5, 2.5, 0.9, 0.3, 0.0, 0.0],  # 6 CSh
        [0.0, 0.0, 0.2, 0.6, 1.5, 2.3, 2.3, 1.7, 0.6, 0.2, 0.0, 0.0],  # 7 OSh
        [0.2, 0.2, 0.4, 1.0, 2.4, 4.1, 4.1, 2.7, 1.0, 0.4, 0.2, 0.2],  # 8 WSav
        [0.3, 0.3, 0.5, 0.8, 1.8, 3.6, 3.8, 2.1, 0.9, 0.5, 0.3, 0.3],  # 9 Sav
        [0.4, 0.5, 0.6, 0.7, 1.2, 3.0, 3.5, 1.5, 0.7, 0.6, 0.5, 0.4],  # 10 Grass
        [0.2, 0.3, 0.3, 0.5, 1.5, 2.9, 3.5, 2.7, 1.2, 0.3, 0.3, 0.2],  # 11 PWet
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 1.5, 0.0, 0.0, 0.0],  # 12 Crop
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 13 Urban
        [0.2, 0.3, 0.3, 0.4, 1.1, 2.5, 3.2, 2.2, 1.1, 0.3, 0.3, 0.2],  # 14 CropNat
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 15 Ice
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 16 Barren
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 17 Water
        [1.0, 1.0, 1.1, 1.3, 1.7, 2.1, 2.1, 1.8, 1.3, 1.1, 1.0, 1.0],  # 18 WTundra
        [0.6, 0.6, 0.7, 0.8, 1.2, 1.8, 1.8, 1.3, 0.8, 0.7, 0.6, 0.6],  # 19 MTundra
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 20 BTundra
    ],
    dtype=np.float64,
)
_SAIM = np.array(
    [
        [0.4, 0.4, 0.4, 0.3, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.6, 0.5],  # 1
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # 2
        [0.3, 0.3, 0.3, 0.4, 0.4, 0.7, 1.3, 1.2, 1.0, 0.8, 0.6, 0.5],  # 3
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.9, 1.2, 1.6, 1.4, 0.6, 0.4],  # 4
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.8, 1.0, 1.0, 0.5, 0.4],  # 5
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6, 0.9, 1.2, 0.9, 0.4, 0.3],  # 6
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.6, 0.8, 0.7, 0.3, 0.2],  # 7
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 1.2, 1.4, 1.1, 0.5, 0.4],  # 8
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.8, 1.2, 1.3, 0.7, 0.4, 0.4],  # 9
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.8, 1.3, 1.1, 0.4, 0.4, 0.4],  # 10
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.6, 0.9, 0.9, 0.6, 0.4, 0.3],  # 11
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.4, 0.3, 0.3, 0.3],  # 12
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 13
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.6, 0.9, 0.7, 0.3, 0.3, 0.3],  # 14
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 15
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 16
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 17
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.6, 0.8, 0.7, 0.3, 0.2],  # 18
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.7, 0.5, 0.3, 0.2],  # 19
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 20
    ],
    dtype=np.float64,
)
_HVT = np.array(
    [20.0, 20.0, 18.0, 16.0, 16.0, 1.10, 1.10, 13.0, 10.0, 1.00,
     5.00, 2.00, 15.0, 1.50, 0.00, 0.00, 0.00, 4.00, 2.00, 0.50],
    dtype=np.float64,
)
_HVB = np.array(
    [8.50, 8.00, 7.00, 11.5, 10.0, 0.10, 0.10, 0.10, 0.10, 0.05,
     0.10, 0.10, 1.00, 0.10, 0.00, 0.00, 0.00, 0.30, 0.20, 0.10],
    dtype=np.float64,
)
_TMIN = np.array(
    [265, 273, 268, 273, 268, 273, 273, 273, 273, 273,
     268, 273, 0, 273, 0, 0, 0, 268, 268, 268],
    dtype=np.float64,
)
# A representative per-category SHDMAX (yearly-max green vegetation fraction).
# Values are arbitrary-but-fixed test inputs (SHDMAX is a 2-D wrfinput field, not
# a parameter table); the port copies whatever lands in parameters.shdfac.
_SHDMAX = np.array(
    [0.80, 0.90, 0.70, 0.85, 0.80, 0.40, 0.30, 0.65, 0.55, 0.60,
     0.50, 0.70, 0.10, 0.55, 0.00, 0.02, 0.00, 0.40, 0.30, 0.10],
    dtype=np.float64,
)


def _wrf_phenology_oracle(vegtyp, snowh, tv, lat, julian, yearlen):
    """Line-for-line WRF PHENOLOGY (dveg=4, croptype=0) + caller FVEG block.

    Scalar reference. ``vegtyp`` is the 1-based MODIS category id.
    """
    c = vegtyp - 1  # 0-based row into the parameter tables

    # PHENOLOGY :1299-1316 — hemisphere day shift + monthly interpolation.
    if lat >= 0.0:
        day = julian
    else:
        day = math.fmod(julian + 0.5 * yearlen, yearlen)
    t = 12.0 * day / yearlen
    it1 = int(t + 0.5)          # Fortran INT truncates toward zero; t+0.5 >= 0
    it2 = it1 + 1
    wt1 = (it1 + 0.5) - t
    wt2 = 1.0 - wt1
    if it1 < 1:
        it1 = 12
    if it2 > 12:
        it2 = 1
    lai = wt1 * _LAIM[c, it1 - 1] + wt2 * _LAIM[c, it2 - 1]
    sai = wt1 * _SAIM[c, it1 - 1] + wt2 * _SAIM[c, it2 - 1]

    # :1324-1325 floor checks
    if sai < 0.05:
        sai = 0.0
    if lai < 0.05 or sai == 0.0:
        lai = 0.0

    # :1327-1331 water/barren/ice/urban -> 0
    if vegtyp in (ISWATER_MODIS, ISBARREN_MODIS, ISICE_MODIS, ISURBAN_MODIS):
        lai = 0.0
        sai = 0.0

    # :1337-1343 snow burial
    hvt, hvb = _HVT[c], _HVB[c]
    db = min(max(snowh - hvb, 0.0), hvt - hvb)
    fb = db / max(1.0e-06, hvt - hvb)
    if 0.0 < hvt <= 1.0:
        snowhc = hvt * math.exp(-snowh / 0.2)
        fb = min(snowh, snowhc) / snowhc

    # :1345-1348 exposed + floor
    elai = lai * (1.0 - fb)
    esai = sai * (1.0 - fb)
    if esai < 0.05:
        esai = 0.0
    if elai < 0.05 or esai == 0.0:
        elai = 0.0

    # caller :863-875 — FVEG for dveg=4 = SHDMAX
    fveg = _SHDMAX[c]
    if fveg <= 0.05:
        fveg = 0.05
    if vegtyp in (ISURBAN_MODIS, ISBARREN_MODIS):
        fveg = 0.0
    if (elai + esai) == 0.0:
        fveg = 0.0

    # :1352-1356 growing season index
    igs = 1.0 if tv > _TMIN[c] else 0.0

    return lai, sai, elai, esai, fveg, igs


def _build_inputs(vegtyp_grid, snowh_grid, tv_grid, lat_grid, julian, yearlen):
    """Pack scalar test grids into the frozen pytrees the port consumes."""
    ny, nx = vegtyp_grid.shape
    c = (vegtyp_grid - 1).astype(np.int64)  # 0-based gather into MODIS rows

    def per_col(table_2d):  # table_2d: (20, 12) -> (12, ny, nx)
        return jnp.asarray(np.moveaxis(table_2d[c], -1, 0), dtype=jnp.float64)

    def per_col_scalar(table_1d):  # (20,) -> (ny, nx)
        return jnp.asarray(table_1d[c], dtype=jnp.float64)

    z2 = jnp.zeros((ny, nx), dtype=jnp.float64)
    s0 = jnp.float64(0.0)
    # Build the real (S0b) NoahMPParameters schema. Phenology reads only
    # hvt/hvb/saim/laim from parameters; SHDMAX is NOT a table param — it is a 2-D
    # wrfinput field passed via NoahMPStatic.shdmax (arbiter module_sf_noahmplsm.F:864).
    # All other table fields are zero placeholders (unread by phenology).
    veg_kw = dict(
        rhol=z2, rhos=z2, taul=z2, taus=z2, xl=z2, z0mvt=z2,
        hvt=per_col_scalar(_HVT), hvb=per_col_scalar(_HVB),
        dleaf=z2, rc=z2, den=z2, cwpvt=z2,
        saim=per_col(_SAIM), laim=per_col(_LAIM), sla=z2, ch2op=z2, nroot=z2,
        mfsno=z2, scffac=z2,
        rsmin=z2, rsmax=z2, rgl=z2, hs=z2, topt=z2, bp=z2, mp=z2, c3psn=z2,
        kc25=z2, akc=z2, ko25=z2, ako=z2, vcmx25=z2, avcmx=z2, qe25=z2, aqe=z2,
        folnmx=z2,
    )
    soil_kw = dict(
        bexp=z2, smcmax=z2, smcref=z2, smcwlt=z2, smcdry=z2, dksat=z2, dwsat=z2,
        psisat=z2, quartz=z2, albsat=z2, albdry=z2,
    )
    gen_kw = dict(
        csoil=s0, zbot=s0, czil=s0, refdk=s0, refkdt=s0, frzk=s0,
        slope=jnp.zeros(1, dtype=jnp.float64), eg=jnp.zeros(2, dtype=jnp.float64),
        omegas=jnp.zeros(2, dtype=jnp.float64), betads=s0, betais=s0,
        swemx=s0, z0sno=s0, ssi=s0, snow_ret_fac=s0, snow_emis=s0,
        iswater=ISWATER_MODIS, isbarren=ISBARREN_MODIS, isice=ISICE_MODIS,
        iscrop=0, isurban=ISURBAN_MODIS,
    )
    params = NoahMPParameters(**veg_kw, **soil_kw, **gen_kw)

    # NoahMPLandState: only snowh + tv are read by phenology; rest are placeholders.
    def land_field(shape):
        return jnp.zeros(shape, dtype=jnp.float64)

    soil = (4, ny, nx)
    snow = (3, ny, nx)
    snowsoil = (7, ny, nx)
    land = NoahMPLandState(
        tslb=land_field(soil), smois=land_field(soil), sh2o=land_field(soil),
        smcwtd=z2, isnow=jnp.zeros((ny, nx), dtype=jnp.int32),
        tsno=land_field(snow), snice=land_field(snow), snliq=land_field(snow),
        zsnso=land_field(snowsoil), snowh=jnp.asarray(snowh_grid, dtype=jnp.float64),
        sneqv=z2, sneqvo=z2, tauss=z2, albold=z2,
        tv=jnp.asarray(tv_grid, dtype=jnp.float64), tg=z2, tah=z2, eah=z2,
        canliq=z2, canice=z2, fwet=z2, lai=z2, sai=z2, cm=z2, ch=z2,
        t_skin=z2, qsfc=z2, znt=z2, emiss=z2, albedo=z2, sfcrunoff=z2, udrunoff=z2,
    )

    static = NoahMPStatic(
        ivgtyp=jnp.asarray(vegtyp_grid, dtype=jnp.int32),
        isltyp=jnp.ones((ny, nx), dtype=jnp.int32),
        xland=z2, landmask=z2, lakemask=z2,
        lu_index=jnp.asarray(vegtyp_grid, dtype=jnp.int32),
        tbot=z2, dzs=jnp.zeros(4, dtype=jnp.float64),
        zsoil=jnp.zeros(4, dtype=jnp.float64),
        lat=jnp.asarray(lat_grid, dtype=jnp.float64),
        dx_m=3000.0, parameters=params,
        # SHDMAX is the dveg=4 FVEG source (2-D wrfinput field, = VEGMAX/100).
        shdmax=per_col_scalar(_SHDMAX),
        shdfac=per_col_scalar(_SHDMAX),
    )

    forcing = NoahMPForcing(
        sfctmp=z2, sfcprs=z2, psfc=z2, uu=z2, vv=z2, qair=z2, qc=z2,
        soldn=z2, lwdn=z2, prcpconv=z2, prcpnonc=z2, prcpsnow=z2, prcpgrpl=z2,
        prcphail=z2, cosz=z2, zlvl=z2,
        julian=jnp.float64(julian), yearlen=jnp.float64(yearlen),
    )
    return land, forcing, static


def test_phenology_annual_cycle_all_categories():
    """LAI/SAI/ELAI/ESAI/FVEG/IGS parity over the full year x all 20 categories."""
    yearlen = 365.0
    cats = np.arange(1, 21, dtype=np.int64)
    # snow-free, warm canopy (so IGS=1 where TMIN>0); NH.
    tv = 290.0
    lat = 28.0  # Canary latitude (NH)
    julians = np.linspace(0.0, yearlen - 1.0, 60)  # ~6-day stride through the year

    max_abs = {k: 0.0 for k in ("lai", "sai", "elai", "esai", "fveg", "igs")}
    for julian in julians:
        veg = cats.reshape(1, -1)
        snowh = np.zeros_like(veg, dtype=np.float64)
        tvg = np.full_like(veg, tv, dtype=np.float64)
        latg = np.full_like(veg, lat, dtype=np.float64)
        land, forcing, static = _build_inputs(veg, snowh, tvg, latg, float(julian), yearlen)
        out = noahmp_phenology_table(land, forcing, static)
        got = {k: np.asarray(getattr(out, k)) for k in max_abs}
        for j, vt in enumerate(cats):
            o = _wrf_phenology_oracle(int(vt), 0.0, tv, lat, float(julian), yearlen)
            for k, ref in zip(("lai", "sai", "elai", "esai", "fveg", "igs"), o):
                max_abs[k] = max(max_abs[k], abs(got[k][0, j] - ref))

    for k, v in max_abs.items():
        assert v < 1.0e-12, f"{k} max abs error {v:.3e} exceeds 1e-12"


def test_phenology_snow_burial_and_short_canopy():
    """Snow-burial reduction incl. the short-canopy (0<HVT<=1) exponential override."""
    yearlen = 365.0
    # categories spanning tall canopy (forest), short canopy (shrub/grass), no-veg.
    cats = np.array([[1, 6, 7, 10, 13, 16, 17]], dtype=np.int64)
    julian = 196.0  # mid-July (peak NH LAI)
    lat = 45.0
    for snowh_val in (0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0):
        snowh = np.full_like(cats, snowh_val, dtype=np.float64)
        tvg = np.full_like(cats, 285.0, dtype=np.float64)
        latg = np.full_like(cats, lat, dtype=np.float64)
        land, forcing, static = _build_inputs(cats, snowh, tvg, latg, julian, yearlen)
        out = noahmp_phenology_table(land, forcing, static)
        got = {k: np.asarray(getattr(out, k)) for k in ("lai", "sai", "elai", "esai", "fveg", "igs")}
        for j, vt in enumerate(cats[0]):
            o = _wrf_phenology_oracle(int(vt), snowh_val, 285.0, lat, julian, yearlen)
            for k, ref in zip(("lai", "sai", "elai", "esai", "fveg", "igs"), o):
                err = abs(got[k][0, j] - ref)
                assert err < 1.0e-12, (
                    f"cat {vt} snowh {snowh_val}: {k} err {err:.3e} (got {got[k][0,j]}, ref {ref})"
                )


def test_phenology_southern_hemisphere_shift():
    """SH latitude shifts the day-of-year by half a year (DAY=mod(J+0.5*YL, YL))."""
    yearlen = 365.0
    cats = np.array([[4, 9, 10, 18]], dtype=np.int64)  # deciduous/savanna/grass/tundra
    lat = -33.0  # Southern Hemisphere
    for julian in (15.0, 105.0, 196.0, 288.0, 350.0):
        veg = cats
        snowh = np.zeros_like(veg, dtype=np.float64)
        tvg = np.full_like(veg, 288.0, dtype=np.float64)
        latg = np.full_like(veg, lat, dtype=np.float64)
        land, forcing, static = _build_inputs(veg, snowh, tvg, latg, julian, yearlen)
        out = noahmp_phenology_table(land, forcing, static)
        got = {k: np.asarray(getattr(out, k)) for k in ("lai", "sai", "elai", "esai", "fveg", "igs")}
        for j, vt in enumerate(cats[0]):
            o = _wrf_phenology_oracle(int(vt), 0.0, 288.0, lat, julian, yearlen)
            for k, ref in zip(("lai", "sai", "elai", "esai", "fveg", "igs"), o):
                assert abs(got[k][0, j] - ref) < 1.0e-12, f"SH cat {vt} julian {julian}: {k}"


def test_phenology_igs_threshold_at_tmin():
    """IGS flips exactly at TV > TMIN (per-category photosynthesis floor)."""
    yearlen = 365.0
    cats = np.array([[1, 5, 10]], dtype=np.int64)  # TMIN = 265, 268, 273
    julian = 196.0
    lat = 28.0
    for tv in (260.0, 265.0, 265.001, 268.0, 270.0, 273.0, 274.0):
        veg = cats
        snowh = np.zeros_like(veg, dtype=np.float64)
        tvg = np.full_like(veg, tv, dtype=np.float64)
        latg = np.full_like(veg, lat, dtype=np.float64)
        land, forcing, static = _build_inputs(veg, snowh, tvg, latg, julian, yearlen)
        out = noahmp_phenology_table(land, forcing, static)
        igs = np.asarray(out.igs)
        for j, vt in enumerate(cats[0]):
            _, _, _, _, _, ref = _wrf_phenology_oracle(int(vt), 0.0, tv, lat, julian, yearlen)
            assert igs[0, j] == ref, f"IGS cat {vt} tv {tv}: got {igs[0,j]} ref {ref}"


def test_phenology_output_dtype_and_shape():
    """fp64 outputs, correct (ny, nx) shape, finite."""
    veg = np.array([[1, 6, 10], [13, 16, 17]], dtype=np.int64)
    snowh = np.zeros_like(veg, dtype=np.float64)
    tvg = np.full_like(veg, 290.0, dtype=np.float64)
    latg = np.full_like(veg, 28.0, dtype=np.float64)
    land, forcing, static = _build_inputs(veg, snowh, tvg, latg, 100.0, 365.0)
    out = noahmp_phenology_table(land, forcing, static)
    for k in ("lai", "sai", "elai", "esai", "fveg", "igs"):
        arr = getattr(out, k)
        assert arr.dtype == jnp.float64, f"{k} not fp64"
        assert arr.shape == (2, 3), f"{k} shape {arr.shape}"
        assert bool(jnp.all(jnp.isfinite(arr))), f"{k} not finite"


def test_phenology_jit_compiles():
    """The port traces and jit-compiles (operational fp64 path, no host transfer)."""
    veg = np.array([[1, 6, 10, 16]], dtype=np.int64)
    snowh = np.full_like(veg, 0.2, dtype=np.float64)
    tvg = np.full_like(veg, 285.0, dtype=np.float64)
    latg = np.full_like(veg, 28.0, dtype=np.float64)
    land, forcing, static = _build_inputs(veg, snowh, tvg, latg, 200.0, 365.0)
    jitted = jax.jit(noahmp_phenology_table)
    out = jitted(land, forcing, static)
    ref = noahmp_phenology_table(land, forcing, static)
    for k in ("lai", "sai", "elai", "esai", "fveg", "igs"):
        assert np.allclose(np.asarray(getattr(out, k)), np.asarray(getattr(ref, k)), atol=0.0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
