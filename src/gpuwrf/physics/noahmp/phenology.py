"""Noah-MP table phenology (Sprint S5, dveg=4) — IMPLEMENTATION.

Faithful JAX port of PHENOLOGY (``module_sf_noahmplsm.F:1255-1358``) restricted
to the **dveg=4** branch with **croptype == 0** (the v0.2.0 scope, ADR-NOAHMP-
INTERFACES.md §1). The dynamic-vegetation / prognostic-LAI / carbon branches
(dveg ∈ {2,5,6} and the dveg ∈ {7,8,9} read-LAI overrides) are CUT.

What the dveg=4 branch does, in WRF source order:

1. Day-of-year shift by hemisphere (NH: ``DAY = JULIAN``; SH: shifted half a
   year), then linear interpolation of the monthly LAI/SAI tables (``LAIM``/
   ``SAIM``) about the nearest month centres (``module_sf_noahmplsm.F:1299-1316``).
2. SAI/LAI floor checks (``:1324-1325``).
3. Water / barren / ice / urban categories force LAI = SAI = 0 (``:1327-1331``).
4. Snow-burial reduction to exposed ELAI/ESAI, with the short-canopy
   (``0 < HVT <= 1``) exponential override and the ELAI/ESAI floor checks
   (``:1335-1348``).
5. FVEG for dveg=4 is set in the *caller* (``NOAHMP_SFLX``, ``:863-875``):
   ``FVEG = SHDMAX``; floor at 0.05; forced to 0 for urban/barren and when
   ELAI+ESAI == 0. We fold that block in here so the returned ``NoahMPPhenology``
   already carries the consumer-ready FVEG (the frozen output type owns ``fveg``).
6. Growing-season index IGS = 1 where ``TV > TMIN`` else 0 (``:1352-1356``);
   the croptype>0 / PGS branch is dead under the scope.

Inputs come only from the frozen ``land_state`` / ``forcing`` / ``static``
pytrees (ADR §3.2): ``land_state.snowh`` (SNOWH), ``land_state.tv`` (TV),
``forcing.julian`` / ``forcing.yearlen`` (clock), ``static.lat`` (LAT, **degrees**
in our state — WRF tests LAT in radians but only the sign drives the hemisphere
shift, and the sign is preserved), ``static.ivgtyp`` (VEGTYP), and the parameter
tables ``static.parameters`` (LAIM/SAIM/HVT/HVB/SHDFAC).

Parameter-table conventions (S0b ``tables.py`` provides ``NoahMPParameters``
already **gathered per land column** by ``ivgtyp``):
  * ``laim`` / ``saim`` : monthly tables, leading axis = 12 months, shape
    ``(12, ny, nx)`` (or ``(12,)`` broadcastable). 1-based WRF months map to
    0-based array rows here.
  * ``hvt`` / ``hvb`` / ``shdfac`` : per-column scalars, shape ``(ny, nx)``.

Two fields PHENOLOGY/dveg=4 reads are NOT in the frozen ``NoahMPParameters``
(``tables.py`` is owned by S0b and frozen against this sprint): the category
sentinels (ISWATER/ISBARREN/ISICE/urban) and the per-category photosynthesis
floor TMIN. The Canary domain runs the corpus ``MODIFIED_IGBP_MODIS_NOAH`` 20-
category land-use set, so those are pinned here as module constants taken
verbatim from ``run/MPTABLE.TBL`` (``&noahmp_modis_parameters``) and gathered by
``ivgtyp``. If a later sprint widens ``NoahMPParameters`` with ``tmin`` / the
category indices via the patch protocol, swap the gathers for the table fields.

fp64 throughout (matches ``State`` / land-state construction).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.types import NoahMPForcing, NoahMPPhenology

NMONTH: int = 12  # monthly LAI/SAI table length (module_sf_noahmplsm.F)

# --- MODIS (MODIFIED_IGBP_MODIS_NOAH, NVEG=20) category sentinels --------------
# Verbatim from /home/enric/src/wrf_pristine/WRF/run/MPTABLE.TBL
# &noahmp_modis_parameters (1-based VEGTYP indices).
ISURBAN_MODIS: int = 13
ISWATER_MODIS: int = 17
ISBARREN_MODIS: int = 16
ISICE_MODIS: int = 15

# Per-category minimum temperature for photosynthesis TMIN(20) [K], MODIS block
# (MPTABLE.TBL TMIN row). Index 0 is a 1-based pad so TMIN_MODIS[VEGTYP] works
# directly with the WRF 1-based category id. Zeros (cat 13/15/16/17 = urban/ice/
# barren/water, the no-veg classes) reproduce WRF's table values exactly.
_TMIN_MODIS_1BASED = (
    0.0,                       # 0  (pad — VEGTYP is 1-based)
    265.0, 273.0, 268.0, 273.0, 268.0, 273.0, 273.0, 273.0, 273.0, 273.0,  # 1..10
    268.0, 273.0, 0.0, 273.0, 0.0, 0.0, 0.0, 268.0, 268.0, 268.0,          # 11..20
)


def _gather_per_category(table_1based: tuple[float, ...], vegtyp: jax.Array) -> jax.Array:
    """Gather a 1-based per-category constant table to per-column (ny, nx)."""

    arr = jnp.asarray(table_1based, dtype=jnp.float64)
    idx = jnp.clip(vegtyp.astype(jnp.int32), 0, arr.shape[0] - 1)
    return arr[idx]


def noahmp_phenology_table(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
) -> NoahMPPhenology:
    """Table phenology for dveg=4 (croptype == 0). See module docstring.

    Returns ``NoahMPPhenology(lai, sai, elai, esai, fveg, igs)`` over all land
    columns. Pure functional / jit-friendly; no host transfer; no state mutation
    (the driver writes the returned LAI/SAI back into ``NoahMPLandState``).
    """

    p = static.parameters

    snowh = jnp.asarray(land_state.snowh, dtype=jnp.float64)   # SNOWH [m]
    tv = jnp.asarray(land_state.tv, dtype=jnp.float64)         # TV [K]
    lat = jnp.asarray(static.lat, dtype=jnp.float64)           # LAT (deg; sign test)
    vegtyp = jnp.asarray(static.ivgtyp, dtype=jnp.int32)       # VEGTYP (1-based)

    # Monthly LAI/SAI + canopy heights. The frozen S0b NoahMPParameters provides
    # PER-CATEGORY tables (laim/saim shape (nveg+1, 12); hvt/hvb shape (nveg+1,)),
    # so gather by VEGTYP here. A caller that passes PRE-GATHERED tables
    # (laim shape (12, ny, nx); hvt shape (ny, nx)) — e.g. the unit oracle — is also
    # supported: detected by the leading axis being the 12-month axis / a non-1D map.
    nveg_p1 = jnp.asarray(p.laim, dtype=jnp.float64).shape[0]

    def _gather_monthly(tbl):
        a = jnp.asarray(tbl, dtype=jnp.float64)
        if a.ndim == 2 and a.shape[0] == NMONTH and a.shape[1] != NMONTH:
            return a  # already (12, ...) pre-gathered (and not a (12,12) ambiguity)
        if a.ndim == 2 and a.shape[1] == NMONTH:
            # per-category (nveg+1, 12) -> (12, ny, nx) gathered by vegtyp.
            idx = jnp.clip(vegtyp, 0, a.shape[0] - 1)
            return jnp.moveaxis(a[idx], -1, 0)
        return a  # (12,) broadcastable

    def _gather_scalar(tbl):
        a = jnp.asarray(tbl, dtype=jnp.float64)
        if a.ndim == 1 and a.shape[0] == nveg_p1:
            idx = jnp.clip(vegtyp, 0, a.shape[0] - 1)
            return a[idx]                       # (ny, nx)
        return a                                # already per-column or scalar

    laim = _gather_monthly(p.laim)                            # (12, ...) monthly LAI
    saim = _gather_monthly(p.saim)                            # (12, ...) monthly SAI
    hvt = _gather_scalar(p.hvt)                               # canopy top [m]
    hvb = _gather_scalar(p.hvb)                               # canopy bottom [m]
    # FVEG source for dveg=4 = SHDMAX (annual-max green-veg fraction), a per-column
    # wrfinput field carried on NoahMPStatic (= VEGMAX/100), NOT an MPTABLE param.
    # Arbiter: pristine WRF module_sf_noahmplsm.F:864 (FVEG=SHDMAX) +
    # module_sf_noahmpdrv.F:753 (FVGMAX=VEGMAX/100). Fall back to instantaneous
    # SHDFAC then to TV's shape if a caller omits both (defensive; driver supplies it).
    shdmax_src = static.shdmax if static.shdmax is not None else static.shdfac
    if shdmax_src is None:
        shdmax = jnp.zeros_like(tv)
    else:
        shdmax = jnp.asarray(shdmax_src, dtype=jnp.float64)    # FVEG source (dveg=4)

    julian = jnp.asarray(forcing.julian, dtype=jnp.float64)    # day-of-year (fractional)
    yearlen = jnp.asarray(forcing.yearlen, dtype=jnp.float64)  # days in year

    # --- monthly table interpolation (module_sf_noahmplsm.F:1299-1316) ---------
    # Hemisphere day shift: SH shifted by half a year. WRF tests LAT (radians) but
    # only the sign matters; LAT here is degrees with the same sign.
    day = jnp.where(
        lat >= 0.0,
        julian,
        jnp.mod(julian + 0.5 * yearlen, yearlen),
    )

    t = 12.0 * day / yearlen                       # current month in [0, 12)
    # Fortran INT() truncates toward zero; T + 0.5 >= 0 here so floor == trunc.
    it1 = jnp.floor(t + 0.5).astype(jnp.int32)     # IT1
    it2 = it1 + 1                                  # IT2
    wt1 = (it1.astype(jnp.float64) + 0.5) - t      # WT1
    wt2 = 1.0 - wt1                                # WT2
    # Month wrap (IT1 < 1 -> 12 ; IT2 > 12 -> 1), then to 0-based rows.
    it1 = jnp.where(it1 < 1, 12, it1)
    it2 = jnp.where(it2 > 12, 1, it2)
    i1 = it1 - 1
    i2 = it2 - 1

    # Gather the two bracketing months along the leading (month) axis. Works for
    # laim shape (12,) or (12, ny, nx); take_along_axis broadcasts per column.
    def _pick(month_table: jax.Array, idx: jax.Array) -> jax.Array:
        if month_table.ndim == 1:
            return month_table[idx]
        # (ny, nx) indices -> add a length-1 month axis for take_along_axis.
        return jnp.take_along_axis(month_table, idx[None, ...], axis=0)[0]

    lai = wt1 * _pick(laim, i1) + wt2 * _pick(laim, i2)
    sai = wt1 * _pick(saim, i1) + wt2 * _pick(saim, i2)

    # --- SAI / LAI floor checks (:1324-1325) ----------------------------------
    sai = jnp.where(sai < 0.05, 0.0, sai)
    lai = jnp.where((lai < 0.05) | (sai == 0.0), 0.0, lai)

    # --- water / barren / ice / urban -> LAI = SAI = 0 (:1327-1331) ------------
    # urban_flag is the urban category under MODIS (driver pins ICE=0/IST=1 land;
    # no separate urban physics in scope — the category test is what WRF applies).
    no_veg = (
        (vegtyp == ISWATER_MODIS)
        | (vegtyp == ISBARREN_MODIS)
        | (vegtyp == ISICE_MODIS)
        | (vegtyp == ISURBAN_MODIS)
    )
    lai = jnp.where(no_veg, 0.0, lai)
    sai = jnp.where(no_veg, 0.0, sai)

    # --- snow burial -> exposed ELAI / ESAI (:1335-1348) ----------------------
    db = jnp.minimum(jnp.maximum(snowh - hvb, 0.0), hvt - hvb)
    fb = db / jnp.maximum(1.0e-06, hvt - hvb)

    # short-canopy override (0 < HVT <= 1): SNOWHC = HVT*exp(-SNOWH/0.2);
    # FB = min(SNOWH, SNOWHC) / SNOWHC.
    short_canopy = (hvt > 0.0) & (hvt <= 1.0)
    snowhc = hvt * jnp.exp(-snowh / 0.2)
    fb_short = jnp.minimum(snowh, snowhc) / snowhc
    fb = jnp.where(short_canopy, fb_short, fb)

    elai = lai * (1.0 - fb)
    esai = sai * (1.0 - fb)
    # croptype == 0 in scope so both ESAI/ELAI floor checks always apply.
    esai = jnp.where(esai < 0.05, 0.0, esai)
    elai = jnp.where((elai < 0.05) | (esai == 0.0), 0.0, elai)

    # --- FVEG for dveg=4 (NOAHMP_SFLX caller, :863-875) -----------------------
    # dveg=4: FVEG = SHDMAX; floor 0.05; urban/barren -> 0; ELAI+ESAI==0 -> 0.
    fveg = jnp.where(shdmax <= 0.05, 0.05, shdmax)
    fveg = jnp.where((vegtyp == ISURBAN_MODIS) | (vegtyp == ISBARREN_MODIS), 0.0, fveg)
    fveg = jnp.where((elai + esai) == 0.0, 0.0, fveg)

    # --- growing-season index IGS (:1352-1356) --------------------------------
    tmin = _gather_per_category(_TMIN_MODIS_1BASED, vegtyp)
    igs = jnp.where(tv > tmin, 1.0, 0.0)

    return NoahMPPhenology(
        lai=lai,
        sai=sai,
        elai=elai,
        esai=esai,
        fveg=fveg,
        igs=igs,
    )


__all__ = ["noahmp_phenology_table"]
