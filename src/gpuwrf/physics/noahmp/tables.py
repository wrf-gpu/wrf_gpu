"""Noah-MP parameter table loader (Sprint 0b) — ACTIVE-OPTION SUBSET.

Loads the per-category Noah-MP lookup tables once per run from the WRF run/
parameter files (``MPTABLE.TBL`` / ``SOILPARM.TBL`` / ``GENPARM.TBL``) and bundles
them as a frozen, device-resident ``NoahMPParameters`` pytree indexed by
``ivgtyp`` (vegetation) and ``isltyp`` (soil), exactly mirroring the WRF
``read_mp_veg_parameters`` / ``read_mp_soil_parameters`` /
``read_mp_rad_parameters`` + ``TRANSFER_MP_PARAMETERS`` semantics
(``module_sf_noahmplsm.F:11389-11761``, ``module_sf_noahmpdrv.F:1434-1709``).

SCOPE (LAND-ONLY, ADR-NOAHMP-INTERFACES.md §1/§2.5). Only the parameters the
ACTIVE Canary option set reads are loaded:

    dveg=4, opt_crs=1 (Ball-Berry), opt_btr=1, opt_run=3 (Schaake96), opt_sfc=1,
    opt_frz=1, opt_inf=1, opt_rad=3, opt_alb=2 (CLASS), opt_snf=1 (Jordan91),
    opt_tbot=2, opt_stc=1.

CUT (NOT loaded — dead on the Canary domain): carbon / dynamic-vegetation
(LFMASS/STMASS/respiration), crop (``&noahmp_crop_parameters``), irrigation,
tile-drainage, groundwater (SIMGM/SIMTOP), glacier, and the VIC/XAJ runoff
parameters (``BVIC/AXAJ/BXAJ/XXAJ/BDVIC/BBVIC/GDVIC`` — used only by
``opt_run in {5,6,7,8}``). ``F1`` is also dropped (only the OLD Noah thermal
conductivity uses it; Noah-MP THERMOPROP does not).

LAYOUT. Every per-category table is loaded as a device-resident fp64 array of
length ``NVEG+1`` (vegetation) or ``NSOIL_CAT+1`` (soil) with a junk slot 0, so a
column gather is the WRF-faithful 1-based ``table[category]`` with no off-by-one
shuffling. Two-band (vis/nir) and 12-month tables carry that extra trailing axis.
No host transfer at runtime: the bundle is built once and gathered per column by
category index inside the jitted physics step.

DATASET. The Canary corpus uses ``MMINLU = MODIFIED_IGBP_MODIS_NOAH``
(``num_land_cat = 21``); WRF reads the ``&noahmp_modis_parameters`` block (NVEG=20).
SOILPARM uses the first (``STAS``, 19-category) block, matching the WRF read order.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

# WRF fixed dimensions (module_sf_noahmplsm.F).
MBAND: int = 2   # radiation bands: 1=vis, 2=nir
NMONTH: int = 12  # monthly LAI/SAI tables
MSC: int = 8     # soil-color classes for soil albedo (noahmp_rad_parameters)

# The frozen active-option set this loader serves (recorded into the restart,
# ADR §5). Mirrors module_sf_noahmpdrv.F default IOPT_* for the Canary run.
DEFAULT_SCOPE_OPTIONS: dict = {
    "dveg": 4,
    "opt_crs": 1,
    "opt_btr": 1,
    "opt_run": 3,
    "opt_sfc": 1,
    "opt_frz": 1,
    "opt_inf": 1,
    "opt_rad": 3,
    "opt_alb": 2,
    "opt_snf": 1,
    "opt_tbot": 2,
    "opt_stc": 1,
}

# WRF dataset identifier for the Canary domain (wrfinput MMINLU attribute).
DEFAULT_DATASET = "MODIFIED_IGBP_MODIS_NOAH"


class NoahMPParameters(NamedTuple):
    """FROZEN Noah-MP parameter bundle — active-option subset (ADR §2.5).

    Vegetation tables are indexed by ``ivgtyp`` (axis 0 length ``nveg + 1``);
    soil tables by ``isltyp`` (axis 0 length ``nsoil_cat + 1``); rad-soil-albedo
    tables by soil-color index (axis 0 length ``MSC + 1``); general parameters are
    fp64 scalars. Slot 0 of each per-category axis is a junk filler so a WRF-style
    1-based gather ``table[category]`` is exact.

    Carbon / crop / irrigation / tile-drain / groundwater / glacier / VIC params
    are CUT (dead under the active option set).
    """

    # --- vegetation, indexed by ivgtyp (shape (nveg+1,) unless noted) ----------
    rhol: jnp.ndarray       # (nveg+1, MBAND) leaf reflectance (vis, nir)
    rhos: jnp.ndarray       # (nveg+1, MBAND) stem reflectance
    taul: jnp.ndarray       # (nveg+1, MBAND) leaf transmittance
    taus: jnp.ndarray       # (nveg+1, MBAND) stem transmittance
    xl: jnp.ndarray         # leaf/stem orientation index
    z0mvt: jnp.ndarray      # momentum roughness by veg type [m]
    hvt: jnp.ndarray        # canopy top height [m]
    hvb: jnp.ndarray        # canopy bottom height [m]
    dleaf: jnp.ndarray      # characteristic leaf dimension [m]
    rc: jnp.ndarray         # tree crown radius [m]
    den: jnp.ndarray        # tree density [trunks/m2]
    cwpvt: jnp.ndarray      # empirical canopy wind parameter
    saim: jnp.ndarray       # (nveg+1, NMONTH) monthly stem area index
    laim: jnp.ndarray       # (nveg+1, NMONTH) monthly leaf area index
    sla: jnp.ndarray        # single-side leaf area per kg [m2/kg]
    ch2op: jnp.ndarray      # max intercepted h2o per unit lai+sai [mm]
    nroot: jnp.ndarray      # number of soil layers with roots (int as fp64)
    mfsno: jnp.ndarray      # snowmelt m-parameter (opt_alb snow-cover fraction)
    scffac: jnp.ndarray     # snow-cover factor [m]
    # CANRES / Ball-Berry stomatal-resistance parameters (opt_crs=1, opt_btr=1)
    rsmin: jnp.ndarray      # minimum stomatal resistance [s/m]  (= RS_TABLE)
    rsmax: jnp.ndarray      # maximum stomatal resistance [s/m]
    rgl: jnp.ndarray        # radiation-stress parameter
    hs: jnp.ndarray         # vapor-pressure-deficit parameter
    topt: jnp.ndarray       # optimum transpiration air temperature [K]
    bp: jnp.ndarray         # minimum leaf conductance [umol/m2/s]
    mp: jnp.ndarray         # slope of conductance-to-photosynthesis
    c3psn: jnp.ndarray      # photosynthetic pathway (1=c3, 0=c4)
    kc25: jnp.ndarray       # co2 Michaelis-Menten constant at 25C [Pa]
    akc: jnp.ndarray        # q10 for kc25
    ko25: jnp.ndarray       # o2 Michaelis-Menten constant at 25C [Pa]
    ako: jnp.ndarray        # q10 for ko25
    vcmx25: jnp.ndarray     # max carboxylation rate at 25C [umol/m2/s]
    avcmx: jnp.ndarray      # q10 for vcmx25
    qe25: jnp.ndarray       # quantum efficiency at 25C
    aqe: jnp.ndarray        # q10 for qe25
    folnmx: jnp.ndarray     # foliage N when f(n)=1 [%]

    # --- soil, indexed by isltyp (shape (nsoil_cat+1,)) ------------------------
    bexp: jnp.ndarray       # Clapp-Hornberger b
    smcmax: jnp.ndarray     # saturated (porosity) soil moisture [m3/m3]
    smcref: jnp.ndarray     # reference (field capacity) soil moisture [m3/m3]
    smcwlt: jnp.ndarray     # wilting-point soil moisture [m3/m3]
    smcdry: jnp.ndarray     # dry soil moisture [m3/m3]
    dksat: jnp.ndarray      # saturated hydraulic conductivity [m/s]
    dwsat: jnp.ndarray      # saturated soil diffusivity [m2/s]
    psisat: jnp.ndarray     # saturated matric potential MAGNITUDE [m] (>0; used as -psisat)
    quartz: jnp.ndarray     # soil quartz content

    # --- soil-albedo, indexed by soil-color (shape (MSC+1, MBAND)) -------------
    albsat: jnp.ndarray     # saturated soil albedo (vis, nir)
    albdry: jnp.ndarray     # dry soil albedo (vis, nir)

    # --- general (GENPARM + rad + global scalars) ------------------------------
    csoil: jnp.ndarray      # soil heat capacity [J/m3/K]
    zbot: jnp.ndarray       # deep-soil temperature depth [m] (<0)
    czil: jnp.ndarray       # Zilitinkevich thermal-roughness coefficient
    refdk: jnp.ndarray      # reference saturated conductivity (KDT base)
    refkdt: jnp.ndarray     # reference KDT factor
    frzk: jnp.ndarray       # frozen-soil parameter (base; adjusted to FRZX per soil)
    slope: jnp.ndarray      # (nslope+1,) subsurface-runoff slope factor (opt_run=3)
    eg: jnp.ndarray         # (2,) ground emissivity (1=soil, 2=lake)
    # two-stream snow params (opt_alb=2 CLASS uses ground/snow blend; rad block)
    omegas: jnp.ndarray     # (MBAND,) two-stream omega for snow
    betads: jnp.ndarray     # two-stream betad for snow
    betais: jnp.ndarray     # two-stream betaI for snow
    # snow water / aging globals (noahmp_global_parameters)
    swemx: jnp.ndarray      # new-snow mass to fully cover old snow [mm]
    z0sno: jnp.ndarray      # snow surface roughness [m]
    ssi: jnp.ndarray        # snowpack liquid water holding capacity [m3/m3]
    snow_ret_fac: jnp.ndarray  # snowpack water release timescale [1/s]
    snow_emis: jnp.ndarray  # snow emissivity

    # --- bookkeeping (frozen ints; recorded for restart-schema guard) ----------
    iswater: int
    isbarren: int
    isice: int
    iscrop: int
    isurban: int


# ---------------------------------------------------------------------------
# WRF text-table parsers (faithful to the Fortran read order).
# ---------------------------------------------------------------------------

def _parse_mptable_block(text: str, block: str) -> dict[str, list[float]]:
    """Parse a single ``&block ... /`` Fortran-namelist section of MPTABLE.TBL.

    Returns ``{KEY: [floats]}``. ``!``-comment lines and trailing inline comments
    are stripped; continuation across blank lines is handled implicitly because
    each parameter is one full line in the WRF tables. Integer scalars (the
    ``IS*`` indices) are returned as single-element lists.
    """
    # isolate the requested namelist block: from the &block header to the first
    # line whose (comment-stripped) content is a bare "/" terminator. A naive
    # search for "/" is wrong because MPTABLE comments contain slashes
    # (e.g. ! 'Open Shrublands' -> USGS 9 "shrubland/grassland").
    start = text.index("&" + block)
    lines = text[start:].splitlines()
    body: list[str] = []
    for raw in lines[1:]:  # drop the &block header line
        if raw.split("!", 1)[0].strip() == "/":
            break
        body.append(raw)

    out: dict[str, list[float]] = {}
    for raw in body:
        line = raw.split("!", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, rhs = line.split("=", 1)
        key = key.strip().upper()
        rhs = rhs.strip().rstrip(",")
        if not rhs:
            continue
        vals = [v for v in re.split(r"[,\s]+", rhs) if v]
        try:
            out[key] = [float(v) for v in vals]
        except ValueError:
            # non-numeric (e.g. a category-description string) -> skip
            continue
    return out


def _veg_col(block: dict[str, list[float]], key: str, nveg: int) -> np.ndarray:
    """1-based per-veg column with a junk slot 0; truncated/checked to ``nveg``."""
    raw = block[key]
    if len(raw) < nveg:
        raise ValueError(f"MPTABLE key {key}: {len(raw)} values < NVEG={nveg}")
    arr = np.zeros(nveg + 1, dtype=np.float64)
    arr[1:] = np.asarray(raw[:nveg], dtype=np.float64)
    return arr


def _veg_band(block: dict, vis_key: str, nir_key: str, nveg: int) -> np.ndarray:
    """(nveg+1, MBAND) band table from the VIS/NIR rows (RHOL_VIS/RHOL_NIR ...)."""
    out = np.zeros((nveg + 1, MBAND), dtype=np.float64)
    out[:, 0] = _veg_col(block, vis_key, nveg)
    out[:, 1] = _veg_col(block, nir_key, nveg)
    return out


def _veg_monthly(block: dict, prefix: str, nveg: int) -> np.ndarray:
    """(nveg+1, NMONTH) monthly table from the 12 ``<prefix>_JAN..DEC`` rows."""
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    out = np.zeros((nveg + 1, NMONTH), dtype=np.float64)
    for m, mon in enumerate(months):
        out[:, m] = _veg_col(block, f"{prefix}_{mon}", nveg)
    return out


def _parse_soilparm(path: Path) -> tuple[int, dict[str, np.ndarray]]:
    """Parse the first (STAS) block of SOILPARM.TBL per the WRF read order.

    WRF read order (module_sf_noahmplsm.F:11704-11709):
        ITMP, BEXP, SMCDRY, F1, SMCMAX, SMCREF, PSISAT, DKSAT, DWSAT,
        SMCWLT, QUARTZ, BVIC, AXAJ, BXAJ, XXAJ, BDVIC, BBVIC, GDVIC
    Returns (slcats, {field: (slcats+1,) array}) with the scoped subset.
    """
    lines = path.read_text().splitlines()
    # line 0: "Soil Parameters"; line 1: SLTYPE (e.g. STAS); line 2: "NCATS,1 'header'"
    slcats = int(re.split(r"[,\s]+", lines[2].strip())[0])
    fields = ["bexp", "smcdry", "f1", "smcmax", "smcref", "psisat",
              "dksat", "dwsat", "smcwlt", "quartz"]
    data = {f: np.zeros(slcats + 1, dtype=np.float64) for f in fields}
    row = 3
    read = 0
    while read < slcats and row < len(lines):
        line = lines[row].strip()
        row += 1
        if not line:
            continue
        toks = [t.strip().strip("'") for t in line.split(",")]
        idx = int(toks[0])
        # toks[1..10] map to bexp..quartz (toks[11:] are VIC/XAJ -> CUT)
        vals = [float(t) for t in toks[1:11]]
        for f, v in zip(fields, vals):
            data[f][idx] = v
        read += 1
    if read != slcats:
        raise ValueError(f"SOILPARM STAS: read {read} of {slcats} categories")
    return slcats, data


def _parse_genparm(path: Path) -> dict:
    """Parse GENPARM.TBL (the fixed line-by-line WRF read order, :11728-11757).

    Returns SLOPE (1-based array with junk slot 0), CSOIL, REFDK, REFKDT, FRZK,
    ZBOT, CZIL scalars. SBETA/FXEXP/SALP/SMLOW/SMHIGH/LVCOEF are present in the
    file but NOT read by Noah-MP (old Noah only) -> not returned.
    """
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    # locate SLOPE_DATA block
    i = lines.index("SLOPE_DATA")
    nslope = int(lines[i + 1])
    slope = np.zeros(nslope + 1, dtype=np.float64)
    for k in range(nslope):
        slope[k + 1] = float(lines[i + 2 + k])

    def scalar_after(tag: str) -> float:
        j = lines.index(tag)
        return float(lines[j + 1])

    return {
        "slope": slope,
        "csoil": scalar_after("CSOIL_DATA"),
        "refdk": scalar_after("REFDK_DATA"),
        "refkdt": scalar_after("REFKDT_DATA"),
        "frzk": scalar_after("FRZK_DATA"),
        "zbot": scalar_after("ZBOT_DATA"),
        "czil": scalar_after("CZIL_DATA"),
    }


def _color_band(block: dict, vis_key: str, nir_key: str) -> np.ndarray:
    """(MSC+1, MBAND) soil-albedo table from the VIS/NIR rows (1-based color)."""
    out = np.zeros((MSC + 1, MBAND), dtype=np.float64)
    vis = np.asarray(block[vis_key][:MSC], dtype=np.float64)
    nir = np.asarray(block[nir_key][:MSC], dtype=np.float64)
    out[1:, 0] = vis
    out[1:, 1] = nir
    return out


# ---------------------------------------------------------------------------
# Public loader.
# ---------------------------------------------------------------------------

def load_noahmp_parameters(
    table_dir,
    *,
    scope_options: dict | None = None,
    dataset: str = DEFAULT_DATASET,
) -> NoahMPParameters:
    """Load and bundle the active-option Noah-MP parameter tables.

    Parameters
    ----------
    table_dir
        Directory holding ``MPTABLE.TBL`` / ``SOILPARM.TBL`` / ``GENPARM.TBL``
        (e.g. ``/home/enric/src/wrf_pristine/WRF/run``).
    scope_options
        Frozen iopt map recorded into the restart (ADR §5). Defaults to the
        Canary active set; a non-default map raises if it selects a CUT option
        that needs parameters this loader does not provide.
    dataset
        WRF land-use dataset identifier (wrfinput ``MMINLU``); selects the
        MPTABLE namelist block. Default ``MODIFIED_IGBP_MODIS_NOAH``.
    """
    table_dir = Path(table_dir)
    scope = dict(DEFAULT_SCOPE_OPTIONS if scope_options is None else scope_options)
    _check_scope(scope)

    # --- vegetation block (MPTABLE) -----------------------------------------
    mp_text = (table_dir / "MPTABLE.TBL").read_text()
    if dataset == "MODIFIED_IGBP_MODIS_NOAH":
        cats_block = _parse_mptable_block(mp_text, "noahmp_modis_veg_categories")
        veg = _parse_mptable_block(mp_text, "noahmp_modis_parameters")
    elif dataset == "USGS":
        cats_block = _parse_mptable_block(mp_text, "noahmp_usgs_veg_categories")
        veg = _parse_mptable_block(mp_text, "noahmp_usgs_parameters")
    else:
        raise ValueError(f"unsupported land-use dataset {dataset!r}")
    nveg = int(cats_block["NVEG"][0])

    rad = _parse_mptable_block(mp_text, "noahmp_rad_parameters")
    glob = _parse_mptable_block(mp_text, "noahmp_global_parameters")

    # --- soil + general blocks ----------------------------------------------
    slcats, soil = _parse_soilparm(table_dir / "SOILPARM.TBL")
    gen = _parse_genparm(table_dir / "GENPARM.TBL")

    f = lambda key: jnp.asarray(_veg_col(veg, key, nveg))  # noqa: E731

    bundle = NoahMPParameters(
        # vegetation
        rhol=jnp.asarray(_veg_band(veg, "RHOL_VIS", "RHOL_NIR", nveg)),
        rhos=jnp.asarray(_veg_band(veg, "RHOS_VIS", "RHOS_NIR", nveg)),
        taul=jnp.asarray(_veg_band(veg, "TAUL_VIS", "TAUL_NIR", nveg)),
        taus=jnp.asarray(_veg_band(veg, "TAUS_VIS", "TAUS_NIR", nveg)),
        xl=f("XL"), z0mvt=f("Z0MVT"), hvt=f("HVT"), hvb=f("HVB"),
        dleaf=f("DLEAF"), rc=f("RC"), den=f("DEN"), cwpvt=f("CWPVT"),
        saim=jnp.asarray(_veg_monthly(veg, "SAI", nveg)),
        laim=jnp.asarray(_veg_monthly(veg, "LAI", nveg)),
        sla=f("SLA"), ch2op=f("CH2OP"), nroot=f("NROOT"),
        mfsno=f("MFSNO"), scffac=f("SCFFAC"),
        # CANRES / Ball-Berry: RSMIN comes from the RS row (drv:1525 RSMIN=RS_TABLE)
        rsmin=f("RS"), rsmax=f("RSMAX"), rgl=f("RGL"), hs=f("HS"), topt=f("TOPT"),
        bp=f("BP"), mp=f("MP"), c3psn=f("C3PSN"), kc25=f("KC25"), akc=f("AKC"),
        ko25=f("KO25"), ako=f("AKO"), vcmx25=f("VCMX25"), avcmx=f("AVCMX"),
        qe25=f("QE25"), aqe=f("AQE"), folnmx=f("FOLNMX"),
        # soil
        bexp=jnp.asarray(soil["bexp"]), smcmax=jnp.asarray(soil["smcmax"]),
        smcref=jnp.asarray(soil["smcref"]), smcwlt=jnp.asarray(soil["smcwlt"]),
        smcdry=jnp.asarray(soil["smcdry"]), dksat=jnp.asarray(soil["dksat"]),
        dwsat=jnp.asarray(soil["dwsat"]), psisat=jnp.asarray(soil["psisat"]),
        quartz=jnp.asarray(soil["quartz"]),
        # soil albedo by color
        albsat=jnp.asarray(_color_band(rad, "ALBSAT_VIS", "ALBSAT_NIR")),
        albdry=jnp.asarray(_color_band(rad, "ALBDRY_VIS", "ALBDRY_NIR")),
        # general
        csoil=jnp.asarray(gen["csoil"]), zbot=jnp.asarray(gen["zbot"]),
        czil=jnp.asarray(gen["czil"]), refdk=jnp.asarray(gen["refdk"]),
        refkdt=jnp.asarray(gen["refkdt"]), frzk=jnp.asarray(gen["frzk"]),
        slope=jnp.asarray(gen["slope"]),
        eg=jnp.asarray(np.asarray(rad["EG"][:2], dtype=np.float64)),
        omegas=jnp.asarray(np.asarray(rad["OMEGAS"][:MBAND], dtype=np.float64)),
        betads=jnp.asarray(rad["BETADS"][0]), betais=jnp.asarray(rad["BETAIS"][0]),
        swemx=jnp.asarray(glob["SWEMX"][0]), z0sno=jnp.asarray(glob["Z0SNO"][0]),
        ssi=jnp.asarray(glob["SSI"][0]),
        snow_ret_fac=jnp.asarray(glob["SNOW_RET_FAC"][0]),
        snow_emis=jnp.asarray(glob["SNOW_EMIS"][0]),
        # bookkeeping (faithful to the parsed MODIS/USGS indices)
        iswater=int(veg["ISWATER"][0]), isbarren=int(veg["ISBARREN"][0]),
        isice=int(veg["ISICE"][0]), iscrop=int(veg["ISCROP"][0]),
        isurban=int(veg["ISURBAN"][0]),
    )
    _validate(bundle, nveg, slcats)
    return bundle


def _check_scope(scope: dict) -> None:
    """Fail closed if the requested options need a CUT parameter family."""
    run = scope.get("opt_run")
    if run in (1, 2):
        raise ValueError(
            f"opt_run={run} (groundwater SIMGM/SIMTOP) is CUT; loader supports "
            "opt_run=3 (Schaake96) only.")
    if run in (5, 6, 7, 8):
        raise ValueError(
            f"opt_run={run} (VIC/XinAnJiang) needs BVIC/AXAJ/... params that are "
            "CUT; loader supports opt_run=3 (Schaake96) only.")
    if scope.get("dveg") not in (None, 4):
        raise ValueError(
            f"dveg={scope.get('dveg')}: only dveg=4 (table LAI/SAI, no dynamic "
            "vegetation/carbon) is supported by this scoped loader.")
    if scope.get("opt_crs") not in (None, 1):
        raise ValueError("only opt_crs=1 (Ball-Berry) is supported.")


def _validate(b: NoahMPParameters, nveg: int, slcats: int) -> None:
    """Reject WRF's sentinel ``-1.0e36`` fill landing in any LIVE category.

    A faithful read leaves only the trailing junk slot 0 unset; every real
    category (1..nveg / 1..slcats) must carry a finite, non-sentinel value.
    """
    def live_veg(arr):
        a = np.asarray(arr)
        return a[1:nveg + 1]

    for name in ("rhol", "rsmin", "z0mvt", "laim", "hvt"):
        a = live_veg(getattr(b, name))
        if not np.all(np.isfinite(a)) or np.any(a <= -1.0e30):
            raise ValueError(f"veg param {name!r} has unset/sentinel live entries")
    for name in ("bexp", "smcmax", "dksat", "psisat"):
        a = np.asarray(getattr(b, name))[1:slcats + 1]
        if not np.all(np.isfinite(a)) or np.any(a <= -1.0e30):
            raise ValueError(f"soil param {name!r} has unset/sentinel live entries")
    # porosity must dominate the moisture thresholds for every live soil class
    smcmax = np.asarray(b.smcmax)[1:slcats + 1]
    smcref = np.asarray(b.smcref)[1:slcats + 1]
    if np.any(smcref > smcmax + 1e-9):
        raise ValueError("soil SMCREF exceeds SMCMAX in a live category")


__all__ = [
    "NoahMPParameters",
    "load_noahmp_parameters",
    "DEFAULT_SCOPE_OPTIONS",
    "DEFAULT_DATASET",
    "MBAND",
    "NMONTH",
    "MSC",
]
