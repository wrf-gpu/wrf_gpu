"""Host-only WRF land history mapping; never advances land physics at output."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import jax
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.runtime.history_accumulators import LAND_FLUX_FIELDS, RADIATION_SOURCES, SURFACE_SOURCES, SNOW_ACCUMULATORS

# Registry names whose prognostic source is the Noah-MP sibling carry.
LAND_LEAVES = {
    "TSLB": "tslb", "SMOIS": "smois", "SH2O": "sh2o",
    "TSNO": "tsno", "SNICE": "snice", "SNLIQ": "snliq", "ZSNSO": "zsnso",
    "ISNOW": "isnow", "SNOW": "sneqv", "SNOWH": "snowh", "SNEQVO": "sneqvo",
    "TV": "tv", "TG": "tg", "TAH": "tah", "EAH": "eah",
    "CANLIQ": "canliq", "CANICE": "canice", "FWET": "fwet",
    "LAI": "lai", "XSAI": "sai", "CM": "cm", "CH": "ch",
    "ALBOLD": "albold", "TAUSS": "tauss", "ALBEDO": "albedo", "EMISS": "emiss",
}
STATIC_FIELDS = (
    # Noah-MP receives VEGFRA (%) as INTENT(IN), separately from its FVEG
    # phenology output (module_sf_noahmpdrv.F:102,752,1289). The shared writer
    # overlays the held auxinput4 value after these wrfinput fields when enabled.
    "ISLTYP", "IVGTYP", "VEGFRA", "SHDMAX", "SHDMIN", "SHDAVG", "SNOALB", "TMN",
    "VAR", "CON", "VAR_SSO", "OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4", "SMCWTD", "CHSTAR",
    "WA", "WT", "ZWT", "GRAIN",
)
CARBON_FIELDS = ("LFMASS", "RTMASS", "STMASS", "WOOD", "STBLCP", "FASTCP")
# These diagnostics are undefined after the glacier branch, including when
# standard SFLX emits values on the port's unimplemented glacier runtime tile.
# This is WRF history definedness; it does not change the carried model state.
# module_sf_noahmpdrv.F:1065-1128 assigns undefined_value (-1.E36, :629) and :1244-1325
# copies them to these Registry history fields (PLAI->LAI, PSAI->XSAI, T2MV->T2V); the same
# branch later resets CANICE/CANLIQ/QTLDRN to 0 (:1150-1152), ZWT stays commented out,
# RS/RB/LAISUN/LAISHA are not history. == CPU-WRF V4.7.1 Swiss SN02 history, all frames > t0.
GLACIER_UNDEFINED_FIELDS = (
    "APAR", "PSN", "BGAP", "WGAP", "GDD", "GRAIN", "GPP", "NPP", "NEE", "WA", "WT", "WSLAKE",
    "TV", "EAH", "TAH", "FWET", "LFMASS", "RTMASS", "STMASS", "WOOD", "STBLCP", "FASTCP",
    "LAI", "XSAI", "T2V", "RSSUN", "RSSHA", "TGV", "CHV", "CHLEAF", "CHUC", "CHV2",
)
LAND_HISTORY_FIELDS = frozenset((*LAND_LEAVES, *STATIC_FIELDS, *LAND_FLUX_FIELDS,
                                 *RADIATION_SOURCES, *SURFACE_SOURCES, *SNOW_ACCUMULATORS, *CARBON_FIELDS,
                                 *GLACIER_UNDEFINED_FIELDS,
                                 "ALBBCK", "CANWAT", "SFROFF", "UDROFF"))


def load_land_history_inputs(path: Path, *, table_dir: Path, parameters=None) -> dict[str, np.ndarray]:
    """Read statics once, with WRF LANDUSE initialization of ALBBCK/EMISS.

    ``real.exe`` inputs precede physics initialization: ALBBCK is overwritten
    from LANDUSE.TBL unless usemonalb is enabled (physics_init.F:1958-1971).
    The supported WN3 namelist uses the seasonal table, with no sea ice.
    """
    result = {}
    with Dataset(path) as ds:
        for name in (*STATIC_FIELDS, "ALBBCK", "ALBEDO", "EMISS", "LANDMASK", "LAI", "TSK", "SNOWC"):
            if name in ds.variables:
                var = ds[name]
                result[name] = np.asarray(var[0] if var.dimensions[0] == "Time" else var[:]).copy()
        if "SEAICE" in ds.variables and np.any(ds["SEAICE"][0] != 0):
            raise ValueError("land history initialization requires the supported ice-free tile")
        result["ALBBCK"], result["ALBEDO"], result["EMISS"] = _landuse_init(
            ds, table_dir, result["IVGTYP"], result.get("SNOWC", 0))
    index = result["IVGTYP"].astype(np.int32)
    # NOAHMP_INIT:2158-2200. CARBON runs only for DVEG=2/5/6; the supported
    # DVEG=4 keeps these initialized pools, so they are genuine host statics.
    if parameters is None:
        from gpuwrf.physics.noahmp.tables import load_noahmp_parameters
        parameters = load_noahmp_parameters(table_dir)
    no_veg = np.isin(index, (parameters.iswater, parameters.isbarren, parameters.isice, parameters.isurban))
    no_veg |= (index >= 51) & (index <= 61)
    f32 = np.float32
    lai = np.where(no_veg, f32(0), np.maximum(result["LAI"], f32(.05)))
    sai = np.where(no_veg, f32(0), np.maximum(f32(.1) * lai, f32(.05)))
    sla = np.asarray(jax.device_get(parameters.sla), dtype=np.float32)
    result["LFMASS"] = lai * (f32(1000) / np.maximum(sla[index], f32(1)))
    result["STMASS"] = sai * (f32(1000) / f32(3))
    for name, constant in (("RTMASS", 500), ("WOOD", 500), ("STBLCP", 1000), ("FASTCP", 1000)):
        result[name] = np.where(no_veg, f32(0), f32(constant))
    result["LAI_INIT"], result["XSAI_INIT"] = lai, sai
    result["CHSTAR"] = np.full_like(lai, f32(.1))  # NOAHMP_INIT:2138; no step driver update.
    result.setdefault("SMCWTD", np.zeros_like(lai))  # Inactive groundwater state for OPT_RUN=3.
    # NOAHMP_INIT:2144-2147,2175,2195. Supported OPT_RUN=3/DVEG=4 keeps
    # these initialized diagnostics, including over open water. ZWT remains
    # defined over glacier columns (the driver's undefined assignment is commented out).
    for name, constant in (("WA", 4900), ("WT", 4900), ("ZWT", 2.5), ("GRAIN", 1e-10)):
        result[name] = np.full_like(lai, f32(constant))
    result["_GLACIER_MASK"] = (index == parameters.isice) & (result["LANDMASK"] > .5)
    return result


def _landuse_init(ds, table_dir: Path, ivgtyp, snowc):
    """WRF landuse_init with usemonalb=.false. (physics_init.F:1958-1971): ALBBCK = ALBD(IS,ISN)/100,
    ALBEDO = ALBBCK, times (1 + SCFX) where SNOWC > 0.5, EMISS = SFEM. Returns (ALBBCK, ALBEDO, EMISS)."""
    mminlu = str(ds.getncattr("MMINLU"))
    cen_lat = float(ds.getncattr("CEN_LAT"))
    raw = np.asarray(ds["Times"][0]).astype("S1")
    start = datetime.strptime(b"".join(raw.tolist()).decode().strip("\x00 "), "%Y-%m-%d_%H:%M:%S")
    season = 2 if start.timetuple().tm_yday < 105 or start.timetuple().tm_yday > 288 else 1
    if cen_lat < 0:
        season = 3 - season
    lines = (table_dir / "LANDUSE.TBL").read_text().splitlines()
    begin = next(i for i, line in enumerate(lines) if line.strip() == mminlu)
    count = int(lines[begin + 1].split(",")[0])
    first = begin + 3 + (season - 1) * (count + 1)
    rows = [line.split(",") for line in lines[first:first + count]]
    albedo = np.array([0, *(float(row[1]) for row in rows)], dtype=np.float32) / np.float32(100)
    emiss = np.array([0, *(float(row[3]) for row in rows)], dtype=np.float32)
    scfx = np.array([0, *(float(row[6]) for row in rows)], dtype=np.float32)
    index = np.asarray(ivgtyp).astype(np.int32)
    if np.any(index < 1) or np.any(index > count):
        raise ValueError("land history has an invalid LANDUSE category")
    albbck = albedo[index]
    return albbck, np.where(np.asarray(snowc) > .5, albbck * (np.float32(1) + scfx[index]), albbck), emiss[index]


def wrf_initial_albedo(ds, table_dir: Path) -> np.ndarray:
    """ALBEDO entering WRF's first step (= its lead-zero history value) from an open wrfinput."""
    snowc = np.asarray(ds["SNOWC"][0]) if "SNOWC" in ds.variables else 0
    return _landuse_init(ds, table_dir, np.asarray(ds["IVGTYP"][0]), snowc)[1]


def land_history_diagnostics(land, initial, inputs, *, own_step: int, history=None, water_sst=None):
    """WRF land outputs on land; retain initialized fields on the water tile.

    Noah-MP is vectorized over the grid in the port, but WRF's driver writes only
    land columns. Its dummy water computations are never history values. The
    initial sibling carry supplies those unchanged water values; TSLB is the
    exception because surface_driver updates its ocean top layer from aux4 SST.
    ``water_sst`` is the held prescribed skin temperature already applied to
    the water State, never the dummy Noah-MP ocean-soil result.
    WRF's first Noah-MP call sets the lower ocean layers to 273.16 K
    (module_sf_noahmpdrv.F:697-705); lead-zero history precedes that call.
    """
    land, initial, history, water_sst = jax.device_get((land, initial, history, water_sst))
    mask = np.asarray(inputs["LANDMASK"]) > .5
    fields = {name: inputs[name] for name in STATIC_FIELDS if name in inputs}
    fields.update({name: inputs[name] for name in CARBON_FIELDS if name in inputs})
    fields["ALBBCK"] = inputs["ALBBCK"]
    for name, attr in LAND_LEAVES.items():
        value, seed = np.asarray(getattr(land, attr)), np.asarray(getattr(initial, attr))
        if name in {"ALBEDO", "EMISS"}:
            seed = inputs[name]
            if own_step == 0:
                value = seed
        elif name in {"CM", "CH"} and own_step == 0:
            # WRF NOAHMP_INIT:2143-2144 initializes these diagnostics to zero;
            # the model's positive first-call seed is not an output diagnostic.
            value = seed = np.zeros_like(value)
        elif name in {"LAI", "XSAI"} and name + "_INIT" in inputs:
            seed = inputs[name + "_INIT"]
            if own_step == 0:
                value = seed
        if name == "TSLB":
            seed = seed.copy()
            if own_step > 0:
                # The first Noah call resets all ocean layers. SST_UPDATE
                # overwrites the top layer on subsequent surface-driver calls;
                # with SST_UPDATE off it keeps the first-call value.
                seed[:] = np.asarray(273.16, dtype=seed.dtype)
                if own_step > 1 and water_sst is not None:
                    seed[0] = np.asarray(water_sst, dtype=seed.dtype)
            else:
                seed[0] = value[0]
        fields[name] = np.where(mask, value, seed)
    fields["CANWAT"] = fields["CANLIQ"] + fields["CANICE"]
    # The model accumulates runoff in metres; WRF history is millimetres.
    for name, attr in (("SFROFF", "sfcrunoff"), ("UDROFF", "udrunoff")):
        fields[name] = np.where(mask, np.asarray(getattr(land, attr)) * 1000, 0)
    if history is not None:
        fields.update(history)
        for name in ("T2V", "T2B"):
            fields[name] = np.where(mask, history[name], np.asarray(initial.tv))
        if own_step == 0:
            fields["T2V"] = fields["T2B"] = np.asarray(initial.tv)
            fields["SNOWC"] = inputs.get("SNOWC", np.zeros_like(mask, dtype=np.float32))
    if own_step > 0 and "_GLACIER_MASK" in inputs:
        # module_sf_noahmpdrv.F:1065-1120: only after the first solve;
        # preserve every computed non-glacier diagnostic and all t0 values.
        undefined = np.float32(-1e36)
        for name in GLACIER_UNDEFINED_FIELDS:
            value = fields.get(name)
            if value is None:
                if history is None and name in LAND_FLUX_FIELDS:
                    continue  # no in-step packet: another writer source owns this whole field
                value = np.zeros_like(mask, dtype=np.float32)
            fields[name] = np.where(inputs["_GLACIER_MASK"], undefined, value)
        if "Q2V" in fields:
            # :1283 Q2MVXY = Q2MV/(1.0 - Q2MV) with Q2MV undefined -> -1 in REAL(4).
            fields["Q2V"] = np.where(inputs["_GLACIER_MASK"], undefined / (np.float32(1) - undefined), fields["Q2V"])
    return fields, land
