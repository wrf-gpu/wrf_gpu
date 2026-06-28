"""Minimal WRF-compatible NetCDF wrfout writer for M7 output handoff."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from datetime import date, datetime
from math import cos, pi
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import numpy as np
from netCDF4 import Dataset

from gpuwrf.physics.surface_constants import CP_D, KARMAN, P0_PA, R_D_OVER_CP, XLV
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

P0_THETA_OFFSET_K = 300.0
# WRF rvovrd = R_v/R_d (share/module_model_constants.F:41). Operational
# State.theta is MOIST theta_m (use_theta_m=1); WRF wrfout variable ``T`` is
# the DRY perturbation theta, so the writer decouples
# theta_dry = theta_m / (1 + rvovrd*qv) and emits theta_m itself as ``THM``.
RVOVRD = 461.6 / 287.0
CP_AIR_J_KG_K = CP_D
LV_J_KG = XLV
DATE_STR_LEN = 19
DEFAULT_SOIL_LAYERS = 4
DEFAULT_SNOW_LAYERS = 3
DEFAULT_SNSO_LAYERS = 7
DEFAULT_SEED_DIM = 8
_WRFOUT_HOST_ARRAYS: contextvars.ContextVar["_WrfoutHostArrays | None"] = (
    contextvars.ContextVar("_WRFOUT_HOST_ARRAYS", default=None)
)


DOWNSTREAM_CRITICAL_VARIABLES: tuple[str, ...] = (
    "Times",
    "XTIME",
    "XLAT",
    "XLONG",
    "HGT",
    "LANDMASK",
    "LU_INDEX",
    "U10",
    "V10",
    "T2",
    "Q2",
    "PSFC",
    "RAINC",
    "RAINNC",
    "RAINSH",
    "SWDOWN",
    "GLW",
    "PBLH",
    "UST",
    "HFX",
    "LH",
    "TSK",
    "CLDFRA",
    "QCLOUD",
    "QICE",
    "QRAIN",
)

MINIMUM_WRFOUT_VARIABLES: tuple[str, ...] = (
    *DOWNSTREAM_CRITICAL_VARIABLES,
    "XLAT_U",
    "XLONG_U",
    "XLAT_V",
    "XLONG_V",
    "U",
    "V",
    "W",
    "T",
    "THM",
    "QVAPOR",
    "P",
    "PB",
    "PH",
    "PHB",
    "MU",
    "MUB",
)

# --- P0-5a operational completeness additions (closing the gap to WRF's field
# list for the Canary operational product). Every name below is a real WRF-ARW
# wrfout variable (verified against
# <DATA_ROOT>/canairy_meteo/runs/wrf_l3/.../wrfout_d02_*): names, units, dims,
# staggering, and descriptions are copied from the reference file -- no invented
# fields. They are populated ONLY from real model state / grid metrics /
# operational diagnostics; a field with no available source is left out (it does
# NOT silently emit zeros for a quantity we do not actually compute). See
# proofs/p0_5/FINDINGS.md for the field-by-field source/justification table.

# (a) Always-available: extra hydrometeors + number concentrations (Thompson
#     prognostic state) + the MYNN TKE. Source = prognostic State leaves.
MICROPHYSICS_EXTRA_VARIABLES: tuple[str, ...] = (
    "QSNOW",
    "QGRAUP",
    "QNICE",
    "QNRAIN",
    "QNSNOW",
    "QNGRAUPEL",
    "QNCLOUD",
    "QNCCN",
    # v0.17 ADR-032 graupel/hail substrate (State qh/Nh/qvolg/qvolh); emitted
    # only when the source leaves carry hail (optional-extra; a non-hail run
    # never gets a fabricated hail field).
    "QHAIL",
    "QNHAIL",
    "QVGRAUPEL",
    "QVHAIL",
    # v0.16 aerosol-aware Thompson (mp=28) prognostic aerosol numbers
    # (State nwfa/nifa); emitted only when the source leaves are present.
    "QNWFA",
    "QNIFA",
    "QKE",
)

# (b) Grid-static coordinate / projection / map-factor / Coriolis fields. Source
#     = GridSpec.metrics + GridSpec.vertical (device-resident static arrays). WRF
#     repeats these every history frame; they let downstream tools reconstruct
#     the projection, eta coordinate, and Coriolis terms without a separate
#     geogrid/wrfinput read.
GRID_COORDINATE_VARIABLES: tuple[str, ...] = (
    "ZNU",
    "ZNW",
    "ZS",
    "DZS",
    "MAPFAC_M",
    "MAPFAC_U",
    "MAPFAC_V",
    "F",
    "E",
    "SINALPHA",
    "COSALPHA",
    "XLAND",
    "P_TOP",
)

# (b2) Extended grid-static vertical-coordinate + map-factor metric arrays.
#     v0.12.0 A1: PURE PAYLOAD of arrays ALREADY resident on ``DycoreMetrics`` /
#     the eta column -- no recompute, no physics. Vertical eta-spacing metrics
#     (DN/DNW/RDN/RDNW/FNM/FNP), the hybrid-eta C-coefficients (C1H..C4H,
#     C1F..C4F), the scalar extrapolation constants (CF1/CF2/CF3 already +
#     CFN/CFN1), the directional map factors (MAPFAC_MX/MY/UX/UY/VX/VY ==
#     msftx/msfty/msfux/msfuy/msfvx/msfvy), and the inverse grid lengths
#     (RDX/RDY) from the projection. Each is emitted ONLY when its source array
#     is present on the grid (dict/synthetic grids carry no metrics -> none are
#     written), so no field is ever fabricated. Names/dims/units/descriptions are
#     copied from the reference wrfout_d02 (375-var operational WRF file).
GRID_METRIC_EXTRA_VARIABLES: tuple[str, ...] = (
    "DN",
    "DNW",
    "RDN",
    "RDNW",
    "FNM",
    "FNP",
    "CFN",
    "CFN1",
    "CF1",
    "CF2",
    "CF3",
    "C1H",
    "C2H",
    "C3H",
    "C4H",
    "C1F",
    "C2F",
    "C3F",
    "C4F",
    "MAPFAC_MX",
    "MAPFAC_MY",
    "MAPFAC_UX",
    "MAPFAC_UY",
    "MAPFAC_VX",
    "MAPFAC_VY",
    "RDX",
    "RDY",
)

# (c) Accumulated precipitation partition (grid-scale snow/ice + graupel). Source
#     = State precip accumulators (coupling.physics_couplers writes snow_acc /
#     graupel_acc / ice_acc each microphysics step). WRF SNOWNC carries snow+ice.
PRECIP_PARTITION_VARIABLES: tuple[str, ...] = (
    "SNOWNC",
    "GRAUPELNC",
)

# (h) Trivially-derived diagnostics (v0.12.0 A2). WRF-FAITHFUL closed forms from
#     fields the writer already has -- NO physics, NO GPU. Each self-gates on the
#     availability of its inputs:
#       CLAT   = XLAT (computational-grid latitude; identical to XLAT for the
#                operational lat/lon grid).
#       COSZEN = WRF radconst/calc_coszen cosine solar zenith from XLAT/XLONG +
#                the forecast clock (run_start + lead) -- the exact transcription
#                already in coupling.physics_couplers._compute_coszen.
#       SR     = frozen-precipitation fraction = solid_acc/(solid_acc + rain_acc),
#                from the State precip accumulators (snow+graupel+ice vs rain).
#       SNOWC  = snow-cover flag (1 where SWE > 0) from the land carry's bulk SWE.
DERIVED_DIAGNOSTIC_VARIABLES: tuple[str, ...] = (
    "CLAT",
    "COSZEN",
    "SR",
    "SNOWC",
)

# (d) Surface energy/moisture-budget fluxes routed from the operational
#     diagnostics map (QFX/GRDFLX) plus the 2-m potential temperature TH2.
#     QFX/GRDFLX appear only when the caller supplies them via ``diagnostics``
#     (see the operational_mode hook spec in proofs/p0_5/FINDINGS.md); TH2 is
#     always derivable from T2 + PSFC. No invented surface flux.
SURFACE_FLUX_EXTRA_VARIABLES: tuple[str, ...] = (
    "QFX",
    "GRDFLX",
    "TH2",
)

# (e) Prognostic Noah-MP soil/snow land columns + land diagnostics. Source = the
#     optional ``land_state`` (NoahMPLandState) handed to the writer; WRF-faithful
#     4-layer soil + bulk snow. Absent (not written) when no land carry is
#     supplied -- the writer never fabricates a soil profile.
LAND_SOIL_VARIABLES: tuple[str, ...] = (
    "TSLB",
    "SMOIS",
    "SH2O",
    "SNOW",
    "SNOWH",
    "CANWAT",
    "SFROFF",
    "UDROFF",
    "ALBEDO",
    "EMISS",
)

# (f) Noah-MP internal snow-layer + canopy diagnostics. Source = the optional
# ``NoahMPLandState`` handed to the writer; absent when no land carry exists.
# The four 3-D snow-column fields (TSNO/SNICE/SNLIQ/ZSNSO) plus the scalar
# snow/canopy prognostics ISNOW/SNEQVO/CANLIQ/CANICE are all genuine device-
# resident slots of the carry (contracts/noahmp_state.py) — each self-gates on a
# present source so a partial carry never fabricates a profile.
LAND_SNOW_DIAGNOSTIC_VARIABLES: tuple[str, ...] = (
    "TSNO",
    "SNICE",
    "SNLIQ",
    "ZSNSO",
    "ISNOW",
    "SNEQVO",
    "CANLIQ",
    "CANICE",
)

# (g) Restart seed arrays for WRF stochastic perturbation options. The Canary
# supported namelist has stochastic physics disabled, so these are emitted only
# when an explicit caller-provided source is present (for example diagnostics).
STOCHASTIC_SEED_VARIABLES: tuple[str, ...] = (
    "ISEEDARR_SPPT",
    "ISEEDARR_SKEBS",
    "ISEEDARRAY_SPP_CONV",
    "ISEEDARRAY_SPP_PBL",
    "ISEEDARRAY_SPP_LSM",
)

# (h) B1 (v0.12.0) RRTMG up/down all-sky radiation flux diagnostics. Source =
# the operational radiation diagnostics map (``M9Diagnostics`` -> wrfout via
# ``diagnostics``): the SW/LW surface (bottom-of-atmosphere) and TOA up/down
# flux slices the RRTMG column solvers already compute, plus the slope-normal
# surface SW flux (SWNORM) and the derived OLR (== LWUPT). All ADD-only (no
# physical fallback) so they appear only when the radiation diagnostics supply
# them, never fabricated. The WRF clear-sky ``...C`` flux vars are deliberately
# absent: this port runs no separate clear-sky radiative-transfer pass, so a
# clear-sky flux value cannot be produced without fabrication.
RADIATION_FLUX_DIAGNOSTIC_VARIABLES: tuple[str, ...] = (
    "SWDNB",
    "SWUPB",
    "LWDNB",
    "LWUPB",
    "SWDNT",
    "SWUPT",
    "LWDNT",
    "LWUPT",
    "OLR",
    "SWNORM",
)

# The full operational field set the writer KNOWS how to emit. Each frame writes
# the subset for which a real source is present (grid metrics + state are always
# present; diagnostics / land carry are optional), so a missing optional source
# cannot manufacture a field. ``write_prepared_wrfout`` iterates the prepared
# payload keys, not this tuple, so optional fields self-gate.
OPERATIONAL_WRFOUT_VARIABLES: tuple[str, ...] = (
    *MINIMUM_WRFOUT_VARIABLES,
    *MICROPHYSICS_EXTRA_VARIABLES,
    *GRID_COORDINATE_VARIABLES,
    *GRID_METRIC_EXTRA_VARIABLES,
    *PRECIP_PARTITION_VARIABLES,
    *SURFACE_FLUX_EXTRA_VARIABLES,
    *DERIVED_DIAGNOSTIC_VARIABLES,
    *LAND_SOIL_VARIABLES,
    *LAND_SNOW_DIAGNOSTIC_VARIABLES,
    *STOCHASTIC_SEED_VARIABLES,
    *RADIATION_FLUX_DIAGNOSTIC_VARIABLES,
)


# Coordinate / dimension variables that make a history frame self-describing
# (geometry + eta levels + staggered lat/lon). ``Times`` and ``XTIME`` are always
# written by ``_write_times`` / ``_write_xtime`` regardless; the rest go through
# the normal variable loop, so ``write_prepared_wrfout`` force-includes THESE in
# any ``variable_subset`` (a name still absent from the prepared payload is simply
# skipped -- the file never fabricates a coordinate). This guarantees a subset
# stream (e.g. the training stream) is a self-contained, stream-valid WRF file.
MANDATORY_WRFOUT_COORDINATES: frozenset[str] = frozenset(
    {"Times", "XTIME", "ZNU", "ZNW", "XLAT_U", "XLAT_V", "XLONG_U", "XLONG_V"}
)


# Lossless NetCDF4 (zlib) compression settings for SUBSET streams only (see
# write_prepared_wrfout). complevel 4 is a strong size/CPU tradeoff for float
# fields; the writer thread runs off the GPU step thread so the CPU cost is hidden.
_SUBSET_STREAM_COMPRESSION: dict[str, object] = {"zlib": True, "complevel": 4}


# A compact, training-ready variable subset for the B200 3km->1km nest dataset
# (issue #122). These 36 names are the model fields a downstream learner needs;
# the writer ALWAYS additionally emits MANDATORY_WRFOUT_COORDINATES so each frame
# is self-describing. Every name is a real entry of WRFOUT_VARIABLE_SPECS; a name
# whose source is absent from a given run is silently skipped (never fabricated).
# OPT-IN ONLY -- selected via the GPUWRF_TRAINING_OUTPUT_SUBSET env flag on the
# nest output path; the default full output is unchanged and byte-identical.
MINIMAL_TRAINING_SET: tuple[str, ...] = (
    # 3D prognostic / diagnostic (16)
    "U", "V", "W", "T", "P", "PB", "PH", "PHB",
    "QVAPOR", "QCLOUD", "QICE", "QRAIN", "QSNOW", "QGRAUP", "CLDFRA", "QKE",
    # 2D surface (12)
    "T2", "Q2", "U10", "V10", "PSFC", "RAINNC",
    "SWDOWN", "GLW", "HFX", "LH", "PBLH", "TSK",
    # 2D cloud-validation (v0.20.2, output-only): OLR (TOA outgoing LW == cloud-top,
    # MSG-satellite-observable), RAINC (convective precip; completes RAINNC for the
    # 3 km cumulus parent), SWDNB (RRTMG instantaneous surface downwelling SW) (3)
    "OLR", "RAINC", "SWDNB",
    # static geography (5)
    "HGT", "XLAT", "XLONG", "LANDMASK", "LU_INDEX",
    # wind-rotation / map-factor (3)
    "SINALPHA", "COSALPHA", "MAPFAC_M",
)


# Opt-in WRF primary-history field set for the Canary WRFv4 configuration used as
# the v0.22 compatibility target. Order and names match the reference WRF
# ``wrfout_d02`` stream (375 variables including ``Times``), cross-checked against
# <DATA_ROOT>/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON and the EM Registry
# include tree that contributes Noah-MP, stochastic, hybrid-coordinate, and mask
# history fields.
# The default writer still uses OPERATIONAL_WRFOUT_VARIABLES; this heavy list is
# activated only by ``full_variable_set=True`` on the writer/prepared payload.
FULL_WRFOUT_VARIABLES: tuple[str, ...] = (
    "Times", "XLAT", "XLONG", "LU_INDEX", "ZNU", "ZNW", "ZS", "DZS",
    "VAR_SSO", "BATHYMETRY_FLAG", "U", "V", "W", "PH", "PHB", "T",
    "THM", "HFX_FORCE", "LH_FORCE", "TSK_FORCE", "HFX_FORCE_TEND", "LH_FORCE_TEND", "TSK_FORCE_TEND", "MU",
    "MUB", "NEST_POS", "P", "PB", "FNM", "FNP", "RDNW", "RDN",
    "DNW", "DN", "CFN", "CFN1", "THIS_IS_AN_IDEAL_RUN", "P_HYD", "Q2", "T2",
    "TH2", "PSFC", "U10", "V10", "RDX", "RDY", "AREA2D", "DX2D",
    "RESM", "ZETATOP", "CF1", "CF2", "CF3", "ITIMESTEP", "XTIME", "QVAPOR",
    "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE", "QNRAIN", "SHDMAX",
    "SHDMIN", "SHDAVG", "SNOALB", "TSLB", "SMOIS", "SH2O", "SEAICE", "XICEM",
    "SFROFF", "UDROFF", "IVGTYP", "ISLTYP", "VEGFRA", "GRDFLX", "ACGRDFLX", "ACSNOM",
    "SNOW", "SNOWH", "CANWAT", "SSTSK", "WATER_DEPTH", "COSZEN", "LAI", "QKE",
    "MAXMF", "MAXWIDTH", "ZTOP_PLUME", "DTAUX3D", "DTAUY3D", "DUSFCG", "DVSFCG", "VAR",
    "CON", "OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3",
    "OL4", "TKE_PBL", "EL_PBL", "O3_GFS_DU", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "MAPFAC_MX",
    "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MF_VX_INV", "MAPFAC_VY", "F", "E",
    "SINALPHA", "COSALPHA", "HGT", "TSK", "P_TOP", "GOT_VAR_SSO", "T00", "P00",
    "TLP", "TISO", "TLP_STRAT", "P_STRAT", "MAX_MSFTX", "MAX_MSFTY", "RAINC", "RAINSH",
    "RAINNC", "SNOWNC", "GRAUPELNC", "HAILNC", "CLDFRA", "SWDOWN", "GLW", "SWNORM",
    "ACSWUPT", "ACSWUPTC", "ACSWDNT", "ACSWDNTC", "ACSWUPB", "ACSWUPBC", "ACSWDNB", "ACSWDNBC",
    "ACLWUPT", "ACLWUPTC", "ACLWDNT", "ACLWDNTC", "ACLWUPB", "ACLWUPBC", "ACLWDNB", "ACLWDNBC",
    "SWUPT", "SWUPTC", "SWDNT", "SWDNTC", "SWUPB", "SWUPBC", "SWDNB", "SWDNBC",
    "LWUPT", "LWUPTC", "LWDNT", "LWDNTC", "LWUPB", "LWUPBC", "LWDNB", "LWDNBC",
    "OLR", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V", "ALBEDO", "CLAT", "ALBBCK",
    "EMISS", "NOAHRES", "TMN", "XLAND", "UST", "PBLH", "HFX", "QFX",
    "LH", "ACHFX", "ACLHF", "SNOWC", "SR", "SAVE_TOPO_FROM_REAL", "ISEEDARR_SPPT", "ISEEDARR_SKEBS",
    "ISEEDARR_RAND_PERTURB", "ISEEDARRAY_SPP_CONV", "ISEEDARRAY_SPP_PBL", "ISEEDARRAY_SPP_LSM", "ISNOW", "TV", "TG", "CANICE",
    "CANLIQ", "EAH", "TAH", "CM", "CH", "FWET", "SNEQVO", "ALBOLD",
    "QSNOWXY", "QRAINXY", "WSLAKE", "ZWT", "WA", "WT", "TSNO", "ZSNSO",
    "SNICE", "SNLIQ", "LFMASS", "RTMASS", "STMASS", "WOOD", "STBLCP", "FASTCP",
    "XSAI", "TAUSS", "T2V", "T2B", "Q2V", "Q2B", "TRAD", "NEE",
    "GPP", "NPP", "FVEG", "QIN", "RUNSF", "RUNSB", "ECAN", "EDIR",
    "ETRAN", "FSA", "FIRA", "APAR", "PSN", "SAV", "SAG", "RSSUN",
    "RSSHA", "BGAP", "WGAP", "TGV", "TGB", "CHV", "CHB", "SHG",
    "SHC", "SHB", "EVG", "EVB", "GHV", "GHB", "IRG", "IRC",
    "IRB", "TR", "EVC", "CHLEAF", "CHUC", "CHV2", "CHB2", "CHSTAR",
    "SMCWTD", "RECH", "QRFS", "QSPRINGS", "QSLAT", "ACINTS", "ACINTR", "ACDRIPR",
    "ACTHROR", "ACEVAC", "ACDEWC", "FORCTLSM", "FORCQLSM", "FORCPLSM", "FORCZLSM", "FORCWLSM",
    "ACRAINLSM", "ACRUNSB", "ACRUNSF", "ACECAN", "ACETRAN", "ACEDIR", "ACQLAT", "ACQRF",
    "ACETLSM", "ACSNOWLSM", "ACSUBC", "ACFROC", "ACFRZC", "ACMELTC", "ACSNBOT", "ACSNMELT",
    "ACPONDING", "ACSNSUB", "ACSNFRO", "ACRAINSNOW", "ACDRIPS", "ACTHROS", "ACSAGB", "ACIRB",
    "ACSHB", "ACEVB", "ACGHB", "ACPAHB", "ACSAGV", "ACIRG", "ACSHG", "ACEVG",
    "ACGHV", "ACPAHG", "ACSAV", "ACIRC", "ACSHC", "ACEVC", "ACTR", "ACPAHV",
    "ACSWDNLSM", "ACSWUPLSM", "ACLWDNLSM", "ACLWUPLSM", "ACSHFLSM", "ACLHFLSM", "ACGHFLSM", "ACPAHLSM",
    "ACCANHS", "SOILENERGY", "SNOWENERGY", "ACEFLXB", "GRAIN", "GDD", "CROPCAT", "PGS",
    "QTDRAIN", "IRNUMSI", "IRNUMMI", "IRNUMFI", "IRSIVOL", "IRMIVOL", "IRFIVOL", "IRELOSS",
    "IRRSPLH", "C1H", "C2H", "C1F", "C2F", "C3H", "C4H", "C3F",
    "C4F", "PCB", "PC", "LANDMASK", "LAKEMASK", "SST", "SST_INPUT",
)


@dataclass(frozen=True)
class WrfoutVariableSpec:
    """WRF variable schema metadata for the v0 minimum output subset."""

    name: str
    dimensions: tuple[str, ...]
    memory_order: str
    description: str
    units: str
    stagger: str = ""
    coordinates: str | None = None
    dtype: str = "f4"


def _spec(
    name: str,
    dimensions: tuple[str, ...],
    memory_order: str,
    description: str,
    units: str,
    *,
    stagger: str = "",
    coordinates: str | None = None,
    dtype: str = "f4",
) -> WrfoutVariableSpec:
    return WrfoutVariableSpec(
        name=name,
        dimensions=dimensions,
        memory_order=memory_order,
        description=description,
        units=units,
        stagger=stagger,
        coordinates=coordinates,
        dtype=dtype,
    )


XY = ("Time", "south_north", "west_east")
XYZ = ("Time", "bottom_top", "south_north", "west_east")
U_XYZ = ("Time", "bottom_top", "south_north", "west_east_stag")
V_XYZ = ("Time", "bottom_top", "south_north_stag", "west_east")
W_XYZ = ("Time", "bottom_top_stag", "south_north", "west_east")
Z_XYZ = ("Time", "bottom_top_stag", "south_north", "west_east")
U_XY = ("Time", "south_north", "west_east_stag")
V_XY = ("Time", "south_north_stag", "west_east")
# P0-5a additions: soil column (4-layer, Z-staggered) + 1-D eta level columns
# + scalar Time-only fields, matching WRF wrfout dimension layout exactly.
SOIL = ("Time", "soil_layers_stag", "south_north", "west_east")
SNOW = ("Time", "snow_layers_stag", "south_north", "west_east")
SNSO = ("Time", "snso_layers_stag", "south_north", "west_east")
SEED = ("Time", "seed_dim_stag")
Z_HALF = ("Time", "bottom_top")
Z_FULL = ("Time", "bottom_top_stag")
SOIL_1D = ("Time", "soil_layers_stag")
TIME_ONLY = ("Time",)
MAPFAC_U_XY = ("Time", "south_north", "west_east_stag")
MAPFAC_V_XY = ("Time", "south_north_stag", "west_east")

WRFOUT_VARIABLE_SPECS: dict[str, WrfoutVariableSpec] = {
    "XTIME": _spec(
        "XTIME",
        ("Time",),
        "0  ",
        "minutes since simulation start",
        "minutes since simulation start",
    ),
    "XLAT": _spec("XLAT", XY, "XY ", "LATITUDE, SOUTH IS NEGATIVE", "degree_north", coordinates="XLONG XLAT"),
    "XLONG": _spec("XLONG", XY, "XY ", "LONGITUDE, WEST IS NEGATIVE", "degree_east", coordinates="XLONG XLAT"),
    "XLAT_U": _spec(
        "XLAT_U",
        U_XY,
        "XY ",
        "LATITUDE, SOUTH IS NEGATIVE",
        "degree_north",
        stagger="X",
        coordinates="XLONG_U XLAT_U",
    ),
    "XLONG_U": _spec(
        "XLONG_U",
        U_XY,
        "XY ",
        "LONGITUDE, WEST IS NEGATIVE",
        "degree_east",
        stagger="X",
        coordinates="XLONG_U XLAT_U",
    ),
    "XLAT_V": _spec(
        "XLAT_V",
        V_XY,
        "XY ",
        "LATITUDE, SOUTH IS NEGATIVE",
        "degree_north",
        stagger="Y",
        coordinates="XLONG_V XLAT_V",
    ),
    "XLONG_V": _spec(
        "XLONG_V",
        V_XY,
        "XY ",
        "LONGITUDE, WEST IS NEGATIVE",
        "degree_east",
        stagger="Y",
        coordinates="XLONG_V XLAT_V",
    ),
    "HGT": _spec("HGT", XY, "XY ", "Terrain Height", "m", coordinates="XLONG XLAT XTIME"),
    "LANDMASK": _spec(
        "LANDMASK",
        XY,
        "XY ",
        "LAND MASK (1 FOR LAND, 0 FOR WATER)",
        "",
        coordinates="XLONG XLAT XTIME",
    ),
    "LU_INDEX": _spec("LU_INDEX", XY, "XY ", "LAND USE CATEGORY", "", coordinates="XLONG XLAT XTIME"),
    "U10": _spec("U10", XY, "XY ", "U at 10 M", "m s-1", coordinates="XLONG XLAT XTIME"),
    "V10": _spec("V10", XY, "XY ", "V at 10 M", "m s-1", coordinates="XLONG XLAT XTIME"),
    "T2": _spec("T2", XY, "XY ", "TEMP at 2 M", "K", coordinates="XLONG XLAT XTIME"),
    "Q2": _spec("Q2", XY, "XY ", "QV at 2 M", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "PSFC": _spec("PSFC", XY, "XY ", "SFC PRESSURE", "Pa", coordinates="XLONG XLAT XTIME"),
    "RAINC": _spec(
        "RAINC",
        XY,
        "XY ",
        "ACCUMULATED TOTAL CUMULUS PRECIPITATION",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "RAINNC": _spec(
        "RAINNC",
        XY,
        "XY ",
        "ACCUMULATED TOTAL GRID SCALE PRECIPITATION",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "RAINSH": _spec(
        "RAINSH",
        XY,
        "XY ",
        "ACCUMULATED SHALLOW CUMULUS PRECIPITATION",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWDOWN": _spec(
        "SWDOWN",
        XY,
        "XY ",
        "DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "GLW": _spec(
        "GLW",
        XY,
        "XY ",
        "DOWNWARD LONG WAVE FLUX AT GROUND SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "PBLH": _spec("PBLH", XY, "XY ", "PBL HEIGHT", "m", coordinates="XLONG XLAT XTIME"),
    "UST": _spec("UST", XY, "XY ", "U* IN SIMILARITY THEORY", "m s-1", coordinates="XLONG XLAT XTIME"),
    "HFX": _spec(
        "HFX",
        XY,
        "XY ",
        "UPWARD HEAT FLUX AT THE SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LH": _spec(
        "LH",
        XY,
        "XY ",
        "LATENT HEAT FLUX AT THE SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "TSK": _spec(
        "TSK",
        XY,
        "XY ",
        "SURFACE SKIN TEMPERATURE",
        "K",
        coordinates="XLONG XLAT XTIME",
    ),
    "CLDFRA": _spec("CLDFRA", XYZ, "XYZ", "CLOUD FRACTION", "", coordinates="XLONG XLAT XTIME"),
    "QCLOUD": _spec(
        "QCLOUD",
        XYZ,
        "XYZ",
        "Cloud water mixing ratio",
        "kg kg-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "QICE": _spec("QICE", XYZ, "XYZ", "Ice mixing ratio", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "QRAIN": _spec(
        "QRAIN",
        XYZ,
        "XYZ",
        "Rain water mixing ratio",
        "kg kg-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "U": _spec(
        "U",
        U_XYZ,
        "XYZ",
        "x-wind component",
        "m s-1",
        stagger="X",
        coordinates="XLONG_U XLAT_U XTIME",
    ),
    "V": _spec(
        "V",
        V_XYZ,
        "XYZ",
        "y-wind component",
        "m s-1",
        stagger="Y",
        coordinates="XLONG_V XLAT_V XTIME",
    ),
    "W": _spec(
        "W",
        W_XYZ,
        "XYZ",
        "z-wind component",
        "m s-1",
        stagger="Z",
        coordinates="XLONG XLAT XTIME",
    ),
    "T": _spec(
        "T",
        XYZ,
        "XYZ",
        "perturbation potential temperature theta-t0",
        "K",
        coordinates="XLONG XLAT XTIME",
    ),
    "THM": _spec(
        "THM",
        XYZ,
        "XYZ",
        "either 1) pert moist pot temp=(1+Rv/Rd Qv)*(theta+T0)-T0; or 2) pert dry pot temp=theta; based on use_theta_m setting",
        "K",
        coordinates="XLONG XLAT XTIME",
    ),
    "QVAPOR": _spec(
        "QVAPOR",
        XYZ,
        "XYZ",
        "Water vapor mixing ratio",
        "kg kg-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "P": _spec("P", XYZ, "XYZ", "perturbation pressure", "Pa", coordinates="XLONG XLAT XTIME"),
    "PB": _spec("PB", XYZ, "XYZ", "BASE STATE PRESSURE", "Pa", coordinates="XLONG XLAT XTIME"),
    "PH": _spec(
        "PH",
        Z_XYZ,
        "XYZ",
        "perturbation geopotential",
        "m2 s-2",
        stagger="Z",
        coordinates="XLONG XLAT XTIME",
    ),
    "PHB": _spec(
        "PHB",
        Z_XYZ,
        "XYZ",
        "base-state geopotential",
        "m2 s-2",
        stagger="Z",
        coordinates="XLONG XLAT XTIME",
    ),
    "MU": _spec(
        "MU",
        XY,
        "XY ",
        "perturbation dry air mass in column",
        "Pa",
        coordinates="XLONG XLAT XTIME",
    ),
    "MUB": _spec(
        "MUB",
        XY,
        "XY ",
        "base state dry air mass in column",
        "Pa",
        coordinates="XLONG XLAT XTIME",
    ),
    # --- P0-5a (a) extra hydrometeors + number concentrations + MYNN TKE ---
    "QSNOW": _spec("QSNOW", XYZ, "XYZ", "Snow mixing ratio", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "QGRAUP": _spec("QGRAUP", XYZ, "XYZ", "Graupel mixing ratio", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "QNICE": _spec("QNICE", XYZ, "XYZ", "Ice Number concentration", "  kg-1", coordinates="XLONG XLAT XTIME"),
    "QNRAIN": _spec("QNRAIN", XYZ, "XYZ", "Rain Number concentration", "  kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QNSNOW": _spec("QNSNOW", XYZ, "XYZ", "Snow Number concentration", "  kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QNGRAUPEL": _spec(
        "QNGRAUPEL",
        XYZ,
        "XYZ",
        "Graupel Number concentration",
        "  kg(-1)",
        coordinates="XLONG XLAT XTIME",
    ),
    "QNCLOUD": _spec(
        "QNCLOUD",
        XYZ,
        "XYZ",
        "cloud water Number concentration",
        "  kg(-1)",
        coordinates="XLONG XLAT XTIME",
    ),
    "QNCCN": _spec("QNCCN", XYZ, "XYZ", "CCN Number concentration", "  kg(-1)", coordinates="XLONG XLAT XTIME"),
    # v0.17 ADR-032 graupel/hail substrate (WRF Registry hail-family
    # qh/qnh/qvolg/qvolh names/descriptions/units).
    "QHAIL": _spec("QHAIL", XYZ, "XYZ", "Hail mixing ratio", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "QNHAIL": _spec("QNHAIL", XYZ, "XYZ", "Hail Number concentration", "  kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QVGRAUPEL": _spec("QVGRAUPEL", XYZ, "XYZ", "Graupel Particle Volume", "m(3) kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QVHAIL": _spec("QVHAIL", XYZ, "XYZ", "Hail Particle Volume", "m(3) kg(-1)", coordinates="XLONG XLAT XTIME"),
    # v0.16 aerosol-aware Thompson (mp=28) aerosol numbers (WRF Registry
    # thompsonaero qnwfa/qnifa names/descriptions).
    "QNWFA": _spec(
        "QNWFA",
        XYZ,
        "XYZ",
        "water-friendly aerosol number con",
        "  kg-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "QNIFA": _spec(
        "QNIFA",
        XYZ,
        "XYZ",
        "ice-friendly aerosol number con",
        "  kg-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "QKE": _spec("QKE", XYZ, "XYZ", "twice TKE from MYNN", "m2 s-2", coordinates="XLONG XLAT XTIME"),
    # --- P0-5a (b) grid-static coordinate / map-factor / Coriolis fields ---
    "ZNU": _spec("ZNU", Z_HALF, "Z  ", "eta values on half (mass) levels", ""),
    "ZNW": _spec("ZNW", Z_FULL, "Z  ", "eta values on full (w) levels", "", stagger="Z"),
    # Static Noah/Noah-MP soil-layer geometry (init_soil_depth_2; the same arrays
    # WRF repeats every history frame). ZS = layer-center depths, DZS = layer
    # thicknesses; 1-D columns of length soil_layers_stag. No coordinates attr
    # (matches the reference wrfout, which omits it for these 1-D soil fields).
    "ZS": _spec("ZS", SOIL_1D, "Z  ", "DEPTHS OF CENTERS OF SOIL LAYERS", "m", stagger="Z"),
    "DZS": _spec("DZS", SOIL_1D, "Z  ", "THICKNESSES OF SOIL LAYERS", "m", stagger="Z"),
    "MAPFAC_M": _spec("MAPFAC_M", XY, "XY ", "Map scale factor on mass grid", "", coordinates="XLONG XLAT XTIME"),
    "MAPFAC_U": _spec(
        "MAPFAC_U", MAPFAC_U_XY, "XY ", "Map scale factor on u-grid", "",
        stagger="X", coordinates="XLONG_U XLAT_U XTIME",
    ),
    "MAPFAC_V": _spec(
        "MAPFAC_V", MAPFAC_V_XY, "XY ", "Map scale factor on v-grid", "",
        stagger="Y", coordinates="XLONG_V XLAT_V XTIME",
    ),
    "F": _spec("F", XY, "XY ", "Coriolis sine latitude term", "s-1", coordinates="XLONG XLAT XTIME"),
    "E": _spec("E", XY, "XY ", "Coriolis cosine latitude term", "s-1", coordinates="XLONG XLAT XTIME"),
    "SINALPHA": _spec("SINALPHA", XY, "XY ", "Local sine of map rotation", "", coordinates="XLONG XLAT XTIME"),
    "COSALPHA": _spec("COSALPHA", XY, "XY ", "Local cosine of map rotation", "", coordinates="XLONG XLAT XTIME"),
    "XLAND": _spec(
        "XLAND", XY, "XY ", "LAND MASK (1 FOR LAND, 2 FOR WATER)", "",
        coordinates="XLONG XLAT XTIME",
    ),
    "P_TOP": _spec("P_TOP", TIME_ONLY, "0  ", "PRESSURE TOP OF THE MODEL", "Pa"),
    # --- v0.12.0 A1 (b2) extended grid-static vertical/map-factor metrics ---
    # Vertical eta-spacing metrics (DycoreMetrics.dn/dnw/rdn/rdnw/fnm/fnp).
    "DN": _spec("DN", Z_HALF, "Z  ", "d(eta) values between half (mass) levels", ""),
    "DNW": _spec("DNW", Z_HALF, "Z  ", "d(eta) values between full (w) levels", ""),
    "RDN": _spec("RDN", Z_HALF, "Z  ", "inverse d(eta) values between half (mass) levels", ""),
    "RDNW": _spec("RDNW", Z_HALF, "Z  ", "inverse d(eta) values between full (w) levels", ""),
    "FNM": _spec("FNM", Z_HALF, "Z  ", "upper weight for vertical stretching", ""),
    "FNP": _spec("FNP", Z_HALF, "Z  ", "lower weight for vertical stretching", ""),
    # Scalar vertical extrapolation constants (DycoreMetrics.cf1/cf2/cf3 + the
    # WRF top-level cfn/cfn1 derived from dnw/dn).
    "CFN": _spec("CFN", TIME_ONLY, "0  ", "extrapolation constant", ""),
    "CFN1": _spec("CFN1", TIME_ONLY, "0  ", "extrapolation constant", ""),
    "CF1": _spec("CF1", TIME_ONLY, "0  ", "2nd order extrapolation constant", ""),
    "CF2": _spec("CF2", TIME_ONLY, "0  ", "2nd order extrapolation constant", ""),
    "CF3": _spec("CF3", TIME_ONLY, "0  ", "2nd order extrapolation constant", ""),
    # Hybrid-eta C-coefficients (DycoreMetrics.c1h..c4h / c1f..c4f).
    "C1H": _spec("C1H", Z_HALF, "Z  ", "half levels, c1h = d bf / d eta, using znw", "Dimensionless"),
    "C2H": _spec("C2H", Z_HALF, "Z  ", "half levels, c2h = (1-c1h)*(p0-pt)", "Pa"),
    "C3H": _spec("C3H", Z_HALF, "Z  ", "half levels, c3h = bh", "Dimensionless"),
    "C4H": _spec("C4H", Z_HALF, "Z  ", "half levels, c4h = (eta-bh)*(p0-pt), using znu", "Pa"),
    "C1F": _spec("C1F", Z_FULL, "Z  ", "full levels, c1f = d bf / d eta, using znu", "Dimensionless", stagger="Z"),
    "C2F": _spec("C2F", Z_FULL, "Z  ", "full levels, c2f = (1-c1f)*(p0-pt)", "Pa", stagger="Z"),
    "C3F": _spec("C3F", Z_FULL, "Z  ", "full levels, c3f = bf", "Dimensionless", stagger="Z"),
    "C4F": _spec("C4F", Z_FULL, "Z  ", "full levels, c4f = (eta-bf)*(p0-pt), using znw", "Pa", stagger="Z"),
    # Directional map-scale factors (DycoreMetrics.msftx/msfty/msfux/msfuy/msfvx/msfvy).
    "MAPFAC_MX": _spec(
        "MAPFAC_MX", XY, "XY ", "Map scale factor on mass grid, x direction", "",
        coordinates="XLONG XLAT XTIME",
    ),
    "MAPFAC_MY": _spec(
        "MAPFAC_MY", XY, "XY ", "Map scale factor on mass grid, y direction", "",
        coordinates="XLONG XLAT XTIME",
    ),
    "MAPFAC_UX": _spec(
        "MAPFAC_UX", MAPFAC_U_XY, "XY ", "Map scale factor on u-grid, x direction", "",
        stagger="X", coordinates="XLONG_U XLAT_U XTIME",
    ),
    "MAPFAC_UY": _spec(
        "MAPFAC_UY", MAPFAC_U_XY, "XY ", "Map scale factor on u-grid, y direction", "",
        stagger="X", coordinates="XLONG_U XLAT_U XTIME",
    ),
    "MAPFAC_VX": _spec(
        "MAPFAC_VX", MAPFAC_V_XY, "XY ", "Map scale factor on v-grid, x direction", "",
        stagger="Y", coordinates="XLONG_V XLAT_V XTIME",
    ),
    "MAPFAC_VY": _spec(
        "MAPFAC_VY", MAPFAC_V_XY, "XY ", "Map scale factor on v-grid, y direction", "",
        stagger="Y", coordinates="XLONG_V XLAT_V XTIME",
    ),
    # Inverse grid lengths (1/dx, 1/dy) from the projection.
    "RDX": _spec("RDX", TIME_ONLY, "0  ", "INVERSE X GRID LENGTH", "m-1"),
    "RDY": _spec("RDY", TIME_ONLY, "0  ", "INVERSE Y GRID LENGTH", "m-1"),
    # --- v0.12.0 A2 (h) trivially-derived diagnostics ---
    "CLAT": _spec(
        "CLAT", XY, "XY ", "COMPUTATIONAL GRID LATITUDE, SOUTH IS NEGATIVE", "degree_north",
        coordinates="XLONG XLAT XTIME",
    ),
    "COSZEN": _spec(
        "COSZEN", XY, "XY ", "COS of SOLAR ZENITH ANGLE", "dimensionless",
        coordinates="XLONG XLAT XTIME",
    ),
    "SR": _spec(
        "SR", XY, "XY ", "fraction of frozen precipitation", "-",
        coordinates="XLONG XLAT XTIME",
    ),
    "SNOWC": _spec(
        "SNOWC", XY, "XY ", "FLAG INDICATING SNOW COVERAGE (1 FOR SNOW COVER)", "",
        coordinates="XLONG XLAT XTIME",
    ),
    # --- P0-5a (c) accumulated precipitation partition ---
    "SNOWNC": _spec(
        "SNOWNC", XY, "XY ", "ACCUMULATED TOTAL GRID SCALE SNOW AND ICE", "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "GRAUPELNC": _spec(
        "GRAUPELNC", XY, "XY ", "ACCUMULATED TOTAL GRID SCALE GRAUPEL", "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    # v0.17 hail microphysics (WSM7/WDM7): accumulated grid-scale hail.
    "HAILNC": _spec(
        "HAILNC", XY, "XY ", "ACCUMULATED TOTAL GRID SCALE HAIL", "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    # --- P0-5a (d) surface energy/moisture-budget fluxes + 2-m theta ---
    "QFX": _spec(
        "QFX", XY, "XY ", "UPWARD MOISTURE FLUX AT THE SURFACE", "kg m-2 s-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "GRDFLX": _spec("GRDFLX", XY, "XY ", "GROUND HEAT FLUX", "W m-2", coordinates="XLONG XLAT XTIME"),
    "TH2": _spec("TH2", XY, "XY ", "POT TEMP at 2 M", "K", coordinates="XLONG XLAT XTIME"),
    # --- P0-5a (e) prognostic Noah-MP soil/snow land columns + land diagnostics ---
    "TSLB": _spec("TSLB", SOIL, "XYZ", "SOIL TEMPERATURE", "K", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "SMOIS": _spec("SMOIS", SOIL, "XYZ", "SOIL MOISTURE", "m3 m-3", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "SH2O": _spec("SH2O", SOIL, "XYZ", "SOIL LIQUID WATER", "m3 m-3", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "SNOW": _spec("SNOW", XY, "XY ", "SNOW WATER EQUIVALENT", "kg m-2", coordinates="XLONG XLAT XTIME"),
    "SNOWH": _spec("SNOWH", XY, "XY ", "PHYSICAL SNOW DEPTH", "m", coordinates="XLONG XLAT XTIME"),
    "CANWAT": _spec("CANWAT", XY, "XY ", "CANOPY WATER", "kg m-2", coordinates="XLONG XLAT XTIME"),
    "SFROFF": _spec("SFROFF", XY, "XY ", "SURFACE RUNOFF", "mm", coordinates="XLONG XLAT XTIME"),
    "UDROFF": _spec("UDROFF", XY, "XY ", "UNDERGROUND RUNOFF", "mm", coordinates="XLONG XLAT XTIME"),
    "ALBEDO": _spec("ALBEDO", XY, "XY ", "ALBEDO", "-", coordinates="XLONG XLAT XTIME"),
    "EMISS": _spec("EMISS", XY, "XY ", "SURFACE EMISSIVITY", "", coordinates="XLONG XLAT XTIME"),
    "TSNO": _spec("TSNO", SNOW, "XYZ", "snow temperature", "K", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "SNICE": _spec("SNICE", SNOW, "XYZ", "snow layer ice", "mm", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "SNLIQ": _spec("SNLIQ", SNOW, "XYZ", "snow layer liquid", "mm", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "ZSNSO": _spec(
        "ZSNSO",
        SNSO,
        "XYZ",
        "layer-bottom depth from snow surf",
        "m",
        stagger="Z",
        coordinates="XLONG XLAT XTIME",
    ),
    # WRF integer field (FieldType 106): active snow-layer count in {-2,-1,0}.
    # The reference wrfout labels its units "m3 m-3" (a known upstream WRF
    # Registry typo); we reproduce it byte-for-byte for schema conformance.
    "ISNOW": _spec(
        "ISNOW",
        XY,
        "XY ",
        "no. of snow layer",
        "m3 m-3",
        coordinates="XLONG XLAT XTIME",
        dtype="i4",
    ),
    "SNEQVO": _spec(
        "SNEQVO",
        XY,
        "XY ",
        "snow mass at last time step",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "CANLIQ": _spec(
        "CANLIQ",
        XY,
        "XY ",
        "intercepted liquid water",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "CANICE": _spec(
        "CANICE",
        XY,
        "XY ",
        "intercepted ice mass",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "ISEEDARR_SPPT": _spec(
        "ISEEDARR_SPPT",
        SEED,
        "Z  ",
        "Array to hold seed for restart, SPPT",
        "",
        stagger="Z",
        dtype="i4",
    ),
    "ISEEDARR_SKEBS": _spec(
        "ISEEDARR_SKEBS",
        SEED,
        "Z  ",
        "Array to hold seed for restart, SKEBS",
        "",
        stagger="Z",
        dtype="i4",
    ),
    "ISEEDARRAY_SPP_CONV": _spec(
        "ISEEDARRAY_SPP_CONV",
        SEED,
        "Z  ",
        "Array to hold seed for restart, RAND_PERT2",
        "",
        stagger="Z",
        dtype="i4",
    ),
    "ISEEDARRAY_SPP_PBL": _spec(
        "ISEEDARRAY_SPP_PBL",
        SEED,
        "Z  ",
        "Array to hold seed for restart, RAND_PERT3",
        "",
        stagger="Z",
        dtype="i4",
    ),
    "ISEEDARRAY_SPP_LSM": _spec(
        "ISEEDARRAY_SPP_LSM",
        SEED,
        "Z  ",
        "Array to hold seed for restart, RAND_PERT4",
        "",
        stagger="Z",
        dtype="i4",
    ),
    # --- B1 (v0.12.0) RRTMG up/down all-sky radiation flux diagnostics --------
    # Instantaneous surface (bottom-of-atmosphere) and top-of-atmosphere up/down
    # SW/LW fluxes + the slope-normal surface SW flux (SWNORM) + outgoing LW (OLR).
    # Schemas (dims/MemoryOrder/stagger/units/description/dtype) are copied verbatim
    # from the reference Gen2 wrfout_d02; every field maps to a flux slice the RRTMG
    # SW/LW column solvers already compute (surface == bottom interface, TOA == top
    # interface). OLR == LWUPT (WRF's TOA outgoing LW) and is derived in
    # ``prepare_wrfout_payload`` from the diagnostic LWUPT.
    #
    # v0.13.0: the WRF clear-sky ``...C`` flux vars (SWDNBC/SWUPBC/LWDNBC/LWUPBC
    # /SWDNTC/SWUPTC/LWDNTC/LWUPTC) ARE now specced/emitted -- the RRTMG SW/LW
    # solvers run the WRF second clear-sky (cloud-free) radiative-transfer pass
    # (``solve_rrtmg_*_column(..., with_clear_sky=True)``; WRF ``pbbcd/pbbcu``,
    # ``totdclfl/totuclfl``).  Still ADD-only: emitted only when the radiation
    # diagnostics supply them, never fabricated, and the all-sky fluxes above are
    # byte-identical with or without the clear-sky pass.
    "SWDNB": _spec(
        "SWDNB", XY, "XY ", "INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWUPB": _spec(
        "SWUPB", XY, "XY ", "INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWDNB": _spec(
        "LWDNB", XY, "XY ", "INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWUPB": _spec(
        "LWUPB", XY, "XY ", "INSTANTANEOUS UPWELLING LONGWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWDNT": _spec(
        "SWDNT", XY, "XY ", "INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWUPT": _spec(
        "SWUPT", XY, "XY ", "INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWDNT": _spec(
        "LWDNT", XY, "XY ", "INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWUPT": _spec(
        "LWUPT", XY, "XY ", "INSTANTANEOUS UPWELLING LONGWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "OLR": _spec(
        "OLR", XY, "XY ", "TOA OUTGOING LONG WAVE", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWNORM": _spec(
        "SWNORM", XY, "XY ", "NORMAL SHORT WAVE FLUX AT GROUND SURFACE (SLOPE-DEPENDENT)", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    # --- v0.13.0 RRTMG CLEAR-SKY (cloud-free) up/down radiation flux diagnostics.
    # Schemas copied verbatim from the reference Gen2 wrfout; each maps to a
    # clear-sky flux slice the RRTMG SW/LW clear-sky pass computes (WRF `pbbcd/
    # pbbcu`, `totdclfl/totuclfl`).  Bottom == surface interface, top == model-top
    # interface.  ADD-only (no fallback): emitted only when the radiation
    # diagnostics supply the clear-sky fluxes. ---
    "SWDNBC": _spec(
        "SWDNBC", XY, "XY ", "INSTANTANEOUS DOWNWELLING CLEAR SKY SHORTWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWUPBC": _spec(
        "SWUPBC", XY, "XY ", "INSTANTANEOUS UPWELLING CLEAR SKY SHORTWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWDNBC": _spec(
        "LWDNBC", XY, "XY ", "INSTANTANEOUS DOWNWELLING CLEAR SKY LONGWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWUPBC": _spec(
        "LWUPBC", XY, "XY ", "INSTANTANEOUS UPWELLING CLEAR SKY LONGWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWDNTC": _spec(
        "SWDNTC", XY, "XY ", "INSTANTANEOUS DOWNWELLING CLEAR SKY SHORTWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWUPTC": _spec(
        "SWUPTC", XY, "XY ", "INSTANTANEOUS UPWELLING CLEAR SKY SHORTWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWDNTC": _spec(
        "LWDNTC", XY, "XY ", "INSTANTANEOUS DOWNWELLING CLEAR SKY LONGWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWUPTC": _spec(
        "LWUPTC", XY, "XY ", "INSTANTANEOUS UPWELLING CLEAR SKY LONGWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
}


_FULL_XYZ_GENERIC_NAMES: tuple[str, ...] = (
    "P_HYD", "DTAUX3D", "DTAUY3D",
)
_FULL_Z_XYZ_GENERIC_NAMES: tuple[str, ...] = (
    "TKE_PBL", "EL_PBL",
)
_FULL_XY_GENERIC_NAMES: tuple[str, ...] = (
    "VAR_SSO", "NEST_POS", "AREA2D", "DX2D", "SHDMAX", "SHDMIN", "SHDAVG", "SNOALB",
    "SEAICE", "XICEM", "VEGFRA", "ACGRDFLX", "ACSNOM", "SSTSK", "WATER_DEPTH", "LAI",
    "MAXMF", "MAXWIDTH", "ZTOP_PLUME", "DUSFCG", "DVSFCG", "VAR", "CON", "OA1",
    "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4", "O3_GFS_DU",
    "ACSWUPT", "ACSWUPTC", "ACSWDNT", "ACSWDNTC", "ACSWUPB", "ACSWUPBC", "ACSWDNB", "ACSWDNBC",
    "ACLWUPT", "ACLWUPTC", "ACLWDNT", "ACLWDNTC", "ACLWUPB", "ACLWUPBC", "ACLWDNB", "ACLWDNBC",
    "ALBBCK", "NOAHRES", "TMN", "ACHFX", "ACLHF", "TV", "TG", "EAH",
    "TAH", "CM", "CH", "FWET", "ALBOLD", "QSNOWXY", "QRAINXY", "WSLAKE",
    "ZWT", "WA", "WT", "LFMASS", "RTMASS", "STMASS", "WOOD", "STBLCP",
    "FASTCP", "XSAI", "TAUSS", "T2V", "T2B", "Q2V", "Q2B", "TRAD",
    "NEE", "GPP", "NPP", "FVEG", "QIN", "RUNSF", "RUNSB", "ECAN",
    "EDIR", "ETRAN", "FSA", "FIRA", "APAR", "PSN", "SAV", "SAG",
    "RSSUN", "RSSHA", "BGAP", "WGAP", "TGV", "TGB", "CHV", "CHB",
    "SHG", "SHC", "SHB", "EVG", "EVB", "GHV", "GHB", "IRG",
    "IRC", "IRB", "TR", "EVC", "CHLEAF", "CHUC", "CHV2", "CHB2",
    "CHSTAR", "SMCWTD", "RECH", "QRFS", "QSPRINGS", "QSLAT", "ACINTS", "ACINTR",
    "ACDRIPR", "ACTHROR", "ACEVAC", "ACDEWC", "FORCTLSM", "FORCQLSM", "FORCPLSM", "FORCZLSM",
    "FORCWLSM", "ACRAINLSM", "ACRUNSB", "ACRUNSF", "ACECAN", "ACETRAN", "ACEDIR", "ACQLAT",
    "ACQRF", "ACETLSM", "ACSNOWLSM", "ACSUBC", "ACFROC", "ACFRZC", "ACMELTC", "ACSNBOT",
    "ACSNMELT", "ACPONDING", "ACSNSUB", "ACSNFRO", "ACRAINSNOW", "ACDRIPS", "ACTHROS", "ACSAGB",
    "ACIRB", "ACSHB", "ACEVB", "ACGHB", "ACPAHB", "ACSAGV", "ACIRG", "ACSHG",
    "ACEVG", "ACGHV", "ACPAHG", "ACSAV", "ACIRC", "ACSHC", "ACEVC", "ACTR",
    "ACPAHV", "ACSWDNLSM", "ACSWUPLSM", "ACLWDNLSM", "ACLWUPLSM", "ACSHFLSM", "ACLHFLSM", "ACGHFLSM",
    "ACPAHLSM", "ACCANHS", "SOILENERGY", "SNOWENERGY", "ACEFLXB", "GRAIN", "GDD", "QTDRAIN",
    "IRSIVOL", "IRMIVOL", "IRFIVOL", "IRELOSS", "IRRSPLH", "PCB", "PC", "LAKEMASK",
    "SST", "SST_INPUT",
)
_FULL_XY_INT_GENERIC_NAMES: tuple[str, ...] = (
    "IVGTYP", "ISLTYP", "CROPCAT", "PGS", "IRNUMSI", "IRNUMMI", "IRNUMFI",
)
_FULL_V_XY_GENERIC_NAMES: tuple[str, ...] = ("MF_VX_INV",)
_FULL_TIME_GENERIC_NAMES: tuple[str, ...] = (
    "HFX_FORCE", "LH_FORCE", "TSK_FORCE", "HFX_FORCE_TEND", "LH_FORCE_TEND", "TSK_FORCE_TEND",
    "RESM", "ZETATOP", "T00", "P00", "TLP", "TISO", "TLP_STRAT", "P_STRAT",
    "MAX_MSFTX", "MAX_MSFTY",
)
_FULL_TIME_INT_GENERIC_NAMES: tuple[str, ...] = (
    "BATHYMETRY_FLAG", "THIS_IS_AN_IDEAL_RUN", "ITIMESTEP", "GOT_VAR_SSO", "SAVE_TOPO_FROM_REAL",
)
_FULL_SEED_INT_GENERIC_NAMES: tuple[str, ...] = ("ISEEDARR_RAND_PERTURB",)


def _register_full_wrfout_generic_specs() -> None:
    for name in _FULL_XYZ_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(
            name, _spec(name, XYZ, "XYZ", name, "", coordinates="XLONG XLAT XTIME")
        )
    for name in _FULL_Z_XYZ_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(
            name, _spec(name, Z_XYZ, "XYZ", name, "", stagger="Z", coordinates="XLONG XLAT XTIME")
        )
    for name in _FULL_XY_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(
            name, _spec(name, XY, "XY ", name, "", coordinates="XLONG XLAT XTIME")
        )
    for name in _FULL_XY_INT_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(
            name, _spec(name, XY, "XY ", name, "", coordinates="XLONG XLAT XTIME", dtype="i4")
        )
    for name in _FULL_V_XY_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(
            name,
            _spec(
                name,
                V_XY,
                "XY ",
                name,
                "",
                stagger="Y",
                coordinates="XLONG_V XLAT_V XTIME",
            ),
        )
    for name in _FULL_TIME_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(name, _spec(name, TIME_ONLY, "0  ", name, ""))
    for name in _FULL_TIME_INT_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(name, _spec(name, TIME_ONLY, "0  ", name, "", dtype="i4"))
    for name in _FULL_SEED_INT_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(name, _spec(name, SEED, "Z  ", name, "", stagger="Z", dtype="i4"))


_register_full_wrfout_generic_specs()


def write_wrfout_netcdf(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    path: str | Path,
    *,
    valid_time: datetime | date | str,
    lead_hours: float,
    run_start: datetime | date | str,
    diagnostics: Mapping[str, Any] | None = None,
    land_state: Any | None = None,
    variable_subset: tuple[str, ...] | frozenset[str] | None = None,
    include_mandatory_coords: bool = False,
    compress: bool = False,
    full_variable_set: bool = False,
) -> Path:
    """Write one WRF-style ``wrfout`` NetCDF file for the M7 minimum variable set.

    The function accepts plain Python/numpy objects as well as the project
    ``State``/``GridSpec`` objects. Device arrays, if passed after an operational
    run, are converted only at this output boundary.

    ``diagnostics`` optionally carries host-only output diagnostics/metadata.
    Static latitude/longitude payloads (``XLAT``/``XLONG`` and staggered variants)
    are selected from this map before the legacy State/projection lookup, so real
    WRF statics can be routed to output without adding JIT-visible state leaves.
    It also carries operational surface-layer diagnostics
    (e.g. the M9 surface map: ``T2``/``U10``/``V10``/``Q2``/``PSFC``/``SWDOWN``/
    ``GLW``/``PBLH``/``TSK`` plus the P0-5a additions ``QFX``/``GRDFLX``). When a
    name is present there, it OVERRIDES the state/default for that output field --
    the writer otherwise falls back to raw lowest-level fields which are
    physically wrong over terrain (e.g. raw level-1 wind/theta read far too
    strong/warm at a high summit). When ``diagnostics`` is ``None`` the behaviour
    is byte-for-byte identical to the legacy path, so no other caller regresses.

    ``land_state`` optionally carries the prognostic Noah-MP land carry
    (``NoahMPLandState``: 4-layer ``tslb``/``smois``/``sh2o``, bulk snow, canopy
    water, runoff, albedo, emissivity). When supplied, the WRF soil/snow land
    fields (``TSLB``/``SMOIS``/``SH2O``/``SNOW``/``SNOWH``/``CANWAT``/``SFROFF``/
    ``UDROFF``/``ALBEDO``/``EMISS``) are written; when ``None`` they are simply
    absent from the file (the writer never fabricates a soil profile).

    ``variable_subset`` optionally restricts the emitted variables to the named
    subset -- the compact training-output path (#122). ``include_mandatory_coords``
    additionally force-emits the geometry/eta/lat-lon coordinates and ``compress``
    applies lossless NetCDF4 compression; both default OFF. When ``variable_subset``
    is ``None`` (the default) the full uncompressed output is byte-identical to
    before.

    ``full_variable_set`` is the opt-in heavy WRF compatibility stream. When True
    the payload is expanded to :data:`FULL_WRFOUT_VARIABLES` (375 names including
    ``Times``), using real state/diagnostics where present and inactive defaults
    for WRF stream slots this port does not currently carry. Default False keeps
    the historical operational subset unchanged.
    """

    prepared = prepare_wrfout_payload(
        state,
        grid,
        namelist,
        path,
        valid_time=valid_time,
        lead_hours=lead_hours,
        run_start=run_start,
        diagnostics=diagnostics,
        land_state=land_state,
        variable_subset=variable_subset,
        include_mandatory_coords=include_mandatory_coords,
        full_variable_set=full_variable_set,
    )
    return write_prepared_wrfout(
        prepared,
        variable_subset=variable_subset,
        include_mandatory_coords=include_mandatory_coords,
        compress=compress,
    )


@dataclass(frozen=True)
class PreparedWrfout:
    """Fully host-materialized wrfout payload (v0.2.0 wall-clock win #3).

    Everything needed to write one wrfout NetCDF file with NO device-array or live
    model-state dependency: every field is already a host ``np.float32`` array
    (the device->host pull happened in :func:`prepare_wrfout_payload` while the
    GPU result was still resident). This object can therefore be handed to a
    background writer thread while the GPU advances the next forecast hour, with
    no risk of racing a donated/reused device buffer. The NetCDF bytes written are
    byte-for-byte identical to the synchronous path -- only the wall-clock timing
    of the write changes.
    """

    target: Path
    dimensions: Mapping[str, int | None]
    fields: Mapping[str, np.ndarray]
    run_start_dt: datetime
    valid_dt: datetime
    lead_hours: float
    grid: Any
    namelist: Any
    full_variable_set: bool = False


@dataclass(frozen=True)
class _WrfoutHostArrays:
    by_id: Mapping[int, Any]

    def host_value(self, value: Any) -> Any:
        return self.by_id.get(id(value), value)


@contextmanager
def _batched_wrfout_device_get(
    *,
    state: Any,
    grid: Any,
    diagnostics: Mapping[str, Any] | None,
    land_state: Any | None,
    requested_names: frozenset[str] | None = None,
) -> Iterator[None]:
    """Materialize device leaves once for one wrfout payload build."""

    context = _build_wrfout_host_arrays(
        state,
        grid,
        diagnostics,
        land_state,
        requested_names=requested_names,
    )
    token = _WRFOUT_HOST_ARRAYS.set(context)
    try:
        yield
    finally:
        _WRFOUT_HOST_ARRAYS.reset(token)


def _build_wrfout_host_arrays(
    *sources: Any, requested_names: frozenset[str] | None = None
) -> _WrfoutHostArrays:
    try:
        import jax
    except Exception:  # pragma: no cover - JAX is a project dependency.
        return _WrfoutHostArrays({})

    array_type = getattr(jax, "Array", ())
    if not array_type:
        return _WrfoutHostArrays({})

    leaves: list[Any] = []
    seen_objects: set[int] = set()
    seen_arrays: set[int] = set()

    def visit(value: Any) -> None:
        if value is None or isinstance(
            value, (str, bytes, bool, int, float, complex, date, datetime, Path)
        ):
            return
        if isinstance(value, array_type):
            value_id = id(value)
            if value_id not in seen_arrays:
                seen_arrays.add(value_id)
                leaves.append(value)
            return
        if isinstance(value, np.ndarray):
            return
        value_id = id(value)
        if value_id in seen_objects:
            return
        seen_objects.add(value_id)
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in dataclass_fields(value):
                visit(getattr(value, field.name, None))
            return
        namespace = getattr(value, "__dict__", None)
        if namespace:
            for item in namespace.values():
                visit(item)

    scan_sources = sources
    if requested_names is not None:
        state, grid, diagnostics, land_state = sources
        scan_sources = tuple(
            _requested_wrfout_source_values(
                state=state,
                grid=grid,
                diagnostics=diagnostics,
                land_state=land_state,
                requested_names=requested_names,
            )
        )
    for source in scan_sources:
        visit(source)
    if not leaves:
        return _WrfoutHostArrays({})
    host_leaves = jax.device_get(tuple(leaves))
    return _WrfoutHostArrays(
        {id(device_value): host_value for device_value, host_value in zip(leaves, host_leaves)}
    )


def _host_materialized_value(value: Any) -> Any:
    context = _WRFOUT_HOST_ARRAYS.get()
    if context is None:
        return value
    return context.host_value(value)


def _prepare_requested_names(
    variable_subset: tuple[str, ...] | frozenset[str] | None,
    *,
    include_mandatory_coords: bool,
) -> frozenset[str] | None:
    if variable_subset is None:
        return None
    requested = set(variable_subset)
    if include_mandatory_coords:
        requested.update(MANDATORY_WRFOUT_COORDINATES)
    requested.discard("Times")
    requested.discard("XTIME")
    return frozenset(requested)


_STATE_SOURCE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "U": ("U", "u"),
    "V": ("V", "v"),
    "W": ("W", "w"),
    "T": ("theta", "THETA"),
    "THM": ("theta", "THETA"),
    "QVAPOR": ("QVAPOR", "qv", "qvapor"),
    "QCLOUD": ("QCLOUD", "qc", "qcloud"),
    "QICE": ("QICE", "qi", "qice"),
    "QRAIN": ("QRAIN", "qr", "qrain"),
    "QSNOW": ("QSNOW", "qs", "qsnow"),
    "QGRAUP": ("QGRAUP", "qg", "qgraup"),
    "QKE": ("QKE", "qke"),
    "CLDFRA": ("CLDFRA", "cldfra", "cloud_fraction"),
    "P": ("p_perturbation", "P"),
    "PB": ("pb", "p_base", "PB"),
    "PH": ("ph_perturbation", "PH"),
    "PHB": ("phb", "ph_base", "PHB"),
    "MU": ("mu_perturbation", "MU"),
    "MUB": ("mub", "mu_base", "MUB"),
    "U10": ("U10", "u10"),
    "V10": ("V10", "v10"),
    "T2": ("T2", "t2"),
    "Q2": ("Q2", "q2"),
    "PSFC": ("PSFC", "psfc"),
    "RAINC": ("RAINC", "rainc", "rainc_acc"),
    "RAINNC": ("RAINNC", "rainnc", "rain_acc"),
    "RAINSH": ("RAINSH", "rainsh"),
    "SWDOWN": ("SWDOWN", "swdown"),
    "GLW": ("GLW", "glw"),
    "PBLH": ("PBLH", "pblh"),
    "UST": ("UST", "ustar"),
    "HFX": ("HFX", "hfx"),
    "LH": ("LH", "lh"),
    "TSK": ("TSK", "tsk", "t_skin"),
    "HGT": ("HGT", "hgt", "terrain_height"),
    "LANDMASK": ("LANDMASK", "landmask"),
    "LU_INDEX": ("LU_INDEX", "lu_index", "ivgtyp"),
    "XLAT": ("XLAT", "xlat", "lat"),
    "XLONG": ("XLONG", "xlong", "lon"),
    "XLAT_U": ("XLAT_U", "xlat_u", "lat_u"),
    "XLONG_U": ("XLONG_U", "xlong_u", "lon_u"),
    "XLAT_V": ("XLAT_V", "xlat_v", "lat_v"),
    "XLONG_V": ("XLONG_V", "xlong_v", "lon_v"),
}


def _requested_wrfout_source_values(
    *,
    state: Any,
    grid: Any,
    diagnostics: Mapping[str, Any] | None,
    land_state: Any | None,
    requested_names: frozenset[str],
) -> list[Any]:
    del land_state
    values: list[Any] = []

    def add(value: Any) -> None:
        if value is not None:
            values.append(value)

    def add_aliases(source: Any, aliases: tuple[str, ...]) -> None:
        for alias in aliases:
            add(_lookup_raw(source, alias, None))

    names = _expand_requested_source_names(requested_names)
    for name in names:
        aliases = _STATE_SOURCE_ALIASES.get(name, (name, name.lower()))
        add_aliases(state, aliases)
        add_aliases(diagnostics, aliases)

    # Total/base aliases used by _perturbation_base_pair fallbacks.
    if names & {"P", "PB", "PSFC", "T2", "TSK", "HFX", "LH"}:
        add_aliases(state, ("p_total", "p", "P_total"))
    if names & {"PH", "PHB", "PSFC", "HFX", "LH"}:
        add_aliases(state, ("ph_total", "ph", "PH_total"))
    if names & {"MU", "MUB", "PSFC"}:
        add_aliases(state, ("mu_total", "mu", "MU_total"))

    if names & {"HGT"}:
        add_aliases(grid, ("HGT", "hgt", "terrain_height"))
    if names & {"XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V"}:
        for coord in ("XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V"):
            add_aliases(diagnostics, _STATE_SOURCE_ALIASES[coord])
            add_aliases(state, _STATE_SOURCE_ALIASES[coord])

    if names & {"ZNU", "ZNW"}:
        add(_lookup_raw(grid, "eta_levels", None))

    metrics = _lookup_raw(grid, "metrics", None)
    metric_attrs = {
        "MAPFAC_M": ("msftx",),
        "MAPFAC_U": ("msfux",),
        "MAPFAC_V": ("msfvx",),
        "SINALPHA": ("sina",),
        "COSALPHA": ("cosa",),
        "F": ("f",),
        "E": ("e",),
    }
    if "PSFC" in names:
        metric_attrs.setdefault("PSFC", ())
        for attr in ("c1h", "c2h", "dnw", "p_top"):
            add(_lookup_raw(metrics, attr, None))
    for name, attrs in metric_attrs.items():
        if name in names:
            for attr in attrs:
                add(_lookup_raw(metrics, attr, None))
    return values


def _expand_requested_source_names(requested_names: frozenset[str]) -> set[str]:
    names = set(requested_names)
    if names & {"T", "THM", "T2", "TSK", "HFX", "LH"}:
        names.update({"T", "QVAPOR", "P", "PB"})
    if names & {"PSFC"}:
        names.update({"P", "PB", "PH", "PHB", "MU", "MUB", "QVAPOR", "QCLOUD", "QICE", "QSNOW", "QGRAUP"})
    if names & {"CLDFRA"}:
        names.update({"QCLOUD", "QICE", "QRAIN"})
    if names & {"U10", "HFX", "LH"}:
        names.add("U")
    if names & {"V10", "HFX", "LH"}:
        names.add("V")
    if names & {"HFX", "LH"}:
        names.update({"HGT", "LANDMASK", "TSK", "Q2", "UST"})
    if names & {"LU_INDEX"}:
        names.add("LANDMASK")
    if names & {"RAINNC"}:
        names.update({"QSNOW", "QGRAUP"})
    if names & {"OLR"}:
        names.add("LWUPT")
    return names


def prepare_wrfout_payload(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    path: str | Path,
    *,
    valid_time: datetime | date | str,
    lead_hours: float,
    run_start: datetime | date | str,
    diagnostics: Mapping[str, Any] | None = None,
    land_state: Any | None = None,
    variable_subset: tuple[str, ...] | frozenset[str] | None = None,
    include_mandatory_coords: bool = False,
    full_variable_set: bool = False,
) -> PreparedWrfout:
    """Materialize all wrfout fields to host numpy (the device->host boundary).

    This is the part of :func:`write_wrfout_netcdf` that touches the live model
    state / device arrays. Calling it eagerly (before the GPU advances the next
    hour) lets the subsequent NetCDF write run on a background thread. ``grid`` and
    ``namelist`` are kept by reference for the global-attribute scalars only --
    those reads are pure host metadata (projection floats / dimensions), never
    device arrays, so they are safe to read on the writer thread.
    """

    target = Path(path)
    run_start_dt = _coerce_datetime(run_start)
    valid_dt = _coerce_datetime(valid_time)
    nx, ny, nz = _grid_extent(grid)
    dimensions = _dimension_sizes(nx=nx, ny=ny, nz=nz, namelist=namelist)
    requested_names = _prepare_requested_names(
        variable_subset,
        include_mandatory_coords=include_mandatory_coords,
    )
    # The single device->host boundary: materialize all reachable device leaves in
    # one jax.device_get call, then let the existing field builder coerce host
    # arrays in its historical order.
    with _batched_wrfout_device_get(
        state=state,
        grid=grid,
        diagnostics=diagnostics,
        land_state=land_state,
        requested_names=requested_names,
    ):
        if requested_names is None:
            fields = _build_output_fields(
                state, grid, namelist, dimensions,
                diagnostics=diagnostics, land_state=land_state,
                run_start=run_start_dt, lead_hours=float(lead_hours),
            )
        else:
            fields = _build_subset_output_fields(
                state, grid, namelist, dimensions,
                requested_names=requested_names,
                diagnostics=diagnostics, land_state=land_state,
                run_start=run_start_dt, lead_hours=float(lead_hours),
            )
        if full_variable_set:
            _add_full_wrfout_fields(
                fields,
                state=state,
                grid=grid,
                namelist=namelist,
                dimensions=dimensions,
                diagnostics=diagnostics,
                land_state=land_state,
                run_start=run_start_dt,
                lead_hours=float(lead_hours),
                requested_names=requested_names,
            )
    return PreparedWrfout(
        target=target,
        dimensions=dimensions,
        fields=fields,
        run_start_dt=run_start_dt,
        valid_dt=valid_dt,
        lead_hours=float(lead_hours),
        grid=grid,
        namelist=namelist,
        full_variable_set=bool(full_variable_set),
    )


def write_prepared_wrfout(
    prepared: PreparedWrfout,
    *,
    variable_subset: tuple[str, ...] | frozenset[str] | None = None,
    target_override: Path | None = None,
    include_mandatory_coords: bool = False,
    compress: bool = False,
) -> Path:
    """Write a :class:`PreparedWrfout` to NetCDF. Pure host work; thread-safe.

    Contains NO device-array access, so it is safe to run on a background writer
    thread while the GPU advances. The bytes are identical to the synchronous
    :func:`write_wrfout_netcdf` path.

    ``variable_subset`` optionally restricts the emitted variables to the named
    subset -- a stream-generic hook for a secondary WRF ``auxhist`` history stream
    (e.g. a surface-only set). When ``None`` (the default) EVERY prepared field is
    written exactly as before, so the main wrfout stream is byte-for-byte
    unchanged. The ``Times``/``XTIME`` time coordinates and the global attributes
    are ALWAYS written regardless of the subset (a stream-valid WRF history frame
    always carries its time stamp -- matching how WRF stamps every auxhist frame).
    A name in ``variable_subset`` that is absent from the prepared payload is
    simply skipped (the file never fabricates a field), and the canonical
    operational write order is preserved.

    ``include_mandatory_coords`` (#122 training output): additionally force-emit the
    geometry/eta/staggered-lat-lon coordinates in :data:`MANDATORY_WRFOUT_COORDINATES`
    so a subset frame is a self-contained, stream-valid WRF file. OFF by default so
    the existing auxhist surface stream keeps its exact prior variable set (a coord
    still absent from the payload is skipped -- never fabricated).

    ``compress`` (#122 training output): apply lossless NetCDF4 zlib compression to
    the written variables (shrinking the ~10 GB/day training target). OFF by default
    so every existing caller's on-disk bytes are unchanged; compression is lossless
    so values read back are bit-identical regardless.

    ``target_override`` writes the same host payload to a different path without a
    second device->host pull -- used by the auxhist stream, which reuses the main
    stream's already-materialized :class:`PreparedWrfout`.
    """

    target = Path(target_override) if target_override is not None else prepared.target
    target.parent.mkdir(parents=True, exist_ok=True)
    subset = None if variable_subset is None else frozenset(variable_subset)
    if subset is not None and include_mandatory_coords:
        # Self-contained training frame: also emit the coordinate/dimension vars (a
        # coord still absent from the payload is skipped below, never fabricated).
        subset = subset | MANDATORY_WRFOUT_COORDINATES
    # Lossless NetCDF4 zlib compression, opt-in (#122 training stream). Default OFF
    # keeps every existing caller's on-disk bytes unchanged.
    compression = _SUBSET_STREAM_COMPRESSION if compress else None
    dimensions = prepared.dimensions
    with Dataset(target, "w", format="NETCDF4") as dataset:
        _create_dimensions(dataset, dimensions)
        _write_global_attrs(
            dataset, prepared.grid, prepared.namelist, dimensions,
            prepared.run_start_dt, prepared.valid_dt,
        )
        _write_times(dataset, prepared.valid_dt)
        # Write in the canonical operational order, but emit ONLY the fields that
        # were actually prepared. Optional sources (operational diagnostics, the
        # Noah-MP land carry) self-gate: an absent source leaves its fields out of
        # ``prepared.fields`` so the file never carries a fabricated quantity.
        # ``subset`` (when set) further restricts to a stream's requested vars.
        if prepared.full_variable_set:
            write_order = FULL_WRFOUT_VARIABLES
        else:
            _write_xtime(dataset, prepared.run_start_dt, prepared.lead_hours)
            write_order = OPERATIONAL_WRFOUT_VARIABLES
        for name in write_order:
            if name == "Times":
                continue
            if name == "XTIME":
                if prepared.full_variable_set:
                    _write_xtime(dataset, prepared.run_start_dt, prepared.lead_hours)
                continue
            if name not in prepared.fields:
                continue
            if subset is not None and name not in subset:
                continue
            spec = WRFOUT_VARIABLE_SPECS[name]
            _write_float_variable(dataset, spec, prepared.fields[name], dimensions, compression)
    return target


def _dimension_sizes(*, nx: int, ny: int, nz: int, namelist: Mapping[str, Any] | Any | None) -> dict[str, int | None]:
    soil_layers = int(_lookup(namelist, "soil_layers_stag", DEFAULT_SOIL_LAYERS))
    snow_layers = int(_lookup(namelist, "snow_layers_stag", DEFAULT_SNOW_LAYERS))
    snso_layers = int(_lookup(namelist, "snso_layers_stag", DEFAULT_SNSO_LAYERS))
    seed_dim = int(_lookup(namelist, "seed_dim_stag", DEFAULT_SEED_DIM))
    return {
        "Time": None,
        "DateStrLen": DATE_STR_LEN,
        "west_east": int(nx),
        "west_east_stag": int(nx) + 1,
        "south_north": int(ny),
        "south_north_stag": int(ny) + 1,
        "bottom_top": int(nz),
        "bottom_top_stag": int(nz) + 1,
        "soil_layers_stag": soil_layers,
        "snow_layers_stag": snow_layers,
        "snso_layers_stag": snso_layers,
        "seed_dim_stag": seed_dim,
    }


def _create_dimensions(dataset: Dataset, dimensions: Mapping[str, int | None]) -> None:
    for name in (
        "Time",
        "DateStrLen",
        "west_east",
        "west_east_stag",
        "south_north",
        "south_north_stag",
        "bottom_top",
        "bottom_top_stag",
        "soil_layers_stag",
        "snow_layers_stag",
        "snso_layers_stag",
        "seed_dim_stag",
    ):
        dataset.createDimension(name, dimensions[name])


def _write_times(dataset: Dataset, valid_time: datetime) -> None:
    times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
    encoded = _wrf_time_string(valid_time).encode("ascii")
    times[0, :] = np.frombuffer(encoded, dtype="S1")


def _write_xtime(dataset: Dataset, run_start: datetime, lead_hours: float) -> None:
    spec = WRFOUT_VARIABLE_SPECS["XTIME"]
    xtime = dataset.createVariable("XTIME", spec.dtype, spec.dimensions)
    xtime[:] = np.asarray([float(lead_hours) * 60.0], dtype=np.float32)
    _set_variable_attrs(xtime, spec)
    time_label = run_start.strftime("%Y-%m-%d %H:%M:%S")
    xtime.description = f"minutes since {time_label}"
    xtime.units = f"minutes since {time_label}"


def _write_float_variable(
    dataset: Dataset,
    spec: WrfoutVariableSpec,
    data: np.ndarray,
    dimensions: Mapping[str, int | None],
    compression: Mapping[str, object] | None = None,
) -> None:
    expected_shape = _shape_for_dimensions(spec.dimensions, dimensions)
    array = _coerce_array(spec.name, data, expected_shape, dtype=_numpy_dtype_for_spec(spec))
    # ``compression`` (zlib/complevel) is passed ONLY for subset streams; the
    # default full-output path leaves it None so the uncompressed bytes are
    # unchanged for existing callers. Compression is lossless, so values read back
    # are bit-identical regardless.
    kwargs = dict(compression) if compression else {}
    variable = dataset.createVariable(spec.name, spec.dtype, spec.dimensions, **kwargs)
    _set_variable_attrs(variable, spec)
    variable[0, ...] = array


def _set_variable_attrs(variable: Any, spec: WrfoutVariableSpec) -> None:
    # WRF tags real fields FieldType=104 and integer fields FieldType=106
    # (module_io_int / Registry). Derive it from the spec dtype so integer
    # diagnostics (ISNOW, the stochastic seed arrays) carry the correct code.
    variable.FieldType = np.int32(106 if str(spec.dtype).startswith("i") else 104)
    variable.MemoryOrder = spec.memory_order
    variable.description = spec.description
    variable.units = spec.units
    variable.stagger = spec.stagger
    if spec.coordinates is not None:
        variable.coordinates = spec.coordinates


def _write_global_attrs(
    dataset: Dataset,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    dimensions: Mapping[str, int | None],
    run_start: datetime,
    valid_time: datetime,
) -> None:
    projection = _lookup(grid, "projection")
    lat_0 = float(_lookup(projection, "lat_0", _lookup(namelist, "cen_lat", 0.0)))
    lon_0 = float(_lookup(projection, "lon_0", _lookup(namelist, "cen_lon", 0.0)))
    dx_m = float(_lookup(projection, "dx_m", _lookup(namelist, "dx", 0.0)))
    dy_m = float(_lookup(projection, "dy_m", _lookup(namelist, "dy", dx_m)))
    kind = str(_lookup(projection, "kind", "lambert")).lower()
    map_proj = {"lambert": 1, "polar": 2, "mercator": 3}.get(kind, 1)

    attrs: dict[str, Any] = {
        "TITLE": str(_lookup(namelist, "title", " OUTPUT FROM GPUWRF WRF-COMPATIBLE NETCDF WRITER")),
        "START_DATE": _wrf_time_string(run_start),
        "SIMULATION_START_DATE": _wrf_time_string(run_start),
        "WEST-EAST_GRID_DIMENSION": np.int32(int(dimensions["west_east_stag"])),
        "SOUTH-NORTH_GRID_DIMENSION": np.int32(int(dimensions["south_north_stag"])),
        "BOTTOM-TOP_GRID_DIMENSION": np.int32(int(dimensions["bottom_top_stag"])),
        "DX": np.float32(dx_m),
        "DY": np.float32(dy_m),
        "GRIDTYPE": "C",
        "MAP_PROJ": np.int32(map_proj),
        "CEN_LAT": np.float32(_lookup(namelist, "cen_lat", lat_0)),
        "CEN_LON": np.float32(_lookup(namelist, "cen_lon", lon_0)),
        "TRUELAT1": np.float32(_lookup(namelist, "truelat1", lat_0)),
        "TRUELAT2": np.float32(_lookup(namelist, "truelat2", lat_0)),
        "MOAD_CEN_LAT": np.float32(_lookup(namelist, "moad_cen_lat", lat_0)),
        "STAND_LON": np.float32(_lookup(namelist, "stand_lon", lon_0)),
        "GMT": np.float32(run_start.hour + run_start.minute / 60.0),
        "JULYR": np.int32(valid_time.year),
        "JULDAY": np.int32(valid_time.timetuple().tm_yday),
        "ISWATER": np.int32(_lookup(namelist, "iswater", 17)),
        "ISLAKE": np.int32(_lookup(namelist, "islake", 21)),
        "ISICE": np.int32(_lookup(namelist, "isice", 15)),
        "ISURBAN": np.int32(_lookup(namelist, "isurban", 13)),
        "ISOILWATER": np.int32(_lookup(namelist, "isoilwater", 14)),
        "Conventions": "WRF-ARW",
        "history": "Created by gpuwrf.io.wrfout_writer.write_wrfout_netcdf",
    }
    for name, value in attrs.items():
        dataset.setncattr(name, value)


def _build_subset_output_fields(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    dimensions: Mapping[str, int | None],
    *,
    requested_names: frozenset[str],
    diagnostics: Mapping[str, Any] | None = None,
    land_state: Any | None = None,
    run_start: datetime | None = None,
    lead_hours: float = 0.0,
) -> dict[str, np.ndarray]:
    del land_state, run_start, lead_hours
    shape_xy = _shape_for_dimensions(XY, dimensions)
    shape_xyz = _shape_for_dimensions(XYZ, dimensions)
    shape_u = _shape_for_dimensions(U_XYZ, dimensions)
    shape_v = _shape_for_dimensions(V_XYZ, dimensions)
    shape_w = _shape_for_dimensions(W_XYZ, dimensions)
    shape_u_xy = _shape_for_dimensions(U_XY, dimensions)
    shape_v_xy = _shape_for_dimensions(V_XY, dimensions)
    shape_z = _shape_for_dimensions(Z_XYZ, dimensions)

    fields: dict[str, np.ndarray] = {}
    requested = set(requested_names)

    def want(name: str) -> bool:
        return name in requested

    need_u = bool(requested & {"U", "U10", "HFX", "LH"})
    need_v = bool(requested & {"V", "V10", "HFX", "LH"})
    need_w = want("W")
    need_theta = bool(requested & {"T", "THM", "T2", "TSK", "HFX", "LH"})
    need_qv = bool(requested & {"T", "THM", "QVAPOR", "Q2", "T2", "TSK", "HFX", "LH", "PSFC"})
    need_qc = bool(requested & {"QCLOUD", "CLDFRA", "PSFC"})
    need_qi = bool(requested & {"QICE", "CLDFRA", "PSFC"})
    need_qr = bool(requested & {"QRAIN", "CLDFRA"})
    need_pressure = bool(requested & {"P", "PB", "T2", "TSK", "PSFC", "HFX", "LH"})
    need_geopotential = bool(requested & {"PH", "PHB", "PSFC", "HFX", "LH"})
    need_mu = want("PSFC")

    u = _field_array(state, ("U", "u"), shape_u) if need_u else None
    v = _field_array(state, ("V", "v"), shape_v) if need_v else None
    w = _field_array(state, ("W", "w"), shape_w) if need_w else None
    theta = (
        _field_array(state, ("theta", "THETA"), shape_xyz, default=300.0)
        if need_theta
        else None
    )
    qv = _field_array(state, ("QVAPOR", "qv", "qvapor"), shape_xyz) if need_qv else None
    qc = _field_array(state, ("QCLOUD", "qc", "qcloud"), shape_xyz) if need_qc else None
    qi = _field_array(state, ("QICE", "qi", "qice"), shape_xyz) if need_qi else None
    qr = _field_array(state, ("QRAIN", "qr", "qrain"), shape_xyz) if need_qr else None
    if theta is not None and qv is not None:
        theta_dry = theta / (1.0 + RVOVRD * np.maximum(qv, 0.0))
    else:
        theta_dry = theta

    p_pert = p_base = None
    if need_pressure:
        p_pert, p_base = _perturbation_base_pair(
            state,
            total_names=("p_total", "p", "P_total"),
            perturbation_names=("p_perturbation", "P"),
            base_names=("pb", "p_base", "PB"),
            shape=shape_xyz,
        )
    ph_pert = ph_base = None
    if need_geopotential:
        ph_pert, ph_base = _perturbation_base_pair(
            state,
            total_names=("ph_total", "ph", "PH_total"),
            perturbation_names=("ph_perturbation", "PH"),
            base_names=("phb", "ph_base", "PHB"),
            shape=shape_z,
        )
    mu_pert = mu_base = None
    if need_mu:
        mu_pert, mu_base = _perturbation_base_pair(
            state,
            total_names=("mu_total", "mu", "MU_total"),
            perturbation_names=("mu_perturbation", "MU"),
            base_names=("mub", "mu_base", "MUB"),
            shape=shape_xy,
        )

    if want("U") and u is not None:
        fields["U"] = u
    if want("V") and v is not None:
        fields["V"] = v
    if want("W") and w is not None:
        fields["W"] = w
    if want("T") and theta_dry is not None:
        fields["T"] = theta_dry - P0_THETA_OFFSET_K
    if want("THM") and theta is not None:
        fields["THM"] = theta - P0_THETA_OFFSET_K
    if want("QVAPOR") and qv is not None:
        fields["QVAPOR"] = qv
    if want("QCLOUD") and qc is not None:
        fields["QCLOUD"] = qc
    if want("QICE") and qi is not None:
        fields["QICE"] = qi
    if want("QRAIN") and qr is not None:
        fields["QRAIN"] = qr
    for wrf_name, state_names in (
        ("QSNOW", ("QSNOW", "qs", "qsnow")),
        ("QGRAUP", ("QGRAUP", "qg", "qgraup")),
        ("QKE", ("QKE", "qke")),
    ):
        if want(wrf_name):
            value = _optional_field_array(state, state_names, shape_xyz)
            if value is not None:
                fields[wrf_name] = value
    if want("CLDFRA"):
        cldfra_default = None
        if qc is not None and qi is not None and qr is not None:
            cldfra_default = np.where((qc + qi + qr) > 1.0e-8, 1.0, 0.0)
        fields["CLDFRA"] = _field_array(
            state,
            ("CLDFRA", "cldfra", "cloud_fraction"),
            shape_xyz,
            default=0.0 if cldfra_default is None else cldfra_default,
        )

    if want("P") and p_pert is not None:
        fields["P"] = p_pert
    if want("PB") and p_base is not None:
        fields["PB"] = p_base
    if want("PH") and ph_pert is not None:
        fields["PH"] = ph_pert
    if want("PHB") and ph_base is not None:
        fields["PHB"] = ph_base
    if want("MU") and mu_pert is not None:
        fields["MU"] = mu_pert
    if want("MUB") and mu_base is not None:
        fields["MUB"] = mu_base

    if requested & {"XLAT", "XLONG"}:
        xlat, xlong = _latlon_fields(state, grid, namelist, shape_xy, diagnostics=diagnostics)
        if want("XLAT"):
            fields["XLAT"] = xlat
        if want("XLONG"):
            fields["XLONG"] = xlong
    if requested & {"XLAT_U", "XLONG_U"}:
        xlat_u, xlong_u = _latlon_fields(
            state, grid, namelist, shape_u_xy, suffix="_u", diagnostics=diagnostics
        )
        if want("XLAT_U"):
            fields["XLAT_U"] = xlat_u
        if want("XLONG_U"):
            fields["XLONG_U"] = xlong_u
    if requested & {"XLAT_V", "XLONG_V"}:
        xlat_v, xlong_v = _latlon_fields(
            state, grid, namelist, shape_v_xy, suffix="_v", diagnostics=diagnostics
        )
        if want("XLAT_V"):
            fields["XLAT_V"] = xlat_v
        if want("XLONG_V"):
            fields["XLONG_V"] = xlong_v

    terrain = None
    if requested & {"HGT", "HFX", "LH"}:
        terrain = _grid_or_state_array(state, grid, ("HGT", "hgt", "terrain_height"), shape_xy)
        if want("HGT"):
            fields["HGT"] = terrain
    landmask = None
    if requested & {"LANDMASK", "LU_INDEX", "HFX", "LH"}:
        landmask = _landmask(state, shape_xy)
        if want("LANDMASK"):
            fields["LANDMASK"] = landmask
    if want("LU_INDEX"):
        default_lu = 17.0 if landmask is None else np.where(landmask > 0.5, 2.0, 17.0)
        fields["LU_INDEX"] = _field_array(
            state, ("LU_INDEX", "lu_index", "ivgtyp"), shape_xy, default=default_lu
        )

    if want("U10") and u is not None:
        fields["U10"] = _field_array(state, ("U10", "u10"), shape_xy, default=_unstagger_x(u)[0])
    if want("V10") and v is not None:
        fields["V10"] = _field_array(state, ("V10", "v10"), shape_xy, default=_unstagger_y(v)[0])
    if want("Q2") and qv is not None:
        fields["Q2"] = _field_array(state, ("Q2", "q2"), shape_xy, default=qv[0])
    if want("T2") and theta_dry is not None and p_pert is not None and p_base is not None:
        t_default = theta_dry[0] * (np.maximum(p_pert[0] + p_base[0], 1.0) / P0_PA) ** R_D_OVER_CP
        fields["T2"] = _field_array(state, ("T2", "t2"), shape_xy, default=t_default)
    if want("TSK") and theta_dry is not None and p_pert is not None and p_base is not None:
        tsk_default = theta_dry[0] * (np.maximum(p_pert[0] + p_base[0], 1.0) / P0_PA) ** R_D_OVER_CP
        fields["TSK"] = _field_array(state, ("TSK", "tsk", "t_skin"), shape_xy, default=tsk_default)
    if want("PSFC"):
        psfc_default = 0.0
        if p_pert is not None and p_base is not None and ph_pert is not None and ph_base is not None:
            p_total = p_pert + p_base
            phi = ph_pert + ph_base
            phi0 = phi[0]
            phi1 = 0.5 * (phi[0] + phi[1])
            phi2 = 0.5 * (phi[1] + phi[2])
            weight = (phi0 - phi2) / (phi1 - phi2)
            psfc_default = weight * p_total[0] + (1.0 - weight) * p_total[1]
        fields["PSFC"] = _field_array(state, ("PSFC", "psfc"), shape_xy, default=psfc_default)

    for name, aliases in (
        ("RAINC", ("RAINC", "rainc", "rainc_acc")),
        ("RAINSH", ("RAINSH", "rainsh")),
        ("SWDOWN", ("SWDOWN", "swdown")),
        ("GLW", ("GLW", "glw")),
        ("PBLH", ("PBLH", "pblh")),
        ("UST", ("UST", "ustar")),
        ("HFX", ("HFX", "hfx")),
        ("LH", ("LH", "lh")),
    ):
        if want(name):
            fields[name] = _field_array(state, aliases, shape_xy)
    if want("RAINNC"):
        rain_acc = _optional_field_array(state, ("rain_acc", "RAINNC"), shape_xy)
        snow_acc = _optional_field_array(state, ("snow_acc", "SNOWNC"), shape_xy)
        ice_acc = _optional_field_array(state, ("ice_acc",), shape_xy)
        graupel_acc = _optional_field_array(state, ("graupel_acc", "GRAUPELNC"), shape_xy)
        hail_acc = _optional_field_array(state, ("hail_acc", "HAILNC"), shape_xy)
        if any(a is not None for a in (snow_acc, graupel_acc, ice_acc, hail_acc)):
            def _zero_xy(a: np.ndarray | None) -> np.ndarray:
                return np.asarray(a, dtype=np.float64) if a is not None else np.zeros(shape_xy, dtype=np.float64)

            rainnc_total = (
                _zero_xy(rain_acc)
                + _zero_xy(snow_acc)
                + _zero_xy(graupel_acc)
                + _zero_xy(ice_acc)
                + _zero_xy(hail_acc)
            )
            fields["RAINNC"] = _coerce_array("RAINNC", rainnc_total, shape_xy)
        else:
            fields["RAINNC"] = _field_array(state, ("RAINNC", "rainnc", "rain_acc"), shape_xy)

    if requested & {"ZNU", "ZNW", "MAPFAC_M", "SINALPHA", "COSALPHA"}:
        _add_subset_grid_fields(fields, grid, dimensions, requested_names=requested_names, shape_xy=shape_xy)

    _apply_subset_diagnostics(fields, diagnostics, requested_names=requested_names, shape_xy=shape_xy)

    ordered: dict[str, np.ndarray] = {}
    for name in OPERATIONAL_WRFOUT_VARIABLES:
        if name in fields and name in requested:
            ordered[name] = _materialized_dtype(name, fields[name])
    for name, value in fields.items():
        if name in requested and name not in ordered:
            ordered[name] = _materialized_dtype(name, value)
    return ordered


def _add_subset_grid_fields(
    fields: dict[str, np.ndarray],
    grid: Any,
    dimensions: Mapping[str, int | None],
    *,
    requested_names: frozenset[str],
    shape_xy: tuple[int, int],
) -> None:
    nz = int(dimensions["bottom_top"])
    eta = _lookup(grid, "eta_levels")
    if eta is not None and ({"ZNU", "ZNW"} & requested_names):
        eta_host = np.asarray(eta, dtype=np.float64)
        if eta_host.shape == (nz + 1,):
            if "ZNW" in requested_names:
                fields["ZNW"] = _coerce_array("ZNW", eta_host, (nz + 1,))
            if "ZNU" in requested_names:
                fields["ZNU"] = _coerce_array("ZNU", 0.5 * (eta_host[:-1] + eta_host[1:]), (nz,))

    metrics = _lookup(grid, "metrics")
    if metrics is None:
        return
    for wrf_name, attr in (
        ("MAPFAC_M", "msftx"),
        ("SINALPHA", "sina"),
        ("COSALPHA", "cosa"),
    ):
        if wrf_name not in requested_names:
            continue
        value = _lookup(metrics, attr)
        if value is not None:
            fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), shape_xy)


def _apply_subset_diagnostics(
    fields: dict[str, np.ndarray],
    diagnostics: Mapping[str, Any] | None,
    *,
    requested_names: frozenset[str],
    shape_xy: tuple[int, int],
) -> None:
    if diagnostics is None:
        return
    diagnostic_surface_fields = {
        "T2", "U10", "V10", "Q2", "PSFC", "SWDOWN", "GLW", "PBLH", "UST",
        "HFX", "LH", "TSK", "QFX", "GRDFLX",
        "SWDNB", "SWUPB", "LWDNB", "LWUPB",
        "SWDNT", "SWUPT", "LWDNT", "LWUPT", "SWNORM",
        "SWDNBC", "SWUPBC", "LWDNBC", "LWUPBC",
        "SWDNTC", "SWUPTC", "LWDNTC", "LWUPTC",
        "OLR",
    }
    for name in requested_names & diagnostic_surface_fields:
        value = diagnostics.get(name)
        if value is not None:
            fields[name] = _coerce_array(name, value, shape_xy)
    if "OLR" in requested_names and "OLR" not in fields:
        lwupt = diagnostics.get("LWUPT")
        if lwupt is not None:
            fields["OLR"] = _coerce_array("OLR", lwupt, shape_xy)


def _build_output_fields(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    dimensions: Mapping[str, int | None],
    *,
    diagnostics: Mapping[str, Any] | None = None,
    land_state: Any | None = None,
    run_start: datetime | None = None,
    lead_hours: float = 0.0,
) -> dict[str, np.ndarray]:
    shape_xy = _shape_for_dimensions(XY, dimensions)
    shape_xyz = _shape_for_dimensions(XYZ, dimensions)
    shape_u = _shape_for_dimensions(U_XYZ, dimensions)
    shape_v = _shape_for_dimensions(V_XYZ, dimensions)
    shape_w = _shape_for_dimensions(W_XYZ, dimensions)
    shape_u_xy = _shape_for_dimensions(U_XY, dimensions)
    shape_v_xy = _shape_for_dimensions(V_XY, dimensions)
    shape_z = _shape_for_dimensions(Z_XYZ, dimensions)
    shape_soil = _shape_for_dimensions(SOIL, dimensions)
    shape_snow = _shape_for_dimensions(SNOW, dimensions)
    shape_snso = _shape_for_dimensions(SNSO, dimensions)
    shape_seed = _shape_for_dimensions(SEED, dimensions)
    shape_mapu = _shape_for_dimensions(MAPFAC_U_XY, dimensions)
    shape_mapv = _shape_for_dimensions(MAPFAC_V_XY, dimensions)

    u = _field_array(state, ("U", "u"), shape_u)
    v = _field_array(state, ("V", "v"), shape_v)
    w = _field_array(state, ("W", "w"), shape_w)
    theta = _field_array(state, ("theta", "THETA"), shape_xyz, default=300.0)
    qv = _field_array(state, ("QVAPOR", "qv", "qvapor"), shape_xyz)
    # State.theta is moist theta_m (use_theta_m=1); WRF-compatible ``T`` and the
    # temperature-derived diagnostics below need the DRY theta view (identical
    # when qv = 0, e.g. synthetic/test states).
    theta_dry = theta / (1.0 + RVOVRD * np.maximum(qv, 0.0))
    qc = _field_array(state, ("QCLOUD", "qc", "qcloud"), shape_xyz)
    qi = _field_array(state, ("QICE", "qi", "qice"), shape_xyz)
    qr = _field_array(state, ("QRAIN", "qr", "qrain"), shape_xyz)

    p_pert, p_base = _perturbation_base_pair(
        state,
        total_names=("p_total", "p", "P_total"),
        perturbation_names=("p_perturbation", "P"),
        base_names=("pb", "p_base", "PB"),
        shape=shape_xyz,
    )
    ph_pert, ph_base = _perturbation_base_pair(
        state,
        total_names=("ph_total", "ph", "PH_total"),
        perturbation_names=("ph_perturbation", "PH"),
        base_names=("phb", "ph_base", "PHB"),
        shape=shape_z,
    )
    mu_pert, mu_base = _perturbation_base_pair(
        state,
        total_names=("mu_total", "mu", "MU_total"),
        perturbation_names=("mu_perturbation", "MU"),
        base_names=("mub", "mu_base", "MUB"),
        shape=shape_xy,
    )
    # WRF-faithful surface-pressure fallback (used only when the operational M9
    # PSFC diagnostic is absent from ``state``). WRF's runtime PSFC is the MOIST
    # hydrostatic surface pressure, NOT an extrapolation of the nonhydrostatic
    # total pressure: PSFC = p8w(kts) (module_surface_driver.F:1988) where the
    # surface driver's p8w argument is grid%p_hyd_w
    # (module_first_rk_step_part1.F:1400), built in phy_prep
    # (module_big_step_utilities_em.F:4946-4958) as
    #   p_hyd_w(kte) = p_top
    #   p_hyd_w(k)   = p_hyd_w(k+1) - (1+qtot)*(c1h(k)*MUT+c2h(k))*dnw(k)
    # with qtot summed over ALL moist species and MUT = mu+mub the full dry
    # column mass. When the hybrid-eta metrics are resident on ``grid.metrics``
    # we evaluate that integral exactly (CPU-truth residual <= 0.18 Pa RMSE,
    # proofs/v014/psfc_moist_pressure_state_closure.*). Without metrics
    # (synthetic/test states) fall back to the height extrapolation of total
    # pressure, which tracks WRF only to ~14 Pa on a moist-consistent state and
    # misses the vapor column load (~200-230 Pa) on a dry-balanced one.
    _psfc_default = None
    _psfc_metrics = _lookup(grid, "metrics")
    if _psfc_metrics is not None:
        _c1h = _lookup(_psfc_metrics, "c1h")
        _c2h = _lookup(_psfc_metrics, "c2h")
        _dnw = _lookup(_psfc_metrics, "dnw")
        _p_top = _lookup(_psfc_metrics, "p_top")
        if all(v is not None for v in (_c1h, _c2h, _dnw, _p_top)):
            _c1h = np.asarray(_c1h, dtype=np.float64)
            _c2h = np.asarray(_c2h, dtype=np.float64)
            _dnw = np.asarray(_dnw, dtype=np.float64)
            _nz = qv.shape[0]
            if _c1h.shape == (_nz,) and _c2h.shape == (_nz,) and _dnw.shape == (_nz,):
                _qs = _optional_field_array(state, ("QSNOW", "qs", "qsnow"), shape_xyz)
                _qg = _optional_field_array(state, ("QGRAUP", "qg", "qgraup"), shape_xyz)
                _qtot = qv + qc + qr + qi
                if _qs is not None:
                    _qtot = _qtot + _qs
                if _qg is not None:
                    _qtot = _qtot + _qg
                _mut = mu_pert + mu_base
                _dp_dry = (
                    _c1h[:, None, None] * _mut[None, :, :] + _c2h[:, None, None]
                ) * (-_dnw[:, None, None])
                _terms = np.asarray((1.0 + _qtot) * _dp_dry, dtype=np.float64)
                # v0.20 fp32 INTEGRATION bit-identity fix (secondary source): the S4
                # merge unconditionally switched this PSFC column integral to Kahan
                # compensated summation, which rounds differently from the historical
                # plain .sum(axis=0) and broke fp64_default PSFC bit-identity. Kahan
                # is only needed for the perturbation-authoritative fp32 mode (its
                # accumulator is fp32-sensitive); gate on the perturbation storage
                # dtype so fp64_default re-emits the exact pre-S4 plain sum.
                _mixed_fp32_psfc = (
                    np.asarray(mu_pert).dtype == np.float32
                    or np.asarray(p_pert).dtype == np.float32
                )
                if _mixed_fp32_psfc:
                    _total = np.zeros_like(_terms[0], dtype=np.float64)
                    _comp = np.zeros_like(_total, dtype=np.float64)
                    for _k in range(_terms.shape[0]):
                        _y = _terms[_k] - _comp
                        _t = _total + _y
                        _comp = (_t - _total) - _y
                        _total = _t
                    _psfc_default = float(np.asarray(_p_top).reshape(-1)[0]) + (_total - _comp)
                else:
                    # fp64_default: historical plain vertical sum (byte-identical).
                    _psfc_default = (
                        float(np.asarray(_p_top).reshape(-1)[0]) + _terms.sum(axis=0)
                    )
    if _psfc_default is None:
        _p_total = p_pert + p_base
        _phi = ph_pert + ph_base
        _phi0 = _phi[0]
        _phi1 = 0.5 * (_phi[0] + _phi[1])
        _phi2 = 0.5 * (_phi[1] + _phi[2])
        _w1 = (_phi0 - _phi2) / (_phi1 - _phi2)
        _psfc_default = _w1 * _p_total[0] + (1.0 - _w1) * _p_total[1]

    xlat, xlong = _latlon_fields(state, grid, namelist, shape_xy, diagnostics=diagnostics)
    xlat_u, xlong_u = _latlon_fields(
        state, grid, namelist, shape_u_xy, suffix="_u", diagnostics=diagnostics
    )
    xlat_v, xlong_v = _latlon_fields(
        state, grid, namelist, shape_v_xy, suffix="_v", diagnostics=diagnostics
    )

    terrain = _grid_or_state_array(state, grid, ("HGT", "hgt", "terrain_height"), shape_xy)
    landmask = _landmask(state, shape_xy)
    lu_index = _field_array(state, ("LU_INDEX", "lu_index", "ivgtyp"), shape_xy, default=np.where(landmask > 0.5, 2.0, 17.0))
    hfx = _optional_field_array(state, ("HFX", "hfx"), shape_xy)
    lh = _optional_field_array(state, ("LH", "lh"), shape_xy)
    if hfx is None or lh is None:
        surface_fluxes = _surface_flux_fallbacks(
            state=state,
            grid=grid,
            theta=theta_dry,
            qv=qv,
            p_total=p_pert + p_base,
            ph_total=ph_pert + ph_base,
            u=u,
            v=v,
            landmask=landmask,
            shape_xy=shape_xy,
        )
        if hfx is None:
            hfx = surface_fluxes["HFX"]
        if lh is None:
            lh = surface_fluxes["LH"]

    fields = {
        "XLAT": xlat,
        "XLONG": xlong,
        "XLAT_U": xlat_u,
        "XLONG_U": xlong_u,
        "XLAT_V": xlat_v,
        "XLONG_V": xlong_v,
        "HGT": terrain,
        "LANDMASK": landmask,
        "LU_INDEX": lu_index,
        "U": u,
        "V": v,
        "W": w,
        "T": theta_dry - P0_THETA_OFFSET_K,
        "THM": theta - P0_THETA_OFFSET_K,
        "QVAPOR": qv,
        "P": p_pert,
        "PB": p_base,
        "PH": ph_pert,
        "PHB": ph_base,
        "MU": mu_pert,
        "MUB": mu_base,
        "QCLOUD": qc,
        "QICE": qi,
        "QRAIN": qr,
        "CLDFRA": _field_array(
            state,
            ("CLDFRA", "cldfra", "cloud_fraction"),
            shape_xyz,
            default=np.where((qc + qi + qr) > 1.0e-8, 1.0, 0.0),
        ),
        "U10": _field_array(state, ("U10", "u10"), shape_xy, default=_unstagger_x(u)[0]),
        "V10": _field_array(state, ("V10", "v10"), shape_xy, default=_unstagger_y(v)[0]),
        # T2/TSK fall back to lowest-level *actual* temperature (theta -> T via the
        # local Exner factor), NOT raw potential temperature: theta[0] over high
        # terrain (e.g. Teide ~3.4km, psfc~660hPa) reads ~+34K too warm when mislabeled
        # as a 2-m/skin temperature. Level-1 air T is a sane proxy when the real
        # surface diagnostic is absent. (The proper fix routes the operational
        # surface-layer T2/U10/V10 diagnostics into the writer state; see task.)
        "T2": _field_array(state, ("T2", "t2"), shape_xy, default=theta_dry[0] * (np.maximum(p_pert[0] + p_base[0], 1.0) / P0_PA) ** R_D_OVER_CP),
        "Q2": _field_array(state, ("Q2", "q2"), shape_xy, default=qv[0]),
        "PSFC": _field_array(state, ("PSFC", "psfc"), shape_xy, default=_psfc_default),
        "RAINC": _field_array(state, ("RAINC", "rainc", "rainc_acc"), shape_xy),
        "RAINNC": _field_array(state, ("RAINNC", "rainnc", "rain_acc"), shape_xy),
        "RAINSH": _field_array(state, ("RAINSH", "rainsh"), shape_xy),
        "SWDOWN": _field_array(state, ("SWDOWN", "swdown"), shape_xy),
        "GLW": _field_array(state, ("GLW", "glw"), shape_xy),
        "PBLH": _field_array(state, ("PBLH", "pblh"), shape_xy),
        "UST": _field_array(state, ("UST", "ustar"), shape_xy),
        "HFX": hfx,
        "LH": lh,
        "TSK": _field_array(state, ("TSK", "tsk", "t_skin"), shape_xy, default=theta_dry[0] * (np.maximum(p_pert[0] + p_base[0], 1.0) / P0_PA) ** R_D_OVER_CP),
    }

    # --- P0-5a (a) extra hydrometeors + number concentrations + MYNN TKE. ---
    # Source = prognostic State leaves (Thompson qs/qg/Ni/Nr, MYNN qke). These
    # leaves always exist in the operational State; when absent (synthetic/test
    # state) _optional_field_array returns None and the field is skipped, so a
    # reduced state never gets a fabricated hydrometeor.
    for wrf_name, state_names in (
        ("QSNOW", ("QSNOW", "qs", "qsnow")),
        ("QGRAUP", ("QGRAUP", "qg", "qgraup")),
        ("QNICE", ("QNICE", "Ni", "qnice")),
        ("QNRAIN", ("QNRAIN", "Nr", "qnrain")),
        ("QNSNOW", ("QNSNOW", "Ns", "qnsnow")),
        ("QNGRAUPEL", ("QNGRAUPEL", "Ng", "qngraupel")),
        ("QNCLOUD", ("QNCLOUD", "Nc", "qnc", "qnc_cloud")),
        ("QNCCN", ("QNCCN", "Nn", "qnn", "qccn")),
        # v0.17 ADR-032 graupel/hail substrate (optional; zero/absent in a
        # non-hail run, so a reduced/non-hail state never gets a fabricated
        # hail field).
        ("QHAIL", ("QHAIL", "qh", "qhail")),
        ("QNHAIL", ("QNHAIL", "Nh", "qnh", "qnhail")),
        ("QVGRAUPEL", ("QVGRAUPEL", "qvolg")),
        ("QVHAIL", ("QVHAIL", "qvolh")),
        # v0.16 aerosol-aware Thompson (mp=28) aerosol numbers (optional).
        ("QNWFA", ("QNWFA", "nwfa", "qnwfa")),
        ("QNIFA", ("QNIFA", "nifa", "qnifa")),
        ("QKE", ("QKE", "qke")),
    ):
        value = _optional_field_array(state, state_names, shape_xyz)
        if value is not None:
            fields[wrf_name] = value

    # --- P0-5a (c) accumulated precipitation partition. Source = the State
    # precip accumulators that coupling.physics_couplers advances each
    # microphysics step as DISJOINT channels (rain_acc=liquid, snow_acc, ice_acc,
    # graupel_acc). WRF's wrfout convention (module_mp_thompson.F:1298-1306) is:
    #   RAINNC    = rain + snow + graupel + ice   (TOTAL accumulated precip, all phases)
    #   SNOWNC    = snow + ice                     (frozen subset)
    #   GRAUPELNC = graupel
    # i.e. SNOWNC/GRAUPELNC are OVERLAPPING subsets of RAINNC, not extra channels.
    # The internal accumulators stay disjoint (conservation budget / SR / restart
    # rely on that); only the wrfout mapping folds them into WRF's total here.
    # Emitted only when present (microphysics active). ---
    rain_acc = _optional_field_array(state, ("rain_acc", "RAINNC"), shape_xy)
    snow_acc = _optional_field_array(state, ("snow_acc", "SNOWNC"), shape_xy)
    ice_acc = _optional_field_array(state, ("ice_acc",), shape_xy)
    graupel_acc = _optional_field_array(state, ("graupel_acc", "GRAUPELNC"), shape_xy)
    # v0.17 hail (WSM7/WDM7): HAILNC is a distinct accumulated channel folded into
    # the RAINNC all-phase total, mirroring graupel. Zero/absent in a non-hail run.
    hail_acc = _optional_field_array(state, ("hail_acc", "HAILNC"), shape_xy)
    if snow_acc is not None or ice_acc is not None:
        snowice = (snow_acc if snow_acc is not None else 0.0) + (
            ice_acc if ice_acc is not None else 0.0
        )
        fields["SNOWNC"] = _coerce_array("SNOWNC", snowice, shape_xy)
    if graupel_acc is not None:
        fields["GRAUPELNC"] = graupel_acc
    if hail_acc is not None:
        fields["HAILNC"] = hail_acc
    # WRF RAINNC is the all-phase total; overwrite the rain-only default written
    # above (RAINNC: rain_acc) when any frozen channel is present so SNOWNC never
    # exceeds RAINNC and the domain precip total matches CPU-WRF.
    if any(a is not None for a in (snow_acc, graupel_acc, ice_acc, hail_acc)):
        def _zero_xy(a: np.ndarray | None) -> np.ndarray:
            return np.asarray(a, dtype=np.float64) if a is not None else np.zeros(shape_xy, dtype=np.float64)

        rainnc_total = (
            _zero_xy(rain_acc)
            + _zero_xy(snow_acc)
            + _zero_xy(graupel_acc)
            + _zero_xy(ice_acc)
            + _zero_xy(hail_acc)
        )
        fields["RAINNC"] = _coerce_array("RAINNC", rainnc_total, shape_xy)

    # --- P0-5a (b) grid-static coordinate / map-factor / Coriolis fields. Source
    # = GridSpec.metrics + GridSpec.vertical (always present on a real GridSpec).
    # Skipped wholesale when the grid carries no metrics (dict/synthetic grid). ---
    _add_grid_coordinate_fields(
        fields, grid, dimensions,
        shape_xy=shape_xy, shape_mapu=shape_mapu, shape_mapv=shape_mapv,
        landmask=landmask,
    )

    # --- P0-5a (d) 2-m potential temperature TH2 = T2 * (P0/PSFC)^(Rd/cp). Always
    # derivable from the (possibly diagnostic-overridden) T2 + PSFC below; computed
    # after the diagnostics override so it tracks the operational 2-m fields. ---

    # --- P0-5a (e) prognostic Noah-MP soil/snow land columns + land diagnostics.
    # Source = the optional NoahMPLandState carry. Never fabricated. ---
    if land_state is not None:
        _add_land_soil_fields(
            fields,
            land_state,
            shape_soil=shape_soil,
            shape_snow=shape_snow,
            shape_snso=shape_snso,
            shape_xy=shape_xy,
        )

    # The set of surface-map names the diagnostics dict is allowed to write. Most
    # OVERRIDE a raw lowest-level fallback already in ``fields``; QFX/GRDFLX are
    # ADD-only operational fluxes (the writer has no physical fallback for them, so
    # they appear only when the operational coupler supplies them -- never
    # fabricated). All are mass-point (ny, nx).
    _DIAGNOSTIC_SURFACE_FIELDS = {
        "T2", "U10", "V10", "Q2", "PSFC", "SWDOWN", "GLW", "PBLH", "UST",
        "HFX", "LH", "TSK", "QFX", "GRDFLX",
        # --- B1 (v0.12.0) RRTMG up/down all-sky surface + TOA flux diagnostics ---
        # ADD-only (no physical fallback): appear only when the operational
        # radiation diagnostics supply them, never fabricated. SWDNB == SWDOWN in
        # the no-slope config; SWNORM is the slope-normal surface SW flux. OLR is
        # derived below from LWUPT (== WRF's TOA outgoing LW).
        "SWDNB", "SWUPB", "LWDNB", "LWUPB",
        "SWDNT", "SWUPT", "LWDNT", "LWUPT", "SWNORM",
        # --- v0.13.0 RRTMG clear-sky (cloud-free) flux diagnostics. ADD-only:
        # appear only when the radiation diagnostics supply the WRF clear-sky pass
        # outputs (``with_clear_sky=True``). ---
        "SWDNBC", "SWUPBC", "LWDNBC", "LWUPBC",
        "SWDNTC", "SWUPTC", "LWDNTC", "LWUPTC",
    }
    if diagnostics is not None:
        # WRF stochastic-perturbation restart seed arrays are 1-D integer state.
        # They are not active in the supported Canary suite, but if a caller carries
        # real seed state, write it faithfully instead of leaving the KI-3 dimension
        # unsupported.
        for name in STOCHASTIC_SEED_VARIABLES:
            value = diagnostics.get(name) if isinstance(diagnostics, Mapping) else None
            if value is not None:
                fields[name] = _coerce_array(name, value, shape_seed, dtype=np.int32)

        # Operational surface-layer diagnostics OVERRIDE the raw lowest-level
        # fallbacks for the surface map. These are physically diagnosed 2-m / 10-m /
        # skin / surface fields (mass-point ``(ny, nx)``), not raw level-1 values.
        for name, value in diagnostics.items():
            if value is None or name not in _DIAGNOSTIC_SURFACE_FIELDS:
                continue
            fields[name] = _coerce_array(name, value, shape_xy)

        # --- B1 (v0.12.0) OLR = WRF's TOA outgoing longwave == LWUPT (the
        # top-of-atmosphere upwelling LW flux). WRF carries OLR as a separate
        # history var but its value is identically the upward LW flux at the model
        # top; derived here from the diagnostic LWUPT so it is always consistent
        # and never an independent fabricated quantity. Emitted only when LWUPT is. ---
        if "LWUPT" in fields:
            fields["OLR"] = _coerce_array("OLR", np.asarray(fields["LWUPT"]), shape_xy)

    # --- P0-5a (d) TH2: 2-m potential temperature = T2 * (P0/PSFC)^(Rd/cp). Built
    # from the final (diagnostic-overridden when present) T2 + PSFC so it is the
    # operational 2-m theta, consistent with WRF's TH2 = T2/pi2 diagnostic. ---
    if "T2" in fields and "PSFC" in fields:
        psfc = np.maximum(np.asarray(fields["PSFC"], dtype=np.float64), 1.0)
        t2 = np.asarray(fields["T2"], dtype=np.float64)
        fields["TH2"] = _coerce_array("TH2", t2 * (P0_PA / psfc) ** R_D_OVER_CP, shape_xy)

    # --- v0.12.0 A2 trivially-derived diagnostics (no physics, no GPU). ---
    _add_derived_diagnostic_fields(
        fields,
        state=state,
        land_state=land_state,
        xlat=xlat,
        xlong=xlong,
        shape_xy=shape_xy,
        run_start=run_start,
        lead_hours=lead_hours,
    )

    return {name: _materialized_dtype(name, value) for name, value in fields.items()}


def _add_full_wrfout_fields(
    fields: dict[str, np.ndarray],
    *,
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    dimensions: Mapping[str, int | None],
    diagnostics: Mapping[str, Any] | None,
    land_state: Any | None,
    run_start: datetime,
    lead_hours: float,
    requested_names: frozenset[str] | None = None,
) -> None:
    """Expand ``fields`` to the opt-in 375-variable WRF history stream.

    This is an output-compatibility layer, not a physics path. It first preserves
    all fields already mapped by the operational writer, then accepts explicit
    same-name payloads from diagnostics/state/land/grid/namelist, derives simple
    WRF algebraic statics, and finally fills unsupported inactive stream slots
    with the zero/default value WRF would carry for disabled diagnostics.
    """

    for name in FULL_WRFOUT_VARIABLES:
        if requested_names is not None and name not in requested_names:
            continue
        if name in {"Times", "XTIME"} or name in fields:
            continue
        spec = WRFOUT_VARIABLE_SPECS.get(name)
        if spec is None:
            raise KeyError(f"FULL_WRFOUT_VARIABLES contains {name} without a WRFOUT_VARIABLE_SPECS entry")
        shape = _shape_for_dimensions(spec.dimensions, dimensions)
        dtype = _numpy_dtype_for_spec(spec)
        value = _full_source_value(
            name,
            shape,
            dtype=dtype,
            diagnostics=diagnostics,
            state=state,
            land_state=land_state,
            grid=grid,
            namelist=namelist,
        )
        if value is None:
            value = _full_derived_value(
                name,
                fields,
                grid=grid,
                namelist=namelist,
                shape=shape,
                dtype=dtype,
                run_start=run_start,
                lead_hours=lead_hours,
            )
        if value is None:
            value = _full_default_value(name, shape, dtype=dtype)
        fields[name] = _coerce_array(name, value, shape, dtype=dtype)


def _full_source_value(
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: Any,
    diagnostics: Mapping[str, Any] | None,
    state: Any,
    land_state: Any | None,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
) -> np.ndarray | None:
    aliases = _full_source_aliases(name)
    for source in (diagnostics, state, land_state, grid, namelist):
        if source is None:
            continue
        for alias in aliases:
            value = _lookup(source, alias, None)
            if value is None:
                continue
            return _coerce_array(name, value, shape, dtype=dtype)
    return None


def _full_source_aliases(name: str) -> tuple[str, ...]:
    lower = name.lower()
    aliases = [name, lower]
    # Common WRF Registry output-name aliases that differ from gpuwrf's carry names.
    extras = {
        "SEAICE": ("xice", "seaice"),
        "SST": ("sst", "t_sst"),
        "SST_INPUT": ("sst_input", "sst"),
        "SSTSK": ("sstsk", "t_skin"),
        "TMN": ("tmn", "deep_soil_temperature"),
        "LAKEMASK": ("lakemask",),
        "VAR_SSO": ("var_sso",),
        "NEST_POS": ("nest_pos",),
        "IVGTYP": ("ivgtyp", "lu_index"),
        "ISLTYP": ("isltyp", "soil_type"),
        "ITIMESTEP": ("itimestep", "step_index"),
        "THIS_IS_AN_IDEAL_RUN": ("this_is_an_ideal_run",),
        "GOT_VAR_SSO": ("got_var_sso",),
        "SAVE_TOPO_FROM_REAL": ("save_topo_from_real",),
        "ISEEDARR_RAND_PERTURB": ("iseedarr_rand_perturb",),
        "TKE_PBL": ("tke_pbl",),
        "EL_PBL": ("el_pbl",),
        "P_HYD": ("p_hyd",),
    }.get(name, ())
    aliases.extend(extras)
    return tuple(dict.fromkeys(aliases))


def _full_derived_value(
    name: str,
    fields: Mapping[str, np.ndarray],
    *,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    shape: tuple[int, ...],
    dtype: Any,
    run_start: datetime,
    lead_hours: float,
) -> np.ndarray | float | int | None:
    if name == "P_HYD" and "P" in fields and "PB" in fields:
        return np.asarray(fields["P"], dtype=np.float64) + np.asarray(fields["PB"], dtype=np.float64)
    if name == "AREA2D":
        return _full_area2d(fields, grid, shape)
    if name == "DX2D":
        area = fields.get("AREA2D")
        if area is None:
            area = _full_area2d(fields, grid, shape)
        if area is not None:
            return np.sqrt(np.maximum(np.asarray(area, dtype=np.float64), 0.0))
    if name == "MF_VX_INV" and "MAPFAC_VX" in fields:
        return 1.0 / np.maximum(np.asarray(fields["MAPFAC_VX"], dtype=np.float64), 1.0e-12)
    if name == "MAX_MSFTX" and "MAPFAC_MX" in fields:
        return float(np.nanmax(np.asarray(fields["MAPFAC_MX"], dtype=np.float64)))
    if name == "MAX_MSFTY" and "MAPFAC_MY" in fields:
        return float(np.nanmax(np.asarray(fields["MAPFAC_MY"], dtype=np.float64)))
    if name in {"IVGTYP", "CROPCAT"} and "LU_INDEX" in fields:
        return np.rint(np.asarray(fields["LU_INDEX"], dtype=np.float64)).astype(dtype)
    if name == "LAKEMASK":
        return np.zeros(shape, dtype=dtype)
    if name in {"SST", "SST_INPUT", "SSTSK", "TMN", "TG", "TGV", "TGB"} and "TSK" in fields:
        return np.asarray(fields["TSK"], dtype=np.float64)
    if name in {"ALBBCK", "ALBOLD"} and "ALBEDO" in fields:
        return np.asarray(fields["ALBEDO"], dtype=np.float64)
    if name == "TV" and "T2" in fields and "Q2" in fields:
        return np.asarray(fields["T2"], dtype=np.float64) * (
            1.0 + 0.608 * np.maximum(np.asarray(fields["Q2"], dtype=np.float64), 0.0)
        )
    if name in {"T2V", "T2B"} and "T2" in fields:
        return np.asarray(fields["T2"], dtype=np.float64)
    if name in {"Q2V", "Q2B"} and "Q2" in fields:
        return np.asarray(fields["Q2"], dtype=np.float64)
    if name == "QSNOWXY" and "QSNOW" in fields:
        return np.sum(np.asarray(fields["QSNOW"], dtype=np.float64), axis=0)
    if name == "QRAINXY" and "QRAIN" in fields:
        return np.sum(np.asarray(fields["QRAIN"], dtype=np.float64), axis=0)
    if name == "ITIMESTEP":
        dt_s = _lookup(namelist, "dt_s", _lookup(namelist, "time_step", None))
        if dt_s not in (None, 0):
            return int(round(float(lead_hours) * 3600.0 / float(dt_s)))
        return int(round(float(lead_hours) * 60.0))
    if name == "THIS_IS_AN_IDEAL_RUN":
        return int(bool(_lookup(namelist, "this_is_an_ideal_run", False)))
    if name == "GOT_VAR_SSO":
        return int(any(k in fields for k in ("VAR", "CON", "OA1", "OL1", "VAR_SSO")))
    if name == "T00":
        return float(_lookup(namelist, "t00", P0_THETA_OFFSET_K))
    if name == "P00":
        return float(_lookup(namelist, "p00", P0_PA))
    if name == "TLP":
        return float(_lookup(namelist, "tlp", 50.0))
    if name == "TISO":
        return float(_lookup(namelist, "tiso", 200.0))
    if name == "P_STRAT":
        return float(_lookup(namelist, "p_strat", 0.0))
    if name == "TLP_STRAT":
        return float(_lookup(namelist, "tlp_strat", 0.0))
    if name == "ZETATOP":
        return float(_lookup(namelist, "zetatop", 0.0))
    if name == "RESM":
        return float(_lookup(namelist, "resm", 0.0))
    if name in {"HFX_FORCE", "LH_FORCE", "TSK_FORCE", "HFX_FORCE_TEND", "LH_FORCE_TEND", "TSK_FORCE_TEND"}:
        return 0.0
    if name == "SAVE_TOPO_FROM_REAL":
        return int(bool(_lookup(namelist, "save_topo_from_real", False)))
    if name == "BATHYMETRY_FLAG":
        return int(bool(_lookup(namelist, "bathymetry_flag", False)))
    del run_start  # currently only lead_hours contributes to derived full fields.
    return None


def _full_area2d(fields: Mapping[str, np.ndarray], grid: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    projection = _lookup(grid, "projection")
    dx_m = _lookup(projection, "dx_m", _lookup(grid, "dx", None))
    dy_m = _lookup(projection, "dy_m", _lookup(grid, "dy", dx_m))
    if dx_m is None or dy_m is None:
        return None
    area = float(dx_m) * float(dy_m)
    if "MAPFAC_MX" in fields and "MAPFAC_MY" in fields:
        mapx = np.maximum(np.asarray(fields["MAPFAC_MX"], dtype=np.float64), 1.0e-12)
        mapy = np.maximum(np.asarray(fields["MAPFAC_MY"], dtype=np.float64), 1.0e-12)
        return area / (mapx * mapy)
    return np.full(shape, area, dtype=np.float64)


def _full_default_value(name: str, shape: tuple[int, ...], *, dtype: Any) -> np.ndarray:
    del name
    return np.zeros(shape, dtype=dtype)


def _add_derived_diagnostic_fields(
    fields: dict[str, np.ndarray],
    *,
    state: Any,
    land_state: Any | None,
    xlat: np.ndarray,
    xlong: np.ndarray,
    shape_xy: tuple[int, int],
    run_start: datetime | None,
    lead_hours: float,
) -> None:
    """Populate the cheap WRF-faithful derived diagnostics (v0.12.0 A2).

    Each field self-gates on the availability of its inputs and uses a closed
    WRF-faithful form -- never a fabricated quantity:

    - ``CLAT`` (computational-grid latitude) == ``XLAT`` for the operational
      lat/lon grid; always present because the writer always resolves XLAT.
    - ``COSZEN`` is WRF's ``radconst``/``calc_coszen`` cosine solar zenith angle
      (module_radiation_driver.F:3594-3666) evaluated from XLAT/XLONG and the
      forecast clock (``run_start`` + ``lead_hours``). Routed through the exact
      transcription in ``coupling.physics_couplers._compute_coszen``. Skipped
      (not fabricated) if the clock is absent.
    - ``SR`` is the WRF surface frozen-precipitation fraction
      = solid_acc / (solid_acc + liquid_acc), built from the State precip
      accumulators (snow+graupel+ice vs rain). Zero where no precip has fallen;
      skipped entirely when the State carries no accumulators (dry/synthetic).
    - ``SNOWC`` is WRF's snow-cover flag (1 where bulk SWE > 0) from the land
      carry's ``sneqv``; skipped when there is no land carry (never fabricated).
    """

    # CLAT == XLAT (computational == geographic latitude on the operational grid).
    fields["CLAT"] = _coerce_array("CLAT", np.asarray(xlat), shape_xy)

    # COSZEN from the WRF solar-geometry transcription + the forecast clock.
    if run_start is not None:
        try:
            from gpuwrf.coupling.physics_couplers import _compute_coszen

            coszen = _compute_coszen(
                np.asarray(xlat, dtype=np.float64),
                np.asarray(xlong, dtype=np.float64),
                run_start,
                lead_seconds=float(lead_hours) * 3600.0,
            )
            fields["COSZEN"] = _coerce_array("COSZEN", np.asarray(coszen), shape_xy)
        except Exception:  # noqa: BLE001 -- best-effort diagnostic; never blocks output.
            pass

    # SR = frozen-precip fraction from the State accumulators (mm). solid = snow +
    # graupel + ice; total = solid + rain. WRF SR is in [0,1]; zero where dry.
    rain_acc = _optional_field_array(state, ("rain_acc", "RAINNC"), shape_xy)
    snow_acc = _optional_field_array(state, ("snow_acc", "SNOWNC"), shape_xy)
    graupel_acc = _optional_field_array(state, ("graupel_acc", "GRAUPELNC"), shape_xy)
    ice_acc = _optional_field_array(state, ("ice_acc",), shape_xy)
    if any(a is not None for a in (rain_acc, snow_acc, graupel_acc, ice_acc)):
        def _z(a: np.ndarray | None) -> np.ndarray:
            return np.asarray(a, dtype=np.float64) if a is not None else np.zeros(shape_xy, dtype=np.float64)

        solid = _z(snow_acc) + _z(graupel_acc) + _z(ice_acc)
        total = solid + _z(rain_acc)
        sr = np.where(total > 0.0, solid / np.maximum(total, 1.0e-12), 0.0)
        fields["SR"] = _coerce_array("SR", sr, shape_xy)

    # SNOWC = snow-cover flag (1 where bulk SWE > 0) from the land carry SWE.
    sneqv = _lookup(land_state, "sneqv") if land_state is not None else None
    if sneqv is None and land_state is not None:
        sneqv = _lookup(land_state, "snow")
    if sneqv is not None:
        swe = np.asarray(sneqv, dtype=np.float64)
        fields["SNOWC"] = _coerce_array("SNOWC", np.where(swe > 0.0, 1.0, 0.0), shape_xy)


def _add_grid_coordinate_fields(
    fields: dict[str, np.ndarray],
    grid: Any,
    dimensions: Mapping[str, int | None],
    *,
    shape_xy: tuple[int, int],
    shape_mapu: tuple[int, int],
    shape_mapv: tuple[int, int],
    landmask: np.ndarray,
) -> None:
    """Populate the WRF grid-static coordinate / map-factor / Coriolis fields.

    Source = the real ``GridSpec`` (``eta_levels`` + ``DycoreMetrics``). Each
    field is added ONLY when its source array is actually present on the grid, so
    a dict-driven / synthetic grid with no metrics emits none of them (rather than
    fabricating unit map factors). All map/Coriolis arrays are
    ``DycoreMetrics`` fp64 device arrays; ``_coerce_array`` pulls them to host
    np.float32 at this output boundary.
    """

    nz = int(dimensions["bottom_top"])
    # ZNU/ZNW: eta on half (mass) / full (w) levels. WRF znw = full eta levels;
    # znu = mass-level midpoints. eta_levels is the authoritative (nz+1,) column.
    eta = _lookup(grid, "eta_levels")
    if eta is not None:
        eta_host = np.asarray(eta, dtype=np.float64)
        if eta_host.shape == (nz + 1,):
            fields["ZNW"] = _coerce_array("ZNW", eta_host, (nz + 1,))
            fields["ZNU"] = _coerce_array("ZNU", 0.5 * (eta_host[:-1] + eta_host[1:]), (nz,))

    # ZS/DZS: static Noah/Noah-MP soil-layer geometry. Pure constants of the soil
    # configuration (WRF init_soil_depth_2, module_soil_pre.F:1128-1151): DZS =
    # layer thicknesses, ZS = layer-center depths derived from DZS. WRF carries
    # them in every history frame so downstream tools can reconstruct the soil
    # column. They depend ONLY on the soil_layers_stag count, so they are always
    # emittable (no metrics/state source needed) -- the SAME arrays the real-init
    # + Noah-MP land hook consume. Computed in real(4) to bit-match the Fortran
    # init_soil_depth_2, whose fp32 cumulative sum rounds ZS(3)=0.70000005
    # (0x3f333334) -- an fp64 accumulation rounds the other way (0x3f333333).
    n_soil = int(dimensions["soil_layers_stag"])
    if n_soil == 4:
        f32 = np.float32
        dzs_host = np.array([0.1, 0.3, 0.6, 1.0], dtype=np.float32)
        zs_host = np.zeros(n_soil, dtype=np.float32)
        zs_host[0] = f32(0.5) * dzs_host[0]
        for _l in range(1, n_soil):
            zs_host[_l] = zs_host[_l - 1] + f32(0.5) * dzs_host[_l - 1] + f32(0.5) * dzs_host[_l]
        fields["ZS"] = _coerce_array("ZS", zs_host, (n_soil,))
        fields["DZS"] = _coerce_array("DZS", dzs_host, (n_soil,))

    metrics = _lookup(grid, "metrics")
    if metrics is not None:
        # Map-scale factors. WRF emits both the primary MAPFAC_M/U/V (the x-direction
        # factors the dycore/PGF consume) and the directional MAPFAC_{M,U,V}{X,Y}.
        # All map onto DycoreMetrics.msf{t,u,v}{x,y}; the primary M/U/V alias the
        # x-direction factors (msftx/msfux/msfvx), matching WRF for the conformal
        # Canary projection.
        for wrf_name, attr, shape in (
            ("MAPFAC_M", "msftx", shape_xy),
            ("MAPFAC_U", "msfux", shape_mapu),
            ("MAPFAC_V", "msfvx", shape_mapv),
            ("MAPFAC_MX", "msftx", shape_xy),
            ("MAPFAC_MY", "msfty", shape_xy),
            ("MAPFAC_UX", "msfux", shape_mapu),
            ("MAPFAC_UY", "msfuy", shape_mapu),
            ("MAPFAC_VX", "msfvx", shape_mapv),
            ("MAPFAC_VY", "msfvy", shape_mapv),
            ("F", "f", shape_xy),
            ("E", "e", shape_xy),
            ("SINALPHA", "sina", shape_xy),
            ("COSALPHA", "cosa", shape_xy),
        ):
            value = _lookup(metrics, attr)
            if value is not None:
                fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), shape)

        # v0.12.0 A1: vertical eta-spacing + hybrid-eta C-coefficient column metrics.
        # PURE PAYLOAD of arrays already resident on DycoreMetrics -- no recompute.
        for wrf_name, attr in (
            ("DN", "dn"),
            ("DNW", "dnw"),
            ("RDN", "rdn"),
            ("RDNW", "rdnw"),
            ("FNM", "fnm"),
            ("FNP", "fnp"),
            ("C1H", "c1h"),
            ("C2H", "c2h"),
            ("C3H", "c3h"),
            ("C4H", "c4h"),
        ):
            value = _lookup(metrics, attr)
            if value is not None and np.asarray(value).shape == (nz,):
                fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), (nz,))
        for wrf_name, attr in (
            ("C1F", "c1f"),
            ("C2F", "c2f"),
            ("C3F", "c3f"),
            ("C4F", "c4f"),
        ):
            value = _lookup(metrics, attr)
            if value is not None and np.asarray(value).shape == (nz + 1,):
                fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), (nz + 1,))

        # Scalar extrapolation constants. cf1/cf2/cf3 are stored directly; the WRF
        # top-of-model constants cfn/cfn1 are the standard dnw/dn ratios at the top
        # mass level (module_initialize_real.F:3757-3758, 0-indexed nz-1):
        #   cfn  = (0.5*dnw[nz-1] + dn[nz-1]) / dn[nz-1]
        #   cfn1 = -0.5*dnw[nz-1] / dn[nz-1]
        for wrf_name, attr in (("CF1", "cf1"), ("CF2", "cf2"), ("CF3", "cf3")):
            value = _lookup(metrics, attr)
            if value is not None:
                fields[wrf_name] = _coerce_array(wrf_name, float(np.asarray(value).reshape(-1)[0]), ())
        dn = _lookup(metrics, "dn")
        dnw = _lookup(metrics, "dnw")
        if dn is not None and dnw is not None:
            dn_top = float(np.asarray(dn).reshape(-1)[-1])
            dnw_top = float(np.asarray(dnw).reshape(-1)[-1])
            if dn_top != 0.0:
                fields["CFN"] = _coerce_array("CFN", (0.5 * dnw_top + dn_top) / dn_top, ())
                fields["CFN1"] = _coerce_array("CFN1", -0.5 * dnw_top / dn_top, ())

        p_top = _lookup(metrics, "p_top")
        if p_top is not None:
            fields["P_TOP"] = _coerce_array("P_TOP", float(np.asarray(p_top).reshape(-1)[0]), ())

    # Inverse grid lengths RDX = 1/dx, RDY = 1/dy from the projection (WRF scalars).
    projection = _lookup(grid, "projection")
    dx_m = _lookup(projection, "dx_m", _lookup(grid, "dx", None))
    dy_m = _lookup(projection, "dy_m", _lookup(grid, "dy", dx_m))
    if dx_m is not None and float(dx_m) != 0.0:
        fields["RDX"] = _coerce_array("RDX", 1.0 / float(dx_m), ())
    if dy_m is not None and float(dy_m) != 0.0:
        fields["RDY"] = _coerce_array("RDY", 1.0 / float(dy_m), ())

    # XLAND (1 land / 2 water) = WRF land/water flag. Derive from the landmask the
    # writer already resolved (1 land / 0 water) so it is always consistent.
    fields["XLAND"] = _coerce_array("XLAND", np.where(landmask > 0.5, 1.0, 2.0), shape_xy)


def _add_land_soil_fields(
    fields: dict[str, np.ndarray],
    land_state: Any,
    *,
    shape_soil: tuple[int, int, int],
    shape_snow: tuple[int, int, int],
    shape_snso: tuple[int, int, int],
    shape_xy: tuple[int, int],
) -> None:
    """Populate the prognostic Noah-MP soil/snow land + diagnostic fields.

    Source = the ``NoahMPLandState`` carry (4-layer ``tslb``/``smois``/``sh2o``,
    bulk ``sneqv``/``snowh``, intercepted canopy water, accumulated runoff, broadband
    ``albedo``/``emiss``). Each field is added ONLY when the carry exposes its
    source attribute, so a partial land carry never fabricates a soil profile.
    WRF mapping (module_sf_noahmpdrv.F): SNOW=SNEQV, SNOWH=SNOWH,
    CANWAT=CANLIQ+CANICE, SFROFF=SFCRUNOFF*1e3, UDROFF=UDRUNOFF*1e3 (m->mm).
    """

    for wrf_name, attr in (("TSLB", "tslb"), ("SMOIS", "smois"), ("SH2O", "sh2o")):
        value = _lookup(land_state, attr)
        if value is not None:
            fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), shape_soil)

    for wrf_name, attr, shape in (
        ("TSNO", "tsno", shape_snow),
        ("SNICE", "snice", shape_snow),
        ("SNLIQ", "snliq", shape_snow),
        ("ZSNSO", "zsnso", shape_snso),
    ):
        value = _lookup(land_state, attr)
        if value is not None:
            fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), shape)

    # Scalar (ny, nx) snow/canopy prognostics. ISNOW is the int32 active-layer
    # count (-2..0); SNEQVO is the prior-step SWE; CANLIQ/CANICE are the canopy
    # interception reservoirs (their sum is the CANWAT bulk written below).
    isnow = _lookup(land_state, "isnow")
    if isnow is not None:
        fields["ISNOW"] = _coerce_array("ISNOW", np.asarray(isnow), shape_xy, dtype=np.int32)
    sneqvo = _lookup(land_state, "sneqvo")
    if sneqvo is not None:
        fields["SNEQVO"] = _coerce_array("SNEQVO", np.asarray(sneqvo), shape_xy)

    sneqv = _lookup(land_state, "sneqv")
    if sneqv is not None:
        fields["SNOW"] = _coerce_array("SNOW", np.asarray(sneqv), shape_xy)
    snowh = _lookup(land_state, "snowh")
    if snowh is not None:
        fields["SNOWH"] = _coerce_array("SNOWH", np.asarray(snowh), shape_xy)
    canliq = _lookup(land_state, "canliq")
    canice = _lookup(land_state, "canice")
    if canliq is not None:
        fields["CANLIQ"] = _coerce_array("CANLIQ", np.asarray(canliq), shape_xy)
    if canice is not None:
        fields["CANICE"] = _coerce_array("CANICE", np.asarray(canice), shape_xy)
    if canliq is not None or canice is not None:
        canwat = (np.asarray(canliq) if canliq is not None else 0.0) + (
            np.asarray(canice) if canice is not None else 0.0
        )
        fields["CANWAT"] = _coerce_array("CANWAT", canwat, shape_xy)
    # WRF carries SFROFF/UDROFF in mm; the Noah-MP carry accumulates runoff in m.
    sfcrunoff = _lookup(land_state, "sfcrunoff")
    if sfcrunoff is not None:
        fields["SFROFF"] = _coerce_array("SFROFF", np.asarray(sfcrunoff) * 1.0e3, shape_xy)
    udrunoff = _lookup(land_state, "udrunoff")
    if udrunoff is not None:
        fields["UDROFF"] = _coerce_array("UDROFF", np.asarray(udrunoff) * 1.0e3, shape_xy)
    albedo = _lookup(land_state, "albedo")
    if albedo is not None:
        fields["ALBEDO"] = _coerce_array("ALBEDO", np.asarray(albedo), shape_xy)
    emiss = _lookup(land_state, "emiss")
    if emiss is not None:
        fields["EMISS"] = _coerce_array("EMISS", np.asarray(emiss), shape_xy)


def _surface_flux_fallbacks(
    *,
    state: Any,
    grid: Any,
    theta: np.ndarray,
    qv: np.ndarray,
    p_total: np.ndarray,
    ph_total: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    landmask: np.ndarray,
    shape_xy: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Recompute sfclay-style HFX/LH when direct WRF flux diagnostics are absent."""

    u_mass = _unstagger_x(u)
    v_mass = _unstagger_y(v)
    dz = np.maximum((ph_total[1:, :, :] - ph_total[:-1, :, :]) / 9.80665, 1.0)
    t_air0 = theta[0] * (np.maximum(p_total[0], 1.0) / P0_PA) ** R_D_OVER_CP
    t_skin = _field_array(state, ("TSK", "tsk", "t_skin"), shape_xy, default=t_air0)
    xland = _field_array(state, ("xland", "XLAND"), shape_xy, default=np.where(landmask > 0.5, 1.0, 2.0))
    soil_moisture = _optional_field_array(state, ("soil_moisture", "SMOIS"), shape_xy)
    mavail = _optional_field_array(state, ("mavail", "MAVAIL"), shape_xy)
    if mavail is None:
        mavail = soil_moisture if soil_moisture is not None else np.ones(shape_xy, dtype=np.float32)
    column_state = SimpleNamespace(
        u=np.moveaxis(u_mass, 0, -1),
        v=np.moveaxis(v_mass, 0, -1),
        theta=np.moveaxis(theta, 0, -1),
        qv=np.moveaxis(qv, 0, -1),
        p=np.moveaxis(p_total, 0, -1),
        dz=np.moveaxis(dz, 0, -1),
        t_skin=t_skin,
        soil_moisture=soil_moisture,
        xland=xland,
        lakemask=_field_array(state, ("lakemask", "LAKEMASK"), shape_xy),
        mavail=mavail,
        roughness_m=_optional_field_array(state, ("roughness_m", "ZNT"), shape_xy),
        ustar=_field_array(state, ("UST", "ustar"), shape_xy),
        dx_m=float(_lookup(_lookup(grid, "projection"), "dx_m", _lookup(grid, "dx", 3000.0))),
    )
    diag = surface_layer_with_diagnostics(column_state)
    # SurfaceLayerDiagnostics already carries the WRF-faithful sfclayrev surface
    # fluxes (hfx = flhc*(thgb-thx), lh = XLV*qfx; sf_sfclayrev.F90:856-878), both
    # W m^-2 positive-upward -- exactly the wrfout HFX/LH contract. Use them directly
    # instead of re-deriving HFX from an aerodynamic resistance: the old path read a
    # nonexistent diag.fh and raised AttributeError on the full operational/d03 path.
    hfx = np.asarray(diag.hfx, dtype=np.float64)
    lh = np.asarray(diag.lh, dtype=np.float64)
    return {"HFX": hfx.astype(np.float32), "LH": lh.astype(np.float32)}


def _perturbation_base_pair(
    state: Any,
    *,
    total_names: tuple[str, ...],
    perturbation_names: tuple[str, ...],
    base_names: tuple[str, ...],
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    total = _optional_field_array(state, total_names, shape)
    perturbation = _optional_field_array(state, perturbation_names, shape)
    base = _optional_field_array(state, base_names, shape)
    zeros = np.zeros(shape, dtype=np.float32)

    if total is None and perturbation is None and base is None:
        return zeros, zeros
    if total is None and perturbation is not None and base is not None:
        total = perturbation + base
    if perturbation is None and total is not None and base is not None:
        perturbation = total - base
    if base is None and total is not None and perturbation is not None:
        base = total - perturbation
    if perturbation is None and total is not None:
        perturbation = total
    if base is None:
        base = zeros
    return perturbation.astype(np.float32), base.astype(np.float32)


def _field_array(
    obj: Any,
    names: tuple[str, ...],
    shape: tuple[int, ...],
    *,
    default: Any = 0.0,
) -> np.ndarray:
    value = _optional_field_array(obj, names, shape)
    if value is None:
        value = default
    return _coerce_array(names[0], value, shape)


def _optional_field_array(obj: Any, names: tuple[str, ...], shape: tuple[int, ...]) -> np.ndarray | None:
    for name in names:
        value = _lookup(obj, name, None)
        if value is not None:
            return _coerce_array(name, value, shape)
    return None


def _grid_or_state_array(
    state: Any,
    grid: Any,
    names: tuple[str, ...],
    shape: tuple[int, ...],
    *,
    default: Any = 0.0,
) -> np.ndarray:
    value = _optional_field_array(state, names, shape)
    if value is None:
        value = _optional_field_array(grid, names, shape)
    if value is None:
        value = default
    return _coerce_array(names[0], value, shape)


def _landmask(state: Any, shape: tuple[int, ...]) -> np.ndarray:
    explicit = _optional_field_array(state, ("LANDMASK", "landmask"), shape)
    if explicit is not None:
        return explicit
    xland = _optional_field_array(state, ("xland", "XLAND"), shape)
    if xland is None:
        return np.ones(shape, dtype=np.float32)
    return np.where(xland <= 1.5, 1.0, 0.0).astype(np.float32)


def _latlon_fields(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    shape: tuple[int, int],
    *,
    suffix: str = "",
    diagnostics: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lat_names = ("XLAT" + suffix.upper(), "xlat" + suffix, "lat" + suffix)
    lon_names = ("XLONG" + suffix.upper(), "xlong" + suffix, "lon" + suffix)
    lat = _optional_field_array(diagnostics, lat_names, shape)
    lon = _optional_field_array(diagnostics, lon_names, shape)
    if lat is not None and lon is not None:
        return lat, lon
    lat = _optional_field_array(state, lat_names, shape)
    lon = _optional_field_array(state, lon_names, shape)
    if lat is not None and lon is not None:
        return lat, lon
    projection = _lookup(grid, "projection")
    lat_0 = float(_lookup(projection, "lat_0", _lookup(namelist, "cen_lat", 0.0)))
    lon_0 = float(_lookup(projection, "lon_0", _lookup(namelist, "cen_lon", 0.0)))
    dx_m = float(_lookup(projection, "dx_m", _lookup(namelist, "dx", 3000.0)))
    dy_m = float(_lookup(projection, "dy_m", _lookup(namelist, "dy", dx_m)))
    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0
    x = np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0
    lat_step = dy_m / 111_320.0
    lon_step = dx_m / max(111_320.0 * cos(lat_0 * pi / 180.0), 1.0)
    lat_grid = lat_0 + y[:, None] * lat_step
    lon_grid = lon_0 + x[None, :] * lon_step
    return (
        np.broadcast_to(lat_grid, shape).astype(np.float32),
        np.broadcast_to(lon_grid, shape).astype(np.float32),
    )


def _unstagger_x(u: np.ndarray) -> np.ndarray:
    return 0.5 * (u[..., :-1] + u[..., 1:])


def _unstagger_y(v: np.ndarray) -> np.ndarray:
    return 0.5 * (v[..., :-1, :] + v[..., 1:, :])


def _shape_for_dimensions(dimensions: tuple[str, ...], sizes: Mapping[str, int | None]) -> tuple[int, ...]:
    return tuple(int(sizes[name]) for name in dimensions if name != "Time")


def _materialized_dtype(name: str, value: Any) -> np.ndarray:
    spec = WRFOUT_VARIABLE_SPECS.get(name)
    if spec is not None and str(spec.dtype).startswith("i"):
        return np.asarray(value, dtype=np.int32)
    return np.asarray(value, dtype=np.float32)


def _numpy_dtype_for_spec(spec: WrfoutVariableSpec) -> Any:
    if str(spec.dtype).startswith("i"):
        return np.int32
    if spec.dtype == "f8":
        return np.float64
    return np.float32


def _coerce_array(name: str, value: Any, shape: tuple[int, ...], *, dtype: Any = np.float32) -> np.ndarray:
    array = np.asarray(_host_materialized_value(value), dtype=dtype)
    if array.shape == (1, *shape):
        array = array[0]
    if array.shape == shape:
        return array.astype(dtype, copy=False)
    if array.shape == ():
        return np.full(shape, array.item(), dtype=dtype)
    try:
        return np.broadcast_to(array, shape).astype(dtype)
    except ValueError as exc:
        raise ValueError(f"{name} shape {array.shape} cannot be written to WRF shape {shape}") from exc


def _grid_extent(grid: Any) -> tuple[int, int, int]:
    nx = _lookup(grid, "nx", None)
    ny = _lookup(grid, "ny", None)
    nz = _lookup(grid, "nz", None)
    projection = _lookup(grid, "projection")
    vertical = _lookup(grid, "vertical")
    if nx is None:
        nx = _lookup(projection, "nx", None)
    if ny is None:
        ny = _lookup(projection, "ny", None)
    if nz is None:
        nz = _lookup(vertical, "nz", None)
    missing = [name for name, value in {"nx": nx, "ny": ny, "nz": nz}.items() if value is None]
    if missing:
        raise ValueError(f"grid is missing {', '.join(missing)}")
    return int(nx), int(ny), int(nz)


def _lookup(obj: Any, name: str, default: Any = None) -> Any:
    return _host_materialized_value(_lookup_raw(obj, name, default))


def _lookup_raw(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _coerce_datetime(value: datetime | date | str) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip().replace("Z", "")
    for fmt in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(text).replace(tzinfo=None)


def _wrf_time_string(value: datetime) -> str:
    return value.strftime("%Y-%m-%d_%H:%M:%S")
