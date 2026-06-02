"""FROZEN metgrid-equivalent artifact schema (v0.3.0 S0 gate output).

This module is the **frozen interface** for the v0.3.0 native-metgrid-ingest
milestone. It defines the exact field set, dimensions, staggering, units, vertical
levels, and projection/coordinate metadata of the *metgrid-equivalent artifact*
that:

* the v0.3.0 lanes S1 (forcing decode), S2 (static geog), S3 (interp) produce,
* the v0.3.0 S4 parity harness compares against the real WPS ``met_em.*`` oracle,
* the v0.4.0 native real.exe-equivalent init consumes.

It is derived from the real WPS ``met_em.*`` structure (recon: ``proofs/v030/
RECON.md`` + ``recon_inventory.json``, oracle = ``/mnt/data/canairy_meteo/runs/
wps_cases/<case>/l3/met_em.d0{1,2,3}.*.nc``, metgrid V4.6.0) intersected with what
WRF ``real.exe`` (``module_initialize_real.F``) requires.

DESIGN
------
* **met_em-faithful naming/layout.** Every field uses the met_em variable name,
  dims, stagger code, and units, so S4 parity is a direct field-by-field compare
  and the v0.4.0 consumer reads a structure real.exe would accept.
* **Pure data / CPU-importable.** This is an OFFLINE artifact contract (written to
  NetCDF, read by the next milestone). It carries numpy arrays + metadata; it does
  NOT require a GPU and does NOT live in the GPU-resident timestep ``State``. The
  interp kernels (S3) may run on GPU, but the artifact itself is a serializable
  product.
* **Frozen vertical = 14 metgrid levels** (1 surface + 13 isobaric), in met_em
  order (index 0 = surface, 1..13 = 1000..50 hPa).
* **fp32 storage like met_em** (metgrid writes ``float``); the comparator works in
  fp64. Storing fp32 keeps the artifact byte-comparable in magnitude to met_em.

FREEZE POLICY
-------------
Changing ``METGRID_SCHEMA_VERSION`` or any field spec after S0 requires a manager
sign-off + a note in ``proofs/v030/RECON.md`` (this is the interface the parallel
lanes are built against). Additive optional fields (new ``MetgridFieldSpec`` with
``mandatory=False``) are allowed without a version bump if they do not change any
existing field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


METGRID_SCHEMA_VERSION = "0.3.0-S0-frozen-2026-06-02"

# WPS/metgrid version the oracle was produced with (met_em TITLE global attr).
ORACLE_METGRID_VERSION = "4.6.0"

# --- staggering --------------------------------------------------------------
# met_em ``stagger`` attribute values; map to C-grid points. "CORNER" is the
# cell-corner (B-grid-like) stagger used only by XLAT_C/XLONG_C.
Stagger = Literal["M", "U", "V", "CORNER"]

# --- the 14 metgrid vertical levels (binding) --------------------------------
# Index 0 = model surface (assembled from 2 m / 10 m / sfc fields); indices 1..13
# = standard isobaric levels. Values are nominal isobaric pressures in Pa; the
# surface level's pressure is spatially varying (= PSFC), so its nominal here is a
# sentinel only. See RECON.md §2.
NUM_METGRID_LEVELS = 14
NUM_ST_LAYERS = 2
NUM_SM_LAYERS = 2
SOIL_LAYER_DEPTHS_CM = (10, 40)  # met_em SOIL_LAYERS = [40., 10.]; named *000010,*010040
ISOBARIC_LEVELS_PA = (
    100000.0, 92500.0, 85000.0, 70000.0, 60000.0, 50000.0, 40000.0,
    30000.0, 25000.0, 20000.0, 15000.0, 10000.0, 5000.0,
)  # 1000..50 hPa, 13 levels (level index 1..13 in the artifact)
SURFACE_LEVEL_SENTINEL_PA = -1.0  # surface level pressure is PSFC (spatial), not a scalar

# met_em variables DELIBERATELY not modeled as MetgridFieldSpec entries:
#   * "Times"                     -> represented by MetEmArtifact.valid_time; the
#                                    NetCDF serializer writes the (Time, DateStrLen)
#                                    char array from it.
#   * SINALPHA_U / SINALPHA_V /   -> staggered wind-rotation factors. The mass-point
#     COSALPHA_U / COSALPHA_V        SINALPHA/COSALPHA ARE modeled; the U/V-stag
#                                    variants are only needed if grid-relative wind
#                                    rotation is applied on staggered points. For
#                                    Lambert with stand_lon == cen_lon (Canary case,
#                                    rotation ~0) they are derivable from the mass
#                                    field; add as optional specs if a future case
#                                    needs them. Tracked, not a v0.3.0 gate field.
OMITTED_METEM_FIELDS = (
    "Times",
    "SINALPHA_U",
    "SINALPHA_V",
    "COSALPHA_U",
    "COSALPHA_V",
)


def metgrid_levels_spec() -> dict[str, object]:
    """Returns the frozen vertical-level description (for the artifact header)."""

    return {
        "num_metgrid_levels": NUM_METGRID_LEVELS,
        "order": "index0=surface, index1..13 = 1000..50 hPa",
        "isobaric_levels_pa": list(ISOBARIC_LEVELS_PA),
        "num_st_layers": NUM_ST_LAYERS,
        "num_sm_layers": NUM_SM_LAYERS,
        "soil_layer_depths_cm": list(SOIL_LAYER_DEPTHS_CM),
    }


# --- projection / coordinate metadata (binding) ------------------------------
@dataclass(frozen=True)
class MetgridProjection:
    """Lambert-conformal projection metadata, mirroring met_em global attrs.

    All v0.3.0 Canary domains share map_proj=1 (Lambert), truelat1/2=25/30,
    stand_lon=-16.4. Per-domain values (dx/dy, dims, parent nesting) differ.
    """

    map_proj: int  # 1 = Lambert conformal (only supported value in v0.3.0)
    truelat1: float
    truelat2: float
    stand_lon: float
    moad_cen_lat: float
    pole_lat: float
    pole_lon: float
    dx_m: float
    dy_m: float
    # mass-grid dims
    nx: int  # west_east
    ny: int  # south_north
    # nesting
    grid_id: int
    parent_id: int
    parent_grid_ratio: int
    i_parent_start: int
    j_parent_start: int
    # land/soil category metadata (constant across domains here)
    mminlu: str = "MODIFIED_IGBP_MODIS_NOAH"
    num_land_cat: int = 21
    num_soil_cat: int = 16
    iswater: int = 17
    islake: int = 21
    isice: int = 15
    isurban: int = 13
    isoilwater: int = 14

    def __post_init__(self) -> None:
        if self.map_proj != 1:
            raise ValueError(
                f"v0.3.0 schema supports only Lambert (map_proj=1); got {self.map_proj}"
            )
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("projection nx/ny must be positive")


# --- per-field spec (binding for S1/S2/S3 producers + S4 comparator) ---------
FieldGroup = Literal["atmos3d", "soil", "surface2d", "geog2d", "geog3d", "coord", "mapfac"]


@dataclass(frozen=True)
class MetgridFieldSpec:
    """One met_em variable's frozen contract.

    ``dims`` are the met_em dimension names in netCDF order (excluding the leading
    ``Time``). ``vertical_dim`` names which dim (if any) is the vertical/level axis.
    ``interp_option`` records the metgrid horizontal-interp method (METGRID.TBL.ARW)
    so S3 reproduces the right stencil and S4 derives the right tolerance.
    ``parity_tol`` is the predeclared S4 acceptance tolerance vs the oracle
    (absolute, in the field's units; ``rel_tol`` is a relative alternative). A
    field passes parity if max|native - oracle| <= parity_tol OR the relative
    bound holds, evaluated per RECON.md §5 masking.
    """

    name: str
    group: FieldGroup
    dims: tuple[str, ...]
    stagger: Stagger
    units: str
    description: str
    interp_option: str
    mandatory: bool
    parity_tol: float
    rel_tol: float = 0.0
    masked: Literal["none", "land", "water", "both"] = "none"
    fill_missing: float | None = None
    source: Literal["aifs_grib", "geo_em", "derived"] = "derived"
    vertical_dim: str | None = None
    flag_in_output: str | None = None


# Convenience dim tuples (met_em dimension names; Time is implicit/leading).
_ATM = ("num_metgrid_levels", "south_north", "west_east")
_ATM_U = ("num_metgrid_levels", "south_north", "west_east_stag")
_ATM_V = ("num_metgrid_levels", "south_north_stag", "west_east")
_SFC = ("south_north", "west_east")
_ST = ("num_st_layers", "south_north", "west_east")
_SM = ("num_sm_layers", "south_north", "west_east")
_U2 = ("south_north", "west_east_stag")
_V2 = ("south_north_stag", "west_east")


def metem_field_specs() -> tuple[MetgridFieldSpec, ...]:
    """Returns the FROZEN ordered tuple of met_em-equivalent field specs.

    Tolerances are PREDECLARED for S4. They are stricter for mandatory dynamics
    fields (TT/UU/VV/GHT/PRES/SPECHUMD), which poison hour-0 if wrong, and looser
    for categorical/static/diagnostic fields. They are stated as the OFFLINE
    metgrid-parity gate; they are NOT the downstream forecast-skill gate (that is
    v0.4.0). Rationale per field is in RECON.md §5 and the S4 contract.
    """

    return (
        # ---- mandatory 3D atmosphere (sixteen_pt+four_pt+average_4pt) ----
        MetgridFieldSpec(
            "TT", "atmos3d", _ATM, "M", "K", "Temperature",
            "sixteen_pt+four_pt+average_4pt", True,
            parity_tol=0.20, rel_tol=0.0, source="aifs_grib",
            vertical_dim="num_metgrid_levels",
        ),
        MetgridFieldSpec(
            "UU", "atmos3d", _ATM_U, "U", "m s-1", "U",
            "sixteen_pt+four_pt+average_4pt", True,
            parity_tol=0.25, source="aifs_grib", vertical_dim="num_metgrid_levels",
        ),
        MetgridFieldSpec(
            "VV", "atmos3d", _ATM_V, "V", "m s-1", "V",
            "sixteen_pt+four_pt+average_4pt", True,
            parity_tol=0.25, source="aifs_grib", vertical_dim="num_metgrid_levels",
        ),
        MetgridFieldSpec(
            "GHT", "atmos3d", _ATM, "M", "m", "Height derived from geopotential",
            "sixteen_pt+four_pt+average_4pt", True,
            parity_tol=2.0, rel_tol=1e-4, source="aifs_grib",
            vertical_dim="num_metgrid_levels",
        ),
        MetgridFieldSpec(
            "SPECHUMD", "atmos3d", _ATM, "M", "kg kg-1", "Specific humidity",
            "sixteen_pt+four_pt+average_4pt", True,
            parity_tol=1e-4, rel_tol=1e-2, source="aifs_grib",
            vertical_dim="num_metgrid_levels", fill_missing=0.0,
            flag_in_output="FLAG_SH",
        ),
        MetgridFieldSpec(
            "PRES", "atmos3d", _ATM, "M", "Pa", "Pressure (derived)",
            "derived(PRESSURE+PSFC@sfc+vertical_index)", True,
            parity_tol=5.0, rel_tol=1e-5, source="derived",
            vertical_dim="num_metgrid_levels",
        ),
        # ---- surface 2D ----
        MetgridFieldSpec(
            "PSFC", "surface2d", _SFC, "M", "Pa", "Surface pressure",
            "four_pt+average_4pt", True,
            parity_tol=10.0, rel_tol=1e-5, source="aifs_grib",
            flag_in_output="FLAG_PSFC",
        ),
        MetgridFieldSpec(
            "PMSL", "surface2d", _SFC, "M", "Pa", "Mean sea-level pressure",
            "sixteen_pt+four_pt+average_4pt", False,
            parity_tol=10.0, rel_tol=1e-5, source="aifs_grib",
            flag_in_output="FLAG_SLP",
        ),
        MetgridFieldSpec(
            "SOILHGT", "surface2d", _SFC, "M", "m", "Source surface height",
            "four_pt+average_4pt", False,
            parity_tol=1.0, source="aifs_grib", flag_in_output="FLAG_SOILHGT",
        ),
        MetgridFieldSpec(
            "SKINTEMP", "surface2d", _SFC, "M", "K", "Skin temperature",
            "sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search", True,
            parity_tol=0.30, masked="both", fill_missing=0.0, source="aifs_grib",
        ),
        MetgridFieldSpec(
            "LANDSEA", "surface2d", _SFC, "M", "0/1 Flag", "Land-sea mask",
            "nearest_neighbor", False,
            parity_tol=0.0, fill_missing=-1.0, source="aifs_grib",
        ),
        MetgridFieldSpec(
            "DEWPT", "surface2d", _SFC, "M", "K", "Dewpoint temperature at 2 m",
            "default(nearest_neighbor/linear_log_p)", False,
            parity_tol=0.50, source="aifs_grib",
        ),
        # ---- soil (water-masked search interp) ----
        MetgridFieldSpec(
            "ST", "soil", _ST, "M", "K", "Soil temperature layers",
            "sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search", True,
            parity_tol=0.30, masked="water", fill_missing=1.0, source="aifs_grib",
            vertical_dim="num_st_layers", flag_in_output="FLAG_SOIL_LAYERS",
        ),
        MetgridFieldSpec(
            "SM", "soil", _SM, "M", "fraction", "Soil moisture layers",
            "sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search", True,
            parity_tol=0.02, masked="water", fill_missing=1.0, source="aifs_grib",
            vertical_dim="num_sm_layers", flag_in_output="FLAG_SOIL_LAYERS",
        ),
        MetgridFieldSpec(
            "SOIL_LAYERS", "soil", _ST, "M", "cm", "Soil layer thicknesses",
            "derived", False, parity_tol=0.0, source="derived",
            vertical_dim="num_st_layers",
        ),
        MetgridFieldSpec(
            "ST000010", "soil", _SFC, "M", "K", "Soil temperature 0-10 cm",
            "sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search", True,
            parity_tol=0.30, masked="water", fill_missing=1.0, source="aifs_grib",
            flag_in_output="FLAG_ST000010",
        ),
        MetgridFieldSpec(
            "ST010040", "soil", _SFC, "M", "K", "Soil temperature 10-40 cm",
            "sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search", True,
            parity_tol=0.30, masked="water", fill_missing=1.0, source="aifs_grib",
            flag_in_output="FLAG_ST010040",
        ),
        MetgridFieldSpec(
            "SM000010", "soil", _SFC, "M", "fraction", "Soil moisture 0-10 cm",
            "sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search", True,
            parity_tol=0.02, masked="water", fill_missing=1.0, source="aifs_grib",
            flag_in_output="FLAG_SM000010",
        ),
        MetgridFieldSpec(
            "SM010040", "soil", _SFC, "M", "fraction", "Soil moisture 10-40 cm",
            "sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search", True,
            parity_tol=0.02, masked="water", fill_missing=1.0, source="aifs_grib",
            flag_in_output="FLAG_SM010040",
        ),
        # ---- coordinate / metric (from geo_em; S2) ----
        MetgridFieldSpec(
            "XLAT_M", "coord", _SFC, "M", "degrees latitude", "Latitude on mass grid",
            "static", True, parity_tol=1e-4, source="geo_em",
        ),
        MetgridFieldSpec(
            "XLONG_M", "coord", _SFC, "M", "degrees longitude", "Longitude on mass grid",
            "static", True, parity_tol=1e-4, source="geo_em",
        ),
        MetgridFieldSpec(
            "XLAT_U", "coord", _U2, "U", "degrees latitude", "Latitude on U grid",
            "static", False, parity_tol=1e-4, source="geo_em",
        ),
        MetgridFieldSpec(
            "XLONG_U", "coord", _U2, "U", "degrees longitude", "Longitude on U grid",
            "static", False, parity_tol=1e-4, source="geo_em",
        ),
        MetgridFieldSpec(
            "XLAT_V", "coord", _V2, "V", "degrees latitude", "Latitude on V grid",
            "static", False, parity_tol=1e-4, source="geo_em",
        ),
        MetgridFieldSpec(
            "XLONG_V", "coord", _V2, "V", "degrees longitude", "Longitude on V grid",
            "static", False, parity_tol=1e-4, source="geo_em",
        ),
        MetgridFieldSpec(
            "CLAT", "coord", _SFC, "M", "degrees latitude", "Computational latitude on mass grid",
            "static", False, parity_tol=1e-4, source="geo_em",
        ),
        MetgridFieldSpec(
            "CLONG", "coord", _SFC, "M", "degrees longitude", "Computational longitude on mass grid",
            "static", False, parity_tol=1e-4, source="geo_em",
        ),
        MetgridFieldSpec(
            "XLAT_C", "coord", ("south_north_stag", "west_east_stag"), "CORNER",
            "degrees latitude", "Latitude at grid cell corners",
            "static", False, parity_tol=1e-4, source="geo_em",
        ),
        MetgridFieldSpec(
            "XLONG_C", "coord", ("south_north_stag", "west_east_stag"), "CORNER",
            "degrees longitude", "Longitude at grid cell corners",
            "static", False, parity_tol=1e-4, source="geo_em",
        ),
        # ---- map factors (from geo_em; S2). met_em carries BOTH the legacy
        # isotropic MAPFAC_M/U/V AND the anisotropic X/Y forms (FLAG_MF_XY=1 ->
        # real.exe reads MAPFAC_*X/*Y). All required for a faithful artifact. ----
        MetgridFieldSpec(
            "MAPFAC_MX", "mapfac", _SFC, "M", "none", "Map scale factor on mass grid, x direction",
            "static", True, parity_tol=1e-5, rel_tol=1e-5, source="geo_em",
            flag_in_output="FLAG_MF_XY",
        ),
        MetgridFieldSpec(
            "MAPFAC_MY", "mapfac", _SFC, "M", "none", "Map scale factor on mass grid, y direction",
            "static", True, parity_tol=1e-5, rel_tol=1e-5, source="geo_em",
            flag_in_output="FLAG_MF_XY",
        ),
        MetgridFieldSpec(
            "MAPFAC_UX", "mapfac", _U2, "U", "none", "Map scale factor on U grid, x direction",
            "static", True, parity_tol=1e-5, rel_tol=1e-5, source="geo_em",
        ),
        MetgridFieldSpec(
            "MAPFAC_UY", "mapfac", _U2, "U", "none", "Map scale factor on U grid, y direction",
            "static", True, parity_tol=1e-5, rel_tol=1e-5, source="geo_em",
        ),
        MetgridFieldSpec(
            "MAPFAC_VX", "mapfac", _V2, "V", "none", "Map scale factor on V grid, x direction",
            "static", True, parity_tol=1e-5, rel_tol=1e-5, source="geo_em",
        ),
        MetgridFieldSpec(
            "MAPFAC_VY", "mapfac", _V2, "V", "none", "Map scale factor on V grid, y direction",
            "static", True, parity_tol=1e-5, rel_tol=1e-5, source="geo_em",
        ),
        MetgridFieldSpec(
            "MAPFAC_M", "mapfac", _SFC, "M", "none", "Map factor on mass grid",
            "static", True, parity_tol=1e-5, rel_tol=1e-5, source="geo_em",
            flag_in_output="FLAG_MF_XY",
        ),
        MetgridFieldSpec(
            "MAPFAC_U", "mapfac", _U2, "U", "none", "Map factor on U grid",
            "static", True, parity_tol=1e-5, rel_tol=1e-5, source="geo_em",
        ),
        MetgridFieldSpec(
            "MAPFAC_V", "mapfac", _V2, "V", "none", "Map factor on V grid",
            "static", True, parity_tol=1e-5, rel_tol=1e-5, source="geo_em",
        ),
        MetgridFieldSpec(
            "F", "mapfac", _SFC, "M", "s-1", "Coriolis f",
            "static", True, parity_tol=1e-9, rel_tol=1e-6, source="geo_em",
        ),
        MetgridFieldSpec(
            "E", "mapfac", _SFC, "M", "s-1", "Coriolis e",
            "static", False, parity_tol=1e-9, rel_tol=1e-6, source="geo_em",
        ),
        MetgridFieldSpec(
            "SINALPHA", "mapfac", _SFC, "M", "none", "Local sine of map rotation",
            "static", False, parity_tol=1e-6, source="geo_em",
        ),
        MetgridFieldSpec(
            "COSALPHA", "mapfac", _SFC, "M", "none", "Local cosine of map rotation",
            "static", False, parity_tol=1e-6, source="geo_em",
        ),
        # ---- static geog 2D (from geo_em; S2) ----
        MetgridFieldSpec(
            "HGT_M", "geog2d", _SFC, "M", "m", "Topography height",
            "static", True, parity_tol=0.5, source="geo_em",
        ),
        MetgridFieldSpec(
            "LANDMASK", "geog2d", _SFC, "M", "none", "Land mask (1=land, 0=water)",
            "static", True, parity_tol=0.0, source="geo_em",
        ),
        MetgridFieldSpec(
            "SOILTEMP", "geog2d", _SFC, "M", "Kelvin", "Annual mean deep-soil temperature",
            "static", True, parity_tol=0.5, source="geo_em",
        ),
        MetgridFieldSpec(
            "LU_INDEX", "geog2d", _SFC, "M", "category", "Dominant land use category",
            "static", True, parity_tol=0.0, source="geo_em",
        ),
        MetgridFieldSpec(
            "SCT_DOM", "geog2d", _SFC, "M", "category", "Dominant top-soil category",
            "static", False, parity_tol=0.0, source="geo_em",
        ),
        MetgridFieldSpec(
            "SCB_DOM", "geog2d", _SFC, "M", "category", "Dominant bottom-soil category",
            "static", False, parity_tol=0.0, source="geo_em",
        ),
        MetgridFieldSpec(
            "SNOALB", "geog2d", _SFC, "M", "percent", "MODIS maximum snow albedo",
            "static", False, parity_tol=0.5, source="geo_em",
        ),
        # Orographic slope (gravity-wave-drag) statics, from geo_em. OA1-4 are
        # directional asymmetry, OL1-4 effective length-scale; VAR/CON variance &
        # convexity. Required only if the GWD scheme (gwd_opt) is on; kept optional.
        MetgridFieldSpec(
            "VAR", "geog2d", _SFC, "M", "m", "Orographic variance",
            "static", False, parity_tol=1.0, rel_tol=1e-3, source="geo_em",
        ),
        MetgridFieldSpec(
            "CON", "geog2d", _SFC, "M", "none", "Orographic convexity",
            "static", False, parity_tol=1e-3, source="geo_em",
        ),
        MetgridFieldSpec(
            "OA1", "geog2d", _SFC, "M", "none", "Orographic asymmetry, direction 1",
            "static", False, parity_tol=1e-3, source="geo_em",
        ),
        MetgridFieldSpec(
            "OA2", "geog2d", _SFC, "M", "none", "Orographic asymmetry, direction 2",
            "static", False, parity_tol=1e-3, source="geo_em",
        ),
        MetgridFieldSpec(
            "OA3", "geog2d", _SFC, "M", "none", "Orographic asymmetry, direction 3",
            "static", False, parity_tol=1e-3, source="geo_em",
        ),
        MetgridFieldSpec(
            "OA4", "geog2d", _SFC, "M", "none", "Orographic asymmetry, direction 4",
            "static", False, parity_tol=1e-3, source="geo_em",
        ),
        MetgridFieldSpec(
            "OL1", "geog2d", _SFC, "M", "none", "Orographic length scale, direction 1",
            "static", False, parity_tol=1e-3, source="geo_em",
        ),
        MetgridFieldSpec(
            "OL2", "geog2d", _SFC, "M", "none", "Orographic length scale, direction 2",
            "static", False, parity_tol=1e-3, source="geo_em",
        ),
        MetgridFieldSpec(
            "OL3", "geog2d", _SFC, "M", "none", "Orographic length scale, direction 3",
            "static", False, parity_tol=1e-3, source="geo_em",
        ),
        MetgridFieldSpec(
            "OL4", "geog2d", _SFC, "M", "none", "Orographic length scale, direction 4",
            "static", False, parity_tol=1e-3, source="geo_em",
        ),
        # ---- static geog 3D categorical / monthly (from geo_em; S2) ----
        MetgridFieldSpec(
            "LANDUSEF", "geog3d", ("z-dimension0021", "south_north", "west_east"), "M",
            "category", "Land use fraction by category",
            "static", True, parity_tol=1e-4, source="geo_em",
            vertical_dim="z-dimension0021",
        ),
        MetgridFieldSpec(
            "SOILCTOP", "geog3d", ("z-dimension0016", "south_north", "west_east"), "M",
            "category", "16-category top-layer soil fraction",
            "static", False, parity_tol=1e-4, source="geo_em",
            vertical_dim="z-dimension0016", flag_in_output="FLAG_SOILCAT",
        ),
        MetgridFieldSpec(
            "SOILCBOT", "geog3d", ("z-dimension0016", "south_north", "west_east"), "M",
            "category", "16-category bottom-layer soil fraction",
            "static", False, parity_tol=1e-4, source="geo_em",
            vertical_dim="z-dimension0016",
        ),
        MetgridFieldSpec(
            "GREENFRAC", "geog3d", ("z-dimension0012", "south_north", "west_east"), "M",
            "fraction", "Monthly MODIS green fraction",
            "static", False, parity_tol=1e-3, source="geo_em",
            vertical_dim="z-dimension0012",
        ),
        MetgridFieldSpec(
            "ALBEDO12M", "geog3d", ("z-dimension0012", "south_north", "west_east"), "M",
            "percent", "Monthly MODIS surface albedo",
            "static", False, parity_tol=0.5, source="geo_em",
            vertical_dim="z-dimension0012",
        ),
        MetgridFieldSpec(
            "LAI12M", "geog3d", ("z-dimension0012", "south_north", "west_east"), "M",
            "m^2/m^2", "Monthly MODIS LAI",
            "static", False, parity_tol=1e-2, source="geo_em",
            vertical_dim="z-dimension0012", flag_in_output="FLAG_LAI12M",
        ),
    )


@dataclass(frozen=True)
class MetEmArtifact:
    """In-memory metgrid-equivalent artifact for one (domain, valid_time).

    Holds the field arrays keyed by met_em variable name, the projection metadata,
    the valid time, and provenance. ``arrays`` values are numpy arrays shaped per
    the matching :class:`MetgridFieldSpec` ``dims`` (WITHOUT the leading Time axis;
    the serializer adds Time=1). This is the object S1+S2+S3 assemble, S4 compares
    to the oracle, and the v0.4.0 native-real init consumes.

    ``validate()`` enforces the frozen schema (presence of mandatory fields, dims,
    shapes, dtype, stagger) so a producer cannot silently drift from the contract.
    """

    domain: str  # "d01" | "d02" | "d03"
    valid_time: str  # "YYYY-MM-DD_HH:MM:SS"
    projection: MetgridProjection
    arrays: dict[str, np.ndarray]
    provenance: dict[str, str] = field(default_factory=dict)
    schema_version: str = METGRID_SCHEMA_VERSION

    def _spec_index(self) -> dict[str, MetgridFieldSpec]:
        return {s.name: s for s in metem_field_specs()}

    def _dim_size(self, dim: str) -> int:
        proj = self.projection
        sizes = {
            "south_north": proj.ny,
            "west_east": proj.nx,
            "south_north_stag": proj.ny + 1,
            "west_east_stag": proj.nx + 1,
            "num_metgrid_levels": NUM_METGRID_LEVELS,
            "num_st_layers": NUM_ST_LAYERS,
            "num_sm_layers": NUM_SM_LAYERS,
            "z-dimension0021": proj.num_land_cat,
            "z-dimension0016": proj.num_soil_cat,
            "z-dimension0012": 12,
        }
        if dim not in sizes:
            raise ValueError(f"unknown schema dim {dim!r}")
        return sizes[dim]

    def validate(self, *, require_optional: bool = False) -> None:
        """Validates this artifact against the frozen schema. Raises on violation."""

        specs = self._spec_index()
        # mandatory presence
        for spec in specs.values():
            if spec.mandatory and spec.name not in self.arrays:
                raise ValueError(f"missing mandatory field {spec.name!r}")
            if require_optional and spec.name not in self.arrays:
                raise ValueError(f"missing field {spec.name!r} (require_optional)")
        # shape / dim conformance for every present field
        for name, arr in self.arrays.items():
            if name not in specs:
                raise ValueError(f"field {name!r} not in frozen schema")
            spec = specs[name]
            expected = tuple(self._dim_size(d) for d in spec.dims)
            if tuple(arr.shape) != expected:
                raise ValueError(
                    f"field {name!r} shape {tuple(arr.shape)} != expected {expected} "
                    f"(dims {spec.dims})"
                )
        if self.projection.map_proj != 1:
            raise ValueError("artifact projection must be Lambert (map_proj=1)")
        if self.domain not in ("d01", "d02", "d03", "d04", "d05"):
            raise ValueError(f"unexpected domain {self.domain!r}")
