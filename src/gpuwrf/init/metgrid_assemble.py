"""metgrid-equivalent artifact assembly (v0.3.0 S3).

Drives the S1 forcing arrays + S2 static-geog arrays through the
``interp_metgrid`` kernels onto the target Lambert C-grid, builds the derived
fields (PRES, the surface metgrid level, soil layer packing), de/re-staggers the
wind components to the C-grid U/V points, and returns a validated
:class:`gpuwrf.init.metgrid_schema.MetEmArtifact`. This is the metgrid
equivalent's field-loop (``process_domain_module.F::process_domain``), minus the
GRIB decode (S1) and geo_em read (S2).

INTERFACES (frozen for S5 wiring)
---------------------------------
``assemble_met_em(domain, valid_time, projection, forcing, static, target_grid,
source_grid)`` is the single entry-point S5 calls.

* ``forcing`` (from S1): a :class:`ForcingFields` with the AIFS source arrays on
  the *source* 0.25-deg lat-lon grid (NOT yet interpolated). 3D fields are
  ``(nlev_iso, ny_src, nx_src)`` in met_em isobaric order (1000..50 hPa, i.e.
  index 0 = 1000 hPa); 2D surface fields are ``(ny_src, nx_src)``. The source
  grid orientation is described by ``source_grid``.
* ``static`` (from S2): a dict of static geog arrays ALREADY on the target grid
  (geo_em is produced on the target grid; metgrid copies it through). Keys are
  met_em variable names; arrays match the schema dims (no Time axis).
* ``target_grid`` (from S2): :class:`TargetGrid` with lat/lon arrays for each
  stagger (mass / U / V), shaped per the schema 2D dims.
* ``source_grid``: :class:`gpuwrf.init.interp_metgrid.LatLonSourceGrid`.

The source slab convention for the kernels is ``(nx_src, ny_src)`` (i=lon first,
j=lat second), so we transpose the ``(ny_src, nx_src)`` forcing arrays at the
boundary. Output arrays are ``(ny_tgt, nx_tgt)`` (met_em south_north, west_east).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gpuwrf.init import interp_metgrid as im
from gpuwrf.init.metgrid_schema import (
    ISOBARIC_LEVELS_PA,
    NUM_METGRID_LEVELS,
    NUM_SM_LAYERS,
    NUM_ST_LAYERS,
    SOIL_LAYER_DEPTHS_CM,
    MetEmArtifact,
    MetgridFieldSpec,
    MetgridProjection,
    metem_field_specs,
)

DEFAULT_MSGVAL = im.DEFAULT_MSGVAL


# =============================================================================
# Inputs from the sibling lanes (frozen for S5)
# =============================================================================
@dataclass
class ForcingFields:
    """AIFS forcing on the SOURCE 0.25-deg grid (S1's product, pre-interp).

    3D isobaric arrays: ``(13, ny_src, nx_src)`` in 1000..50 hPa order (index 0 =
    1000 hPa), matching ``ISOBARIC_LEVELS_PA``. 2D surface arrays:
    ``(ny_src, nx_src)``. ``landsea`` is the source land-sea mask (0=water,
    1=land) used as the soil-field interp mask. ``missing_value`` is the source
    fill for masked/absent cells (ungrib uses -1.e30).
    """

    # 3D isobaric (13 levels)
    t_iso: np.ndarray  # TT
    u_iso: np.ndarray  # UU
    v_iso: np.ndarray  # VV
    gh_iso: np.ndarray  # GHT (gpm == m, recon 2b: no /g)
    q_iso: np.ndarray  # SPECHUMD
    # 2D surface
    t2: np.ndarray
    u10: np.ndarray
    v10: np.ndarray
    q2: np.ndarray  # surface specific humidity (S1 derives from 2d dewpoint)
    psfc: np.ndarray
    pmsl: np.ndarray
    soilhgt: np.ndarray  # orog
    skintemp: np.ndarray
    landsea: np.ndarray
    dewpt: np.ndarray | None = None
    # soil (2 layers; named bands)
    st000010: np.ndarray | None = None
    st010040: np.ndarray | None = None
    sm000010: np.ndarray | None = None
    sm010040: np.ndarray | None = None
    missing_value: float = DEFAULT_MSGVAL


@dataclass
class TargetGrid:
    """Target Lambert C-grid lat/lon arrays (S2's product).

    Mass arrays ``(ny, nx)``; U arrays ``(ny, nx+1)``; V arrays ``(ny+1, nx)``.
    Degrees; longitudes in -180..180 (metgrid convention).
    """

    lat_m: np.ndarray
    lon_m: np.ndarray
    lat_u: np.ndarray
    lon_u: np.ndarray
    lat_v: np.ndarray
    lon_v: np.ndarray


# =============================================================================
# Field -> interp-method routing (the METGRID.TBL.ARW spec, via the schema)
# =============================================================================
def _chain_for(spec: MetgridFieldSpec) -> list[tuple[int, int]]:
    """Parses the schema's ``interp_option`` into the kernel chain. Derived /
    static specs yield an empty chain (handled out-of-band)."""

    return im.parse_interp_string(spec.interp_option)


def _stagger_latlon(stagger: str, tg: TargetGrid):
    if stagger == "U":
        return tg.lat_u, tg.lon_u
    if stagger == "V":
        return tg.lat_v, tg.lon_v
    return tg.lat_m, tg.lon_m


def _to_source_slab(arr2d: np.ndarray) -> np.ndarray:
    """(ny_src, nx_src) -> (nx_src, ny_src) for the kernel (i=lon first)."""

    return np.ascontiguousarray(np.asarray(arr2d).T)


def _interp_2d(
    src2d: np.ndarray,
    stagger: str,
    tg: TargetGrid,
    sg: im.LatLonSourceGrid,
    chain: list[tuple[int, int]],
    msgval: float,
    mask_src: np.ndarray | None = None,
    maskval: float | None = None,
    mask_relational: str | None = None,
) -> np.ndarray:
    """Interpolate one source 2D slab to the target grid for the given stagger.
    Returns ``(ny_tgt, nx_tgt)`` numpy fp64."""

    lat, lon = _stagger_latlon(stagger, tg)
    slab = _to_source_slab(src2d)
    mask = _to_source_slab(mask_src) if mask_src is not None else None
    out = im.interp_field_to_grid(
        slab, lat, lon, sg, chain, msgval,
        mask_array=mask, maskval=maskval, mask_relational=mask_relational,
    )
    return np.asarray(out, dtype=np.float64)


def _apply_target_mask(
    field: np.ndarray,
    target_landmask: np.ndarray,
    masked: str,
    fill_missing: float | None,
) -> np.ndarray:
    """Replicates process_domain_module.F target-point masking: for masked=water
    fields, target water points (LANDMASK==0) are set to fill_missing; for
    masked=land, target land points (LANDMASK==1) -> fill_missing. masked=both /
    none leave the interpolated field (SKINTEMP has no interp_land/water mask in
    this TBL, so 'both' is a no-op on the target). Any remaining msgval is
    backfilled with fill_missing too (the 'user asked to fill' path)."""

    out = np.array(field, dtype=np.float64)
    fm = 0.0 if fill_missing is None else float(fill_missing)
    if masked == "water":
        out = np.where(target_landmask == 0, fm, out)
    elif masked == "land":
        out = np.where(target_landmask == 1, fm, out)
    # backfill any leftover source-missing
    out = np.where(np.abs(out) > 1e29, fm, out)
    return out


# =============================================================================
# 3D atmosphere assembly (TT/UU/VV/GHT/SPECHUMD): isobaric interp + surface lvl
# =============================================================================
def _assemble_atmos3d(
    src_iso: np.ndarray,  # (13, ny_src, nx_src)
    src_sfc: np.ndarray,  # (ny_src, nx_src) surface-level value
    stagger: str,
    tg: TargetGrid,
    sg: im.LatLonSourceGrid,
    chain: list[tuple[int, int]],
    msgval: float,
) -> np.ndarray:
    """Builds a (NUM_METGRID_LEVELS, ny_tgt, nx_tgt) field: index 0 = surface
    (from src_sfc), indices 1..13 = the isobaric levels interpolated. Matches
    the met_em level order (recon: index0=surface, 1..13 = 1000..50 hPa)."""

    lat, _ = _stagger_latlon(stagger, tg)
    ny_t, nx_t = lat.shape
    out = np.empty((NUM_METGRID_LEVELS, ny_t, nx_t), dtype=np.float64)
    # surface level (index 0)
    out[0] = _interp_2d(src_sfc, stagger, tg, sg, chain, msgval)
    # isobaric levels: schema index 1..13 == src_iso 0..12 (1000..50 hPa)
    for lev in range(13):
        out[lev + 1] = _interp_2d(src_iso[lev], stagger, tg, sg, chain, msgval)
    return out


# =============================================================================
# main entry-point
# =============================================================================
def assemble_met_em(
    domain: str,
    valid_time: str,
    projection: MetgridProjection,
    forcing: ForcingFields,
    static: dict[str, np.ndarray],
    target_grid: TargetGrid,
    source_grid: im.LatLonSourceGrid,
    *,
    provenance: dict[str, str] | None = None,
) -> MetEmArtifact:
    """Assemble a metgrid-equivalent :class:`MetEmArtifact` for one
    (domain, valid_time). This is the function S5 calls. Static geog fields in
    ``static`` are copied through (already on the target grid); forcing fields
    are interpolated from the source grid via the METGRID.TBL.ARW interp chains.
    """

    specs = {s.name: s for s in metem_field_specs()}
    sg = source_grid
    tg = target_grid
    msg = forcing.missing_value
    arrays: dict[str, np.ndarray] = {}

    # target landmask drives soil-field target masking. Prefer the geo_em
    # LANDMASK (S2) on the target grid; fall back to the interpolated LANDSEA.
    if "LANDMASK" in static:
        target_landmask = np.asarray(static["LANDMASK"], dtype=np.float64)
    else:
        target_landmask = None

    # ---- 3D atmosphere: TT, UU, VV, GHT, SPECHUMD ----
    chain_atm = im.parse_interp_string("sixteen_pt+four_pt+average_4pt")
    arrays["TT"] = _assemble_atmos3d(forcing.t_iso, forcing.t2, "M", tg, sg, chain_atm, msg)
    arrays["UU"] = _assemble_atmos3d(forcing.u_iso, forcing.u10, "U", tg, sg, chain_atm, msg)
    arrays["VV"] = _assemble_atmos3d(forcing.v_iso, forcing.v10, "V", tg, sg, chain_atm, msg)
    arrays["GHT"] = _assemble_atmos3d(forcing.gh_iso, forcing.soilhgt, "M", tg, sg, chain_atm, msg)
    arrays["SPECHUMD"] = _assemble_atmos3d(forcing.q_iso, forcing.q2, "M", tg, sg, chain_atm, msg)

    # ---- surface 2D ----
    chain_4 = im.parse_interp_string("four_pt+average_4pt")
    arrays["PSFC"] = _interp_2d(forcing.psfc, "M", tg, sg, chain_4, msg)
    arrays["PMSL"] = _interp_2d(forcing.pmsl, "M", tg, sg, chain_atm, msg)
    arrays["SOILHGT"] = _interp_2d(forcing.soilhgt, "M", tg, sg, chain_4, msg)
    arrays["LANDSEA"] = _interp_2d(
        forcing.landsea, "M", tg, sg, im.parse_interp_string("nearest_neighbor"), msg,
    )
    # SKINTEMP: masked=both; no interp_land/water mask in TBL -> plain chain
    chain_soil = im.parse_interp_string(
        "sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search"
    )
    arrays["SKINTEMP"] = _interp_2d(forcing.skintemp, "M", tg, sg, chain_soil, msg)
    if forcing.dewpt is not None:
        arrays["DEWPT"] = _interp_2d(
            forcing.dewpt, "M", tg, sg, im.parse_interp_string("nearest_neighbor"), msg,
        )

    # ---- PRES (derived): index0=PSFC ; index1..13 = isobaric constants ----
    arrays["PRES"] = _build_pres(arrays["PSFC"])

    # ---- soil (water-masked search) ----
    if target_landmask is None:
        # fall back to interpolated LANDSEA rounded to 0/1
        target_landmask = np.round(arrays["LANDSEA"]).astype(np.float64)
    _assemble_soil(arrays, forcing, tg, sg, chain_soil, msg, target_landmask, specs)

    # ---- static geog: copy through from S2 (already on target grid) ----
    for name, spec in specs.items():
        if spec.source == "geo_em" and name in static:
            arrays[name] = np.asarray(static[name], dtype=np.float64)

    art = MetEmArtifact(
        domain=domain,
        valid_time=valid_time,
        projection=projection,
        arrays={k: v.astype(np.float32) for k, v in arrays.items()},
        provenance=provenance or {"assembler": "gpuwrf.init.metgrid_assemble"},
    )
    return art


def _build_pres(psfc: np.ndarray) -> np.ndarray:
    """PRES on the 14 metgrid levels: index 0 = PSFC (spatial), 1..13 = the
    isobaric level constants (recon: confirmed PRES[0]==PSFC, PRES[1:]==levels)."""

    ny, nx = psfc.shape
    pres = np.empty((NUM_METGRID_LEVELS, ny, nx), dtype=np.float64)
    pres[0] = psfc
    for lev, p in enumerate(ISOBARIC_LEVELS_PA):
        pres[lev + 1] = p
    return pres


def _assemble_soil(
    arrays: dict[str, np.ndarray],
    forcing: ForcingFields,
    tg: TargetGrid,
    sg: im.LatLonSourceGrid,
    chain_soil: list[tuple[int, int]],
    msg: float,
    target_landmask: np.ndarray,
    specs: dict[str, MetgridFieldSpec],
) -> None:
    """Soil fields ST/SM/ST000010/.../SOIL_LAYERS. The named-band 2D fields are
    interpolated with the water-mask (interp_mask=LANDSEA(0), equality) + target
    water masking + fill_missing=1.0. ST/SM 3D arrays stack bottom-band first
    (recon: ST[0]==ST010040, ST[1]==ST000010; SOIL_LAYERS=[40,10])."""

    landsea_src = forcing.landsea

    def interp_soil(src2d, name):
        spec = specs[name]
        raw = _interp_2d(
            src2d, "M", tg, sg, chain_soil, msg,
            mask_src=landsea_src, maskval=0.0, mask_relational=" ",
        )
        return _apply_target_mask(raw, target_landmask, spec.masked, spec.fill_missing)

    have_named = all(
        getattr(forcing, a) is not None
        for a in ("st000010", "st010040", "sm000010", "sm010040")
    )
    if have_named:
        arrays["ST000010"] = interp_soil(forcing.st000010, "ST000010")
        arrays["ST010040"] = interp_soil(forcing.st010040, "ST010040")
        arrays["SM000010"] = interp_soil(forcing.sm000010, "SM000010")
        arrays["SM010040"] = interp_soil(forcing.sm010040, "SM010040")

        ny, nx = arrays["ST000010"].shape
        # 3D stack: index0 = 010040 (40 cm band), index1 = 000010 (10 cm band)
        st = np.empty((NUM_ST_LAYERS, ny, nx), dtype=np.float64)
        st[0] = arrays["ST010040"]
        st[1] = arrays["ST000010"]
        arrays["ST"] = st
        sm = np.empty((NUM_SM_LAYERS, ny, nx), dtype=np.float64)
        sm[0] = arrays["SM010040"]
        sm[1] = arrays["SM000010"]
        arrays["SM"] = sm

        # SOIL_LAYERS thickness field: index0=40, index1=10 (recon: [40,10]),
        # constant across the horizontal grid.
        sl = np.empty((NUM_ST_LAYERS, ny, nx), dtype=np.float64)
        sl[0] = float(SOIL_LAYER_DEPTHS_CM[1])  # 40
        sl[1] = float(SOIL_LAYER_DEPTHS_CM[0])  # 10
        arrays["SOIL_LAYERS"] = sl
