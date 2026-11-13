"""Host-only static land patches for per-nest WRF inputs (v0.5.0 item 0.3).

``patch_land`` writes NEW copies of ``wrfinput_dNN`` / ``wrflowinp_dNN`` with
land-use class, soil-class and soil-state patches whose dependent fields are
updated consistently or the patch is refused. It never repairs, clips or
creates variables, never touches terrain/base state (HGT goes through the
geo_em path, ``gpuwrf.io.geo_em_patch``) and imports no JAX: it is outside the
traced model program.

Dependent-field policy is mandatory. A land-use change requires an explicit
policy for every vegetation-climatology field present (``donor`` = copy from
the nearest unpatched cell of the target class; a value; or ``keep`` with a
recorded ``dependent_keep`` waiver). Land/sea flips additionally require the
full flip set; real.exe's measured water conventions are applied for the
fields whose water value is fixed (SMOIS=SH2O=1, ISLTYP=14, ALBBCK=SNOALB=0.08,
vegetation 0, SEAICE/CANWAT/SNOW 0), sea or land temperatures/soil water must
come from a donor or explicit values.

Compile impact (code-read, ``runtime/operational_mode.py`` NoahMPStatic content
digest and ``contracts/grid.py``): IVGTYP/LU_INDEX/ISLTYP/LANDMASK/XLAND/
LAKEMASK/SHDMAX are compile constants of the step program; SMOIS/TSLB/TSK
(land carry), VEGFRA (shdfac), TMN (tbot), SST and wrflowinp records are traced.
The protocol records which class each patch falls in.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from gpuwrf.io.field_override import _publish, _values, load_spec, sha256
from gpuwrf.io.netcdf_lock import Dataset

SCHEMA = "gpuwrf-static-patch-v1"
# NoahMPStatic / GridSpec content: changing any value builds a new step program.
BAKED_FIELDS = frozenset({"LU_INDEX", "IVGTYP", "ISLTYP", "LANDMASK", "XLAND", "LAKEMASK", "SHDMAX", "HGT"})
# Traced inputs/carry of the step program (same compiled program).
TRACED_FIELDS = frozenset({"SMOIS", "TSLB", "SH2O", "TSK", "TMN", "VEGFRA", "SST", "LAI", "CANWAT", "SNOW", "SNOWH", "SEAICE"})
# Read by neither the Noah-MP step nor its static bundle (kept consistent for CPU-WRF/provenance).
UNREAD_FIELDS = frozenset({"LANDUSEF", "SOILCTOP", "SOILCBOT", "SHDMIN", "SHDAVG", "ALBBCK", "SNOALB", "FRC_URB2D"})
VEG_FIELDS = ("SHDMAX", "SHDMIN", "SHDAVG", "VEGFRA", "LAI", "ALBBCK", "SNOALB")
AUX_VEG = ("VEGFRA", "ALBBCK", "LAI")
SEA_FIELDS = ("SST", "TSK", "TMN", "TSLB")
LAND_FIELDS = ("SMOIS", "TSLB", "TSK", "TMN")
SOIL_FIELDS = ("SMOIS", "TSLB")
# real.exe water values measured on WN3 0227 d01-d03 (every water cell).
WATER_CONVENTION = {"SMOIS": 1.0, "SH2O": 1.0, "ALBBCK": 0.08, "SNOALB": 0.08, "VEGFRA": 0.0, "SHDMAX": 0.0,
                    "SHDMIN": 0.0, "SHDAVG": 0.0, "LAI": 0.0, "SEAICE": 0.0, "CANWAT": 0.0, "SNOW": 0.0,
                    "SNOWH": 0.0, "LAKEMASK": 0.0, "FRC_URB2D": 0.0}
AUX_WATER = {"VEGFRA": 0.0, "ALBBCK": 0.08, "LAI": 0.0}
# Base state / atmosphere: always byte-identical (terrain only via geo_em + real.exe).
NEVER_TOUCHED = frozenset({"HGT", "PHB", "PH", "MUB", "MU", "PB", "P", "ALB", "AL", "T_INIT", "T", "THM", "U", "V", "W",
                           "QVAPOR", "XLAT", "XLONG", "MAPFAC_M", "VAR", "CON", "OA1", "OA2", "OA3", "OA4", "OL1",
                           "OL2", "OL3", "OL4", "VAR_SSO"})
WAIVABLE = frozenset({"dependent_keep", "field_ranges", "soil_porosity", "vegetation_order", "nest_coverage",
                      "nest_boundary_zone"})
# Hard physical bounds: never waivable (soil water fraction, temperatures).
HARD_RANGES = {"SMOIS": (0.0, 1.0), "SH2O": (0.0, 1.0), "TSLB": (180.0, 350.0), "TSK": (180.0, 350.0),
               "TMN": (180.0, 350.0), "SST": (180.0, 350.0)}
RANGES = {"VEGFRA": (0.0, 100.0), "SHDMAX": (0.0, 100.0),
          "SHDMIN": (0.0, 100.0), "SHDAVG": (0.0, 100.0), "LAI": (0.0, 15.0), "ALBBCK": (0.0, 1.0),
          "SNOALB": (0.0, 1.0)}


class StaticPatchError(ValueError):
    """A failed consistency check; ``report`` holds every check."""

    def __init__(self, report):
        self.report = report
        failed = [f"{c['nest']}:{c['name']}: {c['detail']}" for c in report["checks"] if c["status"] == "FAIL"]
        super().__init__("; ".join(failed) or "static patch refused")


def _keys(obj, allowed, label):
    if not isinstance(obj, Mapping) or set(obj) - set(allowed):
        raise ValueError(f"{label}: expected mapping with keys from {sorted(allowed)}")


def read_namelist_ints(path, names):
    """Minimal &domains integer-list reader (namelist.input or namelist.wps)."""
    text = re.sub(r"!.*", "", Path(path).read_text())
    out = {}
    for name in names:
        m = re.search(rf"^\s*{name}\s*=\s*([^\n/&]*)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            out[name] = [int(v) for v in re.split(r"[,\s]+", m.group(1).strip()) if v and re.fullmatch(r"-?\d+", v)]
    return out


# ----------------------------------------------------------------------------- masks
def _point_in_polygon(lat, lon, polygon):
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.all(np.isfinite(poly)):
        raise ValueError("polygon must be >= 3 finite [lat, lon] pairs")
    inside = np.zeros(lat.shape, bool)
    y, x = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        crosses = ((y[i] > lat) != (y[j] > lat))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x[j] - x[i]) * (lat - y[i]) / (y[j] - y[i]) + x[i]
        inside ^= crosses & (lon < xint)
        j = i
    return inside


def resolve_mask(description, grid, base):
    """Horizontal boolean mask from a named/geo/array description (list = intersection).

    ``grid`` holds 2-D ``lat``, ``lon``, ``landmask``, ``lu``. Names: ``all``, ``land``,
    ``water``, ``class:N``. Geo: ``{"polygon": [[lat, lon], ...]}``, ``{"bbox": [lat0, lat1, lon0, lon1]}``
    (cell centres). Arrays: boolean/0-1 literal or ``{"file", "variable"}`` (default ``mask``).
    """
    shape = grid["lat"].shape
    if isinstance(description, list) and description and not isinstance(description[0], (int, float, bool, list)):
        mask = np.ones(shape, bool)
        provenance = []
        for item in description:
            m, p = resolve_mask(item, grid, base)
            mask &= m
            provenance.append(p)
        return mask, {"kind": "intersection", "parts": provenance}
    if isinstance(description, str):
        if description == "all":
            return np.ones(shape, bool), {"kind": "named", "mask": "all"}
        if description in ("land", "water"):
            return (grid["landmask"] == (1 if description == "land" else 0)), {"kind": "named", "mask": description}
        m = re.fullmatch(r"class:(\d+)", description)
        if m:
            return grid["lu"] == int(m.group(1)), {"kind": "named", "mask": description}
        raise ValueError(f"unknown named mask {description!r}")
    if isinstance(description, Mapping) and "polygon" in description:
        _keys(description, {"polygon"}, "polygon mask")
        return _point_in_polygon(grid["lat"], grid["lon"], description["polygon"]), {"kind": "polygon", "polygon": description["polygon"]}
    if isinstance(description, Mapping) and "bbox" in description:
        _keys(description, {"bbox"}, "bbox mask")
        lat0, lat1, lon0, lon1 = (float(v) for v in description["bbox"])
        if not lat0 < lat1 or not lon0 < lon1:
            raise ValueError("bbox must be [lat0 < lat1, lon0 < lon1]")
        lat, lon = grid["lat"], grid["lon"]
        return (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1), {"kind": "bbox", "bbox": description["bbox"]}
    a, provenance = _values(description, base, "mask")
    a = np.asarray(a)
    if a.dtype.kind not in "biuf" or not np.all(np.isfinite(a)) or not np.all(np.isin(a, [0, 1])):
        raise ValueError("mask must contain only finite boolean/0/1 values")
    if a.shape == (1, *shape):
        a = a[0]
    if a.shape != shape:
        raise ValueError(f"mask shape {a.shape} != horizontal grid {shape}")
    return a.astype(bool), provenance


# ----------------------------------------------------------------------------- nest geometry
def nest_set_check(namelist, present):
    """Not-waivable: exactly nests 1..max_dom of the namelist are present."""
    md = (read_namelist_ints(namelist, ["max_dom"]).get("max_dom") or [None])[0]
    ok = md is not None and sorted(present) == list(range(1, md + 1))
    return {"nest": "all", "name": "nest_set", "status": "PASS" if ok else "FAIL",
            "detail": f"namelist max_dom={md}, nests present {sorted(present)} (not waivable)"}


def nest_geometry(grids, *, spec_bdy_width=5):
    """{nest: parent/ratio/offsets} from WRF attributes, verified against cell-centre lat/lon.

    Child mass cell (j, i) (0-based) lies in parent cell (j0 + j // r, i0 + i // r) with
    i0 = I_PARENT_START - 1. For odd r the middle child cell's centre is the parent centre.
    """
    geo, checks = {}, []
    for nest, g in grids.items():
        parent, ratio = g.get("parent_id"), g.get("ratio")
        if nest == 1:
            continue
        if parent is None or parent < 1 or parent == nest or parent not in grids:
            checks.append({"nest": f"d{nest:02d}", "name": "nest_parent_missing", "status": "FAIL",
                           "detail": f"parent d{(parent or 0):02d} of d{nest:02d} is not in the file set (or its geometry is unknown): "
                                     "parent/child checks cannot run (not waivable)"})
            continue
        i0, j0 = g["i_parent_start"] - 1, g["j_parent_start"] - 1
        ny, nx = g["lat"].shape
        pny, pnx = grids[parent]["lat"].shape
        ok = ratio >= 1 and i0 >= 0 and j0 >= 0 and i0 + nx // ratio <= pnx and j0 + ny // ratio <= pny and nx % ratio == 0 and ny % ratio == 0
        detail = f"parent d{parent:02d} ratio {ratio} start ({g['i_parent_start']},{g['j_parent_start']})"
        if ok and ratio % 2 == 1:
            mid = ratio // 2
            clat, clon = g["lat"][mid::ratio, mid::ratio], g["lon"][mid::ratio, mid::ratio]
            plat = grids[parent]["lat"][j0:j0 + clat.shape[0], i0:i0 + clat.shape[1]]
            plon = grids[parent]["lon"][j0:j0 + clat.shape[0], i0:i0 + clat.shape[1]]
            err = float(max(np.max(np.abs(clat - plat)), np.max(np.abs(clon - plon))))
            ok = err < 2e-3
            detail += f"; centre lat/lon max error {err:.2e} deg"
        checks.append({"nest": f"d{nest:02d}", "name": "nest_geometry", "status": "PASS" if ok else "FAIL", "detail": detail})
        if ok:
            geo[nest] = {"parent": parent, "ratio": ratio, "i0": i0, "j0": j0, "shape": (ny, nx), "bdy": int(spec_bdy_width)}
    return geo, checks


def child_footprint(g):
    """Child (j, i) -> parent (jp, ip) index arrays."""
    ny, nx = g["shape"]
    r = g["ratio"]
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return g["j0"] + jj // r, g["i0"] + ii // r


def nest_consistency(geo, requested, changed, *, label, waivers, zone_width=None):
    """Parent/child coverage checks for per-nest boolean cell maps ``requested/changed[nest][key]``.

    nest_coverage: a CHANGED parent cell whose r x r children are all child-interior must have at
    least one child cell REQUESTED with the same key (the child never sees parent statics there).
    nest_boundary_zone: a CHANGED child cell inside its boundary zone (spec_bdy_width, or
    ``zone_width``) needs the parent cell above it REQUESTED with the same key (the zone is forced
    from the parent). Maps missing for a nest count as all-False.
    """
    checks = []
    for nest, g in sorted(geo.items()):
        parent = g["parent"]
        ny, nx = g["shape"]
        r = g["ratio"]
        width = g["bdy"] if zone_width is None else zone_width
        jp, ip = child_footprint(g)
        interior = np.zeros((ny, nx), bool)
        interior[width:ny - width, width:nx - width] = True
        pny, pnx = ny // r, nx // r
        child_interior_parent = interior.reshape(pny, r, pnx, r).all(axis=(1, 3))
        keys = {k for m in (requested, changed) for d in (nest, parent) for k in m.get(d, {})}
        def get(maps, d, key, shape):
            m = maps.get(d, {}).get(key)
            return np.zeros(shape, bool) if m is None else m
        pshape = None
        for maps in (requested, changed):
            for m in maps.get(parent, {}).values():
                pshape = m.shape
        pshape = pshape or (g["j0"] + pny, g["i0"] + pnx)
        for key in sorted(keys, key=str):
            preq, pchg = get(requested, parent, key, pshape), get(changed, parent, key, pshape)
            creq, cchg = get(requested, nest, key, (ny, nx)), get(changed, nest, key, (ny, nx))
            under = pchg[g["j0"]:g["j0"] + pny, g["i0"]:g["i0"] + pnx]
            child_any = creq.reshape(pny, r, pnx, r).any(axis=(1, 3))
            orphan = under & child_interior_parent & ~child_any
            status = "PASS" if not orphan.any() else ("WAIVED" if "nest_coverage" in waivers else "FAIL")
            checks.append({"nest": f"d{nest:02d}", "name": "nest_coverage", "status": status, "key": str(key),
                           "detail": f"{label} {key}: {int(orphan.sum())} changed d{parent:02d} cells over the d{nest:02d} interior have no requested child cell"})
            zone = cchg & ~interior & ~preq[jp, ip]
            status = "PASS" if not zone.any() else ("WAIVED" if "nest_boundary_zone" in waivers else "FAIL")
            checks.append({"nest": f"d{nest:02d}", "name": "nest_boundary_zone", "status": status, "key": str(key),
                           "detail": f"{label} {key}: {int(zone.sum())} changed d{nest:02d} cells in the {width}-cell boundary zone lack the d{parent:02d} request above them",
                           "child_only_interior_cells": int((cchg & interior & ~preq[jp, ip]).sum())})
    return checks


# ----------------------------------------------------------------------------- donors
def _donor_index(candidates, targets):
    """Nearest candidate (grid-index distance) for each target cell -> (flat index, distance cells)."""
    from scipy.spatial import cKDTree
    cy, cx = np.nonzero(candidates)
    ty, tx = np.nonzero(targets)
    if ty.size == 0 or cy.size == 0:
        return None, None
    dist, idx = cKDTree(np.column_stack([cy, cx])).query(np.column_stack([ty, tx]))
    return (cy[idx], cx[idx]), dist


# ----------------------------------------------------------------------------- land patch
class _Nest:
    def __init__(self, path, aux_path):
        self.path, self.aux_path = path, aux_path
        with Dataset(path) as ds:
            ds.set_auto_mask(False)
            self.attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
            self.dims = {v: ds[v].dimensions for v in ds.variables}
            self.orig = {v: np.array(ds[v][:]) for v in ds.variables if v != "Times"}
        self.aux_orig = {}
        if aux_path is not None:
            with Dataset(aux_path) as ds:
                ds.set_auto_mask(False)
                self.aux_orig = {v: np.array(ds[v][:]) for v in ds.variables if v != "Times"}
        self.new = {k: v.copy() for k, v in self.orig.items()}
        self.aux_new = {k: v.copy() for k, v in self.aux_orig.items()}
        self.changed = {}      # field -> horizontal bool
        self.aux_changed = {}

    def grid(self, current=False):
        o = self.new if current else self.orig
        return {"lat": o["XLAT"][0], "lon": o["XLONG"][0], "landmask": o["LANDMASK"][0], "lu": o["LU_INDEX"][0],
                "parent_id": int(self.attrs.get("PARENT_ID", 0)), "ratio": int(self.attrs.get("PARENT_GRID_RATIO", 1)),
                "i_parent_start": int(self.attrs.get("I_PARENT_START", 1)), "j_parent_start": int(self.attrs.get("J_PARENT_START", 1))}

    def has(self, field):
        return field in self.new

    def set(self, field, cells, values, *, aux=False):
        """Set ``values`` (per changed cell: shape (n,) or (levels, n) or (records, n)) on cells."""
        store, changed = (self.aux_new, self.aux_changed) if aux else (self.new, self.changed)
        a = store[field]
        view = a[0] if (not aux and a.shape[0] == 1) else a
        if view.ndim == 2:
            view[cells] = np.asarray(values, view.dtype)
        else:
            view[:, cells[0], cells[1]] = np.asarray(values, view.dtype)
        mark = changed.setdefault(field, np.zeros(a.shape[-2:], bool))
        mark[cells] = True


def _cells(mask):
    return np.nonzero(mask)


def _value_columns(nest, field, spec, cells, base, *, aux=False):
    """Explicit policy values -> per-cell arrays."""
    store = nest.aux_new if aux else nest.new
    a = store[field]
    view = a[0] if (not aux and a.shape[0] == 1) else a
    n = cells[0].size
    if isinstance(spec, Mapping):
        arr, _ = _values(spec, base, field)
    else:
        arr = np.asarray(spec, dtype=np.float64)
    if arr.dtype.kind not in "fiu" or not np.all(np.isfinite(arr)):
        raise ValueError(f"{field}: explicit values must be finite numbers")
    if view.ndim == 2:
        if arr.ndim == 0:
            return np.full(n, arr, np.float64)
        if arr.shape == view.shape:
            return arr[cells]
    else:
        levels = view.shape[0]
        if arr.ndim == 0:
            return np.full((levels, n), arr, np.float64)
        if arr.shape == (levels,):
            return np.repeat(arr[:, None], n, axis=1)
        if arr.shape == view.shape:
            return arr[:, cells[0], cells[1]]
    raise ValueError(f"{field}: explicit values must be a scalar{', a per-level list' if view.ndim == 3 else ''} or the full field shape {view.shape}")


def _apply_policy(nest, field, policy, cells, donors, base, *, label):
    """Apply donor/value to wrfinput (and the wrflowinp mirror) for one field."""
    if policy == "donor":
        if donors is None:
            raise ValueError(f"{label} {field}: no donor cell available (give explicit values)")
        dy, dx = donors
        src = nest.orig[field][0] if nest.orig[field].shape[0] == 1 else nest.orig[field]
        vals = src[dy, dx] if src.ndim == 2 else src[:, dy, dx]
        nest.set(field, cells, vals)
        if field in nest.aux_orig and field in AUX_VEG + ("SST",):
            nest.set(field, cells, nest.aux_orig[field][:, dy, dx], aux=True)
        return
    vals = _value_columns(nest, field, policy, cells, base)
    nest.set(field, cells, vals)
    if field in nest.aux_orig and field in AUX_VEG + ("SST",):
        records = nest.aux_new[field].shape[0]
        nest.set(field, cells, np.repeat(vals[None, :], records, axis=0), aux=True)


def _landuse(nest, item, base, all_masks, waivers, records):
    _keys(item, {"nest", "kind", "mask", "class", "soil_class", "dependents"}, "landuse patch")
    attrs = nest.attrs
    label = f"d{item['nest']:02d} landuse"
    if str(attrs.get("MMINLU", "")) != "MODIFIED_IGBP_MODIS_NOAH" or int(attrs.get("NUM_LAND_CAT", 0)) != 21:
        raise ValueError(f"{label}: only MODIFIED_IGBP_MODIS_NOAH (21 classes) inputs are supported")
    target = item.get("class")
    if isinstance(target, bool) or not isinstance(target, int) or not 1 <= target <= 21:
        raise ValueError(f"{label}: class must be a MODIS category 1..21")
    iswater, islake, isice = int(attrs.get("ISWATER", 17)), int(attrs.get("ISLAKE", 21)), int(attrs.get("ISICE", 15))
    if target in (isice, islake):
        raise ValueError(f"{label}: target class {target} (glacier/lake) refused: init paths and glacier program "
                         "structure belong to real.exe/the lake model")
    mask, provenance = resolve_mask(item.get("mask", "all"), nest.grid(), base)
    lu0 = nest.orig["LU_INDEX"][0]
    lm0 = nest.orig["LANDMASK"][0]
    if np.any(mask & np.isin(lu0, [isice, islake])):
        raise ValueError(f"{label}: mask contains glacier/lake cells (classes {isice}/{islake}); refused")
    if np.any(mask & (nest.orig["SEAICE"][0] > 0)) if "SEAICE" in nest.orig else False:
        raise ValueError(f"{label}: mask contains sea-ice cells; refused")
    change = mask & (lu0 != target)
    to_water = target == iswater
    l2w = change & (lm0 == 1) & to_water
    w2l = change & (lm0 == 0) & (not to_water)
    l2l = change & (lm0 == 1) & (not to_water)
    soil_class = item.get("soil_class")
    if soil_class is not None and (isinstance(soil_class, bool) or not isinstance(soil_class, int) or not 1 <= soil_class <= 19
                                   or soil_class in (14, 16)):
        raise ValueError(f"{label}: soil_class must be a STAS land category 1..19 except 14 (water) and 16 (land ice)")
    if w2l.any() and soil_class is None:
        raise ValueError(f"{label}: water->land cells need soil_class")
    if to_water and soil_class is not None:
        raise ValueError(f"{label}: soil_class does not apply to a water target")
    # Required dependent fields per transition (only fields present in the file).
    required = set()
    if (l2l | w2l).any():
        required |= {f for f in VEG_FIELDS if nest.has(f)}
    if l2w.any():
        required |= {f for f in SEA_FIELDS if nest.has(f)}
    if w2l.any():
        required |= {f for f in LAND_FIELDS if nest.has(f)}
    pre_init = bool(np.all(nest.orig["SH2O"][:, :, lm0 == 1] == 0)) if nest.has("SH2O") else True
    if w2l.any() and nest.has("SH2O") and not pre_init:
        required.add("SH2O")
    deps = item.get("dependents")
    if isinstance(deps, str):
        if deps != "donor":
            raise ValueError(f"{label}: dependents shorthand must be 'donor'")
        deps = dict.fromkeys(required, "donor")
    if deps is None:
        deps = {}
    if not isinstance(deps, Mapping):
        raise ValueError(f"{label}: dependents must be 'donor' or a mapping field -> donor|keep|values")
    unknown = set(deps) - required
    missing = required - set(deps)
    if missing:
        raise ValueError(f"{label}: dependent fields need an explicit policy (donor | keep | values): {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label}: policies for fields that this transition does not use: {sorted(unknown)}")
    keeps = sorted(f for f, p in deps.items() if isinstance(p, str) and p == "keep")
    for f, p in deps.items():
        if isinstance(p, str) and p not in ("donor", "keep"):
            raise ValueError(f"{label} {f}: policy must be donor, keep or values")
    hard_keep = [f for f in keeps if f in SEA_FIELDS + LAND_FIELDS + ("SH2O",) and (l2w | w2l).any()]
    if hard_keep:
        raise ValueError(f"{label}: land/sea flips cannot keep {hard_keep} (the full flip set must be patched)")
    # Donors from the ORIGINAL file, excluding every land-use patch mask of this nest.
    eligible = ~all_masks
    donor_log = {}
    def donors_for(cells_mask, candidate):
        found, dist = _donor_index(candidate & eligible, cells_mask)
        return found, dist
    veg_donor, veg_dist = donors_for(l2l | w2l, (lu0 == target) & (lm0 == 1))
    sea_donor, sea_dist = donors_for(l2w, (lu0 == iswater) & (lm0 == 0))
    if veg_dist is not None:
        donor_log["vegetation"] = {"cells": int(veg_dist.size), "max_distance_cells": float(veg_dist.max()), "mean_distance_cells": float(veg_dist.mean())}
    if sea_dist is not None:
        donor_log["sea"] = {"cells": int(sea_dist.size), "max_distance_cells": float(sea_dist.max()), "mean_distance_cells": float(sea_dist.mean())}
    # Class fields.
    cells = _cells(change)
    nest.set("LU_INDEX", cells, np.full(cells[0].size, target))
    if nest.has("IVGTYP"):
        nest.set("IVGTYP", cells, np.full(cells[0].size, target))
    if nest.has("LANDUSEF"):
        onehot = np.zeros((nest.new["LANDUSEF"].shape[1], cells[0].size))
        onehot[target - 1] = 1.0
        nest.set("LANDUSEF", cells, onehot)
    # Land -> land.
    for sub, donor in ((l2l, veg_donor), (w2l, veg_donor)):
        if not sub.any():
            continue
        c = _cells(sub)
        d = None
        if donor is not None:
            sel = np.isin(np.ravel_multi_index(_cells(l2l | w2l), lu0.shape), np.ravel_multi_index(c, lu0.shape))
            d = (donor[0][sel], donor[1][sel])
        for f in sorted(f for f in VEG_FIELDS if f in deps and not (isinstance(deps[f], str) and deps[f] == "keep")):
            _apply_policy(nest, f, deps[f], c, d, base, label=label)
        if sub is w2l:
            for f in sorted(f for f in LAND_FIELDS if f in deps):
                _apply_policy(nest, f, deps[f], c, d, base, label=label)
            nest.set("LANDMASK", c, np.ones(c[0].size))
            nest.set("XLAND", c, np.ones(c[0].size))
            if nest.has("SH2O"):
                if "SH2O" in deps:
                    _apply_policy(nest, "SH2O", deps["SH2O"], c, d, base, label=label)
                else:
                    nest.set("SH2O", c, np.zeros((nest.new["SH2O"].shape[1], c[0].size)))
    if soil_class is not None and (l2l | w2l).any():
        c = _cells((l2l | w2l))
        nest.set("ISLTYP", c, np.full(c[0].size, soil_class))
        for f in ("SOILCTOP", "SOILCBOT"):
            if nest.has(f):
                onehot = np.zeros((nest.new[f].shape[1], c[0].size))
                if soil_class > onehot.shape[0]:
                    raise ValueError(f"{label}: soil_class {soil_class} outside {f} categories")
                onehot[soil_class - 1] = 1.0
                nest.set(f, c, onehot)
    # Land -> water: convention + sea temperatures.
    if l2w.any():
        c = _cells(l2w)
        for f in sorted(f for f in SEA_FIELDS if f in deps):
            _apply_policy(nest, f, deps[f], c, sea_donor, base, label=label)
        for f, v in WATER_CONVENTION.items():
            if nest.has(f):
                a = nest.new[f]
                levels = a.shape[1] if a.ndim == 4 else None
                nest.set(f, c, np.full((levels, c[0].size) if levels else c[0].size, v))
        for f, v in AUX_WATER.items():
            if f in nest.aux_orig:
                nest.set(f, c, np.full((nest.aux_new[f].shape[0], c[0].size), v), aux=True)
        nest.set("LANDMASK", c, np.zeros(c[0].size))
        nest.set("XLAND", c, np.full(c[0].size, 2.0))
        nest.set("ISLTYP", c, np.full(c[0].size, int(attrs.get("ISOILWATER", 14))))
        for f in ("SOILCTOP", "SOILCBOT"):
            if nest.has(f):
                onehot = np.zeros((nest.new[f].shape[1], c[0].size))
                onehot[int(attrs.get("ISOILWATER", 14)) - 1] = 1.0
                nest.set(f, c, onehot)
    transitions = {}
    for a, b in zip(lu0[change].astype(int), np.full(int(change.sum()), target)):
        transitions[f"{a}->{b}"] = transitions.get(f"{a}->{b}", 0) + 1
    records.append({"nest": item["nest"], "kind": "landuse", "class": target, "soil_class": soil_class,
                    "mask": provenance, "mask_cells": int(mask.sum()), "changed_cells": int(change.sum()),
                    "land_to_land": int(l2l.sum()), "land_to_water": int(l2w.sum()), "water_to_land": int(w2l.sum()),
                    "transitions": transitions, "dependents": {f: (p if isinstance(p, str) else "values") for f, p in sorted(deps.items())},
                    "kept_fields": keeps, "donors": donor_log,
                    "pre_noahmp_init_sh2o": pre_init})
    return ("landuse", target), mask, change, keeps


def _soil(nest, item, base, records):
    _keys(item, {"nest", "kind", "field", "op", "values", "file", "variable", "mask", "layers"}, "soil patch")
    field = item.get("field")
    label = f"d{item['nest']:02d} soil"
    if field not in SOIL_FIELDS:
        raise ValueError(f"{label}: field must be one of {SOIL_FIELDS} (SH2O is rebuilt by Noah-MP init)")
    op = item.get("op", "set")
    if op not in ("set", "scale", "add"):
        raise ValueError(f"{label}: op must be set, scale or add")
    if ("values" in item) == ("file" in item):
        raise ValueError(f"{label}: provide exactly one of values or file")
    # Soil patches see the land/sea mask AFTER land-use patches of the same spec (flipped cells count).
    mask, provenance = resolve_mask(item.get("mask", "land"), nest.grid(current=True), base)
    lm0 = nest.new["LANDMASK"][0]
    if np.any(mask & (lm0 != 1)):
        raise ValueError(f"{label}: soil patches apply to land cells only (mask has {int((mask & (lm0 != 1)).sum())} water cells)")
    current = nest.new[field][0].copy()
    levels = current.shape[0]
    layers = item.get("layers", list(range(levels)))
    if not isinstance(layers, list) or not layers or any(isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < levels for k in layers):
        raise ValueError(f"{label}: layers must be a list of 0-based soil levels < {levels}")
    spec = item["values"] if "values" in item else {"file": item["file"], **({"variable": item["variable"]} if "variable" in item else {})}
    arr, value_provenance = (_values(spec, base, field) if isinstance(spec, Mapping) else (np.asarray(spec, np.float64), {"kind": "literal", "values": spec}))
    if arr.dtype.kind not in "fiu" or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label}: values must be finite numbers")
    full = np.broadcast_to(arr, current.shape) if arr.shape in ((), current.shape, current.shape[1:]) else None
    if full is None and arr.shape == (len(layers),):
        full = np.zeros(current.shape)
        full[layers] = arr[:, None, None]
    if full is None:
        raise ValueError(f"{label}: values must be scalar, per-listed-layer, (ny,nx) or {current.shape}")
    old = current.astype(np.float64)
    new = full if op == "set" else old * full if op == "scale" else old + full
    sel = np.zeros(current.shape, bool)
    sel[layers] = mask
    out = current.copy()
    out[sel] = new[sel].astype(current.dtype)
    cells = _cells(mask)
    nest.set(field, cells, out[:, cells[0], cells[1]])
    changed = np.any(out != current, axis=0)
    records.append({"nest": item["nest"], "kind": "soil", "field": field, "op": op, "layers": layers,
                    "values": value_provenance, "mask": provenance, "mask_cells": int(mask.sum()),
                    "changed_cells": int(changed.sum())})
    return ("soil", field), mask, mask & changed, []


def _check(report, nest, name, ok, detail, waivers=(), **extra):
    status = "PASS" if ok else ("WAIVED" if name in waivers else "FAIL")
    report["checks"].append({"nest": nest, "name": name, "status": status, "detail": detail, **extra})


def _porosity_table():
    from gpuwrf.io.wrfinput_consistency import _porosity
    return _porosity(None)


def validate_nest(nest, label, report, waivers, keeps, *, allowed=None):
    """Consistency checks on the patched arrays of one nest (fail closed)."""
    n, o, a = nest.new, nest.orig, nest.attrs
    lu, lm = n["LU_INDEX"][0], n["LANDMASK"][0]
    changed_any = np.zeros(lu.shape, bool)
    for m in list(nest.changed.values()) + list(nest.aux_changed.values()):
        changed_any |= m
    iswater, islake = int(a.get("ISWATER", 17)), int(a.get("ISLAKE", 21))
    isoilwater = int(a.get("ISOILWATER", 14))
    if "IVGTYP" in n:
        _check(report, label, "lu_alias", np.array_equal(n["IVGTYP"][0], lu.astype(n["IVGTYP"].dtype)),
               "IVGTYP == LU_INDEX everywhere")
    xl_ok = bool(np.all(np.isin(lm, [0, 1])) and np.all(n["XLAND"][0] == 2 - lm))
    _check(report, label, "land_water_mask", xl_ok and not np.any(changed_any & (np.isin(lu, [iswater, islake]) != (lm == 0))),
           "LANDMASK in {0,1}, XLAND == 2 - LANDMASK, water/lake LU <=> LANDMASK == 0 on changed cells")
    if "LANDUSEF" in nest.changed:
        c = nest.changed["LANDUSEF"]
        f = n["LANDUSEF"][0][:, c]
        _check(report, label, "landusef", bool(np.all(f.argmax(0) + 1 == lu[c]) and np.allclose(f.sum(0), 1.0)),
               "patched LANDUSEF sums to 1 with argmax == LU_INDEX")
    soil = n["ISLTYP"][0]
    soil_ok = bool(np.all((soil >= 1) & (soil <= 19)) and not np.any(changed_any & ((soil == isoilwater) != (lm == 0))))
    for f in ("SOILCTOP", "SOILCBOT"):
        if f in nest.changed:
            c = nest.changed[f]
            soil_ok = soil_ok and bool(np.all(n[f][0][:, c].argmax(0) + 1 == soil[c]))
    _check(report, label, "soil_category", soil_ok, "ISLTYP 1..19, water soil (14) <=> LANDMASK == 0 and SOILCTOP/SOILCBOT argmax == ISLTYP on changed cells")
    _check(report, label, "dependent_keep", not keeps, f"kept vegetation-climatology fields at re-classed cells: {keeps}", waivers)
    # Flip completeness: every water-convention field equals the convention on land->water cells.
    flipped = lm != o["LANDMASK"][0]
    if flipped.any():
        to_water = flipped & (lm == 0)
        to_land = flipped & (lm == 1)
        bad = []
        for f, v in WATER_CONVENTION.items():
            if f in n and to_water.any():
                arr = n[f][0]
                vals = arr[:, to_water] if arr.ndim == 3 else arr[to_water]
                if not np.all(vals == np.asarray(v, arr.dtype)):
                    bad.append(f)
        for f in SEA_FIELDS:
            if f in n and to_water.any() and not np.all(nest.changed.get(f, np.zeros_like(flipped))[to_water]):
                bad.append(f + "(unpatched)")
        for f in LAND_FIELDS + VEG_FIELDS:
            if f in n and to_land.any() and f not in keeps and not np.all(nest.changed.get(f, np.zeros_like(flipped))[to_land]):
                bad.append(f + "(unpatched)")
        if "SST" in nest.aux_orig and to_water.any() and not np.all(nest.aux_changed.get("SST", np.zeros_like(flipped))[to_water]):
            bad.append("wrflowinp SST(unpatched)")
        _check(report, label, "flip_set", not bad, f"land/sea flip set complete ({int(to_water.sum())} to water, {int(to_land.sum())} to land); missing/wrong: {bad}")
    # Value ranges on changed cells (wrfinput and wrflowinp records): hard bounds never waivable.
    def out_of(table):
        bad = []
        for f, c in list(nest.changed.items()):
            if c.any() and (f in table or table is HARD_RANGES):
                arr = n[f][0]
                vals = arr[:, c] if arr.ndim == 3 else arr[c]
                if vals.dtype.kind == "f" and not np.all(np.isfinite(vals)):
                    bad.append(f + "(nonfinite)")
                elif f in table:
                    lo, hi = table[f]
                    if not np.all((vals >= lo) & (vals <= hi)):
                        bad.append(f)
        for f, c in nest.aux_changed.items():
            if c.any() and f in table:
                vals = nest.aux_new[f][:, c]
                lo, hi = table[f]
                if not (np.all(np.isfinite(vals)) and np.all((vals >= lo) & (vals <= hi))):
                    bad.append("wrflowinp " + f)
        return bad
    bad = out_of(HARD_RANGES)
    _check(report, label, "hard_ranges", not bad, f"changed values finite, soil water in [0,1], temperatures in [180,350] K (not waivable); violations: {bad}")
    bad = out_of(RANGES)
    _check(report, label, "field_ranges", not bad, f"changed values inside physical ranges; out of range: {bad}", waivers)
    veg_c = np.zeros(lu.shape, bool)
    for f in ("SHDMIN", "VEGFRA", "SHDMAX"):
        veg_c |= nest.changed.get(f, np.zeros(lu.shape, bool))
    if veg_c.any() and {"SHDMIN", "VEGFRA", "SHDMAX"} <= n.keys():
        smin, v, smax = n["SHDMIN"][0][veg_c], n["VEGFRA"][0][veg_c], n["SHDMAX"][0][veg_c]
        _check(report, label, "vegetation_order", bool(np.all((smin <= v + 1e-3) & (v <= smax + 1e-3))),
               "SHDMIN <= VEGFRA <= SHDMAX on changed cells", waivers)
    # Soil water vs porosity on changed land cells; SH2O <= SMOIS when SH2O is carried.
    soil_c = (nest.changed.get("SMOIS", np.zeros(lu.shape, bool)) | nest.changed.get("ISLTYP", np.zeros(lu.shape, bool))) & (lm == 1)
    if soil_c.any():
        porosity, provenance = _porosity_table()
        report.setdefault("soil_table", provenance)
        cap = porosity[np.clip(soil[soil_c].astype(int), 1, len(porosity) - 1)]
        sm = n["SMOIS"][0][:, soil_c]
        ok = bool(np.all((sm >= 0) & (sm <= cap[None, :])))
        if "SH2O" in n:
            sh = n["SH2O"][0][:, soil_c]
            ok_liquid = bool(np.all((sh >= 0) & (sh <= sm)))
            _check(report, label, "soil_liquid", ok_liquid, "0 <= SH2O <= SMOIS on changed land cells")
        _check(report, label, "soil_porosity", ok, f"0 <= SMOIS <= STAS porosity(ISLTYP) on {int(soil_c.sum())} changed land cells", waivers)
    # Untouched: base state, terrain, atmosphere and every field outside the patch's declared set.
    moved = [f for f in NEVER_TOUCHED if f in n and n[f].tobytes() != o[f].tobytes()]
    _check(report, label, "base_state_untouched", not moved, f"terrain/base state/atmosphere byte-identical; changed: {moved}")
    declared, aux_declared = (nest.changed, nest.aux_changed) if allowed is None else allowed
    stray = [f for f in n if f not in declared and n[f].tobytes() != o[f].tobytes()]
    stray += ["wrflowinp " + f for f in nest.aux_new if f not in aux_declared and nest.aux_new[f].tobytes() != nest.aux_orig[f].tobytes()]
    outside = []
    for f, c in nest.changed.items():
        diff = (n[f] != o[f]).reshape(-1, *lu.shape).any(axis=0)
        if np.any(diff & ~c):
            outside.append(f)
    _check(report, label, "untouched_elsewhere", not stray and not outside,
           f"only declared fields on declared cells change; stray fields {stray}, off-mask {outside}")


def compile_impact(fields):
    baked = sorted(set(fields) & BAKED_FIELDS)
    if baked:
        return {"class": "new_program", "baked_fields": baked,
                "note": "NoahMPStatic/GridSpec content changes: new compiled step program (cold compile + new AOT cache per variant) [I, code-read]"}
    unread = sorted(set(fields) & UNREAD_FIELDS)
    return {"class": "same_program", "baked_fields": [], "unread_fields": unread,
            "note": "only traced inputs (land carry, shdfac, tbot, lower boundary) change: same compiled program expected [I, code-read; GPU unverified]"}


def patch_land(input_dir, output_dir, spec, *, spec_base=None, namelist=None):
    """Publish NEW wrfinput/wrflowinp copies with checked static land patches.

    Output must not exist; originals are read-only; staging is removed on refusal. An empty
    spec copies every selected file byte for byte (no file is opened writable).
    """
    _keys(spec, {"patches", "waivers", "spec_bdy_width"}, "spec")
    entries = spec.get("patches", [])
    if not isinstance(entries, list):
        raise ValueError("patches must be a list")
    waivers = spec.get("waivers", {}) or {}
    if not isinstance(waivers, Mapping) or any(k not in WAIVABLE or not isinstance(v, str) or not v.strip() for k, v in waivers.items()):
        raise ValueError(f"waivers must map {sorted(WAIVABLE)} to nonempty reasons")
    root, output = Path(input_dir).resolve(), Path(output_dir).absolute()
    base = Path(spec_base or ".").resolve()
    if not root.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("input directory must exist and output must be new")
    files = {p.name: p for p in root.iterdir() if re.fullmatch(r"wrfinput_d[0-9]{2}|wrflowinp_d[0-9]{2}", p.name)}
    inputs = sorted(int(n[-2:]) for n in files if n.startswith("wrfinput_"))
    if not inputs:
        raise ValueError("input directory has no wrfinput_dNN")
    if entries and namelist is None:
        raise ValueError("non-empty land specs need --namelist (namelist.input): the nest set (max_dom) and "
                         "spec_bdy_width must be known, otherwise a missing child nest goes unchecked")
    by_nest = {}
    for item in entries:
        if not isinstance(item, Mapping) or item.get("kind") not in ("landuse", "soil"):
            raise ValueError("each patch needs kind landuse or soil (z0: gpuwrf patch z0; HGT: gpuwrf patch geo-em)")
        nest = item.get("nest")
        if isinstance(nest, bool) or not isinstance(nest, int) or nest not in inputs:
            raise ValueError(f"patch nest {nest!r} has no wrfinput_dNN in the input directory")
        by_nest.setdefault(nest, []).append(item)
    bdy = spec.get("spec_bdy_width")
    if bdy is None and namelist is not None:
        bdy = (read_namelist_ints(namelist, ["spec_bdy_width"]).get("spec_bdy_width") or [5])[0]
    bdy = 5 if bdy is None else int(bdy)
    report = {"schema": SCHEMA, "tool": "gpuwrf patch land", "spec": spec, "waivers": dict(waivers),
              "spec_bdy_width": bdy, "files": {}, "patches": [], "checks": [],
              "scope": "wrfinput/wrflowinp copies only; carry namelist/wrfbdy/tables from the source case"}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.patch-", dir=output.parent))
    try:
        nests, requested, changes = {}, {}, {}
        for d in inputs:
            aux = files.get(f"wrflowinp_d{d:02d}")
            if d in by_nest:
                nests[d] = _Nest(files[f"wrfinput_d{d:02d}"], aux)
        all_keeps = {}
        for d, items in sorted(by_nest.items()):
            nest = nests[d]
            landuse_masks = np.zeros(nest.orig["LU_INDEX"].shape[-2:], bool)
            for item in items:
                if item["kind"] == "landuse":
                    landuse_masks |= resolve_mask(item.get("mask", "all"), nest.grid(), base)[0]
            keeps = []
            for item in items:
                if item["kind"] == "landuse":
                    key, req, chg, kept = _landuse(nest, item, base, landuse_masks, waivers, report["patches"])
                else:
                    key, req, chg, kept = _soil(nest, item, base, report["patches"])
                for maps, cells in ((requested, req), (changes, chg)):
                    maps.setdefault(d, {})
                    maps[d][key] = maps[d].get(key, np.zeros(cells.shape, bool)) | cells
                keeps += kept
            all_keeps[d] = sorted(set(keeps))
            validate_nest(nest, f"d{d:02d}", report, waivers, all_keeps[d])
        if by_nest:
            grids = {}
            for d in inputs:
                path = files[f"wrfinput_d{d:02d}"]
                if d in nests:
                    grids[d] = nests[d].grid()
                else:
                    with Dataset(path) as ds:
                        ds.set_auto_mask(False)
                        grids[d] = {"lat": np.array(ds["XLAT"][0]), "lon": np.array(ds["XLONG"][0]),
                                    "parent_id": int(ds.getncattr("PARENT_ID")) if "PARENT_ID" in ds.ncattrs() else 0,
                                    "ratio": int(ds.getncattr("PARENT_GRID_RATIO")) if "PARENT_GRID_RATIO" in ds.ncattrs() else 1,
                                    "i_parent_start": int(ds.getncattr("I_PARENT_START")) if "I_PARENT_START" in ds.ncattrs() else 1,
                                    "j_parent_start": int(ds.getncattr("J_PARENT_START")) if "J_PARENT_START" in ds.ncattrs() else 1}
            geo, geo_checks = nest_geometry(grids, spec_bdy_width=bdy)
            report["checks"] += geo_checks
            report["checks"].append(nest_set_check(namelist, inputs))
            report["checks"] += nest_consistency(geo, requested, changes, label="patch", waivers=waivers)
        for name, source in sorted(files.items()):
            original_sha = sha256(source)
            target = staging / name
            shutil.copyfile(source, target)
            if sha256(target) != original_sha:
                raise ValueError(f"source changed while copying: {source}")
            d = int(name[-2:])
            aux = name.startswith("wrflowinp_")
            nest = nests.get(d)
            changed_fields = {}
            if nest is not None:
                store, orig, marks = (nest.aux_new, nest.aux_orig, nest.aux_changed) if aux else (nest.new, nest.orig, nest.changed)
                touched = [f for f in marks if store[f].tobytes() != orig[f].tobytes()]
                if touched:
                    with Dataset(target, "r+") as ds:
                        ds.set_auto_maskandscale(False)
                        for f in touched:
                            var = ds[f]
                            if any(k in var.ncattrs() for k in ("scale_factor", "add_offset", "_Unsigned")):
                                raise ValueError(f"{name}:{f}: packed variables are unsupported")
                            var[:] = store[f]
                        ds.sync()
                    with Dataset(target) as ds:
                        ds.set_auto_mask(False)
                        for f in touched:
                            if np.asarray(ds[f][:]).tobytes() != store[f].tobytes():
                                raise ValueError(f"{name}:{f}: stored values differ from the patched array")
                for f in touched:
                    diff = (store[f] != orig[f]).reshape(-1, *store[f].shape[-2:]).any(axis=0)
                    changed_fields[f] = {"cells_changed": int(diff.sum()),
                                         "max_abs_change": float(np.max(np.abs(store[f].astype(np.float64) - orig[f]))),
                                         "dtype": str(store[f].dtype)}
            report["files"][name] = {"source": str(source), "output": str(output / name),
                                     "sha256_before": original_sha, "sha256_after": sha256(target),
                                     "byte_identical": original_sha == sha256(target), "fields_changed": changed_fields}
        fields = {f for r in report["files"].values() for f in r["fields_changed"]}
        report["compile_impact"] = compile_impact(fields)
        glacier = [d for d, n in nests.items() if np.any(n.new["IVGTYP"] == int(n.attrs.get("ISICE", 15))) and "IVGTYP" in n.changed]
        if glacier:
            report["compile_impact"]["glacier_note"] = f"glacier cells present on re-classed nests {glacier}: glacier index partition unchanged (class 15 never a source/target)"
        report["passed"] = not any(c["status"] == "FAIL" for c in report["checks"])
        if not report["passed"]:
            raise StaticPatchError(report)
        if any(sha256(files[name]) != record["sha256_before"] for name, record in report["files"].items()):
            raise ValueError("source inputs changed during patch")
        (staging / "PATCH_PROTOCOL.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=_json) + "\n")
        _publish(staging, output)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


# ----------------------------------------------------------------------------- z0 (traced control)
def patch_z0(input_dir, basis_path, spec, output, *, spec_base=None):
    """Write a ``z0mvt`` surface-control file: basis x (factor on mask, 1 elsewhere) per nest.

    Same compiled program for any factors (Track P traced control). The basis must come from the
    SAME (possibly land-patched) case: its IVGTYP must equal the wrfinput IVGTYP.
    """
    from gpuwrf.surface_params.cli import expand_from_basis, load_basis
    from gpuwrf.surface_params.controls import save_controls

    _keys(spec, {"patches"}, "z0 spec")
    entries = spec.get("patches", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("z0 spec needs a nonempty patches list")
    output = Path(output)
    if output.exists():
        raise ValueError("output control file must be new")
    base = Path(spec_base or ".").resolve()
    basis, manifest = load_basis(basis_path)
    multipliers, records = {}, []
    for item in entries:
        _keys(item, {"nest", "mask", "factor"}, "z0 patch")
        nest, factor = item.get("nest"), item.get("factor")
        if isinstance(nest, bool) or not isinstance(nest, int):
            raise ValueError("z0 patch nest must be an integer")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not np.isfinite(factor) or factor <= 0:
            raise ValueError("z0 factor must be a positive finite number")
        domain = f"d{nest:02d}"
        path = Path(input_dir) / f"wrfinput_{domain}"
        with Dataset(path) as ds:
            ds.set_auto_mask(False)
            grid = {"lat": np.array(ds["XLAT"][0]), "lon": np.array(ds["XLONG"][0]),
                    "landmask": np.array(ds["LANDMASK"][0]), "lu": np.array(ds["LU_INDEX"][0])}
            ivgtyp = np.array(ds["IVGTYP"][0])
        key = f"{domain}/_ivgtyp"
        if key not in basis.files or not np.array_equal(basis[key], ivgtyp):
            raise ValueError(f"{domain}: basis IVGTYP differs from {path.name} (rebuild the basis on the patched case)")
        mask, provenance = resolve_mask(item.get("mask", "land"), grid, base)
        field = multipliers.setdefault(domain, {}).setdefault("z0mvt", np.ones(ivgtyp.shape))
        field[mask] *= float(factor)
        records.append({"nest": nest, "factor": float(factor), "mask": provenance, "cells": int(mask.sum())})
    controls = expand_from_basis(basis, manifest, multipliers)
    provenance = {"tool": "gpuwrf patch z0", "basis": str(Path(basis_path).resolve()), "basis_sha256": sha256(basis_path),
                  "input_dir": str(Path(input_dir).resolve()), "patches": records,
                  "compile_impact": "traced surface control: same compiled program as any other z0mvt values [M, Track P]"}
    saved = save_controls(output, controls, provenance=provenance)
    return {"output": str(output), "sha256": sha256(output), "manifest": saved, "patches": records}


LAND_PATCH_FIELDS = frozenset({"LU_INDEX", "IVGTYP", "LANDUSEF", "ISLTYP", "SOILCTOP", "SOILCBOT", "LANDMASK", "XLAND",
                               "SH2O", *VEG_FIELDS, *SEA_FIELDS, *LAND_FIELDS, *WATER_CONVENTION})
AUX_PATCH_FIELDS = frozenset({*AUX_VEG, "SST"})


def verify_patched(source_dir, patched_dir, *, namelist=None, spec_bdy_width=None, waivers=None):
    """Re-audit an edited case against its source (files only, no protocol trusted).

    Every differing cell counts as changed; fields outside the land-patch field set must be
    byte-identical. Same checks as ``patch_land`` (requested == changed for the nest checks).
    """
    waivers = dict(waivers or {})
    src, dst = Path(source_dir), Path(patched_dir)
    report = {"schema": SCHEMA, "tool": "gpuwrf patch verify", "source_dir": str(src.resolve()),
              "patched_dir": str(dst.resolve()), "checks": [], "files": {}, "waivers": waivers}
    if spec_bdy_width is None:
        spec_bdy_width = (read_namelist_ints(namelist, ["spec_bdy_width"]).get("spec_bdy_width") or [5])[0] if namelist else 5
    nests, changes = {}, {}
    for path in sorted(src.glob("wrfinput_d[0-9][0-9]")):
        d = int(path.name[-2:])
        aux = src / f"wrflowinp_d{d:02d}"
        if aux.exists() and not (dst / aux.name).exists():
            report["checks"].append({"nest": f"d{d:02d}", "name": "aux_present", "status": "FAIL",
                                     "detail": f"source has {aux.name} but the edited case does not (forcing records would be lost)"})
        nest = _Nest(path, aux if aux.exists() and (dst / aux.name).exists() else None)
        with Dataset(dst / path.name) as ds:
            ds.set_auto_mask(False)
            for v in nest.new:
                nest.new[v] = np.array(ds[v][:])
        if nest.aux_path is not None:
            with Dataset(dst / aux.name) as ds:
                ds.set_auto_mask(False)
                for v in nest.aux_new:
                    nest.aux_new[v] = np.array(ds[v][:])
        shape = nest.orig["LU_INDEX"].shape[-2:]
        for store, orig, marks in ((nest.new, nest.orig, nest.changed), (nest.aux_new, nest.aux_orig, nest.aux_changed)):
            for v in store:
                if store[v].shape != orig[v].shape or store[v].dtype != orig[v].dtype:
                    raise ValueError(f"d{d:02d} {v}: shape/dtype differs from the source")
                if store[v].tobytes() != orig[v].tobytes():
                    marks[v] = (store[v] != orig[v]).reshape(-1, *shape).any(axis=0)
        report["files"][path.name] = {"sha256_source": sha256(path), "sha256_patched": sha256(dst / path.name),
                                      "fields_changed": {v: int(m.sum()) for v, m in sorted(nest.changed.items())}}
        lu_changed = nest.changed.get("LU_INDEX", np.zeros(shape, bool))
        if lu_changed.any():
            # A re-classed land cell must carry climatology of its NEW class: an unchanged value is stale unless
            # some unchanged cell of that class in the source has exactly this value (donor copies are such values).
            land_lu = lu_changed & (nest.new["LANDMASK"][0] == 1)
            lu_new, lu_old = nest.new["LU_INDEX"][0], nest.orig["LU_INDEX"][0]
            stale = {}
            for f in VEG_FIELDS:
                if f not in nest.new:
                    continue
                vals, src_vals = nest.new[f][0], nest.orig[f][0]
                same = land_lu & ~nest.changed.get(f, np.zeros(shape, bool))
                bad = 0
                for cls in np.unique(lu_new[same]):
                    pool = src_vals[(lu_old == cls) & (nest.orig["LANDMASK"][0] == 1) & ~lu_changed]
                    sel = same & (lu_new == cls)
                    bad += int(np.count_nonzero(~np.isin(vals[sel], pool)))
                if bad:
                    stale[f] = bad
            ok = not stale
            report["checks"].append({"nest": f"d{d:02d}", "name": "dependents_updated",
                                     "status": "PASS" if ok else ("WAIVED" if "dependent_keep" in waivers else "FAIL"),
                                     "detail": f"{int(land_lu.sum())} re-classed land cells; stale vegetation values (unchanged and not "
                                               f"a value of the new class in the source) per field: {stale}"})
        validate_nest(nest, f"d{d:02d}", report, waivers, [], allowed=(LAND_PATCH_FIELDS, AUX_PATCH_FIELDS))
        nests[d] = nest
        changes[d] = {"any": np.any(np.stack([m for v, m in nest.changed.items()] or [np.zeros(shape, bool)]), axis=0)}
    if namelist is not None:
        report["checks"].append(nest_set_check(namelist, list(nests)))
    grids = {d: n.grid() for d, n in nests.items()}
    geo, geo_checks = nest_geometry(grids, spec_bdy_width=spec_bdy_width)
    report["checks"] += geo_checks
    report["checks"] += nest_consistency(geo, changes, changes, label="edit", waivers=waivers)
    report["passed"] = not any(c["status"] == "FAIL" for c in report["checks"])
    return report


__all__ = ["patch_land", "verify_patched", "patch_z0", "resolve_mask", "nest_geometry", "nest_consistency", "load_spec",
           "StaticPatchError", "compile_impact", "read_namelist_ints"]
