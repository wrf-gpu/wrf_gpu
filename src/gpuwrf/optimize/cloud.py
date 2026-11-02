"""CPU cloud scoring for MSG3 facade copies; no JAX or model-kernel changes.

Native-cell CWP threshold -> area-weighted footprint occurrence (a spatial
fraction, never a cloud probability). MSG3 has flat centres, no row/column grid
or exact polygons. Centre assignment uses EPSG:6933, areas DX*DY/MAPFAC_M².
View geometry is limited to corrected geolocation; parallax is not applied.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gpuwrf.physics.surface_constants import R_D_OVER_CP

FORMAT = "alisios.msg3.v1"
LEGACY_MAX_OBSERVATIONS = 2048  # each old observation-space f64 matrix <=32MiB


def finite(value, name):
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError(f"{name}: masked/fill values")
    out = np.asarray(value, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{name}: nonfinite values")
    return out


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name}: expected positive finite number")
    return float(value)


def cwp(species, rho_dry, dz):
    """Signed ∑q*rho_dry*dz, g/m²; q kg/kg dry air. No clipping/repair."""
    rho, dz = finite(rho_dry, "dry density"), finite(dz, "layer thickness")
    if rho.ndim != 3 or dz.shape != rho.shape or (rho <= 0).any() or (dz <= 0).any():
        raise ValueError("density/dz require matching positive (level,y,x) arrays")
    if not species:
        raise ValueError("need at least one condensate species")
    total = np.zeros_like(rho)
    for name, values in species.items():
        values = finite(values, name)
        if values.shape != rho.shape:
            raise ValueError(f"{name}: condensate/density shape mismatch")
        total += values
    return finite(np.sum(total*rho*dz, axis=0)*1000., "CWP")


def _spatial():
    try:
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from pyproj import Transformer
    except ImportError as exc:
        raise ValueError("cloud scoring requires gpuwrf[optimize] (scipy, pyproj)") from exc
    return cKDTree, csr_matrix, Transformer


@dataclass
class Footprints:
    model_shape: tuple
    index: np.ndarray
    area: np.ndarray
    weight: np.ndarray
    cell_count: np.ndarray
    total_area: float
    max_distance_m: float

    def identity(self, lat, lon):
        """Bind cell assignment, areas and native pixel order across candidates."""
        import hashlib
        digest = hashlib.sha256(str(self.model_shape).encode())
        for value in (self.index, self.area, lat, lon):
            digest.update(np.ascontiguousarray(value).view(np.uint8))
        return digest.hexdigest()

    @classmethod
    def build(cls, lat, lon, cell_area, pixel_lat, pixel_lon, *, mask=None, max_distance_m=2263.2503):
        lat, lon, area = (finite(v, n) for v, n in
                          ((lat, "latitude"), (lon, "longitude"), (cell_area, "cell area")))
        plat, plon = finite(pixel_lat, "pixel latitude"), finite(pixel_lon, "pixel longitude")
        if lat.ndim != 2 or lon.shape != lat.shape or area.shape != lat.shape or (area <= 0).any():
            raise ValueError("model coordinates/areas require matching 2D grids and positive areas")
        if (plat.ndim != 1 or plon.shape != plat.shape or not plat.size or
                (abs(plat) > 90).any() or (abs(plon) > 180).any() or
                (abs(lat) > 90).any() or (abs(lon) > 180).any()):
            raise ValueError("MSG3 requires flat geographic pixel centres")
        limit = positive(max_distance_m, "max_distance_m")
        mask = np.ones(lat.shape, bool) if mask is None else np.asarray(mask)
        if mask.shape != lat.shape or mask.dtype != bool or not mask.any():
            raise ValueError("model region must be a nonempty boolean mass-grid mask")
        tree_cls, _, transformer = _spatial()
        project = transformer.from_crs(4326, 6933, always_xy=True)
        pixels = np.column_stack(project.transform(plon, plat))
        if len(np.unique(pixels, axis=0)) != len(pixels):
            raise ValueError("duplicate MSG3 centres")
        points = np.column_stack(project.transform(lon[mask], lat[mask]))
        distance, nearest = tree_cls(pixels).query(points, k=min(2, len(pixels)))
        if len(pixels) > 1:
            # Frozen rule for an exact tie: lowest flat MSG index.
            ties = np.isclose(distance[:, 0], distance[:, 1], rtol=0, atol=1e-8)
            if ties.any():
                tree = tree_cls(pixels)
                for i in np.flatnonzero(ties):
                    choices = tree.query_ball_point(points[i], distance[i, 0]+1e-8)
                    nearest[i, 0] = min(choices)
            distance, nearest = distance[:, 0], nearest[:, 0]
        index = np.full(lat.shape, -1, dtype=np.int64)
        index[mask] = np.where(distance <= limit, nearest, -1)
        use = index >= 0
        weights = np.bincount(index[use], weights=area[use], minlength=len(plat))
        finite(weights, "assigned area")
        finite(area[mask].sum(), "total model area")
        counts = np.bincount(index[use], minlength=len(plat))
        return cls(lat.shape, index.ravel(), area.ravel(), weights, counts, float(area[mask].sum()),
                   float(distance[distance <= limit].max()) if (distance <= limit).any() else 0.)

    def apply(self, values):
        """Area mean. Pass native-cell occurrence, not raw CWP."""
        values = finite(values, "model occurrence")
        if values.shape != self.model_shape or ((values < 0) | (values > 1)).any():
            raise ValueError("occurrence requires mass-grid values in [0,1]")
        return self.mean(values)

    def mean(self, values):
        """Signed area mean for diagnostic quantities on the SAME footprint."""
        values = finite(values, "footprint diagnostic")
        if values.shape != self.model_shape:
            raise ValueError("diagnostic/footprint shape mismatch")
        use = self.index >= 0
        sums = np.bincount(self.index[use], weights=values.ravel()[use]*self.area[use], minlength=len(self.weight))
        return finite(np.divide(sums, self.weight, out=np.zeros_like(sums), where=self.weight > 0), "footprint fraction")


def contingency(hits, false_alarms, misses, correct_negatives):
    h, f, m, c = [np.asarray(v, dtype=np.int64) for v in (hits, false_alarms, misses, correct_negatives)]
    n = h+f+m+c

    def ratio(a, b):
        out = np.divide(a, b, out=np.zeros(np.broadcast_shapes(np.shape(a), np.shape(b))), where=b != 0)
        return np.where(b != 0, out, None).tolist()

    random = np.divide((h+f).astype(float)*(h+m), n, out=np.zeros_like(n, dtype=float), where=n > 0)
    return {"hits": h.tolist(), "false_alarms": f.tolist(), "misses": m.tolist(),
            "correct_negatives": c.tolist(), "valid": n.tolist(),
            "frequency_bias": ratio(h+f, h+m), "csi": ratio(h, h+f+m),
            "ets": ratio(h-random, h+f+m-random)}


def _xyz(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))


def neighbourhoods(lat, lon, radii_km):
    """Sparse great-circle neighbourhoods on native flat centres; no invented grid."""
    tree_cls, sparse, _ = _spatial()
    tree = tree_cls(_xyz(lat, lon))
    out = {}
    if not radii_km or len(set(radii_km)) != len(radii_km):
        raise ValueError("need distinct FSS radii in km")
    for radius in radii_km:
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 <= radius <= 200:
            raise ValueError("FSS radii must be finite km in [0,200]")
        near = tree.query_ball_tree(tree, 2*np.sin(radius/(2*6371.0088)))
        lengths = np.array([len(v) for v in near])
        out[str(float(radius))] = sparse((np.ones(int(lengths.sum())), np.concatenate(near),
                                         np.r_[0, np.cumsum(lengths)]), shape=(len(lat), len(lat)))
    return out


def verify(occurrence, clm, daylight, weights, lat, lon, *, pixel_threshold=.5, radii_km=(0, 5, 10), regions=None):
    """Report categorical/spatial diagnostics + continuous residuals to binary CLM.

    Continuous loss is equal-pixel MSE of model spatial fraction against binary
    satellite occurrence; it is NOT a probabilistic score. Counts are unweighted.
    FSS uses area-weighted neighbour fractions and area-weighted valid centres.
    Undefined event ratios/centroids are null; empty support is an error.
    """
    values, area = finite(occurrence, "occurrence"), finite(weights, "footprint area")
    obs, day = np.asarray(clm), np.asarray(daylight)
    lat, lon = finite(lat, "pixel latitude"), finite(lon, "pixel longitude")
    if (values.ndim != 2 or obs.shape != values.shape or not np.isin(obs, (0, 1, 255)).all() or
            ((values < 0) | (values > 1)).any() or area.shape != values.shape[1:] or (area < 0).any() or
            lat.shape != area.shape or lon.shape != area.shape):
        raise ValueError("cloud score requires (slot,pixel) fractions, CLM0/1/255 and pixel coordinates/areas")
    finite(area.sum(), "total footprint area")
    if (abs(lat) > 90).any() or (abs(lon) > 180).any():
        raise ValueError("invalid geographic pixel centres")
    if day.dtype != bool or day.shape != values.shape:
        raise ValueError("daylight requires frozen boolean (slot,pixel) mask")
    pixel_threshold = positive(pixel_threshold, "pixel_threshold")
    if pixel_threshold > 1:
        raise ValueError("pixel_threshold must be in (0,1]")
    valid = (obs != 255) & day & (area[None, :] > 0)
    if not valid.any():
        raise ValueError("no valid daytime cloud support")
    yes, pred = obs == 1, values >= pixel_threshold
    counts = [np.sum(v, axis=0) for v in
              (pred & yes & valid, pred & ~yes & valid, ~pred & yes & valid, ~pred & ~yes & valid)]
    masks = {"all": np.ones(area.shape, bool)}
    for name, mask in sorted((regions or {}).items()):
        mask = np.asarray(mask)
        if name == "all" or not name or mask.dtype != bool or mask.shape != area.shape or not mask.any():
            raise ValueError("regions require named nonempty boolean pixel masks (all reserved)")
        masks[name] = mask
    active = area > 0
    active_area = area[active]
    matrices = neighbourhoods(lat[active], lon[active], radii_km)
    reports = {}
    xyz = _xyz(lat, lon)
    for name, region in masks.items():
        use = valid & region
        fss = {r: np.zeros(2) for r in matrices}
        displacement = []
        for p, o, v in zip(pred, yes, use):
            av, ap, ao = v[active], p[active], o[active]
            for radius, matrix in matrices.items():
                denominator = matrix@(av*active_area)
                pf = np.divide(matrix@(ap*av*active_area), denominator, out=np.zeros_like(active_area), where=denominator > 0)
                of = np.divide(matrix@(ao*av*active_area), denominator, out=np.zeros_like(active_area), where=denominator > 0)
                fss[radius] += [np.sum(active_area[av]*(pf[av]-of[av])**2), np.sum(active_area[av]*(pf[av]**2+of[av]**2))]
            a, b = p & v, o & v
            if a.any() and b.any():
                ca, cb = np.average(xyz[a], axis=0, weights=area[a]), np.average(xyz[b], axis=0, weights=area[b])
                if min(np.linalg.norm(ca), np.linalg.norm(cb)) < 1e-12:
                    continue  # antipodal sets have no defined centroid
                ca, cb = ca/np.linalg.norm(ca), cb/np.linalg.norm(cb)
                displacement.append(float(6371.0088*np.arctan2(np.linalg.norm(np.cross(ca, cb)), np.dot(ca, cb))))
        report = contingency(*(int(v[region].sum()) for v in counts))
        report.update(loss=float(np.mean((values[use]-yes[use])**2)) if use.any() else None,
                      fss={r: float(1-t[0]/t[1]) if t[1] else (1. if use.any() else None) for r, t in fss.items()},
                      centroid_displacement_km=float(np.mean(displacement)) if displacement else None,
                      displacement_slots=len(displacement))
        reports[name] = report
    report = {"loss": reports["all"]["loss"], "regions": reports, "pixels": contingency(*counts),
              "counts": {"total": int(values.size), "invalid_clm": int((obs == 255).sum()),
                         "night": int((~day).sum()), "uncovered": int((area == 0).sum()*len(values)),
                         "valid": int(valid.sum())},
              "pixel_threshold": pixel_threshold, "continuous_operator": "area fraction versus binary CLM, equal-pixel MSE",
              "fss_operator": "great-circle radius km, area-weighted valid neighbours/centres, pooled terms",
              "displacement_operator": "area-weighted spherical cloud centroids, mean km over defined slots"}
    return report, values[valid], yes[valid].astype(float), np.argwhere(valid)


def _volume(ds, index, name, units):
    var = ds.variables[name]
    if str(getattr(var, "units", "")).strip() not in units:
        raise ValueError(f"{name}: unsupported/missing units")
    if var.dimensions != ("Time", "bottom_top_stag" if name in ("PH", "PHB") else "bottom_top",
                          "south_north", "west_east"):
        raise ValueError(f"{name}: unsupported WRF dimensions")
    return finite(var[index], name)


def wrf_columns(ds, index, species=("QCLOUD", "QICE"), *, return_geometry=False):
    """Dry density p/(Rd*Tair*(1+qv/epsilon)); WRF q is per kg dry air."""
    def read(name, units):
        return _volume(ds, index, name, units)

    p, theta = read("P", {"Pa"})+read("PB", {"Pa"}), read("T", {"K"})+300.
    vapor = read("QVAPOR", {"kg kg-1", "kg/kg", "kg kg^-1"})
    if (p <= 0).any() or (theta <= 0).any() or (vapor < 0).any():
        raise ValueError("invalid WRF thermodynamic state")
    temperature = theta*(p/100000.)**R_D_OVER_CP
    rho = p/(287.*temperature*(1+vapor/(287./461.6)))
    height = (read("PH", {"m2 s-2", "m2 s^-2", "m^2 s^-2"})+
              read("PHB", {"m2 s-2", "m2 s^-2", "m^2 s^-2"}))/9.81
    dz = np.diff(height, axis=0)
    result = cwp({s: read(s, {"kg kg-1", "kg/kg", "kg kg^-1"}) for s in species}, rho, dz)
    return (result, rho, dz) if return_geometry else result


def channel_diagnostics(ds, index, op, rho, dz, columns, valid, regions):
    """Separate prognostic/PBL/actual held RRTMG inputs; never fill missing taps.

    Regional profiles preserve solver-input layer fractions/condensates without
    storing full 3D histories. RRTMG condensates are held from its last call;
    their column integral uses CURRENT history density/dz, explicitly a proxy.
    The actual call clock is unavailable in existing history and stays unknown.
    """
    masks = {"all": np.ones(len(op.weight), bool), **regions}

    def summarize(column, profiles):
        pixel = op.mean(column)
        occurrence = op.apply(column >= 10.)  # fixed bound process threshold
        results = {}
        for name, region in masks.items():
            use = valid & region & (op.weight > 0)
            if not use.any():
                results[name] = {"status": "UNAVAILABLE", "reason": "no common daytime area"}
                continue
            cells = op.index >= 0
            cells[cells] &= use[op.index[cells]]
            weights = np.where(cells, op.area, 0.)
            profile = {field: (np.einsum("kj,j->k", value.reshape((value.shape[0], -1)), weights)/weights.sum()).tolist()
                       for field, value in profiles.items()}
            results[name] = {"status": "AVAILABLE", "column_condensate_g_m2": float(np.average(pixel[use], weights=op.weight[use])),
                             "bound_cwp10_area_fraction": float(np.average(occurrence[use], weights=op.weight[use])),
                             "layer_area_means": profile}
        return results

    result = {"prognostic": {"status": "AVAILABLE", "regions": summarize(columns, {})}}
    definitions = {"pbl_subgrid": ("QC_BL", "QI_BL", "CLDFRA_BL"),
                   "radiative_active": ("QC_RAD", "QI_RAD", "QS_RAD", "QG_RAD", "CLDFRA")}
    for channel, names in definitions.items():
        present = {}
        for name in names:
            if name in ds.variables:
                value = _volume(ds, index, name, {"", "1"} if name.startswith("CLDFRA") else {"kg kg-1", "kg/kg", "kg kg^-1"})
                if value.shape != rho.shape:
                    raise ValueError(f"{name}: geometry shape mismatch")
                if name.startswith("CLDFRA") and ((value < 0) | (value > 1)).any():
                    raise ValueError(f"{name}: invalid layer fraction")
                present[name] = value
        missing = sorted(set(names)-set(present))
        result[channel] = {"status": "UNAVAILABLE" if missing else "AVAILABLE", "missing_fields": missing}
        if not missing:
            column = cwp({n: v for n, v in present.items() if not n.startswith("CLDFRA")}, rho, dz)
            result[channel]["regions"] = summarize(column, present)
        if channel == "radiative_active":
            result[channel].update(call_time="UNAVAILABLE: not recorded in existing history",
                time_semantics="held solver-input mixing ratios/fraction from latest radiation call",
                column_semantics="held condensate at current history density/dz; not call-time CWP")
    return result


def load_config(raw, resolve):
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs
    from .data import identity

    allowed = {"truth", "format", "daylight", "threshold_g_m2", "pixel_threshold", "species", "fss_radii_km",
               "max_distance_m", "model_region", "regions", "min_common_area", "superobs"}
    if not isinstance(raw, dict) or set(raw)-allowed or raw.get("format") != FORMAT:
        raise ValueError(f"cloud requires format={FORMAT}, truth path and frozen daylight slots")
    paths = truth_paths(raw["truth"], resolve)
    refuse_protected_inputs("cloud truth (stage ALISIOS facade)", paths)
    species = raw.get("species", ["QCLOUD", "QICE"])
    if (not isinstance(species, list) or not species or len(set(species)) != len(species) or
            any(s not in ("QCLOUD", "QICE", "QRAIN", "QSNOW", "QGRAUP", "QHAIL") for s in species)):
        raise ValueError("species must name distinct WRF condensates")
    region = raw.get("model_region", {})
    if not isinstance(region, dict) or set(region)-{"land_only", "max_height_m"}:
        raise ValueError("model_region supports land_only and max_height_m")
    if "land_only" in region and not isinstance(region["land_only"], bool):
        raise ValueError("land_only must be boolean")
    if "max_height_m" in region:
        positive(region["max_height_m"], "max_height_m")
    minimum = positive(raw.get("min_common_area", .9), "min_common_area")
    if not .9 <= minimum <= 1:
        raise ValueError("min_common_area requires [0.9,1]")
    threshold = positive(raw.get("pixel_threshold", .5), "pixel_threshold")
    if threshold > 1:
        raise ValueError("pixel_threshold must be in (0,1]")
    superobs = raw.get("superobs")
    if superobs is not None:
        _superobs_contract(superobs)
    config = {"format": FORMAT, "truth": [str(p) for p in paths], "truth_identity": [identity(p) for p in paths],
              "daylight": raw["daylight"], "species": species, "model_region": region,
              "threshold_g_m2": positive(raw.get("threshold_g_m2", 10.), "threshold_g_m2"),
              "pixel_threshold": threshold, "fss_radii_km": raw.get("fss_radii_km", [0, 5, 10]),
              "max_distance_m": positive(raw.get("max_distance_m", 2263.2503), "max_distance_m"),
              "regions": raw.get("regions", {}), "min_common_area": minimum, "superobs": superobs}
    truth = read_truth(config)
    if superobs is not None:
        _superobs_regions(truth)
    neighbourhoods(np.array([0.]), np.array([0.]), config["fss_radii_km"])  # validate scales without a full native graph
    return config


def truth_paths(value, resolve):
    """One native D18 file or several dates; never require an unbound merge copy."""
    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(v, (str, Path)) for v in values):
        raise ValueError("cloud truth requires a path or nonempty list of paths")
    paths = [resolve(v) for v in values]
    if len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ValueError("duplicate cloud truth file")
    return paths


def read_truth(config):
    """Native MSG3 four-key NPZ; exact epoch seconds and explicit day-slot IDs."""
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs
    from .data import identity

    paths = [Path(p) for p in config["truth"]]
    refuse_protected_inputs("cloud truth", paths)
    if [identity(p) for p in paths] != config["truth_identity"]:
        raise ValueError("cloud truth changed since study load")
    parts = [_read_native(p) for p in paths]
    lat, lon = parts[0][2:]
    if any(not np.array_equal(a, b) for part in parts[1:] for a, b in zip(part[2:], (lat, lon))):
        raise ValueError("MSG3 native grid/order differs across truth files")
    times = np.concatenate([part[0] for part in parts])
    clm = np.concatenate([part[1] for part in parts])
    order = np.argsort(times, kind="stable")
    times, clm = times[order], clm[order]
    duplicate = np.r_[False, np.diff(times) == 0]
    if any(not np.array_equal(clm[n], clm[n-1]) for n in np.flatnonzero(duplicate)):
        raise ValueError("conflicting duplicate MSG3 endpoint slot")
    times, clm = times[~duplicate], clm[~duplicate]
    daylight = np.zeros(clm.shape, bool)
    slots = config["daylight"]
    if not isinstance(slots, list) or not slots:
        raise ValueError("daylight requires frozen [{time: UTC ISO, pixel_ids: [...]|all}] slots")
    seen = set()
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != {"time", "pixel_ids"}:
            raise ValueError("daylight slot needs time and pixel_ids")
        time = pd.Timestamp(slot["time"])
        if time.tz is None or time.value % 10**9 or time.value//10**9 not in times or time.value in seen:
            raise ValueError("daylight slot must be distinct explicit-zone exact native time")
        seen.add(time.value)
        ids = slot["pixel_ids"]
        if ids == "all":
            ids = list(range(len(lat)))
        if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or
                any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(lat) for i in ids)):
            raise ValueError("daylight pixel_ids must be distinct native flat indices")
        daylight[np.searchsorted(times, time.value//10**9), ids] = True
    regions = {}
    if not isinstance(config["regions"], dict):
        raise ValueError("regions require {name: [native_pixel_ids]}")
    for name, ids in config["regions"].items():
        if (not name or name == "all" or not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or
                any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(lat) for i in ids)):
            raise ValueError("invalid region pixel IDs")
        regions[name] = np.isin(np.arange(len(lat)), ids)
    return {"times": pd.to_datetime(times, unit="s", utc=True), "clm": clm, "daylight": daylight,
            "lat": lat, "lon": lon, "regions": regions}


def _read_native(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"times", "cloud", "lat", "lon"}:
            raise ValueError("MSG3 requires exactly times/cloud/lat/lon")
        times, clm, lat, lon = (data[k] for k in ("times", "cloud", "lat", "lon"))
    if times.dtype != np.dtype("<i8") or times.ndim != 1 or not len(times) or (np.diff(times) <= 0).any() or (times % 900).any():
        raise ValueError("MSG3 times require increasing distinct int64 15-minute UTC epoch seconds")
    if (lat.dtype != np.dtype("<f4") or lon.dtype != np.dtype("<f4") or lat.ndim != 1 or lon.shape != lat.shape or
            not len(lat) or not np.isfinite(lat).all() or not np.isfinite(lon).all() or
            (abs(lat) > 90).any() or (abs(lon) > 180).any()):
        raise ValueError("MSG3 coordinates require finite float32 flat geographic centres")
    if clm.dtype != np.uint8 or clm.shape != (len(times), len(lat)) or not np.isin(clm, (0, 1, 255)).all():
        raise ValueError("MSG3 cloud requires uint8 (slot,pixel), 0/1/255")
    if len(np.unique(np.column_stack((lat, lon)), axis=0)) != len(lat):
        raise ValueError("duplicate MSG3 centres")
    return times, clm, lat, lon


def _inventory(paths):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.verify.wrfout import _times
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs

    paths = [Path(p) for p in paths]
    refuse_protected_inputs("cloud wrfout", paths)
    records, domains, issues = [], set(), set()
    for path in paths:
        with Dataset(path) as ds:
            domain = getattr(ds, "GRID_ID", None)
            issue = getattr(ds, "SIMULATION_START_DATE", getattr(ds, "START_DATE", None))
            if domain is None or issue is None:
                raise ValueError("cloud wrfout needs GRID_ID and initialization date")
            domains.add(domain)
            issues.add(issue)
            records.extend((t, path, i) for i, t in enumerate(_times(ds)))
    records.sort()
    if not records or len(domains) != 1 or len(issues) != 1 or len({r[0] for r in records}) != len(records):
        raise ValueError("select distinct wrfout times from one domain and initialization")
    return records, pd.Timestamp(next(iter(issues)).replace("_", "T"), tz="UTC")


def score_case(paths, config, issue, hours, *, before=None):
    """Stream WRF frames and match exact MSG3 slots only; no interpolation.

    Returns report,prediction,observation,keys, keys=(slot ISO,flat pixel index).
    Slots below frozen common-area minimum are excluded and explicitly counted.
    """
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.verify.wrfout import _field

    truth = read_truth(config)
    inventory, actual_issue = _inventory(paths)
    issue = pd.Timestamp(issue)
    if issue.tz is None or actual_issue != issue:
        raise ValueError("cloud forecast initialization disagrees with case or lacks timezone")
    end = issue+pd.Timedelta(hours=positive(hours, "hours"))
    cutoff = pd.Timestamp(before) if before is not None else end+pd.Timedelta(nanoseconds=1)
    if any(t < issue or t > end for t, _, _ in inventory):
        raise ValueError("cloud forecast outside declared horizon")
    slots = truth["times"]
    eligible = np.flatnonzero((slots > issue) & (slots <= end) & (slots < cutoff))
    records = {t: (path, index) for t, path, index in inventory if t > issue and t < cutoff}
    matches = [s for s in eligible if slots[s] in records]
    if any(truth["daylight"][s].any() and slots[s] not in records for s in eligible):
        raise ValueError("missing required exact daytime model frame; frozen support cannot shrink")
    if not matches:
        raise ValueError("no exact cloud time matches after initialization/date split")
    op, reference, values, coverages, channels = None, None, [], [], []
    day = truth["daylight"][matches].copy()
    below = 0
    for j, s in enumerate(matches):
        path, index = records[slots[s]]
        with Dataset(path) as ds:
            lat, lon, mf = (_field(ds, k, index) for k in ("XLAT", "XLONG", "MAPFAC_M"))
            dx, dy = positive(float(ds.DX), "DX"), positive(float(ds.DY), "DY")
            if (mf <= 0).any():
                raise ValueError("MAPFAC_M must be positive")
            mask = np.ones(lat.shape, bool)
            region = config["model_region"]
            if region.get("land_only"):
                mask &= _field(ds, "LANDMASK", index) == 1
            if "max_height_m" in region:
                mask &= _field(ds, "HGT", index) <= region["max_height_m"]
            grid = (lat, lon, mf, mask, dx, dy)
            if op is None:
                op = Footprints.build(lat, lon, dx*dy/mf**2, truth["lat"], truth["lon"], mask=mask,
                                      max_distance_m=config["max_distance_m"])
                reference = grid
            elif any(not np.array_equal(a, b) for a, b in zip(grid, reference)):
                raise ValueError("moving/changing cloud model grid unsupported")
            columns, rho, dz = wrf_columns(ds, index, config["species"], return_geometry=True)
            values.append(op.apply(columns >= config["threshold_g_m2"]))
            area = np.sum(op.weight[(truth["clm"][s] != 255) & day[j]])
            expected = np.sum(op.weight[day[j]])
            coverage = float(area/expected) if expected else 0.
            coverages.append(coverage)
            if day[j].any() and coverage < config["min_common_area"]:
                below += 1
                day[j] = False
            common = (truth["clm"][s] != 255) & day[j] & (op.weight > 0)
            channels.append({"time": slots[s].isoformat(), "daytime_pixels": int(truth["daylight"][s].sum()),
                "common_pixels": int(common.sum()), "channels": channel_diagnostics(ds, index, op, rho, dz, columns,
                                                                                      common, truth["regions"])})
    report, pred, obs, indices = verify(np.array(values), truth["clm"][matches], day, op.weight, truth["lat"], truth["lon"],
        pixel_threshold=config["pixel_threshold"], radii_km=config["fss_radii_km"], regions=truth["regions"])
    keys = [(slots[matches[t]].isoformat(), int(p)) for t, p in indices]
    report["matching"] = {"eligible_slots": len(eligible), "matched_slots": len(matches),
                          "unmatched_slots": len(eligible)-len(matches), "below_common_area_slots": below,
                          "common_area_fraction": coverages, "minimum_common_area": config["min_common_area"],
                          "original_night_pixels": int((~truth["daylight"][matches]).sum())}
    report["footprints"] = {"operator": "nearest centre EPSG:6933; native CWP threshold then area-weighted occurrence",
        "area_role": "assigned model area, not satellite polygon area", "parallax": "not applied",
        "view_geometry": "corrected geolocation only", "cell_count": op.cell_count.tolist(),
        "assigned_area_m2": op.weight.tolist(), "selected_model_area_m2": op.total_area,
        "max_centre_distance_m": op.max_distance_m, "threshold_g_m2": config["threshold_g_m2"],
        "sha256": op.identity(truth["lat"], truth["lon"])}
    report["channel_diagnostics"] = channels
    return report, pred, obs, keys


def guard_observations(count, operator="superobservations"):
    """Memory capacity only; this never grants statistical/operator admission."""
    from gpuwrf.surface_params import eki
    available = bool(getattr(eki, "ENSEMBLE_SPACE_UPDATE", False))
    if operator not in ("superobservations", "native"):
        raise ValueError("cloud EKI observation_operator must be superobservations or native")
    if operator == "native" and not available:
        raise ValueError("full-native cloud EKI vectors are refused until ensemble-space EKI merges; use admitted superobs")
    if count > LEGACY_MAX_OBSERVATIONS and not available:
        raise ValueError(f"cloud EKI m={count} exceeds legacy safe cap {LEGACY_MAX_OBSERVATIONS}; "
                         "ensemble-space EKI must merge before this admitted scope can run")


def _superobs_contract(spec):
    if (not isinstance(spec, dict) or set(spec) != {"correlation_length_km", "contract"} or
            not isinstance(spec["contract"], str) or not spec["contract"].strip()):
        raise ValueError("cloud EKI requires admitted superobs.correlation_length_km and contract")
    positive(spec["correlation_length_km"], "superobs correlation_length_km")
    return spec


def require_correlation_contract(config, operator="superobservations"):
    """ALISIOS P1.5 is separate from memory capacity, including direct callers.

    No admitted native correlation contract/reader has been delivered. A scalar
    sigma (even beside a length/reference label) is not a bound native Gamma.
    Keep native fitting closed until that correlation operator is implemented;
    the existing superobservation operator actually applies its frozen length.
    """
    if operator == "native":
        raise ValueError("native correlated MSG EKI vectors require the admitted ALISIOS correlation contract "
                         "and its bound operator; ensemble-space EKI removes only the memory limit. "
                         "Use admitted superobservations; native correlation admission is pending")
    if config is not None:
        _superobs_contract(config.get("superobs"))


def _superobs_regions(truth):
    region_names = ["outside_registered_regions", *sorted(truth["regions"])]
    region = np.zeros(len(truth["lat"]), np.int64)
    occupied = np.zeros(len(region), bool)
    for n, name in enumerate(region_names[1:], 1):
        mask = truth["regions"][name]
        if (occupied & mask).any():
            raise ValueError("superobs requires disjoint registered regions; overlapping pixels cannot be counted twice")
        region[mask] = n
        occupied |= mask
    return region_names, region


def superobserve(report, pred, obs, keys, config, truth):
    """Fixed region × EPSG6933 correlation-length blocks at each native time.

    Both sides use the SAME assigned model-area weights. A pixel enters one
    observation only. Sigma is NOT divided by sqrt(pixel count): those pixels
    are correlated, and the external contract supplies representation error.
    """
    require_correlation_contract(config)
    spec = config["superobs"]
    if not keys:
        raise ValueError("no common superobservation support")
    length = positive(spec["correlation_length_km"], "superobs correlation_length_km")*1000.
    _, _, transformer = _spatial()
    x, y = transformer.from_crs(4326, 6933, always_xy=True).transform(truth["lon"], truth["lat"])
    block_coordinates = finite(np.floor(np.column_stack((x, y))/length), "superobs block coordinates")
    if (abs(block_coordinates) > np.iinfo(np.int64).max//2).any():
        raise ValueError("superobs correlation length too small for stable block coordinates")
    blocks = block_coordinates.astype(np.int64)
    region_names, region = _superobs_regions(truth)
    times = sorted({k[0] for k in keys})
    time_id = {t: n for n, t in enumerate(times)}
    pixels = np.fromiter((k[1] for k in keys), dtype=np.int64, count=len(keys))
    codes = np.column_stack((np.fromiter((time_id[k[0]] for k in keys), dtype=np.int64, count=len(keys)),
                             region[pixels], blocks[pixels]))
    groups, inverse = np.unique(codes, axis=0, return_inverse=True)
    guard_observations(len(groups))
    weights = finite(report["footprints"]["assigned_area_m2"], "superobs footprint weights")[pixels]
    total = np.bincount(inverse, weights=weights, minlength=len(groups))
    count = np.bincount(inverse, minlength=len(groups))
    if (total <= 0).any():
        raise ValueError("superobs has no common positive footprint area")
    prediction = finite(np.bincount(inverse, weights=weights*pred)/total, "superobs prediction")
    observation = finite(np.bincount(inverse, weights=weights*obs)/total, "superobs truth")
    output_keys = [(f"r{r}:{region_names[r]}:x{bx}:y{by}", times[t], "cloud") for t, r, bx, by in groups]
    report["superobservations"] = {"operator": "disjoint region x EPSG6933 correlation-length block x native time",
        "correlation_length_km": spec["correlation_length_km"], "contract": spec["contract"],
        "native_samples": len(pred), "observations": len(groups), "uncertainty": "supplied sigma, no sqrt(n) reduction",
        "pixel_counts": count.tolist(), "assigned_area_m2": total.tolist(), "keys": output_keys}
    return prediction, observation, output_keys


def collect(members_paths, config, issue, hours, *, before=None, sigma=1., reports=None, operator="superobservations"):
    """Default EKI superobservations; exact shared native support and footprint."""
    guard_observations(0, operator)
    require_correlation_contract(config, operator)
    sigma = positive(sigma, "cloud sigma")
    if not members_paths:
        raise ValueError("cloud collector requires members")
    rows, truth, keys, footprint = [], None, None, None
    observation_keys = None
    native_truth = read_truth(config) if operator == "superobservations" else None
    for paths in members_paths:
        report, pred, obs, current = score_case(paths, config, issue, hours, before=before)
        if keys is not None and (current != keys or not np.array_equal(obs, truth)):
            raise ValueError("cloud support/truth differs between members")
        current_footprint = report["footprints"]["sha256"]
        if footprint is not None and current_footprint != footprint:
            raise ValueError("cloud footprint differs between members")
        truth, keys, footprint = obs, current, current_footprint
        if operator == "superobservations":
            pred, obs, current = superobserve(report, pred, obs, current, config, native_truth)
        else:
            current = [(p, t, "cloud") for t, p in current]
            guard_observations(len(pred), operator)
        if observation_keys is not None and current != observation_keys:
            raise ValueError("superobs support differs between members")
        observation_keys = current
        rows.append(pred)
        if reports is not None:
            reports.append(report)
        observations = obs
    return np.array(rows), observations, np.full(observations.shape, sigma), observation_keys


def evaluate(study, fetch, *, prefetch=None, on_round=None):
    """Discrete successive halving with the same cloud collector as EKI."""
    import json

    train = [c for c in study["cases"] if c["split"] == "train"]
    test = [c for c in study["cases"] if c["split"] == "test"]
    candidates, cache, rounds = study["candidates"], {}, []
    provenance = list(study["inputs"])

    def get(candidate, cases):
        missing = [c for c in cases if (candidate["id"], c["id"]) not in cache]
        results = fetch(candidate, missing) if missing else {}
        for case in missing:
            paths, audit, identities = results[case["id"]]
            result = score_case(paths, study["cloud"], case["issue"], study["hours"],
                                before=study["test_start"] if case["split"] == "train" else None)
            cache[candidate["id"], case["id"]] = result
            provenance.append({"candidate": candidate["id"], "case": case["id"], "extraction": audit, "inputs": identities})
        return [cache[candidate["id"], c["id"]] for c in cases]

    def score(results):
        pred, truth = np.concatenate([r[1] for r in results]), np.concatenate([r[2] for r in results])
        return {"loss": float(np.mean((pred-truth)**2))/study["scales"]["cloud"]**2,
                "common_rows": len(pred), "cloud_cases": [r[0] for r in results]}

    for n, spec in enumerate(study["rounds"]):
        dates = train[:spec["train_dates"]]
        pairs = [(c, d) for c in candidates for d in dates if (c["id"], d["id"]) not in cache]
        if prefetch is not None and pairs:
            prefetch(pairs)
        results = {c["id"]: get(c, dates) for c in candidates}
        support = [[r[3] for r in result] for result in results.values()]
        if any(s != support[0] for s in support):
            raise ValueError("cloud support differs across discrete candidates")
        footprints = [[r[0]["footprints"]["sha256"] for r in result] for result in results.values()]
        if any(f != footprints[0] for f in footprints):
            raise ValueError("cloud footprint differs across discrete candidates")
        scores = sorted(({"id": c["id"], **score(results[c["id"]])} for c in candidates), key=lambda s: (s["loss"], s["id"]))
        keep = spec.get("keep", max(1, math.ceil(len(scores)/2)))
        record = {"round": n, "train_dates": len(dates), "dates": [d["id"] for d in dates], "scores": scores,
                  "common_rows": scores[0]["common_rows"], "keep": keep, "survivors": [s["id"] for s in scores[:keep]]}
        rounds.append(record)
        if on_round is not None:
            on_round(record)
        candidates = [c for c in candidates if c["id"] in record["survivors"]]
    winner = next(c for c in study["candidates"] if c["id"] == rounds[-1]["scores"][0]["id"])
    pairs = [(winner, d) for d in test if (winner["id"], d["id"]) not in cache]
    if prefetch is not None and pairs:
        prefetch(pairs)
    held = get(winner, test)
    rng = np.random.default_rng(0)
    losses = np.array([np.sum((r[1]-r[2])**2) for r in held])
    counts = np.array([len(r[1]) for r in held])
    selected = rng.integers(len(held), size=(400, len(held)))
    boot = losses[selected].sum(1)/counts[selected].sum(1)/study["scales"]["cloud"]**2
    report = {"schema": "gpuwrf.optimize.result.v1", "region": study["region"], "objective": "cloud", "recommended": winner,
              "rounds": rounds, "held_out": {**score(held), "uncertainty": {"method": "issue-date block bootstrap, seed0",
                             "interval95": np.quantile(boot, [.025, .975]).tolist(), "dates": len(held)}},
              "budget_case_hours": study["budget_case_hours"], "planned_case_hours": study["planned_case_hours"],
              "evaluated_case_hours": len(cache)*study["hours"], "test_start": study["test_start"],
              "train_cases": [c["id"] for c in train], "test_cases": [c["id"] for c in test],
              "model_sha256": study["model_sha256"], "provenance": provenance,
              "status": "HELD_OUT_DIAGNOSTIC; no release promotion or regional superiority claim"}
    json.dumps(report, allow_nan=False)
    return report
