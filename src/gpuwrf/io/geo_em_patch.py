"""Host-only terrain path for RE experiments: patch geo_em HGT_M, then check real.exe output.

Terrain is never edited in wrfinput: the base state (MUB/PB/PHB/ALB/T_INIT), the
hydrostatic initial state and wrfbdy all follow from HGT inside real.exe. The
supported route is: ``patch_geo_em`` writes NEW geo_em files -> ALISIOS re-runs
metgrid.exe (met_em carries HGT_M) and real.exe -> ``check_real`` verifies the new
wrfinput against the patched geo_em and, optionally, an unpatched reference run.
No JAX import; outside the traced model program. A changed HGT is a compile
constant of the step program (GridSpec terrain/metrics): every terrain variant is a
new compiled program.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from gpuwrf.io.field_override import _publish, _values, sha256
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.static_patch import _keys, _json, nest_consistency, nest_geometry, nest_set_check, read_namelist_ints, resolve_mask

SCHEMA = "gpuwrf-geo-em-patch-v1"
CHECK_SCHEMA = "gpuwrf-real-terrain-check-v1"
GEO_WAIVABLE = frozenset({"water_terrain", "nest_coverage", "nest_blend_zone", "parent_under_child_zone", "gwd_subgrid_orography"})
# WRF constants (share/module_model_constants.F, RWORDSIZE 4 values).
G, R_D = 9.81, 287.0
CP = 7.0 * R_D / 2.0
P1000MB, T0 = 100000.0, 300.0
CVPM = -(CP - R_D) / CP
# Land statics real.exe must not change when only terrain changed.
LAND_STATICS = ("LU_INDEX", "IVGTYP", "ISLTYP", "LANDMASK", "XLAND", "LANDUSEF", "SOILCTOP", "SOILCBOT", "SHDMAX",
                "SHDMIN", "ALBBCK", "SNOALB", "XLAT", "XLONG", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E",
                "SINALPHA", "COSALPHA")


def smooth_121(field, passes):
    """WPS/WRF-style 1-2-1 smoother (x then y per pass); outermost rows/columns unchanged."""
    a = np.asarray(field, np.float64).copy()
    for _ in range(int(passes)):
        b = a.copy()
        b[:, 1:-1] = 0.25 * a[:, :-2] + 0.5 * a[:, 1:-1] + 0.25 * a[:, 2:]
        c = b.copy()
        c[1:-1, :] = 0.25 * b[:-2, :] + 0.5 * b[1:-1, :] + 0.25 * b[2:, :]
        a = c
    return a


def _taper(mask, cells):
    """Weight 1 on the mask, linear to 0 over ``cells`` grid cells outside it."""
    if cells <= 0:
        return mask.astype(np.float64)
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~mask)
    return np.clip(1.0 - dist / (cells + 1.0), 0.0, 1.0)


def _geo_files(root):
    files = {}
    for p in Path(root).iterdir():
        m = re.fullmatch(r"geo_em\.d([0-9]{2})\.nc", p.name)
        if m:
            files[int(m.group(1))] = p
    if not files:
        raise ValueError(f"{root}: no geo_em.dNN.nc files")
    return files


def _geo_grids(files, namelist):
    nl = read_namelist_ints(namelist, ["parent_id", "i_parent_start", "j_parent_start", "parent_grid_ratio"]) if namelist else {}
    grids = {}
    for d, path in files.items():
        with Dataset(path) as ds:
            ds.set_auto_mask(False)
            grids[d] = {"lat": np.array(ds["XLAT_M"][0]), "lon": np.array(ds["XLONG_M"][0]),
                        "landmask": np.array(ds["LANDMASK"][0]), "lu": np.array(ds["LU_INDEX"][0]),
                        "hgt": np.array(ds["HGT_M"][0])}
        if nl.get("parent_id") and len(nl["parent_id"]) >= d:
            grids[d].update(parent_id=nl["parent_id"][d - 1], ratio=nl["parent_grid_ratio"][d - 1],
                            i_parent_start=nl["i_parent_start"][d - 1], j_parent_start=nl["j_parent_start"][d - 1])
    return grids


def patch_geo_em(geo_dir, output_dir, spec, *, spec_base=None, namelist=None):
    """Publish NEW geo_em files with HGT_M patched on cell masks (smooth | set | add | scale).

    ``taper_cells`` blends the patched field into the original over a ring outside the mask.
    Only HGT_M changes; every other variable stays byte-identical. Empty spec = byte copies.
    Multi-nest checks need ``namelist`` (namelist.wps or namelist.input) for the nest geometry.
    """
    _keys(spec, {"patches", "waivers", "spec_bdy_width", "blend_width"}, "geo-em spec")
    entries = spec.get("patches", [])
    if not isinstance(entries, list):
        raise ValueError("patches must be a list")
    waivers = spec.get("waivers", {}) or {}
    if not isinstance(waivers, Mapping) or any(k not in GEO_WAIVABLE or not isinstance(v, str) or not v.strip() for k, v in waivers.items()):
        raise ValueError(f"waivers must map {sorted(GEO_WAIVABLE)} to nonempty reasons")
    root, output = Path(geo_dir).resolve(), Path(output_dir).absolute()
    base = Path(spec_base or ".").resolve()
    if not root.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("geo directory must exist and output must be new")
    files = _geo_files(root)
    if entries and namelist is None:
        raise ValueError("non-empty geo-em specs need --namelist (namelist.input: max_dom, nest geometry, gwd_opt, spec_bdy_width)")
    nlv = read_namelist_ints(namelist, ["spec_bdy_width", "blend_width", "gwd_opt"]) if namelist else {}
    bdy = int(spec.get("spec_bdy_width", (nlv.get("spec_bdy_width") or [5])[0]))
    blend = int(spec.get("blend_width", (nlv.get("blend_width") or [5])[0]))
    grids = _geo_grids(files, namelist)
    multi = bool(entries)
    if entries and len(files) > 1 and not all("parent_id" in grids[d] for d in grids):
        raise ValueError("multi-nest geo_em patches need --namelist (namelist.wps/namelist.input) for the nest geometry")
    report = {"schema": SCHEMA, "tool": "gpuwrf patch geo-em", "spec": spec, "waivers": dict(waivers),
              "spec_bdy_width": bdy, "blend_width": blend, "files": {}, "patches": [], "checks": [],
              "next_steps": "ALISIOS: metgrid.exe (met_em carries HGT_M) -> real.exe -> gpuwrf patch check-real",
              "compile_impact": {"class": "new_program", "baked_fields": ["HGT"],
                                 "note": "GridSpec terrain/metrics are compile constants: new compiled program per terrain variant [I, code-read]"}}
    new_hgt, changed = {}, {}
    for item in entries:
        _keys(item, {"nest", "mask", "op", "passes", "values", "file", "variable", "taper_cells"}, "geo-em patch")
        d = item.get("nest")
        if isinstance(d, bool) or not isinstance(d, int) or d not in files:
            raise ValueError(f"geo-em patch nest {d!r} has no geo_em.dNN.nc")
        op = item.get("op")
        g = grids[d]
        current = new_hgt.get(d, g["hgt"].astype(np.float64))
        mask, provenance = resolve_mask(item.get("mask", "all"), g, base)
        if op == "smooth":
            passes = item.get("passes", 1)
            if isinstance(passes, bool) or not isinstance(passes, int) or not 1 <= passes <= 100:
                raise ValueError("smooth passes must be an integer 1..100")
            if {"values", "file"} & set(item):
                raise ValueError("smooth takes no values")
            candidate = smooth_121(current, passes)
        elif op in ("set", "add", "scale"):
            if ("values" in item) == ("file" in item):
                raise ValueError(f"{op}: provide exactly one of values or file")
            spec_v = item["values"] if "values" in item else {"file": item["file"], **({"variable": item["variable"]} if "variable" in item else {})}
            arr, _ = _values(spec_v, base, "HGT_M") if isinstance(spec_v, Mapping) else (np.asarray(spec_v, np.float64), None)
            arr = np.asarray(arr, np.float64)
            if arr.shape == (1, *current.shape):
                arr = arr[0]
            if arr.shape not in ((), current.shape) or not np.all(np.isfinite(arr)):
                raise ValueError(f"{op}: values must be a finite scalar or the full HGT_M grid {current.shape}")
            candidate = np.broadcast_to(arr, current.shape) if op == "set" else current + arr if op == "add" else current * arr
        else:
            raise ValueError("op must be smooth, set, add or scale (replace a DEM with set + file)")
        taper = item.get("taper_cells", 0)
        if isinstance(taper, bool) or not isinstance(taper, int) or not 0 <= taper <= 50:
            raise ValueError("taper_cells must be an integer 0..50")
        w = _taper(mask, taper)
        result = w * candidate + (1.0 - w) * current
        new_hgt[d] = result
        touched = w > 0
        changed[d] = changed.get(d, np.zeros(mask.shape, bool)) | (touched & (result.astype(np.float32) != g["hgt"]))
        report["patches"].append({"nest": d, "op": op, "mask": provenance, "mask_cells": int(mask.sum()),
                                  "taper_cells": taper, "touched_cells": int(touched.sum())})
    # Checks.
    for d, h in new_hgt.items():
        g = grids[d]
        h32 = h.astype(np.float32)
        c = changed[d]
        label = f"d{d:02d}"
        ok = bool(np.all(np.isfinite(h32)) and np.all((h32 >= -500) & (h32 <= 9000)))
        report["checks"].append({"nest": label, "name": "hgt_range", "status": "PASS" if ok else "FAIL",
                                 "detail": "patched HGT_M finite and in [-500, 9000] m"})
        wet = c & (g["landmask"] == 0)
        status = "PASS" if not wet.any() else ("WAIVED" if "water_terrain" in waivers else "FAIL")
        report["checks"].append({"nest": label, "name": "water_terrain", "status": status,
                                 "detail": f"{int(wet.sum())} water cells (LANDMASK 0) change HGT_M"})
        delta = h32.astype(np.float64) - g["hgt"]
        report["patches"].append({"nest": d, "summary": True, "cells_changed": int(c.sum()),
                                  "max_abs_change_m": float(np.max(np.abs(delta))) if delta.size else 0.0,
                                  "mean_change_on_changed_m": float(delta[c].mean()) if c.any() else 0.0})
    gwd = nlv.get("gwd_opt")
    for d, c in sorted(changed.items()):
        if not c.any():
            continue
        g = None if not gwd else (gwd[d - 1] if len(gwd) >= d else gwd[0])
        ok = g is not None and g not in (1, 3)
        status = "PASS" if ok else ("WAIVED" if "gwd_subgrid_orography" in waivers else "FAIL")
        report["checks"].append({"nest": f"d{d:02d}", "name": "gwd_subgrid_orography", "status": status,
                                 "detail": f"gwd_opt={g if g is not None else 'unknown (pass namelist.input)'}: geo_em VAR/CON/OA1-4/OL1-4 "
                                           "keep the old DEM statistics after an HGT_M patch; gravity-wave drag would use them"})
    if multi:
        report["checks"].append(nest_set_check(namelist, list(files)))
        geo, geo_checks = nest_geometry(grids, spec_bdy_width=bdy)
        report["checks"] += geo_checks
        # Parent terrain under a child's spec+blend zone reaches the child through wrf.exe blend_terrain,
        # where the child can never be patched consistently: refuse (waivable).
        from gpuwrf.io.static_patch import child_footprint
        for d, g in sorted(geo.items()):
            ny, nx = g["shape"]
            width = bdy + blend
            zone = np.ones((ny, nx), bool)
            zone[width:ny - width, width:nx - width] = False
            jp, ip = child_footprint(g)
            under = np.zeros(grids[g["parent"]]["lat"].shape, bool)
            under[jp[zone], ip[zone]] = True
            hit = under & changed.get(g["parent"], np.zeros(under.shape, bool))
            status = "PASS" if not hit.any() else ("WAIVED" if "parent_under_child_zone" in waivers else "FAIL")
            report["checks"].append({"nest": f"d{d:02d}", "name": "parent_under_child_zone", "status": status,
                                     "detail": f"{int(hit.sum())} changed d{g['parent']:02d} cells lie under the d{d:02d} {width}-cell spec+blend zone "
                                               "(wrf.exe blends them into the child; check-real would reject the run)"})
        req = {d: {"hgt": changed.get(d, np.zeros(grids[d]["lat"].shape, bool))} for d in grids}
        for chk in nest_consistency(geo, req, req, label="terrain", waivers=waivers, zone_width=bdy + blend):
            if chk["name"] == "nest_boundary_zone":
                chk["name"] = "nest_blend_zone"
                chk["status"] = "PASS" if chk["status"] == "PASS" else ("WAIVED" if "nest_blend_zone" in waivers else "FAIL")
                chk["detail"] += " (wrf.exe blend_terrain replaces this zone with the parent terrain)"
            report["checks"].append(chk)
    report["passed"] = not any(c["status"] == "FAIL" for c in report["checks"])
    if not report["passed"]:
        from gpuwrf.io.static_patch import StaticPatchError
        raise StaticPatchError(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.geo-", dir=output.parent))
    try:
        for d, source in sorted(files.items()):
            before = sha256(source)
            target = staging / source.name
            shutil.copyfile(source, target)
            if sha256(target) != before:
                raise ValueError(f"source changed while copying: {source}")
            if d in new_hgt and changed[d].any():
                with Dataset(target, "r+") as ds:
                    ds.set_auto_maskandscale(False)
                    var = ds["HGT_M"]
                    var[0] = new_hgt[d].astype(var.dtype)
                    ds.sync()
                with Dataset(target) as ds, Dataset(source) as orig:
                    ds.set_auto_mask(False)
                    orig.set_auto_mask(False)
                    stray = [v for v in ds.variables if v != "HGT_M" and np.asarray(ds[v][:]).tobytes() != np.asarray(orig[v][:]).tobytes()]
                    if stray:
                        raise ValueError(f"{source.name}: variables other than HGT_M changed: {stray}")
            report["files"][source.name] = {"source": str(source), "output": str(output / source.name),
                                            "sha256_before": before, "sha256_after": sha256(target),
                                            "byte_identical": before == sha256(target)}
        if any(sha256(files[d]) != report["files"][files[d].name]["sha256_before"] for d in files):
            raise ValueError("source geo_em changed during patch")
        (staging / "GEO_PATCH_PROTOCOL.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=_json) + "\n")
        _publish(staging, output)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


# ----------------------------------------------------------------------------- post-real checks
def base_state(hgt, consts):
    """real.exe base state (module_initialize_real.F:3790-3830) in float64.

    Returns MUB (y,x), PB/T_INIT/ALB (k,y,x) and PHB (k+1,y,x) for HYPSOMETRIC_OPT 1 or 2.
    """
    p00, t00, a = consts["P00"], consts["T00"], consts["TLP"]
    tiso, a_strat, p_strat, ptop = consts["TISO"], consts["TLP_STRAT"], consts["P_STRAT"], consts["P_TOP"]
    c3f, c4f, c3h, c4h = (np.asarray(consts[k], np.float64) for k in ("C3F", "C4F", "C3H", "C4H"))
    h = np.asarray(hgt, np.float64)
    p_surf = p00 * np.exp(-t00 / a + np.sqrt((t00 / a) ** 2 - 2.0 * G * h / a / R_D))
    mub = p_surf - ptop
    pb = c3h[:, None, None] * mub[None] + c4h[:, None, None] + ptop
    temp = np.maximum(tiso, t00 + a * np.log(pb / p00))
    if p_strat > 0:
        strat = pb < p_strat
        temp = np.where(strat, tiso + a_strat * np.log(np.where(strat, pb, p_strat) / p_strat), temp)
    t_init = temp * (p00 / pb) ** (R_D / CP) - T0
    alb = (R_D / P1000MB) * (t_init + T0) * (pb / P1000MB) ** CVPM
    nz = c3h.size
    phb = np.empty((nz + 1, *h.shape))
    phb[0] = h * G
    if int(consts.get("HYPSOMETRIC_OPT", 2)) == 1:
        dnw, c1h, c2h = (np.asarray(consts[k], np.float64) for k in ("DNW", "C1H", "C2H"))
        for k in range(1, nz + 1):
            phb[k] = phb[k - 1] - dnw[k - 1] * (c1h[k - 1] * mub + c2h[k - 1]) * alb[k - 1]
    else:
        for k in range(1, nz + 1):
            pfu = c3f[k] * mub + c4f[k] + ptop
            pfd = c3f[k - 1] * mub + c4f[k - 1] + ptop
            phm = c3h[k - 1] * mub + c4h[k - 1] + ptop
            phb[k] = phb[k - 1] + alb[k - 1] * phm * np.log(pfd / pfu)
    return {"MUB": mub, "PB": pb, "T_INIT": t_init, "ALB": alb, "PHB": phb}


# Relative tolerances: float64 recompute vs real.exe REAL storage (calibrated on WN3 0227 d01-d03, see docs).
# Measured max on the originals: MUB/PB/ALB 5.8e-7, theta (T_INIT+T0) see docs, PHB 2.0e-5. A 1 m HGT error moves MUB by ~1e-4.
BASE_TOL = {"MUB": 2e-6, "PB": 2e-6, "T_INIT": 2e-6, "ALB": 2e-6, "PHB": 5e-5}


def _read_real(path):
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        out = {v: np.array(ds[v][:]) for v in ds.variables if v != "Times"}
        attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
    return out, attrs


def check_real(geo_dir, real_dir, *, reference_dir=None, namelist=None):
    """Check real.exe output produced from a patched geo_em (read-only; returns a report)."""
    files = _geo_files(geo_dir)
    if namelist is None:
        raise ValueError("check-real needs --namelist (namelist.input: max_dom, spec_bdy_width, blend_width)")
    report = {"schema": CHECK_SCHEMA, "tool": "gpuwrf patch check-real", "geo_dir": str(Path(geo_dir).resolve()),
              "real_dir": str(Path(real_dir).resolve()), "reference_dir": None if reference_dir is None else str(Path(reference_dir).resolve()),
              "files": {}, "checks": [], "diagnostics": {}}
    nlv = read_namelist_ints(namelist, ["spec_bdy_width", "blend_width"]) if namelist else {}
    bdy, blend = (nlv.get("spec_bdy_width") or [5])[0], (nlv.get("blend_width") or [5])[0]
    report["spec_bdy_width"], report["blend_width"] = bdy, blend

    def check(nest, name, ok, detail, **extra):
        report["checks"].append({"nest": nest, "name": name, "status": "PASS" if ok else "FAIL", "detail": detail, **extra})

    real, attrs, ref = {}, {}, {}
    for d, geo_path in sorted(files.items()):
        path = Path(real_dir) / f"wrfinput_d{d:02d}"
        if not path.exists():
            raise ValueError(f"missing {path}")
        label = f"d{d:02d}"
        real[d], attrs[d] = _read_real(path)
        report["files"][path.name] = {"path": str(path), "sha256": sha256(path), "geo_em": str(geo_path), "geo_sha256": sha256(geo_path)}
        with Dataset(geo_path) as ds:
            ds.set_auto_mask(False)
            hgt_m = np.array(ds["HGT_M"][0])
        hgt = real[d]["HGT"][0]
        diff = np.abs(hgt.astype(np.float64) - hgt_m)
        check(label, "hgt_matches_geo_em", bool(np.all(diff == 0)), f"wrfinput HGT == geo_em HGT_M (max |diff| {float(diff.max()):.3g} m)",
              max_abs_diff_m=float(diff.max()), cells_differing=int((diff > 0).sum()))
        consts = {k: float(np.asarray(real[d][k]).reshape(-1)[0]) for k in ("P00", "T00", "TLP", "TISO", "TLP_STRAT", "P_STRAT", "P_TOP")}
        for k in ("C3F", "C4F", "C3H", "C4H", "DNW", "C1H", "C2H"):
            consts[k] = np.asarray(real[d][k]).reshape(-1)
        consts["HYPSOMETRIC_OPT"] = int(attrs[d].get("HYPSOMETRIC_OPT", 2))
        recomputed = base_state(hgt, consts)
        errs = {}
        for k, tol in BASE_TOL.items():
            stored = real[d][k][0].astype(np.float64)
            ref_value = recomputed[k] + (T0 if k == "T_INIT" else 0.0)  # T_INIT is a perturbation of theta from T0
            rel = np.abs(stored - recomputed[k]) / np.maximum(np.abs(ref_value), 1.0)
            errs[k] = float(rel.max())
        check(label, "base_state_consistent", all(errs[k] <= BASE_TOL[k] for k in errs),
              f"MUB/PB/T_INIT/ALB/PHB recomputed from HGT (hypsometric_opt {consts['HYPSOMETRIC_OPT']}) within tolerance",
              max_relative_error=errs, tolerance=BASE_TOL)
        phb0 = np.abs(real[d]["PHB"][0][0].astype(np.float64) - hgt.astype(np.float64) * G)
        check(label, "phb_surface", bool(phb0.max() <= 1e-6 * max(1.0, float(np.abs(hgt).max()) * G)),
              f"PHB(surface) == g*HGT (max |diff| {float(phb0.max()):.3g})")
        mass = real[d]["MU"][0].astype(np.float64) + real[d]["MUB"][0]
        thick = np.diff(real[d]["PH"][0].astype(np.float64) + real[d]["PHB"][0], axis=0)
        check(label, "mass_geometry", bool(np.all(mass > 0) and np.all(thick > 0)), "positive dry column mass and layer thickness")
        if reference_dir is not None:
            ref[d], _ = _read_real(Path(reference_dir) / f"wrfinput_d{d:02d}")
            moved = [v for v in LAND_STATICS if v in real[d] and v in ref[d] and real[d][v].tobytes() != ref[d][v].tobytes()]
            check(label, "land_statics_unchanged", not moved, f"land/grid statics identical to the reference real.exe run; changed: {moved}")
            dh = real[d]["HGT"][0].astype(np.float64) - ref[d]["HGT"][0]
            changed_vars = sorted(v for v in real[d] if v in ref[d] and real[d][v].shape == ref[d][v].shape and real[d][v].tobytes() != ref[d][v].tobytes())
            report["diagnostics"][label] = {"hgt_cells_changed": int((dh != 0).sum()), "hgt_max_abs_change_m": float(np.abs(dh).max()),
                                            "variables_changed_vs_reference": changed_vars}
    # Parent/child terrain zone: identical to the reference in the child spec+blend zone and the parent cells under it.
    grids = {d: {"lat": real[d]["XLAT"][0], "lon": real[d]["XLONG"][0], "parent_id": int(attrs[d].get("PARENT_ID", 0)),
                 "ratio": int(attrs[d].get("PARENT_GRID_RATIO", 1)), "i_parent_start": int(attrs[d].get("I_PARENT_START", 1)),
                 "j_parent_start": int(attrs[d].get("J_PARENT_START", 1))} for d in real}
    report["checks"].append(nest_set_check(namelist, list(real)))
    geo, geo_checks = nest_geometry(grids, spec_bdy_width=bdy)
    report["checks"] += geo_checks
    width = bdy + blend
    for d, g in sorted(geo.items()):
        p = g["parent"]
        ny, nx = g["shape"]
        r = g["ratio"]
        zone = np.ones((ny, nx), bool)
        zone[width:ny - width, width:nx - width] = False
        from gpuwrf.io.static_patch import child_footprint
        jp, ip = child_footprint(g)
        # Bilinear-free diagnostic: child HGT minus the parent cell HGT containing it, in the zone.
        jump = real[d]["HGT"][0].astype(np.float64) - real[p]["HGT"][0][jp, ip]
        diag = report["diagnostics"].setdefault(f"d{d:02d}", {})
        diag["zone_child_minus_parent_cell_hgt_max_abs_m"] = float(np.abs(jump[zone]).max())
        if reference_dir is None:
            check(f"d{d:02d}", "nest_terrain_zone_unchanged", False,
                  "no --reference-dir: identity of the child spec+blend zone terrain (and the parent cells under it) is unproven")
        else:
            same_child = bool(np.array_equal(real[d]["HGT"][0][zone], ref[d]["HGT"][0][zone]))
            pz = np.zeros(real[p]["HGT"][0].shape, bool)
            pz[jp[zone], ip[zone]] = True
            same_parent = bool(np.array_equal(real[p]["HGT"][0][pz], ref[p]["HGT"][0][pz]))
            check(f"d{d:02d}", "nest_terrain_zone_unchanged", same_child and same_parent,
                  f"child {width}-cell spec+blend zone HGT and the d{p:02d} cells under it identical to the reference (child {same_child}, parent {same_parent})")
            ref_jump = ref[d]["HGT"][0].astype(np.float64) - ref[p]["HGT"][0][jp, ip]
            diag["reference_zone_child_minus_parent_cell_hgt_max_abs_m"] = float(np.abs(ref_jump[zone]).max())
    report["passed"] = not any(c["status"] == "FAIL" for c in report["checks"])
    return report


__all__ = ["patch_geo_em", "check_real", "base_state", "smooth_121"]
