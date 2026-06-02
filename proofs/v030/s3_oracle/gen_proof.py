"""Generate proofs/v030/s3_interp_report.json: the S3 proof object.

Records, against the REAL interp_module.F oracle:
  * per-kernel analytic-oracle max relative error (the AC: <= 1e-6),
  * the interp_module.F file:line <-> JAX-function map,
  * the +-chain dispatcher fall-through check,
  * the PRES-build check,
  * the writer structural-diff vs a real met_em file,
  * the full-assemble validate result for d01/d02/d03,
  * a one-shot interp wall-clock timing.
Run:  PYTHONPATH=src python3 proofs/v030/s3_oracle/gen_proof.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

import jax.numpy as jnp  # noqa: E402
import netCDF4  # noqa: E402

from gpuwrf.init import interp_metgrid as im  # noqa: E402
from gpuwrf.init.metem_writer import FIELD_TYPE, write_met_em  # noqa: E402
from gpuwrf.init.metgrid_assemble import (  # noqa: E402
    ForcingFields,
    TargetGrid,
    assemble_met_em,
)
from gpuwrf.init.metgrid_schema import (  # noqa: E402
    ISOBARIC_LEVELS_PA,
    MetgridProjection,
    metem_field_specs,
)
from oracle import Oracle  # noqa: E402

ORACLE_GLOB = "/mnt/data/canairy_meteo/runs/wps_cases/*/l3/met_em.d01.*.nc"

# interp_module.F file:line <-> JAX function map (line numbers verified in the
# WPS source read during S3).
KERNEL_MAP = {
    "oned": "interp_module.F:1342 -> interp_metgrid.oned",
    "nearest_neighbor": "interp_module.F:376 -> interp_metgrid.nearest_neighbor",
    "four_pt": "interp_module.F:1055 -> interp_metgrid.four_pt",
    "sixteen_pt": "interp_module.F:1179 -> interp_metgrid.sixteen_pt",
    "four_pt_average": "interp_module.F:623 -> interp_metgrid.four_pt_average",
    "wt_four_pt_average": "interp_module.F:742 -> interp_metgrid.wt_four_pt_average",
    "sixteen_pt_average": "interp_module.F:861 -> interp_metgrid.sixteen_pt_average",
    "wt_sixteen_pt_average": "interp_module.F:958 -> interp_metgrid.wt_sixteen_pt_average",
    "search_extrap": "interp_module.F:451 -> interp_metgrid.search_extrap",
    "interp_sequence(+-chain)": "interp_module.F:304 -> interp_metgrid.interp_sequence",
    "interp_to_latlon(lltoxy)": "process_domain_module.F:2594 -> interp_metgrid.latlon_to_source_xy",
}


def _smooth(nx, ny):
    ii = np.arange(1, nx + 1)[:, None]
    jj = np.arange(1, ny + 1)[None, :]
    return (290.0 + 5.0 * np.sin(0.15 * ii) + 3.0 * np.cos(0.2 * jj) + 0.05 * ii * jj) * np.ones((nx, ny))


def kernel_errors(orc):
    nx, ny = 28, 22
    slab = _smooth(nx, ny)
    rng = np.random.default_rng(5)
    rx = rng.uniform(3.0, nx - 3.0, 600)
    ry = rng.uniform(3.0, ny - 3.0, 600)
    out = {}
    methods = {
        "nearest_neighbor": [(im.N_NEIGHBOR, 0)],
        "four_pt": [(im.FOUR_POINT, 0)],
        "sixteen_pt": [(im.SIXTEEN_POINT, 0)],
        "four_pt_average": [(im.AVERAGE4, 0)],
        "sixteen_pt_average": [(im.AVERAGE16, 0)],
        "wt_four_pt_average": [(im.W_AVERAGE4, 0)],
        "wt_sixteen_pt_average": [(im.W_AVERAGE16, 0)],
    }
    for name, chain in methods.items():
        o = np.asarray(orc.interp(slab, rx, ry, chain), dtype=np.float64)
        j = np.asarray(im.interp_sequence(jnp.asarray(rx), jnp.asarray(ry), jnp.asarray(slab), chain), dtype=np.float64)
        rel = float(np.max(np.abs(j - o) / (np.abs(o) + 1e-9)))
        out[name] = {"max_rel_err": rel, "pass_1e-6": rel <= 1e-6}
    # oned
    x = rng.uniform(0, 1, 500); a = rng.uniform(1, 5, 500); b = rng.uniform(1, 5, 500)
    c = rng.uniform(1, 5, 500); d = rng.uniform(1, 5, 500)
    a[:100] = 0.0; d[100:200] = 0.0
    o = np.asarray(orc.oned(x, a, b, c, d), dtype=np.float64)
    jj = np.asarray(im.oned(jnp.asarray(x), jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d)), dtype=np.float64)
    rel = float(np.max(np.abs(jj - o) / (np.abs(o) + 1e-9)))
    out["oned"] = {"max_rel_err": rel, "pass_1e-6": rel <= 1e-6}
    return out


def chain_and_search(orc):
    res = {}
    nx, ny = 30, 24
    slab = _smooth(nx, ny)
    rng = np.random.default_rng(11)
    rx = rng.uniform(3.0, nx - 3.0, 500)
    ry = rng.uniform(3.0, ny - 3.0, 500)
    chain = im.parse_interp_string("sixteen_pt+four_pt+average_4pt")
    o = np.asarray(orc.interp(slab, rx, ry, chain), dtype=np.float64)
    j = np.asarray(im.interp_sequence(jnp.asarray(rx), jnp.asarray(ry), jnp.asarray(slab), chain), dtype=np.float64)
    res["atmos_chain_16+4+avg4_max_rel"] = float(np.max(np.abs(j - o) / (np.abs(o) + 1e-9)))

    # masked soil chain + search
    nx2, ny2 = 26, 22
    ii = np.arange(1, nx2 + 1)[:, None]; jj2 = np.arange(1, ny2 + 1)[None, :]
    s = (281.0 + 0.4 * ii + 0.25 * jj2) * np.ones((nx2, ny2))
    landsea = np.where(ii <= 9, 0.0, 1.0) * np.ones((nx2, ny2))
    msg = im.DEFAULT_MSGVAL
    sm = s.copy(); sm[landsea == 0] = msg
    soilchain = im.parse_interp_string("sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search")
    rx2 = rng.uniform(3.0, nx2 - 3.0, 400); ry2 = rng.uniform(3.0, ny2 - 3.0, 400)
    o2 = np.asarray(orc.interp(sm, rx2, ry2, soilchain, msgval=msg, mask_array=landsea, maskval=0.0, mask_relational=" "), dtype=np.float64)
    j2 = np.asarray(im.interp_sequence(jnp.asarray(rx2), jnp.asarray(ry2), jnp.asarray(sm), soilchain, msgval=msg, mask_array=jnp.asarray(landsea), maskval=0.0, mask_relational=" "), dtype=np.float64)
    finmatch = bool(np.array_equal(np.abs(o2) < 1e29, np.abs(j2) < 1e29))
    m = (np.abs(o2) < 1e29) & (np.abs(j2) < 1e29)
    res["soil_masked_chain_max_rel"] = float(np.max(np.abs(j2[m] - o2[m]) / (np.abs(o2[m]) + 1e-9)))
    res["soil_finite_mask_agree"] = finmatch
    return res


def assemble_checks():
    sg = im.LatLonSourceGrid(lon0_deg=-22.0, dlon_deg=0.25, lat0_deg=32.0, dlat_deg=-0.25, nx=49, ny=33, global_wrap=False)
    lonc = sg.lon0_deg + sg.dlon_deg * np.arange(sg.nx)
    latc = sg.lat0_deg + sg.dlat_deg * np.arange(sg.ny)
    LAT, LON = np.meshgrid(latc, lonc, indexing="ij")

    def s2(amp, base):
        return base + amp * (np.sin(np.radians(LON)) + np.cos(np.radians(LAT)))

    forcing = ForcingFields(
        t_iso=np.stack([s2(3, 280 - 8 * k) for k in range(13)]),
        u_iso=np.stack([s2(2, 5 + k) for k in range(13)]),
        v_iso=np.stack([s2(2, -3 + 0.5 * k) for k in range(13)]),
        gh_iso=np.stack([s2(20, 100 + 1500 * k) for k in range(13)]),
        q_iso=np.stack([np.clip(s2(0.001, 0.005 - 3e-4 * k), 0, None) for k in range(13)]),
        t2=s2(5, 290), u10=s2(1, 4), v10=s2(1, -2), q2=np.clip(s2(0.001, 0.006), 0, None),
        psfc=s2(500, 100500), pmsl=s2(300, 101300), soilhgt=np.clip(s2(200, 300), 0, None),
        skintemp=s2(4, 291), landsea=(LON > -18).astype(float), dewpt=s2(5, 285),
        st000010=s2(2, 292), st010040=s2(2, 290),
        sm000010=np.clip(s2(0.05, 0.18), 0, 1), sm010040=np.clip(s2(0.05, 0.16), 0, 1),
    )
    results = {}
    geom = {"d01": (12, 9), "d02": (16, 11), "d03": (14, 13)}
    for dom, (nx, ny) in geom.items():
        lon = np.linspace(-19, -15, nx); lat = np.linspace(27, 29, ny)
        lat_m, lon_m = np.meshgrid(lat, lon, indexing="ij")
        lonu = np.linspace(-19, -15, nx + 1); lat_u, lon_u = np.meshgrid(lat, lonu, indexing="ij")
        latv = np.linspace(27, 29, ny + 1); lat_v, lon_v = np.meshgrid(latv, lon, indexing="ij")
        tg = TargetGrid(lat_m, lon_m, lat_u, lon_u, lat_v, lon_v)
        proj = MetgridProjection(map_proj=1, truelat1=25, truelat2=30, stand_lon=-16.4,
                                 moad_cen_lat=28.3, pole_lat=90, pole_lon=0, dx_m=9000, dy_m=9000,
                                 nx=nx, ny=ny, grid_id=1, parent_id=1, parent_grid_ratio=1,
                                 i_parent_start=1, j_parent_start=1)
        landmask = (lon[None, :] > -17).astype(float) * np.ones((ny, nx))
        static = {
            "XLAT_M": lat_m, "XLONG_M": lon_m, "HGT_M": np.abs(lat_m) * 10,
            "LANDMASK": landmask, "SOILTEMP": 285 + 0 * lat_m,
            "LU_INDEX": np.round(np.abs(lon_m)) % 20,
            "MAPFAC_M": np.ones((ny, nx)), "MAPFAC_U": np.ones((ny, nx + 1)), "MAPFAC_V": np.ones((ny + 1, nx)),
            "MAPFAC_MX": np.ones((ny, nx)), "MAPFAC_MY": np.ones((ny, nx)),
            "MAPFAC_UX": np.ones((ny, nx + 1)), "MAPFAC_UY": np.ones((ny, nx + 1)),
            "MAPFAC_VX": np.ones((ny + 1, nx)), "MAPFAC_VY": np.ones((ny + 1, nx)),
            "F": 7e-5 * np.ones((ny, nx)), "LANDUSEF": np.ones((proj.num_land_cat, ny, nx)) / proj.num_land_cat,
        }
        t0 = time.time()
        art = assemble_met_em(dom, "2026-04-28_18:00:00", proj, forcing, static, tg, sg)
        dt = time.time() - t0
        try:
            art.validate(require_optional=False)
            valid = True
            err = None
        except Exception as e:  # noqa
            valid = False
            err = str(e)
        pres_ok = bool(np.allclose(art.arrays["PRES"][0], art.arrays["PSFC"], atol=1e-2))
        for lev, p in enumerate(ISOBARIC_LEVELS_PA):
            pres_ok = pres_ok and bool(np.allclose(art.arrays["PRES"][lev + 1], p, atol=1e-2))
        ght_ok = bool(np.allclose(art.arrays["GHT"][0], art.arrays["SOILHGT"], atol=1e-3))
        soil_ok = (
            bool(np.allclose(art.arrays["SOIL_LAYERS"][0], 40.0))
            and bool(np.allclose(art.arrays["SOIL_LAYERS"][1], 10.0))
            and bool(np.allclose(art.arrays["ST"][0], art.arrays["ST010040"], atol=1e-4))
            and bool(np.allclose(art.arrays["ST"][1], art.arrays["ST000010"], atol=1e-4))
        )
        results[dom] = {
            "validate_pass": valid, "validate_error": err,
            "pres_build_ok": pres_ok, "ght_surface_ok": ght_ok,
            "soil_packing_ok": soil_ok,
            "UU_shape": list(art.arrays["UU"].shape), "VV_shape": list(art.arrays["VV"].shape),
            "n_fields": len(art.arrays), "assemble_wallclock_s": round(dt, 3),
        }
    return results


def structural_diff():
    hits = sorted(glob.glob(ORACLE_GLOB))
    if not hits:
        return {"oracle_found": False}
    ref = netCDF4.Dataset(hits[0])
    try:
        nx = ref.dimensions["west_east"].size
        ny = ref.dimensions["south_north"].size
        sg = im.LatLonSourceGrid(lon0_deg=-22.0, dlon_deg=0.25, lat0_deg=32.0, dlat_deg=-0.25, nx=49, ny=33, global_wrap=False)
        # reuse assemble_checks' forcing builder via a minimal inline copy
        lonc = sg.lon0_deg + sg.dlon_deg * np.arange(sg.nx); latc = sg.lat0_deg + sg.dlat_deg * np.arange(sg.ny)
        LAT, LON = np.meshgrid(latc, lonc, indexing="ij")
        s2 = lambda amp, base: base + amp * (np.sin(np.radians(LON)) + np.cos(np.radians(LAT)))
        forcing = ForcingFields(
            t_iso=np.stack([s2(3, 280 - 8 * k) for k in range(13)]),
            u_iso=np.stack([s2(2, 5 + k) for k in range(13)]),
            v_iso=np.stack([s2(2, -3) for k in range(13)]),
            gh_iso=np.stack([s2(20, 100 + 1500 * k) for k in range(13)]),
            q_iso=np.stack([np.clip(s2(0.001, 0.005), 0, None) for k in range(13)]),
            t2=s2(5, 290), u10=s2(1, 4), v10=s2(1, -2), q2=np.clip(s2(0.001, 0.006), 0, None),
            psfc=s2(500, 100500), pmsl=s2(300, 101300), soilhgt=np.clip(s2(200, 300), 0, None),
            skintemp=s2(4, 291), landsea=(LON > -18).astype(float), dewpt=s2(5, 285),
            st000010=s2(2, 292), st010040=s2(2, 290),
            sm000010=np.clip(s2(0.05, 0.18), 0, 1), sm010040=np.clip(s2(0.05, 0.16), 0, 1),
        )
        lon = np.linspace(-19, -15, nx); lat = np.linspace(27, 29, ny)
        lat_m, lon_m = np.meshgrid(lat, lon, indexing="ij")
        lonu = np.linspace(-19, -15, nx + 1); lat_u, lon_u = np.meshgrid(lat, lonu, indexing="ij")
        latv = np.linspace(27, 29, ny + 1); lat_v, lon_v = np.meshgrid(latv, lon, indexing="ij")
        tg = TargetGrid(lat_m, lon_m, lat_u, lon_u, lat_v, lon_v)
        proj = MetgridProjection(map_proj=1, truelat1=25, truelat2=30, stand_lon=-16.4,
                                 moad_cen_lat=28.3, pole_lat=90, pole_lon=0, dx_m=9000, dy_m=9000,
                                 nx=nx, ny=ny, grid_id=1, parent_id=1, parent_grid_ratio=1,
                                 i_parent_start=1, j_parent_start=1)
        landmask = (lon[None, :] > -17).astype(float) * np.ones((ny, nx))
        static = {
            "XLAT_M": lat_m, "XLONG_M": lon_m, "HGT_M": np.abs(lat_m) * 10, "LANDMASK": landmask,
            "SOILTEMP": 285 + 0 * lat_m, "LU_INDEX": np.round(np.abs(lon_m)) % 20,
            "MAPFAC_M": np.ones((ny, nx)), "MAPFAC_U": np.ones((ny, nx + 1)), "MAPFAC_V": np.ones((ny + 1, nx)),
            "MAPFAC_MX": np.ones((ny, nx)), "MAPFAC_MY": np.ones((ny, nx)),
            "MAPFAC_UX": np.ones((ny, nx + 1)), "MAPFAC_UY": np.ones((ny, nx + 1)),
            "MAPFAC_VX": np.ones((ny + 1, nx)), "MAPFAC_VY": np.ones((ny + 1, nx)),
            "F": 7e-5 * np.ones((ny, nx)), "LANDUSEF": np.ones((proj.num_land_cat, ny, nx)) / proj.num_land_cat,
        }
        art = assemble_met_em("d01", "2026-04-28_18:00:00", proj, forcing, static, tg, sg)
        out = "/tmp/s3_struct_parity.nc"
        write_met_em(art, out)
        ours = netCDF4.Dataset(out)
        try:
            problems = []
            for vname in ours.variables:
                if vname == "Times":
                    continue
                if vname not in ref.variables:
                    problems.append(f"var {vname} not in oracle"); continue
                ov, rv = ours.variables[vname], ref.variables[vname]
                if ov.dimensions != rv.dimensions:
                    problems.append(f"{vname} dims ours={ov.dimensions} ref={rv.dimensions}")
                for a in ("FieldType", "MemoryOrder", "stagger"):
                    if ov.getncattr(a) != rv.getncattr(a):
                        problems.append(f"{vname}.{a} ours={ov.getncattr(a)!r} ref={rv.getncattr(a)!r}")
            return {
                "oracle_found": True, "oracle_file": hits[0],
                "vars_written": len([v for v in ours.variables if v != "Times"]),
                "structural_problems": problems,
                "structural_diff_zero": len(problems) == 0,
                "oracle_var_count": len(ref.variables),
            }
        finally:
            ours.close()
    finally:
        ref.close()


def main():
    orc = Oracle()
    report = {
        "sprint": "v0.3.0-S3-interp-kernels",
        "oracle": "REAL WPS interp_module.F (liboracle.so) compiled via proofs/v030/s3_oracle/build.sh",
        "kernel_rel_tol_AC": 1e-6,
        "kernel_fortran_map": KERNEL_MAP,
        "per_kernel_error": kernel_errors(orc),
        "chain_and_search": chain_and_search(orc),
        "assemble_validate": assemble_checks(),
        "writer_structural": structural_diff(),
        "notes": [
            "Residual ~1e-7 rel is the Fortran single-precision (real) rounding vs JAX fp64.",
            "sixteen_pt n/=16 averaging branch is dead in interp_module.F (n always 16); omitted.",
            "search_extrap realized as nearest-usable within L1 BFS depth; matches Fortran BFS tie-break.",
            "Static geog fields (XLAT/MAPFAC/LANDUSEF/...) are S2's; assembly copies them through.",
        ],
    }
    out_path = os.path.join(_ROOT, "proofs", "v030", "s3_interp_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", out_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
