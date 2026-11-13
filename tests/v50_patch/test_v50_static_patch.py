"""gpuwrf patch (v0.5.0 item 0.3): host-only static land/terrain patches on a synthetic 2-nest case.

Real WN3 0227 3-nest evidence (round trip, 22 seeded mutants) lives in the lane proof; these tests
pin the rules on small real-structured files: byte-identical no-op, donor known answers, fail-closed
dependents/flips/nest checks, verify re-audit, geo_em patch and the post-real base-state check.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.static_patch import StaticPatchError, patch_land, verify_patched
from gpuwrf.io.geo_em_patch import base_state, check_real, patch_geo_em, smooth_121

ROOT = Path(__file__).resolve().parents[2]
NY1 = NX1 = 12
R, I0, J0, NC = 3, 3, 3, 5          # child = 5 x 5 parent cells at 0-based parent (3, 3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid(nest):
    if nest == 1:
        j, i = np.meshgrid(np.arange(NY1), np.arange(NX1), indexing="ij")
        return 28.0 + 0.09 * j, -17.0 + 0.09 * i
    n = NC * R
    j, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return 28.0 + 0.09 * (J0 + (j - 1) / R), -17.0 + 0.09 * (I0 + (i - 1) / R)


def parent_lu(lat, lon):
    i = np.rint((lon + 17.0) / 0.09).astype(int)
    lu = np.where(i < 5, 17, np.where(i % 2 == 0, 10, 16))
    return lu.astype(np.float32)


CONSTS = {"P00": 1e5, "T00": 290.0, "TLP": 50.0, "TISO": 216.65, "TLP_STRAT": -11.0, "P_STRAT": 5500.0, "P_TOP": 5000.0}
ETA_F = np.array([1.0, 0.7, 0.4, 0.0])


def make_nest(path, nest, aux_path=None):
    lat, lon = grid(nest)
    ny, nx = lat.shape
    lu = parent_lu(lat, lon)
    land = lu != 17
    hgt = np.where(land, 200.0 + 50 * np.sin(lat * 7), 0.0).astype(np.float32)
    rng = np.random.default_rng(nest)
    eta_h = 0.5 * (ETA_F[:-1] + ETA_F[1:])
    c = dict(CONSTS, C3F=ETA_F, C4F=np.zeros(4), C3H=eta_h, C4H=np.zeros(3), DNW=np.diff(ETA_F), C1H=np.ones(3), C2H=np.zeros(3), HYPSOMETRIC_OPT=2)
    b = base_state(hgt, c)
    with Dataset(path, "w") as ds:
        for name, size in (("Time", None), ("DateStrLen", 19), ("south_north", ny), ("west_east", nx), ("bottom_top", 3),
                           ("bottom_top_stag", 4), ("soil_layers_stag", 4), ("land_cat_stag", 21), ("soil_cat_stag", 16)):
            ds.createDimension(name, size)
        ds.setncatts({"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "NUM_LAND_CAT": np.int32(21), "ISWATER": np.int32(17),
                      "ISLAKE": np.int32(21), "ISICE": np.int32(15), "ISURBAN": np.int32(13), "ISOILWATER": np.int32(14),
                      "GRID_ID": np.int32(nest), "PARENT_ID": np.int32(max(nest - 1, 0) or 1),
                      "I_PARENT_START": np.int32(I0 + 1 if nest == 2 else 1), "J_PARENT_START": np.int32(J0 + 1 if nest == 2 else 1),
                      "PARENT_GRID_RATIO": np.int32(R if nest == 2 else 1), "HYPSOMETRIC_OPT": np.int32(2)})
        t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0] = np.array(list("2026-02-27_18:00:00"), dtype="S1")
        xy, soil = ("Time", "south_north", "west_east"), ("Time", "soil_layers_stag", "south_north", "west_east")

        def var(name, value, dims=xy, dtype="f4"):
            v = ds.createVariable(name, dtype, dims)
            v[:] = np.asarray(value)[None]
        var("XLAT", lat); var("XLONG", lon); var("LU_INDEX", lu); var("IVGTYP", lu.astype(np.int32), dtype="i4")
        var("ISLTYP", np.where(land, 8, 14), dtype="i4"); var("LANDMASK", land * 1.0); var("XLAND", np.where(land, 1.0, 2.0))
        luf = np.zeros((21, ny, nx)); luf[lu.astype(int) - 1, np.arange(ny)[:, None], np.arange(nx)[None]] = 1.0
        var("LANDUSEF", luf, ("Time", "land_cat_stag", "south_north", "west_east"))
        sct = np.zeros((16, ny, nx)); sct[np.where(land, 7, 13), np.arange(ny)[:, None], np.arange(nx)[None]] = 1.0
        var("SOILCTOP", sct, ("Time", "soil_cat_stag", "south_north", "west_east")); var("SOILCBOT", sct, ("Time", "soil_cat_stag", "south_north", "west_east"))
        var("SMOIS", np.where(land, 0.1 + 0.01 * np.arange(4)[:, None, None], 1.0), soil)
        var("SH2O", np.where(land, 0.0, 1.0) * np.ones((4, 1, 1)), soil)
        var("TSLB", 285.0 + rng.random((4, ny, nx)), soil)
        shdmax = np.where(land, np.where(lu == 10, 60.0, 5.0) + rng.random((ny, nx)), 0.0)
        var("SHDMAX", shdmax); var("SHDMIN", 0.5 * shdmax); var("SHDAVG", 0.7 * shdmax); var("VEGFRA", 0.8 * shdmax)
        var("LAI", np.where(land, 0.02 * shdmax, 0.0)); var("ALBBCK", np.where(land, 0.15 + 0.001 * shdmax, 0.08))
        var("SNOALB", np.where(land, 0.5, 0.08))
        sst = 291.0 + 0.1 * lat + 0.2 * (lon + 17.0)
        var("SST", sst); var("TSK", np.where(land, 288.0, sst)); var("TMN", np.where(land, 287.0, sst))
        for name in ("SEAICE", "CANWAT", "SNOW", "SNOWH", "LAKEMASK"):
            var(name, np.zeros((ny, nx)))
        var("HGT", hgt); var("T2", 290.0 * np.ones((ny, nx)))
        var("MUB", b["MUB"]); var("MU", np.zeros((ny, nx)))
        for name, dims in (("PB", "bottom_top"), ("ALB", "bottom_top"), ("T_INIT", "bottom_top"), ("PHB", "bottom_top_stag")):
            var(name, b[name], ("Time", dims, "south_north", "west_east"))
        var("PH", np.zeros((4, ny, nx)), ("Time", "bottom_top_stag", "south_north", "west_east"))
        for name, value in CONSTS.items():
            var(name, value, ("Time",))
        for name, dims, value in (("C3F", "bottom_top_stag", ETA_F), ("C4F", "bottom_top_stag", np.zeros(4)), ("C3H", "bottom_top", eta_h),
                                  ("C4H", "bottom_top", np.zeros(3)), ("DNW", "bottom_top", np.diff(ETA_F)), ("C1H", "bottom_top", np.ones(3)),
                                  ("C2H", "bottom_top", np.zeros(3))):
            var(name, value, ("Time", dims))
    if aux_path is not None:
        with Dataset(aux_path, "w") as ds:
            for name, size in (("Time", None), ("DateStrLen", 19), ("south_north", ny), ("west_east", nx)):
                ds.createDimension(name, size)
            t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
            for k in range(3):
                t[k] = np.array(list(f"2026-02-27_{18 + k:02d}:00:00"), dtype="S1")
            for name, value in (("VEGFRA", 0.8 * shdmax), ("ALBBCK", np.where(land, 0.15 + 0.001 * shdmax, 0.08)),
                                ("LAI", np.where(land, 0.02 * shdmax, 0.0)), ("SST", sst), ("SEAICE", np.zeros((ny, nx)))):
                v = ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))
                v[:] = np.stack([value + 0.01 * k * (name != "SEAICE") for k in range(3)])
    return lat, lon


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    for nest in (1, 2):
        make_nest(root / f"wrfinput_d{nest:02d}", nest, root / f"wrflowinp_d{nest:02d}")
    write_case_namelist(root / "namelist.input", 2)
    return root


def read(path, name):
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        return np.array(ds[name][:])


def run(case, tmp_path, patches, name="out", **spec):
    spec.setdefault("spec_bdy_width", 1)       # 15 x 15 child: keep a real interior
    return patch_land(case, tmp_path / name, {"patches": patches, **spec}, namelist=case / "namelist.input")


BOX = {"bbox": [28.225, 28.585, -16.685, -16.415]}  # parent-cell edges: d01 cells (3..6, 4..6) and exactly their d02 children


def test_empty_spec_is_byte_identity(case, tmp_path):
    report = patch_land(case, tmp_path / "out", {})
    for name in ("wrfinput_d01", "wrfinput_d02", "wrflowinp_d01", "wrflowinp_d02"):
        assert sha(case / name) == sha(tmp_path / "out" / name)
        assert report["files"][name]["byte_identical"]
    assert report["compile_impact"]["class"] == "same_program"
    assert json.loads((tmp_path / "out/PATCH_PROTOCOL.json").read_text())["schema"] == "gpuwrf-static-patch-v1"


def test_landuse_donor_known_answers_and_untouched_storage(case, tmp_path):
    patches = [{"nest": n, "kind": "landuse", "class": 10, "mask": [BOX, "class:16"], "dependents": "donor"} for n in (1, 2)]
    report = run(case, tmp_path, patches)
    out = tmp_path / "out"
    lu0, lu1 = read(case / "wrfinput_d02", "LU_INDEX")[0], read(out / "wrfinput_d02", "LU_INDEX")[0]
    changed = lu0 != lu1
    assert changed.any() and np.all(lu1[changed] == 10)
    assert np.array_equal(read(out / "wrfinput_d02", "IVGTYP")[0], lu1.astype(np.int32))
    luf = read(out / "wrfinput_d02", "LANDUSEF")[0]
    assert np.all(luf[9][changed] == 1) and np.all(luf.sum(0)[changed] == 1)
    # Donor = nearest original class-10 land cell outside the mask: copy of its SHDMAX and of every aux record.
    from scipy.spatial import cKDTree
    mask = changed
    shd0 = read(case / "wrfinput_d02", "SHDMAX")[0]
    cand = np.argwhere((lu0 == 10))
    tgt = np.argwhere(mask)
    _, idx = cKDTree(cand).query(tgt)
    donors = cand[idx]
    assert np.array_equal(read(out / "wrfinput_d02", "SHDMAX")[0][mask], shd0[donors[:, 0], donors[:, 1]])
    aux0, aux1 = read(case / "wrflowinp_d02", "VEGFRA"), read(out / "wrflowinp_d02", "VEGFRA")
    assert np.array_equal(aux1[:, mask], aux0[:, donors[:, 0], donors[:, 1]])
    for name in ("HGT", "PHB", "MUB", "T2", "SST", "SMOIS", "TSLB"):
        assert read(out / "wrfinput_d02", name).tobytes() == read(case / "wrfinput_d02", name).tobytes()
    assert report["compile_impact"]["class"] == "new_program"
    assert set(report["compile_impact"]["baked_fields"]) == {"IVGTYP", "LU_INDEX", "SHDMAX"}
    assert verify_patched(case, out, spec_bdy_width=1)["passed"]


def test_dependents_are_mandatory_and_keep_needs_waiver(case, tmp_path):
    item = {"nest": 2, "kind": "landuse", "class": 10, "mask": [BOX, "class:16"]}
    with pytest.raises(ValueError, match="explicit policy"):
        run(case, tmp_path, [item])
    keep = dict(item, dependents={f: "keep" for f in ("SHDMAX", "SHDMIN", "SHDAVG", "VEGFRA", "LAI", "ALBBCK", "SNOALB")})
    with pytest.raises(StaticPatchError, match="dependent_keep"):
        run(case, tmp_path, [keep, dict(keep, nest=1)])
    assert not (tmp_path / "out").exists()
    report = run(case, tmp_path, [keep, dict(keep, nest=1)], waivers={"dependent_keep": "climatology retained on purpose"})
    assert {c["status"] for c in report["checks"] if c["name"] == "dependent_keep"} == {"WAIVED"}


@pytest.mark.parametrize("target", [15, 21])
def test_glacier_and_lake_targets_refused(case, tmp_path, target):
    with pytest.raises(ValueError, match="glacier/lake"):
        run(case, tmp_path, [{"nest": 2, "kind": "landuse", "class": target, "mask": BOX, "dependents": "donor"}])


def test_no_donor_of_target_class_fails_closed(case, tmp_path):
    with pytest.raises(ValueError, match="no donor"):
        run(case, tmp_path, [{"nest": 2, "kind": "landuse", "class": 7, "mask": [BOX, "land"], "dependents": "donor"}])


def test_land_to_water_flip_uses_convention_and_sea_donor(case, tmp_path):
    lat, lon = grid(2)
    cell = np.zeros(lat.shape, bool)
    cell[7, 6] = True          # land, interior of d02
    np.save(tmp_path / "cell.npy", cell)
    item = {"nest": 2, "kind": "landuse", "class": 17, "mask": {"file": str(tmp_path / "cell.npy")}, "dependents": "donor"}
    report = run(case, tmp_path, [item])
    out = tmp_path / "out/wrfinput_d02"
    assert read(out, "LANDMASK")[0][7, 6] == 0 and read(out, "XLAND")[0][7, 6] == 2 and read(out, "ISLTYP")[0][7, 6] == 14
    assert np.all(read(out, "SMOIS")[0][:, 7, 6] == 1) and read(out, "VEGFRA")[0][7, 6] == 0
    assert read(out, "ALBBCK")[0][7, 6] == np.float32(0.08)
    assert "SST" in report["files"]["wrflowinp_d02"]["fields_changed"]
    assert report["patches"][0]["land_to_water"] == 1
    with pytest.raises(ValueError, match="cannot keep"):
        run(case, tmp_path, [dict(item, dependents={"SST": "keep", "TSK": "donor", "TMN": "donor", "TSLB": "donor"})], name="o2",
            waivers={"dependent_keep": "x"})


def test_water_to_land_needs_soil_class(case, tmp_path):
    item = {"nest": 2, "kind": "landuse", "class": 16, "mask": [{"bbox": [28.2, 28.6, -16.8, -16.6]}, "water"], "dependents": "donor"}
    with pytest.raises(ValueError, match="soil_class"):
        run(case, tmp_path, [item])


def test_nest_boundary_zone_and_coverage_fail_closed(case, tmp_path):
    lat, lon = grid(2)
    zone = np.zeros(lat.shape, bool)
    zone[0, 10] = True         # child boundary zone (width 1), land
    np.save(tmp_path / "zone.npy", zone)
    child_only = {"nest": 2, "kind": "soil", "field": "SMOIS", "op": "scale", "values": 1.1, "mask": {"file": str(tmp_path / "zone.npy")}}
    with pytest.raises(StaticPatchError, match="nest_boundary_zone"):
        run(case, tmp_path, [child_only])
    report = run(case, tmp_path, [child_only], waivers={"nest_boundary_zone": "deliberate"})
    assert any(c["status"] == "WAIVED" for c in report["checks"] if c["name"] == "nest_boundary_zone")
    parent_only = {"nest": 1, "kind": "soil", "field": "SMOIS", "op": "scale", "values": 1.1, "mask": [BOX, "land"]}
    with pytest.raises(StaticPatchError, match="nest_coverage"):
        run(case, tmp_path, [parent_only], name="o3")
    both = run(case, tmp_path, [parent_only, dict(parent_only, nest=2)], name="o4")
    assert both["passed"] and both["compile_impact"]["class"] == "same_program"


def test_soil_patch_porosity_and_water_refusal(case, tmp_path):
    with pytest.raises(StaticPatchError, match="soil_porosity"):
        run(case, tmp_path, [{"nest": n, "kind": "soil", "field": "SMOIS", "values": 0.95, "mask": [BOX, "land"]} for n in (1, 2)])
    with pytest.raises(ValueError, match="land cells only"):
        run(case, tmp_path, [{"nest": 2, "kind": "soil", "field": "SMOIS", "values": 0.2, "mask": "all"}])


def test_verify_catches_seeded_inconsistencies(case, tmp_path):
    patches = [{"nest": n, "kind": "landuse", "class": 10, "mask": [BOX, "class:16"], "dependents": "donor"} for n in (1, 2)]
    run(case, tmp_path, patches)
    out = tmp_path / "out"
    j, i = np.argwhere(read(case / "wrfinput_d02", "LU_INDEX")[0] != read(out / "wrfinput_d02", "LU_INDEX")[0])[0]
    for name, value, check in (("IVGTYP", 16, "lu_alias"), ("HGT", 999.0, "base_state_untouched"), ("T2", 1.0, "untouched_elsewhere"),
                               ("ISLTYP", 14, "soil_category")):
        bad = tmp_path / f"bad_{name}"
        bad.mkdir()
        for p in out.glob("wr*"):
            bad.joinpath(p.name).write_bytes(p.read_bytes())
        with Dataset(bad / "wrfinput_d02", "r+") as ds:
            a = ds[name][:]
            a[0, j, i] = value
            ds[name][:] = a
        failed = {c["name"] for c in verify_patched(case, bad, spec_bdy_width=1)["checks"] if c["status"] == "FAIL"}
        assert check in failed, (name, failed)


def test_smooth_121_known_answer():
    a = np.zeros((5, 5))
    a[2, 2] = 16.0
    s = smooth_121(a, 1)
    assert s[2, 2] == 4.0 and s[2, 1] == 2.0 and s[1, 1] == 1.0 and s.sum() == 16.0


def make_geo(root, nest):
    lat, lon = grid(nest)
    lu = parent_lu(lat, lon)
    with Dataset(root / f"geo_em.d{nest:02d}.nc", "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("south_north", lat.shape[0])
        ds.createDimension("west_east", lat.shape[1])
        for name, value in (("XLAT_M", lat), ("XLONG_M", lon), ("LU_INDEX", lu), ("LANDMASK", (lu != 17) * 1.0),
                            ("HGT_M", read(root.parent / "case" / f"wrfinput_d{nest:02d}", "HGT")[0]), ("VAR", np.ones(lat.shape))):
            v = ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            v[:] = np.asarray(value)[None]


@pytest.fixture
def geo(case, tmp_path):
    root = tmp_path / "geo"
    root.mkdir()
    for nest in (1, 2):
        make_geo(root, nest)
    nl = tmp_path / "namelist.wps"
    nl.write_text(f"&share\n max_dom = 2,\n/\n&geogrid\n parent_id = 1, 1,\n parent_grid_ratio = 1, {R},\n"
                  f" i_parent_start = 1, {I0 + 1},\n j_parent_start = 1, {J0 + 1},\n/\n&physics\n gwd_opt = 0, 0,\n/\n")
    return root, nl


def test_geo_em_patch_noop_smooth_and_refusals(case, tmp_path, geo):
    root, nl = geo
    noop = patch_geo_em(root, tmp_path / "g0", {}, namelist=nl)
    assert all(v["byte_identical"] for v in noop["files"].values())
    spec = {"patches": [{"nest": 2, "op": "add", "values": 30.0, "taper_cells": 0, "mask": [{"bbox": [28.4, 28.5, -16.6, -16.5]}, "land"]}],
            "blend_width": 1, "spec_bdy_width": 1}
    report = patch_geo_em(root, tmp_path / "g1", spec, namelist=nl)
    assert report["passed"] and not report["files"]["geo_em.d02.nc"]["byte_identical"]
    with Dataset(tmp_path / "g1/geo_em.d02.nc") as a, Dataset(root / "geo_em.d02.nc") as b:
        assert np.array_equal(a["VAR"][:], b["VAR"][:]) and not np.array_equal(a["HGT_M"][:], b["HGT_M"][:])
    with pytest.raises(StaticPatchError, match="nest_blend_zone"):
        patch_geo_em(root, tmp_path / "g2", {"patches": [{"nest": 2, "op": "add", "values": 30.0, "mask": [{"bbox": [28.0, 28.4, -16.8, -16.0]}, "land"]}]}, namelist=nl)
    with pytest.raises(StaticPatchError, match="water_terrain"):
        patch_geo_em(root, tmp_path / "g3", {"patches": [{"nest": 1, "op": "add", "values": 3.0, "mask": "water"}]}, namelist=nl)
    with pytest.raises(ValueError, match="namelist"):
        patch_geo_em(root, tmp_path / "g4", {"patches": [{"nest": 1, "op": "add", "values": 3.0, "mask": "land"}]})


def test_check_real_base_state_and_reference(case, tmp_path, geo):
    root, nl = geo
    good = check_real(root, case, reference_dir=case, namelist=nl)
    assert good["passed"], [c for c in good["checks"] if c["status"] == "FAIL"]
    bad = tmp_path / "bad"
    bad.mkdir()
    for p in case.glob("wrfinput_d*"):
        bad.joinpath(p.name).write_bytes(p.read_bytes())
    with Dataset(bad / "wrfinput_d02", "r+") as ds:
        a = ds["MUB"][:]
        a[0, 7, 9] *= 1.001
        ds["MUB"][:] = a
    failed = {c["name"] for c in check_real(root, bad, reference_dir=case, namelist=nl)["checks"] if c["status"] == "FAIL"}
    assert failed == {"base_state_consistent"}


def test_cli_routes_through_real_parser(case, tmp_path):
    spec = tmp_path / "empty.json"
    spec.write_text("{}")
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT / "src"))
    run_ = subprocess.run([sys.executable, "-m", "gpuwrf", "patch", "land", "--input-dir", str(case), "--output-dir",
                           str(tmp_path / "cli"), "--spec", str(spec)], capture_output=True, text=True, env=env, timeout=300)
    assert run_.returncode == 0, run_.stderr
    assert json.loads(run_.stdout)["files"]["wrfinput_d02"]["byte_identical"]
    bad = subprocess.run([sys.executable, "-m", "gpuwrf", "patch", "verify", "--input-dir", str(case), "--patched-dir", str(tmp_path / "cli")],
                         capture_output=True, text=True, env=env, timeout=300)
    assert bad.returncode == 0 and json.loads(bad.stdout)["passed"]


def test_z0_control_from_basis_masks_factor_and_refuses_stale_basis(case, tmp_path):
    from gpuwrf.io.static_patch import patch_z0
    from gpuwrf.surface_params.cli import BASIS_SCHEMA
    from gpuwrf.surface_params.controls import load_controls
    ivg = read(case / "wrfinput_d02", "IVGTYP")[0]
    base = np.where(ivg == 10, 0.12, np.where(ivg == 16, 0.01, 0.0))
    manifest = {"schema": BASIS_SCHEMA, "domains": {"d02": {"grid_shape": list(ivg.shape),
                "controls": {"z0mvt": {"active": True, "dtype": "float32", "shape": list(ivg.shape)}}}}}
    def write(path, ivgtyp):
        np.savez(path, **{"d02/_ivgtyp": ivgtyp.astype(np.int32), "d02/z0mvt": base,
                          "manifest": np.frombuffer(json.dumps(manifest).encode(), np.uint8)})
    write(tmp_path / "basis.npz", ivg)
    spec = {"patches": [{"nest": 2, "mask": [BOX, "land"], "factor": 2.5}]}
    report = patch_z0(case, tmp_path / "basis.npz", spec, tmp_path / "z0.npz")
    controls, _ = load_controls(tmp_path / "z0.npz")
    lat, lon = grid(2)
    from gpuwrf.io.static_patch import resolve_mask
    mask, _ = resolve_mask([BOX, "land"], {"lat": lat, "lon": lon, "landmask": read(case / "wrfinput_d02", "LANDMASK")[0],
                                           "lu": read(case / "wrfinput_d02", "LU_INDEX")[0]}, tmp_path)
    z0 = controls["d02"]["z0mvt"]
    assert z0.dtype == np.float32 and report["patches"][0]["cells"] == int(mask.sum())
    assert np.array_equal(z0[mask], (base[mask] * 2.5).astype(np.float32))
    assert np.array_equal(z0[~mask], base[~mask].astype(np.float32))
    with pytest.raises(ValueError, match="must be new"):
        patch_z0(case, tmp_path / "basis.npz", spec, tmp_path / "z0.npz")
    stale = ivg.copy()
    stale[0, 0] = 5
    write(tmp_path / "stale.npz", stale)
    with pytest.raises(ValueError, match="rebuild the basis"):
        patch_z0(case, tmp_path / "stale.npz", spec, tmp_path / "z1.npz")


# ---------------------------------------------------------------- critic FIX round 1 (v50-patch-cr 2026-11-11)
def copy_case(src, dst, names):
    dst.mkdir()
    for n in names:
        (dst / n).write_bytes((src / n).read_bytes())
    return dst


def edit_cell(path, values):
    with Dataset(path, "r+") as ds:
        for name, (cell, value) in values.items():
            a = ds[name][:]
            if a.ndim == 4:
                a[0, :, cell[0], cell[1]] = value
            else:
                a[0, cell[0], cell[1]] = value
            ds[name][:] = a


def test_fix1_missing_parent_fails_closed_not_waivable(case, tmp_path, geo):
    only = copy_case(case, tmp_path / "only_d02", ["wrfinput_d02", "wrflowinp_d02"])
    zone = {"nest": 2, "kind": "soil", "field": "SMOIS", "op": "scale", "values": 1.1, "mask": [BOX, "land"]}
    with pytest.raises(StaticPatchError, match="nest_parent_missing"):
        patch_land(only, tmp_path / "o1", {"patches": [zone], "spec_bdy_width": 1,
                                         "waivers": {"nest_boundary_zone": "x", "nest_coverage": "x"}}, namelist=case / "namelist.input")
    edited = copy_case(only, tmp_path / "edited", ["wrfinput_d02", "wrflowinp_d02"])
    edit_cell(edited / "wrfinput_d02", {"SMOIS": ((0, 10), 0.11)})
    failed = {c["name"] for c in verify_patched(only, edited, spec_bdy_width=1)["checks"] if c["status"] == "FAIL"}
    assert "nest_parent_missing" in failed
    root, nl = geo
    g2 = copy_case(root, tmp_path / "g_only", ["geo_em.d02.nc"])
    with pytest.raises(StaticPatchError, match="nest_parent_missing"):
        patch_geo_em(g2, tmp_path / "g_out", {"patches": [{"nest": 2, "op": "add", "values": 5.0, "mask": [BOX, "land"]}],
                                             "waivers": {"nest_blend_zone": "x", "gwd_subgrid_orography": "x"}}, namelist=nl)
    real2 = copy_case(case, tmp_path / "real_only", ["wrfinput_d02"])
    failed = {c["name"] for c in check_real(g2, real2, namelist=nl)["checks"] if c["status"] == "FAIL"}
    assert "nest_parent_missing" in failed


def test_fix2_parent_terrain_under_child_zone_and_refless_check_real(case, tmp_path, geo):
    root, nl = geo
    lat, lon = grid(1)
    cell = np.zeros(lat.shape, bool)
    cell[J0, I0 + 2] = True                    # parent cell under the child's spec+blend zone (land)
    np.save(tmp_path / "pz.npy", cell)
    spec = {"patches": [{"nest": 1, "op": "add", "values": 30.0, "mask": {"file": str(tmp_path / "pz.npy")}}],
            "spec_bdy_width": 1, "blend_width": 1, "waivers": {"gwd_subgrid_orography": "test"}}
    with pytest.raises(StaticPatchError, match="parent_under_child_zone"):
        patch_geo_em(root, tmp_path / "g1", spec, namelist=nl)
    refless = check_real(root, case, namelist=nl)
    assert {c["name"] for c in refless["checks"] if c["status"] == "FAIL"} == {"nest_terrain_zone_unchanged"}


def test_fix3_verify_fails_stale_dependents_and_missing_aux(case, tmp_path):
    names = ["wrfinput_d01", "wrfinput_d02", "wrflowinp_d01", "wrflowinp_d02"]
    stale = copy_case(case, tmp_path / "stale", names)
    lu = read(case / "wrfinput_d02", "LU_INDEX")[0]
    j, i = np.argwhere(lu[4:11, 4:11] == 16)[0] + 4      # interior class-16 land cell
    luf = np.zeros(21); luf[9] = 1.0
    edit_cell(stale / "wrfinput_d02", {"LU_INDEX": ((j, i), 10), "IVGTYP": ((j, i), 10), "LANDUSEF": ((j, i), luf)})
    report = verify_patched(case, stale, spec_bdy_width=1)
    assert "dependents_updated" in {c["name"] for c in report["checks"] if c["status"] == "FAIL"}
    waived = verify_patched(case, stale, spec_bdy_width=1, waivers={"dependent_keep": "hand edit keeps climatology"})
    assert waived["passed"] and any(c["status"] == "WAIVED" for c in waived["checks"] if c["name"] == "dependents_updated")
    no_aux = copy_case(case, tmp_path / "no_aux", ["wrfinput_d01", "wrfinput_d02", "wrflowinp_d01"])
    assert "aux_present" in {c["name"] for c in verify_patched(case, no_aux, spec_bdy_width=1)["checks"] if c["status"] == "FAIL"}


def test_fix4_incomplete_flip_is_caught(case, tmp_path):
    names = ["wrfinput_d01", "wrfinput_d02", "wrflowinp_d01", "wrflowinp_d02"]
    bad = copy_case(case, tmp_path / "flip", names)
    cell = (7, 7)
    assert read(case / "wrfinput_d02", "LANDMASK")[0][cell] == 1
    luf = np.zeros(21); luf[16] = 1.0
    sct = np.zeros(16); sct[13] = 1.0
    edit_cell(bad / "wrfinput_d02", {"LU_INDEX": (cell, 17), "IVGTYP": (cell, 17), "LANDUSEF": (cell, luf), "LANDMASK": (cell, 0),
                                     "XLAND": (cell, 2), "ISLTYP": (cell, 14), "SOILCTOP": (cell, sct), "SOILCBOT": (cell, sct)})
    failed = {c["name"] for c in verify_patched(case, bad, spec_bdy_width=1)["checks"] if c["status"] == "FAIL"}
    assert "flip_set" in failed


def test_fix5_hard_ranges_not_waivable_and_post_landuse_soil_mask(case, tmp_path):
    waive = {"field_ranges": "x", "soil_porosity": "x"}
    huge = [{"nest": n, "kind": "soil", "field": "SMOIS", "op": "scale", "values": 1e39, "mask": [BOX, "land"]} for n in (1, 2)]
    with pytest.raises(StaticPatchError, match="hard_ranges"):
        run(case, tmp_path, huge, waivers=waive)
    hot = [{"nest": n, "kind": "soil", "field": "TSLB", "values": 400.0, "mask": [BOX, "land"]} for n in (1, 2)]
    with pytest.raises(StaticPatchError, match="hard_ranges"):
        run(case, tmp_path, hot, name="o2", waivers=waive)
    lat, lon = grid(2)
    cell = np.zeros(lat.shape, bool); cell[7, 6] = True
    np.save(tmp_path / "c.npy", cell)
    flip = {"nest": 2, "kind": "landuse", "class": 17, "mask": {"file": str(tmp_path / "c.npy")}, "dependents": "donor"}
    soil = {"nest": 2, "kind": "soil", "field": "TSLB", "op": "add", "values": 1.0, "mask": {"file": str(tmp_path / "c.npy")}}
    with pytest.raises(ValueError, match="land cells only"):
        run(case, tmp_path, [flip, soil], name="o3")


def test_fix6_d01_terrain_with_gwd_needs_waiver(case, tmp_path, geo):
    root, nl = geo
    gwd = tmp_path / "namelist.input"
    gwd.write_text(nl.read_text().replace("&geogrid", "&physics\n gwd_opt = 1, 0,\n/\n&geogrid"))
    lat, lon = grid(1)
    cell = np.zeros(lat.shape, bool); cell[1, 10] = True          # d01 land, outside the child
    np.save(tmp_path / "d01.npy", cell)
    spec = {"patches": [{"nest": 1, "op": "add", "values": 10.0, "mask": {"file": str(tmp_path / "d01.npy")}}]}
    with pytest.raises(StaticPatchError, match="gwd_subgrid_orography"):
        patch_geo_em(root, tmp_path / "g1", spec, namelist=gwd)
    ok = patch_geo_em(root, tmp_path / "g2", dict(spec, waivers={"gwd_subgrid_orography": "VAR/OA/OL kept from the old DEM"}), namelist=gwd)
    assert ok["passed"]


# ---------------------------------------------------------------- critic Delta 1 (child-missing variant)
def write_case_namelist(path, max_dom):
    path.write_text(f"&domains\n max_dom = {max_dom},\n spec_bdy_width = 1,\n parent_id = 1, 1,\n"
                    f" parent_grid_ratio = 1, {R},\n i_parent_start = 1, {I0 + 1},\n j_parent_start = 1, {J0 + 1},\n/\n"
                    "&physics\n gwd_opt = 0, 0,\n/\n")
    return path


def test_delta1_land_needs_namelist_and_child_missing_fails(case, tmp_path):
    only = copy_case(case, tmp_path / "only_d01", ["wrfinput_d01", "wrflowinp_d01"])
    spec = {"patches": [{"nest": 1, "kind": "soil", "field": "SMOIS", "op": "scale", "values": 1.1, "mask": [BOX, "land"]}]}
    with pytest.raises(ValueError, match="namelist"):
        patch_land(only, tmp_path / "o1", spec)
    nl = write_case_namelist(tmp_path / "namelist.input", 2)
    with pytest.raises(StaticPatchError, match="nest_set"):
        patch_land(only, tmp_path / "o2", spec, namelist=nl)
    assert patch_land(only, tmp_path / "o3", {}, namelist=None)["files"]["wrfinput_d01"]["byte_identical"]


def test_delta1_geo_em_and_check_real_child_missing_fail(case, tmp_path, geo):
    root, nl = geo
    g1 = copy_case(root, tmp_path / "g_d01", ["geo_em.d01.nc"])
    spec = {"patches": [{"nest": 1, "op": "add", "values": 5.0, "mask": [BOX, "land"]}]}
    with pytest.raises(StaticPatchError, match="nest_set"):
        patch_geo_em(g1, tmp_path / "g_out", spec, namelist=nl)
    with pytest.raises(ValueError, match="namelist"):
        patch_geo_em(g1, tmp_path / "g_out2", spec)
    r1 = copy_case(case, tmp_path / "r_d01", ["wrfinput_d01"])
    failed = {c["name"] for c in check_real(g1, r1, reference_dir=case, namelist=nl)["checks"] if c["status"] == "FAIL"}
    assert "nest_set" in failed
    with pytest.raises(ValueError, match="namelist"):
        check_real(g1, r1, reference_dir=case)
