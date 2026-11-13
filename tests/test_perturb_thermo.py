"""Seeded PBL theta/QV wrfinput noise (v0.5.0 item 0.2a): reproducibility, spread, OFF identity, THM consistency."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.perturb.thermo import SCHEMA, load_config, noise, prepare, vertical_weight

REPO = Path(__file__).resolve().parents[1]
NZ, NY, NX = 6, 14, 16
RVOVRD = np.float32(461.6 / 287.0)


def _case(path, *, nests=2, theta_m=1, width=2):
    path.mkdir()
    (path / "namelist.input").write_text(f"&domains\n max_dom={nests},\n/\n&bdy_control\n spec_bdy_width={width},\n/\n")
    rng = np.random.default_rng(7)
    for number in range(1, nests + 1):
        with Dataset(path / f"wrfinput_d{number:02d}", "w", format="NETCDF3_64BIT_OFFSET") as ds:
            for name, size in (("Time", 1), ("bottom_top", NZ), ("bottom_top_stag", NZ + 1),
                               ("south_north", NY), ("west_east", NX), ("west_east_stag", NX + 1)):
                ds.createDimension(name, size)
            xyz, stag = ("Time", "bottom_top", "south_north", "west_east"), ("Time", "bottom_top_stag", "south_north", "west_east")
            hgt = rng.uniform(0, 300, (1, NY, NX)).astype("f4")
            heights = np.array([0, 60, 200, 450, 800, 1500, 3000], "f4")  # AGL faces
            phb = np.broadcast_to((heights[None, :, None, None] + hgt[:, None]) * np.float32(9.81), (1, NZ + 1, NY, NX))
            ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))[:] = hgt
            ds.createVariable("PHB", "f4", stag)[:] = phb
            ds.createVariable("PH", "f4", stag)[:] = 0
            t = rng.uniform(-10, 20, (1, NZ, NY, NX)).astype("f4")
            qv = rng.uniform(1e-3, 1.5e-2, (1, NZ, NY, NX)).astype("f4")
            ds.createVariable("T", "f4", xyz)[:] = t
            ds.createVariable("QVAPOR", "f4", xyz)[:] = qv
            thm = (t + np.float32(300)) * (np.float32(1) + RVOVRD * qv) - np.float32(300) if theta_m else t
            ds.createVariable("THM", "f4", xyz)[:] = thm
            ds.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))[:] = 3.5
            ds.setncattr("USE_THETA_M", np.int32(theta_m))
            ds.description = "unchanged metadata"
    (path / "wrfbdy_d01").write_bytes(b"lateral")
    (path / "wrflowinp_d01").write_bytes(b"lower")
    return path


def _config(path, **change):
    cfg = {"schema": SCHEMA, "seed": 20260227, "theta_std_k": 0.2, "qv_rel_std": 0.02, "full_m": 300.0, "top_m": 1000.0}
    cfg.update(change)
    path.write_text(json.dumps(cfg))
    return path


def _arrays(path):
    with Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        return {n: np.asarray(ds[n][:]) for n in ds.variables}, {a: ds.getncattr(a) for a in ds.ncattrs()}


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_seed_reproducibility_and_member_independence(tmp_path):
    case = _case(tmp_path / "case")
    cfg = _config(tmp_path / "cfg.json")
    a = prepare(case, tmp_path / "a", members=3, config=cfg)
    b = prepare(case, tmp_path / "b", members=3, config=cfg)
    for ra, rb in zip(a["members"], b["members"]):
        assert {k: v["sha256"] for k, v in ra["prepared"].items()} == {k: v["sha256"] for k, v in rb["prepared"].items()}
    shas = {(m["id"], k): v["sha256"] for m in a["members"] for k, v in m["prepared"].items()}
    assert len(set(shas.values())) == len(shas)  # every member and nest differs
    other = prepare(case, tmp_path / "c", members=3, config=_config(tmp_path / "cfg2.json", seed=1))
    assert all(m["prepared"][k]["sha256"] != shas[(m["id"], k)] for m in other["members"] for k in m["prepared"])
    # The stream is the documented public PCG64 SeedSequence([seed, member, domain]) draw.
    gen = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260227, 1, 2])))
    expected = np.clip(gen.standard_normal((1, NZ, NY, NX)), -3, 3)
    np.testing.assert_array_equal(noise(20260227, 1, 2, (1, NZ, NY, NX), 3.)[0], expected)


def test_spread_positive_only_inside_pbl_interior_and_everything_else_byte_identical(tmp_path):
    case = _case(tmp_path / "case")
    result = prepare(case, tmp_path / "out", members=4, config=_config(tmp_path / "cfg.json"))
    src, src_attrs = _arrays(case / "wrfinput_d01")
    members = []
    for row in result["members"]:
        arrays, attrs = _arrays(Path(row["case"]) / "wrfinput_d01")
        assert attrs == src_attrs
        for name in set(src) - {"T", "QVAPOR", "THM"}:
            assert arrays[name].tobytes() == src[name].tobytes(), name
        members.append(arrays)
        for linked in ("namelist.input", "wrfbdy_d01", "wrflowinp_d01"):
            assert (Path(row["case"]) / linked).is_symlink()
    z_faces = (src["PH"] + src["PHB"]) / np.float32(9.81) - src["HGT"][:, None]
    z = 0.5 * (z_faces[:, :-1] + z_faces[:, 1:])
    weight = vertical_weight(z.astype(np.float64), 300., 1000.)
    weight[..., :2, :] = weight[..., -2:, :] = weight[..., :, :2] = weight[..., :, -2:] = 0
    t = np.stack([m["T"] for m in members])
    q = np.stack([m["QVAPOR"] for m in members])
    changed = np.any(t != src["T"], axis=0)
    assert np.array_equal(changed, weight > 0)            # ring and above top_m untouched
    full = weight == 1
    assert full.sum() > 100
    spread_t = t.std(axis=0, ddof=1)[full]
    spread_q = (q / src["QVAPOR"]).std(axis=0, ddof=1)[full]
    assert np.all(spread_t > 0) and np.all(spread_q > 0)
    # 4 members x cells: the pooled sample std matches the requested amplitude within 15 %.
    pooled_t = float(np.sqrt(np.mean(((t - src["T"])[:, full]) ** 2)))
    pooled_q = float(np.sqrt(np.mean(((q / src["QVAPOR"] - 1)[:, full]) ** 2)))
    assert abs(pooled_t / 0.2 - 1) < .15 and abs(pooled_q / 0.02 - 1) < .15
    assert float(np.max(np.abs(t - src["T"]))) <= 3 * 0.2 + 1e-5
    assert np.all(q >= 0)


@pytest.mark.parametrize("theta_m", [1, 0])
def test_thm_follows_real_exe_theta_m_and_untouched_cells_keep_bytes(tmp_path, theta_m):
    case = _case(tmp_path / "case", theta_m=theta_m)
    row = prepare(case, tmp_path / "out", members=2, config=_config(tmp_path / "cfg.json"))["members"][1]
    src, _ = _arrays(case / "wrfinput_d02")
    new, _ = _arrays(Path(row["case"]) / "wrfinput_d02")
    touched = new["T"] != src["T"]
    assert touched.any()
    assert new["THM"][~touched].tobytes() == src["THM"][~touched].tobytes()
    if theta_m:
        expected = (new["T"] + np.float32(300)) * (np.float32(1) + RVOVRD * new["QVAPOR"]) - np.float32(300)
        np.testing.assert_allclose(new["THM"][touched], expected[touched], atol=1e-4, rtol=0)
    else:
        assert np.array_equal(new["THM"][touched], new["T"][touched])


def test_refusals(tmp_path):
    case = _case(tmp_path / "case")
    for change, message in ((dict(theta_std_k=0, qv_rel_std=0), "both zero"), (dict(top_m=200.), "top_m"),
                            (dict(qv_rel_std=.5), "qv_rel_std"), (dict(domains=[3]), "beyond max_dom"),
                            (dict(seed=-1), "seed"), (dict(extra=1), "unknown")):
        with pytest.raises(ValueError, match=message):
            prepare(case, tmp_path / f"o{len(message)}{message[:3]}", members=2, config=_config(tmp_path / "c.json", **change))
    with pytest.raises(ValueError, match="schema"):
        load_config(_config(tmp_path / "s.json", schema="v0"))
    (tmp_path / "exists").mkdir()
    with pytest.raises(ValueError, match="exists"):
        prepare(case, tmp_path / "exists", members=2, config=_config(tmp_path / "c.json"))
    assert not any(p.name.startswith("o") for p in tmp_path.iterdir() if p.is_dir())  # failed runs leave nothing


def test_domains_subset_links_unedited_nest(tmp_path):
    case = _case(tmp_path / "case")
    row = prepare(case, tmp_path / "out", members=2, config=_config(tmp_path / "c.json", domains=[2]))["members"][0]
    assert (Path(row["case"]) / "wrfinput_d01").is_symlink()
    assert not (Path(row["case"]) / "wrfinput_d02").is_symlink()


def test_cli_routes_thermo_and_keeps_wind_contract(tmp_path):
    case = _case(tmp_path / "case")
    env = {"JAX_PLATFORMS": "cpu", "PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"}
    run = lambda *args: subprocess.run([sys.executable, "-m", "gpuwrf.cli", "perturb", *args],
                                       capture_output=True, text=True, env=env, cwd=REPO)
    ok = run("--case", str(case), "--output", str(tmp_path / "t"), "--members", "2",
             "--thermo-config", str(_config(tmp_path / "c.json")))
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["schema"] == SCHEMA
    both = run("--case", str(case), "--output", str(tmp_path / "x"), "--members", "2", "--wind-std", ".05",
               "--thermo-config", str(tmp_path / "c.json"))
    assert both.returncode == 2 and "not allowed with" in both.stderr
    seedless = run("--case", str(case), "--output", str(tmp_path / "y"), "--members", "2", "--wind-std", ".05")
    assert seedless.returncode == 2 and "requires --seed" in seedless.stderr
    seeded = run("--case", str(case), "--output", str(tmp_path / "z"), "--members", "2",
                 "--thermo-config", str(tmp_path / "c.json"), "--seed", "4")
    assert seeded.returncode == 2 and "drop --seed" in seeded.stderr


def test_refuses_recipe_that_changes_no_cell(tmp_path):
    """Critic finding 6: a layer below the first mass level (30 m AGL here) is an ensemble of source copies."""
    case = _case(tmp_path / "case")
    with pytest.raises(ValueError, match="changes no T/QVAPOR cell"):
        prepare(case, tmp_path / "out", members=2, config=_config(tmp_path / "c.json", full_m=0., top_m=1.))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("declared", [False, True])
def test_refuses_fill_values_and_never_publishes_nonfinite_thm(tmp_path, declared):
    """Critic finding 7: masking is off for byte-exact copies, so a fill value is a large finite number."""
    import netCDF4
    case = _case(tmp_path / "case")
    with Dataset(case / "wrfinput_d01", "a") as ds:
        ds.set_auto_maskandscale(False)
        fill = np.float32(-999.) if declared else np.float32(netCDF4.default_fillvals["f4"])
        if declared:
            ds["QVAPOR"].setncattr("missing_value", fill)
        q = ds["QVAPOR"][:]
        q[0, 0, NY // 2, NX // 2] = fill
        ds["QVAPOR"][:] = q
    with pytest.raises(ValueError, match="fill/missing values"):
        prepare(case, tmp_path / "out", members=2, config=_config(tmp_path / "c.json"))
    assert not (tmp_path / "out").exists()
