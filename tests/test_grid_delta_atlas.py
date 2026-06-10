from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_grid_delta_atlas.py"


def _load_atlas_module():
    spec = importlib.util.spec_from_file_location("build_grid_delta_atlas", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_char_var(ds: Dataset, name: str, dims: tuple[str, ...], text: str) -> None:
    var = ds.createVariable(name, "S1", dims)
    arr = np.asarray(list(text), dtype="S1")
    if len(dims) == 2:
        arr = arr.reshape(1, len(text))
    var[:] = arr


def _write_num_var(ds: Dataset, name: str, dims: tuple[str, ...], values: np.ndarray) -> None:
    var = ds.createVariable(name, "f8", dims)
    var.units = "test_units"
    var[:] = np.asarray(values, dtype=np.float64)


def _create_wrfout(path: Path, valid: datetime, *, side: str, lead: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = valid.strftime("%Y-%m-%d_%H:%M:%S")
    cpu_t2 = np.arange(4, dtype=np.float64).reshape(2, 2) + 10.0 * lead
    delta = np.asarray([[0.0, 0.1], [0.2, -0.2]], dtype=np.float64) * (lead + 1)
    t2 = cpu_t2 if side == "cpu" else cpu_t2 + delta
    cpu_u = np.arange(12, dtype=np.float64).reshape(2, 2, 3) + lead
    u = cpu_u if side == "cpu" else cpu_u + 0.5
    finite = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64) + lead
    nanfield = finite.copy()
    if side == "gpu" and lead == 1:
        nanfield[0, 1] = np.nan

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("bad_strlen", 4)
        ds.createDimension("south_north", 2)
        ds.createDimension("west_east", 2)
        ds.createDimension("west_east_stag", 3)
        ds.createDimension("bottom_top", 2)
        _write_char_var(ds, "Times", ("Time", "DateStrLen"), stamp)
        _write_char_var(ds, "BADTEXT", ("Time", "bad_strlen"), "abcd")
        _write_num_var(ds, "XTIME", ("Time",), np.asarray([lead * 60.0]))
        _write_num_var(ds, "T2", ("Time", "south_north", "west_east"), t2.reshape(1, 2, 2))
        _write_num_var(ds, "U", ("Time", "bottom_top", "south_north", "west_east_stag"), u.reshape(1, 2, 2, 3))
        _write_num_var(ds, "NANFIELD", ("Time", "south_north", "west_east"), nanfield.reshape(1, 2, 2))
        if side == "cpu":
            _write_num_var(ds, "SHAPE", ("Time", "south_north", "west_east"), np.ones((1, 2, 2)))
            _write_num_var(ds, "ONLY_CPU", ("Time", "south_north", "west_east"), np.ones((1, 2, 2)))
        else:
            _write_num_var(ds, "SHAPE", ("Time", "south_north", "west_east_stag"), np.ones((1, 2, 3)))
            _write_num_var(ds, "ONLY_GPU", ("Time", "south_north", "west_east"), np.ones((1, 2, 2)))


def _create_case(tmp_path: Path) -> tuple[Path, Path, datetime]:
    init = datetime(2026, 5, 1, 18, tzinfo=timezone.utc)
    cpu_dir = tmp_path / "cpu" / "20260501_18z_case"
    gpu_dir = tmp_path / "gpu" / "20260501_18z_case"
    for lead in (0, 1):
        valid = init + timedelta(hours=lead)
        name = f"wrfout_d02_{valid.strftime('%Y-%m-%d_%H:%M:%S')}"
        _create_wrfout(cpu_dir / name, valid, side="cpu", lead=lead)
        _create_wrfout(gpu_dir / name, valid, side="gpu", lead=lead)
    extra_valid = init + timedelta(hours=2)
    _create_wrfout(
        cpu_dir / f"wrfout_d02_{extra_valid.strftime('%Y-%m-%d_%H:%M:%S')}",
        extra_valid,
        side="cpu",
        lead=2,
    )
    return cpu_dir, gpu_dir, init


def test_grid_delta_atlas_records_inventory_and_metrics(tmp_path: Path) -> None:
    atlas = _load_atlas_module()
    cpu_dir, gpu_dir, init = _create_case(tmp_path)
    proof_dir = tmp_path / "proof"
    asset_dir = tmp_path / "assets"

    rc = atlas.main(
        [
            "--cpu-dir",
            str(cpu_dir),
            "--gpu-dir",
            str(gpu_dir),
            "--case-id",
            "synthetic_case",
            "--domain",
            "d02",
            "--init",
            init.isoformat(),
            "--proof-dir",
            str(proof_dir),
            "--asset-dir",
            str(asset_dir),
            "--no-plots",
            "--no-default-mandatory-fields",
        ]
    )
    assert rc == 0

    manifest = json.loads((proof_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((proof_dir / "grid_delta_summary.json").read_text(encoding="utf-8"))
    report = (proof_dir / "GRID_DELTA_ATLAS.md").read_text(encoding="utf-8")

    assert manifest["schema"] == "grid-delta-atlas-manifest-v1"
    assert manifest["pairing"]["paired_file_count"] == 2
    assert manifest["pairing"]["cases"][0]["domains_coverage"][0]["unmatched_cpu_count"] == 1
    assert manifest["plots"]["status"] == "disabled"
    assert summary["schema"] == "grid-delta-atlas-summary-v1"
    assert summary["inventory"]["compared_numeric_field_count"] >= 3
    assert summary["inventory"]["missing_record_count"] >= 2
    assert summary["inventory"]["non_numeric_record_count"] >= 1
    assert summary["inventory"]["nonfinite_record_count"] == 1
    assert summary["inventory"]["shape_mismatch_record_count"] == 2

    t2 = summary["field_metrics"]["T2"]["overall"]
    assert t2["count"] == 8
    assert t2["finite_pair_count"] == 8
    assert np.isclose(t2["max_abs"], 0.4)
    assert t2["p50_abs"] is not None
    assert t2["p95_abs"] is not None
    assert t2["p99_abs"] is not None
    assert t2["p999_abs"] is not None
    assert t2["safe_relative"]["count"] > 0
    assert t2["correlation"] is not None
    assert t2["worst"]["field"] if "field" in t2["worst"] else t2["worst"]["lead_h"] == 1

    nanfield = summary["field_metrics"]["NANFIELD"]
    assert nanfield["overall"]["nonfinite_gpu_count"] == 1
    assert nanfield["issues"]["nonfinite"][0]["nonfinite_gpu_count"] == 1
    assert "Grid-Delta Atlas" in report
    assert "Top Field Differences" in report


def test_grid_delta_atlas_case_json_and_lead_filter(tmp_path: Path) -> None:
    atlas = _load_atlas_module()
    cpu_dir, gpu_dir, init = _create_case(tmp_path)
    case_json = tmp_path / "cases.json"
    case_json.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "from_manifest",
                        "cpu_dir": str(cpu_dir),
                        "gpu_dir": str(gpu_dir),
                        "domains": ["d02"],
                        "init_time_utc": init.isoformat(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    proof_dir = tmp_path / "proof_case_json"

    assert atlas.main(
        [
            "--case-json",
            str(case_json),
            "--min-lead",
            "1",
            "--max-lead",
            "1",
            "--proof-dir",
            str(proof_dir),
            "--asset-dir",
            str(tmp_path / "assets_case_json"),
            "--no-plots",
            "--no-default-mandatory-fields",
        ]
    ) == 0

    manifest = json.loads((proof_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((proof_dir / "grid_delta_summary.json").read_text(encoding="utf-8"))
    assert manifest["pairing"]["paired_file_count"] == 1
    assert manifest["pairing"]["lead_hours"] == [1]
    assert summary["field_metrics"]["T2"]["overall"]["count"] == 4


def test_grid_delta_atlas_plot_generation_when_available(tmp_path: Path) -> None:
    atlas = _load_atlas_module()
    cpu_dir, gpu_dir, init = _create_case(tmp_path)
    proof_dir = tmp_path / "proof_plots"
    asset_dir = tmp_path / "assets_plots"

    assert atlas.main(
        [
            "--cpu-dir",
            str(cpu_dir),
            "--gpu-dir",
            str(gpu_dir),
            "--domain",
            "d02",
            "--init",
            init.isoformat(),
            "--proof-dir",
            str(proof_dir),
            "--asset-dir",
            str(asset_dir),
            "--no-default-mandatory-fields",
            "--spatial-plot-limit",
            "2",
        ]
    ) == 0
    manifest = json.loads((proof_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["plots"]["status"] == "ok":
        paths = {Path(plot["path"]).name for plot in manifest["plots"]["plots"] if "path" in plot}
        assert "dashboard.png" in paths
        assert "heatmap_rmse.png" in paths
        assert (asset_dir / "dashboard.png").is_file()
    else:
        assert manifest["plots"]["status"] == "skipped_missing_dependency"
