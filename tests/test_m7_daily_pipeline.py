from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpuwrf.integration.daily_pipeline import (
    DailyCase,
    DailyPipelineConfig,
    build_wrfout_inventory,
    compare_wrfouts_xarray,
    execute_daily_pipeline,
    hour_steps,
    resolve_run_dir,
)


def _synthetic_state(nx: int = 3, ny: int = 2, nz: int = 2) -> SimpleNamespace:
    y2, x2 = np.indices((ny, nx), dtype=np.float32)
    z3 = np.arange(nz, dtype=np.float32)[:, None, None]
    zf = np.arange(nz + 1, dtype=np.float32)[:, None, None]
    terrain = 10.0 + y2 + x2
    pb = 90_000.0 - 500.0 * z3 + np.zeros((nz, ny, nx), dtype=np.float32)
    p_pert = 100.0 + np.zeros((nz, ny, nx), dtype=np.float32)
    phb = 9.81 * (terrain[None, :, :] + 500.0 * zf)
    ph_pert = np.ones((nz + 1, ny, nx), dtype=np.float32)
    mub = 85_000.0 + terrain
    mu_pert = 10.0 + np.zeros((ny, nx), dtype=np.float32)
    return SimpleNamespace(
        u=np.full((nz, ny, nx + 1), 4.0, dtype=np.float32),
        v=np.full((nz, ny + 1, nx), 1.0, dtype=np.float32),
        w=np.zeros((nz + 1, ny, nx), dtype=np.float32),
        theta=300.0 + z3 + np.zeros((nz, ny, nx), dtype=np.float32),
        qv=np.full((nz, ny, nx), 0.008, dtype=np.float32),
        qc=np.zeros((nz, ny, nx), dtype=np.float32),
        qi=np.zeros((nz, ny, nx), dtype=np.float32),
        qr=np.zeros((nz, ny, nx), dtype=np.float32),
        p_total=pb + p_pert,
        p_perturbation=p_pert,
        ph_total=phb + ph_pert,
        ph_perturbation=ph_pert,
        mu_total=mub + mu_pert,
        mu_perturbation=mu_pert,
        u10=np.full((ny, nx), 4.0, dtype=np.float32),
        v10=np.full((ny, nx), 1.0, dtype=np.float32),
        t2=np.full((ny, nx), 289.0, dtype=np.float32),
        q2=np.full((ny, nx), 0.007, dtype=np.float32),
        psfc=pb[0] + p_pert[0],
        rainc=np.zeros((ny, nx), dtype=np.float32),
        rain_acc=np.zeros((ny, nx), dtype=np.float32),
        rainsh=np.zeros((ny, nx), dtype=np.float32),
        swdown=np.full((ny, nx), 500.0, dtype=np.float32),
        glw=np.full((ny, nx), 300.0, dtype=np.float32),
        pblh=np.full((ny, nx), 800.0, dtype=np.float32),
        ustar=np.full((ny, nx), 0.3, dtype=np.float32),
        hfx=np.full((ny, nx), 20.0, dtype=np.float32),
        lh=np.full((ny, nx), 70.0, dtype=np.float32),
        t_skin=np.full((ny, nx), 290.0, dtype=np.float32),
        cldfra=np.zeros((nz, ny, nx), dtype=np.float32),
        landmask=np.ones((ny, nx), dtype=np.float32),
        lu_index=np.ones((ny, nx), dtype=np.float32) * 2.0,
    )


def _synthetic_case(run_dir: Path) -> DailyCase:
    state = _synthetic_state()
    grid = SimpleNamespace(
        nx=3,
        ny=2,
        nz=2,
        projection=SimpleNamespace(kind="lambert", lat_0=28.3, lon_0=-16.1, dx_m=3000.0, dy_m=3000.0),
        vertical=SimpleNamespace(nz=2, top_pressure_pa=5000.0),
        terrain_height=np.ones((2, 3), dtype=np.float32) * 10.0,
    )
    namelist = {
        "title": " OUTPUT FROM GPUWRF DAILY PIPELINE TEST",
        "truelat1": 25.0,
        "truelat2": 30.0,
        "stand_lon": -16.4,
        "moad_cen_lat": 28.3,
        "cen_lat": 28.3,
        "cen_lon": -16.1,
        "soil_layers_stag": 4,
    }
    return DailyCase(
        state=state,
        grid=grid,
        namelist=namelist,
        run_start=datetime(2026, 5, 21, 18, tzinfo=timezone.utc),
        metadata={"run_id": "synthetic", "run_dir": str(run_dir), "grid": {"mass_shape": [2, 2, 3]}},
    )


def test_resolve_run_dir_accepts_existing_path(tmp_path: Path) -> None:
    assert resolve_run_dir(str(tmp_path), Path("/does/not/matter")) == tmp_path
    assert resolve_run_dir("abc", tmp_path) == tmp_path / "abc"


def test_hour_steps_matches_ten_second_operational_dt() -> None:
    assert hour_steps(1, 10.0) == 360
    assert hour_steps(24, 10.0) == 8640


def test_execute_pipeline_writes_hourly_wrfouts_and_inventory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "rsl.error.0000").write_text(
        "\n".join(
            [
                "Timing for main: time 2026-05-21_18:00:10 on domain   2: 1.0 elapsed seconds",
                "Timing for main: time 2026-05-21_18:00:20 on domain   2: 1.0 elapsed seconds",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def case_builder(config: DailyPipelineConfig):
        return _synthetic_case(run_dir), run_dir

    def forecast_fn(state: SimpleNamespace, namelist, hours: float) -> SimpleNamespace:
        del namelist, hours
        state.t2 = state.t2 + 0.5
        state.theta = state.theta + 0.5
        return state

    config = DailyPipelineConfig(
        run_id="synthetic",
        hours=2,
        output_dir=tmp_path / "outputs",
        proof_dir=tmp_path / "proof",
        score=False,
    )
    payload = execute_daily_pipeline(config, forecast_fn=forecast_fn, case_builder=case_builder)

    assert payload["hours"] == 2
    assert len(payload["wrfout_files"]) == 2
    inventory = build_wrfout_inventory(payload["wrfout_files"])
    assert inventory["status"] == "PASS"
    assert (tmp_path / "proof" / "pipeline_run_20260521.json").is_file()
    assert (tmp_path / "proof" / "wrfout_inventory.json").is_file()


def test_compare_wrfouts_xarray_identical_final_file_passes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    case = _synthetic_case(run_dir)
    path = tmp_path / "wrfout_d02_2026-05-21_19:00:00"
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf

    write_wrfout_netcdf(
        case.state,
        case.grid,
        case.namelist,
        path,
        valid_time=datetime(2026, 5, 21, 19),
        lead_hours=1.0,
        run_start=case.run_start,
    )

    result = compare_wrfouts_xarray(path, path)

    assert result["status"] == "PASS"
    assert result["fields"]["T2"]["pass"] is True
