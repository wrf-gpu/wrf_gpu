from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset, chartostring

from gpuwrf.paths import reference_path
from gpuwrf.io.wrfout_writer import (
    MINIMUM_WRFOUT_VARIABLES,
    WRFOUT_VARIABLE_SPECS,
    write_wrfout_netcdf,
)


REFERENCE = reference_path(
    "runs",
    "wrf_l3",
    "20260525_18z_l3_24h_20260526T221207Z",
    "wrfout_d02_2026-05-25_18:00:00",
)


def synthetic_case(nx: int = 5, ny: int = 4, nz: int = 3):
    y2, x2 = np.indices((ny, nx), dtype=np.float32)
    z3 = np.arange(nz, dtype=np.float32)[:, None, None]
    zf = np.arange(nz + 1, dtype=np.float32)[:, None, None]
    terrain = 100.0 + y2 + 0.5 * x2
    pb = (90_000.0 - 800.0 * z3 + terrain[None, :, :]).astype(np.float32)
    p_pert = (100.0 + 0.5 * z3 + 0.1 * x2[None, :, :]).astype(np.float32)
    phb = (9.81 * (terrain[None, :, :] + 600.0 * zf)).astype(np.float32)
    ph_pert = (3.0 + 0.2 * zf + 0.01 * y2[None, :, :]).astype(np.float32)
    mub = (85_000.0 + terrain).astype(np.float32)
    mu_pert = (40.0 + 0.2 * x2 + 0.1 * y2).astype(np.float32)
    landmask = np.where((x2 + y2) % 3 == 0, 0.0, 1.0).astype(np.float32)

    state = SimpleNamespace(
        u=(4.0 + np.zeros((nz, ny, nx + 1), dtype=np.float32)),
        v=(1.5 + np.zeros((nz, ny + 1, nx), dtype=np.float32)),
        w=np.zeros((nz + 1, ny, nx), dtype=np.float32),
        theta=(300.0 + z3 + 0.1 * y2[None, :, :]).astype(np.float32),
        qv=(0.009 + np.zeros((nz, ny, nx), dtype=np.float32)),
        qc=(1.0e-5 + np.zeros((nz, ny, nx), dtype=np.float32)),
        qi=(2.0e-6 + np.zeros((nz, ny, nx), dtype=np.float32)),
        qr=(3.0e-6 + np.zeros((nz, ny, nx), dtype=np.float32)),
        p_total=pb + p_pert,
        p_perturbation=p_pert,
        ph_total=phb + ph_pert,
        ph_perturbation=ph_pert,
        mu_total=mub + mu_pert,
        mu_perturbation=mu_pert,
        u10=(4.2 + 0.01 * x2).astype(np.float32),
        v10=(1.1 + 0.01 * y2).astype(np.float32),
        t2=(289.0 + 0.1 * y2).astype(np.float32),
        q2=(0.008 + np.zeros((ny, nx), dtype=np.float32)),
        psfc=(pb[0] + p_pert[0]).astype(np.float32),
        rainc=np.zeros((ny, nx), dtype=np.float32),
        rain_acc=(0.2 + 0.01 * x2).astype(np.float32),
        rainsh=np.zeros((ny, nx), dtype=np.float32),
        swdown=(500.0 + x2).astype(np.float32),
        glw=(300.0 + y2).astype(np.float32),
        pblh=(800.0 + y2).astype(np.float32),
        ustar=(0.3 + np.zeros((ny, nx), dtype=np.float32)),
        hfx=(20.0 + y2).astype(np.float32),
        lh=(70.0 + x2).astype(np.float32),
        t_skin=(290.0 + y2).astype(np.float32),
        cldfra=(0.25 + np.zeros((nz, ny, nx), dtype=np.float32)),
        landmask=landmask,
        lu_index=np.where(landmask > 0.5, 2.0, 17.0).astype(np.float32),
    )
    grid = SimpleNamespace(
        nx=nx,
        ny=ny,
        nz=nz,
        projection=SimpleNamespace(kind="lambert", lat_0=28.34, lon_0=-16.12, dx_m=3000.0, dy_m=3000.0),
        vertical=SimpleNamespace(nz=nz, top_pressure_pa=5_000.0),
        terrain_height=terrain.astype(np.float32),
    )
    namelist = {
        "title": " OUTPUT FROM GPUWRF WRF-COMPATIBLE NETCDF WRITER",
        "truelat1": 25.0,
        "truelat2": 30.0,
        "stand_lon": -16.4,
        "moad_cen_lat": 28.3,
        "cen_lat": 28.34,
        "cen_lon": -16.12,
        "soil_layers_stag": 4,
    }
    return state, grid, namelist


def write_small(tmp_path: Path) -> tuple[Path, SimpleNamespace]:
    state, grid, namelist = synthetic_case()
    path = tmp_path / "wrfout_d02_2026-05-25_21:00:00"
    write_wrfout_netcdf(
        state,
        grid,
        namelist,
        path,
        valid_time=datetime(2026, 5, 25, 21),
        lead_hours=3.0,
        run_start=datetime(2026, 5, 25, 18),
    )
    return path, state


def test_roundtrip_writes_minimum_variables_and_time_coordinates(tmp_path: Path):
    path, _ = write_small(tmp_path)

    with Dataset(path) as dataset:
        missing = [name for name in MINIMUM_WRFOUT_VARIABLES if name not in dataset.variables]
        assert missing == []
        assert dataset.file_format == "NETCDF4"
        assert len(dataset.dimensions["Time"]) == 1
        assert len(dataset.dimensions["DateStrLen"]) == 19
        assert len(dataset.dimensions["west_east"]) == 5
        assert len(dataset.dimensions["west_east_stag"]) == 6
        assert len(dataset.dimensions["south_north"]) == 4
        assert len(dataset.dimensions["south_north_stag"]) == 5
        assert len(dataset.dimensions["bottom_top"]) == 3
        assert len(dataset.dimensions["bottom_top_stag"]) == 4
        assert chartostring(dataset["Times"][:])[0] == "2026-05-25_21:00:00"
        assert float(dataset["XTIME"][0]) == 180.0
        assert dataset.getncattr("START_DATE") == "2026-05-25_18:00:00"
        assert dataset.getncattr("WEST-EAST_GRID_DIMENSION") == 6


@pytest.mark.skipif(not REFERENCE.is_file(), reason="Gen2 reference wrfout unavailable")
def test_dim_and_attr_conformance_against_reference_schema(tmp_path: Path):
    path, _ = write_small(tmp_path)

    with Dataset(REFERENCE) as reference, Dataset(path) as candidate:
        for name in MINIMUM_WRFOUT_VARIABLES:
            assert name in reference.variables
            ref_var = reference.variables[name]
            out_var = candidate.variables[name]
            assert tuple(out_var.dimensions) == tuple(ref_var.dimensions)
            assert str(np.dtype(out_var.dtype)) == str(np.dtype(ref_var.dtype))
            for attr in ("units", "description", "MemoryOrder", "stagger"):
                if attr not in ref_var.ncattrs() and attr not in out_var.ncattrs():
                    continue
                assert str(out_var.getncattr(attr)) == str(ref_var.getncattr(attr))


def test_total_state_split_writes_base_and_perturbation_pairs(tmp_path: Path):
    path, state = write_small(tmp_path)

    with Dataset(path) as dataset:
        np.testing.assert_allclose(dataset["P"][0], state.p_perturbation)
        np.testing.assert_allclose(dataset["PB"][0], state.p_total - state.p_perturbation)
        np.testing.assert_allclose(dataset["P"][0] + dataset["PB"][0], state.p_total)
        np.testing.assert_allclose(dataset["PH"][0], state.ph_perturbation)
        np.testing.assert_allclose(dataset["PHB"][0], state.ph_total - state.ph_perturbation)
        np.testing.assert_allclose(dataset["PH"][0] + dataset["PHB"][0], state.ph_total)
        np.testing.assert_allclose(dataset["MU"][0], state.mu_perturbation)
        np.testing.assert_allclose(dataset["MUB"][0], state.mu_total - state.mu_perturbation)
        np.testing.assert_allclose(dataset["MU"][0] + dataset["MUB"][0], state.mu_total)


def test_variable_specs_cover_minimum_set():
    assert len(MINIMUM_WRFOUT_VARIABLES) == 41
    assert set(MINIMUM_WRFOUT_VARIABLES) - {"Times"} == set(WRFOUT_VARIABLE_SPECS)
