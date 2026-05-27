from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np

from gpuwrf.coupling.boundary_apply import BoundaryConfig, SIDE_INDEX, apply_lateral_boundaries
from gpuwrf.integration.d02_replay import _field_sides_2d, _field_sides_3d
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.land_state import load_hourly_land_state
from gpuwrf.runtime.operational_mode import _limit_theta_by_level


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m6_run_dummy_coupled.py"
_SPEC = importlib.util.spec_from_file_location("m6_run_dummy_coupled", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
make_dummy_grid = _MODULE.make_dummy_grid
make_initial_state = _MODULE.make_initial_state


def test_theta_guard_keeps_200_floor_and_widens_lower_column_ceiling_to_450k():
    origin = jnp.ones((44, 1, 1), dtype=jnp.float64) * 300.0
    origin = origin.at[0, 0, 0].set(150.0).at[29, 0, 0].set(500.0).at[31, 0, 0].set(800.0)
    candidate = origin.at[0, 0, 0].set(100.0).at[29, 0, 0].set(500.0).at[31, 0, 0].set(800.0)

    limited = _limit_theta_by_level(candidate, origin)

    assert float(limited[0, 0, 0]) == 200.0
    assert float(limited[29, 0, 0]) == 450.0
    assert float(limited[31, 0, 0]) == 700.0


def test_d02_side_helpers_return_wrf_ordered_five_row_strips():
    field3 = np.arange(2 * 4 * 6, dtype=np.float64).reshape(2, 4, 6)
    strips3 = _field_sides_3d(field3, width=3)

    assert strips3["W"].shape == (3, 2, 4)
    assert np.array_equal(strips3["W"][0], field3[:, :, 0])
    assert np.array_equal(strips3["W"][2], field3[:, :, 2])
    assert np.array_equal(strips3["E"][0], field3[:, :, -1])
    assert np.array_equal(strips3["E"][2], field3[:, :, -3])
    assert np.array_equal(strips3["S"][1], field3[:, 1, :])
    assert np.array_equal(strips3["N"][1], field3[:, -2, :])

    field2 = np.arange(4 * 6, dtype=np.float64).reshape(4, 6)
    strips2 = _field_sides_2d(field2, width=3)
    assert np.array_equal(strips2["W"][0], field2[:, 0])
    assert np.array_equal(strips2["E"][2], field2[:, -3])
    assert np.array_equal(strips2["N"][0], field2[-1, :])


def test_lateral_boundary_apply_uses_strip_matching_relax_offset():
    grid = make_dummy_grid(8, 8, 3)
    state = make_initial_state(grid)
    side_len = max(grid.nx + 1, grid.ny + 1)
    width = 5

    def leaf(z_len: int, dtype=jnp.float32):
        values = jnp.zeros((1, 4, width, z_len, side_len), dtype=dtype)
        for offset, value in enumerate((10.0, 20.0, 30.0, 40.0, 50.0)):
            values = values.at[:, SIDE_INDEX["W"], offset, :, :].set(value)
        return values

    zero_u = jnp.zeros_like(state.u)
    candidate = state.replace(
        u=zero_u,
        u_bdy=leaf(grid.nz),
        v_bdy=leaf(grid.nz),
        theta_bdy=leaf(grid.nz),
        qv_bdy=leaf(grid.nz),
        ph_bdy=leaf(grid.nz + 1, dtype=jnp.float64),
        mu_bdy=leaf(1, dtype=jnp.float64),
    )
    out = apply_lateral_boundaries(
        candidate,
        0.0,
        10.0,
        BoundaryConfig(spec_bdy_width=5, spec_zone=1, relax_zone=4, update_cadence_s=3600.0),
    )

    assert float(out.u[0, 4, 0]) == 10.0
    assert abs(float(out.u[0, 4, 1]) - 2.2) < 1.0e-6


def _write_land_wrfout(path: Path, *, tsk: float, sst: float, smois: float) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("soil_layers_stag", 4)
        dataset.createDimension("south_north", 2)
        dataset.createDimension("west_east", 3)

        def var2(name: str, value: float):
            var = dataset.createVariable(name, "f8", ("Time", "south_north", "west_east"))
            var[0, :, :] = value
            return var

        def var_soil(name: str, value: float):
            var = dataset.createVariable(name, "f8", ("Time", "soil_layers_stag", "south_north", "west_east"))
            var[0, :, :, :] = value
            return var

        var2("TSK", tsk)
        var2("SST", sst)
        var_soil("SMOIS", smois)
        var_soil("SH2O", smois * 0.8)
        var_soil("TSLB", tsk - 1.0)
        xland = var2("XLAND", 1.0)
        landmask = var2("LANDMASK", 1.0)
        xland[0, 0, 0] = 2.0
        landmask[0, 0, 0] = 0.0
        var2("LAKEMASK", 0.0)
        var2("IVGTYP", 2.0)
        var2("ISLTYP", 4.0)
        var2("LU_INDEX", 2.0)
        var2("VEGFRA", 50.0)
        var2("CM", 0.0)


def test_hourly_land_state_uses_wrfout_time_slice_and_sst_over_water(tmp_path: Path):
    _write_land_wrfout(tmp_path / "wrfout_d02_2026-05-21_18:00:00", tsk=280.0, sst=290.0, smois=0.20)
    _write_land_wrfout(tmp_path / "wrfout_d02_2026-05-21_19:00:00", tsk=281.0, sst=299.0, smois=0.42)

    land = load_hourly_land_state(Gen2Run(tmp_path), "d02", time=1)

    assert land.source["time_index"] == 1
    assert float(np.asarray(land.t_skin)[0, 0]) == 299.0
    assert float(np.asarray(land.t_skin)[1, 1]) == 281.0
    assert float(np.asarray(land.soil_moisture)[0, 1, 1]) == 0.42
