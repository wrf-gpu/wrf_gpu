"""v0234 SP2 regression: WRF-seasonal Noah-MP cold-start surface fields."""

from __future__ import annotations

from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import jax.numpy as jnp

from gpuwrf.io.land_state import load_prescribed_land_state
from gpuwrf.physics.noah_mp import (
    mavail_from_prescribed_fields,
    roughness_from_prescribed_fields,
)
from gpuwrf.physics.mynn_edmf import _wrf_superadiabatic_gate


def test_wrf_tskin_gate_parity_on_port_only_mass_flux_discriminator() -> None:
    """Real TSK reproduces WRF's gate across the 4,526-column discriminator."""

    ncol = 4526
    thv0 = np.full(ncol, 300.0, dtype=np.float64)
    qv0 = np.full(ncol, 0.01, dtype=np.float64)
    dz0 = np.full(ncol, 50.0, dtype=np.float64)
    is_water = (np.arange(ncol) % 3) == 0
    ts = np.where(np.arange(ncol) % 2, 300.2, 299.0)
    fltv2 = np.full(ncol, 0.01, dtype=np.float64)

    expected = np.where(
        ts > 0.0,
        (thv0 - ts * (1.0 + 0.608 * qv0)) / (0.5 * dz0)
        < np.where(is_water, -0.001, -0.003),
        fltv2 > 0.0,
    )
    actual = np.asarray(_wrf_superadiabatic_gate(
        jnp.asarray(thv0), jnp.asarray(ts), jnp.asarray(qv0), jnp.asarray(dz0),
        jnp.asarray(is_water), jnp.asarray(fltv2),
    ))
    np.testing.assert_array_equal(actual, expected)

    fallback = np.asarray(_wrf_superadiabatic_gate(
        jnp.asarray(thv0), jnp.full(ncol, -1.0), jnp.asarray(qv0), jnp.asarray(dz0),
        jnp.asarray(is_water), jnp.asarray(fltv2),
    ))
    np.testing.assert_array_equal(fallback, fltv2 > 0.0)


def test_winter_landuse_column_changes_mynn_drag_inputs() -> None:
    """Categories active in d03 use the pristine winter SFZ0/SLMO column."""

    xland = np.ones((1, 4), dtype=np.float64)
    landmask = np.ones_like(xland)
    lu = np.asarray([[5, 10, 12, 16]], dtype=np.float64)
    soil = np.ones((1, 1, 4), dtype=np.float64)

    summer_z0 = np.asarray(roughness_from_prescribed_fields(
        xland, landmask, lu_index=lu, season=1,
    ))
    winter_z0 = np.asarray(roughness_from_prescribed_fields(
        xland, landmask, lu_index=lu, season=2,
    ))
    winter_mavail = np.asarray(mavail_from_prescribed_fields(
        xland, landmask, soil, lu_index=lu, season=2,
    ))

    np.testing.assert_array_equal(summer_z0, [[0.50, 0.12, 0.15, 0.01]])
    np.testing.assert_array_equal(winter_z0, [[0.20, 0.10, 0.05, 0.01]])
    np.testing.assert_array_equal(winter_mavail, [[0.60, 0.30, 0.60, 0.05]])


class _Run:
    def __init__(self, path: Path, fields: dict[str, np.ndarray]):
        self.path = path
        self.fields = fields
        self.run_id = "winter-fixture"

    def wrfinput_file(self, domain: str) -> Path:
        assert domain == "d03"
        return self.path

    def wrfinput_variables(self, domain: str) -> list[str]:
        assert domain == "d03"
        return list(self.fields)

    def load_wrfinput(self, domain: str, name: str, lazy: bool = False):
        assert domain == "d03" and lazy is False
        return self.fields[name]


def test_loader_selects_northern_winter_from_wrfinput(tmp_path: Path) -> None:
    """March 1 at positive CEN_LAT must reproduce WRF ``ISN=2``."""

    path = tmp_path / "wrfinput_d03"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.frombuffer(b"2025-03-01_00:00:00", dtype="S1")
        dataset.setncattr("CEN_LAT", 28.279888)

    s2 = np.ones((1, 1), dtype=np.float64)
    s3 = np.ones((4, 1, 1), dtype=np.float64)
    fields = {
        "XLAND": s2,
        "LANDMASK": s2,
        "IVGTYP": np.full_like(s2, 5),
        "ISLTYP": np.full_like(s2, 8),
        "LU_INDEX": np.full_like(s2, 5),
        "SST": np.full_like(s2, 290.0),
        "TSK": np.full_like(s2, 289.0),
        "SMOIS": s3 * 0.2,
        "SH2O": s3 * 0.2,
        "TSLB": s3 * 288.0,
    }
    land = load_prescribed_land_state(_Run(path, fields), "d03")

    assert land.source["landuse_season"] == 2
    assert land.source["landuse_julday"] == 60
    assert float(land.roughness_m[0, 0]) == 0.20
    assert float(land.mavail[0, 0]) == 0.60
