"""P0-5b: full-carry wrfrst-equivalent restart CPU bit-fidelity (GPU-free).

These run on CPU: the State is built from the frozen field-shape contract (NOT
State.zeros, which hard-requires a GPU), and the carry via
``initial_operational_carry``.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.noahmp_state import NoahMPLandState
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.io import restart as restart_mod
from gpuwrf.io.restart import read_restart, read_restart_metadata, write_restart
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry


def _state(grid: GridSpec) -> State:
    def pattern(shape, dtype, offset):
        values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape) + offset / 991.0 + 0.25
        return jnp.asarray(values, dtype=dtype)

    return State(
        **{
            field: pattern(shape, DEFAULT_DTYPES.dtype_for(field), index)
            for index, (field, shape) in enumerate(_state_field_shapes(grid).items(), start=1)
        }
    )


def _land(grid: GridSpec) -> NoahMPLandState:
    ny, nx = grid.ny, grid.nx

    def s2(off):
        return jnp.asarray(np.arange(ny * nx, dtype=np.float64).reshape(ny, nx) + off)

    def soil(off):
        return jnp.asarray(np.arange(4 * ny * nx, dtype=np.float64).reshape(4, ny, nx) + off)

    def snow(off):
        return jnp.asarray(np.arange(3 * ny * nx, dtype=np.float64).reshape(3, ny, nx) + off)

    return NoahMPLandState(
        tslb=soil(285.0), smois=soil(0.3), sh2o=soil(0.28), smcwtd=s2(0.31),
        isnow=jnp.asarray(np.zeros((ny, nx), dtype=np.int32)),
        tsno=snow(270.0), snice=snow(0.0), snliq=snow(0.0),
        zsnso=jnp.asarray(np.full((7, ny, nx), -0.1)),
        snowh=s2(0.0), sneqv=s2(0.0), sneqvo=s2(0.0), tauss=s2(0.0), albold=s2(0.2),
        tv=s2(288.0), tg=s2(287.0), tah=s2(288.5), eah=s2(1000.0),
        canliq=s2(0.01), canice=s2(0.0), fwet=s2(0.0), lai=s2(2.0), sai=s2(0.5),
        cm=s2(0.012), ch=s2(0.011), t_skin=s2(287.2), qsfc=s2(0.008), znt=s2(0.1),
        emiss=s2(0.98), albedo=s2(0.2), sfcrunoff=s2(0.001), udrunoff=s2(0.0005),
    )


def _equal(a, b) -> bool:
    a_h, b_h = np.asarray(a), np.asarray(b)
    return bool(a_h.shape == b_h.shape and a_h.dtype == b_h.dtype and np.array_equal(a_h, b_h))


def _namelist(grid: GridSpec) -> OperationalNamelist:
    return OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=10.0, acoustic_substeps=10)


def test_full_carry_roundtrip_bit_identical(tmp_path: Path) -> None:
    grid = GridSpec.canary_3km_template()
    rad = (
        jnp.asarray(np.full((grid.ny, grid.nx), 412.0)),
        jnp.asarray(np.full((grid.ny, grid.nx), 305.0)),
        jnp.asarray(np.full((grid.ny, grid.nx), 0.42)),
    )
    carry = initial_operational_carry(_state(grid), noahmp_land=_land(grid), noahmp_rad=rad)
    path = tmp_path / "full.wrfrst"
    write_restart(carry, _namelist(grid), grid, 137, path)

    restored, _, restored_grid, step = read_restart(path)
    assert step == 137
    assert restored.state.active_field_names() == carry.state.active_field_names()
    for field in carry.state.active_field_names():
        assert _equal(getattr(carry.state, field), getattr(restored.state, field)), field
    for field in restart_mod._CARRY_SCRATCH_FIELDS:
        assert _equal(getattr(carry, field), getattr(restored, field)), field
    for field in NoahMPLandState.__slots__:
        assert _equal(getattr(carry.noahmp_land, field), getattr(restored.noahmp_land, field)), field
    for i in range(3):
        assert _equal(carry.noahmp_rad[i], restored.noahmp_rad[i]), i


def test_held_rthraten_survives_restart(tmp_path: Path) -> None:
    """rthraten is the WRF held radiative tendency: it MUST round-trip non-zero,
    not be re-seeded to zero (the mid-radiation-interval continuity guarantee)."""
    grid = GridSpec.canary_3km_template()
    carry = initial_operational_carry(_state(grid))
    held = jnp.asarray(np.full_like(np.asarray(carry.rthraten), 1.234e-4))
    carry = carry.replace(rthraten=held)
    path = tmp_path / "held.wrfrst"
    write_restart(carry, _namelist(grid), grid, 9, path)
    restored, _, _, _ = read_restart(path)
    assert _equal(carry.rthraten, restored.rthraten)
    assert float(np.asarray(restored.rthraten).flat[0]) == pytest.approx(1.234e-4)


def test_landless_roundtrip_and_metadata(tmp_path: Path) -> None:
    grid = GridSpec.canary_3km_template()
    carry = initial_operational_carry(_state(grid))
    path = tmp_path / "landless.wrfrst"
    write_restart(carry, _namelist(grid), grid, 3, path)
    meta = read_restart_metadata(path)
    assert meta["has_noahmp_land"] is False
    assert meta["has_noahmp_rad"] is False
    assert meta["step_index"] == 3
    restored, _, _, step = read_restart(path)
    assert step == 3
    assert restored.state.active_field_names() == carry.state.active_field_names()
    for field in carry.state.active_field_names():
        assert _equal(getattr(carry.state, field), getattr(restored.state, field)), field


def test_schema_drift_fails_closed(tmp_path: Path) -> None:
    grid = GridSpec.canary_3km_template()
    carry = initial_operational_carry(_state(grid))
    path = tmp_path / "drift.wrfrst"
    write_restart(carry, _namelist(grid), grid, 1, path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["carry"]["scratch_field_order"] = ["BOGUS"] + list(payload["carry"]["scratch_field_order"])
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError):
        read_restart(path)


def test_bad_format_rejected(tmp_path: Path) -> None:
    path = tmp_path / "junk.wrfrst"
    with path.open("wb") as handle:
        pickle.dump({"format": "not-a-restart"}, handle)
    with pytest.raises(ValueError):
        read_restart(path)
