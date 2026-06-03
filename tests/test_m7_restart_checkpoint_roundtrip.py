"""Unit tests for M7 checkpoint writer/reader round-trips."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.checkpoint import read_checkpoint, read_checkpoint_with_runtime_state, write_checkpoint
from gpuwrf.runtime.operational_mode import OperationalNamelist


def _pattern(shape: tuple[int, ...], dtype: object, offset: int) -> jax.Array:
    values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    values = values + float(offset) / 1000.0
    return jnp.asarray(values, dtype=dtype)


def _synthetic_state(grid: GridSpec) -> State:
    fields = {
        field: _pattern(shape, DEFAULT_DTYPES.dtype_for(field), index)
        for index, (field, shape) in enumerate(_state_field_shapes(grid).items(), start=1)
    }
    return State(**fields)


def _synthetic_tendencies(grid: GridSpec) -> Tendencies:
    shapes = _state_field_shapes(grid)
    return Tendencies(
        u=jnp.zeros(shapes["u"], dtype=DEFAULT_DTYPES.dtype_for("u")),
        v=jnp.zeros(shapes["v"], dtype=DEFAULT_DTYPES.dtype_for("v")),
        w=jnp.zeros(shapes["w"], dtype=DEFAULT_DTYPES.dtype_for("w")),
        theta=jnp.zeros(shapes["theta"], dtype=DEFAULT_DTYPES.dtype_for("theta")),
        qv=jnp.zeros(shapes["qv"], dtype=DEFAULT_DTYPES.dtype_for("qv")),
        p=jnp.zeros(shapes["p"], dtype=DEFAULT_DTYPES.dtype_for("p")),
        ph=jnp.zeros(shapes["ph"], dtype=DEFAULT_DTYPES.dtype_for("ph")),
        mu=jnp.zeros(shapes["mu"], dtype=DEFAULT_DTYPES.dtype_for("mu")),
    )


def test_checkpoint_roundtrip_preserves_all_state_fields_bitwise(tmp_path: Path) -> None:
    grid = GridSpec.canary_3km_template()
    state = _synthetic_state(grid)
    namelist = OperationalNamelist(
        grid=grid,
        tendencies=_synthetic_tendencies(grid),
        metrics=grid.metrics,
        dt_s=10.0,
        acoustic_substeps=10,
    )

    checkpoint_path = tmp_path / "restart.pkl"
    write_checkpoint(state, namelist, grid, 17, checkpoint_path)
    restored_state, restored_namelist, restored_grid, restored_step = read_checkpoint(checkpoint_path)

    assert restored_step == 17
    # v0.6.0 S0 appended 3 additive physics leaves (Nc, Nn, rainc_acc) to the
    # original 53-leaf schema. The guard tracks the authoritative count.
    assert len(State.__slots__) == 56
    assert State.__slots__[-3:] == ("Nc", "Nn", "rainc_acc")
    assert restored_grid == grid
    assert restored_namelist.grid == restored_grid
    assert restored_namelist.dt_s == namelist.dt_s
    assert restored_namelist.acoustic_substeps == namelist.acoustic_substeps

    for field in State.__slots__:
        original = np.asarray(getattr(state, field))
        restored = np.asarray(getattr(restored_state, field))
        assert restored.dtype == original.dtype, field
        assert restored.shape == original.shape, field
        assert np.array_equal(restored, original), field

    leaves = jax.tree_util.tree_leaves(restored_state)
    assert leaves
    assert all(hasattr(leaf, "devices") for leaf in leaves)


def test_checkpoint_can_preserve_optional_runtime_state(tmp_path: Path) -> None:
    grid = GridSpec.canary_3km_template()
    state = _synthetic_state(grid)
    namelist = OperationalNamelist(
        grid=grid,
        tendencies=_synthetic_tendencies(grid),
        metrics=grid.metrics,
        dt_s=10.0,
        acoustic_substeps=10,
    )
    runtime_state = {"ww": jnp.ones_like(state.w), "step_tag": jnp.asarray(17, dtype=jnp.int32)}

    checkpoint_path = tmp_path / "restart_with_runtime.pkl"
    write_checkpoint(state, namelist, grid, 17, checkpoint_path, runtime_state=runtime_state)
    _, _, _, _, restored_runtime = read_checkpoint_with_runtime_state(checkpoint_path)

    assert restored_runtime is not None
    assert np.array_equal(np.asarray(restored_runtime["ww"]), np.asarray(runtime_state["ww"]))
    assert int(np.asarray(restored_runtime["step_tag"])) == 17
