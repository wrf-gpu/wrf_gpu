"""Unit tests for M7 checkpoint writer/reader round-trips."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import (
    CONDITIONAL_STATE_LEAVES,
    SCALAR_BOUNDARY_OPTIONAL_LEAVES,
    State,
    Tendencies,
    _state_field_shapes,
)
from gpuwrf.runtime.checkpoint import read_checkpoint, read_checkpoint_with_runtime_state, write_checkpoint
from gpuwrf.runtime.operational_mode import OperationalNamelist


def _pattern(shape: tuple[int, ...], dtype: object, offset: int) -> jax.Array:
    values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    values = values + float(offset) / 1000.0
    return jnp.asarray(values, dtype=dtype)


def _synthetic_state(grid: GridSpec, *, mp_physics: int = 8) -> State:
    fields = {
        field: _pattern(shape, DEFAULT_DTYPES.dtype_for(field), index)
        for index, (field, shape) in enumerate(_state_field_shapes(grid, mp_physics=mp_physics).items(), start=1)
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
        # v0.20 S1: legacy p/ph/mu shapes dropped from _state_field_shapes; the
        # Tendencies pressure/geopotential/mass buffers share the total shapes.
        p=jnp.zeros(shapes["p_total"], dtype=DEFAULT_DTYPES.dtype_for("p")),
        ph=jnp.zeros(shapes["ph_total"], dtype=DEFAULT_DTYPES.dtype_for("ph")),
        mu=jnp.zeros(shapes["mu_total"], dtype=DEFAULT_DTYPES.dtype_for("mu")),
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
    # appended 4 MYNN SGS-cloud leaves (qsq, qc_bl, qi_bl, cldfra_bl), v0.17
    # ADR-032 appended the graupel/hail substrate (qh, Nh, qvolg, qvolh), v0.16
    # appended the aerosol-aware Thompson (mp=28) nwfa/nifa leaves, and the v0.17
    # hail microphysics appended the hail surface accumulator (hail_acc), and v0.21.1
    # appended optional wrfbdy scalar leaves. The guard tracks the
    # authoritative consolidated count (53 + 3 + 4 + 4 + 2 + 1 = 67), minus the 3
    # legacy p/ph/mu duplicate aliases removed in v0.20 S1 = 64, plus 7 optional
    # scalar boundary leaves = 71.
    assert len(State.__slots__) == 71
    assert State.__slots__[-21:-7] == (
        "Nc", "Nn", "rainc_acc", "qsq", "qc_bl", "qi_bl", "cldfra_bl",
        "qh", "Nh", "qvolg", "qvolh", "nwfa", "nifa", "hail_acc",
    )
    assert State.__slots__[-7:] == SCALAR_BOUNDARY_OPTIONAL_LEAVES
    assert restored_grid == grid
    assert restored_namelist.grid == restored_grid
    assert restored_namelist.dt_s == namelist.dt_s
    assert restored_namelist.acoustic_substeps == namelist.acoustic_substeps

    assert state.active_field_names() == tuple(name for name in State.__slots__ if name not in CONDITIONAL_STATE_LEAVES)
    assert restored_state.active_field_names() == state.active_field_names()
    assert len(state.active_field_names()) == 57  # v0.20 S1: -3 legacy p/ph/mu aliases
    for field in state.active_field_names():
        original = np.asarray(getattr(state, field))
        restored = np.asarray(getattr(restored_state, field))
        assert restored.dtype == original.dtype, field
        assert restored.shape == original.shape, field
        assert np.array_equal(restored, original), field
    for field in CONDITIONAL_STATE_LEAVES:
        assert getattr(restored_state, field) is None, field

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


def test_checkpoint_roundtrip_preserves_hail_conditional_state(tmp_path: Path) -> None:
    grid = GridSpec.canary_3km_template()
    state = _synthetic_state(grid, mp_physics=24)
    namelist = OperationalNamelist(
        grid=grid,
        tendencies=_synthetic_tendencies(grid),
        metrics=grid.metrics,
        dt_s=10.0,
        acoustic_substeps=10,
        mp_physics=24,
    )

    checkpoint_path = tmp_path / "restart_hail.pkl"
    write_checkpoint(state, namelist, grid, 23, checkpoint_path)
    restored_state, _, _, restored_step = read_checkpoint(checkpoint_path)

    assert restored_step == 23
    assert state.active_field_names() == restored_state.active_field_names()
    assert len(restored_state.active_field_names()) == 62  # v0.20 S1: 57 base + 5 hail
    for field in ("qh", "Nh", "qvolg", "qvolh", "hail_acc"):
        assert getattr(restored_state, field) is not None, field
    for field in ("nwfa", "nifa"):
        assert getattr(restored_state, field) is None, field
    for field in state.active_field_names():
        assert np.array_equal(np.asarray(getattr(restored_state, field)), np.asarray(getattr(state, field))), field
