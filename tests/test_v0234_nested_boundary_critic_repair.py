"""Independent CPU gates for the critic-required full-SINT/scalar-RK repair.

The numerical expectations come from the two NumPy source translations under
``scripts/``.  Those modules import no production code.  Every JAX command in
this suite is CPU-selectable and the suite contains no CUDA or lock operation.
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

import gpuwrf.nesting.boundary_construction as boundary_construction
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.coupling.boundary_apply import (
    BoundaryConfig,
    NESTED_BOUNDARY_SCALAR_SPECIES,
    nested_scalar_boundary_tendencies,
)
from gpuwrf.nesting.interp import (
    build_sint_weights,
    interp_sint_full,
)
from gpuwrf.nesting.moving import _SHIFTABLE_STATE_FIELDS
from gpuwrf.nesting.moving_driver import (
    _CARRY_SHIFT_FIELDS,
    MovingNestConfig,
    MovingNestDriver,
    PrescribedMove,
)
from gpuwrf.dynamics.advection import halo_spec
from gpuwrf.runtime.domain_tree import DomainEdge
from gpuwrf.runtime.operational_mode import (
    _MOISTURE_SPECIES,
    _apply_moisture_large_step,
    _initial_carry_for_run,
    _moisture_coupled_tendencies,
    _nested_number_scalar_coupled_tendencies,
    _nested_scalar_species_batch_width,
    _nested_scalar_stage_tendencies,
    _scalar_transport_coupled_tendencies,
    _stage_transport_velocities,
)
from gpuwrf.validation.moving_nest_testbed import (
    build_domain_namelist,
    build_flat_grid,
    build_nested_pair,
    build_neutral_state,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from v0234_wrf_scalar_boundary_oracle import (
    rk_scalar_sequence,
    scalar_boundary_tendency,
)
from v0234_wrf_sint_source_oracle import sint_full, wrf_sides


@pytest.mark.parametrize(
    ("shape", "dtype", "expected"),
    (
        ((44, 93, 195), jnp.float64, 1),  # release d02: 797,940 cells
        ((44, 93, 195), jnp.float32, 1),  # dtype-independent d02 guard
        ((44, 69, 102), jnp.float64, 2),  # release d03: 309,672 cells
        ((44, 69, 102), jnp.float32, 2),
        ((44, 87, 99), jnp.float64, 2),  # release d08: 378,972 cells
        ((1, 1, 524288), jnp.float64, 2),  # exact 2**19-cell boundary
        ((1, 1, 524288), jnp.float32, 2),
        ((1, 1, 524289), jnp.float64, 1),
        ((1, 1, 524289), jnp.float32, 1),
    ),
)
def test_nested_scalar_species_batch_width_is_static_and_bounded(
    shape: tuple[int, ...], dtype, expected: int
):
    field = jax.ShapeDtypeStruct(shape, dtype)
    assert _nested_scalar_species_batch_width(field) == expected


@pytest.mark.parametrize("shape", ((44, 69, 102), (44, 87, 99)))
def test_nested_plain_scalar_stage_forces_reference_width(shape: tuple[int, ...]):
    field = jax.ShapeDtypeStruct(shape, jnp.float64)
    assert _nested_scalar_species_batch_width(field, use_limiter=False) == 1


CPU_WRF_DIR = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf"
)


def _sint_weights(
    parent_shape: tuple[int, int],
    child_shape: tuple[int, int],
    *,
    ratio: int,
    start: tuple[int, int],
):
    return build_sint_weights(
        parent_grid_ratio=ratio,
        i_parent_start=start[0],
        j_parent_start=start[1],
        parent_ny=parent_shape[0],
        parent_nx=parent_shape[1],
        child_ny=child_shape[0],
        child_nx=child_shape[1],
    )


def _synthetic_parent(pattern: str, shape: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float64)
    if pattern == "constant":
        return np.full(shape, 7.25, dtype=np.float64)
    if pattern == "affine":
        return 3.0 + 2.0 * x - 1.5 * y
    if pattern == "impulse":
        value = np.zeros(shape, dtype=np.float64)
        value[shape[0] // 2, shape[1] // 2] = 1.0
        return value
    if pattern == "extrema":
        checker = ((x.astype(np.int64) + y.astype(np.int64)) % 2).astype(np.float64)
        return np.where(checker > 0, 9.0, -4.0) + 0.01 * x
    raise AssertionError(pattern)


@pytest.mark.parametrize("ratio,start", ((3, (6, 6)), (4, (6, 6))))
@pytest.mark.parametrize(
    "xstag,ystag",
    ((False, False), (True, False), (False, True)),
    ids=("mass", "u", "v"),
)
@pytest.mark.parametrize("pattern", ("constant", "affine", "impulse", "extrema"))
def test_full_sint_matches_source_literal_oracle_all_patterns_staggers_ratios(
    ratio, start, xstag, ystag, pattern
):
    parent = _synthetic_parent(pattern, (28, 30))
    child_shape = (13, 15)
    weights = _sint_weights(parent.shape, child_shape, ratio=ratio, start=start)
    actual = np.asarray(
        interp_sint_full(
            jnp.asarray(parent),
            weights,
            parent_grid_ratio=ratio,
            xstag=xstag,
            ystag=ystag,
        )
    )
    expected = sint_full(
        parent,
        ratio=ratio,
        i_parent_start=start[0],
        j_parent_start=start[1],
        child_ny=child_shape[0],
        child_nx=child_shape[1],
        xstag=xstag,
        ystag=ystag,
    )
    np.testing.assert_array_equal(actual, expected)
    if pattern == "constant":
        np.testing.assert_array_equal(actual, np.full(child_shape, 7.25))
    if pattern in ("impulse", "extrema"):
        # The donor/TR4 correction is flux-limited: it cannot manufacture a
        # global extremum outside the parent stencil range.
        assert float(actual.min()) >= float(parent.min())
        assert float(actual.max()) <= float(parent.max())


@pytest.mark.parametrize(
    "parent_domain,child_shape,start",
    (
        ("d01", (117, 267), (16, 16)),
        ("d02", (93, 111), (92, 36)),
    ),
)
@pytest.mark.parametrize(
    "variable,xstag,ystag",
    (("T", False, False), ("U", True, False), ("V", False, True)),
)
def test_full_sint_real_cpu_wrf_fields_both_operational_edges_all_staggers(
    parent_domain, child_shape, start, variable, xstag, ystag
):
    path = CPU_WRF_DIR / f"wrfout_{parent_domain}_2025-03-01_15:00:00"
    assert path.is_file(), f"missing authenticated CPU-WRF field: {path}"
    with Dataset(path) as dataset:
        parent = np.asarray(dataset.variables[variable][0, 0], dtype=np.float32)
    output_shape = (
        child_shape[0] + int(ystag),
        child_shape[1] + int(xstag),
    )
    weights = _sint_weights(parent.shape, output_shape, ratio=3, start=start)
    expected = sint_full(
        parent,
        ratio=3,
        i_parent_start=start[0],
        j_parent_start=start[1],
        child_ny=output_shape[0],
        child_nx=output_shape[1],
        xstag=xstag,
        ystag=ystag,
    )
    actual = np.asarray(
        interp_sint_full(
            jnp.asarray(parent),
            weights,
            parent_grid_ratio=3,
            xstag=xstag,
            ystag=ystag,
        )
    )
    np.testing.assert_array_equal(actual, expected)
    side_len = max(output_shape) + 1
    np.testing.assert_array_equal(
        wrf_sides(actual, width=5, side_len=side_len),
        wrf_sides(expected, width=5, side_len=side_len),
    )


def test_full_sint_lowering_has_no_observer_callback_or_custom_call():
    parent = jnp.arange(2 * 28 * 30, dtype=jnp.float64).reshape(2, 28, 30)
    weights = _sint_weights((28, 30), (13, 15), ratio=4, start=(6, 6))

    def run(value):
        return interp_sint_full(
            value,
            weights,
            parent_grid_ratio=4,
            xstag=True,
        )

    lowered = jax.jit(run).lower(parent)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo")).lower()
    for forbidden in (
        "host_callback",
        "io_callback",
        "pure_callback",
        "outside_compilation",
        "xla_python_cpu_callback",
        "custom_call",
    ):
        assert forbidden not in stablehlo


def _moving_pattern(shape: tuple[int, ...], seed: int, dtype: np.dtype) -> np.ndarray:
    """Nonlinear deterministic field that activates SINT's residual limiter."""

    index = np.indices(shape, dtype=np.float64)
    value = np.full(shape, 0.125 * (seed + 1), dtype=np.float64)
    for axis, coord in enumerate(index):
        value += (axis + 1.0) * (0.17 + 0.003 * seed) * coord
    checker = np.mod(np.sum(index.astype(np.int64), axis=0) + seed, 3) == 0
    value += np.where(checker, 2.0 + 0.01 * seed, -0.35)
    return value.astype(dtype)


def _nearest_source_oracle(
    parent: np.ndarray,
    *,
    ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    child_shape: tuple[int, int],
) -> np.ndarray:
    """Independent categorical registration used by WRF moving initialization."""

    def axis(start, length, parent_length):
        destination = np.arange(length, dtype=np.float64)
        coordinate = (
            float(start - 1)
            + (destination - float(ratio // 2)) / float(ratio)
        )
        return np.clip(np.rint(coordinate), 0, parent_length - 1).astype(np.int64)

    yi = axis(j_parent_start, child_shape[0], parent.shape[-2])
    xi = axis(i_parent_start, child_shape[1], parent.shape[-1])
    return np.take(np.take(parent, yi, axis=-2), xi, axis=-1)


def test_candidate_moving_fill_uses_full_sint_for_state_and_persistent_scratch(
    monkeypatch,
):
    """The real move callback never falls back to SINT-linear when candidate-on."""

    tree = build_nested_pair()
    carries = {
        name: _initial_carry_for_run(bundle.state, bundle.namelist)
        for name, bundle in tree.domains.items()
    }
    parent = carries["d01"]
    child = carries["d02"]

    # Populate every represented shiftable parent state leaf and every
    # persistent carry scratch with nonlinear data.  Categorical state remains
    # discrete so its independently checked nearest-neighbour path is visible.
    state_updates: dict[str, jax.Array] = {}
    seed = 0
    for name in _SHIFTABLE_STATE_FIELDS:
        parent_value = getattr(parent.state, name, None)
        child_value = getattr(child.state, name, None)
        if (
            parent_value is None
            or child_value is None
            or getattr(parent_value, "ndim", 0) < 2
            or getattr(child_value, "ndim", 0) < 2
        ):
            continue
        if name in ("xland", "lakemask"):
            y, x = np.indices(parent_value.shape[-2:], dtype=np.int64)
            categorical = np.where((x + 2 * y) % 4 < 2, 1.0, 2.0)
            state_updates[name] = jnp.asarray(categorical, dtype=parent_value.dtype)
        else:
            state_updates[name] = jnp.asarray(
                _moving_pattern(tuple(parent_value.shape), seed, np.dtype(parent_value.dtype))
            )
        seed += 1
    parent = parent.replace(state=parent.state.replace(_cast=False, **state_updates))

    scratch_updates: dict[str, jax.Array] = {}
    for name in _CARRY_SHIFT_FIELDS:
        parent_value = getattr(parent, name, None)
        child_value = getattr(child, name, None)
        if (
            parent_value is None
            or child_value is None
            or getattr(parent_value, "ndim", 0) < 2
            or getattr(child_value, "ndim", 0) < 2
        ):
            continue
        scratch_updates[name] = jnp.asarray(
            _moving_pattern(tuple(parent_value.shape), seed, np.dtype(parent_value.dtype))
        )
        seed += 1
    parent = parent.replace(**scratch_updates)

    driver = MovingNestDriver.for_tree(
        tree,
        [
            MovingNestConfig(
                child="d02",
                mode="prescribed",
                prescribed_moves=(PrescribedMove(1, 1, 1),),
                corral_dist=1,
            )
        ],
    )
    released_edge = tree.edges["d01"][0]
    candidate_edge = dataclasses.replace(released_edge, coupled_forcedown=True)

    def forbidden_linear(*_args, **_kwargs):
        raise AssertionError("candidate-on moving fill reached interp_sint_linear")

    monkeypatch.setattr(boundary_construction, "interp_sint_linear", forbidden_linear)
    result = driver.make_move_fn()(candidate_edge, parent, child, 1)
    assert isinstance(result, tuple)
    new_edge, moved = result
    assert isinstance(new_edge, DomainEdge)
    assert new_edge.coupled_forcedown is True
    assert (new_edge.spec.i_parent_start, new_edge.spec.j_parent_start) == (6, 6)

    ratio = int(new_edge.spec.parent_grid_ratio)
    child_ny = int(tree.domains["d02"].grid.ny)
    child_nx = int(tree.domains["d02"].grid.nx)

    def assert_exposed(actual, expected, name):
        np.testing.assert_array_equal(
            np.asarray(actual)[..., -ratio:],
            np.asarray(expected)[..., -ratio:],
            err_msg=f"{name} east exposed band",
        )
        np.testing.assert_array_equal(
            np.asarray(actual)[..., -ratio:, :],
            np.asarray(expected)[..., -ratio:, :],
            err_msg=f"{name} north exposed band",
        )

    checked_state: set[str] = set()
    checked_state_names: set[str] = set()
    for name in _SHIFTABLE_STATE_FIELDS:
        parent_value = getattr(parent.state, name, None)
        child_value = getattr(child.state, name, None)
        moved_value = getattr(moved.state, name, None)
        if (
            parent_value is None
            or child_value is None
            or moved_value is None
            or getattr(child_value, "ndim", 0) < 2
        ):
            continue
        shape = tuple(child_value.shape[-2:])
        if shape == (child_ny, child_nx):
            stagger = "mass"
        elif shape == (child_ny, child_nx + 1):
            stagger = "u"
        elif shape == (child_ny + 1, child_nx):
            stagger = "v"
        else:
            continue
        if name in ("xland", "lakemask"):
            expected = _nearest_source_oracle(
                np.asarray(parent_value),
                ratio=ratio,
                i_parent_start=new_edge.spec.i_parent_start,
                j_parent_start=new_edge.spec.j_parent_start,
                child_shape=shape,
            )
        else:
            expected = sint_full(
                np.asarray(parent_value),
                ratio=ratio,
                i_parent_start=new_edge.spec.i_parent_start,
                j_parent_start=new_edge.spec.j_parent_start,
                child_ny=shape[0],
                child_nx=shape[1],
                xstag=stagger == "u",
                ystag=stagger == "v",
            )
        assert_exposed(moved_value, expected, f"state.{name}")
        checked_state.add(stagger if name not in ("xland", "lakemask") else name)
        checked_state_names.add(name)

    checked_scratch: set[str] = set()
    checked_scratch_names: set[str] = set()
    for name in _CARRY_SHIFT_FIELDS:
        parent_value = getattr(parent, name, None)
        child_value = getattr(child, name, None)
        moved_value = getattr(moved, name, None)
        if parent_value is None or child_value is None or moved_value is None:
            continue
        shape = tuple(child_value.shape[-2:])
        if shape == (child_ny, child_nx):
            stagger = "mass"
        elif shape == (child_ny, child_nx + 1):
            stagger = "u"
        elif shape == (child_ny + 1, child_nx):
            stagger = "v"
        else:
            continue
        expected = sint_full(
            np.asarray(parent_value),
            ratio=ratio,
            i_parent_start=new_edge.spec.i_parent_start,
            j_parent_start=new_edge.spec.j_parent_start,
            child_ny=shape[0],
            child_nx=shape[1],
            xstag=stagger == "u",
            ystag=stagger == "v",
        )
        assert_exposed(moved_value, expected, f"carry.{name}")
        checked_scratch.add(stagger)
        checked_scratch_names.add(name)

    assert {"mass", "u", "v", "xland", "lakemask"} <= checked_state
    assert {"mass", "u", "v"} <= checked_scratch
    expected_state_names = {
        name
        for name in _SHIFTABLE_STATE_FIELDS
        if getattr(parent.state, name, None) is not None
        and getattr(child.state, name, None) is not None
        and getattr(getattr(child.state, name), "ndim", 0) >= 2
        and tuple(getattr(child.state, name).shape[-2:])
        in ((child_ny, child_nx), (child_ny, child_nx + 1), (child_ny + 1, child_nx))
    }
    expected_scratch_names = {
        name
        for name in _CARRY_SHIFT_FIELDS
        if getattr(parent, name, None) is not None
        and getattr(child, name, None) is not None
        and tuple(getattr(child, name).shape[-2:])
        in ((child_ny, child_nx), (child_ny, child_nx + 1), (child_ny + 1, child_nx))
    }
    assert checked_state_names == expected_state_names
    assert checked_scratch_names == expected_scratch_names


def _scalar_fixture(*, cadence_s: float):
    grid0 = build_flat_grid(nx=14, ny=13, nz=3, dx_m=1000.0)
    y, x = np.indices((grid0.ny, grid0.nx), dtype=np.float64)
    metrics = dataclasses.replace(
        grid0.metrics,
        c1h=jnp.asarray((0.73, 0.82, 0.91), dtype=jnp.float64),
        c2h=jnp.asarray((31.0, 47.0, 66.0), dtype=jnp.float64),
        msfty=jnp.asarray(
            1.04 + 0.11 * np.sin(0.21 * x) + 0.07 * np.cos(0.17 * y),
            dtype=jnp.float64,
        ),
        provenance="v0234-independent-scalar-rk",
    )
    grid = dataclasses.replace(grid0, metrics=metrics)
    state = build_neutral_state(grid)
    mu = np.asarray(state.mu_total) + 8.0 * np.sin(0.3 * x) + 5.0 * np.cos(0.2 * y)
    side_len = max(grid.nx + 1, grid.ny + 1)
    updates = {"mu_total": jnp.asarray(mu), "mu_perturbation": jnp.asarray(mu - np.asarray(state.mu_total))}
    records: dict[str, np.ndarray] = {}
    for index, name in enumerate(NESTED_BOUNDARY_SCALAR_SPECIES):
        scale = 1.0e-4 * (index + 1) if index < 6 else 2.0e3 * (index - 5)
        field = scale * (
            1.0
            + 0.02 * np.arange(grid.nz)[:, None, None]
            + 0.001 * y[None]
            + 0.0007 * x[None]
        )
        target0 = (scale * 1.2 + 0.03 * field) * (
            35.0 + 0.74 * mu[None]
        )
        target1 = target0 + scale * (
            0.3
            + 0.01 * np.arange(grid.nz)[:, None, None]
            + 0.002 * y[None]
            - 0.001 * x[None]
        )
        record = np.stack(
            (
                wrf_sides(target0, width=5, side_len=side_len),
                wrf_sides(target1, width=5, side_len=side_len),
            ),
            axis=0,
        ).astype(np.float32)
        updates[name] = jnp.asarray(field)
        updates[f"{name}_bdy"] = jnp.asarray(record)
        records[name] = record
    state = state.replace(**updates)
    config = BoundaryConfig(
        spec_bdy_width=5,
        spec_zone=1,
        relax_zone=4,
        update_cadence_s=float(cadence_s),
        spec_exp=0.0,
        force_geopotential=False,
        nested_frozen_wrf_boundary_bundle=True,
    )
    return grid, state, records, config


@pytest.mark.parametrize("cadence_s", (6.0, 10.0), ids=("ratio3", "ratio5"))
def test_scalar_rk1_frozen_tendencies_match_source_oracle_all_families_corners(
    cadence_s
):
    grid, state, records, config = _scalar_fixture(cadence_s=cadence_s)
    dt_full = 2.0
    lead = cadence_s
    actual = nested_scalar_boundary_tendencies(
        state,
        lead,
        grid.metrics,
        dt_full,
        config,
    )
    for name, tendency in zip(NESTED_BOUNDARY_SCALAR_SPECIES, actual, strict=True):
        expected = scalar_boundary_tendency(
            np.asarray(getattr(state, name)),
            np.asarray(state.mu_total),
            np.asarray(grid.metrics.c1h),
            np.asarray(grid.metrics.c2h),
            records[name],
            lead_seconds=lead,
            cadence_s=cadence_s,
            dt_full=dt_full,
            spec_zone=1,
            relax_zone=4,
        )
        np.testing.assert_allclose(np.asarray(tendency), expected, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("with_advection", (False, True), ids=("zero_adv", "nonzero_adv"))
def test_three_stage_scalar_update_matches_source_oracle_with_changing_mass(
    with_advection
):
    grid, origin, records, config = _scalar_fixture(cadence_s=6.0)
    frozen = nested_scalar_boundary_tendencies(origin, 6.0, grid.metrics, 2.0, config)
    frozen_oracle = tuple(
        scalar_boundary_tendency(
            np.asarray(getattr(origin, name)),
            np.asarray(origin.mu_total),
            np.asarray(grid.metrics.c1h),
            np.asarray(grid.metrics.c2h),
            records[name],
            lead_seconds=6.0,
            cadence_s=6.0,
            dt_full=2.0,
            spec_zone=1,
            relax_zone=4,
        )
        for name in NESTED_BOUNDARY_SCALAR_SPECIES
    )
    for actual, expected in zip(frozen, frozen_oracle, strict=True):
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-7, atol=2e-7)

    y, x = np.indices(origin.mu_total.shape, dtype=np.float64)
    mu_new = tuple(
        np.asarray(origin.mu_total) + delta * (1.0 + 0.01 * y - 0.02 * x)
        for delta in (0.4, -0.7, 1.1)
    )
    dt_rk = (2.0 / 3.0, 1.0, 2.0)
    adv_by_species: dict[str, tuple[np.ndarray, ...]] = {}
    for index, name in enumerate(NESTED_BOUNDARY_SCALAR_SPECIES):
        template = np.asarray(getattr(origin, name), dtype=np.float64)
        if with_advection:
            adv_by_species[name] = tuple(
                np.full_like(template, (index + 1) * stage * 1.0e-3)
                for stage in (1.0, -0.5, 0.25)
            )
        else:
            adv_by_species[name] = tuple(np.zeros_like(template) for _ in range(3))

    expected_by_species = {
        name: rk_scalar_sequence(
            np.asarray(getattr(origin, name)),
            np.asarray(origin.mu_total),
            mu_new,
            np.asarray(grid.metrics.c1h),
            np.asarray(grid.metrics.c2h),
            frozen_oracle[index],
            adv_by_species[name],
            dt_rk,
            spec_zone=1,
            msfty=np.asarray(grid.metrics.msfty),
        )
        for index, name in enumerate(NESTED_BOUNDARY_SCALAR_SPECIES)
    }

    for stage_index in range(3):
        advected = tuple(
            jnp.asarray(adv_by_species[name][stage_index])
            for name in NESTED_BOUNDARY_SCALAR_SPECIES
        )
        species, merged = _nested_scalar_stage_tendencies(
            advected,
            NESTED_BOUNDARY_SCALAR_SPECIES,
            frozen,
            config,
            grid.metrics.msfty,
        )
        stage_state = origin.replace(mu_total=jnp.asarray(mu_new[stage_index]))
        actual = _apply_moisture_large_step(
            stage_state,
            origin,
            q_tendencies=merged,
            dt_rk=dt_rk[stage_index],
            metrics=grid.metrics,
            species=species,
        )
        for name in NESTED_BOUNDARY_SCALAR_SPECIES:
            np.testing.assert_allclose(
                np.asarray(getattr(actual, name)),
                expected_by_species[name][stage_index],
                rtol=3e-6,
                atol=3e-6,
            )


@pytest.mark.parametrize("advection_opt", (1, 2))
def test_live_moist_and_number_helpers_apply_nonunit_map_at_rk1_rk2_rk3(
    advection_opt: int,
):
    """Exercise the production moist/QNI/QNR transport and WRF RK merge."""

    grid, origin0, records, config = _scalar_fixture(cadence_s=6.0)
    nz, ny, nx = origin0.qv.shape
    z, y, x = np.indices((nz, ny, nx), dtype=np.float64)
    yu, xu = np.indices((ny, nx + 1), dtype=np.float64)
    yv, xv = np.indices((ny + 1, nx), dtype=np.float64)
    scalar_updates: dict[str, jax.Array] = {
        "u": jnp.asarray(
            9.0 + 0.21 * xu - 0.13 * yu + 0.8 * np.sin(0.31 * xu),
            dtype=jnp.float64,
        )[None].repeat(nz, axis=0),
        "v": jnp.asarray(
            -3.0 + 0.17 * yv + 0.09 * xv + 0.6 * np.cos(0.27 * yv),
            dtype=jnp.float64,
        )[None].repeat(nz, axis=0),
    }
    for index, name in enumerate(NESTED_BOUNDARY_SCALAR_SPECIES):
        base = np.asarray(getattr(origin0, name), dtype=np.float64)
        positive_scale = 1.0e-5 * (index + 1) if index < 6 else 100.0 * (index - 5)
        blob = np.exp(
            -(
                ((x - 0.57 * nx) / 3.1) ** 2
                + ((y - 0.46 * ny) / 2.7) ** 2
                + ((z - 0.43 * nz) / 1.4) ** 2
            )
        )
        scalar_updates[name] = jnp.asarray(
            np.maximum(base + positive_scale * (0.3 + blob), positive_scale * 0.05)
        )
    origin0 = origin0.replace(**scalar_updates)
    namelist = build_domain_namelist(
        grid,
        dt_s=2.0,
        is_child=True,
        parent_dt_s=6.0,
    )
    namelist = dataclasses.replace(
        namelist,
        grid=grid,
        metrics=grid.metrics,
        boundary_config=config,
        run_physics=False,
        run_boundary=True,
        use_flux_advection=True,
        moist_adv_opt=advection_opt,
        scalar_adv_opt=advection_opt,
    )
    origin = apply_halo(origin0, halo_spec(grid))
    frozen = nested_scalar_boundary_tendencies(
        origin,
        6.0,
        grid.metrics,
        2.0,
        config,
    )
    frozen_oracle = tuple(
        scalar_boundary_tendency(
            np.asarray(getattr(origin, name)),
            np.asarray(origin.mu_total),
            np.asarray(grid.metrics.c1h),
            np.asarray(grid.metrics.c2h),
            records[name],
            lead_seconds=6.0,
            cadence_s=6.0,
            dt_full=2.0,
            spec_zone=1,
            relax_zone=4,
        )
        for name in NESTED_BOUNDARY_SCALAR_SPECIES
    )
    for actual, expected in zip(frozen, frozen_oracle, strict=True):
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-7, atol=2e-7)

    dt_rk = (2.0 / 3.0, 1.0, 2.0)
    mass_delta = (0.45, -0.65, 1.05)
    raw_by_name: dict[str, list[np.ndarray]] = {
        name: [] for name in NESTED_BOUNDARY_SCALAR_SPECIES
    }
    actual_by_name: dict[str, list[np.ndarray]] = {
        name: [] for name in NESTED_BOUNDARY_SCALAR_SPECIES
    }
    mu_by_stage: list[np.ndarray] = []
    y2, x2 = np.indices((ny, nx), dtype=np.float64)
    map_factor = np.asarray(grid.metrics.msfty)

    for stage_index, rk_step in enumerate((1, 2, 3)):
        mu_new = np.asarray(origin.mu_total) + mass_delta[stage_index] * (
            1.0 + 0.013 * y2 - 0.009 * x2
        )
        stage_updates: dict[str, jax.Array] = {
            "mu_total": jnp.asarray(mu_new),
        }
        for species_index, name in enumerate(NESTED_BOUNDARY_SCALAR_SPECIES):
            stage_updates[name] = getattr(origin, name) * (
                1.0
                + (stage_index + 1) * (species_index + 1) * 2.0e-4
                + jnp.asarray(3.0e-4 * np.sin(0.19 * x + 0.23 * y))
            )
        haloed = apply_halo(origin.replace(**stage_updates), halo_spec(grid))
        mu_by_stage.append(np.asarray(haloed.mu_total))
        velocities = _stage_transport_velocities(haloed, namelist)
        moist = _moisture_coupled_tendencies(
            haloed,
            namelist,
            rk_step=rk_step,
            step_origin=origin,
            transport_velocities=velocities,
        )
        numbers = _nested_number_scalar_coupled_tendencies(
            haloed,
            namelist,
            rk_step=rk_step,
            step_origin=origin,
            transport_velocities=velocities,
        )
        raw = moist + numbers
        assert len(raw) == 8
        combined = _scalar_transport_coupled_tendencies(
            haloed,
            namelist,
            rk_step=rk_step,
            step_origin=origin,
            species=NESTED_BOUNDARY_SCALAR_SPECIES,
            advection_opt=advection_opt,
            transport_velocities=velocities,
            species_batch_width=2,
        )
        for separate, batched in zip(raw, combined, strict=True):
            np.testing.assert_array_equal(np.asarray(batched), np.asarray(separate))
        for name, tendency in zip(NESTED_BOUNDARY_SCALAR_SPECIES, raw, strict=True):
            value = np.asarray(tendency)
            assert np.isfinite(value).all(), (rk_step, name)
            assert float(np.max(np.abs(value))) > 0.0, (rk_step, name)
            raw_by_name[name].append(value)

        species, merged = _nested_scalar_stage_tendencies(
            raw,
            _MOISTURE_SPECIES + ("Ni", "Nr"),
            frozen,
            config,
            grid.metrics.msfty,
        )
        assert species == NESTED_BOUNDARY_SCALAR_SPECIES
        for name, raw_tendency, boundary_tendency, combined in zip(
            species, raw, frozen, merged, strict=True
        ):
            expected_adv = np.zeros_like(np.asarray(raw_tendency))
            expected_adv[:, 1:-1, 1:-1] = (
                np.asarray(raw_tendency)[:, 1:-1, 1:-1]
                * map_factor[None, 1:-1, 1:-1]
            )
            expected = expected_adv + np.asarray(boundary_tendency)
            np.testing.assert_array_equal(np.asarray(combined), expected)
            # All four specified edges are exactly the unscaled frozen sc_tend.
            np.testing.assert_array_equal(
                np.asarray(combined)[:, 0, :], np.asarray(boundary_tendency)[:, 0, :]
            )
            np.testing.assert_array_equal(
                np.asarray(combined)[:, -1, :], np.asarray(boundary_tendency)[:, -1, :]
            )
            np.testing.assert_array_equal(
                np.asarray(combined)[:, :, 0], np.asarray(boundary_tendency)[:, :, 0]
            )
            np.testing.assert_array_equal(
                np.asarray(combined)[:, :, -1], np.asarray(boundary_tendency)[:, :, -1]
            )

        updated = _apply_moisture_large_step(
            haloed,
            origin,
            q_tendencies=merged,
            dt_rk=dt_rk[stage_index],
            metrics=grid.metrics,
            species=species,
        )
        for name in species:
            actual_by_name[name].append(np.asarray(getattr(updated, name)))

    for index, name in enumerate(NESTED_BOUNDARY_SCALAR_SPECIES):
        expected_stages = rk_scalar_sequence(
            np.asarray(getattr(origin, name)),
            np.asarray(origin.mu_total),
            tuple(mu_by_stage),
            np.asarray(grid.metrics.c1h),
            np.asarray(grid.metrics.c2h),
            frozen_oracle[index],
            tuple(raw_by_name[name]),
            dt_rk,
            spec_zone=1,
            msfty=map_factor,
        )
        for rk_step, (actual, expected) in enumerate(
            zip(actual_by_name[name], expected_stages, strict=True), start=1
        ):
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=3e-6,
                atol=3e-6,
                err_msg=f"{name} RK{rk_step}",
            )


def test_scalar_frozen_bundle_is_restart_deterministic_and_aot_observer_free():
    grid, state, _records, config = _scalar_fixture(cadence_s=6.0)

    def build(reference):
        return nested_scalar_boundary_tendencies(
            reference,
            jnp.asarray(6.0, dtype=jnp.float64),
            grid.metrics,
            2.0,
            config,
        )

    first = build(state)
    restarted = pickle.loads(pickle.dumps(first, protocol=5))
    second = build(pickle.loads(pickle.dumps(state, protocol=5)))
    for lhs, mid, rhs in zip(first, restarted, second, strict=True):
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(mid))
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))

    lowered = jax.jit(build).lower(state)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo")).lower()
    for forbidden in (
        "host_callback",
        "io_callback",
        "pure_callback",
        "outside_compilation",
        "xla_python_cpu_callback",
        "custom_call",
    ):
        assert forbidden not in stablehlo
    assert len(jax.tree_util.tree_leaves(lowered.out_info)) == 8
