from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.halo import HaloSpec, apply_halo
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.dynamics.explicit_diffusion import sixth_order_diffusion_tendency
from gpuwrf.dynamics.flux_advection import flux5_face_periodic
from gpuwrf.dynamics.sharded_horizontal import (
    sharded_flux5_face_periodic_x,
    sharded_sixth_order_diffusion_tendency,
    sharded_x_face_pressure_dpn,
    sharded_x_staggered_divergence,
)
from gpuwrf.runtime.sharding import ShardingConfig, partition_state_x, x_partition_bounds


pytestmark = pytest.mark.skipif(len(jax.local_devices()) < 4, reason="requires fake or real 4-device mesh")


def _test_grid(*, nx: int = 32, ny: int = 4, nz: int = 6) -> GridSpec:
    projection = Projection("lambert", 28.3, -15.6, 3000.0, 3000.0, int(nx), int(ny))
    terrain = TerrainProvenance(
        source_path="synthetic-dgx-operator-test",
        sha256="synthetic-dgx-operator-test",
        shape=(projection.ny, projection.nx),
        units="m",
        projection_transform="native-lambert",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta_levels = jnp.linspace(1.0, 0.0, int(nz) + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", int(nz), 5000.0, eta_levels)
    bc = BCMetadata(
        source="ideal",
        fields=("u", "v", "theta", "qv", "p"),
        update_cadence_h=6,
        interpolation="linear",
        restart_compatible=True,
    )
    terrain_height = jnp.zeros(terrain.shape, dtype=jnp.float64)
    return GridSpec(projection, terrain, vertical, bc, eta_levels, terrain_height)


def _deterministic_state(grid: GridSpec) -> State:
    fields = {}
    for index, (name, shape) in enumerate(_state_field_shapes(grid).items()):
        dtype = DEFAULT_DTYPES.dtype_for(name)
        size = 1
        for dim in shape:
            size *= int(dim)
        values = jnp.sin(jnp.arange(size, dtype=jnp.float64).reshape(shape) * 0.017 + index)
        if dtype == jnp.int32:
            fields[name] = jnp.mod(jnp.arange(size, dtype=jnp.int32).reshape(shape), 7)
        else:
            fields[name] = values.astype(dtype)
    return State(**fields)


def _stack_mass_slices(field: jax.Array, bounds: tuple[tuple[int, int], ...]) -> jax.Array:
    return jnp.stack([field[..., start:end] for start, end in bounds], axis=0)


def _stack_face_slices(field: jax.Array, bounds: tuple[tuple[int, int], ...]) -> jax.Array:
    return jnp.stack([field[..., start : end + 1] for start, end in bounds], axis=0)


def _assert_close(got: jax.Array, want: jax.Array, *, rtol: float, atol: float) -> None:
    max_abs = float(jnp.max(jnp.abs(got.astype(jnp.float64) - want.astype(jnp.float64))))
    assert jnp.allclose(got, want, rtol=rtol, atol=atol), max_abs


def _global_x_face_pressure_dpn(
    p: jax.Array,
    *,
    fnm: jax.Array,
    fnp: jax.Array,
    cf1: jax.Array,
    cf2: jax.Array,
    cf3: jax.Array,
) -> jax.Array:
    left = jnp.concatenate([p[:, :, :1], p], axis=2)
    right = jnp.concatenate([p, p[:, :, -1:]], axis=2)
    pair_sum = left + right
    _, ny, nx_face = pair_sum.shape
    bottom = 0.5 * (cf1 * pair_sum[0] + cf2 * pair_sum[1] + cf3 * pair_sum[2])
    interior = 0.5 * (
        fnm[1:, None, None] * pair_sum[1:, :, :]
        + fnp[1:, None, None] * pair_sum[:-1, :, :]
    )
    top = jnp.zeros((ny, nx_face), dtype=p.dtype)
    return jnp.concatenate([bottom[None, :, :], interior, top[None, :, :]], axis=0)


def test_sharded_flux5_x_matches_global_owned_columns():
    grid = _test_grid()
    state = _deterministic_state(grid)
    bounds = x_partition_bounds(grid.nx, 4)
    sharded = partition_state_x(state, grid, num_partitions=4, halo_width=3, fill_halos=True)
    global_flux = flux5_face_periodic(state.theta, state.qv + 0.25, axis=2)

    got = jax.pmap(
        lambda field, vel: sharded_flux5_face_periodic_x(field, vel + 0.25, halo_width=3),
        axis_name="shard",
    )(sharded.theta, sharded.qv)

    _assert_close(got, _stack_mass_slices(global_flux, bounds), rtol=2.0e-6, atol=2.0e-7)


def test_sharded_sixth_order_diffusion_matches_global_owned_columns():
    grid = _test_grid()
    state = _deterministic_state(grid)
    bounds = x_partition_bounds(grid.nx, 4)
    sharded = partition_state_x(state, grid, num_partitions=4, halo_width=3, fill_halos=True)
    kwargs = {"dt": 10.0, "diff_6th_factor": 0.12, "horizontal_only": True, "monotonic": True}
    global_tendency = sixth_order_diffusion_tendency(state.theta, **kwargs)

    got = jax.pmap(
        lambda field: sharded_sixth_order_diffusion_tendency(field, halo_width=3, **kwargs),
        axis_name="shard",
    )(sharded.theta)

    _assert_close(got, _stack_mass_slices(global_tendency, bounds), rtol=2.0e-6, atol=2.0e-9)


def test_sharded_acoustic_x_staggered_divergence_matches_global_owned_columns():
    grid = _test_grid()
    state = _deterministic_state(grid)
    bounds = x_partition_bounds(grid.nx, 4)
    sharded = partition_state_x(state, grid, num_partitions=4, halo_width=1, fill_halos=True)
    global_div = 0.1 * (state.u[:, :, 1 : grid.nx + 1] - state.u[:, :, : grid.nx])

    got = jax.pmap(
        lambda u: sharded_x_staggered_divergence(u, rdx=0.1, halo_width=1),
        axis_name="shard",
    )(sharded.u)

    _assert_close(got, _stack_mass_slices(global_div, bounds), rtol=0.0, atol=0.0)


def test_sharded_acoustic_x_face_pressure_dpn_matches_global_owned_faces():
    grid = _test_grid()
    state = _deterministic_state(grid)
    bounds = x_partition_bounds(grid.nx, 4)
    sharded = partition_state_x(state, grid, num_partitions=4, halo_width=1, fill_halos=True)
    fnm = jnp.linspace(0.2, 0.8, grid.nz, dtype=state.p.dtype)
    fnp = 1.0 - fnm
    cf1 = jnp.asarray(0.55, dtype=state.p.dtype)
    cf2 = jnp.asarray(0.30, dtype=state.p.dtype)
    cf3 = jnp.asarray(0.15, dtype=state.p.dtype)
    global_dpn = _global_x_face_pressure_dpn(state.p, fnm=fnm, fnp=fnp, cf1=cf1, cf2=cf2, cf3=cf3)
    owned_width = grid.nx // 4

    def local_dpn(p):
        rank = jax.lax.axis_index("shard")
        return sharded_x_face_pressure_dpn(
            p,
            fnm=fnm,
            fnp=fnp,
            cf1=cf1,
            cf2=cf2,
            cf3=cf3,
            halo_width=1,
            global_start=rank * owned_width,
            global_nx=grid.nx,
        )

    got = jax.pmap(local_dpn, axis_name="shard")(sharded.p)

    _assert_close(got, _stack_face_slices(global_dpn, bounds), rtol=1.0e-12, atol=1.0e-12)


def test_ppermute_halo_then_sharded_operators_match_global_owned_outputs():
    grid = _test_grid()
    state = _deterministic_state(grid)
    bounds = x_partition_bounds(grid.nx, 4)

    halo3_unfilled = partition_state_x(state, grid, num_partitions=4, halo_width=3, fill_halos=False)
    cfg3 = ShardingConfig(enabled=True, num_partitions=4, halo_width=3)
    spec3 = HaloSpec(
        width=3,
        fields_to_exchange=("theta", "qv"),
        edge_type="periodic",
        sharding=cfg3,
    )

    def exchange3(local_state):
        return apply_halo(local_state, spec3)

    halo3 = jax.pmap(exchange3, axis_name=cfg3.axis_name)(halo3_unfilled)
    flux_global = flux5_face_periodic(state.theta, state.qv + 0.25, axis=2)
    flux_got = jax.pmap(
        lambda field, vel: sharded_flux5_face_periodic_x(field, vel + 0.25, halo_width=3),
        axis_name=cfg3.axis_name,
    )(halo3.theta, halo3.qv)
    _assert_close(flux_got, _stack_mass_slices(flux_global, bounds), rtol=2.0e-6, atol=2.0e-7)

    kwargs = {"dt": 10.0, "diff_6th_factor": 0.12, "horizontal_only": True, "monotonic": True}
    diffusion_global = sixth_order_diffusion_tendency(state.theta, **kwargs)
    diffusion_got = jax.pmap(
        lambda field: sharded_sixth_order_diffusion_tendency(field, halo_width=3, **kwargs),
        axis_name=cfg3.axis_name,
    )(halo3.theta)
    _assert_close(diffusion_got, _stack_mass_slices(diffusion_global, bounds), rtol=2.0e-6, atol=2.0e-9)

    halo1_unfilled = partition_state_x(state, grid, num_partitions=4, halo_width=1, fill_halos=False)
    cfg1 = ShardingConfig(enabled=True, num_partitions=4, halo_width=1)
    spec1 = HaloSpec(
        width=1,
        fields_to_exchange=("u", "p"),
        edge_type="periodic",
        sharding=cfg1,
    )

    def exchange1(local_state):
        return apply_halo(local_state, spec1)

    halo1 = jax.pmap(exchange1, axis_name=cfg1.axis_name)(halo1_unfilled)
    div_global = 0.1 * (state.u[:, :, 1 : grid.nx + 1] - state.u[:, :, : grid.nx])
    div_got = jax.pmap(
        lambda u: sharded_x_staggered_divergence(u, rdx=0.1, halo_width=1),
        axis_name=cfg1.axis_name,
    )(halo1.u)
    _assert_close(div_got, _stack_mass_slices(div_global, bounds), rtol=0.0, atol=0.0)

    fnm = jnp.linspace(0.2, 0.8, grid.nz, dtype=state.p.dtype)
    fnp = 1.0 - fnm
    cf1 = jnp.asarray(0.55, dtype=state.p.dtype)
    cf2 = jnp.asarray(0.30, dtype=state.p.dtype)
    cf3 = jnp.asarray(0.15, dtype=state.p.dtype)
    global_dpn = _global_x_face_pressure_dpn(state.p, fnm=fnm, fnp=fnp, cf1=cf1, cf2=cf2, cf3=cf3)
    owned_width = grid.nx // 4

    def local_dpn(p):
        rank = jax.lax.axis_index("shard")
        return sharded_x_face_pressure_dpn(
            p,
            fnm=fnm,
            fnp=fnp,
            cf1=cf1,
            cf2=cf2,
            cf3=cf3,
            halo_width=1,
            global_start=rank * owned_width,
            global_nx=grid.nx,
        )

    dpn_got = jax.pmap(local_dpn, axis_name=cfg1.axis_name)(halo1.p)
    _assert_close(dpn_got, _stack_face_slices(global_dpn, bounds), rtol=1.0e-12, atol=1.0e-12)
