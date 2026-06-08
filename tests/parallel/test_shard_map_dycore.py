"""Fake-mesh bit-identity tests for the ``jax.shard_map`` dycore decomposition.

CPU fake-device only.  Requires a fake multi-device mesh, e.g.::

    PYTHONPATH=src JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    taskset -c 0-3 pytest -q tests/parallel/test_shard_map_dycore.py

Real multi-GPU throughput is NOT exercised here (one physical GPU); these tests
validate that the domain decomposition introduces zero numerical change.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.dynamics.flux_advection import flux5_face_periodic
from gpuwrf.runtime.shard_map_dycore import (
    make_x_mesh,
    periodic_ring_halo_x,
    sharded_flux5_advection_x,
    sharded_sixth_order_diffusion_x,
    single_device_flux5_advection_x,
    single_device_sixth_order_diffusion_x,
)

_NEED_MESH = pytest.mark.skipif(
    len(jax.local_devices()) < 4,
    reason="requires fake or real >=4-device mesh (XLA_FLAGS=--xla_force_host_platform_device_count)",
)


def _field(seed: int, shape=(8, 6, 48)) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=jnp.float64)


@_NEED_MESH
@pytest.mark.parametrize("partitions", [2, 4])
def test_flux5_advection_partition_invariant_bit_identical(partitions):
    field, vel = _field(0), _field(1)
    base = np.asarray(sharded_flux5_advection_x(field, vel, make_x_mesh(1, halo_width=3)))
    got = np.asarray(sharded_flux5_advection_x(field, vel, make_x_mesh(partitions, halo_width=3)))
    assert np.array_equal(got, base), f"P={partitions} not bit-identical to P=1"


@_NEED_MESH
@pytest.mark.parametrize("partitions", [2, 4])
def test_sixth_order_diffusion_partition_invariant_bit_identical(partitions):
    field = _field(2)
    kwargs = dict(dt=12.0, diff_6th_factor=0.12)
    base = np.asarray(sharded_sixth_order_diffusion_x(field, make_x_mesh(1, halo_width=3), **kwargs))
    got = np.asarray(
        sharded_sixth_order_diffusion_x(field, make_x_mesh(partitions, halo_width=3), **kwargs)
    )
    assert np.array_equal(got, base), f"P={partitions} not bit-identical to P=1"


@_NEED_MESH
@pytest.mark.parametrize("partitions", [2, 4])
def test_ppermute_halo_reconstructs_analytic_field(partitions):
    nx = 48
    h = 3
    i = np.arange(nx, dtype=np.float64)
    analytic = np.sin(2.0 * np.pi * i / nx)
    field = jnp.asarray(analytic.reshape(1, 1, nx))
    shard = make_x_mesh(partitions, halo_width=h)

    def body(local):
        return periodic_ring_halo_x(
            local, width=h, axis_name=shard.axis_name, num_partitions=shard.num_partitions
        )

    sm = jax.shard_map(body, mesh=shard.mesh, in_specs=shard.in_spec, out_specs=shard.in_spec)
    haloed = np.asarray(jax.jit(sm)(jax.device_put(field, shard.sharding)))

    owned = nx // partitions
    per_shard = haloed.reshape(1, 1, partitions, owned + 2 * h)
    for rank in range(partitions):
        start = rank * owned
        expected = analytic[(np.arange(start - h, start + owned + h)) % nx]
        assert np.array_equal(per_shard[0, 0, rank], expected), f"rank {rank} halo mismatch"


@_NEED_MESH
def test_decomposition_is_exact_in_eager_mode():
    """Padded-local stencil == global stencil bit-for-bit (eager): proves the
    sharded-vs-eager fp residual is XLA jit reassociation, not decomposition."""
    field, vel = _field(4), _field(5)
    h = 3
    global_eager = np.asarray(single_device_flux5_advection_x(field, vel))

    fp = jnp.concatenate([field[..., -h:], field, field[..., :h]], axis=-1)
    vp = jnp.concatenate([vel[..., -h:], vel, vel[..., :h]], axis=-1)
    ff = flux5_face_periodic(fp, vp, axis=-1)
    padded = np.asarray((-(jnp.roll(ff, -1, axis=-1) - ff))[..., h:-h])
    assert np.array_equal(padded, global_eager)


@_NEED_MESH
def test_sharded_matches_eager_global_within_fp64_roundoff():
    field, vel = _field(6), _field(7)
    global_eager = np.asarray(single_device_flux5_advection_x(field, vel))
    sharded = np.asarray(sharded_flux5_advection_x(field, vel, make_x_mesh(2, halo_width=3)))
    assert float(np.max(np.abs(sharded - global_eager))) < 1e-12


@_NEED_MESH
def test_sixth_order_matches_eager_global_within_fp64_roundoff():
    field = _field(8)
    kwargs = dict(dt=12.0, diff_6th_factor=0.12)
    global_eager = np.asarray(single_device_sixth_order_diffusion_x(field, **kwargs))
    sharded = np.asarray(
        sharded_sixth_order_diffusion_x(field, make_x_mesh(2, halo_width=3), **kwargs)
    )
    assert float(np.max(np.abs(sharded - global_eager))) < 1e-12


def test_make_x_mesh_rejects_too_many_partitions():
    with pytest.raises(ValueError, match="partitions"):
        make_x_mesh(len(jax.devices()) + 1, halo_width=3)


def test_flux5_requires_three_halo_cells():
    field, vel = _field(0), _field(1)
    with pytest.raises(ValueError, match="3 halo"):
        sharded_flux5_advection_x(field, vel, make_x_mesh(1, halo_width=2))
