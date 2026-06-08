"""``jax.shard_map`` domain decomposition for the dominant dycore stencils.

This module is the v0.13 extension of the v0.11 DGX ``pmap`` substrate
(:mod:`gpuwrf.runtime.sharding`).  Where the v0.11 path partitions a real
operational ``State`` and traces ``run_forecast_operational`` under a leading
device axis via ``jax.pmap``, this module expresses the *single-program*
domain-decomposition directly with :class:`jax.sharding.Mesh` +
:func:`jax.shard_map`:

* the horizontal grid is sharded along a 1-D mesh axis (x);
* halos are refreshed *inside* the ``shard_map`` body with collective
  ``jax.lax.ppermute`` (a periodic ring exchange across the mesh axis);
* the dominant dycore horizontal stencils (5th-order flux advection and the
  6th-order numerical diffusion filter) are applied on the haloed local shard
  and trimmed back to the owned interior.

The deliverable is **bit-identity** versus the single-device path on a
fake/CPU multi-device mesh.  Real multi-GPU throughput is *not* measured here:
this workstation has one physical GPU, so per-watt / whole-Earth claims stay
PROJECTED, never MEASURED.  The fake-mesh bit-identity is the validatable
contract.

Why ``shard_map`` and not only ``pmap``:

* ``shard_map`` is the current single-program-multiple-data primitive; it maps a
  per-shard function over a named ``Mesh`` and lets collective primitives
  (``ppermute``, ``all_gather`` ...) appear in the body referring to the mesh
  axis by name.  It composes under ``jit`` without a separate leading device
  axis on the inputs, which is the shape a future real WRF timestep loop wants
  (the global array is sharded, not pre-split host-side).
* The collective halo exchange is identical in spirit to the v0.11 ppermute
  ring; the bit-identity gate proves the SPMD lowering reproduces the global
  stencil exactly.

Nothing here is wired into the default forecast path: importing this module is
inert and the default single-GPU compiled graph is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from gpuwrf.dynamics.explicit_diffusion import sixth_order_diffusion_tendency
from gpuwrf.dynamics.flux_advection import flux5_face_periodic

# Importing the dynamics modules above already enables jax_enable_x64.  No GPU
# context is initialised by importing this module.

__all__ = [
    "ShardMapMesh",
    "make_x_mesh",
    "periodic_ring_halo_x",
    "shard_x_field",
    "unshard_x_field",
    "sharded_flux5_advection_x",
    "sharded_sixth_order_diffusion_x",
    "single_device_flux5_advection_x",
    "single_device_sixth_order_diffusion_x",
]


@dataclass(frozen=True)
class ShardMapMesh:
    """A 1-D x-decomposition mesh built from (fake or real) devices.

    ``num_partitions`` devices are taken from ``jax.devices()`` and named on a
    single mesh axis.  ``halo_width`` is the per-side ghost-cell depth the
    stencil bodies require (>= the stencil reach).
    """

    mesh: Mesh
    axis_name: str
    num_partitions: int
    halo_width: int

    @property
    def in_spec(self) -> PartitionSpec:
        """PartitionSpec that shards the last (x) axis over the mesh axis."""

        return PartitionSpec(None, None, self.axis_name)

    @property
    def replicated_spec(self) -> PartitionSpec:
        return PartitionSpec()

    @property
    def sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, self.in_spec)


def make_x_mesh(
    num_partitions: int,
    *,
    axis_name: str = "x",
    halo_width: int = 3,
) -> ShardMapMesh:
    """Build a 1-D x mesh over ``num_partitions`` visible devices.

    Works for both fake CPU devices (``--xla_force_host_platform_device_count``)
    and real GPUs; the device platform is whatever JAX exposes.
    """

    n = int(num_partitions)
    if n < 1:
        raise ValueError("num_partitions must be positive")
    devices = jax.devices()
    if n > len(devices):
        raise ValueError(f"requested {n} partitions but only {len(devices)} devices visible")
    if int(halo_width) < 1:
        raise ValueError("halo_width must be positive")
    mesh = jax.make_mesh((n,), (axis_name,), devices=tuple(devices[:n]))
    return ShardMapMesh(
        mesh=mesh,
        axis_name=str(axis_name),
        num_partitions=n,
        halo_width=int(halo_width),
    )


# --------------------------------------------------------------------------- #
# Collective periodic halo exchange (inside shard_map body)
# --------------------------------------------------------------------------- #


def periodic_ring_halo_x(
    interior: jax.Array,
    *,
    width: int,
    axis_name: str,
    num_partitions: int,
) -> jax.Array:
    """Append periodic x halos to a local *interior* shard via ``ppermute``.

    ``interior`` is the owned x interval (no ghost cells) for one shard.  The
    returned array is ``interior`` flanked by ``width`` ghost cells on each side,
    pulled from the ring neighbours so the global periodic wrap is preserved.

    Must be called inside a ``shard_map`` body so ``axis_name`` resolves to the
    mesh axis.
    """

    h = int(width)
    if h <= 0:
        return interior
    if int(interior.shape[-1]) < h:
        raise ValueError("local x interior is smaller than the halo width")
    n = int(num_partitions)
    # left_edge of this shard feeds the right halo of the left neighbour;
    # right_edge feeds the left halo of the right neighbour.  ppermute perm maps
    # source rank -> destination rank.
    right_edge = interior[..., -h:]
    left_edge = interior[..., :h]
    send_right = tuple((rank, (rank + 1) % n) for rank in range(n))
    send_left = tuple((rank, (rank - 1) % n) for rank in range(n))
    left_halo = jax.lax.ppermute(right_edge, axis_name=axis_name, perm=send_right)
    right_halo = jax.lax.ppermute(left_edge, axis_name=axis_name, perm=send_left)
    return jnp.concatenate((left_halo, interior, right_halo), axis=-1)


# --------------------------------------------------------------------------- #
# Shard / unshard a global x-mass field across the mesh
# --------------------------------------------------------------------------- #


def shard_x_field(field: jax.Array, shard: ShardMapMesh) -> jax.Array:
    """Distribute a global ``(nz, ny, nx)`` field across the mesh x axis."""

    nx = int(field.shape[-1])
    if nx % shard.num_partitions != 0:
        raise ValueError(f"nx={nx} must be divisible by num_partitions={shard.num_partitions}")
    return jax.device_put(field, shard.sharding)


def unshard_x_field(field: jax.Array) -> jax.Array:
    """Gather a sharded field back to a single addressable host array."""

    return jnp.asarray(jax.device_get(field))


# --------------------------------------------------------------------------- #
# Single-device references (the bit-identity targets)
# --------------------------------------------------------------------------- #


def single_device_flux5_advection_x(field: jax.Array, vel_face: jax.Array) -> jax.Array:
    """Global 5th-order flux advection tendency on a periodic x axis.

    Returns the flux divergence ``-(F_{i+1} - F_i)`` on mass cells where ``F`` is
    the WRF ``flux5`` left-face value.  ``vel_face`` is the mass-coupled face
    velocity collocated with the flux (periodic ``nx`` faces).  This is the
    single-domain reference the sharded path must match bit-for-bit.
    """

    face_flux = flux5_face_periodic(field, vel_face, axis=-1)
    east_face = jnp.roll(face_flux, -1, axis=-1)
    return -(east_face - face_flux)


def single_device_sixth_order_diffusion_x(
    field: jax.Array,
    *,
    dt: float,
    diff_6th_factor: float,
    monotonic: bool = True,
) -> jax.Array:
    """Global 6th-order x-diffusion tendency (single-domain reference)."""

    return sixth_order_diffusion_tendency(
        field,
        dt=dt,
        diff_6th_factor=diff_6th_factor,
        horizontal_only=True,
        monotonic=monotonic,
    )


# --------------------------------------------------------------------------- #
# Sharded stencil application via shard_map
# --------------------------------------------------------------------------- #


def _local_flux5_owned(
    field_interior: jax.Array,
    vel_interior: jax.Array,
    *,
    width: int,
    axis_name: str,
    num_partitions: int,
) -> jax.Array:
    """Per-shard flux5 advection: refresh halos, apply stencil, trim to owned."""

    haloed_field = periodic_ring_halo_x(
        field_interior, width=width, axis_name=axis_name, num_partitions=num_partitions
    )
    haloed_vel = periodic_ring_halo_x(
        vel_interior, width=width, axis_name=axis_name, num_partitions=num_partitions
    )
    # flux5 face value at the left face of each local cell; the east face is the
    # next cell's left face (roll within the haloed local field is exact because
    # the halos carry the neighbour values).
    face_flux = flux5_face_periodic(haloed_field, haloed_vel, axis=-1)
    east_face = jnp.roll(face_flux, -1, axis=-1)
    tend = -(east_face - face_flux)
    h = int(width)
    return tend[..., h:-h]


def _local_sixth_owned(
    field_interior: jax.Array,
    *,
    width: int,
    axis_name: str,
    num_partitions: int,
    dt: float,
    diff_6th_factor: float,
    monotonic: bool,
) -> jax.Array:
    """Per-shard 6th-order x-diffusion: halo refresh, stencil, trim to owned."""

    haloed = periodic_ring_halo_x(
        field_interior, width=width, axis_name=axis_name, num_partitions=num_partitions
    )
    tend = sixth_order_diffusion_tendency(
        haloed,
        dt=dt,
        diff_6th_factor=diff_6th_factor,
        horizontal_only=True,
        monotonic=monotonic,
    )
    h = int(width)
    return tend[..., h:-h]


def sharded_flux5_advection_x(
    field: jax.Array,
    vel_face: jax.Array,
    shard: ShardMapMesh,
) -> jax.Array:
    """Domain-decomposed flux5 advection over the mesh, gathered to a global array.

    Bit-identity target: :func:`single_device_flux5_advection_x`.

    Both inputs are global ``(nz, ny, nx)`` arrays.  ``vel_face`` here is the
    periodic ``nx``-face mass-coupled velocity (same convention as the
    single-device reference); each shard owns the same ``nx/P`` x columns.
    """

    if shard.halo_width < 3:
        raise ValueError("flux5 needs at least 3 halo cells")
    body = partial(
        _local_flux5_owned,
        width=shard.halo_width,
        axis_name=shard.axis_name,
        num_partitions=shard.num_partitions,
    )
    sm = jax.shard_map(
        body,
        mesh=shard.mesh,
        in_specs=(shard.in_spec, shard.in_spec),
        out_specs=shard.in_spec,
    )
    field_d = shard_x_field(field, shard)
    vel_d = shard_x_field(vel_face, shard)
    return jax.jit(sm)(field_d, vel_d)


def sharded_sixth_order_diffusion_x(
    field: jax.Array,
    shard: ShardMapMesh,
    *,
    dt: float,
    diff_6th_factor: float,
    monotonic: bool = True,
) -> jax.Array:
    """Domain-decomposed 6th-order x-diffusion over the mesh, gathered to global.

    Bit-identity target: :func:`single_device_sixth_order_diffusion_x`.
    """

    if shard.halo_width < 3:
        raise ValueError("6th-order diffusion needs at least 3 halo cells")
    body = partial(
        _local_sixth_owned,
        width=shard.halo_width,
        axis_name=shard.axis_name,
        num_partitions=shard.num_partitions,
        dt=float(dt),
        diff_6th_factor=float(diff_6th_factor),
        monotonic=bool(monotonic),
    )
    sm = jax.shard_map(
        body,
        mesh=shard.mesh,
        in_specs=(shard.in_spec,),
        out_specs=shard.in_spec,
    )
    field_d = shard_x_field(field, shard)
    return jax.jit(sm)(field_d)
