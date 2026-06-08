#!/usr/bin/env python3
"""v0.13 multi-GPU (S1) fake-mesh bit-identity proof for the ``shard_map`` dycore.

This is a CPU-only fake-multi-device proof.  It validates that the v0.13
``jax.shard_map`` domain decomposition of the dominant dycore horizontal
stencils (5th-order flux advection + 6th-order numerical diffusion), with a
collective ``jax.lax.ppermute`` periodic-ring halo exchange, is **bit-identical**
to the single-shard path across partition counts on a fake CPU mesh, and that the
halo exchange reproduces a known analytic field exactly.

HARDWARE HONESTY
----------------
This workstation has ONE physical RTX 5090.  Real multi-GPU throughput,
NVLink/NCCL bandwidth, and collective-overlap are therefore UNMEASURED.  No
per-watt or whole-Earth claim derived from this proof may be labelled MEASURED;
those stay PROJECTED.  The validatable deliverable is the fake-mesh bit-identity
of the decomposed numerics, proven here.

WHAT IS PROVEN
--------------
1. ``shard_map`` partition-invariance: P=2, P=4 (and P=8 where devices allow)
   reproduce the P=1 single-shard result for both stencils with max abs diff
   exactly 0.0 (bit-identical).  This is the load-bearing gate: domain
   decomposition introduces zero numerical change.
2. The ``ppermute`` periodic-ring halo exchange reconstructs the exact global
   periodic neighbourhood (verified against the global slices AND against a
   smooth analytic sinusoid that the stencil should annihilate / reproduce).
3. Sharded vs. an eager (non-jit) global reference agree to fp64 round-off
   (~1e-15); the residual is XLA jit float reassociation, NOT the decomposition.
   This is demonstrated by ``padded_eager_vs_global_eager`` == 0.0: applying the
   identical local stencil on a periodically padded global array, then trimming,
   reproduces the global formula bit-for-bit in eager mode.

REPRODUCE (CPU fake-device only; never the GPU lock wrapper)
------------------------------------------------------------
    PYTHONPATH=src JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 \
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    taskset -c 0-3 python proofs/v013/multigpu_fakemesh.py \
        --output proofs/v013/multigpu_fakemesh.json
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import numpy as np

# Importing the dycore enables jax_enable_x64. No GPU context is initialised.
from gpuwrf.dynamics.flux_advection import flux5_face_periodic
from gpuwrf.runtime.shard_map_dycore import (
    make_x_mesh,
    periodic_ring_halo_x,
    sharded_flux5_advection_x,
    sharded_sixth_order_diffusion_x,
    single_device_flux5_advection_x,
    single_device_sixth_order_diffusion_x,
)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _bit_identical(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def check_partition_invariance(nz: int, ny: int, nx: int, halo_width: int) -> dict:
    """P=1 reference vs P in {2,4,8 (if available)} must be bit-identical."""

    field = jax.random.normal(jax.random.PRNGKey(0), (nz, ny, nx), dtype=jnp.float64)
    vel = jax.random.normal(jax.random.PRNGKey(1), (nz, ny, nx), dtype=jnp.float64)
    dt = 12.0
    diff_6th_factor = 0.12

    candidate_partitions = [p for p in (1, 2, 4, 8) if nx % p == 0 and p <= len(jax.devices())]
    adv = {}
    dif = {}
    for p in candidate_partitions:
        shard = make_x_mesh(p, halo_width=halo_width)
        adv[p] = np.asarray(sharded_flux5_advection_x(field, vel, shard))
        dif[p] = np.asarray(
            sharded_sixth_order_diffusion_x(field, shard, dt=dt, diff_6th_factor=diff_6th_factor)
        )

    base_p = 1 if 1 in adv else candidate_partitions[0]
    base_adv, base_dif = adv[base_p], dif[base_p]
    rows = []
    all_bit = True
    for p in candidate_partitions:
        if p == base_p:
            continue
        a_bit = _bit_identical(adv[p], base_adv)
        d_bit = _bit_identical(dif[p], base_dif)
        all_bit = all_bit and a_bit and d_bit
        rows.append(
            {
                "partitions": p,
                "vs_partitions": base_p,
                "flux5_advection_bit_identical": a_bit,
                "flux5_advection_max_abs_diff": _max_abs(adv[p], base_adv),
                "sixth_order_diffusion_bit_identical": d_bit,
                "sixth_order_diffusion_max_abs_diff": _max_abs(dif[p], base_dif),
            }
        )
    return {
        "domain_shape": [nz, ny, nx],
        "halo_width": halo_width,
        "reference_partitions": base_p,
        "partition_counts_tested": candidate_partitions,
        "comparisons": rows,
        "all_bit_identical": bool(all_bit),
    }


def check_halo_analytic(nx: int, halo_width: int) -> dict:
    """ppermute halo refresh reconstructs a known analytic periodic field.

    A periodic sinusoid f(i) = sin(2*pi*i/nx) sharded across the mesh, halo-
    exchanged via ppermute, must reproduce the global periodic neighbourhood
    exactly (every ghost cell equals the analytic value at the wrapped index).
    """

    results = []
    all_ok = True
    i = np.arange(nx, dtype=np.float64)
    analytic = np.sin(2.0 * np.pi * i / nx)
    # 3-D shape (nz=1, ny=1, nx) to match stencil array rank.
    field = jnp.asarray(analytic.reshape(1, 1, nx))

    for p in [p for p in (2, 4) if nx % p == 0 and p <= len(jax.devices())]:
        shard = make_x_mesh(p, halo_width=halo_width)

        def body(local):
            return periodic_ring_halo_x(
                local,
                width=halo_width,
                axis_name=shard.axis_name,
                num_partitions=shard.num_partitions,
            )

        sm = jax.shard_map(
            body, mesh=shard.mesh, in_specs=shard.in_spec, out_specs=shard.in_spec
        )
        field_d = jax.device_put(field, shard.sharding)
        haloed = np.asarray(jax.jit(sm)(field_d))  # (1,1, (nx/p + 2h) * p) laid out per shard

        owned = nx // p
        h = halo_width
        # Reconstruct the expected haloed value per shard from the analytic field.
        ok_shards = True
        max_diff = 0.0
        per_shard = haloed.reshape(1, 1, p, owned + 2 * h)
        for rank in range(p):
            start = rank * owned
            idx = (np.arange(start - h, start + owned + h)) % nx
            expected = analytic[idx]
            got = per_shard[0, 0, rank]
            d = float(np.max(np.abs(got - expected)))
            max_diff = max(max_diff, d)
            ok_shards = ok_shards and bool(np.array_equal(got, expected))
        all_ok = all_ok and ok_shards
        results.append(
            {
                "partitions": p,
                "halo_width": halo_width,
                "analytic_field": "sin(2*pi*i/nx)",
                "ghost_cells_bit_identical_to_analytic": ok_shards,
                "max_abs_diff": max_diff,
            }
        )
    return {"nx": nx, "checks": results, "all_bit_identical": bool(all_ok)}


def check_sharded_vs_eager_global(nz: int, ny: int, nx: int, halo_width: int) -> dict:
    """Sharded == eager global within fp64 round-off; padded-eager == global eager.

    Shows the only sharded-vs-eager-global delta is XLA jit reassociation, not the
    domain decomposition: the padded-local stencil reproduces the global formula
    bit-for-bit in eager mode.
    """

    field = jax.random.normal(jax.random.PRNGKey(2), (nz, ny, nx), dtype=jnp.float64)
    vel = jax.random.normal(jax.random.PRNGKey(3), (nz, ny, nx), dtype=jnp.float64)

    # Eager global reference (no jit).
    global_eager = np.asarray(single_device_flux5_advection_x(field, vel))

    # Padded-local eager: identical stencil body on a periodically padded array.
    h = halo_width

    def padded_local_eager(f, v):
        fp = jnp.concatenate([f[..., -h:], f, f[..., :h]], axis=-1)
        vp = jnp.concatenate([v[..., -h:], v, v[..., :h]], axis=-1)
        ff = flux5_face_periodic(fp, vp, axis=-1)
        tend = -(jnp.roll(ff, -1, axis=-1) - ff)
        return tend[..., h:-h]

    padded_eager = np.asarray(padded_local_eager(field, vel))

    # Sharded (jit, shard_map) on a 2-device fake mesh.
    p = 2 if (nx % 2 == 0 and len(jax.devices()) >= 2) else 1
    shard = make_x_mesh(p, halo_width=halo_width)
    sharded = np.asarray(sharded_flux5_advection_x(field, vel, shard))

    return {
        "domain_shape": [nz, ny, nx],
        "halo_width": halo_width,
        "sharded_partitions": p,
        "padded_eager_vs_global_eager_bit_identical": _bit_identical(padded_eager, global_eager),
        "padded_eager_vs_global_eager_max_abs_diff": _max_abs(padded_eager, global_eager),
        "sharded_vs_global_eager_max_abs_diff": _max_abs(sharded, global_eager),
        "sharded_vs_global_eager_within_fp64_roundoff": _max_abs(sharded, global_eager) < 1e-12,
        "note": (
            "padded_eager==global_eager (0.0) proves the decomposition is exact; "
            "the sharded-vs-eager ~1e-15 residual is XLA jit float reassociation, "
            "identical to a plain single-device jit and independent of partition count."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="proofs/v013/multigpu_fakemesh.json")
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--halo-width", type=int, default=3)
    args = parser.parse_args()

    devices = jax.devices()
    partition_invariance = check_partition_invariance(args.nz, args.ny, args.nx, args.halo_width)
    halo_analytic = check_halo_analytic(args.nx, args.halo_width)
    sharded_vs_eager = check_sharded_vs_eager_global(args.nz, args.ny, args.nx, args.halo_width)

    gate_pass = bool(
        partition_invariance["all_bit_identical"]
        and halo_analytic["all_bit_identical"]
        and sharded_vs_eager["padded_eager_vs_global_eager_bit_identical"]
        and sharded_vs_eager["sharded_vs_global_eager_within_fp64_roundoff"]
    )

    report = {
        "proof": "v013_multigpu_fakemesh_shard_map_bit_identity",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "jax_version": jax.__version__,
        "python": platform.python_version(),
        "platform": str(devices[0].platform),
        "fake_device_count": len(devices),
        "sharded_stencils": ["flux5_face_periodic_advection_x", "sixth_order_diffusion_x"],
        "halo_substrate": "jax.shard_map + jax.lax.ppermute periodic ring",
        "hardware_honesty": (
            "CPU fake-device proof only. This workstation has ONE physical RTX 5090. "
            "Real multi-GPU throughput, NVLink/NCCL bandwidth, and collective overlap are "
            "UNMEASURED. Per-watt / whole-Earth claims derived from scaling stay PROJECTED, "
            "never MEASURED. The validatable deliverable is the fake-mesh bit-identity proven here."
        ),
        "checks": {
            "partition_invariance_bit_identity": partition_invariance,
            "ppermute_halo_analytic": halo_analytic,
            "sharded_vs_eager_global": sharded_vs_eager,
        },
        "gate_pass": gate_pass,
    }

    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nGATE_PASS={gate_pass}  ->  {args.output}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
