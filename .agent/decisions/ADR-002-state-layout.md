# ADR-002 — State Layout for M3 GPU Skeleton

Date: 2026-05-19
Status: worker draft for manager finalization
Scope: M3 single-GPU JAX/XLA state, grid, halo, and dummy timestep loop

## Decision

Decision: Use a structure-of-arrays `State` pytree with one fp64 JAX array per prognostic field, C-grid staggering, C-order array layout, separate preallocated `Tendencies`, and a single no-op `HaloSpec` call shape that can later be backed by MPI/GPU-aware halo exchange without changing dycore callers.

Layout: SoA, not AoS. Each prognostic is a separate JAX leaf: `u`, `v`, `w`, `theta`, `qv`, `p`, `ph`, and `mu`. The fastest-changing dimension is `x`, so 3D fields use `(z, y, x-like)` and column mass uses `(y, x)`. This matches JAX/XLA's default row-major layout in the HLO dump and keeps field-level fusion simple. AoS is rejected for M3 because the dycore and physics kernels normally touch selected named fields, not packed structs of all variables; packing unrelated prognostics would increase memory traffic and make precision overrides harder.

Staggering: Arakawa C-grid. The mass fields `theta`, `qv`, `p`, and `mu` live on mass points. `u` is x-face staggered with shape `(nz, ny, nx+1)`, `v` is y-face staggered with shape `(nz, ny+1, nx)`, `w` and `ph` are vertical-face staggered with shape `(nz+1, ny, nx)`. `GridSpec.staggering` is fixed to `c-grid` in M3 so later dycore work can rely on one convention. The grid contract carries map projection, terrain provenance, vertical metadata, halo width, and boundary-condition provenance as named machine-readable fields.

Halo packing: M3 does not allocate halo buffers and does not exchange data. `apply_halo(state, halo) -> state` is intentionally the only call shape, and the single-GPU implementation returns the exact same `State` object. Future multi-GPU work can replace the function body with pack/exchange/unpack while preserving every caller. The planned packing strategy is field-selective SoA packing: exchange only `HaloSpec.fields_to_exchange`, pack contiguous edge slabs per field, and keep edge type (`periodic`, `open`, `nest_boundary`) in `HaloSpec` so boundary semantics do not leak into dycore kernels. M3 records `halo_buffer_bytes = 0`; a later multi-GPU ADR must account for persistent pack buffers explicitly.

## Precision

All M3 prognostic and tendency leaves are fp64. This follows the project precision policy default for the mass-conservation pathway. The dtype registry is deliberately small and per-field so later M4/M5 work can propose validated overrides without changing the `State` API. No mixed precision is authorized by this ADR draft.

## Residency And Timestep Carry

`State.zeros(grid)` and `Tendencies.zeros(grid)` allocate every leaf once on the first visible JAX GPU and raise if no GPU backend is visible. The timestep loop accepts an already allocated `State`, an already allocated `Tendencies`, scalar `dt`, and static `n_steps`. The loop is one `jax.jit` around `jax.lax.scan`; the scanned body has no `jnp.array`, `jnp.zeros`, or `jnp.empty` calls. The dummy operation updates `theta` through an add/subtract chain using preallocated tendency data, forcing a real fused HLO body while remaining physics-neutral.

## Consequences

The positive consequence is a compact, auditable API: every persistent byte appears in the budget JSON, every state field has a single owner, and XLA sees a simple pytree carry. The main cost is that future kernels must explicitly pass any grouped field views they need instead of indexing a packed struct. That is acceptable because the project values clear memory traffic over convenience packing. Multi-GPU halo buffers are deferred, but the caller interface is frozen now so M4 dycore code should not need a refactor when exchange becomes real.
