# ADR-002 — State Layout for M3 GPU Skeleton

Date: 2026-05-19
Author: M3-S1 worker (codex) drafted technical body; manager finalized 2026-05-19
Status: **proposed by manager**; awaiting Codex critical-review per `.agent/rules/cross-model-review-policy.md`; manager will apply findings or record dissent before M3 milestone closeout
Scope: M3 single-GPU JAX/XLA state, grid, halo, and dummy timestep loop
Reversibility: irreversible per `.agent/rules/architecture-decision-policy.md` (state layout, halo contract). Per manager-autonomy directive, manager exercises with Codex cross-model review; reports to user at M3 closeout.

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

## Risks

- **SoA may underperform for kernels that touch all prognostics at once** (e.g. some physics couplers). Mitigation: M5 first-physics-suite decision-gate sprint will identify if any scheme materially needs an AoS view; `jax.tree_util.tree_map` can construct a temporary AoS view without changing storage.
- **C-grid staggering is the WRF convention and most NWP papers**; switching to C-D or A-grid later would be a major rewrite. Locked here because the project explicitly targets WRF compatibility per the constitution.
- **fp64 across all prognostics**: on RTX 5090 Blackwell consumer, fp64 is 1:64 throughput vs fp32. This is the M4 precision-policy concern; M3 just establishes the reference path. ADR-003 (M4) will propose validated per-field downcasts where safe.
- **`apply_halo` no-op for single-GPU** could mask a future MPI-version bug if the no-op return isn't structurally identical to a pack/exchange/unpack round-trip. Mitigation: M3-S1 test asserts `apply_halo(state, halo) is state` (identity, not just equality) for the no-op case; future multi-GPU implementation MUST satisfy `tree_all(state == apply_halo(state, halo))` after round-trip.
- **`State.zeros(grid)` on first GPU device** is correct for single-GPU but will need a per-device factory when multi-GPU lands. Caller interface is fine; init code will need a `device_id: int = 0` parameter at M3.x or M6.

## Cross-model challenge

To be populated from `.agent/decisions/REVIEW-codex-ADR-002/critical-review.md` after the Codex `gpt-5.5 xhigh` critical-review runs. Per `.agent/rules/cross-model-review-policy.md`, dissent is preserved verbatim.

**Placeholder until critical-review returns:** none recorded.

## Trigger for revisiting

ADR-002 must be revisited only if:
1. M4 dycore implementation reveals that the SoA layout produces a kernel-launch explosion that hybrid AoS-views cannot mitigate (would mean the pytree-flatten cost is XLA-pathological at the dycore scale — would need a separate measurement).
2. M5 first physics suite requires a packed-struct view that cannot be cheaply constructed via `tree_map` (very unlikely with modern JAX).
3. Multi-GPU lands (post-v0) and the halo-as-function-pointer abstraction proves insufficient — would mean the round-trip identity assertion failed, forcing a halo redesign.

Outside these three, M4-M7 work on this layout without revisiting ADR-002.

## Audit trail

- M3-S1 sprint contract: `.agent/sprints/2026-05-19-m3-state-grid-halo-skeleton/sprint-contract.md`
- Worker implementation (attempt 2 final): commits since `df7fce3` on `worker/gpt/m3-state-grid-halo-skeleton`
- HLO evidence: `artifacts/m3/hlo_dump/dummy_loop.txt`
- Spacetime budget: `artifacts/m3/spacetime_budget.json` (state=38656 B, tendency=38656 B, total persistent=77312 B; kernel_launches_per_step=3; wall_time_per_step_us≈2.6)
- Transfer audit: `artifacts/m3/transfer_audit.json` (host_to_device_bytes_post_init=0, device_to_host_bytes_post_init=0 after `dt`-static fix)
- Cross-model review (forthcoming): `.agent/decisions/REVIEW-codex-ADR-002/critical-review.md`
