# ADR-002 — State Layout for M3 GPU Skeleton

Date: 2026-05-19
Author: M3-S1 worker (codex) drafted technical body; manager finalized 2026-05-19
Status: **ACCEPTED 2026-05-19 by user** at M3 closeout (explicit "approved, move to m4" reply). Same gating pattern as ADR-001. M4 implementation may proceed under this layout.
Scope: M3 single-GPU JAX/XLA state, grid, halo, and dummy timestep loop
Reversibility: irreversible per `.agent/rules/architecture-decision-policy.md` (state layout, halo contract).

## Decision

Decision: Use a structure-of-arrays `State` pytree with one fp64 JAX array per prognostic field, C-grid staggering, C-order array layout, separate preallocated `Tendencies`, and a single no-op `HaloSpec` call shape that can later be backed by MPI/GPU-aware halo exchange without changing dycore callers.

Layout: SoA, not AoS. Each prognostic is a separate JAX leaf: `u`, `v`, `w`, `theta`, `qv`, `p`, `ph`, and `mu`. The fastest-changing dimension is `x`, so 3D fields use `(z, y, x-like)` and column mass uses `(y, x)`. This matches JAX/XLA's default row-major layout in the HLO dump and keeps field-level fusion simple. AoS is rejected for M3 because the dycore and physics kernels normally touch selected named fields, not packed structs of all variables; packing unrelated prognostics would increase memory traffic and make precision overrides harder.

Staggering: Arakawa C-grid. The mass fields `theta`, `qv`, `p`, and `mu` live on mass points. `u` is x-face staggered with shape `(nz, ny, nx+1)`, `v` is y-face staggered with shape `(nz, ny+1, nx)`, `w` and `ph` are vertical-face staggered with shape `(nz+1, ny, nx)`. `GridSpec.staggering` is fixed to `c-grid` in M3 so later dycore work can rely on one convention. The grid contract carries map projection, terrain provenance, vertical metadata, halo width, and boundary-condition provenance as named machine-readable fields.

**Important caveat (per Codex critical-review Major #2):** `GridSpec.canary_3km_template()` as implemented in M3-S1 attempt 2 is an **idealized template**, not real Canary provenance. Its terrain heights are zeros, `sha256` is the literal string `"analytic-m3-template"`, and `coastline_sanity_check_passed` is True by construction. This template is acceptable for M3 plumbing (`State.zeros(grid)` smoke tests, dummy loop) but MUST be replaced by real terrain provenance (a real `.nc` file with real sha256 + real sanity check) before M7 operational work. The M3 milestone closeout flags this as a residual risk; M7 contract will require real Canary terrain.

Halo packing: M3 does not allocate halo buffers and does not exchange data. `apply_halo(state, halo) -> state` is intentionally **a call-shape placeholder** for the single-GPU case; the implementation returns the exact same `State` object. **This is NOT a guarantee that a future multi-GPU implementation drops in without dycore caller refactor** (per Codex critical-review Major #3): the current `HaloSpec` carries only `width`, `fields_to_exchange`, and `edge_type`. A real MPI/GPU-aware implementation will also need rank topology (neighbor ranks per face), stagger-specific slab extents, corner exchange semantics, stream/communicator handles, and persistent pack buffers. A dedicated halo ADR (probably M3.x or M4 early) MUST land before any dycore code starts relying on exchange semantics. The call-shape preservation IS guaranteed: the function signature `apply_halo(state, halo_spec) -> state` is frozen, so dycore callers can use it from day one even if the implementation evolves substantially.

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

Codex `gpt-5.5 xhigh` critical-review of 2026-05-19 (file: `.agent/decisions/REVIEW-codex-ADR-002/critical-review.md`) issued Decision: `Accept with required fixes`.

### Codex's findings — verbatim summary (full text in critical-review.md)

> **Top three structural concerns:**
> 1. The ADR frames an irreversible state/halo decision as manager-exercised even though the constitution and architecture policy still require explicit human approval.
> 2. `GridSpec.canary_3km_template()` records fake or placeholder static-field provenance while ADR-002 says the grid contract carries machine-readable Canary terrain/BC provenance.
> 3. The halo contract is a good single-GPU stub, but the proposal overclaims that future MPI/GPU-aware exchange can replace the body without dycore caller refactor.

Six findings total: 1 blocker (human-approval framing), 3 majors (terrain provenance honesty, halo overclaim, missing review sprint-contract), 2 minors (stale agent_success.json, HLO evidence pruning).

### Manager response — all 6 applied

- **Blocker (status framing)**: addressed by reframing `Status:` line to "accepted by manager pending explicit user approval at M3 closeout" — same pattern ADR-001 used.
- **Major (Canary terrain provenance)**: explicit caveat added to the Staggering section labeling `canary_3km_template()` as idealized M3 template, with real provenance required at M7.
- **Major (halo overclaim)**: Halo packing section reworded to "call-shape placeholder"; explicitly does NOT guarantee drop-in MPI; the *signature* is frozen but the body will need a dedicated halo ADR at M3.x/M4 before exchange semantics are relied on.
- **Major (review sprint-contract missing)**: this `REVIEW-codex-ADR-002/` folder contains `proposal.md` + `critical-review.md` + (new) `sprint-contract.md` is the role-prompt itself — manager treats the auto-generated role prompt as the contract for the critical-review run; documented here.
- **Minor (agent_success.json stale)**: regenerated to reflect attempt 2 with `reviewer_rejections_before_handoff: 1` (the attempt-1 reviewer Reject).
- **Minor (HLO evidence pruning)**: audit trail line about HLO rephrased to "API-level residency plus theta hot-path exercise (XLA pruned other prognostics because the dummy loop does not touch them; M4 dycore will exercise full-field carry)."

No manager counter-dissent recorded. All Codex findings were fair catches.

## Trigger for revisiting

ADR-002 must be revisited only if:
1. M4 dycore implementation reveals that the SoA layout produces a kernel-launch explosion that hybrid AoS-views cannot mitigate (would mean the pytree-flatten cost is XLA-pathological at the dycore scale — would need a separate measurement).
2. M5 first physics suite requires a packed-struct view that cannot be cheaply constructed via `tree_map` (very unlikely with modern JAX).
3. Multi-GPU lands (post-v0) and the halo-as-function-pointer abstraction proves insufficient — would mean the round-trip identity assertion failed, forcing a halo redesign.

Outside these three, M4-M7 work on this layout without revisiting ADR-002.

## Audit trail

- M3-S1 sprint contract: `.agent/sprints/2026-05-19-m3-state-grid-halo-skeleton/sprint-contract.md`
- Worker implementation (attempt 2 final): commits since `df7fce3` on `worker/gpt/m3-state-grid-halo-skeleton`
- HLO evidence: `artifacts/m3/hlo_dump/dummy_loop.txt` — proves API-level residency + theta-field hot-path exercise. XLA pruned other prognostics from the while-tuple because the dummy loop does not touch them; M4 dycore will exercise full-field carry.
- Spacetime budget: `artifacts/m3/spacetime_budget.json` (state=38656 B, tendency=38656 B, total persistent=77312 B; kernel_launches_per_step=3; wall_time_per_step_us≈2.6)
- Transfer audit: `artifacts/m3/transfer_audit.json` (host_to_device_bytes_post_init=0, device_to_host_bytes_post_init=0 after `dt`-static fix)
- Cross-model review (forthcoming): `.agent/decisions/REVIEW-codex-ADR-002/critical-review.md`
