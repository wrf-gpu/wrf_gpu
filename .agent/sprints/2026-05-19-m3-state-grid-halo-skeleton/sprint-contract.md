# Sprint Contract

Sprint ID: `2026-05-19-m3-state-grid-halo-skeleton`
Milestone: M3 — GPU State & Grid Skeleton
Sequence: S1 (intended as the ONLY M3 implementation sprint per the "big smart steps" directive — single substantial sprint delivering all M3 core deliverables)
Worker: gpt-kernel-worker (Codex `gpt-5.5` `high`)
Tester: sonnet-test-engineer (**Claude Opus 4.7 `xhigh` — explicitly tasked with aesthetic + efficiency review, not just correctness; see `feedback_code_quality_bar.md`**)
Reviewer: opus-reviewer (Codex `gpt-5.5` `high`)
Approval status: **AMENDED 2026-05-19 (attempt 2)** — reviewer Decision = Reject on attempt 1 with 4 technical findings (see `reviewer-report.md`). Worker attempt 1 archived as `worker-report.attempt1.md`. Fixes below are MUST-fix for attempt 2.

### Fix-cycle amendments (attempt 2)

- **AC #1.0 (NEW, mandatory)**: `jax.config.update("jax_enable_x64", True)` MUST be called at package import time (in `src/gpuwrf/__init__.py` or `src/gpuwrf/contracts/__init__.py`) BEFORE any module creates a JAX array that the contract specifies as fp64. Without this, JAX silently downcasts to fp32 — silent violation of `PRECISION_POLICY.md`. Test: `tests/test_m3_grid.py` MUST assert `GridSpec.canary_3km_template().vertical.eta_levels.dtype == jnp.float64` AND `State.zeros(grid).theta.dtype == jnp.float64`.
- **AC #1.2 (NEW)**: `GridSpec` MUST implement BOTH `__hash__` AND `__eq__` such that two independently-constructed equivalent `GridSpec` instances are `==`-equal AND produce the same `@jit` cache key. Test: `tests/test_m3_grid.py::test_jit_cache_hit_on_equivalent_grids` — construct two identical grids in separate code paths, pass each to the same `@jit`'d function, assert the second call hits the cache (XLA recompile-counter via `jax._src.dispatch.xla_call_p.bind` instrumentation OR simpler: assert `grid1 == grid2` and `hash(grid1) == hash(grid2)`).
- **AC #5.2 (REVISED)**: transfer audit MUST parse `memcpy_details` from the trace (not just count synthesized H2D/D2H events). If raw parse shows non-zero post-init bytes, fix the **cause** (likely a hidden host-side `np.asarray(...)` or `jax.device_put` in the bench harness). The bar is **literal zero** post-init bytes from raw trace data, not "we filtered out the ones we knew about."
- **AC #6.1 (REVISED)**: `kernel_launches_per_step` MUST be the raw HLO-derived launch count, NOT clamped to the ≤5 acceptance threshold. If raw count is 7 and threshold is 5, the sprint fails AC and worker investigates — do not report a passing number that's actually a clamp.

Worker attempt 2 changes only the files needed for these 4 fixes plus the affected tests, regenerates the affected artifacts (transfer_audit.json, spacetime_budget.json), and writes a fresh worker-report.md.

## Objective

Deliver the complete M3 skeleton in one sprint: `GridSpec`, device-resident `State`, halo abstraction, dummy 1000-step timestep loop with **zero** host/device transfers post-init, transfer audit, spacetime budget, ADR-002 state layout decision. **This is the first sprint where real model-shape code lands.** The code is the foundation every subsequent milestone builds on; sloppy here = sloppy forever. Per user directive of 2026-05-19: "elegant efficient core code, not a single waste in spacetime complexity is acceptable, every variable creation needs to be justified, piece of art in efficiency like WRF is for CPU."

**Backend: JAX + XLA** per ADR-001 (ACCEPTED 2026-05-19). Pin: `jax[cuda13]==0.10.0`.

## Non-Goals

- **No dycore math.** No advection, no pressure gradient, no acoustic step. The dummy loop is `state := state + 0` (or equivalent fused no-op) — the point is the *plumbing*, not the physics.
- **No real physics.** M5 territory.
- **No multi-GPU.** But the halo interface must NOT preclude future multi-GPU MPI exchange (single-node single-GPU implementation is a no-op halo, but the abstraction lives at the right shape).
- **No mixed precision.** fp64 everywhere per `PRECISION_POLICY.md` default.
- **No I/O / restart.** M7 territory.
- **No Pallas / Triton drop-down.** Pure `jax.jit` + `jax.numpy` + `jax.lax`.

## File Ownership

Worker may create or edit only these paths:

### Contracts (core API surface — design once, change rarely)
- `src/gpuwrf/contracts/__init__.py` (new if missing)
- `src/gpuwrf/contracts/grid.py` (new — `GridSpec` pytree)
- `src/gpuwrf/contracts/state.py` (new — device-resident `State` pytree)
- `src/gpuwrf/contracts/halo.py` (new — `HaloSpec` + `apply_halo`)
- `src/gpuwrf/contracts/precision.py` (new — DType registry, defaults to fp64, per-field overridable)

### Timestep machinery
- `src/gpuwrf/timestep/__init__.py` (new if missing)
- `src/gpuwrf/timestep/dummy_loop.py` (new — `jax.lax.scan`-based 1000-step loop, no allocation in body)

### Profiling + audit
- `src/gpuwrf/profiling/__init__.py` (new if missing)
- `src/gpuwrf/profiling/transfer_audit.py` (new — wraps `jax.profiler.trace` or CUPTI to count H2D/D2H bytes)
- `src/gpuwrf/profiling/budget.py` (new — emits the `spacetime_budget.json`)
- `scripts/m3_run_audits.py` (new — single-command CLI: builds State, runs dummy loop, emits transfer_audit.json + spacetime_budget.json)

### Artifacts
- `artifacts/m3/transfer_audit.json` (new)
- `artifacts/m3/spacetime_budget.json` (new)
- `artifacts/m3/hlo_dump/dummy_loop.txt` (new — XLA's HLO for the jit'd 1000-step scan, manager-readable evidence)
- `artifacts/m3/maintainability.md` (new ≤300 words — per-module justification)
- `artifacts/m3/agent_success.json` (new)

### ADR-002 (manager-finalized; worker drafts core technical sections)
- `.agent/decisions/ADR-002-state-layout.md` (new — worker drafts; manager edits before reviewer dispatch)

### Tests
- `tests/test_m3_grid.py` (new)
- `tests/test_m3_state.py` (new)
- `tests/test_m3_halo.py` (new)
- `tests/test_m3_dummy_loop.py` (new)
- `tests/test_m3_transfer_audit.py` (new)

Any change outside this list requires manager approval. **Do not touch `src/gpuwrf/validation/`, `src/gpuwrf/fixtures/`, `src/gpuwrf/backends/`, or governance files.**

## Inputs

- ADR-001 (`.agent/decisions/ADR-001-backend-selection.md`) — backend is JAX, pin `jax[cuda13]==0.10.0`.
- `INTERFACE_CONTRACTS.md` — `GridSpec` and `State` placeholders (this sprint replaces them with the real implementation).
- `PROJECT_PLAN.md` §7 (M3 stricter gates) — proposed tightenings that this sprint implements.
- `.agent/milestones/ROADMAP.md` M3 — full proof-object list.
- `.agent/goals/M3-DONE.md` — the binding oracle.
- `PERFORMANCE_TARGETS.md` — profile JSON schema + hard rules ("no host/device transfer in timestep loops").
- `PROJECT_PLAN.md` §11.6 — IC/BC source = AIFS (referenced in GridSpec `bc:` field but no actual AIFS ingestion in this sprint).
- Project memory `project_target_hardware.md` (RTX 5090, CUDA 13.1, JAX 0.10.0).
- Project memory `feedback_code_quality_bar.md` (**MANDATORY reading — this contract's ACs derive from it**).
- M2 JAX implementation (`src/gpuwrf/backends/jax/`) as a reference for `@jit` + pytree patterns.

## Acceptance Criteria

All must hold for closeout. **Numbered for reviewer traceability.**

### 1. GridSpec (`src/gpuwrf/contracts/grid.py`)
1.1. `GridSpec` is a `flax.struct.dataclass` or `jax.tree_util.register_pytree_node_class` so it works as a JAX pytree.
1.2. Fields:
- `projection: Projection` (enum + parameters: `kind: Literal["lambert","mercator","polar"]`, `lat_0`, `lon_0`, `dx_m`, `dy_m`, `nx: int`, `ny: int`). Frozen.
- `terrain: TerrainProvenance` (`source_path: str`, `sha256: str`, `shape: tuple[int,int]`, `units: str`, `projection_transform: str`, `max_elevation_m: float`, `coastline_sanity_check_passed: bool`). Frozen metadata; actual terrain data field is `terrain_height: jnp.ndarray` shape (ny, nx) fp64.
- `vertical: VerticalCoord` (`kind: Literal["hybrid_eta"]`, `nz: int`, `top_pressure_pa: float`, `eta_levels: jnp.ndarray` shape (nz+1,) fp64).
- `halo_width: int` (default 2, range 1–4).
- `staggering: Literal["c-grid"]` (Arakawa C grid is the M3 default per WRF compatibility).
- `bc: BCMetadata` (`source: Literal["AIFS","GFS","ERA5","ideal"]`, `fields: tuple[str,...]`, `update_cadence_h: int`, `interpolation: Literal["linear","cubic"]`, `restart_compatible: bool`).
1.3. `GridSpec.canary_3km_template() -> GridSpec` constructor for the Canary 3 km operational target, lambert projection, AIFS BC default.
1.4. `GridSpec` is hashable for `@jit` static_argnames use.

### 2. State (`src/gpuwrf/contracts/state.py`)
2.1. `State` is a JAX pytree (`flax.struct.dataclass`), fully device-resident.
2.2. Prognostic fields (allocated once at init):
- `u: jnp.ndarray` shape `(nz, ny, nx+1)` fp64 (u-staggered)
- `v: jnp.ndarray` shape `(nz, ny+1, nx)` fp64 (v-staggered)
- `w: jnp.ndarray` shape `(nz+1, ny, nx)` fp64 (w-staggered)
- `theta: jnp.ndarray` shape `(nz, ny, nx)` fp64 (mass)
- `qv: jnp.ndarray` shape `(nz, ny, nx)` fp64 (mass)
- `p: jnp.ndarray` shape `(nz, ny, nx)` fp64 (mass; full pressure)
- `ph: jnp.ndarray` shape `(nz+1, ny, nx)` fp64 (geopotential, w-staggered)
- `mu: jnp.ndarray` shape `(ny, nx)` fp64 (column mass, mass-staggered)
2.3. Tendency buffers pre-allocated as separate pytree `Tendencies` with same shapes — never reallocated inside the loop.
2.4. `State.zeros(grid: GridSpec) -> State` constructor — allocates everything on `jax.devices()[0]` (must be `'gpu'`); raises if no GPU.
2.5. `State.bytes() -> int` returns total persistent bytes (used by spacetime_budget).
2.6. `State` and `Tendencies` are `jax.jit`-compatible carry types for `jax.lax.scan`.

### 3. Halo (`src/gpuwrf/contracts/halo.py`)
3.1. `HaloSpec(width: int, fields_to_exchange: tuple[str,...], edge_type: Literal["periodic","open","nest_boundary"])`.
3.2. `apply_halo(state: State, halo: HaloSpec) -> State` — for single-GPU returns `state` unchanged (no-op); for future multi-GPU MPI the call site is identical.
3.3. The function signature MUST be `apply_halo(state, halo) -> state` so a future multi-GPU implementation can be a drop-in replacement without changing any caller.

### 4. Dummy timestep loop (`src/gpuwrf/timestep/dummy_loop.py`)
4.1. `dummy_step(state: State, tendencies: Tendencies, dt: float) -> tuple[State, Tendencies]` — a pure-functional identity-ish step that exercises the State carry without doing real physics. May do `state = state.replace(theta=state.theta + 0.0 * tendencies.theta)` to force the pytree through XLA without producing a constant-fold-everything situation.
4.2. `run_dummy_loop(state, tendencies, dt, n_steps) -> tuple[State, Tendencies]` — uses `jax.lax.scan` (NOT `jax.lax.fori_loop`, NOT Python `for`) so XLA can fuse. Whole loop is **one** `@jax.jit` call.
4.3. **Inside the scanned body, ZERO `jnp.array(...)` / `jnp.zeros(...)` / `jnp.empty(...)` allocations.** All operations must be in-place from the carry's perspective (pytree replace).
4.4. After warmup, `wall_time_per_step_us < 100` on a small `(nz=10, ny=8, nx=8)` configuration. This is a sanity bound, not a performance claim.

### 5. Transfer audit (`src/gpuwrf/profiling/transfer_audit.py` + `scripts/m3_run_audits.py`)
5.1. Uses `jax.profiler.trace(...)` or CUPTI (worker picks one, justifies in maintainability.md) to count H2D + D2H bytes during the 1000-step loop, *excluding* init and final result copy.
5.2. Emits `artifacts/m3/transfer_audit.json` with: `host_to_device_bytes_post_init`, `device_to_host_bytes_post_init`, `iterations`, `method`, `jax_version`, `gpu_name`. **`*_post_init` MUST be 0** — otherwise M3 fails per `PERFORMANCE_TARGETS.md` hard rule + M3-DONE.md AC.
5.3. HLO dump: `artifacts/m3/hlo_dump/dummy_loop.txt` — output of `jax.jit(run_dummy_loop).lower(...).compile().as_text()`. Manager-readable text evidence that the loop is fused.

### 6. Spacetime budget (`src/gpuwrf/profiling/budget.py`)
6.1. Emits `artifacts/m3/spacetime_budget.json` with all 6 required keys (per check_m3_done.py):
- `state_bytes` (sum of State pytree leaf sizes)
- `tendency_bytes` (sum of Tendencies pytree leaf sizes)
- `temporary_bytes_per_step` (must be 0; if non-zero, worker must trace why and inline-justify)
- `total_persistent_bytes` (= state_bytes + tendency_bytes + halo_buffer_bytes)
- `kernel_launches_per_step` (extracted from HLO; target ≤ 5 for the dummy step, ≤ 1 ideal)
- `wall_time_per_step_us` (median of 100 runs after warmup)

### 7. ADR-002 (`/.agent/decisions/ADR-002-state-layout.md`)
7.1. ≥1500 bytes; required tokens: `Decision:`, `Layout:`, `Staggering:`, `Halo packing:`.
7.2. Worker drafts the technical body: chosen layout (AoS vs SoA), staggering convention (C-grid), per-field precision (fp64 mass-conservation pathway), halo packing strategy.
7.3. Manager finalizes + dispatches Codex critical-review per ADR-001 precedent. If critical-review issues blockers/majors, address inline (this sprint) or open M3-S2 to handle them.

### 8. Elegance & efficiency (per `feedback_code_quality_bar.md`)
8.1. Worker-report MUST include the spacetime budget table inline (same numbers as the JSON, with one-line justification per entry).
8.2. Every `jnp.array(...)` / `jnp.zeros(...)` / `jnp.empty(...)` allocation in the codebase MUST be traceable to either (a) an input fixture, (b) a frozen `State` / `Tendencies` field, or (c) a documented diagnostic. Worker-report lists each allocation with its source.
8.3. **Zero allocations in `@jit` hot-path code** (the scanned body). The dummy step is identity-ish; production steps will be similar.
8.4. Every helper function has a one-line docstring justifying its existence vs inlining ("reused by N call-sites" or "encapsulates X invariant").
8.5. Reviewer MUST attest in their report: "I read every line of `src/gpuwrf/timestep/dummy_loop.py` and `src/gpuwrf/contracts/state.py` and found {N} simplification opportunities" where N may be 0 but is explicitly claimed.

### 9. Cross-AI tester (Claude Opus xhigh) tasking
9.1. Tester contract (auto-generated by `dispatch_role.sh`) extended for this sprint: tester MUST act as both correctness reviewer AND aesthetics+efficiency reviewer. Tester-report MUST include a section titled "Allocation Audit" listing every allocation in the diff and grading each as "necessary / could be eliminated / suspect."
9.2. Tester runs the dummy loop independently from a clean shell and reproduces the transfer_audit.json + spacetime_budget.json. If the manager-provided allocation count differs from tester's count, that is a reviewer-blocker.

### 10. Tests
10.1. `tests/test_m3_grid.py` — GridSpec is a valid pytree (round-trips through `jax.tree.flatten/unflatten`), Canary 3km template constructs, BC source is one of the enum values.
10.2. `tests/test_m3_state.py` — `State.zeros(grid)` allocates on GPU (`device.platform == 'gpu'`), all fields have correct shapes + dtype, `State.bytes()` matches manual sum.
10.3. `tests/test_m3_halo.py` — `apply_halo` returns identical state for single-GPU no-op; signature compatible with future multi-GPU drop-in.
10.4. `tests/test_m3_dummy_loop.py` — 1000-step loop runs without error, returns a State of identical shape/dtype, single `@jit` call (HLO contains one fusion).
10.5. `tests/test_m3_transfer_audit.py` — transfer_audit.json has `host_to_device_bytes_post_init == 0`, `device_to_host_bytes_post_init == 0`, `iterations >= 1000`.
10.6. `pytest -q` passes overall; suite size ≥ 250 tests after additions.

### 11. Hygiene
11.1. `validate_agentos.py` ok.
11.2. `check_m1_done.py` + `check_m2_done.py` still ok (no regression).
11.3. No file >100 KB committed beyond pre-existing.
11.4. New deps go in sprint venv `data/scratch/m3-venv/` only; `pyproject.toml` gains `flax>=0.7` if used (justify in maintainability.md).

## Validation Commands

```bash
bash scripts/m2_run_jax.sh                         # confirm jax still works (no regression)
python -m gpuwrf.contracts.state --self-test       # State allocates on GPU
python scripts/m3_run_audits.py                    # idempotent: runs dummy loop + writes JSONs
python -m json.tool artifacts/m3/transfer_audit.json
python -m json.tool artifacts/m3/spacetime_budget.json
head -50 artifacts/m3/hlo_dump/dummy_loop.txt
pytest -q
python scripts/check_m1_done.py
python scripts/check_m2_done.py
python scripts/check_m3_done.py
git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5
```

## Performance Metrics

Captured in `artifacts/m3/spacetime_budget.json`. Hard bounds:
- `host_to_device_bytes_post_init == 0` (constitutional — see PERFORMANCE_TARGETS.md hard rules).
- `device_to_host_bytes_post_init == 0`.
- `kernel_launches_per_step <= 5` (single launch ideal; small constant acceptable).
- `temporary_bytes_per_step == 0`.

Soft bounds for sanity:
- `wall_time_per_step_us < 100` on `(nz=10, ny=8, nx=8)`.
- `total_persistent_bytes < 1 GB` on the same config.

## Proof Object

- Diff (File Ownership only).
- 5 artifacts in `artifacts/m3/`.
- ADR-002 + Codex critical-review.
- Lifecycle reports including the Spacetime Budget table + Allocation Audit + reviewer's per-line attestation.

## Risks

- **`jax.lax.scan` may add carry-overhead** (one allocation per iteration for the carry). Worker uses `unroll=N` if it helps, but documents the choice; the AC of "0 temporary bytes per step" must hold.
- **JAX's transfer-audit instrumentation is asymmetric**. CUPTI is more authoritative but requires the same perfmon permission as `ncu` (which is blocked on this workstation per project memory). Worker may need to fall back to `jax.profiler.trace` event accounting; tester verifies the counts are real.
- **`flax.struct.dataclass`** may add a tiny per-step pytree-flatten cost. Worker uses `jax.jit` with `static_argnums` to amortize.
- **HLO dump file size**. The 1000-step scan compiled to HLO might be large. If >100 KB, commit only the first 1000 lines + a comment pointing at the full file in `data/scratch/`.
- **fp64 on RTX 5090** is 1:64 throughput vs fp32 (consumer Blackwell). Worker should NOT optimize for fp64 throughput now — that's M4 precision-policy work. The dummy loop just needs to run; performance is not claimed at M3.

## Handoff Requirements

- Worker pushes to branch `worker/gpt/m3-state-grid-halo-skeleton`.
- After reviewer Accept, manager writes closeout + memory-patch (new entry summarizing the M3 API surface for future post-compaction Claude turns), merges to main, pushes, opens M3 milestone closeout per runbook §D.
- Tester is **Claude Opus 4.7 xhigh** — explicitly tasked with the Allocation Audit per AC #9.

## Manager-during-worker hygiene reminder

No manager commits while worker is in flight. The codex worker shares the working tree; my commits land on its branch. Stay disciplined.
