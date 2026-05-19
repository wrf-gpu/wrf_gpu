# Goal Condition — M3 Done

Used by the self-paced `/loop` manager to detect M3 completion. End state is **objective and machine-checkable**. M3 is done when `python scripts/check_m3_done.py` returns `{"ok": true}`.

## Single-command status check

```bash
python scripts/check_m3_done.py
```

## Explicit assertions (binding)

### A. Repository hygiene
- `python scripts/validate_agentos.py` ok.
- `pytest -q` all pass.
- M1 + M2 oracles still ok (no regression).

### B. Sprint completeness
For every `dir = .agent/sprints/2026-*-m3-*/`: `python scripts/close_sprint.py <dir>` ok.

### C. M3 deliverables (proof objects)

1. `src/gpuwrf/contracts/grid.py` — `GridSpec` dataclass / pytree with named, machine-readable fields:
   - `projection: {Lambert, Mercator, polar}` enum + parameters (lat_0, lon_0, dx, dy, nx, ny)
   - `terrain: {source_path, sha256, shape, units, projection_transform, max_elevation_m, coastline_sanity_check_passed}` provenance
   - `vertical: {scheme=hybrid_eta, nz, top_pressure_pa, eta_levels}` metadata
   - `halo_width: int`
   - `staggering: {mass, u, v, w}` indicator
   - `bc: {source=AIFS, fields=[u,v,T,qv,p_s], update_cadence_h, interpolation=linear, restart_compatible}` metadata
2. `src/gpuwrf/contracts/state.py` — device-resident `State` (JAX pytree):
   - Allocated once at init (no allocation in hot path)
   - Fields: prognostics (u, v, w, T, qv, p, ph) + diagnostics (selectable) + tendency buffers (pre-allocated)
   - `State.from_init(grid: GridSpec, ic: Path) -> State` constructor
   - All fields are `jnp.ndarray` on the GPU (verified by `device.platform == 'gpu'`)
3. `src/gpuwrf/contracts/halo.py` — halo abstraction stub (implementation deferred to M3.x or M4):
   - `HaloSpec` with width, fields-to-exchange, edge type (periodic / open / nest-boundary)
   - `apply_halo(state, halo_spec) -> state` no-op for single-GPU; the **interface** must support multi-GPU MPI exchange without future refactor
4. `src/gpuwrf/timestep/dummy_loop.py` — pure-functional 1000-step dummy timestep loop using `jax.lax.scan`:
   - One `@jax.jit` call wrapping the whole loop
   - No `jnp.array(...)` allocations inside the scanned body
   - State is carried as the scan carry; only the carry is allocated
5. `artifacts/m3/transfer_audit.json` — produced by running the dummy loop with CUPTI or `jax.profiler` tracing:
   - `host_to_device_bytes_post_init: 0`
   - `device_to_host_bytes_post_init: 0`
   - `iterations: 1000`
6. `artifacts/m3/spacetime_budget.json` — manager-mandated budget table:
   - `state_bytes`, `tendency_bytes`, `temporary_bytes_per_step`, `total_persistent_bytes`
   - `flops_per_cell_per_step` (estimated)
   - `kernel_launches_per_step`
   - `wall_time_per_step_us` (median of 100 runs after warmup)
7. `.agent/decisions/ADR-002-state-layout.md` — selects: AoS vs SoA for prognostics, the staggering convention, the storage layout (C-order vs F-order for each variable class), the halo packing strategy, the precision per field (fp64 mass-conservation pathway). Includes Codex cross-model critical-review.

### D. Cross-AI provenance
Each M3 sprint's `tester-report.md` produced by **Claude Opus 4.7 xhigh** (`ai=claude` in completion log).

### E. Bounds
- Per-role retry cap: 5.
- M3 wall-time cap: 48 h.
- Spend cap: ~150 agent calls.

## Escalation triggers

- A worker cannot achieve `host_to_device_bytes_post_init: 0` for the dummy loop → write `BLOCKER-m3-transfer-audit.md`. This would mean JAX/XLA has a hidden transfer pathology on cc120; needs research-scout sprint.
- ADR-002 cross-model review deadlocks → escalate to user.
- M2 (JAX) regresses on M1 fixtures → halt; would mean JAX install drifted.
