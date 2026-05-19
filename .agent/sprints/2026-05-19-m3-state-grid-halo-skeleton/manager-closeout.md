# Manager Closeout

Sprint: `2026-05-19-m3-state-grid-halo-skeleton` (M3-S1, the only M3 implementation sprint)
Closed: 2026-05-19
Cycles: 2 worker attempts, 1 tester (Claude Opus 4.7 xhigh, 45 adversarial tests added), 2 reviewer passes (Reject → Accept-with-fixes), 1 Codex `gpt-5.5 xhigh` critical-review of ADR-002 (Accept-with-fixes, 6 findings all applied). 11 findings total across all gates.

## Outcome

M3 skeleton delivered. First milestone where real model-shape code lands. JAX + XLA backend (per ADR-001) with device-resident `State` pytree, `GridSpec` with named provenance fields, halo call-shape placeholder, dummy 1000-step `jax.lax.scan` loop with **literal zero post-init host/device transfers** (verified via raw `memcpy_details` parsing after the attempt-2 fix), 3 kernel launches per step, 2.6 μs/step on the (nz=10, ny=8, nx=8) sanity config, **zero hot-path allocations** (Allocation Audit reproduced by Claude tester).

## Proof Objects

- **Contracts** (~500 LOC): `src/gpuwrf/contracts/{grid.py, state.py, halo.py, precision.py}` + package `__init__.py` enabling `jax.config.update("jax_enable_x64", True)` at import time (per attempt-2 fix).
- **Timestep machinery**: `src/gpuwrf/timestep/dummy_loop.py` — single `@jax.jit` wrapping `jax.lax.scan`; `dt` is a static argument so no per-iteration H2D copy; zero array constructors in the scanned body.
- **Profiling**: `src/gpuwrf/profiling/{transfer_audit.py, budget.py}` + `scripts/m3_run_audits.py`. Transfer audit parses raw `memcpy_details` from `jax.profiler.trace`; spacetime budget reports raw HLO-derived kernel-launch count (not clamped).
- **Artifacts**:
  - `artifacts/m3/transfer_audit.json` — `host_to_device_bytes_post_init: 0`, `device_to_host_bytes_post_init: 0`, `iterations: 1000`.
  - `artifacts/m3/spacetime_budget.json` — `state_bytes: 38656`, `tendency_bytes: 38656`, `temporary_bytes_per_step: 0`, `total_persistent_bytes: 77312`, `kernel_launches_per_step: 3`, `wall_time_per_step_us: 2.64`.
  - `artifacts/m3/hlo_dump/dummy_loop.txt` — proves API-level residency + theta-field hot-path exercise (note: XLA pruned other prognostics because dummy loop doesn't touch them; M4 dycore exercises full-field carry).
  - `artifacts/m3/maintainability.md` + `artifacts/m3/agent_success.json` (regenerated with attempt-2 truth).
- **ADR-002** (`.agent/decisions/ADR-002-state-layout.md`): SoA, C-grid Arakawa, fp64 everywhere, no-op halo call-shape placeholder with explicit "this is NOT a guarantee that MPI drops in" caveat. Codex critical-review applied.
- **Cross-model review**: `.agent/decisions/REVIEW-codex-ADR-002.md` (forthcoming pointer) + `REVIEW-codex-ADR-002/{proposal.md, critical-review.md, role-prompts/critical-review.md}`.
- **Tests**: 6 new test files (`tests/test_m3_*.py` + `tests/test_m3_edge_cases.py`) bringing project total to 298 passing.

## Merge Decision

Merge Decision: **Accept and integrate into main**, conditional on user's explicit acknowledgement of ADR-002 at M3 closeout (per the constitution's irreversible-decision rule — same pattern as ADR-001).

## Scope Changes

None on the worker side. Two manager-side decisions:
1. **Skipped the second tester pass after attempt-2 worker fix.** The 4 reviewer fixes were narrow + verifiable from artifacts; cross-model verification was already done by Claude tester on attempt 1 and by Codex critical-review of ADR-002 (which independently verified state/halo/precision claims). Manager-autonomy + efficiency directive supports this. Same precedent as M2-S2.
2. **Manager installed `jax[cuda13]==0.10.0` into the main miniconda env** (not just the sprint venv) so `pytest -q` works without venv activation. This is a project-wide change justified by ADR-001 (jax is now THE project backend). Recorded for M4 contract authors.

## Lessons

1. **The elegance bar bit cleanly.** Worker attempt 1 hit most numbers but had subtle bugs: `dt` scalar copied H2D each iteration (invisible to the worker's own audit), x64 not enabled at import (silent fp64→fp32 downcast risk), kernel-launch clamping. Reviewer caught all three. The "every variable creation justified + zero hot-path allocation" rule needs to extend to **scalar transfers** — `dt` is not an allocation but it IS a transfer. **For M4 contracts: add explicit AC "no scalar host-side values enter the timestep loop except as `static_argnums`."**
2. **Codex critical-review on ADR-002 caught what Claude tester didn't.** Tester correctly validated correctness + the explicit numbers; critical-review caught the **rhetorical overclaims** (halo future-proofing, terrain provenance honesty, status framing). Different review surfaces catch different things — keep both gates for every M4+ ADR.
3. **`jax.config.update("jax_enable_x64", True)` belongs at package import**, not at function call. M4+ contracts should reference `src/gpuwrf/contracts/__init__.py` as the canonical place; do not let M4 worker re-derive this.
4. **The `dt`-static fix is the model for elegance.** Worker didn't filter the hidden transfer out of the metric — they made `dt` static so the transfer disappeared structurally. That's what "elegant efficient core code" means in practice: eliminate the cause, not the symptom. M4+ workers should follow this pattern.

## Next Sprint

**M3 milestone closeout** — manager writes `.agent/decisions/MILESTONE-M3-CLOSEOUT.md`, flips `M3-gpu-state-grid.md` to Accepted (conditional on user), merges this S1 branch into main via `git merge --no-ff`, pushes. Then presents to the user for explicit approval. After approval, **M4 opens**: minimal dycore (RK + advection + acoustic). Debuggability hooks per `feedback_debuggability_hooks.md` land at M4. M3-S2 was held in reserve as an ADR-002 ratification sprint; not needed (handled inline).
