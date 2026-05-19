# Milestone M3 Closeout — GPU State & Grid Skeleton

Date: 2026-05-19
Status: **CLOSED on manager side; M4 dispatch gated on explicit user approval of ADR-002.**

## Summary

M3 delivers the first real model-shape code: device-resident `State` pytree, `GridSpec` with named provenance fields, halo call-shape placeholder, and a 1000-step `jax.lax.scan` dummy timestep loop with **literal zero post-init host/device transfers** (verified by raw `memcpy_details` parsing). ADR-002 selects SoA + Arakawa C-grid + fp64. M4 is unblocked architecturally.

## Closed Sprints

| ID | Outcome | Cycles |
|---|---|---|
| `2026-05-19-m3-state-grid-halo-skeleton` (S1) | Accept attempt 2 + ADR-002 critical-review applied | 2 worker + 1 tester (Claude xhigh, 45 adversarial tests) + 2 reviewer (Reject → Accept-with-fixes) + 1 Codex critical-review on ADR-002 (6 fixes applied) |

S2 was held in reserve (ADR-002 ratification); not needed — handled inline in S1.

## Proof Objects (all on `main` after this closeout)

- **Contracts**: `src/gpuwrf/contracts/{grid.py, state.py, halo.py, precision.py}` + `__init__.py` (enables `jax_enable_x64` at import).
- **Timestep**: `src/gpuwrf/timestep/dummy_loop.py` — single `@jax.jit`, `dt` static, zero hot-path allocations, `jax.lax.scan`-based 1000 steps.
- **Profiling**: `src/gpuwrf/profiling/{transfer_audit.py, budget.py}`, `scripts/m3_run_audits.py`.
- **Artifacts**:
  - `artifacts/m3/transfer_audit.json` — `host_to_device_bytes_post_init: 0`, `device_to_host_bytes_post_init: 0`, `iterations: 1000`. **Constitutional hard rule satisfied.**
  - `artifacts/m3/spacetime_budget.json` — state=38656 B, tendency=38656 B, total persistent=77312 B, `temporary_bytes_per_step: 0`, `kernel_launches_per_step: 3` (raw HLO count), `wall_time_per_step_us: 2.64`.
  - `artifacts/m3/hlo_dump/dummy_loop.txt` — API-level residency + theta hot-path exercise (full-field carry exercised at M4).
- **ADR-002** (`.agent/decisions/ADR-002-state-layout.md`): SoA, C-grid, fp64, halo call-shape placeholder.
- **Codex cross-model review**: `.agent/decisions/REVIEW-codex-ADR-002.md` + folder.
- **Tests**: 298 passing (up from 250 pre-M3); 6 new test files specifically for M3 + cross-AI tester's 45 adversarial additions + worker's 4-fix tests in attempt 2.
- **Memory entries** for future post-compaction continuity: `feedback_debuggability_hooks.md` (M4+ binding) + `project_state_layout.md` (ADR-002 summary + critical M4 practices).

## Numbers worth remembering

- **Zero** post-init host/device transfers — literal, raw-trace-verified, after the `dt`-static elegance fix.
- **3** kernel launches per step (raw HLO count, not clamped).
- **0** temporary bytes per step (no allocations in the scanned body).
- **2.64 μs** per step on the (nz=10, ny=8, nx=8) sanity config.

These are the architectural baseline against which M4 dycore + M5 physics will be measured. Regressing any of them is a flag.

## Residual Risks

- **`canary_3km_template()` uses idealized terrain** (zeros, `sha256="analytic-m3-template"`). Acceptable for M3/M4/M5/M6 plumbing; **real Canary terrain provenance required at M7**. M7 contract must include a "replace template with real .nc + real sha256 + real coastline sanity check" AC.
- **Halo `apply_halo()` is single-GPU no-op only.** The function signature is frozen, but a dedicated halo ADR (probably M3.x or M4 early) is required before any multi-GPU exchange semantics are relied on. Single-GPU work through M7 doesn't need it; M7+ multi-GPU does.
- **HLO evidence shows theta hot-path** only; XLA pruned other prognostics because the dummy loop doesn't touch them. M4 dycore will exercise full-field carry — that's where the real "no spill" claim across all 8 prognostics gets validated.
- **NVIDIA performance counters still blocked** (ERR_NVGPUCTRPERM). Transfer audit uses `jax.profiler.trace` event accounting; this is sufficient for M3-M4 architectural claims but the M3-closeout follow-up from ADR-001 (get `nvidia-driver-perfmon-allow=1` set) still stands before any M4+ absolute-performance claim.
- **fp64 throughput on RTX 5090 Blackwell is 1:64 vs fp32.** Not an M3 problem (we don't make performance claims). Becomes a real M4 question — ADR-003 (M4 dycore precision) will propose validated per-field downcasts.

## Top Lessons

1. **Elegance bar held.** Worker attempt 2 produced 0 hot-path allocations + 0 post-init transfers + 3 launches/step + 2.6 μs/step + every variable creation justified. The `dt`-static fix (eliminating the cause, not filtering the symptom) is the model for M4+ work.
2. **Cross-model gates pay off again.** Tester (Claude Opus) caught environment debt + did Allocation Audit. Binding reviewer caught 4 technical bugs (x64, __eq__, audit parsing, kernel-launch clamping) that the worker didn't see. Codex ADR-002 critical-review caught 6 rhetorical/audit overclaims. Different gates catch different things — keep all of them for M4+.
3. **Manager-during-worker hygiene held this time.** No contamination commits — manager waited for agent reports before touching the working tree.
4. **`/loop` autonomous mode works.** 16+ manager turns across M3 with full sprint lifecycle, no deadlocks, no need for user intervention except at the constitutional approval gate.

## Required Next Action — Explicit User Approval

Per the constitution (`PROJECT_CONSTITUTION.md:16`) and `.agent/rules/architecture-decision-policy.md:13`, **ADR-002 cannot be treated as locked until the user explicitly approves it.** Same pattern as ADR-001. M4 implementation is dispatch-blocked until user reply.

## Recommended Next Milestone

**M4 — Minimal Dycore** opens immediately on user approval:
- Reduced RK + advection + acoustic step in JAX (per ADR-001 + ADR-002)
- **Debuggability hooks land here** per `feedback_debuggability_hooks.md` — every hot-path `@jit` gets a `debug: bool = False` static argument; XLA dead-code-eliminates the debug branch in production; snapshot + assert-finite + assert-physical-bounds at every dycore stage.
- Tier 1 + Tier 2 + Tier 3 validation per `VALIDATION_STRATEGY.md` on at least one idealized case (em_hill2d_x or equivalent).
- ADR-003 dycore precision (worker drafts, manager finalizes, Codex critical-reviews).
- M4 stop/go gate against ADR-001's binding thresholds (`local_memory_bytes ≤ 256, registers ≤ 128, kernel_launches ≤ 10`) on the first real physics scheme proxy.

Estimated wall time: ~3-5 manager turns + agent runs.
