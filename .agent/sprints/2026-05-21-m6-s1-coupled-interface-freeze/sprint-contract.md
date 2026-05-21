# Sprint Contract — M6-S1 Coupled Interface and Precision Boundary Freeze

**Sprint ID**: `2026-05-21-m6-s1-coupled-interface-freeze`
**Created**: 2026-05-21 11:05 by manager (Claude Opus 4.7 1M-context)
**Status**: ACTIVE — first M6 implementation sprint (dispatched after M5-prologue closure)
**Trigger**: M5-S1.y closed UNBLOCKED-WITH-DEBT; M5-S2.x closed ACCEPT; M5-S3.x closed GROUNDWORK-PHASE-2 (M5-S3.y is parallel debt, not a blocker for interface freeze)
**M6 plan scout**: `worker/codex/m6-milestone-plan-scout @ 3392d04` (scout's recommended scope)

## Objective

Define the coupled state, column adapters, precision boundaries, and file ownership BEFORE forecast-loop work begins. Per plan scout's M6-S1 spec:

1. Extend ADR-002 SoA pytree (not AoS rewrite) with: hydrometeors (`qv, qc, qr, qi, qs, qg`), number concentrations (`Ni, Nr, Ns, Ng`), TKE (`qke`), surface state handles, precipitation accumulators.
2. Define precision boundaries per ADR-007 Authorization Matrix: `mu`, pressure/geopotential, pressure-gradient accumulation, acoustic accumulators stay FP64; `u/v/theta/qv` may run FP32 only under ADR-007 gates; Thompson hydrometeors + numbers + source/sink arithmetic may run FP32 under gate; `w` stays empirical-test-gated.
3. Define column adapter contracts: how Thompson microphysics, MYNN PBL, RRTMG radiation, surface stub each consume + produce state slices in the coupled driver.
4. Produce a 100-step dummy coupled carry on a small domain (e.g. 16×16×30) with zero post-init host/device transfer bytes and a machine-readable spacetime budget per `PERFORMANCE_TARGETS.md`.

## Acceptance (pre-M6-S2-coupled-driver gate)

- **AC1 — State pytree extension.** `src/gpuwrf/contracts/state.py` extended with the named hydrometeor + number + TKE + surface + precipitation leaves. SoA preserved (one JAX array per field). Type annotations + units in docstrings. Test asserting all new leaves are device-resident JAX arrays with expected dtypes.
- **AC2 — Precision boundary memo + code.** `src/gpuwrf/contracts/precision.py` extended with the Authorization Matrix per field; FP64 fields tagged; FP32-allowed fields tagged with gate flag. Test asserting each field's dtype matches Authorization Matrix.
- **AC3 — Coupling adapter contracts.** New `src/gpuwrf/coupling/` module with `physics_couplers.py` defining: `thompson_adapter(state, dt) -> state`, `mynn_adapter(state, dt) -> state`, `rrtmg_adapter(state, dt) -> state`, `surface_adapter(state, dt) -> state`. Each adapter is a type-checked wrapper around the existing physics kernel, slicing state pytree to physics input + reassembling state output. Synchronous RK3 substep timing (no async).
- **AC4 — 100-step dummy coupled carry.** `scripts/m6_run_dummy_coupled.py` runs 100 steps on a 16×16×30 domain, alternating dycore + 4 physics adapters. Output: `artifacts/m6/coupled_dummy_carry.json` with `wall_time_per_step_ms, hlo_bytes, kernel_launches_per_step, host_to_device_bytes_post_init=0, device_to_host_bytes_post_init=0, temporary_bytes_per_step=0`. Transfer audit must show ZERO.
- **AC5 — Spacetime budget table.** Machine-readable per-step budget: total wall, per-kernel wall share, per-kernel launches, per-kernel HLO bytes. JSON schema match `PERFORMANCE_TARGETS.md` budget schema.
- **AC6 — ADR (small).** New `.agent/decisions/ADR-010-coupled-state-extension.md` documenting state extension + precision boundaries + adapter contracts + 100-step proof object reference. Cross-reference ADR-002 (state layout) and ADR-007 (precision policy).
- **AC7 — File ownership freeze.** After this sprint closes, M6-S2..S8 sprints can run in parallel with file-disjointness. List the freeze in ADR-010.

## Files Worker May Modify

- `src/gpuwrf/contracts/state.py` (extend SoA pytree)
- `src/gpuwrf/contracts/precision.py` (Authorization Matrix per field)
- `src/gpuwrf/coupling/__init__.py`, `coupling/physics_couplers.py` (NEW module)
- `scripts/m6_run_dummy_coupled.py` (NEW)
- `tests/test_m6_state_extension.py`, `tests/test_m6_precision_matrix.py`, `tests/test_m6_dummy_coupled.py` (NEW)
- `.agent/decisions/ADR-010-coupled-state-extension.md` (NEW)
- `artifacts/m6/coupled_dummy_carry.json`, `artifacts/m6/spacetime_budget.json` (NEW)
- Worker report

## Files Worker Must NOT Modify

- Any physics kernel (`src/gpuwrf/physics/**`) — only WRAP, do not change physics; M5-S1.y, M5-S2.x, M5-S3.x physics is FROZEN until M5-S3.y or M6-S3 dispatch
- `src/gpuwrf/dynamics/**` — dycore is frozen; only the coupling adapter touches it
- `ADR-001, ADR-002, ADR-007` — only AMEND with cross-reference, do not REWRITE
- Any other ADR or governance file
- `.agent/rules/**`, `.agent/skills/**`

## Dispatch

- Primary worker: codex gpt-5.5 xhigh (per frontrunner role)
- Reviewer (mandatory per sprint-lifecycle hard rule): Claude Opus 4.7 xhigh
- Wall-time: 12-18 hours
- Worktree: `/tmp/wrf_gpu2_m6s1` (isolated)
- Branch: `worker/codex/m6-s1-coupled-interface-freeze`

## Hard rules

- **NO** physics-kernel modification — wrap only.
- **NO** AoS rewrite of state pytree — extend the existing SoA pytree.
- **NO** `min(raw, cap)` launch fudge.
- **NO** non-zero post-init transfer. AC4's `host_to_device_bytes_post_init = 0 and device_to_host_bytes_post_init = 0` is a HARD CHECK in the gate, not informational.
- Cite ADR-002 + ADR-007 sections for every architectural claim.
- 100-step dummy carry must use REAL physics kernels (not stubs) so the proof object is load-bearing.
