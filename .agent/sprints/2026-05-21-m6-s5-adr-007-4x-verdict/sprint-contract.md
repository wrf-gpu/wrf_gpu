# Sprint Contract — M6-S5 ADR-007 4× Verdict + Dycore Cap Lift

**Sprint ID**: `2026-05-21-m6-s5-adr-007-4x-verdict`
**Created**: 2026-05-21 17:20
**Status**: ACTIVE — dispatching now (final M6 validation swarm)
**Trigger**: M6-S4 Opus accept + binding F-min-3 (dycore cap must lift); inherited M6-S2 R-15 (CPU denominator choice) + R-20 (end-to-end wall vs audit-extrapolated)

## Objective

Produce the binding ADR-007 4× speedup verdict for M6 close. Three load-bearing prereqs:

1. **Lift the 1s dycore cap** (`driver.py:757 dycore_dt_s = min(dt_s, 1.0)`). Either (a) stabilise M4 dycore to integrate 60s coupled step (likely needs 3:1 acoustic substep ratio with CFL diagnostics) OR (b) drop coupled `dt_s` to 6-10s (WRF-like for 3km). Document choice + stability evidence.
2. **End-to-end wall measurement** (not per-kernel audit extrapolated). Use observed 24h forecast wall (worker reports `output_run_wall_s=302.77s` from M6-S2; recompute with dycore cap lifted).
3. **Bind one CPU denominator**: grid-points-attributed (`3012.25s` from M6-S4 v2 manifest) OR raw-timing-subtraction (`4859.53s` from M6-S2a). Document rationale.

## Acceptance

- **AC1 Dycore cap lifted**: `dycore_dt_s = min(dt_s, 1.0)` removed or replaced with proper CFL-stable integration. Stability evidence: 24h forecast finite (sanitize_state firing rate documented).
- **AC2 End-to-end wall measured**: real cold-start + compile + 24h coupled forecast + output wall time recorded. NOT audit-step extrapolation.
- **AC3 CPU denominator binding**: explicit choice with documented rationale; respect FP32/FP64 evidence from M6-S2a (-r4 caveat).
- **AC4 ADR-007 verdict artifact**: `artifacts/m6/performance/full_domain_batching_verdict.json` with PASS/FAIL decision, GPU wall, CPU wall, speedup ratio, profiler raw paths, transfer audit, op/kernel/HLO size/temp peak/compile retries/cache size/allocator fragmentation per M7 critic §5.
- **AC5 Profiler raw artifacts**: nsys/ncu (or JAX profiler trace) raw paths persisted.
- **AC6 Verdict decision**: PASS if speedup ≥4× AND M6-S4 Tier-2 invariants hold under lifted-cap forecast. FAIL with documented root cause if not.
- **AC7 ADR-007 amendment**: update §Status with verdict outcome + binding evidence path.

## Files Worker May Modify

- `src/gpuwrf/coupling/driver.py` (dycore cap lift + CFL guard or coupled dt reduction)
- `src/gpuwrf/dynamics/{step.py, rk3.py, acoustic.py}` (if stabilization required — ADR amendment required for dycore changes; alternatively just lower coupled dt)
- `src/gpuwrf/profiling/budget.py` (extend for end-to-end wall + op-count + retrace count)
- `scripts/m6_full_domain_batching.py` (NEW — 24h cold-start + warm-run wall benchmarks)
- `tests/test_m6_dycore_cap_lift.py`, `test_m6_4x_verdict.py` (NEW)
- `.agent/decisions/ADR-007-precision-policy.md` (Status amendment)
- `artifacts/m6/performance/full_domain_batching_verdict.json` (NEW)
- Worker report

## HARD RULES

1. NO `min(raw, cap)` budget fudge
2. NO synthesized speedup (must be real measured wall)
3. End-to-end (cold-start + compile + 24h forecast + output) wall is the binding numerator
4. CPU denominator binding choice MUST cite the M6-S2a v2 manifest + `-r4` precision caveat
5. File-disjoint with M6-S6 (validation/tier3_coupled) + M6-S7 (validation/tier4_probtest)
6. `/exit` slash-command; watchdog + multi-Enter

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (mandatory)
- Wall: **12-20h**
- Worktree: `/tmp/wrf_gpu2_m6s5`
- Branch: `worker/codex/m6-s5-adr-007-4x-verdict`

## End-goal context

M6-S5 produces the binding answer to "does GPU-resident WRF-compatible NWP run ≥4× faster than 28-rank CPU WRF on the same workstation?" If PASS, M7 dispatch authorized; if FAIL, project scope decision required.
