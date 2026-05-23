# Sprint Contract — M6.x Warm-Bubble Failure Diagnostic (Opus 4.7)

## Objective

The unification sprint (commit `e2391d3`) successfully unified the public nonhydrostatic scan path through the MPAS conservative recurrence — but the honest warm-bubble test shows `w_max` at 600s = **0.041 m/s** (target [5, 10] m/s; prototype with simplified `_wrf_buoyancy_column_update` hit 8.52). 23× too small.

Two interpretations:
- **(A) Architectural gap (ADR-021 fallback)**: the conservative MPAS-recurrence with the 6-leaf `AcousticScanCarry` genuinely lacks the WRF small-step scratch fields (`t_2ave`, `ww`, `muave`, `ph_tend`, `_1`/`_save` families) needed to drive coupled warm-bubble dynamics. Architecture step-back's 3rd pivot criterion fires for real.
- **(B) Wiring bug**: a specific coupling between the warm-bubble θ-perturbation initialization and the MPAS-recurrence's buoyancy term is mis-threaded. Slice oracle works because slice initial conditions are consistent with the operator's expected layout; coupled warm-bubble harness may inject θ' through a channel the recurrence doesn't read.

This sprint is **diagnostic, not implementation**. Opus 4.7 (different model than prior codex sprints — fresh angle) reads the unified code, the warm-bubble harness, the slice oracle, and the MPAS source, and returns a verdict between (A), (B), or (C) something else. The verdict drives the next dispatch.

## Non-Goals

- No new code or refactoring. **READ-ONLY** except for adding diagnostic instrumentation in a single file (e.g., a `scripts/diagnostic_warm_bubble_vs_slice.py`) that runs comparison probes.
- No ADR amendments.
- No dispatch of other agents.
- No re-running the unification sprint or production-grade sprint.
- No claim to "fix" anything — only to diagnose.

## File Ownership

Write-only to this sprint folder + ONE new diagnostic script under `scripts/`:
- `scripts/diagnostic_warm_bubble_vs_slice.py` (new — instrumentation only, no production-code changes)
- `.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md` (the deliverable)
- `.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/probe_*.json` (instrumentation captures)

Read-only everywhere else, especially: all `src/gpuwrf/dynamics/` files, all oracle files, all test files.

## Inputs

Required reading (in this order):
1. `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/worker-report.md` — what the unification sprint did + measured w_max numbers per epssm
2. `src/gpuwrf/dynamics/acoustic_wrf.py` — current unified state. Focus on:
   - `_mpas_recurrence_vertical_update` (whatever name the worker chose)
   - The unified `vertical_acoustic_update` dispatcher
   - `_mu_continuity_increment` (the temp stabilizer)
   - How `pressure_scale=-1.0` and `pressure_scale=0.0` both reach the recurrence
3. `src/gpuwrf/dynamics/vertical_implicit_solver.py` — Thomas solver
4. `scripts/m6_warm_bubble_test.py` — how the warm-bubble harness initializes + advances the system. Critical: HOW is the warm-bubble θ-perturbation injected into the State leaves?
5. `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py` — how the slice initializes its θ-perturbation. Critical: WHY does this produce a peak w_max=1.327 m/s while the warm-bubble harness's unified path produces 0.041?
6. `tests/test_m6x_vertical_acoustic_oracle.py` — R7 oracle (passes on unified path)
7. `tests/test_m6x_adr023_production_grade.py` — production-gate (passes on unified path, slice RMSE 1.69%)
8. `.agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/reviewer-report.md` — the reject that prompted unification
9. `.agent/decisions/ADR-023-conservative-column-solver.md` — current state (PROPOSED, with §"Fallback trigger")
10. WRF source `module_small_step_em.F:1340-1597` — `advance_w` canonical buoyancy term
11. MPAS source `mpas_atm_time_integration.F:2146-2208` — buoyancy + recurrence body

## Diagnostic instrumentation (the one allowed code addition)

Create `scripts/diagnostic_warm_bubble_vs_slice.py`. Pure-Python, no production-code changes. It should:

1. Run both setups with identical perturbation amplitude:
   - the MPAS slice oracle with a warm-bubble theta perturbation (peak Δθ matching the warm-bubble harness)
   - the warm-bubble harness via `scripts/m6_warm_bubble_test.py` machinery (or equivalent)
2. At each substep capture and log to `probe_warm_bubble_vs_slice.json`:
   - The Δθ field shape and peak value at each layer
   - The buoyancy term `g·θ'/θ_base` (or whatever the recurrence's effective buoyancy is) at each layer
   - The implicit-coupling RHS for the w equation
   - The w_new produced
3. If they diverge, log the first substep where divergence is significant and the source variable
4. Also probe: with `epssm=0.0` (centered), does warm-bubble pass? With `pressure_scale=0.0` (forces the original ADR-023 spec branch — same kernel post-unification), does warm-bubble pass?

## Acceptance Criteria

Produce `.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md` containing:

1. **§1 Reproduction.** Independently rerun the unification's warm-bubble failure. Confirm w_max ≈ 0.04 m/s at 600s on unified path. Capture output.

2. **§2 Comparison: slice vs warm-bubble harness.** What is structurally different about how the slice oracle's column trajectory generates w_max≈1.3 m/s vs the warm-bubble harness producing 0.04? Cite file:line for both initialization paths.

3. **§3 Buoyancy-term trace.** With the diagnostic script, trace the buoyancy coupling: does the warm-bubble θ-perturbation actually arrive at the recurrence's buoyancy RHS? Cite the actual probe output. If yes, what's the magnitude? Is it ~30× too small (consistent with a missing factor), or zero (consistent with a missing channel)?

4. **§4 Sign convention check.** Is the buoyancy sign in `_mpas_recurrence_vertical_update` consistent with what `scripts/m6_warm_bubble_test.py` expects? Warm parcel (θ' > 0) should produce positive vertical acceleration (rising). A sign flip would explain the deadness on a warm-bubble while still passing analytic oracles (which test linear-mode propagation, not bubble lifting).

5. **§5 `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38` audit.** The unification worker introduced this constant. Is `0.38` derived from a MPAS source line, or is it a magic number? If derived, cite the line. If a magic number, can it be 1.0 (un-scaled)? Run the diagnostic with the scale at 1.0 and report w_max.

6. **§6 `_mu_continuity_increment` audit.** The tanh CFL limiter on mu remains as a "temp validation stabilizer." Without it, the unified warm-bubble run goes nonfinite at step 2. Does this suggest the mu coupling has a sign error, an off-by-one in indexing, or genuinely needs the missing WRF small-step scratch?

7. **§7 Verdict** (exactly one):
   - `WIRING-BUG-WITH-FIX-PROPOSAL` — Specific file:line + proposed fix. Manager dispatches the fix sprint instead of ADR-021. Estimate worker hours for fix.
   - `SIGN-ERROR-WITH-FIX-PROPOSAL` — Same shape as above but for a sign convention issue.
   - `ARCHITECTURAL-GAP-FALLBACK-TO-ADR-021` — Genuine architectural deficiency that requires carry expansion or fundamentally different operator structure. Manager proceeds with ADR-021 prototype (already dispatched in parallel as Plan B).
   - `MAGIC-NUMBER-ADJUSTMENT` — `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE` or another scalar is mis-derived; specific value change closes the gap. Manager dispatches a small fix sprint.
   - `MIXED` — partial findings of (B) wiring + (A) gap; manager picks the next move.

8. **§8 Confidence.** Strength of evidence for the verdict. "High" / "Medium" / "Low" with one paragraph of justification.

9. **§9 Open questions for the next sprint** (if applicable).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_diag
python scripts/diagnostic_warm_bubble_vs_slice.py --output .agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/probe_warm_bubble_vs_slice.json
# Plus targeted single-shot runs as the diagnostic dictates
```

## Performance Metrics

N/A — diagnostic sprint.

## Proof Object

- `.agent/sprints/.../diagnostic-report.md` (2000-5000 words)
- `.agent/sprints/.../probe_*.json` (instrumentation output)
- `scripts/diagnostic_warm_bubble_vs_slice.py` (new)

Time budget: **2-4 hours**.

## Risks

- **Bias toward "small fix"**: Opus may want to find a wiring bug to avoid recommending ADR-021. Counter: explicitly require §3 buoyancy-term magnitude trace — if the buoyancy is right but w_max is still 30× too small, that's evidence for architectural gap, not wiring.
- **Bias toward "fundamental gap"**: opposite bias. Counter: §5 explicit audit of `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE`. Magic-number debugs are common in port work.
- **Diagnostic script becoming production code**: keep it under `scripts/`, not under `src/gpuwrf/`.

## Handoff Requirements

When `diagnostic-report.md` and probe JSON are on disk, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [tester / m6x-warm-bubble-failure-diagnostic / opus] exit=<ec>`.
