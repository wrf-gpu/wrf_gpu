# Sprint Contract — F6: First-12-Step Transaction Audit (the cheap direct debug)

**Sprint ID**: `2026-05-28-f6-first-12-step-transaction-audit`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/f6-first-12-step-audit`
**Worktree**: `/tmp/wrf_gpu2_f6`
**Wall-time**: 1-2 days
**GPU usage**: YES (essential)
**Sandbox**: TRY `--sandbox danger-full-access` first; if classifier blocks, fall back to workspace-write and report blocked

## Why this sprint

F4 critique Q4: "We are missing the direct first-principles check for the actual observed failure: M11.3 restored the three suspected fixes and made the run fail earlier than baseline. The signal is at step 11/12. F1/F2 do not directly answer why step 11/12 fails."

The cheapest test that directly localizes the M11.3 blow-up is a per-substep
transaction audit over the first 12 steps with physics/boundary/guards
selectively disabled. Goal: pinpoint which JAX operator first violates which
algebraic invariant.

## Binding goal (universal)

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72h RMSE on T2/U10/V10
**statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins
on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs (read in order)

1. `.agent/sprints/2026-05-28-f3-agy-architecture-followup/findings.md` — Opus arch review (which JAX functions are suspect)
2. `.agent/sprints/2026-05-28-f4-gpt55-plan-grounding/critique.md` — Q4 spec for this audit
3. `proofs/m11p3/diagnostic_report_after_fix.json` — current failure pattern (step 11/12)
4. `proofs/m11p3/limiter_diagnostics_24h.json` — limiter saturation evidence
5. `src/gpuwrf/runtime/operational_mode.py` — `_rk_scan_step`, `_physics_boundary_step`, `_limit_guarded_dynamics_state`
6. `src/gpuwrf/dynamics/core/acoustic.py` — `acoustic_substep_core`, `_decouple_theta_after_advance`, `_diagnose_pressure`
7. `src/gpuwrf/dynamics/mu_t_advance.py`
8. `src/gpuwrf/diagnostics/comprehensive_harness.py` — existing harness can be extended

## Approach

### Phase 1 — Build the audit driver (0.5 day)

Add `scripts/f6_transaction_audit.py` that:
- Runs the M11.3 path for **first 12 steps only** with selectable toggles
- Toggles available: `physics_off`, `boundary_off`, `guards_off` (independently)
- Dumps per-RK-stage AND per-acoustic-substep budget rows to JSON for each step
- Budget row contains: `mu`, `muts`, `muave`, `theta`, `theta_1`, `p`, `ph`, `w`, `u`, `v`, `theta_tend`, `mu_tend` — stats: mean, min, max, abs-max, finite-count
- Re-uses comprehensive_harness wherever possible (do not duplicate)

### Phase 2 — Run the 4 toggle combinations (0.5 day)

a. `physics_off + boundary_off + guards_off`: pure dycore. If THIS fails at step 11/12, the bug is dycore-internal.
b. `physics_on + boundary_off + guards_off`: dycore + physics. If a-passes and b-fails, physics coupling.
c. `physics_off + boundary_on + guards_off`: dycore + BC. If a-passes and c-fails, BC.
d. `physics_off + boundary_off + guards_on`: dycore + limiter. If a-passes and d-fails, limiter.

Emit `proofs/f6/audit_combination_{a,b,c,d}.json`.

### Phase 3 — Algebraic invariants (0.5 day)

For each step + each substep, check algebraic invariants (NOT WRF comparison):
- All values finite
- Dry mass nonnegative: `mu_perturbation + mut >= 0` everywhere
- Pressure bounded: `|p_perturbation| < some_threshold * pb`
- Theta mass residual: `theta_mass_after - theta_mass_before` within 1e-10 of explicit theta tendency
- `muts = mut + work_mu` consistency
- RK2/RK3 starts from intended saved state: e.g. `theta_1 == theta_at_RK1_start`

Emit `proofs/f6/invariant_violations.json` — list first-violation step + operator + invariant + which toggle combination.

### Phase 4 — Three cheap regression tests (per F4 Q1)

Add to `tests/unit/`:
- `test_rk_scan_step_advection_active.py` — initialize nonzero velocity + theta gradient, run one step, assert advected fields change
- `test_mu_persistence_two_substeps.py` — nonzero `mu_perturbation` + zero tendencies, run two acoustic substeps, assert mu_save preserved across substeps
- `test_decouple_theta_state_reference.py` — analytic check that `_decouple_theta_after_advance` uses `theta_1` (not `theta`) — direct equation check

These are deterministic unit tests; they should fail under the OLD code (M11.2 baseline) and PASS under M11.3+.

### Phase 5 — Honest report

`proofs/f6/audit_summary.md` answering:
- Where does the blow-up actually start? (operator + invariant + which toggle reproduces it)
- Does it match F3 Opus's diagnosis (missing advance_uv, cross-stage mu_save loss, _diagnose_pressure stub)?
- Does it match agy's diagnosis or contradict it?
- What is the most targeted next-fix scope?

## Acceptance

- **AC1**: `scripts/f6_transaction_audit.py` works for the 4 toggle combinations.
- **AC2**: `proofs/f6/audit_combination_{a,b,c,d}.json` produced for all 4.
- **AC3**: `proofs/f6/invariant_violations.json` lists first violation per combination.
- **AC4**: 3 unit tests added under `tests/unit/`, all passing on current HEAD or with explicit `xfail` documentation if M11.3 hasn't reached them.
- **AC5**: `proofs/f6/audit_summary.md` answers the 4 honest report questions.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES — try `--sandbox danger-full-access` first.
3. **Files writable**: `scripts/f6_transaction_audit.py`, `tests/unit/test_rk_scan_step_advection_active.py`, `tests/unit/test_mu_persistence_two_substeps.py`, `tests/unit/test_decouple_theta_state_reference.py`, `proofs/f6/**`, `.agent/sprints/2026-05-28-f6-.../**`.
4. **Files NOT writable**: any dycore source code (this is read-only audit), governance, plan, ADRs.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Auto-notify on exit**: `tmux send-keys -t 0:0 "AGENT REPORT: f6 DONE exit=$?" Enter`.
8. **End with verdict**: `F6_COMPLETE` if AC1-AC5 pass; `F6_PARTIAL` with explicit gaps.

## If GPU sandbox blocks

If `--sandbox danger-full-access` is denied by the classifier:
1. Try without the flag — see if codex can ask for permission interactively.
2. If still blocked, run the audit on CPU JAX (`JAX_PLATFORMS=cpu`). Document that the GPU run is pending. The audit is still valuable on CPU — it would reveal the same algebraic invariant violations.
