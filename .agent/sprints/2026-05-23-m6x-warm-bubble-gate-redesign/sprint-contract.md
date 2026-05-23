# Sprint Contract — M6.x Warm-Bubble Gate Redesign (Stage 1 of critic's recommendation)

## Objective

Per `m6x-warm-bubble-gate-strategy-critic` (verdict `CHANGE-THE-GATE`, commit `c80b622`), the [5, 10] m/s warm-bubble amplitude target is **not sourced** for our pure-small-step Gaussian harness. Both "passing" implementations achieved their pass via unphysical clamps (ADR-021's `w_next = 9.0 * tanh(max(w_next, 0.0) / 9.0)` clamp to exactly 9.0; ADR-023 prototype's `NONHYDROSTATIC_BUOYANCY_SCALE` + drag + mu gating). The actual M6 close gate per MILESTONES.md + VALIDATION_STRATEGY.md + ADR-007 is Tier-3 convergence + initial Tier-4 RMSE — not warm-bubble amplitude.

This sprint implements **Stage 1** of the critic's two-stage recommendation: convert the warm-bubble harness from an amplitude-pass gate into an **operator-sanity gate**.

## Non-Goals

- No new physics or stabilization scheme.
- No modification of the analytic R7 oracle, MPAS slice oracle, or their tests.
- No modification of c2-A2 horizontal PGF or `mu_continuity_tendency`.
- No removal of the `_mu_continuity_increment` tanh limiter in this sprint (it's a separate operator concern — keep it documented as a temp stabilizer; this sprint just makes the gate honest about whether it's saturating).
- No d02 or 24h forecast.
- No remote push.
- No carry expansion or Newton outer.
- No ADR-021 work — that prototype stays on its branch (unmerged) as evidence; this sprint operates on the ADR-023 unified path that's on main.
- No host/device transfer regression.

## File Ownership

Write-only on this sprint's branch `worker/gpt/m6x-warm-bubble-gate-redesign`:

- `scripts/m6_warm_bubble_test.py` — REWRITE the verdict logic + reporting. Keep the integration loop and grid setup. Change the verdict from amplitude-band check to operator-sanity check.
- `tests/test_m6x_warm_bubble_operator_sanity.py` (new) — pytest tests asserting the new verdict semantics + anti-clamp static checks.
- `.agent/decisions/ADR-024-warm-bubble-gate-policy.md` (new) — record the gate policy change as a project-level ADR. Reference the critic verdict + Opus diagnostic.
- `.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/` — proofs + worker-report.

Read-only everywhere else, especially `src/gpuwrf/`.

## Inputs

Required reading:
- **`.agent/sprints/2026-05-23-m6x-warm-bubble-gate-strategy-critic/reviewer-report.md`** — your spec, §5 recommendation + §6 cost estimate + anti-tautology gates
- **`.agent/sprints/2026-05-23-m6x-warm-bubble-failure-diagnostic/diagnostic-report.md`** — Opus's evidence on what the current verdict hides (max|w|, theta/p extrema, mu limiter saturation)
- `.agent/decisions/ADR-023-conservative-column-solver.md` — current architecture state
- `scripts/m6_warm_bubble_test.py` — current harness (you're rewriting the verdict half)
- `src/gpuwrf/dynamics/acoustic_wrf.py` — current unified state for static-anti-clamp scanning
- `MILESTONES.md` § M6 — the actual binding gates
- `VALIDATION_STRATEGY.md` — the 4-tier pyramid
- `ADR-007` precision-policy — operational RMSE binds
- `tests/test_m6x_vertical_acoustic_oracle.py` — R7 oracle (must remain green as a precondition of the new gate)

## Acceptance Criteria

### 1. New harness verdict structure

`scripts/m6_warm_bubble_test.py` outputs a JSON proof object with these mandatory keys per the critic's §5 spec:
- `verdict`: one of `PASS_OPERATOR_SANITY` / `FAIL_FINITENESS` / `FAIL_PHYSICAL_BOUNDS` / `FAIL_ANTI_CLAMP_DETECTION`
- `first_nonfinite_step`: int or null
- `samples`: per-checkpoint (300s, 600s, plus optional intermediate samples) with:
  - `w_max_m_s` (signed positive peak)
  - `w_min_m_s` (signed negative peak)
  - `w_abs_max_m_s` (max |w|)
  - `theta_perturbation_max_K`
  - `theta_perturbation_min_K`
  - `p_perturbation_max_Pa`
  - `p_perturbation_min_Pa`
  - `mu_perturbation_max_Pa`
  - `centroid_z_m`
  - `mass_residual_kg_per_m2` or `mu_residual_Pa` (column-integrated mass change vs t=0)
- `bound_violations`: list of bound-violation entries (empty list = no violations); each entry has `field`, `step`, `value`, `bound`, `tolerance`
- `anti_clamp_warnings`: list of static-scan warnings (e.g., found `tanh(.../9.0)` style clamp on production path)

### 2. Operator-sanity verdict logic

`verdict` is:
- `FAIL_FINITENESS` if `first_nonfinite_step is not None`
- `FAIL_PHYSICAL_BOUNDS` if any of: `theta_perturbation_max > 50 K`, `theta_perturbation_min < -50 K`, `p_perturbation_max > 50000 Pa`, `p_perturbation_min < -50000 Pa`, `mu_perturbation_max > 50000 Pa` (these bounds are conservative; tighten in follow-up sprint if needed). Bound violations populated.
- `FAIL_ANTI_CLAMP_DETECTION` if the static scan finds a target-shaped clamp on the production path (see AC3)
- `PASS_OPERATOR_SANITY` otherwise

**The amplitude band [5, 10] is NOT a gate.** `w_max_m_s` is reported, but does NOT bind pass/fail.

### 3. Anti-clamp static scan

Add a static-scan helper that scans the production code path (`src/gpuwrf/dynamics/acoustic_wrf.py`, `src/gpuwrf/dynamics/vertical_implicit_solver.py`) for:
- Constants tied to `[5, 10]` (e.g., `9.0`, `10.0`, `5.0` appearing as bounds)
- Patterns `tanh(... / X)` where X is in `[5, 10]`
- `jnp.maximum(..., 0.0)` on `w` (positive-only velocity)
- `jnp.minimum(..., theta_target)` style theta clipping
- Explicit `lift_bias` or `updraft_drag` named constants
- Magic numbers `0.38`, `1.35` that are present in current code — these may already exist in the unified path (MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE, MPAS_OMEGA_TO_W_METRIC); the scan should WARN (not fail) on these, since they're documented in ADR-023 as inherited from the slice oracle.

The scan returns warnings, not hard failures, unless it finds patterns specifically matching the [5, 10] amplitude band — those are hard fails.

### 4. R7 oracle + hydrostatic-rest pre-requisite check

The harness must run the R7 oracle tests and a hydrostatic-rest invariance test as preconditions of returning any verdict. If either fails, the verdict is `FAIL_FINITENESS` (the conservative path that has a broken R7 has no business reporting operator-sanity).

### 5. ADR-024 policy

Write `.agent/decisions/ADR-024-warm-bubble-gate-policy.md` documenting:
- The gate change rationale (cite the critic verdict, Opus diagnostic §9.2, the published WRF em_squall2d_x evidence)
- The new verdict semantics
- The anti-tautology gates (must come from external reference; must fail on clamps; etc.)
- The two-stage path: Stage 1 (this sprint) + Stage 2 (build sourced WRF/CM1/MPAS reference if amplitude gate is later desired)
- Status: PROPOSED. (Manager will ratify after reviewer concurs.)

### 6. New pytest gate

`tests/test_m6x_warm_bubble_operator_sanity.py` with at least four tests:
- `test_warm_bubble_runs_finite_on_unified_path` — current path runs the harness to 600s without NaN
- `test_warm_bubble_extrema_reported_correctly` — verdict JSON contains all mandatory keys with finite values
- `test_anti_clamp_scan_detects_known_patterns_in_test_fixtures` — feed the scanner a test fixture containing a known clamp pattern; assert it's detected
- `test_r7_oracle_is_prerequisite_for_pass` — assert that the harness verdict is `FAIL_FINITENESS` if R7 oracle test would fail

### 7. Re-run on current main

Run the redesigned harness on the current main (commit at start of this sprint) and capture the resulting verdict JSON to `proof_current_state_verdict.json`. The expected outcome on the post-wiring-fix unified ADR-023 path:
- `verdict = FAIL_PHYSICAL_BOUNDS` (mu_perturbation max likely > 50 kPa per Opus probe data) OR `PASS_OPERATOR_SANITY` (if the wiring fix reduced theta blowup enough that the bounds hold)
- Both outcomes are valid evidence for the next sprint

### 8. No-regression on existing tests

All prior tests still PASS. ≥ 29 tests pass (27 from wiring-fix sprint's regression + 2 from wiring-fix's gate + new operator-sanity tests, minus the 4 production-grade tests that may be affected if they depended on the amplitude verdict).

If the new gate makes prior tests fail because they depended on the amplitude verdict, REPORT in the worker report — don't paper over.

### 9. Worker report

`worker-report.md` with: summary, the gate change rationale (echoing critic + Opus), the new JSON verdict structure, the static-scan rules, ADR-024 highlights, re-run verdict on current main, files changed, commands, proof objects, risks, handoff.

### 10. Branch commit

On `worker/gpt/m6x-warm-bubble-gate-redesign`. Multiple commits OK.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_gate_redesign
pytest tests/test_m6x_warm_bubble_operator_sanity.py -v | tee .agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_new_gate.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py -v | tee .agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_no_regression.txt
python scripts/m6_warm_bubble_test.py --output .agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_current_state_verdict.json | tee .agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_current_state_verdict.txt
pytest tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_transfer_audit.txt
```

## Performance Metrics

None.

## Proof Object

- `proof_new_gate.txt` — new test PASS
- `proof_no_regression.txt` — prior tests still PASS
- `proof_current_state_verdict.json` + `.txt` — re-run on current main
- `proof_transfer_audit.txt`
- `worker-report.md`
- `.agent/decisions/ADR-024-warm-bubble-gate-policy.md`
- New code on `worker/gpt/m6x-warm-bubble-gate-redesign`

Time budget: **4-8 hours**.

## Risks

- **`test_m6x_adr023_production_grade.py:test_mpas_slice_trajectory_rmse_under_production_target` depends on the prior verdict semantics**: this test may need to be left alone (it's about the slice RMSE, not the warm-bubble amplitude) — but verify it still passes.
- **Existing references to the old verdict** in worker-reports, sprint contracts, etc.: do NOT update old artifacts; their content is historical record. Only update the harness + tests + new ADR.
- **Scope creep into operator changes**: this sprint changes the GATE, not the OPERATOR. The mu_continuity_increment limiter, the magic-number constants (`0.38`, `1.35`), and the architecture remain as-is — separate concerns for follow-up sprints.
- **Spec-gaming**: be honest about the re-run verdict on current main. If it's `FAIL_PHYSICAL_BOUNDS`, report it; do NOT tune the bounds to pass.

## Handoff Requirements

When all proof files are on disk, ADR-024 is on disk, worker-report.md is committed on `worker/gpt/m6x-warm-bubble-gate-redesign`, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [worker / m6x-warm-bubble-gate-redesign / codex] exit=<ec>`.

## Failure modes the manager will reject

- Tuning the new bounds to make `current main` magically PASS.
- Modifying the R7 oracle, MPAS slice oracle, or their tests.
- Removing or weakening the `mu_continuity_increment` limiter (separate concern).
- Modifying `_mpas_recurrence_vertical_update` or the c2-A2 horizontal PGF.
- Self-marking ADR-024 ACCEPTED — that's the next reviewer's call.
- Host transfer regression.
