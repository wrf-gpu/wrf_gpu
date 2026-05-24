# Sprint Contract — M6.x S3-hunt: Operator Bug Hunt (per critic's exit-rule verdict)

## Objective

The exit-rule critic (commit `aad5635`) read the catastrophic S2.1-redo baseline (T2 1h RMSE 136K, 17B nonfinites/run, theta=550K post-sanitize) and recommended `DISPATCH-OPERATOR-BUG-HUNT` with frozen scope: find or disprove a specific operator bug before any architectural change or fix package. The failure is early (step 2 pre-sanitize), saturating (cap hits on all guarded fields), equation-shaped — consistent with a sign/unit/staging/missing-term bug, not necessarily an architectural inadequacy.

This sprint's job: **prove or disprove a concrete operator bug in 8-16 hours, using sanitizer-bypass diagnostics.** It is NOT a fix sprint. A fix may be implemented only if a specific bug is named with source citation AND an A/B improvement is demonstrated on a pre-sanitize diagnostic.

## Non-Goals

- NO architecture promotion (ADR-023 stays PROPOSED).
- NO new stabilizers, NO new clamps, NO target-shaped damping.
- NO acceptance based on post-sanitize finiteness.
- NO multi-suspect changes (every A/B toggles one suspect at a time).
- NO new physics scheme.
- NO 1h or 24h forecast — short replay only (1-10 steps).
- NO remote push.
- NO removing _mu_continuity_increment unless the bug-hunt specifically proves it's the first cause.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_s3hunt` on branch `worker/gpt/m6x-s3hunt-operator-bug-hunt`.

Write-only:
- `scripts/m6_d02_short_replay_sanitizer_off.py` (NEW) — short replay variant that disables `_sanitize_replay_candidate` OR aborts on first bad candidate. Used for diagnostic capture only.
- `scripts/m6_bughunt_ab_toggle.py` (NEW) — A/B harness that runs the short replay with exactly one suspect toggled
- `scripts/diagnostic_first_bad_step_tracer.py` (NEW) — per-step pre-sanitize tensor dump for steps 1-5: field, location, value, immediately preceding state. Can be added under scripts/ or as a sidecar.
- `src/gpuwrf/integration/d02_replay.py` — may add a sanitizer-bypass mode (read-only on operator; the bypass is integration-layer)
- `src/gpuwrf/dynamics/acoustic_wrf.py` — touchable ONLY if a specific bug is named with source citation. Otherwise read-only.
- `src/gpuwrf/dynamics/vertical_implicit_solver.py` — same rule (named bug + cite or read-only)
- `tests/test_m6x_s3hunt_bug_named.py` (NEW) — test that asserts the named bug fix improves the first-bad metric
- `.agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/` — proofs + worker-report

Read-only everywhere else.

## Inputs

Required reading (in this order):
- **`.agent/sprints/2026-05-24-m6x-exit-rule-critic/reviewer-report.md`** — your binding spec; especially §4 and §5 (9 grep targets)
- `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/worker-report.md` + proofs — the catastrophic baseline you're diagnosing
- `.agent/decisions/ADR-023-conservative-column-solver.md` (PROPOSED) — operator design intent
- `.agent/decisions/source_mining_operator_table.md` — what each operator term should be doing canonically
- WRF source `module_small_step_em.F` (especially `:619-651`, `:828-868`, `:902-942`, `:1094-1175`, `:1340-1597`)
- MPAS source `mpas_atm_time_integration.F` (especially `:1589-1656`, `:2146-2208`, `:2491-2495`)

## Acceptance Criteria

### Stage 1: Sanitizer-bypass short replay diagnostic (MANDATORY)

Run a short replay (1, 2, 5, 10 steps) on real Gen2 d02 IC + boundary with `_sanitize_replay_candidate` either disabled OR converted to abort-on-first-bad. Capture per-step pre-sanitize:
- First bad field (name)
- First bad cell location `(i, j, k)`
- First bad value
- Immediately preceding state (the field that produced it, prior step's value)
- Which stage of the timestep produced it: pre-vertical-recurrence / inside-recurrence / post-recurrence / mu-update / physics / boundary-application

Output: `proof_first_bad_trace.json`. If first-bad step IS step 2 as previously measured, document the EXACT term + value. If first-bad step is later than step 2 (e.g., the bypass exposes different behavior), document the discrepancy.

### Stage 2: A/B suspect toggles (MANDATORY, 4-7 toggles)

For each of these suspects, run a 10-step sanitizer-off replay with the suspect toggled (off, neutral form, or scaled), compare to baseline:

1. **MPAS recurrence sign-check** (`acoustic_wrf.py:763-827`): verify each ±sign in rhs_interior + cofwz/cofwr/cofwt usage against the cited MPAS lines.
2. **`_mu_continuity_increment`** (`:473-495`): run with `dmu = 0`, then with no-bound raw update + abort, then with WRF MUAVE-style sourced damping (if straightforward).
3. **`_mpas_w_metric_faces`** (`:512-534`): run with a fixed reference column metric vs the current per-level computation. If results differ wildly, the metric is the bug.
4. **n_acoustic sweep** (`d02_replay.py:489-520` + `acoustic_wrf.py:954-987`): vary `n_acoustic ∈ {1, 4, 8, 16}` on 10-step replay. If increasing acoustic substeps delays/removes first nonfinite, it's a cadence/stability bug.
5. **Physics disable**: run dycore-only (physics tendencies = 0) for 10 steps. If clean, the coupling between physics and dycore is at fault.
6. **Boundary disable**: skip lateral boundary application for 10 steps. If clean, the boundary application has a bug. (Note: previous evidence showed interior > boundary error, so this is expected to NOT clean things.)
7. **Branch verification** (`:686-695, :742-752`): confirm real d02 takes `pressure_scale <= 0.0` → `_mpas_recurrence_vertical_update`, NOT the analytic-slice branch. If wrong branch, that's the bug.

Output: `proof_ab_toggles.json` with per-toggle: name, change description, first-bad-step before, first-bad-step after, fields-on-cap before, fields-on-cap after, recommendation.

### Stage 3: Coefficient sanity check on one d02 column (RECOMMENDED)

Dump `build_epssm_column_coefficients` output for one real d02 column (e.g., center of grid). Compare values numerically against expected MPAS/WRF ranges. Output: `proof_column_coefficients.json`.

### Stage 4: Verdict

Write `verdict.md` with one of:
- `BUG-FOUND-AND-FIXED`: specific bug named (file:line + source citation), fix implemented, 10-step sanitize-off replay now passes acceptance bar (first nonfinite step → null, no fields on caps)
- `BUG-FOUND-NEEDS-DESIGN`: specific bug named but fix requires broad change (e.g., WRF MUAVE scratch). Document; do NOT implement. Recommend either a separate fix sprint or `M6-DYCORE-BLOCKER-MEMO`.
- `NO-BUG-LOCALIZED`: 4-7 A/B toggles all match baseline (no single suspect dominates). Architecture is the issue. Recommend `M6-DYCORE-BLOCKER-MEMO`.

### Stage 5: No regression

`pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m6x_tier3_convergence_infra.py tests/test_m3_transfer_audit.py -v` — all PASS.

### Stage 6: Worker report

`worker-report.md` with: verdict + first-bad trace + A/B table + coefficient sanity findings + files changed + commands + risks + handoff.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_s3hunt
# Stage 1: short replay with sanitizer bypass
python scripts/m6_d02_short_replay_sanitizer_off.py --steps 10 \
  --output .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_first_bad_trace.json \
  | tee .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_first_bad_log.txt
# Stage 2: A/B toggles
python scripts/m6_bughunt_ab_toggle.py \
  --output .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_ab_toggles.json \
  | tee .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_ab_log.txt
# Stage 5: no-regression
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py tests/test_m6x_s3narrow_stabilizer_audit.py tests/test_m6x_tier3_convergence_infra.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_no_regression.txt
```

## Performance Metrics

- Time per A/B toggle: ≤ 5 min wall (10-step replay should be fast post-S2.2)
- Transfer audit: 0 H2D/D2H bytes binding

## Proof Object

- `proof_first_bad_trace.json`, `proof_ab_toggles.json`, `proof_column_coefficients.json`
- `proof_no_regression.txt`
- `verdict.md`
- `worker-report.md`
- Branch `worker/gpt/m6x-s3hunt-operator-bug-hunt`

Time budget: **8-16 hours**. The critic's bar.

## Risks

- **Multi-suspect changes**: if the worker changes >1 suspect at a time, the result is uninterpretable. Hard reject.
- **Adding new stabilizers**: any new clamp/damping that wasn't already in the code = reject.
- **Post-sanitize "PASS"**: only sanitizer-off acceptance counts.
- **Spec-gaming**: every claim cites file:line + proof JSON path.
- **CPU budget**: bound to cores 0-3 via dispatch_role_session2.sh wrapper.

## Handoff Requirements

When verdict.md + all proof files on disk + no-regression passes + worker-report.md committed: `/exit`. Wrapper sends AGENT REPORT to manager pane (session 2).

## Failure modes the manager will reject

- Verdict that doesn't match one of {BUG-FOUND-AND-FIXED, BUG-FOUND-NEEDS-DESIGN, NO-BUG-LOCALIZED}.
- Acceptance based on post-sanitize finiteness.
- Multi-suspect changes claimed as "localized."
- Renaming a stabilizer to avoid the anti-clamp scanner.
- Skipping any of Stage 1, 2, or 4.
