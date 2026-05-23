# Sprint Contract — M6.x S2: Current ADR-023 1h d02 Replay Baseline (Instrumented)

## Objective

Critic HYBRID plan, Sprint S2. Run the **current unified ADR-023 path** through a **1h Gen2 d02 boundary replay**, instrumented with the diagnostic sidecars from S1. Goal: produce an honest diagnosis of where the current operator drifts, what bounds it violates, and how the sanitizer/limiter are masking instability. NO operator-code changes — we are measuring the unchanged baseline first.

This is the cleanest possible "where do we stand" measurement. The result drives S3's targeted fix (which mu-limiter replacement, which magic number, which boundary handling).

## Non-Goals

- NO edits to `src/gpuwrf/dynamics/`, `src/gpuwrf/contracts/`, `src/gpuwrf/physics/`, or any production code. Strictly READ-ONLY on the operator.
- No new sidecars (S1 built them; S2 USES them).
- No fix attempts — even tempting small fixes. Report findings; S3 acts on them.
- No 24h or 6h forecast — 1h only.
- No remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_s2_baseline` on branch `worker/gpt/m6x-s2-d02-baseline-instrumented`.

Write-only:
- `scripts/m6_d02_baseline_run_instrumented.py` (new — orchestration that runs the d02 1h replay with all S1 sidecars active)
- `.agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/` — proofs + worker-report

Read-only everywhere else. Especially `src/gpuwrf/`.

## Inputs

Required reading:
- **`.agent/sprints/2026-05-23-m6x-close-strategy-plan-critic/reviewer-report.md`** §2 sprint S2 + §6 — critic's spec for this sprint
- `.agent/decisions/source_mining_operator_table.md` (built in S1) — reference for what each operator term should be doing canonically
- `scripts/m6_d02_boundary_replay_1h.py` (from earlier halted sprint, preserved on main) — existing 1h replay scaffold
- `src/gpuwrf/integration/d02_replay.py` — d02 replay core (read-only)
- `scripts/diagnostic_*.py` (the 12 from S1) — your instrumentation toolkit
- `tests/test_m6x_d02_boundary_replay.py` (from earlier sprint, preserved) — existing smoke test
- `.agent/references/cpu-wrf-baseline.md` — Gen2 d02 location for the comparison reference
- `data/fixtures/gen2_baseline/rmse_summary.csv` — Gen2 noise floor anchors (T2 24h 0.628 K, U10 24h 1.46 m/s, V10 24h 1.59 m/s)

## Acceptance Criteria

### Part A: Instrumented 1h d02 baseline run

`scripts/m6_d02_baseline_run_instrumented.py` orchestrates:
1. Run `scripts/m6_d02_boundary_replay_1h.py` (or equivalent) against Gen2 d02 IC + boundary, 1h forecast, current ADR-023 unified operator unchanged
2. Capture pre-sanitize candidate state at every substep (NOT just post-sanitize)
3. Pipe replay outputs through these S1 sidecars and capture their JSON outputs:
   - `diagnostic_bound_violation_tracer.py` — first violation per field
   - `diagnostic_sanitizer_audit.py` — per-step clip/changed counts
   - `diagnostic_limiter_activation_tracker.py` — mu_continuity_increment saturation fraction
   - `diagnostic_field_rmse_timeline.py` — T2/U10/V10/w/theta vs Gen2 at hourly leads (Gen2 only has hourly truth; sub-hourly are diagnostic-only)
   - `diagnostic_spatial_divergence_map.py` — where errors grow
   - `diagnostic_boundary_ring_error_profiler.py` — boundary vs interior split
   - `diagnostic_vertical_column_phase_space.py` — picked columns (1 boundary-zone, 1 over Mount Teide, 1 ocean)
   - `diagnostic_operator_term_budget_tracer.py` — which RHS term dominates per substep
   - `diagnostic_transfer_launch_timeline.py` — kernel launches + transfer audit
4. Aggregate into a single `proof_s2_baseline_summary.json` for manager consumption

### Part B: Findings classification

Write `findings.md` in the sprint folder classifying each diagnostic finding as:
- `EXPECTED-BAD` — known pre-existing issue (e.g., mu blowup at saturation)
- `NEW-FINDING-NEEDS-S3-FIX` — surfaced by this sprint, will inform S3 sprint design
- `OK-WITHIN-NOISE` — within Gen2 baseline noise floor (cite `rmse_summary.csv` values)
- `BLOCKER` — completely nonfinite / sanitizer cannot keep run going (would stop S3 dispatch)

### Part C: S3 input memo

Write `s3_input_memo.md` in the sprint folder with:
- The 3 highest-priority operator concerns from the source-mining table that this baseline run confirms need fixing (with proof JSON cites)
- For each: recommended source-cited fix (referencing `source_mining_operator_table.md`)
- For each: expected effect on the baseline numbers (which Gen2 RMSE delta should improve)
- The exit-rule status: would S3 + 1 fix sprint plausibly produce a Tier-3 PASS, or is BLOCKER memo warranted?

### Part D: No regression

`pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m3_transfer_audit.py -v` — all PASS (45+ tests).

### Part E: Worker report

`worker-report.md` summary with:
- 1h forecast verdict: did it complete, did sanitizer mask anything
- Top-5 numerical findings with proof cites
- Files changed (just the orchestration script + sprint folder)
- Commands run, exit codes
- Risks
- Handoff to S3

## Validation Commands

```bash
cd /tmp/wrf_gpu2_s2_baseline
python scripts/m6_d02_baseline_run_instrumented.py \
  --duration-s 3600 \
  --output-dir .agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/ \
  | tee .agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_baseline_run.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_path_unification.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_warm_bubble_operator_sanity.py tests/test_m6x_s1_diagnostic_sidecars.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented/proof_no_regression.txt
```

## Performance Metrics

- Wall time for 1h forecast: report informational
- Peak GPU memory: report informational
- Transfer audit: must be 0 H2D/D2H bytes in the timestep loop (binding)

## Proof Object

- `proof_baseline_run.txt` — main orchestration log
- `proof_s2_baseline_summary.json` — aggregated diagnostic findings
- Per-sidecar JSON outputs (named `proof_<sidecar-name>.json`)
- `proof_no_regression.txt`
- `findings.md` — finding classification
- `s3_input_memo.md` — recommendations for next sprint
- `worker-report.md`

Time budget: **6-10 hours**.

## Risks

- **Gen2 d02 IC/BC unavailable in the test environment**: if the data isn't accessible, fall back to a synthetic Gen2-shaped fixture and document loudly. The sanity diagnostics still produce useful info on synthetic IC.
- **Sanitizer masking masks itself**: the audit sidecar must capture PRE-sanitize candidate state. If the replay scaffold doesn't expose that, document the gap — that's a finding.
- **1h forecast blows up before completion**: if so, that's data — capture the first-nonfinite step + the diagnostic state right before, write the BLOCKER finding, halt cleanly. Don't add stabilization to "fix" it.
- **Scope creep into operator fixing**: resist any temptation to "just fix this one thing". S3 is the fix sprint.
- **Spec-gaming**: every numerical finding cites the JSON proof path + the sidecar that produced it.

## Handoff Requirements

When all proof files are on disk, `findings.md` and `s3_input_memo.md` are committed, no-regression passes, worker-report on disk: type `/exit` as a slash command. Wrapper sends `AGENT REPORT [worker / m6x-s2-d02-baseline-instrumented / codex] exit=<ec>` to manager pane.

## Failure modes the manager will reject

- Modifying any file under `src/gpuwrf/`.
- "Fixing" the baseline operator instead of measuring it.
- Skipping the sanitizer-masking audit (the critic explicitly flagged this as the most important hidden risk).
- Sub-hourly Gen2 RMSE claims with no truth file (sub-hourly is diagnostic-only).
- s3_input_memo without explicit source-mining table cross-references.
