# Sprint Contract — M6b Operational-vs-Validation Composition Bisection

## Objective

M6b RETRY (commit `worker/gpt/m6b-honest-1h-canary-RETRY`) returned BLOCKER again: operational mode with carry-expanded scratch families passes 10s probe but **theta grows to 1343 K at step 62 (~10 min sim time)**. Same composition path in validation mode (B6 = `coupled_step.py`) was 0.0 bitwise on 10 timesteps.

This is the failure-critic's Hypothesis B (localized composition bug in operational), now empirically confirmed. **Operational-mode composition diverges from validation-mode composition at some specific step/operator/RK-stage.** This sprint runs a structured bisection to localize the bug.

## Non-Goals

- NO modifications to validation-mode code (the bitwise baseline must stay locked).
- NO modifications to operational `wrf.exe`.
- NO speculative fixes — only localize the divergence. Fix is a follow-up sprint.
- NO sanitizer / clamps / stabilizers added to operational mode.
- NO new physics or solver.
- NO new operator semantics.
- NO 1h forecast in this sprint (the bisection is short).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_opbisect` on branch `worker/gpt/m6b-operational-composition-bisection`.

Write-only:
- `scripts/m6b_operational_vs_validation_compare.py` (NEW) — runs BOTH operational_mode + validation `coupled_step` on the SAME real Gen2 d02 IC for N timesteps; per-step per-field max-abs delta
- `src/gpuwrf/runtime/operational_mode.py` — **read-only with diagnostics only**. Allowed: one optional `debug=False` static-arg to enable per-step snapshot emission (DCE-eliminated in production). NO runtime semantic changes.
- `tests/test_m6b_operational_vs_validation_compare.py` (NEW)
- `.agent/sprints/2026-05-25-m6b-operational-composition-bisection/` — proofs + worker-report including the bisection table

Read-only everywhere else.

## Inputs (mandatory)

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-honest-1h-canary-RETRY/worker-report.md` (the BLOCKER details — theta growth at step 62)
3. `.agent/sprints/2026-05-25-m6b-fix-carry-expansion/worker-report.md` (the operational carry expansion + inline scratch copy/adapt; **suspect of the bug**)
4. `.agent/sprints/2026-05-25-m6b6-coupled-step-parity/worker-report.md` (the validation 0.0 bitwise baseline)
5. `src/gpuwrf/runtime/operational_mode.py` + `src/gpuwrf/runtime/operational_state.py`
6. `src/gpuwrf/dynamics/coupled_step.py` + `acoustic_loop.py` + `dycore_step.py` + `small_step_scratch.py` (validation-only — the correct reference)
7. `.agent/sprints/2026-05-25-m6b-failure-critic/reviewer-report.md` (§4 said bisection if Hypothesis A insufficient — now activates)

## Acceptance Criteria

### Stage 1 — Side-by-side runner (MANDATORY)

`scripts/m6b_operational_vs_validation_compare.py`:
- Load the SAME real Gen2 d02 IC (e.g., 20260523, the one that failed at step 62)
- Build TWO state objects: operational (with promoted scratch) + validation
- Run BOTH for N timesteps (start with N=10, max N=70 to cover the step-62 failure point)
- At each timestep, compute per-field max-abs delta between operational and validation
- Capture the FIRST step where any field's delta exceeds 1e-10 (the "divergence step")
- For that step, drill down: per-RK-stage delta, per-acoustic-substep delta, per-operator delta

Capture proof: `proof_bisection_step_level.json`.

### Stage 2 — Sub-step bisection at the divergence step (MANDATORY)

At the divergence step identified in Stage 1:
- Within the RK loop, find the diverging RK stage
- Within the acoustic loop of that RK stage, find the diverging substep
- Within that substep, find the diverging operator: `calc_coef_w`, `advance_uv`, `advance_mu_t`, `tridiag_fwd`, `tridiag_back`, scratch updates, etc.
- Per-field delta at each granularity level

Capture proof: `proof_bisection_substep_level.json`.

### Stage 3 — Field+operator localization (MANDATORY)

The divergence localizes to ONE of:
- A specific scratch field (`t_2ave`/`ww`/`muave`/`muts`/`ph_tend`/`_save`) has wrong update timing or formula
- A specific operator (`calc_coef_w` or other) is called with wrong inputs in operational
- A specific RK stage interleaving is wrong (e.g., physics tendency applied at wrong time)

Name the offending element + cite the WRF source line that validation gets right.

Outcome: `OPERATIONAL-COMPOSITION-DEFECT-LOCALIZED-AT-<field/operator/stage>`.

### Stage 4 — No fix attempt (MANDATORY)

Do NOT fix the bug in this sprint. Document the defect + a 1-paragraph "recommended minimal fix" + cite WRF source for the correct behavior. The fix is a follow-up M6b-operational-composition-fix sprint.

### Stage 5 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py tests/test_m6b_honest_v2_*.py tests/test_m6b_operational_vs_validation_*.py -v
```

### Stage 6 — Worker report

`worker-report.md`:
- Stage 1 step-level bisection table (which step diverges)
- Stage 2 sub-step bisection (which RK / acoustic / operator)
- Stage 3 named defect + WRF source citation + recommended fix paragraph
- Files changed (just the comparator + tests; operational_mode.py debug arg only)
- Outcome verdict

## Validation Commands

```bash
cd /tmp/wrf_gpu2_opbisect
taskset -c 0-3 python scripts/m6b_operational_vs_validation_compare.py --gen2-run-id 20260523_18z_l3_24h_20260524T004313Z --steps 70 2>&1 | tee .agent/sprints/2026-05-25-m6b-operational-composition-bisection/proof_bisection_run.txt
pytest <full test list> -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-operational-composition-bisection/proof_no_regression.txt
```

## Performance Metrics

N/A — diagnostic sprint.

## Kill Gates

- Bisection cannot find any diverging step within 70 timesteps → escalate (the failure may not be reproducible at smaller scale; or operational mode IS already correct and M6b RETRY had an instrumentation bug). Escalate to manager.
- Operational sha changes → STOP.
- Validation-mode code modified → REJECT.

## Risks

- Operational state and validation state may have DIFFERENT field shapes/layouts; comparator must normalize them on a common grid before comparing. Use the validation-mode shapes as canonical.
- The bisection runner may compile slowly the first time due to building both paths in one process. Acceptable; document the warm-up cost.

## Handoff Requirements

When defect localized + worker-report committed: `/exit`. Manager dispatches the named fix sprint (`m6b-operational-composition-fix-<localized-defect>`).

Time budget: **45-90 min**.
