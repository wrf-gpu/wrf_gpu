# Sprint Contract — M6 Dycore Pressure Step-Back (CODEX Critic)

## Objective

We accreted 4 fix sprints today (V suppress, vertical_implicit coftz, op-theta-fix decoupling, microphysics guards) but the boundary audit + M6 acceptance both verdict NO-GO:

- p_perturbation hits IEEE-754 overflow (1.7×10³⁰⁸ Pa) in 2-11 min on all 3 V3 ICs
- W reaches 1180 m/s on 20260521 step 75
- u escapes guard to 10¹⁵ m/s on 20260509 step 14
- T2 RMSE = 10⁸⁵ K on 2 of 3 ICs
- Only v, theta, qc are bounded — by guards, not physics
- Multi-step CPU parity 0.0 bitwise is misleading: both validation_wrappers and operational explode the same way

**You are the codex critic. Do not implement. Critically review:**

1. Have we been fixing symptoms instead of root cause?
2. Should some of today's 4 fixes be REVERTED (suspect: op-theta-fix's "decoupling formula at composition boundary" — that change re-routes `p` through `p_total` and may have unbalanced p_perturbation integration; boundary audit explicitly fingered this)?
3. Where is the single MOST LIKELY operator-level root cause of the positive p feedback?
4. Should we go back to the savepoint ladder (B0-B6) and verify it's actually validating against WRF on REAL ICs (not just synthetic fixtures)?
5. Is there a fundamental flaw in the dynamics/core/acoustic loop that the reframe sprint papered over?

## Non-Goals

- NO code changes.
- NO new sprint dispatched by you — your output is a critique memo.
- NO remote push.

## File Ownership

Worktree at `/tmp/wrf_gpu2_critic` on branch `critic/codex/m6-dycore-pressure-stepback`.
FIRST: `cd /tmp/wrf_gpu2_critic`.

Write-only:
- `.agent/sprints/2026-05-26-m6-dycore-pressure-stepback/critic-report.md` (NEW — your output)

Read-only:
- Everything else.

## Inputs (all merged on manager-2026-05-23 as of HEAD c4d5a12)

1. `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/worker-report.md` (V suppress workaround)
2. `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/worker-report.md` (vertical_implicit fix landed in wrong file)
3. `.agent/sprints/2026-05-26-m6b-operational-theta-fix/worker-report.md` (advance_mu_t_wrf theta-flux + "decoupling formula at composition boundary")
4. `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/worker-report.md` (Thompson + operational coupling guards)
5. `.agent/sprints/2026-05-26-m6-boundary-dynamics-audit/tester-report.md` (the NO-GO + suspect localization)
6. `.agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/worker-report.md` (the FAIL acceptance)
7. The recent 8 commits in `git log --oneline manager-2026-05-23 -8`
8. The source code that was touched: `src/gpuwrf/dynamics/core/acoustic.py`, `src/gpuwrf/dynamics/mu_t_advance.py`, `src/gpuwrf/dynamics/vertical_implicit_solver.py`, `src/gpuwrf/runtime/operational_mode.py`, `src/gpuwrf/physics/thompson_column.py`, `src/gpuwrf/dynamics/acoustic_wrf.py`.

## Acceptance Criteria

Write `critic-report.md` with these sections (you must include literal `Decision:` token):

### 1. Symptom-vs-root-cause assessment

For each of the 4 fixes today, classify:
- ROOT-CAUSE-CORRECT (truly fixes the underlying defect; should keep)
- SYMPTOM-MASK (papers over a symptom; should consider reverting)
- MIXED (real fix but with unwanted side-effects)

Provide a 1-paragraph rationale per fix.

### 2. Suspect localization

Identify the SINGLE most-likely operator-level root cause of the pressure positive feedback. Cite file:line. Explain mechanistically why this operator amplifies p_perturbation by ~10× per timestep.

### 3. Reverts to consider

List specific commits (by hash) that should be considered for revert and why. If none, justify keeping all 4 fixes.

### 4. Savepoint ladder validity check

Are B0-B6 actually validating against WRF Fortran, or just against another JAX path? If the ladder isn't catching real-IC explosions, what's missing? Recommend a "B7-real-IC" rung or equivalent.

### 5. Strategic decision

Decision: ONE of:
- `Decision: REVERT-AND-RESTART — revert commit X, then start fresh dycore-pressure-feedback fix sprint at <named operator>`
- `Decision: FOCUSED-FIX — keep all 4 fixes, dispatch focused fix at <named operator>`
- `Decision: ARCHITECTURAL-PIVOT — the dynamics/core composition has a structural flaw, recommend <pivot plan> before continuing`
- `Decision: INSUFFICIENT-EVIDENCE — recommend <specific instrumentation> before deciding`

### 6. Risks of your recommendation

3-5 bullets on what could go wrong if the manager follows your decision.

>=600 bytes total.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_critic
# Read-only. No code/test commands required.
git log --oneline manager-2026-05-23 -10
# Write critic-report.md
git add .agent/sprints/2026-05-26-m6-dycore-pressure-stepback/critic-report.md
git commit -m "[critic memo] $(date -u +%FT%TZ)"
```
