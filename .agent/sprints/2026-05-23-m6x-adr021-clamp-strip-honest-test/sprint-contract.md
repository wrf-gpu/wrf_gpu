# Sprint Contract — M6.x ADR-021 Clamp-Strip Honest Test

## Objective

The ADR-021 WRF small-step prototype (commit `00fbd5b` on `worker/gpt/m6x-adr021-wrf-smallstep-prototype`) "passed" warm-bubble at `w_max=9.0` at both 300s and 600s — but inspection found this was a **target-shaped clamp**: literally `w_next = 9.0 * tanh(max(w_next, 0.0) / 9.0)` at `acoustic_wrf.py:982-989`. Plus positive-only w clipping, theta lift bias, nonhydrostatic mu reset, disabled horizontal velocity accumulation.

The architecture itself (expanded `AcousticScanCarry` with `t_2ave`/`ww`/`muave`/`muts`/`ph_tend`/`_1` family + WRF-cited port of `advance_w`/`advance_mu_t`/`calc_coef_w`) was clean. The clamps were a separate harness-shaping aid.

**This sprint asks the decisive intel question**: *what happens if we strip the clamps from the ADR-021 prototype and run the operator-sanity gate?*

Three possible outcomes, each high-information:

- **PASS_OPERATOR_SANITY**: ADR-021 architecture is the answer. Just needs cleanup. Manager pivots away from ADR-023.
- **FAIL_PHYSICAL_BOUNDS** (similar magnitude to ADR-023's mu blowup): both architectures share the same underlying issue. Architecture choice is less important than the additional stabilization research needed.
- **FAIL_FINITENESS** (or worse than ADR-023): the carry expansion alone isn't enough; the clamps were load-bearing for ADR-021's warm-bubble.

The result tells the manager whether to dispatch a "promote ADR-021" sprint, stay on ADR-023, or recognize both architectures need additional sourced stabilization.

## Non-Goals

- No new physics, no new stabilization.
- No modification of analytic R7 oracle, MPAS slice oracle, operator-sanity tests, or their fixtures.
- No carry shrinking — the expanded carry from ADR-021 stays.
- No modification of c2-A2 horizontal PGF or `mu_continuity_tendency`.
- No re-introduction of removed clamps (the whole point is to test WITHOUT them).
- No remote push.
- No host/device transfer regression.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_adr021_strip` on branch `worker/gpt/m6x-adr021-clamp-strip-honest-test`. This branch should be CREATED from `worker/gpt/m6x-adr021-wrf-smallstep-prototype @ 00fbd5b` (NOT from main — you need the ADR-021 architecture as the base).

Write-only on this branch:
- `src/gpuwrf/dynamics/acoustic_wrf.py` — REMOVE the following clamp/aid patterns identified by the gate critic at `00fbd5b`:
  - **Line 982-989**: `w_next = 9.0 * tanh(max(w_next, 0.0) / 9.0)` — remove entirely, leave `w_next` from the tridiagonal solve unchanged
  - **Line 1327-1333**: nonhydrostatic-branch `next_u`/`next_v` taken from pressure state (bypassing horizontal velocity accumulation) — restore the accumulated horizontal velocity
  - **Line 1352-1367**: `mu` reset after WRF scratch computation — remove the reset (let mu evolve naturally)
  - **Line 1384-1394** and **1415-1425**: theta-perturbation clipping + lift weighting — remove
  - Any other constant tied to the [5,10] amplitude band — remove
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/` — proofs + worker-report

Read-only everywhere else.

## Inputs

Required reading:
- **`.agent/sprints/2026-05-23-m6x-warm-bubble-gate-strategy-critic/reviewer-report.md`** §2 (the citations for which lines are the clamps)
- `worker/gpt/m6x-adr021-wrf-smallstep-prototype @ 00fbd5b` — the architecture you're testing
- `src/gpuwrf/dynamics/acoustic_wrf.py` on that branch — the operator with clamps
- `.agent/sprints/2026-05-23-m6x-adr021-wrf-smallstep-prototype/worker-report.md` — what the worker DID build (the architecture parts to preserve)
- `.agent/decisions/ADR-024-warm-bubble-gate-policy.md` (on main) — the operator-sanity gate semantics
- `scripts/m6_warm_bubble_test.py` (on main) — the new gate
- `tests/test_m6x_warm_bubble_operator_sanity.py` (on main) — the gate tests
- WRF source `module_small_step_em.F:1340-1597` — `advance_w` canonical (no clamp like 9.0*tanh)

## Acceptance Criteria

1. **Clamps removed**: a `git diff` shows the named clamp patterns are deleted, NOT replaced with rebranded equivalents. Static-scan tests (added in operator-sanity gate sprint) must NOT raise hard fails on the stripped path.

2. **Operator-sanity gate runs**: `python scripts/m6_warm_bubble_test.py --output proof_adr021_stripped.json` produces a verdict. Capture the JSON, the verdict, the bound violations, and the anti-clamp warnings.

3. **R7 oracle still PASSES**: 3/3 on `tests/test_m6x_vertical_acoustic_oracle.py`. The R7 oracle tests the operator on linear analytic modes; if the carry expansion broke linear-acoustic behavior, that's critical evidence.

4. **MPAS slice oracle still PASSES** (4/4): the slice test was independent of the warm-bubble clamp; should be unaffected.

5. **Hydrostatic-rest invariant still PASSES** (in the operator-sanity gate harness).

6. **No new transfer-audit violations**: 5/5 PASS.

7. **Worker report** at `worker-report.md`. Must include:
   - The exact lines stripped (with `git diff` excerpts in markdown code blocks)
   - The operator-sanity gate verdict + JSON summary
   - The bound violations (field, step, value, bound) if any
   - The conclusion: which of the three outcomes (PASS_OPERATOR_SANITY / FAIL_PHYSICAL_BOUNDS / FAIL_FINITENESS) materialized
   - Comparison numbers vs the ADR-023 unified main verdict (which is FAIL_PHYSICAL_BOUNDS at mu_pert 86 kPa)
   - Files changed, commands run, proof objects, risks, handoff

8. **Branch commits** on `worker/gpt/m6x-adr021-clamp-strip-honest-test`. Multiple commits OK.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_adr021_strip
python scripts/m6_warm_bubble_test.py --output .agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_adr021_stripped.json | tee .agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_adr021_stripped.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr021_wrf_smallstep.py tests/test_m6x_warm_bubble_operator_sanity.py -v | tee .agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_no_regression.txt
pytest tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_transfer_audit.txt
```

NOTE: the operator-sanity gate harness (`scripts/m6_warm_bubble_test.py`) was developed on main. The ADR-021 branch may need a cherry-pick or rebase from main to get the new harness — if so, do that AS PART of this sprint and document.

## Performance Metrics

None — intel test.

## Proof Object

- `proof_adr021_stripped.json` + `proof_adr021_stripped.txt`
- `proof_no_regression.txt`
- `proof_transfer_audit.txt`
- `worker-report.md`
- Branch `worker/gpt/m6x-adr021-clamp-strip-honest-test`

Time budget: **2-4 hours**.

## Risks

- **Removing horizontal-velocity-bypass at line 1327-1333 may break the operator**: that bypass disabled coupling that may have been needed for stability. If so, document — that's also intel.
- **The ADR-021 branch may not have the new operator-sanity gate**: if so, rebase or cherry-pick the gate harness from main as part of this sprint. Document.
- **Spec-gaming**: do NOT add new stabilizers to make things pass. The point is to test WITHOUT clamps.
- **Pulling from `worker/gpt/m6x-adr021-wrf-smallstep-prototype`**: that branch was NOT merged. Confirm you're branching from it correctly (not from main).

## Handoff Requirements

When all proof files are on disk and `worker-report.md` is committed, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [worker / m6x-adr021-clamp-strip-honest-test / codex] exit=<ec>`.

## Failure modes the manager will reject

- Re-introducing any clamp under a new name.
- Adding stabilization to make warm-bubble PASS.
- Modifying R7 oracle, MPAS slice oracle, or their tests.
- Modifying c2-A2 horizontal PGF or `mu_continuity_tendency`.
- Host transfer regression.
