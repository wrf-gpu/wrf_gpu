# Worker Report

## Summary

Summary: Completed the pure research scout for M6.x third-path substrate options. No model code, tests, governance files, reviewer/tester/manager reports, or memory patch files were modified. The comparison memo scores Options A/B/C across all required dimensions and names `C-primary = ICON4Py-pattern JAX rewrite`, while recommending Option B as the highest-confidence immediate route to an honest 1h coupled Gen2 d02 forecast.

## Files Changed

- `.agent/sprints/2026-05-24-m6x-third-path-substrate-scout/option_comparison.md`
- `.agent/sprints/2026-05-24-m6x-third-path-substrate-scout/worker-report.md`
- `.agent/sprints/2026-05-24-m6x-third-path-substrate-scout/proof_no_touch.txt`

No code files were edited. The pre-existing untracked `scripts/dispatch_role_session2.sh` was not touched.

## Commands Run

- `sed -n ... PROJECT_CONSTITUTION.md`, `AGENTS.md`, `CLAUDE.md`, `PROJECT_PLAN.md`, `.agent/milestones/ROADMAP.md`, `.agent/goals/M1-DONE.md`, sprint contract, and local skills.
  Output: confirmed the constitutional constraints, local-skill authority, sprint scout scope, JAX-primary context, M6 gate, and no-code/no-push contract.
- `nl -ba .agent/decisions/blockers/M6-DYCORE-BLOCKER-MEMO.md | sed -n '1,260p'`
  Output: confirmed the active blocker, 1h d02 RMSE collapse, sanitizer counts, Options A/B/C/D, and manager dispatch of this scout.
- `nl -ba .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/verdict.md | sed -n '1,260p'`
  Output: `NO-BUG-LOCALIZED`; seven one-suspect buckets failed to move first nonfinite beyond step 2.
- `nl -ba .agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/worker-report.md | sed -n '1,260p'`
  Output: confirmed sanitizer-off replay first nonfinite step 2, field `u`, stage `post-recurrence`, and 54-test regression proof from that sprint.
- `nl -ba` / `rg` reads for `.agent/sprints/2026-05-24-m6x-s2dot1redo-real-baseline/findings_real.md`, `.agent/sprints/2026-05-24-m6x-exit-rule-critic/reviewer-report.md`, `.agent/sprints/2026-05-23-m6x-dycore-alt-methods-scout/worker-report.md`, ADR-001, ADR-023, ADR-024, source-mining table, `MILESTONES.md`, and `VALIDATION_STRATEGY.md`.
  Output: confirmed M6 close gate, Gen2 replay failure magnitudes, ADR-023/ADR-021 limitations, JAX primary backend, and Tier 1-4 validation posture.
- Local source inspection with `nl -ba` / `git show` for WRF, MPAS, Dinosaur, ICON4Py, and Pace source files.
  Output: produced the file:line citations used in `option_comparison.md`, including WRF `module_small_step_em.F:1102-1489`, MPAS `mpas_atm_time_integration.F:1824-2208`, Dinosaur `time_integration.py:74-405`, ICON4Py `solve_nonhydro.py:139-1378`, and Pace `dyn_core.py:472-965`.
- `git ls-remote https://github.com/google-research/neuralgcm.git HEAD`
  Output: `66842d977249081f07086fb699dfcffddb27f6d4 HEAD`.
- `curl -fsSL` reads against NeuralGCM public GitHub raw source at that commit.
  Output: confirmed README/license/pyproject and `legacy/api.py` JAX advance/unroll citations; marked as public, not locally verified.
- `pytest --collect-only 2>&1 | tail -3 | tee .agent/sprints/2026-05-24-m6x-third-path-substrate-scout/proof_no_touch.txt`
  Output:
  ```
          <Function test_all_skills_have_metadata_and_evals>

  ========================= 579 tests collected in 5.92s =========================
  ```
- `wc -c .agent/sprints/2026-05-24-m6x-third-path-substrate-scout/{worker-report.md,option_comparison.md,proof_no_touch.txt}`
  Output: worker report 6495 bytes before this final command-log patch, comparison memo 16072 bytes, proof file 141 bytes.
- `git status --short --branch`
  Output: on `scout/codex/m6x-third-path-substrate`; modified/added deliverables only, plus pre-existing untracked `scripts/dispatch_role_session2.sh` not touched.
- `git diff --check -- .agent/sprints/2026-05-24-m6x-third-path-substrate-scout/{worker-report.md,option_comparison.md,proof_no_touch.txt}`
  Output: exit 0, no whitespace errors.
- `git commit -m "Add M6x third-path substrate scout report"` followed by amend after adding this command log.
  Output: three scout files committed on `scout/codex/m6x-third-path-substrate`; no remote push performed per sprint contract.

## Proof Objects

- `.agent/sprints/2026-05-24-m6x-third-path-substrate-scout/option_comparison.md`
- `.agent/sprints/2026-05-24-m6x-third-path-substrate-scout/worker-report.md`
- `.agent/sprints/2026-05-24-m6x-third-path-substrate-scout/proof_no_touch.txt`

## Risks

- Option C evidence is strongest as an ICON4Py-pattern rewrite into JAX, not as a direct dependency. Direct ICON4Py/Pace import would reopen backend/toolchain/state-layout decisions under ADR-001.
- NeuralGCM was not locally cloned; I used public GitHub source at `google-research/neuralgcm@66842d9` and marked it as not locally verified.
- Time estimates are scout-level estimates, not implementation measurements. The source evidence supports relative risk, not calendar guarantees.
- The role wrapper asked for a worker branch and remote push, but the sprint contract and user clarification say branch `scout/codex/m6x-third-path-substrate` and `NO remote push`; I followed the contract.

## Handoff

Objective: compare Options A, B, and C for escaping the M6.x dycore blocker, and deep-dive candidate substrates for Option C.

Files changed: `option_comparison.md`, `worker-report.md`, and `proof_no_touch.txt` in `.agent/sprints/2026-05-24-m6x-third-path-substrate-scout/`.

Commands run: mandatory read-order commands, blocker/bug-hunt reads, source citation reads, public NeuralGCM source reads, and the required `pytest --collect-only` validation command above.

Proof objects produced: comparison memo, this report, and `proof_no_touch.txt`.

Unresolved risks: C-primary is not a direct drop-in substrate; it is an ICON4Py-pattern JAX rewrite. Option A remains faster than B but may repeat partial-scratch ambiguity. Option B is slower but has the clearest correctness path.

Next decision needed: manager should decide whether to pursue the recommended Option B, use A as a one-sprint fast probe before B, or open a separate ADR for the C-primary ICON4Py-pattern JAX rewrite.

Dissent: The strongest case against Option B is that it spends the most time preserving the WRF small-step family after two WRF-shaped attempts already failed. Option A could close faster if the missing state really is only `t_2ave`, `ww`, `muave`, `muts`, and `ph_tend`; Option C could avoid the brittle scratch-state trap entirely by adopting the ICON4Py vertical-implicit organization that already has limited-area hooks, predictor/corrector split, tridiagonal `w` solve, Rayleigh/divergence controls, and explicit exchanges. If the project values escaping the current architecture over source-parity confidence, C-primary deserves a prototype sprint.

RECOMMEND-OPTION-B. The full WRF small-step port with a savepoint harness is the best immediate M6 close path because it turns the current ambiguous step-2 failure into recurrence-by-recurrence numerical comparisons against WRF, preserves M5 physics and d02 replay infrastructure, adds no external dependency, and keeps Tier-4 Gen2 RMSE interpretation strongest. C-primary should be retained as the fallback architecture, specifically an ICON4Py-pattern JAX rewrite, if B proves too large or exposes a deeper incompatibility.
