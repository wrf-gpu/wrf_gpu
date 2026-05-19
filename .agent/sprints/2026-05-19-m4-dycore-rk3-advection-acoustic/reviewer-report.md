# Reviewer Report — M4 Dycore RK3 Advection Acoustic

Role: reviewer (opus-reviewer / Codex gpt-5.5). Branch: `reviewer/opus/m4-dycore-rk3-advection-acoustic`.

## Findings

1. **blocker — RK3 over-applies nonzero base tendencies.** `rk3_step` advances sequentially by `dt / 3`, then `dt / 2`, then `dt` on the already-updated state (`src/gpuwrf/dynamics/rk3.py:45`, `src/gpuwrf/dynamics/rk3.py:51`, `src/gpuwrf/dynamics/rk3.py:57`), while `add_scaled_tendencies` adds the full passed tendency each time (`src/gpuwrf/dynamics/tendencies.py:11`). A constant RHS should integrate to `state + dt*tendency`; my spot-check with `dt=6` and `theta_tendency=1` produced mean `theta` delta `11.0`, not `6.0`. This is a core integrator correctness failure and is not covered by `tests/test_m4_rk3.py`, which tests only the separate helper `rk3_scalar_decay` (`tests/test_m4_rk3.py:16`).

2. **blocker — Tier-1 parity does not test the dycore advection operator.** `run_tier1` imports and runs `fixture_reference_update` (`src/gpuwrf/validation/tier1.py:15`, `src/gpuwrf/validation/tier1.py:47`), then reports an operator string that explicitly says the dycore uses a different 5H/3V upwind scheme (`src/gpuwrf/validation/tier1.py:53`). The actual M4 mass-scalar advection path is `advect_mass_scalar` / `compute_advection_tendencies` (`src/gpuwrf/dynamics/advection.py:125`, `src/gpuwrf/dynamics/advection.py:174`). The artifact therefore proves the M1 fixture generator matches its own committed `phi_next`, not that the M4 dycore has Tier-1 parity.

3. **blocker — Tier-2 invariant proof is a degenerate steady-state run.** `density_current_state` starts from `State.zeros`, leaves `u/v/w` zero, sets uniform `p`, and sets `mu` to ones (`src/gpuwrf/validation/tier2.py:48`, `src/gpuwrf/validation/tier2.py:60`, `src/gpuwrf/validation/tier2.py:61`). `compute_advection_tendencies` never updates `mu` (`src/gpuwrf/dynamics/advection.py:179`), and `invariant_record` only checks final-state `qv`/finite values (`src/gpuwrf/validation/tier2.py:74`). My 10-step spot-check on the same IC had max state delta `0.0`. This does not satisfy the contract’s density-current or nontrivial invariant evidence.

4. **blocker — Tier-3 convergence bypasses the integrated dycore.** The convergence engine imports `ddx4_centered` directly (`src/gpuwrf/validation/tier3.py:13`) and advances a local 1D RK helper (`src/gpuwrf/validation/tier3.py:22`) with the centered derivative (`src/gpuwrf/validation/tier3.py:28`). It does not call `step`, `run`, `rk3_step`, `compute_advection_tendencies`, the 5H/3V operator, or the acoustic substep. The artifact label confirms this is a centered-reference test (`artifacts/m4/tier3_convergence.json:2`). It cannot close the M4 Tier-3 dycore convergence requirement.

5. **major — HLO stripped evidence is contract-noncompliant and one HLO artifact is mislabeled.** The contract requested a hand-stripped sibling file, but the implementation exposes `step_stripped_reference` in the same file and sends it through `_step_impl(..., False)` (`src/gpuwrf/dynamics/step.py:54`, `src/gpuwrf/dynamics/step.py:58`). The maintainability note acknowledges the requested file was not created (`artifacts/m4/maintainability.md:7`). Also, `scripts/m4_hlo_diff.py` writes `prod` to both the production and stripped HLO outputs (`scripts/m4_hlo_diff.py:54`, `scripts/m4_hlo_diff.py:55`). I independently confirmed production HLO contains no `isfinite` tokens, so the debug no-leak property is likely true, but the proof object does not match AC 2.4.

6. **major — Velocity advection is materially incomplete relative to AC 1.2.** `advect_u_face` only applies x self-advection (`src/gpuwrf/dynamics/advection.py:135`), `advect_v_face` only y self-advection (`src/gpuwrf/dynamics/advection.py:141`), and `advect_w_face` only vertical self-advection (`src/gpuwrf/dynamics/advection.py:147`). The contract called for horizontal advection of `u`, `v`, `w`, and `theta` plus vertical advection. This may be acceptable only if the sprint contract is amended to define a narrower toy dycore.

7. **minor — M5 dry-run reports unavailable register/local-memory metrics as zero.** `scripts/m4_m5_gate_dryrun.py` hard-codes `local = 0` and `registers = 0` (`scripts/m4_m5_gate_dryrun.py:33`). The artifact records those zeros while the rationale says profiler follow-up is required (`artifacts/m4/m5_gate_dryrun.json:4`). This is not a sprint blocker because the gate trip is reporting-only, but it must not be read as evidence that local memory and registers pass.

## Contract Compliance

The committed worker diff stays inside the implementation/proof paths listed by the sprint contract. The worktree also contains uncommitted tester/manager-side changes outside the worker diff; I did not modify or rely on those for this decision except for reading `tester-report.md` and the adversarial test summary.

Required proof objects are present, and the JSON artifacts parse, but Tier-1, Tier-2, Tier-3, and HLO-stripped evidence do not satisfy the contract’s stated meaning. Hard transfer/temp-byte artifacts report zero post-init transfers and zero temporary bytes, but the correctness gates fail before performance can be accepted.

I read every line of `src/gpuwrf/dynamics/step.py` and `src/gpuwrf/debug/asserts.py` and found 1 simplification opportunity: replace the same-code-path stripped wrapper with either a real hand-stripped sibling or a manager-approved amended evidence path.

## Correctness Risks

The actual integrated dycore has no nontrivial oracle coverage for its 5H/3V advection, no nontrivial mass/positivity trajectory, and no convergence proof. The RK3/base-tendency scaling bug is independently reproducible and would corrupt any later physics tendencies passed through the public API.

## Performance Risks

The M5 dry-run trips on kernel launches (`29 > 10`) as documented in `artifacts/m4/m5_gate_dryrun.json:2`. Registers, local memory, occupancy, and bandwidth remain unknown; no performance claim should be made from this sprint beyond the reported zero transfer/temp-byte artifacts.

## Independent Spot-Checks Run

- `git diff --stat main...HEAD` and `git diff --check main...HEAD`.
- Re-ran tier validators to `/tmp/wrf_gpu2_reviewer_m4/*.json`; reproduced Tier-1/2/3 `pass: true`.
- Independent probes: Tier-1 wrapper max abs error `0.0`; Tier-2 10-step max state delta `0.0`; constant theta tendency RK spot-check delta `11.0` vs expected `6.0`.
- HLO spot-check on reduced grid: normalized production/stripped HLO equal, production HLO contained no `isfinite` tokens.
- `PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false pytest -q tests/test_m4_debug_hooks.py tests/test_m4_tier1.py tests/test_m4_tier2_invariants.py tests/test_m4_tier3_convergence.py -p no:cacheprovider` → `14 passed in 13.91s`.

## Required Fixes

1. Fix `rk3_step` so constant tendencies integrate exactly over one `dt`; add a public `step`/`run` regression for this.
2. Replace Tier-1 with a real oracle for the dycore’s 5H/3V operator, preferably a sibling analytic fixture, or explicitly amend the contract before accepting a narrower reference-wrapper proof.
3. Replace Tier-2 with a nontrivial density-current or documented analytic invariant trajectory; check qv/finiteness over the trajectory, not just final state.
4. Replace Tier-3 with a convergence test that uses the integrated dycore public API or the actual dycore advection operator under a documented analytic setup.
5. Provide the literal hand-stripped HLO sibling or amend AC 2.4, and fix `scripts/m4_hlo_diff.py` to write the stripped HLO text to the stripped artifact.

Decision: Reject
