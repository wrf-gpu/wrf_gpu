# Critical Review - M6 Dycore Pressure Step-Back

Decision: Reject

## Top Three Structural Concerns

1. The required decision artifact is missing. The role prompt asks for `proposal.md`, but this folder contains only the sprint contract, role prompt, retry marker, and completion helper. That makes the review target ambiguous, and the sprint contract itself conflicts with the active role scope by requesting `critic-report.md` plus a commit while the role prompt allows only `critical-review.md` and no commit.
2. The four-fix sequence is converging on bounded diagnostics, not physical dynamics. The strongest evidence is not marginal: real-IC runs drive pressure to double-precision overflow scale and fail M6 acceptance on both bounds and T2 RMSE.
3. The savepoint ladder is not yet proving the operational real-IC pressure path against WRF Fortran. Current parity claims can be true while the operational forecast is unphysical, because the path is still allowed to pass through guards, field projections, and shared JAX comparisons that do not isolate WRF `pressure_perturbation` terms.

## Findings

1. Critical - The available evidence rejects any M6 continuation or closeout based on today's fixes. The boundary audit reports final-step `p_perturbation` and `p_total` around `1.76e308` to `1.79e308` Pa on all three ICs, plus `u` up to `8e37` m/s and `w` up to `1e60` m/s; it explicitly classifies all three ICs as `FAIL_B_OR_C`. See `.agent/sprints/2026-05-26-m6-boundary-dynamics-audit/tester-report.md:64`, `.agent/sprints/2026-05-26-m6-boundary-dynamics-audit/tester-report.md:70`, and `.agent/sprints/2026-05-26-m6-boundary-dynamics-audit/tester-report.md:84`. The acceptance sprint independently reports Stage 1 `FAIL` and Stage 2 `FAIL`, with T2 RMSE as high as `7.36e85` K. See `.agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/worker-report.md:36` and `.agent/sprints/2026-05-26-m6-acceptance-tier4-all3-ics/worker-report.md:41`.

2. Critical - The most likely structural root cause is an inconsistent operational composition around acoustic/RK state projection, not a single bounded microphysics defect. `_physics_boundary_step` runs the RK/acoustic path, then restores `theta`, `mu`, `mu_total`, and `mu_perturbation` from `physical_origin` while leaving other evolved dynamic fields in place. See `src/gpuwrf/runtime/operational_mode.py:499` and `src/gpuwrf/runtime/operational_mode.py:504`. Separately, `_carry_from_acoustic_core` reconstructs `p_total` from template total/base difference plus `acoustic.p`, while `State` declares `p`/`p_total` as total pressure and `p_perturbation` as a distinct WRF perturbation field. See `src/gpuwrf/runtime/operational_mode.py:271` and `src/gpuwrf/contracts/state.py:331`. That is a plausible pressure positive-feedback mechanism: pressure/geopotential/wind are advanced or reconstructed under one thermodynamic state, then the next step re-enters with mass/theta fields reset to another state.

3. High - The V fix is a symptom mask. `_m6b_acoustic_tendencies` replaces the computed V tendency with the base V tendency, which suppresses an unvalidated reduced-dycore self-advection term rather than correcting the WRF operator. See `src/gpuwrf/runtime/operational_mode.py:219`. The worker report says the fix is narrow and operational-only and that full 1h acceptance remains blocked by thermodynamic instability. See `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/worker-report.md:56`.

4. High - The microphysics fix is useful as a guardrail but cannot be counted as root-cause progress on pressure dynamics. Thompson now masks invalid thermodynamic columns with `_thermodynamically_admissible` and `_select_state`; operational mode also rejects nonfinite and out-of-range moisture or boundary replay values. See `src/gpuwrf/physics/thompson_column.py:386`, `src/gpuwrf/physics/thompson_column.py:400`, `src/gpuwrf/runtime/operational_mode.py:186`, and `src/gpuwrf/runtime/operational_mode.py:525`. The worker report is honest that this guard masks bad coupling values and leaves pressure/wind dynamics suspect. See `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/worker-report.md:40`.

5. High - The savepoint ladder is not yet a real-IC WRF pressure validation gate. The sprint contract itself says multi-step CPU parity is misleading because validation wrappers and operational explode the same way. See `.agent/sprints/2026-05-26-m6-dycore-pressure-stepback/sprint-contract.md:12`. The boundary audit also notes that per-step WRF truth and per-term Fortran pressure cross-checks are missing, and recommends per-substep `p` tendency comparison against `module_small_step_em.F::pressure_perturbation`. See `.agent/sprints/2026-05-26-m6-boundary-dynamics-audit/tester-report.md:130` and `.agent/sprints/2026-05-26-m6-boundary-dynamics-audit/tester-report.md:147`.

6. Medium - The coftz fix landed in a path that the operational replay did not exercise. The worker report states the fix is correct for `vertical_implicit_solver.py` / `_mpas_recurrence_vertical_update`, but it does not affect the current operational replay and requires changing or routing through `dynamics/core`. See `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/worker-report.md:47` and `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/worker-report.md:53`.

7. Medium - The op-theta decoupling should not be accepted or reverted blindly. It fixed the step 2/5/10 parity symptom and kept B6 green, but the later audit specifically calls out the composition-boundary decoupling and `p_total` routing as a suspect in unbalanced `p_perturbation` integration. See `.agent/sprints/2026-05-26-m6b-operational-theta-fix/worker-report.md:10` and `.agent/sprints/2026-05-26-m6-boundary-dynamics-audit/tester-report.md:116`. Treat this as an ablation candidate, not a proven root cause.

8. Medium - The review process is internally inconsistent. The active role prompt requires `critical-review.md` only and no commit, while the sprint contract asks for `critic-report.md` and a commit. See `.agent/sprints/2026-05-26-m6-dycore-pressure-stepback/role-prompts/critical-review.md:20`, `.agent/sprints/2026-05-26-m6-dycore-pressure-stepback/role-prompts/critical-review.md:28`, and `.agent/sprints/2026-05-26-m6-dycore-pressure-stepback/sprint-contract.md:52`. I followed the active role prompt.

## Dissent

The strongest argument for accepting with required fixes is that the step-back direction is right: the contract names symptom-fix risk, calls out the suspect op-theta composition change, and asks whether B0-B6 should be revalidated on real ICs. The op-theta fix also has nontrivial positive evidence: controlled real-IC steps 2/5/10 passed with zero final delta and B6 stayed green.

That dissent does not overcome the evidence. The pressure overflow is common across ICs, two cases are interior-dominant at onset, and the current guard/projection architecture lets surface diagnostics look bounded while core dynamics are not physical.

## Closing Recommendation

Reject the implied decision to continue from the four-fix stack as an M6 path. Do not close M6, do not start M6c, and do not add more production guards as acceptance evidence.

Open an explicit B7-real-IC pressure rung before further fixes: one WRF Fortran-instrumented real-IC savepoint around the first bad cells for 20260509 and 20260521, with per-substep `p_perturbation`, `p_total/PB`, `theta`, `mu/muts`, `ww`, vertical coefficients, RHS, and pressure-tendency terms. Run the operational path with guards disabled and compare operator-by-operator. As a separate ablation, test reverting or gating the op-theta composition change and the V/microphysics masks, but use those only to localize the defect, not as a blind rollback plan.
