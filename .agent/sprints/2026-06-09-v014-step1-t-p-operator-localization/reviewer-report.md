# Reviewer Report

## Decision:

Accept. The sprint used the right wall-clock method: a focused Step-1 substage
truth/comparator instead of another long validation run. It produced a
specific earlier boundary and correctly avoids treating the issue as acoustic
until the first-RK physics/source boundary is resolved.

## Findings

- Verdict is
  `STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_T_STATE`.
- First strict and first material T/P-family mismatch is `T_STATE` at
  `after_rk_addtend_before_small_step_prep`, RK1.
- Same boundary has large tendency-family residuals, including `PH_TEND`,
  `RW_TEND`, `PH_TENDF`, `T_TEND`, and `T_TENDF`.
- RK1 `after_small_step_prep_calc_p_rho` work arrays `T_WORK` and `P_WORK` both
  have max_abs `0.0`, so the immediate next debug target is before
  `small_step_prep`.
- Production `src/gpuwrf/**` remained unchanged.

## Weaknesses

The WRF truth surfaces start after `rk_addtend_dry/spec_bdy_dry`; they do not
yet split WRF `first_rk_step_part1`, `first_rk_step_part2`, `rk_tendency`,
`relax_bdy_dry`, `rk_addtend_dry`, and `spec_bdy_dry` individually. That split
is the correct next sprint, not a reason to discount this proof.

## Required Next Sprint

Emit or compare the immediate post-`first_rk_step_part1/part2` WRF boundary and
the matching JAX `_physics_step_forcing` output. Determine whether the mismatch
is in physics state mutation, dry `*_tendf` construction, carry/state handoff,
or boundary tendency application.
