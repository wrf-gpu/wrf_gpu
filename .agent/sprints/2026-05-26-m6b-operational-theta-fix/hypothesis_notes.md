# Hypothesis Notes

Summary: BLOCKED_ACCEPTANCE_NOT_MET. The focused operational theta patch fixed the
real-IC controlled parity failure at steps 2/5/10 and preserved B6 parity, but it
did not clear the full 20260509 1h replay. That run still fails at step 11 with
the existing `MATH:coftz` classification.

## H1 - `advance_mu_t_wrf` theta flux source

Matched partially. `advance_mu_t_wrf` was using `theta_1` directly for the
horizontal and vertical theta-flux source. I switched the flux source to
`theta_ave`, matching the coftz diagnosis that the operational path must not
build theta-face coefficients from the just-advanced instantaneous theta path.

## H2 - `acoustic_substep_core` theta carry between substeps

Matched for the controlled parity nonfinite. The larger concrete defect was
that `advance_mu_t_wrf` emits WRF's mass-coupled small-step theta quantity, but
`acoustic_substep_core` was carrying that value forward as perturbation theta.
Adding the WRF-style decoupling boundary:

`(theta_mass + theta_1 * (c1h * mut + c2h)) / (c1h * muts_new + c2h)`

reduced the first RK1 substep from O(1e3 K perturbation) back to the input
perturbation range and made steps 2/5/10 bitwise-clean.

## H3 - `build_scratch_state` feeds wrong `t_2ave`

Not the primary line-level defect after H2. Once theta is decoupled before the
scratch update, `t_2ave = 0.5 * (theta_old + theta_new)` no longer inherits the
mass-coupled runaway and the controlled parity probes pass.

## H4 - `_ph_tend_increment` unit/sign issue

Not localized here. `ph_tend` became finite and bitwise-clean after the theta
decoupling. No independent ph_tend formula change was made.

## H5 - `dts` substep time

Not matched. The operational path was already using `dt_s / acoustic_substeps`;
the fixed parity proof retains that cadence and passes steps 2/5/10.

## Remaining failure

The 20260509 1h localizer still fails at step 11, cell k=28 j=59 i=72, with
theta O(2.4e12 K), qv depleted to the floor, and qc O(3.2e7). The script still
classifies this as `MATH:coftz`. The nearest next investigation is the physics
and vertical-implicit/coftz interaction on that run, not the controlled
`advance_mu_t`/shared acoustic-core parity path fixed here.
