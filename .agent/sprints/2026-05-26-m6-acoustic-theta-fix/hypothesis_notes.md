# Hypothesis Notes

Summary: BLOCKED_ACCEPTANCE_NOT_MET. The sprint found and fixed the named
step-18 acoustic theta algebra bug, but the full 360-step guard-disabled
acceptance still exposes a later acoustic theta failure at step 47.

## H1 - Mass-coupled theta mismatch

Matched. WRF `small_step_prep` mass-couples `t_2` before `advance_mu_t`, and
`small_step_finish` projects it back using the saved pre-coupled theta. The JAX
core was feeding perturbation theta directly into `advance_mu_t` and only
decoupling afterward. At the step-17 bad cell `[12,30,62]`, the one-substep
theta delta was about 442 K before the WRF prep/finish algebra and about 9.46 K
after it.

## H2 - `theta_1` reference state stale

Partially matched as a remaining risk. The local `advance_mu_t` formula now uses
WRF `t_1` for horizontal and vertical theta fluxes, but the later step-47
failure suggests the operational RK/acoustic save-family may still be updating
the reference state at the wrong boundary. That wrapper is outside this sprint's
file ownership.

## H3 - `fnm` / `fnp` face weights wrong

Not matched. The weights are used with the same index pattern as
`module_small_step_em.F:1151-1155`; the large bad-cell delta was caused by
missing mass coupling, not by an off-by-one in the face interpolation.

## H4 - `ww` sign error in `wdtn`

Not matched by the step-17 cross-check. Changing the theta mass-coupling path
reduces the bad-cell delta without changing the `ww * face_theta` sign.

## H5 - `t_2ave` running-average update wrong

Not the root cause of the named blocker. `t_2ave` remains a later-risk signal:
at the new step-46 precursor, `t_2ave` and instantaneous theta disagree sharply,
but variant probes still localize the explosive projection to the near-singular
`c1h*muts+c2h` denominator.

## H6 - Missing `msfty` map factor on horizontal theta flux

Not matched. The WRF formula uses `msftx` inside the flux divergence and the
outer `msfty` multiplier when applying the tendency; the implemented formula
keeps that structure.
