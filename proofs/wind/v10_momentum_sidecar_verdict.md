# case3 V10 momentum-budget sidecar verdict

**VERDICT:** Run the profile/budget/MYNN-momentum-off experiment, but do not treat
MYNN-off as a clean binary isolation. It is a perturbation test: improvement proves
"PBL momentum matters", not "MYNN over-mixing is the cause". Current evidence makes
interior dycore momentum/PGF/advection the higher-prior culprit because the k0
vector direction is wrong, especially `u0`; MYNN remains fixable only if the
profile and signed tendencies prove it is pushing the wind in the error direction.

## Findings

1. **MYNN-off is confounded.** In `_apply_mean_tendencies`, the same implicit
   solve applies both `dfm` vertical diffusion and the `bottom_drag` lower
   boundary sink. Zeroing the coupler's `du_mass/dv_mass` removes both. Over water,
   bottom stress generally damps the current k0 wind, while vertical diffusion can
   strengthen or weaken k0 depending on the shear above. Therefore a better MYNN-off
   V10 score is not enough to blame `dfm`, `el`, or over-mixing.

2. **Use signed vector attribution, not speed-only attribution.** The final water
   error vector is approximately `E = GPU - WRF = (+1.81 u, +2.26 v) m/s`. A harmful
   PBL momentum increment must be aligned with `(+u, +v)` over the failing interior.
   Surface drag on the current `(u>0, v<0)` wind would tend to push `u` negative
   and `v` positive, so it can worsen the weak southerly `v` but should help the
   wrong-sign `u`. A wrong-sign `u0` is therefore a dycore-smelling signal unless
   vertical mixing is importing positive `u` from aloft.

3. **The planned vertical profile is the decisive discriminator.**
   - Dycore-favored: `u/v` errors have the same sign through k0..k5, or k2+ is
     already wrong versus WRF before MYNN can matter.
   - PBL-favored: k2+ matches WRF reasonably, but k0/k1 are selectively pulled away
     from WRF by MYNN increments aligned with `E`.
   - Mixed: `v` improves with MYNN-off but `u` remains wrong-sign. That is not a
     MYNN root cause; it is at most a PBL contribution on top of dycore direction
     error.

4. **The current harness must not be described as a per-step budget.** The script
   records a single MYNN increment on the final post-forecast state. That is useful
   for sign and scale, but it is not the 24 h integrated tendency split promised in
   the task. If the conclusion depends on cumulative attribution, accumulate
   pre-PBL, post-PBL, and post-dycore k0 `u/v` over water inside the forecast loop
   or at least at hourly segment boundaries.

5. **Add one more split before naming a MYNN lever.** For the final-state one-step
   diagnostic, report three MYNN variants over water/deep interior: full momentum,
   no-bottom-drag with `dfm` active, and bottom-drag-only with `dfm=0`. This separates
   surface stress from vertical redistribution. Without this split, "MYNN-off helps"
   does not identify whether the candidate fix is lower boundary stress, vertical
   diffusivity, or neither.

## Answers

**Q1:** MYNN-momentum-off is valid as a sensitivity test, not as a clean isolation.
Avoid mis-attribution by checking signed `du0/dv0` against the error vector, and by
separating bottom drag from vertical diffusion. If MYNN-off strengthens `v` only
because it removed otherwise WRF-faithful surface drag, that is a force-balance
problem upstream of MYNN, not proof of over-mixing.

**Q2:** Yes, the `u0` sign error argues for a dycore prior. PBL mixing can only own
the direction error if the aloft GPU profile is WRF-like or WRF-opposite in a way
that the MYNN increment demonstrably rotates k0 away from WRF. If the whole lower
column is wrong-sign/too-weak, defer to dycore/interior momentum transport.

**Q3:** No empirical MYNN tuning lever is approved for a `-0.099` residual. The only
low-regression MYNN changes are source-backed WRF fidelity fixes: wrong length-scale
option/constant, wrong `dfm` units or vertical indexing, wrong rho/dz weighting,
wrong lower-boundary stress vector, wrong A-grid-to-C-grid increment, or a missing
WRF limiter. Do not tune `el`, scale `dfm`, or alter Pr/prlim unless a WRF savepoint
or column oracle proves the current value is non-faithful and a multi-case gate
clears wind, T2/RH, PBLH, and stability.

## Falsifiers Required

A MYNN-fixable conclusion requires all of:
- k2+ lower-column profile is close to WRF while k0/k1 are selectively wrong.
- Integrated MYNN `du/dv` over the failing water interior is aligned with the final
  error vector and large enough to matter.
- MYNN-off improves vector/component skill, not just V10 speed.
- Bottom-drag-vs-vertical-diffusion split identifies the offending term.

If any of those fail, classify the case3 residual as dycore/interior momentum and
defer MYNN code changes.
