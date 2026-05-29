# F7C — 12-step operational-dt audit (AC3)

Command:
`taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 12 --dt-s 6
--acoustic-substeps 4 --epssm 0.5 --combination a --damping
--output-dir proofs/f7c/audit_operational_dt`

(real Gen2 d02 replay fixture; physics_off + boundary_off + guards_off; WRF
damping ON: w_damping=1, damp_opt=3, dampcoef=0.2, zdamp=5000.)

## Result — NOT CLEAN (honest)

`first_critical_violation` (combination a):
- step 8, RK stage 2, acoustic substep 1, operator `advance_mu_t`
- invariant `pressure_bounded`: `abs_p_over_base_max = 3.92` (threshold 2.0)

No masking clamp was used. The audit cadence was updated to match the production
RK cadence (large-step PGF + `rk_addtend_dry`, no `add_scaled_tendencies`
double-application).

## Interpretation

- Sprint B's first critical was step 6-7; with the corrected circulation cadence
  it moved to **step 8** (later, physical transients for the first ~7 steps), but
  the audit does not reach a clean 12 steps.
- The violation is `pressure_bounded` (perturbation pressure exceeds 2× base),
  i.e. the same acoustic restoring-balance weakness diagnosed for the idealized
  cases: the delta-from-reference perturbation-pressure work array does not build
  the restoring gradient fast enough against the (now active) large-step momentum
  forcing on the real d02 IC at dt=6 s. This is the residual acoustic-core
  formulation issue documented in `rk_addtend_dry_proof.md §3`, not the
  rk_addtend_dry/PGF cadence gap.

AC3 = NOT MET (residual instability past step 7, named with evidence).
