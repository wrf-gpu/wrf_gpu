# F7D — 12-step operational-dt audit (AC4)

Command (identical to Sprint C for apples-to-apples comparison):
`taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 12 --dt-s 6
--acoustic-substeps 4 --epssm 0.5 --combination a --damping
--output-dir proofs/f7d/audit_operational_dt`

(real Gen2 d02 replay fixture; physics_off + boundary_off + guards_off; WRF
damping ON: w_damping=1, damp_opt=3, dampcoef=0.2, zdamp=5000.)

## Result — NOT IMPROVED (honest)

| | first_critical | invariant | abs_p_over_base | operator |
|--|--|--|--|--|
| Sprint C (pre-fix) | **step 8**, RK2, sub1 | pressure_bounded | 3.92 | advance_mu_t |
| F7D (post mass-fix) | **step 5**, RK2, sub2 | pressure_bounded | 3.38 | advance_mu_t |

The MUT/MUTS mass-semantics fix moved the first `pressure_bounded` violation
**earlier** (step 5 vs 8), with a slightly **smaller** overshoot (3.38 vs 3.92).
The contract target — move first_critical *past* step 8 toward clean — was NOT met.

## Interpretation

On the real d02 IC (nonzero `mu_perturbation`), the fix replaces the wrong base-mass
(MUB) denominator with WRF's correct full small-step total `muts` in `calc_p_rho`.
This is the WRF-faithful formulation and is required for the operational path, but
it surfaces the same `pressure_bounded` acoustic-restoring weakness *sooner*: the
perturbation pressure still exceeds 2× base by step 5. This is the **same residual
class** as the idealized-case runaway (a perturbation-driven acoustic/vertical-mode
overshoot the current restoring + WRF damping does not fully control), and is
independent of the (now correct) total-mass semantics. No masking clamp was used.
