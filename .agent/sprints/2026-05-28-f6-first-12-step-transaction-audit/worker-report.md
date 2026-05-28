# F6 Worker Report

Summary: F6_COMPLETE. The first-12-step transaction audit was implemented and run on `cuda:0` under `taskset -c 0-3`. All four toggle combinations produced proof JSONs, the aggregate invariant file, and the audit summary.

## Objective

Localize the M11.3 first-12-step blow-up by instrumenting the operational RK/acoustic path with physics, boundary, and guard toggles, then add the three cheap regression tests from F4 Q1.

## Files Changed

- `scripts/f6_transaction_audit.py`
- `tests/unit/test_rk_scan_step_advection_active.py`
- `tests/unit/test_mu_persistence_two_substeps.py`
- `tests/unit/test_decouple_theta_state_reference.py`
- `proofs/f6/audit_combination_a.json`
- `proofs/f6/audit_combination_b.json`
- `proofs/f6/audit_combination_c.json`
- `proofs/f6/audit_combination_d.json`
- `proofs/f6/invariant_violations.json`
- `proofs/f6/audit_summary.md`
- `.agent/sprints/2026-05-28-f6-first-12-step-transaction-audit/worker-report.md`

## Commands Run

- `taskset -c 0-3 python -c "import jax; print([str(d) + ':' + d.platform for d in jax.devices()])"` inside sandbox: saw CPU only.
- Escalated `taskset -c 0-3 python -c "import jax; print([str(d) + ':' + d.platform for d in jax.devices()])"`: saw `cuda:0`.
- `taskset -c 0-3 python -m py_compile scripts/f6_transaction_audit.py tests/unit/test_rk_scan_step_advection_active.py tests/unit/test_mu_persistence_two_substeps.py tests/unit/test_decouple_theta_state_reference.py`
- `taskset -c 0-3 pytest -q tests/unit/test_rk_scan_step_advection_active.py tests/unit/test_mu_persistence_two_substeps.py tests/unit/test_decouple_theta_state_reference.py`
- `taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 1 --combination a --output-dir /tmp/f6_smoke`
- Throwaway full run with an incorrect perturbation-theta sanity check; artifacts were overwritten.
- `taskset -c 0-3 python -m py_compile scripts/f6_transaction_audit.py`
- `taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 12 --output-dir proofs/f6`

## Proof Objects Produced

- `proofs/f6/audit_combination_a.json`
- `proofs/f6/audit_combination_b.json`
- `proofs/f6/audit_combination_c.json`
- `proofs/f6/audit_combination_d.json`
- `proofs/f6/invariant_violations.json`
- `proofs/f6/audit_summary.md`

## Result

All four combinations reproduce an immediate pure-dycore critical violation at step 1, RK1 acoustic substep 1, inside `advance_mu_t`: total-theta sanity leaves the `[200, 700] K` range. The same run also records zero u/v delta inside acoustic substeps, RK2 saved-theta mismatch at step 1, `muts`/work-mu inconsistency at step 1, dry-mass negativity by step 8, pressure bound violation by step 9 or 12 depending on toggle, and nonfinite theta by step 10 or 12 depending on toggle.

## Unresolved Risks

- The theta-mass residual invariant is intentionally strict and trips immediately; it is useful as an algebraic alarm, not as a WRF parity claim.
- This audit is algebraic/JAX-internal only; it does not replace the independent WRF savepoint oracle work.
- The supplemental total-theta and wind sanity bounds are diagnostic guards, not formal WRF equivalence criteria.

## Next Decision Needed

Do a targeted dycore repair sprint before physics or boundary work: carry the RK1 saved family across RK2/RK3, add/validate small-step `advance_uv`, and then replace the `_diagnose_pressure` stub once save-family invariants are clean.
