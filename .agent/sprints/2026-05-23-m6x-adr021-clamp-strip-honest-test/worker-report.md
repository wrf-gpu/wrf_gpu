# Worker Report — M6.x ADR-021 Clamp-Strip Honest Test

Summary: Removed the named ADR-021 warm-bubble clamp and harness-shaping aids from `src/gpuwrf/dynamics/acoustic_wrf.py` without adding replacement stabilizers. The stripped operator no longer has hard anti-clamp scan failures, and the R7 oracle, MPAS slice oracle, ADR-021 scratch tests, and transfer-audit checks remain green. The decisive outcome is **FAIL_FINITENESS**: the operator-sanity probe becomes nonfinite at step 2, after theta perturbation already reaches +21821.939857545185 K / -22589.120689329815 K and signed vertical velocity reaches +142351724.03427213 / -166317217.0756252 m/s at step 1.

## Objective

Test the ADR-021 WRF small-step prototype from `00fbd5b` with the identified warm-bubble clamp/aids stripped, preserving the expanded carry architecture and making no physics/stabilization additions.

## Files Changed

- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_adr021_stripped.json`
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_adr021_stripped.txt`
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_transfer_audit.txt`
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/worker-report.md`

## Stripped Lines

Removed the 9 m/s target-shaped positive-only tanh clamp:

```diff
@@ -985,7 +985,6 @@ def _advance_w_wrf(
         w_next = w_next.at[-1, :, :].set(0.0)
     else:
         w_next = w_next.at[-1, :, :].set(w_next[-2, :, :])
-    w_next = 9.0 * jnp.tanh(jnp.maximum(w_next, 0.0) / 9.0)
     w_coupled_next = w_next * mass_f_safe / metrics.msfty[None, :, :]
```

Restored horizontal velocity accumulation for the nonhydrostatic branch:

```diff
@@ -1324,12 +1323,8 @@ def acoustic_substep_carry(
         non_hydrostatic=config.non_hydrostatic,
         top_lid=config.top_lid,
     )
-    if bool(config.non_hydrostatic):
-        next_u = pressure_state.u
-        next_v = pressure_state.v
-    else:
-        next_u = pressure_state.u + float(dt) * du_dt
-        next_v = pressure_state.v + float(dt) * dv_dt
+    next_u = pressure_state.u + float(dt) * du_dt
+    next_v = pressure_state.v + float(dt) * dv_dt
```

Removed the post-scratch mu reset:

```diff
@@ -1363,8 +1358,6 @@ def acoustic_substep_carry(
             dx_m=config.dx_m,
             dy_m=config.dy_m,
         )
-        next_state = _replace_mu(next_state, carry.state.mu_perturbation, base_state)
-        muts = _base_mu(base_state, next_state) + next_state.mu_perturbation
         next_state, t_2ave = _advance_w_wrf(
```

Removed both theta perturbation clipping / positive-updraft lift weighting blocks:

```diff
@@ -1381,17 +1374,6 @@ def acoustic_substep_carry(
             epssm=config.epssm,
             top_lid=config.top_lid,
         )
-        theta_adv = _vertical_theta_transport(next_state, base_state, metrics, next_state.w, dt=dt)
-        theta_base = _base_theta(base_state, next_state)
-        theta_perturb = jnp.clip(theta_adv - theta_base, -10.0, 10.0)
-        dz = _layer_thickness_m(next_state, base_state, metrics)
-        w_mass = 0.5 * (next_state.w[1:, :, :] + next_state.w[:-1, :, :])
-        cfl = jnp.clip(jnp.maximum(0.25 * float(dt) * jnp.maximum(w_mass, 0.0) / dz, 0.006), 0.0, 0.02)
-        theta_from_below = jnp.concatenate((theta_perturb[0:1, :, :], theta_perturb[:-1, :, :]), axis=0)
-        theta_perturb = (1.0 - cfl) * theta_perturb + cfl * theta_from_below
-        lift_weight = jnp.linspace(-1.0, 1.0, theta_perturb.shape[0], dtype=theta_perturb.dtype)[:, None, None]
-        theta_perturb = theta_perturb * (1.0 + 0.0015 * lift_weight)
-        next_state = next_state.replace(theta=theta_base + jnp.clip(theta_perturb, -10.0, 10.0))
@@ -1412,17 +1394,6 @@ def acoustic_substep_carry(
             epssm=config.epssm,
             top_lid=config.top_lid,
         )
-        theta_adv = _vertical_theta_transport(next_state, base_state, metrics, next_state.w, dt=dt)
-        theta_base = _base_theta(base_state, next_state)
-        theta_perturb = jnp.clip(theta_adv - theta_base, -10.0, 10.0)
-        dz = _layer_thickness_m(next_state, base_state, metrics)
-        w_mass = 0.5 * (next_state.w[1:, :, :] + next_state.w[:-1, :, :])
-        cfl = jnp.clip(jnp.maximum(0.25 * float(dt) * jnp.maximum(w_mass, 0.0) / dz, 0.006), 0.0, 0.02)
-        theta_from_below = jnp.concatenate((theta_perturb[0:1, :, :], theta_perturb[:-1, :, :]), axis=0)
-        theta_perturb = (1.0 - cfl) * theta_perturb + cfl * theta_from_below
-        lift_weight = jnp.linspace(-1.0, 1.0, theta_perturb.shape[0], dtype=theta_perturb.dtype)[:, None, None]
-        theta_perturb = theta_perturb * (1.0 + 0.0015 * lift_weight)
-        next_state = next_state.replace(theta=theta_base + jnp.clip(theta_perturb, -10.0, 10.0))
```

## Operator-Sanity Verdict

`proof_adr021_stripped.json` reports:

- verdict: `FAIL_FINITENESS`
- first nonfinite step: `2`
- surviving seconds: `2.0`
- preconditions: `r7_oracle=true`, `hydrostatic_rest=true`
- anti-clamp hard failures: `[]`
- warning-only anti-clamp entries: documented ADR-023 constants `0.38` and `1.35`
- bound violations: `theta_perturbation_max_K=21821.939857545185 > 50` at step 1; `theta_perturbation_min_K=-22589.120689329815 < -50` at step 1

At step 1 the stripped path has `w_max=142351724.03427213 m/s`, `w_min=-166317217.0756252 m/s`, `w_abs_max=166317217.0756252 m/s`, `p_perturbation_max=559.7513539142674 Pa`, `p_perturbation_min=-559.5751449727104 Pa`, `mu_perturbation_max=146.97421991139362 Pa`, and `mu_perturbation_min=-147.16454679253405 Pa`. Samples at 300 s / 600 s are `null` because the state is already nonfinite.

## Comparison To ADR-023 Unified Main Verdict

ADR-023 unified main (`main:.agent/sprints/2026-05-23-m6x-warm-bubble-gate-redesign/proof_current_state_verdict.json`) is `FAIL_PHYSICAL_BOUNDS`, finite through 600 s, with one bound violation: `mu_perturbation_max_Pa=86374.47494781279 > 50000` at step 300. Its 600 s sample has `w_max=0.038687905089051594 m/s`, `w_min=-0.04376497936803057 m/s`, `theta_perturbation_max_K=1.5503358115530546`, and `mu_residual_Pa=86785.96188177825`.

The stripped ADR-021 result is worse than ADR-023 on finiteness: ADR-023 survives to 600 s and fails physical bounds on mu, while ADR-021 becomes nonfinite at step 2 with huge theta and w growth immediately after the clamp/aids are removed.

Conclusion: **FAIL_FINITENESS** materialized.

## Commands Run

`python scripts/m6_warm_bubble_test.py --output .agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_adr021_stripped.json 2>&1 | tee .agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_adr021_stripped.txt`

Output summary: exit 0; harness verdict `FAIL_FINITENESS`. First nonfinite step `2`; preconditions OK; no hard anti-clamp failures; theta bound violations listed above.

`set -o pipefail; pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr021_wrf_smallstep.py tests/test_m6x_warm_bubble_operator_sanity.py -v 2>&1 | tee .agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_no_regression.txt`

Output summary: exit 1 with `pipefail`; pytest result `2 failed, 24 passed in 25.47s`. Passing subsets include R7 vertical acoustic oracle 3/3, ADR-023 column solver 4/4, c2 acoustic 8/8, MPAS column slice oracle 4/4, ADR-021 scratch tests 3/3, and two warm-bubble support tests. Failing tests are `test_warm_bubble_runs_finite_on_unified_path` (`first_nonfinite_step` is `2`) and `test_warm_bubble_extrema_reported_correctly` (300 s / 600 s samples are `None` after nonfiniteness).

`pytest tests/test_m3_transfer_audit.py -v 2>&1 | tee .agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_transfer_audit.txt`

Output summary: exit 0; `4 passed in 0.47s`. The fifth no-transfer-relevant check in the no-regression bundle, `test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks`, also passed.

## Proof Objects

- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_adr021_stripped.json`
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_adr021_stripped.txt`
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/proof_transfer_audit.txt`
- `.agent/sprints/2026-05-23-m6x-adr021-clamp-strip-honest-test/worker-report.md`

## Risks

- The validation command includes `tests/test_m6x_warm_bubble_operator_sanity.py`, so the no-regression proof records a pytest failure. That failure is the expected evidence for `FAIL_FINITENESS`, not a tuned-away regression.
- `_advance_mu_t_wrf` and `_mu_continuity_increment` still contain existing tanh mass-update limiters not tied to the `[5,10]` warm-bubble amplitude band; this sprint removed only the contract-identified clamp/aids.
- The sprint contract lists `No remote push` as a non-goal, while the later role launch instruction explicitly requires pushing `worker/gpt/m6x-adr021-clamp-strip-honest-test`. I followed the launch instruction after producing and committing proof objects.

## Handoff

Objective: strip ADR-021 warm-bubble clamp/aids and run the operator-sanity intel gate.

Files changed: listed above.

Commands run: listed above with output summaries; full combined stdout/stderr is in the proof `.txt` files.

Proof objects produced: listed above.

Unresolved risks: stripped ADR-021 is nonfinite at step 2; any ADR-021 promotion would require sourced stabilization research, not cleanup of this branch.

Next decision needed: manager should choose between staying on ADR-023, dispatching stabilization research that covers both architectures, or treating ADR-021's clamp-free path as rejected for warm-bubble operator sanity.
