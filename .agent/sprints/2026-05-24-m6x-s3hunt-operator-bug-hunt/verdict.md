# Verdict

NO-BUG-LOCALIZED

The sanitizer-bypass replay reproduced the catastrophic short-run failure: first nonfinite step is 2, first nonfinite field is `u`, and the localized stage is `post-recurrence`. The first guard-limit hit appears earlier at step 1, so the 10-step sanitizer-off acceptance bar is not met (`first_nonfinite_step_null=false`, `no_fields_on_caps=false`).

Stage 2 ran seven one-suspect buckets. None moved the first nonfinite beyond step 2 or produced a 10-step sanitizer-off pass:

| Suspect | Variants | Result |
|---|---|---|
| MPAS recurrence sign check | `mpas_recurrence_cofwr_sign_flip` | first nonfinite step 2 |
| `_mu_continuity_increment` | `dmu=0`, raw unbounded `dmu` | first nonfinite step 2 |
| `_mpas_w_metric_faces` | fixed center-column reference metric | first nonfinite step 2 |
| `n_acoustic` sweep | 1, 4, 8, 16 | first nonfinite step 2 |
| Physics disable | dycore-only after initial state | first nonfinite step 2 |
| Boundary disable | skip lateral boundary application | first nonfinite step 2 |
| Branch verification | force positive pressure branch | first nonfinite step 2 |

Coefficient sanity on the center d02 column did not expose a scalar coefficient blow-up: all dumped coefficients were finite, the metric was finite-positive, `tri_b` was positive/nonzero, and weak diagonal dominance held.

Recommendation: write `M6-DYCORE-BLOCKER-MEMO` or dispatch a design sprint for the broader dycore recurrence/state shape. This sprint did not find a single local operator toggle that justifies an implementation fix.

