# Hypothesis Notes

## Matched hypothesis

Matched: missing density / momentum-coupling decoupling in `horizontal_pressure_gradient`.

WRF `advance_uv` does not update plain velocity at the pressure-gradient line. `small_step_prep` first converts `u_2` and `v_2` into mass-coupled small-step momentum (`module_small_step_em.F:238-254`). `advance_uv` then applies `u -= dts*cqu*dpxy` and `v -= dts*cqv*dpxy` to that mass-coupled variable (`module_small_step_em.F:868,942`). The JAX `State.u` and `State.v` fields are velocities, so returning `-cqu*dpx` / `-cqv*dpy` as velocity tendencies missed the final division by the same face dry-column mass used in `dpxy`.

Fix: keep WRF `dpx`/`dpy` construction unchanged, but return velocity tendencies as:

- `du_dt = -cqu * dpx / (c1h*muu + c2h)`
- `dv_dt = -cqv * dpy / (c1h*muv + c2h)`

## Other hypotheses checked

- Sign error on `dv_dt`: WRF line `942` has the same negative pressure-gradient sign as `u` at line `868`; sign flip would not match source.
- Stagger mismatch: current interior pairings match WRF `p(i)-p(i-1)` at lines `828-831` and `p(i,j)-p(i,j-1)` at lines `902-905`.
- Missing `top_lid`: step-49 bad cell is not the top row; `top_lid=True` does not change the local fixture output.
- `dpn` face pressure: current bottom/interior construction follows WRF lines `836-851` and `910-925`; focused c2 PGF tests still pass.
- `cqu`/`cqv`: WRF applies these at final update lines `868` and `942`; the fixed velocity tendency still applies them there.

## Remaining risk

The HPG fixture regression passes and the old step-49 pressure explosion is not the first failure anymore, but the required guard-disabled replay still fails the contract stability gate with theta at step 18 inside `acoustic`. This is outside the HPG algebra fixed here and remains unresolved.
