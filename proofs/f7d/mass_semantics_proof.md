# F7D Mass-Semantics Proof — MUT/MUTS total-mass fix

## The bug (verified against WRF source)

WRF's `grid%mut` is the **full stage-entry dry mass** `MUB + MU_current`
(`calculate_full` in `rk_step_prep`, `dyn_em/module_em.F:184-187`;
`module_big_step_utilities_em.F:3912-3916`).  `grid%muts` is the full
small-step total `MUT + MU_work` (`module_small_step_em.F:1102-1107`).

`calc_p_rho` is called with `grid%mu_2` (the perturbation/work mu) as its `mu`
argument and **`grid%muts`** as its `Mut` (total-mass denominator) argument
(`solve_em.F:2628-2635` at stage entry, `solve_em.F:4164-4171` each substep).
`calc_coef_w` is called with `grid%mut` (`solve_em.F:2676-2681`); `advance_w`
with `grid%mut` (current) + `grid%muts` (live) (`module_small_step_em.F:1178-1185`).

Before this sprint the JAX `SmallStepPrepState.mut` was `MUB` (base dry mass),
so `calc_p_rho` / `calc_coef_w` / `advance_w` were fed the wrong (base) total
mass — the acoustic restoring loop used the wrong denominator.

## The fix (`src/gpuwrf/dynamics/core/small_step_prep.py`)

| field      | before (BUG)          | after (WRF-faithful)                       | WRF ref |
|------------|-----------------------|--------------------------------------------|---------|
| `prep.mub` | (did not exist)       | `MUB`                                       | INTENT(IN) `mub` |
| `prep.mut` | `MUB`                 | `MUB + MU_current`  (= `grid%mut`)          | `module_em.F:184-187` |
| `mu_work`  | `0` / `MU_ref-MU_cur` | `0` (RK1) / `MU_ref-MU_current` (else)      | `module_small_step_em.F:187-190,213-214` |
| `prep.muts`| `MUB + mu_work`       | `mut + mu_work` = `MUB + MU_ref`            | `module_small_step_em.F:172-175,196-199` |
| `muu/muv`  | face-avg(MUB+MU_cur)  | face-avg(`prep.mut`)                        | `:172-207` |
| `muus/muvs`| face-avg(MUB+MU_ref)  | face-avg(`prep.muts`)                       | `:172-207` |

The θ/u/v/w work arrays are then built from the full current/stage mass pairs
(`(c1h*muts+c2h)*ref - (c1h*mut+c2h)*cur`), matching `module_small_step_em.F:238-276`.

## calc_p_rho call-site fix (`src/gpuwrf/dynamics/core/calc_p_rho.py`, `acoustic.py`)

- `_calc_al_p` parameter `mut` renamed `muts_total`; numerator kept as the work
  variables (`mu_work`, `ph_work`, `theta_work`) — **not** made absolute.
- `calc_p_rho_wrf` feeds `prep.muts` (was `prep.mut`).
- `acoustic_substep_core` feeds the **live** `muts_new` from `advance_mu_t` into
  `calc_p_rho_step` (was `uv_state.mut`), matching `solve_em.F:4164-4171`.
- `advance_w` receives `mut=uv_state.mut` (now full stage-entry) + `muts=muts_new`
  (live); the WRF split for `mut` (RHS / coef denominators) vs `muts` (t_2ave /
  geopotential finish) is preserved (`module_small_step_em.F:1343,1395,1584`).
- `pg_buoy_w` vertical-buoyancy split kept unchanged (frozen stage-entry absolute
  `p_buoy`), per `module_em.F:1361-1368`.

## Verification (numeric, fp64, cuda:0) — `proofs/f7d/rk1_source_parity.json`

Synthetic nonzero-`mu_perturbation`, reference≠state (RK2 pattern):

```
max_abs( mut  - (mub + mu_current) )  = 0.0   (machine zero)
max_abs( muts - (mub + mu_ref) )      = 0.0
max_abs( mu_work - (mu_ref-mu_current)) = 0.0
max_abs( (muts - mut) - mu_work )     = 0.0   (WRF MUTS = MUT + MU_work identity)
```

The idealized warm/cold-bubble ICs encode the perturbation in θ + ph
(`mu_perturbation == 0`), so for those `mut == mub` numerically while the
semantic `mut = MUB + MU_current` holds (verified above on the synthetic case).
