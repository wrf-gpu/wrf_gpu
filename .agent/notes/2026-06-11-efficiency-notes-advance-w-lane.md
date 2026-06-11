# Efficiency notes — advance_w / acoustic lane (v0.14, 2026-06-11, fable)

Collected while building the WRF-native intra-`advance_w` oracle. These are
observations only; none were changed in this sprint unless stated in the
sprint review. Purpose: the project endpoint is a highly efficient GPU
rewrite — flag inefficiencies as they are found.

## 1. Per-substep `w_damp` inside `advance_w_wrf` (unfaithful placement + wasted work)

`src/gpuwrf/dynamics/core/advance_w.py` applies the `w_damping=1` vertical-CFL
limiter *inside* `advance_w_wrf`, i.e. **once per acoustic substep** with
`dt=dts` (`advance_w.py:310-323`). WRF applies it **once per RK stage** in
`rk_tendency` on the large-step `rw_tend` with the full `dt`
(`module_em.F:738`, `module_big_step_utilities_em.F:2686-2692`).

* Faithfulness: with `dts = dt/ns` the JAX activation threshold is effectively
  `ns`× stricter — proven inert at the Switzerland h36 RK1 substep (0 activated
  cells at dt=18 either way, see `proofs/v014/wrf_native_advance_w_dump.json`),
  but it would fire differently from WRF in stronger-CFL regimes.
* Efficiency: the limiter (abs/div/compare/where over the full 3-D face grid)
  runs `1+ns/2+ns` times per step instead of 3; it belongs in the once-per-stage
  `rw_tend` assembly in `operational_mode._acoustic_core_state_from_prep`
  (where `pg_buoy + tendencies.w` is already built). Moving it deletes
  ~7/3 of that work per step and removes 5 of the 8 kernel launches.

## 2. `safe_*` `jnp.where` floors in the hot acoustic path

`advance_w_wrf` floors `mass_h`, `mass_f_mut`, `mass_f_muts`, `theta_total_ref`
with `jnp.where(|x|>eps, x, eps)` on every substep. WRF divides directly; the
hybrid-coordinate column masses are bounded away from zero by construction
(`c1f*mut+c2f >= ptop`-scale). Four full-grid `abs`+`where` per substep are
pure overhead in production (XLA cannot drop them). If a guard is wanted it
should be a debug-mode branch (per the M4 debuggability-hooks convention), not
an unconditional hot-path op.

## 3. `t_2ave` denominator recomputed per substep

`mass_h = c1h*muts + c2h` is rebuilt (broadcast multiply+add) every substep
inside `advance_w_wrf`, but `muts` is constant across the acoustic loop for a
stage (WRF treats it as stage-constant). It can be hoisted to stage prep
alongside `a/alpha/gamma` (calc_coef_w already consumes the same quantity).
Same for `mass_f_mut`/`mass_f_muts`.

## 4. Two Thomas `lax.scan`s with `unroll=False`

The forward/backward sweeps in `advance_w_wrf` use `jax.lax.scan(...,
unroll=False)` over ~44 levels. On Blackwell, `unroll=8` (or a fully unrolled
fori) usually removes most of the scan-carry latency for nz≈45 and fuses with
the surrounding element-wise ops. Worth a profiled A/B in the next perf sprint
(no claim without profiler artifact, per GPU rules).

## 5. Proof-local CPU shim allocations

(Not production.) The h36 replay context builds the full operational state on
CPU twice per proof run (`_build_stage1` + jit validity). Caching the stage
context as an NPZ would cut proof iteration time ~3 min/run; debug-only, fine
to leave.

## 6. (FIXED this sprint) Redundant per-stage `diagnose_pressure_al_alt`

`operational_mode._acoustic_core_state_from_prep` recomputed the full
`p/al/alt` diagnostics at EVERY RK stage entry only to feed `p` to
`pg_buoy_w`, while the carry already held the same diagnostic
(`_refresh_grid_p_from_finished`, WRF `calc_p_rho_phi` cadence). Besides being
WRF-unfaithful at RK1 (the dominant h36 rw_tend error), this wasted a full
3-D diagnostics pass per stage (3 per step). Replaced with the carried
`state.p_perturbation` — one fewer full-grid kernel chain per stage.
