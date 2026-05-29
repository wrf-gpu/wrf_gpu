# GPT Bug Hunt Findings

## 1. What the probes showed

`ph` is not frozen inside the acoustic loop. The report that `max|ph_perturbation|` stays at 131.827 Pa-m2/s2 is a misleading max statistic: the initial hydrostatic warm-bubble perturbation remains the largest value, while the live work geopotential changes every substep and is carried.

Warm-bubble, first physical timestep, cuda:0, fp64, `PYTHONPATH=src taskset -c 0-3`:

| RK stage/substep | max\|w_work\| | max\|ph_work\| | max\|ph_perturbation\| | max\|p_work\| |
| --- | ---: | ---: | ---: | ---: |
| initial physical | 0.000000e+00 | n/a | 1.318273828e+02 | 4.365574569e-11 |
| RK1 sub1 | 1.285156999e-02 | 2.764656743e-03 | 1.318273828e+02 | 1.766017551e-02 |
| RK2 entry | n/a | 2.764656743e-03 | 1.318273828e+02 | 1.605469770e-02 |
| RK2 sub1 | 1.148030e-02 | 3.345231561e-03 | 1.318273828e+02 | 1.533234849e-02 |
| RK2 sub2 | 9.736707e-03 | 3.676961991e-03 | 1.318273828e+02 | 1.319532342e-02 |
| RK2 sub3 | 7.261942e-03 | 3.759833887e-03 | 1.318273828e+02 | 1.937592632e-02 |
| RK2 sub4 | 4.043387e-03 | 3.593840773e-03 | 1.318273828e+02 | 2.554402365e-02 |
| RK2 sub5 | 1.118981e-03 | 3.178982241e-03 | 1.318273828e+02 | 3.167213729e-02 |
| RK3 entry | n/a | 4.143254978e-04 | 1.318273828e+02 | 3.092419428e-02 |
| RK3 sub1 | 2.060505e-03 | 5.810428725e-04 | 1.318273828e+02 | 2.064348659e-02 |
| RK3 sub2 | 5.046700e-03 | 1.327488905e-03 | 1.318273828e+02 | 1.586316041e-02 |
| RK3 sub3 | 7.292268e-03 | 1.824984645e-03 | 1.318273828e+02 | 1.230617229e-02 |
| RK3 sub4 | 8.787616e-03 | 2.073509204e-03 | 1.318273828e+02 | 7.573015943e-03 |
| RK3 sub5 | 9.531119e-03 | 2.073048395e-03 | 1.318273828e+02 | 1.652983976e-02 |
| RK3 sub6 | 9.523017e-03 | 1.823595118e-03 | 1.318273828e+02 | 2.575390958e-02 |
| RK3 sub7 | 8.763009e-03 | 1.325148981e-03 | 1.318273828e+02 | 3.493918866e-02 |
| RK3 sub8 | 7.251121e-03 | 5.777166871e-04 | 1.318273828e+02 | 4.408744059e-02 |
| RK3 sub9 | 5.004685e-03 | 4.186883460e-04 | 1.318273828e+02 | 5.318925837e-02 |
| RK3 sub10 | 2.018800e-03 | 1.664045599e-03 | 1.318273828e+02 | 6.224643873e-02 |

The source path also shows the carry is live:

- `src/gpuwrf/dynamics/core/advance_w.py:424-434` updates `ph_next` from the solved `w`.
- `src/gpuwrf/dynamics/core/acoustic.py:620-634` feeds that `ph_next` into `calc_p_rho_step`.
- `src/gpuwrf/dynamics/core/acoustic.py:639-650` returns `ph=ph_next` and `p=p_rho.p` in the scan carry.

Longer warm-bubble probe, current code, showed the physical geopotential is changing while the pressure diagnostic remains far too small:

| time | max\|w\| | theta prime max | max\|state.p_perturbation\| | max\|delta ph_perturbation\| |
| ---: | ---: | ---: | ---: | ---: |
| 0 s | 0.000000e+00 | 2.000000 | 4.365575e-11 | 0.000000e+00 |
| 10 s | 1.512619e-01 | 2.000369 | 4.542883e+00 | 1.176837e-01 |
| 50 s | 8.778600e+00 | 2.007933 | 7.594635e+00 | 5.540896e-01 |
| 100 s | 4.467220e+01 | 2.017800 | 7.239664e+00 | 1.503175e+00 |
| 150 s | 1.056720e+02 | 2.009563 | 7.958498e+00 | 2.726015e+00 |
| 200 s | 1.718553e+02 | 2.008259 | 1.731212e+01 | 8.486338e+00 |

The decisive pressure probe compared carried `state.p_perturbation` to the WRF `calc_p_rho_phi` pressure implied by the same live physical `ph_perturbation` and theta. They agree at initialization, then diverge by kilopascals:

| time | max\|state.p_perturbation\| | max\|calc_p_rho_phi p'\| | max\|difference\| | max\|w\| |
| ---: | ---: | ---: | ---: | ---: |
| 0 s | 4.365575e-11 | 2.910383e-11 | 5.820766e-11 | 0.000000e+00 |
| 10 s | 4.542883e+00 | 5.547318e+02 | 5.501889e+02 | 1.512619e-01 |
| 20 s | 5.993590e+00 | 1.592908e+03 | 1.586915e+03 | n/a |
| 50 s | 7.594635e+00 | 5.509939e+03 | 5.502344e+03 | 8.778600e+00 |
| 100 s | 7.239664e+00 | 1.216948e+04 | 1.216248e+04 | 4.467220e+01 |

I also found one real secondary carry bug in theta work: after one RK1 acoustic substep, the next-substep recoupling formula in `src/gpuwrf/dynamics/core/acoustic.py:431-437` differs from the actual carried `theta_coupled_work` by `6.537688778132e-02` absolute, while the exact WRF inverse reconstruction differs by only `2.103516666407e-11`. A read-only monkeypatch that preserved `theta_coupled_work` made theta transport active, but did not materially change `max|w|` through 100 s (`44.70668` patched vs `44.67220` current). That makes it a real bug, but not the single most likely cause of this linear `w` runaway.

## 2. Single most likely root cause

The root cause is that the physical-state perturbation pressure used by the next large-step pressure-gradient and vertical `pg_buoy_w` path is stale/wrong. The acoustic small-step loop evolves `ph` correctly, but the RK/stage boundary carries `p_perturbation = acoustic_out.p`, which is the small-step work pressure from `calc_p_rho_step`, not the WRF `calc_p_rho_phi` diagnostic pressure recomputed from the finished physical `ph_perturbation` and theta.

File/line evidence:

- `src/gpuwrf/dynamics/core/small_step_finish.py:37-40` reads `ph_work` and `p_perturbation` directly from the acoustic output.
- `src/gpuwrf/dynamics/core/small_step_finish.py:56-73` adds `ph_save` to produce the finished physical `ph_perturbation`, but returns the acoustic work `p_perturbation` unchanged as `state.p_perturbation`.
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py:93-125` then treats `state.p_perturbation` as WRF `grid%p` for large-step diagnostics.
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py:165-213` uses that stale `p_abs` in horizontal PGF terms.
- `src/gpuwrf/runtime/operational_mode.py:678-687` builds the stage `rw_tend_stage` for `pg_buoy_w` from `pressure.p`; in this path the pressure remains the small-step/work diagnostic rather than a refreshed physical `calc_p_rho_phi` pressure.
- WRF ground truth recomputes `al` and `p` from physical `ph` and theta in `calc_p_rho_phi`: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_big_step_utilities_em.F:1023-1029` for `al`, and `:1082-1087` for `p`.

This matches the symptom better than a frozen-`ph` hypothesis: mass is conserved and `ph` does respond, but the restoring pressure feedback seen by the next-stage PGF/`pg_buoy_w` is suppressed from O(10^3-10^4 Pa) to O(1-10 Pa), leaving an effectively constant net vertical force.

## 3. Concrete fix

Refresh WRF physical pressure diagnostics at the RK/stage boundary from the finished state, not from the acoustic work pressure:

1. After `small_step_finish_wrf` has reconstructed physical `theta`, `ph_perturbation`, and `mu_perturbation`, run the WRF `calc_p_rho_phi` equivalent on that finished physical state using the explicit base fields (`pb`, `phb`, `mub`, `theta_base`) and metrics.
2. Store the resulting `p'` back into `state.p_perturbation`, `state.p_total`, and any carried `al`/`alt` diagnostics needed by later kernels.
3. Ensure `_absolute_diagnostics` and `_acoustic_core_state_from_prep` consume this refreshed WRF `grid%p` for large-step horizontal PGF and once-per-stage `pg_buoy_w`, while the acoustic substep may still use `calc_p_rho_step` for its work-array pressure/smdiv memory.
4. Separately fix the proven theta-work carry bug by preserving `state.theta_coupled_work` across acoustic substeps, or reconstructing the next coupled theta work with the exact inverse of `small_step_finish` (`muts_coef * theta_phys - mut_coef * theta_1`). Do not use `muts_coef * theta_1 - mut_coef * theta_phys`.

## 4. Falsifiable check

The pressure-refresh fix is confirmed only if both checks pass:

1. Repeat the pressure diagnostic probe. At every RK stage and every 10 s warm-bubble sample, `max|state.p_perturbation - calc_p_rho_phi(state.ph_perturbation, state.theta)|` must stay near floating-point/truncation tolerance. It must not grow to `5.50e+02 Pa` at 10 s or `1.216e+04 Pa` at 100 s.
2. Repeat the warm-bubble integration to at least 200 s and 500 s. `max|w|` must stop the observed near-linear growth (`0.151` at 10 s, `8.78` at 50 s, `44.67` at 100 s, `171.86` at 200 s), while `max|delta ph_perturbation|` and refreshed `p_perturbation` co-evolve.

For the secondary theta carry fix, the one-substep invariant is: the next-substep coupled theta input must match the previous `theta_coupled_work` to about `1e-10`; the current `6.537688778132e-02` mismatch must vanish.

F7H_BUGHUNT_COMPLETE
