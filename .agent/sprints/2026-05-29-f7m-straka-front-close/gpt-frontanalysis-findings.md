# F7M Straka Front Analysis Findings

## Objective

Identify the most likely operator/coupling defect behind the Straka descending-cold-front failure: excess vertical velocity at the head plus deficient cold-front propagation speed.

## Probe Evidence At The Front

Read-only CUDA/fp64 probes were run from `/tmp/wrf_gpu2_straka_audit` with `PYTHONPATH=src taskset -c 0-3`, `cuda:0`, `JAX_ENABLE_X64=1`, `nu=75`, `dt=0.1`, `acoustic_substeps=10`, and the Straka setup from `src/gpuwrf/ic_generators/idealized.py:275`.

The first-order signal is not missing low-level acceleration. At the front threshold, low-level `u` grows much faster than the cold-pool front advances:

| t (s) | theta front x (m) | last-20s front speed (m/s) | low-front max u (m/s) | domain low max u (m/s) | domain max \|w\| (m/s) | front-window max \|w\| (m/s) | front theta_min (K) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1250 | 5 | 1.41 | 1.46 | 2.73 | 1.16 | -14.98 |
| 80 | 1750 | 10 | 5.72 | 5.86 | 9.68 | 3.82 | -8.10 |
| 160 | 2250 | 5 | 12.48 | 14.06 | 19.22 | 6.50 | -5.63 |
| 200 | 2450 | 5 | 17.56 | 19.34 | 22.62 | 11.20 | -6.61 |
| 220 | 2550 | 5 | 20.68 | 23.57 | 23.49 | 14.15 | -6.74 |
| 240 | 2650 | 5 | 25.02 | 28.32 | 23.90 | 17.04 | -7.49 |
| 260 | 2750 | 5 | 30.98 | 32.76 | 23.86 | 20.10 | -8.60 |
| 280 | 3050 | 15 | 34.78 | 36.05 | 25.75 | 22.60 | -12.16 |

At `t=240 s`, the front is still only at `x=2650 m`, with an apparent 20 s speed of `5 m/s`, while the low-level front-window `u` is already `25.0 m/s`. The front-window `|w|` is climbing rapidly: `11.20 -> 14.15 -> 17.04 m/s` from `t=200` to `240 s`.

Horizontal pressure-gradient evidence does not support an under-forced-`u` diagnosis. The front-window large-step `u` PGF range evolves from positive forcing early to head-opposing pressure after the pile-up:

| t (s) | front-window u PGF min/max (m/s2) |
|---:|---:|
| 20 | `0.038 / 0.090` |
| 160 | `-0.036 / 0.100` |
| 200 | `-0.202 / 0.100` |
| 220 | `-0.351 / 0.066` |
| 240 | `-0.509 / 0.027` |
| 260 | `-0.690 / 0.009` |
| 280 | `-0.823 / -0.028` |

A separate read-only probe recomputed the large-step horizontal PGF with the suspected WRF-style `al/mass` denominator variant. In the front window it was numerically identical to the current JAX PGF at `t=200,220,240 s`:

| t (s) | current PGF min/max | corrected-al PGF min/max |
|---:|---:|---:|
| 200 | `-0.202348496387 / 0.099693137989` | `-0.202348496387 / 0.099693137989` |
| 220 | `-0.351006186935 / 0.066434947909` | `-0.351006186935 / 0.066434947909` |
| 240 | `-0.508820296487 / 0.027075425396` | `-0.508820296487 / 0.027075425396` |

The vertical buoyancy/PGF operator and lower boundary condition also do not look like the primary front defect. At `t=0`, `pg_buoy_w` is balanced to roundoff in the domain (`max_abs ~= 7.7e-13`), and the flat-terrain lower boundary in `src/gpuwrf/dynamics/core/advance_w.py:274` sets `w_surface = 0`, matching the WRF flat lower-boundary form in `dyn_em/module_small_step_em.F:1372`.

The scalar-limiter hypothesis is not supported by the Straka namelist. `/home/enric/src/wrf_pristine/WRF/test/em_grav2d_x/namelist.input.100m` uses `h_sca_adv_order = 5` and `v_sca_adv_order = 3`, with no `scalar_adv_opt` override; WRF therefore uses ordinary high-order scalar advection, not a monotonic scalar limiter, for this fixture. JAX is already routing theta through flux-form scalar advection at `src/gpuwrf/runtime/operational_mode.py:1089`.

## Single Most Likely Defect

The most likely cold-front defect is that JAX still advances momentum with the reduced primitive M4 advection operators, while WRF advances coupled momentum with mass-flux-form `advect_u`, `advect_v`, and `advect_w`.

JAX evidence:

- `src/gpuwrf/dynamics/advection.py:202` (`advect_u_face`) computes primitive `u * derivative5_upwind(u)` style advection on physical velocity.
- `src/gpuwrf/dynamics/advection.py:228` (`advect_w_face`) does the same for physical `w`.
- `src/gpuwrf/dynamics/advection.py:262` (`compute_advection_tendencies`) still returns these primitive `u/v/w` tendencies.
- `src/gpuwrf/runtime/operational_mode.py:1081` consumes those tendencies and later `rk_addtend_dry` couples them back into WRF-style momentum updates.

WRF comparison:

- WRF calls momentum advection from `dyn_em/module_em.F:493` for `advect_u`, `dyn_em/module_em.F:540` for `advect_v`, and `dyn_em/module_em.F:589` for `advect_w`.
- The WRF advection module uses flux operators such as `flux5`/`flux3` in `dyn_em/module_advect_em.F:199`.
- WRF's `advect_w` implementation starts at `dyn_em/module_advect_em.F:4364` and operates on coupled mass/momentum transports, not primitive `w * grad(w)` alone.

This fits the observed failure better than PGF, lower `w` BC, or scalar limiter. The model creates strong low-level outflow, but the scalar front only crawls forward while `u` piles up and drives excessive convergence/updraft at the head. A smooth warm bubble can pass while this remains wrong, because the Straka descending head is a much sharper momentum-transport/coupling test.

## Concrete Fix

Port the WRF dry, periodic/flat-terrain momentum advection path for `u`, `v`, and `w`, then replace the primitive `compute_advection_tendencies` momentum outputs in `_augment_large_step_tendencies`.

Implementation shape:

1. Add WRF-compatible momentum flux-form routines, preferably beside `src/gpuwrf/dynamics/flux_advection.py`, for the dry Straka subset first.
2. Build the same coupled transports used by WRF: `ru`, `rv`, and vertical `ww`/`wwE` from the coupled mass fields and velocities, using the existing coupling helpers rather than physical velocities alone.
3. Match `h_mom_adv_order = 5` and `v_mom_adv_order = 3` for this fixture, with periodic horizontal boundaries and no artificial boundary-order degradation in the interior front region.
4. Return coupled momentum tendencies directly, or clearly mark the units so `rk_addtend_dry` does not multiply/divide by mass a second time.
5. Leave the current theta `advect_scalar_flux` path in place unless a separate WRF-vs-JAX scalar budget probe contradicts it.

The critical replacement is not another scalar-front clamp. It is to stop feeding primitive `u/v/w` advection into a WRF coupled-momentum update.

## Falsifiable Check

After the fix, rerun the Straka fixture on CUDA/fp64 with `nu=75`, `dt=0.1`, `acoustic_substeps=10`, and the same diagnostics.

The fix is supported if:

- By `t=240 s`, the front-window low-level `u` is no longer more than roughly 5x the observed front speed while front-window `|w|` is still ramping upward.
- The `t=200 -> 240 s` front-window `|w|` sequence drops substantially from the current `11.20 -> 14.15 -> 17.04 m/s`.
- The cold-front speed after descent is closer to the gravity-current scale instead of remaining at repeated `5 m/s` increments while low-level `u` exceeds `25 m/s`.
- The run remains finite to `900 s`, with the front no longer suffering the documented speed deficit, and the warm-bubble acceptance suite still passes.

A stronger direct proof is a momentum-budget probe at the front: the new WRF flux-form `u/w` advection tendency should oppose the head pile-up where the current primitive operator permits accelerating convergence and rising `w`.

## Commands Run

- `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 JAX_ENABLE_X64=1 PYTHONPATH=src taskset -c 0-3 python scripts/f7l_straka_probe.py --end 20 --interval 20 --nu 75 --dt 0.1`
- `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 JAX_ENABLE_X64=1 PYTHONPATH=src taskset -c 0-3 python - <<'PY' ...` custom read-only front diagnostic to `280 s`
- `CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 JAX_ENABLE_X64=1 PYTHONPATH=src taskset -c 0-3 python - <<'PY' ...` custom read-only current-vs-corrected horizontal PGF diagnostic at `200/220/240 s`
- Read-only source inspection with `rg`, `sed`, `nl`, and `wc`.

F7M_FRONTANALYSIS_COMPLETE
