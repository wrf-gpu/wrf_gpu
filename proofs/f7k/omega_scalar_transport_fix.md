# F7K — root cause of the warm-bubble under-translation: per-substep theta re-coupling

**Status: the F7J-localized "omega / vertical-scalar-transport" residual is FIXED.**
The bug is a **theta mass-coupling cadence error in the acoustic small-step loop**,
objectively pinpointed by an in-pipeline diff (integrated `dtheta/dt` was *exactly*
`1/acoustic_substeps` of the correct large-step advective rate) and confirmed
against the WRF small-step source. One-line scoped fix; no clamps/caps/tuning.

## Symptom (F7J residual, re-measured)

Skamarock warm bubble, fp64, cuda:0, before fix:

| t (s) | peak θ′ z (m) | centroid z (m) | max\|w\| (m/s) |
|------:|--------------:|---------------:|---------------:|
| 0 | 1875 | 2000 | 0.00 |
| 100 | 2125 | 2009 | 2.99 |
| 200 | 2125 | 2034 | 5.95 |
| 300 | 2125 | 2077 | 8.94 |
| 500 | 2375 | 2213 | 14.8 |

The updraft `w` grew to ~15 m/s but the warm θ′ anomaly did **not translate** with
it — the peak stayed pinned near z≈2125 m (thermal_rise 213 m vs ≥500 m target).
"Deformation, not translation," exactly as F7J localized.

## Objective localization (the decisive diff)

At t=200 s I computed the large-step flux-form scalar tendency
(`advect_scalar_flux` with the continuity omega `rom` from
`couple_velocities_periodic`) at the bubble-center column and compared it to the
**actual** integrated `dθ/dt` over one full RK timestep:

```
k   z(m)   dθ/dt_largeadv    dθ/dt_ACTUAL     ratio
4  1125     -0.00615         -0.00061         0.100
5  1375     -0.00652         -0.00065         0.100
8  2125     +0.00089         +0.00009         0.104   <- peak
10 2625     +0.00711         +0.00071         0.101
11 2875     +0.00813         +0.00082         0.100
```

The advective **direction and vertical profile were perfectly correct** (cooling
below the peak, warming above → textbook upward translation; ratio constant), but
the **magnitude was exactly 0.100× = 1/`acoustic_substeps` (=10)** too small at
every level. The omega `rom` (verified WRF-faithful against `calc_ww_cp`,
`module_big_step_utilities_em.F:744-778`) and the flux-form `advect_scalar`
(verified vs `module_advect_em.F:4306-4333`, v_sca_adv_order=3) were both correct
— the defect was a **factor of N_sound applied to the accumulated theta**.

## Root cause (WRF file:line)

WRF couples the work potential temperature `t_2` to dry mass **ONCE per RK stage**
in `small_step_prep` (`module_small_step_em.F:263`):

```
t_2(i,k,j) = (c1h*muts+c2h)*t_1(i,k,j) - (c1h*mut+c2h)*t_2(i,k,j)
```

then advances that **persistent coupled array in place across every acoustic
substep** in `advance_mu_t` (`:1141-1172`: `t = t + msfty*dts*ft` plus the
omega/flux vertical+horizontal transport), and **decouples ONCE at the end** in
`small_step_finish` (`:295+`). Over `N_sound` substeps the work theta therefore
accumulates `N_sound * dts = dt_rk` worth of the large-step tendency + transport.

The JAX `acoustic_substep_core` (`core/acoustic.py:497`) instead **re-coupled the
work theta from the (nearly static) perturbation theta on EVERY substep**
(`_mass_couple_theta_before_advance`) and **decoupled on every substep**. So each
substep applied a fresh `dts`-worth of tendency, then threw away the accumulation
when it decoupled back to the perturbation theta. Trace of `theta_coupled_work` at
the bubble center across the 10 RK3 substeps, before fix (k11, z≈2875 m):

```
sub1: 6.079   sub2: -0.034   sub5: 6.011   sub10: -0.170   -> final dθ' ≈ 0.00000
```

It **oscillated and never accumulated** (only ~1 substep's worth survived), so the
net theta change was ≈`1/N_sound` of correct. u/v/w/ph are not affected — their
coupled work arrays already carry forward across substeps; only theta was reset.

## The fix (1 line, scoped)

`src/gpuwrf/dynamics/core/acoustic.py` `acoustic_substep_core`:

```python
# before:
coupled_state = uv_state.replace(theta=_mass_couple_theta_before_advance(uv_state))
# after (advance the PERSISTENT coupled work theta, coupled once at stage entry
# via prep.theta_work -> AcousticCoreState.theta_coupled_work):
coupled_state = uv_state.replace(theta=uv_state.theta_coupled_work)
```

The stage-entry coupling is already done WRF-faithfully in
`small_step_prep_wrf` (`theta_work = mass_h_ref*theta_ref - mass_h_cur*theta_cur`,
`core/small_step_prep.py:215`) and carried as `theta_coupled_work`; the final
decouple is the existing `small_step_finish_wrf`. The per-substep
`theta_coupled_work=theta_coupled` carry (`:646`) was already present, so the fix
just feeds it back as the next substep's coupled input instead of re-coupling.

After fix, `theta_coupled_work` accumulates monotonically (k11: 3.33 → 6.67 →
16.66 → 33.33 over substeps 1/2/5/10), and the final one-step `dθ'` matches the
correct large-step rate to ratio 1.00 (k11: 0.000449 actual vs 0.000446 expected).

## No masking

No clamps, caps, diffusion fudge, tolerance widening, or tuning. The change is a
WRF-faithful cadence correction (couple-once / advance-persistent / decouple-once),
identical to `module_small_step_em.F` `small_step_prep` → `advance_mu_t` (×N) →
`small_step_finish`.

## WRF-vs-JAX center-column note

`proofs/f7k/wrf_vs_jax_center_diff.json` records a per-(rk,iteration) center-column
attempt against the pristine WRF v4.7.1 `em_quarter_ss` savepoints. That harness is
**not the binding evidence** for this close: the WRF case is a 3-D, open-lateral-BC,
stably-stratified-sounding run, while the JAX gate is a periodic doubly-symmetric
slab on a neutral base, and the re-derived stratified IC is not discretely
consistent with the JAX `calc_p_rho_phi` perturbation split (documented IC mismatch,
see `wrf_vs_jax_center_diff.json` `note`). The decisive, objective localization was
the in-pipeline `1/N_sound` ratio above; the idealized-case verdicts (AC2/AC3) are
the binding gates.
