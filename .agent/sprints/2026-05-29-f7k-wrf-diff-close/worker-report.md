# F7K Worker Report — WRF-ground-truth diff + omega/scalar-transport close

**Status: `F7K_PARTIAL` (strong).** The F7J-localized warm-bubble under-translation
residual was **root-caused and fixed** (a theta mass-coupling cadence bug in the
acoustic small-step loop), objectively pinpointed by an in-pipeline diff. The
**Skamarock warm bubble now PASSES all 6 AC2 checks** (thermal_rise 1925 m, was
213 m). m4 regression 10/10. **Straka (AC3) still goes non-finite** — a separate,
documented stiff-regime residual. Per the F7K hard rule (fix pinpoints + closes
warm bubble but not Straka), mark PARTIAL with the divergence evidence and STOP.

## Objective

Build the definitive WRF-vs-JAX center-column per-substep diff on the pristine
WRF v4.7.1 `em_quarter_ss` ground truth, find the first/largest divergence to
pinpoint the broken operator, fix it WRF-faithfully, re-run the idealized cases.

## Root cause (the close)

The residual was **NOT** the omega `ww` formula (verified WRF-faithful vs
`calc_ww_cp`, `module_big_step_utilities_em.F:744-778`) nor the flux-form scalar
advection (verified vs `module_advect_em.F:4306-4333`). It was a **theta
mass-coupling cadence error**: `acoustic_substep_core`
(`core/acoustic.py:497`) **re-coupled the work theta from the (near-static)
perturbation theta on every acoustic substep and decoupled every substep**,
resetting the coupled work array each substep and discarding the accumulated
large-step tendency + vertical/horizontal transport. WRF couples ONCE per RK
stage (`small_step_prep`, `module_small_step_em.F:263`), advances the PERSISTENT
coupled `t_2` across all substeps (`advance_mu_t`, `:1141-1172`), and decouples
ONCE (`small_step_finish`). Net effect of the bug: theta advanced only ~1/`N_sound`
of correct → warm bubble rose 213 m not ~2000 m.

**Decisive objective localization** (`proofs/f7k/omega_scalar_transport_fix.md`):
at t=200 s the large-step flux-form advective `dθ/dt` and the actual integrated
one-step `dθ/dt` had **identical vertical profile/direction** but the actual was
**exactly 0.100× = 1/`acoustic_substeps` (=10)** at every level — a textbook
"correct operator, wrong cadence" signature. After fix, `theta_coupled_work`
accumulates monotonically across substeps (k11 z≈2875 m: 3.33→6.67→16.66→33.33
over substeps 1/2/5/10) and the one-step `dθ'` matches the correct rate (ratio 1.00).

## The fix (1 line, WRF-faithful, no masking)

`src/gpuwrf/dynamics/core/acoustic.py` `acoustic_substep_core`:
```python
- coupled_state = uv_state.replace(theta=_mass_couple_theta_before_advance(uv_state))
+ coupled_state = uv_state.replace(theta=uv_state.theta_coupled_work)
```
Advance the persistent stage-coupled work theta (coupled once in
`small_step_prep_wrf`, decoupled once in `small_step_finish_wrf`) instead of
re-coupling each substep. u/v/w/ph were never affected — only theta was reset.
No clamps/caps/diffusion-fudge/tolerance-widening.

## Files changed
- **M** `src/gpuwrf/dynamics/core/acoustic.py` (+13/-1; the 1-line fix + WRF
  citation comment).
- **NEW** `scripts/f7k_wrf_vs_jax_center_diff.py` — WRF-aligned IC builder +
  per-(rk,iteration) center-column diff harness.
- **NEW** `proofs/f7k/` artifacts.

## Commands run (CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src taskset -c 0-3, cuda:0, fp64)
- `scripts/f7k_wrf_vs_jax_center_diff.py` → `wrf_vs_jax_center_diff.json`.
- in-pipeline `dθ/dt` ratio probe (1/N_sound localization).
- per-RK3-substep `theta_coupled_work` accumulation trace (before/after).
- `run_warm_bubble_case(proof_dir='proofs/f7k')` → PASS 6/6.
- `run_density_current_case(proof_dir='proofs/f7k')` → FAIL (non-finite).
- Straka detonation-timing trace.
- `pytest tests/test_m4_acoustic.py test_m4_dycore_step.py test_m4_tier2_invariants.py` → 10 passed.

## Acceptance gates
- **AC1 (WRF center-column parity): NOT a clean parity.** The diff harness
  (`wrf_vs_jax_center_diff.json`) documents a real IC mismatch: WRF `em_quarter_ss`
  is 3-D, open-lateral-BC, stratified-sounding; the JAX gate is a periodic
  doubly-symmetric slab on a neutral base, and the re-derived stratified IC is not
  discretely consistent with the JAX `calc_p_rho_phi` perturbation split (p' came
  out ~100× off feeding WRF's exact IC through the JAX al/p relation — see
  `wrf_vs_jax_center_diff.json` note). The binding, objective localization was the
  in-pipeline `1/N_sound` ratio, not the cross-model column parity.
- **AC2 (warm bubble): PASS 6/6.** finite-500s; θ'max 1.92 K; max|w| 11.7 m/s;
  **thermal_rise 1925 m (≥500)**; drift 0; mass drift 0. (was 213 m FAIL.)
- **AC3 (Straka): FAIL (non-finite).** Separate stiff-regime residual; see
  `straka_density_current_diagnostics.json` + timing trace.
- **AC4 (no regression): PASS.** m4 10/10; all prior F7 fixes (F7D/F/H/I/J)
  untouched; no clamps/caps/diffusion-fudge.

## Unresolved risk / next decision
Straka detonates even with the corrected (now physically stronger) vertical
scalar transport. This is a **distinct** residual from the warm-bubble cadence
bug just fixed — likely a stiff-regime acoustic/CFL or diffusion-balance issue
specific to the dx=100 m, −15 K cold-pool configuration (the warm bubble, the
cleaner buoyancy test, is fully closed). Recommended next: a focused Straka
stability probe (acoustic CFL audit + WRF-faithful diffusion/divergence-damping
check) — NOT a diffusion fudge. The dynamical-core translation defect that has
blocked F7 is resolved; the warm bubble is the headline close.

F7K_PARTIAL
