# F7J Worker Report — implement `rhs_ph`/`ph_tend` + `advect_w`, close idealized cases

**Status: `F7J_PARTIAL`.** The prime-suspect operator was real and is now
implemented WRF-faithfully. It **eliminates the e-fold-29s buoyancy-driven
vertical standing mode** (AC1 ✅), and turns the warm bubble from a
detonate-at-180s FAIL into a finite-to-500s run passing **5/6** AC2 checks. It
does **not** fully close AC2 (thermal rise 213 m vs ≥500 m) or AC3 (Straka
detonates ~240 s). The residual is now in a **different operator** (theta↔omega
vertical-transport coupling), not the w/phi restoring loop. STOP per the F7J
hard rule.

## Objective
Implement the stubbed large-step geopotential-equation RHS `rhs_ph`/`ph_tend`
(prime suspect: it closes the w/phi restoring loop so the warm-bubble buoyancy
saturates) and fold `advect_w` into `rw_tend` per WRF; verify the warm-bubble
standing mode saturates and close Skamarock + Straka.

## Files changed
- **NEW** `src/gpuwrf/dynamics/core/rhs_ph.py` — `rhs_ph_wrf`, WRF-faithful
  geopotential-equation RHS (terms 1,2 horizontal phi advection 2nd order;
  term 3 `-omega d/d_eta phi` phi_adv_z==1; term 4 `gw` non-hydrostatic).
  Source: WRF `module_big_step_utilities_em.F:1365-1612`.
- **M** `src/gpuwrf/runtime/operational_mode.py` (`_acoustic_core_state_from_prep`):
  `rw_tend_stage += tendencies.w` (advect_w fold, item 2); `ph_tend_stage =
  rhs_ph_wrf(...)` once per RK stage from the stage state + stage omega
  `carry.ww` (= WRF `wwE=grid%ww`); `AcousticCoreState.ph_tend = ph_tend_stage`
  (was `carry.ph_tend`=0). +43/-1 lines, scoped.
- **NEW** `scripts/f7j_rhs_ph_mode_probe.py` — full-rhs_ph vs terms-1,2-only
  decision probe (never weakens invariants).
- **NEW** proofs under `proofs/f7j/`.

## Commands run (all `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src taskset -c 0-3`, cuda:0, fp64)
- `python -u scripts/f7i_center_column_w_trace.py --steps {200,2000,6000} --stride {20,100,300}` (before/after standing-mode trace)
- `python -c "run_warm_bubble_case(proof_dir='proofs/f7j')"` → 5/6 PASS, RAN_TO_COMPLETION
- `python -c "run_density_current_case(proof_dir='proofs/f7j')"` → FAIL, RAN_TO_COMPLETION (NaN ~240s)
- `python -u scripts/f7j_rhs_ph_mode_probe.py --case straka --gw {full,h12}` (term-selection decision)
- Straka detonation-time trace (stride 200) → NaN at 240 s
- theta-flux-tendency vs `-w·dθ/dz` consistency probe → matches to 3 sig figs
- flat-rest `rhs_ph_wrf` probe → `max|ph_tend| == 0.0`
- `pytest tests/test_m4_acoustic.py test_m4_dycore_step.py test_m4_tier2_invariants.py` → 10 passed (×2)

## Proof objects (`proofs/f7j/`)
- `center_column_w_trace_after.json` — AFTER: smooth upward single-lobe, finite
  past 600 s, `sign_alt≈0`, growth linear (mode gone). BEFORE = F7I
  `center_column_w_trace.json` (exp e-fold 29 s, NaN 180 s).
- `skamarock_bubble_diagnostics.json` + `skamarock_bubble_verdict.md` — AC2 5/6.
- `straka_density_current_diagnostics.json` + `straka_density_current_verdict.md` — AC3 FAIL.
- `rhs_ph_fix.md` — full WRF file:line + before/after + double-count refutation.
- `regression_recheck.json` — AC4 10/10 + flat-rest=0 + no clamps/caps.

## Acceptance gates
- **AC1 (standing mode gone): ✅.** e-fold-29s exponential mode eliminated.
  Warm-bubble center w now smooth, upward, single-lobe; `sign_alt≈0`; finite
  past 600 s; growth linear (`max|w| 0.9→17.7` over 30→600 s ≈ 20× / 20×).
- **AC2 (warm bubble): 5/6 PASS, FAIL overall.** finite-500s ✅; max|w|=14.8 ✅;
  θ'max=2.02 K ✅; drift 1.8e-12 ✅; mass drift 0 ✅; **thermal_rise=213 m ❌**
  (≥500 needed). Was a detonate-at-180s hard FAIL before.
- **AC3 (Straka): FAIL.** finite + steady `max|w|` ramp 2.7→24.3 then NaN at
  240 s. Full rhs_ph decisively better than terms-1,2-only (240 s vs 40 s).
- **AC4 (no regression): ✅.** m4 10/10 (×2); flat-rest=0; grid%p-refresh /
  full-p pg_buoy_w preserved; **NO clamps/caps/diffusion-fudge**; change limited
  to the WRF operator + advect_w fold.

## Root cause confirmed + double-count refuted
`ph_tend` was init 0 and never updated (`accumulate_ph_tend` was a
validation-scratch stub). The large-step geopotential RHS — the explicit,
frozen half of the w/phi restoring loop — contributed nothing, so buoyancy
pumped w without the geopotential saturating it. Implementing the real four-term
`rhs_ph` closes the loop and kills the mode. The apparent term-3/term-4
double-count vs `advance_w` is NOT a double count: WRF's `rhs_ph` uses the
stage (frozen) omega `wwE`+stage `ph`; `advance_w` re-adds with the small-step
evolving omega `ww`+reference `ph_1` — different fields, both required.
Empirically: terms-1,2-only detonates Straka at 40 s, full at 240 s.

## Unresolved risk / next decision needed
A **separate, localized** residual remains: with healthy `max|w|≈15 m/s` the
warm anomaly advects only ~200 m in 500 s, and Straka's stronger regime ramps
`max|w|` to detonation. The flux-advection omega `rom`
(`couple_velocities_periodic`) is sign/magnitude-consistent with `w` and the
theta-flux tendency matches `-w·dθ/dz` to 3 sig figs — yet the bubble
under-translates. This points to the **prognostic-`w` ↔ continuity-`omega` ↔
scalar-transport** consistency (deformation vs translation), distinct from the
now-fixed geopotential loop. **Manager decision:** route the WRF-ground-truth
`em_quarter_ss` center-column diff at this residual (omega/continuity + scalar
vertical transport), not the implicit w/ph solve (cleared F7I) nor the
geopotential RHS (fixed F7J). Keep the F7J `rhs_ph` + advect_w fix — it is
WRF-correct, regression-safe, and the mode-killer.

F7J_PARTIAL
