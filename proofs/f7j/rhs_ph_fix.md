# F7J — `rhs_ph`/`ph_tend` large-step geopotential RHS + `advect_w` into `rw_tend`

**Status: F7J_PARTIAL.** The prime-suspect operator was real and is now
implemented WRF-faithfully. It **eliminates the e-fold-29s buoyancy-driven
vertical standing mode** (AC1 achieved) and turns the warm bubble from a
detonate-at-180s FAIL into a finite-to-500s run that passes 5/6 AC2 checks.
It does **not** fully close AC2 (thermal rise 213 m vs ≥500 m) or AC3 (Straka
detonates at ~240 s vs <40 s for the wrong reading). A separate, localized
residual remains in the **theta↔omega vertical-transport coupling**, not in the
w/phi restoring loop. Per the F7J hard rule, STOP for manager review.

## What was stubbed (the bug)

`ph_tend` was initialised to zero (`operational_state.py:94`) and **never
updated**: `accumulate_ph_tend` (`small_step_scratch.py:66`) was a
validation-only scratch helper, never the real operator. The acoustic core fed
`ph_tend=carry.ph_tend` (always 0) into `advance_w_wrf`
(`core/acoustic.py:570`). So the large-step geopotential-equation RHS — the
explicit, frozen-during-the-acoustic-loop half of the w/phi restoring loop —
contributed nothing. The buoyancy pumped `w` without the geopotential
saturating it.

Separately, WRF's `rk_tendency` builds `rw_tend = advect_w(w) + pg_buoy_w(...)`
(`module_em.F:1011-1067` then `:1361-1368`); the JAX `rw_tend_pg_buoy` carried
**only** `pg_buoy_w`. The large-step w advection (`tendencies.w`, already
computed and coupled in `_augment_large_step_tendencies`) was never folded in.

## The fix (WRF file:line)

New module `src/gpuwrf/dynamics/core/rhs_ph.py` — `rhs_ph_wrf`, a faithful
translation of WRF `rhs_ph` (`module_big_step_utilities_em.F:1365-1612`) for the
idealized doubly-periodic gate (unit map factors, `phi_adv_z==1`,
`non_hydrostatic==True`, 2nd-order horizontal advection):

| WRF term | WRF lines | JAX |
| --- | --- | --- |
| Term 3: `-omega d/d_eta(phi)` (destaggered, phi_adv_z==1) | `:1500-1518` | `rhs_ph.py` `wdwn_mass` + `fnm/fnp` restagger |
| Term 4: `+(c1f*mut+c2f) g w / my` ("gw", non-hydrostatic; `ph_tend(kde)=0`) | `:1522-1540` | `rhs_ph.py` `gw`, top face zeroed |
| Terms 1,2: `-(1/my) rdx/rdy * muuf/muvf * (u/v) * d(phi)` (2nd order) | `:1546-1612` | `rhs_ph.py` `flux_x`/`flux_y` (periodic `roll`) |

Wiring in `src/gpuwrf/runtime/operational_mode.py`
(`_acoustic_core_state_from_prep`):
- `rw_tend_stage = pg_buoy_w_dry(...) + tendencies.w` (advect_w fold, item 2).
- `ph_tend_stage = rhs_ph_wrf(u=state.u, v=state.v, ww=carry.ww,
  ph=state.ph_perturbation, phb=ph_base, w=state.w, mut=prep.mut, muu=prep.muu,
  muv=prep.muv, ...)`, computed **once per RK stage** from the stage state and
  the stage explicit omega `carry.ww` (= WRF `wwE = grid%ww`, `solve_em.F:762`),
  then carried unchanged through the acoustic loop (matches WRF rk_tendency
  cadence). Returned `AcousticCoreState.ph_tend = ph_tend_stage` (was
  `carry.ph_tend` = 0).

## Why this is NOT a double count (it is the WRF time split)

WRF zeroes `ph_tend` (`module_em.F:651`) and `rhs_ph` accumulates **all four**
terms with the **stage (large-step, frozen) omega `wwE`** and stage `ph`.
`advance_w` then re-adds term-3/term-4 contributions using the **small-step
evolving omega `ww`** and the **reference geopotential `ph_1`**
(`module_small_step_em.F:1357-1382`, `:1345`, `:1583`). Different fields ⇒ both
required. **Empirically confirmed:** Straka with `rhs_ph` terms 1,2-only
(the literal advance_w self-comment reading) detonates at **40 s**; with the
full four-term `rhs_ph` it survives to **240 s** — the full operator is
decisively more stabilising, so it is not a double count.
(`scripts/f7j_rhs_ph_mode_probe.py`.)

## Evidence

### AC1 — standing mode eliminated (`center_column_w_trace_after.json`)
Warm-bubble bubble-center column (ic=40), fp64, cuda:0:

| t (s) | BEFORE max\|w\| (F7I) | BEFORE sign_alt | AFTER max\|w\| | AFTER sign_alt |
|------:|---------------------:|----------------:|---------------:|---------------:|
| 100 | 4.386 | 0.075 | 2.99 | 0.000 |
| 150 | 17.27 | 0.075 | (n/a) | — |
| 170 | 26.55 | 0.075 | (n/a) | — |
| 180 | **NaN** | — | finite | — |
| 200 | — | — | 5.98 | ~0.000 |
| 600 | — | — | 17.69 | 0.000 |

BEFORE: exp growth e-fold≈29 s, fixed multi-lobe structure (`sign_alt` pinned
0.075), DOWNWARD center w, NaN at 180 s. AFTER: smooth single-lobe **upward** w
peaking at the bubble center, `sign_alt≈0` (no checkerboard), finite past 600 s,
growth **linear** (`max|w| ~0.9→17.7` over 30→600 s ≈ 20× over 20× time), not
exponential. The 2Δz/standing mode is GONE.

### AC2 — warm bubble 5/6 PASS (`skamarock_bubble_diagnostics.json`)
| Check | Value | Threshold | Pass |
| --- | ---: | --- | :-: |
| all_snapshots_finite | finite to 500 s | finite | ✅ |
| theta_prime_max_500s | 2.016 K | 0.5–2.5 | ✅ |
| max_abs_w_500s | 14.81 m/s | 1–30 | ✅ |
| horizontal_drift_500s | 1.8e-12 m | ≤250 | ✅ |
| relative_mass_drift | 0.0 | ≤1e-8 | ✅ |
| **thermal_rise_500s** | **213 m** | **≥500** | ❌ |

### AC3 — Straka FAIL but improved (`straka_density_current_diagnostics.json`)
Finite + steady max\|w\| growth 2.7→24.3 m/s through 220 s, then NaN at 240 s
(was non-finite far earlier; terms-1,2-only detonates at 40 s). Still
detonates ⇒ FAIL.

### AC4 — no regression (`regression_recheck.json`)
- `tests/test_m4_acoustic.py test_m4_dycore_step.py test_m4_tier2_invariants.py`
  → **10 passed** (twice).
- **flat-rest = exactly 0**: `rhs_ph_wrf` on a balanced rest column returns
  `max|ph_tend| == 0.0` (no spurious forcing).
- grid%p-refresh / full-p pg_buoy_w (F7D/F7H) preserved; **no clamps/caps/
  diffusion-fudge added**; production change limited to the `rhs_ph` operator +
  the `advect_w` fold.

## Residual (localized for the manager / WRF-ground-truth agent)

The w/phi restoring loop now closes (mode gone, bubble rises coherently). The
remaining shortfall is in **vertical scalar transport**: with healthy
`max|w|≈15 m/s` the warm anomaly advects only ~200 m in 500 s. The center-column
diagnostic shows the flux-advection omega `rom`
(`couple_velocities_periodic`, from horizontal mass-flux divergence) is
sign-consistent with `w` but the theta' blob barely moves — the
prognostic-`w` ↔ continuity-`omega` ↔ scalar-transport coupling is too weak.
In Straka's much stronger regime (15 K cold pool, dx=100 m) the same
weak/inconsistent coupling lets `max|w|` ramp unbounded to detonation. This is
a transport/continuity-consistency issue, distinct from the (now-fixed)
geopotential restoring loop — the natural next WRF-ground-truth diff target.
