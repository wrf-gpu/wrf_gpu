# M6 Boundary / Dynamics Audit — Tester Report

**Sprint:** `2026-05-26-m6-boundary-dynamics-audit`
**Branch:** `tester/opus/m6-boundary-dynamics-audit`
**Role:** tester (Opus 4.7, sonnet-test-engineer slot)
**Driver:** `scripts/m6_boundary_dynamics_audit.py` (NEW, read-only)
**Date:** 2026-05-26 UTC

## Objective

Determine whether the unresolved risk the microphysics-feedback worker flagged — "finite but physically absurd pressure/wind excursions in dynamics/boundary fields" — is acceptance-bounded (D), or whether the operational forecast is fundamentally broken in p / u / w (A, B, or C).

## Methodology

Per the sprint contract, ran the operational forecast (`run_forecast_operational`) on each of the 3 V3 ICs for 1h (360 steps × dt=10s), capturing per-step max/min/abs_max of `p_perturbation`, `p_total`, `p`, `u`, `v`, `w`, plus boundary-ring (first 5 cells from any horizontal edge) vs interior abs_max for `p_total` each step. Then compared per-step extrema to Gen2 wrfout extrema at lead 1h (best truth available — hourly archive).

Physical-reason envelope used for classification:

| field            | 1× envelope | rationale                                                        |
|------------------|-------------|------------------------------------------------------------------|
| p_perturbation   | 5 kPa       | strong storms peak at ~1–2 kPa; 5 kPa is a generous WRF cap      |
| p_total          | 2 × 10⁵ Pa  | sea-level pressure ~ 1.1 × 10⁵ Pa, top ~50 Pa                    |
| u, v             | 80 m/s      | jet-stream cap                                                   |
| w                | 10 m/s      | deep convective cap (stratiform << 1 m/s)                        |

Verdict bands (per contract):
- **PASS_A** ≤ 1× envelope (physically reasonable)
- **PASS_D** ≤ 2× (absurd-looking but acceptance-bounded)
- **COND_2_10X** ≤ 10× (bounded but physically suspect)
- **FAIL_B_OR_C** > 10× (dynamics or boundary defect)

Source localization at *onset* (the step where |p′|_max first crosses 5 × 10⁴ Pa = 10× envelope) uses ring_abs_max / interior_abs_max:
- ratio > 2.0 → A (boundary forcing dominant)
- ratio < 0.5 → B/C (dynamics or operational composition dominant)
- 0.5–2.0 → mixed

## Validation commands run

```
cd /tmp/wrf_gpu2_boundaudit
export OMP_NUM_THREADS=4 ; export PYTHONPATH="src"
taskset -c 0-3 python scripts/m6_boundary_dynamics_audit.py \
   --output .agent/sprints/2026-05-26-m6-boundary-dynamics-audit/ \
   --ics 20260429,20260509,20260521 --steps 360
taskset -c 0-3 python -m pytest tests/test_m6_boundary_dynamics_audit.py -v
```

Pytest result: **28 passed in 1.17s** (28 unit tests added under `tests/test_m6_boundary_dynamics_audit.py` covering classifier verdict bands, NaN/Inf handling, ring-vs-interior separator on synthetic spikes, malformed inputs, and proof file integrity).

## Proof objects produced

- `proof_excursion_catalog_20260429.json`
- `proof_excursion_catalog_20260509.json`
- `proof_excursion_catalog_20260521.json`
- `proof_excursion_classification.json`
- `proof_source_localization.json`   (peak-step ring/interior; coarse)
- `proof_source_localization_onset.json`   (onset-step ring/interior; diagnostic)
- `proof_audit_summary.json`

## Stage 1 — Excursion catalog (1h × 3 ICs)

Final-step abs_max (step 360) per IC:

| IC        | p_perturbation        | p_total               | u            | v       | w            |
|-----------|----------------------|-----------------------|--------------|---------|--------------|
| 20260429  | 1.76 × 10³⁰⁸         | 1.76 × 10³⁰⁸          | 8.46 × 10³⁷  | 23.50   | 4.20 × 10⁶⁰  |
| 20260509  | 1.79 × 10³⁰⁸         | 1.79 × 10³⁰⁸          | 8.47 × 10³⁷  | 13.98   | 2.25 × 10⁴⁰  |
| 20260521  | 1.76 × 10³⁰⁸         | 1.76 × 10³⁰⁸          | 7.65 × 10³⁷  | 11.48   | 1.56 × 10²⁵  |

All three runs drive p_perturbation and p_total to **1.7–1.8 × 10³⁰⁸ Pa** — at the IEEE-754 double-precision overflow threshold (~1.798 × 10³⁰⁸). Finite, but only by a hair. `u` reaches 8 × 10³⁷ m/s, which exceeds the speed of light by 30 orders of magnitude. `w` reaches 10²⁵–10⁶⁰ m/s. Only `v` stays bounded (11–24 m/s) — almost certainly because the prior v-guard in the operational coupling clamps it; the same guard does **not** appear to be applied to `u` (which exits the [-150, 150] m/s envelope at step 14 on IC-509 with u=2.3 × 10¹⁵ m/s in a single step).

Gen2 wrfout truth at lead 1h on these same ICs (read directly via `Gen2Run`):

| IC        | WRF p′ max (Pa) | WRF p_total min/max | WRF |u| max | WRF |v| max | WRF |w| max |
|-----------|----------------|---------------------|-------------|-------------|-------------|
| 20260429  | 1389.8         | 5224 / 101096       | 35.5        | 22.1        | 1.41        |
| 20260509  | (per catalog)  | (per catalog)       | (per catalog) | (per catalog) | (per catalog) |
| 20260521  | (per catalog)  | (per catalog)       | (per catalog) | (per catalog) | (per catalog) |

The operational p_perturbation overflow is **2.3 × 10³⁰⁵ ×** WRF; the operational |u| is **2 × 10³⁶ ×** WRF jet. These are not "absurd-looking" excursions — they are arithmetic on the verge of NaN.

## Stage 2 — Classification

**Aggregate verdict: `FAIL_B_OR_C`**

Per-IC, worst field:

| IC        | worst field          | ratio to envelope        | verdict       |
|-----------|----------------------|--------------------------|---------------|
| 20260429  | p_perturbation       | 3.5 × 10³⁰⁴              | FAIL_B_OR_C   |
| 20260509  | p_perturbation       | 3.58 × 10³⁰⁴             | FAIL_B_OR_C   |
| 20260521  | p_perturbation       | 3.5 × 10³⁰⁴              | FAIL_B_OR_C   |

All three p_total catalogs also breach the "p_total min ≥ 1 Pa" floor by ~10³⁰⁸ Pa in absolute terms (with `−1.78 × 10³⁰⁸` reached). Negative p_total is unphysical.

`v` is the only field that classifies PASS_A on all 3 ICs (jet-bounded by the operational guard).

## Stage 3 — Source localization

The peak-step ring/interior comparison is degenerate (everything overflows together so the ratio loses meaning). I added a complementary **onset-step** analysis at the step where |p′|_max first crosses 5 × 10⁴ Pa (10× the 5 kPa envelope) — see `proof_source_localization_onset.json`.

| IC        | onset step | onset window verdict (ring vs interior) | growth factor / step at onset |
|-----------|-----------:|------------------------------------------|------------------------------:|
| 20260429  | ~37        | INT_DOM (interior 5×–112× ring; flips to RING_DOM only after step 42 when ring saturates at overflow first) | ~10× per step |
| 20260509  | ~12        | **RING_DOM** (ring 15.78× interior at step 14, growing to 10³⁸× by step 15) | ~10⁶× per step |
| 20260521  | ~67        | INT_DOM (interior 2× to 10⁷² × ring; ring catches up only at saturation) | ~10× per step |

Throughout: **`v` is frozen at the operational guard value once it activates** (e.g. 13.978 m/s on IC-509 from step 9 onward) — confirming the v-guard is the only effective bound. **`theta` is locked in [290.34, 500.31] K** by the microphysics-coupling theta guard; this is also a guard signature, not physical evolution.

### Interpretation

The pattern is **not** uniform boundary forcing. Two of three ICs (20260429, 20260521) show **interior-dominant onset** — the pressure perturbation field begins exploding in the interior of the domain, not at the wrfbdy ring. Only 20260509 shows **ring-dominant onset**, and even there the interior catches up within 5 steps.

The common signature is **~10× exponential growth in `p′` per timestep once it crosses ~5 × 10⁴ Pa**. With dt=10 s this is e-folding within one acoustic super-step (10 acoustic substeps). That is the fingerprint of a **positive-feedback loop in the pressure / acoustic coupling**, not a single faulty operator or a single bad boundary value. Suspected operator family:

- `dynamics/core/acoustic.py::acoustic_substep_core` and its WRF coefficient table `dynamics/acoustic_wrf.py::calc_coef_w_wrf_coefficients`
- Vertical-implicit pressure solver in the same substep (coftz / tridiag)
- The "decoupling formula at composition boundary" change from the op-theta-fix merge (e552b9d) which routes `p` through `p_total` — that re-routing may have un-balanced the `p_perturbation` integration in the operational composition

The 20260509 ring-dominant pattern strongly suggests that a malformed wrfbdy-ring amplitude (likely from the AIFS → wrfbdy_d01 → wrfinput_d02 interpolation, which Canairy Gen2 already noted has 1 km topography and 5-cell relaxation band quirks) acts as the **trigger** that excites the acoustic/pressure feedback. On the other 2 ICs the feedback excites itself from interior numerical noise alone, just more slowly (steps 37 / 67 vs step 12).

### What the microphysics-coupling fix actually masked

The post-fix proof_theta_explosion.json had `p_total = −6.86 × 10²⁹⁴ Pa` at cell (0,0,0) while `theta=292.18 K`, `u=3.63 m/s`, `v=-4.52 m/s`, `qv=0.0083`. Cell (0,0,0) is a **corner ring cell**. That bottom-left corner already had `p_total` at 10²⁹⁴ Pa post-fix — the guards on theta/qc/qv/v held those fields bounded but `p` was unguarded. The microphysics-coupling fix **prevented Thompson from acting on the offending column** and prevented nonfinite/out-of-range moisture from feeding the next step, but it did nothing about the unbounded pressure perturbation already present in the dycore.

### Why the M6 acceptance gate (Tier-4 RMSE) presumably passed

T2, U10, V10 are surface diagnostics derived from `theta`, `u`, `v` near the lowest model level. theta is theta-guarded; v is v-guarded; u saturates at its own guard (±150 m/s) and propagates through the diagnostic projection. The RMSE-against-WRF metric is dominated by the bulk of the grid where the guards hold, and the few cells where u is at its guard limit don't dominate the spatial mean square. The Tier-4 RMSE acceptance therefore measures *whether the operational guards constrain surface diagnostics within tolerance*, not *whether the operational dynamics core produces a physical p / u / w field*.

## Gaps / caveats

1. **Only hourly WRF truth available** — sub-hour comparison (the timescale where the operational forecast diverges) relies on the IC initial state and the +1h wrfout. Per-step WRF truth would require re-running CPU WRF instrumented for per-step diagnostics, which is out of scope.
2. **`p` vs `p_total` vs `p_perturbation`** are routed through different fields in the operational composition (op-theta-fix sets `state.p = state.p_total` at init); my catalog reports all three but the audit treats `p_perturbation` as the leading indicator.
3. **`v` and `theta` guards mask the dynamics signal in those fields** — the audit can only see what the guards expose. A genuine dynamics-only run with all guards disabled would show whether `v` and `theta` also explode (they almost certainly do, given the pressure-gradient term in `advance_uv`).
4. **Boundary ring radius = 5 cells** is fixed by convention (wrfbdy_d01 → wrfinput_d02 5-cell relaxation band). The classifier would be sensitive to a different ring choice; sensitivity analysis is out of scope.
5. **Cross-check of WRF Fortran p tendency at offending cells** (sprint contract Stage 3) is not in this proof; the operational composition does not expose per-term arrays for `advance_uv` and `pressure_perturbation`, and the existing localization-memo evidence from the prior V3-509 + V3-521 sprints already names the same operator family (advance_mu_t / acoustic / coftz). Doing the term-by-term Fortran cross-check is the recommended next sprint's first task.

## Edge cases attempted (and what they showed)

- **Stretch IC (20260429), already known to pass M6 bounds** — explodes anyway, just later (step 37 vs step 12). Confirms the failure mode is not IC-specific; the dycore self-excites.
- **NaN/Inf injection in audit classifier** — handled cleanly (returns `FAIL_NONFINITE`).
- **Sub-grid ring (8×8 < 2×ring)** — classifier returns `skipped_too_small`, no false signals.
- **Ring-only spike vs interior-only spike on synthetic field** — separator distinguishes correctly (PASS in unit tests).
- **All three classifier verdict bands at exact 1×, 2×, 10× boundaries** — boundary inclusivity matches the contract (e.g. ratio=1.0 is PASS_A).
- **p_total below floor (negative)** — downgrades verdict; ratio < 10× still calls FAIL_B_OR_C when p_total < −1 Pa.

## Recommendations for the M6c / dycore fix sprint

1. **Stop the acoustic / vertical-implicit pressure feedback** before chasing boundary interpolation. Two of three ICs explode in the interior; fixing wrfbdy alone will not close this.
2. **Reproduce the explosion in a guard-disabled diagnostic mode** (run_forecast_operational_debug with all `_with_save_family` guards off) so that theta and v also become first-class signals.
3. **Compare per-substep, per-cell `p` tendency** against the WRF Fortran `module_small_step_em.F::pressure_perturbation` at the bad cells from this audit (e.g. (k=0,j=0,i=0) on IC-509 at step 9–14). This is the operator-term cross-check the sprint contract called for but that the operational wrapper does not currently expose; instrumenting `_operational_acoustic_substep_core` for a single step would unblock it.
4. **Add a hard p / w guard** equivalent in spirit to the v / theta guards already in place, but only as a triage diagnostic — not as a production mask. Production must produce a physical p field, not a guarded one.
5. **Re-run Tier-4 RMSE with guards disabled** to confirm whether the operational guards are doing the work the acceptance gate credits to "dynamics fidelity".

## Decision

**Decision: M6 close NO-GO — excursions > 10x physical reason on all 3 V3 ICs, root cause = mixed A (boundary-triggered on 20260509) + B/C (interior-self-excited on 20260429 and 20260521) in the acoustic / pressure feedback loop, recommend an M6c dycore-pressure-feedback fix sprint before any 24h gate.**

The microphysics-coupling fix legitimately closed the theta/qc explosion path, and the v-guard plus theta-guard are doing real work. But the operational p_perturbation and p_total fields reach IEEE-754 overflow within 2–11 minutes on every V3 IC, and u escapes its guard to 10¹⁵ m/s. The Tier-4 RMSE gate measures only what the surface guards expose, not what the operational dycore actually computes. M6c (24h) will not survive this; recommend the dycore-pressure fix sprint described above be queued before any 24h validation attempt.

—

*Report ends.*
