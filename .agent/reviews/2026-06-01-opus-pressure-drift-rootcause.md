# +2.6 kPa pressure-drift ROOT CAUSE — base-state inverse density `alb` recomputed at the WRONG base potential temperature (constant 300 K instead of WRF's `t0+t_init` profile)

Date: 2026-06-01
Agent: Opus 4.8 MAX (worker/opus/final-verdict, main working tree)
Owned: dycore pressure/geopotential path + the replay-loader IC hydrostatic balance.
Files: `src/gpuwrf/integration/d02_replay.py` (the fix), `src/gpuwrf/dynamics/acoustic_wrf.py`
(the consumer), diag under `scripts/diag/`.

## VERDICT (one line)

The unified +2.6 kPa perturbation-pressure / Exner-T2 offset on BOTH d02
(`force_geopotential=True`) and d03 is **candidate 1 + candidate 3 combined**: the
replay loader recomputes the dycore's base inverse density `alb` from a CONSTANT
300 K base potential temperature, while the loaded base geopotential `phb` was
hydrostatically integrated by WRF from `alb` of the realistic, height-varying base
profile `t0+t_init` (~290 K surface → ~465 K near the lid). The loaded IC is
therefore NOT in OUR dycore's discrete hydrostatic balance; the prognostic
perturbation geopotential `ph'` equilibrates over the first forecast hour to absorb
the base-state mismatch, producing the steady near-uniform +2.6 kPa `p'` offset.
WRF's dycore does not drift from the same IC because its `alb` matches its `phb`.

## The mechanism (file:line + why it drifts +2.6 kPa)

1. **The loaded base state is in WRF's discrete hydrostatic balance.** WRF
   `dyn_em/module_initialize_real.F:3793-3818` builds the base state so that
   ```
   pb(k)    = c3h(k)*(p_surf - p_top) + c4h(k) + p_top                 (:3795)
   t_init(k)= (t00 + A*log(pb/p00)) * (p00/pb)^(Rd/cp) - t0            (:3796-3801)
   alb(k)   = (Rd/p1000)*(t_init(k)+t0)*(pb(k)/p1000)^cvpm            (:3802)
   phb(k+1) = phb(k) - dnw(k)*(c1h(k)*mub + c2h(k))*alb(k)            (:3817)
   ```
   i.e. `phb` is EXACTLY the hydrostatic integral of `alb` of the realistic
   `t0+t_init` profile. The corpus wrfout stores PB/PHB/MUB directly (and `T00=290`,
   `TLP(=A)=50`, `P00=1e5`, `P_TOP=5000`, `TISO=200`), and the replay loader reads
   them verbatim → the IC base state IS WRF's, balanced against WRF's `alb`.

2. **Our dycore recomputes `alb` from the WRONG base theta.**
   `src/gpuwrf/dynamics/acoustic_wrf.py:253`,
   `diagnose_pressure_al_alt` →
   `alb = _inverse_density_from_theta_pressure(base_state.theta_base, base_state.pb)`
   `= (Rd/p0)*theta_base*(pb/p0)^cvpm`. This is the SAME EOS as WRF :3802, but
   `theta_base` was set to a CONSTANT 300 K by the loader
   (`src/gpuwrf/integration/d02_replay.py:788`,
   `theta_base = jnp.full_like(theta, P0_THETA_OFFSET_K)`), NOT WRF's `t0+t_init`.
   Aloft `t0+t_init` reaches ~465 K, so our `alb` is up to ~35 % too small there.

3. **Consequence: the loaded `phb` is NOT hydrostatic against OUR `alb`.**
   Reintegrating `phb` from our constant-300 `alb` diverges from the loaded `phb`
   by **-1504 m^2/s^2 @ k=20, -15828 @ k=40, -22373 @ k=44** (vs **<10-80 m^2/s^2**
   round-off when using WRF's `alb`). Our `alt = al + alb` (used in the acoustic w-ph
   buoyancy solve and the EOS pressure) is on a wrong base reference, so the
   prognostic `ph'` — which `calc_p_rho`/`advance_w` advance — relaxes the interior
   to OUR balance. That depresses mid-column `ph'` (the exact "bowed-low ph',
   uniform p' offset" signature both bisections found) and the EOS inflates `p` by
   a near-uniform +2.6 kPa. This is in-flight (hour-1 equilibration), level-uniform,
   entirely in `p'` (PB matches corpus to 0 Pa), and INDEPENDENT of the lateral
   boundary — so it is present on d02 (`force_geopotential=True`) too, exactly as
   the d02 diagnosis observed.

This is why the earlier "round-trips to 1e-10 Pa" claim was true yet did not catch
it: the round-trip used the SAME wrong `alb` for forward and inverse, so it was
self-consistent — but on the wrong base reference. The drift only manifests when the
LOADED `ph'` (balanced against WRF's `alb`) meets OUR `alt` in the time advancement.

## Why it is NOT the other candidates

- NOT candidate 2 (mub / p_top / c1c2 / p0): PB base matches corpus to 0 Pa; mub,
  c1/c2, p_top are loaded verbatim and reproduce WRF's `phb` exactly when paired
  with WRF's `alb`. The defect is purely the recomputed `alb`'s base-theta argument.
- NOT candidate 4 (smdiv / acoustic pressure memory): the offset is a static
  base-state-balance error present from the first hour and steady, not a slow
  smdiv accumulation; `calc_p_rho_step` smdiv matches WRF (:557-567).
- NOT candidate 5 (ph' surface anchor / top BC): `ph'[0]=0` and the lid are WRF-
  faithful; the bow is interior, driven by the `alb` mismatch, not the anchor.
- NOT a boundary/nesting issue: confirmed by the prior d02 (force_geopotential=True
  still drifts) and the failed boundary-forcing attempts.

## THE DRIVER is the OPERATIONAL-path diagnostic alb, not just the loader

CRITICAL refinement after the first 6 h gate (below): fixing the LOADER's
`BaseState.theta_base` alone did NOT collapse the drift, because the OPERATIONAL
stepping does not read the loader's `BaseState`. Every RK stage,
`runtime/operational_mode.py::_refresh_grid_p_from_finished` (line ~1088-1099)
RE-DIAGNOSES and OVERWRITES the prognostic `state.p_perturbation` via
`diagnose_pressure_al_alt(next_state, base, metrics)` where it builds
`base = BaseState(..., theta_base = jnp.full_like(next_state.theta, prep.theta_offset))`
with `prep.theta_offset = _theta_base_offset(...) = CONSTANT 300 K`. (The legacy
`_dycore_step_adr023` path does the same at line ~771.) So the in-flight prognostic
pressure is recomputed each stage with the constant-300 `alb` — the exact bug,
re-injected every step regardless of the loader. Isolation at the loaded IC:
the const-300 `alb` diagnostic gives a surface `p'` that differs from the correct
(phb-derived) `alb` diagnostic by **-3974 Pa at the surface / +1615 Pa column-mean**;
that wrong vertical `p'` gradient feeds `pg_buoy_w` and drives the `ph'`/`w`
equilibration to the steady +2.6 kPa.

## The WRF-faithful fix (applied) — make `alb` exact in `diagnose_pressure_al_alt`

`src/gpuwrf/dynamics/acoustic_wrf.py::diagnose_pressure_al_alt`: recover the base
inverse density `alb` by INVERTING the base geopotential the state already carries,
instead of recomputing it from a (possibly-constant-300) `theta_base`:
```
alb(k) = -(phb(k+1)-phb(k)) / ( dnw(k)*(c1h(k)*mub + c2h(k)) )   # invert WRF :3817
```
This is the EXACT discrete `alb` WRF's `module_initialize_real.F:3817` integrated
`phb` from, is grid-agnostic (uses the file's own hybrid c1h/c2h/dnw, correct for
the hybrid coordinate), needs no base-profile params, and is CALLER-AGNOSTIC: it no
longer matters whether a caller passes `theta_base=300` or the true profile — the
diagnostic now always uses WRF's real `alb`. This corrects BOTH the operational
`_refresh_grid_p_from_finished`/`_dycore_step` diagnostic (the in-flight driver) and
the loader at once.

Also applied (defence in depth, now redundant-but-harmless): the loader
`src/gpuwrf/integration/d02_replay.py::_wrf_base_theta_from_loaded_state` recovers the
true `t0+t_init` base profile for `BaseState.theta_base` (and sets `base_state.t0`
to the WRF constant 300 K). Either fix alone makes the loaded IC consistent; both
together leave no constant-300 `alb` anywhere on the operational path.

The `base_state=None` branch (used by `small_step_prep_wrf` and the legacy
`_acoustic_core_state` for the PROGNOSTIC `alt = EOS(theta, p_total)`) is unchanged
and was never the bug — WRF's small-step `calc_p_rho` (`module_small_step_em.F:522`)
uses full `alt`, which our prognostic recompute matches. Only the `base_state`-aware
DIAGNOSTIC `p` overwrite carried the wrong `alb`.

Shared by BOTH domains: d02/d03/operational all run the one operational stepping path,
so the fix corrects the unified bug on both at once. The idealized warm-bubble /
Straka path builds its own self-consistent neutral-300 K base
(`ic_generators/idealized.py:507-515`, `phb` integrated FROM its own 300 K `alb`),
and on the doubly-periodic grid the phb-inversion returns that same neutral `alb`
to round-off, so idealized is unaffected (re-run gate below).

## Static validation of the fix (CPU/GPU, no stepping) — PASS

Via the production loader on the d03 nested case (`build_replay_case(..., domain='d03')`):
- recovered `theta_base` = WRF base profile: surface ~291 K, lid ~465 K (= `t0+t_init`).
- **`phb` reintegration error with the NEW `alb`: 0.018 m^2/s^2** (was -22373 with
  constant 300 K) → loaded IC now in OUR discrete hydrostatic balance to round-off.
- single-shot surface `p'` diagnosis at t=0: bias vs corpus **-1381 Pa** (was
  **-5356 Pa** with constant 300 K) — the fix shifts the diagnosed surface `p'` by
  **+3974 Pa toward corpus**, the same magnitude as the +2.6 kPa drift it removes.
  (The residual -1381 Pa is the dry-vs-moist / fp32-corpus-`al` single-shot offset,
  not the dynamical drift; the in-flight gate below is the decisive measurement.)

## GATES (falsifiable — STOP+report on failure; nothing forced)

(a) Idealized warm-bubble + Straka 6/6 — PASS, both before AND after the
`diagnose_pressure_al_alt` `alb`-from-`phb` change, BIT-IDENTICAL to baseline
(STRAKA: front 14150 m, 4 rotors, theta' min -9.970995032355745, max|w|
14.574919073009704, mass drift 2.25e-9; BUBBLE: thermal_rise 1924.35 m, theta'
1.92 K, mass drift 0). The fix is a strict no-op for the doubly-periodic neutral-300
idealized base (phb-inversion returns its own neutral `alb`). Dycore protected.

(b) d03 6 h short run — **alb-from-phb fix: PASS, the +2.6 kPa COLLAPSES, T2 < 1 K.**
 - LOADER-FIX-ONLY (alb-from-phb NOT yet applied): D03_1KM_VALIDATED, all_finite,
   NO blow-up — but +2.6 kPa did NOT collapse (mean +2656 Pa; T2 RMSE 1.45, bias
   +1.27 K) ≈ baseline. This proved the loader `BaseState` is not read by the
   operational stepping → exposed the `_refresh_grid_p_from_finished` const-300
   `alb` driver.
 - alb-from-phb fix: D03_1KM_VALIDATED PASS, all_finite, stable, no blow-up.
   psfc bias **+2656 → -293 Pa** (mean; hour1 +2720 → -230); T2 RMSE **1.45 → 0.72 K**;
   T2 bias **+1.27 → -0.29 K** (warm → slight-cool). The +2.6 kPa offset is GONE.
   Per-lead psfc: -230/-254/-276/-297/-333/-370 Pa; T2 RMSE: 0.68/0.55/0.63/0.76/0.83/0.88.
   Proof: /tmp/d03_albfix2_proofs/d03_validation_albfix2_6h.json, wrfouts in
   /tmp/d03_albfix2_runs/...albfix2_6h/. Scorer scripts/diag/d03_psfc_t2_check.py.

(c) d02 24 h full re-score (primary product): <PENDING — running>.

## Files / diag artifacts
- `src/gpuwrf/integration/d02_replay.py` — the fix (`_wrf_base_theta_from_loaded_state`
  + use it for `theta_base`; `t0` = WRF constant 300 K).
- `scripts/diag/ic_pressure_check.py`, `d03_pressure_knockout.py`,
  `d03_psfc_t2_check.py` — reused for validation.
