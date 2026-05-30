# FP32 Downcast Plan (PLAN ONLY — not implemented in this sprint)

Sprint: `worker/opus/perf-diag` (from `manager-2026-05-23` @ `17e0039`).
Scope: a concrete, risk-assessed plan to move the coupled real-case operational
forecast from its current **forced-FP64** execution to a **mixed FP32/FP64** path,
without re-introducing the prior fp32-defeat bugs, gated by the idealized close
gates and the +1h/+3h operational RMSE. **No precision code is changed here.**

This plan operationalizes the binding authorization matrix in
`.agent/decisions/ADR-007-precision-policy.md` for the *current* coupled path. It
does not re-litigate ADR-007; it specifies exactly which knobs to flip and how to
gate the flip.

---

## 0. Why this matters / current state

Task 1 (`proofs/perf/warmed_timing.json`) showed the warmed coupled scan runs at
**~45.5 ms/step (16.4 s per forecast-hour)** entirely in **FP64** today. The
real-case path is forced to FP64 by design:

* `gpuwrf.integration.daily_pipeline._build_real_case` builds the namelist with
  `force_fp64=True` (daily_pipeline.py:204).
* `operational_mode._enforce_operational_precision(state, force_fp64=True)` upcasts
  **every** `STATE_FIELD_ORDER` field to `float64` at the public entry AND once per
  step inside the scan (operational_mode.py:319-336), using `state.replace(_cast=False, ...)`.

So today even the fields ADR-007 marks `FP32-OK` (u, v, theta, qv, hydrometeors)
run in FP64. The FP32 opportunity is precisely to stop forcing those fields up.

On the RTX 5090 (Blackwell, consumer) FP64:FP32 throughput is ~1:64 (ADR-007
Context). Any arithmetic that stays FP64 sees no benefit; the win comes from the
FP32-eligible advection/physics arithmetic and from halving the memory traffic and
the device working set (Task 1 peak ≈ 9.0 GB at 180 steps; FP32 storage of the
gated fields roughly halves the per-step traffic for those leaves).

---

## 1. Field-by-field downcast classification

Source of truth = `src/gpuwrf/contracts/precision.py::PRECISION_MATRIX` (already the
ADR-007 matrix). The current operational path **ignores** it (force_fp64). The plan
is to **honor** it. Summary:

### MUST STAY FP64 (`FP64-locked` — do not downcast, ever, in this plan)

These were FP64-critical in the dycore close and are catastrophic-cancellation- or
mass-conservation-prone:

| Field / path | Reason |
|---|---|
| `mu`, `mu_total`, `mu_perturbation` | column dry-mass continuity; mass residual must stay ≤1e-10 fractional |
| `p`, `p_total`, `p_perturbation` | pressure; PGF subtracts nearly-equal adjacent-level pressures (Δp/p ~ 1e-3..1e-5) — FP32 leaves ~2 sig digits, insufficient |
| `ph`, `ph_total`, `ph_perturbation` | geopotential; same hydrostatic-cancellation argument as pressure |
| `w` | vertical wind — `needs-empirical-test` in ADR-007; couples into the **implicit w/ph acoustic solve** (`calc_coef_w`, `advance_w`), which was FP64-critical at dycore close. Keep FP64 in phase 1. |
| **Acoustic small-step internals** | `acoustic_substep_core`, `small_step_prep`, `calc_p_rho`, `calc_coef_w`, `diagnose_pressure_al_alt`, `horizontal_pressure_gradient`, the implicit tridiagonal w/ph solve — **all FP64**. The acoustic loop runs `acoustic_substeps` (=10) times per RK substep and accumulates pressure/geopotential deltas; this is the single most cancellation-sensitive kernel in the model. |
| `ustar, theta_flux, qv_flux, tau_u, tau_v, rhosfc, fltv, t_skin, soil_moisture, roughness_m` | surface-layer stability/flux handles (Monin-Obukhov is iterative and ill-conditioned near neutral) |
| `*_acc` (rain/snow/graupel/ice) | monotone accumulators; FP32 loses small increments over 24-72h (swamping) |
| FP64 boundary leaves (`w_bdy, p_bdy, ph_bdy, mu_bdy, pb_bdy, phb_bdy, mub_bdy`) | feed the locked fields |

### CAN GO FP32 (`FP32-OK` — storage + non-acoustic arithmetic)

| Field | Notes |
|---|---|
| `u`, `v` | horizontal winds; storage + non-acoustic (advection/diffusion/physics) arithmetic FP32; **FP64 at the acoustic/PGF boundary** (upcast on entry to the acoustic core). |
| `theta` | potential temperature; FP32 storage with FP64 conservation boundary (the theta limiter already runs its mass-conservation math in fp64 — `_positive_definite_theta_increment_limiter` casts to float64 internally; that stays). |
| `qv` | water vapor; FP32 storage/tendency. |
| `qc, qr, qi, qs, qg, Ni, Nr, Ns, Ng` | Thompson hydrometeor mass/number; already `float32` in `DEFAULT_DTYPES`. |
| `qke` | MYNN TKE. |
| `xland, lakemask, mavail` | static land masks. |
| FP32 boundary leaves (`u_bdy, v_bdy, theta_bdy, qv_bdy`) | feed FP32 fields. |

### KEY ARCHITECTURAL RULE — the FP64 conservation/acoustic boundary

The model is **mixed**, not FP32. The acoustic core, pressure, geopotential, mass,
and the implicit w solve are FP64 *islands*; FP32 fields are **upcast on entry**
to those islands and the FP64 result is **downcast on exit**. The existing code
already does most of this: `_acoustic_core_state*` casts u/v/theta into FP64
(`.astype(jnp.float64)`) before the acoustic solve. So the FP32 plan does not touch
the acoustic island at all — it only changes the *resident storage dtype* of the
FP32-OK fields between RK steps.

---

## 2. How to GATE the flip (binding pass/fail before any production merge)

Run BOTH gate families on the mixed-precision build; ALL must pass or the field is
reverted to FP64.

### 2a. Idealized close gates (MUST still PASS, hard floor)

These are the dycore correctness floor and must be re-run with the mixed-precision
state (note: the idealized cases set `force_fp64=True` deliberately, so they will
need a parallel `force_fp64=False` variant ONLY for this gate):

* **Warm bubble** — `proofs/f2/` warm-bubble θ′ evolution must stay physical
  (no detonation, max|w| saturates) and match the FP64 reference field within the
  idealized tolerance. Prior history: **fp32 lost the perturbation and detonated**
  (daily_pipeline.py:168-170) — this is the canonical failure to watch.
* **Density current / Straka** — `proofs/sprintU/straka_canonical_parity.json` and
  `straka_deformation_gate.json`: front position ~4.25 km @ 300 s, max|w| ~22 m/s.
  FP32 must reproduce front propagation (the FP64 reference is WRF em_grav2d_x).

Because the acoustic island stays FP64, the **expectation** is these PASS — the
perturbation is carried through the acoustic solve in FP64. The gate exists to
catch the case where FP32 storage of θ between RK steps quantizes the perturbation
below the noise floor on a near-rest idealized state (the historical detonation).
If warm-bubble fails, θ stays FP64 and only u/v/q*/physics go FP32.

### 2b. Operational RMSE gate (binding, the real fitness test)

Re-run the +1h/+3h skill signal (`proofs/coupled/task2_skill_signal.py`) with the
mixed-precision build and require the gridded T2/U10/V10 RMSE to stay within
tolerance of the **current FP64 numbers**:

| Field | FP64 +1h | FP64 +3h | FP32 tolerance (ADR-007 §Decision pt5 / w-test plan) |
|---|---|---|---|
| T2  | 1.33 | 1.83 | `|RMSE_fp32 − RMSE_fp64| ≤ 0.10 K` at each lead |
| U10 | 2.22 | 1.91 | `≤ 0.10 m/s` |
| V10 | 3.70 | 2.75 | `≤ 0.10 m/s` |

Plus: every emitted field finite, and the dry-mass residual / total-water residual
must stay ≤ the FP64 run's value (no conservation regression). Extend to 6h/12h/24h
once the OOM-fixed M9 diagnostics path (Task 2) is used for multi-lead scoring.

### 2c. w-field empirical sub-gate (phase 2 only)

`state.w` is `needs-empirical-test`. Keep it FP64 in phase 1. Phase-2 w→FP32
follows the ADR-007:70 plan exactly: paired 24h forecasts FP64-w vs FP32-w, gate
`|ΔRMSE| < 0.10`, inspect vertical-velocity column spectra at sea/lee/ridge/peak
for spurious 2Δx noise, water-budget residual ≤ 1e-10. Pass-all → w FP32; else keep
FP64.

---

## 3. Implementation outline (for the gated follow-up sprint)

1. **Single switch already exists.** `_enforce_operational_precision(state, force_fp64=False)`
   already routes through `DEFAULT_DTYPES.dtype_for(field)` (the FP32 matrix). The
   change is: `_build_real_case` sets `force_fp64=False`, and the public entries
   (`run_forecast_operational` etc.) keep their existing `force_fp64` plumbing. No
   new dtype logic is needed — the matrix and the cast helper are already there.
2. **Verify the acoustic-island upcast is complete.** Audit every entry into the
   acoustic/pressure/mass kernels and confirm FP32 leaves are `.astype(float64)`
   before use and the FP64 result is the stored authority. (`_acoustic_core_state`
   / `_acoustic_core_state_from_prep` already do this for u/v/theta/ph/p; confirm
   no FP32 leaf reaches `calc_coef_w` / `diagnose_pressure_al_alt` un-upcast.)
3. **Conservation boundaries stay FP64.** The theta limiter
   (`_positive_definite_theta_increment_limiter`) already does its mass math in
   float64 internally and casts the *output* back to the field dtype — leave it.
4. **Halo / boundary apply.** Confirm `apply_halo` and `apply_lateral_boundaries`
   do not silently re-canonicalize dtypes (see §4 bug #2).

---

## 4. Prior fp32-defeat bugs — DO NOT reintroduce

These three are the reason a naive `force_fp64=False` historically became a silent
no-op or a detonation. The plan must explicitly defend against each:

1. **x64-at-import (global enable).** `jax.config.update("jax_enable_x64", True)`
   is set at import in `contracts/state.py:17` and `operational_mode.py:69`. With
   x64 enabled, FP32 arrays are NOT auto-promoted, but any *Python float literal*
   or `jnp.asarray(x)` without an explicit dtype becomes FP64 and **promotes the
   whole expression back to FP64** — silently defeating the downcast and erasing
   the speedup while still "looking" fp32 in storage. **Defense:** every constant
   in an FP32 arithmetic path must carry an explicit `dtype=` matching the field
   (`jnp.asarray(c, dtype=field.dtype)`), and the gate must assert the *compiled*
   HLO has fp32 fusions for the gated fields (not just fp32 storage). This is the
   subtlest of the three and the one that quietly kills the perf win.

2. **`State.replace(_cast=...)` canonicalization.** `State.replace` casts each
   updated value back to the *current* field dtype by default (`_cast=True`,
   state.py:566-588). This means: (a) if a field's resident dtype is FP64, writing
   an FP32 value silently upcasts it back to FP64 (downcast no-op); (b) conversely
   `_enforce_operational_precision(force_fp64=True)` MUST use `_cast=False` to make
   the upcast stick (it does). **Defense:** the FP32 build must ensure the *initial*
   carry already has the FP32 fields stored as FP32 (so `_cast=True` keeps them
   FP32 thereafter), and `_enforce_operational_precision(force_fp64=False)` must use
   the same `replace` semantics that land each field at its matrix dtype. Add a
   one-step assertion that `final_state.u.dtype == float32` etc. — the historical
   failure mode was the build "claiming" fp32 while every leaf was silently FP64.

3. **Scatter-buffer dtype.** `*.at[idx].set(value)` / `dynamic_update_slice`
   (the HLO scan stacking, halo writes, boundary relaxation) take the dtype of the
   **buffer**, not the value. If a scratch/output buffer is allocated FP64 (e.g.
   `jnp.zeros_like(some_fp64_field)`) and an FP32 value is scattered in, the buffer
   stays FP64 and re-promotes. **Defense:** every scatter target in an FP32 path
   must be allocated with the field's dtype; the gate must scan the compiled HLO
   for unexpected `convert(f32->f64)` ops on the gated fields inside the loop body
   (the `proofs/perf/fusion_transfer_audit.py` HLO introspection already greps the
   compiled text and can be extended to count `convert` ops f32↔f64).

---

## 5. Expected speedup (honest, bounded)

* The acoustic/pressure/mass island stays FP64 and is the per-step hot path
  (10 acoustic substeps × the implicit w/ph solve per RK substep × 3 RK substeps).
  Those kernels see **no** FP32 benefit. So the *dynamics* per-step cost is
  dominated by FP64 work that this plan does not touch.
* The FP32 win is concentrated in: (a) the **physics** column kernels (Thompson,
  MYNN, RRTMG, surface) which ADR-007 micro-benched at ~3× FP64→FP32 on GPU; (b)
  the non-acoustic advection/diffusion arithmetic (ADR-007 M4 dycore micro-bench:
  713 µs→203 µs, ~3.5×, but that micro-bench downcast the *whole* state incl.
  pressure, which this plan does NOT — so the realized dycore gain is much smaller);
  (c) halved memory traffic / working set for the gated leaves.
* **Honest projection:** because the FP64 acoustic island is not downcastable, the
  realistic end-to-end warmed speedup from mixed precision alone is **~1.3–1.8×**
  (physics + advection arithmetic + memory traffic), NOT the 3–4× a full-FP32
  micro-bench suggests. Reaching the larger speedups requires the *separate* XLA
  single-scan compile fix (Task 3) + kernel fusion work, which is orthogonal to
  precision. CPU-WRF denominator for any speedup framing = **28-rank CPU WRF on
  this workstation** (no CPU wall re-measured here).
* Combined with the Task-1 finding (warmed 24h ≈ 6.6 min/case, ensemble ≈ 3.3 h),
  mixed precision is a **nice-to-have for the ensemble throughput, not a blocker**:
  the 24-72h × 30-case ensemble is already feasible warmed in FP64 once the compile
  cost is paid once per static-hours value.

---

## 6. Recommended sequencing

1. **Phase 0 (now):** ship the Task-2 M9 OOM fix (done) so multi-lead scoring works.
2. **Phase 1 (gated sprint):** flip `force_fp64=False` for u/v/theta/qv + the
   already-fp32 hydrometeors; keep w + acoustic + pressure/mass/geopotential FP64.
   Gate with §2a (warm-bubble/Straka) + §2b (+1h/+3h T2/U10/V10) + §4 HLO
   convert-op audit. Expected ~1.3–1.8×.
3. **Phase 2 (separate gated sprint):** w→FP32 empirical sub-gate (§2c) ONLY if
   phase 1 passes and the w-spectra are clean.
4. **Never (this plan):** the acoustic small-step, the implicit w/ph solve, and the
   pressure/EOS stay FP64 indefinitely unless a dedicated sound-wave + mass-residual
   proof authorizes otherwise (ADR-007 would need amendment).
