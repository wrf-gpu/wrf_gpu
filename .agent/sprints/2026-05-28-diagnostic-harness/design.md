# Comprehensive Diagnostic Harness — Design

**Status**: in-flight 2026-05-28
**Owner**: manager-Opus (Claude Opus 4.7, 1M context)
**Authority**: principal directive 2026-05-28 — "one test that runs all couplings and dynamics over many timesteps and produces relatively clear-to-exact indications of which coupling/dynamic is failing or completely missing".
**Constraint**: must not modify `src/gpuwrf/dycore/**` or `src/gpuwrf/coupling/physics_couplers.py` (M11/M12/M13 workers own them). All instrumentation is wrapper-only, behind a `diagnostic_on: bool` static-arg switch on `_physics_boundary_step` so that XLA dead-code-eliminates the entire instrumentation tree in production.

---

## 1. Problem statement

We already have two diagnostic surfaces:

- **Bitwise dycore parity** at 100 coupled steps (`tests/savepoint/test_dycore_100_steps.py`) — proves the pure dynamics path is bitwise but ignores physics and boundary forcing.
- **Hourly wrfout RMSE trace** (`scripts/operational_trace_compare.py`) — proves *end-to-end* divergence but says nothing about which operator caused it.

The blind spot is between these: per-step, per-operator attribution. Nothing today distinguishes a Thompson failure from a MYNN failure from a missing surface-flux coupling, and nothing detects an operator that is wired but silently identity (Δ-state = 0). That blind spot is the reason M11/M12/M13 are three parallel sprints with no shared "where is the bug?" surface.

The harness fills this exact gap.

---

## 2. Required artifact (`diagnostic_report.json`) — schema

The harness writes a single self-contained JSON at `proofs/diagnostic_harness/diagnostic_report.json`. A human reader opens it top-to-bottom and decides next sprint priorities without reading any other file.

```jsonc
{
  "schema_version": "diagnostic-harness-1.0",
  "generated_utc": "2026-05-28T14:00:00Z",
  "commit": "<git rev-parse HEAD>",
  "run_config": {
    "case_run_id": "20260521_18z_l3_24h_20260522T133443Z",
    "domain": "d02",
    "hours": 1.0,
    "dt_s": 10.0,
    "steps_total": 360,
    "radiation_cadence_steps": 60,
    "rk_order": 3,
    "acoustic_substeps": 10,
    "diagnostic_on": true,
    "disable_guards": false,
    "run_physics": true,
    "run_boundary": true,
    "platform": "<JAX_PLATFORMS env>",
    "wall_seconds_total": 0.0,
    "wall_seconds_diagnostic_overhead": 0.0
  },

  "headline_diagnosis": "<one-paragraph human-readable verdict; auto-generated from the per-operator + invariant + RMSE summaries>",

  "first_failure_attribution": {
    "first_invariant_break": {
      "step": 17,
      "operator": "microphysics_thompson",
      "invariant": "qv_nonnegative",
      "magnitude": -1.2e-08,
      "cell_zyx": [5, 30, 70]
    },
    "first_nonfinite": {
      "step": 23,
      "operator": "dycore_rk3",
      "field": "w",
      "cell_zyx": [40, 12, 15]
    },
    "first_significant_anchor_divergence": {
      "hour": 1,
      "field": "U",
      "rmse_vs_wrf": 38.4,
      "operator_attribution_guess": "dycore_rk3"
    }
  },

  "operator_attribution_24h": {
    "dycore_rk3": {
      "verdict": "ACTIVE",
      "steps_with_finite_delta": 360,
      "mean_abs_delta_per_step": { "u": 0.42, "v": 0.31, "w": 0.018, "theta": 0.013 },
      "max_abs_delta_per_step": { "u": 9.8, "v": 6.1, "w": 0.42, "theta": 2.3 },
      "first_zero_delta_step": null,
      "comments": "delta nonzero on every step for all dynamics state vars"
    },
    "microphysics_thompson": {
      "verdict": "ACTIVE",
      "steps_with_finite_delta": 360,
      "mean_abs_delta_per_step": { "qv": 3.1e-7, "qc": 1.0e-6, "qr": 4.0e-7, "qi": 1.0e-8, "qs": 5.0e-8, "qg": 1.0e-9, "theta": 6.0e-4 },
      "comments": "..."
    },
    "surface_layer": {
      "verdict": "MISSING",
      "steps_with_finite_delta": 360,
      "mean_abs_delta_per_step": { "theta_flux": 0.0, "qv_flux": 0.0, "tau_u": 0.0, "tau_v": 0.0, "ustar": 0.0, "rhosfc": 0.0 },
      "first_zero_delta_step": 1,
      "comments": "surface adapter wired into _physics_boundary_step but produces identically-zero flux fields; downstream MYNN bottom-BC therefore has no effect on theta"
    },
    "mynn_pbl":      { "verdict": "ACTIVE",   "...": "..." },
    "rrtmg":         { "verdict": "INACTIVE", "reason": "radiation_cadence_steps=999999 — cadence never fires in this 1h run", "...": "..." },
    "lateral_boundary": { "verdict": "ACTIVE", "...": "..." }
  },

  "internal_consistency_24h": {
    "all_state_finite": {
      "violated": false,
      "first_violation_step": null,
      "violation_count": 0,
      "first_violation_field": null
    },
    "qv_nonnegative": {
      "violated": true,
      "first_violation_step": 17,
      "violation_count": 28,
      "first_violation_operator": "microphysics_thompson",
      "min_value_observed": -1.2e-08
    },
    "theta_in_bounds": { "lower_30_levels_k": [200, 400], "upper_14_levels_k": [250, 700], "violated": false, "first_violation_step": null },
    "wind_in_bounds":  { "u_abs_max_m_s": 100.0, "v_abs_max_m_s": 100.0, "w_abs_max_m_s": 50.0, "violated": false, "first_violation_step": null },
    "mu_nonnegative":  { "violated": false, "first_violation_step": null },
    "dry_mass_drift_per_step": { "p50": 1.0e-12, "p99": 1.0e-9, "max": 1.0e-7, "max_step": 47 },
    "column_water_drift_per_step": { "p50": 1.0e-12, "p99": 1.0e-9, "max": 1.0e-8, "max_step": 47 }
  },

  "wrf_anchor_comparison": {
    "source": "proofs/m9/operational_trace_hourly.json or inline if absent",
    "per_field": {
      "U":     { "rmse_over_all_hours": 38.4, "max_abs_diff": 71.2, "first_divergence_hour": 1 },
      "V":     { "...": "..." },
      "T2":    { "...": "..." },
      "U10":   { "...": "..." }
    }
  },

  "coupling_chain_audit": {
    "surface_layer__to__mynn_theta_bottom_bc": {
      "verdict": "BROKEN",
      "evidence": "surface_layer produced theta_flux ≡ 0; MYNN bottom-BC therefore added zero increment to theta[..., 0] on every step",
      "first_broken_step": 1
    },
    "thompson__to__theta_via_latent_heat": {
      "verdict": "ACTIVE",
      "evidence": "thompson produced mean(|d theta|) = 6.0e-4 K/step",
      "first_broken_step": null
    },
    "rrtmg__to__theta_via_heating_rate": {
      "verdict": "INACTIVE",
      "evidence": "radiation cadence never fired in 1h run; theta delta from rrtmg = N/A",
      "first_broken_step": null
    }
  },

  "verdict": "DIAGNOSIS_PRODUCED" | "NO_DATA" | "HARNESS_BROKEN",
  "next_sprint_recommendations": [
    "M12 (surface flux): surface adapter writes theta_flux ≡ 0; investigate surface_layer.surface_layer fixed inputs",
    "M11 (theta limiter): no theta-bound violations — OK to keep limiter conservative",
    "M13 (radiation): increase radiation_cadence_steps cadence to fire at least 4x in operational run"
  ]
}
```

The `verdict` field is the single boolean the manager checks: any value other than `DIAGNOSIS_PRODUCED` means the harness itself failed and the rest of the JSON is not usable.

---

## 3. Operators instrumented

The harness intercepts `_physics_boundary_step` between each major operator and records pre/post state snapshots. The hooks are introduced via a wrapper module — the existing `_physics_boundary_step` is **not modified beyond a single `diagnostic_on: bool` kwarg pass-through** and a final call to the harness's recorder function.

| Operator key             | Source                                          | Δ-state fields recorded                                                |
|--------------------------|-------------------------------------------------|------------------------------------------------------------------------|
| `dycore_rk3`             | `_rk_scan_step` (entire RK + acoustic envelope) | `u`, `v`, `w`, `theta`, `p_perturbation`, `ph_perturbation`, `mu_perturbation` |
| `dynamics_guards`        | `_limit_guarded_dynamics_state` + qx clamps     | hits-per-step (count of clipped cells) + which field                  |
| `microphysics_thompson`  | `thompson_adapter`                              | `qv`, `qc`, `qr`, `qi`, `qs`, `qg`, `Ni`, `Nr`, `theta`               |
| `surface_layer`          | `surface_adapter`                               | `ustar`, `theta_flux`, `qv_flux`, `tau_u`, `tau_v`, `rhosfc`, `fltv`  |
| `mynn_pbl`               | `mynn_adapter`                                  | `u`, `v`, `w`, `theta`, `qv`, `qke`                                   |
| `rrtmg`                  | `rrtmg_adapter` (only steps where cadence fires)| `theta`                                                                |
| `lateral_boundary`       | `apply_lateral_boundaries`                      | `u`, `v`, `w`, `theta`, `qv`, `p_perturbation`, `ph_perturbation`     |
| `boundary_guards`        | post-boundary `_finite_or_origin` family        | hits-per-step + which field                                            |

Per operator per step, the harness records (computed inside the JIT, reduced via `jax.lax.scan` carries, no Python-level callbacks in the hot path):

- `mean_abs_delta` per recorded field — scalar reduction over the full grid.
- `max_abs_delta` per recorded field — scalar reduction.
- `nonfinite_count_post` — number of nonfinite cells after the operator.
- `guard_hit_count` (where applicable) — number of cells the limiter touched.

These are stored as a single dense `(steps, field_index)` array per operator, returned from the scan and written to the JSON at the end.

---

## 4. Invariants checked + thresholds

Computed **once per step**, after the full `_physics_boundary_step` returns. Same JIT, no host callbacks.

| Invariant key                      | Definition                                                                               | Threshold                                       |
|------------------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------|
| `all_state_finite`                 | every leaf of State has `jnp.all(jnp.isfinite)`                                          | violation = any nonfinite                       |
| `qv_nonnegative`                   | `jnp.min(state.qv) >= 0`                                                                 | `qv >= -1e-12` (FP slack)                       |
| `qc_nonnegative` (and qr, qi, qs, qg) | same                                                                                  | `q >= -1e-12`                                   |
| `theta_in_bounds`                  | lower 30 levels ∈ [200, 450] K; upper levels ∈ [250, 700] K                              | hard violation if outside                       |
| `wind_in_bounds`                   | `|u| ≤ 100`, `|v| ≤ 100`, `|w| ≤ 50` m/s                                                 | hard violation if outside                       |
| `mu_nonnegative`                   | `state.mu_total > 0`                                                                     | hard violation if any cell ≤ 0                  |
| `dry_mass_drift_per_step`          | `|sum(mu_total_post) - sum(mu_total_pre)| / sum(mu_total_pre)`                           | warning ≥ 1e-9 per step (logged, not fatal)     |
| `column_water_drift_per_step`      | `|sum(qv + qc + qr + qi + qs + qg) - prev| / prev` weighted by mass                      | warning ≥ 1e-9 per step (logged, not fatal)     |

The harness reports **first violation step** and **total violation count** per invariant. It does NOT abort: the run completes so we see late-step behavior too.

---

## 5. Missing-coupling detection methodology

For each operator, after the run, we compute `total_abs_delta_over_run = sum_t mean_abs_delta_per_step[t]` for each Δ-field. We classify:

- `verdict = MISSING`: the operator is wired (called by `_physics_boundary_step`) but `total_abs_delta_over_run < eps_missing` for ALL its recorded fields. `eps_missing = 1e-30` per-field-value scale (i.e. literally zero up to FP noise).
- `verdict = INACTIVE`: the operator was not called this run (e.g. radiation when cadence > steps_total). Detected by a static analysis hook: the operator's "called" flag is set to `False` upfront in the harness recorder.
- `verdict = ACTIVE`: at least one recorded field has `total_abs_delta_over_run > eps_missing`.
- `verdict = NOISY_ZERO`: `total_abs_delta_over_run` is below `eps_missing` for some fields but above it for others — partial coupling failure. Reported as a comment.

The first_zero_delta_step per operator-field also lets us detect operators that became inactive partway through the run (e.g. a clip that fired and saturated for the rest of the run).

---

## 6. First-failure attribution algorithm

The attribution narrative is built top-down:

1. **First nonfinite cell in the entire state**: scan over steps, find smallest `t` where any leaf has `~isfinite`. Operator attribution = the operator after which the nonfinite first appeared (recorded by per-operator post-state nonfinite_count diff).
2. **First invariant violation**: smallest `t` where any invariant trips. Operator = the operator after which the invariant transitioned from satisfied to violated, found by comparing `invariant_value_after_operator[t]` for each operator in `_physics_boundary_step` order.
3. **First significant anchor divergence**: smallest hour where any wrfout-anchored field exceeds a tolerance threshold (cross-referenced from the operational_trace_hourly.json result that is reused if present).
4. **Coupling-chain attribution**: ordered pairs of (upstream → downstream) operators where upstream produces a state field consumed by downstream as a boundary condition. If upstream's Δ is zero AND downstream's Δ has the expected "depends on upstream" signature missing, flag the chain BROKEN.

Result: each of these four narratives appears in the report with the offending step + operator + field + magnitude.

---

## 7. JIT integration + overhead budget

The harness must run inside one JIT trace. The principal-mandated `diagnostic_on: bool` static-argname is the gate: when False, XLA dead-code-eliminates every harness call.

- **Hot-path discipline**: zero `jax.device_get`, zero `jax.debug.print`, zero `host_callback`. All scalar reductions are produced inside the scan body and carried in the scan carry.
- **JIT key**: `(hours, diagnostic_on)` are both static; `(diagnostic_on=True, hours=1.0)` produces one compiled program; `(diagnostic_on=False, hours=1.0)` produces a completely different (smaller) compiled program; production never sees the True branch.
- **Overhead budget**: ≤30% wall-clock vs `diagnostic_on=False` for a 24h Canary d02 run. Measured by a smoke test that runs the same 1h case both ways and reports the ratio in the report's `wall_seconds_diagnostic_overhead` field.

The harness adds, per step, O(K) scalar reductions where K = (≈8 fields × 7 operators × 2 metrics) ≈ 112 device-side reductions per step. Each is a single `jnp.mean(jnp.abs(...))` over the device array, so the dominant cost is one reduction kernel per metric, batched by XLA. Expected overhead on the production GPU is well under 30%.

---

## 8. Replacement of M9 placeholder tests

The two `_PLACEHOLDER.py` tests in `tests/savepoint/` are replaced with:

- `test_physics_couplers_PLACEHOLDER.py` → `test_physics_couplers.py`: runs `comprehensive_harness.run_diagnostic_harness(hours=1/60)` (1-minute case = 6 steps) on CPU and asserts that all four physics-coupler operators (`microphysics_thompson`, `surface_layer`, `mynn_pbl`, `rrtmg` if cadence allows) have non-`NO_DATA` verdicts in the resulting report.
- `test_operational_variables_PLACEHOLDER.py` → `test_operational_variables.py`: runs the same harness call and asserts the report contains valid `wrf_anchor_comparison` keys for the 16 operational fields (or marks them MISSING with a documented reason).

Both tests are CPU-only and run in < 60 seconds in CI.

A new `test_diagnostic_harness.py` validates the harness itself: runs 1h, asserts `verdict == "DIAGNOSIS_PRODUCED"`, asserts all eight invariants are present in the report.

---

## 9. Design call rationale (manager autonomy)

The principal is off the computer; this design is final without sign-off. Decisions that could have been escalated and the reasoning for the call made:

- **No new entry point in `physics_couplers.py`**: M12 worker is actively editing that file. Wrapping `_physics_boundary_step` in a thin recorder is non-conflicting.
- **No per-operator timing collection in v1**: per-operator wall time would require synchronous `block_until_ready` between operators inside the scan, which defeats the JIT. Profile that separately with Nsight, not here.
- **No spatial pattern (heat-maps) in v1**: scalar reductions only. Spatial divergence maps are already produced by `scripts/diagnostic_spatial_divergence_map.py` and `proofs/m9/divergence_map_v2.json`. The harness links to those by file path, not duplicates them.
- **`NOISY_ZERO` verdict added** between ACTIVE and MISSING because the partial-coupling case (one field of a multi-field operator is silently zero) is exactly what we expect to find first in M12.
- **JSON schema is versioned (`schema_version: diagnostic-harness-1.0`)**: future fields are additive; the manager-Opus does not need to re-read consumers when the schema is extended.

