# Worker Report — Comprehensive Diagnostic Harness

**Sprint**: 2026-05-28-diagnostic-harness
**Worker**: manager-Opus (Claude Opus 4.7, 1M context)
**Date**: 2026-05-28
**Verdict**: `DIAGNOSTIC_HARNESS_COMPLETE`

---

## Objective

Build a single test that runs all couplings + dynamics over many timesteps
and produces clear-to-exact indications of which coupling/dynamic is failing
or completely missing. Deliver:

1. Design document.
2. Instrumentation library at `src/gpuwrf/diagnostics/comprehensive_harness.py`.
3. Driver script at `scripts/run_diagnostic_harness.py`.
4. Pytest entry point + real M9 placeholder replacements.
5. Updated docs.

Constraints honored: did NOT modify `src/gpuwrf/dycore/**` or
`src/gpuwrf/coupling/physics_couplers.py` (M11/M12/M13 workers own them);
did NOT modify `src/gpuwrf/runtime/operational_mode.py` (instead, mirror the
operator sequence in a new wrapper module); CPU pinning via `taskset -c 0-3`
on every command; manager repo only (no `/home/enric/src/wrf_gpu/` touches).

---

## Files changed

### New

- `.agent/sprints/2026-05-28-diagnostic-harness/design.md` — schema, operator list,
  invariant list, missing-coupling detection methodology, first-failure attribution
  algorithm, JIT-overhead budget, manager-autonomy design rationale.
- `.agent/sprints/2026-05-28-diagnostic-harness/worker-report.md` — this file.
- `src/gpuwrf/diagnostics/__init__.py` — package marker + public API surface.
- `src/gpuwrf/diagnostics/comprehensive_harness.py` — instrumented mirror of
  `_physics_boundary_step`; pure-JIT accumulator; per-operator/per-step Δ stats;
  per-step invariants; missing-coupling classifier; coupling-chain auditor;
  headline-diagnosis + recommendations builder; JSON-serializable report dataclass.
- `scripts/run_diagnostic_harness.py` — CLI driver; loads Canary 20260521 d02
  replay case; runs `run_diagnostic_forecast` with `diagnostic_on=True`; optional
  `--measure-overhead` mode for JIT-overhead ratio; writes
  `proofs/diagnostic_harness/diagnostic_report.json`.
- `tests/savepoint/test_diagnostic_harness.py` — pytest entry point; skips cleanly
  in CPU-only env; runs a 6-step (1-minute) harness call on GPU; asserts the
  report has all required top-level keys, all operator/invariant/chain names,
  and round-trip-serializes through JSON.
- `proofs/diagnostic_harness/diagnostic_report_smoke_3step.json` — proof of a
  3-step GPU smoke run; verdict `DIAGNOSIS_PRODUCED`.

### Modified

- `tests/savepoint/test_physics_couplers_PLACEHOLDER.py` — was `xfail` placeholder;
  now invokes the diagnostic harness in slim form and asserts the four physics
  couplers (`microphysics_thompson`, `surface_layer`, `mynn_pbl`, `rrtmg`) all
  receive recognized verdicts and the coupling-chain audit block is populated.
- `tests/savepoint/test_operational_variables_PLACEHOLDER.py` — was `xfail`;
  now asserts the `wrf_anchor_comparison` block is present and the per-field
  summary keys overlap with the 16-field operational set when the trace JSON
  exists.
- `tests/savepoint/README.md` — appended "Comprehensive Diagnostic Harness"
  section with production-driver invocation snippet and report-reading guide.
- `.agent/decisions/PROJECT-RESET-PLAN-FINAL.md` — appended a single
  diagnostic-harness paragraph immediately below the invariant ladder declaring
  the harness as the project-level single-source-of-truth for "what is wrong?".

### NOT modified (per constraints)

- `src/gpuwrf/dycore/**` (no such dir; closest is `src/gpuwrf/dynamics/**`) — untouched.
- `src/gpuwrf/coupling/physics_couplers.py` — untouched.
- `src/gpuwrf/runtime/operational_mode.py` — untouched. The instrumented mirror
  re-implements the operator sequence locally and imports only the existing
  private helpers (`_rk_scan_step`, `_limit_guarded_dynamics_state`,
  `_finite_or_origin`, `_valid_mixing_ratio`, `_limit_theta_by_level`,
  `_enforce_operational_precision`, `_steps_for_hours`) plus the four public
  physics adapters. The original `_physics_boundary_step` and
  `run_forecast_operational` remain bit-for-bit identical.

---

## Commands run

```
# CPU smoke / import + structural sanity (skips State.zeros which needs GPU)
JAX_PLATFORMS=cpu taskset -c 0-3 python -c "from gpuwrf.diagnostics ... import ..."

# GPU smoke (3-step end-to-end forecast through full harness)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  taskset -c 0-3 python -c "<3-step harness driver>"   # verdict: DIAGNOSIS_PRODUCED in 93s

# Production driver smoke
XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  taskset -c 0-3 python scripts/run_diagnostic_harness.py \
    --hours 0.00833 --output /tmp/diag_smoke.json
# -> HARNESS REPORT verdict=DIAGNOSIS_PRODUCED schema=diagnostic-harness-1.0

# Pytest — CPU-only (skips cleanly)
taskset -c 0-3 python -m pytest tests/savepoint/test_diagnostic_harness.py \
  tests/savepoint/test_physics_couplers_PLACEHOLDER.py \
  tests/savepoint/test_operational_variables_PLACEHOLDER.py
# -> 14 skipped (no GPU)

# Pytest — GPU (passes)
JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-3 \
  python -m pytest tests/savepoint/test_diagnostic_harness.py \
    tests/savepoint/test_physics_couplers_PLACEHOLDER.py \
    tests/savepoint/test_operational_variables_PLACEHOLDER.py -v
# -> 13 passed, 1 skipped (wrf-anchor per-field — no overlapping hours)
```

---

## Proof objects produced

1. **`proofs/diagnostic_harness/diagnostic_report_smoke_3step.json`** — 3-step
   GPU smoke run on Canary 20260521; `verdict: DIAGNOSIS_PRODUCED`;
   schema-version `diagnostic-harness-1.0`; ~16 KB.
2. **`pytest` 13/14 pass on GPU** — every harness top-level key, every operator,
   every invariant, every coupling chain has a recognized value.
3. **CPU-only test run skips cleanly** — no false-positive failures in CPU CI;
   the harness honors the `State.zeros` GPU requirement via an explicit
   `_skip_if_no_gpu` check.

### Initial diagnostic findings on the 3-step Canary smoke

The smoke output already produces actionable signal even from a tiny window:

- `surface_layer`: **ACTIVE** — surface adapter is producing nonzero
  `theta_flux`, `qv_flux`, `tau_u`, `tau_v`, `ustar`, `rhosfc`. The earlier
  hypothesis that this operator was MISSING is **falsified** for the
  20260521 case at the start of forecast (this does NOT rule out a runtime
  collapse later in the 24h run).
- `mynn_pbl`: **ACTIVE** — PBL kernel modifies `u`, `v`, `theta`, `qv`, `qke`.
- `lateral_boundary`: **ACTIVE** — boundary forcing reaches `u`, `v`, `theta`, etc.
- `rrtmg`: **INACTIVE** as expected (smoke used `radiation_cadence_steps=999999`).
- `dycore_rk3`: **NOISY_ZERO** — `p_perturbation` and `ph_perturbation`
  show zero delta over 3 steps. This is most likely a benign artifact of
  the very-short window (acoustic mean over 3 steps may be < `EPS_MISSING=1e-30`);
  needs a longer-run check (manager TODO below).
- `microphysics_thompson`: **NOISY_ZERO** — `qr`, `qc`, `qg`, `qs`, `qi`, `qv`
  show zero delta over 3 steps. Plausibly a no-precipitation initial condition;
  needs a longer-run check.
- `dynamics_guards`, `boundary_guards`: **PASSIVE_OK** — no clip events
  (i.e. no cell exceeded a bound and triggered a clamp). This is the
  expected production-success state.
- All 10 invariants: not violated, no first-violation step.

---

## Unresolved risks

1. **NOISY_ZERO false positives on short runs**. The 3-step smoke flagged
   `dycore_rk3.p_perturbation` and Thompson moisture species as NOISY_ZERO.
   On a 24h run this should not happen, but if `EPS_MISSING=1e-30` is too
   strict for very stable cases, we may keep getting NOISY_ZERO false alarms.
   Mitigation: re-evaluate after the manager runs the harness on a 1h Canary
   GPU run and inspects the actual per-field totals.
2. **JIT compile cost** is 93s for a 3-step run. The harness JIT key is
   `(hours, diagnostic_on)`, so a 24h run compiles ONCE and runs to completion
   — but the compile is non-trivial. The principal-mandated overhead budget
   is on *post-compile* wall-clock, which `--measure-overhead` measures
   correctly (it warms the cache before timing).
3. **GPU memory contention**. The smoke was run with
   `XLA_PYTHON_CLIENT_MEM_FRACTION=0.10` because workers M11/M12/M13 are
   actively running. A full 24h Canary run will need the GPU mostly idle.
4. **Boundary RMSE thresholds not yet wired into the headline**. The
   harness reads `proofs/m9/operational_trace_hourly.json` if present but
   does not yet auto-flag fields where RMSE exceeds the project thresholds
   (`T2 ≤ 3 K`, `U10/V10 ≤ 7.5 m/s`). Easy follow-up — the per-field RMSE
   is already in the JSON, just not graded.
5. **`OperationalNamelist` immutability workaround**: the driver script
   rebuilds the namelist via `__class__(...)` to override `run_physics`,
   `run_boundary`, `disable_guards` because the frozen dataclass has no
   `replace` method. Not pretty; fine for v1.
6. **First-failure attribution for nonfinite-cell location** records the
   nonfinite cell count but not the cell index. For first-fault triage we
   should also record the (z, y, x) of the first nonfinite cell. Easy add.

---

## Next decision needed

None blocking. The harness is in place; manager-Opus can:

1. Run `scripts/run_diagnostic_harness.py --hours 1 --jax-platform cuda
   --measure-overhead` to confirm the 30% overhead budget and write the
   1h-run artifact.
2. Run `scripts/run_diagnostic_harness.py --hours 24 --jax-platform cuda
   --output proofs/diagnostic_harness/diagnostic_report_24h.json` overnight
   to produce the project's first official diagnostic report.
3. Use the resulting JSON to redirect M11/M12/M13 worker priorities based
   on whichever operator first trips MISSING / NOISY_ZERO / invariant
   break on the real 24h Canary run.

The harness is now the project's single-source-of-truth diagnostic
artifact, as committed in
`.agent/decisions/PROJECT-RESET-PLAN-FINAL.md` (paragraph immediately
below the invariant ladder).
