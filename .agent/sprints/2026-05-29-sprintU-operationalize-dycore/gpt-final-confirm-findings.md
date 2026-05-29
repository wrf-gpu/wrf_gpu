# Sprint U Final Confirm Findings

Reviewer: GPT-5.5 xhigh (Codex)
Date: 2026-05-29
Branch reviewed: `worker/opus/f7d-pressure-mass-fix` at `79a4e48`
Scope: final decisive confirmation of the dry dycore close, per `gpt-final-confirm-prompt.md`.

## Decision

**CLOSE-CONFIRMED** confidence 9/10.

The remaining P0-1 blocker is fixed. The public operational entries now initialize the scan carry with `force_fp64=bool(namelist.force_fp64)`, matching the in-scan precision enforcement, and both public entry points run a mixed-precision real Canary d02 state to fp64 finite output with no scan dtype crash and no `FutureWarning`.

## Findings

### 1. Public-entry fp64 blocker is closed

The production real-case namelist sets `force_fp64=True` along with the F7 operational dycore flags at `src/gpuwrf/integration/daily_pipeline.py:179` through `src/gpuwrf/integration/daily_pipeline.py:195`, and the daily path calls `run_forecast_operational` at `src/gpuwrf/integration/daily_pipeline.py:268` through `src/gpuwrf/integration/daily_pipeline.py:270`.

The actual blocker line is now fixed:

- `run_forecast_operational` builds `initial_operational_carry(_enforce_operational_precision(state, force_fp64=bool(namelist.force_fp64)))` at `src/gpuwrf/runtime/operational_mode.py:1553` through `src/gpuwrf/runtime/operational_mode.py:1560`.
- `run_forecast_operational_with_limiter_diagnostics` does the same at `src/gpuwrf/runtime/operational_mode.py:1613` through `src/gpuwrf/runtime/operational_mode.py:1620`.
- The diagnostic debug entry is also fixed at `src/gpuwrf/runtime/operational_mode.py:1673` through `src/gpuwrf/runtime/operational_mode.py:1680`.
- The in-scan step still enforces the same fp64 policy before returning the next carry at `src/gpuwrf/runtime/operational_mode.py:1471`.
- The fp64 enforcement is real, not canonicalized back to loaded dtype: `_enforce_operational_precision(force_fp64=True)` casts every `STATE_FIELD_ORDER` leaf and calls `state.replace(_cast=False, **updates)` at `src/gpuwrf/runtime/operational_mode.py:299` through `src/gpuwrf/runtime/operational_mode.py:311`; `State.replace(_cast=False)` preserves supplied dtypes at `src/gpuwrf/contracts/state.py:566` through `src/gpuwrf/contracts/state.py:588`.

The submitted public proof says the real d02 input has raw `theta/u/v` as float32, the public entry ran without scan dtype crash or FutureWarning, and all 10 checked prognostics are float64 and finite at `proofs/sprintU/public_entry_fp64.txt:16` through `proofs/sprintU/public_entry_fp64.txt:21`.

I independently reproduced the public-entry proof:

```text
taskset -c 0-3 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY'
# _build_real_case(DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10))
# run_forecast_operational(case.state, case.namelist, nl.dt_s / 3600.0)
PY
```

Observed:

```text
force_fp64 True
raw theta=float32 u=float32 v=float32
future_warnings 0
theta/u/v/w/mu/ph/p_total/ph_total/mu_total/qv: all float64, all finite
ALL_FP64 True
ALL_FINITE True
```

I also ran the same one-step smoke through `run_forecast_operational_with_limiter_diagnostics`; it produced the same all-fp64/all-finite result and returned limiter diagnostic keys without scan dtype failure.

### 2. Same-class operator dtype risk is resolved

By my prior reasoning, the residual risk in operators allocating buffers from incoming field dtype was conditional on the public entry handing them fp32 `theta/u/v`. That condition is gone: the public entry upcasts before the first scan step, and the scan output is enforced again at `src/gpuwrf/runtime/operational_mode.py:1471`.

I do not have a remaining specific operator precision-drop blocker to name. The prior same-class risk is resolved by the entry-upcast fix.

### 3. Idealized regression gate is honest and unregressed

The older regression proof records the expected pytest result, `4 passed`, at `proofs/sprintU/fp64_regression_gate.txt:1` through `proofs/sprintU/fp64_regression_gate.txt:3`. The close-gate tests assert `verdict == "PASS"` for warm bubble and density current at `tests/idealized/test_dycore_close_gate.py:47` through `tests/idealized/test_dycore_close_gate.py:76`; the original idealized tests also assert `verdict == "PASS"` at `tests/idealized/test_warm_bubble.py:24` through `tests/idealized/test_warm_bubble.py:34` and `tests/idealized/test_density_current.py:24` through `tests/idealized/test_density_current.py:35`.

The named post-entry artifact `proofs/sprintU/fp64_regression_gate_postentry.txt` is zero bytes in this worktree, so I did not rely on it. I independently reran the two underlying idealized cases into `/tmp/wrf_gpu2_sprintu_final_gate` to avoid modifying repo proof artifacts:

```text
warm_status RAN_TO_COMPLETION
warm_verdict PASS
warm thermal_rise_500s 1924.3475674059687 True
warm max_abs_w_500s 11.680187856366164 True
warm theta_prime_max_500s 1.9200603163246228 True
warm relative_mass_drift 0.0 True
density_status RAN_TO_COMPLETION
density_verdict PASS
density front_position_900s 14150.0 True
density theta_prime_min_900s -9.970995032353756 True
density max_abs_w_900s 14.57491907301182 True
density relative_mass_drift 2.2509471651212643e-09 True
```

Those values match the prior gate within roundoff. This is not a numeric regression. The empty post-entry text file is proof hygiene, not a dycore close blocker, because the gate was independently reproduced here and the PASS assertions are in code.

### 4. P0-2 full 3D deformation deferral is acceptable

The operational real-case path uses `diff_6th_opt=2` at `src/gpuwrf/integration/daily_pipeline.py:187` through `src/gpuwrf/integration/daily_pipeline.py:190`, and the real-case proof records `deformation_momentum_diffusion_P0_2=false`, `diff_6th_opt=2`, and `precision=fp64` at `proofs/sprintU/real_case_smoke.json:2` through `proofs/sprintU/real_case_smoke.json:17`.

The deferral is now explicit and honest: full 3D `u/v/w` deformation tensor work is deferred to Phase B at `proofs/f7/DYCORE_STATUS.md:51` through `proofs/f7/DYCORE_STATUS.md:66`, with remaining Phase-B scope listed at `proofs/f7/DYCORE_STATUS.md:68` through `proofs/f7/DYCORE_STATUS.md:73`. That is acceptable because operational real-case runs do not enable deformation momentum diffusion, and the implemented one-row `u/w` subcase is only load-bearing for the 2D Straka gate.

### 5. No remaining true dycore-close blocker found

The canonical Straka WRF-vs-JAX proof remains a PASS and finite through touchdown at `proofs/sprintU/straka_canonical_parity.json:196` through `proofs/sprintU/straka_canonical_parity.json:201`, with the comparison series finite at `proofs/sprintU/straka_canonical_parity.json:15` through `proofs/sprintU/straka_canonical_parity.json:130`. The operational status file honestly names Phase-B exclusions rather than overclaiming terrain/map-factor/LBC/moist/per-cell parity closure at `proofs/f7/DYCORE_STATUS.md:68` through `proofs/f7/DYCORE_STATUS.md:73`.

I do not see a concrete remaining blocker that should prevent declaring the dry dycore operational-ready and moving to Phase B.

## Commands Run

```text
sed/nl inspections of PROJECT_CONSTITUTION.md, AGENTS.md, sprint prompt, Sprint U reports, local skills, source, tests, and proofs
git status --short
git rev-parse --short HEAD
git log --oneline -5
taskset -c 0-3 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY' ... run_forecast_operational public-entry dtype smoke ... PY
taskset -c 0-3 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY' ... run_forecast_operational_with_limiter_diagnostics dtype smoke ... PY
taskset -c 0-3 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python - <<'PY' ... warm bubble + density current gates into /tmp ... PY
git restore -- tracked proof files refreshed by the /tmp idealized run side effect
```

## Proof Objects Produced

This report is the proof object for the final confirmation:

`/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-29-sprintU-operationalize-dycore/gpt-final-confirm-findings.md`

SPRINTU_FINAL_COMPLETE
**CLOSE-CONFIRMED** confidence 9/10
