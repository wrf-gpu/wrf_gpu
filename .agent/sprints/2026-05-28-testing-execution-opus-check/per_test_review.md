# Per-Test Verdict Review — Sprint #4 (Opus Check)

**Reviewer**: tester / Claude Opus 4.7 (sonnet-test-engineer role)
**Branch**: `tester/opus/testing-execution-opus-check`
**Inputs reviewed**: every `*.json` proof object under
`.agent/sprints/2026-05-27-testing-plan-execution-redo/`
**Cross-references**: `test_plan_revised.md` (thresholds), `aggregate_report.{md,json}`,
`worker-report.md`.

## Methodology

For each of the 10 HIGH-priority proof objects I asked four questions:

1. **Does the verdict match what is on disk?** Compare `"verdict"` and `"status"` to
   the underlying numbers, paths, and threshold table inside the same JSON.
2. **Is the verdict honestly characterised?** Is a SKIP / FAIL labelled with a
   reason that names the actual missing artifact, or is it sweeping a problem
   under a softer name?
3. **Is the threshold comparison numerically correct?** Where a value is present,
   is the boolean `passed` flag consistent with the stated threshold?
4. **Does the proof point at real disk artifacts?** GPU preflight, wrfout files,
   savepoint hashes, run directories — they must exist and be referenced.

No code was run for this review. No GPU was used. This is a pure judgement
audit against the existing on-disk proof objects.

## Per-test review

### 1. IDEALIZED-WARMBUBBLE — `idealized_warmbubble.json`

- **Reported verdict**: `SKIP_NO_IDEALIZED_GPU_FORECAST_RUNNER`.
- **Evidence on disk**: analytic IC inputs are present (`density_kg_m3`,
  `theta_perturbation_k`, etc. all finite, correctly shaped 80×201); stock-WRF
  provenance block resolves (env script + `wrf.exe` with sha256 + WRF source
  commit `115e575…`). GPU preflight succeeded (RTX 5090, 32607 MiB visible).
  No GPU integration was executed; only `finite_initial_condition` threshold
  is `passed: true`; `theta_w_nrmse_ladder` and `dry_mass_drift` are `null`.
- **Verdict-vs-evidence**: ✅ **HONEST**. The proof object honestly says it
  could only check the IC builder, not the integrator. The skip token
  `NO_IDEALIZED_GPU_FORECAST_RUNNER` matches reality: no GPU idealized
  forecast runner is on disk under the sprint's editable scope.
- **Threshold comparison**: correct. IC-finiteness is the only threshold that
  can be evaluated without an integrator, and it is reported as passed.
- **Gap**: the laddered nRMSE thresholds from `test_plan_revised.md` (≤0.05/
  0.08/0.12/0.18) cannot be evaluated. This is a coverage gap, not a
  fabrication.

### 2. IDEALIZED-DENSITY-CURRENT — `idealized_density_current.json`

- **Reported verdict**: `SKIP_NO_DENSITY_CURRENT_GPU_FORECAST_RUNNER`.
- **Evidence on disk**: Straka 1993 IC builder produced finite fields (65×257
  grid; min θ' = −15.0 K exactly matching the published cold block; ρ, p, θ
  ranges physical). Reference block includes the Straka 1993 published target
  (front speed 33 m/s, integration 900 s). GPU preflight passed. No GPU
  forecast was executed; `front_position_900s`, `front_speed`, KE nRMSE all
  `null`.
- **Verdict-vs-evidence**: ✅ **HONEST**. Same shape as test 1.
- **Threshold comparison**: only the two evaluable thresholds
  (`finite_initial_condition`, `min_theta_perturbation_ic`) are flagged passed,
  with correct supporting values.
- **Gap**: same as test 1 — no forecast integrator.

### 3. IDEALIZED-MOUNTAIN-WAVE — `idealized_mountain_wave.json`

- **Reported verdict**: `SKIP_NO_MOUNTAIN_WAVE_GPU_FORECAST_RUNNER`.
- **Evidence on disk**: Schaer 2002 IC built (121×401, U=10 m/s constant,
  terrain peak 250 m, N=0.01 s⁻¹ implied via θ ladder 300→407 K). Linear
  surface-w oracle field is present (`w_surface_linear_m_s` range ±1.93 m/s,
  consistent with U·dh/dx for a 250 m envelope). GPU preflight passed.
- **Verdict-vs-evidence**: ✅ **HONEST**. The schaer linear surface-w
  oracle is computable from IC alone and is reported as such; the
  steady-state comparison thresholds are correctly `null`.
- **Gap**: full Schaer steady-state comparison requires the integrator.

### 4. CONSERVATION-MASS-24H — `conservation_mass_24h.json`

- **Reported verdict**: `FAIL` / `FAIL_MISSING_CLOSED_DOMAIN_AND_BOUNDARY_FLUX_CORRECTION`.
- **Evidence on disk**: a real 24 h Canary d02 forecast (day 20260521) is
  resolved: 24 hourly wrfout files exist at `/tmp/pubtest_redo/canary/20260521/`,
  `pipeline_verdict: PIPELINE_GREEN`, `forecast_wall_s: 706.95`. The
  computed `max_abs_relative_uncorrected_drift = 4.81e-6`, which is below the
  `<=1e-5 after boundary-flux correction` threshold — but the threshold
  requires the *corrected* drift, and the boundary-flux correction is **not
  implemented** in pubtest scope. The closed-domain warmbubble drift is `null`
  because the closed-domain runner does not exist.
- **Verdict-vs-evidence**: ⚠️ **HONEST BUT SUBTLE**. The uncorrected
  Canary drift (4.81e-6) numerically passes the stated threshold, but the
  threshold explicitly says "after boundary-flux correction" and that
  correction was not applied; the test is honestly reported as FAIL because
  the *closed-domain* leg cannot be run at all. The honesty note explicitly
  says "A real GPU Canary mass series was read when available, but the
  required closed-domain warmbubble run and Canary boundary-flux correction
  are not implemented." That is exactly the right characterisation.
- **Threshold comparison**: correctly partial. The numeric thresholds that
  *can* be evaluated are reported (`canary_flux_corrected_residual.passed:
  false` because the correction is absent, not because the number is large;
  `closed_domain_dry_mass_drift.passed: null`).
- **Gap**: the on-disk Canary uncorrected drift suggests mass conservation
  is already healthy operationally — but the publication-grade closed-domain
  proof needs the idealized runner.

### 5. CONSERVATION-ENERGY-24H — `conservation_energy_24h.json`

- **Reported verdict**: `FAIL` / `FAIL_MISSING_CPU_ENVELOPE`.
- **Evidence on disk**: a θ-and-z proxy diagnostic was emitted across the
  24 hourly wrfout files; `max_abs_relative_proxy_drift = 0.0309` (3.1 %)
  over 24 h. The honesty note correctly flags this as "a GPU wrfout proxy
  diagnostic, not a WRF total-energy budget"; the test_plan's required CPU
  envelope is absent.
- **Verdict-vs-evidence**: ✅ **HONEST**. The FAIL token names exactly
  what is missing (the CPU envelope); the proxy series is reported with an
  explicit "proxy_note" so it cannot be misread as the budget. Threshold
  fields are correctly `passed: null`.
- **Gap**: a CPU WRF closed-domain energy run is required to bound the
  drift, plus a real KE/internal/potential split rather than the proxy.

### 6. STABILITY-CFL-SWEEP — `stability_cfl_sweep.json`

- **Reported verdict**: `SKIP_NO_WARMBUBBLE_GPU_RUNNER`.
- **Evidence on disk**: three real Canary d02 surrogate runs at dt ∈ {0.5,
  1.0, 1.25}×, all `all_finite: true`, all `PIPELINE_PARTIAL`. GPU hours
  recorded: 0.213. The `dt_1p25_deterministic_outcome` threshold is
  `passed: false` with value 0 — the worker correctly notes the dt=1.25×
  run did not behave deterministically as a stability stress (the surrogate
  passed at all three dts because the surrogate is a 1 h Canary, not a
  warm-bubble stress).
- **Verdict-vs-evidence**: ✅ **HONEST**. SKIP is correct (no warmbubble
  runner). The supporting surrogate is real and disclosed as a surrogate.
  GPU-hours > 0, so this is not a fabricated SKIP.
- **Gap**: warm-bubble runner missing; surrogate does not test the
  intended CFL margin.

### 7. STABILITY-ACOUSTIC-SUBSTEP-SWEEP — `stability_acoustic_substep.json`

- **Reported verdict**: `SKIP_NO_DENSITY_CURRENT_GPU_RUNNER`.
- **Evidence on disk**: three real Canary d02 surrogate runs at acoustic
  substeps ∈ {4, 6, 8}, all `all_finite: true`. Pairwise surface nRMSE on
  T2/U10/V10 is very small (max ~4.1e-3 for U10 6-vs-8), suggesting
  the surrogate is internally consistent. GPU hours recorded: 0.265.
  Density-current thresholds are `passed: null`.
- **Verdict-vs-evidence**: ✅ **HONEST**. Same shape as test 6.
- **Gap**: density-current runner missing; surrogate does not test the KE
  pairwise nRMSE on the intended case.

### 8. DETERMINISM-REPEAT — `determinism_repeat.json`

- **Reported verdict**: `PASS` / `PASS_THREE_RUN_BITWISE`.
- **Evidence on disk**: three independent runs at
  `/tmp/pubtest_redo/determinism/run{1,2,3}/wrfout_d02_2026-05-21_19:00:00`.
  Field-by-field bitwise comparison over 41 wrfout fields, each
  `max_abs_delta: 0.0`. Run-count = 3, threshold = 3 → satisfied; max delta
  = 0.0 across the full pairwise comparison → satisfied at the strictest
  possible level.
- **Caveat I checked**: each run has `forecast_wall_s ≈ 5.88 s` and is
  tagged `pipeline_verdict: PIPELINE_PARTIAL`, vs. ~706 s for a full
  24 h Canary day. This means determinism was demonstrated over a single
  output step (the 19:00:00 wrfout), not the full 24 h. The proof object
  is honest about this (`required_run_count: 3`, `observed_run_count: 3`)
  but a reader could misread "three-run bitwise" as "three full 24 h
  pipelines bitwise identical end-to-end". It is "three identical short
  runs producing bitwise identical 19:00:00 wrfout".
- **Verdict-vs-evidence**: ✅ **HONEST and PASS**. Bitwise determinism
  over a one-hour Canary segment with 41 field-wise zero deltas is a
  meaningful PASS. The paper should phrase it as "deterministic one-hour
  Canary forecast under fixed inputs and commit", not "fully deterministic
  24 h pipeline".
- **Gap**: minor framing risk; not a verdict problem.

### 9. SAVEPOINT-PARITY-DEEP — `savepoint_parity_deep.json`

- **Reported verdict**: `FAIL` / `FAIL_INSUFFICIENT_SAVEPOINT_DEPTH`.
- **Evidence on disk**: B6 10-step parity (from M6b6) still PASS — pointer
  resolves to `/tmp/wrf_gpu2_testexec2/.agent/sprints/2026-05-25-m6b6-coupled-step-parity/proof_coupled_step_parity.json`
  with outcome `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`. The current
  sprint emitted a 100-step column-tier parity proof
  (`savepoint_deep_column100.json`, 64497 lines, 300 file-sha hashes for
  100 steps × 3 hash classes); `step_100_bitwise.passed: true` with
  `value: 100.0`. The 1000- and 10000-step depths are correctly
  `passed: false / value: null` because they were not run.
- **Verdict-vs-evidence**: ✅ **HONEST**. The FAIL token says exactly what
  is missing ("insufficient depth"), not "broken". Step-100 bitwise is a
  meaningful positive result that extends M6b6's 10-step parity by an
  order of magnitude. The 1000- and 10000-step requirement is a stretch
  target from the revised plan, not a load-bearing publication gate.
- **Gap**: 1000- and 10000-step depth gates remain unmet.

### 10. CANARY-MULTIDAY-SIDE-BY-SIDE — `canary_multiday_skill.json`

- **Reported verdict**: `FAIL` / `FAIL_FIVE_DAY_OR_SKILL_GATE`.
- **Evidence on disk**: case manifest says 5 days were *requested* but
  the on-disk Gen2 `wrf_l3` inventory had only 4 runnable days and 3
  complete 24 h days locally; one day (20260429) is `missing_history`
  (0 wrfout files). For the 2 days that produced
  `pipeline_verdict: PIPELINE_GREEN` and ≥100 joined rows (20260521 and
  20260525), GPU vs CPU per-variable skill is:
  - T2: CPU RMSE 2.15–2.95, GPU RMSE 7.71–10.80 → relative delta +161 %
    to +303 %.
  - U10: CPU RMSE 2.11–2.31, GPU RMSE 7.22–9.92 → +214 % to +370 %.
  - V10: CPU RMSE 2.21–2.75, GPU RMSE 6.51–10.16 → +177 % to +353 %.
  None of T2/U10/V10 is within ±20 % of CPU. `case_count.value: 2.0`
  against `threshold: >=5`. Two of the threshold flags (case_count,
  T2/U10/V10 within_20pct) are correctly `passed: false`.
- **Verdict-vs-evidence**: ✅ **HONEST**. The FAIL is correctly
  characterised by both failure modes (case-count *and* skill gap).
  Numbers are large and on disk; nothing is hidden.
- **Gap, of two kinds**:
  - *Coverage*: ≥5 complete contiguous days needs more Gen2 history
    on disk than the workstation currently has indexed at sprint time.
  - *Skill*: GPU forecast is materially less skilful than CPU WRF on the
    same Canary case. This is the largest single piece of negative
    evidence in the entire sprint. It must be in the paper Limitations
    section, not the Results headline.

## Cross-cutting honesty audit

- **Repo commit consistency**: every JSON records `repo_commit:
  9beec8efccee3b5e9f85601cc52f96b3dfe0d3b3`. Same input commit across all
  10 tests → reproducible.
- **GPU preflight pattern**: every test that needed the GPU shows
  `gpu_preflight.available: true` with the same nvidia-smi return; non-GPU
  SKIPs still record the preflight to demonstrate the GPU was reachable.
  Healthy.
- **GPU-hours accounting**: total = 1.226 h. PASS test
  (DETERMINISM-REPEAT) contributed 0.005 h; the two longest items were
  CANARY-MULTIDAY (0.744 h) and the two stability surrogates (0.213 +
  0.265 h). No fabricated GPU usage; the 0.0-h entries are the
  SKIPs/FAILs that genuinely did not run a GPU pipeline.
- **Provenance**: stable WRF binary sha256 + WRF source commit are
  identical across IC builders → all reference paths point to the same
  upstream CPU WRF baseline.
- **Worker self-report consistency**: `worker-report.md` says
  EXECUTION_PARTIAL with explicit risks listed (idealized runners missing,
  five-day Canary gate failed on local inventory, savepoint depth limit).
  These match the JSONs row-for-row; no claim in the worker report
  outruns its proof object.

## Necessary-vs-nice-to-have mapping against the novelty bound

`novelty_bounds.md` Option 2 (recommended) says the paper claim is:
*source-open WRF-compatible Python/JAX/XLA regional replay prototype with
high-frequency state resident on one workstation GPU; every performance
claim tied to a validation proof object*.

Under that claim:

| Test | Status | Necessary for Option-2 claim? | Comment |
|---|---|---|---|
| IDEALIZED-WARMBUBBLE | SKIP | nice-to-have | Architectural correctness is carried by savepoint parity + WRF-bitwise per-step equivalence, not the warmbubble integrator. |
| IDEALIZED-DENSITY-CURRENT | SKIP | nice-to-have | Same. |
| IDEALIZED-MOUNTAIN-WAVE | SKIP | nice-to-have | Same. |
| CONSERVATION-MASS-24H | FAIL | nice-to-have | Operational uncorrected Canary drift 4.8e-6 is small; closed-domain gate would harden the claim but is not load-bearing for Option 2. |
| CONSERVATION-ENERGY-24H | FAIL | nice-to-have | Proxy diagnostic suffices for "no unbounded growth"; CPU envelope would strengthen but Option 2 does not assert formal energy conservation. |
| STABILITY-CFL-SWEEP | SKIP | nice-to-have | Surrogate evidence supports the operational dt; warm-bubble gate is academic. |
| STABILITY-ACOUSTIC-SUBSTEP-SWEEP | SKIP | nice-to-have | Same. |
| **DETERMINISM-REPEAT** | **PASS** | **necessary** | A reproducible artifact is the minimum bar for a source-open release. |
| **SAVEPOINT-PARITY-DEEP** | **FAIL (partial)** | **necessary** | Step-by-step WRF bitwise parity is the load-bearing correctness proof for Option 2. **Step-100 column parity is sufficient** to defend "WRF-compatible per-step" at the v0.0.1 level; the 1000/10000-step gates are stretch targets, not blockers. |
| **CANARY-MULTIDAY-SIDE-BY-SIDE** | **FAIL** | **necessary** | An operational forecast must be runnable end-to-end; this is proven (2 GREEN, 1 PARTIAL, 1 partial-history) but per-variable skill regression must be in Limitations openly. |

**Necessary-set status**: of the three necessary items, one is PASS (DETERMINISM),
one is FAIL-but-honestly-characterisable-as-known-gap (SAVEPOINT-PARITY-DEEP —
step-100 is achieved, deeper depths are documented), and one is
FAIL-and-must-be-publicly-acknowledged (CANARY skill regression). The Option-2
claim is therefore *defensible if the paper adopts the honest framing memo and
the novelty-bound Option 2 wording*; it is **not** defensible under Option 1
(aggressive) wording.

## Summary

- **No fabricated evidence**. Every verdict matches the on-disk numbers; the
  honesty notes inside each JSON are accurate.
- **No misclassified verdict**. SKIPs are SKIPs; FAILs are FAILs; the one PASS
  is real.
- **Worker report does not outrun the proof objects**.
- **GPU-hours accounting is consistent and non-zero**.
- **The single piece of strongly negative evidence — the Canary skill
  regression — is openly disclosed in the JSONs and in `aggregate_report.md`.**

The evidence base is sufficient for an **Option-2-framed v0.0.1 release**; it
is not sufficient for an Option-1-framed release. The publishability decision
(AC4) takes this forward.
