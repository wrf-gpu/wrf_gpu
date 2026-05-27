# Executable Publication Testing Plan

## Execution Target

Recommended execution sprint ID: `2026-05-27-publication-testing-plan-execution`.

All proof paths below use:

`.agent/sprints/2026-05-27-publication-testing-plan-execution/`

The execution sprint should write code only after its own sprint contract freezes file ownership. This plan intentionally does not modify model code or run fresh measurements.

## Priority Policy

- `HIGH`: required before an arXiv submission can responsibly frame the project as an open-source WRF-compatible GPU port.
- `MEDIUM`: materially strengthens the paper but can be deferred if the paper is framed as a prototype with explicit limitations.
- `LOW`: future work or release hardening beyond the first paper.

The HIGH-priority GPU budget is designed to stay under 24 GPU-hours. Current local evidence suggests a 24 h Canary GPU pipeline is about 12.2 min in iteration 2, but the plan budgets conservatively because idealized cases, diagnostics, I/O, and reruns add overhead.

## Test Matrix

### IDEALIZED-WARMBUBBLE

- Priority: HIGH
- What it proves: Buoyant response, acoustic-step stability, vertical-velocity growth, symmetry, and finite-state behavior in a controlled convective trigger.
- Inputs needed: WRF warm-bubble or supercell-style ideal namelist; matching JAX initial-state builder; CPU WRF reference output at 5, 10, 20, and 30 min; grid and timestep metadata; no active microphysics unless explicitly testing moist behavior.
- How to run it:
  - Write `scripts/pubtest_prepare_wrf_ideal.py --case warmbubble --output <proof>/inputs/warmbubble`.
  - Write `scripts/pubtest_run_wrf_reference.py --case warmbubble --minutes 30 --output <proof>/wrf/warmbubble`.
  - Write `scripts/pubtest_run_gpu_ideal.py --case warmbubble --minutes 30 --output <proof>/gpu/warmbubble`.
  - Write `scripts/pubtest_compare_ideal.py --case warmbubble --wrf <proof>/wrf/warmbubble --gpu <proof>/gpu/warmbubble --output <proof>/idealized_warmbubble.json`.
  - Commands must be run with `taskset -c 0-3`.
- Pass/fail criteria: all fields finite; domain dry-mass relative residual `<= 1e-10` for closed-domain mode or boundary-budget residual `<= 1e-6`; normalized RMSE vs CPU WRF for theta perturbation and vertical velocity `<= 0.05` at each saved lead; max vertical-velocity lead-time error `<= 10%`; horizontal symmetry error `<= 1e-6` of peak perturbation for symmetric setup.
- Proof object: `.agent/sprints/2026-05-27-publication-testing-plan-execution/idealized_warmbubble.json`, plus `idealized_warmbubble_summary.md`.
- Estimated wall-time + GPU budget: 20-40 min wall; 0.5 GPU-hour budget.

### IDEALIZED-DENSITY-CURRENT

- Priority: HIGH
- What it proves: Cold-pool propagation, sharp-gradient handling, near-surface dynamics, diffusion behavior, and mass/energy budget sanity.
- Inputs needed: Analytic density-current initial condition or published benchmark configuration; CPU WRF/reference run; output at 5, 10, 15, 30, and 60 min.
- How to run it:
  - Write `scripts/pubtest_prepare_density_current.py --output <proof>/inputs/density_current`.
  - Run CPU reference and GPU path with identical grid, timestep, and diffusion options.
  - Compare with `scripts/pubtest_compare_ideal.py --case density_current`.
- Pass/fail criteria: all fields finite; dry-mass relative residual `<= 1e-10` closed-domain or `<= 1e-6` flux-corrected; cold-front location within `1` horizontal grid cell of reference at each lead; minimum theta perturbation within `0.5 K`; kinetic-energy time series normalized RMSE `<= 0.10`; no negative pressure or water species.
- Proof object: `.agent/sprints/2026-05-27-publication-testing-plan-execution/idealized_density_current.json`.
- Estimated wall-time + GPU budget: 30-60 min wall; 1.0 GPU-hour budget.

### IDEALIZED-MOUNTAIN-WAVE

- Priority: HIGH
- What it proves: Terrain-following coordinate, pressure-gradient force, vertical propagation, and top/bottom boundary behavior.
- Inputs needed: WRF `em_hill2d_x` or equivalent mountain-wave case; terrain and map-factor metadata; CPU WRF reference; vertical cross-section outputs.
- How to run it:
  - Write `scripts/pubtest_prepare_wrf_ideal.py --case mountain_wave`.
  - Run CPU WRF and GPU path for 1 h and 3 h saved intervals if feasible.
  - Compare with `scripts/pubtest_compare_mountain_wave.py`.
- Pass/fail criteria: all fields finite; vertical velocity cross-section normalized RMSE `<= 0.10`; dominant wave phase shift `<= 1` grid cell horizontally and vertically; pressure perturbation normalized RMSE `<= 0.10`; dry-mass budget residual `<= 1e-6` if open-boundary flux correction is used.
- Proof object: `.agent/sprints/2026-05-27-publication-testing-plan-execution/idealized_mountain_wave.json`.
- Estimated wall-time + GPU budget: 45-90 min wall; 1.5 GPU-hours budget.

### CONSERVATION-MASS-24H

- Priority: HIGH
- What it proves: The implementation does not silently create or destroy dry air over long integrations except through explicit boundary fluxes.
- Inputs needed: One closed-domain idealized case and one Canary replay case; initial/final dry mass; boundary mass-flux diagnostics for open-boundary run; FP64 mass path.
- How to run it:
  - Write `scripts/pubtest_mass_budget.py --case warmbubble --hours 24 --closed-domain --output <proof>/conservation_mass_ideal.json`.
  - Write `scripts/pubtest_mass_budget.py --case canary-20260521 --hours 24 --boundary-flux-corrected --output <proof>/conservation_mass_canary.json`.
- Pass/fail criteria: closed-domain relative dry-mass drift `<= 1e-10`; Canary boundary-flux-corrected residual `<= 1e-5`; no nonfinite mass fields; residual time series monotonic drift must be explained if present.
- Proof object: `.agent/sprints/2026-05-27-publication-testing-plan-execution/conservation_mass_24h.json`.
- Estimated wall-time + GPU budget: 45-90 min wall if reusing ideal runs and one Canary run; 1.5 GPU-hours budget.

### CONSERVATION-ENERGY-24H

- Priority: HIGH
- What it proves: Dynamics and physics changes are not producing unbounded or unexplained energy drift.
- Inputs needed: Closed-domain dry ideal case; optional physics-off Canary run; definitions for kinetic, internal/dry thermodynamic, and potential energy terms; boundary and surface-flux terms if open boundaries or physics are enabled.
- How to run it:
  - Write `scripts/pubtest_energy_budget.py --case warmbubble --hours 24 --closed-domain --output <proof>/conservation_energy_ideal.json`.
  - If Canary is included, run `--flux-corrected` and emit an assumptions block.
- Pass/fail criteria: closed-domain dry-energy drift `<= 0.1%` over 24 h; flux-corrected Canary residual `<= 0.5%`; no unexplained step jump over `0.05%`; assumptions recorded in JSON.
- Proof object: `.agent/sprints/2026-05-27-publication-testing-plan-execution/conservation_energy_24h.json`.
- Estimated wall-time + GPU budget: 30-60 min wall if reusing saved states; 1.0 GPU-hour budget.

### BENCHMARK-WRF-STOCK-IDEAL

- Priority: HIGH
- What it proves: The port can reproduce at least one stock WRF idealized benchmark beyond project-local fixtures.
- Inputs needed: Stock WRF ideal case selected during execution, recommended first choice `em_hill2d_x` if it aligns with implemented terrain handling, otherwise `em_quarter_ss` for thermal bubble/supercell-style dynamics; WRF namelist; CPU WRF output.
- How to run it:
  - Write `scripts/pubtest_run_wrf_stock_case.py --case <case> --lead-hours <n>`.
  - Write `scripts/pubtest_import_wrf_stock_case.py --case <case>`.
  - Run GPU analog with fixed timestep and compare fields.
- Pass/fail criteria: WRF input metadata imported without manual edits; all GPU fields finite; selected prognostic fields normalized RMSE `<= 0.10` at agreed lead; dry-mass budget passes; deviations file generated for unsupported WRF features.
- Proof object: `.agent/sprints/2026-05-27-publication-testing-plan-execution/benchmark_wrf_stock_ideal.json`.
- Estimated wall-time + GPU budget: 1-2 h wall; 2.0 GPU-hours budget.

### BENCHMARK-DYCORE-BAROCLINIC-WAVE

- Priority: MEDIUM
- What it proves: Large-scale balanced flow and synoptic wave growth are not only local Canary artifacts.
- Inputs needed: Published baroclinic-wave benchmark configuration or WRF analog; CPU/reference fields; output at day 5 and day 9 if feasible.
- How to run it:
  - Write `scripts/pubtest_run_baroclinic_wave.py --backend wrf|gpu`.
  - Compare geopotential, temperature, wind, surface pressure, and vorticity metrics.
- Pass/fail criteria: no nonfinite values; normalized RMSE for core fields `<= 0.10` at day 5 and `<= 0.20` at day 9, or inside a CPU perturbation envelope if available; mass budget residual passes.
- Proof object: `.agent/sprints/2026-05-27-publication-testing-plan-execution/benchmark_baroclinic_wave.json`.
- Estimated wall-time + GPU budget: 2-4 h wall; 4.0 GPU-hours budget.

### CANARY-MULTIDAY-SIDE-BY-SIDE

- Priority: HIGH
- What it proves: The current 2026-05-21 single-day result is not the whole story; skill is measured against CPU WRF and observations across a compact multi-regime corpus.
- Inputs needed: 7-10 retained Gen2/WRF d02 CPU runs with matching GPU input path; AEMET station observations; run IDs chosen to cover trade-wind, strong wind, stable nocturnal, calima/dry, warm daytime surface, and precipitation/cloud cases where available.
- How to run it:
  - Write `scripts/pubtest_select_canary_cases.py --min-cases 7 --output <proof>/canary_case_manifest.json`.
  - Run `scripts/m7_daily_pipeline.py` or successor for each case with proof-dir subfolders.
  - Run `scripts/m7_gpu_vs_cpu_skill_diff.py` for every case.
  - Aggregate with `scripts/pubtest_aggregate_skill.py`.
- Pass/fail criteria: at least 7 cases complete; station count and valid hours recorded per case; GPU within `+/-20%` of CPU RMSE for T2/U10/V10 on the aggregate OR the paper must keep a "skill gap" claim; no case with nonfinite output; per-hour error curves emitted.
- Proof object: `.agent/sprints/2026-05-27-publication-testing-plan-execution/canary_multiday_skill.json`.
- Estimated wall-time + GPU budget: 3-6 h wall depending on retained data; 4.0 GPU-hours budget for 10 GPU 24 h runs and rerun overhead.

### PRECIP-FSS-SAL-EVENT

- Priority: MEDIUM
- What it proves: High-resolution precipitation verification avoids pointwise double penalty and object-location ambiguity.
- Inputs needed: At least one precipitation event with CPU WRF, GPU output, and gridded or station-derived precipitation observations; threshold set such as 1 mm/h and 5 mm/3h; radii such as 3, 9, and 15 grid cells.
- How to run it:
  - Write `scripts/pubtest_precip_fss.py --case <case> --thresholds 1,5 --radii 3,9,15`.
  - Write `scripts/pubtest_precip_sal.py --case <case>`.
- Pass/fail criteria: FSS and SAL computed for CPU and GPU; GPU FSS no worse than CPU by more than `20%` at at least one event-relevant radius, or explicitly reported as a limitation; SAL components finite and interpreted.
- Proof object: `.agent/sprints/2026-05-27-publication-testing-plan-execution/precip_fss_sal_event.json`.
- Estimated wall-time + GPU budget: 1-2 h wall; 1.5 GPU-hours budget.

### REPRO-CROSS-HARDWARE

- Priority: MEDIUM if second GPU access exists; LOW otherwise
- What it proves: Results are not an RTX 5090 / driver-only accident.
- Inputs needed: Second NVIDIA GPU or cloud runner with compatible JAX/CUDA; same fixture bundle and one small ideal case; environment metadata.
- How to run it:
  - On local RTX 5090: `scripts/pubtest_repro_bundle.py --case warmbubble --output <proof>/local`.
  - On second GPU: same command with same commit and fixture checksums.
  - Compare with `scripts/pubtest_compare_repro_bundle.py`.
- Pass/fail criteria: same commit and fixture checksums; field deltas within deterministic tolerance envelope; if exact cross-hardware reproducibility is not expected, statistical/metric equivalence must be defined before running.
- Proof object: `.agent/sprints/2026-05-27-publication-testing-plan-execution/repro_cross_hardware.json`.
- Estimated wall-time + GPU budget: 1 h on each GPU; 2 GPU-hours total if hardware exists.

### PUBLIC-RELEASE-CHECKLIST

- Priority: HIGH
- What it proves: "Open source" is a usable release, not a code dump.
- Inputs needed: repository URL, release branch/commit, license decision, citation metadata, install docs, tutorial scripts, proof-object manifest, data policy, CI status.
- How to run it:
  - Write `.agent/sprints/2026-05-27-publication-testing-plan-execution/public_release_checklist.md`.
  - Write `scripts/pubtest_release_audit.py` or extend the existing publication audit to check required release files.
  - Run `taskset -c 0-3 python scripts/pubtest_release_audit.py --proof-dir <proof>`.
- Pass/fail criteria: `LICENSE`, `CITATION.cff`, `README`, `INSTALL`, `CONTRIBUTING`, environment lock/container instructions, tutorial entrypoint, proof manifest, and data-availability statement exist; all cited proof objects exist; no paper claim points to a missing artifact; CI smoke tests are documented.
- Proof object: `.agent/sprints/2026-05-27-publication-testing-plan-execution/public_release_checklist.md` and `public_release_audit.json`.
- Estimated wall-time + GPU budget: no GPU; 30-60 min wall.

## Documentation Checklist

Required before public submission:

- `README.md`: one-paragraph scope, current limitations, supported cases, quickstart.
- `INSTALL.md`: tested OS, Python, CUDA/JAX, GPU memory, CPU-only smoke path.
- `LICENSE`: explicit open-source license selected by the human owner.
- `CITATION.cff`: title, human author, version, DOI placeholder or Zenodo DOI.
- `CONTRIBUTING.md`: test/proof-object expectations, no binary fixture rule, issue template.
- `docs/validation.md`: four-tier validation strategy mapped to current proof files.
- `docs/data.md`: where large data live, how to fetch or regenerate, checksums, licensing caveats.
- `examples/`: one idealized case and one Canary replay notebook or script.
- `proof_manifest.md`: every paper claim mapped to file paths and commands.
- `known_limitations.md`: current AEMET skill regression, data-replay limitations, missing cross-hardware test, incomplete precipitation verification.

## Public Access Plan

The public release should be staged as a tagged source release plus external artifact archive:

1. Freeze a release commit after execution tests finish.
2. Generate `proof_manifest.md` with claim, proof object, command, and reviewer status.
3. Keep large binary fixtures and profiler reports out of git; archive them externally with checksums and scripts to regenerate.
4. Include a minimal fixture bundle small enough for CI.
5. Add CI jobs for schema validation, unit tests, fixture comparison, release audit, and paper citation/proof audit.
6. Publish exact environment: Python, JAX, jaxlib, CUDA driver/toolkit, GPU, OS, and key environment variables.
7. Make limitations visible in the README and abstract: current result is a fast prototype with a documented skill gap, unless the HIGH-priority skill tests recover.

## HIGH-Priority Cost Estimate

| Test | GPU-hours budget |
|---|---:|
| IDEALIZED-WARMBUBBLE | 0.5 |
| IDEALIZED-DENSITY-CURRENT | 1.0 |
| IDEALIZED-MOUNTAIN-WAVE | 1.5 |
| CONSERVATION-MASS-24H | 1.5 |
| CONSERVATION-ENERGY-24H | 1.0 |
| BENCHMARK-WRF-STOCK-IDEAL | 2.0 |
| CANARY-MULTIDAY-SIDE-BY-SIDE | 4.0 |
| PUBLIC-RELEASE-CHECKLIST | 0.0 |
| Rerun/debug reserve | 6.0 |
| Total HIGH budget | 17.5 |

This leaves about 6.5 GPU-hours of margin under a 24 GPU-hour overnight cap. CPU WRF reference generation may require additional CPU wall-time and should not be hidden inside the GPU-hour estimate.

## Execution Order

1. Freeze case manifest and thresholds before running.
2. Run public release audit in "expected fail" mode to expose missing docs early.
3. Run the three idealized cases.
4. Run mass and energy budgets, reusing saved idealized states where possible.
5. Run the stock WRF benchmark.
6. Run the Canary multi-day side-by-side set.
7. Aggregate HIGH-priority results into `publication_testing_summary.md`.
8. Only then decide whether the paper can move from prototype framing to stronger WRF-port framing.
