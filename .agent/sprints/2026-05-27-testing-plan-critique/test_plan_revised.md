# Executable Publication Testing Plan — Revised (sprint #2 output)

**Status**: PLAN_REVISED, ready for sprint #3 (execution).
**Supersedes**: `.agent/sprints/2026-05-27-publication-testing-plan-research/test_plan.md`.
**Authority**: revisions in this file take precedence; the original plan remains the source of community-acceptance rationale and gap analysis.
**Revisions applied from**: `plan_critique.md` (this sprint).

## Execution Target

Recommended execution sprint ID: `2026-05-27-publication-testing-plan-execution`.

All proof paths use:

`.agent/sprints/2026-05-27-publication-testing-plan-execution/`

The execution sprint should write code only after its own sprint contract freezes file ownership. This revised plan does not modify model code or run fresh measurements.

## Priority Policy

- `HIGH`: required before an arXiv submission can responsibly frame the project as an open-source WRF-compatible GPU port.
- `MEDIUM`: materially strengthens the paper but can be deferred if the paper is framed as a prototype with explicit limitations.
- `LOW`: future work or release hardening beyond the first paper.

24 GPU-hour cap retained. Updated HIGH budget at the bottom of this file.

## Revisions Summary (delta from original plan)

- **Removed** `BENCHMARK-WRF-STOCK-IDEAL` (redundant); replaced with a stock-WRF provenance requirement applied per idealized case.
- **Replaced** primary mountain-wave test with **Schaer 2002** analytic-oracle case; `em_hill2d_x` kept as secondary smoke.
- **Anchored** each idealized case to a published reference (Bryan & Fritsch 2002, Straka 1993, Schaer 2002).
- **Re-laddered** warm-bubble nRMSE threshold per lead time; chaotic case cannot hold ≤ 0.05 at 30 min.
- **Re-framed** symmetry threshold in physical, fp64-appropriate units.
- **Reframed** energy budget as Tier-4-style CPU envelope, not absolute drift.
- **Added** water-substance budget for microphysics-on runs (MEDIUM unless paper claims precip).
- **Tightened** Canary multi-day corpus from "7–10 cases" to "≥ 14 continuous days" using the 34-day inventory on disk.
- **Added** per-variable skill pass/fail (was aggregate-only).
- **Added** STABILITY-CFL-SWEEP, STABILITY-ACOUSTIC-SUBSTEP-SWEEP, DETERMINISM-REPEAT, SAVEPOINT-PARITY-DEEP, COMPILE-COLD-START-TIME, VRAM-FOOTPRINT-1KM-FRESH.
- **Demoted** REPRO-CROSS-HARDWARE to LOW + future-work declaration.
- **Expanded** PUBLIC-RELEASE-CHECKLIST: DOI minting, Software Heritage archival, reviewer 5-minute drive, signed proof-manifest checksums, pip-audit, coverage report, AI_USE.md.
- **Flagged** that WRF idealized cases require recompiling `ideal.exe + wrf.exe` per case; analytic-only path preferred for two of three idealized cases.

## Test Matrix

### IDEALIZED-WARMBUBBLE

- Priority: HIGH
- Reference: **Bryan & Fritsch 2002 MWR** dry warm-bubble (alt: Wicker & Skamarock 1998); ARW technical note also describes this case. Reference must be cited in `proof.reference`.
- What it proves: buoyant response, acoustic-step stability, vertical-velocity growth, symmetry, finite behaviour.
- Stock-WRF provenance requirement: WRF reference must come from `compile em_quarter_ss` (or equivalent stock ARW target), unmodified source, commit + compile command captured in `proof.wrf_provenance`.
- Inputs: matching JAX initial-state builder under `src/gpuwrf/fixtures/idealized/` (TO BE WRITTEN; no IC builder exists today); CPU WRF reference output at 5, 10, 20, 30 min; grid/timestep metadata; microphysics off unless explicitly testing moist.
- Steps:
  - Write `scripts/pubtest_prepare_wrf_ideal.py --case warmbubble --output <proof>/inputs/warmbubble`.
  - Compile stock WRF `em_quarter_ss` from `/home/enric/src/wrf_gpu/sources/...` (verify commit), then run `ideal.exe` and `wrf.exe` for 30 min; record `proof.wrf_provenance`.
  - Write `scripts/pubtest_run_wrf_reference.py --case warmbubble --minutes 30 --output <proof>/wrf/warmbubble`.
  - Write `scripts/pubtest_run_gpu_ideal.py --case warmbubble --minutes 30 --output <proof>/gpu/warmbubble`.
  - Write `scripts/pubtest_compare_ideal.py --case warmbubble --wrf <proof>/wrf/warmbubble --gpu <proof>/gpu/warmbubble --output <proof>/idealized_warmbubble.json`.
  - All under `taskset -c 0-3`.
- Pass/fail (revised):
  - All fields finite.
  - **Laddered nRMSE θ' and w vs CPU WRF**: ≤ 0.05 @ 5 min, ≤ 0.08 @ 10 min, ≤ 0.12 @ 20 min, ≤ 0.18 @ 30 min.
  - **w_max lead-time error** ≤ 10%.
  - **Closed-domain dry-mass relative drift** ≤ 1e-10 (fp64; diagnostic = `sum_xy(MUTS)`).
  - **Horizontal symmetry** for symmetric setup: `|w_max + w_min_reflected| / max(|w|) ≤ 1e-10` at fp64.
- Proof object: `idealized_warmbubble.json` + `idealized_warmbubble_summary.md`.
- Wall-time + GPU budget: 1.5–2.5 h wall (compile + run); **0.5 GPU-hour** + ~1 h CPU compile.

### IDEALIZED-DENSITY-CURRENT

- Priority: HIGH
- Reference: **Straka et al. 1993 IJNMF** density-current benchmark (Δx = 100 m, initial −15 K block, integration 900 s).
- What it proves: cold-pool propagation, sharp-gradient handling, near-surface dynamics, diffusion, mass/energy budget.
- WRF reference: **not required** — Straka 1993 publishes a converged reference numerical solution; compare against published values. Optional CPU WRF `em_grav2d_x` run as cross-check, but not required.
- Inputs: analytic cold-block IC (must be implemented under `src/gpuwrf/fixtures/idealized/density_current.py`); output at 300, 600, 900 s (and optional 1800, 3600 s).
- Steps:
  - Write `scripts/pubtest_prepare_density_current.py --output <proof>/inputs/density_current`.
  - Run GPU density current with fixed Δx = 100 m, dt set by CFL ≤ 0.5.
  - Compare with `scripts/pubtest_compare_density_current.py` against Straka 1993 published reference values (front position, front speed, min θ').
- Pass/fail (revised):
  - All fields finite.
  - **Front position within 1 horizontal grid cell** of Straka 1993 reference at t = 900 s.
  - **Front speed within 5%** of Straka 1993 reference (~33 m/s for Δx=100m).
  - **Min θ' within 0.5 K**.
  - **KE time-series nRMSE laddered**: ≤ 0.05 @ 5–10 min, ≤ 0.10 @ 15–30 min, ≤ 0.15 @ 60 min.
  - **Dry-mass closed-domain drift** ≤ 1e-10.
  - No negative pressure or water species.
- Proof object: `idealized_density_current.json`.
- Wall-time + GPU budget: 30–60 min wall; **1.0 GPU-hour**.

### IDEALIZED-MOUNTAIN-WAVE (Schaer primary)

- Priority: HIGH
- Reference (primary): **Schaer et al. 2002 MWR** sinusoidal-terrain non-hydrostatic test (envelope half-width 5 km, U = 10 m/s, N = 0.01 s⁻¹). Analytic linear-regime steady-state solution for w(x,z) provides the oracle.
- Reference (secondary): WRF `em_hill2d_x` bell-shaped hill, kept as a stock-WRF-binary smoke test only.
- What it proves: terrain-following coordinate, pressure-gradient force, vertical propagation, top/bottom boundary behaviour. The Schaer test is widely regarded as the most rigorous published probe of the σ/η coordinate.
- Steps:
  - Write `scripts/pubtest_prepare_mountain_wave.py --case schaer` (build Schaer IC + sinusoidal terrain). IC builder under `src/gpuwrf/fixtures/idealized/schaer.py`.
  - Write `scripts/pubtest_prepare_mountain_wave.py --case em_hill2d_x` for the smoke.
  - Run GPU for 5 h (steady-state regime) on both.
  - Optional: compile stock WRF `em_hill2d_x` for cross-check on the smoke case.
  - Compare with `scripts/pubtest_compare_mountain_wave.py` against analytic for Schaer, against CPU WRF for em_hill2d_x.
- Pass/fail (revised, Schaer):
  - All fields finite.
  - **Peak vertical-velocity amplitude** `|w_peak|` within 10% of analytic linear-regime solution at t = 5 h (steady state).
  - **Dominant wave phase shift** ≤ 1 grid cell horizontal and vertical.
  - **Pressure perturbation nRMSE vs analytic** ≤ 0.10.
  - **Dry-mass open-boundary flux-corrected residual** ≤ 1e-6.
  - **No vertical-coordinate instability** (no grid-point oscillations exceeding 5× background w).
- Pass/fail (em_hill2d_x smoke):
  - All fields finite; w cross-section qualitatively matches CPU WRF; no instability.
- Proof object: `idealized_mountain_wave.json` covering both cases.
- Wall-time + GPU budget: 1.5–2 h wall; **1.5 GPU-hours**.

### CONSERVATION-MASS-24H

- Priority: HIGH
- What it proves: implementation does not silently create/destroy dry air over long integrations except through explicit boundary fluxes.
- Inputs: one closed-domain warm-bubble extended to 24 h; one Canary 24 h run; boundary mass-flux diagnostics for the Canary case; fp64 mass path.
- Steps:
  - **Reuse** `scripts/diagnostic_conservation_tracker.py` (mass + KE + dry-static-energy totals are already implemented there) — wrap, do not rewrite.
  - Write `scripts/pubtest_mass_budget.py --case warmbubble --hours 24 --closed-domain --output <proof>/conservation_mass_ideal.json` (thin wrapper).
  - Write `scripts/pubtest_mass_budget.py --case canary-<run_id> --hours 24 --boundary-flux-corrected --output <proof>/conservation_mass_canary.json`.
- Pass/fail (revised):
  - **Closed-domain relative dry-mass drift** ≤ 1e-10 (diagnostic = `sum_xy(MUTS)` vs initial).
  - **Canary boundary-flux-corrected residual** ≤ 1e-5.
  - No nonfinite mass fields.
  - Residual time-series monotonic drift must be explained if present.
  - **Optional moisture/water budget (MEDIUM)**: closed-domain water-substance total mass closure ≤ 1e-8 with sedimentation/precipitation accounted; Canary ≤ 1e-4 open-domain. Only required if paper claims precipitation; otherwise mark deferred.
- Proof object: `conservation_mass_24h.json`.
- Wall-time + GPU budget: 45–90 min wall reusing ideal + Canary runs; **1.5 GPU-hours**.

### CONSERVATION-ENERGY-24H

- Priority: HIGH
- What it proves: dynamics and physics do not produce unbounded or unexplained energy drift, **bounded by CPU WRF behaviour** on the same configuration.
- Inputs: closed-domain dry idealized case (reuse warm bubble); optional physics-off Canary run; CPU WRF reference run on the same warm-bubble closed-domain configuration.
- Steps:
  - **Reuse** `scripts/diagnostic_conservation_tracker.py` (KE + dry-static-energy totals).
  - Write `scripts/pubtest_energy_budget.py --case warmbubble --hours 24 --closed-domain --output <proof>/conservation_energy_ideal.json`.
  - Run CPU WRF on identical configuration; record CPU drift as the envelope.
  - If Canary included, run `--flux-corrected` and emit assumptions block.
- Pass/fail (revised — Tier-4 style):
  - **GPU total-energy drift within ±20% of CPU WRF drift** on the same closed-domain warm-bubble configuration over 24 h.
  - **Per-component split reported**: KE, internal (cv·T), potential (g·z); each component drift within CPU envelope ±20%; no single component diverges from CPU by > 0.5%.
  - **No unexplained step jump** > 0.05%.
  - Assumptions and CPU envelope numerical values recorded in JSON.
- Proof object: `conservation_energy_24h.json`.
- Wall-time + GPU budget: 1–1.5 h wall; **1.0 GPU-hour** (GPU side) + ~1 h CPU WRF reference.

### STABILITY-CFL-SWEEP

- Priority: HIGH
- What it proves: stable behaviour at nominal CFL plus margin; required by community-acceptance §4 "stability margins."
- Steps:
  - Reuse warm-bubble case; run at dt = 0.5×, 1.0×, 1.25× nominal.
  - Optionally also Canary 1-h smoke at the three dts.
  - Write `scripts/pubtest_stability_cfl_sweep.py`.
- Pass/fail:
  - 0.5× and 1.0× runs complete with finite fields, mass drift ≤ 1e-10.
  - 1.25× either completes within tolerance, or fails *deterministically* (NaN at a specific step) — not silently producing garbage. The largest stable dt must be reported.
- Proof object: `stability_cfl_sweep.json`.
- Wall-time + GPU budget: 30 min wall (three short runs); **0.3 GPU-hour**.

### STABILITY-ACOUSTIC-SUBSTEP-SWEEP

- Priority: HIGH
- What it proves: behaviour is not load-bearing on a specific `time_step_sound` value.
- Steps:
  - Reuse density-current case; vary substep count `n` ∈ {4, 6, 8}.
  - Write `scripts/pubtest_acoustic_substep_sweep.py`.
- Pass/fail:
  - All three runs finite; front position varies by ≤ 1 cell across the three settings; KE time-series nRMSE pairwise ≤ 0.05.
- Proof object: `stability_acoustic_substep.json`.
- Wall-time + GPU budget: 30 min wall; **0.3 GPU-hour**.

### DETERMINISM-REPEAT

- Priority: HIGH
- What it proves: full-pipeline determinism (currently only restart determinism is asserted by `restart_continuity.json`).
- Steps:
  - Run the Canary 1-h pipeline three times on identical inputs, identical commit, identical environment.
  - Write `scripts/pubtest_determinism_repeat.py` that wraps three calls.
- Pass/fail:
  - Max delta across three runs = **0.0 bitwise** for every State field at the final step.
  - If non-bitwise: explain why (e.g. cuDNN nondeterministic kernels) and either gate non-determinism off or document the bound.
- Proof object: `determinism_repeat.json`.
- Wall-time + GPU budget: ~30 min wall; **0.3 GPU-hour**.

### SAVEPOINT-PARITY-DEEP

- Priority: HIGH
- What it proves: bitwise savepoint parity does not silently degrade with depth.
- Steps:
  - Reuse B6 savepoint harness; extend to compare GPU vs WRF at step 100, 1000, 10000.
  - Write `scripts/pubtest_savepoint_parity_deep.py`.
- Pass/fail:
  - Max delta = 0.0 bitwise at step 100.
  - Max delta documented and bounded at step 1000 and 10000 (may not be 0.0 due to accumulated round-off; bound must be < 1e-12 relative for all fields).
- Proof object: `savepoint_parity_deep.json`.
- Wall-time + GPU budget: 1–2 h wall; **1.0 GPU-hour**.

### CANARY-MULTIDAY-SIDE-BY-SIDE

- Priority: HIGH
- What it proves: skill is measured across regimes, not a single day.
- Inputs: **≥ 14 continuous days** from `/mnt/data/canairy_meteo/runs/wrf_l3/`. Inventory shows 34 days available (20260428 → 20260525), so this is bankable; up to 21 days if budget allows.
- Steps:
  - Write `scripts/pubtest_select_canary_cases.py --window-days 14 --output <proof>/canary_case_manifest.json` (selector over the existing inventory).
  - **Reuse** `scripts/m7_daily_pipeline.py` per case (subfolders per run_id).
  - **Reuse** `scripts/m7_gpu_vs_cpu_skill_diff.py` per case.
  - Write `scripts/pubtest_aggregate_skill.py` (cross-case aggregator).
  - Write `scripts/pubtest_first_error_growth.py` (per-hour curves + first-divergence metric).
- Pass/fail (revised, per-variable):
  - At least 14 cases complete; station count and valid hours recorded per case.
  - **Per-variable pass/fail** (not aggregate): for each of T2, U10, V10, GPU within ±20% of CPU RMSE.
  - Per-variable failures explicitly enumerated in the paper.
  - No case with nonfinite output.
  - Per-hour error curves emitted; first-divergence hour reported per variable.
- Proof object: `canary_multiday_skill.json`.
- Wall-time + GPU budget: 4–6 h wall (14 × ~12 min GPU + overhead); **4.0 GPU-hours**.

### COMPILE-COLD-START-TIME

- Priority: MEDIUM
- What it proves: the "compile-once, scan-the-loop" claim is quantified.
- Steps:
  - Write `scripts/pubtest_compile_cold_start.py`: spawn fresh Python, time first compile of operational forecast loop, time subsequent run.
- Pass/fail:
  - Cold compile time measured; subsequent run time measured; ratio reported. No threshold — diagnostic only.
- Proof object: `compile_cold_start.json`.
- Wall-time + GPU budget: 20 min wall; **0.1 GPU-hour**.

### VRAM-FOOTPRINT-1KM-FRESH

- Priority: MEDIUM
- What it proves: the 7.28 GB 1 km claim is current.
- Steps:
  - Write `scripts/pubtest_vram_footprint.py` covering 3 km d02 and 1 km full-domain configurations on current commit.
- Pass/fail:
  - Peak VRAM ≤ 32 GB for the 1 km case; baseline number reported.
- Proof object: `vram_footprint.json`.
- Wall-time + GPU budget: 20 min wall; **0.2 GPU-hour**.

### BENCHMARK-DYCORE-BAROCLINIC-WAVE

- Priority: MEDIUM (deferred to v1 paper)
- Stays in plan as future work. Execution sprint may skip.

### PRECIP-FSS-SAL-EVENT

- Priority: MEDIUM
- Implementation already available at `src/gpuwrf/validation/forecast_vs_obs.py:467` (`compute_fractions_skill_score`). The execution sprint should **wrap and extend** this, not rewrite. SAL implementation absent — add if scope allows; otherwise mark deferred.
- Thresholds: as in original plan.
- Wall-time + GPU budget: 1–2 h wall; **1.5 GPU-hours**.

### REPRO-CROSS-HARDWARE

- Priority: **LOW** (was MEDIUM)
- Hardware not available on user's workstation. **Mark in paper as future work**; do not block.

### PUBLIC-RELEASE-CHECKLIST

- Priority: HIGH
- Required files (original plan retained):
  - `LICENSE` (human-owner decision required — execution sprint must flag, not pick).
  - `CITATION.cff` with Zenodo DOI placeholder.
  - `README.md` with quickstart.
  - `INSTALL.md` with tested OS/Python/CUDA/JAX/GPU memory + CPU-only smoke.
  - `CONTRIBUTING.md` with proof-object expectations, no-binary-fixture rule, issue template, triage SLA.
  - `docs/validation.md` mapping the four-tier strategy to current proof files.
  - `docs/data.md` with per-dataset license + retrieval path (AEMET, Gen2 wrf_l3, CAMS, MAIAC enumerated separately).
  - `examples/` with one idealized + one Canary case.
  - `proof_manifest.md` mapping every paper claim to file paths + commands.
  - `known_limitations.md` with the current AEMET skill regression, data-replay limits, missing cross-hardware test, incomplete precipitation verification.
- **Added requirements** (from AC6 critique):
  - **DOI minting plan**: Zenodo–GitHub integration; release tag → automatic deposit → DOI in CITATION.cff.
  - **Software Heritage SWHID** for the release commit.
  - **Reviewer 5-minute test drive**: `scripts/release_smoke.sh` that performs install + one idealized smoke + PASS on a laptop in < 5 min.
  - **Signed proof-manifest checksums**: `sha256sums.txt.asc` or git-signed release tag.
  - **`pip-audit` / `safety` CI job**.
  - **Coverage report**: `coverage xml` produced in CI; number visible.
  - **`AI_USE.md`** documenting the AI-agent build methodology (per strategic-framing memo §13).
  - **Hardware reproducibility statement**: all results from RTX 5090 + pinned JAX/CUDA; bit-different results expected elsewhere.
- Steps:
  - Write `.agent/sprints/2026-05-27-publication-testing-plan-execution/public_release_checklist.md`.
  - Write `scripts/pubtest_release_audit.py` that verifies each required file exists and that every paper claim has a matching proof object.
  - Run `taskset -c 0-3 python scripts/pubtest_release_audit.py --proof-dir <proof>`.
- Pass/fail:
  - All required files exist; all cited proof objects resolve; reviewer 5-minute drive completes; no paper claim points to a missing artifact.
- Proof object: `public_release_checklist.md` + `public_release_audit.json`.
- Wall-time + GPU budget: 30–60 min wall; **0 GPU-hour**.

## Documentation Checklist

(Unchanged from original plan; one addition.)

- All items in original plan plus:
- `AI_USE.md`: AI-agent build methodology disclosure, mirroring §13 of the paper.

## Public Access Plan

(Unchanged from original plan in shape; tightened items now in PUBLIC-RELEASE-CHECKLIST.)

## HIGH-Priority Cost Estimate (revised)

| Test | GPU-hours budget |
|---|---:|
| IDEALIZED-WARMBUBBLE | 0.5 |
| IDEALIZED-DENSITY-CURRENT | 1.0 |
| IDEALIZED-MOUNTAIN-WAVE (Schaer + em_hill2d_x smoke) | 1.5 |
| CONSERVATION-MASS-24H | 1.5 |
| CONSERVATION-ENERGY-24H | 1.0 |
| STABILITY-CFL-SWEEP | 0.3 |
| STABILITY-ACOUSTIC-SUBSTEP-SWEEP | 0.3 |
| DETERMINISM-REPEAT | 0.3 |
| SAVEPOINT-PARITY-DEEP | 1.0 |
| CANARY-MULTIDAY-SIDE-BY-SIDE (≥ 14 days) | 4.0 |
| PUBLIC-RELEASE-CHECKLIST | 0.0 |
| Rerun/debug reserve | 6.0 |
| **Total HIGH budget** | **17.4** |

Stays under the 24 GPU-hour overnight cap with ~6.6 GPU-hours margin.

Additional CPU wall-time (not GPU-hour budgeted):

- 3 × ~1 h for stock WRF idealized compiles + reference runs.
- 1 × ~1 h for CPU WRF closed-domain warm-bubble energy-budget reference.
- 14 × ~30 min for Canary CPU WRF reference (already on disk — no recompile needed).

## Execution Order

1. Freeze case manifest and thresholds before running.
2. Run public release audit in "expected fail" mode to expose missing docs early.
3. Stock WRF idealized compiles in background (em_quarter_ss; em_grav2d_x optional cross-check; em_hill2d_x).
4. Run STABILITY-CFL-SWEEP first to confirm the dycore is well-conditioned at the planned dt.
5. Run the three idealized cases (warm bubble, Schaer mountain, density current).
6. Run STABILITY-ACOUSTIC-SUBSTEP-SWEEP on the density current.
7. Run DETERMINISM-REPEAT on a 1-h Canary smoke.
8. Run SAVEPOINT-PARITY-DEEP.
9. Run mass and energy budgets, reusing saved idealized states where possible.
10. Run CANARY-MULTIDAY-SIDE-BY-SIDE (≥ 14 days from disk).
11. Run COMPILE-COLD-START-TIME and VRAM-FOOTPRINT-1KM-FRESH (cheap; can be slotted between other runs).
12. Run PRECIP-FSS-SAL-EVENT (MEDIUM; if time).
13. Aggregate HIGH-priority results into `publication_testing_summary.md`.
14. Run final PUBLIC-RELEASE-CHECKLIST audit; iterate on missing files until clean.
15. Only then decide whether the paper can move from prototype framing toward stronger WRF-port framing.

## Cross-cutting Execution Notes

- **CPU pinning**: every script must use `taskset -c 0-3`. Per user memory, WRF gets cores 4–31.
- **No host/device transfer in timestep loop**: existing ADR-027 invariant; reaffirmed for every test that touches the operational forecast loop.
- **Reuse before rewrite**: `m7_daily_pipeline.py`, `m7_gpu_vs_cpu_skill_diff.py`, `diagnostic_conservation_tracker.py`, `compute_fractions_skill_score` in `forecast_vs_obs.py` are all already implemented. Pubtest wrappers should be thin.
- **WRF idealized recompile**: budget ~1 h compile per case. The existing `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` is a real-data build and **cannot** run idealized namelists.
- **LICENSE choice**: human-owner decision per the strategic-framing memo. Execution sprint must flag this for the user before running the release audit.
- **Honest decision rule** if any HIGH test fails: paper limitations section gets a new bullet; the headline claim is not changed but the supporting evidence is qualified.
