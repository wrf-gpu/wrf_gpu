# V0.14.0 Validation Plan - 16h Campaign

Date: 2026-06-08
Owner: GPT-5.5 xhigh validation architect
Branch: `worker/gpt/v013-valplan`

## Positioning

2026-06-08 23:11 WEST manager update: this plan is now **grid-parity-first**.
The powered TOST scorer remains a final gate, but it is not the next GPU
campaign. The next campaign must first explain and reduce the CPU-WRF vs GPU-WRF
cell-level divergence across all written fields. Station TOST cannot hide a broad
spatial field mismatch.

This is the deeper validation campaign after the v0.13 3h gate. The model is a
fast, GPU-native, GPU-scalable WRF-compatible implementation. The campaign must
not overclaim bit-truth or perfect efficiency.

Primary objective: prove that implemented couplings run stably across longer
horizons, more schemes, nesting, GWD, feedback, restart/reproducibility, and
multi-region data without NaNs, OOMs, or crashes.

Secondary objective: collect strong, honest CPU-WRF and AEMET equivalence
evidence. The direct grid-cell envelope is the first gate; powered TOST is the
final gate-keeper-facing artifact after the grid-field gap is no longer radical.

## Corpus Reality

Current pairable truth found on disk:

- Canary L2 9/3 km:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output` contains the current
  powered-TOST `n=15` CPU-WRF truth corpus with 72h d01/d02 wrfout.
- Canary L2 and L3 operational cases:
  `/mnt/data/canairy_meteo/runs/wrf_l2` and `/mnt/data/canairy_meteo/runs/wrf_l3`
  contain retained inputs and some retained CPU-WRF outputs. L3 has full 24h
  examples such as `20260509_18z_l3_24h_20260511T190519Z` and
  `20260521_18z_l3_24h_20260522T133443Z`.
- AEMET:
  `/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations`.
- Switzerland:
  `/mnt/data/wrf_gpu_switzerland_big/run_cpu` and
  `/mnt/data/wrf_gpu_switzerland_128/run_cpu` contain 24h CPU truth. This gives
  a non-Canary winter/Alps region. It does not give Canary seasonal coverage.
- Pristine WRF:
  `/home/enric/src/wrf_pristine/WRF` exists with CPU-WRF binaries for oracle or
  backfill work.

Limitations:

- Current Canary truth is effectively MAM 2026. A true Canary multi-season claim
  is not available from the retained corpus.
- Powered `n=15` is available now. ADR-029 indicates roughly `n=27` is needed
  for a 10 percent MDE at the current margins, so `n=30` requires additional
  CPU-WRF truth/backfill before this campaign can honestly claim that sample
  size.
- 9/3/1 km plus GWD plus 2-way feedback for 24h is known 32GB-VRAM-marginal and
  has OOMed around hour 14. The 16h plan uses a bounded 12h 2-way slice and a
  24h one-way 1 km run.

## Common Setup

```bash
export OUT=/mnt/data/wrf_gpu_validation/v0140_campaign_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
```

GPU jobs must use `scripts/run_gpu_lowprio.sh` (repo-versioned lock wrapper);
do not depend on `/tmp/wrf_gpu_run_lowprio.sh`. Long detached campaigns should
follow `docs/GPU_RUNBOOK.md`.

GPU jobs are serial. CPU jobs may run in parallel on cores 0-23. CPU-only
commands must force `JAX_PLATFORMS=cpu`.

## Tests

### B1 - 72h Canary L2 9/3 km, GWD, 2-Way Feedback

Type: PRIMARY RUNS-confidence, SECONDARY long-run equivalence support

Resource: GPU, one serial job

Estimate: 1h45m

Command:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  PYTHONPATH=src \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  GPUWRF_GWD_NESTED=1 \
  python -m gpuwrf.cli run \
    --input-dir /mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z \
    --output-dir "$OUT/b1_canary_l2_72h_feedback_gwd" \
    --max-dom 2 \
    --hours 72 \
    --feedback \
    --proof-dir "$OUT/b1_canary_l2_72h_feedback_gwd/proofs" \
    --score
```

Pass criterion:

- d01 and d02 each emit 72 hourly wrfout frames.
- Core surface, precipitation, moisture, thermodynamic, and 3D wind fields are
  finite for all frames.
- No OOM, NaN, or crash occurs.
- Runtime and memory proof metadata are retained.

What it proves:

- The heaviest fitting nested GWD plus 2-way configuration can survive a
  multi-day operational horizon.

### B2 - 24h Canary L3 9/3/1 km, GWD, One-Way Nesting

Type: PRIMARY RUNS-confidence

Resource: GPU, one serial job

Estimate: 2h00m

Command:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  PYTHONPATH=src \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  GPUWRF_GWD_NESTED=1 \
  python -m gpuwrf.cli run \
    --input-dir /mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z \
    --output-dir "$OUT/b2_canary_l3_24h_oneway_gwd" \
    --max-dom 3 \
    --hours 24 \
    --proof-dir "$OUT/b2_canary_l3_24h_oneway_gwd/proofs" \
    --score
```

Pass criterion:

- d01, d02, and d03 each emit 24 hourly wrfout frames.
- All core fields are finite.
- GWD diagnostics and nested boundary updates are present.

What it proves:

- Full 24h 1 km Canary production geometry runs with GWD when feedback is not
  enabled.

### B3 - 12h Canary L3 9/3/1 km, GWD, 2-Way Feedback Slice

Type: PRIMARY RUNS-confidence

Resource: GPU, one serial job

Estimate: 1h15m

Command:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  PYTHONPATH=src \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  GPUWRF_GWD_NESTED=1 \
  python -m gpuwrf.cli run \
    --input-dir /mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z \
    --output-dir "$OUT/b3_canary_l3_12h_feedback_gwd" \
    --max-dom 3 \
    --hours 12 \
    --feedback \
    --proof-dir "$OUT/b3_canary_l3_12h_feedback_gwd/proofs" \
    --score
```

Pass criterion:

- d01, d02, and d03 each emit 12 hourly wrfout frames.
- All fields are finite.
- No OOM occurs before hour 12.

What it proves:

- The heaviest known coupling combination is stable through the largest horizon
  that is expected to fit reliably on the current 32GB workstation.

### B4 - Grid-Cell Envelope First, Then Powered TOST n=15

Type: PRIMARY CPU-WRF field-parity gate, SECONDARY gate-keeper equivalence

Resource: GPU serial campaign plus CPU scoring

Estimate: 6h00m

Current status:

- TOST is paused after 3 durable cases.
- `proofs/v014/v10_grid_diagnostics.json` shows V10 grid RMSE above 1.5 m/s in
  3/3 cases. This is too large to continue treating station TOST as the next
  decision point.
- 2026-06-09 live-nest debug update: `proofs/v014/live_nest_base_hook.json`
  classifies the current root-cause surface as `NATIVE_PORT_PLAN_READY`. The
  d02 base-state mismatch is traced to missing WRF live-nest initialization:
  parent interpolation through `med_interp_domain` /
  `nest_interpdown_interp.inc`, `blend_terrain`, and `start_domain_em` base
  recomputation. Native `wrfinput_d02` differs from CPU-WRF h0 by about
  `1047` Pa `PB` and `1050` Pa `MUB` on the target patch, while WRF base
  formulas on CPU-WRF h0 terrain reproduce `PB/MUB/PHB` within `0.1`. The next
  validation-enabling source sprint is therefore a GPU-native initialization
  port, not a CPU-WRF h0 production shortcut.
- 2026-06-09 Opus critic update:
  `.agent/reviews/2026-06-09-v014-debug-method-critic.md` accepts the live-nest
  base port as a real correctness fix but rejects treating it as the
  V10/grid-field symptom closer without a direct falsifier. The plan now
  requires separation of two claims: (1) base-state agreement improved and
  (2) V10/grid-field divergence materially improved. A source port may not be
  used to resume TOST or claim grid parity unless an init-override or direct
  grid-field proof closes the symptom.
- 2026-06-09 direct proof update: the direct falsifier has now run and did
  **not** close the symptom. `proofs/v014/grid_after_live_nest_base.json`
  verdict is `GRID_SYMPTOM_NOT_CLOSED` after one h12 GPU run with
  `L2_D02_GREEN`. h1-h12 `V10` RMSE remains `2.55039100124724` m/s, worst h11
  RMSE `4.277008742661733`; `PSFC`, `P`, `MU`, and `PH` still have large
  residuals. `proofs/v014/same_state_momentum_mass.json` independently shows
  the selected h10 `U` mismatch already exists at
  `post_after_all_rk_steps_pre_halo`. Therefore B4 remains blocked before
  powered TOST; next validation-enabling work is dynamic same-state
  localization/fix, reviewed by the targeted Opus critic sprint.
- 2026-06-09 Opus critic closeout: the next validation-enabling proof boundary
  is **not** final-RK output localization from the old JAX carry. Opus found the
  pre-RK input is already divergent, so the active next sprint is strict
  same-input single-RK-step parity using WRF's own pre-RK input savepoint. Only
  if that proof shows a same-input mismatch should a dynamics source edit be
  considered.
- 2026-06-09 same-input closeout: the strict comparison is currently blocked by
  missing instrumentation, not by a proven model source defect.
  `proofs/v014/same_input_single_rk_parity.json` verdict is
  `SAME_INPUT_TENDENCY_INPUT_BLOCKED_PRE_RK_FULL_NATIVE_STATE_RK_TENDF_AND_HISTORY_SOURCE_FIELDS`.
  The next validation-enabling sprint is the full pre-RK native-state/tendency
  WRF hook plus proof-only JAX loader. Powered TOST and Switzerland validation
  remain paused until this boundary either runs cleanly or localizes a fix.
- 2026-06-09 full pre-RK hook closeout:
  `proofs/v014/same_input_single_rk_parity_full.json` verdict is
  `FULL_PRE_RK_JAX_LOADER_BLOCKED_RK_FIXED_SOURCE_BOUNDARY`. The CPU-WRF hook
  now provides full native pre-RK state at `d02` step `6000`, but current-step
  `*_tendf`/`*_save` source leaves are not available at the step-entry boundary.
  B4 remains blocked. The next validation-enabling proof is a WRF source/save
  boundary after those leaves exist and before any dynamics state mutation.
- 2026-06-09 source/save-boundary closeout:
  `proofs/v014/same_input_single_rk_parity_sources.json` verdict is
  `SOURCE_SAVE_BOUNDARY_READY_NO_JAX_WRAPPER_FULL_DOMAIN_PATCH_AND_SCALAR_OLD_LIMITER`.
  WRF now emits the current-step dry source/save leaves at a consistent
  pre-mutation boundary, and native dry state preservation versus the full
  pre-RK savepoint is exact on overlap. B4 remains blocked because no strict JAX
  comparison has executed. The next validation-enabling work is a proof-only
  full-domain wrapper/truth-surface sprint, including same-boundary
  carry/boundary leaves, full-domain/full-vertical post-RK truth, and a
  consistent old-field strategy.
- 2026-06-09 full-domain wrapper closeout:
  `proofs/v014/same_input_single_rk_parity_wrapped.json` verdict is
  `FULL_DOMAIN_WRAPPER_BLOCKED_TRUTH_SURFACE_PATCH_ONLY_AND_CARRY_LEAVES`.
  Existing h10/step-6000 surfaces are patch-only and do not satisfy the full
  same-input wrapper contract. B4 remains blocked; per management review, do
  not continue the step-6000 wrapper ladder.
- 2026-06-09 early-step discriminator closeout:
  `proofs/v014/early_step_discriminator.json` verdict is
  `EARLY_STEP_DISCRIMINATOR_BLOCKED_CPU_REALCASE_LOADER_GPU_ONLY_NO_CANDIDATE_WRF_PREHALO_TRUTH_NO_SAME_INPUT_CARRY_CONTRACT`.
  It covers steps `1`, `60`, `600`, `3000`, and `5999` in one fail-closed proof.
  The current blocker is comparison infrastructure: CPU-only real-case replay
  loading is GPU-only at `State.zeros`, no candidate-step WRF post-RK/pre-halo
  full-field truth surface exists, no WRF-controlled same-input carry sequence
  exists, and the field/staggering schema is not frozen. B4 remains blocked
  before powered TOST or Switzerland. The next validation-enabling sprint must
  build that comparison contract/tooling and rerun the discriminator before any
  production dynamics source edit.
- 2026-06-09 same-input contract-builder closeout:
  `proofs/v014/same_input_contract_builder.json` verdict is
  `SAME_INPUT_CONTRACT_BLOCKED_NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH_STEP_1`.
  The initial d02 CPU/JAX same-input loader and 16-field schema are now ready.
  B4 remains blocked because the full-domain CPU-WRF d02 step-1
  `post_after_all_rk_steps_pre_halo` truth surface is missing. The next
  validation-enabling task is a disposable CPU-WRF hook that emits the accepted
  npz truth contract, followed by a rerun of the contract builder for the first
  strict WRF-vs-JAX residual table.
- 2026-06-09 management-review correction:
  `.agent/reviews/2026-06-09-v014-management-review-01.md` records
  `NO_GOAL_CHANGE` but criticizes the step-6000 same-input path as a blocked
  micro-sprint ladder at the hardest instance. B4 remains blocked. After the
  already-running full-domain wrapper sprint closes, the next validation-enabling
  discriminator must be a consolidated early-step/drift-onset sprint, starting
  from shared `wrfinput` where instrumentation is cheap, with at least one strict
  same-input comparison executed or all remaining blockers named in one pass.
- 2026-06-09 full-domain wrapper closeout:
  `proofs/v014/same_input_single_rk_parity_wrapped.json` verdict is
  `FULL_DOMAIN_WRAPPER_BLOCKED_TRUTH_SURFACE_PATCH_ONLY_AND_CARRY_LEAVES`.
  Existing step-6000 source/save and post-RK surfaces are patch-only and lack the
  same-boundary full wrapper carry/boundary leaves. B4 remains blocked and the
  next validation-enabling work is now the staged early-step same-input
  discriminator, not more step-6000 wrapper instrumentation.
- 2026-06-09 source sprint update:
  `proofs/v014/live_nest_base_source_fix.json` classifies the landed candidate
  as `LIVE_NEST_BASE_SOURCE_PARTIAL_NO_GRID_SYMPTOM_PROOF`. Native live-nest
  base fields now match CPU-WRF h0 as validation oracle to formula-level
  residuals on the target patch (PB `0.0489` Pa, MUB `0.0444` Pa, PHB
  `0.0933`, HGT `2.42e-05` m), without a CPU-WRF h0 production dependency and
  without timestep-loop host/device transfer. This is accepted only as a
  base-state source fix. It does not close or materially reduce V10/grid-field
  divergence until an init-override/direct grid-field proof shows that symptom
  improvement. TOST remains paused.
- Before resuming the n=15 campaign, the project must either fix the responsible
  operators or record an operator-specific root cause and accepted residual.

Command:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  PYTHONPATH=src \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  GPUWRF_AEMET_ROOT=/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations \
  python proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py \
    --resume
```

B4 has two distinct proof pillars. They answer different questions and both must
be reported before any v0.14 equivalence claim. Their order is now binding:

- **B4b direct grid-cell envelope comes first:** compare GPU wrfout against
  CPU-WRF wrfout directly over every common grid cell, lead hour, and written
  field. This is not a station-skill test; it is the field-by-field numerical
  divergence envelope reviewers expect when asking whether "millions of grid
  values" remain close enough.
- **B4a station-skill TOST comes after B4b is acceptable:** compare CPU-WRF skill
  vs AEMET and GPU skill vs AEMET on complete station/time pairs (`tost_pairs`).
  This is the ADR-029 statistical TOST artifact.

The immediate B4b implementation target is broader than the current T2/U10/V10
`cell_level` summary: compare every field written by `wrfout_writer.py` for which
the CPU truth has a matching variable, including surface fields, precipitation,
pressure diagnostics, 3D winds, thermodynamic state, and moisture fields.

Pass criterion:

- The grid-cell envelope report exists before any resumed TOST run and includes
  all comparable written fields, not only T2/U10/V10.
- All 15 available manifest cases are either scored or have documented,
  reproducible exclusions once TOST is resumed.
- For T2, U10, and V10, the aggregate report includes complete-pair counts,
  case-level RMSE deltas, confidence intervals, TOST p-values, and ADR-029
  margins:
  - T2 margin: 0.2148692978020805 K
  - U10 margin: 0.23064713972582307 m/s
  - V10 margin: 0.2752320537920854 m/s
- AEMET pair counts are nonzero for each scored case where observations exist.
- Any `NOT_EQUIVALENT` result is kept as a result, not treated as a harness
  failure.
- `cell_level_stats.json` or its v0.14 successor is emitted and contains pooled
  grid-cell RMSE, bias, MAE, p95, p99, max, Pearson r, fraction-within-tolerance,
  per-lead blocks, and spatial/terrain splits for all comparable fields.
- The direct grid-cell envelope is reported separately from station TOST. The
  current script's `cell_tol` values (`T2=2.0 K`, `U10/V10=2.5 m/s`) are
  diagnostic tolerances, not a historical pass claim. For v0.14 closure, promote
  them into a predeclared envelope or replace them with an ADR-bound envelope
  before the final run; do not tune them after seeing the campaign.
- No v0.14 "CPU-WRF equivalent" claim is allowed unless both B4a and B4b pass,
  or unless the release explicitly says which pillar failed and why.

What it proves:

- This is the main gate-keeper-facing equivalence artifact currently available
  from retained truth.
- It also provides repeated 9/3 km nested GPU stability evidence across cases.
- It separates station skill from direct grid-field agreement so a station
  TOST result cannot hide a broad spatial divergence, and a good grid envelope
  cannot be mistaken for an observation-skill proof.

n=30 path:

- Current disk truth does not support n=30. To run n=30, first backfill and
  retain 15 additional CPU-WRF L2 truth cases with pristine WRF, then generate a
  new manifest with complete d01/d02 72h wrfout and AEMET-pairable dates.
- A full n=30 GPU campaign is expected to cost roughly 11-12h and should replace
  B1, B2, B3, and B7 in a 16h window, or run as a second overnight campaign.

### B5 - Forecast-Skill Closure A/B: rad_rk_tendf And Wind-Error Levers

Type: SECONDARY forecast-skill measurement, PRIMARY variant stability

Resource: GPU serial plus CPU scoring

Estimate: 2h00m

Precondition:

- Land a small measurement harness at `proofs/v014/skill_ab_runner.py`. The
  harness must only toggle existing operational namelist/options and reuse the
  existing CPU-WRF/AEMET scorer. It must not introduce new physics.

CPU precheck:

```bash
taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  JAX_ENABLE_X64=true \
  PYTHONPATH=src \
  python proofs/v013/skill_closure.py
```

GPU A/B command:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  PYTHONPATH=src \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  GPUWRF_AEMET_ROOT=/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations \
  python proofs/v014/skill_ab_runner.py \
    --case-dir /mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z \
    --truth-dir /mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z \
    --domain d02 \
    --hours 24 \
    --variants baseline,rad_rk_tendf1,moist_adv_opt2,rad_rk_tendf1_moist_adv_opt2 \
    --out "$OUT/b5_skill_ab_20260509.json"
```

Pass criterion:

- Every variant run finishes and is finite.
- The report includes T2, U10, V10, QVAPOR, PSFC, and precipitation RMSE versus
  CPU-WRF, plus AEMET T2/U10/V10 where available.
- The report includes per-lead wind error growth slopes.
- The report states whether `rad_rk_tendf`, moisture-advection wiring, or their
  combination improves U10/V10 without materially degrading T2/QVAPOR/PSFC.

What it proves:

- This directly measures the #7 forecast-skill closure hypothesis instead of
  inferring it from CPU-only closure proofs.

### B6 - Full Implemented Scheme Operational Forecast Gates

Type: PRIMARY RUNS-confidence

Resource: GPU serial short forecasts

Estimate: 1h30m

Precondition:

- Land `proofs/v014/operational_suite_runner.py`. The existing
  `proofs/v060/forecast_gate_harness.py --run` refuses manager-scheduled runs,
  so the campaign needs a runner that launches short forecasts and records
  per-scheme active diagnostics.

Command:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  PYTHONPATH=src \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python proofs/v014/operational_suite_runner.py \
    --case-dir /mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z \
    --hours 3 \
    --matrix full_v013 \
    --out-dir "$OUT/b6_scheme_forecast_gates"
```

Pass criterion:

- Every implemented operational scheme in the v0.13 suite runs a 3h forecast
  without NaNs or crashes.
- The matrix includes:
  - Microphysics: Thompson, WSM6, WDM6, Morrison, Kessler, Lin, WSM3, WSM5,
    plus WSM7 as column-oracle-proven but `recognized_fail_closed` unless the
    qh State/dynamics/I-O leaf is operationalized.
  - PBL: MYNN, MYJ, YSU, ACM2, BouLac, MRF.
  - Surface: sfclayrev, MYNN, Janjic, GFS, old-MM5.
  - LSM: Noah-MP, Noah-classic.
  - Cumulus: KF, BMJ, GF, Tiedtke.
  - Radiation: RRTMG SW/LW, Dudhia SW, RRTM LW, GSFC SW.
  - GWD on/off.
- Each run records scheme-active diagnostics so a disabled branch cannot pass as
  a silent no-op.
- Reference-only and unported schemes are fail-closed and listed explicitly.

What it proves:

- The whole implemented physics surface is exercised in operational forecast
  form, not just in isolated oracles.

### B7 - Switzerland 24h Winter/Alps Run And CPU-WRF Compare

Type: PRIMARY RUNS-confidence, SECONDARY multi-region equivalence support

Resource: GPU serial plus CPU compare

Estimate: 45 min GPU plus 10 min CPU

GPU command:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  PYTHONPATH=src \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m gpuwrf.cli run \
    --input-dir /mnt/data/wrf_gpu_switzerland_128/run_cpu \
    --output-dir "$OUT/b7_switzerland_128_gpu" \
    --domain d01 \
    --hours 24 \
    --proof-dir "$OUT/b7_switzerland_128_gpu/proofs"
```

CPU compare command:

```bash
taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  PYTHONPATH=src \
  python scripts/equivalence_switzerland_compare.py \
    --gpu-dir "$OUT/b7_switzerland_128_gpu" \
    --cpu-dir /mnt/data/wrf_gpu_switzerland_128/run_cpu \
    --domain d01 \
    --hours 24 \
    --out "$OUT/b7_switzerland_equivalence.json"
```

Pass criterion:

- GPU run emits 24 finite d01 frames.
- Comparator finds pairable CPU-WRF and GPU-WRF frames and reports finite RMSE
  and status.
- `EQUIVALENT` and `NOT_EQUIVALENT` are both valid scientific outcomes; `NO_DATA`
  or missing pairs is a failure.

What it proves:

- The model is not only a Canary-specific runner.
- It gives a winter/Alps rough-equivalence data point, while avoiding a false
  Canary multi-season claim.

### B8 - CPU Reproducibility, Community, And Operational Smoke Sweep

Type: PRIMARY RUNS-confidence and SECONDARY reviewer support

Resource: CPU parallel lane

Estimate: 2h00m

Commands:

```bash
taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  PYTHONPATH=src \
  bash scripts/verify_reproducibility.sh

taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  PYTHONPATH=src \
  bash scripts/community_validation.sh

taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  JAX_ENABLE_X64=true \
  PYTHONPATH=src \
  python proofs/v060/multicfg_operational_smoke.py \
    --steps 16 \
    --out "$OUT/b8_multicfg_operational_smoke.json"
```

Pass criterion:

- Reproducibility proof suite passes.
- Community validation passes Straka, Skamarock, conservation, and restart
  checks.
- 16-step multicfg operational smoke passes with finite outputs and active
  scheme reporting.

What it proves:

- The core cheap proof surface stays green while GPU campaign work is running.

### B9 - Focused V0.13 Oracle, Wiring, Restart, And Fake-Mesh Sweep

Type: PRIMARY RUNS-confidence and SECONDARY reviewer support

Resource: CPU parallel lane

Estimate: 45 min

Commands:

```bash
taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  JAX_ENABLE_X64=true \
  PYTHONPATH=src \
  python -m pytest -q \
    tests/test_v013_myj_janjic_operational.py \
    tests/test_v013_mrf_operational.py \
    tests/test_v013_t3_surface_lsm_wiring.py \
    tests/test_v013_ra_sw_gsfc.py \
    tests/test_v060_ra_sw_dudhia.py \
    tests/test_rrtm_lw_operational_wiring.py \
    tests/test_gwd_operational_wiring.py \
    tests/test_v0110_boundary_feedback.py \
    tests/test_p0_1a_nesting.py \
    tests/test_p0_5_restart_full_carry.py \
    tests/test_v0110_wrfrst_netcdf.py \
    tests/test_m7_restart_checkpoint_roundtrip.py

taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/t3_microphysics_oracle.py

taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/myj_janjic_oracle.py

taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/mrf_oracle.py

taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/t3_surface_lsm_oracle.py

taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/t3_radiation_oracle.py

taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/t3_cumulus_oracle.py

taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  JAX_ENABLE_X64=true \
  XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  PYTHONPATH=src \
  python proofs/v013/multigpu_fakemesh.py
```

Pass criterion:

- All focused tests and proof scripts exit 0.
- Restart checkpoint roundtrip and wrfrst NetCDF handling are green.
- Fake mesh is partition bit-identical.
- Reference-only schemes are named honestly and fail closed.

What it proves:

- The deeper campaign has a complete CPU-side proof floor for the v0.13
  additions.

### B10 - Carry-Over Scheme Admission Path

Type: CONDITIONAL PRIMARY RUNS-confidence path

Resource: CPU oracle first, then GPU short forecast once ported

Estimate: not included in the mandatory 16h wall unless a scheme is newly
ported before campaign start

Scope:

- Cumulus JAX kernels beyond the current operational set.
- CAM, NUWRF, and GFDL radiation.
- RUC LSM.
- Shin-Hong and QNSE PBL.

Admission commands, per newly ported scheme:

```bash
taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  JAX_ENABLE_X64=true \
  WRF_PRISTINE_ROOT=/home/enric/src/wrf_pristine/WRF \
  PYTHONPATH=src \
  python proofs/v014/<family>_<scheme>_oracle.py \
    --out "$OUT/b10_<scheme>_oracle.json"

scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  PYTHONPATH=src \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python proofs/v014/operational_suite_runner.py \
    --case-dir /mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z \
    --hours 3 \
    --matrix full_v013_plus_carryovers \
    --only <scheme> \
    --out-dir "$OUT/b10_<scheme>_forecast_gate"
```

Pass criterion:

- Oracle parity or source-derived tolerance proof passes.
- Operational forecast emits 3 finite hourly outputs with active diagnostics.
- Until both gates pass, the scheme remains reference-only or fail-closed in
  public validation reports.

What it proves:

- Carry-over schemes have a concrete, falsifiable path into the campaign without
  being smuggled into operational claims.

## 16h Budget

GPU is serial. CPU lanes run concurrently on cores 0-23 and provide proof
coverage while GPU jobs run.

| Lane | Time window | Test | Estimate | Notes |
| --- | ---: | --- | ---: | --- |
| GPU | 00:00-01:45 | B1 L2 72h GWD 2-way | 1h45m | Multi-day heaviest fitting run |
| GPU | 01:45-03:45 | B2 L3 24h one-way GWD | 2h00m | Full 1 km production geometry |
| GPU | 03:45-05:00 | B3 L3 12h 2-way GWD | 1h15m | Bounded heaviest slice |
| GPU | 05:00-09:00 | B4 grid-cell envelope + targeted reruns | 4h00m | First gate; no station-only decision |
| GPU | 09:00-11:00 | B5 skill A/B | 2h00m | #7 closure measurement, only if it helps attribution |
| GPU | 11:00-12:30 | B6 scheme forecast gates | 1h30m | Full implemented suite |
| GPU | 12:30-13:15 | B7 Switzerland | 45 min | Non-Canary region |
| GPU | 13:15-16:00 | Slack / targeted probe retry / conditional TOST resume | 2h45m | TOST only after B4b is acceptable |
| CPU | 00:00-02:00 | B8 CPU proof sweep | 2h00m | Parallel |
| CPU | 02:00-02:45 | B9 focused proof sweep | 45 min | Parallel |
| CPU | 05:00-15:30 | B4/B5/B7 scoring and field reports as outputs appear | included | Does not extend GPU critical path |

Mandatory test count: 9.

Conditional carry-over admission path: 1 additional template test family.

Budgeted wall time: 16h00.

## Acceptance Summary

The 16h campaign passes only if:

- B1, B2, B3, B6, B7, B8, and B9 pass all RUNS-confidence criteria.
- B4 emits an all-comparable-field grid-cell envelope and either narrows the
  current V10/U10/PSFC divergence or records the next operator-specific falsifier.
  Powered TOST completion is conditional on this result, not mandatory before it.
- B5 reports finite A/B skill measurements, including wind error growth, even if
  no variant improves skill.
- The report clearly separates:
  - stable finite operational runs,
  - rough CPU-WRF equivalence,
  - AEMET forecast skill,
  - powered TOST pass/fail,
  - reference-only or fail-closed schemes.

The highest-value test is B4: it is now the direct field-parity gate using
retained CPU-WRF truth. Powered TOST remains the final gate after B4b is no
longer radically red.
