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

Release-gate update 2026-06-09 16:08 WEST:

TOST is still required before v0.14 close, but it is not sufficient as a
station-only artifact. The final validation output must have two pillars:

- **Pillar A — ADR-029 powered station TOST** for AEMET T2/U10/V10 RMSE
  equivalence.
- **Pillar B — Grid-Delta Atlas** comparing GPU wrfout against CPU-WRF wrfout
  for every paired case, lead time, grid cell, and common numeric field.

The atlas requirement is now recorded in
`.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`. It must generate compact
release plots and a README-embeddable dashboard before tag:

- field x lead heatmaps for RMSE, bias, p99, and max_abs;
- lead-time drift/stability plots for mandatory core fields;
- delta-distribution plots and worst-case spatial maps;
- a machine-readable pass/fail manifest and tolerance envelope.

The correct claim is not "bitwise identity" unless proven. The v0.14 claim
should be near-equivalence under predeclared field envelopes, stable bounded
drift, and full inventory transparency for every common numeric wrfout field.
Station TOST must be interpreted together with this grid-delta atlas.

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
- 2026-06-09 15:44 WEST manager update: validation remains paused behind the
  Step-1 grid-parity ladder. The current active boundary is the JAX live-nest
  loader/carry `T_STATE` construction before `_physics_step_forcing`
  (`STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`). The next proof must split
  raw d02 state, live-nest base-init state, boundary package, initial carry, and
  haloed step-entry. Do not run Switzerland, TOST, FP32, memory source work, or
  GPU validation until this stage is explained or fixed.
- 2026-06-09 15:58 WEST manager update: the JAX loader split is closed as
  `STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`. Boundary
  package, carry, and halo are ruled out. The next blocker is WRF live-nest
  `t_2`/theta initialization after `med_nest_initial` terrain/base blending and
  `start_domain_em`; validation remains paused until an initialization-only
  source fix is proven or the residual is otherwise explained.
- 2026-06-09 16:03 WEST manager update: the active proof/fix sprint is
  `v014-step1-live-nest-theta-semantics`. It must prove WRF
  `adjust_tempqv` candidate semantics against WRF pre-call truth before any
  production source edit. Validation remains paused behind this gate.
- 2026-06-09 16:08 WEST principal validation update: when validation resumes,
  powered TOST must produce a full Grid-Delta Atlas over all paired CPU/GPU
  wrfout fields, times, cells, and cases, with release plots committed and
  README-linked. The existing limited cell-level stats in the old powered-TOST
  runner are not enough for v0.14.
- 2026-06-09 16:18 WEST manager update: live-nest theta semantics is a partial,
  not a source-fix close. `proofs/v014/step1_live_nest_theta_semantics.json`
  verdict is
  `STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_TSTATE_MILLIKELVIN_RESIDUAL`.
  WRF `USE_THETA_M=1` dry-to-moist theta conversion plus `adjust_tempqv`
  reduces `T_STATE` max_abs from `5.490173101425171` to
  `0.00541785382188209`, but remains above the prior `1e-3 K` material gate.
  The QVAPOR schema proof
  `proofs/v014/step1_qvapor_precall_truth_schema.json` closed as
  `STEP1_QVAPOR_PRECALL_TRUTH_MISSING_SAVEPOINT_SPEC_READY`: same-boundary
  pre-call `QVAPOR` truth is missing and existing QVAPOR artifacts are
  post-RK/pre-halo. B4 remains blocked; next validation-enabling sprint is the
  minimal CPU-WRF `before_first_rk_step_part1_call` QVAPOR savepoint, then a
  rerun of the theta proof.
- 2026-06-09 18:17 WEST manager update: production live-nest theta/QV wiring is
  now closed by `proofs/v014/step1_live_nest_theta_qv_wiring.json` with verdict
  `STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD`. The live-nest
  `build_replay_case` path applies WRF `USE_THETA_M=1` theta conversion plus
  `adjust_tempqv`; corrected theta max_abs is `5.788684885033035e-05 K` and
  QVAPOR max_abs is `5.970267497393267e-08`. B4 remains blocked because the
  Step-1 16-field comparison still diverges: first divergent schema field `T`,
  largest residual `P` max_abs `974.9820434775493`, worst Fortran
  `i=1,j=30,k=1`, boundary band true. The next validation-enabling sprint is
  Step-1 `P/PH/MU` boundary/operator localization, not TOST, Switzerland, FP32,
  or memory work.
- 2026-06-09 18:22 WEST manager update: the active validation-enabling sprint is
  `.agent/sprints/2026-06-09-v014-step1-p-ph-mu-boundary-localization`. Its
  gate is a focused Step-1 boundary/operator proof or narrow before/after fix
  for the current `P/PH/MU` residual. B4 remains paused until this sprint either
  closes the residual, names the exact next operator, or records a precise
  missing-truth blocker.
- 2026-06-09 18:50 WEST manager update: P/PH/MU boundary localization closed as
  `STEP1_P_PH_MU_LOCALIZED_FIRST_RK_STEP_PART1_P_STATE`. The current first
  material P-family state residual is WRF `after_first_rk_step_part1` versus
  JAX `_physics_step_forcing.carry.state`, `P_STATE` max_abs `69.96875`;
  `MU_STATE` and `W_STATE` are material at the same checked boundary. RK1
  `small_step_prep`/`calc_p_rho(step=0)` work arrays are exact for the checked
  work fields. B4 remains blocked because final strict Step-1 still has `P`
  max_abs `974.9820434775493`. Next validation-enabling sprint is an internal
  WRF `first_rk_step_part1` split around `phy_prep`/`calc_p_rho_phi` state
  writes for `P/MU/W`, or a post-acoustic/pre-refresh pressure split before any
  source edit.
- 2026-06-09 18:57 WEST manager update: the active validation-enabling sprint is
  `.agent/sprints/2026-06-09-v014-step1-first-rk-part1-p-state-split`. It is a
  CPU-first internal boundary split around WRF `first_rk_step_part1`
  `phy_prep` / `calc_p_rho_phi` and the matching JAX `_physics_step_forcing`
  surfaces. B4, Switzerland, TOST, FP32 source work, and memory source work
  remain paused until this sprint returns an exact boundary, narrow fix, or
  exact blocker.
- 2026-06-09 19:08 WEST manager update: first-RK part1 P-state split closed as
  `STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`. WRF
  `before_first_rk_step_part1_call -> after_first_rk_step_part1` is exact for
  `P_STATE/MU_STATE/W_STATE/PH_STATE`; JAX is already off at
  `raw_child_state` and preserves the same `P/MU/W` residuals through
  live-child, boundary package, carry, halo, and `_physics_step_forcing`.
  Validation remains paused. The next validation-enabling work is a narrow
  live-nest perturbation-state initialization proof/fix for
  `P_STATE/MU_STATE/W_STATE`.
- 2026-06-09 Step-1 debug update:
  `proofs/v014/step1_rk1_source_boundary.json` localizes the first material
  Step-1 mismatch to WRF `after_first_rk_step_part1`, field `T_STATE`, not to
  acoustic/small-step pressure refresh. The residual is material against both
  JAX operational carry and `_physics_step_forcing.state` (max_abs about
  `5.49` K, RMSE about `1.92` K), while `small_step_prep` `T_WORK/P_WORK`
  continuity remains exact. Validation campaigns, Switzerland, TOST, FP32, and
  memory work stay behind the grid-parity gate until the internal
  `first_rk_step_part1` mutation is explained or fixed.
- 2026-06-09 Part1 update:
  `proofs/v014/step1_part1_physics_state_mutation.json` rules out
  `first_rk_step_part1` internals for the current `T_STATE` residual. The full
  residual is already present at `part1_entry_before_init_zero_tendency`
  (max_abs `5.490173101425171`, RMSE `1.9175184863907806`), and WRF's largest
  internal `T_STATE` delta from that entry is max_abs `0.0`. The next validation
  enabler is upstream call-site/state-handoff localization before
  `first_rk_step_part1`.
- 2026-06-09 Pre-part1 update:
  `proofs/v014/step1_pre_part1_handoff.json` localizes the current `T_STATE`
  residual to JAX live-nest Step-1 loader/carry construction before
  `_physics_step_forcing`. WRF solve_em does not change `grid%t_2` before
  `first_rk_step_part1` (max_abs `0.0`), the solve_em pre-call hook is
  continuous with prior part1-entry truth (max_abs `0.0`), and WRF `T_STATE`
  maps to JAX `State.theta - 300 K`, not full theta. Validation campaigns stay
  paused until the JAX loader/carry source is split and fixed or explained.
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
- 2026-06-09 step-1 live-nest init rerun closeout:
  `proofs/v014/step1_live_nest_init_rerun.json` verdict is
  `STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_CLOSED_NEXT_T`. The live-nest/base
  initialization residual is closed in the strict Step-1 proof:
  `MUB/PB/PHB` max_abs are about `0.05/0.05/0.11`, respectively. B4 remains
  blocked because the Step-1 comparison still diverges: first divergent field
  `T`, largest residual `P` max_abs `1561.2503728885986`, RMSE
  `305.9413510899027`, with `PH/MU/W` also material. The next
  validation-enabling work is Step-1 operator/source localization. Do not resume
  powered TOST, Switzerland validation, FP32 source work, or memory follow-ups
  until this grid-parity boundary is explained and reduced.
- 2026-06-09 step-1 T/P operator-localization closeout:
  `proofs/v014/step1_t_p_operator_localization.json` verdict is
  `STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_T_STATE`.
  Disposable WRF instrumentation produced 168 substage truth files and localized
  the first strict/material T/P-family mismatch to `T_STATE` at
  `after_rk_addtend_before_small_step_prep`, RK1. The top residual at that
  boundary is `PH_TEND` max_abs `794096.1875`; `RW_TEND`, `PH_TENDF`,
  `T_TEND`, and `T_TENDF` are also large. RK1 `after_small_step_prep_calc_p_rho`
  work arrays `T_WORK` and `P_WORK` then match exactly. B4 remains blocked. The
  next validation-enabling work is WRF `first_rk_step_part1/part2` versus JAX
  `_physics_step_forcing` / dry `*_tendf` construction, not acoustic or final
  pressure refresh.
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
- 2026-06-09 step-1 same-input truth closeout:
  `proofs/v014/step1_same_input_truth.json` verdict is
  `STEP1_SAME_INPUT_COMPARISON_EXECUTED_FIRST_DIVERGENT_T`. The first strict
  full-domain d02 step-1 comparison executed against CPU-WRF
  `post_after_all_rk_steps_pre_halo` truth. First divergent schema field is `T`,
  while the largest residuals are `MUB/PB/PHB/P` (`MUB` max_abs `2635.640625`,
  `PB` `2627.3828125`, `PHB` `2237.9423828125`). B4 remains blocked. The next
  validation-enabling work is native live-nest child base-state initialization
  or a decisive init-override falsifier, followed by a rerun of
  `proofs/v014/step1_same_input_truth.py`.
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

## Update 2026-06-09 16:45 WEST

The validation plan remains gated by grid parity, not by station-only TOST.
Before any long validation campaign, the current Step-1 live-nest divergence
chain must close or be explicitly bounded.

Latest closed proof:

- `proofs/v014/step1_qvapor_precall_savepoint.{py,json,md}` with verdict
  `STEP1_QVAPOR_PRECALL_SAVEPOINT_READY`.
- Same-boundary pre-call QVAPOR root:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.
- Old fields stayed text-identical to the accepted pre-call dump, max_abs
  `0.0`; QVAPOR is full shape `[44,66,159]`, all finite.

Immediate pre-validation gate:

1. Rerun the theta semantics proof using this QVAPOR root.
2. Classify the remaining worst `T_STATE` residual cell as boundary/interior.
3. Continue to the larger base-state split/V10 driver if the theta tail is
   bounded and not the main grid-delta cause.

Final v0.14 validation must still include the Grid-Delta Atlas gate from
`.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`: all paired CPU/GPU wrfout
times, all cells, all common numeric fields, deterministic plots, and README
dashboard. TOST station scores are a pillar, not a substitute for cell-field
delta stability.

## Update 2026-06-09 17:00 WEST

Same-boundary QVAPOR validation is complete, but it does not yet unblock long
validation. Verdict:
`STEP1_THETA_SAME_QVAPOR_INTERIOR_RESIDUAL_NEEDS_WRF_INTERMEDIATE`.

Validation implications:

- The QVAPOR truth gap is closed: candidate QVAPOR versus WRF pre-call QVAPOR
  max_abs is `3.838436518426372e-06`.
- The remaining `T_STATE` max_abs `0.00541785382188209 K` is an interior
  residual, not a horizontal boundary-only tail.
- Long GPU validation, TOST, and Switzerland remain paused until the next
  WRF-intermediate proof explains or bounds this residual.

Immediate pre-validation gate:

1. Emit WRF exact `adjust_tempqv` intermediates for the residual path.
2. Decide whether an init-only production patch is warranted or whether the
   lane returns to the larger base-state/V10 driver.
3. Only after this grid-parity branch is fixed or explicitly bounded should the
   v0.14 Grid-Delta Atlas and TOST/Switzerland campaign resume.

## Update 2026-06-09 17:16 WEST

The WRF-intermediate gate is now classified, not blocked. Verdict:
`STEP1_ADJUST_TEMPQV_INTERMEDIATE_PRESSURE_INPUT_MISMATCH`.

Validation remains paused. The exact WRF hook showed that saved inputs match,
but current `mub`/`pb_new_equiv`/`p_new` differ by about `17.5 Pa` at the
interior worst cell. This is enough to explain the remaining `0.0054 K`
theta residual and means the next validation-enabling work is a current
live-nest base-input split/fix, not TOST, Switzerland, FP32, or memory source
work.

Immediate pre-validation gate:

1. Split WRF current `MUB/PB` construction around live-nest `blend_terrain`,
   base recomputation, and `adjust_tempqv` call-site inputs.
2. Compare against the JAX live-nest base-init reconstruction used by
   `step1_theta_same_qvapor`.
3. Only resume long validation after this pressure/base-input mismatch is
   patched with field proof or explicitly bounded as not driving the larger
   grid deltas.

## Update 2026-06-09 17:32 WEST

The current `MUB/PB` split gate is closed and points to a small init/theta
source fix. Verdict:
`STEP1_CURRENT_MUB_BASE_SPLIT_WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED`.

Validation remains paused. The next pre-validation gate is now:

1. Add or prove a transient post-`blend_terrain`/pre-`start_domain` current
   `MUB` path for WRF `adjust_tempqv`.
2. Use that transient field only for live-nest theta/QV adjustment; keep final
   post-`start_domain` BaseState unchanged for step-entry.
3. Rerun the Step-1 theta/QV proof and require a field-level guard before
   returning to Grid-Delta Atlas, Switzerland, or TOST.

## Update 2026-06-09 17:48 WEST

The transient adjust-base helper proof is green:
`STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_CLOSED`.

Validation remains paused because the helper is not yet wired into the
production live-nest init consumer. The next pre-validation gate is now:

1. Wire WRF theta_m conversion plus `adjust_tempqv` into live-nest init using
   `_wrf_live_nest_transient_adjust_mub`.
2. Run the full Step-1 same-input d02 comparison across the 16-field schema.
3. Only after that comparison closes or names the next boundary should we resume
   Grid-Delta Atlas, Switzerland, or TOST planning.
