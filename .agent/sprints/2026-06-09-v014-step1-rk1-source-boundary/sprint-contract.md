# Sprint Contract: V0.14 Step-1 RK1 Source Boundary

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Split the newly localized Step-1 mismatch boundary between CPU-WRF
`first_rk_step_part1/part2` and JAX `_physics_step_forcing` / dry tendency
construction.

Trigger evidence:

- `proofs/v014/step1_t_p_operator_localization.json`
- verdict
  `STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_T_STATE`
- first strict/material T/P-family mismatch:
  `after_rk_addtend_before_small_step_prep`, RK1, `T_STATE`
- same boundary has large `PH_TEND`, `RW_TEND`, `PH_TENDF`, `T_TEND`, and
  `T_TENDF` residuals
- RK1 `after_small_step_prep_calc_p_rho` work arrays `T_WORK` and `P_WORK`
  match exactly, so do not continue acoustic debugging yet

The sprint must determine whether the source is:

- WRF/JAX physics state mutation mismatch;
- WRF/JAX `*_tendf` dry tendency mismatch;
- JAX carry/state handoff mismatch after `_physics_step_forcing`;
- `rk_tendency` / `_augment_large_step_tendencies` mismatch;
- `rk_addtend_dry` / boundary tendency application mismatch;
- or a narrowly provable production bug in those paths.

## Method Rule

Use the fastest rigorous wall-clock method: extend the existing Step-1 substage
truth/comparator, not a long validation run.

Accepted comparisons:

1. Emit WRF d02 Step-1 RK1 truth surfaces at the smallest useful boundaries:
   - after `first_rk_step_part1`;
   - after `first_rk_step_part2`;
   - after `rk_tendency`, before `relax_bdy_dry` / `rk_addtend_dry`;
   - after `rk_addtend_dry/spec_bdy_dry` if needed for continuity with the
     previous proof.
2. Compare those surfaces to JAX:
   - initial live-nest Step-1 carry/state;
   - `_physics_step_forcing(...).state`, `.carry`, and `.dry_tendencies`;
   - `compute_advection_tendencies` plus `_augment_large_step_tendencies`;
   - pre-`small_step_prep` state and tendencies.
3. If a concrete production bug is proven, apply the smallest targeted fix and
   rerun the Step-1 strict proof before reporting success.

Forbidden comparisons:

- no WRF final truth vs JAX initial state;
- no JAX-vs-JAX-only conclusion;
- no one-cell/station proxy;
- no broad acoustic debugging unless the new boundary proof closes the earlier
  source boundary and residuals remain.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision source work.
- No memory source work.
- No GPU.
- No Hermes or Telegram.
- No broad dycore rewrite or performance-regressing source change.

## Inputs

- `proofs/v014/step1_t_p_operator_localization.py`
- `proofs/v014/step1_t_p_operator_localization.json`
- `proofs/v014/step1_t_p_operator_localization_wrf_patch.diff`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/same_input_contract_builder.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/dynamics/core/small_step_prep.py`
- `src/gpuwrf/dynamics/core/calc_p_rho.py`
- `src/gpuwrf/dynamics/core/small_step_finish.py`
- `src/gpuwrf/dynamics/flux_advection.py`
- `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/**`
- `/mnt/data/wrf_gpu2/v014_step1_same_input_truth/**`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/**`

## Write Scope

Required repo files:

- `proofs/v014/step1_rk1_source_boundary.py`
- `proofs/v014/step1_rk1_source_boundary.json`
- `proofs/v014/step1_rk1_source_boundary.md`
- `.agent/reviews/2026-06-09-v014-step1-rk1-source-boundary.md`

Optional repo files:

- `proofs/v014/step1_rk1_source_boundary_wrf_patch.diff`
- targeted source edits only if an exact, narrow, performance-compatible bug is
  proven:
  - `src/gpuwrf/runtime/operational_mode.py`
  - `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
  - `src/gpuwrf/dynamics/flux_advection.py`
  - `src/gpuwrf/dynamics/core/small_step_prep.py`
  - `src/gpuwrf/dynamics/core/calc_p_rho.py`
  - `src/gpuwrf/dynamics/core/small_step_finish.py`

Do not touch unrelated source, TOST outputs, Switzerland outputs, FP32 work,
memory source work, or old untracked artifacts.

## Required Work

1. Reuse the live-nest Step-1 input path and the existing substage parser logic
   where practical.
2. Emit or consume WRF truth for RK1 boundaries before `small_step_prep`, with
   enough fields to compare:
   - `T/P/PB/PH/PHB/MU/MUB/W`;
   - `t_tend/t_tendf`, `ph_tend/ph_tendf`, `rw_tend/rw_tendf`,
     `mu_tend/mu_tendf`;
   - any WRF `h_diabatic`, `rthften`, or relevant dry-physics leaves needed to
     explain `T_STATE` and `T_TENDF`.
3. Capture matching JAX leaves:
   - before `_physics_step_forcing`;
   - after `_physics_step_forcing`;
   - after `_augment_large_step_tendencies` for RK1;
   - after `rk_addtend_dry` equivalent if separated.
4. Classify the first mismatch boundary. The result must say whether the issue
   is in physics state output, dry tendency output, state/carry handoff,
   tendency augmentation, boundary tendency application, or still unresolved
   because a specific truth leaf is missing.
5. If a source fix is made, rerun:
   - this sprint proof;
   - `proofs/v014/step1_t_p_operator_localization.py`;
   - `proofs/v014/step1_live_nest_init_rerun.py`;
   and report before/after top residuals.

## Verdicts

Emit exactly one final verdict:

- `STEP1_RK1_SOURCE_LOCALIZED_<boundary_or_leaf>`
- `STEP1_RK1_SOURCE_FIXED_<boundary_or_leaf>`
- `STEP1_RK1_SOURCE_BLOCKED_<specific_missing_truth_or_contract>`
- `STEP1_RK1_SOURCE_NO_REMAINING_DIVERGENCE`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_rk1_source_boundary.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_rk1_source_boundary.py
python -m json.tool proofs/v014/step1_rk1_source_boundary.json \
  >/tmp/step1_rk1_source_boundary.validated.json
git diff -- src/gpuwrf
```

If production source changes:

```bash
python -m py_compile \
  src/gpuwrf/runtime/operational_mode.py \
  src/gpuwrf/dynamics/core/rk_addtend_dry.py \
  src/gpuwrf/dynamics/flux_advection.py \
  src/gpuwrf/dynamics/core/small_step_prep.py \
  src/gpuwrf/dynamics/core/calc_p_rho.py \
  src/gpuwrf/dynamics/core/small_step_finish.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_t_p_operator_localization.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_init_rerun.py
```

## Acceptance Criteria

- JSON validates and records CPU-only execution.
- The proof names the exact WRF/JAX boundary and field/leaf for the first
  material mismatch or exact blocker.
- Any source fix is narrow and performance-compatible: no host/device transfer
  inside timestep loops, no CPU-WRF wrapper, no broad de-optimization.
- Production `src/gpuwrf/**` remains unchanged unless a concrete bug is proven.
- Review report includes objective, files changed, commands run, proof objects,
  unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_RK1_SOURCE_BOUNDARY DONE - see proofs/v014/step1_rk1_source_boundary.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
