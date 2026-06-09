# Sprint Contract: V0.14 Step-1 T/P Operator Localization

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Localize the remaining d02 Step-1 strict same-input WRF-vs-JAX divergence after
live-nest base initialization closure.

The known state is:

- `proofs/v014/step1_live_nest_init_rerun.json` verdict
  `STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_CLOSED_NEXT_T`.
- Base residuals are closed: `MUB/PB/PHB` max_abs about `0.05/0.05/0.11`.
- The strict Step-1 comparison still diverges: first divergent schema field `T`;
  largest residual `P` max_abs `1561.2503728885986`; `PH/MU/W` are also
  material.

This sprint must find the earliest dynamic/operator boundary that introduces
the `T/P/PH/MU` residuals, or deliver a narrowly justified fix with before/after
proof. It must not resume long validation runs.

## Method Rule

At top level, answer the manager's tooling question: are we using the fastest
rigorous wall-clock method? The expected method is a focused Step-1 substage
truth/comparator harness, not slow full-runtime reproduction.

Accepted paths:

1. Extend the existing CPU-only proof harness to capture JAX Step-1 boundaries:
   physics forcing output, RK stage entries, large-step tendencies after
   `_augment_large_step_tendencies` / `rk_addtend_dry`, `small_step_prep_wrf`,
   `calc_p_rho_wrf`, per-stage post-acoustic/pre-halo state, and final pre-halo
   state.
2. If WRF substage truth is needed, patch only a disposable WRF copy under
   `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/**` to emit the
   matching d02 Step-1 substage arrays. Reuse existing built scratch trees when
   safe; do not modify pristine WRF or production source for instrumentation.
3. If a concrete production bug is proven, apply only the smallest targeted fix
   inside the allowed source scope and rerun the Step-1 proof before reporting
   success.

Forbidden paths:

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision work.
- No memory source work.
- No GPU.
- No Hermes or Telegram.
- No station-score proxy, JAX-vs-JAX-only conclusion, one-cell proof, or
  initial-state-vs-post-step comparison.
- No broad dycore rewrite or performance-regressing source change.

## Inputs

- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/step1_same_input_truth.py`
- `proofs/v014/step1_same_input_truth_wrf_patch.diff`
- `proofs/v014/same_input_contract_builder.py`
- `proofs/v014/source_save_boundary_hook_wrf_patch.diff`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/dynamics/core/small_step_prep.py`
- `src/gpuwrf/dynamics/core/calc_p_rho.py`
- `src/gpuwrf/dynamics/core/small_step_finish.py`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/dynamics/core/advance_w.py`
- `src/gpuwrf/dynamics/mu_t_advance.py`
- `src/gpuwrf/dynamics/flux_advection.py`
- `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`
- `/mnt/data/wrf_gpu2/v014_step1_same_input_truth/**`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/**`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/**`

## Write Scope

Required repo files:

- `proofs/v014/step1_t_p_operator_localization.py`
- `proofs/v014/step1_t_p_operator_localization.json`
- `proofs/v014/step1_t_p_operator_localization.md`
- `.agent/reviews/2026-06-09-v014-step1-t-p-operator-localization.md`

Optional repo files:

- `proofs/v014/step1_t_p_operator_localization_wrf_patch.diff` if WRF scratch
  instrumentation is created.
- Targeted production source edits only if an exact bug is proven and the fix is
  narrow:
  - `src/gpuwrf/runtime/operational_mode.py`
  - `src/gpuwrf/dynamics/core/small_step_prep.py`
  - `src/gpuwrf/dynamics/core/calc_p_rho.py`
  - `src/gpuwrf/dynamics/core/small_step_finish.py`
  - `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
  - `src/gpuwrf/dynamics/core/advance_w.py`
  - `src/gpuwrf/dynamics/mu_t_advance.py`
  - `src/gpuwrf/dynamics/flux_advection.py`

Do not touch unrelated source, TOST outputs, Switzerland outputs, FP32 work, or
old untracked artifacts.

## Required Work

1. Reuse or factor the live-nest Step-1 input construction from
   `proofs/v014/step1_live_nest_init_rerun.py`.
2. Emit a compact ranked residual table for the final accepted comparison and
   detailed per-boundary arrays/metrics in JSON.
3. Create substage comparisons sufficient to distinguish at least these
   hypotheses:
   - physics forcing / RK-fixed dry tendency injection creates `T` residual;
   - `rk_addtend_dry` or map/mass coupling creates `T/P` residual;
   - `small_step_prep_wrf` work/reference split creates residuals;
   - `calc_p_rho_wrf` pressure diagnostic/work pressure mismatch creates `P`;
   - acoustic scan / `small_step_finish_wrf` creates `PH/MU/W/P` residuals;
   - final pressure refresh `_refresh_grid_p_from_finished` creates `P`.
4. If WRF substage truth is emitted, compare full d02 arrays for the same frozen
   16-field schema where available, plus diagnostics for `theta_work`,
   `theta_tend`, `mu_work`, `p_work`, `ph_work`, `t_save`, `mu_save`, `ph_save`,
   and WRF `grid%p` if emitted.
5. If a source fix is made, rerun the Step-1 strict comparison after the fix and
   report before/after top residuals. State explicitly why the fix does not
   undermine the high-performance GPU design.

## Verdicts

Emit exactly one final verdict:

- `STEP1_TP_LOCALIZED_<operator_or_boundary>`
- `STEP1_TP_FIXED_<operator_or_boundary>`
- `STEP1_TP_BLOCKED_<specific_missing_truth_or_contract>`
- `STEP1_TP_NO_REMAINING_DIVERGENCE`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_t_p_operator_localization.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_t_p_operator_localization.py
python -m json.tool proofs/v014/step1_t_p_operator_localization.json \
  >/tmp/step1_t_p_operator_localization.validated.json
git diff -- src/gpuwrf
```

If production source changes:

```bash
python -m py_compile \
  src/gpuwrf/runtime/operational_mode.py \
  src/gpuwrf/dynamics/core/small_step_prep.py \
  src/gpuwrf/dynamics/core/calc_p_rho.py \
  src/gpuwrf/dynamics/core/small_step_finish.py \
  src/gpuwrf/dynamics/core/rk_addtend_dry.py \
  src/gpuwrf/dynamics/core/advance_w.py \
  src/gpuwrf/dynamics/mu_t_advance.py \
  src/gpuwrf/dynamics/flux_advection.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_init_rerun.py
python -m json.tool proofs/v014/step1_live_nest_init_rerun.json \
  >/tmp/step1_live_nest_init_rerun.after_tp_fix.validated.json
```

## Acceptance Criteria

- JSON validates and records CPU-only execution.
- The proof avoids weak comparisons and names the exact WRF/JAX boundary used.
- The result is either a precise localized boundary/operator, a narrow
  before/after fix, or an exact blocker that says what truth surface/contract is
  missing.
- Any source fix is performance-compatible: no host/device transfer in timestep
  loops, no CPU-WRF wrapper, no broad de-optimization.
- Review report includes objective, files changed, commands run, proof objects,
  unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_TP_OPERATOR_LOCALIZATION DONE - see proofs/v014/step1_t_p_operator_localization.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
