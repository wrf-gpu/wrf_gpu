# Sprint Contract: V0.14 Step-1 P/PH/MU Boundary Localization

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`
Base commit: `3aa5f15b`

## Objective

Localize or narrowly fix the remaining d02 Step-1 strict same-input divergence
after production live-nest theta/QV initialization closure.

Current accepted state:

- `proofs/v014/step1_live_nest_theta_qv_wiring.json` verdict:
  `STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD`.
- Live-nest theta/QV initialization is now production-wired and closes against
  same-boundary WRF pre-call truth:
  - theta max_abs `5.788684885033035e-05 K`;
  - QVAPOR max_abs `5.970267497393267e-08`.
- The Step-1 16-field comparison still diverges:
  - first divergent schema field: `T`;
  - largest residual: `P` max_abs `974.9820434775493`, RMSE
    `135.98147360593399`;
  - worst `P` cell: Fortran `i=1,j=30,k=1`, boundary band true;
  - next material fields: `PH` max_abs `67.3623167023926`, `MU`
    `14.125275642998986`, `W` `2.640715693903735`, `U`
    `0.7835467705023085`.

This sprint must distinguish whether the remaining residual is a boundary
application/package issue, RK tendency/source issue, small-step/acoustic
pressure issue, pressure refresh issue, or comparison-boundary/schema issue.
If a precise, narrow source bug is proven, the sprint may apply the smallest
allowed fix and rerun the Step-1 comparison.

## Method Rule

At top level, answer the tooling question: are we still using the fastest
rigorous wall-clock method? Prefer a focused Step-1 boundary/substage
comparator over slow free-running forecasts.

The expected path is:

1. Reuse the CPU-only live-nest Step-1 input construction from
   `proofs/v014/step1_live_nest_theta_qv_wiring.py` and
   `proofs/v014/step1_live_nest_init_rerun.py`.
2. Reuse prior WRF substage truth and patch mechanics from
   `proofs/v014/step1_t_p_operator_localization.*` and
   `proofs/v014/step1_rk1_source_boundary.*`.
3. Build the smallest new comparator or WRF scratch hook needed to pinpoint the
   first boundary/operator that introduces the current `P/PH/MU` residual after
   theta/QV closure.

Forbidden paths:

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision source work.
- No memory source work.
- No long GPU forecast; avoid GPU entirely unless the manager explicitly
  authorizes a short check later.
- No Hermes or Telegram.
- No station-score proxy, JAX-vs-JAX-only conclusion, one-cell-only proof, or
  initial-state-vs-post-step comparison masquerading as Step-1 parity.
- No broad dycore rewrite, CPU-WRF wrapper, timestep-loop host/device transfer,
  or performance-regressing fix.

## Inputs

- `proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `proofs/v014/step1_live_nest_theta_qv_wiring.json`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/step1_t_p_operator_localization.py`
- `proofs/v014/step1_t_p_operator_localization.json`
- `proofs/v014/step1_t_p_operator_localization_wrf_patch.diff`
- `proofs/v014/step1_rk1_source_boundary.py`
- `proofs/v014/step1_rk1_source_boundary.json`
- `proofs/v014/step1_rk1_source_boundary_wrf_patch.diff`
- `proofs/v014/step1_same_input_truth.py`
- `proofs/v014/step1_same_input_truth_wrf_patch.diff`
- `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`
- `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth/**`
- `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/**`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_p_ph_mu_boundary_localization/**`

## Write Scope

Required repo files:

- `proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `proofs/v014/step1_p_ph_mu_boundary_localization.json`
- `proofs/v014/step1_p_ph_mu_boundary_localization.md`
- `.agent/reviews/2026-06-09-v014-step1-p-ph-mu-boundary-localization.md`

Optional repo file:

- `proofs/v014/step1_p_ph_mu_boundary_localization_wrf_patch.diff` if a scratch
  WRF instrumentation patch is created.

Optional production source edits only if the proof names an exact bug and the
fix is narrow:

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/boundary_apply.py`
- `src/gpuwrf/nesting/boundary_construction.py`
- `src/gpuwrf/contracts/halo.py`
- `src/gpuwrf/dynamics/core/small_step_prep.py`
- `src/gpuwrf/dynamics/core/calc_p_rho.py`
- `src/gpuwrf/dynamics/core/small_step_finish.py`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/dynamics/core/advance_w.py`
- `src/gpuwrf/dynamics/mu_t_advance.py`
- `src/gpuwrf/dynamics/flux_advection.py`

Do not edit unrelated source, release docs, memory roadmap, FP32 roadmap, TOST
outputs, Switzerland outputs, or old untracked artifacts.

## Required Work

1. Verify `3aa5f15b` is an ancestor and record branch/head in JSON.
2. Load the current Step-1 residual table after theta/QV closure and use it as
   the before-fix baseline.
3. Produce a compact boundary/operator table that names, for `T/P/PH/MU/W/U`,
   the earliest checked boundary where each field becomes material.
4. Specifically distinguish:
   - boundary-package construction versus boundary application;
   - RK source/tendency injection versus small-step prep;
   - `calc_p_rho`/pressure diagnostic refresh versus acoustic scan finish;
   - horizontal boundary band only versus interior spread;
   - old stale pre-theta-fix proof surfaces versus current post-theta-fix
     state.
5. If new WRF scratch instrumentation is needed, emit a diff file and keep all
   generated truth under the allowed scratch root.
6. If a source fix is made, rerun
   `proofs/v014/step1_live_nest_theta_qv_wiring.py` or a strict equivalent and
   report before/after top residuals. State explicitly why the fix preserves
   the GPU-native performance model.

## Verdicts

Emit exactly one final verdict:

- `STEP1_P_PH_MU_LOCALIZED_<operator_or_boundary>`
- `STEP1_P_PH_MU_FIXED_<operator_or_boundary>`
- `STEP1_P_PH_MU_BLOCKED_<specific_missing_truth_or_contract>`
- `STEP1_P_PH_MU_NO_REMAINING_DIVERGENCE`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_p_ph_mu_boundary_localization.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_p_ph_mu_boundary_localization.py
python -m json.tool proofs/v014/step1_p_ph_mu_boundary_localization.json \
  >/tmp/step1_p_ph_mu_boundary_localization.validated.json
git diff -- src/gpuwrf
```

If production source changes:

```bash
python -m py_compile \
  src/gpuwrf/runtime/operational_mode.py \
  src/gpuwrf/coupling/boundary_apply.py \
  src/gpuwrf/nesting/boundary_construction.py \
  src/gpuwrf/contracts/halo.py \
  src/gpuwrf/dynamics/core/small_step_prep.py \
  src/gpuwrf/dynamics/core/calc_p_rho.py \
  src/gpuwrf/dynamics/core/small_step_finish.py \
  src/gpuwrf/dynamics/core/rk_addtend_dry.py \
  src/gpuwrf/dynamics/core/advance_w.py \
  src/gpuwrf/dynamics/mu_t_advance.py \
  src/gpuwrf/dynamics/flux_advection.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_theta_qv_wiring.py
python -m json.tool proofs/v014/step1_live_nest_theta_qv_wiring.json \
  >/tmp/step1_live_nest_theta_qv_wiring.after_p_ph_mu_fix.validated.json
```

## Acceptance Criteria

- JSON validates and records CPU-only execution unless a manager-authorized GPU
  check was added later.
- The proof names the exact WRF/JAX boundary/operator used for every conclusion.
- The result is either a precise localized boundary/operator, a narrow
  before/after fix, or an exact blocker saying which truth surface/contract is
  missing.
- Any source fix is performance-compatible: no timestep-loop host/device
  transfer, no CPU-WRF runtime dependency, no broad de-optimization.
- The review report includes objective, files changed, commands run, proof
  objects, unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'OPUS STEP1_P_PH_MU_BOUNDARY_LOCALIZATION DONE - see proofs/v014/step1_p_ph_mu_boundary_localization.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
