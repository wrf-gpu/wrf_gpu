# Sprint Contract: V0.14 Step-1 First-RK Part1 P-State Split

Date: 2026-06-09 18:57 WEST
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`
Base commit: `ebedb3c1`

## Objective

Find the first internal WRF/JAX boundary that introduces the current Step-1
`P/MU/W` state residual after the live-nest theta/QV fix.

Current accepted predecessor:

- Commit `ebedb3c1`: `v014 close p ph mu boundary localization`.
- Proof verdict: `STEP1_P_PH_MU_LOCALIZED_FIRST_RK_STEP_PART1_P_STATE`.
- Current first material P-family residual: WRF
  `after_first_rk_step_part1` vs JAX `_physics_step_forcing.carry.state`,
  `P_STATE` max_abs `69.96875`.
- `MU_STATE` max_abs `13.256103515625` and `W_STATE` max_abs
  `0.7605466246604919` are material at the same checked boundary.
- RK1 `small_step_prep` / `calc_p_rho(step=0)` work arrays are exact for
  checked `T_WORK/P_WORK/PH_WORK/MU_WORK/W_WORK`.
- Final strict Step-1 residual remains red: `P` max_abs
  `974.9820434775493`.

This sprint must split inside WRF `first_rk_step_part1` around the state writes
that feed `P/MU/W`, especially `phy_prep` and `calc_p_rho_phi`, or produce a
post-acoustic/pre-refresh pressure split if that is the first falsifiable
boundary. The result must be one exact internal boundary, one narrow
performance-compatible source fix with before/after proof, or one exact
missing-truth blocker.

## Method Rule

At top level, answer whether this is still the fastest rigorous wall-clock
method. The expected answer should be falsifiable from proof artifacts, not
intuition.

Preferred method:

1. Reuse the post-theta/QV Step-1 comparator from
   `proofs/v014/step1_p_ph_mu_boundary_localization.py`.
2. Add the smallest disposable WRF instrumentation needed to emit internal
   `first_rk_step_part1` surfaces for `P`, `MU`, and `W`.
3. Compare those surfaces against the exact current JAX boundary that enters or
   leaves `_physics_step_forcing`.
4. If the internal WRF boundary proves JAX already differs before any candidate
   source line, split the JAX adapter/carry path rather than guessing.

Avoid slow full forecasts, station scores, and broad source rewrites. This is a
kernel-level/debug-boundary sprint; one good savepoint is worth more than
another free-running run.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision source work.
- No memory source work.
- No long GPU forecast; prefer CPU-only proof.
- No Hermes or Telegram.
- No JAX-vs-JAX-only conclusion, one-cell-only conclusion, or station-score
  proxy.
- No broad dycore rewrite, CPU-WRF runtime dependency, timestep-loop
  host/device transfer, or performance-regressing fix.

## Inputs

- `proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `proofs/v014/step1_p_ph_mu_boundary_localization.json`
- `proofs/v014/step1_p_ph_mu_boundary_localization.md`
- `proofs/v014/step1_rk1_source_boundary.py`
- `proofs/v014/step1_rk1_source_boundary.json`
- `proofs/v014/step1_rk1_source_boundary_wrf_patch.diff`
- `proofs/v014/step1_t_p_operator_localization.py`
- `proofs/v014/step1_t_p_operator_localization.json`
- `proofs/v014/step1_t_p_operator_localization_wrf_patch.diff`
- `/mnt/data/wrf_gpu2/v014_step1_p_ph_mu_boundary_localization/**`
- `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/**`
- `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth/**`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_first_rk_part1_p_state_split/**`

## File Ownership

Required repo files:

- `proofs/v014/step1_first_rk_part1_p_state_split.py`
- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- `proofs/v014/step1_first_rk_part1_p_state_split.md`
- `.agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`

Optional repo files:

- `proofs/v014/step1_first_rk_part1_p_state_split_wrf_patch.diff`
- `proofs/v014/step1_first_rk_part1_p_state_split_source_patch.diff`

Optional production source edits only if the proof names an exact bug and the
fix is narrow and performance-compatible:

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/scan_adapters.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/dynamics/core/calc_p_rho.py`
- `src/gpuwrf/dynamics/core/small_step_prep.py`
- `src/gpuwrf/dynamics/core/small_step_finish.py`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/dynamics/mu_t_advance.py`
- `src/gpuwrf/dynamics/flux_advection.py`

Do not edit release docs, FP32 roadmap, memory roadmap, TOST outputs,
Switzerland outputs, or unrelated dirty/untracked artifacts.

## Required Work

1. Verify `ebedb3c1` is an ancestor and record branch/head in JSON.
2. Rerun or load the predecessor baseline and preserve its top residual table.
3. Emit or reuse WRF internal surfaces sufficient to split:
   - entry to `first_rk_step_part1`;
   - after `phy_prep`-related state/source writes;
   - before and after `calc_p_rho_phi` or equivalent pressure/mass/geopotential
     refresh;
   - before the `after_first_rk_step_part1` surface used by the predecessor.
4. Compare WRF internal surfaces against the matching JAX state/carry/tendency
   surfaces. If an exact JAX surface is missing, create it proof-locally where
   possible; otherwise name the missing contract exactly.
5. Distinguish:
   - state mutation versus tendency/source mutation;
   - boundary-band-only versus interior spread;
   - pressure refresh versus acoustic/small-step prep;
   - stale pre-theta/QV artifacts versus current post-fix state.
6. If a source fix is made, rerun a strict Step-1 comparison and report
   before/after top residuals plus why GPU performance is preserved.

## Verdicts

Emit exactly one final verdict:

- `STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_<operator_or_boundary>`
- `STEP1_FIRST_RK_PART1_P_STATE_FIXED_<operator_or_boundary>`
- `STEP1_FIRST_RK_PART1_P_STATE_BLOCKED_<specific_missing_truth_or_contract>`
- `STEP1_FIRST_RK_PART1_P_STATE_NO_REMAINING_DIVERGENCE`

## Validation Commands

At minimum:

```bash
python -m py_compile proofs/v014/step1_first_rk_part1_p_state_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_first_rk_part1_p_state_split.py
python -m json.tool proofs/v014/step1_first_rk_part1_p_state_split.json \
  >/tmp/step1_first_rk_part1_p_state_split.validated.json
git diff -- src/gpuwrf
```

If production source changes:

```bash
python -m py_compile \
  src/gpuwrf/runtime/operational_mode.py \
  src/gpuwrf/coupling/scan_adapters.py \
  src/gpuwrf/coupling/physics_couplers.py \
  src/gpuwrf/dynamics/core/calc_p_rho.py \
  src/gpuwrf/dynamics/core/small_step_prep.py \
  src/gpuwrf/dynamics/core/small_step_finish.py \
  src/gpuwrf/dynamics/core/rk_addtend_dry.py \
  src/gpuwrf/dynamics/mu_t_advance.py \
  src/gpuwrf/dynamics/flux_advection.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_theta_qv_wiring.py
python -m json.tool proofs/v014/step1_live_nest_theta_qv_wiring.json \
  >/tmp/step1_live_nest_theta_qv_wiring.after_part1_p_state_fix.validated.json
```

## Acceptance Criteria

- JSON validates and records CPU-only execution unless the manager later
  authorizes a short GPU check.
- The proof names exact WRF and JAX boundaries for every claimed residual.
- The result is a precise boundary/operator, a narrow before/after fix, or an
  exact missing-truth blocker.
- Any source fix preserves the GPU-native performance model: no timestep-loop
  host/device transfer, no CPU-WRF runtime dependency, no broad de-optimization.
- The review report includes objective, files changed, commands run, proof
  objects, unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_FIRST_RK_PART1_P_STATE_SPLIT DONE - see proofs/v014/step1_first_rk_part1_p_state_split.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
