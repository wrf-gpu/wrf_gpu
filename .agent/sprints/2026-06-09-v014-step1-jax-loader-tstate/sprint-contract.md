# Sprint Contract: V0.14 Step-1 JAX Loader T-State

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Split the JAX live-nest Step-1 loader/carry construction for `T_STATE` against
the accepted WRF solve_em pre-`first_rk_step_part1` truth.

Trigger evidence:

- `proofs/v014/step1_pre_part1_handoff.json`
- verdict `STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`
- WRF `T_STATE` is unchanged from `after_step_increment` to
  `before_first_rk_step_part1_call`: max_abs `0.0`
- WRF `grid%t_2` maps to JAX `State.theta - 300 K`
- WRF pre-call `T_STATE` vs raw JAX live-nest state has max_abs
  `5.490173101425171`, RMSE `1.9175184863907806`

This sprint must determine whether the exact source is:

- raw d02 `wrfinput`/loaded `State.theta`;
- live-nest base initialization updating `PB/PHB/MUB` without corresponding
  `T_STATE`/theta semantics;
- parent boundary package construction;
- initial `OperationalCarry` construction;
- halo/step-entry view construction;
- or one exact, narrow, performance-compatible production bug.

## Method Rule

Use the fastest rigorous wall-clock method: a CPU-only JAX loader-stage
comparator against the existing WRF pre-call truth. Do not rebuild WRF or run a
long validation campaign unless the existing truth is proven invalid.

At planning time, explicitly ask whether this is the right tool. For this
sprint it is: the mismatch is already in raw JAX loader/carry state, and all
candidate stages can be split in one proof process without GPU.

Accepted stages:

1. `raw_child_state`: proof-local `_state_from_wrfinput` / raw d02 state.
2. `live_child_state`: after `_apply_live_nest_base_init` and recomputed
   `BaseState`.
3. `boundary_packaged_state`: after `build_child_boundary_package`.
4. `initial_carry_state`: after `initial_operational_carry`.
5. `haloed_step_entry_state`: after the exact halo path used by
   `_physics_step_forcing`.

For each stage, compare at least:

- `T_STATE = State.theta - 300 K`;
- `THETA_FULL = State.theta` where useful for semantic checks;
- `P_STATE`, `PB`, `MU_STATE`, `MUB`, `MUT`;
- `PH_STATE`, `PHB`, and `W_STATE` if available without expanding scope.

The proof must also split interior versus lateral boundary bands for `T_STATE`
so a boundary-only package issue is not confused with a full-domain loader
issue.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision source work.
- No memory source work.
- No GPU.
- No Hermes or Telegram.
- No broad dycore rewrite or performance-regressing source change.
- No new WRF instrumentation unless existing WRF pre-call truth is proven
  insufficient.

## Inputs

- `proofs/v014/step1_pre_part1_handoff.py`
- `proofs/v014/step1_pre_part1_handoff.json`
- `proofs/v014/step1_pre_part1_handoff.md`
- `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/same_input_contract_builder.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/nesting/boundary_construction.py`
- `src/gpuwrf/runtime/operational_state.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/contracts/state.py`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_jax_loader_tstate/**`

## Write Scope

Required repo files:

- `proofs/v014/step1_jax_loader_tstate.py`
- `proofs/v014/step1_jax_loader_tstate.json`
- `proofs/v014/step1_jax_loader_tstate.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md`

Optional repo files:

- targeted production source edits only if an exact, narrow,
  performance-compatible loader bug is proven:
  - `src/gpuwrf/integration/d02_replay.py`
  - `src/gpuwrf/nesting/boundary_construction.py`
  - `src/gpuwrf/runtime/operational_state.py`
  - `src/gpuwrf/runtime/operational_mode.py`
  - `src/gpuwrf/contracts/state.py`

Do not touch unrelated source, TOST outputs, Switzerland outputs, FP32 work,
memory source work, or old untracked artifacts.

## Required Work

1. Verify branch/head and that `99df65e0` is an ancestor.
2. Load/parse the accepted WRF pre-call truth from
   `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`.
3. Reconstruct the JAX Step-1 live-nest inputs in named stages without using the
   GPU.
4. Compare each stage to the WRF pre-call truth with full-domain and
   interior-vs-boundary-band metrics.
5. Explicitly report theta semantics:
   `WRF T_STATE` vs `State.theta - 300 K`, and the wrong full-theta comparison.
6. Classify the first material stage and field.
7. If a source fix is made, rerun:
   - this sprint proof;
   - `proofs/v014/step1_pre_part1_handoff.py`;
   - `proofs/v014/step1_part1_physics_state_mutation.py`;
   - `proofs/v014/step1_rk1_source_boundary.py`;
   - `proofs/v014/step1_t_p_operator_localization.py`;
   - `proofs/v014/step1_live_nest_init_rerun.py`;
   and report before/after top residuals.

## Verdicts

Emit exactly one final verdict:

- `STEP1_JAX_LOADER_TSTATE_LOCALIZED_RAW_WRFINPUT`
- `STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`
- `STEP1_JAX_LOADER_TSTATE_LOCALIZED_BOUNDARY_PACKAGE`
- `STEP1_JAX_LOADER_TSTATE_LOCALIZED_CARRY_CONSTRUCTION`
- `STEP1_JAX_LOADER_TSTATE_LOCALIZED_HALO_ENTRY`
- `STEP1_JAX_LOADER_TSTATE_FIXED_<stage_or_leaf>`
- `STEP1_JAX_LOADER_TSTATE_BLOCKED_<specific_missing_truth_or_contract>`
- `STEP1_JAX_LOADER_TSTATE_NO_REMAINING_DIVERGENCE`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_jax_loader_tstate.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_jax_loader_tstate.py
python -m json.tool proofs/v014/step1_jax_loader_tstate.json \
  >/tmp/step1_jax_loader_tstate.validated.json
git diff -- src/gpuwrf
```

If production source is edited, also run the proof chain listed in Required
Work item 7 and compile the edited Python modules.

## Acceptance Criteria

- The proof is CPU-only and records `gpu_used=false`.
- The proof compares WRF pre-call truth against the named JAX loader stages.
- The proof identifies the first stage where `T_STATE` becomes materially
  different, or applies a narrow fix and proves the residual closes.
- Detailed field tables live in JSON; markdown is concise.
- No weak station, one-cell, JAX-vs-JAX-only, or long-run validation conclusion
  is used.
