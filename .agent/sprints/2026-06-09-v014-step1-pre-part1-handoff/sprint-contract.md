# Sprint Contract: V0.14 Step-1 Pre-Part1 Handoff

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Move one boundary upstream from WRF `first_rk_step_part1` and classify why
`T_STATE` already diverges at part1 entry.

Trigger evidence:

- `proofs/v014/step1_part1_physics_state_mutation.json`
- verdict `STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE`
- `part1_entry_before_init_zero_tendency` `T_STATE` vs JAX live-nest
  step-entry state: max_abs `5.490173101425171`, RMSE
  `1.9175184863907806`
- WRF internal `first_rk_step_part1` `T_STATE` delta from entry: max_abs `0.0`

The sprint must determine whether the source is:

- WRF/JAX field semantic mapping error at the call boundary
  (`grid%t_2` perturbation theta vs JAX full/perturbation theta, offset, dtype,
  halo, or transpose);
- JAX live-nest Step-1 loader/carry state already differs from WRF call-site
  state;
- WRF mutates `grid%t_2` before the call through physical boundary setup,
  halo/DM exchange, parent/nest feedback, or solve_em pre-part1 state assembly;
- the previous accepted Step-1 loader proof compared the wrong surface;
- or a narrowly provable production bug.

## Method Rule

Use the fastest rigorous wall-clock method: extend the Step-1 savepoint and
JAX-capture comparator with WRF solve_em call-site surfaces immediately before
`first_rk_step_part1`, plus the smallest upstream surfaces needed to determine
whether WRF or JAX moved.

Accepted WRF surfaces include:

- `solve_em_entry_or_post_step_increment` if cheap and unambiguous;
- after dry/state physical boundary setup for `grid%t_2`, `grid%p`, `grid%mu_2`,
  `grid%ph_2`;
- immediately before `CALL first_rk_step_part1`;
- optional re-use of `part1_entry_before_init_zero_tendency` for continuity.

Accepted JAX surfaces include:

- raw `build_live_nest_step1_inputs()` state/carry before `_physics_step_forcing`;
- the prior `capture_jax_boundaries()` `step_entry_state_zero_dry`;
- explicit full-theta and perturbation-theta views, with the 300 K offset named;
- any halo/interior-crop variants needed to distinguish mapping from state
  divergence.

The proof must explicitly report whether WRF `T_STATE` is being compared to
JAX full theta or perturbation theta and why that mapping is correct.

Forbidden comparisons:

- no WRF final truth vs JAX initial state;
- no JAX-vs-JAX-only conclusion;
- no one-cell/station proxy;
- no acoustic, TOST, Switzerland, FP32, or memory work;
- no production source fix unless an exact performance-compatible bug is proven.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision source work.
- No memory source work.
- No GPU.
- No Hermes or Telegram.
- No broad dycore rewrite or performance-regressing source change.

## Inputs

- `proofs/v014/step1_part1_physics_state_mutation.py`
- `proofs/v014/step1_part1_physics_state_mutation.json`
- `proofs/v014/step1_part1_physics_state_mutation_wrf_patch.diff`
- `proofs/v014/step1_rk1_source_boundary.py`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_same_input_truth.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/contracts/state.py`
- `/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/**`
- `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/**`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/**`

## Write Scope

Required repo files:

- `proofs/v014/step1_pre_part1_handoff.py`
- `proofs/v014/step1_pre_part1_handoff.json`
- `proofs/v014/step1_pre_part1_handoff.md`
- `.agent/reviews/2026-06-09-v014-step1-pre-part1-handoff.md`

Optional repo files:

- `proofs/v014/step1_pre_part1_handoff_wrf_patch.diff`
- targeted source edits only if an exact, narrow, performance-compatible bug is
  proven:
  - `src/gpuwrf/runtime/operational_mode.py`
  - `src/gpuwrf/contracts/state.py`
  - specific loader/coupling files under `src/gpuwrf/**`

Do not touch unrelated source, TOST outputs, Switzerland outputs, FP32 work,
memory source work, or old untracked artifacts.

## Required Work

1. Verify branch/head and that `588686d6` is an ancestor.
2. Reuse parser/comparator conventions from the prior Step-1 proofs where
   practical.
3. Emit or consume WRF full d02 pre-part1 call-site truth with enough fields to
   compare:
   - `T_STATE`, `P_STATE`, `PB`, `MU_STATE`, `MUB`, `MUT`;
   - `PH_STATE`, `PHB`, `W_STATE` if needed for state assembly;
   - tile bounds and owned-cell policy.
4. Capture matching JAX live-nest Step-1 surfaces:
   - raw state/carry;
   - zero-dry-tendency state if used by prior proof;
   - perturbation theta view (`theta - 300`) and full theta view.
5. Classify the first material mismatch. The result must say whether the issue
   is field mapping, JAX loader/carry construction, WRF pre-part1 state mutation,
   wrong prior comparison surface, or one exact missing truth/capture blocker.
6. If a source fix is made, rerun:
   - this sprint proof;
   - `proofs/v014/step1_part1_physics_state_mutation.py`;
   - `proofs/v014/step1_rk1_source_boundary.py`;
   - `proofs/v014/step1_t_p_operator_localization.py`;
   - `proofs/v014/step1_live_nest_init_rerun.py`;
   and report before/after top residuals.

## Verdicts

Emit exactly one final verdict:

- `STEP1_PRE_PART1_LOCALIZED_FIELD_MAPPING_<field>`
- `STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_<field>`
- `STEP1_PRE_PART1_LOCALIZED_WRF_PRECALL_MUTATION_<field>`
- `STEP1_PRE_PART1_LOCALIZED_WRONG_PRIOR_SURFACE_<field>`
- `STEP1_PRE_PART1_FIXED_<field_or_leaf>`
- `STEP1_PRE_PART1_BLOCKED_<specific_missing_truth_or_contract>`
- `STEP1_PRE_PART1_NO_REMAINING_DIVERGENCE`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_pre_part1_handoff.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_pre_part1_handoff.py
python -m json.tool proofs/v014/step1_pre_part1_handoff.json \
  >/tmp/step1_pre_part1_handoff.validated.json
git diff -- src/gpuwrf
```

If production source changes:

```bash
python -m py_compile src/gpuwrf/runtime/operational_mode.py src/gpuwrf/contracts/state.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_part1_physics_state_mutation.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_rk1_source_boundary.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_init_rerun.py
```

## Acceptance Criteria

- JSON validates and records CPU-only execution.
- The proof names the exact WRF/JAX pre-part1 boundary and field/mapping for
  the first material mismatch or exact blocker.
- The proof explicitly validates full-vs-perturbation theta semantics.
- Any source fix is narrow and performance-compatible: no host/device transfer
  inside timestep loops, no CPU-WRF wrapper, no broad de-optimization.
- Production `src/gpuwrf/**` remains unchanged unless a concrete bug is proven.
- Review report includes objective, files changed, commands run, proof objects,
  unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_PRE_PART1_HANDOFF DONE - see proofs/v014/step1_pre_part1_handoff.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
