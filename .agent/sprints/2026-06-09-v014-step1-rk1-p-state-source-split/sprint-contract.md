# Sprint Contract: V0.14 Step-1 RK1 P-State Source Split

Date: 2026-06-09 23:05 WEST
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`
Base commit: `7cad8577`

## Objective

Localize or fix the remaining Step-1 RK1 stage-entry `P_STATE` material
divergence after the Mythos live-nest/start-domain init fix.

The accepted fresh comparator says:

- verdict:
  `STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_P_STATE`;
- first strict substage mismatch:
  `after_rk_addtend_before_small_step_prep`, RK1, `T_STATE`;
- first material T/P-family mismatch:
  `after_rk_addtend_before_small_step_prep`, RK1, `P_STATE`;
- top material residuals at the same boundary are tendency-family:
  `PH_TEND max_abs=794096.1875`, `RW_TEND max_abs=131390.765625`,
  `PH_TENDF max_abs=27082.453125`;
- RK1 `small_step_prep` then has `T_WORK=0.0` and `P_WORK=0.0`, so do not jump
  into acoustic substeps until this earlier boundary is closed.

## Method Rule

Use the fastest rigorous wall-clock path. Prefer focused proof tooling and
small same-boundary comparisons over slow full-run chasing.

Do not merely answer the manager's current hypothesis. If the `P_STATE` source
hypothesis is wrong, rank the next likely causes, run cheap falsifiers where
possible, and return excluded hypotheses plus the best next exact boundary.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No GPU.
- No memory/FP32 source work; Mythos owns that in tmux `0:1`.
- No Hermes/Telegram.
- No broad dycore rewrite.
- No acoustic-substep continuation until RK1 stage-entry/source boundary is
  explained.

## Inputs

- `proofs/v014/mythos_kernel_fix_260609.{py,json,md}`
- `proofs/v014/step1_t_p_operator_localization.{py,json,md}`
- `proofs/v014/step1_rk1_source_boundary.{py,json,md}`
- `proofs/v014/step1_part1_physics_state_mutation.{py,json,md}`
- `proofs/v014/step1_pre_part1_handoff.{py,json,md}`
- WRF substage truth root:
  `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth`
- Accepted Step-1 final truth:
  `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`
- Source likely needed for reading/proof:
  `src/gpuwrf/runtime/operational_mode.py`,
  `src/gpuwrf/integration/d02_replay.py`,
  `src/gpuwrf/dynamics/**`, and WRF `dyn_em/solve_em.F`,
  `dyn_em/module_first_rk_step_part1.F`,
  `dyn_em/module_first_rk_step_part2.F`, `dyn_em/module_big_step_utilities_em.F`.

## File Ownership

Required proof artifacts:

- `proofs/v014/step1_rk1_p_state_source_split.py`
- `proofs/v014/step1_rk1_p_state_source_split.json`
- `proofs/v014/step1_rk1_p_state_source_split.md`
- `.agent/reviews/2026-06-09-v014-step1-rk1-p-state-source-split.md`

Optional disposable WRF patch diff:

- `proofs/v014/step1_rk1_p_state_source_split_wrf_patch.diff`

Optional production source edit only if exact and narrow:

- `src/gpuwrf/runtime/operational_mode.py`
- or another single proven source file, with before/after proof.

Do not edit Mythos' worktree or memory/FP32 files.

## Required Work

1. Verify current branch/head and record them.
2. Reuse or extend the existing substage truth/comparator so the proof starts
   from the current post-Mythos init state.
3. Split the RK1 stage-entry `P_STATE` residual into the smallest available
   source families:
   - state entering WRF `first_rk_step_part1/part2`;
   - WRF `first_rk_step_part1/part2` direct state mutation;
   - `rk_addtend_dry` / `spec_bdy_dry`;
   - JAX `_physics_step_forcing`, `_augment_large_step_tendencies`, and
     carry/source leaves;
   - pressure/total-vs-perturb mapping;
   - boundary relaxation at the same stage boundary.
4. Include the huge `PH_TEND/RW_TEND/PH_TENDF` residuals in the attribution.
   Decide whether they are causal, stale tendency/carry values, different
   source ordering, boundary forcing, or a downstream consequence.
5. If a narrow production fix is proven, implement it and rerun before/after
   proof. Otherwise leave production source untouched and return the next exact
   source boundary.

## Verdicts

Emit exactly one:

- `STEP1_RK1_P_STATE_SOURCE_FIXED_<source>`
- `STEP1_RK1_P_STATE_SOURCE_LOCALIZED_<source>`
- `STEP1_RK1_P_STATE_SOURCE_BLOCKED_<exact_missing_truth_or_contract>`
- `STEP1_RK1_P_STATE_SOURCE_REFUTED_<next_best_hypothesis>`

## Validation

Minimum:

```bash
python -m py_compile proofs/v014/step1_rk1_p_state_source_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_rk1_p_state_source_split.py
python -m json.tool proofs/v014/step1_rk1_p_state_source_split.json \
  >/tmp/step1_rk1_p_state_source_split.validated.json
git diff --check
```

If production source changes, also run the fresh
`proofs/v014/step1_t_p_operator_localization.py` comparator and the focused
module tests for the touched source file.

## Acceptance Criteria

- JSON validates.
- No GPU was used.
- Report names the first still-failing boundary or the exact source fix.
- If no source fix is made, the next sprint can start without rediscovering the
  same boundary.

## Completion Signal

```bash
tmux send-keys -t 0:2 'GPT STEP1_RK1_P_STATE_SOURCE_SPLIT DONE - see proofs/v014/step1_rk1_p_state_source_split.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
