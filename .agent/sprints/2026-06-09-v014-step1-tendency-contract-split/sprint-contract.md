# Sprint Contract: V0.14 Step-1 Tendency Contract Split

Date: 2026-06-09 23:35 WEST
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`
Base commit: `d4e83f2a`

## Objective

Localize or fix the remaining Step-1 tendency-family divergence after the
Mythos init fix and the stale RK1 `P_STATE` proof-loader issue were closed.

Current accepted state:

- `P_STATE` is below material gate under patched-init capture:
  `0.0390625 Pa <= 1.0 Pa`.
- The next exact boundary from
  `proofs/v014/step1_rk1_p_state_source_split.md` is:
  split WRF `first_rk_step_part2` `T_TENDF`, then RK1
  `after_rk_addtend` `T_TEND/PH_TEND/RW_TEND`, against JAX
  `compute_advection_tendencies` and `_augment_large_step_tendencies`.
- Do not enter acoustic substeps until this earlier tendency boundary is
  explained.

## Method Rule

Use the fastest rigorous wall-clock method. Prefer focused same-boundary
comparators, WRF truth-surface parsing, and cheap proof-local falsifiers.

Do not only execute the manager's hypothesis. If the `T_TENDF/T_TEND` lane is
not causal, rank alternate causes, run cheap falsifiers if possible, and return
excluded hypotheses plus the next exact boundary.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No GPU.
- No memory/FP32 source work; Mythos owns that in tmux `0:1`.
- No Hermes/Telegram.
- No broad dycore rewrite.
- No acoustic-substep debugging until the tendency boundary is closed.

## Inputs

- `proofs/v014/step1_rk1_p_state_source_split.{py,json,md}`
- `proofs/v014/step1_t_p_operator_localization.{py,json,md}`
- `proofs/v014/step1_rk1_source_boundary.{py,json,md}`
- `proofs/v014/step1_part1_physics_state_mutation.{py,json,md}`
- `proofs/v014/step1_pre_part1_handoff.{py,json,md}`
- WRF substage truth root:
  `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth`
- Source likely needed:
  `src/gpuwrf/runtime/operational_mode.py`,
  `src/gpuwrf/dynamics/**`,
  WRF `dyn_em/module_first_rk_step_part1.F`,
  `dyn_em/module_first_rk_step_part2.F`,
  `dyn_em/module_big_step_utilities_em.F`, and `dyn_em/solve_em.F`.

## File Ownership

Required artifacts:

- `proofs/v014/step1_tendency_contract_split.py`
- `proofs/v014/step1_tendency_contract_split.json`
- `proofs/v014/step1_tendency_contract_split.md`
- `.agent/reviews/2026-06-09-v014-step1-tendency-contract-split.md`

Optional disposable WRF patch diff:

- `proofs/v014/step1_tendency_contract_split_wrf_patch.diff`

Optional production source edit only if exact and narrow:

- `src/gpuwrf/runtime/operational_mode.py`
- or a single proven tendency/source file.

Do not edit Mythos' memory worktree or memory/FP32 files.

## Required Work

1. Verify branch/head and record them.
2. Build or reuse a patched-init capture path so stale live-nest proof helpers
   cannot reintroduce the old init residual.
3. Split WRF/JAX tendency construction for Step-1 RK1:
   - WRF `first_rk_step_part2` `T_TENDF`;
   - WRF/JAX RK1 `T_TEND`;
   - WRF/JAX RK1 `PH_TEND`, `RW_TEND`, `PH_TENDF`, `RW_TENDF`;
   - JAX raw `compute_advection_tendencies`;
   - JAX `_augment_large_step_tendencies`;
   - physics dry source leaves, proving whether empty-dry/full-dry equivalence
     is causal or not.
4. Decide whether the remaining tendency residual is:
   - stale/incorrect proof carry;
   - WRF source-order mismatch;
   - boundary forcing/relaxation;
   - old-field/history input mismatch;
   - JAX advection operator mismatch;
   - physics tendency injection mismatch;
   - or a still-missing WRF truth boundary.
5. If a narrow production fix is proven, implement it and rerun before/after
   proof. Otherwise leave source untouched and name the next exact source
   boundary.

## Verdicts

Emit exactly one:

- `STEP1_TENDENCY_CONTRACT_FIXED_<source>`
- `STEP1_TENDENCY_CONTRACT_LOCALIZED_<source>`
- `STEP1_TENDENCY_CONTRACT_BLOCKED_<exact_missing_truth_or_contract>`
- `STEP1_TENDENCY_CONTRACT_REFUTED_<next_best_hypothesis>`

## Validation

Minimum:

```bash
python -m py_compile proofs/v014/step1_tendency_contract_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_tendency_contract_split.py
python -m json.tool proofs/v014/step1_tendency_contract_split.json \
  >/tmp/step1_tendency_contract_split.validated.json
git diff --check
```

If production source changes, rerun:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_t_p_operator_localization.py
```

## Acceptance Criteria

- JSON validates.
- No GPU was used.
- Report ranks hypotheses and exclusions.
- Report names the first still-failing boundary or exact source fix.
- If no source fix is made, the next sprint can start without rediscovery.

## Completion Signal

```bash
tmux send-keys -t 0:2 'GPT STEP1_TENDENCY_CONTRACT_SPLIT DONE - see proofs/v014/step1_tendency_contract_split.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
