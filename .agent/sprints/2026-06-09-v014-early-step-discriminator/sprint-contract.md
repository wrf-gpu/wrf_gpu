# Sprint Contract: V0.14 Early-Step Same-Input Discriminator

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Run the consolidated early-step same-input discriminator recommended by the
management review. Start from shared `wrfinput` where instrumentation is cheap,
execute at least one strict same-input comparison, and locate the first
divergence window. If strict execution still cannot run, name all blockers in one
proof pass.

This replaces the h10 / step-6000 wrapper ladder. Do not open another
one-blocker micro-sprint.

## Trigger Evidence

- `.agent/reviews/2026-06-09-v014-management-review-01.md`
- `.agent/decisions/V0140-EARLY-STEP-DISCRIMINATOR-PLAN.md`
- `proofs/v014/same_input_single_rk_parity_wrapped.json`
- `proofs/v014/same_input_single_rk_parity_wrapped.md`
- `.agent/sprints/2026-06-09-v014-full-domain-source-wrapper/manager-closeout.md`

## Non-Goals

- No production `src/gpuwrf/**` edits.
- No GPU unless manager explicitly reauthorizes inside a follow-up contract.
- No TOST.
- No Switzerland validation.
- No FP32 or memory source work.
- No Hermes or Telegram.
- No source fix. This is a discriminator/proof sprint.

## Inputs

Use existing Canary L2 d02 inputs and current replay loaders:

- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/wrfinput_d02`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/namelist.input`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/wrfbdy_d01`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/contracts/state.py`
- `scripts/compare_wrfout_grid.py`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_early_step_discriminator/**`

## Write Scope

Repository files:

- `proofs/v014/early_step_discriminator.py`
- `proofs/v014/early_step_discriminator.json`
- `proofs/v014/early_step_discriminator.md`
- optional `proofs/v014/early_step_discriminator.csv`
- optional `proofs/v014/early_step_discriminator_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-early-step-discriminator.md`

Do not touch:

- production `src/gpuwrf/**`
- unrelated untracked artifacts
- TOST outputs

## Required Work

1. Inventory the cheapest strict same-input route from shared `wrfinput_d02`.
   Prefer existing replay loaders and proof helpers over new WRF hooks.
2. Candidate step sequence:
   - `1`
   - `60`
   - `600`
   - `3000`
   - `5999`
3. Execute at least one strict same-input comparison if technically possible.
   Strict means:
   - WRF and JAX start from the same state for that step;
   - tendencies are either computed from that same state by both paths or
     controlled from WRF output without mixing JAX-produced carry;
   - comparison is against matching WRF post-RK/pre-halo or an exactly named
     equivalent boundary;
   - no JAX-vs-JAX self-compare and no one-cell proof.
4. If the first compared step is clean, bisect forward until a divergent window
   is found or the candidate list is exhausted.
5. If the first compared step diverges, report the first divergent field and
   operator/boundary hypothesis; do not start a source edit.
6. If strict execution cannot run, emit one consolidated blocker covering all
   candidate steps and all missing fields/contracts. Do not emit a one-blocker
   ladder result.
7. Headline selectors must use dynamic/perturbation fields. Exclude static
   writer/base artifacts from dominant-field decisions.

## Required Metrics

For each strict comparison that executes, emit at least:

- field
- count
- max_abs
- RMSE
- bias
- p95
- p99
- first/worst index

Minimum fields: `T`, `P`, `PB`, `PH`, `PHB`, `MU`, `MUB`, `U`, `V`, `W`,
and active moisture if included by the chosen entry point.

## Verdicts

Emit exactly one final manager-facing verdict:

- `EARLY_STEP_DYNAMICS_CLEAN_THROUGH_<step>`
- `FIRST_DIVERGENT_STEP_<step>_<field_or_operator>`
- `EARLY_STEP_DISCRIMINATOR_BLOCKED_<all_blockers_named>`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/early_step_discriminator.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/early_step_discriminator.py
python -m json.tool proofs/v014/early_step_discriminator.json \
  >/tmp/early_step_discriminator.validated.json
git diff -- src/gpuwrf
```

If a disposable WRF hook/build/run is needed, record exact commands and log paths
in JSON. CPU-only WRF is allowed inside scratch.

## Acceptance Criteria

- CPU-only and no GPU use.
- JSON validates.
- `git diff -- src/gpuwrf` is empty.
- At least one strict same-input comparison executes, or one proof names all
  blockers across the early-step sequence.
- No weak comparison is emitted.
- Review report includes objective, files changed, commands run, proof objects,
  unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT EARLY_STEP_DISCRIMINATOR DONE - see proofs/v014/early_step_discriminator.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```

If tmux socket access is blocked, still write all artifacts and leave the DONE
marker visible in the worker TUI.
