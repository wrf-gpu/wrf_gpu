# Sprint Contract: V0.14 Same-Input Contract Builder

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Build the missing same-input comparison contract/tooling exposed by
`proofs/v014/early_step_discriminator.json`, then rerun a strict early-step
candidate comparison if technically possible.

This is a tooling sprint, not a dynamics source-fix sprint. The goal is to make
future grid-parity debugging fast, reproducible, and falsifiable.

## Trigger Evidence

- `proofs/v014/early_step_discriminator.json`
- `proofs/v014/early_step_discriminator.md`
- `.agent/reviews/2026-06-09-v014-early-step-discriminator.md`
- `.agent/skills/managing-sprints/SKILL.md` debug-tooling check

## Non-Goals

- No production `src/gpuwrf/**` edits.
- No production dycore/runtime/physics fix.
- No TOST.
- No Switzerland validation.
- No FP32 or memory source work.
- No Hermes or Telegram.
- No weak comparison, JAX-vs-JAX self-compare, one-cell proof, or mixed
  JAX-produced carry with WRF truth leaves.

## Inputs

- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/wrfinput_d02`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/namelist.input`
- `/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/rsl.error.0000`
- prior WRF hook diffs under `proofs/v014/*wrf_patch.diff`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/runtime/operational_state.py`
- `src/gpuwrf/runtime/operational_mode.py`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/**`

## Write Scope

Repository files:

- `proofs/v014/same_input_contract_builder.py`
- `proofs/v014/same_input_contract_builder.json`
- `proofs/v014/same_input_contract_builder.md`
- optional `proofs/v014/same_input_contract_builder_wrf_patch.diff`
- optional updates to `proofs/v014/early_step_discriminator.py`
- optional regenerated `proofs/v014/early_step_discriminator.json`
- optional regenerated `proofs/v014/early_step_discriminator.md`
- `.agent/reviews/2026-06-09-v014-same-input-contract-builder.md`

Do not touch:

- production `src/gpuwrf/**`
- unrelated untracked artifacts
- TOST outputs

## Required Work

1. Build or precisely block a CPU-compatible proof-local loader/checkpoint reader
   that constructs the real d02 `State`, `Tendencies`, `BaseState`/metrics,
   `OperationalNamelist`, and initial `OperationalCarry` without calling
   GPU-only `State.zeros`.
2. Define a frozen WRF/JAX field map for `T`, `P`, `PB`, `PH`, `PHB`, `MU`,
   `MUB`, `U`, `V`, `W`, and active moisture:
   - units
   - staggering
   - shape/count
   - WRF variable/source
   - JAX leaf/source
   - first/worst index semantics
   - which static/base fields are excluded from headline dynamic selectors
3. Produce a disposable CPU-WRF hook patch or an exact blocker for candidate
   post-RK/pre-halo full-field truth at step `1` first. Steps `60`, `600`,
   `3000`, and `5999` may be added only after step `1` is cheap.
4. If the loader and WRF surface are both available, rerun or extend
   `proofs/v014/early_step_discriminator.py` to execute the first strict
   comparison.
5. If strict execution still cannot run, emit an implementation-ready blocker:
   exact missing field, exact WRF source location, exact JAX loader/source
   location, and the smallest next patch/tool required.

## Required Metrics

If a strict comparison executes, emit per-field:

- count
- max_abs
- RMSE
- bias
- p95
- p99
- first mismatch index
- worst mismatch index

## Verdicts

Emit exactly one final manager-facing verdict:

- `SAME_INPUT_CONTRACT_READY_STEP_<step>`
- `SAME_INPUT_CONTRACT_EXECUTED_FIRST_DIVERGENT_STEP_<step>_<field_or_operator>`
- `SAME_INPUT_CONTRACT_EXECUTED_CLEAN_THROUGH_<step>`
- `SAME_INPUT_CONTRACT_BLOCKED_<specific_blockers>`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/same_input_contract_builder.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/same_input_contract_builder.py
python -m json.tool proofs/v014/same_input_contract_builder.json \
  >/tmp/same_input_contract_builder.validated.json
git diff -- src/gpuwrf
```

If `early_step_discriminator.py` is updated or rerun:

```bash
python -m py_compile proofs/v014/early_step_discriminator.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/early_step_discriminator.py
python -m json.tool proofs/v014/early_step_discriminator.json \
  >/tmp/early_step_discriminator.validated.json
```

If a disposable WRF hook/build/run is needed, record exact commands, patch path,
run directory, and log path in JSON. CPU-WRF is allowed inside scratch.

## Acceptance Criteria

- CPU/JAX proof path runs with `JAX_PLATFORMS=cpu` and no visible GPU.
- JSON validates.
- `git diff -- src/gpuwrf` is empty.
- Either at least one strict same-input comparison executes, or the proof emits
  exact implementation-ready blockers for the loader, WRF surface, and field
  schema.
- Review report includes objective, files changed, commands run, proof objects,
  unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT SAME_INPUT_CONTRACT_BUILDER DONE - see proofs/v014/same_input_contract_builder.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```

If tmux socket access is blocked, still write all artifacts and leave the DONE
marker visible in the worker TUI.
