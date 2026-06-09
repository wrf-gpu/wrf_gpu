# Sprint Contract: V0.14 Step-1 Base-State Boundary

Date: 2026-06-09 20:40 WEST
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`
Base commit: pending manager commit after input-split closeout

## Objective

Emit or reproduce the exact WRF `start_domain_em` base-state boundary before
the hypsometric `AL/ALT` pass for the live-nest Step-1 d02 case.

The predecessor sprint proved:

- current pressure formula vs WRF P max_abs `3.9458582235092763`;
- replacing diagnosed ALT with WRF ALT reduces pressure max_abs to
  `0.07605321895971429`;
- fp32 ALT diagnosis with WRF `PHB+MUB` reduces pressure max_abs to `0.0859375`;
- best local WRF-order fp32/cp=1004.5 base candidate still leaves
  `P_STATE` max_abs `2.828125` and `MU_STATE` max_abs `0.011962890625`.

Therefore the remaining unknown is exact WRF base-state reconstruction/source
order feeding `AL/ALT`, especially `PHB+MUB`, not broad dycore or acoustic.

## Method Rule

Use the fastest rigorous wall-clock path: disposable WRF instrumentation or
source-reproduction proof around `start_domain_em`, then CPU/JAX/NumPy
comparison against current JAX live-nest inputs and the predecessor truth
surfaces.

Do not make a production `src/gpuwrf/**` edit in this sprint. If a source fix is
obvious, write the next patch contract with before/after metrics; do not patch
from an incomplete base-state truth surface.

If the manager hypothesis is wrong, do not stop at "blocked": rank alternate
causes, try cheap proof-local falsifiers, and name the next exact truth surface
or source line.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No GPU run.
- No FP32 production source work.
- No memory production source work.
- No Hermes/Telegram.
- No broad dycore rewrite.
- No production CPU-WRF runtime dependency.
- No timestep-loop host/device transfer.

## Inputs

- `proofs/v014/step1_jax_start_domain_input_split.py`
- `proofs/v014/step1_jax_start_domain_input_split.json`
- `proofs/v014/step1_jax_start_domain_input_split.md`
- `proofs/v014/step1_start_domain_perturb_subsurface.py`
- `proofs/v014/step1_start_domain_perturb_subsurface.json`
- `src/gpuwrf/integration/d02_replay.py`
- WRF source, especially `dyn_em/start_em.F` and `share/module_model_constants.F`
- Existing WRF truth root:
  `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715/wrf_truth`

## File Ownership

Required repo files:

- `proofs/v014/step1_base_state_boundary.py`
- `proofs/v014/step1_base_state_boundary.json`
- `proofs/v014/step1_base_state_boundary.md`
- `.agent/reviews/2026-06-09-v014-step1-base-state-boundary.md`

Optional proof-only WRF patch diff:

- `proofs/v014/step1_base_state_boundary_wrf_patch.diff`

Do not edit production `src/gpuwrf/**` files in this sprint.

## Required Work

1. Verify branch/head and record the manager base commit once launched.
2. Locate the exact WRF source boundary before hypsometric `P/al/alt`
   computation in `start_domain_em`.
3. Emit or reproduce the following WRF values at that boundary for d02 Step-1:
   - `p_surf` / surface pressure expression input and output;
   - post-assignment `MUB`;
   - `PB`;
   - `T_INIT` or the equivalent base-state temperature/potential-temperature
     intermediate;
   - `ALB`;
   - `PHB` after base geopotential integration;
   - active `C3F/C4F/C3H/C4H` coefficients as used in memory;
   - active scalar constants (`r_d`, `cp`, `cvpm`, `p1000mb`, `t0`, `g`,
     `p_top`, `p00`, `t00`, lapse/isothermal/stratosphere constants);
   - relevant flags or masks affecting this branch.
4. Compare WRF boundary values against:
   - current production JAX live-nest inputs;
   - proof-local fp64/cp=1004.0 base recompute;
   - proof-local fp64/cp=1004.5 base recompute;
   - proof-local fp32/cp=1004.0 base recompute;
   - proof-local fp32/cp=1004.5 base recompute.
5. Split residuals by field and source-order family. The report must state
   whether the remaining mismatch is caused by constants, dtype/evaluation
   order, coefficient indexing, terrain/blend input, pressure-surface formula,
   PHB integration order, or a still-missing WRF truth surface.
6. If exact production source semantics are now proven, produce a next-sprint
   patch plan with the exact source lines, expected before/after metrics, and
   validation gates. Do not patch in this sprint.

## Verdicts

Emit exactly one final verdict:

- `STEP1_BASE_STATE_BOUNDARY_READY_FOR_D02_REPLAY_PATCH_<source>`
- `STEP1_BASE_STATE_BOUNDARY_LOCALIZED_<source_or_missing_contract>`
- `STEP1_BASE_STATE_BOUNDARY_BLOCKED_<specific_truth_or_source_gap>`
- `STEP1_BASE_STATE_BOUNDARY_REFUTED_<hypothesis_and_next_best>`

## Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_base_state_boundary.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_base_state_boundary.py
python -m json.tool proofs/v014/step1_base_state_boundary.json \
  >/tmp/step1_base_state_boundary.validated.json
git diff -- src/gpuwrf
```

If disposable WRF instrumentation is used, also record the WRF patch diff and
the exact compile/run commands in the JSON/MD/review. Scratch WRF output may
live under `/mnt/data/wrf_gpu2/`.

## Acceptance Criteria

- JSON validates and records CPU-only execution.
- No production `src/gpuwrf/**` diff.
- Report includes a ranked residual split by base-state field/source family.
- If no patch contract is ready, the next truth surface/source boundary is exact
  enough for a worker to start without rediscovery.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_BASE_STATE_BOUNDARY DONE - see proofs/v014/step1_base_state_boundary.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
