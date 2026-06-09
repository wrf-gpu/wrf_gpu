# Sprint Contract: V0.14 Step-1 Live-Nest Perturbation-State Init

Date: 2026-06-09 19:15 WEST
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`
Base commit: `131b27cd`

## Objective

Close or precisely localize the live-nest `raw_child_state -> live_child_state`
perturbation-state mismatch for `P_STATE/MU_STATE/W_STATE`.

Current accepted predecessor:

- Commit `131b27cd`: `v014 close first rk part1 p state split`.
- Verdict:
  `STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`.
- WRF `before_first_rk_step_part1_call -> after_first_rk_step_part1` is exact
  for `P_STATE/MU_STATE/W_STATE/PH_STATE`.
- JAX `raw_child_state`, `live_child_state`, boundary package, initial carry,
  haloed entry, and `_physics_step_forcing.carry.state` all preserve the same
  residuals versus WRF pre-call:
  - `P_STATE=69.96875`
  - `MU_STATE=13.256103515625`
  - `W_STATE=0.7605466246604919`
  - `PH_STATE=0.00048828125`

The expected first hypothesis is: live-nest base/theta/QV correction now fixes
`PB/MUB/PHB/T/QV`, but does not apply WRF-equivalent live-nest perturbation
state initialization for `P/MU/W`. This hypothesis may be wrong; the worker must
try to disprove it and report alternative likely causes with evidence.

## Method Rule

Use the fastest rigorous wall-clock path. Prefer CPU-only savepoint comparison,
WRF formula transcription, and small proof-local experiments over full forecasts.

The worker is expected to act like a senior debugger, not a narrow log parser:

- Start with the manager hypothesis, but if evidence contradicts it, broaden
  within this boundary and identify the most likely alternate causes.
- Return both proven exclusions and ranked hypotheses.
- If a likely formula/source bug is proven, attempt a narrow GPU-native fix and
  rerun the strict Step-1 proof.
- If the exact formula is not yet recoverable, produce the smallest next truth
  surface or source location needed; do not guess.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision source work.
- No memory source work.
- No long GPU forecast; prefer CPU-only.
- No Hermes or Telegram.
- No broad dycore rewrite, CPU-WRF runtime dependency, timestep-loop
  host/device transfer, or performance-regressing fix.

## Inputs

- `proofs/v014/step1_first_rk_part1_p_state_split.py`
- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- `proofs/v014/step1_first_rk_part1_p_state_split.md`
- `proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `proofs/v014/step1_live_nest_theta_qv_wiring.json`
- `proofs/v014/step1_current_mub_base_input_split.py`
- `proofs/v014/step1_current_mub_base_input_split.json`
- `proofs/v014/step1_transient_adjust_base_fix.md`
- `proofs/v014/step1_pre_part1_handoff.py`
- `proofs/v014/step1_pre_part1_handoff.json`
- `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth/**`
- `/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/wrf_truth/**`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_live_nest_perturb_state_init/**`

## File Ownership

Required repo files:

- `proofs/v014/step1_live_nest_perturb_state_init.py`
- `proofs/v014/step1_live_nest_perturb_state_init.json`
- `proofs/v014/step1_live_nest_perturb_state_init.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-perturb-state-init.md`

Optional repo files:

- `proofs/v014/step1_live_nest_perturb_state_init_source_patch.diff`
- `proofs/v014/step1_live_nest_perturb_state_init_wrf_patch.diff`

Optional production source edits only after exact proof and only if narrow:

- `src/gpuwrf/integration/d02_replay.py`
- a directly relevant live-nest/init helper if the repo already has one

Do not edit `src/gpuwrf/dynamics/**`, FP32 files, memory files, TOST outputs,
Switzerland outputs, or unrelated dirty/untracked artifacts.

## Required Work

1. Verify `131b27cd` is an ancestor and record branch/head.
2. Reuse predecessor baseline and WRF pre-call truth.
3. Identify WRF semantics for live-nest `P/MU/W` perturbation state at
   `before_first_rk_step_part1_call`. Sources to inspect include WRF
   `med_nest_initial`, `start_domain_em`, pressure/base recomputation, and any
   current JAX live-nest code.
4. Build a proof-local candidate formula or exact blocker for:
   - `P_STATE`
   - `MU_STATE`
   - `W_STATE`
5. Compare candidate outputs against WRF pre-call truth and current JAX stages.
6. If the candidate closes the boundary, apply the smallest GPU-native source
   patch and rerun at least:
   - `proofs/v014/step1_live_nest_perturb_state_init.py`
   - `proofs/v014/step1_first_rk_part1_p_state_split.py`
   - the strict Step-1 comparison proof if available and not excessive.
7. If the candidate fails, report ranked alternate hypotheses and the exact next
   proof surface/source lines.

## Verdicts

Emit exactly one final verdict:

- `STEP1_LIVE_NEST_PERTURB_STATE_FIXED_<source_or_formula>`
- `STEP1_LIVE_NEST_PERTURB_STATE_LOCALIZED_<source_or_missing_contract>`
- `STEP1_LIVE_NEST_PERTURB_STATE_BLOCKED_<specific_truth_or_source_gap>`
- `STEP1_LIVE_NEST_PERTURB_STATE_REFUTED_<hypothesis_and_next_best>`

## Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_live_nest_perturb_state_init.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_perturb_state_init.py
python -m json.tool proofs/v014/step1_live_nest_perturb_state_init.json \
  >/tmp/step1_live_nest_perturb_state_init.validated.json
git diff -- src/gpuwrf
```

If production source changes:

```bash
python -m py_compile src/gpuwrf/integration/d02_replay.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_first_rk_part1_p_state_split.py
python -m json.tool proofs/v014/step1_first_rk_part1_p_state_split.json \
  >/tmp/step1_first_rk_part1_p_state_split.after_perturb_state_fix.validated.json
```

## Acceptance Criteria

- JSON validates and records CPU-only execution unless manager-authorized GPU
  check is added later.
- The report includes ranked hypotheses, exclusions, and why the chosen next
  surface is the fastest rigorous path.
- Any source fix has before/after residuals and preserves GPU-native execution:
  no CPU-WRF runtime dependency and no timestep-loop host/device transfer.
- The review report includes objective, files changed, commands run, proof
  objects, unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_LIVE_NEST_PERTURB_STATE_INIT DONE - see proofs/v014/step1_live_nest_perturb_state_init.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
