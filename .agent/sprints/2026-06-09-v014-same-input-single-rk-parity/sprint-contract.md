# Sprint Contract: V0.14 Same-Input Single-RK-Step Parity

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Produce the decisive proof boundary recommended by the Opus critic:
WRF pre-RK input at d02 step 6000 -> one JAX dynamics step -> WRF
post-RK/pre-halo output at the same step.

This must remove the stale-base and 5999-step drift confounds. If a strict
same-input comparison cannot be made without reconfounding tendencies or patch
stencils, report the exact missing input/wrapper instead of producing a weak
comparison.

## Trigger Evidence

- `.agent/reviews/2026-06-09-v014-dynamic-root-cause-opus-critic.md`
- `proofs/v014/dynamic_root_cause_opus_critic.json`
- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/wrf_post_rk_refresh_localization.json`
- `proofs/v014/same_state_momentum_mass.json`
- `proofs/v014/grid_after_live_nest_base.json`

## Non-Goals

- No production `src/` edits.
- No GPU.
- No TOST.
- No Switzerland validation.
- No FP32 or memory implementation.
- No Hermes or Telegram.

## Inputs

WRF pre-RK input savepoints:

- `/tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output/pre_rk_input_d2_step_6000_is_1_ie_23_js_18_je_33.txt`
- `/tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output/pre_rk_input_d2_step_6000_is_1_ie_23_js_1_je_17.txt`

WRF post-RK/pre-halo truth:

- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/refresh_output/refresh_post_after_all_rk_steps_pre_halo_d2_step_6000_is_1_ie_23_js_18_je_33.txt`
- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/refresh_output/refresh_post_after_all_rk_steps_pre_halo_d2_step_6000_is_1_ie_23_js_1_je_17.txt`

Optional dynamics-only comparison surface:

- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/refresh_output/refresh_post_final_calc_p_rho_phi_d2_step_6000_is_1_ie_23_js_18_je_33.txt`
- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/refresh_output/refresh_post_final_calc_p_rho_phi_d2_step_6000_is_1_ie_23_js_1_je_17.txt`

Reference helpers to inspect/reuse:

- `proofs/v014/pre_rk_input_boundary.py`
- `proofs/v014/wrf_post_rk_refresh_localization.py`
- `proofs/v014/same_state_momentum_mass.py`
- `src/gpuwrf/dynamics/step.py`
- `src/gpuwrf/dynamics/rk3.py`

## Write Scope

- `proofs/v014/same_input_single_rk_parity.py`
- `proofs/v014/same_input_single_rk_parity.json`
- `proofs/v014/same_input_single_rk_parity.md`
- `.agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md`

Scratch if needed:

- `/mnt/data/wrf_gpu2/v014_same_input_single_rk_parity/**`

## Required Work

1. Parse the WRF pre-RK input and post-RK/pre-halo savepoints, preserving native
   staggering and zero/one-based coordinate conventions.
2. Determine whether the current repo has enough fields to build a strict
   same-input JAX `State` plus tendencies for exactly one step. Inspect source
   only as needed.
3. **Tendency-control rule:** if the proof cannot feed WRF-equivalent current
   step tendencies, physics tendencies, and required history/source fields, do
   not compare against full post-RK output as if it were a dycore proof. Either:
   - compare against a narrower dynamics-only WRF surface that matches the
     available JAX boundary, or
   - emit `SAME_INPUT_TENDENCY_INPUT_BLOCKED_<missing>` naming exact missing
     fields and the next WRF/JAX hook needed.
4. **Patch-width rule:** score only cells/levels with enough halo for one-step
   stencils. If the existing patch is too narrow, emit
   `SAME_INPUT_PATCH_WIDTH_BLOCKED_<needed>` with the exact wider WRF hook.
5. If the strict comparison can run, emit per-field max_abs/RMSE and a ranked
   residual table for at least `T/P/PB/PH/PHB/MU/MUB/U/V/W`; include `U10/V10`
   only if they are actually computable at this boundary without writer or
   surface-layer confounds.
6. Emit one of:
   - `DYNAMICS_CLEAN_SINGLE_STEP`
   - `SAME_INPUT_SINGLE_STEP_MISMATCH_<first_or_dominant_field>`
   - `SAME_INPUT_TENDENCY_INPUT_BLOCKED_<missing>`
   - `SAME_INPUT_PATCH_WIDTH_BLOCKED_<needed>`
7. Recommend the next exact action:
   - If clean: upstream drift/carry-producer bisection.
   - If mismatch: one narrower term-localization sprint.
   - If blocked: exact WRF/JAX savepoint or wrapper to add.

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/same_input_single_rk_parity.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/same_input_single_rk_parity.py
python -m json.tool proofs/v014/same_input_single_rk_parity.json \
  >/tmp/same_input_single_rk_parity.validated.json
git diff -- src
```

## Acceptance Criteria

- CPU-only.
- JSON validates.
- Repo `src/` unchanged.
- No vague conclusion: either strict same-input parity ran, or the exact
  blocker is named.
- The report explicitly states whether the result supports upstream drift,
  final-RK PGF/mass-wind, theta/tendency source, or blocked instrumentation.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT SAME_INPUT_SINGLE_RK DONE - see proofs/v014/same_input_single_rk_parity.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
