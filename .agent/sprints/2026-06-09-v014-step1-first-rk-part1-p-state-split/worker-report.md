# Worker Report: V0.14 Step-1 First-RK Part1 P-State Split

Date: 2026-06-09
Worker: GPT-5.5 xhigh tmux `0:4`

## Objective

Split the Step-1 `P/MU/W` residual around WRF `first_rk_step_part1`, especially
`phy_prep`, and name the exact upstream surface or contract if WRF part1 is
clean.

Summary: Worker produced a CPU-only localization proof and did not edit
production source.

## Files Changed

- `proofs/v014/step1_first_rk_part1_p_state_split.py`
- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- `proofs/v014/step1_first_rk_part1_p_state_split.md`
- `.agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`

## Commands Run

- `python -m py_compile proofs/v014/step1_first_rk_part1_p_state_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_first_rk_part1_p_state_split.py`
- `python -m json.tool proofs/v014/step1_first_rk_part1_p_state_split.json >/tmp/step1_first_rk_part1_p_state_split.validated.json`
- `git diff -- src/gpuwrf`
- attempted manager tmux notification; it failed with `Operation not permitted`
  and is recorded in the review.

## Proof Objects

- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- `proofs/v014/step1_first_rk_part1_p_state_split.md`
- `.agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`

## Result

Verdict:
`STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`.

WRF `before_first_rk_step_part1_call -> after_first_rk_step_part1` is exact for
`P_STATE/MU_STATE/W_STATE/PH_STATE`. JAX `raw_child_state`, `live_child_state`,
boundary package, carry, halo, and `_physics_step_forcing.carry.state` all
retain the same material residuals versus WRF pre-call:

- `P_STATE=69.96875`
- `MU_STATE=13.256103515625`
- `W_STATE=0.7605466246604919`
- `PH_STATE=0.00048828125`

## Unresolved Risks

- The WRF-equivalent live-nest `P/MU/W` initialization formula is not yet
  implemented.
- No post-acoustic/pre-refresh run was needed because the first material
  residual is upstream of `first_rk_step_part1`.

## Next Decision

Open a narrow live-nest perturbation-state sprint for `P_STATE/MU_STATE/W_STATE`.
Patch only after proving a GPU-native WRF-equivalent initialization formula.
