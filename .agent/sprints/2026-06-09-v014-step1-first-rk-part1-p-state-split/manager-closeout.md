# Manager Closeout: V0.14 Step-1 First-RK Part1 P-State Split

Date: 2026-06-09 19:08 WEST

## Outcome

Closed with verdict
`STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`.

Merge Decision: commit and push proof artifacts, roadmap updates, and sprint
closeout. Do not merge any production source change from this sprint.

The current `P/MU/W` residual is not introduced by WRF
`first_rk_step_part1`, `phy_prep`, boundary package, carry, halo, or
`_physics_step_forcing`. It is already present in JAX `raw_child_state` versus
WRF `before_first_rk_step_part1_call` and remains unchanged through the JAX
live-nest/carry path.

## Proof Objects

- `proofs/v014/step1_first_rk_part1_p_state_split.py`
- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- `proofs/v014/step1_first_rk_part1_p_state_split.md`
- `.agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`

## Key Metrics

- WRF pre-call to after-part1:
  `P_STATE/MU_STATE/W_STATE/PH_STATE = 0.0`.
- WRF part1 entry to `after_phy_prep`:
  `P_STATE=0.0`, `MU_STATE=0.0`.
- JAX raw/live/halo state versus WRF pre-call:
  `P_STATE=69.96875`, `MU_STATE=13.256103515625`,
  `W_STATE=0.7605466246604919`, `PH_STATE=0.00048828125`.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_first_rk_part1_p_state_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_first_rk_part1_p_state_split.py`
- `python -m json.tool proofs/v014/step1_first_rk_part1_p_state_split.json >/tmp/step1_first_rk_part1_p_state_split.manager.validated.json`
- `git diff -- src/gpuwrf` with no output

## Scope

No production source, GPU validation, TOST, Switzerland, FP32 source work,
memory source work, or Hermes was used.

## Next Sprint

Open a narrow live-nest perturbation-state sprint for
`P_STATE/MU_STATE/W_STATE`: transcribe/prove WRF raw-child to pre-part1
initialization semantics and apply only a narrow GPU-native fix if the formula
is proven and closes the boundary.
