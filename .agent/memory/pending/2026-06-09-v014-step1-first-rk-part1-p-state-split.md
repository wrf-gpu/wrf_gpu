# Pending Memory: V0.14 Step-1 First-RK Part1 P-State Split

Date: 2026-06-09 19:08 WEST

Verdict:
`STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`.

Accepted proof objects:

- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- `proofs/v014/step1_first_rk_part1_p_state_split.md`
- `.agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`

Manager validation:

- `python -m py_compile proofs/v014/step1_first_rk_part1_p_state_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_first_rk_part1_p_state_split.py`
- `python -m json.tool proofs/v014/step1_first_rk_part1_p_state_split.json >/tmp/step1_first_rk_part1_p_state_split.manager.validated.json`
- `git diff -- src/gpuwrf` with no output

Key memory:

- WRF `before_first_rk_step_part1_call` to `after_first_rk_step_part1` is exact
  for `P_STATE/MU_STATE/W_STATE/PH_STATE`.
- WRF part1 entry to `after_phy_prep` is exact for `P_STATE/MU_STATE`.
- JAX `raw_child_state` already differs from WRF pre-call for `P_STATE`,
  `MU_STATE`, and `W_STATE`.
- JAX `live_child_state`, boundary package, initial carry, haloed step entry,
  and `_physics_step_forcing.carry.state` preserve the same residuals rather
  than introducing them.
- First material residuals versus WRF pre-call:
  `P_STATE=69.96875`, `MU_STATE=13.256103515625`,
  `W_STATE=0.7605466246604919`, `PH_STATE=0.00048828125`.
- Exact missing contract: live-nest `raw_child_state -> live_child_state`
  perturbation-state initialization for `P_STATE/MU_STATE/W_STATE`.

Manager decision:

- Do not edit `first_rk_step_part1`, `phy_prep`, boundary package, carry, halo,
  or acoustic refresh from this evidence.
- Open a narrow live-nest perturbation-state sprint to transcribe/prove WRF
  `P/MU/W` initialization semantics and apply only a GPU-native fix if that
  proof closes the boundary.
