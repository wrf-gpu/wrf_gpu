# Worker Report

Summary: localized the Step-1 tendency boundary to WRF `first_rk_step_part2`
`T_TENDF` source leaves without production source edits.

objective: localize or fix the remaining Step-1 tendency-family divergence
after patched-init `P_STATE` closure.

files changed:

- `proofs/v014/step1_tendency_contract_split.py`
- `proofs/v014/step1_tendency_contract_split.json`
- `proofs/v014/step1_tendency_contract_split.md`
- `.agent/reviews/2026-06-09-v014-step1-tendency-contract-split.md`

commands run:

- `python -m py_compile proofs/v014/step1_tendency_contract_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_tendency_contract_split.py`
- `python -m json.tool proofs/v014/step1_tendency_contract_split.json >/tmp/step1_tendency_contract_split.validated.json`
- `git diff --check`

proof objects produced:

- `proofs/v014/step1_tendency_contract_split.py`
- `proofs/v014/step1_tendency_contract_split.json`
- `proofs/v014/step1_tendency_contract_split.md`
- `.agent/reviews/2026-06-09-v014-step1-tendency-contract-split.md`

unresolved risks:

- Source-save PH/RW evidence is patch-only.
- WRF `after_rk_addtend_before_small_step_prep` is post dry boundary/addtend/spec
  work; JAX `_augment_large_step_tendencies` is not that exact boundary.

next decision needed: split WRF `first_rk_step_part2` internals around
`calculate_phy_tend`, `update_phy_ten`, and `conv_t_tendf_to_moist`.
