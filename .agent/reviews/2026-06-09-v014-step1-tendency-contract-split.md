# Review: V0.14 Step-1 Tendency Contract Split

Verdict: `STEP1_TENDENCY_CONTRACT_LOCALIZED_FIRST_RK_STEP_PART2_T_TENDF_SOURCE_LEAVES`.

objective: localize or fix the remaining Step-1 tendency-family divergence after patched-init `P_STATE` closure.

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
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_tendency_contract_split.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_tendency_contract_split.md`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-tendency-contract-split.md`

unresolved risks:
- The source-save hook is patch-only, so PH/RW source-save metrics are sparse falsifiers; full-domain conclusions use the full WRF surfaces.
- The full WRF after_rk_addtend surface is after relax_bdy_dry/rk_addtend_dry/spec_bdy_dry, while JAX _augment is not an exact post-spec boundary.
- No production source edit was made because the first exact failure is inside WRF first_rk_step_part2 source-leaf construction, not a proven narrow JAX source line.

next decision needed: WRF first_rk_step_part2 internals: emit after calculate_phy_tend, after update_phy_ten, and after conv_t_tendf_to_moist for the theta source leaves feeding T_TENDF. Include the raw RTH*TEN/T_HIST_SRC contributors and the current JAX dry physics source bundle.
