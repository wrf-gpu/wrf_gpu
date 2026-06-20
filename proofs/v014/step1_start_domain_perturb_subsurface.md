# V0.14 Step-1 Start-Domain Perturbation Subsurface

Verdict: `STEP1_START_DOMAIN_PERTURB_SUBSURFACE_LOCALIZED_CURRENT_JAX_AL_ALT_BASE_INPUT_GAP`.

## Result

- CPU backend: `cpu`; GPU used: `False`.
- Required ancestor `ee6cbbe1` present: `True`.
- Clean workdir: `<DATA_ROOT>/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715`; prefilled root WRF/run ignored: `True`.
- WRF emitted 28 d02 patch files for each requested surface: after hypsometric P/al/alt, before press_adj, after press_adj, and after W surface branch.
- Production source patch applied: `False`. No production patch was applied because current JAX AL/ALT/base/PH inputs still leave P max_abs 3.9458582235092763 Pa, above the 1 Pa material gate.

## Key Metrics

| Check | max_abs | RMSE | Interpretation |
|---|---:|---:|---|
| WRF P from internal ALT fp32 vs WRF after_hypsometric P | 0.015625 | 0.0017691372004962024 | source formula/order closed |
| WRF press_adj fp32 vs WRF after_press MU | 4.547473508864641e-13 | 9.500192094660529e-14 | source formula/order closed |
| WRF after W branch vs accepted pre-call W | 5.960464477539063e-08 | 7.351352471267985e-10 | W branch closed |
| Current JAX pressure formula vs WRF after_hypsometric P | 3.9458582235092763 | 0.3832298992869327 | current-input patch falsifier |
| Current JAX press_adj formula vs WRF after_press MU | 0.047773029698646496 | 0.0010454860097534014 | current-input patch falsifier |

## Ranked Hypotheses

- 1. WRF live-nest start_domain recomputes P/al/alt, then press_adj updates MU, then set_w_surface updates W. Status: `SUPPORTED_BY_INTERNAL_SURFACES`. WRF internal P-from-ALT fp32 max_abs=0.015625; press_adj fp32 max_abs=4.547473508864641e-13; after-w-surface vs pre-call W max_abs=5.960464477539063e-08.
- 2. A narrow production patch using current JAX inputs is exact enough now. Status: `REFUTED_FOR_P_IF_THRESHOLD_EXCEEDED`. Current JAX pressure formula vs WRF internal P max_abs=3.9458582235092763; current JAX press_adj formula vs WRF after_press MU max_abs=0.047773029698646496. Patch threshold for P is 1 Pa and MU is 0.01 Pa.
- 3. Remaining P gap is in current JAX start_domain input surfaces, not source ordering. Status: `SUPPORTED`. WRF internal formula/order closes against WRF internal truth, while current JAX AL/ALT/base/PH inputs still have ranked residuals headed by [{'field': 'P_STATE_raw_current_vs_wrf_after_hyp', 'max_abs': 69.96875, 'rmse': 1.161700383780558, 'bias': 0.0187159801204043, 'threshold': None, 'worst_mismatch_fortran': {'i': 156, 'j': 55, 'k': 1}}, {'field': 'W_STATE_raw_current_vs_wrf_after_w', 'max_abs': 0.7605466842651367, 'rmse': 0.014709815601114631, 'bias': -0.0001511840907329694, 'threshold': None, 'worst_mismatch_fortran': {'i': 62, 'j': 31, 'kstag': 1}}, {'field': 'PHB_current_vs_wrf_after_hyp', 'max_abs': 0.10811684231157415, 'rmse': 0.02459295000326211, 'bias': -0.01747610370992694, 'threshold': None, 'worst_mismatch_fortran': {'i': 146, 'j': 34, 'kstag': 42}}].

## Exclusions

- The prefilled scratch-root WRF/run was ignored; all trusted WRF files came from the timestamped clean workdir.
- The hook is gated on grid%press_adj=.TRUE. and grid id 2, so ordinary non-live-nest d02 start_domain calls are excluded.
- WRF after_hypsometric P_STATE is continuous with accepted pre-call P_STATE; press_adj and W branch do not mutate P.
- WRF after_press_adj MU_STATE is continuous with accepted pre-call MU_STATE; remaining current-JAX MU gap is formula/input, not later solve_em.
- WRF after_w_surface_branch W_STATE is continuous with accepted pre-call W_STATE; W is not an acoustic or physics tendency source here.
- Boundary package, carry, halo, first_rk_step_part1, phy_prep, and acoustic refresh remain excluded by predecessor proofs.

## Next Step

Split current JAX live-nest start_domain input construction for AL/ALT: compare final blended HT, PB/MUB/PHB, PH_STATE, MU before press_adj, and diagnosed AL/ALT against the WRF internal after_hypsometric surface. Patch P/MU only after that current-input gap is below the material gate.

Detailed metrics are in `proofs/v014/step1_start_domain_perturb_subsurface.json`.
