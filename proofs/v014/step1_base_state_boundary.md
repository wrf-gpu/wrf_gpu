# V0.14 Step-1 Base-State Boundary

Verdict: `STEP1_BASE_STATE_BOUNDARY_LOCALIZED_P_SURF_MUB_FP32_SOURCE_ARITHMETIC`.

## Result

- CPU backend: `cpu`; GPU used: `False`.
- Required ancestor `6ced5a8e` present: `True`.
- Production source patch applied: `False`. Current/proof-local p_surf formula still leaves P_STATE 2.828125 and MU_STATE 0.011962890625; only WRF-emitted MUB closes P/MU (0.40625, 0.001220703125).
- WRF branch: multi-domain real `start_domain_em`, `hypsometric_opt=2`, `rebalance=0`, `use_theta_m=1`.
- Dominant remaining source: exact WRF `p_surf -> MUB` arithmetic before the `AL/ALT` pass.

## Key Metrics

| Source family | MUB max_abs | PHB max_abs | P max_abs | MU max_abs | Interpretation |
|---|---:|---:|---:|---:|---|
| `production_current_live_child_base_fp64_cp1004_0_jax_hgt` | 0.05002361937658861 | 0.10811684231157415 | 4.40625 | 0.025634765624545253 | current production formula, not patch-ready |
| `proof_formula_fp32_cp1004_5_jax_hgt` | 0.0546875 | 0.046875 | 2.828125 | 0.011962890625 | WRF constants/fp32 help but do not close |
| `proof_formula_fp32_cp1004_5_wrf_ht` | 0.0546875 | 0.046875 | 2.828125 | 0.011962890625 | terrain substitution does not improve |
| `wrf_boundary_mub_fp32_cp1004_5_wrf_ht` | 0.0 | 0.03125 | 0.40625 | 0.001220703125 | WRF MUB closes P/MU gates |

## Source Split

- `pressure_surface_formula`: `DOMINANT_REMAINING_SOURCE`. {'formula_fp32_cp1004_5_jax_hgt_P_STATE': 2.828125, 'formula_fp32_cp1004_5_jax_hgt_MU_STATE': 0.011962890625, 'formula_fp32_cp1004_5_jax_hgt_MUB': 0.0546875, 'wrf_boundary_mub_P_STATE': 0.40625, 'wrf_boundary_mub_MU_STATE': 0.001220703125}
- `dtype_evaluation_order`: `SECONDARY_SUPPORTED_NOT_SUFFICIENT`. {'fp64_cp1004_5_P_STATE': 4.40625, 'fp32_cp1004_5_P_STATE': 2.828125}
- `coefficient_indexing`: `REFUTED_AS_DOMINANT`. Current metrics with WRF MUB close P/MU below gates.
- `terrain_blend_input`: `REFUTED_AS_DOMINANT`. WRF HT substitution leaves the fp32 formula P residual unchanged.
- `PHB_integration_order`: `SMALL_RESIDUAL_NOT_DOMINANT_DOWNSTREAM`. {'wrf_boundary_mub_candidate_PHB': 0.03125, 'wrf_boundary_mub_candidate_P_STATE': 0.40625}
- `missing_truth_surface`: `EXACT_P_SURF_BEFORE_MUB_NOT_EMITTED`. This proof recovers p_surf from MUB+P_TOP but does not have a WRF-emitted p_surf_before_mub scalar field.

## Ranked Hypotheses

- 1. The remaining Step-1 base gap is the exact WRF p_surf/MUB source arithmetic feeding AL/ALT. Status: `SUPPORTED_LOCALIZED`. The best local fp32/cp=1004.5 formula still leaves MUB max_abs 0.0546875, P_STATE 2.828125, and MU_STATE 0.011962890625; substituting WRF-emitted MUB into the same base/AL/ALT path reduces P_STATE to 0.40625 and MU_STATE to 0.001220703125.
- 2. The WRF branch is multi-domain real start_domain with rebalance disabled: PB/T_INIT/ALB are reconstituted from MUB, PHB is not re-integrated in that later block. Status: `SUPPORTED_BY_SOURCE_AND_FLAGS`. Truth headers report input_from_file=T, hypsometric_opt=2, rebalance=0, restart=F, use_theta_m=1; namelist max_dom=2. Source lines show the initial p_surf/MUB/PHB integration block followed by the max_dom>1 real reconstitution block.
- 3. Terrain/blend input is the dominant residual. Status: `REFUTED`. JAX HT vs WRF HT max_abs 2.682152282318384e-05; using WRF HT instead of JAX HT in the fp32/cp=1004.5 formula leaves P_STATE unchanged at 2.828125.
- 4. Constants or cp=1004.0 vs WRF cp=1004.5 are the dominant residual. Status: `REFUTED_AS_DOMINANT`. Changing cp affects PHB modestly but not the MUB/PB source. fp32 cp=1004.0 and cp=1004.5 both leave P_STATE 2.828125.
- 5. Coefficient indexing or PH/MU time-level selection is the cause. Status: `REFUTED_BY_PREDECESSOR_AND_CURRENT_ABLATION`. The predecessor proved JAX PH and pre-press MU match WRF PH1/MU1 to roundoff/zero. With WRF MUB, current coefficients produce P_STATE 0.40625; with WRF PHB+MUB direct substitution the pressure residual is 0.0859375.

## Exclusions

- No production src/gpuwrf edit was made.
- No GPU, TOST, Switzerland, FP32 production source, memory production source, or Hermes path was used.
- Terrain was falsified as dominant by substituting WRF HT into the proof-local fp32 formula with no P improvement.
- cp/constants were falsified as dominant: cp=1004.0 vs 1004.5 does not move MUB/PB and leaves the same downstream P gap.
- Coefficient indexing is unlikely: exact WRF MUB with current metrics closes downstream P/MU gates.
- PHB integration order remains a small base residual, but not the dominant downstream P/MU blocker after WRF MUB substitution.

## Next Decision

Do not patch d02_replay from the current p_surf formula yet. The next source contract should either instrument one disposable WRF boundary immediately around the p_surf expression/MUB assignment to capture p_surf_before_mub and MUB exactly, or implement a narrowly gated WRF-compatible fp32/libm p_surf helper and require P_STATE <= 1 Pa and MU_STATE <= 0.01 Pa in this same proof before production patching.

Detailed metrics are in `proofs/v014/step1_base_state_boundary.json`.
