# V0.14 Step-1 JAX Start-Domain Input Split

Verdict: `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_LOCALIZED_BASE_STATE_RECONSTRUCTION_FP32_ALT_SOURCE_ORDER_GAP`.

## Result

- CPU backend: `cpu`; GPU used: `False`.
- Required ancestor `66c091fc` present: `True`.
- Production source patch applied: `False`. Direct WRF AL/ALT substitution closes P/MU below gates, but current/proof-local production inputs do not: best local base candidate leaves P max_abs 2.828125 and MU max_abs 0.011962890625.
- Dominant family: base-state reconstruction feeding fp32 `AL/ALT` diagnosis.

## Key Metrics

| Check | max_abs | RMSE | Interpretation |
|---|---:|---:|---|
| Current pressure formula vs WRF P | 3.9458582235092763 | 0.3832298992869327 | current inputs not patch-ready |
| Replace ALT with WRF ALT | 0.07605321895971429 | 0.006830944106223064 | diagnosed ALT is dominant |
| FP32 ALT with WRF PHB+MUB | 0.0859375 | 0.009877167668418278 | base PHB+MUB closes pressure |
| WRF fields with FP64 ALT diagnosis | 2.961779549412313 | 0.30906526361285835 | dtype/order matters |
| Best local fp32/cp=1004.5 base candidate P | 2.828125 | 0.04812588194483578 | not patch-ready |
| Best local fp32/cp=1004.5 base candidate MU | 0.011962890625 | 0.0001283031870446325 | slightly above MU gate |

## Ranked Hypotheses

- 1. Dominant current JAX formula residual is diagnosed AL/ALT, fed by base-state reconstruction. Status: `SUPPORTED_LOCALIZED_NOT_PATCH_READY`. Direct WRF ALT substitution reduces P max_abs to 0.07605321895971429; direct WRF AL/ALT/ALB substitution reduces MU max_abs to 6.14762238910771e-05. Replacing WRF PHB+MUB in the fp32 ALT diagnosis reduces P max_abs to 0.0859375.
- 2. The missing production contract is WRF start_domain base reconstruction precision/source order. Status: `SUPPORTED_BY_FALSIFIER`. Using WRF fields with fp64 ALT diagnosis leaves P max_abs 2.961779549412313, while WRF fields with fp32 diagnosis leave P max_abs 0.0625. A local fp32/cp=1004.5 base recompute still leaves P max_abs 2.828125 and MU max_abs 0.011962890625, so the exact base source/order is not closed.
- 3. Final blended terrain is the dominant source. Status: `REFUTED`. HT max_abs is 2.682152282318384e-05 m; HT_FINE max_abs is 4.547473508864641e-13 m; replacing terrain in press_adj does not improve MU (0.04777337170366991).
- 4. Time-level selection, PH_STATE, or pre-press MU is the source. Status: `REFUTED`. T1/T2, MU1/MU2, and PH1/PH2 are exact at after_hyp for checked fields; JAX PH vs WRF PH1 max_abs 5.329070518200751e-15; JAX MU vs WRF MU1 max_abs 0.0.
- 5. PB or theta alone is the dominant pressure source. Status: `REFUTED_AS_DOMINANT`. PB-only substitution leaves P max_abs 3.9035602698713774; theta-only substitution leaves P max_abs 3.9520774969569175.

## Exclusions

- WRF start_domain P/press_adj/W source ordering remains accepted from the predecessor proof.
- Time-level selection is not the P/ALT cause: T1/T2, MU1/MU2, and PH1/PH2 match exactly at the hypsometric surface.
- PH_STATE and pre-press MU are not the input gap: current JAX PH and MU match WRF PH1/MU1 to round-off/zero.
- Terrain blend is not dominant: HT and HT_FINE residuals are tiny, and terrain substitution does not improve press_adj MU.
- A narrow production patch is not safe yet: proof-local WRF-like fp32 base recompute does not close P/MU gates.

## Next Step

Emit or reproduce the exact WRF start_domain base-state source boundary before the hypsometric AL/ALT pass: p_surf, MUB immediately after assignment, PB/T_INIT/ALB after the multi-domain reconstitution block, PHB after base integration, C3F/C4F/C3H/C4H as used in memory, imask/rebalance/hybrid flags, and scalar constants. The next worker should close that base reconstruction to WRF PHB+MUB, then apply the P/MU/W perturbation init patch already proven by direct AL/ALT substitution.

Detailed metrics are in `proofs/v014/step1_jax_start_domain_input_split.json`.
