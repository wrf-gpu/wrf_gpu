# Reviewer Report: V0.14 Step-1 JAX Start-Domain Input Split

Decision: `ACCEPT_LOCALIZATION_OPEN_NEXT_BASE_STATE_BOUNDARY`.

Review scope:

- Proof artifacts and worker review for
  `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_LOCALIZED_BASE_STATE_RECONSTRUCTION_FP32_ALT_SOURCE_ORDER_GAP`.
- Sprint contract requirements.
- Manager rerun results.
- Production source diff status.

Findings:

- The proof satisfies the sprint objective by splitting the live-nest
  `start_domain` inputs rather than continuing broad dycore debugging.
- The result is methodologically useful: direct WRF `AL/ALT` substitution
  closes most of the old `P/MU` path, while proof-local production inputs still
  cannot generate WRF-equivalent `AL/ALT`.
- The no-patch decision is correct. The best local fp32/cp=1004.5 base
  candidate still leaves `P_STATE` max_abs `2.828125` and `MU_STATE` max_abs
  `0.011962890625`; patching now would guess at WRF source-order details.
- The next boundary is exact enough for a new worker: WRF `start_domain`
  base-state values before hypsometric `AL/ALT`, including `p_surf`, `MUB`,
  `PB/T_INIT/ALB`, `PHB`, coefficients, flags, and scalar constants.

Risks:

- The proof reuses existing WRF truth surfaces and does not yet emit the
  missing base-state boundary.
- No production code was changed, so the grid divergence remains open.
- The constant discrepancy in `d02_replay.py` (`cp=1004.0` versus WRF
  `1004.5`) is real but not sufficient by itself; it must be handled under the
  next source-order proof, not as an isolated patch.

Recommendation:

Commit the proof and formal closeout. Open a new narrow sprint for the exact
WRF base-state boundary. Keep TOST, Switzerland, FP32 source work, and broad
memory source work paused until this live-nest grid-parity issue is fixed or
bounded by proof.
