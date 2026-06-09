# Manager Closeout

## Outcome

Closed as:
`STEP1_BASE_STATE_BOUNDARY_LOCALIZED_P_SURF_MUB_FP32_SOURCE_ARITHMETIC`.

The manager hypothesis is supported but refined. The WRF branch is the
multi-domain real `start_domain_em` path with `hypsometric_opt=2`,
`rebalance=0`, and `use_theta_m=1`. Exact WRF `MUB` is the decisive remaining
surface feeding the `AL/ALT` pass: substituting WRF-emitted `MUB` into the same
proof-local base/AL/ALT path reduces downstream `P_STATE` to `0.40625 Pa` and
`MU_STATE` to `0.001220703125 Pa`, below gates. The best local fp32/cp=1004.5
p_surf formula still leaves `P_STATE=2.828125 Pa` and
`MU_STATE=0.011962890625 Pa`, so no production patch is authorized.

## Proof Objects

- `proofs/v014/step1_base_state_boundary.py`
- `proofs/v014/step1_base_state_boundary.json`
- `proofs/v014/step1_base_state_boundary.md`
- `.agent/reviews/2026-06-09-v014-step1-base-state-boundary.md`
- Reused WRF truth root:
  `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715/wrf_truth`

## Merge Decision:

Merge the proof artifacts and sprint closeout files. Do not patch
`src/gpuwrf/**` from this sprint.

## Scope Changes

No scope expansion. No TOST, Switzerland, GPU, FP32 production source, memory
production source, Hermes, or production CPU-WRF dependency was used.

## Lessons

The missing boundary is narrower than broad base-state reconstruction:
terrain, cp constants, coefficient indexing, and PH/MU time-level selection are
not dominant. The next useful artifact is exact WRF p_surf/MUB arithmetic.

## Next Sprint

Emit a disposable WRF truth surface immediately around the `p_surf` expression
and `grid%MUB(i,j) = p_surf - grid%p_top`, or implement a proof-local
WRF-compatible fp32/libm helper. Gate any production `d02_replay.py` patch on
`P_STATE <= 1 Pa`, `MU_STATE <= 0.01 Pa`, and no `src/gpuwrf` diff before the
patch.
