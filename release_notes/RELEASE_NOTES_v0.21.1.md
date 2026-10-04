# Release notes - wrf_gpu v0.21.1

**v0.21.1 is a focused stability point release off v0.21.0.** It packages the
manager-verified Mont-Blanc boundary fix.

## What changed

v0.21.0 could run the most-extreme Mont-Blanc-class fixture without immediate
floating-point overflow, but the solution was physically invalid: d01 physical
vertical velocity (`W`) diverged at the outer specified-boundary corner while the
interior remained bounded. The false-pass sequence was approximately:

`44 -> 288 -> 2066 -> 14531 -> 99325 m/s`.

v0.21.1 fixes the boundary mechanism:

- standalone `wrfbdy` roots now use WRF specified-boundary cadence and edge
  advection degradation;
- hydrometeor and number scalar `wrfbdy` leaves are decoded and applied when the
  file actually carries the corresponding WRF side base/tendency arrays;
- specified-domain `zero_grad_bdy(W)` is applied to the reconstructed physical
  `W`, by solving the acoustic work value that `small_step_finish_wrf`
  reconstructs back to WRF's copied physical boundary value;
- WRF's corner source-index behavior is preserved: y-side rows own the corners
  and use the nearest interior source column.

This is a WRF boundary-copy semantics fix, not a limiter. There is no `W`
masking, `nan_to_num`, finite guard, or value clip/clamp in the fix.

## Validation state

Manager-verified source-branch gates before release integration:

- Mont-Blanc native-dt max-dom2 2 h: d01 max `|W|` bounded at
  `2.31, 4.13, 3.57, 3.62, 3.58 m/s`; d02 max `|W| <= 8.04 m/s`.
- 20250121 native-dt max-dom3: `PIPELINE_GREEN`; T/U/V correlations vs CPU-WRF
  `0.999995 / 0.999839 / 0.999751`.
- Canary native-dt max-dom3: `PIPELINE_GREEN`.
- Focused CPU suite: `40 passed, 3 skipped`.

Release-branch acceptance artifacts are under `proofs/v022/nest_dycore/`:
Mont-Blanc native-dt max-dom2 confirms d01 `W` remains bounded during the short
release gate, and Canary native-dt max-dom3 is `PIPELINE_GREEN`. The artifacts
are summarized in `proofs/v022/nest_dycore/V0211_RELEASE_INTEGRATION_REPORT.md`.

## Compatibility

v0.21.1 is intended to be `v0.21.0 + the Mont-Blanc boundary fix` only. It does
not change the v0.21.0 AOT/fused-cache release story, does not tag or push from
this worktree, and does not update the curated `wrfgpu` distribution tree.

## Honest scope

- The Mont-Blanc fixture in this release gate is max-dom2; d03 no-regression is
  covered by the manager-verified 20250121 gate and the release-branch Canary
  max-dom3 gate.
- The nested CLI still treats `--compare-cpu-dir` as a single-domain post-step;
  CPU-WRF T/U/V comparison proofs for nested runs are generated separately when
  needed.
- Long-horizon OOM and broader v0.22 performance work remain outside this point
  release.
