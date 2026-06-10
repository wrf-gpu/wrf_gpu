# Memory Patch

Reviewer Status: ACCEPTED_BY_MANAGER.

## Durable Facts

- The v0.14 3D pressure-state dry-vs-moist dynamics blocker is closed by a
  WRF-anchored production fix.
- WRF source anchors:
  `module_big_step_utilities_em.F:856-870` (`calc_cq` w-face `cqw`) and
  `module_big_step_utilities_em.F:2474-2497` (`pg_buoy_w` `cq1/cq2` plus
  water-mass loading).
- Production files changed:
  `src/gpuwrf/dynamics/core/advance_w.py` and
  `src/gpuwrf/runtime/operational_mode.py`.
- The operational default is now moist-cqw ON. Use `GPUWRF_MOIST_CQW=0` only
  for bisection/back-compat checks.
- CPU proof:
  `proofs/v014/moist_cqw_pressure_dynamics_closure.{py,json,md}`.
- GPU proof:
  `proofs/v014/moist_cqw_gpu_h4_validation.{py,json,md}` and run root
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_moistcqw_h4_20260610T165255Z`.
- Key validation: new GPU `P+PB(k0)` vs moist hydrostatic half-level residual
  mean/RMSE `-9.492/11.758 Pa`, CPU truth `-13.349/13.444 Pa`, old GPU baseline
  `-201.492/204.437 Pa`.
- h1-h4 `P` RMSE improved `55.125 -> 22.642 Pa`; wind/temperature fields also
  improved; peak VRAM was `16921 MiB`.

## Next Action

Run Canary d02 72h GPU-vs-CPU field-parity/stability from the default-ON branch
with resource CSV logging. Continue to Switzerland 72h only after Canary does
not show renewed unacceptable drift, or after a fresh root-cause sprint closes
any new blocker.
