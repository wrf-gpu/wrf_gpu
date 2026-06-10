# Reviewer Report

Decision: ACCEPT AFTER MANAGER GPU GATE.

The worker's WRF anchoring and CPU proof are coherent. The code path matches
WRF's `calc_cq` w-face total-moisture average and `pg_buoy_w` `cq1/cq2`
water-loading form, while preserving the dry specialization when `qtot=0`.
The source diff is appropriately narrow: one dynamics helper module and the
operational acoustic prep wiring.

Manager-added GPU proof:

- `proofs/v014/moist_cqw_gpu_h4_validation.{py,json,md}`
- Run root:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_moistcqw_h4_20260610T165255Z`
- GPU rc `0`, harness verdict `L2_D02_GREEN`, peak VRAM `16921 MiB`.

Acceptance evidence:

- New GPU `P+PB(k0)` residual versus its own moist hydrostatic half-level:
  mean/RMSE `-9.492/11.758 Pa`; CPU truth is `-13.349/13.444 Pa`.
- Previous PSFC-fix GPU baseline was dry-balanced:
  moist residual `-201.492/204.437 Pa`, dry residual `-5.990/9.268 Pa`.
- h1-h4 field RMSE improves materially:
  `P 55.125 -> 22.642 Pa`, `W 0.0281 -> 0.0250`, `T 0.310 -> 0.256 K`,
  `U 0.505 -> 0.384 m/s`, `V 0.442 -> 0.272 m/s`, `U10 0.695 -> 0.482 m/s`,
  `V10 0.861 -> 0.519 m/s`.
- `QVAPOR` is essentially unchanged and finite; `PSFC` remains good
  (`45.775 -> 41.013 Pa` RMSE).

Remaining all-field comparator `FAIL` is not a rejection of this sprint: the
top field is static/base-state `MUB/PB` plus surface/radiation lanes, while the
assigned 3D pressure-state dry-column bug is closed by direct hydrostatic and
field evidence.
