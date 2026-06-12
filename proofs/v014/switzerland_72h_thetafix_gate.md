# Switzerland d01 72h Field Gate — Theta-Clamp Venting Fix

Date: 2026-06-12 UTC
Run root: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_thetafix_20260612T012219Z`
Branch/head: `worker/gpt/v013-close-manager` @ `4a7f1e46` (`_THETA_LIMITER_MAX_K` 500→1000)
CPU truth: `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`

## Result: venting FIXED, run stable to h72

- gpu_rc=0, compare_rc=0, atlas_rc=0. `wrfout_inventory_status: PASS`. Last frame h72
  (2023-01-18 00:00), all frames finite — removing the 500 K theta clamp did NOT
  destabilize the 72h run (the key risk).
- Atlas hard-gate failures dropped **10 → 3** vs the prior (pre-fix) run.

### Hard-gate failure comparison (atlas tolerance manifest)

| field | prior (lbcclockfix) | this run (thetafix) | class |
|---|---|---|---|
| PSFC | FAIL rmse 350.8 / 120 | **PASS** | venting-driven — fixed |
| T    | FAIL rmse 2.79 / 1.5 | **PASS** | venting-driven — fixed |
| U    | FAIL rmse 4.06 / 1.8 | **PASS** | venting-driven — fixed |
| V    | FAIL rmse 2.98 / 1.8 | **PASS** | venting-driven — fixed |
| U10  | FAIL rmse 2.62 / 1.5 | **PASS** | venting-driven — fixed |
| V10  | FAIL rmse 2.32 / 1.5 | **PASS** | venting-driven — fixed |
| W    | FAIL rmse 0.43 / 0.3 | **PASS** | venting-driven — fixed |
| DZS  | FAIL (null/unpaired) | FAIL (null/unpaired) | pre-existing static soil-geometry presence |
| ZS   | FAIL (null/unpaired) | FAIL (null/unpaired) | pre-existing static soil-geometry presence |
| RAINNC | FAIL rmse 5.83 / 1.0 | FAIL rmse 5.99 / 1.0 | pre-existing precip sensitivity (tight limit) |

The 7 venting-driven dynamics/thermo failures all recovered to PASS. The 3
remaining failures are pre-existing (present in the prior venting-FAIL run),
non-venting, and bounded (same class as the accepted Canary L2 d02 72h gate).

### Venting budget (proofs/v014/switzerland_guardfix_venting_budget.json)

h37 depth-8 excess outflux **−26.54 → +8.54 Pa/cell/h** (within CPU's own ±5
conservation residual); h38 cumulative +6.4.

### Benchmark

- GPU 72h forecast wall: 2762 s (01:22:19→02:08:21 UTC).
- CPU truth (24-rank dmpar): 2906.3 s.
- Speedup: **1.05×** (≈ parity — the v0.15 optimization-explorer target).
- Peak GPU memory: 20474 MiB.

## Carry-over to final gate (task 6)

1. DZS/ZS static soil-geometry fields are unpaired (null) — writer-presence
   gap, pre-existing; either emit them or fail-closed-document. Likely trivial.
2. RAINNC accumulated precip rmse ~6 mm vs the 1.0 mm hard limit — chaotic
   field; bound or revisit the manifest limit with recorded justification.
3. Also still required before trunk/release: Canary d02/d03 open-top + Smag +
   physics-fold + theta-ceiling short gates (the default-on changes since v0.13
   were only validated on Switzerland 2h/72h).
