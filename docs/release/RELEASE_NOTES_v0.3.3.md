# v0.3.3 — WRF-compatible forecasts on one NVIDIA GPU

v0.3.3 improves agreement with original Fortran WRF and fixes expensive GPU layout problems. It keeps familiar WRF inputs and history files, using JAX/Pallas on the GPU.

**Current release: v0.3.3.** Validation is accepted with the disclosed deviations below.

## What changed

- **Faster Noah-MP radiation:** a native fp32 kernel accelerates land-surface radiation. The published speed and energy figures remain dated v0.3.2 measurements; these science runs are not a new quiet-host throughput benchmark.
- **Compiler layout fixed:** the new default avoids oversized fused kernels while preserving the physics.
- **Snow fall speeds match WRF:** Thompson now uses the correct denominator for tiny active snow amounts.
- **Small-grid stability barriers:** additional phase boundaries cost about 12% more kernels and approximately 8% more ordinary-step time on grids with ≤4,096 horizontal columns only. Larger grids do not use them; this is not a universal whole-run slowdown.

## Validation

Seven 72-hour forecasts—six Tenerife runs including initial-condition members and a genuine replicate, plus Storm Monica—and the shipped Swiss 24-hour case are compared with **original CPU-WRF**, using every paired hour and native cell.

| Comparison | Forecast | Strict 24 h (raw) | Full-window D6 (raw) | Interpretation |
|---|---:|---|---|---|
| Tenerife 0115 primary | 72 h | FAIL | FAIL | 61 floor-limited rows; one real V10 miss. Rain and cirrus lifetime differences disclosed (L4/L6/L7). |
| Tenerife 0227 primary | 72 h | PASS | FAIL | All eight breaches pass the frozen CPU-pair spread annex (L5). |
| Tenerife 0408 primary | 72 h | PASS | PASS | D6 passes; trace graupel integrity miss disclosed (L1). |
| 0227 IC member | 72 h | PASS | FAIL | All six breaches pass the frozen CPU-pair spread annex; integrity annex passes. |
| 0115 IC member | 72 h | FAIL | FAIL | 54 floor-limited rows; strict rain FAIL retained. Integrity annex passes; IC difference included. |
| 0115 same-IC replica | 72 h | PASS | FAIL | Eight floor-limited rows, two real V10 misses (h42/h44). Cirrus lifetime integrity FAIL disclosed; same-IC fresh-compile replica. |
| Storm Monica | 72 h | PASS | PASS | D6 and all-frame integrity pass. |
| Shipped Swiss | 24 h | PASS | PASS | D6 and all-frame integrity pass; 25 frames on one domain. |

[Hourly plots](evidence/v033/final_w3/index.html) show RMSE, signed bias and spatial error bands. [The technical appendix](TECHNICAL_VALIDATION_APPENDIX_v0.3.3.md) retains source identities, unchanged limits, raw failures and replica qualifications.

## Known deviations

- **Rain:** 0115 reaches 1.042–1.045 mm RMSE versus a 1 mm limit. A genuine same-IC fresh-compile replica reaches 0.845–0.886 mm: small compile/autotune round-off moves this chaotic forecast across the bound. The miss is disclosed; 0408 rain passes and 0227 d03 rain rises from 1.10 mm at h67 to 2.465 mm at h72 (up to 2.47× the 1 mm limit). All eight raw breaches are FLOOR-LIMITED under the frozen A3 CPU-pair annex: C/X 0.81–0.95, so the two original CPU runs differ almost as much. [Rain evidence](TECHNICAL_VALIDATION_APPENDIX_v0.3.3.md#final-manager-ledger).
- **Trace graupel:** one 0408 frame has CPU values of 2.46×10⁻¹⁰ kg/kg in five boundary cells, with GPU output zero. Ledger L1 records this as disclosed and nonblocking. [Trace assessment](TECHNICAL_VALIDATION_APPENDIX_v0.3.3.md#final-eight-arm-results).
- **Cloud ice:** the 0115 integrity report has a separate hour-60 population difference: CPU ice mass about 2.28×10⁻⁷ kg/kg and number 2.40×10⁴/kg in 50 cells, with GPU zero. The fresh replica has the same class in d03 at hour 63: CPU mass 2.14×10⁻⁸ kg/kg and number 4.50×10⁴/kg across 112 columns, GPU zero. The manager records this late cirrus-patch lifetime difference as disclosed, with the station assessment meeting the no-material-skill-loss criterion; the raw failed check remains visible. [Ice findings](TECHNICAL_VALIDATION_APPENDIX_v0.3.3.md#final-eight-arm-results).
- **Brief negative moisture:** unclipped transport/source updates shared with WRF can produce undershoots before physics cleanup. Monitoring found minima −3.84×10⁻⁶kg/kg cloud water and −4.20×10⁻⁶kg/kg water vapour, without nonfinite values. [Counts and limitations](TECHNICAL_VALIDATION_APPENDIX_v0.3.3.md#f2-nonzero-counts-explicitly-disclosed).
- **Coastal wind speed:** 0115 station RMSE is about 1.8% higher, while temperature, humidity and wind direction improve. The primary has one late V10 miss (1.692 m/s versus 1.5); the fresh replica has two (1.725 and 1.623 m/s). This persistent small excess is disclosed for follow-up, not renamed PASS. [Station assessment](TECHNICAL_VALIDATION_APPENDIX_v0.3.3.md#final-manager-ledger).

Small misses can be disclosed only under the Owner's full rule: plausible trajectory divergence, no unresolved port bug, physically valid finite results, and no material station-skill loss. Large errors, new failure types, crashes and failed native-kernel gates still block. Raw failed checks remain visible.

## Upgrading

`GPUWRF_LAYOUT_PIN` now defaults to `ac_carry_cols_cum`. Remove explicit old layout settings to use the release default. A changed layout may require fresh compatible executable caches; do not bypass admission checks. See the [user guide](https://wrf-gpu.github.io/wrf_gpu/) for installation and your first forecast.

**Release identity:** v0.3.3 · `be0dd543c9f7f6d619a8b504348b11267ffe913f` · measured-source mappings in the appendix.
