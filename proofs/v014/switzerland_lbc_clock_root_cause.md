# V0.14 Switzerland LBC Clock Root Cause

Date: 2026-06-11
Verdict: `LBC_CLOCK_ROOT_CAUSE_PROVEN_FIX_GATE_PASS`

## Root Cause

The single-domain `gpuwrf.cli run` path advances forecasts in hourly calls to
`run_forecast_operational`. Each call restarts the internal forecast step clock
at lead zero. Before this fix, the state still carried the full 73-record
boundary time axis, so each hourly call consumed boundary records 0 -> 1 again.

The effect was a frozen lateral boundary: the GPU outer `MU` ring matched CPU
truth hour 1 exactly at every probed lead through h72. That boundary freeze
drove the Switzerland d01 72h pressure/mass drift.

## Proof

- Broken run: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_20260611T001250Z/gpu_output`
- CPU truth: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`
- Fixed h6 run: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_h6_lbcfix_20260611T013851Z/gpu_output`
- JSON proof: `proofs/v014/switzerland_lbc_clock_root_cause.json`
- Proof script: `proofs/v014/switzerland_lbc_clock_root_cause.py`

Gates:

| Gate | Result |
|---|---|
| Broken GPU `MU` boundary ring equals CPU h01 ring at h1/h2/h3/h6/h12/h24/h48/h72 | PASS, max abs `0.0` |
| Mechanism emulation reproduces broken target and fixed target | PASS, max abs `0.0` |
| Fixed h6 GPU rerun boundary ring follows same-hour CPU truth | PASS, max abs `0.0` at h1-h6 |
| Fixed h6 PSFC drift collapses vs broken run | PASS, h6 RMSE `37.5365 Pa` vs broken h6 `245.419 Pa` |
| Fixed h1-h6 grid comparator | PASS |

Selected fixed h6 metrics:

| Lead | MU RMSE Pa | PSFC RMSE Pa | T RMSE K | Boundary ring max abs Pa |
|---:|---:|---:|---:|---:|
| 1 | 27.9498 | 28.9827 | 0.2865 | 0.0 |
| 2 | 30.2536 | 31.2754 | 0.4286 | 0.0 |
| 3 | 32.2720 | 32.8453 | 0.5332 | 0.0 |
| 4 | 33.9918 | 35.4393 | 0.5905 | 0.0 |
| 5 | 35.4195 | 36.8337 | 0.6220 | 0.0 |
| 6 | 34.5023 | 37.5365 | 0.6407 | 0.0 |

Performance note: the h6 proof run included two compile/initialization-heavy
segments (`452.2 s`, `480.3 s`), then stable hot-step segments around `37.8 s`
per forecast hour. The fix reduces device boundary-leaf time-axis carry from
full-run records to two records per segment; it does not introduce an obvious
hot-loop GPU performance regression in this h6 proof.

## Next Step

Merge the fix branch and rerun the full Switzerland d01 72h field gate. The
expected result is removal of the monotonic `PSFC/MU/PH` drift class; remaining
bounded physics residuals should be adjudicated with the same policy as Canary.
