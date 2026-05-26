# M6b V3 20260509 Theta Localization Memo

**Verdict**: `MATH:coftz`

**Evidence**:
- V3 operational replay on `20260509_18z_l3_24h_20260511T190519Z` first breached theta bounds at step 11 / lead 110.0 s.
- Failure cell was k=28 j=59 i=72 with theta=2444869763072.0 K and bound=400.0 K.
- WRF hourly reference status at the failure site: `WRF_REFERENCE_BENIGN_AT_FAILURE_SITE`.
- WRF selected-cell theta at nearest hourly reference: 348.63739013671875 K.
- First divergence against interpolated hourly WRF reference: {'step': 1, 'lead_seconds': 10.0, 'forecast_theta_k': 348.63739013671875, 'interpolated_wrf_theta_k': 348.63606516520184, 'delta_k': 0.0013249715169081355, 'abs_delta_k': 0.0013249715169081355}.
- Boundary-ring profiler large-boundary classification: False.
- Operator budget top term: `coftz`.

**Recommended next sprint**: `m6b-v3-fix-20260509-theta-math` — isolate the named cause without touching `dynamics/core/` or the operational-mode body.

**Risks / caveats**:
- WRF truth is hourly, so sub-hour divergence uses interpolation between hourly wrfout files rather than an acoustic-substep savepoint.
- The diagnostic first-bad helper is the sanitizer-off replay helper, while Stage 1 uses the V3 operational entry point.
- No GPU-vs-CPU step-2 NaN cross-link was found in this sprint's inputs; if a sister sprint reproduces that NaN, compare its first bad cell against this proof.
