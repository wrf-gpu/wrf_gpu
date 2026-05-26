# M6b V3 20260509 Theta Localization Memo

**Verdict**: `IC-SPECIFIC`

**Evidence**:
- V3 operational replay on `20260509_18z_l3_24h_20260511T190519Z` first breached theta bounds at step None / lead None s.
- Failure cell was k=None j=None i=None with theta=None K and bound=None K.
- WRF hourly reference status at the failure site: `WRF_REFERENCE_BENIGN_AT_FAILURE_SITE`.
- WRF selected-cell theta at nearest hourly reference: 292.177490234375 K.
- First divergence against interpolated hourly WRF reference: {'step': 1, 'lead_seconds': 10.0, 'forecast_theta_k': 292.1775817871094, 'interpolated_wrf_theta_k': 292.17756746080187, 'delta_k': 1.4326307507417368e-05, 'abs_delta_k': 1.4326307507417368e-05}.
- Boundary-ring profiler large-boundary classification: True.
- Operator budget top term: `pressure_restoring`.

**Recommended next sprint**: `m6b-v3-ic-boundary-forcing-audit` — isolate the named cause without touching `dynamics/core/` or the operational-mode body.

**Risks / caveats**:
- WRF truth is hourly, so sub-hour divergence uses interpolation between hourly wrfout files rather than an acoustic-substep savepoint.
- The diagnostic first-bad helper is the sanitizer-off replay helper, while Stage 1 uses the V3 operational entry point.
- No GPU-vs-CPU step-2 NaN cross-link was found in this sprint's inputs; if a sister sprint reproduces that NaN, compare its first bad cell against this proof.
