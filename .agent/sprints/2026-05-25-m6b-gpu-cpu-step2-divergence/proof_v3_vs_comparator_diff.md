# V3 vs Comparator Diff

- Aggregate matrix complete: True.
- Current verdict: (A)-SENTINEL-COINCIDENCE.
- V3 script path: `scripts/m6b_canary_1h_honest_v3.py`.
- V3 uses `run_forecast_operational`: True.
- V3 imports validation wrappers: False.
- Prior CPU bisect anchor in worker report: 2, 5, 10 steps; report path `.agent/sprints/2026-05-25-m6b-standalone-vs-comparator-bisect/worker-report.md`.
- V3 bounds path was a stepwise operational-mode bounds audit, not a validation-wrapper comparator.
- Comparator/bisect path checked bitwise harness agreement; V3 checked physical bounds and fail-fast validity.
- Tolerance/field-order difference: comparator proofs use max-abs field deltas over broad state snapshots; V3 records theta/u/v/w physical bounds and does not compute `max_abs_delta`.
- Warm/cold cache difference: V3 compiled during the stepwise audit and did not record a JIT-cache warm-state discriminator. This sprint records wall time per step but does not claim performance.

## V3 Bounds Summary

```json
{
  "available": true,
  "path": "/tmp/wrf_gpu2_gpucpu/.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/proof_bounds.json",
  "runs": [
    {
      "blocker": "WIND_BOUNDS",
      "first_bad_all_leaves_finite": true,
      "first_bad_lead_seconds": 460.0,
      "first_bad_step": 46,
      "run_id": "20260521_18z_l3_24h_20260522T072630Z",
      "status": "FAIL",
      "steps_checked": 46
    },
    {
      "blocker": "WIND_BOUNDS",
      "first_bad_all_leaves_finite": true,
      "first_bad_lead_seconds": 460.0,
      "first_bad_step": 46,
      "run_id": "20260521_18z_l3_24h_20260522T133443Z",
      "status": "FAIL",
      "steps_checked": 46
    },
    {
      "blocker": "THETA_BOUNDS",
      "first_bad_all_leaves_finite": true,
      "first_bad_lead_seconds": 110.0,
      "first_bad_step": 11,
      "run_id": "20260509_18z_l3_24h_20260511T190519Z",
      "status": "FAIL",
      "steps_checked": 11
    }
  ],
  "status": "BLOCKER"
}
```
