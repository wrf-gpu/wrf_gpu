# Worker Report — M6b V3 Localize 20260509 Theta

Summary: Implemented the pinned 20260509-only localization driver and produced the required proof objects. Verdict is `MATH:coftz`: the operational V3 replay breaches theta bounds at step 11 / lead 110 s at k=28 j=60 i=73, while the Gen2 WRF reference is benign at that site, the first divergence against interpolated hourly WRF reference appears at step 1, and the boundary-ring profiler is not boundary-dominated.

## objective

Pinpoint whether the 20260509 theta explosion in Gen2 ID `20260509_18z_l3_24h_20260511T190519Z` is physical, math/operator, IC-specific, numerical drift, or insufficient evidence.

## files changed

- `scripts/m6b_v3_localize_509.py`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/worker-report.md`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/localization_memo.md`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_theta_explosion.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_wrf_reference_theta.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_math_vs_ic.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_first_bad_step_tracer.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_cell_divergence_trace.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_vertical_column_phase_space.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_boundary_ring_error.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_operator_term_budget.json`
- Diagnostic sidecar inputs and captured validation stdout/stderr in the sprint folder.

## commands run + output

`python -m py_compile scripts/m6b_v3_localize_509.py`

Output: no stdout/stderr; exit 0.

`cd /tmp/wrf_gpu2_loc_509 && export OMP_NUM_THREADS=4 && export PYTHONPATH="src" && taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/`

Stdout:

```json
{
  "artifact_type": "m6b_v3_localize_20260509_summary",
  "proofs": [
    ".agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_theta_explosion.json",
    ".agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_wrf_reference_theta.json",
    ".agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_math_vs_ic.json",
    ".agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/localization_memo.md"
  ],
  "status": "MATH:coftz"
}
```

Stderr: XLA reported a slow compile for `jit_run_forecast_operational`; no traceback; exit 0.

`git add -A && git commit -m "[V3 localize 20260509] $(date -u +%FT%TZ)"`

Stdout:

```text
[worker/gpt/m6b-v3-localize-20260509-theta b3b6649] [V3 localize 20260509] 2026-05-25T23:53:54Z
 16 files changed, 25091 insertions(+)
```

Stderr: empty; exit 0.

## proof objects produced

- `proof_theta_explosion.json`: first theta violation at step 11, lead 110 s, level 28, cell i=73 j=60, theta `2.604313608192e12 K`.
- `proof_wrf_reference_theta.json`: WRF reference benign at the failure site; selected-cell theta `348.6410217285156 K`.
- `proof_math_vs_ic.json`: verdict `MATH:coftz`; first divergence step 1; boundary-ring large classification false.
- `proof_operator_term_budget.json`: top ranking `coftz`, then `theta_transport`, `pressure_restoring`, `buoyancy`, `cofwr`.
- `localization_memo.md`: handoff recommendation for the next sprint.

## risks

- WRF truth is hourly, so sub-hour divergence uses linear interpolation rather than savepoint truth.
- The first-bad helper is the sanitizer-off replay helper, while Stage 1 uses the operational V3 entry point.
- A separate GPU-vs-CPU step-2 NaN sprint was active in this worktree environment; this sprint did not consume its result.

## handoff

Objective complete. Next decision needed: dispatch `m6b-v3-fix-20260509-theta-math` to isolate the `coftz`/theta-transport recurrence path without touching `dynamics/core/` or `operational_mode.py` body.
