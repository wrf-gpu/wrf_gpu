# Worker Report

Summary: Implemented `scripts/m6b_v3_localize_521.py` and produced the targeted 20260521 V3 wind-bound localization proofs. Verdict is `NAMED-FIX:dycore_rk_acoustic`, not `BOUND-REVISION`: Gen2 WRF reference V near the bad cell is only 4.272483 m/s, nearby same-level max is 4.307302 m/s, and domain vertical max |V| is 11.398314 m/s at the interpolated step-46 time.

## Objective

Localize whether the step-46 `|V|=103.720413 m/s` bound violation for Gen2 ID `20260521_18z_l3_24h_20260522T072630Z` is physical or an operator defect.

## Files changed

- `scripts/m6b_v3_localize_521.py`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_step46_violation.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_wrf_reference_compare.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_operator_decomposition_input.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_operator_decomposition.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_first_divergent_step.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/localization_memo.md`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/worker-report.md`

## Commands run

Validation command:

```bash
cd /tmp/wrf_gpu2_loc_521
export OMP_NUM_THREADS=4
export PYTHONPATH="src"
taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/
```

Exit: `0`.

stderr head/tail:

```text
E0526 00:29:18.250137 ... slow_operation_alarm.cc:73]
[Compiling module jit_run_forecast_operational for GPU] Very slow compile?
E0526 00:29:19.766735 ... slow_operation_alarm.cc:140] The operation took 2m1.516675192s
[Compiling module jit_run_forecast_operational for GPU] Very slow compile?
```

stdout:

```json
{
  "artifact_type": "m6b_v3_521_localization_summary",
  "status": "OK",
  "verdict": "NAMED-FIX:dycore_rk_acoustic"
}
```

Inspection commands:

```bash
jq '{first_bad:.first_bad_step,bad_cell:.bad_cell}' .agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_step46_violation.json
jq '{status,recommendation,cell:.wrf_reference_v_at_cell_m_s,near:.wrf_reference_abs_v_max_same_level_nearby_radius2_m_s,domain:.wrf_reference_vertical_max_abs_v_anywhere_domain_m_s}' .agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_wrf_reference_compare.json
jq '{dominant_term,named_fix_operator,values_m_s,ranking:.term_budget.measurements.ranking}' .agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_operator_decomposition.json
jq '{status,earliest:.earliest_detectable_divergence_step_in_window,per_step}' .agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_first_divergent_step.json
```

Key output:

```text
first_bad_step=46, bad_cell=[33,53,36], v=103.72041320800781 m/s
WRF reference: cell=4.272482872009277 m/s, nearby=4.307302474975586 m/s, domain vertical max |V|=11.398313522338867 m/s
operator ranking: dycore_rk_acoustic max_abs=2.198419952392578 m/s/s; mynn max_abs=0.050250244140625; boundary_application=0.0
first detectable divergence in step-40..46 window: step 40
```

Runtime notes: two early attempts were interrupted because a diagnostic 46-step carry scan compiled too slowly. One later attempt reached Stage 3 and failed with a JAX donated-buffer reuse error; the script was fixed to reload a fresh initial state for that path. The final validation command above completed cleanly.

## Proof objects produced

- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_step46_violation.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_wrf_reference_compare.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_operator_decomposition.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_first_divergent_step.json`
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/localization_memo.md`

## Risks

- WRF history output is hourly, so step-46 WRF reference values use linear interpolation between 18:00 and 19:00 UTC files.
- The V3 operational wrapper does not expose separate pressure-gradient, Coriolis, and vertical-advection arrays; Stage 3 localizes to the available WRF-ordered stage boundary and names `dycore_rk_acoustic`.
- Step-45/46 `ww` snapshots are reconstructed from the operational carry path used by the localizer; no core code was modified.

## Handoff

Next decision: dispatch `2026-05-25-m6b-v3-dycore-rk-acoustic-fix` to isolate and repair the V acceleration inside `dycore_rk_acoustic`, then rerun this 20260521 localizer plus the V3 1h acceptance.
