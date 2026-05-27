# Worker Report - M7 LU_INDEX Audit

Summary: Completed the LU_INDEX source audit and mismatch proof objects. Verdict: `BLOCKED`.

## Summary

`src/gpuwrf/io/land_state.py` is not the source of the observed LU_INDEX mismatch. It loads `LU_INDEX` directly from Gen2 `wrfinput_d02`, and the loaded distribution exactly matches both `wrfinput_d02` and Gen2 CPU WRF first-hour `wrfout_d02` for the 20260521 case: `{5: 164, 9: 83, 10: 251, 13: 15, 16: 255, 17: 9726}`.

The GPU first-hour wrfout emits `{2: 768, 17: 9726}`. Every land cell is collapsed to category `2`; water remains `17`. This matches the current wrfout writer fallback when `State` has no `LU_INDEX` / `lu_index` field. The real fix requires carrying LU_INDEX through `State` / replay builders or passing a static-output sidecar into `wrfout_writer`; those files are outside this sprint's allowed edit set. I did not touch dycore, physics, runtime, contracts, validation, governance, reviewer, tester, manager, or memory files.

## Files Changed

- `tests/test_m7_lu_index_audit.py`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_source_audit.md`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_diff_map.nc`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_diff_summary.json`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_fix_lead1_verification.json`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/invariant_preservation.json`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/pipeline_run_20260521.json`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/wrfout_inventory.json`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/station_scores_20260521.json`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/restart_in_pipeline.json`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/repeatability.json`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/speedup_vs_cpu_24h.json`
- `.agent/sprints/2026-05-27-m7-lu-index-audit/worker-report.md`

## Commands Run And Output

`taskset -c 0-3 python - <<'PY' ... generate lu_index_diff_map.nc + lu_index_diff_summary.json`

```json
{
  "classification": "CATEGORICAL_COLLAPSE_TO_LAND_WATER_DEFAULT",
  "land_mismatch_fraction": 1.0,
  "map_path": ".agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_diff_map.nc",
  "max_abs_diff": 14.0,
  "mismatch_count": 768,
  "summary_path": ".agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_diff_summary.json"
}
```

`PYTHONPATH=src taskset -c 0-3 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 1 --output-dir /tmp/m7_lu_index_audit_20260521 --proof-dir .agent/sprints/2026-05-27-m7-lu-index-audit`

```text
exit code: 2
stderr: empty
stdout key fields: verdict=PIPELINE_PARTIAL, device=cuda:0, hours=1, output_dir=/tmp/m7_lu_index_audit_20260521, wall_clock_total_s=88.39677902800031, wrfout_files=["/tmp/m7_lu_index_audit_20260521/wrfout_d02_2026-05-21_19:00:00"]
note: script exits 2 unless the full daily-pipeline green path is requested; the 1h wrfout was emitted and used by lu_fix_lead1_verification.json.
```

`taskset -c 0-3 python - <<'PY' ... generate lu_fix_lead1_verification.json`

```json
{
  "hfx_rmse": 65.25587803869163,
  "lu_max_abs_diff": 14.0,
  "output": ".agent/sprints/2026-05-27-m7-lu-index-audit/lu_fix_lead1_verification.json",
  "verdict": "BLOCKED"
}
```

`PYTHONPATH=src taskset -c 0-3 pytest -q tests/test_m7_lu_index_audit.py`

```text
.                                                                        [100%]
1 passed in 1.71s
```

`taskset -c 0-3 python scripts/validate_agentos.py`

```json
{
  "errors": [],
  "ok": true,
  "required_files_checked": 31,
  "skills_checked": 13
}
```

`git diff --check`

```text
<no output>
```

## Proof Objects Produced

- AC1: `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_source_audit.md`
- AC2: `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_diff_map.nc`
- AC2 summary: `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_index_diff_summary.json`
- AC5 blocked probe: `.agent/sprints/2026-05-27-m7-lu-index-audit/lu_fix_lead1_verification.json`
- AC6 scope/invariant audit: `.agent/sprints/2026-05-27-m7-lu-index-audit/invariant_preservation.json`
- Test proof: `tests/test_m7_lu_index_audit.py`

## Risks

- The end-to-end LU_INDEX fix is not applied. It needs a follow-up contract that owns `State`, replay builders, and/or `wrfout_writer`.
- HFX/LH/PBLH/T2 do not improve in this sprint because the real propagation fix is blocked.
- AC6 step-2 bitwise parity was not rerun; instead, the invariant proof records that no dycore/physics/runtime/core files were touched. Running the historical step-2 script would write aggregate files into a different sprint directory.
- Remote push was not performed because the sprint contract hard rule says "No remote push. Local commit only."

## Handoff

- objective: audit and fix 20260521 LU_INDEX mismatch between GPU wrfout and Gen2 reference.
- files changed: listed above; no forbidden files modified.
- commands run: listed above with stdout/stderr status.
- proof objects produced: source audit, NetCDF diff map, diff summary, 1h blocked verification, invariant-preservation audit, focused pytest.
- unresolved risks: real fix requires widening file ownership to preserve LU_INDEX into wrfout output.
- next decision needed: authorize a follow-up LU_INDEX propagation sprint that may edit `src/gpuwrf/contracts/state.py`, `src/gpuwrf/integration/d02_replay.py`, `src/gpuwrf/coupling/driver.py`, and/or `src/gpuwrf/io/wrfout_writer.py`.
