# Performance Summary Table

| Metric | Value | Proof object |
|---|---:|---|
| Warm 1 h d02 forecast wall time | 5.71 s | `.agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json` |
| Three-run warm reproducibility CV | 0.42 percent | `.agent/sprints/2026-05-27-m7-profiler-window-fix/reproducibility_v2.json` |
| 24 h d02 pipeline wall time | 324.78 s | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` |
| Forecast-only wall time | 310.27 s | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` |
| CPU d02-only timing denominator | 16305 s | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` |
| Corrected apples-to-apples speedup | 50.20x | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` |
| Inter-kernel D2H inside loop | 0 copies, 0 bytes | `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json` |
| Restart continuity | max delta 0.0 | `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json` |
| 1 km memory probe | 7278 MiB of 32607 MiB | `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json` |
