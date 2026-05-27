# Performance Summary Table

Table 1. Performance summary separated by system state. The post-fix corrected-physics path is the current result; the pre-fix path is retained as diagnostic history.

| System state | Metric | Value | Proof object |
|---|---|---:|---|
| Post-fix corrected-physics path | 24 h d02 pipeline wall time | 708.32 s | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/pipeline_run_20260521.json` |
| Post-fix corrected-physics path | 24 h forecast-only wall time | 700.73 s | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/pipeline_run_20260521.json` |
| Post-fix corrected-physics path | CPU d02-only timing denominator | 16305 s | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json` |
| Post-fix corrected-physics path | Apples-to-apples d02-only speedup | 23.02x | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json` |
| Post-fix corrected-physics path | Full five-domain CPU aggregate framing | 63.39x, not apples-to-apples | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json` |
| Pre-fix diagnostic path | 24 h d02 pipeline wall time | 324.78 s | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` |
| Pre-fix diagnostic path | 24 h forecast-only wall time | 310.27 s | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` |
| Pre-fix diagnostic path | Apples-to-apples d02-only speedup | 50.20x | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` |
| Pre-fix diagnostic path | Full five-domain CPU aggregate framing | 138.24x, not apples-to-apples | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` |
| Shared systems evidence | Warm 1 h d02 forecast wall time | 5.71 s | `.agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json` |
| Shared systems evidence | Three-run warm reproducibility CV | 0.42 percent | `.agent/sprints/2026-05-27-m7-profiler-window-fix/reproducibility_v2.json` |
| Shared systems evidence | Inter-kernel D2H inside forecast loop | 0 copies, 0 bytes | `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json` |
| Shared systems evidence | Restart continuity | max delta 0.0 | `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json` |
| Shared systems evidence | 1 km one-step memory probe | 7278 MiB of 32607 MiB | `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json` |
