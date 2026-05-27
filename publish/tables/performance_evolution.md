# Performance Evolution Table

All timings use the 20260521 Canary 3 km d02 case. The d02-only speedup row is the defensible apples-to-apples denominator; full five-domain CPU aggregate ratios are retained only as context.

| System state | 24 h pipeline wall time | 24 h forecast-only wall time | CPU d02 denominator | d02-only speedup | Full five-domain framing | Verdict / caveat | Proof objects |
|---|---:|---:|---:|---:|---:|---|---|
| Pre-fix diagnostic path | 324.78 s | 310.27 s | 16305.31 s | 50.20x | 138.24x, not apples-to-apples | Rejected as current headline: fast path had material skill regression. | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json`; `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` |
| Iter-1 post-fix corrected-physics path | 708.32 s | 700.73 s | 16305.31 s | 23.02x | 63.39x, not apples-to-apples | SKILL_IMPROVED_PARTIAL: winds improved, T2 still outside tolerance. | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/pipeline_run_20260521.json`; `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json` |
| Iter-2 theta/land/boundary path | 732.63 s | 687.90 s | 16305.31 s | 22.26x | 61.28x, not apples-to-apples | BLOCKED: speed invariant preserved, all three skill variables still outside tolerance. | `.agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json`; `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_speedup.json` |
