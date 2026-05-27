# Skill Regression Summary Table

| Variable | CPU WRF RMSE | GPU RMSE | Relative change | Proof object |
|---|---:|---:|---:|---|
| T2 | 2.15 K | 7.86 K | +266 percent | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json` |
| U10 | 2.31 m s-1 | 11.31 m s-1 | +390 percent | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json` |
| V10 | 2.75 m s-1 | 9.44 m s-1 | +243 percent | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json` |
| L2 d02 T2 bounded check | threshold 3.0 K | 4.07 K | FAIL | `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json` |
| L2 d02 U10 bounded check | threshold 7.5 m s-1 | 10.78 m s-1 | FAIL | `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json` |
| L2 d02 V10 bounded check | threshold 7.5 m s-1 | 7.83 m s-1 | FAIL | `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json` |
