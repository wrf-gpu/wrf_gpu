# Skill Regression and Post-fix Recovery Summary Table

Table 2. Skill summary separated by pre-fix and post-fix state. The post-fix path improves 6 of 9 aggregate T2/U10/V10 metrics versus the pre-fix GPU baseline, but all variables remain outside the pre-declared 20 percent tolerance against CPU WRF.

| System state | Variable | CPU WRF RMSE | GPU RMSE | Relative change vs CPU | Proof object |
|---|---|---:|---:|---:|---|
| Pre-fix diagnostic path | T2 | 2.15 K | 7.86 K | +266 percent | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json` |
| Pre-fix diagnostic path | U10 | 2.31 m s-1 | 11.31 m s-1 | +390 percent | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json` |
| Pre-fix diagnostic path | V10 | 2.75 m s-1 | 9.44 m s-1 | +243 percent | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json` |
| Post-fix corrected-physics path | T2 | 2.15 K | 8.85 K | +312 percent | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json` |
| Post-fix corrected-physics path | U10 | 2.31 m s-1 | 6.75 m s-1 | +193 percent | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json` |
| Post-fix corrected-physics path | V10 | 2.75 m s-1 | 7.23 m s-1 | +163 percent | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json` |
| L2 d02 replay validation | T2 bounded check | threshold 3.0 K | 4.07 K | FAIL | `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json` |
| L2 d02 replay validation | U10 bounded check | threshold 7.5 m s-1 | 10.78 m s-1 | FAIL | `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json` |
| L2 d02 replay validation | V10 bounded check | threshold 7.5 m s-1 | 7.83 m s-1 | FAIL | `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json` |
