# Skill Evolution Table

The same 20260521 side-by-side AEMET scoring setup is used for all rows: 73 stations, 24 common valid hours, and 1747 joined station-time rows. Relative deltas compare each GPU metric to the CPU WRF metric; BIAS deltas use absolute bias magnitude, matching the proof JSON schema.

| Variable | System | BIAS | BIAS delta vs CPU | MAE | MAE delta vs CPU | RMSE | RMSE delta vs CPU | Proof object |
|---|---|---:|---:|---:|---:|---:|---:|---|
| T2 (K) | CPU WRF baseline | 0.34 | 0% | 1.68 | 0% | 2.15 | 0% | `proofs/2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json` |
| T2 (K) | Pre-fix GPU | 5.46 | +1487% | 5.76 | +243% | 7.86 | +266% | `proofs/2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json` |
| T2 (K) | Iter-1 GPU | 6.90 | +1908% | 7.46 | +344% | 8.85 | +312% | `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_skill_diff.json` |
| T2 (K) | Iter-2 GPU | 6.35 | +1749% | 8.24 | +390% | 10.80 | +403% | `proofs/2026-05-27-m7-skill-fix-iter2__post_iter2_skill_diff.json` |
| U10 (m s-1) | CPU WRF baseline | -0.14 | 0% | 1.71 | 0% | 2.31 | 0% | `proofs/2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json` |
| U10 (m s-1) | Pre-fix GPU | 8.01 | +5824% | 9.17 | +436% | 11.31 | +390% | `proofs/2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json` |
| U10 (m s-1) | Iter-1 GPU | 3.01 | +2127% | 5.29 | +209% | 6.75 | +193% | `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_skill_diff.json` |
| U10 (m s-1) | Iter-2 GPU | 3.11 | +2196% | 5.63 | +229% | 7.24 | +214% | `proofs/2026-05-27-m7-skill-fix-iter2__post_iter2_skill_diff.json` |
| V10 (m s-1) | CPU WRF baseline | -0.67 | 0% | 1.97 | 0% | 2.75 | 0% | `proofs/2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json` |
| V10 (m s-1) | Pre-fix GPU | -3.65 | +444% | 7.43 | +277% | 9.44 | +243% | `proofs/2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json` |
| V10 (m s-1) | Iter-1 GPU | 0.12 | -83% | 5.75 | +191% | 7.23 | +163% | `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_skill_diff.json` |
| V10 (m s-1) | Iter-2 GPU | -0.03 | -95% | 6.02 | +205% | 7.62 | +177% | `proofs/2026-05-27-m7-skill-fix-iter2__post_iter2_skill_diff.json` |
