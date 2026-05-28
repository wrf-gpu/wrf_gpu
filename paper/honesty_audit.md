# Publication Honesty Audit

Date: 2026-05-27
Scope: `paper/paper.md` after the publication revision pass.

Summary: The revised paper leads with the current corrected-physics result, not the faster pre-fix diagnostic path. Quantitative claims below are either backed by proof objects or explicitly marked as release-time placeholders.

| Claim in revised paper | Proof object or source | Status |
|---|---|---|
| Current post-fix 24 h d02 pipeline wall time is 708.32 s | `proofs/2026-05-27-m7-skill-fix-algorithmic__pipeline_run_20260521.json` | backed |
| Current post-fix forecast-only wall time is 700.73 s | `proofs/2026-05-27-m7-skill-fix-algorithmic__pipeline_run_20260521.json` | backed |
| Current apples-to-apples d02-only speedup is 23.02x | `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_speedup.json` | backed |
| CPU d02-only denominator is 16305 s | `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_speedup.json` | backed |
| Post-fix full five-domain aggregate framing is 63.39x and not apples-to-apples | `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_speedup.json` | backed with caveat |
| Pre-fix 24 h d02 pipeline wall time was 324.78 s | `proofs/2026-05-27-m7-daily-pipeline-integration__pipeline_run_20260521.json` | backed; not current headline |
| Pre-fix apples-to-apples d02-only speedup was 50.20x | `proofs/2026-05-27-m7-honest-speedup-skill-diff__honest_speedup_table.json` | backed; diagnostic history |
| Original 156.82x celebration was rejected | `<development-history-not-included-in-public-repo>/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md` and honest-speedup sprint artifacts | backed |
| Warm 1 h d02 forecast wall time is 5.71 s | `proofs/2026-05-26-m7-gpu-profile-prep__wall_clock.json` | backed |
| Three-run warm reproducibility CV is 0.42 percent | `proofs/2026-05-27-m7-profiler-window-fix__reproducibility_v2.json` | backed |
| Inter-kernel D2H inside forecast loop is 0 copies, 0 bytes | `proofs/2026-05-27-m7-profiler-window-fix__d2h_audit_v2.json` | backed; parsed JSON, original nsys binary absent in this checkout |
| Restart continuity max delta is 0.0 | `proofs/2026-05-27-m7-restart-continuity__restart_continuity.json`; `<development-history-not-included-in-public-repo>/2026-05-27-m7-skill-fix-algorithmic/restart_in_pipeline.json` | backed |
| Pipeline wrfout inventory is 24/24 readable | `<development-history-not-included-in-public-repo>/2026-05-27-m7-daily-pipeline-integration/wrfout_inventory.json` | backed |
| 1 km one-step memory probe reports 7278 MiB of 32607 MiB | `proofs/2026-05-27-m7-1km-memory-audit__step_feasibility.json` | backed; not peak allocator evidence |
| 20260521 station comparison used 73 stations, 24 common valid hours, 1747 joined rows | `proofs/2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json`; `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_skill_diff.json` | backed |
| Pre-fix T2 RMSE CPU/GPU was 2.15/7.86 K | `proofs/2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json` | backed |
| Pre-fix U10 RMSE CPU/GPU was 2.31/11.31 m s-1 | `proofs/2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json` | backed |
| Pre-fix V10 RMSE CPU/GPU was 2.75/9.44 m s-1 | `proofs/2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json` | backed |
| Post-fix T2 RMSE CPU/GPU is 2.15/8.85 K | `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_skill_diff.json` | backed |
| Post-fix U10 RMSE CPU/GPU is 2.31/6.75 m s-1 | `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_skill_diff.json` | backed |
| Post-fix V10 RMSE CPU/GPU is 2.75/7.23 m s-1 | `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_skill_diff.json` | backed |
| Six of nine post-fix aggregate metrics improved versus the pre-fix GPU baseline | `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_skill_diff.json`; worker report | backed |
| All three variables remain outside 20 percent tolerance | `proofs/2026-05-27-m7-skill-fix-algorithmic__post_fix_skill_diff.json` | backed |
| L2 d02 T2/U10/V10 bounded checks failed at 4.07 K, 10.78 m s-1, 7.83 m s-1 | `<development-history-not-included-in-public-repo>/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json` | backed |
| Radiation cadence changed from pre-fix 999999 to post-fix 180 | `<development-history-not-included-in-public-repo>/2026-05-27-m7-skill-fix-algorithmic/worker-report.md` and pipeline JSON | backed |
| RRTMG runs 48 times in a 24 h integration at cadence 180 and dt=10 s | 8640 steps / 180 from post-fix configuration | inferred arithmetic |
| Current environment is Python 3.13.11, JAX 0.10.0, jaxlib 0.10.0, CUDA toolkit 13.1.115, driver 595.71.05, RTX 5090 | local environment commands run during revision pass | backed locally; must be frozen at release |
| Public repository URL is `github.com/<TBD>` | sprint contract requires placeholder | release-time placeholder |
| Release commit is TBD | release has not been cut | release-time placeholder |
| AI systems are disclosed as systems, not humans | manuscript byline and Section 12 | policy framing, not empirical claim |

Rejected or softened claims:

- The paper no longer headlines 324.78 s / 50.20x as the current result.
- The paper no longer calls the architecture "correct"; it says systems evidence keeps the architecture viable.
- The paper no longer describes radiation as disabled in the current post-fix path.
- The paper no longer describes 7278 MiB as peak process memory.
- The unresolved Mollick TODO citation was removed rather than fabricated.
