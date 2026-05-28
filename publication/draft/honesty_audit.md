# Publication Honesty Audit

Date: 2026-05-28
Scope: `publication/draft/paper.md` after the port-first rewrite sprint.

Summary: The revised paper leads with the source-open JAX/XLA WRF-compatible port and whole-state GPU residency. It uses the iteration-2 22.26x speedup as the current result, preserves the 156.82x overclaim as a rejected methodology example, and discloses the multi-day Canary station-skill regression in Abstract, Results, Limitations, and Discussion.

| Claim in revised paper | Proof object or source | Status |
|---|---|---|
| v0.0.1 keeps high-frequency forecast state resident on a single consumer-grade GPU | `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json`; `.agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md` | backed |
| Operational forecast loop performs zero host-device transfers inside the loop | `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json` | backed |
| Inter-kernel D2H inside forecast loop is 0 copies, 0 bytes | `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json` | backed |
| Per-step bitwise parity against unmodified WRF v4 is demonstrated to 100 coupled steps on the column savepoint tier | `.agent/sprints/2026-05-27-testing-plan-execution-redo/savepoint_deep_column100.json`; `.agent/sprints/2026-05-27-testing-plan-execution-redo/savepoint_parity_deep.json` | backed |
| 1000- and 10000-step savepoint depths are deferred to v0.1 | `.agent/sprints/2026-05-28-testing-execution-opus-check/paper_rewrite_input.md`; `.agent/sprints/2026-05-28-testing-execution-opus-check/skip_fail_triage.md` | backed as limitation |
| Three independent 1-hour Canary d02 pipeline runs are bitwise identical across 41 archived fields | `.agent/sprints/2026-05-27-testing-plan-execution-redo/determinism_repeat.json` | backed |
| Determinism proof total recorded GPU runtime is 17.6 s for the three runs | `.agent/sprints/2026-05-27-testing-plan-execution-redo/determinism_repeat.json` | backed |
| Publication-test harness consumed 1.226 GPU-hours across the HIGH-priority set | `.agent/sprints/2026-05-27-testing-plan-execution-redo/aggregate_report.json` | backed |
| RTX 5090 had 32607 MiB total and about 26200 MiB used at sprint time | `.agent/sprints/2026-05-27-testing-plan-execution-redo/aggregate_report.json`; `paper_rewrite_input.md` | backed by rewrite input |
| Four Canary d02 forecasts were executed: 20260428 partial-history 2 h, and complete 24 h cases 20260509, 20260521, 20260525 | `.agent/sprints/2026-05-27-testing-plan-execution-redo/aggregate_report.json`; `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_case_manifest.json` | backed |
| Complete-day forecast wall clocks were between 572 s and 713 s | `.agent/sprints/2026-05-27-testing-plan-execution-redo/aggregate_report.json`; `paper_rewrite_input.md` | backed by rewrite input |
| Iteration-2 24 h d02 pipeline wall time is 732.63 s | `.agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json` | backed |
| Iteration-2 24 h forecast-only wall time is 687.90 s | `.agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json` | backed |
| Iteration-2 apples-to-apples d02-only speedup is 22.26x | `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_speedup.json` | backed |
| Iteration-1 24 h d02 pipeline wall time is 708.32 s | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/pipeline_run_20260521.json` | backed |
| Iteration-1 forecast-only wall time is 700.73 s | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/pipeline_run_20260521.json` | backed |
| Iteration-1 apples-to-apples d02-only speedup is 23.02x | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json` | backed |
| Pre-fix 24 h d02 pipeline wall time was 324.78 s | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` | backed; diagnostic history only |
| Pre-fix forecast-only wall time was 310.27 s | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` | backed; diagnostic history only |
| Pre-fix apples-to-apples speedup was 50.20x | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` | backed; diagnostic history only |
| Original 156.82x headline speedup was rejected | `.agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md`; `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/verdict.md` | backed |
| Warm 1 h d02 forecast wall time is about 5.71 s | `.agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json`; `.agent/sprints/2026-05-27-m7-profiler-window-fix/reproducibility_v2.json` | backed |
| 1 km one-step feasibility probe reports 7278 MiB of 32607 MiB | `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json` | backed; not a full 1 km forecast |
| Restart continuity max delta is 0.0 | `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json`; `.agent/sprints/2026-05-27-m7-skill-fix-iter2/restart_in_pipeline.json` | backed |
| Operational dry-mass relative drift is 4.81e-6 on the 2026-05-21 24 h Canary d02 forecast | `.agent/sprints/2026-05-27-testing-plan-execution-redo/conservation_mass_24h.json` | backed |
| Proxy total-energy drift is bounded at 3.09 % over 24 h | `.agent/sprints/2026-05-27-testing-plan-execution-redo/conservation_energy_24h.json` | backed |
| Stability surrogates ran dt in {0.5x, 1.0x, 1.25x} and acoustic substeps in {4, 6, 8} with finite output | `.agent/sprints/2026-05-27-testing-plan-execution-redo/stability_cfl_sweep.json`; `.agent/sprints/2026-05-27-testing-plan-execution-redo/stability_acoustic_substep.json` | backed |
| Pairwise surface nRMSE across acoustic settings is bounded by 4.16e-3 | `.agent/sprints/2026-05-27-testing-plan-execution-redo/stability_acoustic_substep.json` | backed |
| 2026-05-09 T2 CPU/GPU RMSE is 2.51/11.97 K, +378 % | `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_multiday_skill.json` | backed |
| 2026-05-09 U10 CPU/GPU RMSE is 2.12/7.21 m/s, +240 % | `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_multiday_skill.json` | backed |
| 2026-05-09 V10 CPU/GPU RMSE is 2.21/6.51 m/s, +195 % | `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_multiday_skill.json` | backed |
| 2026-05-21 T2 CPU/GPU RMSE is 2.15/10.80 K, +303 % | `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_multiday_skill.json` | backed |
| 2026-05-21 U10 CPU/GPU RMSE is 2.31/7.24 m/s, +214 % | `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_multiday_skill.json` | backed |
| 2026-05-21 V10 CPU/GPU RMSE is 2.75/7.62 m/s, +177 % | `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_multiday_skill.json` | backed |
| 2026-05-25 T2 CPU/GPU RMSE is 2.95/7.71 K, +161 % | `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_multiday_skill.json` | backed |
| 2026-05-25 U10 CPU/GPU RMSE is 2.11/9.92 m/s, +370 % | `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_multiday_skill.json` | backed |
| 2026-05-25 V10 CPU/GPU RMSE is 2.24/10.16 m/s, +353 % | `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_multiday_skill.json` | backed |
| None of T2, U10, V10 is within +/-20 % of CPU WRF RMSE on the three complete days | `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_multiday_skill.json`; `.agent/sprints/2026-05-28-testing-execution-opus-check/skip_fail_triage.md` | backed |
| 20260428 partial-history case has zero valid joined station pairs and is excluded | `.agent/sprints/2026-05-27-testing-plan-execution-redo/canary_multiday_skill.json` | backed |
| Radiation cadence is 180 steps, giving 48 RRTMG calls in a 24 h integration | `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/worker-report.md`; arithmetic from 8640 / 180 | backed/inferred |
| Lateral boundary pack was widened to `spec_bdy_width=5` in iteration 2 | `.agent/sprints/2026-05-27-m7-skill-fix-iter2/worker-report.md` | backed |
| Public release target is `github.com/wrf-gpu/wrf_gpu` under AGPL-3.0 | `.agent/sprints/2026-05-28-paper-rewrite-port-first/sprint-contract.md` | release-plan claim |
| Versioning policy is 0.0.x prototype, 0.1.0 arXiv companion, 1.0.0 reserved for operational claim | `.agent/sprints/2026-05-28-paper-rewrite-port-first/sprint-contract.md` | release-plan claim |
| Current environment is Python 3.13.11, JAX 0.10.0, jaxlib 0.10.0, CUDA toolkit 13.1.115, NVIDIA driver 595.71.05, RTX 5090 | prior publication audit environment; current draft inherited values | backed locally by prior revision pass; must be frozen at release |
| AI systems are disclosed as systems and Enric R.G. retains human submission responsibility | `publication/draft/paper.md`; `PAPER-REWRITE-FRAMING-MEMO.md` | disclosure claim |

Rejected or softened claims:

- The paper does not claim to be the first GPU-enabled WRF.
- The paper does not claim skill equivalence with CPU WRF.
- The paper does not use 156.82x as a valid performance result.
- The paper does not headline the faster pre-fix 50.20x path as current.
- The paper does not claim formal energy conservation or community-benchmark idealized-case validation.
- The paper does not claim full 24 h pipeline determinism; only the 1 h segment is demonstrated.
- The paper does not claim a full 1 km forecast; only one-step memory feasibility is reported.
