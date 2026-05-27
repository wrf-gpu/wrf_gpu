# Honesty Audit - Publication First Draft

Sprint: `2026-05-27-publication-first-draft`
Draft: `publication/draft/paper.md`
Verdict: `DRAFT_READY_WITH_KNOWN_LIMITATION`

This audit lists quantitative or externally sensitive claims in the draft and the support used. Claims without a local proof object are either tied to cited literature from the research briefs or explicitly softened.

## Project Performance Claims

| Claim in draft | Support | Treatment |
|---|---|---|
| Warm 1 h d02 forecast wall time is 5.71 s | `.agent/sprints/2026-05-26-m7-gpu-profile-prep/wall_clock.json`; `.agent/sprints/2026-05-27-m7-profiler-window-fix/reproducibility_v2.json` mean 5.7058455703330155 s | Kept |
| 24 h d02 pipeline wall time is 324.78 s | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` wall_clock_total_s 324.77563990700037 | Kept |
| 24 h forecast-only wall time is 310.27 s | Same pipeline JSON wall_clock_forecast_only_s 310.2703010409932 | Kept |
| Corrected apples-to-apples speedup is 50.20x | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/honest_speedup_table.json` row `cpu_d02_only_24h` ratio 50.20484702814852 | Kept as headline |
| 138.24x full-nest framing exists but is not apples-to-apples | Same JSON row `cpu_full_nest_5_domain_aggregate_24h`, fairness verdict says not apples-to-apples | Kept only with caveat |
| 156.82x was wrong | `.agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md`; `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/verdict.md` | Mentioned once only as rejected number |
| Three-run warm reproducibility CV is 0.42 percent | `reproducibility_v2.json` coefficient_of_variation 0.0042480879014396965 | Kept |
| Cold JIT compile-inclusive wall is 102.58-106.18 s | `wall_clock.json` cold_start_jit_compile_inclusive_wall_s for 20260509 and 20260521 | Kept as operational caveat |
| Inter-kernel D2H inside loop is zero | `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json` counts.d2h_inter_kernel_inside_window=0 and bytes.d2h_inter_kernel_inside_window=0 | Kept |
| Restart continuity max delta is 0.0 | `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json` all listed fields max_delta 0.0, verdict PASS | Kept |
| 24/24 hourly wrfouts readable | `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/wrfout_inventory.json` and pipeline JSON wrfout files list | Kept |
| 1 km memory probe uses 7278 MiB of 32607 MiB with roughly 78 percent headroom | `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json`; `nvidia_smi_after_warm.memory_used_mib=7278`, total 32607 | Kept as memory probe, not forecast validation |

## Forecast Quality Claims

| Claim in draft | Support | Treatment |
|---|---|---|
| Side-by-side AEMET scoring used 73 stations, 24 common valid hours, 1747 joined rows | `.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json` station_count_scored, common_valid_time_count, station_observation_rows | Kept |
| T2 CPU/GPU RMSE is 2.15 K vs 7.86 K, +266 percent | Same JSON aggregate_comparison.variables.T2.metrics.rmse | Kept |
| U10 CPU/GPU RMSE is 2.31 m s-1 vs 11.31 m s-1, +390 percent | Same JSON aggregate_comparison.variables.U10.metrics.rmse | Kept |
| V10 CPU/GPU RMSE is 2.75 m s-1 vs 9.44 m s-1, +243 percent | Same JSON aggregate_comparison.variables.V10.metrics.rmse | Kept |
| GPU is materially worse on all aggregate metrics | Same JSON verdict `FAIL_SKILL_DIFF`, aggregate_comparison.gpu_materially_worse_than_cpu=true | Kept |
| L2 d02 replay independently failed bounded Tier-4 checks | `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json` status FAIL | Kept |
| L2 d02 RMSE values 4.07 K, 10.78 m s-1, 7.83 m s-1 | Same L2 JSON fields T2/U10/V10 | Kept |
| Root cause is not yet known | `.agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md` lists RCA sprints as in flight; no RCA reports existed when inspected | Kept; hypotheses are named as hypotheses |
| Radiation cadence is effectively disabled by `radiation_cadence_steps=999999` | `wall_clock.json` and `pipeline_run_20260521.json` namelist field | Kept as plausible suspect only |

## Methods and Setup Claims

| Claim in draft | Support | Treatment |
|---|---|---|
| Claude Opus 4.7 used a 1M-token context manager role | Sprint contract and closeout authorship text | Kept as role description |
| GPT-5.5 Codex was worker/critic | Sprint contract and role prompt | Kept as role description |
| Python environment reports Python 3.13.11 and JAX 0.10.0 | Command run during this sprint: `taskset -c 0-3 python --version`; `import jax; print(jax.__version__)` | Kept; notes conflict with brief's JAX 0.4.x placeholder |
| Project package requires Python >=3.10 and jax>=0.4 | `pyproject.toml` | Kept |
| CPU comparison baseline is 28-rank WRF v4.7.1 | `.agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md`; `.agent/goals/M1-DONE.md`; sprint contract predecessor context | Kept |
| M6 B6 savepoint parity and multi-step parity were 0.0 bitwise | `.agent/decisions/MILESTONE-M6-CLOSEOUT.md` | Kept as lower-level evidence, not operational proof |
| M6 used three V3 ICs and aggregate RMSE T2/U10/V10 0.62/3.07/3.17 | `.agent/decisions/MILESTONE-M6-CLOSEOUT.md` | Mentioned only generally; detailed values not emphasized in paper |

## Literature Comparator Claims

| Claim in draft | Source in references | Treatment |
|---|---|---|
| AceCAST speedup range 5-14x | `tempoquest2025acecast` from English brief | Kept with citation |
| Pace 3.5-4x speedup context | `dahm2023pace` plus brief contract text | Kept with citation |
| ICON GPU speedup 5.5x socket-to-socket | `fuhrer2026icon`, `lapillonne2026benchmarking` | Kept with citation |
| SCREAM 1.26 SYPD at 3.25 km on Frontier-class resources | `bertagna2024scream` | Kept with citation |
| NIM up to 34x dynamics-only | `govett2017parallelization` | Kept with citation |
| ML weather models named as adjacent work | GraphCast, Pangu, FourCastNet, GenCast, Aurora, NeuralGCM, Stormer, AIFS entries from English/German briefs | Kept with citations |
| arXiv/AI authorship policy caveat | `arxiv2026policy`, `pcmag2026arxiv`, `nature2024editorial`, `schmidt2025senior` from briefs | Kept with caveat; manager should verify before submission |

## Softened or TODO Claims

| Draft item | Reason | Treatment |
|---|---|---|
| Mollick et al. authorship/collaboration citation | Requested by sprint contract but not present in provided research-brief text | Marked as `\cite{TODO_Mollick_AI_authorship}` and not fabricated in BibTeX |
| "First" or priority claims | Prior art is broad and citation extraction is incomplete | Avoided; draft explicitly says no first-GPU-regional-NWP or operational-replacement claim |
| Direct AIFS ingestion | Measured run used Gen2 d02 replay | Softened; AIFS is future/planned path, not measured result |
| 1 km operational readiness | Only memory/one-step feasibility measured | Softened to memory probe, not forecast validation |
| Operational replacement | Skill regression blocks it | Rejected throughout abstract, Results, Limitations |
