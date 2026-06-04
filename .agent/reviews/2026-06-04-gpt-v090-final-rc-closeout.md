# GPT Final RC Closeout - v0.9.0

## Objective

Finalize the v0.9.0 release-candidate hygiene pass without changing frozen kernel,
State, precision, dycore, or coupler code. Scope was limited to README framing,
known-issues documentation, and re-scoring existing proof JSONs.

## Files changed

- `README.md`
- `docs/KNOWN_ISSUES.md`
- `proofs/v090/naive_agent_gate.json`
- `proofs/v090/speedup_benchmark.json`
- `.agent/reviews/2026-06-04-gpt-v090-final-rc-closeout.md`

## Exact changes

- `README.md`: reframed v0.9.0 as fp64 operational mode. Source:
  `src/gpuwrf/integration/daily_pipeline.py:198-200` documents the ADR-007
  rationale and `src/gpuwrf/integration/daily_pipeline.py:234` sets
  `force_fp64=True`. The prompt's `:230` line citation is stale by four lines.
- `README.md`: separated real-user speedup from kernel/compute-only ceiling.
  Real-user d02 speedup is `2.16` warm conservative, `2.41` midpoint, `2.59`
  high, and `1.33` cold conservative from
  `proofs/v090/speedup_benchmark.json:/cases/nested_9_3km/results`. The warm
  wall clock is `2149.88616062497 s` and `29.859530008680142 s/fc-hr` from
  `proofs/v090/speedup_d02_72h/pipeline_run_l2_d02.json:/wall_clock_total_s`
  and `/wall_clock_per_forecast_hour_s`.
- `README.md`: stated why that speedup is precision-independent today. Source:
  `proofs/perf/compute_cycle_analysis.md:13-17` and `:118-126` say the workload
  is launch/memory-bound and fp32 measured about `1.00x`; `:148-154` identifies
  fusion / launch-count reduction as the lever.
- `README.md`: kept the kernel/compute-only ceiling separate. Source:
  `publish/runtime_optimization_analysis.md:13-18` gives `5.29x` conservative
  and `7.84x` realistic compile-excluded kernel/compute framing.
- `README.md`: updated d02 science numbers from
  `proofs/v090/d02_coupled_skill_72h.json:/field_summary` and `/lead_count`.
  Values cited: `72` leads; T2 `72/72`, mean `1.0643340472268767 K`, max
  `1.4232233285372622 K`, final `0.8128759757122581 K`, bar `3.0 K`; V10
  `72/72`, mean `3.206825141907839 m/s`, final `2.9739276356930175 m/s`, bar
  `7.5 m/s`; U10 `66/72`, mean `4.7910817885018595 m/s`, final
  `3.9998579990063505 m/s`, max `8.041450483217423 m/s`, bar `7.5 m/s`;
  HFX mean/final `60.45938515385001` / `45.091941676632025 W m-2`; PBLH
  mean/final `182.1028436126301` / `265.2520672925367 m`.
- `docs/KNOWN_ISSUES.md`: documented KI-1 d03 1 km gated-fp32 qke nonfinite
  after hour `1`, grid `44 x 75 x 93`, `dt=3.0 s`, `10` acoustic substeps,
  and `3036` qke cells from `proofs/v090/d03_1km_validation.json`. It also
  documents the later contradiction: `proofs/v090/d03_1km_validation_qkefix.json`
  and `proofs/v090/pipeline_run_d03_qkefix_gated_fp32.json` show qke promoted
  to float64 but still failing at hour `1` with the same `3036` cells and finite
  min/max `0.00002330710837569365` / `27.360868568649096`. Full-fp64 d03 short
  finite evidence is `0.3 h / 360 steps` from
  `proofs/v090/d03_replay_finite_check.json:/survived_hours` and
  `/survived_steps`.
- `docs/KNOWN_ISSUES.md`: documented KI-2 as a case-sensitive long single-call
  qke robustness edge. Source:
  `proofs/v090/d02_gated_fp32_recheck.json:/regression_test` records hour `1`
  failure with `2024` qke cells in both base-qke and promoted-qke variants.
  Caveat: see factual concerns below.
- `docs/KNOWN_ISSUES.md`: documented KI-3 writer scope. Source:
  `proofs/v090/naive_agent_gate.json:/dimension_mismatch_detail` records `64`
  generated variables vs `375` reference variables. Source:
  `proofs/v090/naive_gate_run/dimension_compare.json:/files/0/generated_dims`
  and `/reference_dims` shows core dims match and only `seed_dim_stag=8`,
  `snow_layers_stag=3`, and `snso_layers_stag=7` are missing from the generated
  file.
- `proofs/v090/naive_agent_gate.json`: rescored the gate PASS under the
  correct-forecast contract. Source data preserved: process exit remains `1`,
  pipeline verdict remains `PIPELINE_PARTIAL`, strict dimension comparison
  remains FAIL as diagnostic data, and the gate now records why the old
  `375`-variable criterion conflated diagnostic writer coverage with forecast
  correctness. Source:
  `proofs/v090/naive_gate_run/pipeline_payload.json:/all_finite_check` shows
  `all_finite=true` across `56` written fields and `/metadata/namelist`
  shows `force_fp64=true`.
- `proofs/v090/speedup_benchmark.json`: reconciled precision provenance. Source:
  `proofs/v090/speedup_d02_72h/pipeline_run_l2_d02.json:/metadata/namelist/force_fp64`
  is `false`, so the recorded wall-clock was gated-fp32 replay, not a direct
  fp64 timing. The file now says `precision_provenance_uncertain=false` because
  that provenance is known, and says the wall applies to fp64 ship mode only
  because committed roofline/profiling evidence says fp32 vs fp64 is about
  `1.00x` on this launch/bandwidth-bound workload.

## Commands run

All shell commands were pinned with `taskset -c 0-3`. No GPU forecast was
launched. No Python execution was needed, so no command required `PYTHONPATH=src`.

- `taskset -c 0-3 git status --short --branch`
- `taskset -c 0-3 git diff --stat`
- `taskset -c 0-3 git diff --check`
- `taskset -c 0-3 jq empty proofs/v090/naive_agent_gate.json proofs/v090/speedup_benchmark.json`
- `taskset -c 0-3 sed -n '1,220p' README.md`
- `taskset -c 0-3 sed -n '1,260p' docs/KNOWN_ISSUES.md`
- `taskset -c 0-3 sed -n '1,220p' proofs/v090/naive_agent_gate.json`
- `taskset -c 0-3 sed -n '1,260p' proofs/v090/speedup_benchmark.json`
- `taskset -c 0-3 jq ... proofs/v090/d02_coupled_skill_72h.json`
- `taskset -c 0-3 jq ... proofs/v090/d03_1km_validation.json`
- `taskset -c 0-3 jq ... proofs/v090/d03_1km_validation_qkefix.json`
- `taskset -c 0-3 jq ... proofs/v090/d03_replay_finite_check.json`
- `taskset -c 0-3 jq ... proofs/v090/d02_gated_fp32_recheck.json`
- `taskset -c 0-3 jq ... proofs/v090/naive_gate_run/dimension_compare.json`
- `taskset -c 0-3 jq ... proofs/v090/naive_gate_run/pipeline_payload.json`
- `taskset -c 0-3 nl -ba src/gpuwrf/integration/daily_pipeline.py | sed -n '190,240p'`
- `taskset -c 0-3 nl -ba proofs/perf/compute_cycle_analysis.md | sed -n '1,180p'`
- `taskset -c 0-3 nl -ba publish/runtime_optimization_analysis.md | sed -n '1,60p'`

## Proof objects produced

- Updated `proofs/v090/naive_agent_gate.json`
- Updated `proofs/v090/speedup_benchmark.json`
- This review: `.agent/reviews/2026-06-04-gpt-v090-final-rc-closeout.md`

## Factual concerns

- The manager-requested KI-1 wording "qke fp32 overflow at 1 km" is contradicted
  by the later committed qke-fp64 proof. The original proof
  `proofs/v090/d03_1km_validation.json:/nonfinite_fields/qke` says float32 qke
  overflow with `3036` cells, but
  `proofs/v090/d03_1km_validation_qkefix.json:/full_24h_gated_fp32_run` says qke
  was float64 and still failed at hour `1` with the same `3036` cells. I did not
  repeat the stale overflow claim; I documented the corrected diagnosis.
- The manager prompt cited `src/gpuwrf/integration/daily_pipeline.py:230` for
  `force_fp64=True`; in this checkout the actual assignment is line `234`.
- The manager prompt says the 20260521 long single-call qke edge was verified in
  both full fp64 and gated-fp32. The clean evidence I found directly supports
  base-qke vs qke-promoted A/B. In this checkout, the subordinate promoted proof
  path named by `proofs/v090/d02_gated_fp32_recheck.json` points to a different
  20260509 grid-mismatch artifact, so I treated the full-fp64 daily-pipeline
  wording as insufficiently proven by committed artifacts.
- The speedup wall-clock was not directly measured in fp64 ship mode:
  `proofs/v090/speedup_d02_72h/pipeline_run_l2_d02.json:/metadata/namelist/force_fp64`
  is `false`. I reconciled this honestly by stating that the measurement is
  gated-fp32 replay and is applicable to fp64 only because the committed
  compute-cycle evidence says precision is not the bottleneck.

## Unresolved risks

- d03 1 km operational scoring remains blocked until a v0.10.0 numerics fix
  produces a finite full-length proof.
- The long single-call daily-pipeline qke edge needs a clean full-fp64 vs
  gated-fp32 proof pair if the project wants to keep saying "both fp64 and
  gated-fp32" without caveat.
- The v0.9.0 naive gate now passes the correct-forecast contract, but a future
  writer/gate hygiene sprint should decide whether selected Noah-MP snow-layer
  diagnostics should be emitted or explicitly excluded by schema.

## Next decision

Release manager should decide whether the RC can proceed with status CONCERN
because the corrected qke diagnosis and speedup provenance are honest but differ
from the exact manager-set wording.
