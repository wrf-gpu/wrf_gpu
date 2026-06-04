# v0.9.0 Gap-Critic Review

Reviewer: GPT-5.5 xhigh gap-analysis critic
Date: 2026-06-04
Worktree: `/home/enric/src/wrf_gpu2/.claude/worktrees/gpt-v090-final`
Branch observed: `worker/gpt/v090-release-final`

## Verdict

**SHIP.** I found **zero FIX-NOW issues** that genuinely undercut the release claim: a tested stable v0.9.0 that can run the backfilled Canary Islands cases through the supported path, with fp64 as the operational mode and gated-fp32 as an experimental performance preview.

The release is not perfect and should not be described as such. The remaining issues are real, but they are either explicitly documented as known issues, scoped out by the principal's release bar, or backed by honest proof caveats. I classify them as CARRY-OVER items.

## FIX-NOW Findings

None.

## CARRY-OVER Findings

| ID | Classification | Finding | Why It Does Not Block v0.9.0 |
| --- | --- | --- | --- |
| C1 | CARRY-OVER | `proofs/v090/d02_coupled_skill_72h.json` has `status: FAIL` because U10 breaches the all-leads bar at six evening leads. | README and `docs/KNOWN_ISSUES.md` disclose this. The proof still establishes all 72 leads finite, final-hour T2/U10/V10 within bars, and skillful coupled behavior on the backfilled d02 case. The release claim is finite + skillful supported operation, not a powered paper-grade all-leads TOST pass. |
| C2 | CARRY-OVER | d03 1km gated-fp32 remains unstable after hour 1, and the qke-fp64 experiment falsifies the simple "qke fp32 overflow" hypothesis. | `docs/KNOWN_ISSUES.md` correctly frames this as a d03 steep-terrain dynamics/numerics issue, not an fp64 ship-mode failure. The d03 full-fp64 evidence is only short-run finite, so d03 1km should remain deferred to v0.10.0 numerics work. |
| C3 | CARRY-OVER | Some autonomous long single-call daily-pipeline inits, notably 20260521, show a qke-numerics edge in both fp64 and gated-fp32. | This is documented as KI-2. The supported v0.9.0 cadence advances in output-interval segments with finite checks, and the committed d02 72h replay proof plus naive-agent run demonstrate finite supported operation on the validated path. |
| C4 | CARRY-OVER | The operational writer emits a focused variable set rather than CPU-WRF's full 375-variable inventory; strict dimension/variable comparison exits nonzero. | This is documented as KI-3 and the re-scored naive-agent gate is honest: core dimensions match, fields are finite, and the strict full-inventory check is retained as a diagnostic rather than hidden. |
| C5 | CARRY-OVER | The headline d02 wall-clock speedup is directly measured in gated-fp32 replay and applied to fp64 through the committed precision-equivalence analysis, not by a direct fp64 72h timing run. | The README and `proofs/v090/speedup_benchmark.json` disclose the provenance. `proofs/perf/compute_cycle_analysis.md` measures a warmed coupled real-d02 operational step with dycore, physics, boundary, and guard cost, not a narrow single-kernel microbenchmark. Direct fp64 72h timing would improve provenance, but is not required before the immutable tag. |
| C6 | CARRY-OVER | Some proof artifacts still contain stale labels, e.g. gated-fp32 described as "operational SHIP mode" and the original d03 proof retaining the old fp32-overflow hypothesis. | The release-facing README, speedup benchmark metadata, and known-issues document correct these claims. This is proof-hygiene debt, not a live release over-claim. |
| C7 | CARRY-OVER | Older GF/Tiedtke proof/comment artifacts are stale: one v060 proof still describes GF/Tiedtke as fail-closed and `scan_adapters.py` has an outdated module comment, while live dispatch and newer proof show GF/Tiedtke coverage. | The live code and `proofs/v060/multicfg_smoke_report.json` show 21/21 run configs passing, 2/2 intended fail-closed configs, and GF adapter triggering. This should be cleaned for maintainability but does not undercut v0.9.0. |
| C8 | CARRY-OVER | The full historical pytest sweep is not globally green; `proofs/v090/release_trunk_greensuite.json` records 89 failures. | The sweep classifies these as base-identical environment, fixture, oracle, comparator, or known dycore residuals with zero real merge regressions. The release has targeted proof objects for the supported path, but the historical-suite cleanup remains release-hygiene debt. |

## Completeness

I did not find undisclosed stubs, TODOs, or silently deferred work masquerading as complete for the v0.9.0 scope.

The release-facing scope is internally consistent: fp64 is the operational mode, gated-fp32 is experimental/performance-preview, d03 1km is not validated as a full release target, and the full WRF physics catalog is not claimed. `src/gpuwrf/integration/daily_pipeline.py` sets `force_fp64=True` in the operational real-case namelist, with the ADR-007 precision policy comment immediately above it. README makes the same distinction.

Proof coverage exists for the stated release claims:

- d02 72h coupled replay: `proofs/v090/d02_coupled_skill_72h.json`
- d02 speed/wall-clock provenance: `proofs/v090/speedup_benchmark.json`
- fp32/fp64 precision-equivalence rationale: `proofs/perf/compute_cycle_analysis.md`
- final-hour Tier-4 RMSE: `proofs/v090/speedup_d02_72h/tier4_rmse_l2_d02.json`
- operational pipeline finite wrfout production: `proofs/v090/speedup_d02_72h/pipeline_run_l2_d02.json`
- naive-agent gate re-score: `proofs/v090/naive_agent_gate.json`
- known issues: `docs/KNOWN_ISSUES.md`

The stale proof annotations in C6 are the main completeness blemish. They do not override the release-facing documents, but they should be corrected after the tag or before a paper-grade artifact bundle.

## Correctness

The validation is conclusive enough for the explicit ship bar, not for stronger claims.

The d02 72h coupled validation is against a CPU-WRF oracle/backfilled case, not a JAX-vs-JAX self-compare. It reports fixed bars and an honest failing overall machine status because U10 misses six all-leads checks. It also reports all leads finite, final-hour binding within bar, and finite/skillful T2/U10/V10/HFX/PBLH behavior. README repeats the important caveat instead of converting the proof into an inflated PASS.

The README speedup numbers are traceable:

- Warm 72h d02 GPU replay total: `2149.88616062497 s`
- Conservative CPU denominator: `64.6 s/fc-hr`
- Conservative compile-amortized headline: `2.16x`
- Midpoint/high context: `2.41x` and `2.59x`
- Cold 24h conservative headline: `1.33x`

The d02 skill counts and caveats also match the proof JSON: 72 finite leads, final-hour T2/U10/V10 within bars, and six U10 all-leads breaches. I did not find invented d02 counts in README.

The naive-agent gate is not gamed. `proofs/v090/naive_agent_gate.json` retains the raw nonzero process exit and the strict dimension failure, then re-scores under the correct forecast contract because the failure is the known 64-variable focused-writer scope rather than a forecast validity failure. The payload was produced in fp64 mode and reports finite fields.

### fp64 Provenance Arbitration

The shipped operational mode is fp64, but the headline 72h d02 wall-clock speedup is measured in gated-fp32 replay. That is explicitly visible in `proofs/v090/speedup_d02_72h/pipeline_run_l2_d02.json` and `proofs/v090/speedup_benchmark.json` where `force_fp64=false`.

The critical question is whether the fp32-vs-fp64 `~1.00x` equivalence is backed by representative evidence or only a narrow per-kernel microbenchmark.

My arbitration: it is closer to **representative full-step evidence**, not a narrow per-kernel-only benchmark. `proofs/perf/compute_cycle_analysis.md` describes warmed profiling of the coupled real-d02 operational forecast, including dycore, physics, boundary, and guard cost. It reports per-step production cost and nsys-level launch/memory-operation counts, then explains why fp32 does not materially speed up the current launch-tax and memory-bandwidth-bound workload.

Limit: this is still not a direct fp64 full 72h pipeline timing. It does not directly capture the full 72h pipeline envelope, compile behavior, writer cadence, or land-refresh cadence in fp64. However, those costs are either precision-independent overheads or explicitly included in the transparent gated-fp32 benchmark provenance. Because the release documents do not claim the 2.16x number was directly measured in fp64, I do **not** consider a direct fp64 d02 72h timing required before the immutable tag.

Answer: `fp64_direct_timing_required=no`.

## Efficiency

I found no release-claim-undercutting efficiency issue in the shipped operational path.

`daily_pipeline.py` calls the operational forecast path with `force_fp64=True`. `src/gpuwrf/runtime/operational_mode.py` uses the jitted operational forecast path with `debug=False`; diagnostic and profiler variants are separate. Host materialization in the daily pipeline occurs for finite summaries, surface diagnostics, and wrfout writing at output boundaries, not as an in-timestep transfer inside the compiled timestep loop.

The known inefficiencies are real: launch/serialization tax, thousands of unfused operations, memory-bandwidth limitation, and fp64 throttle exposure after future fusion. Those are correctly framed as v0.10.0 work and do not falsify the v0.9.0 speedup claim as written.

## Handoff

Objective: perform a read-only pre-release gap analysis for v0.9.0 and decide whether any issue genuinely undercuts the claim that this is a tested stable release capable of running the backfilled Canary Islands WRF cases via the supported path.

Files changed:

- `.agent/reviews/2026-06-04-gpt-v090-gap-critic.md`
- `/tmp/gpt_v090_gapcritic.done`

Commands run:

- Read project rules and skills with `taskset -c 0-3 sed -n ...` and `taskset -c 0-3 rg ...`
- Inspected release files with `taskset -c 0-3 sed -n ... README.md docs/KNOWN_ISSUES.md src/gpuwrf/integration/daily_pipeline.py proofs/perf/compute_cycle_analysis.md`
- Inspected proof JSONs with `taskset -c 0-3 jq ... proofs/v090/*.json`
- Inspected operational runtime and dispatch code with `taskset -c 0-3 sed -n ... src/gpuwrf/runtime/operational_mode.py src/gpuwrf/coupling/scan_adapters.py`
- Searched for debug/profiling/host-transfer risks with `taskset -c 0-3 rg ...`

Proof objects produced:

- This review document.
- `/tmp/gpt_v090_gapcritic.done`

Unresolved risks:

- No direct fp64 72h d02 timing has been committed; the fp64 speedup claim is a documented inference from representative coupled-step profiling plus gated-fp32 full-pipeline timing.
- d03 1km full forecast validation remains unresolved.
- Long single-call qke numerics edge remains unresolved for some inits.
- Full historical pytest suite is not globally green.

Next decision needed:

- Manager decides whether the documented CARRY-OVER items are acceptable for tagging v0.9.0 under the principal's release bar. My recommendation is to tag.

GPT V090 GAP-CRITIC DONE: verdict=SHIP fix_now=0 carry_over=8 fp64_direct_timing_required=no
