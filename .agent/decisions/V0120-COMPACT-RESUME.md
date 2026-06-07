# v0.12.0 — COMPACT-RESUME STATE (2026-06-07 ~17:30 WEST) — READ FIRST ON RESUME

**GOAL:** Ship v0.12.0 "Standalone, Fast & Honest" — release TONIGHT (~22:00–23:30 WEST), tag `v0.12.0`, push to github.com/wrf-gpu/wrf_gpu (home=latest, v0.2.0 stays accessible). The 4.6h flock loss + 3 GPU OOM/hang detours ate the buffer but it's still feasible. Stability + honesty > completeness > deadline.

## TRUNK = `worker/opus/v0120-integration` @ `21c23e0` (worktree `.claude/worktrees/v0120-integration`)
MERGED + committed: standalone native-init CLI + nested-OOM fix (A) · release-docs v0.12.0 (B) · fail-closed scheme catalog incl. radiation silent-wrong-scheme fix (C) · persistent JIT cache (E) · PSFC height-extrap fix · perf (C: warm 16.7 s/fc-hr, cache 147→29s, graph-flag falsified) · Switzerland setup (F) + 150² bench scripts (G) · **tester-hardening (shipped scripts flock-free / canairy-path-free)** · differential-to-100% gap analysis · **the AIFS-1km GATE proof**.

## ✅ THE RELEASE-CRITICAL GATE — DONE/GREEN (commit 9204598)
24h **standalone live-nested d01→d02→d03 (1km)** on the exact prod-failing AIFS case `20260531_18z_l3_24h_...`: PIPELINE_GREEN, 24/24 wrfout per domain, all finite (T2 279-301K, PSFC 68-102 kPa), **peak 20.7 GB / 32**. Proof: `proofs/v0120/nested_24h_1km_gate.json`. The yesterday-failing AIFS-pull→1km scenario now runs out-of-box. **This is the v0.12.0 correctness item — closed.**

## RUNNING NOW (survive compact — auto-notify on completion):
1. ✅ **Coverage agent DONE + MERGED** (trunk `fd41246`): wrfout 74→**105/375** vars. The "64-var" claim was wrong (real baseline 74) → fix to 105/375 in doc-fill.
2. ✅ **Switzerland benchmark RESOLVED** (see ceiling section below): CPU 128² 28-rank = 43.28 s/fc-hr (clean); GPU 128² AND 150² both **OOM** at fp64 → ship d02 operational speedup + honest fp64 grid-ceiling note. Proof: `proofs/v0120/switzerland_128_gpu_result.json`. swiss128 branch `ef4cfdd` = reproducible harness (merge optional). NO more grid-chasing.
3. **B5 sprint — namelist recognition breadth** (in-proc agent, bg, branch `worker/opus/v0120-namelist-recognition`): honest per-key fail-closed verdicts for gwd_opt/adv_opt/cadence/out-of-scope keys. CPU-only, validator-only. NON-BLOCKING: merge into v0.12.0 only if clean+green before release-critic, else bank v0.13.
4. **B3 sprint — Noah-MP snow-layer diags** (in-proc agent, bg, branch `worker/opus/v0120-noahmp-snow`): verify-first route TSNO/SNICE/SNLIQ/ZSNSO if carry holds them, else bank v0.13. CPU-only. NON-BLOCKING.
5. **15-min heartbeat** (ScheduleWakeup, re-arm each tick) — anti-stall; drive to release.

## DEFERRED to POST-RELEASE (principal explicitly approved — Sonnet finishes + nachreichen in 0.12.x):
- **TOST n=15 (KI-5):** the powered-equivalence campaign. GPU path is BROKEN on v0.12.0 — `run_one_case_v0120.py`/`execute_daily_pipeline` fails **rc=2 per case** (`L2_D02_BLOCKED`, 0/15 → ABORT). Also hit a flock-double-wrap deadlock earlier (4.6h, my mistake — fixed by running the orchestrator DIRECTLY, no outer `/tmp/wrf_gpu_run.sh` wrap). NOT a release blocker (the gate proves correctness). Document KI-5 honestly: harness ready (`worker/sonnet/v0120-tostprep`), GPU daily_pipeline path needs the rc=2 fix, deferred to post-release Sonnet. The TOST runner still needs its internal `/tmp/wrf_gpu_run_lowprio.sh` wrapper made only-if-present (tester-safety) — on the tostprep branch, not trunk.

## Switzerland benchmark — honest grid ceiling found (RESOLVED 2026-06-07 ~17:48):
**Both 150² AND 128² fp64 single-GPU OOM** (128² peak 31.2 GB, fails to alloc the RRTMG g-point radiation temp `f64[45,128,128,16]` families). **Single-GPU fp64 grid ceiling is BELOW 128²** on this 32GB RTX 5090 w/ ~4GB desktop — refined DOWN from the earlier "~128-140²" estimate. CPU 128² 28-rank ref = **43.28 s/fc-hr** (mainloop, clean, 4800 steps). **SHIP DECISION:** report operational speedup on the production Canary d02 grid (~2.5× real-user / ~5× warm-kernel) + document the fp64 grid ceiling honestly (150²+128² OOM, RRTMG temp dominates; fp32 would lift it but detonates the acoustic solver; g-point-chunked RRTMG temp = v0.13+ lever). Did NOT chase 112² (marginal, not headline-bearing, release clock). Proof: `proofs/v0120/switzerland_128_gpu_result.json`. NOT a blocker.

## PATH TO RELEASE (remaining):
1. ✅ DONE — coverage merged (105/375).
2. ✅ DONE — Switzerland benchmark resolved (fp64 ceiling <128²; ship d02 + ceiling note). proofs/v0120/switzerland_128_gpu_result.json.
2b. Collect B5 (namelist-recognition) + B3 (noahmp-snow) bg sprints → merge IF clean+green before release-critic, else bank v0.13. NON-BLOCKING.
3. **Fill doc placeholders + fixes** in README/KNOWN_ISSUES/RELEASE_NOTES: gate ✅GREEN proof (proofs/v0120/nested_24h_1km_gate.json); **FIX "64-variable" → 105/375**; correct the FALSE "missing only stochastic+snow" claim (294 of 375 absent — see differential review); Switzerland = **honest fp64 grid-ceiling note** (150²+128² OOM, d02 operational speedup ~2.5×/~5×); TOST = DEFERRED note (KI-5); fill the 2 remaining `<<MANAGER-FILL>>` in KNOWN_ISSUES (~lines 250-251).
4. Re-check ALL gaps closed (smoke gate, tests, fail-closed, docs accurate).
5. **Final release-critic** (Opus) → **release-worker** (Opus MAX — README polish + cleanup + tag, per principal "core gatekeeper = opus maxcode"). 
6. Tag `v0.12.0` → push github.com/wrf-gpu/wrf_gpu (home=latest). → **Final short report to principal** (key changes + positive results).

## v0.13 PREP (saved): `.agent/reviews/2026-06-07-opus-v0120-differential-to-100pct.md` (committed) — the differential-to-100% feature map + cheap-wins (A1/A2 being done now; B1/B2/B3/B5 + scheme-wiring = v0.13). Architectural gaps: 2-way nesting feedback, 3D-LES km_opt=2/3/5, GWD, PD/monotonic advection, auxhist, lat-lon proj; ~58 unported scheme kernels = bulk of v1.0.0. Out-of-scope (keep): Chem/Fire/Hydro/ocean/urban/moving-nests/FDDA/stochastic.

## CONSTRAINTS: GPU one job at a time; nightly scheduler pid NEVER touch; cores 0-3 ours; window 0:3 = principal's separate codex (NEVER touch); honesty (no faked numbers / no tolerance loosening). Lessons in memory `feedback_agent_launch_tmux.md` (flock-double-wrap, anti-stall, pkill -f self-match).
