# v0.12.0 — COMPACT-RESUME STATE (2026-06-07 ~17:30 WEST) — READ FIRST ON RESUME

**GOAL:** Ship v0.12.0 "Standalone, Fast & Honest" — release TONIGHT (~22:00–23:30 WEST), tag `v0.12.0`, push to github.com/wrf-gpu/wrf_gpu (home=latest, v0.2.0 stays accessible). The 4.6h flock loss + 3 GPU OOM/hang detours ate the buffer but it's still feasible. Stability + honesty > completeness > deadline.

## TRUNK = `worker/opus/v0120-integration` @ `21c23e0` (worktree `.claude/worktrees/v0120-integration`)
MERGED + committed: standalone native-init CLI + nested-OOM fix (A) · release-docs v0.12.0 (B) · fail-closed scheme catalog incl. radiation silent-wrong-scheme fix (C) · persistent JIT cache (E) · PSFC height-extrap fix · perf (C: warm 16.7 s/fc-hr, cache 147→29s, graph-flag falsified) · Switzerland setup (F) + 150² bench scripts (G) · **tester-hardening (shipped scripts flock-free / canairy-path-free)** · differential-to-100% gap analysis · **the AIFS-1km GATE proof**.

## ✅ THE RELEASE-CRITICAL GATE — DONE/GREEN (commit 9204598)
24h **standalone live-nested d01→d02→d03 (1km)** on the exact prod-failing AIFS case `20260531_18z_l3_24h_...`: PIPELINE_GREEN, 24/24 wrfout per domain, all finite (T2 279-301K, PSFC 68-102 kPa), **peak 20.7 GB / 32**. Proof: `proofs/v0120/nested_24h_1km_gate.json`. The yesterday-failing AIFS-pull→1km scenario now runs out-of-box. **This is the v0.12.0 correctness item — closed.**

## RUNNING NOW (survive compact — auto-notify on completion):
1. **Coverage agent `a3260739c3cc0784d`** — wrfout A1+A2: routes ~30 device-resident grid-metric/coord arrays + 5 diagnostics into the writer → raises coverage **74→~104+/368**. Worktree `.claude/worktrees/v0120-wrfout-cov` (branch worker/opus/v0120-wrfout-cov). Reports exact new var count → MANAGER then fixes the stale doc claim. STILL RUNNING.
2. **Switzerland-128² CPU ref — DETACHED PID `1171820`** (NOT an agent; the agent a26dbe23 finished + committed `ef4cfdd` on worker/opus/v0120-swiss128). 28-rank CPU-WRF 128², ~15-20 min, writes `/mnt/data/wrf_gpu_switzerland_128/run_cpu/cpu_timing.json` on completion. 128² inputs VALID. **GPU run is the MANAGER's next GPU job (GPU is FREE now — TOST failed/done):**
   `CASE_ROOT=/mnt/data/wrf_gpu_switzerland_128 DOMAIN=d01 HOURS=24 CPU_WALL_BASIS=mainloop PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform bash scripts/equivalence_switzerland.sh` (from a swiss128 worktree). Memory fits (~24.6 GB, ~3.4 GB headroom; if RESOURCE_EXHAUSTED → 112²). Speedup = CPU(28-rank,128²,mainloop/fcst-hr) / GPU(128²,warm,fp64,/fcst-hr). Warm = run twice.
3. **15-min heartbeat** (ScheduleWakeup, re-arm each tick) — anti-stall: verify the current GPU job is progressing; drive to release.

## DEFERRED to POST-RELEASE (principal explicitly approved — Sonnet finishes + nachreichen in 0.12.x):
- **TOST n=15 (KI-5):** the powered-equivalence campaign. GPU path is BROKEN on v0.12.0 — `run_one_case_v0120.py`/`execute_daily_pipeline` fails **rc=2 per case** (`L2_D02_BLOCKED`, 0/15 → ABORT). Also hit a flock-double-wrap deadlock earlier (4.6h, my mistake — fixed by running the orchestrator DIRECTLY, no outer `/tmp/wrf_gpu_run.sh` wrap). NOT a release blocker (the gate proves correctness). Document KI-5 honestly: harness ready (`worker/sonnet/v0120-tostprep`), GPU daily_pipeline path needs the rc=2 fix, deferred to post-release Sonnet. The TOST runner still needs its internal `/tmp/wrf_gpu_run_lowprio.sh` wrapper made only-if-present (tester-safety) — on the tostprep branch, not trunk.

## Switzerland benchmark — honest grid ceiling found:
150² fp64 single-GPU **OOMs** (RRTMG g-point radiation temporary = 25.58 GiB single alloc; even platform allocator can't fit it on 32GB w/ ~4GB desktop). **Single-GPU fp64 grid ceiling ≈ 128–140².** 28-rank CPU ref at 150² = 61.0 s/fc-hr (done) but GPU can't match 150² → 128² rebuild in flight (agent a26dbe23). If 128² GPU run works → ship honest speedup (GPU 128² warm fp64 vs 28-rank 128², per-fc-hr). Else → ship Canary/d02 perf (~2.5× real-user / ~5× warm-kernel) + document the fp64 ceiling. NOT a blocker.

## PATH TO RELEASE (remaining):
1. Collect coverage agent (a3260739) → merge → get new var count.
2. Collect Switzerland-128² agent (a26dbe23) → run the **128² GPU benchmark** (GPU, ~15min, `XLA_PYTHON_CLIENT_ALLOCATOR=platform`) → compare → speedup. Merge.
3. **Fill doc placeholders + fixes** in README/KNOWN_ISSUES/RELEASE_NOTES: gate ✅GREEN proof; Switzerland speedup (128²) or honest ceiling note; **FIX the stale "64-variable" claim → accurate new count (~104/368)**; KI-3 "missing only stochastic+snow" is FALSE → correct; TOST = DEFERRED note (KI-5); fill the 2 remaining `<<MANAGER-FILL>>` in KNOWN_ISSUES (~lines 250-251).
4. Re-check ALL gaps closed (smoke gate, tests, fail-closed, docs accurate).
5. **Final release-critic** (Opus) → **release-worker** (Opus MAX — README polish + cleanup + tag, per principal "core gatekeeper = opus maxcode"). 
6. Tag `v0.12.0` → push github.com/wrf-gpu/wrf_gpu (home=latest). → **Final short report to principal** (key changes + positive results).

## v0.13 PREP (saved): `.agent/reviews/2026-06-07-opus-v0120-differential-to-100pct.md` (committed) — the differential-to-100% feature map + cheap-wins (A1/A2 being done now; B1/B2/B3/B5 + scheme-wiring = v0.13). Architectural gaps: 2-way nesting feedback, 3D-LES km_opt=2/3/5, GWD, PD/monotonic advection, auxhist, lat-lon proj; ~58 unported scheme kernels = bulk of v1.0.0. Out-of-scope (keep): Chem/Fire/Hydro/ocean/urban/moving-nests/FDDA/stochastic.

## CONSTRAINTS: GPU one job at a time; nightly scheduler pid NEVER touch; cores 0-3 ours; window 0:3 = principal's separate codex (NEVER touch); honesty (no faked numbers / no tolerance loosening). Lessons in memory `feedback_agent_launch_tmux.md` (flock-double-wrap, anti-stall, pkill -f self-match).
