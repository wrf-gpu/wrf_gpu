# v0.12.0 — COMPACT-RESUME STATE (2026-06-07 ~17:30 WEST) — READ FIRST ON RESUME

**GOAL:** Ship v0.12.0 "Standalone, Fast & Honest" — release TONIGHT (~22:00–23:30 WEST), tag `v0.12.0`, push to github.com/wrf-gpu/wrf_gpu (home=latest, v0.2.0 stays accessible). The 4.6h flock loss + 3 GPU OOM/hang detours ate the buffer but it's still feasible. Stability + honesty > completeness > deadline.

## TRUNK = `worker/opus/v0120-integration` @ `21c23e0` (worktree `.claude/worktrees/v0120-integration`)
MERGED + committed: standalone native-init CLI + nested-OOM fix (A) · release-docs v0.12.0 (B) · fail-closed scheme catalog incl. radiation silent-wrong-scheme fix (C) · persistent JIT cache (E) · PSFC height-extrap fix · perf (C: warm 16.7 s/fc-hr, cache 147→29s, graph-flag falsified) · Switzerland setup (F) + 150² bench scripts (G) · **tester-hardening (shipped scripts flock-free / canairy-path-free)** · differential-to-100% gap analysis · **the AIFS-1km GATE proof**.

## ✅ THE RELEASE-CRITICAL GATE — DONE/GREEN (commit 9204598)
24h **standalone live-nested d01→d02→d03 (1km)** on the exact prod-failing AIFS case `20260531_18z_l3_24h_...`: PIPELINE_GREEN, 24/24 wrfout per domain, all finite (T2 279-301K, PSFC 68-102 kPa), **peak 20.7 GB / 32**. Proof: `proofs/v0120/nested_24h_1km_gate.json`. The yesterday-failing AIFS-pull→1km scenario now runs out-of-box. **This is the v0.12.0 correctness item — closed.**

## RELEASE TRUNK = `e602122` (post B5+B3 merge). DONE this push:
1. ✅ Coverage 74→**105/375** merged. 2. ✅ Switzerland resolved (fp64 ceiling <128²; ship d02 + ceiling note; proofs/v0120/switzerland_128_gpu_result.json). 3. ✅ **B5 merged** (namelist recognition, 71 passed — honest per-key verdicts; naive-user-test fix). 4. ✅ **B3 merged** (Noah-MP snow/canopy diags, KI-3 CLOSED, +4 vars).

## LANES — outcomes (all off e602122, isolated, proof-verified by manager before merge):
- ✅ **B1 radiation flux** MERGED (wrfout 120→130 vars; schema-exact, consistency-verified).
- ✅ **GWD kernel** MERGED `076b5aa` (bl_gwdo_run port, pristine-WRF Fortran ORACLE pass, 10 tests, **default-OFF** = no-op unless gwd_opt==1 AND gwdo_statics present → validated-path byte-unchanged; gwd_opt {0}→{0,1}).
- ⏳ **GWD coupling-validation RUNNING** `worker/opus/gwd-coupling-validation` (agent a5334e1b, maxcode) — wires GWDOStatics from real geo_em + short real-case GPU run → proves gwd_opt=1 runs finite+physical. **DECIDES: keep gwd_opt IMPLEMENTED in v0.12.0 (if validated) ELSE manager DEMOTES gwd_opt→warn + banks operational GWD to v0.13.** Critical path.
- → **2-way nesting** BANKED v0.13 `dfab32c` (~85%, WRF sm121 smoother + --feedback + CPU-validated TWO_WAY_FEEDBACK_VALIDATED; remaining = 24h real-GPU equivalence).
- → **PD/mono advection** BANKED v0.13 `017e6c1` (~90%, WRF advect_scalar_pd/_mono FCT port, 14 tests + WRF-Fortran-parity <1e-12, default-path byte-identical Straka/Skamarock 6/6; remaining = operational RK3 wiring + real-GPU). NOTE: WRF canonical scalar_adv_opt 1=PD/2=mono (brief had it swapped).
- GPU = ONE job, flock single-wrap. GPT-codex = reserve debugger. 15-min heartbeat = anti-stall + active liveness check.

## REMAINING TO RELEASE (after GWD coupling settles):
1. 🔴 Naive-user namelist fix (cudt/bldt → WARN+run; gwd_opt per coupling outcome; keep hard-fail for reference-only/moist_adv_opt/out-of-scope; mirror pipeline case-prep; WITH proof test) — maxcode.
2. Doc-fill: gate-GREEN proof, **64→130/375** var fix, correct false "missing-only stochastic+snow" claim, Switzerland honest fp64-ceiling, TOST-DEFERRED, v0.13-banked list (nesting/PD/+ GWD-operational if demoted).
3. Gap re-check (full test suite green on trunk, fail-closed honest, docs accurate). 4. Release-critic (Opus). 5. Release-worker (Opus-MAX: README+cleanup+tag v0.12.0). 6. Push github.com/wrf-gpu/wrf_gpu (home=latest). 7. Final report.

## v0.12.0 RELEASE proceeds IN PARALLEL on `e602122` (NOT blocked by the 4 lanes): doc-fill → gap re-check → release-critic (Opus) → release-worker (Opus-MAX) → tag → push. At ~midnight: assess each lane → merge clean+validated ones into v0.12.0, else they become v0.13's head start.

## DEFERRED to POST-RELEASE (principal explicitly approved — Sonnet finishes + nachreichen in 0.12.x):
- **TOST n=15 (KI-5):** the powered-equivalence campaign. GPU path is BROKEN on v0.12.0 — `run_one_case_v0120.py`/`execute_daily_pipeline` fails **rc=2 per case** (`L2_D02_BLOCKED`, 0/15 → ABORT). Also hit a flock-double-wrap deadlock earlier (4.6h, my mistake — fixed by running the orchestrator DIRECTLY, no outer `/tmp/wrf_gpu_run.sh` wrap). NOT a release blocker (the gate proves correctness). Document KI-5 honestly: harness ready (`worker/sonnet/v0120-tostprep`), GPU daily_pipeline path needs the rc=2 fix, deferred to post-release Sonnet. The TOST runner still needs its internal `/tmp/wrf_gpu_run_lowprio.sh` wrapper made only-if-present (tester-safety) — on the tostprep branch, not trunk.

## Switzerland benchmark — honest grid ceiling found (RESOLVED 2026-06-07 ~17:48):
**Both 150² AND 128² fp64 single-GPU OOM** (128² peak 31.2 GB, fails to alloc the RRTMG g-point radiation temp `f64[45,128,128,16]` families). **Single-GPU fp64 grid ceiling is BELOW 128²** on this 32GB RTX 5090 w/ ~4GB desktop — refined DOWN from the earlier "~128-140²" estimate. CPU 128² 28-rank ref = **43.28 s/fc-hr** (mainloop, clean, 4800 steps). **SHIP DECISION:** report operational speedup on the production Canary d02 grid (~2.5× real-user / ~5× warm-kernel) + document the fp64 grid ceiling honestly (150²+128² OOM, RRTMG temp dominates; fp32 would lift it but detonates the acoustic solver; g-point-chunked RRTMG temp = v0.13+ lever). Did NOT chase 112² (marginal, not headline-bearing, release clock). Proof: `proofs/v0120/switzerland_128_gpu_result.json`. NOT a blocker.

## PATH TO RELEASE (remaining):
1. ✅ DONE — coverage merged (105/375).
2. ✅ DONE — Switzerland benchmark resolved (fp64 ceiling <128²; ship d02 + ceiling note). proofs/v0120/switzerland_128_gpu_result.json.
2b. Collect B5 (namelist-recognition) + B3 (noahmp-snow) bg sprints → merge IF clean+green before release-critic, else bank v0.13. NON-BLOCKING.
3. **Fill doc placeholders + fixes** in README/KNOWN_ISSUES/RELEASE_NOTES: gate ✅GREEN proof (proofs/v0120/nested_24h_1km_gate.json); **FIX "64-variable" → 105/375**; correct the FALSE "missing only stochastic+snow" claim (294 of 375 absent — see differential review); Switzerland = **honest fp64 grid-ceiling note** (150²+128² OOM, d02 operational speedup ~2.5×/~5×); TOST = DEFERRED note (KI-5); fill the 2 remaining `<<MANAGER-FILL>>` in KNOWN_ISSUES (~lines 250-251).
4. Re-check ALL gaps + **🔴 CONFIRMED RELEASE-CRITICAL NAIVE-USER FIX (do AFTER GWD lane settles, ONE pass):** real Canary prod namelists (wrf_l3 + wrf_l2) ALL have `cudt=5,5,5` + `gwd_opt=1`. Standalone CLI calls `validate_operational_namelist` (cli.py:395) → post-B5 RAISES on cudt=5 AND gwd_opt=1 → **a naive user pointing `gpuwrf run` at a real namelist is now REJECTED**. (24h gate ran green only because the operational pipeline has a case-prep that rewrites these; the standalone CLI lacks it.) FIX: downgrade `cudt`/`bldt` cadence + `gwd_opt=1`(if GWD not implemented tonight) from hard-raise → **WARN+run** (cumulus/PBL every-step, run-without-GWD — exactly what the pipeline case-prep already does). KEEP hard fail-closed for genuine wrong-substitution: reference-only schemes (RRTM/Dudhia/MYJ/New-Tiedtke), `moist_adv_opt`/`scalar_adv_opt` 2/3/4, out-of-scope features. Mirror the pipeline case-prep so standalone CLI == pipeline. Proof test: feed a REAL Canary namelist (cudt=5,gwd_opt=1) → PROCEEDS with warnings; moist_adv_opt=2 / a reference-only scheme → STILL raises. Sequence AFTER GWD (both touch scheme_catalog gwd_opt — avoid merge conflict). This is the SAME rc=2 (L2_D02_BLOCKED) seen in TOST + nesting GPU smoke.
NESTING lane DONE ~85% (`worker/opus/nesting-2way-feedback` @ dfab32c): WRF sm121 smoother + --feedback wiring + CPU-validated (TWO_WAY_FEEDBACK_VALIDATED, 31 tests); defaults OFF; rec v0.13 (24h real-GPU equivalence remaining). Zero trunk risk.
5. **Final release-critic** (Opus) → **release-worker** (Opus MAX — README polish + cleanup + tag, per principal "core gatekeeper = opus maxcode"). 
6. Tag `v0.12.0` → push github.com/wrf-gpu/wrf_gpu (home=latest). → **Final short report to principal** (key changes + positive results).

## v0.13 PREP (saved): `.agent/reviews/2026-06-07-opus-v0120-differential-to-100pct.md` (committed) — the differential-to-100% feature map + cheap-wins (A1/A2 being done now; B1/B2/B3/B5 + scheme-wiring = v0.13). Architectural gaps: 2-way nesting feedback, 3D-LES km_opt=2/3/5, GWD, PD/monotonic advection, auxhist, lat-lon proj; ~58 unported scheme kernels = bulk of v1.0.0. Out-of-scope (keep): Chem/Fire/Hydro/ocean/urban/moving-nests/FDDA/stochastic.

## CONSTRAINTS: GPU one job at a time; nightly scheduler pid NEVER touch; cores 0-3 ours; window 0:3 = principal's separate codex (NEVER touch); honesty (no faked numbers / no tolerance loosening). Lessons in memory `feedback_agent_launch_tmux.md` (flock-double-wrap, anti-stall, pkill -f self-match).
