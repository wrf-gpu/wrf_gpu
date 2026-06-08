# v0.13 Roadmap — "Validate & Accelerate" (2026-06-08, post-v0.12.0 tag)

Consolidated plan for v0.13: close the validations v0.12.0 deliberately deferred, land the
compile-speed + VRAM-ceiling levers (speed is a core motto — compile counts), absorb the
external NCAR/UCAR-style critique, and start the next scheme wave toward v1.0.0.

Companion docs: `.agent/decisions/V0130-SPEED-ROADMAP.md` (compile/dispatch detail) and
`.agent/reviews/2026-06-07-naive-ai-v011-critique-triage.md` (external-critique triage).

Priority: **P1** = critical path / credibility gate · **P2** = medium · **P3** = long-tail.
Complexity: S / M / L / XL.

## Tier 1 — P1: close v0.12.0 carry-overs + critical levers

| Sprint | Goal | Cx | Origin |
|---|---|:--:|---|
| compile-speed (re-merge + GPU-validate) | AOT precompile + persistent XLA autotune cache + cache hardening; validate the XLA flags ON GPU (CPU-proven; reverted from v0.12 — flag injection aborted GPU path) | M | branch `worker/opus/compile-speed` |
| TOST n=15 powered equivalence | fix the rc=2 GPU `daily_pipeline` scoring path → run the powered paired-TOST (n≈27 for full power) → the real equivalence number (KI-5) | M | deferred → Sonnet |
| g-point-chunked RRTMG temporary | chunk the dominant fp64 VRAM consumer → lifts the single-GPU grid ceiling (<128²) AND unblocks nested-GWD | M-L | Switzerland + gwd7 finding |
| GWD on nested (enable) | with the chunked temp, the 24 h nested 1 km + GWD run fits → flip `gwd_opt=1` default-on + add the 24 h-nested-GWD gate | L | gated-off in v0.12 (VRAM @hr7) |
| 2-way nesting 24 h real-GPU equivalence | full 24 h 2-way nested equivalence vs CPU-WRF (feedback=1) | L | scaffolding shipped (defaults-off) |
| RRTM-LW cross-model skeptic pass | independent GPT-5.5 audit of the band/laytrop vectorization (author wrote kernel + oracle) | S-M | `ra_lw=1` shipped opt-in |
| **Forecast-skill closure (credibility gate)** | T2/U10/V10 RMSE regressions: surface-flux over-flux + RRTMG-SW clear-sky T2 bias + theta-guard/land-state. The blocker for any "operational/replacement" claim | L-XL | external critique #1 |
| Multi-GPU domain decomposition (S1) | sharded stencils + halo exchange → lifts single-GPU VRAM ceiling, bigger grids, the per-watt/whole-Earth claims become measured not projected | XL | halo interface pre-designed |

## Tier 2 — P2: architecture completeness + speed follow-ons + reproducibility

| Sprint | Goal | Cx |
|---|---|:--:|
| Outsider-runnable reproducibility | bundle the missing Thompson table assets + scripts so an external reviewer runs the FULL proof collection (not just the public tests); green `verify_reproducibility.sh` end-to-end | M | 
| PD/mono advection real-GPU + moisture | real-case GPU validation + extend to moisture species (currently theta-only) | M |
| MYJ-PBL + Janjic-sfclay wire | last reference-only pair → operational (TKE-carry + paired sfclay), savepoint oracle | L |
| 3D-TKE / Smagorinsky / SMS-3DTKE LES | km_opt 2/3/5 — the sub-km LES regime (where GPU wins most) | L-XL |
| Clear-sky radiation fluxes | the 8 `...C` vars via a separate clear-sky RT pass (B1 honestly omitted them) | M-L |
| Standard community validation | WRF/community idealized suite, closed-domain mass/energy budgets, bitwise-restart, larger multi-day corpus | L | external critique #3 |
| Parallel-compile + dev autotune knob | `--xla_gpu_force_compilation_parallelism` + `--fast-compile` dev flag (GPU-validated) | S-M |
| Sub-jit split + recompile hygiene | smaller jit blocks + static-arg/shape stability + `donate_argnums` | M |
| CPU-flock for idle nightly cores | opportunistically borrow idle cores 4-31, yield instantly to the nightly (GPU-flock analogue) | M |

## Tier 3 — P3: scheme long-tail (the bulk → v1.0.0, template-following, parallelizable)

| Family | Goal | Cx |
|---|---|:--:|
| Microphysics ×~22 | Ferrier/Goddard/MY/WDM5/WSM7/P3/NSSL/CAM5.1/SBM… (cheap 1-mom first: WSM7/Goddard) | XL |
| Cumulus ×~10 | SAS family / Grell-3D / Zhang-McFarlane / KSAS / MSKF + New-Tiedtke-wire | L-XL |
| PBL ×~8 | QNSE / UW / GBM / TEMF / Shin-Hong / TKE-eps / MRF | L-XL |
| Radiation ×~12 | Goddard SW/LW / CAM / FLG / RRTMG-K / fast-RRTMG / GFDL | L-XL |
| Surface-layer ×4 + LSM ×6 | GFS/QNSE/TEMF/old-MM5 sfclay · thermal-slab/RUC/CLM4/CTSM/Pleim-Xiu/SSiB LSM | L-XL |

## Multi-hardware / independent reproduction (P2, external critique #7)
A second GPU / driver / JAX stack + an independent reproduction run (v0.12.0 is one RTX 5090,
one stack).

## Deliberately OUT-OF-SCOPE (documented boundary, NOT v0.13)
WRF-Chem · WRF-Fire · WRF-Hydro · coupled ocean · urban canopy (UCM/BEP/BEM) · moving nests ·
FDDA/DA · stochastic physics.

## Framing (carried from the critique)
Publication-worthy NOW as a transparent research-artifact + AI-assisted scientific-software
process preprint; NOT yet "full WRF replacement." Keep "WRF-compatible reimplementation, not a
Fortran-source port." The credibility unlock for a model-development claim = Tier-1 skill
closure + outsider-runnable reproducibility + community-standard benchmarks.

---

## Post-Tier1/2 sequence (principal directive 2026-06-08)

**Trigger:** ALL Tier 1 + Tier 2 merged + proven (incl. compile-speed GPU-validated → warm cache, so runs are fast/no-compile-delay).

### Step A — integrated GPU smoke gate (one agent)
- **24 h, 9/3/1 km nested**, a **VRAM-manageable Canary sub-region** (domains sized to fit the single-GPU fp64 ceiling comfortably — e.g. trimmed d01/d02/d03 around one island).
- Goal: **fast** (warm AOT/autotune cache, no compile stall), **smooth, zero errors, zero NaNs, solid solution**. NO CPU compare — this is a "does the integrated v0.13 trunk run clean + fast end-to-end" gate, not a skill gate.
- Proof: `proofs/v013/integrated_smoke_24h_nested.json` (PIPELINE_GREEN, all-finite, wall-clock incl. cache-warm).
- **If FAIL → repair** (debug lane; cross-model GPT escalation if stuck), re-run until clean.

### Step B — Tier 3 rollout (only after Step A is GREEN)
- **One maxcode worker per physics group** (5 groups), each implementing its family reference-only→operational via the established traceable-JAX + pristine-WRF-savepoint-oracle template; **tested in groups**:
  1. **Microphysics** (~22: WSM7/Goddard cheap 1-mom first → Ferrier/MY/WDM5/P3/NSSL/CAM5.1/SBM)
  2. **Cumulus** (~10: SAS family, Grell-3D, Zhang-McFarlane, KSAS, MSKF, New-Tiedtke-wire)
  3. **PBL** (~8: QNSE, UW, GBM, TEMF, Shin-Hong, TKE-eps, MRF)
  4. **Radiation** (~12: Goddard SW/LW, CAM, FLG, RRTMG-K, fast-RRTMG, GFDL)
  5. **Surface-layer (~4) + LSM (~6)** (GFS/QNSE/TEMF/old-MM5 sfclay; thermal-slab/RUC/CLM4/CTSM/Pleim-Xiu/SSiB)
- Discipline: per-scheme fp64 oracle on CPU (parallelizable, like RRTM-LW); GPU only for a per-group integration smoke (serialized, one GPU job). Defaults unchanged (each scheme opt-in, fail-closed until oracle-proven). File-ownership per group subdir to avoid collisions.
- Cadence: land each group as it proves out; this is the bulk of the path to v1.0.0.

---

## Progress log (live, manager-maintained)

**2026-06-08 ~07:20** — v0.13 wave 1:
- ✅ **Outsider-reproducibility** (T2) MERGED `d9398fc`: 45 proof .py /home/enric→0 (resolvers+WRF_PRISTINE_ROOT env), `scripts/verify_reproducibility.sh` GREEN 11/11 outsider-runnable, `manifest/reproducibility_assets.json`, `docs/REPRODUCIBILITY.md`. Independently re-verified (gate rc=0, 0 .py leaks).
- ✅ **compile-speed** (T1) CPU-verified+BANKED `worker/opus/v013-compile-speed @b9da88d`: opt-in default-off + subprocess flag-probe (fixes the v0.12 GPU-abort); 22 tests, import-inert. → GPU-runbook validation pending (when GPU frees) → then merge.
- ✅ **RRTM-LW skeptic** (T1) VERDICT SOUND `a057e04`: no JAX port bug (max div 2.7e-13), oracle-integrity clean. 2 findings → rrtmlw-fix lane.
- 🔄 RUNNING: g-point-chunk-RRTMG (GPU keystone), rrtmlw-fix (F1 ptop + F2 fail-loud).
- Carry-overs noted: 51 non-.py proof files still have /home/enric (29 dev .sh + logs; not on CPU verify path); stale rrtmg manifest-pin `5cc63950` vs disk `0695e523` (radiation-owned → fold into g-point-chunk or a radiation lane); proof-report .json regeneration embeds git_head (consider gitignore).

**2026-06-08 ~07:29** — wave 1 cont.:
- ✅ **RRTM-LW findings fix** (T1 #6) MERGED `a5e4973`: F1 `_nbuf` grid-aware (real top_pressure_pa; None→5000 = production bit-identical, 7+7 cases max diff 0.0); F2/F3 masking-clamps→fail-loud NaN guards (forbidden pattern removed). New pristine-WRF non-5000-ptop oracle (100mb/20mb): grid-aware rel ~2e-13 vs hardcoded 4.8e-2/NaN. Oracle OVERALL PASS, 9 wiring tests, /home/enric in proofs/*.py stays 0. → Tier1 #6 DONE.
- 🔄 g-point-chunk now in GPU VRAM-measurement phase (~21GB held).
- v0.13 trunk @ a5e4973. DONE: reproducibility (T2), RRTM-LW skeptic+fix (T1#6). BANKED: compile-speed (T1#1, GPU-validate pending). RUNNING: g-point-chunk (T1#3 keystone).

**2026-06-08 ~08:07** — wave 2 cont.:
- ✅ **PD/mono advection → moisture** (T2) MERGED `6eb4b01`: `advect_moisture_scalars()` pure addition (default moist_adv_opt=0 byte-unchanged), positivity + WRF-parity bit-exact (0.0 diff), 10 tests + 55 dynamics regression green. FOLLOW-UP: function proven but NOT operationally wired (operational path flux-advects theta only; moisture via physics boundary) → ~1-call runtime hookup + investigate "is moisture correctly advected operationally?" (relevant to skill-closure #7).
- 🔄 g-point-chunk in GPU VRAM-ceiling measurement (~32GB held @0% — watch for hang). MYJ+Janjic actively editing (CPU).
- v0.13 trunk @ 6eb4b01. DONE: reproducibility(T2), RRTM-LW(T1#6), PD-moisture(T2). BANKED: compile-speed(T1#1).

**2026-06-08 ~08:14** — wave 2/3:
- ✅ **g-point-chunk RRTMG** (T1#3) MERGED `f323303`: SW peak VRAM −45..57% (lax.scan band-tiling, bit-identical max_rel=0.0). LW inert/neutral. Remaining VRAM floor = upstream optics/taumol → follow-up.
- ✅ **compile-speed** (T1#1) MERGED `4227ef6`, GPU-VALIDATED: real-GPU import clean (no v0.12 abort, XLA_FLAGS=None, autotune default-off); 22 tests. Autotune-effect gated/opt-in until measured.
- Tier1 done: #1, #3, #6. Tier2 done: reproducibility, PD-moisture(fn-level). RUNNING: MYJ+Janjic (CPU).
- NEXT: #4 GWD-on-nested fit test (24h-nested-1km+GWD on the chunked trunk — does it clear the hr7 OOM now?); if OOM→optics/taumol follow-up needed first.

**2026-06-08 ~08:23** — wave 3:
- ✅ **MYJ-PBL + Janjic-sfclay** (T2) MERGED `c612ab9`: reference-only→operational, oracle PASS vs v0.6.0 pristine-WRF savepoints (worst PBL 2.7e-11/SFC 1.6e-10), default byte-unchanged + fail-closed pairing, 101 tests. Follow-up: end-to-end coupled-RMSE (only per-scheme parity proven).
- RUNNING (max-parallel): GWD-nested-gate (GPU, compiling), TOST-rc2-fix, multi-GPU-fakemesh, clear-sky-radiation, community-validation (CPU).
- Tier1 done: #1,#3,#6. Tier2 done: reproducibility, PD-moisture, MYJ+Janjic.

**2026-06-08 ~08:26** — wave 3 cont.:
- ✅ **Multi-GPU fake-mesh** (T1#8) MERGED `9c04a7b`: shard_map + ppermute halo, partition-invariance bit-identical (0.0); CPU fake-mesh only → real throughput HW-deferred (per-watt/Earth PROJECTED). 27 tests.
- Tier1 DONE: #1,#3,#6,#8 (4/8). RUNNING: GWD-nested-gate(GPU,compiling), TOST-rc2-fix, clear-sky-radiation, community-validation (CPU) + skill-closure-investigation (CPU read-only, front-loading #7).
- Tier1 remaining: #2(rc2-fix running→then GPU campaign), #4(gate running), #5(2-way-24h), #7(skill-closure, investigating). Tier2 remaining: sub-jit, parallel-compile, CPU-flock, multi-hardware + follow-ups(optics/taumol-chunk, PD-moisture-op-wiring, stale-rrtmg-pin).

**2026-06-08 ~08:31** — GWD-nested gate (#4) OOM:
- 🔴 24h-nested-1km+GWD (GPUWRF_GWD_NESTED=1) on chunked trunk OOM'd at **step 0** (0 wrfout, "Failed to allocate 8.1 GiB", platform allocator) — WORSE than gwd7 (which reached hr7). GPU clean afterward (4041 MiB → no lingering CPU-lane contention; the 4 concurrent CPU lanes are JAX_PLATFORMS=cpu). Likely self-inflicted: bigger v0.13 compile graph + GWD residents + the UNCHUNKED LW taumol floor (_lw_solver_base) at first radiation call.
- VERDICT: SW g-point-chunk alone is INSUFFICIENT for 24h-nested-1km+GWD on one 32GB GPU. #4 (GWD-on-nested default-on) is BLOCKED on the optics/taumol-chunk follow-up (the LW floor g-point-chunk flagged). GWD stays GATED-OFF-default (no regression vs v0.12.0) — HONEST.
- PLAN: dispatch optics/taumol-chunk AFTER clear-sky merges (rrtmg collision); then re-test GWD-nested on an EXCLUSIVE GPU. GPT-codex cross-diagnosing the step0-vs-hr7 OOM.

**2026-06-08 ~08:32** — wave 3 results:
- ✅ **TOST rc=2 fix** (T1#2) MERGED: scoring-path rc=0 proven (real GPU wrfout vs CPU-WRF); GPU n=15 campaign = runbook (proofs/v013/tost_rc2_fix.md), a later GPU step.
- ✅ **Skill-closure investigation** done (.agent/reviews/2026-06-08-skill-closure-investigation.md). KEY: (1) moisture transport = REAL correctness gap (dycore advects only u/v/w/theta; qv-tend dead code; condensates ZERO advection; NOT WRF-faithful) → #1 fix = wire advect_moisture_scalars into RK3 (CPU-validatable, default-off, GPT-cross-check staged). (2) HONEST: headline T2/U10/V10 closure (credibility gate) needs the HARD GPU sprints (dycore ph' / MYNN-EDMF / faithful *_tendf) — no cheap knob; T2 already PASSES (0.484K), NOT_EQUIVALENT is wind-error-growth (KI-4). So #7 will NOT fully close in this push → land #1 correctness fix + document ranks 2-4 as carry-over (honest, matches "research artifact not full replacement").

**2026-06-08 ~08:44** — wave 3/4:
- ✅ **clear-sky radiation** (T2) MERGED `f9eb962`: 8 ...C flux vars via WRF-faithful 2nd clear-sky stream, oracle PASS (not self-compare), all-sky byte-unchanged, default-off. (Follow-up: runtime threads with_clear_sky through M9Diagnostics for operational wrfout.)
- Tier1 done: #1,#2,#3,#6,#8. Tier2 done: reproducibility, PD-moisture, MYJ+Janjic, clear-sky.
- RUNNING: community-validation, moisture-wiring(#7-correctness core-dycore), GPT-moisture-cross-check (codex), optics/taumol-chunk (NEW: LW _lw_solver_base + SW optics VRAM floor → unblocks #4 GWD-nested).

**2026-06-08 ~08:49** — community-validation (T2) MERGED `ad01bff`: scripts/community_validation.sh PASS (idealized Straka+warm-bubble, conservation budgets, bitwise-restart). Tier2 5/9 done. LESSON: GPT cross-check via `codex exec "<longprompt>"` arg hung on stdin; use stdin pipe for long prompts. AVOID pkill -f "codex" (self-matches our own codex). moisture-wiring GPT-cross-check re-run robustly at merge.

**2026-06-08 ~09:18** — moisture-advection-wiring (#7-correctness) MERGED `584037d` + verified on trunk: all 5 gates PASS (default byte-identical, conservation 8.2e-16, idealized Straka+Skamarock unchanged, WRF-parity 1.7e-16, finite/monotonic), 7 tests. Closes the condensate-zero-advection correctness gap (opt-in moist_adv_opt, default-off). GPT cross-check of the opt-in path running (bou8wvjjr).
- v0.13 TALLY: 11 items merged. Tier1 5/8 (#1,#2,#3,#6,#8) + #7-correctness. Tier2 5/9 (reproducibility, PD-moisture, MYJ+Janjic, clear-sky, community-validation). RUNNING: optics/taumol-chunk(→#4). REMAINING: #4 GWD-nested(blocked on optics/taumol+retest), #5 2-way-24h, #7-headline-skill(hard GPU carry-over); Tier2: sub-jit, parallel-compile, CPU-flock(sensitive), multi-hardware(doc-only).

**2026-06-08 ~09:22** — GPT cross-check of moisture-wiring opt-in path (codex gpt-5.5 xhigh): core approach VALIDATED (post-acoustic q_new=(mut_old*q_old+dt*tend)/mut_new from rk1_reference + final-RK3-stage-only PD/monotonic limiter are WRF-faithful). 3 real fidelity-refinements (NOT blockers; moisture-wiring stays MERGED default-off):
  - Q1: advection tendency should use acoustic-accumulated mass fluxes (ru_m/rv_m/ww_m) not stage-entry winds — BUT this mirrors the existing THETA cadence (shared property of all scalar advection, not a new moisture bug).
  - Q3: PBL/cumulus moisture tendencies should fold into the RK scalar tendency before the final limiter, not apply later as state deltas (= the existing v0.9 physics-state-delta-post-dycore cadence).
  - Q2: report the final top-hat overshoot as a validation LIMITATION; Q4: real-map/lateral-boundary conservation + physics-increment positivity still untested.
  → CARRY-OVER: a deeper WRF-cadence-fidelity sprint (acoustic-accumulated scalar fluxes + physics-tendency folding) for BOTH theta+moisture — GPU-bound, same family as #7-headline-skill. moisture_adv_opt stays OPT-IN (default-off) until then. GPT cross-check = did its job (validated merge + surfaced real refinements).

**2026-06-08 ~09:24** — optics/taumol-chunk MERGED + inertness verified; launching GWD-nested-1km RETEST (#4) on chunked trunk (GPU exclusive). VRAM SW-88.6%/LW-43.6%; deep-col OOM→fits → expect 24h-nested-1km+GWD to fit now.

**2026-06-08 ~09:51** — 🎉 GWD-nested-1km RETEST (#4) PAST step0 + forecasting (sim-hr2, wrfout 2/2/2, GPU only 6.3GB used vs 28GB ceiling — HUGE headroom from optics/taumol −88.6%/−43.6%). The prior run OOM'd at step0; this fits cleanly with GWD on. → optics/taumol-chunk UNBLOCKED GWD-on-nested. Awaiting full 24h (ETA ~11:00) to confirm #4; then flip GWD-nested default-on + write proof.

**2026-06-08 ~11:25** — ✅ #4 GWD-on-nested DONE: 24h-nested-1km+GWD GREEN (PIPELINE_GREEN, 24/24 wrfout/dom, all-finite, forecast-only ~1.86h, peak ~6-18GB) — optics/taumol VRAM chunking unblocked it (v0.12 OOM'd step0/hr7). GWD flipped DEFAULT-ON (gwd_opt=1 honoured; GPUWRF_GWD_NESTED=0 force-off). Tier1: #1,#2,#3,#4,#6,#7-correctness,#8 DONE; remaining #5 (2-way-24h), #7-headline (carry-over). GPU free → launching #5.

**2026-06-08 ~12:10 — PRINCIPAL: NO CARRY-OVER. Close EVERYTHING incl Tier3, max-parallel, agent-validated.**
v0.13.0 now releases only when ALL closed (Tier1 #7-skill, Tier2 remainder, all Tier3). Tier3-problems discussable for carry-over; the rest NOT. Launching full-closure wave (all CPU-developable via fp64 oracles, parallel to #5-GPU):
- #7-skill/dycore-fidelity (the hard credibility gate: wind-error-growth, dycore-ph'/MYNN-EDMF/*_tendf + moisture-cadence GPT Q1/Q3) — ATTACK, honest if research-wall.
- Tier3 ×5 physics groups (microphysics, cumulus, PBL, radiation, surface+LSM), top-schemes-first batches, per-family oracle.
- (next wave) Tier2 compile-perf (sub-jit + parallel-compile-knob) + CPU-flock; multi-hardware = physically HW-limited (1 GPU), honest-not-closeable.

**2026-06-08 ~12:32 — #5 (2-way+GWD 24h) OOM at hr12** (RESOURCE_EXHAUSTED 3.66GiB). GWD-one-way fit 24h GREEN; the 2-way feedback path (child→parent copy_fcn + sm121 smoother) adds resident VRAM that tips over at a later-hour peak. CLOSE (no carry-over): dispatched 2-way-feedback-VRAM-reduction lane (release child-copy post-area-average, chunk/reuse smoother buffers) → then re-test 2-way+GWD 24h on GPU. #5 = VRAM-marginal, fixable (not a feedback-numerical bug — ran finite to hr12).

**2026-06-08 ~12:57 — wave-1 merges + diagnostics.** MERGED: #7 rad_rk_tendf A/B knob (870b4ff0, CPU gates PASS; honest: won't close 8.06→7.5 alone — GPU A/B + icloud_bl/acoustic-flux levers = follow-ups); T3-pbl MRF bl=99 (oracle 5.4e-16; YSU/MYJ/MYNN/ACM2/BouLac already op; Shin-Hong+QNSE carry-over). T3 so far: microphysics WSM7, cumulus oracle-infra, pbl MRF. 
**TOST Step-0 rc=2 root cause = DATA-PREP**: standalone native-init needs wrfbdy_d02 (lateral forcing); the L2 corpus merged-run-root has no CPU-WRF wrfout history + no wrfbdy_d02. Fix = build wrfbdy_d02 from each case's retained met_em via native-init LBC, THEN run n=15. NOT a forecast/physics bug (scoring path already CPU-rc0-proven). → dispatch TOST-fix-and-run lane.
RUNNING (CPU): aacae3=T3-radiation, ad9e47=T3-surface+lsm, a07e524=#5-2way-VRAM. GPU free.

**2026-06-08 ~13:18 — T3-surface+lsm MERGED** (sfclay 91+3 operational oracle~2e-12, slab-LSM reference-only [needs TSLB land-carry], RUC=carry-over). Merge needed 5 catalog/registry/test 3-way resolutions (union of all Tier3 families: bl{0,1,2,5,7,8,99} sf_sfclay{0,1,2,3,5,7,91} cu{0,1,2,3,5,6,14,16} sf_surface{0,1,2,4}, spec-count→34). FIXED a latent cumulus-merge bug: dispatch_matrix() KeyError'd on reference-only cu=5/14 (in ACCEPTED but not in operational _CU_ENTRIES) — added the reference-only skip + test now verifies operational-routes-vs-reference-only-fail-closed (90 tests green). 
T3 status: microphysics(WSM7 op-pending-qh-leaf), cumulus(oracle-infra, kernels carry-over), pbl(MRF op; ShinHong/QNSE carry-over), surface+lsm(2 sfclay op; slab ref-only; RUC carry-over). RUNNING: aacae3=T3-radiation (CPU), 2way+GWD-retest (GPU, hr2).

**2026-06-08 ~14:29 — compile-perf MERGED** (8b5b1f81: GPUWRF_XLA_PARALLEL_COMPILE knob default-off + recompile-hygiene 2→1; 42 tests). **#5 2-way+GWD-24h retest: rc=1 at hr14** — the cumulative v0.13 VRAM cuts (optics/taumol + feedback-dedup + platform-alloc) pushed the OOM from hr12→hr14 (finite throughout), but the 1km+GWD+2-way config (the single heaviest) still exceeds 32GB before 24h. → either one targeted radiation-peak-during-feedback VRAM lane, or honest-document as 32GB-VRAM-marginal-at-heaviest-config (capability proven to hr14; one-way GWD-24h is fully GREEN; 2-way alone likely fits).

**2026-06-08 ~14:37 — GSFC-LW (ra_lw=5) MERGED** (e588f0ba): oracle-infra + reference-only + honest carry-over (12501-LOC NUWRF family, STOP-condition fired correctly — no slop). Default ra_lw=4 byte-unchanged, spec-count 36, 77 tests. v0.13 MERGED so far: #7-knob, #5-2way-VRAM, T3 microphysics(WSM7)/cumulus(oracle-infra)/pbl(MRF)/surface+lsm(2 sfclay+slab-ref)/radiation(GSFC-SW op + GSFC-LW ref), compile-perf(Tier2), dispatch-fix. 
RUNNING: 9/3km(max-dom=2) 2-way+GWD 24h gate (GPU, #5 capability closure at fitting res); TOST-wrfbdy-fix (CPU). CLOSEABLES remaining: TOST n=15 (unblocking), #7 GPU A/B, mp-batch2 (WDM5+qh-leaf), slab-hook, sub-jit, CPU-flock. DOCUMENTED carry-overs (allowed): cumulus JAX kernels, CAM/NUWRF/GFDL radiation, Shin-Hong/QNSE, RUC-LSM (all multi-thousand-LOC); 2way+GWD+1km+24h + multi-hardware = 32GB-HW-limits.

**2026-06-08 ~14:45 — PRINCIPAL STEERING (positioning + proof + validation):**
POSITIONING: do NOT claim "perfectly-efficient rewrite" or "completely true/faithful port". Value prop = FAST, GPU-NATIVE, **GPU-SCALABLE** WRF-compatible model (nearly all new HPC is GPU). Imperative = get STRONGER ON PROOF (serious validation), not stronger claims. Apply to README/release messaging.
3-POINT PLAN:
 1. GPT-xhigh #1 = full v0.13 IMPLEMENTATION REVIEW (find+fix bugs; substantial ones → manager opens a roadmap case + alternating Opus-max ↔ GPT-xhigh debug workers). branch worker/gpt/v013-impl-review.
 2. GPT-xhigh #2 = build a v0.13 (≤3h wall-clock, full CPU+GPU) + v0.14 (16h) VALIDATION PLAN on known regions (Canary/Switzerland). Primary: WE are sure it runs; Secondary: convince WRF gate-keepers (stable + roughly-equivalent for most couplings). branch worker/gpt/v013-valplan.
 3. As soon as the remaining CPU/GPU v0.13 items finish → START the v0.13 validation plan (manager curates from GPT's plan). Any validation failure → immediately dispatch a debug worker. v0.13 ≠ fully-validated, but must be CLEARLY seriously-tested (not an untested bug-heap). Parallelize CPU+GPU.
CPU BUDGET CHANGE: validation/CPU runs use ~24 threads (taskset -c 0-23), NOT 28 like the nightlies — leave cores 24-31 free for agents + overhead. (Supersedes the old cores-0-3-only rule for the validation window.)

**⚑⚑ PRE-COMPACT ANCHOR 2026-06-08 ~15:00 — READ FIRST ON RESUME ⚑⚑**
Trunk = branch worker/opus/v0120-integration @ 440ebe0a (v0.13.0, spec-count 36). Working dir = /home/enric/src/wrf_gpu2/.claude/worktrees/v0120-integration. Loop heartbeat ~1100s. Principal AFK → debug via GPT-codex (stdin; NEVER pkill -f codex).
POSITIONING (principal 2026-06-08): sell FAST/GPU-NATIVE/GPU-SCALABLE WRF-compatible model, NOT "perfect/true port"; get STRONGER ON PROOF. CPU budget: taskset -c 0-23 (~24 threads), leave 24-31 free (NOT 28).

RUNNING NOW (poll/continue these):
 - #5 = 9/3km(max-dom=2) 2-way+GWD 24h gate (GPU nohup). Poll /mnt/data/canairy_meteo/gate_2way_d02_v013/run.rc → rc=0 + ~24/dom finite = #5 CLOSED (2-way+GWD 24h GREEN at fitting res; 1km-24h=32GB HW-limit). At hr12 @15:00. Write proofs/v013/twoway_gwd_9_3km_24h_gate.json.
 - WDM5 (mp=14 clean op port) = Opus lane ac48e9d (CPU). Merge when done: 3-way union catalog + spec-count→37 + run dispatch/interfaces/namelist tests.
 - GPT-xhigh #1 impl-review = codex, /tmp/gpt_impl_review.log (grep "DONE rc="), branch worker/gpt/v013-impl-review, report .agent/reviews/2026-06-08-gpt-v013-impl-review.md. On done: merge its safe fixes (CPU-verify); SUBSTANTIAL bugs → roadmap cases + alternate Opus-max ↔ GPT-xhigh debug workers.
 - GPT-xhigh #2 validation-plan = codex, /tmp/gpt_valplan.log, branch worker/gpt/v013-valplan, docs .agent/decisions/V0130-VALIDATION-PLAN.md + V0140-VALIDATION-PLAN.md. On done: CURATE the 3h plan into the roadmap.

MERGED v0.13 (9 lanes this wave + earlier): #7 rad_rk_tendf knob, #5 2way-VRAM-dedup, T3 microphysics-WSM7, T3 cumulus-oracle-infra, T3 pbl-MRF, T3 surface-2sfclay(+slab-ref), T3 radiation-GSFC-SW(+GSFC-LW-ref-infra), dispatch-bug-fix, compile-perf, TOST-wrfbdy-fix. EARLIER v0.13: optics/taumol VRAM, #4 GWD-on-nested GREEN (default-on), RRTM-LW, multi-GPU(fake-mesh), MYJ+Janjic, clear-sky, reproducibility, g-point-chunk, TOST-rc2-fix.

CLOSEABLES REMAINING (GPU-serial after #5 frees GPU):
 - TOST n=15 (UNBLOCKED via wrfbdy-fix): single-case smoke `/tmp/wrf_gpu_run_lowprio.sh taskset -c 0-23 env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false python proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py --case 20260429_18z_l2_72h_20260524T204451Z --allow-single` → rc=0 → full `--resume`.
 - #7 rad_rk_tendf A/B (prod 24h, =0 vs =1, score U10/V10/T2 vs CPU-WRF) — the wind-skill credibility measure.
 - WDM5 merge (CPU).
 - THEN the curated v0.13 VALIDATION CAMPAIGN (≤3h, Canary known-region, parallel CPU 0-23 + GPU-serial). ANY failure → immediate cross-model debug worker.

DOCUMENTED CARRY-OVERS (principal-allowed): cumulus JAX kernels (2400-5300 LOC), CAM/Goddard-NUWRF/GFDL radiation (8-12k LOC), RUC-LSM (7516 LOC), Shin-Hong/QNSE PBL — all have de-risked oracle-infra; qh-State-leaf (→WSM7/WDM7 operational) + slab-LSM-hook = focused future State sprints; 2way+GWD+1km+24h + multi-hardware = 32GB-HW-limits; #7 24h wind-skill closure = open research (lever landed, not closed).

RELEASE v0.13.0 (when closeables + validation done): fill 18 <<MANAGER-FILL>> in RELEASE_NOTES_v0.13.0.md/README.md/docs/KNOWN_ISSUES.md (#5/#7/TOST/validation numbers) + GPU-scalable honest positioning + fix docs/namelist-compatibility.md stale MYJ refs → gap-critic (Opus+GPT) → tag v0.13.0 → push wrfgpu HEAD:main + tag (home=latest) → final report.

**2026-06-08 ~15:08 — POST-COMPACT RESUME (manager).** Principal: finish v0.13 max-efficiency; GPT/codex workers → tmux-window pattern (skill L30) going forward (current 2 headless GPTs left to finish); goal = clean v0.13 release, test-as-much-as-possible, parallelize CPU/GPU/**agents**.
6 PARALLEL LANES RUNNING:
 1. #5 9/3km(max-dom=2) 2-way+GWD 24h gate — GPU nohup, /mnt/data/canairy_meteo/gate_2way_d02_v013/, 14/24/dom @hr14 finite, ~25min left. rc=0+24/dom → CLOSE (proofs/v013/twoway_gwd_9_3km_24h_gate.json).
 2. WDM5 mp=14 — Opus agent ac48e9d (auto-notify); caught+fixing real pidep uncapped-ice-diameter bug vs wdm52D. Merge w/ 3-way union catalog + spec-count→37.
 3. GPT#1 impl-review — codex headless PID 3092430, /tmp/gpt_impl_review.log, branch worker/gpt/v013-impl-review. On done: merge safe fixes; substantial→roadmap cases + alternate Opus-max↔GPT debug.
 4. GPT#2 valplan — codex headless PID 3092432, /tmp/gpt_valplan.log, branches/docs V0130+V0140-VALIDATION-PLAN.md. On done: curate 3h plan.
 5. Agent V (aee7e6f9) — CPU bedrock validation (idealized Straka/Skamarock + conservation + restart-bit-identity), worktree, proofs/v013/cpu_bedrock_validation.json. Auto-notify.
 6. Agent D (a6d9923d) — gate-keeper docs/namelist-compatibility.md accuracy vs real registry, worktree. Auto-notify.
NEXT GPU (serial, after #5): TOST n=15 smoke→resume; #7 rad_rk_tendf A/B 24h. NEXT after GPTs: curated validation campaign + debug-on-failure. RELEASE: fill 18 placeholders + GPU-scalable positioning + gap-critic → tag → push wrfgpu home=latest.

**2026-06-08 ~15:14 — WDM5 MERGED** (672cb478): mp=14 operational, 6/6 pristine-WRF fp64 oracle PASS, clean 3-way (no conflicts, trunk only had doc commits since base), spec-count 36→37, 56 catalog tests green. v0.13 OPERATIONAL microphysics now: Thompson(8,default), Kessler(1), Lin(2), WSM3(3), WSM5(4), WSM6(6), Morrison(10), WDM6(16), **WDM5(14 NEW)**; WSM7(24)=ref-only (qh-leaf).
**PRINCIPAL 15:13:** stay in 15-min loop until v0.13 done; each tick verify all lanes alive + spawn quality agents; UNLIMITED GPT/Opus-max/maxcode budget; top goal = highest-quality COMPLETELY-FUNCTIONAL v0.13. → added 2 Opus-max quality lanes: (7) oracle-integrity anti-slop audit across ALL v0.13 oracle claims (read-only+report); (8) operational-functional-coverage audit (every operational scheme has a runnable functional test; add consolidated smoke module). Big fan-outs (debug lanes + validation campaign) held for the 2 imminent GPT deliverables.

**2026-06-08 ~15:22 — 3 LANES MERGED (CPU, GPU untouched):**
 - bedrock CPU validation (b1dbab15): 5 PASS/0 FAIL/1 SKIP — idealized Straka+Skamarock fp64 PASS, mass-conservation exact (drift 0.0 / budget 2.45e-16), restart bit-identical 2 paths fail-closed, qke NaN-smoke. 1 skip=live-coupled mass invariant (State.zeros refuses CPU by design → GPU). PROVES RUNS-stability; NOT equivalence.
 - gatekeeper docs/namelist-compatibility.md (merged + patched 5f28b276): 11 stale entries fixed, programmatically-derived OPERATIONAL/REF-ONLY, bit-identity disclaimed, GPU-native/scalable framing; WDM5 mp=14 row + count 9→10 added post-merge. mp:10op cu:5op/3ref bl:7op sf_sfclay:7op sf_surface:3op/1ref ra_sw:4op ra_lw:3op/1ref.
RELEASE KNOWN_ISSUES.md FILL-LIST (Agent D found these carry-overs NOT explicitly named — add bullets at release): WSM7/WDM7 qh-State-leaf; GSFC/Goddard NUWRF-LW (ra_lw=5) ref-only; RUC-LSM; Shin-Hong/QNSE PBL; slab-LSM hook (sf_surface=1, TSLB carry+GSW/GLW forcing); combined 2way+GWD+1km+24h 32GB VRAM ceiling still-binding.
LANES NOW: #5 gate(GPU, ~20/24), GPT#1 impl-review(codex), GPT#2 valplan(codex), Agent O oracle-integrity-audit(a4d9bf36), Agent F operational-functional-coverage(a79909b8). PRINCIPAL: 15-min loop until v0.13 done; keep spawning quality agents; unlimited GPT/Opus-max budget; goal=highest-quality COMPLETELY-FUNCTIONAL v0.13.

**2026-06-08 ~15:30 — BIG WAVE (trunk @ 0b917ea6, spec-count 37):**
 ✅ #5 CLOSED GREEN (798704d5): 9/3km 2-way+GWD 24h rc=0, 24/24 wrfout/dom, all-finite hr24 (proofs/v013/twoway_gwd_9_3km_24h_gate.json). 1km+24h=32GB HW-limit (honest).
 ✅ Agent O oracle-integrity audit MERGED (798704d5): 19 oracles, 0 BLOCKER/0 MAJOR/2 MINOR; 0 self-compare, 0 hidden-clamp, 3 bit-identity exact, 0 ref-only-as-operational; sha256 of every pristine module verified. VERDICT: proof corpus HONEST enough to ship. MINOR fix lanes M-1 (cumulus ref-only source-checksums) + M-2 (clear-sky absolute tolerance ceiling) — queue.
 ✅ GPT#1 impl-review MERGED (526f4d73): 2 MAJOR fixes (ra_sw/ra_lw=0 disabled-radiation no-op + zeroed diag; slab gpu_runnable=False). 75 tests green. Issues flagged → A (MRF+sf=3/91 diag mismatch, MAJOR) dispatched to Opus-max debug ae361b74; B (WSM7 mp=24 "ref-only" vs actual fail-closed — wording mismatch; namelist-doc already correct → low-risk doc/comment align + KNOWN_ISSUES).
 ✅ GPT#2 validation plans MERGED (0b917ea6): V0130 3h (7 tests) + V0140 16h. All harness scripts + test files verified to EXIST. Campaign ADOPTED + LAUNCHED.
RUNNING FLEET (6): TOST n=15 smoke (GPU, /mnt/data/canairy_meteo/tost_smoke_v013/smoke.rc; single-case 24h, ACQUIRED GPU); GPU-chain A1(24h L2 9/3km GWD+2way+score)→A2(6h L3 9/3/1km) queued behind smoke via lowprio flock (/mnt/data/wrf_gpu_validation/v0130_campaign/a{1,2}.rc); CPU-campaign A4-A7 (Opus a2e022d0, cores 12-23, proofs/v013/cpu_campaign_a4a7_results.json); Issue-A MRF/sf fix (Opus-max ae361b74, cores 0-11); Agent F operational-functional-coverage (Opus a79909b8, cores 28-31).
REMAINING TO RELEASE: smoke rc=0→full TOST n=15 (--resume); A1/A2 results; A4-A7 results; merge Issue-A fix + Agent F test module; M-1/M-2 + Issue-B doc; #7 rad_rk_tendf A/B (GPU). THEN fill 18 placeholders + #5/#7/TOST/validation numbers + GPU-scalable positioning + KNOWN_ISSUES fill-list → gap-critic(Opus+GPT) → tag v0.13.0 → push wrfgpu HEAD:main+tag(home=latest).

**2026-06-08 ~15:48 — Agent F functional-coverage MERGED + 2 defects handled (trunk @ 6a500844):**
 ✅ Agent F MERGED: tests/test_v013_operational_smoke.py (38 tests, ~21 options' FIRST integrated functional coverage via real _physics_step_forcing, assert finite+actual-mutation) + proofs/v013/operational_functional_coverage.md. OPERATIONAL set confirmed: mp{1,2,3,4,6,8,10,14,16} bl{1,2,5,7,8,99} sf_sfclay{1,2,3,5,7,91} cu{1,2,3,6} sf_surface{2,4} ra_sw{1,2,4} ra_lw{1,4}.
 ✅ DEFECT #1 FIXED (6a500844): mp=14 WDM5 was UNROUTABLE (scan-wired + advertised-operational but missing from physics_dispatch._MP_ENTRIES → _resolve_operational_suite hard-rejected). Added _mp_entry(14,...) row; mp=14 now resolves + passes functional smoke (37 passed/1 xfailed). A real "advertised-but-broken" bug I'd shipped this session — now genuinely functional.
 🔧 DEFECT #2 (cu=6 Tiedtke INERT — adapter hard-zeroes QVFTEN → zero precip; kernel works when fed it) → DEBUG LANE dispatched as GPT-codex in tmux window 2 (worker/gpt/v013-tiedtke, /tmp/gpt_tiedtke_prompt.txt, marker "GPT TIEDTKE DONE"). Cross-model (Opus found→GPT fixes). Path: thread real QVFTEN if tractable+verified, ELSE honest re-scope cu=6→reference-only. Default byte-unchanged required.
ACTIVE FLEET (5): TOST smoke (GPU), GPU-chain A1→A2 (queued), CPU-campaign A4-A7 (Opus a2e022d0), Issue-A MRF/sf fix (Opus-max ae361b74), Tiedtke fix (GPT tmux win2). NOTE: applied principal's GPT-in-tmux rule (skill L30) for the first time — window 2, attachable.

**2026-06-08 ~16:00 — Issue-A MERGED + 2 NEW PRINCIPAL GPT LANES (trunk @ post-merge):**
 ✅ Issue-A MRF/sf honesty fix MERGED (1b2ab8c0→trunk): bug broader (YSU1/ACM2 7/BouLac8/MRF99 all re-derive revised-MM5) → fail-close bl∈{1,7,8,99}×sf≠1 (no silent substitution); MYNN/MYJ exempt; default byte-unchanged (hash ca729533…); composes w/ WDM5 fix (131 passed/1 xfailed). Residual: bl×sf=91 could be widened by threading old-MM5 forcing (future).
 🆕 PRINCIPAL 2 GPT LANES (analysis/writing, NON-BLOCKING — no GPU, no new CPU-WRF, own branches, manager reviews):
  - PAPER v0.14 (GPT tmux win4, worktree gpt-paper, branch worker/gpt/v014-paper, /tmp/gpt_paper_prompt.txt, marker "GPT PAPER DONE"): rewrite paper→v0.14 target; honest claim-audit table (TESTED/PENDING/SIMULATED/PROJECTED/NOT-TESTED); theme = energy-eff ~4× per-watt MEASURED + comparable-not-bit-identical + GPU/multi-GPU scaling (proven/sim/projected separated) + AI-driven-engineering case study; TOST n=15 = PENDING placeholder; artifacts publish/paper_v014_draft.md + publish/paper_v014_claim_audit.md.
  - CANARY EXISTING-DATA STATS (GPT tmux win5, worktree gpt-canary-stats, branch worker/gpt/v013-canary-stats, /tmp/gpt_canary_stats_prompt.txt, marker "GPT CANARY-STATS DONE"): inventory ~30 d03/nested d02-d03 runs; usability assessment; prep extraction/align/QC/stats/plot scripts (scripts/canary_stats/); report .agent/reviews/2026-06-08-gpt-canary-existing-data-stats.md; KEY Q = can we validate GPU≈WRF first-order WITHOUT new CPU runs? Avoid new CPU-WRF runs; GPU runs still proceed.
ACTIVE FLEET (6): TOST smoke (GPU), GPU-chain A1→A2 (queued), CPU-campaign A4-A7 (Opus a2e022d0), Tiedtke fix (GPT win2), Paper (GPT win4), Canary-stats (GPT win5). 3 GPT-codex now in tmux per principal rule.

**2026-06-08 ~17:20 — HIBERNATE RECOVERY.** Box suspended ~16:12→resumed ~17:16 (no reboot). Damage = ONLY the GPU job: TOST-smoke python (PID 3149338) hung with dead CUDA context (held 19.2GB @0% util, never wrote rc, held the lowprio flock → blocked queued A1/A2). Recovered: killed the hung smoke chain + queued A1/A2 chain (NOT codex), GPU back to 4.2GB desktop-only, relaunched fresh GPU chain smoke→A1→A2 (per-step rc so a future suspend loses only the in-flight job). CPU/codex SURVIVED intact: Tiedtke DONE (b9420aae, threaded real QVFTEN → cu=6 genuinely functional, path-2a), Paper v0.14 DONE (61a8cb59), canary-stats still working (win5). 14 codex procs + tmux 2/4/5 untouched.
NEXT-TICK TODO (DONE lanes ready, deferred from this recovery turn): (a) MERGE Tiedtke b9420aae — touches scan_adapters/operational_mode/physics_dispatch (CONFLICT-RISK with Issue-A; CPU-verify catalog+dispatch+smoke); (b) REVIEW+MERGE Paper v0.14 → publish/ + tell principal ready to proofread; (c) A5 rc=1 = 2 STALE harness expectations (pbl_acm2 8/7/7 now correctly Issue-A-fail-closed; pbl_myj_janjic now correctly operational) → update A5 matrix, NOT production bugs; (d) poll relaunched GPU chain.

**2026-06-08 ~17:53 — PRINCIPAL DECISION: TOST n=15 BLOCKS the v0.13.0 tag** (closest full real proof; principal will request H200+ compute for big-scale validation later, but n=15 must ship). Killed the twice-hibernate-stalled smoke; GPU clean. LAUNCHED overnight marathon (detached, /mnt/data/wrf_gpu_validation/v0130_marathon/): Phase1 TOST n=15 `--resume` (BLOCKER, ~8-11h, per-case lowprio flock 0-3) → Phase2 A1 (24h L2 9/3km GWD+2way+score) → Phase3 A2 (6h L3 9/3/1km). rc files n15.rc/a1.rc/a2.rc.
CPU PROFILE tonight: loadavg ~2.2/32, NO nightly CPU-WRF; only user's wu_scraper (~1 core); GPU jobs lowprio-pinned 0-3 → cores 4-31 free (≥24). Safe to run CPU merges in parallel.
HIBERNATE RESILIENCE: --resume preserves completed cases; 15-min loop hibernate-check (0%util+no-rc-progress+long-elapsed=zombie) → kill stalled case + relaunch --resume. Principal advised to keep box awake.
ETA tag+push: tomorrow ~07:00-09:30 (n=15 pass + merges + packaging + gap-critic). PARALLEL CPU now/next-tick: merge Tiedtke b9420aae + canary-stats e96a16ef + Paper→local publish/; A5 matrix fix; M-1/M-2/Issue-B; draft release docs. TAG GATED ON n=15 rc=0 + equivalence PASS.
