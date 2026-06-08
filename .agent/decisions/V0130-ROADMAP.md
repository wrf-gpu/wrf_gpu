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
