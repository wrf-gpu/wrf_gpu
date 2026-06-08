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
