# PLAN — v0.12.0: Complete WRF v4 Port ("no open flanks") + Gotthard equivalence suite

Status: DRAFT (manager-written milestone plan, 2026-06-05). Requires cross-model review (Phase 0) before implementation sprints. Authored while v0.11.0 validations run.

## Goal
v0.12.0 = a **complete, faithful WRF v4 ARW port**: every remaining WRF v4 feature/option either **implemented + oracle-validated**, or **explicitly + justifiably scoped out** (documented as a deliberate boundary, not an omission) — so **no WRF v4 developer can credibly say "this is not a valid port."**

Plus a **shipped, self-serve equivalence test** on a **non-Canary region (Gotthard / Central Switzerland)** that demonstrates GPU-port ≈ CPU-WRF at all grid points and all timesteps — proving the port **generalizes** beyond its development region.

## Entry state (v0.11.0 trunk @ 3db9ec6)
Tier-1 (nesting one-way + gated feedback, wrfrst restart, Kain-Fritsch, conservation budgets) ✅ · Tier-2 physics (MYNN-EDMF, RRTMG topo/slope, Thompson debts, terrain-slope diffusion, GWD-deviation) ✅ · DGX-D1 sharding substrate ✅ · recompile precondition ✅ · qke-KI-2 gate closed ✅. d03-1km (KI-1) + powered TOST + DGX-D2 in v0.11.0 validation.

## Phase 0 — Definitive "valid-port" gap audit (cross-model; doubles as the milestone review)
Independent audits (GPT + Opus + agy) answering: *"What would a WRF v4 developer flag as missing, stubbed, approximated, fail-closed, or non-faithful?"* — checked against the full WRF v4 **Registry** (every namelist option, scheme id, dynamics/IO/nesting/DA feature). Output: the authoritative remaining-gap inventory with per-item acceptance criteria. Supersedes the stale `publish/GPU_PORT_GAPS_TODO.md` + the README roadmap table. **This is the binding scope for v0.12.0.**

## Phase 1 — Full physics scheme matrix (consolidate the v0.6.0 wave + complete + oracle-validate)
Bring the major WRF v4 physics options to "implemented + WRF-savepoint-parity + scan-wired + registry=implemented" (much was prototyped in the v0.6.0 scheme wave — consolidate banked branches, finish, validate):
- **Microphysics** mp_physics: Kessler(1), Lin(2), WSM3/5/6, Thompson(8 ✅), Morrison(10), WDM6(16), Goddard(7), Thompson-aerosol(28), NSSL.
- **Cumulus** cu_physics: KF(1 ✅), BMJ(2), GF(3 ✅), Grell-3(5), Tiedtke(6), SAS family(4/14/84).
- **PBL** bl_pbl_physics: YSU(1 ✅), MYJ(2), ACM2(7 ✅), MYNN(5 ✅), BouLac(8), Shin-Hong(11).
- **Surface layer** sf_sfclay: MM5(1), MYJ(2), MYNN(5 ✅), revised-MM5(91 ✅).
- **LSM** sf_surface_physics: Noah(2), RUC(3), Noah-MP(4 ✅).
- **Radiation** ra_lw/sw: RRTMG(4 ✅), RRTM(1)/Dudhia(1), Goddard(5), CAM(3).
Declare the **supported matrix explicitly**. Genuinely out-of-scope (urban BEP/BEM, WRF-Chem, lake, WRF-Hydro) = **documented scope decisions**, fail-closed with a named reason — intentional boundaries, not gaps.

## Phase 2 — Dynamics & numerics option completeness
- **Positive-definite + monotonic** scalar/moisture advection (full moist_adv_opt / scalar_adv_opt family) + **WRF boundary-order degradation** near specified/nested boundaries (closes delta #10).
- Full advection orders (h/v 2–6), diff_opt/km_opt variants, diff_6th_opt, damp_opt(1/2/3), w_damping, epssm, divergence damping, gwd_opt (implement if completeness demands).
- Adaptive timestep (use_adaptive_time_step), DFI (digital-filter init) — implement or justified-scope.

## Phase 3 — I/O, namelist, grid completeness
- **Full wrfout** variable set toward the 375-var reference (closes KI-3) + auxhist/auxinput streams.
- **wrfrst** full-state restart — confirm completeness over 24–72 h.
- **Full WRF namelist** parse + validate + **loud rejection** of unsupported options (`io/namelist_check.py` completeness).
- Map projections beyond Lambert/Mercator/Polar as needed for generality.

## Phase 4 — Nesting & DA completeness
- **Full two-way nesting feedback** (v0.11.0 = one-way + gated; complete + validate two-way), moving/vortex-following nests if claimed.
- **FDDA** (analysis/obs/spectral nudging) — implement or justified-scope.

## Phase 5 — Multi-GPU/DGX finalize (carry from v0.11.0 D2)
Finalize the optional sharded path (real-DGX-ready), single-GPU default stays **zero-overhead** (re-verified).

## Phase 6 — Gotthard / Central-Switzerland equivalence test suite (SHIPPED with v0.12.0)
A packaged, self-serve script (`scripts/equivalence_gotthard.py` + a small suite) that an end-user runs to see GPU≈CPU for themselves, on a **new region** (proves generalization):
- **Domain:** center **46.65°N, 8.55°E**; square **~150–160 km** (auto-size to the LARGEST that fits with headroom in 32 GB at 1 km / fp64 — maximize to make the GPU-vs-CPU **speed difference** as stark as possible); **dx = 1 km**; **24 h**; **AIFS-initialized**.
- **Pipeline:** AIFS → WPS/real.exe (Alpine geog/terrain) → **CPU-WRF v4** run (sensible Alpine 1 km physics suite) → **GPU-port** run (native AIFS init, v0.4.0 path) → comparison.
- **`--date` parameter** selects the day for the CPU-WRF reference.
- **Comparison script:** Python stats + tables, GPU vs CPU at **ALL grid points and ALL output timesteps** — per-field (T2/U10/V10/precip + 3D U/V/W/T/QV…) RMSE / bias / max-abs-diff per timestep + an overall **equivalence verdict** vs a predeclared tolerance. Also reports the **wall-clock speedup** (the demo).
- **Default-day mode:** the CPU solution for a fixed default day is published to a **gdrive link** (optional download) → the tester runs ONLY the GPU pass on their machine and gets the equivalence stats directly — **no CPU-WRF build required on their end.**
- **HONEST equivalence framing (must be explicit in the script + docs):** GPU-port vs CPU-WRF is **numerical/operational equivalence within a predeclared tolerance**, NOT bitwise — the port is a faithful JAX reimplementation, not a bit-clone of the Fortran (per the project's validation philosophy). "Equal at all points/times" = the within-tolerance equivalence criterion holds **everywhere + at every timestep**; separately, the **GPU port is bitwise self-deterministic** (same run reproduces exactly). The script states both clearly so no claim is overstated.
- **Pre-v0.12.0 GATE:** build + run the full suite locally once on the default Gotthard day → confirm the equivalence criterion holds at all grid points + all timesteps (and record any per-field tolerances) → publish the default-day CPU solution to gdrive. The suite ships only after this local confirmation.

## Phase 7 — Final "valid-port" gap-critic + release
- Cross-model gap-critic from the **WRF-developer perspective** ("can anyone credibly call this not a valid port?") → fix-now or explicit-justified-scope (recorded).
- Release protocol: gap-critic → Opus-xhigh release worker (README/docs/cleanup) → **tag v0.12.0** → push github.com/wrf-gpu/wrf_gpu (home=latest). (Tag the pending v0.10.1 + v0.11.0 in sequence first.)

## Honest scope notes
- "Complete" = the full **standard WRF v4 ARW** feature set + the common physics matrix. Deliberately-excluded subsystems (WRF-Chem, urban canopy BEP/BEM, WRF-Hydro, chem/aerosol-coupled beyond Thompson-aerosol) are **documented scope decisions**, so a reviewer sees intentional boundaries.
- The equivalence standard is **numerical/operational (within predeclared tol)**, not bitwise-vs-Fortran.
- Sizing: most of Phase 1 is **consolidation** of the v0.6.0 banked scheme work, not from-scratch — that makes v0.12.0 large but tractable.

---

## LOCKED SCOPE + REVISED SPRINT TABLE (2026-06-05 — SUPERSEDES the draft phase list above)

**GPT-5.5 plan review:** `proofs/v0110/v0120_plan_review.md` — verdict **GAPS:14**. Found many missing WRF v4 scheme IDs (MP `5/9/11/13/18/24/26/38/50-53/55/56/95-97`, CU `7/10/11/16/93/94/96`, PBL `3/4/9/10/12/16/17`, SFC `3/4/7/10`, LSM `1/5/6/7/8`, RA LW/SW variants), shallow-cu, FDDA/spectral-nudging, ndown, moving-nests, aux-streams 1-24, lat-lon/global projection, stochastic physics; plus all the per-scheme control namelist options. Structural fixes folded in below.

**PRODUCT-SCOPE DECISION (principal, 2026-06-05) = SCOPE A — "Common + Fail-Closed":**
- Common operational matrix → **implemented + oracle-validated**.
- Entire remaining WRF v4 catalog → **`recognized_fail_closed`** (loud rejection, named reason) — *never a silent wrong path*.
- Exotic / WRF-Chem / WRF-Fire / WRF-Hydro / BEP-BEM → **`out_of_scope` + reason**.
- Moving/vortex nests → **scoped out with reason** (P5.3 = the reject path).
- **"No open flank" = no SILENT gap** (faithful result OR explicit "recognized, unsupported because X"), NOT every-scheme-bit-ported. Constitution-aligned (not a line-by-line port).
- Sizing: **L–XL milestone** (multi-week, multi-agent).

**Binding precondition (GPT risk #1):** Phase 0 must emit a **machine-readable COVERAGE LEDGER** — every Registry rconfig / scheme ID / I-O stream / boundary mode → `implemented | validated | recognized_fail_closed | out_of_scope+reason` — **before** Phase 1 begins, so implementers can't close a sprint against a short list while unlisted features stay open. **State/IO ABI (P1.1) frozen before broad scheme work** (GPT risk #4 — else the state ABI churns).

**Difficulty:** ◔S=≤1d/1 agent · ◑M=2-4d · ◕L=~1wk/multi-agent · ●XL=1-2wk/wave · ●●XXL=weeks. `[GPU]`=GPU-bound.

| Sprint | Content (gaps folded in) | Diff. | Dep. |
|--------|---------------------------|:----:|------|
| P0.1 Coverage-Ledger | Full Registry enumeration → classify every option/scheme/stream/boundary; cross-model (GPT+Opus+agy). Binding scope. | ◕ L | first |
| P0.2 Product-scope | **LOCKED = Scope A** (principal 2026-06-05) | done | — |
| P1.1 State/IO-ABI freeze | wrfrst full state (domain-tree, per-scheme carry, accumulators, adaptive-dt, DFI, moving-nest, FDDA, stochastic seeds, alarms) + scheme-dependent Registry I/O matrix | ◕ L | P0 |
| P2-MP Microphysics | common set faithful + controls (`mp_zero_out`/`hail_opt`/`progn`/`use_mp_re`); rest fail-closed | ● XL | P1 |
| P2-CU Cumulus | + shallow-cu (`shcu_physics`) + `cu_rad_feedback`/`kfeta_trigger`/`cudt`; fix code `84`→`94` | ◕ L | P1 |
| P2-PBL | missing IDs + MYNN option-fidelity (`bl_mynn_*`/`icloud_bl`/`scalar_pblmix`/`tke_budget`) | ◕ L | P1 |
| P2-SFC Surface-layer | fix `1`=revised-MM5/`91`=old; +`3/4/7/10`; validate PBL/SFC pairings | ◑ M | P1 |
| P2-LSM Land | Noah-MP full `&noah_mp` matrix; +ocean/lake/sea-ice/SST-update; urban-UCM impl-or-reject | ◕ L | P1 |
| P2-RA Radiation | LW/SW missing IDs + controls (`radt`/`cldovrlp`/ozone/aerosol/`slope_rad`/`topo_shading`) | ◕ L | P1 |
| P3.1 Advection | full order matrix + WRF-equivalent PD/monotonic limiters + boundary-order degradation | ◕ L | P1 |
| P3.2 Diffusion/Damping/GWD | `diff_opt`/`km_opt`/`diff_6th`/`sfs_opt` + `damp_opt`/`w_damping`/`epssm`/polar filters + `gwd_opt=1/3` | ◕ L | P1 |
| P3.3 Vertical/Coord/dt/DFI | `hybrid_opt`/`etac`/`use_theta_m`/`non_hydrostatic` + adaptive-dt group + DFI families (impl-or-reject) | ◕ L | P1 |
| P4.1 Boundaries/LBC | specified/nested/open/periodic/symmetric/polar; `spec_bdy_width`/`relax_zone`; `have_bcs_moist/scalar`; tendencies for all state | ◕ L | P1 |
| P5.1 Nesting matrix | full config + multi-child + subcycling + full feedback field-set + `interp_method_type` `[GPU]` | ◕ L | P4 |
| P5.2 ndown/offline | `have_bcs_*`, `vert_refine_*`, `rebalance`, wrfbdy gen/consume | ◑ M | P5.1 |
| P5.3 Moving/Vortex-nests | **REJECT path** (Scope A): recognized_fail_closed + reason | ◔ S | P0.2 |
| P6.1 FDDA/Nudging | grid + spectral + surface FDDA + obs-nudging (`auxinput11`); WRFDA/4DVAR out-of-scope | ◕ L | P1 |
| P7.1 Aux-streams + I/O controls | streams 1-24 (`auxhistN`/`auxinputN`) + `frames_per_outfile`/`iofields_filename`/`tslist`/`nocolons`; `io_form_*` explicit | ◕ L | P1 |
| P7.2 Namelist parser/validator | all groups, per-domain arrays, `max_dom` broadcast, cross-option deps, machine-readable unsupported-ledger | ◑ M | P0.1 |
| P7.3 Projections + static-geo + v-interp | `map_proj=0` lat-lon + rotated/global (impl-or-reject); `mminlu`/`num_land_cat`/SST; `interp_type`/`hypsometric_opt`/`eta_levels` | ◕ L | P1 |
| P8.1 Stochastic + misc fail-closed | `sppt`/`skebs`/`spp` + lightning/HAILCAST/windfarm/trajectories/SCM/Solar/Fire/Chem/Hydro/BEP → named scope decisions, fail-closed | ◑ M | P0.2 |
| P9.1 DGX finalize | finalize optional shard path, default zero-overhead re-verified (carry from v0.11.0 D2) | ◑ M | — |
| P10.1 Gotthard suite build | **FIXED** domain (predeclared e_we/e_sn/e_vert, dx, dt, p_top, physics suite); CPU-ref pinned+checksums; field inventory; tolerances predeclared; AIFS-license note; gdrive+checksum+fallback | ◕ L | P1-P7 |
| P10.2 Gotthard local-confirm + publish | default day local: equivalence at all gridpoints/output-times ≤ tol → publish CPU ref `[GPU]` | ◑ M | P10.1 |
| P11.1 Release gap-critic | cross-model vs the coverage ledger (WRF-dev perspective) → fix-now/carry | ◑ M | all |
| P11.2 Release worker + tag + push | Opus-xhigh: README/docs/cleanup/notes → tag v0.12.0 → push org | ◑ M | P11.1 |

**Sequencing:** P0 (ledger + scope ✅) → **P1 ABI-freeze first** → P2-P8 parallel waves (file-disjoint, GPU-mutex) → P9 → **P10 after P4/P5** (Gotthard uses nested/specified boundaries) → P11.

**Gotthard fixes from the review (binding for P10):** fix the default domain exactly (no auto-size — breaks a single published CPU reference); pin the CPU-WRF reference (version/compiler/precision/namelist/WPS/AIFS-source + `wrfinput`/`wrfbdy`/`wrfout` checksums); predeclare per-variable tolerances *before* the confirm run; define the full compared field inventory; "all timesteps" = **output times** (internal RK only with savepoints); AIFS redistribution needs a license note + manifests + non-link-rot fallback; one default day = an external-region regression/demo, **not** a generalization proof — phrase claims accordingly.
