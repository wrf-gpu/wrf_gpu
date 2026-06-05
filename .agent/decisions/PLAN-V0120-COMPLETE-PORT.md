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
