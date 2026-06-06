# wrf_gpu

A GPU-native, WRF-compatible regional NWP system designed and built almost entirely by an AI agent swarm. The operational target is **Canary Islands daily forecasting** (3 km then 1 km) on a single-workstation RTX 5090.

This is not a port of legacy WRF source. It is a clean JAX rewrite that targets the GPU memory hierarchy from day one and validates against WRF as an oracle rather than inheriting WRF's architecture.

## Current status — v0.11.0

**v0.11.0 is the feature-complete release of wrf_gpu: live multi-domain nesting, full restart continuity, conservation-closed budgets, MYNN-EDMF, topographic/slope radiation, terrain-slope diffusion, Kain-Fritsch cumulus, gravity-wave drag, and optional multi-GPU/DGX sharding on a standalone, JAX-native, single-GPU WRF v4 ARW forecast system.** It also ships the recompile fix (no per-chunk XLA recompile in production), and resolves the prior d03 1 km steep-terrain instability (KI-1, now closed with WRF-faithful qke cold-start seeding) and the long single-call qke edge (KI-2, closed by a WRF-faithful IEEE fmax/fmin fix in MYNN).

The system performs **native real-init** (assembles `wrfinput`/`wrfbdy` from met_em-stage forcing, no `real.exe` and no CPU-WRF artifact for the initial/boundary state), runs a **nonhydrostatic split-explicit ARW dycore** on the GPU, exposes a **WRF-compatible namelist** with a **GPU-operational physics menu** and a **fail-closed boundary** on everything not yet ported.

v0.11.0 also corrects a dry-physics RK-cadence regression in the MYNN-EDMF/conservation integration that had degraded d02 wind skill; after the fix, winds recover to the v0.9.0 level: U10 mean RMSE 4.43 m/s / V10 3.59 m/s, beats persistence 23/24 leads.

This is a deliberate step beyond v0.1.0, which was a single-domain **replay** path that consumed CPU-WRF/Gen2 artifacts for initialization. v0.3.0 added native metgrid; v0.4.0 added native real-init (proven equivalent to `real.exe` at t=0); v0.6.0 expanded the operational physics menu; v0.9.0 consolidated these into a standalone forecast system; v0.10.0 kept those numerics unchanged and removed one faithful Thompson sedimentation inefficiency; **v0.11.0 adds live nesting, restart, conservation, MYNN-EDMF, topographic radiation, and slope diffusion**.

> **Honesty note.** Two distinct claims are kept separate throughout this README and must not be conflated:
> 1. **Native init** is proven equivalent to `real.exe` at t=0 (savepoint parity) and produces a stable forecast.
> 2. The **coupled skill validation** vs CPU-WRF on d02 and on the nested d01→d02→d03 hierarchy is run through the **replay harness** (parent-history replay, which consumes a CPU-WRF `wrfout` for the boundary/skill comparison). The validated coupled-skill runs are *not* from-scratch native-init runs. The standalone AIFS e2e (native real-init → forecast, no CPU-WRF) is proven stable for a 6 h smoke window on a distinct case.

> **Statistical honesty.** Operational equivalence to CPU-WRF is characterised as mean RMSE within operational bars on a single representative MAM case and a single season. Formal TOST equivalence at the ADR-029 predeclared tight margins (T2 ±0.215 K, U10 ±0.231 m/s, V10 ±0.275 m/s) is **underpowered** at the corpus size available (only n≈2-3 pairable cases). **No "TOST PASS" / "statistically-proven equivalence" is claimed.**

### What v0.11.0 is — GPU-operational capability

**GPU-operational physics menu (scan-wired into the operational forecast loop, WRF-oracle-gated).** These are the schemes the operational scan actually dispatches; the exact wiring is in [`src/gpuwrf/runtime/operational_mode.py`](src/gpuwrf/runtime/operational_mode.py) (`_SCAN_WIRED_OPTIONS`) and [`src/gpuwrf/coupling/scan_adapters.py`](src/gpuwrf/coupling/scan_adapters.py); the namelist-accepted matrix is in [`src/gpuwrf/contracts/physics_registry.py`](src/gpuwrf/contracts/physics_registry.py).

| Family | Namelist key | GPU-operational options (scan-wired) |
|---|---|---|
| Microphysics | `mp_physics` | 1 Kessler, 2 Purdue-Lin, 3 WSM3, 4 WSM5, 6 WSM6, 8 Thompson, 10 Morrison, 16 WDM6 |
| PBL | `bl_pbl_physics` | 1 YSU, **5 MYNN-EDMF** (v0.11.0: DMP mass flux + cloud-aware moisture/thermodynamics enabled, matching WRF defaults `bl_mynn_edmf=1`, `bl_mynn_edmf_mom=1`), 7 ACM2, 8 BouLac |
| Surface layer | `sf_sfclay_physics` | 1 revised-MM5, 5 MYNN-SL, 7 Pleim-Xiu |
| Cumulus | `cu_physics` | **1 Kain-Fritsch** (v0.11.0: column/savepoint parity PASS; d01 parent now has faithful KF), 2 BMJ (fp64), 3 Grell-Freitas (scale-aware), 6 Tiedtke |
| Radiation | `ra_sw_physics` / `ra_lw_physics` | RRTMG SW + LW; **v0.11.0: topographic shading (`topo_shading=1`) + slope-corrected surface radiation (`slope_rad=1`) now operational** (`ra_sw=4` / `ra_lw=4`) |
| Land surface | `sf_surface_physics` | 2 Noah classic (explicit static/land bundle), 4 Noah-MP (`use_noahmp=True`) |
| Diffusion | `diff_opt`, `km_opt` | constant-K and 2-D Smagorinsky; **v0.11.0: terrain-slope + map-factor deformation terms now included** (WRF formula parity, max residual `3.78e-15`) |

`mp_physics=0` (passive vapor), `bl_pbl_physics=0`, `sf_sfclay_physics=0`, `cu_physics=0`, and `ra_*=0` are accepted as "disabled" slots.

**Parity-proven but fail-closed (recognized, loudly rejected if selected operationally).** These schemes pass per-scheme savepoint parity against an unmodified-WRF oracle but are **not** scan-wired into the GPU operational loop. Selecting one does **not** silently fall back or silently skip — it raises a specific, named error before any compute (`UnsupportedSchemeSelection` / `UnsupportedNamelistOption`):

- **MYJ PBL** (`bl_pbl_physics=2`) and its mandatory partner **Janjic-Eta surface layer** (`sf_sfclay_physics=2`) — savepoint-parity-proven CPU reference, GPU scan-wire is a post-v0.11.0 item.
- **New-Tiedtke cumulus** (`cu_physics=16`) — interface-compatible/accepted but not separately source-gated by a distinct WRF path.
- **Dudhia shortwave** (`ra_sw_physics=1`) and **classic RRTM longwave** (`ra_lw_physics=1`) — isolated-savepoint parity-proven; the operational radiation slot runs RRTMG only, so these are not yet operationally selectable (post-v0.11.0 jit/vmap rewrite + radiation-family dispatch).

**WRF-compatible namelist + fail-closed behavior.** The port reads WRF-exact namelist names and integer codes (`mp_physics`, `cu_physics`, `bl_pbl_physics`, `sf_sfclay_physics`, `sf_surface_physics`, `ra_lw`, `ra_sw`, `diff_opt`, `km_opt`, `dyn_opt`, …) via `gpuwrf run --namelist namelist.input`. Option validation is **fail-closed before any compute** and reports one of three honest outcomes ([`src/gpuwrf/io/namelist_check.py`](src/gpuwrf/io/namelist_check.py)):

- **implemented** — accepted and operationally wired;
- **recognized-WRF-not-yet-implemented** — a real WRF v4 scheme the port names but does not yet wire (fail-closed, names the scheme);
- **invalid** — not a recognized WRF v4 option at all (fail-closed).

**Dynamics.** Nonhydrostatic ARW mass core, RK3 + split-explicit acoustic substepping, flux-form advection (h=5 / v=3), WRF `w_damping` + Rayleigh upper damping (`damp_opt=3`), monotonic 6th-order filter (`diff_6th_opt=2`), constant-K diffusion (`diff_opt=2`/`km_opt=1`), and the WRF real-data-default 2-D Smagorinsky path (`diff_opt=1`/`km_opt=4`) — **v0.11.0: terrain-slope and map-factor deformation terms added to all diffusion paths**. Idealized gates (Skamarock warm bubble, Straka density current) pass 6/6 against published references + pristine WRF v4.7.1 ground truth; the operational dycore is finite/stable over full d02/d03 forecasts. Full dycore record: [`proofs/f7/DYCORE_STATUS.md`](proofs/f7/DYCORE_STATUS.md).

**Live multi-domain nesting (v0.11.0 new).** The `domain_tree` runtime (`src/gpuwrf/runtime/domain_tree.py`) drives d01→d02→d03 one-way live nesting with per-domain subcycling, WRF-faithful boundary update cadence, multi-domain synchronized output, and an optional two-way feedback gate (`src/gpuwrf/coupling/boundary_feedback.py`, disabled by default). A full nested d01→d02→d03 24 h one-way forecast over the Canary 9/3/1 km hierarchy ran finite and stable with final-lead T2 RMSE vs CPU-WRF of 1.03 K (d02) and 1.10 K (d03) — single case (20260521), one season; the RMSE numbers are from the final lead (h=24), not averages over all leads. Mean T2 RMSE over all 24 leads was 1.31 K (d02) and 1.67 K (d03). These numbers characterize the nesting fidelity on a representative case; no ensemble or TOST equivalence is claimed. Two-way feedback is implemented and unit-proven but has not been enabled in a long live forecast proof.

**WRF restart (v0.11.0 new).** `io/wrfrst_netcdf.py` writes and reads WRF-compatible `wrfrst` files covering all 75 prognostic/carry fields. Restart continuity is bit-identical: a-path (1..2N) vs b1-path (1..N) + b2-path restart (N..2N) produce identical final states on all 75 fields (`proofs/v0110/restart_continuity.json`).

**Conservation budgets closed (v0.11.0 new).** Dry-mass, total-water, and moist-static-energy relative residuals are 0.0 (fp64) on the validated d02 case (`proofs/v0110/conservation_budgets_closed.json`). Dry physics tendencies are now applied through `rk_addtend_dry` at each RK stage; non-dry physics deltas are applied post-dycore.

**Optional multi-GPU/DGX sharding (v0.11.0 new, single-GPU default = zero overhead).** `runtime/sharding.py` + `dynamics/sharded_horizontal.py` implement domain decomposition over a mesh of GPU devices. With `ShardingConfig.disabled()` (the production default), all sharding code is behind early-return guards — the committed proof shows 56/56 State field SHA-256 hashes bit-identical between the reference (single-GPU) and the sharding-disabled DGX path (`dgx_bitident_result.json`). The sharding path itself is verified on a fake-mesh (CPU-multi-device) and requires a DGX or NVLink cluster for real multi-GPU throughput.

**Recompile fix (v0.11.0 new).** `jit(_advance_chunk)` now compiles once and is reused on every subsequent chunk: chunks 2-3 run at 65.7 ms/step (18s/chunk), no per-chunk recompile (`proofs/v0110/recompile_fix2_3chunks.json`). The root cause was a non-JAX-contract-compliant `tree_unflatten` in `State` and `DycoreMetrics`.

**Operational precision.** v0.10.0 ships **fp64 as the operational mode**: the production daily-pipeline case builder hardcodes `force_fp64=True` in [`src/gpuwrf/integration/daily_pipeline.py`](src/gpuwrf/integration/daily_pipeline.py). ADR-007 gated-fp32 is retained only as an **experimental performance preview** and is **not** a v0.10.0 release path. It remains negative / no-go on the committed kernel evidence because the current workload is launch-tax / memory-bandwidth bound, not arithmetic-throughput-bound: the committed roofline and v0.10.0 Wave-B scoping measured fp32 at ~1.00x over fp64.

### Validation (v0.11.0)

Proof objects live under [`proofs/v0110/`](proofs/v0110/) (v0.11.0-specific) and the previous baseline proofs under [`proofs/v090/`](proofs/v090/) and [`proofs/v0100/`](proofs/v0100/).

- **Native real-init equivalence.** Native `wrfinput`/`wrfbdy` assembly is savepoint-parity-proven equivalent to `real.exe` at t=0 (v0.4.0; one-cell categorical-LSM residual documented). Native metgrid passed its gate at v0.3.0. This removes the CPU-WRF dependency for the initial/boundary state.
- **Per-scheme savepoint parity.** Each GPU-operational scheme (including v0.11.0 additions: KF cumulus, MYNN-EDMF mass flux, RRTMG topographic/slope radiation, terrain-slope diffusion) passes an fp64 math-faithfulness gate vs an unmodified-WRF oracle, under `proofs/`.
- **Coupled vs CPU-WRF, d02 (3 km).** Combined-physics GPU forecast (replay harness, radiation-ON) vs 28-rank CPU-WRF v4.7.1 `wrfout`, 24 h, one representative MAM case (`20260507_18z`). **Finite and stable all 24 h, no blow-up** (proof [`proofs/v0110/wind_regression_recovery/baseline/d02_coupled_skill.json`](proofs/v0110/wind_regression_recovery/baseline/d02_coupled_skill.json)). Per-lead RMSE vs CPU-WRF truth: **T2 within bar (3.0 K) at 24/24 leads** (mean 1.11 K, final 1.25 K); **V10 within bar (7.5 m/s) at 24/24 leads** (mean 3.59 m/s, final 4.33 m/s); **U10 within bar at 23/24 leads** (mean 4.43 m/s, final 8.06 m/s) — the final lead (h+24) transiently exceeds the 7.5 m/s bar (8.06 m/s), the same pre-existing episodic westerly under-prediction pattern as v0.9.0. **Beats persistence on 23/24 leads.** This is the **operational equivalence evidence** (single case, single season); no TOST or ensemble equivalence is claimed. The machine proof `status` is `FAIL` solely because the all-leads-within-bar predicate trips on that one final lead.
- **Coupled vs CPU-WRF, nested d01→d02→d03 (9/3/1 km), 24 h one-way.** Full nested hierarchy ran finite and stable on case `20260521_18z` with live parent-produced boundary packages. Final-lead (h=24) T2 RMSE vs CPU-WRF: d02 1.03 K / d03 1.10 K. Mean T2 RMSE over 24 leads: d02 1.31 K / d03 1.67 K. All fields finite on all domains at all leads. Single case, single season; no ensemble or TOST equivalence is claimed. Two-way feedback disabled in this proof. Proof: [`proofs/v0110/`](proofs/v0110/) (on validation branch `worker/gpt/v0110-val-nest24h`).
- **d03 1 km steep-terrain stability (KI-1 RESOLVED).** The prior open issue (gated-fp32 qke non-finite at h+1 over Tenerife steep terrain) is **closed** in v0.11.0 by the WRF-faithful qke cold-start seed (background TKE profile per `module_bl_mynnedmf.F:618-691 mym_initialize`) and the MYNN qke IEEE fmax/fmin fix. The d03 Tenerife replay ran **24 h finite in gated-fp32** with final-lead T2 RMSE 1.61 K (within 3.0 K bar), U10 5.13 m/s, V10 6.63 m/s (both within 7.5 m/s bar). **Requirement:** initial state must carry a WRF-faithful qke cold-start seed; a wrfinput with zero or near-zero qke may still trigger the edge. Proof: [`proofs/v0110/d031km_v0110.json`](proofs/v0110/d031km_v0110.json), [`proofs/v0110/val_d031km.md`](proofs/v0110/val_d031km.md) (on validation branch `worker/gpt/v0110-val-d031km`).
- **Conservation budgets closed (KI-conservation CLOSED).** Dry-mass, total-water, and moist-static-energy relative budget residuals are **0.0** (fp64) on the validated d02 case (`proofs/v0110/conservation_budgets_closed.json`). This uses the conserving physics coupling path (dry tendencies via `rk_addtend_dry`, non-dry physics applied post-dycore). Conservation unit tests (2/2 PASS) and analytical argument both confirm the fix is path-independent with respect to budget closure.
- **Restart bit-identity.** A-path vs B1+B2 (restart at midpoint): 75/75 fields bit-identical (`proofs/v0110/restart_continuity.json`).
- **DGX single-GPU-default bit-identity.** With `ShardingConfig.disabled()` (the production default): 56/56 State field SHA-256 hashes bit-identical between the reference trunk and the DGX-d2 sharding-disabled path.
- **Powered TOST equivalence (n=15).** The MAM corpus is prepared (forcing retained, CPU-WRF references assembled); the formal n=15 TOST has **not been scored for v0.11.0** — it is the powered analysis carried by the paper. **n=15 is honestly underpowered** (n≈27 needed to detect a 10% RMSE difference at α=0.05, β=0.20). The **operational equivalence evidence for v0.11.0 is the d02 coupled-skill result above**. **No "TOST PASS" / "statistical equivalence" is claimed** — doing so on an unscored underpowered corpus would be an over-claim. Margins + power analysis: [`.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md`](.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md).
- **End-to-end wall-clock speedup.** The v0.11.0 warm real-user d02 speedup inherits from v0.10.0 (≈ 2.47x warm vs 28-rank CPU-WRF, same workstation), as v0.11.0 does not change the per-step compute profile relative to v0.10.0 for the d02 single-domain path. The recompile fix removes the previous chunk-1 cost for chunks 2+ in long runs. The d03 1 km speedup is **not remeasured** for v0.11.0 (the d031km validation ran with a low-priority GPU wrapper; no clean timing was extracted). Kernel / compute-only ceiling (≈ 5.3×–7.84×) is a per-step number, not real-user wall-clock — do not conflate it with the 2.47x warm real-user headline.

**Standalone AIFS end-to-end (native init, no CPU-WRF dependency).** The full native pipeline (AIFS met_em → `build_real_init` → native LBC → `run_forecast_operational_segmented` → wrfout) ran stable and finite for 6 h on case `20260428_18z` (`proofs/v0110/standalone_e2e`). This confirms the native-init path is operational. No 24 h or RMSE claim is made from this 6 h smoke.

### Honest boundaries — what v0.11.0 does NOT claim

- **Not a universal WRF v4.** Standard regional ARW configs only. Exotic/rare features are README-TODO and fail-closed.
- **Not the full physics catalog.** WRF v4 has roughly 24 microphysics, 12 PBL, many surface-layer/LSM/cumulus/radiation options; v0.11.0 covers the common operational subset above. Everything else fails closed with a named reason.
- **Not full two-way nesting.** One-way live nesting is proven over a 24 h window. Two-way feedback is implemented behind a runtime gate but has not been enabled in a long live forecast proof. Nested in-loop `w` relaxation is off.
- **Not DFI / FDDA / spectral-nudging / adaptive-Δt** (fixed Δt only), **not aerosol-coupled microphysics** (Thompson-aerosol `mp=28`/Morrison-aerosol `mp=40`/NSSL fail closed), and **not urban (BEP/BEM) / lake / WRF-Chem** (these are rejected, not roadmap).
- **Free-running limited-area (run_boundary=False) on wide domains.** Free-running without lateral-boundary relaxation on wide domains (nx≈160+) can go unstable beyond ~14 h. The validated operational path uses boundary forcing. See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).
- **RRTMG intermediate gas optical depth (`taug`) in 4 UV bands.** The top-layer convention differs from the WRF oracle fixture in 4 UV bands (bands 9, 10, 12, 13). Integrated flux outputs pass tier-1 (< 0.05% rel); this is a pre-existing, isolated oracle-fixture discrepancy. Fix→v0.12.0. See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).
- **Known bounded residual (U10).** U10 final-lead RMSE (h+24) is 8.06 m/s vs the 7.5 m/s operational bar on the validated d02 case — the same pre-existing episodic evening-peak westerly under-prediction as v0.9.0. T2 and V10 are within bar at all leads; U10 beats persistence on 23/24 leads.
- **No powered n=15 TOST PASS.** The corpus is prepared but the formal equivalence analysis is the paper's deliverable, not a v0.11.0 release gate.
- **v0.2.0 paper tag not yet released.** The stable paper-baseline intended at v0.2.0 was never formally tagged. All prior releases (v0.1.0 and up) remain accessible in the git history and in the org repo; v0.2.0 stays accessible for paper claims.

A code-grounded, prioritized inventory of the remaining gap to a complete WRF v4 replacement lives in [`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md) and the v0.11.0+ full-port gap analysis under [`.agent/reviews/`](.agent/reviews/).

## Roadmap — delta to a complete WRF v4 port (post-v0.11.0)

Consolidated, honestly-prioritized ledger of everything still deferred / simplified / not-yet-faithful relative to official WRF v4, sorted by importance for an *optimal complete* port. Complexity: **S** ≈ 1–2 focused sprints · **M** ≈ 3–5 · **L** ≈ 5–10 · **XL** ≈ 10+. (v0.2.0→v0.11.0 already closed native real-init, prognostic Noah-MP, the terrain/map-factor core, the GPU-operational scheme set — Thompson, MYNN-EDMF/YSU/ACM2 PBL, MYNN-sfclay, Grell-Freitas + Kain-Fritsch cumulus, Noah-MP, RRTMG topo/slope, terrain-slope diffusion, live nesting, restart, and conservation budgets.)

| # | Item — delta vs official WRF v4 | Cmplx | Detail |
|---|---|---|---|
| **Tier 1 — blocks a complete standalone WRF v4 replacement** | | | |
| 1 | **Full multi-domain nested equivalence** — 24 h one-way is proven; two-way feedback + radiation-in-loop + in-loop w relaxation + 5-domain long-run equivalence remain carry-overs. | L | GPU_PORT_GAPS P0-1 |
| 2 | **Full `wrfout` variable coverage** — focused 64-variable writer vs WRF's 375 (missing: stochastic-seed arrays, Noah-MP snow-layer diagnostics). Blocks downstream tools. | M | GPU_PORT_GAPS P0-5; KNOWN_ISSUES KI-3 |
| **Tier 2 — physics fidelity (faithful to the pinned Canary suite)** | | | |
| 3 | **MYNN PBL completeness** — EDMF mass flux wired; `icloud_bl=1` cloud PDF (`bl_mynn_cloudpdf=2`) and `cloudmix` (`bl_mynn_cloudmix=1`) partial. Tied to the residual near-surface wind-skill gap. | M | GPU_PORT_GAPS P1-4 |
| 4 | **Thompson microphysics parity debts** — snow fall-speed approx, cloud-water sedimentation, invalid-column fallback. | M | GPU_PORT_GAPS P1-5 |
| 5 | **Positive-definite / monotonic scalar advection** — flux-adv frozen h5/v3; moisture safety via guards. | M | GPU_PORT_GAPS P1-6 |
| 6 | **RRTMG SW taug top-layer convention fix** — 4 UV bands fail intermediate oracle comparison; tier-1 flux outputs faithful; pre-existing. | S | KNOWN_ISSUES KI-6 |
| **Tier 3 — correctness / robustness debts** | | | |
| 7 | **Free-running open-lateral-boundary stability** — wide domains (nx≈160+) can blow up without boundary relaxation beyond ~14 h. Operational path is stable with forcing. | M | KNOWN_ISSUES KI-7 |
| 8 | **U10 episodic under-prediction** — final-lead breach on the validated d02 case (tied to MYNN cloud PDF). | S–M | KNOWN_ISSUES KI-4 |
| **Tier 4 — statistical / release closure** | | | |
| 9 | **Powered n=15 TOST scoring** — corpus prepared, not yet scored (the paper's equivalence claim). | S–M | KNOWN_ISSUES KI-5; ADR-029 |
| 10 | **v0.2.0 stable paper-release tag** — intended stable baseline never formally tagged. | S | `V0.2.0-PLAN.md` |
| **Tier 5 — performance (optional)** | | | |
| 11 | **Hand-fused-kernel rewrite for 1.4–1.8×** — optional Pallas/Triton branch (~30% of project). | XL | [`.agent/reviews/2026-06-05-gpt-hand-fused-kernel-feasibility.md`](.agent/reviews/2026-06-05-gpt-hand-fused-kernel-feasibility.md) |
| 12 | **Real multi-GPU throughput** — sharding code committed and bit-identity proven on fake mesh; DGX/NVLink cluster required for real throughput benefit. | M | `contracts/halo.py` |
| **Tier 6 — breadth / general WRF coverage (beyond the Canary suite)** | | | |
| 13 | **Full physics scheme matrix** — alternate MP/PBL/CU/RA/LSM families beyond the wired set (recognized-but-fail-closed). | XL | GPU_PORT_GAPS P1-2 |
| 14 | **FDDA / grid+obs nudging / spectral nudging** — none (only lateral-BC relaxation). | M–XL | GPU_PORT_GAPS P1-1 |
| 15 | **Map-projection / grid generality** — Lambert/Mercator/Polar + hybrid-eta C-grid only; no moving/global nests. | M | GPU_PORT_GAPS P2-1 |
| 16 | **Full WRF namelist parsing + loud rejection** of unsupported options. | S–M | `io/namelist_check.py` |
| 17 | **Additional wrfout diagnostics / auxhist streams** (downstream-driven). | S–L | GPU_PORT_GAPS P2-3 |

**Critical path to a *complete* port:** items **1 → 2** are the standalone-replacement chain; **3 / 4 / 5** are the highest-value fidelity levers (where the remaining wind/T2 skill lives). The perf rewrite (11) and breadth (13–17) are real but lower-leverage than finishing the nest + the fidelity tier.

## Core goals (immutable)

1. **GPU-native architecture.** Whole-state device residency after init. No host/device transfers inside the timestep loop without an ADR. Fused timestep-scale kernels, not micro-kernel launch storms.
2. **Operational skill parity with CPU WRF v4** on Canary L2/L3 cases: 24–72 h RMSE on T2, U10, V10 statistically equivalent under TOST at predeclared operational margins on a seasonal ensemble (n=15 floor today; n≈27–30 is the powered target).
3. **Performance vs 28-rank CPU WRF** on the same workstation, re-certified after every correctness fix (no stale speedup claims). The headline is the honest command-to-finish wall-clock ratio; kernel-level ratios are reported separately, never as the headline.
4. **Validation against WRF, not bitwise reproducibility.** Tiered pyramid: micro fixture / savepoint parity → physical invariants → short-run / timestep convergence → station-RMSE TOST equivalence.
5. **Forkable and auditable.** Every claim has a proof object on disk. Every architecture decision has an ADR with cross-model review.

## Where to look first (in this order)

| When you want to… | Read |
|---|---|
| Understand the project scope | [`PROJECT_CONSTITUTION.md`](PROJECT_CONSTITUTION.md), [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md), [`PROJECT_SPEC.md`](PROJECT_SPEC.md) |
| See the GPU-operational vs fail-closed physics matrix | [`src/gpuwrf/contracts/physics_registry.py`](src/gpuwrf/contracts/physics_registry.py), [`src/gpuwrf/runtime/operational_mode.py`](src/gpuwrf/runtime/operational_mode.py) (`_SCAN_WIRED_OPTIONS`) |
| Run a forecast | [`src/gpuwrf/cli.py`](src/gpuwrf/cli.py) — `gpuwrf run --namelist namelist.input …` |
| Check current known issues | [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) |
| See v0.11.0 proof objects | [`proofs/v0110/`](proofs/v0110/) |
| See prior release proofs | [`proofs/v090/`](proofs/v090/), [`proofs/v0100/`](proofs/v0100/) |
| See the full WRF v4 gap inventory | [`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md) |
| See prior versions (v0.2.0, v0.9.0, v0.10.0) | Accessible via git tags `v0.2.0`, `v0.9.0`, `v0.10.0` on the org repo; v0.2.0 is the stable paper-claims baseline |

## Run

```bash
# Validate a WRF namelist fail-closed (no GPU / no compile needed):
gpuwrf run --namelist <input-dir>/namelist.input --input-dir <case-dir> \
    --output-dir runs/my_forecast --domain d02 --hours 1

# Development check:
pytest -q
```

## Layout

```
.
├── PROJECT_CONSTITUTION.md          immutable end goal
├── ARCHITECTURE_PRINCIPLES.md       backend / runtime principles
├── VALIDATION_STRATEGY.md           four-tier validation pyramid
├── PRECISION_POLICY.md              FP64/FP32/BF16 rules
├── PERFORMANCE_TARGETS.md           profiler JSON schema + transfer rules
├── INTERFACE_CONTRACTS.md           GridSpec, State, Tendencies
├── RISK_REGISTER.md                 living risk list
├── docs/                            user-facing references
├── fixtures/                        manifest schemas + analytic samples + Canary slice
├── src/gpuwrf/                      implementation code
│   ├── contracts/                   frozen State / grid / physics_registry
│   ├── coupling/                    scan adapters + physics dispatch
│   ├── runtime/                     operational forecast loop
│   ├── physics/                     scheme kernels
│   ├── io/                          namelist check + wrfout/wrfinput I/O
│   └── integration/                 daily pipeline / native init
├── scripts/                         CLIs: check_*_done, validators
├── tests/                           pytest suite
├── proofs/                          per-milestone proof objects (JSON + reports)
└── publish/                         user-facing analysis + gaps TODO
```
