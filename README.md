# wrf_gpu

A GPU-native, WRF-compatible regional NWP system designed and built almost entirely by an AI agent swarm. The operational target is **Canary Islands daily forecasting** (3 km then 1 km) on a single-workstation RTX 5090.

This is not a port of legacy WRF source. It is a clean JAX rewrite that targets the GPU memory hierarchy from day one and validates against WRF as an oracle rather than inheriting WRF's architecture.

## Current status — v0.10.0

**v0.10.0 is the optimized-kernel release of a standalone, JAX-native, single-GPU WRF v4 ARW forecast system for standard regional configurations.** It performs **native real-init** (assembles `wrfinput`/`wrfbdy` from met_em-stage forcing, no `real.exe` and no CPU-WRF artifact for the initial/boundary state), runs a **nonhydrostatic split-explicit ARW dycore** on the GPU, exposes a **WRF-compatible namelist** with a **GPU-operational physics menu** and a **fail-closed boundary** on everything not yet ported, and ships with coupled CPU-WRF validation on the Canary 3 km (d02) case plus documented 1 km (d03) limits.

v0.10.0 is the optimized-kernel release. Relative to v0.9.0 it keeps the validated forecast and wrfout numerics UNCHANGED (bit-identical) while reducing Thompson sedimentation's faithful static substep cap from 64 to 16 — proven cap16==cap64 on the precip oracle and 24 h d02 hydrometeor/precip/skill checks. The warmed coupled d02 step improves 74.25 -> 64.76 ms, a 12.78% reduction (1.146x); the end-to-end real-user speedup vs 28-rank CPU-WRF rises from ~2.16x to ~2.47x (warm). Other candidate levers (acoustic fusion, gated-fp32, MYNN/PBL restructuring, daily-wrapper) were below the 1% exit gate, negative, not bit-identical, or fidelity/precision-gated; on the committed evidence a 2x warmed speedup is NOT WRF-faithfully achievable in this release — the kernel is at its WRF-faithful floor (the launch/occupancy headroom needs a hand-fused-kernel rewrite, a separate evaluated branch).

This is a deliberate step beyond v0.1.0, which was a single-domain **replay** path that consumed CPU-WRF/Gen2 artifacts for initialization. v0.3.0 added native metgrid; v0.4.0 added native real-init (proven equivalent to `real.exe` at t=0); v0.6.0 expanded the operational physics menu; v0.9.0 consolidated these into a standalone forecast system; v0.10.0 keeps those numerics unchanged and removes one faithful Thompson sedimentation inefficiency.

> **Honesty note.** Two distinct claims are kept separate throughout this README and must not be conflated:
> 1. **Native init** is proven equivalent to `real.exe` at t=0 (savepoint parity) and produces a stable forecast.
> 2. The **coupled skill validation** vs CPU-WRF on d02/d03 is run through the **replay harness** (parent-history replay, which consumes a CPU-WRF `wrfout` for the boundary/skill comparison). The validated coupled-skill run is *not* a from-scratch native-init run.

### What v0.10.0 is — GPU-operational capability

**GPU-operational physics menu (scan-wired into the operational forecast loop, WRF-oracle-gated).** These are the schemes the operational scan actually dispatches; the exact wiring is in [`src/gpuwrf/runtime/operational_mode.py`](src/gpuwrf/runtime/operational_mode.py) (`_SCAN_WIRED_OPTIONS`) and [`src/gpuwrf/coupling/scan_adapters.py`](src/gpuwrf/coupling/scan_adapters.py); the namelist-accepted matrix is in [`src/gpuwrf/contracts/physics_registry.py`](src/gpuwrf/contracts/physics_registry.py).

| Family | Namelist key | GPU-operational options (scan-wired) |
|---|---|---|
| Microphysics | `mp_physics` | 1 Kessler, 2 Purdue-Lin, 3 WSM3, 4 WSM5, 6 WSM6, 8 Thompson, 10 Morrison, 16 WDM6 |
| PBL | `bl_pbl_physics` | 1 YSU, 5 MYNN, 7 ACM2, 8 BouLac |
| Surface layer | `sf_sfclay_physics` | 1 revised-MM5, 5 MYNN-SL, 7 Pleim-Xiu |
| Cumulus | `cu_physics` | 1 Kain-Fritsch, 2 BMJ (fp64), 3 Grell-Freitas (scale-aware), 6 Tiedtke |
| Radiation | `ra_sw_physics` / `ra_lw_physics` | RRTMG SW + LW (the operational radiation slot runs RRTMG; `ra_sw=4` / `ra_lw=4`) |
| Land surface | `sf_surface_physics` | 2 Noah classic (explicit static/land bundle), 4 Noah-MP (`use_noahmp=True`) |

`mp_physics=0` (passive vapor), `bl_pbl_physics=0`, `sf_sfclay_physics=0`, `cu_physics=0`, and `ra_*=0` are accepted as "disabled" slots.

**Parity-proven but fail-closed (recognized, loudly rejected if selected operationally).** These schemes pass per-scheme savepoint parity against an unmodified-WRF oracle but are **not** scan-wired into the GPU operational loop. Selecting one does **not** silently fall back or silently skip — it raises a specific, named error before any compute (`UnsupportedSchemeSelection` / `UnsupportedNamelistOption`):

- **MYJ PBL** (`bl_pbl_physics=2`) and its mandatory partner **Janjic-Eta surface layer** (`sf_sfclay_physics=2`) — savepoint-parity-proven CPU reference, GPU scan-wire is a post-v0.10.0 item.
- **New-Tiedtke cumulus** (`cu_physics=16`) — interface-compatible/accepted but not separately source-gated by a distinct WRF path.
- **Dudhia shortwave** (`ra_sw_physics=1`) and **classic RRTM longwave** (`ra_lw_physics=1`) — isolated-savepoint parity-proven; the operational radiation slot runs RRTMG only, so these are not yet operationally selectable (post-v0.10.0 jit/vmap rewrite + radiation-family dispatch).

**WRF-compatible namelist + fail-closed behavior.** The port reads WRF-exact namelist names and integer codes (`mp_physics`, `cu_physics`, `bl_pbl_physics`, `sf_sfclay_physics`, `sf_surface_physics`, `ra_lw`, `ra_sw`, `diff_opt`, `km_opt`, `dyn_opt`, …) via `gpuwrf run --namelist namelist.input`. Option validation is **fail-closed before any compute** and reports one of three honest outcomes ([`src/gpuwrf/io/namelist_check.py`](src/gpuwrf/io/namelist_check.py)):

- **implemented** — accepted and operationally wired;
- **recognized-WRF-not-yet-implemented** — a real WRF v4 scheme the port names but does not yet wire (fail-closed, names the scheme);
- **invalid** — not a recognized WRF v4 option at all (fail-closed).

**Dynamics.** Nonhydrostatic ARW mass core, RK3 + split-explicit acoustic substepping, flux-form advection (h=5 / v=3), WRF `w_damping` + Rayleigh upper damping (`damp_opt=3`), monotonic 6th-order filter (`diff_6th_opt=2`), constant-K diffusion (`diff_opt=2`/`km_opt=1`), and the WRF real-data-default 2-D Smagorinsky path (`diff_opt=1`/`km_opt=4`). Idealized gates (Skamarock warm bubble, Straka density current) pass 6/6 against published references + pristine WRF v4.7.1 ground truth; the operational dycore is finite/stable over full d02/d03 forecasts. Full dycore record: [`proofs/f7/DYCORE_STATUS.md`](proofs/f7/DYCORE_STATUS.md).

**Operational precision.** v0.10.0 ships **fp64 as the operational mode**: the production daily-pipeline case builder hardcodes `force_fp64=True` in [`src/gpuwrf/integration/daily_pipeline.py`](src/gpuwrf/integration/daily_pipeline.py). ADR-007 gated-fp32 is retained only as an **experimental performance preview** and is **not** a v0.10.0 release path. It remains negative / no-go on the committed kernel evidence because the current workload is launch-tax / memory-bandwidth bound, not arithmetic-throughput-bound: the committed roofline and v0.10.0 Wave-B scoping measured fp32 at ~1.00x over fp64.

### Validation (v0.10.0)

The validation lane keeps the v0.9.0 forecast/wrfout numerics bit-identical and adds the v0.10.0 optimized-kernel proof objects under [`proofs/v0100/`](proofs/v0100/) (single representative MAM case; honest about the open d03 1 km gated-fp32 gate and the unscored n=15 TOST).

- **Native real-init equivalence.** Native `wrfinput`/`wrfbdy` assembly is savepoint-parity-proven equivalent to `real.exe` at t=0 (v0.4.0; one-cell categorical-LSM residual documented), and v0.3.0 native metgrid passed its gate. This removes the CPU-WRF dependency for the initial/boundary state.
- **Per-scheme savepoint parity.** Each GPU-operational scheme passes an fp64 math-faithfulness gate vs an unmodified-WRF oracle (regime-robustness insurance), under `proofs/`.
- **Coupled vs CPU-WRF, d02 (3 km).** Combined-physics GPU forecast (replay harness, radiation-ON) vs 28-rank CPU-WRF v4.7.1 `wrfout`, 72 h, one representative MAM case (`20260507_18z`). **Finite and stable all 72 h, no blow-up** (proof [`proofs/v090/d02_coupled_skill_72h.json`](proofs/v090/d02_coupled_skill_72h.json), run via the d02-replay harness with the validated stability namelist). Per-lead RMSE vs CPU-WRF truth: **T2 within bar (3.0 K) at 72/72 leads** (mean 1.06 K, max 1.42 K, final-hour 0.81 K); **V10 within bar (7.5 m/s) at 72/72 leads** (mean 3.21 m/s, final 2.97 m/s); **U10 within bar at 66/72 leads** (mean 4.79 m/s, final 4.00 m/s) — it transiently breaches the 7.5 m/s bar over lead hours 21-26 (2026-05-08T15:00:00 through 20:00:00 UTC; max 8.04 m/s) then recovers, an *episodic near-surface westerly under-prediction*, not a degrading instability. HFX/PBLH stay inside their informational bands (HFX mean 60 / final 45 W m⁻²; PBLH mean 182 / final 265 m). **Honest verdict:** the proof's machine `status` is `FAIL` *solely* because the all-leads-within-bar predicate trips on those 6 U10 leads; the final-hour Tier-4 RMSE passes on all of T2/U10/V10 and T2/V10 pass at every lead. This is the **operational equivalence evidence** (single case, single season), not a green-on-every-lead claim. The d02 gated-fp32 replay path is numerically close to fp64 and stable in the 3 h reverify proof (`proofs/v090/d02replay_2to3h_reverify.json`); only the d03 1 km gated-fp32 preview is known non-finite. `precip` is not separately scored here.
- **Coupled vs CPU-WRF, d03 (1 km).** The 1 km gated-fp32 preview is **OPEN / CARRIED OVER** and is **not** the v0.10.0 default mode. The d03 Tenerife forecast goes **non-finite after forecast hour 1** (qke the sole offending field, 3036 cells over about 69 steep-terrain columns); because no timesteps complete, no T2/U10/V10/PBLH/precip RMSE could be scored. A later qke->fp64 follow-up falsified the pure fp32-range-overflow diagnosis: qke still goes non-finite with the identical 3036-cell signature, so the issue is a qke/dynamics numerics robustness edge over steep 1 km terrain. **1 km is finite in full fp64** over the confirmed 0.3 h / 360-step window, but fp64 is ~1:64-throttled on the RTX 5090 so a full 24 h fp64 validation is impractically slow. Full write-up + carry-over: [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md), proofs [`proofs/v090/d03_1km_validation.json`](proofs/v090/d03_1km_validation.json), [`proofs/v090/d03_1km_validation_qkefix.json`](proofs/v090/d03_1km_validation_qkefix.json), and [`proofs/v090/d03_replay_finite_check.json`](proofs/v090/d03_replay_finite_check.json). *(The CPU 1 km reference is also indicative only — it ran inside a contended 5-domain nest, not a clean standalone.)*
- **Powered TOST equivalence (n=15).** Statistical equivalence of 24–72 h RMSE on **T2 / U10 / V10** under TOST at the ADR-029 predeclared margins (10% of the local CPU-WRF benchmark RMSE: **T2 ±0.215 K, U10 ±0.231 m/s, V10 ±0.275 m/s**). **n=15 is the binding floor and is honestly underpowered** relative to the ADR-029 target (n≈27 to detect a 10% RMSE difference at α=0.05, β=0.20). The result is labeled single-season (MAM) and underpowered, never an unqualified "equivalence PASS." Margins + power analysis: [`.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md`](.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md). **TOST result:** the n=15 CPU-WRF MAM corpus is **prepared** (forcing retained, CPU-WRF reference runs assembled/backfilled), but the **formal n=15 TOST has not yet been scored for v0.10.0** — it is the powered analysis carried by the paper, not a v0.10.0 ship gate. The **operational equivalence evidence for v0.10.0 is the d02 (3 km) coupled-skill result above plus v0.10.0 bit-identity to v0.9.0**. **No "TOST PASS" / "equivalence PASS" is claimed here** — doing so on an unscored n=15 would be an over-claim.
- **End-to-end wall-clock speedup.** Honest command-to-finish wall-clock (CPU wall-clock ÷ GPU wall-clock), single RTX 5090 vs 28-rank CPU-WRF v4.7.1 on the same workstation, compile-inclusive headline, applicable to the fp64 ship mode because fp32-vs-fp64 is measured/analysed at ~1.00× on this launch/bandwidth-bound workload (proof [`proofs/v090/speedup_benchmark.json`](proofs/v090/speedup_benchmark.json); bottleneck proof [`proofs/perf/compute_cycle_analysis.md`](proofs/perf/compute_cycle_analysis.md)).
  - **9/3 km nested (d02), real-user-time, WARM cache (operational daily cadence), 72 h — headline ≈ 2.47x.** This is the v0.9.0 conservative warm real-user ratio (~2.16x) updated by the v0.10.0 warmed coupled-step gain (1.146x), compile-inclusive, with the JAX persistent compile cache warm (the realistic daily-run scenario). The direct proof for the kernel gain is [`proofs/v0100/wave_b1_nsed16_timing.json`](proofs/v0100/wave_b1_nsed16_timing.json); the v0.9.0 real-user denominator is [`proofs/v090/speedup_benchmark.json`](proofs/v090/speedup_benchmark.json).
  - **9/3 km nested (d02), COLD first-ever launch, 24 h — not remeasured for v0.10.0; v0.9.0 was ≈ 1.33x** (pays the one-time RRTMG/physics/dycore XLA compile; amortizes away on subsequent runs via the persistent cache).
  - **1 km (d03) speedup — UNMEASURED / BLOCKED**, because the d03 1 km gated-fp32 preview NaNs at hour 1 and full-fp64 has no complete 24 h timing yet.
  - **Kept clearly separate (NOT the headline):** the prior published **kernel / compute-only (compile-EXCLUDED) ceiling of ≈ 5.3×–7.84×** ([`publish/runtime_optimization_analysis.md`](publish/runtime_optimization_analysis.md)) is a *steady-state per-step* number, not real-user wall-clock; do not conflate it with the v0.10.0 ~2.47x warm real-user headline. The compile-excluded *real-user* steady-state for this benchmark is ≈ 2.33×–2.79× (context only); the strict dt-matched floor (GPU forced to the CPU 6 s step) is ≈ 1.29× warm.

### Honest boundaries — what v0.10.0 does NOT claim

- **Not a universal WRF v4.** Standard regional ARW configs only. Exotic/rare features are README-TODO and fail-closed.
- **Not the full physics catalog.** WRF v4 has roughly 24 microphysics, 12 PBL, many surface-layer/LSM/cumulus/radiation options; v0.10.0 covers the common subset above. Everything else fails closed with a named reason.
- **Not terrain-faithful diffusion.** Both the constant-K and the new Smagorinsky paths are **flat-slab** (map-factor / coordinate-slope deformation terms dropped) — within tolerance for the Canary cases, not fully faithful over steep terrain. Terrain-slope diffusion is a post-v0.10.0 refinement.
- **Not full two-way nesting.** v0.5.0 one-way nesting is operator-proven over a short window; full nested equivalence (24 h / two-way d03 feedback / radiation-in-loop) is a post-v0.10.0 carry-over.
- **Not DFI / FDDA / spectral-nudging / adaptive-Δt** (fixed Δt only), **not aerosol-coupled microphysics** (Thompson-aerosol `mp=28`/Morrison-aerosol `mp=40`/NSSL fail closed; aerosol-State expansion is ADR-gated, post-v0.10.0), and **not urban (BEP/BEM) / lake / WRF-Chem** (these are rejected, not roadmap).
- **Known bounded residual.** A documented near-surface westerly excess persists in the standalone 24 h forecast (T2 correct, stable/finite); after multi-round debugging it is ruled out vs WRF against every faithful ported operator and is characterized as dynamical, not a fidelity bug ([`proofs/f7/DYCORE_STATUS.md`](proofs/f7/DYCORE_STATUS.md), v0.4.0 carry-over). The daytime-T2 behavior is the WRF land HFX behavior plus the Noah-MP T2MB land-T2 overwrite (now implemented) — so the faithful default T2 may differ from the WRF `wrfout` T2 by that LSM-overwrite term.

A code-grounded, prioritized inventory of the remaining gap to a complete WRF v4 replacement lives in [`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md) and the v0.10.0+ full-port gap analysis under [`.agent/reviews/`](.agent/reviews/).

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
| See proof objects | [`proofs/`](proofs/), [`proofs/v0100/`](proofs/v0100/) |
| See the full WRF v4 gap inventory | [`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md) |

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
