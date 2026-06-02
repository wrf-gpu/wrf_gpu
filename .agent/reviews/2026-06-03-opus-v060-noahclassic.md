# Handoff — v0.6.0 Lane 14: Noah-classic LSM JAX port + WRF savepoint parity

Date: 2026-06-03
Author: Opus 4.8 MAX (frontrunner)
Branch: `worker/opus/v060-noahclassic` (worktree `.claude/worktrees/v060-noahclassic-opus`)

## Objective

Port WRF ARW **Noah LSM (classic, `sf_surface_physics=2`)** to JAX, WRF-faithful,
savepoint-parity-gated, against the frozen V0.6.0-S0 interfaces. Distinct from the
already-ported Noah-MP (option 4); shares the forcing/output coupling pattern but
implements the classic soil/veg/snow physics.

## Result: PASS

12 real Canary d03 land columns across 4 regimes (daytime/nighttime vegetated+bare,
snow, wet-precip) reproduce real WRF SFLX to the fp32-vs-fp64 oracle-dust floor:

| field | max abs err | predeclared tol |
| --- | --- | --- |
| TSK (T1) | 3.8e-5 K | 2e-3 K |
| TSLB (STC) | 1.5e-5 K | 2e-3 K |
| SMOIS (SMC) | 3.4e-8 | 5e-6 |
| SH2O | 3.4e-8 | 5e-6 |
| HFX | 8.6e-4 W/m2 | 2e-2 + 1e-4·rel |
| QFX | 2.1e-10 kg/m2/s | 1e-8 + 1e-4·rel |
| LH | 7.3e-4 W/m2 | 2e-2 + 1e-4·rel |
| GRDFLX | 1.3e-3 W/m2 | 2e-2 + 1e-4·rel |
| SNEQV/SNOWH/SNCOVR/ALBEDO | ≤1.7e-8 | ≤1e-5 |

Tolerances were FROZEN before the run; observed residuals sit far below them
(fp64 JAX vs single-precision WRF SFLX). Not loosened to pass.

## Files changed (file-disjoint; S0 interfaces untouched)

- `src/gpuwrf/physics/lsm_noah_classic.py` — the port (SFLX driver + NOPAC/SNOPAC,
  4-layer soil thermo HRT/HSTEP/TDFCND/FRH2O/SNKSRC/TBND/TMPAVG, soil hydrology
  SMFLX/SRT/SSTEP/WDFCND with Schaake infiltration, evapotranspiration
  EVAPO/DEVAP/TRANSP/CANRES, snow SNFRAC/ALCALC/CSNOW/SNOW_NEW/SNOWPACK/SNOWZ0,
  PENMAN, urban override, surface fluxes). fp64; jittable pure column-tile function.
- `proofs/v060/oracle/noahclassic_offline_driver.F90` + `build_driver.sh` — Fortran
  oracle factory: links compiled pristine `module_sf_noahlsm.o` + `module_sf_noahdrv.o`,
  reads real WRF tables via `SOIL_VEG_GEN_PARM`, reconstructs the driver forcing prep,
  calls real `SFLX` + `REDPRM` per column. EXTERNAL oracle, NOT a self-compare.
- `proofs/v060/build_noahclassic_savepoints.py` — column extraction + savepoint build.
- `proofs/v060/savepoints_noahclassic.json` — the gold savepoints (input snapshot +
  WRF output + full REDPRM parameter block, per column).
- `proofs/v060/gen_parity_report.py` + `proofs/v060/noahclassic_savepoint_parity_report.json`
  — the deliverable proof object (per-field/per-column error, regimes, PASS/FAIL,
  provenance, honest residuals).
- `tests/v060/test_noahclassic_parity.py` — the parity gate (predeclared tols).

## Commands run

```
# one-time Fortran oracle build (conda env wrfbuild, cores 0-3)
proofs/v060/oracle/build_driver.sh
# savepoint extraction (base python has netCDF4)
taskset -c 0-3 python3 proofs/v060/build_noahclassic_savepoints.py
# parity gate / report (CPU, fp64)
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 XLA_FLAGS=--xla_cpu_use_thunk_runtime=false \
  taskset -c 0-3 python3 -m pytest tests/v060/test_noahclassic_parity.py -q
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 XLA_FLAGS=--xla_cpu_use_thunk_runtime=false \
  taskset -c 0-3 python3 proofs/v060/gen_parity_report.py
```

## Key bugs found + fixed during the lane

1. **ALB/XLAI/EMBRD/Z0BRD resolution** — SFLX overwrites these in the SHDFAC-interp
   block AFTER REDPRM. The oracle now replicates that block and dumps the resolved
   values (`RESOLVED` line) so the port consumes WRF's exact surface characteristics.
2. **Urban override** — for `VEGTYP==ISURBAN` (SF_URBAN_PHYSICS=0) SFLX overrides
   SHDFAC=0.05, RSMIN=400, SMC{MAX,REF,WLT,DRY}, DF1=3.24, HRT CSOIL=3.0e6. Added.
3. **ZSOIL root cause** (the big one) — the savepoint builder stored the ZS layer-
   MIDPOINT depths `[-0.05,-0.25,-0.70,-1.50]` as ZSOIL, but SFLX's ZSOIL is cumulative
   `-SLDPTH` (= `-DZS`) `[-0.10,-0.40,-1.00,-2.00]`. The 2× shallower L1 concentrated
   canopy-drip infiltration → top-layer soil moisture ~2× over-accumulation, which
   cascaded into the energy balance. Fixed the ZSOIL constant; all fields then
   collapsed to the dust floor. (The oracle always used the correct internal ZSOIL;
   only the value fed to the JAX port was wrong.)

## Coupling / carry contract (S0 lane 14)

- State leaves writable: `t_skin` (TSK), `soil_moisture`, `mavail`.
- 4-layer land carry (`PhysicsCarry.land_surface`): TSLB/SMOIS/SH2O/CANWAT/SNEQV/
  SNOWH/SNCOVR/SNOTIME + S0 members `flx4,fvb,fbur,fgsn,smcrel(=SMAV),xlaidyn(=XLAI)`
  (UA-physics off → flx4/fvb/fbur/fgsn = 0). `num_soil_layers=4`.
- The `NoahClassicState`/`Forcing`/`Params`/`Output` NamedTuples are the lane-local
  pytrees; wiring into `runtime.operational_mode` + `PhysicsCarry.land_surface`
  family-merge is the manager-owned operational dispatch step (post-parity, per S0).

## Unresolved risks / carry-over

- **Operational coupler not wired**: this lane delivers the parity-gated physics
  kernel + adapter types only (per S0, integration is a serial manager step). A thin
  `noah_classic_coupler.py` (mirroring `noahmp_coupler.py`'s land/water masked blend
  + forcing assembly) is the next step before any integrated forecast.
- **No GPU performance claim** (CPU-only lane per resource constraint). The kernel is
  jittable and allocation-light, but the GPU transfer audit / profiler artifact is
  out of scope here.
- **Regime breadth**: 12 columns cover the present d03 land categories + a synthesized
  Teide snowpack + a synthetic wet-precip column. Frozen-soil (sub-freezing STC with
  ice) is exercised lightly (the FRH2O/SNKSRC path runs but the corpus columns are
  mostly unfrozen); a dedicated frozen-soil column would harden the phase-change path.
- **FRH2O Newton loop** is a fixed-10-iteration vectorized form with a converged-freeze;
  faithful for the converged columns used here. The Flerchinger explicit fallback
  (KCOUNT==0 branch) is not implemented (never taken in these regimes).

## Next decision

Accept the lane (PASS, real WRF oracle, frozen tols, honest residuals) and schedule
the operational coupler wiring as a follow-on serial step. No blockers.
