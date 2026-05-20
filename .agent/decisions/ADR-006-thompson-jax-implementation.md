# ADR-006 - Thompson JAX Implementation Mapping

Date: 2026-05-20
Author: M5-S1 worker draft (Codex gpt-5.5)
Status: ACCEPTED worker draft, pending manager closeout integration
Scope: post-hoc implementation record for the M5-S1 Thompson microphysics column source/sink subset.

## Decision

Decision: keep the JAX Thompson source/sink candidate implementation, with attempt-4 WRF-order sequencing and cloud-ice-number fixes, and use a compiled WRF Fortran harness as the Tier-1 oracle. The public JAX API remains `step_thompson_column(state, dt, *, debug=False) -> state`, where `state` is the `ThompsonColumnState` pytree carrying `qv`, `qc`, `qr`, `qi`, `qs`, `qg`, `Ni`, `Nr`, `T`, `p`, and `rho`.

This ADR is not a forward architecture decision. ADR-001 remains the backend decision and ADR-005 remains the first-physics-suite decision. ADR-006 records the implementation and oracle mapping actually used by this sprint.

## WRF source mapping

WRF source mapping: the source of truth is `../wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre`.

The Fortran harness in `scripts/wrf_thompson_harness.f90` uses a locally compiled WRF v4.7.1 `module_mp_thompson` object. It calls `thompson_init` first, because lookup-table allocation and initialization live in lines 604-1064 and must happen before the driver path. It then calls `mp_gt_driver`, whose call boundary is lines 1070-1564. The harness passes the frozen M5-S1 prognostic fields and writes outputs in `ES24.16E3` text format before the Python fixture packager creates `fixtures/samples/analytic-thompson-column-v1.npz`.

Build dependency tree:

- `data/scratch/module_mp_thompson_nosed.o` compiled from `../wrf_gpu/.../module_mp_thompson.F.pre` with the attempt-4 no-sedimentation patch
- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_mp_radar.o`
- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/share/module_model_constants.o`
- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/frame/module_wrf_error.o`
- harness-local stubs for namelist and single-rank DM helpers used by Thompson table initialization

`gfortran` is not installed on this workstation and the WRF `.mod` files are NVHPC-built, so `scripts/wrf_thompson_harness_build.sh` uses `nvfortran` for ABI compatibility. The compiled harness lives only under `data/scratch/` and is referenced by SHA-256 in the fixture manifest's `files` list. The manifest keeps `source: wrf-derived` because the M1 manifest schema only permits `analytic` or `wrf-derived`; the more specific `wrf-thompson-via-fortran-harness` marker is recorded in `source_commit`.

The JAX candidate maps these WRF sections:

- driver prep and density formula: lines 1070-1274
- saturation helpers: lines 5444-5495
- cloud condensation adjustment: lines 3456-3556
- Berry-Reinhardt warm-rain autoconversion: lines 2242-2258
- rain-cloud-water collection shape: lines 2260-2268
- Srivastava-Coen rain evaporation: lines 3561-3636
- cloud-ice/snow/graupel deposition and sublimation: lines 2709-2770
- rain freezing and snow/graupel melting: lines 2658-2669 and 2845-2889
- tendency bookkeeping and final mass/number constraints: lines 2967-3260 and 4033-4142

Attempt 4 changed the JAX process order to follow WRF checkpoints: source/sink staging and conservation (2917-3247), working-state update before condensation (3250-3273), cloud condensation/evaporation (3456-3558), rain evaporation (3561-3638), instant cloud-ice melt/cloud-water freeze (4005-4031), and final write/balance (4033-4142). It also changed cloud-ice number handling to match lines 2719-2727: `pni_ide` is active only in the sublimation branch, while positive deposition partitions mass without creating new `Ni`.

## Sedimentation status

Sedimentation status: OUT for M5-S1 per ADR-005. The WRF sedimentation and precipitation accumulation path starts around lines 3655-4003. Attempt 4 replaced the attempt-3 `dz=1.0e30` workaround with a local source patch: `scripts/wrf_thompson_harness_build.sh` copies `module_mp_thompson.F.pre` to `data/scratch/module_mp_thompson_nosed.F90` and inserts a no-sedimentation patch immediately before the sedimentation flux loops. The patch zeroes `vtrk`, `vtnrk`, `vtik`, `vtnik`, `vtsk`, `vtgk`, `vtngk`, `vtck`, and `vtnck`, so the sedimentation loops at lines 3854-4003 execute with zero terminal velocities and no fallout flux. The harness now passes physical `dz=1000 m`.

The JAX kernel itself contains no sedimentation, terminal-velocity, substepping, or precipitation-accumulation code. Aerosol activation/scavenging, exact generated lookup-table parity, radar/effective-radius diagnostics, and hail/graupel volume state are also outside the M5-S1 candidate.

## Kernel fusion

The JAX implementation uses one public `@jax.jit` with `dt` and `debug` static. The process-order refactor is still a single fused JAX call; it only reorders existing source/sink helpers and splits warm-rain collection from rain evaporation so the checkpoints match WRF source order. `src/gpuwrf/physics/thompson_column_debug_stripped.py` is a hand-stripped sibling with debug calls physically omitted. `python scripts/m5_run_thompson.py` recompiles both paths and rewrites `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`; the diff is 0 bytes on the worker run when HLO identity holds.

There are no `jnp.array`, `jnp.zeros`, or `jnp.empty` calls in the traced Thompson body. Container construction is via `state.replace(...)` around fused expressions over existing leaves.

## HLO Auditability

An auditor can re-derive the HLO identity proof by rerunning `python scripts/m5_run_thompson.py`. The committed truncated HLO files are readability artifacts only; the proof is the regenerated zero-byte diff from the committed source and stripped sibling.

## Tolerances

The Fortran harness gives a structurally independent oracle and attempt 4 restored the ADR-005 strict Tier-1 tolerances: `abs=1e-10, rel=1e-8` for water species and `abs=1e-3, rel=1e-6` for `Ni/Nr`; `output_T` is recorded with strict `abs=1e-8, rel=1e-8`. These strict tolerances currently fail. The attempt-4 max absolute errors are `qv=1.4304079020558032e-05`, `qc=1.517228938283358e-04`, `qr=4.760876436193939e-06`, `qi=1.3708094759935232e-04`, `qs=1.447943623500527e-04`, `qg=1.5218435328806104e-05`, `Ni=126975.12500000044`, `Nr=67300.453125`, and `T=0.040290844661740266 K`. The order fix confirmed the diagnosis by reducing the main temperature error below 0.1 K, but exact table/moment parity remains unresolved.

## Gate dry-run

Gate dry-run: `artifacts/m5/thompson_gate_result.json` is expected to report `FALLBACK`/correctness failure after attempt 4 because Tier-1 does not meet the restored strict tolerances. This is not a backend performance fallback claim; it is a physics-parity blocker recorded in `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/BLOCKER-m5-s1-attempt4-tolerance.md`. Register and local-memory counters remain `null` because Nsight perf counters are blocked by the known workstation perfmon policy.
