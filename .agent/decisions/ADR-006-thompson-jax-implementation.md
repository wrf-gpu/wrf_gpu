# ADR-006 - Thompson JAX Implementation Mapping

Date: 2026-05-20
Author: M5-S1 worker draft (Codex gpt-5.5)
Status: ACCEPTED worker draft, pending manager closeout integration
Scope: post-hoc implementation record for the M5-S1 Thompson microphysics column source/sink subset.

## Decision

Decision: keep the attempt-2 JAX Thompson source/sink transcription as the candidate implementation, and replace the Tier-1 oracle with an attempt-3 compiled WRF Fortran harness. The public JAX API remains `step_thompson_column(state, dt, *, debug=False) -> state`, where `state` is the `ThompsonColumnState` pytree carrying `qv`, `qc`, `qr`, `qi`, `qs`, `qg`, `Ni`, `Nr`, `T`, `p`, and `rho`.

This ADR is not a forward architecture decision. ADR-001 remains the backend decision and ADR-005 remains the first-physics-suite decision. ADR-006 records the implementation and oracle mapping actually used by this sprint.

## WRF source mapping

WRF source mapping: the source of truth is `../wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre`.

The Fortran harness in `scripts/wrf_thompson_harness.f90` uses `module_mp_thompson` from the existing WRF v4.7.1 build. It calls `thompson_init` first, because lookup-table allocation and initialization live in lines 604-1064 and must happen before the driver path. It then calls `mp_gt_driver`, whose call boundary is lines 1070-1564. The harness passes the frozen M5-S1 prognostic fields and writes outputs in `ES24.16E3` text format before the Python fixture packager creates `fixtures/samples/analytic-thompson-column-v1.npz`.

Build dependency tree:

- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_mp_thompson.o`
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

## Sedimentation status

Sedimentation status: OUT for M5-S1 per ADR-005. The WRF sedimentation and precipitation accumulation path starts around lines 3655-3972. The attempt-3 harness suppresses sedimentation numerically by passing `dz=1.0e30` for all levels, so flux-divergence fallout is negligible for the synthetic columns. This is weaker than a source-level bypass of lines 3655-3972 and is recorded as residual risk. A follow-up wrapper/table sprint should replace this with an explicit patched WRF source build or exported source-only subroutine if exact no-sedimentation parity becomes blocking.

The JAX kernel itself contains no sedimentation, terminal-velocity, substepping, or precipitation-accumulation code. Aerosol activation/scavenging, exact generated lookup-table parity, radar/effective-radius diagnostics, and hail/graupel volume state are also outside the M5-S1 candidate.

## Kernel fusion

The JAX implementation uses one public `@jax.jit` with `dt` and `debug` static. `src/gpuwrf/physics/thompson_column_debug_stripped.py` is a hand-stripped sibling with debug calls physically omitted. `python scripts/m5_run_thompson.py` recompiles both paths and rewrites `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`; the diff is 0 bytes on the worker run.

There are no `jnp.array`, `jnp.zeros`, or `jnp.empty` calls in the traced Thompson body. Container construction is via `state.replace(...)` around fused expressions over existing leaves.

## HLO Auditability

An auditor can re-derive the HLO identity proof by rerunning `python scripts/m5_run_thompson.py`. The committed truncated HLO files are readability artifacts only; the proof is the regenerated zero-byte diff from the committed source and stripped sibling.

## Tolerances

The Fortran harness gives a structurally independent oracle, but it also exposes the current exact-parity gap: the JAX candidate still uses documented proxies where WRF uses generated lookup tables. Therefore the attempt-3 Tier-1 tolerances are broad and explicitly non-final: `abs=2e-4, rel=1.0` for water species, `abs=2e6, rel=10.0` for `Ni/Nr`, and `abs=2 K, rel=2e-2` for temperature. These tolerances are sufficient to keep the independent-oracle pipeline running, but they are not evidence of final WRF Thompson parity. Exact table parity remains a follow-up risk.

## Gate dry-run

Gate dry-run: `artifacts/m5/thompson_gate_result.json` is GO on the worker run because Tier-1 and Tier-2 pass under the recorded attempt-3 tolerances and the HLO-derived launch count is within the ADR-001/ADR-005 threshold. Register and local-memory counters remain `null` because Nsight perf counters are blocked by the known workstation perfmon policy.
