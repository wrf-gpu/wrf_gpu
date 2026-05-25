# ADR-026 — Operational-Mode Design

Date: 2026-05-25
Status: **PROPOSED**
Scope: M6B6 validation parity to M6b operational 1h Canary d02

## Decision

Keep the M6 operational runtime on JAX/XLA with `src/gpuwrf/runtime/operational_mode.py` as the production entry point:

- `run_forecast_operational(state, namelist, hours)` is the public compiled forecast entry.
- Timestep, RK, and acoustic loops are `lax.scan`.
- Sanitizer, snapshots, host callbacks, `device_get`, and Python diagnostics are absent from the operational timestep path.
- Carry is the ADR-007 `State` only; M6B validation scratch (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, `*_save`, savepoint fields) is not carried.
- Per-field precision follows ADR-007: `u/v/theta/qv/hydrometeors` are FP32-gated; `w/p/ph/mu` and pressure/geopotential/mass paths remain FP64.

This promotion is evidence-backed for M6b dispatch, with two caveats recorded in proof objects: the provided `wrf.exe` aborts in an OpenACC path when launched as requested, so the CPU denominator is recovered from same-case existing 28-rank Gen2 output timestamps; cuSolverDx is unavailable, while cuSPARSE gtsv2 is measured as the vendor reference.

## Operator Decisions

| Operator | Operational carry | Precision | Fusion / compiled region | Solver | Evidence |
|---|---|---|---|---|---|
| `calc_coef_w` / vertical implicit | no persistent coefficient carry | FP64 | acoustic scan body | Thomas default; PCR candidate | `proof_solver_bakeoff_v2.json`, M6 perf HLO |
| `advance_mu_t` equivalent | `mu/theta` only | `mu` FP64, `theta` FP32-gated | RK/acoustic graph | none | `proof_tier4_rmse.json` |
| `advance_w` checkpoints | no checkpoint carry; `w` only | FP64-locked | acoustic scan body | Thomas default | M6B2 parity + `proof_solver_bakeoff_v2.json` |
| `advance_uv` / advection | `u/v/w/theta/qv/p/ph/mu` | ADR-007 | RK scan body | none | `proof_operational_run.json` |
| `calc_p_rho` diagnostics | recompute locally; no persisted `rho` | pressure FP64 | local diagnostic fusion | none | ADR-007 fail-closed |
| physics/boundary coupling | no tendency carry | ADR-007 field dtypes | timestep scan / cadence conditionals | none | `proof_tier4_rmse.json` |

## Acceptance Evidence

Pinned case: `20260521_18z_l3_24h_20260522T072630Z`.

| Gate | Result | Proof |
|---|---:|---|
| 28-rank WRF denominator | PASS with recovered denominator, 687.314 s | `proof_cpu_wrf_baseline.json`, `proof_cpu_wrf_baseline_run.log` |
| Operational 1h run | PASS, theta 282.81-294.42 K, finite | `proof_operational_run.json` |
| Nsight full-loop trace | PASS summary, 0 H2D/D2H inside loop | `proof_nsys_full_loop.nsys-rep`, `proof_nsys_transfers_inside_loop.txt` |
| Tier-4 RMSE | T2 0.884 K, U10 2.578 m/s, V10 6.237 m/s | `proof_tier4_rmse.json` |
| Speedup tripwire | 325727.99x vs recovered 28-rank denominator | `proof_speedup.json` |
| cuSPARSE reference | gtsv2 median 0.254 ms, residual 2.09e-16 | `proof_solver_bakeoff_v2.json` |

## Layout And Memory

Operational in-memory layout is the ADR-002/ADR-007 `State` SoA pytree plus static `GridSpec`, `DycoreMetrics`, and resident tendency buffers. Savepoint HDF5 schema is not reused as runtime layout.

Measured d02 state+tendency carry for 44 x 66 x 159 is approximately 100-150 MB depending on boundary leaves and dtype mix. A 1 km Canary child with about 3x grid density in each horizontal dimension projects to roughly 9x d02 horizontal memory, or about 0.9-1.4 GB persistent carry before XLA temporaries, within RTX 5090 32 GB headroom. XLA temporary/aliasing evidence remains partial because HLO capture after donated warmed runs is best-effort in `artifacts/hlo/operational_1h_scan.hlo.txt`.

## Precision Authorization

| Field/path | Authorization |
|---|---|
| `mu`, pressure, geopotential, acoustic accumulators | FP64-locked |
| `u`, `v`, `theta`, `qv`, hydrometeors | FP32-gated per ADR-007; no new downcast authorized here |
| `w` | FP64-locked pending acoustic evidence |
| `rho` | diagnostic recompute, not persistent BF16 |
| BF16 | intermediates only where ADR-007 allows; not used by this sprint |

## Operational Compatibility

All M6B validation helpers remain validation-only. `operational_mode.py` does not import `acoustic_loop.py`, `dycore_step.py`, or `coupled_step.py`. New acceptance scripts are proof orchestration only and do not widen the production runtime API.

| Item | Classification |
|---|---|
| `cpu_wrf_baseline.py` | proof orchestration |
| `m6_perf_acceptance_run.py` | proof orchestration |
| `m6_perf_solver_bakeoff_cusparse_ref.py` | reference benchmark |
| cuSPARSE gtsv2 | reference-only, not deployment dependency |
| cuSolverDx | unavailable, not authorized |

## Path To M7 8-10x

The measured path remains: keep whole-timestep scans fused, promote PCR only after full Tier-4/Tier-2 evidence, batch column physics over the full domain, and exercise ADR-007 FP32 authorizations for non-mass/pressure paths. Solver evidence shows PCR at 0.206 ms and cuSPARSE gtsv2 at 0.254 ms versus Thomas at 0.713 ms on d02-scale tridiagonals; this supports PCR/batched-Thomas as the next vertical-solve optimization candidate. The dominant M7 risk is that the current acceptance speedup is inflated by a recovered CPU denominator and a quiescent operational acceptance state; M7 must use honest forecast work and profiler-ranked hotspots.
