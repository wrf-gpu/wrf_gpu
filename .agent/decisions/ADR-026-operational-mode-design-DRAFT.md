# ADR-026 — Operational-Mode Design (DRAFT)

Date: 2026-05-25
Status: **DRAFT — acceptance blocked**
Scope: M6B6 validation parity to M6b operational 1h Canary d02

## Decision

Keep the M6 operational runtime on JAX/XLA and add `src/gpuwrf/runtime/operational_mode.py` as the production entry point:

- `run_forecast_operational(state, namelist, hours)` is the single public `@jit` forecast entry.
- Timestep, RK, and acoustic loops are `lax.scan`.
- Sanitizer, snapshots, host callbacks, `device_get`, and Python diagnostics are absent from the operational path.
- Carry is the ADR-007 `State` only; M6B validation scratch (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, `*_save`, savepoint fields) is not carried.
- Per-field precision follows ADR-007: `u/v/theta/qv/hydrometeors` are FP32-gated; `w/p/ph/mu` and pressure/geopotential/mass paths remain FP64.

This ADR stays DRAFT because Stage 3/4 full acceptance was not completed in this turn: no committed golden 1h operational-vs-WRF RMSE table and no 28-rank CPU WRF wall-clock comparison for the new operational entry point.

## Stage 1 Operator Plan

| Operator | Validation carry | Operational carry | Validation precision | Operational precision | Fusion target | Solver |
|---|---|---|---|---|---|---|
| `calc_coef_w` | `a/alpha/gamma`, `mut`, WRF metric savepoints | no persistent carry; coefficients local to acoustic body when vertical solver enabled | FP64 | FP64 mass/pressure path | fused into acoustic scan body | Thomas retained |
| `advance_mu_t` | `mu/mudf/muts/muave/ww/theta/ph_tend` | `mu/theta` only through `State`; `muts/muave/ww/ph_tend` dropped | FP64 validation | `mu` FP64, `theta` FP32-gated | fused into RK/acoustic graph | none |
| `advance_w` checkpoints | six savepoint/checkpoint families plus `tri_*` | no checkpoint carry; `w` only | FP64 | `w` FP64-locked | fused vertical solve inside acoustic scan | Thomas retained |
| `advance_uv` / advection | full savepoint state | `u/v/w/theta/qv/p/ph/mu` `State` fields | FP64-heavy validation | `u/v/theta/qv` FP32-gated, pressure/mass FP64 | RK scan body | none |
| `calc_p_rho` | diagnostic `p/rho/al/alt` style intermediates | recompute diagnostics locally; no persisted `rho` | FP64 | pressure path FP64; `rho` not persisted | acoustic/local diagnostic fusion | none |
| `sumflux` / physics coupling | physics and boundary tendency savepoints | no tendency carry; physics mutates `State` leaves only | FP64 validation | ADR-007 field dtypes | timestep scan body; radiation cadence via `lax.cond` | none |

Evidence basis: M6B0-R through M6B6 reports prove validation parity but classify scratch/savepoint fields as validation-only or undecided. ADR-007 authorizes only the listed FP32-gated production fields; fail-closed fields remain FP64.

## Stage 1.5 Solver Bakeoff

Proofs:

- `.agent/sprints/2026-05-25-m6-perf-design/artifacts/proof_solver_bakeoff.json`
- `.agent/sprints/2026-05-25-m6-perf-design/artifacts/hlo/*.hlo.txt`
- `.agent/sprints/2026-05-25-m6-perf-design/artifacts/proof_solver_bakeoff_nsight.nsys-rep`
- `.agent/sprints/2026-05-25-m6-perf-design/artifacts/proof_solver_bakeoff_nsight_loop_mem_cuda_gpu_mem_size_sum.json`

Measured d02-shape result on 45 vertical faces × 10,494 columns:

| Algorithm | Wall ms | Relative residual | Delta vs Thomas | HLO launch estimate | Decision |
|---|---:|---:|---:|---:|---|
| M6B2 `lax.scan` Thomas | 0.713 | 1.42e-16 | 0 | 18 | operational default |
| pure PCR | 0.206 | 2.32e-16 | 1.88e-16 | 6 | M7 optimization candidate |
| PCR + Thomas refinement | 0.733 | 8.50e-17 | 1.29e-16 | 24 | reject for now |
| XLA tridiagonal primitive reference | 0.310 | 2.55e-16 | 1.99e-16 | 6 | benchmark only |

Nsight capture was restricted to the warmed solver loop with `cudaProfilerStart/Stop`; the mem summary reports only Device-to-Device copies in the captured range and no Host-to-Device or Device-to-Host copies.

Despite PCR winning the isolated solve, ADR-026 does **not** promote PCR yet because the sprint contract requires cuSPARSE/cuSolverDx references and Tier-4 1h RMSE before solver promotion. Direct cuSPARSE/cuSolverDx bindings do not exist in this repo. Thomas remains the operational solver because it has M6B2 parity and full B-ladder correctness context.

## Stage 2 Runtime Build

Implemented:

- `src/gpuwrf/runtime/operational_mode.py`
- `scripts/m6_perf_design_canary_1h.py`
- `tests/test_m6_operational_mode_no_h2d.py`
- `tests/test_m6_operational_mode_parity_envelope.py`

Operational smoke proof:

- `.agent/sprints/2026-05-25-m6-perf-design/artifacts/proof_operational_smoke.json`
- Result: PASS on the small template grid, sanitizer absent, ADR-007 dtypes confirmed.

## Stage 3/4/5 Acceptance Status

Acceptance remains blocked:

- Tier-4 golden 1h RMSE was not run; no `T2/U10/V10` operational-vs-WRF table exists for the new runtime.
- 28-rank CPU WRF wall-clock comparison was not rerun against this runtime.
- Full operational 1h Nsight trace for the forecast loop was not captured; the committed zero-transfer Nsight proof covers the solver loop only.
- cuSPARSE/cuSolverDx references were not run.

Blocker proof:

- `.agent/sprints/2026-05-25-m6-perf-design/artifacts/proof_acceptance_status.json`

## Measured Path To M7 8-10x

The isolated vertical solve shows the immediate opportunity: pure PCR is 3.46x faster than the M6B2 Thomas scan on d02-shape coefficients while preserving ~1e-16 relative residual. To reach the M7 8-10x target, the next measured path is:

1. Add a real cuSPARSE/cuSolverDx reference or an explicit rejected-integration proof.
2. Run PCR in the full operational forecast and gate on Tier-4 golden 1h plus Tier-2 invariants.
3. Capture a full forecast Nsight trace, not only solver trace, and rank whole-timestep hotspots.
4. If PCR remains a whole-timestep hotspot win, promote it behind ADR-026 review; otherwise keep Thomas and focus on physics/radiation batching.
5. Use the launch budget from the current HLO: PCR has 6 estimated launches vs Thomas 18 for the solve. The forecast target should keep the full timestep in one compiled scan and avoid reintroducing per-operator launches.

## Consequences

Positive:

- Operational and validation modes are now physically separated in code.
- Scratch carry creep is blocked.
- Transfer discipline is machine-tested for the solver loop and source-tested for the operational runtime.

Negative / risks:

- ADR-026 cannot move to PROPOSED until full golden 1h Tier-4, 28-rank CPU comparison, and full forecast Nsight pass.
- The operational vertical solver is still conservative Thomas even though PCR measured faster, because promotion evidence is incomplete.
- The current smoke proof is not a meteorological acceptance proof.
