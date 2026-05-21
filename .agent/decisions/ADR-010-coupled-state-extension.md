# ADR-010 — Coupled State Extension and Adapter Freeze

Date: 2026-05-21
Author: M6-S1 worker (codex gpt-5.5 xhigh)
Status: PROPOSED for M6-S1 reviewer
Scope: M6 coupled state leaves, precision boundary registry, column-physics adapters, dummy coupled carry, and file ownership for M6-S2..S8.

## Context

ADR-002 selects a structure-of-arrays `State` pytree with one JAX array leaf per named field, C-grid staggering, and `(z, y, x-like)` row-major field layout. ADR-002 rejects an AoS rewrite because dycore and physics kernels normally touch selected named fields, not packed structs of all variables. M6-S1 extends that accepted SoA contract; it does not replace it.

ADR-007 narrows the previous fp64 lock into a per-field policy: `mu`, pressure/geopotential, pressure-gradient accumulation, and acoustic accumulators remain FP64; `u/v/theta/qv`, hydrometeors, number concentrations, and Thompson source/sink paths are FP32-authorized only under their gates; `w` remains locked by default until the empirical sound-wave test lands. This ADR converts that policy into the M6 code registry while preserving FP64 storage at mass/pressure/surface-stability/accumulation boundaries.

## Decision

The M6 coupled state remains ADR-002 SoA. The following leaves are added as separate JAX arrays; no tuple/list/dict field packing is introduced:

| Field(s) | Shape | Units | Precision boundary |
|---|---:|---|---|
| `qc, qr, qi, qs, qg` | `(nz, ny, nx)` | kg kg^-1 | ADR-007 FP32-gated |
| `Ni, Nr, Ns, Ng` | `(nz, ny, nx)` | m^-3 | ADR-007 FP32-gated |
| `qke` | `(nz, ny, nx)` | m2 s^-2 | ADR-007 FP32-gated |
| `ustar, theta_flux, qv_flux, tau_u, tau_v, rhosfc, fltv, t_skin, soil_moisture` | `(ny, nx)` | surface-layer scalars/flux handles | FP64 |
| `rain_acc, snow_acc, graupel_acc, ice_acc` | `(ny, nx)` | mm | FP64 |

Existing ADR-002 leaves are preserved: `u`, `v`, `w`, `theta`, `qv`, `p`, `ph`, `mu`. `ph` is the existing geopotential field corresponding to the ADR-007 `pgeop` row; `PRECISION_MATRIX` also contains `pgeop` as an alias for governance clarity.

## Precision Boundary

`src/gpuwrf/contracts/precision.py` now defines `PRECISION_MATRIX` as `(dtype, fp32_gate_required)` per field. The storage registry is built from that matrix. Locked FP64 rows are `mu`, `p`, `ph/pgeop`, `w`, all surface-stability handles, and all precipitation accumulators. FP32-gated rows are `u`, `v`, `theta`, `qv`, hydrometeors, number concentrations, and `qke`.

This implements ADR-007's authorization matrix as code but does not claim the M6-S5 full-domain batching verdict. The M6-S1 proof is interface residency and transfer behavior, not a speedup claim.

## Adapter Contracts

`src/gpuwrf/coupling/physics_couplers.py` freezes four synchronous adapter entry points:

- `thompson_adapter(state: State, dt: float) -> State`
- `mynn_adapter(state: State, dt: float) -> State`
- `rrtmg_adapter(state: State, dt: float) -> State`
- `surface_adapter(state: State, dt: float) -> State`

Each adapter slices the SoA state into transient column-batched views with vertical as the last axis, calls the existing M5 physics kernel unchanged, and reassembles a new `State`. The adapters cast updated fields back to the `PRECISION_MATRIX` storage dtype at the coupling boundary. The M6-S1 dummy driver uses synchronous ordering: dycore, Thompson, MYNN, surface, and RRTMG every tenth step. No asynchronous physics timing is introduced.

The MYNN/RRTMG adapters use an M6-S1 constant `DEFAULT_DZ_M = 100.0` because the contracted adapter signature does not carry `GridSpec`; M6-S2 owns the forecast driver/grid-metric route. This is a dummy-carry interface proof, not the final metric-aware forecast integration.

## Proof Object

The contracted proof objects are:

- `artifacts/m6/coupled_dummy_carry.json`
- `artifacts/m6/spacetime_budget.json`

The generated 100-step dummy carry uses domain `[16, 16, 30]` and reports:

- `host_to_device_bytes_post_init = 0`
- `device_to_host_bytes_post_init = 0`
- `temporary_bytes_per_step = 0`
- `wall_time_per_step_ms = 0.646770759485662`
- `kernel_launches_per_step = 320`
- `hlo_bytes = 5193670`

The zero transfer result was obtained after replacing the initial dynamic radiation-cadence `lax.cond` with static nested ten-step scans. The first version produced 1-byte D2H predicate transfers; the committed version removes that runtime predicate.

## File Ownership Freeze For M6-S2..S8

After M6-S1 review, downstream M6 work must keep the following ownership split unless a new contract explicitly changes it:

| Sprint | Owner files |
|---|---|
| M6-S2 forecast driver | `scripts/m6_run_coupled_forecast.py`, `src/gpuwrf/coupling/driver.py`, forecast-driver tests, `artifacts/m6/coupled_driver/` |
| M6-S3 surface + Noah-MP | `src/gpuwrf/physics/surface_layer.py`, `src/gpuwrf/physics/noah_mp.py`, `scripts/m6_extract_surface_fixture.py`, surface/land fixtures and tests |
| M6-S4 Tier-2 coupling | `src/gpuwrf/validation/tier2_coupled.py`, `scripts/m6_run_tier2_coupled.py`, `artifacts/m6/tier2/` |
| M6-S5 ADR-007 verdict | `scripts/m6_full_domain_batching.py`, `artifacts/m6/performance/`, profiler verdict tests |
| M6-S6 Tier-3 TSC1.0 | `src/gpuwrf/validation/tier3_coupled.py`, `scripts/m6_run_tsc.py`, `artifacts/m6/tier3/` |
| M6-S7 Tier-4 probtest | `src/gpuwrf/validation/tier4_probtest.py`, `scripts/m6_run_tier4.py`, `artifacts/m6/tier4/` |
| M6-S8 operational + closeout | `scripts/m6_compare_gen2.py`, `src/gpuwrf/validation/operational.py`, `artifacts/m6/operational/`, M6 closeout draft |

M6-S2 may use the frozen adapter entry points. M6-S3 may widen the surface-layer protocol through its owned new modules, but it must not rewrite the M6-S1 adapter contract without a reviewed contract amendment. Physics kernels under `src/gpuwrf/physics/thompson_column.py`, `mynn_pbl.py`, `rrtmg_sw.py`, and `rrtmg_lw.py` remain frozen by M6-S1.

## Consequences And Risks

The positive consequence is that M6-S2..S8 can start from a single coupled `State` layout and stable adapter names. Field-level dtype ownership is now machine-readable and test-covered.

Known risks:

- The dummy adapters prove interface and residency only; they do not prove operational correctness, conservation closure, or the M6-S5 4x feasibility verdict.
- RRTMG remains M5-S3.x debt-sensitive; this ADR wraps the current kernels unchanged and does not claim radiation transfer-solver parity.
- `DEFAULT_DZ_M` is a deliberate M6-S1 placeholder. M6-S2 must thread real grid metrics into the coupled driver without changing this sprint's State shape.

## References

- ADR-002 state layout: `.agent/decisions/ADR-002-state-layout.md`
- ADR-007 precision policy: `.agent/decisions/ADR-007-precision-policy.md`
- M6-S1 proof script: `scripts/m6_run_dummy_coupled.py`
- M6-S1 tests: `tests/test_m6_state_extension.py`, `tests/test_m6_precision_matrix.py`, `tests/test_m6_dummy_coupled.py`
