# c2 JAX/WRF Dycore Architecture

Date: 2026-05-22
Worker: codex
Status: AC1 architecture patch produced; executable skeleton and proof harness follow this document.

## Decision

Adopt a WRF-compatible JAX dycore architecture with four explicit data categories:

1. `State`: time-evolving prognostic and coupled physics fields only. The existing M6 `State` remains backward-compatible while c2 modules stop treating boundary and base fields as implicit prognostics.
2. `BaseState`: static WRF base fields such as `pb`, base geopotential `phb`, base theta, and terrain-following base mass. These may vary over terrain but do not belong in the high-frequency timestep carry.
3. `BoundaryState`: lateral forcing grouped outside the prognostic state. The c2 boundary schema includes `u/v/w/theta/qv/p/pb/ph/mu`, while the old six legacy boundary leaves remain only as a transition adapter.
4. `DycoreMetrics`: device-resident static grid data under `GridSpec`: `msftx/msfty`, `msfux/msfuy`, `msfvx/msfvy`, WRF hybrid-eta coefficients `c1h/c2h/c3h/c4h`, `c1f/c2f/c3f/c4f`, `dn/dnw/rdn/rdnw`, and `p_top`.

Map factors and hybrid coefficients are static grid data, not `State` leaves. Previous pressure and flux accumulators are scan carry, not Python globals.

## Module Layout

New c2 modules under `src/gpuwrf/dynamics/`:

- `metrics.py`: WRF map-factor loading at initialization, flat analytic metric fixtures, and JIT-safe metric accessors.
- `hybrid_eta.py`: pressure and mass-weight reconstruction from WRF hybrid coefficients.
- `damping.py`: disabled-by-default `smdiv` and Rayleigh damping skeletons.
- `hyperdiffusion.py`: disabled-by-default sixth-order diffusion skeleton.
- `limiters.py`: disabled-by-default positive-definite and monotonic limiter skeletons with mass diagnostics.
- `acoustic_wrf.py`: WRF-shaped acoustic nested-scan skeleton. It intentionally does not import c1 `acoustic.py`.
- `orchestrator.py`: outer timestep `lax.scan` plus nested acoustic `lax.scan`, carrying state, previous pressure, and flux accumulators.

The old M4/M6 modules remain in place for compatibility and historical tests. c2 implementation sprints should fill these new modules rather than patching c1 acoustic/advection internals.

## Scan Carry Policy

The c2 outer timestep carry is:

- `state`: current prognostic `State`
- `previous_pressure`: pressure memory for WRF `smdiv`
- `fluxes`: explicit pressure and theta accumulator skeletons, to be widened as WRF flux-form implementation lands

Nested acoustic substeps carry `(state, previous_pressure)` through `lax.scan`. No host/device movement is authorized inside either loop. Initialization-only file loaders may read NetCDF on host, then return JAX arrays before timestep entry.

## Numerical Anchors

WRF remains the numerical truth:

- `module_small_step_em.F:95-96`, `522-542`, and `562` anchor hybrid coefficients and `smdiv` pressure memory.
- `module_small_step_em.F:701-705`, `828-942`, `1094-1112`, and `1569-1584` anchor map-factor use in acoustic/small-step terms.
- `module_advect_em.F:157-162`, `1561-1566`, `2816-2920`, and `4364-4400` anchor staggered map factors in advective momentum tendencies.
- `module_big_step_utilities_em.F:1045-1047` anchors hybrid pressure reconstruction with `c3h/c4h` and `ptop`.
- `module_big_step_utilities_em.F:6120-6333` anchors Rayleigh damping as a named wind tendency mechanism.
- `module_big_step_utilities_em.F:6506-6605` and `6753-6898` anchor sixth-order diffusion config, coefficient construction, map-factor scaling, and up-gradient guards.

Architecture patterns, not code, are taken from prior art:

- Pace/FV3 pattern: metric-rich grid data, named acoustic dynamics, damping, hyperdiffusion, and tracer fixer components (`fv_dynamics.py`, `dyn_core.py`, `del2cubed.py`, `ray_fast.py`, `fillz.py`, `grid/helper.py`).
- ICON4Py pattern: explicit nonhydrostatic config, metric state, interpolation state, intermediate fields, and damping enums (`NonHydrostaticConfig`, `SolveNonhydro`, `IntermediateFields`, `dycore_states.py`).
- Dinosaur/NeuralGCM style: JAX pytrees, pure functional transforms, and scan-friendly composition.

## ADR-002 Amendment Summary

The proposed amendment is in `.agent/patches/2026-05-22-c2-adr-002-amendment.md`. It does not rewrite ADR-002 in place. It adds:

- `DycoreMetrics` as a child of `GridSpec`
- `BaseState` and `BoundaryState` as separate pytrees
- WRF hybrid-eta coefficient arrays as grid metrics
- explicit scan-carry policy for previous pressure and accumulators
- a transition rule: legacy boundary leaves may remain until coupling code migrates, but new dycore code should consume `BoundaryState`

## Risks

- The current skeleton proves representation and scan shape, not WRF numerical parity.
- `GridSpec` is still used as a static arg in older M4/M6 functions; c2 functions pass `DycoreMetrics` as a pytree to avoid embedding WRF-sized arrays in static cache keys.
- Warm-bubble integration proof is limited by the absence of `scripts/m6_warm_bubble_test.py` in this worktree; the c2 proof harness records this honestly and runs an analytic finite-state smoke instead.
