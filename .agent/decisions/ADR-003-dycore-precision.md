# ADR-003 — Dycore Precision Draft

Date: 2026-05-19
Author: M4 worker draft (Codex gpt-5.5)
Status: draft for manager finalization
Scope: M4 reduced dry dycore: RK3, advection, acoustic substep, debug hooks, and tier-1/2/3 validation artifacts.

## Decision

Decision: retain fp64 for all M4 prognostic storage and all dycore arithmetic in this sprint. ADR-003 proposes a future validated downcast path only for non-acoustic tendency arithmetic after equivalent tier evidence is produced at fp32. No mixed-precision runtime change is made by M4.

Rationale: the M4 dycore is the reference path for later physics coupling. The proof objects show fp64 parity and invariants are clean: `artifacts/m4/tier1_advection_parity.json` reports `max_abs_err = 0.0`, `max_rel_err = 0.0`, and `pass = true`; `artifacts/m4/tier2_invariants.json` reports `mass_residual_relative = 0.0`, `qv_positivity_violations = 0`, `nan_inf_violations = 0`, and `pass = true`; `artifacts/m4/tier3_convergence.json` reports `observed_order = 3.963913678661959` against `expected_order = 3.0`. These are fp64 reference results, not authorization to downcast.

## Per-field precision:

| Field / component | Current precision | Proposed precision | Validation evidence | WRF-community reference |
|---|---:|---:|---|---|
| `state.u` | fp64 | fp64 retained for M4; fp32 candidate after pressure-gradient regression | Tier-2 mass residual 0.0 with fp64; no fp32 evidence yet | none in repo |
| `state.v` | fp64 | fp64 retained for M4; fp32 candidate after pressure-gradient regression | Tier-2 mass residual 0.0 with fp64; no fp32 evidence yet | none in repo |
| `state.w` | fp64 | fp64 retained for M4; fp32 candidate only if acoustic accumulator remains fp64 | Tier-2 mass residual 0.0 with fp64; no fp32 evidence yet | none in repo |
| `state.theta` | fp64 | fp32 storage candidate for dry advection after tier-1/tier-2 fp32 rerun | Tier-1 advection wrapper max_abs 0.0 at fp64; no fp32 evidence yet | common operational NWP uses mixed precision in selected thermodynamic paths, but no project citation yet |
| `state.qv` | fp64 | fp32 storage candidate in physics tendency path only after tracer positivity validation | Tier-2 qv positivity violations 0 at fp64; no fp32 evidence yet | none in repo |
| `state.p` | fp64 | fp64 retained for pressure-gradient/acoustic path | Tier-2 mass residual 0.0 at fp64; no fp32 evidence sufficient for acoustic | WRF operational experience noted in sprint contract: fp32 acoustic accumulators can introduce mass drift |
| `state.ph` | fp64 | fp64 retained while acoustic/geopotential coupling is validated | Tier-2 no NaN/Inf at fp64; no fp32 evidence yet | none in repo |
| `state.mu` | fp64 | fp64 retained for mass continuity | Tier-2 mass residual 0.0 at fp64 | none in repo |
| advection tendency `u/v/w/theta/qv/p` | fp64 | fp32 arithmetic candidate for non-acoustic tendency construction | Tier-1 and tier-2 fp64 pass; fp32 rerun required before change | none in repo |
| acoustic substep accumulator | fp64 | fp64 retained | Required by contract unless fp32 mass residual <= 1e-10 is proven; not proven | sprint-contract WRF-community warning |
| physics tendency arithmetic | not implemented in M4 | fp32 candidate in M5 with fp64 accumulation at coupling boundary | no M4 physics evidence; this is a plan only | operational NWP commonly tolerates fp32 in selected physics tendencies, but project must validate per scheme |

## Downcast plan:

1. Keep `State` and `Tendencies` constructors fp64 until a dedicated precision sprint reruns tier-1 and tier-2 with explicit fp32 candidate leaves.
2. First candidate downcast: non-acoustic advection tendency arithmetic for `theta` and passive tracers. Required proof: tier-1 max_abs within the fixture tolerance and tier-2 mass residual still <= 1e-10 for fp64 mass fields, with zero qv positivity and NaN/Inf violations.
3. Second candidate downcast: physics tendency arithmetic in M5, because those updates are column-local and often less sensitive than pressure-gradient/acoustic accumulation. Required proof: scheme-specific tier-1 fixture parity, tracer positivity, water/mass budget, and a profile report showing the downcast is worth carrying.
4. Do not downcast `mu`, `p`, `ph`, or the acoustic accumulator unless an explicit follow-up ADR shows fp32 mass residual <= 1e-10 on the M4/M6 short-run envelope. This draft does not provide that evidence.

## Validation evidence:

The M4 evidence package is:

- `artifacts/m4/tier1_advection_parity.json`: `pass=true`, `max_abs_err=0.0`, `max_rel_err=0.0`, fixture `analytic-stencil-3d-advdiff-v1`.
- `artifacts/m4/tier2_invariants.json`: `pass=true`, `mass_residual_relative=0.0`, `qv_positivity_violations=0`, `nan_inf_violations=0`, 100 iterations.
- `artifacts/m4/tier3_convergence.json`: `pass=true`, `observed_order=3.963913678661959`, `expected_order=3.0`.
- `artifacts/m4/transfer_audit.json`: zero post-init host-to-device and device-to-host bytes over 100 iterations.
- `artifacts/m4/spacetime_budget.json`: `temporary_bytes_per_step=0`, `state_bytes=14540800`, `tendency_bytes=14540800`, `wall_time_per_step_us=435.55664946325123`.

This evidence supports the fp64 M4 reference and a future downcast plan. It does not authorize fp32 production dycore arithmetic yet.
