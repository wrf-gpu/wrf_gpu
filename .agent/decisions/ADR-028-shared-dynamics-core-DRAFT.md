# ADR-028 - Shared Dynamics Core

**Status:** DRAFT
**Date:** 2026-05-25
**Author:** GPT-5 Codex worker, M6b reframe shared-core sprint
**Triggered by:** Reframe critic verdict `REFRAME-TO-SHARED-CORE`.

## Context

The M6B savepoint ladder proved the validation composition bitwise, but the
first operational route retyped the same timestep composition in
`runtime/operational_mode.py`. The critic enumerated seven visible interface
mismatches: carry, RK/acoustic schedule, coefficient cadence, boundary lead
time, physics sequence, thermodynamic offsets, and precision. The subsequent
defect cascade showed that "operational composes own variants" prevented direct
validation-wrapper imports but did not prevent duplicated WRF-shaped math.

The principal GPU-core directive still binds: validation savepoint machinery is
not production runtime. The reframe target is therefore not validation-as-
operational. Per the critic, the path is to "reframe to shared core" and let
mode-specific wrappers own validation IO or operational policy.

## Decision

Validation and operational both import `gpuwrf.dynamics.core`.

Validation wrappers may not be imported by operational runtime. The principal
directive, pilots wrong if incompatible with GPU-optimized core, still binds via
the wrapper boundary: savepoint emission, HDF5 layout, fp64 strictness, and
snapshot dictionaries remain validation-only and absent from operational graphs.

Operational wrappers may own carry pruning, fusion, precision policy, kernel
selection, static segmentation, donation, and profiler-backed solver choice.
They may not hand-copy the RK/acoustic/coupled timestep math that is already
owned by `dynamics.core`.

## Implementation

- `src/gpuwrf/dynamics/core/acoustic.py` owns `AcousticCoreState`,
  `advance_mu_t_core`, `w_solve_core`, `acoustic_substep_core`, and
  `acoustic_scan_core`.
- `src/gpuwrf/dynamics/core/dycore.py` owns `rk_stage_core`-equivalent RK
  composition through `dycore_timestep_core` and a schedule encoded as data.
- `src/gpuwrf/dynamics/core/coupled.py` owns `coupled_timestep_core`, including
  dycore, physics, and boundary lead-time composition.
- `src/gpuwrf/dynamics/validation_wrappers.py` and the historical validation
  modules are compatibility wrappers around core.
- `src/gpuwrf/runtime/operational_mode.py` imports `dynamics.core` directly and
  no longer defines `_wrf_small_step_acoustic`.

## Alternatives

`STAY-THE-COURSE` was rejected because it would keep two independently
composed timesteps and continue exposing new interface defects after each local
fix.

`PIVOT-TO-VALIDATION-AS-OPERATIONAL` was rejected because savepoint-shaped,
fp64, HDF5/snapshot-oriented validation code would violate `PROJECT_PLAN`
§14.5.1 and the GPU-optimized-core primacy directive.

`HYBRID` sharing only RK/acoustic templates was rejected as too narrow because
the critic's remaining mismatches also covered boundary lead time, physics
sequence, thermo offsets, and precision.

## Consequences

- The old Amendment #1 is superseded narrowly: validation wrappers stay out of
  production, but pure numerical core code is shared.
- Validation parity remains the guardrail for moved math.
- Operational performance work must happen outside the core through wrapper
  policy and later evidence, not through a second mathematical composition.
- Any future GPU solver substitution must preserve the core call contract and
  carry its own ADR/proof object.

## Proof Objects

Expected proof objects for this sprint:

- `.agent/sprints/2026-05-25-m6b-reframe-shared-core/proof_b6_unchanged.txt`
- `.agent/sprints/2026-05-25-m6b-reframe-shared-core/proof_step1_parity_reframed.json`
- `.agent/sprints/2026-05-25-m6b-reframe-shared-core/proof_10s_bounded.txt`
- `.agent/sprints/2026-05-25-m6b-reframe-shared-core/proof_d2h_warmed.txt`
- `tests/test_m6b_shared_core_contract.py`

## Review

Resolve at PROPOSED after this sprint's B6 unchanged, real-IC step-1 parity,
10 s bounded probe, and warmed inter-kernel D2H gates pass.
