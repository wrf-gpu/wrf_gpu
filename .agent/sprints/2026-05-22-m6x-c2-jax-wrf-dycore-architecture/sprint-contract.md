# Sprint Contract — c2 JAX/WRF Dycore Architecture (USER AUTHORIZED 2026-05-22)

**Sprint ID**: `2026-05-22-m6x-c2-jax-wrf-dycore-architecture`
**Created**: 2026-05-22 ~12:50
**Status**: ACTIVE
**Trigger**: 11 c1 iterations + 4 bug-hunts + plan-critic + methodology meta-review + GPU dycore architecture scout converged on Option C (port architectural patterns, not code, from Pace/ICON4Py with Dinosaur JAX style). User authorized 2026-05-22 ~12:45.
**Branch**: `worker/codex/m6x-c2-jax-wrf-dycore-architecture` (from main)
**Worktree**: `/tmp/wrf_gpu2_c2_arch`

## Objective

Freeze and prove the architecture for a JAX/XLA WRF-compatible GPU dycore that can represent WRF map factors, hybrid-eta coefficients, smdiv, sixth-order hyperdiffusion, Rayleigh damping, and positive-definite/monotonic limiting **without host/device transfers inside timestep loops**.

This sprint does NOT claim operational closure. Primary deliverables: the architecture (typed state taxonomy, named modules) + first executable proof harness.

## Architecture references (per scout consensus)

- **Primary**: Pace/FV3 decomposition pattern (GridData with metrics, named DampingCoefficients, AcousticDynamics, HyperdiffusionDamping, RayleighDamping, FillNegativeTracerValues as explicit modules)
- **Secondary**: ICON4Py (explicit `NonHydrostaticConfig`, `metric_state_nonhydro`, `interpolation_state`, `IntermediateFields` dataclasses)
- **JAX style**: Dinosaur/NeuralGCM (pytrees, scan-friendly pure functions, coordinate-system objects)
- **Numerical truth**: WRF `dyn_em` source (cite for every formula)

## Acceptance criteria (8 ACs from scout)

- **AC1 ADR amendment**: produce ADR or architecture patch that amends ADR-002 with `DycoreMetrics`, WRF hybrid-eta coefficients, boundary-state policy, base-state policy, and scan-carry policy. Human approval required before broad implementation; manager closes the loop on patch protocol.
- **AC2 Metrics proof**: load or synthesize `msftx, msfty, msfux, msfuy, msfvx, msfvy` for both analytic flat fixture AND WRF fixture. Prove shapes, staggering, dtype, provenance, NO implicit host callbacks inside `jit`.
- **AC3 Hybrid-eta proof**: represent `c1h, c2h, c3h, c4h, c1f, c2f, c3f, c4f, dn, dnw, rdn, rdnw` as JAX arrays. Compare pressure/geopotential helper output against WRF savepoint OR analytic oracle.
- **AC4 Damping proof**: implement smdiv, 6th-order diffusion skeleton, Rayleigh/sponge skeleton, limiter skeleton as pure JAX functions with disabled-by-default config. Each module: isolated test proves identity when disabled + nontrivial finite effect when enabled.
- **AC5 Scan proof**: outer timestep `lax.scan` + nested acoustic `lax.scan` carry required diagnostics without Python-side mutation or host/device transfer inside loop. Transfer audit artifact.
- **AC6 Conservation/limiter proof**: for ≥1 scalar field, prove limiter preserves nonnegative mass within tolerance on analytic fixture. Report any deliberate non-conservation.
- **AC7 Integration proof**: warm-bubble OR WRF fixture for short window — first with new modules wired with conservative flags, then with stabilizers enabled. Proof object: finite-state + mass/energy diagnostics + comparison to previous c1 result.
- **AC8 Decision gate**: at closeout, recommend ONE of: continue C implementation, narrow to B throughput-only, or rollback architecture if proof objects show it's incompatible with JAX/XLA or WRF fixtures.

## File ownership

- `src/gpuwrf/contracts/grid.py` (extend with metrics)
- `src/gpuwrf/contracts/state.py` (split into State + BaseState + BoundaryState as needed)
- New files under `src/gpuwrf/dynamics/`:
  - `metrics.py` — WRF map-factor accessors + staggered metric interpolation
  - `hybrid_eta.py` — WRF hybrid-eta coefficient loading + pressure reconstruction
  - `damping.py` — smdiv + Rayleigh/sponge + their config dataclasses
  - `hyperdiffusion.py` — 6th-order WRF-style flux-limited diffusion
  - `limiters.py` — positive-definite + monotonic scalar correction
  - `acoustic_wrf.py` — WRF small-step acoustic scan (replaces current c1 acoustic.py)
  - `orchestrator.py` (or revised `rk3.py`) — RK3 + acoustic substep composition
- `tests/test_m6x_c2_*.py` (per-module isolated tests)
- `.agent/decisions/ADR-002-state-layout.md` AMENDMENT (per patch protocol)
- Optional new ADR `ADR-020-c2-dycore-architecture.md`

## Hard rules

1. NO line-by-line FV3, ICON, HOMME, or NeuralGCM port
2. NO MPI or multi-GPU decomposition this sprint
3. NO hidden sanitize acceptance
4. NO physics retuning to compensate for dycore instability
5. NO claims of GPU performance unless profiler + transfer-audit artifacts produced
6. Map-factors STATIC in GridSpec (NOT in time-step State pytree)
7. Hybrid-eta coefficients STATIC in GridSpec
8. Previous-pressure or smdiv memory in scan carry, NOT Python globals
9. Cite WRF `dyn_em/module_*.F:lineno` for every numerical choice
10. Cite Pace/ICON4Py for architectural pattern when used
11. BEFORE `/exit`: `git add . && git commit && git push`
12. `/exit` slash-command

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (MANDATORY after green AC list)
- Wall-time: **3-5 working days** aggressive / 1 week conservative
- Worktree: `/tmp/wrf_gpu2_c2_arch`
- Branch: `worker/codex/m6x-c2-jax-wrf-dycore-architecture`

## Proof object paths

- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/architecture.md`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/metrics.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/hybrid_eta.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/scan_transfer_audit.md`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/limiter_conservation.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/integration_warm_bubble.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/worker-report.md`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/manager-closeout.md` (after manager review)

## Risks

- c1 branch may not contain all A7/A11 fixes; architecture sprint must reconcile branch state at start
- WRF `wrfbdy` boundary may need broader schema (current 6 leaves possibly insufficient)
- JAX compilation pressure if all config is static — distinguish static enums from dynamic coefficient arrays
- Pace and ICON4Py prove architecture patterns, not WRF compatibility — WRF `dyn_em` remains numerical oracle
- First architecture sprint might prove JAX representation viable but still not close 600s warm-bubble — acceptable if proof objects isolate WHY

## End-goal

If c2-A1 closes GREEN with proof gates passing → c2-A2 implementation sprints (4-5 sprints, ~3 weeks aggressive) → M7 dispatch authorized → Canary 3km daily operational forecast.

If c2-A1 reveals JAX/WRF incompatibility → narrow to B throughput-only closeout + escalate dycore problem to user-decision territory (E3SM/SCREAM port, ML emulator, end-goal pivot).
