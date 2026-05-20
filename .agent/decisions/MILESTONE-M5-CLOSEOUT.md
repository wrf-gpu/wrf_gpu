# Milestone M5 Closeout — First Physics Suite

**Milestone**: M5 — First Physics Suite
**Status**: **CLOSED**
**Date**: 2026-05-21 ~00:45 (post-midnight overnight close)
**Manager**: Claude Opus 4.7 (1M-context)

## Exit gate satisfaction

Per MILESTONES.md M5 goal: "implement one WRF-compatible physics column subset" — minimum already met by M5-S1 alone. **M5 closes with TWO physics column subsets implemented + supporting ADRs**:

| Component | Status | Closing artifact |
|---|---|---|
| M5-S0 (first-scheme decision-gate scout) | ✓ closed (Thompson selected) | `ADR-005-first-physics-suite.md`, commit `09a3738` |
| M5-S1 Thompson microphysics column | ✓ closed | `manager-closeout.md`, merge `d768194` + `00e7ee8` |
| M5-S1.x Thompson lookup-table export | ✓ closed (partial, deferred residual) | `manager-closeout.md`, merge `fe959d2` + `1868545` |
| **ADR-007 precision policy** | ✓ closed | merge `445c49f` + `6c9df22` |
| M5-S2 MYNN2.5 PBL column | ✓ closed | `manager-closeout.md`, merge `989f143` + `e4abc86` |

Per ADR-005 deferred-schemes section: RRTMG radiation and Noah-MP land surface explicitly defer to M5-S3..N or M6/M7 boundary depending on surface-coupling / spectral-table infrastructure readiness. **They are NOT required for M5 close.**

## What M5 proved

1. **JAX backend (ADR-001) handles real branchy WRF physics** end-to-end. Thompson microphysics (8 hydrometeors, ~6 active source/sink processes, lookup-table interpolation) and MYNN2.5 PBL (prognostic TKE + implicit vertical solve) both compiled, ran, and produced physically-consistent column outputs under `@jit` with documented hot-path discipline.
2. **Fortran-harness oracle pattern works** for structural anti-tautology when the WRF object is available (Thompson). When it's not (MYNN-EDMF object path mismatch), source-derived harness is acceptable with explicit anti-tautology gap and reliance on Tier-2 conservation + operational-RMSE validation in M6.
3. **Tier-2 conservation is the load-bearing physical sanity gate** at column-fixture level. Thompson water_residual=2.67e-12; MYNN momentum_residual=2.46e-16. Both at fp64-noise level → kernels are physically sound regardless of Tier-1 strictness.
4. **Tier-1 fixture parity is a transcription-bug sanity check**, not the binding gate. The validation-philosophy memory locked this in: operational RMSE on `U10/V10/T2` at 24h/72h is the binding M6/M7 gate. Per-cell parity below the operational noise floor is operationally irrelevant.
5. **ADR-007 precision policy** (Gemini-triggered): FP64-only infeasible (RTX 5090 throttled to 1.8 TFLOPS FP64 vs 9950X CPU 2.2 TFLOPS); mixed-precision feasible IF full-domain physics batching closes the M5 column-microfixture launch-bound gap (M6 coupled-run is the empirical test). M4 dycore already hits 215× FP32 vs CPU FP64. Authorization Matrix lists per-field permissions gated by operational-RMSE impact.
6. **Multi-AI workflow validated**: Gemini side-runner caught 2 coefficient bugs (lami CIE2, graupel CGG11/CGE11) that codex worker, codex diagnosis, and Claude Opus tester all missed. Claude Opus reviewer caught a Gemini hallucination (CGG11 numerical value) by actually running `math.gamma()`. Bug-fix parallel-pair rule paid off; Gemini policy refined to reactive-only-on-complex-bug-chase + architecture-tiebreak for quota conservation in M6/M7.
7. **Reusable infrastructure**: `tridiagonal_solver.py` (XLA primitive wrapper) is reusable for any future vertical-implicit scheme. `data/fixtures/thompson-tables-v1.npz` is the first WRF lookup-table export with reproducible extractor.

## Known residual debt → M6 prologue (folded into one prologue sprint)

1. **Thompson HLO-safe table-gather/fusion design** (from M5-S1.x). Rain-freezing tables extracted and pinned but not wired into JIT hot path because naive 4-D gathers caused 23 launches.
2. **Thompson process-level residual closure** (from M5-S1.x). Remaining `qr/Nr` rain-evap, `qg` graupel sublim/melt, cloud-water freezing/nucleation, number-balance — process-level not table-level. M6 operational RMSE will determine if these need a targeted fix.
3. **MYNN harness WRF-object-linked rebuild** (from M5-S2). Link against `module_bl_mynnedmf*.o` rather than source-derived standalone. Quantifies the anti-tautology gap from M5-S2 against real WRF.

## Risks carried forward into M6

- **FP64-only mixed-precision conditional**: ADR-007 says feasible IF full-domain batching closes M5 microfixture launch-bound gap. M6 is the empirical test — if it fails, project re-scopes (data-center GPU or ML-hybrid emulator per ADR-007 alternatives).
- **Anti-tautology weakness on MYNN side**: Tier-2 conservation strong but Tier-1 oracle is author-shared. M6 vs Gen2 backfill on `U10/V10/T2` is the corrective.
- **Surface layer + Noah-MP land + RRTMG radiation** all NOT implemented. M6 short forecast is column-physics-only or uses bulk surface stub; M7 needs the surface-coupling work.

## Per ADR-005 follow-on hooks

- **Noah-MP land surface** → M7 (needs surface/SST/static-geog proof object first)
- **RRTMG radiation** → M5-S3 or M6 boundary (needs table governance + solar geometry + gas/cloud optical inputs)
- **Real surface layer (Monin-Obukhov etc.)** → M6/M7 surface-coupling sprint
- **Cumulus parameterization** → revisit after coupled validation (3km explicit convection should suffice for v0)

## Next milestone — M6 Coupled Short Forecast

Per user directive 2026-05-20 evening: "if you get to milestone end, you have my approval to continue with milestone 6 after consensus with gpt on it's detailed plan."

**Next action**: dispatch codex gpt-5.5 xhigh to draft M6 detailed milestone plan. Manager (Claude Opus) reviews for consensus. After consensus → dispatch M6 implementation sprints.

M6 scope per MILESTONES.md:
- Couple dycore (M4) + physics (M5) for short-forecast windows
- Short forecast driver
- Conservation checks (Tier-2)
- Drift envelope (Tier-3 short-run timestep convergence)
- Initial Tier-4 small-ensemble prototype (per PROJECT_PLAN §7)
- Validation source per `.agent/references/cpu-wrf-baseline.md`: Gen2 backfill `wrf_l3/`, AIFS month files in `data/aifs_single/`

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 00:45
