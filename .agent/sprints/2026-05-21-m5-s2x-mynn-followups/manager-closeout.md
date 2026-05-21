# M5-S2.x Manager Closeout — MYNN Follow-Ups

**Sprint**: `2026-05-21-m5-s2x-mynn-followups`
**Status**: **CLOSED — Opus reviewer ACCEPT; merged to main as GO_CARRYFORWARD**
**Date**: 2026-05-21 ~10:42
**Manager**: Claude Opus 4.7 (1M-context)

## What landed

Codex worker (single 27m delivery, commit `7f9f4f1`, merged `9625d73`):

- **AC1 independent budget probe**: WRF-vs-JAX (not JAX-vs-JAX). `wrf_mynn_harness.f90:121-125` appends real `mynn_tendencies` outputs `du, dv, dth, dqv` as cols 19-22; `tier2_mynn.py:110-181` consumes WRF columns and compares against JAX one-step `(state_next-state)/dt`. Max abs residuals: `u=2.6e-5, v=9.4e-6, theta=1.6e-6, qv=2.4e-10` (target ≤1e-3).
- **AC2 radicand**: Path A chosen. `_flux_richardson` at `mynn_pbl.py:175-179` unguarded; matches WRF `module_bl_mynnedmf.F90:1918` plain SQRT. `tests/test_m5_mynn_radicand.py:13-37` exercises discriminant-positive negative-radicand boundary; both WRF and JAX NaN.
- **AC3 surface-layer interface for M6-S3**: ADR-008:34-47 with all 7 inputs (`ustar, theta_flux, qv_flux, tau_u, tau_v, rhosfc, fltv`), outputs, units, sign convention, RK3 timing. `mynn_surface_stub.py:27-43, :76-85` typed `SurfaceFluxes` + `surface_layer(state) -> SurfaceFluxes` hook. WRF citations (`:3421, :3428, :1575, :4436, :4195, :4257, etc.`) reviewer-spot-checked correct.
- **AC4 honest accounting**: 35 raw == 35 reported, HLO 279 KB < 300 KB, 0 post-init transfers, debug-vs-stripped diff 0 bytes.
- **AC5 tests**: 11/11 MYNN tests pass.

## Reviewer verdict

Opus 4.7 reviewer (23m fresh-context): **ACCEPT — close as GO_CARRYFORWARD**. Independent probes ran:
- `nm` confirmed 5 WRF MYNN-EDMF symbols defined in harness ELF text (not worker stubs).
- 45 of 48 array entries genuinely differ between WRF harness output and JAX (cross-AI probe).
- Direct probe at ri∈{1.1, 1.5, 1.9, 2.5, 0.5}: NaN propagation verified through `jnp.minimum(NaN, rfc)`.
- 3 spot-checked WRF citations exact.
- No spec-gaming pattern detected.

## Non-blocking follow-ups (M6-S3 / M6 fold-on)

1. Tighten independent-budget tolerance 1e-3 → ~5e-5 after operational Tier-4 RMSE validation.
2. Widen `SurfaceLayerState` protocol with `z_a, z0, z0h, z0q, theta_skin, soil_state` (or threaded route) before M6-S3 Monin-Obukhov plug-in.
3. EDMF mass-flux extension when daytime convective-BL T2/qv2 RMSE evidence demands.
4. Env hygiene (`/tmp` 97% full) before next full-pytest audit run.

## M6 dispatch impact

**Unblocked** for M6-S3 dispatch (this sprint provided the interface contract). M6-S3 may proceed in parallel with M5-S3.y RRTMG when prologue closes.

## Process notes

Sprint executed cleanly: contract → worker → reviewer → ACCEPT in 50 min total. Verifiability triple (`nm` + non-clipped + non-vacuous-tolerance) all passed. No anti-pattern recurrence. The 2-AI workflow (codex worker + Opus reviewer) functioning as designed.

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 10:42
