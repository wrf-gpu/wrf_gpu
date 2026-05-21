# M6-S1 Manager Closeout — Coupled Interface and Precision Boundary Freeze

**Sprint**: `2026-05-21-m6-s1-coupled-interface-freeze`
**Status**: **CLOSED — Opus reviewer ACCEPT-WITH-MINOR-FOLLOWUPS; M6 implementation UNBLOCKED**
**Date**: 2026-05-21 ~12:05
**Manager**: Claude Opus 4.7 (1M-context)

## What landed (commit `e50bcd2` worker + `7d84bce` reviewer + merge `2c6748a`)

Codex worker (22m clean delivery):
- **AC1** State pytree extension: 23 new SoA leaves (hydrometeors, numbers, qke, surface handles, precipitation accumulators) with units + shapes
- **AC2** Precision matrix: `PRECISION_MATRIX` per field; FP64-locked (mu/p/pgeop/w/surface stability/accumulators) vs FP32-gated (u/v/theta/qv/hydrometeors/numbers/qke)
- **AC3** Four coupling adapters: Thompson/MYNN/RRTMG/surface — wrap-only, **0 physics-kernel diff** verified via `git diff main...HEAD -- src/gpuwrf/physics/`
- **AC4** 100-step dummy coupled carry: **0 H2D / 0 D2H / 0 temporary bytes** (independently reproduced by reviewer)
- **AC5** Spacetime budget: 0.65ms/step (dycore 24 + Thompson 7 + MYNN 32 + surface 1 + RRTMG 170 every 10th step), HLO 5.19 MB, 320 launches/step at 16×16×30 dummy domain
- **AC6** ADR-010 with cross-refs to ADR-002 + ADR-007
- **AC7** File-ownership freeze declared for M6-S2..S8

Opus reviewer (9m) verdict: **12 PASS / 5 FOLLOWUP / 0 REJECT**

## 5 M6-S2 prerequisites (per reviewer R-3/R-5/R-7/R-9/R-13)

Bundled into M6-S2 contract on dispatch:

1. **R-3**: ratify FP32 storage downcast (or revert to FP64 until M6-S7 RMSE gates). Worker chose FP32 storage interpretation of "interface freeze"; defensible but must be ratified, not silently inherited.
2. **R-5**: thread real `GridSpec` metrics — replace `DEFAULT_DZ_M=100.0` placeholder in `physics_couplers.py:29`. On non-uniform terrain-following grid this is wrong by ~10× near surface and ~5× aloft.
3. **R-7**: measure `temporary_bytes_per_step` properly (currently hardcoded literal 0) or downgrade to null with "not measured" tag. H2D/D2H zeros ARE real measurements; the AC4 hard-check is still satisfied.
4. **R-9**: handle non-multiple-of-10 radiation cadence (current trailing `remainder = steps % 10` branch silently omits radiation for non-100-step counts). Cosmetic for the contracted 100-step run; driver must not inherit.
5. **R-13 / plan-critic amendment-3**: bundle boundary-forcing State extension (`u_bdy, theta_bdy, qv_bdy`, time-varying BC port) as planned M6-S1.b appendix work. Required before AIFS BC ingest.

## Reviewer's verifiability triple all PASS

- **0-byte transfer audit**: independently reproduced. Earlier `lax.cond` D2H bug fixed via static nested scans.
- **No physics modification**: `git diff main...HEAD -- src/gpuwrf/physics/` empty. Only `mynn_surface_stub.py` import-touched (existing hook), and `debug/snapshots.py` got a 5-line tree_leaves compatibility patch.
- **No `min(raw, cap)` fudge**: launch counts derived from compiled HLO scrape, not clamps.

## Reviewer's adversarial probes

- **R-15 (precision cast trace)**: State.replace silently downcasts FP64 → field dtype on every update. M4 dycore tests still pass but this is a real behavior change inside RK3 substeps — must be re-validated in M6-S5 (ADR-007 verdict) sprint.
- **R-16 (d02 scale-up)**: static `lax.scan` structure means HLO compile is grid-invariant; cell count grows 62× (16×16×30 → 160×67×45), runtime scales linearly, HBM persistent state ~62× = manageable on RTX 5090 32 GB. **No HLO blow-up risk.**

## M6 dispatch impact

- **M6-S2** (forecast driver): bundle 5 prerequisites; dispatch after M6-S2a closes.
- **M6-S2a** (Gen2 backfill + boundary replay + shared I/O): DISPATCHED in parallel with M5-S3.z.
- **M6-S3..S8**: file-ownership disjoint per ADR-010; can parallel after M6-S2 smoke.
- **M5-S3.z** (RRTMG intermediate-oracles): DISPATCHED in parallel with M6-S2a.

## Process notes

- Worker delivered clean (22m); reviewer caught the 5 minor follow-ups without falling for any spec-gaming.
- **Watchdog fix worked**: M6-S1 Opus reviewer's AGENT REPORT fired via the watchdog (file stable for 60s → multi-step kill + report). User flagged single-Enter unreliable → multi-Enter pattern now encoded in skill file (commit `?`).
- The dummy-coupled-carry pattern (`static nested scans for radiation cadence`) is reusable for M6-S2 — worker correctly diagnosed and fixed the `lax.cond` D2H trap that would have bitten M6-S2 too.

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 12:05
