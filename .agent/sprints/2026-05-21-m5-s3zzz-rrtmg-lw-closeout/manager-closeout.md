# M5-S3.zzz LW Manager Closeout

**Sprint**: M5-S3.zzz RRTMG LW closeout
**Status**: **CLOSED — Opus PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-LW-TAUMOL; M5-S3.zzzzz LW broadband binding next**
**Date**: 2026-05-21 ~15:15

## What landed

Codex worker (~2h35m wall):
- 16/16 LW bands `taug` + `fracs` PASS intermediate-oracle gate (`abs ≤ 1e-8 + rel ≤ 1e-4`)
- All bands FULL_BRANCH_ACCEPTED with WRF source citations per band
- Native LW table loader from pinned raw RRTMG_LW_DATA
- ADR-009 correctly held NOT-PARITY
- No SW touch (M6-S3.zzzz file-disjointness preserved)

## Honest gap

Tier-1 LW broadband still FAILS — root cause now downstream `cldprmc_lw + rtrnmc` transfer/source (analog to SW M5-S3.zz outcome).

## Operational impact

LW residual 47-60 W/m² broadband → 0.5-1 K/day T2 drift → 5-10 K cumulative at 24h. **Operationally unusable for M6 Tier-4 RMSE** until M5-S3.zzzzz closes.

## M5-S3.zzzzz scope (per Opus §3 binding decision)

**Scope**: LW `cldprmc_lw` + `rtrnmc` intermediate-oracle dumps + per-quantity per-band per-layer validators + branch fixes.

**Sequencing**: PARALLEL with M5-S3.zzzz under **manager interface-freeze** of shared files:
- `scripts/wrf_rrtmg_harness.f90` dump record name prefixes: `cldprmc_sw_*` / `spcvmc_*` (SW S3.zzzz) vs `cldprmc_lw_*` / `rtrnmc_*` (LW S3.zzzzz)
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` validator naming: `validate_sw_cldprmc_*` vs `validate_lw_cldprmc_*`
- `artifacts/m5/rrtmg_per_band_status.json` JSON schema: `sw_cldprmc_bands` vs `lw_cldprmc_bands` disjoint keys

File-disjoint production code: SW (`rrtmg_sw.py`) vs LW (`rrtmg_lw.py`).

Estimated wall: 16-32h.

## M5 RRTMG PARITY path

M5 RRTMG PARITY = S3.zzzz (SW broadband) + S3.zzzzz (LW broadband) BOTH close → ADR-009 SW-PARITY+LW-PARITY → M6-S8 operational T2 binding gate meaningful.

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 15:15
