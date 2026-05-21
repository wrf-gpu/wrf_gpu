# M5-S3.zzzz Manager Closeout — SW PARITY ACHIEVED 🎉

**Sprint**: M5-S3.zzzz RRTMG SW cldprmc+spcvmc oracle + broadband closeout
**Status**: **CLOSED — Opus ACCEPT-SW-PARITY; first PARITY claim of M5-S3 cycle**
**Date**: 2026-05-21 ~15:50

## Headline

**First PARITY-class result in the 7-sprint M5-S3 RRTMG arc.** Strict Tier-1 SW PASS with all 9 per-field residuals 14×-2900× under threshold:
- flux_down: 0.0715 W/m² (vs 1.0 = 14× under)
- flux_up: 0.0468 (21× under)
- toa_up: 0.0267 (37× under)
- surface_down: 0.0343 (29× under)
- surface_up: 0.00617 (162× under)
- column_absorbed: 0.0354 (28× under)
- surface_absorbed: 0.0281 (35× under)
- heating_rate: 3.42e-8 K/s (vs 1e-4 = 2900× under)

## Root cause was a STACK, not a single hypothesis

Both M5-S3.zz reviewer-flagged hypotheses (R-8 cloud_safe floor, R-9 double-Eddington-then-blend) were **REJECTED against WRF source**:
- R-8: WRF actually USES `max(0.01, cldfrac)` floor (`:11030-11033`); JAX was correct
- R-9: WRF DOES call reftra separately for clear/cloud then blend outputs (`:8651-8670`); JAX was correct

The actual residual stack (7 small WRF-alignment defects):
1. MCICA pressure seed integer truncation precision
2. Liquid cloud radius table indexing (`int(radliq-1.5)`)
3. WRF climatological ozone branch (`o3input=0`)
4. Reftra `exp_tbl` lookup (not naive `exp()`)
5. Hard optical-depth cap removal (WRF doesn't have one)
6. Native-real (single-precision) at lookup-sensitive sites
7. column_absorbed = TOA_net − surface_net (standard convention)

Each O(1-20 W/m²); together collapsing 87 W/m² to 0.07 W/m². **Intermediate-oracle methodology proven sound end-to-end.**

## Verifiability triple ALL PASS

- `nm` symbols persisted + SHA `2dd3acc8…6c476d` matches manifest (closes M5-S3.zz §1.1 debt)
- 0 clip-pinning across 22 new cldprmc/spcvmc oracle arrays
- raw 443 launches == reported 443 (no fudge); gate honestly FALLBACK driven by LW + launch budget
- ADR-009 amendment ENDORSED: `SW-PARITY, LW-NOT-PARITY`

## Non-blocking follow-ups

- R-13: spcvmc bands 10/13 ztra precision residuals (max_rel 0.002/0.008) — washed out at broadband sum
- R-14: 443 launches → M5 closeout / Pareto-frontier optimization (separate from PARITY)
- ADR-009 attribution line missing M5-S3.zz mention (cosmetic)

## M5 RRTMG PARITY progress

| Sprint | State |
|---|---|
| SW: M5-S3.zzzz | ✓ SW-PARITY ACCEPTED |
| LW: M5-S3.zzzzz | 🟡 codex worker in flight (~30 min) |

After M5-S3.zzzzz LW broadband closes → full PARITY → ADR-009 SW+LW PARITY → M6-S8 operational T2 unblocks (subject to M6-S5 dycore cap lift + denominator selection).

## Process notes

**Intermediate-oracle methodology proven end-to-end**: M5-S3.z oracle infrastructure → M5-S3.zz hypothesis flags → M5-S3.zzzz oracle-driven confrontation. The 3-cycle dividend (S3.z + S3.zz + S3.zzzz) paid off with the first PARITY claim. Pattern transferable to LW S3.zzzzz.

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 15:50
