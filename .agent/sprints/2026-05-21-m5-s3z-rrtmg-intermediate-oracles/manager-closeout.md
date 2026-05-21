# M5-S3.z Manager Closeout — RRTMG Intermediate-Oracle Extraction

**Sprint**: `2026-05-21-m5-s3z-rrtmg-intermediate-oracles`
**Status**: **CLOSED — Opus PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-4; M5-S3.zz Option 1 (SW) binding next**
**Date**: 2026-05-21 ~12:40
**Manager**: Claude Opus 4.7 (1M-context)

## What landed (commit `5f7fa54` + reviewer `d9e9aeb`)

Codex worker (~24min):
- **AC2 PASS** — intermediate-oracle NPZ (121 KB, well under 30 MB budget; SHA-pinned)
- **AC3 SW `taug`+`taur` per-band PASS** at branch level (all 14 SW bands)
- **AC3 LW Planck/dplank state PASS**
- **AC4 PASS** — LW source machinery (`dplankup/dplankdn` + `tfn_tbl`) wired
- **AC5 PARTIAL** — SW HLO reverted to 497 KB (within budget) via production-path revert to nearest-pressure; launches still 42 (target ≤10)
- **AC8 PASS** — per-band debt list landed

Honestly **FAIL** (root-caused):
- AC3 SW `sfluxzen` (band/g-point mis-allocation — found via §3 root-cause)
- AC3 SW `setcoef` precision (single vs double — WRF compile uses `-r4`)
- AC3 LW `taug+fracs` (untranscribed for 16 bands — debt to M5-S3.zzz)
- AC6 strict Tier-1: SW max flux-down 110 W/m², LW 70 W/m²
- AC7 ADR-009 correctly held at NOT-PARITY

## Opus reviewer verdict

**PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-4** with **binding M5-S3.zz Option 1 (SW-focused)** decision.

Rationale (sprint-success probability binds when both sprints needed):
- Option 1 SW-focused: ~85% probability, 8-16h
- Option 2 LW-focused: ~50% probability, 24-48h
- Sequenced 1 then 2 has higher joint success than 2-first

M5-S3.zz scope (per reviewer §4 binding):
1. Fix SW `sfluxzen` band/g-point allocation against WRF intermediate oracle
2. SW `setcoef` precision policy decision (recompile WRF `-r8` OR amend contract to single-precision floor)
3. Re-enable validated 14 SW branches via `lax.scan` fusion (≤10 launches, ≤500 KB HLO)
4. Strict Tier-1 SW pass: `abs ≤ 1 W/m² + rel ≤ 0.05`
5. ADR-009 → "SW-PARITY, LW-NOT-PARITY"

M5-S3.zzz scope (advance-bound): Option 2 LW closeout — 16 `taumol+fracs` branches transcribed against intermediate-oracle.

## Operational impact

- SW heating bias: 2.5 K/day per column peak (down from M5-S3.y's 3.1)
- LW heating bias: 5.2 K/day per column peak (unchanged — LW still nearest-pressure)
- 24h T2 drift: still 1-3 K corridor
- After M5-S3.zz (SW closes): ~0.7-1.5 K (LW-dominated)
- After M5-S3.zzz (LW closes): <0.5 K (M6 UNBLOCKS)

## M6 dispatch impact

- **M6 coupled forecast**: BLOCKED on **BOTH M5-S3.zz AND M5-S3.zzz** close
- M6-S2 + M6-S3 + M6-S4..S8 implementation work CAN proceed in parallel (file-disjoint with `physics/rrtmg_*`)
- M6-S8 OPERATIONAL VALIDATION blocked until RRTMG parity

## Process notes

- M5-S3.z worker delivered the M5-S3.y reviewer §5 methodology exactly: intermediate-oracle NPZ + per-band validation framework + honest per-band debt list. None of the anti-patterns from prior cycles recurred.
- **Watchdog + multi-Enter encoding worked**: Opus reviewer AGENT REPORT fired without manager manual Enter this time.
- Sprint cycle: A1 (synthetic) → A2 (clip-pinning) → A3 (groundwork) → S3.x (Eddington) → S3.y (partial native) → S3.z (intermediate-oracles) → next S3.zz (SW closeout). Each cycle adds permanent infrastructure; nothing wasted.

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 12:40
