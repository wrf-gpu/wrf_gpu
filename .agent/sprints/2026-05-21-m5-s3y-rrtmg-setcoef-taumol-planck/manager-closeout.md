# M5-S3.y Manager Closeout — RRTMG setcoef + taumol + Planck-source attempt-1

**Sprint**: `2026-05-21-m5-s3y-rrtmg-setcoef-taumol-planck`
**Status**: **CLOSED — Opus reviewer PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3; M5-S3.z binding next**
**Date**: 2026-05-21 ~12:00
**Manager**: Claude Opus 4.7 (1M-context)

## What landed

Codex worker (24m delivery; self-flagged "do not accept"):

### Four PERMANENT artifacts (preserved across all future sprints)

1. **AC0 PASS — Eddington oracle rebuild**. `module_ra_rrtmg_sw.F` `kmodts=2→1` patch applied to both `/mnt/data/.../wrf_gpu_src/` AND `/home/enric/src/canairy_meteo/.../wrf_src/` (build-rule path). Harness rebuilt: SHA `25c88aa4...`. `nm` confirms all WRF symbols still resolved (`spcvmc_sw, rtrnmc, taumol_sw, taumol, setcoef_sw, setcoef, cldprmc_sw, cldprmc`).
2. **Native SW + LW table extraction**. `extract_rrtmg_tables.py` extended with WRF `swcmbdat/cmbgb*` reduced-g pattern. New NPZ SHA `9d8bedbf...`, 4.2 MB. Contains: `sw_absa/absb/selfref/forref/sfluxref` (per-g per-band gas absorption + solar source), `lw_totplnk/totplk16` (Planck tables). Real WRF data, byte-for-byte from source READ-lists.
3. **Faithful `_sw_setcoef` port**. `rrtmg_sw.py` adds `_SWSetCoefState` + vectorized JAX port of WRF `setcoef_sw` (jp/jt/fac00..fac11/indfor/indself/colamt*) matching `module_ra_rrtmg_sw.F:2843-3099`.
4. **LW Planck-source replacement**. `rrtmg_lw.py:287-292` replaces the M5-S3.x grey `σT⁴·g_weight` with WRF `totplnk` table interpolation × `delwave·π·10⁴` flux scaling. Quantified improvement: LW column-net-heating residual **88.25 → 73.67 W/m²** (17% improvement).

### Regressed / incomplete (intentional rollback target in M5-S3.z)

- **AC2 14-band SW `taumol_sw` expansion** shipped without per-band oracles to gate against → SW HLO **1.31 MB** (vs 500 KB budget); SW launches **36** alone (vs 10 cap).
- **AC4 LW `taumol` per-band + Planck fractions**: NOT done (dominant remaining LW correctness driver).
- **AC6 per-band WRF harness output**: NOT done. `tier1_rrtmg_per_band.json` `{"produced": false, "reason": "M5-S3.y worker did not complete the WRF harness per-band flux extension."}` — honest deliverable miss. **This is the methodological blocker for M5-S3.z.**
- **AC7 launch fusion**: 40 → 52 total (worse).
- **AC8 strict Tier-1 + ADR-009 PARITY**: FAIL. ADR-009 status correctly held at NOT-PARITY.

## Reviewer verdict

Opus 4.7 reviewer with WATCHDOG launcher (first real test of upgraded auto-notify pattern — fired clean):

- **Verifiability triple all PASS**: `nm` symbols preserved, 0 clip-pinning on new tables, raw counts honest (no `min(raw, cap)` fudge).
- **Verdict: PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3** — accept the 4 permanent artifacts; bind M5-S3.z to intermediate-oracle methodology.
- Rejected alternatives: REJECT (would discard 4 good artifacts), ACCEPT-as-parity (dishonest), REJECT-revert (Eddington rebuild + native tables + LW Planck-source are unrelated to SW regression).

## Operational impact

Per validation-philosophy memory:
- SW heating bias: 3.1 K/day per column peak (slightly worse than M5-S3.x's 2.5 K/day due to SW regression)
- LW heating bias: 5.3 K/day per column peak (essentially flat vs M5-S3.x)
- 24h T2 drift: still **1-3 K** for adversarial profiles — same corridor as M5-S3.x
- **5-10 K drift permanently behind us**; now bouncing in narrower 1-3 K corridor
- Still above 0.5 K threshold where M6-S3 surface-layer/Noah-MP signal would dominate

**M6 coupled-forecast validation REMAINS BLOCKED on M5-S3.z close.**

## M5-S3.z scope (per reviewer §5 — binding)

Updated `.agent/sprints/2026-05-21-m5-s3z-rrtmg-intermediate-oracles/sprint-contract.md` with reviewer's detailed scope. Key changes vs original stub:

1. **WRF harness per-band TOA + surface flux emission** (closes AC6).
2. **WRF harness intermediate-oracle dumps** (per-band, per-layer, per-g-point):
   - SW: `jp, jt, jt1, fac00..fac11, indself, indfor, selffac, forfac, colmol, taug, taur, sfluxzen` at entry to `spcvmc_sw`
   - LW: `jp, jt, planklay, planklev, plankbnd, taug, fracs, secdiff` at entry to `rtrnmc`
3. **Band-by-band JAX validation** with TIGHT tolerances at intermediate level: `abs ≤ 1e-8 + rel ≤ 1e-4` per g-point per layer (NOT broadband flux output).
4. **HARD RULE in worker prompt**: no further hand-transcribed JAX branch code without corresponding WRF intermediate-oracle dump for that band. Failed SW bands REVERT to nearest-pressure approximation per-band, with documented per-band debt.
5. **LW source machinery completion**: `dplankup/dplankdn` non-isothermal + `tfn_tbl` lookup.
6. **SW launch fusion**: 36 SW → ≤6 via `lax.scan` or table-driven compact branches.
7. **Strict Tier-1 pass at flux-output level**: unchanged contract bar.
8. **ADR-009 → PARITY** with per-band intermediate-oracle evidence.

Estimated wall: **24-48h**.

## M6 dispatch impact

- **M6 coupled forecast**: BLOCKED on M5-S3.z close (operational T2 drift gate)
- **M6-S1 interface freeze**: in Opus reviewer (independent of RRTMG)
- **M6-S2a Gen2 backfill accessor**: can dispatch in parallel with M5-S3.z after M6-S1 ACCEPTs (file-disjoint)
- The 4 M5-S3.y permanent artifacts are inherited unconditionally by M5-S3.z

## Process notes

- **WATCHDOG fix worked first try**: fd5c214 launcher pattern eliminates the 4-incident stuck-at-/exit pattern. Both AGENT REPORTs (s3y + m6s1 pending) will fire reliably.
- M5-S3 cycle history: A1 (synthetic tables) → A2 (clip-pinning + vacuous tolerances) → A3 (groundwork) → S3.x (Eddington + δ-scaling) → S3.y (partial native taumol+Planck) → S3.z (intermediate-oracle methodology). Net direction: each cycle adds permanent infrastructure even when worker self-rejects.
- Anti-pattern recurrence: NONE this cycle. Clip-pinning, vacuous tolerances, launch fudge, fabricated tables, ADR mis-set to PARITY — all avoided.

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 12:00
