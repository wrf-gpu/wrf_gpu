# Sprint Contract — M5-S3.z RRTMG Intermediate-Oracle Extraction + Per-Branch Validation

**Sprint ID**: `2026-05-21-m5-s3z-rrtmg-intermediate-oracles`
**Created**: 2026-05-21 11:42 by manager (Claude Opus 4.7 1M-context)
**Status**: STUB — to dispatch after M6-S1 Opus closes (parallel with M6-S2a, file-disjoint)
**Trigger**: M5-S3.y worker self-flagged "do not accept" with recommendation §3: "Scope M5-S3.z to first add WRF per-band harness output and intermediate `taug/taur/fracs/plank*` dumps, then validate each JAX branch against those intermediate oracles before touching transfer fusion."

## Objective

Make M5-S3.z the **WRF intermediate-oracle extraction sprint**, not another round of brute-force JAX taumol/Planck transcription. M5-S3.y proved that naive all-band native branch expansion explodes HLO (1.31 MB > 500 KB) and launches (52 > 10) before correctness is solved.

**Strategy**: extract WRF's *intermediate* state — per-band per-g-point optical depths (`taug, taur`), Planck fractions (`fracs`), Planck-source values (`planklay, planklev, plankbnd, dplankup, dplankdn`) — as proof objects. Then validate JAX branches against those intermediate oracles BEFORE attempting end-to-end transfer fusion.

This is a **per-branch validation infrastructure sprint** that unblocks (later) M5-S3.zz where each JAX branch is fixed in isolation with HLO budget control, instead of attempting all 14 SW bands and 16 LW bands in one shipment.

## Acceptance (per M5-S3.y Opus reviewer §5 — binding)

- **AC1 — WRF harness per-band TOA + surface flux emission**. `scripts/wrf_rrtmg_harness.f90` extends to dump 14 SW + 16 LW per-band TOA-up, TOA-down, surface-up, surface-down arrays per scenario. Closes M5-S3.y AC6 (methodological blocker).
- **AC2 — WRF harness intermediate-oracle dumps** (per-band, per-layer, per-g-point):
  - SW (at entry to `spcvmc_sw`): `jp(lev), jt(lev), jt1(lev), fac00..fac11(lev), indself(lev), indfor(lev), selffac(lev), forfac(lev), colmol(lev,*), taug(lev,igc,iband), taur(lev,igc), sfluxzen(igc,iband)`
  - LW (at entry to `rtrnmc`): `jp(lev), jt(lev), planklay(lev,iband), planklev(lev,iband), plankbnd(iband), taug(lev,igc,iband), fracs(lev,igc), secdiff(iband)`
  - Persist as `data/fixtures/rrtmg-intermediate-oracle-v1.npz` with SHA pinned in manifest.
- **AC3 — Band-by-band JAX validation against intermediate oracle** (TIGHT tolerances — these are intermediate quantities, not flux output):
  - Each of the 14 SW bands: `taug` and `taur` JAX outputs match WRF intermediate within `abs ≤ 1e-8 + rel ≤ 1e-4` per g-point per layer.
  - Each of the 16 LW bands: `taug` and `fracs` JAX outputs match WRF intermediate within `abs ≤ 1e-8 + rel ≤ 1e-4`.
  - JAX `_sw_setcoef` outputs match WRF `setcoef_sw` intermediate within float64 round-off `abs ≤ 1e-12 + rel ≤ 1e-10`.
  - LW Planck-state (`planklay/planklev/plankbnd`) matches WRF `setcoef`/`taumol` Planck path within `abs ≤ 1e-10 + rel ≤ 1e-8`.
- **AC4 — LW source machinery completion**. `dplankup/dplankdn` non-isothermal per-layer correction + `tfn_tbl` source-correction lookup in `rtrnmc` (`module_ra_rrtmg_lw.F:3270-3340`).
- **AC5 — SW launch fusion**. 36 SW → ≤6 SW launches via `lax.scan` over bands OR table-driven compact branches. HLO SW ≤500 KB. **Total combined raw launches ≤10**.
- **AC6 — Strict Tier-1 pass at flux-output level**. `abs ≤ 1 W/m² + rel ≤ 0.05` for fluxes, `abs ≤ 1e-4 K/s + rel ≤ 0.05` for heating (unchanged contract bar).
- **AC7 — ADR-009 finalized to `PARITY`**, citing per-band intermediate-oracle validation evidence.
- **AC8 — Per-band debt list**. `artifacts/m5/rrtmg_per_band_status.json` documents which JAX branches pass intermediate-oracle gate, which are reverted to nearest-pressure approximation, and which carry which debt to future sprints.

## HARD RULES (per reviewer §5 constraints)

1. **NO further hand-transcribed JAX branch code until corresponding WRF intermediate-oracle dump exists for that band.** Worker may NOT validate against broadband-flux output alone. This is the methodological discipline that prevents the M5-S3.y SW regression.
2. **SW `_sw_taumol_*` branches that fail intermediate-oracle gate must be reverted to the M5-S3.x nearest-pressure approximation for that band ONLY**, with a documented per-band debt list. Better to ship a correct broadband approximation than 14 incorrect band branches.
3. Carry forward unconditionally: all M5-S3.y AC0 (Eddington oracle), native table data (`sw_absa/absb/selfref/forref/sfluxref`, `lw_totplnk/totplk16`), faithful JAX `_sw_setcoef` formulas, LW `totplnk` Planck-source replacement.

## Estimated wall

24-48 hours (intermediate-oracle harness extension non-trivial; per-band validation many small gates; LW source completion + SW fusion larger code lifts).

## Inputs

- `data/scratch/wrf_rrtmg_harness` (with Eddington patch from M5-S3.y AC0; SHA `25c88aa4...`) — preserve
- `data/fixtures/rrtmg-tables-v1.npz` (from M5-S3.y, includes native `absa/absb/selfref/forref/sfluxref/totplnk/totplk16`)
- `src/gpuwrf/physics/rrtmg_*.py` from M5-S3.y (preserve or partially revert per AC5)

## Files Worker May Modify

- `scripts/wrf_rrtmg_harness.f90` (extend; preserve real subroutine calls)
- `scripts/wrf_rrtmg_harness_build.sh` (if needed)
- `scripts/extract_rrtmg_tables.py` (if needed for new dimensions)
- `scripts/m5_generate_rrtmg_fixture.py` (parse new records)
- `src/gpuwrf/physics/rrtmg_*.py` (per AC5 revert + intermediate-validation hooks)
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` (NEW)
- `tests/test_m5_rrtmg_intermediate.py` (NEW)
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md` (amend)
- `data/fixtures/rrtmg-tables-v1.npz` (only if AC1/AC2 need extension)
- Worker report

## Files Worker Must NOT Modify

- `src/gpuwrf/contracts/**`, `src/gpuwrf/dynamics/**`, `src/gpuwrf/coupling/**` (M6-S1 owns)
- `src/gpuwrf/physics/thompson_*`, `mynn_*` (other physics, disjoint)
- Other ADR or governance file
- WRF source patches (the Eddington `kmodts=1` patch from M5-S3.y already applied — preserve)

## Dispatch

- Primary worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (mandatory per sprint-lifecycle)
- Wall-time: **16-24 hours** (less than M5-S3.y because no transfer-solver re-architecting; just oracle extraction + branch validation)
- Worktree: `/tmp/wrf_gpu2_s3z` (NEW, isolated)
- Branch: `worker/codex/m5-s3z-rrtmg-intermediate-oracles`

## Hard rules

- NO new transfer-solver code shipping — this sprint extracts oracles + validates branches; M5-S3.zz (or later) does the actual fixes.
- NO synthetic / fabricated / clip-pinned coefficients.
- NO `min(raw, cap)` launch fudge.
- Per-band per-g-point intermediate residual tables are the deliverable, not a single broadband pass/fail.
- Cite `module_ra_rrtmg_*.F:lineno` for every intermediate-quantity extraction.

## Pre-dispatch decision

After Opus reviewer of M5-S3.y returns verdict:
- If **REJECT-bounded-rework as M5-S3.z**: dispatch this sprint as-is.
- If **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3**: adopt M5-S3.y SW setcoef + LW setcoef as carry-forward; dispatch this sprint with AC5 = "preserve" (not "revert").
- If **REJECT-revert**: dispatch this sprint with AC5 = "revert M5-S3.y SW native branch expansion entirely; preserve only AC0 Eddington oracle".
