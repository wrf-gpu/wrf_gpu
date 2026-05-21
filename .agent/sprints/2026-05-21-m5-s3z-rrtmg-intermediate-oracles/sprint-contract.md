# Sprint Contract — M5-S3.z RRTMG Intermediate-Oracle Extraction + Per-Branch Validation

**Sprint ID**: `2026-05-21-m5-s3z-rrtmg-intermediate-oracles`
**Created**: 2026-05-21 11:42 by manager (Claude Opus 4.7 1M-context)
**Status**: STUB — to dispatch after M6-S1 Opus closes (parallel with M6-S2a, file-disjoint)
**Trigger**: M5-S3.y worker self-flagged "do not accept" with recommendation §3: "Scope M5-S3.z to first add WRF per-band harness output and intermediate `taug/taur/fracs/plank*` dumps, then validate each JAX branch against those intermediate oracles before touching transfer fusion."

## Objective

Make M5-S3.z the **WRF intermediate-oracle extraction sprint**, not another round of brute-force JAX taumol/Planck transcription. M5-S3.y proved that naive all-band native branch expansion explodes HLO (1.31 MB > 500 KB) and launches (52 > 10) before correctness is solved.

**Strategy**: extract WRF's *intermediate* state — per-band per-g-point optical depths (`taug, taur`), Planck fractions (`fracs`), Planck-source values (`planklay, planklev, plankbnd, dplankup, dplankdn`) — as proof objects. Then validate JAX branches against those intermediate oracles BEFORE attempting end-to-end transfer fusion.

This is a **per-branch validation infrastructure sprint** that unblocks (later) M5-S3.zz where each JAX branch is fixed in isolation with HLO budget control, instead of attempting all 14 SW bands and 16 LW bands in one shipment.

## Acceptance

- **AC1 — Per-band WRF harness output**. `scripts/wrf_rrtmg_harness.f90` extended to dump:
  - SW: per-band `taug(ig)` (gas optical depth per g-point), `taur(ig)` (Rayleigh per g-point), `cldfmc(lay,ig)` (cloud fraction per Monte Carlo column), `sfluxref(ig)` (solar source)
  - LW: per-band `taug(lay,ig)` (gas optical depth), `fracs(lay,igc)` (Planck fraction), `planklay(lay,iband)`, `planklev(lay+1,iband)`, `plankbnd(iband)`
  - All written as additional unformatted records appended to the existing harness output, with structured indexing.
- **AC2 — Intermediate oracle fixture extension**. `scripts/m5_generate_rrtmg_fixture.py` parses the new harness output records into NPZ leaves. Manifest schema updated. Sample size budget: ≤10 MB total (per-band per-g-point arrays for the 3 scenarios).
- **AC3 — JAX branch validation framework**. New `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` with one function per intermediate quantity that compares JAX intermediate computation to WRF harness output, per band, per g-point. Pass criterion per quantity: abs ≤1e-6 + rel ≤0.01 (tighter than end-to-end flux tolerances; these are intermediate quantities).
- **AC4 — Per-branch validation report**. `artifacts/m5/rrtmg_intermediate_validation.json` with per-quantity per-band pass/fail. Lists which JAX branches need fixing.
- **AC5 — M5-S3.y SW partial REVERT (if Opus reviewer requires)**. Roll back the SW native branch expansion that blew HLO if the Opus reviewer chose REJECT-revert path. Preserve AC0 Eddington oracle and basic setcoef. Keep only what passes intermediate-oracle validation.
- **AC6 — ADR-009 amendment**. Document the intermediate-oracle pattern + per-branch validation strategy. Reset status to "PHASE-3-INTERMEDIATE-ORACLES" (not "PARITY").

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
