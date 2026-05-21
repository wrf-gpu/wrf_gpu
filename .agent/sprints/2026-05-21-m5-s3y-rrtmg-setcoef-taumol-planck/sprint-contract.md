# Sprint Contract — M5-S3.y RRTMG setcoef + taumol + Planck-source (M6 Prologue Phase 2)

**Sprint ID**: `2026-05-21-m5-s3y-rrtmg-setcoef-taumol-planck`
**Created**: 2026-05-21 10:43 by manager (Claude Opus 4.7 1M-context)
**Status**: STUB — to dispatch after M5-S3.x close; can run in parallel with any other M6-prologue items
**Trigger**: M5-S3.x closed ACCEPT-AS-GROUNDWORK-PHASE-2; Opus reviewer §5 named the exact remaining scope below

## Objective

Close the remaining RRTMG transfer-solver gaps so M6 coupled-forecast validation can dispatch:

1. **SW `setcoef_sw` port** — pressure/temperature interpolation factors `fac00/fac01/fac10/fac11`, reference-pressure indices `indfor/indself`, jp/jt lookups (WRF `module_ra_rrtmg_sw.F:2843-3099`). Expose as JAX-table-resident state.
2. **SW `taumol_sw` per-band port** — all 14 bands (16-29) computing `taug(ig) = colamt × k(jp,jt,ig) + selfref + forref` (WRF `module_ra_rrtmg_sw.F:3190-4653`).
3. **LW `setcoef` port** — analog to SW (WRF `module_ra_rrtmg_lw.F:3556-3921`).
4. **LW `taumol` per-band** — all 16 bands computing per-g-point `taug` AND `fracs(lev,igc)` Planck fractions (WRF `module_ra_rrtmg_lw.F:4824-7942`).
5. **LW Planck-source machinery in `rtrnmc`** — `planklay(lev,iband)`, `planklev(lev,iband)`, `plankbnd(iband)` with `dplankup/dplankdn` per-layer non-isothermal correction (WRF `module_ra_rrtmg_lw.F:3270-3340`).
6. **Eddington-vs-PIFM oracle resolution** — local WRF compiles `kmodts=2` (PIFM, not Eddington). Options:
   - (a) Patch local WRF to set `kmodts=1` and rebuild harness (Eddington oracle)
   - (b) Retarget JAX from Eddington to PIFM (matches local oracle but diverges from sprint-S3.x contract)
   Manager decision required BEFORE worker dispatch.
7. **Per-band fixture/harness extension** — WRF harness must emit per-band TOA + surface fluxes; Python validation must compare per-band, not just broadband.
8. **Launch fusion** — 40 → ≤10 per call. Likely requires fusing SW band-loop into single `lax.scan` and similarly for LW.

## Acceptance (M6 coupled-forecast gate)

- Strict Tier-1 pass at contract bar: `flux abs ≤1 W/m² + rel ≤0.05`, `heating abs ≤1e-4 K/s + rel ≤0.05`.
- Raw launches `≤10` per kernel call.
- HLO `≤500 KB` per kernel.
- ADR-009 finalized to "PARITY" status (not "GROUNDWORK").
- Per-band residual table in worker report.
- Verifiability triple: `nm` symbols preserved, no clip-pinning, non-vacuous tolerances.

## Inputs (carried forward from M5-S3.x)

- `data/fixtures/rrtmg-tables-v1.npz` (1.75 MB; preserve SHA — extend only if M5-S3.y needs new table dimensions for setcoef/taumol per-band coefficients)
- `scripts/wrf_rrtmg_harness.f90` (real WRF driver binding; extend for per-band flux dumps)
- `scripts/wrf_rrtmg_harness_build.sh` (preserve link chain)
- `scripts/extract_rrtmg_tables.py` (extend for new table dimensions if needed)
- `scripts/m5_run_rrtmg.py`, `m5_gate_rrtmg.py` (preserve gate logic)
- `src/gpuwrf/physics/rrtmg_*.py` (extend; preserve Eddington / vrtqdr / diffusivity)

## Files Worker May Modify

- `src/gpuwrf/physics/rrtmg_sw.py` (add real setcoef + taumol; preserve Eddington/vrtqdr)
- `src/gpuwrf/physics/rrtmg_lw.py` (add real setcoef + taumol + Planck source; preserve diffusivity recurrence structure)
- `src/gpuwrf/physics/rrtmg_constants.py` (add Planck-fraction constants if needed)
- `src/gpuwrf/physics/rrtmg_tables.py` (extend for setcoef/taumol per-band tables)
- `scripts/wrf_rrtmg_harness.f90` (extend per-band emission)
- `scripts/extract_rrtmg_tables.py` (if new table dimensions needed)
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md` (finalize to "PARITY")
- `tests/test_m5_rrtmg_*` (extend per-band)
- Worker report

## Files Worker Must NOT Modify

- Anything under `src/gpuwrf/physics/thompson_*` or `src/gpuwrf/physics/mynn_*` (P1/P2 owns)
- Anything under `src/gpuwrf/dynamics/**`, `src/gpuwrf/contracts/**`, `src/gpuwrf/timestep/**`, `src/gpuwrf/coupling/**` (M6-S1 owns)
- Any other ADR or governance file

## Dispatch

- Primary worker: codex gpt-5.5 xhigh (per frontrunner role)
- Reviewer (mandatory per sprint-lifecycle hard rule): Claude Opus 4.7 xhigh
- Wall-time: **16-32 hours** (largest M5 prologue item; comparable to M5-S1 Thompson)
- Worktree: `/tmp/wrf_gpu2_s3y` (isolated)
- Branch: `worker/codex/m5-s3y-rrtmg-setcoef-taumol-planck`

## Pre-dispatch decision needed

**Manager must decide Eddington-vs-PIFM** (sprint contract item 6) before worker dispatch. Options:

(a) Patch local WRF + rebuild harness → preserves the M5-S3.x Eddington implementation, adds harness rebuild step
(b) Switch JAX to PIFM → matches local oracle, requires rewriting M5-S3.x `rrtmg_sw.py:211-215` γ coefficients per Meador-Weaver Table 1 PIFM row

Recommended: **(a)** — preserves M5-S3.x progress; PIFM has known accuracy issues for non-conservative scattering vs Eddington. Manager TBD.

## Hard rules (encoded from M5-S3 cycle lessons)

- **NO** synthetic / fabricated / clip-pinned coefficients labeled as real (M5-S3-A1/A2 anti-patterns)
- **NO** vacuous tolerances (M5-S3-A2 had `abs=1200 W/m²`)
- **NO** `min(raw, cap)` launch fudge (M5-S3-A1 anti-pattern)
- **NO** elective bypass of real WRF subroutines — `nm`-verified link preserved
- Cite WRF `module_ra_rrtmg_*.F:lineno` for every formula claim
- Verify any coefficient by computation, not literal copy-paste
