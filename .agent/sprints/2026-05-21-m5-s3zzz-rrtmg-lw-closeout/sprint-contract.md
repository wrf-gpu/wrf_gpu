# Sprint Contract — M5-S3.zzz RRTMG LW Closeout

**Sprint ID**: `2026-05-21-m5-s3zzz-rrtmg-lw-closeout`
**Branch**: `worker/codex/m5-s3zzz-rrtmg-lw-closeout`
**Date**: 2026-05-21
**Worker**: Codex GPT-5.5 xhigh

## Objective

Close the longwave side of the M5 RRTMG implementation by replacing the nearest-pressure LW `taumol`/`fracs` approximation with oracle-validated per-band branches where they pass, fusing the LW band loop with `jax.lax.scan`, and producing honest Tier-1/intermediate proof artifacts.

## Scope

Owned production core file:

- `src/gpuwrf/physics/rrtmg_lw.py`

Allowed support/proof files:

- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py`
- `tests/test_m5_rrtmg_*.py`
- `artifacts/m5/rrtmg_intermediate_validation.json`
- `artifacts/m5/rrtmg_per_band_status.json`
- `artifacts/m5/tier1_rrtmg_lw_parity.json`
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md`
- `.agent/sprints/2026-05-21-m5-s3zzz-rrtmg-lw-closeout/worker-report.md`

Explicitly forbidden:

- Do not touch `src/gpuwrf/physics/rrtmg_sw.py`.
- Do not ship a LW band branch into production unless its `taug` and `fracs` intermediate oracle gates pass.
- Do not report capped launch counts; raw launch markers must equal reported launch counts.

## Acceptance Criteria

### AC1 + AC2 — Sixteen LW `taumol` + `fracs` branches

For each LW band 1-16, transcribe the WRF `taumol` branch from `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_lw.F:4824-7942`.

For each band:

- `taug(lay, igc, iband)` is gas optical depth per g-point per layer.
- `fracs(lay, igc)` is the Planck fraction per g-point per layer.
- Major-species interpolation must follow WRF `jp/jt/fac00..fac11` table interpolation.
- Band-specific minor continua must follow WRF `selfref`, `forref`, and minor-reference table logic.
- Ratio-interpolation and special paths, including band-1 H2O+CO2 ratio interpolation and band-3 low-altitude H2O scaling, must cite WRF source lines.

Methodology:

1. Implement a band branch in JAX.
2. Validate that branch against the intermediate-oracle NPZ for `taug` and `fracs`.
3. If both gates pass at `abs <= 1e-8 + rel <= 1e-4`, deploy the branch to production.
4. If either gate fails, keep that band on the nearest-pressure fallback and document the debt.

### AC3 — Per-band LW intermediate-oracle validation

Extend `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` with:

- `validate_lw_taug_per_band(jax_taug, wrf_taug, band)`
- `validate_lw_fracs_per_band(jax_fracs, wrf_fracs, band)`

Both use `abs <= 1e-8 + rel <= 1e-4`. Update `artifacts/m5/rrtmg_intermediate_validation.json` with per-LW-band pass status.

### AC4 — LW launch fusion

Implement the 16-band LW loop using `jax.lax.scan` over band index, mirroring the M5-S3.zz SW `_sw_taumol_fused` pattern. Target LW raw launches `<=4`; combined SW+LW must not regress against the M5-S3.zz baseline. No `min(raw, cap)` launch reporting.

### AC5 — Strict Tier-1 LW pass

After AC1-AC4, `python scripts/m5_run_rrtmg.py` plus `python scripts/m5_gate_rrtmg.py` should show LW Tier-1 PASS:

- LW broadband and per-band flux: `abs <= 1 W/m^2 + rel <= 0.05`
- LW heating: `abs <= 1e-4 K/s + rel <= 0.05`

### AC6 — SW no regression

`git diff main...HEAD -- src/gpuwrf/physics/rrtmg_sw.py` must be empty.

### AC7 — ADR-009 amendment

If LW Tier-1 passes, amend ADR-009 to `SW-PARTIAL (broadband debt -> S3.zzzz), LW-PARITY`.

If LW Tier-1 fails, keep ADR-009 as NOT-PARITY, document the LW root cause, and recommend a follow-up LW oracle sprint. Do not mark full PARITY while the SW broadband closeout remains in flight.

### AC8 — Per-band debt list

Extend `artifacts/m5/rrtmg_per_band_status.json` with LW band entries containing:

- `taug` PASS/FAIL
- `fracs` PASS/FAIL
- `implementation_status`: `FULL_BRANCH_ACCEPTED` or `FALLBACK_NEAREST_PRESSURE`

## Required Proof

1. `nm` symbols on the WRF RRTMG harness binary are preserved.
2. Intermediate validation has no clip-pinning on per-LW-band gates.
3. Raw launch marker counts equal reported launch counts.

## Validation Commands

```bash
python scripts/m5_run_rrtmg.py
python scripts/m5_gate_rrtmg.py
cat artifacts/m5/rrtmg_intermediate_validation.json | jq '.lw'
cat artifacts/m5/rrtmg_per_band_status.json | jq '.lw_bands'
cat artifacts/m5/tier1_rrtmg_lw_parity.json | jq '.pass, .per_field_max_abs_err'
pytest -q tests/test_m5_rrtmg_*.py
git diff main...HEAD -- src/gpuwrf/physics/rrtmg_sw.py
```

## Reporting

The worker report must include objective, files changed, commands run, proof objects produced, unresolved risks, next decision needed, per-band table, WRF citations, and before/after evidence.
