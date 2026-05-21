# Sprint Contract — M5-S3.zzzz RRTMG cldprmc_sw + spcvmc_sw Intermediate-Oracle + Cloud-Optics/Broadband Transfer Fix

**Sprint ID**: `2026-05-21-m5-s3zzzz-rrtmg-cldprmc-spcvmc-oracle`
**Created**: 2026-05-21 13:55
**Status**: STUB — dispatch after M5-S3.zz Opus accepts
**Trigger**: M5-S3.zz worker discovered new root cause beyond sfluxzen+setcoef+taumol — downstream broadband transfer + cloud optics. Per the intermediate-oracle methodology established by M5-S3.z reviewer (no production edits without WRF intermediate dumps), build the oracle infrastructure for the next layer.

## Objective

Extract WRF intermediate state at the `cldprmc_sw` → `spcvmc_sw` boundary, validate JAX cloud-optics + broadband transfer against those oracles, fix per-quantity branches that fail.

## Acceptance

- **AC1 WRF harness `cldprmc_sw` intermediate dumps**: per-band per-layer per-g-point arrays at exit of `cldprmc_sw` and entry of `spcvmc_sw`:
  - `pcldfmc(lay, ig)` — MCICA cloud mask per Monte Carlo subcolumn
  - `ptaucmc(lay, ig)` — total optical depth (cloud + Rayleigh + gas)
  - `pasycmc(lay, ig)` — asymmetry parameter
  - `pomgcmc(lay, ig)` — single-scattering albedo
  - `ptaormc(lay, ig)` — total optical depth pre-delta-scaling
- **AC2 WRF harness `spcvmc_sw` intermediate dumps**: per-band per-layer arrays at internal stages:
  - Clear/cloud two-stream: `zref, ztra, zrefd, ztrad`
  - Direct-beam transmittance per layer
  - Per-g-point `zfd, zfu` BEFORE broadband accumulation
- **AC3 Intermediate-oracle NPZ**: extend `rrtmg-intermediate-oracle-v1.npz` or create `rrtmg-cldprmc-spcvmc-oracle-v1.npz`; SHA-pinned in manifest; ≤50 MB.
- **AC4 JAX validation against new oracle**: extend `validation/rrtmg_intermediate_oracles.py` with per-quantity per-band per-layer validation; tolerance `abs ≤1e-4 + rel ≤1e-3` (single-precision floor per WRF compile).
- **AC5 Fix branches that fail**: per-band debt list `artifacts/m5/rrtmg_per_band_cloud_status.json` lists which JAX cloud-optics or transfer branches PASS vs need fix. Apply fixes incrementally; revert any branch that re-FAILS Tier-1 SW.
- **AC6 Strict Tier-1 SW PASS at flux-output level**: `abs ≤1 W/m² + rel ≤0.05` for broadband + per-band SW fluxes.
- **AC7 LW NO regression**: do NOT touch `rrtmg_lw.py` production path.
- **AC8 ADR-009 status update**: if Tier-1 SW passes, amend to "SW-PARITY, LW-NOT-PARITY". If not, hold at "NOT-PARITY" with current scope debt + recommend next sprint scope.

## Inputs (carry forward UNCONDITIONALLY)

- All M5-S3.zz outputs: sfluxzen fix, setcoef precision floor, 14 SW branches lax.scan
- `data/fixtures/rrtmg-intermediate-oracle-v1.npz` — gas optical-depth oracles (M5-S3.z)
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` — validation framework
- `scripts/wrf_rrtmg_harness.f90` — already has gas-optical-depth dump infrastructure

## Files Worker May Modify

- `scripts/wrf_rrtmg_harness.f90` (extend with cldprmc_sw + spcvmc_sw internal dumps)
- `scripts/wrf_rrtmg_harness_build.sh` (if needed)
- `scripts/m5_generate_rrtmg_fixture.py` (parse new records)
- `src/gpuwrf/physics/rrtmg_sw.py` (cloud-optics + transfer fixes per-branch per-validation-pass)
- `src/gpuwrf/physics/rrtmg_tables.py` (if new tables needed)
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` (extend)
- `data/fixtures/rrtmg-intermediate-oracle-v1.npz` (extend) OR new `rrtmg-cldprmc-spcvmc-oracle-v1.npz`
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md` (amend per outcome)
- `tests/test_m5_rrtmg_*` (extend)
- Worker report

## Files Worker Must NOT Modify

- `rrtmg_lw.py` production path
- Other physics modules (frozen)
- io/, dynamics/, contracts/, coupling/

## Dispatch

- Codex gpt-5.5 xhigh worker
- Opus 4.7 xhigh reviewer (mandatory)
- Wall-time: **16-32h**
- Worktree: `/tmp/wrf_gpu2_s3zzzz` (NEW)
- Branch: `worker/codex/m5-s3zzzz-rrtmg-cldprmc-spcvmc-oracle`

## HARD RULES

1. **Oracle FIRST**: cldprmc_sw + spcvmc_sw intermediate dumps MUST land before any production cloud-optics edits.
2. **Per-branch validation**: every JAX branch fix must pass intermediate-oracle gate BEFORE production deploy.
3. **NO `min(raw, cap)` fudge.**
4. **NO clip-pinning.**
5. Cite `module_ra_rrtmg_sw.F:lineno` for every formula.
6. WATCHDOG + multi-Enter pattern in launcher.

## REVIEWER-BINDING ADDITIONS (per M5-S3.zz Opus reviewer §4 A1-A5)

- **A1 — R-8 hypothesis confrontation**: explicitly verify `cloud_safe = max(cloud_box, 0.01)` floor against `cldprmc_sw` `ptaucmc/pasycmc/pomgcmc` dumps. If floor is the bias source, replace with `where(cloud_box > 0, ..., 0)` form that does NOT floor the denominator.
- **A2 — R-9 hypothesis confrontation**: explicitly verify "double-Eddington-then-blend" against `spcvmc_sw` per-g-point `zref/ztra/zrefd/ztrad` dumps. Restructure JAX to compute reftra ONCE per g-point with MCICA-selected optical properties (matching WRF accumulator semantics), not blend-then-Eddington.
- **A3 — Per-g-point flux pre-accumulation dump**: dump `zfd/zfu` BEFORE broadband `sum(axis=(-1,-2))` reduction (`rrtmg_sw.py:965-966`); validate JAX `down_band/up_band` matches WRF per-g-point fluxes before sum.
- **A4 — Re-`nm` harness + persist SHA**: rebuild harness with cldprmc+spcvmc dump extensions; run `nm` on rebuilt binary; persist symbol set SHA to `fixtures/manifests/rrtmg-intermediate-oracle-v1.yaml`. Closes M5-S3.zz verifiability-triple §1.1 debt.
- **A5 — ADR-009 status**: amend to `SW-PARITY, LW-NOT-PARITY` ONLY IF strict Tier-1 SW PASS proven. Otherwise hold at `NOT-PARITY` + document new SW broadband root cause + recommend M5-S3.zzzzz scope.

## Attribution table from M5-S3.zz reviewer §3.3 (what your oracle will pin down)

| Source | Est. contribution | Oracle catches |
|---|---|---|
| cldprmc_sw cloud-optics (R-8 + cloud_safe + delta-scaling closure) | 20-50 W/m² | YES — ptaucmc/pomgcmc/pasycmc/ptaormc per band/layer/g-point |
| spcvmc_sw per-layer reftra blending (R-9) | 10-30 W/m² | YES — clear/cloud zref/ztra/zrefd/ztrad |
| Direct-beam transmittance | 5-15 W/m² | YES |
| Per-g-point flux accumulation pre-broadband | 0-10 W/m² | YES — zfd/zfu BEFORE sum |
| Transfer-solver | ≈0 | already PASS (M5-S3.y carry-forward) |
| Gas optical / sfluxzen / setcoef | ≈0 | already PASS (M5-S3.zz) |

## End-goal context

If this sprint closes SW PARITY, M5 RRTMG is half-done (LW still at S3.zzz). Combined S3.zzzz + S3.zzz unblocks M6 operational T2 validation. The 6-sprint M5-S3 cycle (S3 → S3.x → S3.y → S3.z → S3.zz → S3.zzzz) reflects honest discovery: each cycle adds permanent infrastructure (native tables → Eddington oracle → intermediate-oracle NPZ → per-band framework → sfluxzen+setcoef → cldprmc+spcvmc). Trajectory is positive.
