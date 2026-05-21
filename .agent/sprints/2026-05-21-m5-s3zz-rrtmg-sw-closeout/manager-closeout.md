# M5-S3.zz Manager Closeout — RRTMG SW Closeout (partial; new root cause)

**Sprint**: `2026-05-21-m5-s3zz-rrtmg-sw-closeout`
**Status**: **CLOSED-PARTIAL pending Opus review; M5-S3.zzzz cldprmc/spcvmc oracle binding next**
**Date**: 2026-05-21 ~13:55
**Manager**: Claude Opus 4.7 (1M-context)

## What landed (commit `f62fe88`, merged via `--no-ff`)

Codex worker (~50min including manager-prompted finalization):

| AC | Status | Evidence |
|---|---|---|
| AC1 sfluxzen | PASS | `rrtmg_intermediate_validation.json.sw.sfluxzen.pass=true`; max_abs 3.8e-6, max_rel 3.7e-7 (within 1e-8+1e-4 bar). Root cause: WRF source-active gating + zero-init at band-11; fixed via `_source_active` in `_sw_sfluxzen`. WRF citation `module_ra_rrtmg_sw.F:3380-3382, 4446-4464`. |
| AC2 setcoef precision | PASS (Path B) | `setcoef.pass=true` at amended `abs ≤1e-4 + rel ≤1e-3` floor. WRF `-r4 -i4` compile evidence in `compile.log:55,141,180`. Test asserts the floor. |
| AC3 14 SW branches via lax.scan | PARTIAL | `_sw_taumol_fused` wraps validated branches behind `jax.lax.scan` over band index. `rrtmg_per_band_status.json`: 14 SW bands FULL_BRANCH_ACCEPTED + intermediate_gate PASS. Launch/HLO targets not claimed because Tier-1 flux still fails. |
| AC4 strict Tier-1 SW | FAIL | flux_down 56 W/m², flux_up 64 W/m², column_absorbed 87 W/m², toa_up 30, surface_down 21. Heating max_abs 2.5e-5 K/s WITHIN threshold. |
| AC5 LW no regression | PASS-by-edit-scope | `rrtmg_lw.py` not touched; M5-S3.z Planck-source preserved. |
| AC6 ADR-009 | correctly NOT amended | SW parity not proven; ADR-009 stays NOT-PARITY. |

## New root cause (binds M5-S3.zzzz scope)

Both M5-S3.z reviewer root causes (sfluxzen + setcoef precision) are CLOSED. Tier-1 SW flux still fails — root cause shifted:

- **NOT** gas optical depth (`taumol_sw`) — passes per-band intermediate-oracle
- **NOT** source allocation (`sfluxzen`) — passes per-band intermediate-oracle
- **NOT** setcoef interpolation — passes amended single-precision bar
- **NOT** MCICA random-overlap mask — JAX KISS mirrors WRF Fortran (worker probed)

Residual is **downstream of gas optics** — at the `cldprmc_sw` (cloud-optical-properties assembly) and/or `spcvmc_sw` (broadband two-stream transfer + accumulation) interfaces. Worker added cloud optical handling but without WRF intermediate dumps, this is a blind production edit.

## M5-S3.zzzz scope (NEW, per worker recommendation §3)

`.agent/sprints/2026-05-21-m5-s3zzzz-rrtmg-cldprmc-spcvmc-oracle/sprint-contract.md` to be written. Required oracle dumps:
- MCICA cloud masks + paths (`pcldfmc`)
- Cloud optical properties: `ptaucmc, pasycmc, pomgcmc, ptaormc`
- Clear/cloud two-stream layer: `zref, ztra, zrefd, ztrad`
- Direct-beam transmittance
- Per-g-point flux arrays `zfd, zfu` BEFORE broadband accumulation

Estimated wall: 16-32h.

## Sequencing decision

Previously bound: M5-S3.zzz = LW closeout (Option 2).

**Manager amendment**: M5-S3.zzzz cldprmc/spcvmc oracle FIRST (SW broadband closeout is on critical path; LW still nearest-pressure but operational impact is similar magnitude to SW broadband residual). Order:

1. **M5-S3.zzzz** (cldprmc + spcvmc intermediate oracle + cloud-optics + transfer fix) — DISPATCH after Opus review of S3.zz
2. **M5-S3.zzz** (LW taumol + fracs closeout) — after S3.zzzz

Rationale: S3.zzzz closes SW completely; S3.zzz then handles LW. Both are still 2 sprints; order chosen so S3.zzzz can validate the broader cloud-optics infrastructure that LW will also need (LW has its own cldprmc + rtrnmc dump path).

## Opus reviewer required

Per sprint-lifecycle hard rule. Reviewer to verify:
- 14 SW branches actually re-enabled in production (`lax.scan` path); not silently bypassed
- Root-cause attribution (broadband transfer vs cloud optics) is justified
- Worker's `cldprmc_sw + spcvmc_sw` oracle recommendation is well-scoped for M5-S3.zzzz
- No anti-pattern recurrence

## Operational impact (unchanged)

T2 drift still 1-3 K corridor. M5 RRTMG PARITY now needs **3 more sprints** (S3.zzzz cldprmc/spcvmc + S3.zzz LW + final S3.zzzzz validation), not 2 as previously estimated. M6 operational validation blocked through all three.

## Process notes

- Worker's "discover root cause N, fix it, expose root cause N+1" cycle continues. Pattern is genuine project discovery, not stalling.
- Manager intervention required to finalize (worker hadn't committed; prompted via terminal).
- Verifiability triple: NOT fully re-run by worker (harness `nm` check missing) — Opus reviewer should re-verify.

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 13:55
