# v0.18 RADIATION (RA) family — ship-gate report

**Branch:** `worker/gpt/v018-ra` · **Frontrunner:** Opus 4.8 (max) · continues GPT worker's uncommitted start.

## Bar (v0.18, no v0.19): every scheme is EXACTLY ONE of
- **(a)** operational + real physics-pristine WRF module-oracle GREEN, or
- **(b)** reference-only **with a real physics-pristine WRF module oracle wired fail-closed**, or
- **(c)** proven computationally-unavailable / real-world-irrelevant, documented + carried (WRF source cited).

No silent gaps, no reference-only-without-a-real-oracle, no fake greens, no tolerance widening.

## Family status (this report covers the radiation long-tail)

Already shipped operational (pre-v0.18, unchanged): **ra_sw 1 (Dudhia), 2 (GSFC), 4 (RRTMG)**; **ra_lw 1 (classic RRTM), 4 (RRTMG)**; **ra_lw 31 (Held-Suarez)** — all class (a), scan-wired, savepoint-parity-proven.

| Scheme | WRF module (driver dispatch) | Class | Oracle it leans on | Verdict |
|---|---|---|---|
| ra_lw 3 / ra_sw 3 — CAM | `module_ra_cam.F:CAMRAD` (8.1k LOC) | (b) reference-only | `proofs/v018/savepoints/ra_tail_wrf/ra3_wrf_real.json` | GREEN |
| ra_lw 5 / ra_sw 5 — new Goddard NUWRF | `module_ra_goddard.F:goddardrad` (12.5k LOC) | (b) reference-only | `…/ra5_wrf_real.json` (+ v0.13 single-col lwrad oracle) | GREEN |
| ra_lw 7 / ra_sw 7 — FLG/UCLA | `module_ra_flg.F:RAD_FLG` (15.3k LOC) | (b) reference-only | `…/ra7_wrf_real.json` | GREEN |
| ra_lw 99 / ra_sw 99 — GFDL-Eta | `module_ra_gfdleta.F:ETARA` (10.2k LOC) | (b) reference-only | `…/ra99_wrf_real.json` | GREEN |
| ra_lw 14 / ra_sw 14 — RRTMG-K (KIAPS) | `module_ra_rrtmg_{lwk,swk}.F` | (c) compiled-out | n/a (cannot run) | GREEN |
| ra_lw 24 / ra_sw 24 — fast RRTMG (GPU/MIC) | `module_ra_rrtmg_{lwf,swf}.F` | (c) compiled-out | n/a (cannot run) | GREEN |

### Why (b), not (a), for 3/5/7/99
Each is an 8k–15k-LOC monolithic radiation module (CAMRAD/goddardrad/RAD_FLG/ETARA), several sharing SW+LW in one driver with hardcoded correlated-k / cloud-optical tables. A faithful traceable JAX column port within this sprint is not achievable without becoming a self-compare / happy-path (explicitly forbidden). The honest landing the bar names for exactly this case ("architecture-sized faithful port acceptable here") is **reference-only with a real oracle wired fail-closed**: namelist-accepted for a reference comparison, fail-closed in the operational GPU scan with a named reason, and backed by a real physics-pristine WRF **exact-driver** oracle a future port can validate against.

### The oracle: exact-driver real-WRF savepoint
For each scheme N, a **physics-pristine, WRFGPU2_ORACLE-instrumented** `wrf.exe` is run on a real-data fixture with `ra_lw_physics = ra_sw_physics = N`. WRF's `radiation_driver` dispatches the exact module for N (verified in `phys/module_radiation_driver.F`: CAMLWSCHEME→CAMRAD, GODDARDLWSCHEME→goddardrad, FLGLWSCHEME→RAD_FLG, GFDLLWSCHEME→ETARA). The radiation tendency + flux history fields it writes (`RTHRATLW/RTHRATSW`, `GLW`, `OLR`, `SWDOWN`, the LW/SW up/down TOA+surface fluxes) are the exact output of that module — recorded as `raN_wrf_real.json`. This is a real WRF executable running the actual upstream-identical radiation module, not a JAX self-compare. `wrf_source_checksums.txt` records module/driver sha256s, and `raw_hash_manifest.txt` records raw `wrfout`/log hashes plus the dump-only instrumentation audit.

The fixture is an 18 h window (2026-04-28_18:00 → 2026-04-29_12:00 UTC) over the Atlantic near the Canaries (lon ≈ −16, solar noon ≈ 13:00 UTC), so the 09:00/12:00 UTC frames carry strong daytime shortwave — the oracle proves **both** the LW and the SW path of each module actually fired (a 6 h night-only window would leave SW degenerate).

### Why (c) for 14/24 — proven computationally-unavailable
`ra_lw/sw_physics = 14` (RRTMG-K / KIAPS) and `24` (fast RRTMG, GPU/MIC) are **compiled out of standard WRF itself** — not a port gap:
- `phys/module_ra_rrtmg_lwk.F` / `swk.F` are bare `#if( BUILD_RRTMK != 1)` dummy stubs (`dummy=1`); `lwf.F` / `swf.F` likewise guarded by `#if( BUILD_RRTMG_FAST != 1)`.
- Pristine `configure.wrf` sets `-DBUILD_RRTMK=0` (line 205) and `-DBUILD_RRTMG_FAST=0` (line 204).
- The `radiation_driver` `CASE (RRTMK_LWSCHEME)` / `CASE (RRTMG_LWSCHEME_FAST)` are themselves `#if( BUILD_* == 1)`-gated, so selecting 14/24 in unmodified WRF falls through to the default branch and aborts: `'The longwave/shortwave option does not exist'` (`module_radiation_driver.F:2309` LW / `:2971` SW).

There is no real oracle to build because the scheme cannot run in this build. They are **not accepted** at the namelist layer and fail closed with a source-cited reason.

**Object-size corroboration** (independently confirmed against the GPT schemes-worker handoff `/tmp/v018_rrtmg1424_handoff.md`): the compiler emitted stub-sized objects for the variants vs the real base RRTMG, proving the `#else` real code was not compiled in this build:

```
phys/module_ra_rrtmg_lw.o   627784 B   (real base RRTMG LW)
phys/module_ra_rrtmg_sw.o   599056 B   (real base RRTMG SW)
phys/module_ra_rrtmg_lwk.o    1264 B   (14 RRTMG-K LW stub)
phys/module_ra_rrtmg_swk.o    1264 B   (14 RRTMG-K SW stub)
phys/module_ra_rrtmg_lwf.o    1272 B   (24 fast RRTMG LW stub)
phys/module_ra_rrtmg_swf.o    1272 B   (24 fast RRTMG SW stub)
```

A prior `worker/gpt/v017-rrtmg` attempt (checkpoint `da6c2ffd`) built RRTMG-14/24 JAX wiring but its own report marked WRF oracle parity RED/blocked for exactly this reason — the local WRF variant objects are dummy stubs. Two independent investigations + source + object sizes agree: **class (c), compiled-out, no real oracle obtainable without a `BUILD_RRTMK=1` / `BUILD_RRTMG_FAST=1` rebuild** (out of scope; would be a different WRF binary than the validated pristine build).

## Wiring (seams touched, set-UNION — no sibling clobber)
- `contracts/physics_registry.py` — `ACCEPTED_RA_SW_PHYSICS` += {3,7}, `ACCEPTED_RA_LW_PHYSICS` += {3,7}; `RA_*_SCHEMES` catalog entries for CAM/FLG.
- `contracts/physics_interfaces.py` — 4 reference-only `PhysicsStepSpec` (CAM lw/sw, FLG lw/sw) citing the real-WRF savepoint oracle.
- `io/scheme_catalog.py` — `_REFERENCE_ONLY` ra_lw/sw {3,5,7,99} with named reasons; `_PER_CODE_FAIL_CLOSED_REASON` for the compiled-out 14/24 (source-cited).
- `io/namelist_check.py` — `SUPPORTED_OPTIONS` ra docs + actions (3/5/7/99 reference-only, 14/24 compiled-out).
- `runtime/operational_mode.py` — `_SCAN_UNWIRED_REASON` ra_lw/sw {3,5,7,99} → fail-closed with the cited oracle path.

## Tests (proof objects)
- `tests/test_v018_ra_tail_oracle.py` — oracle exists/finite/non-trivial (LW+SW fired) + exact-module attribution + source checksums + every cited oracle path exists on disk + reference-only-and-fail-closed (3/5/7/99) + compiled-out-fail-closed-with-citation (14/24).
- `tests/test_scheme_catalog_fail_closed.py`, `tests/test_namelist_check.py`, `tests/contracts/test_v060_physics_interfaces.py` — catalog/registry/interface consistency green with the additions.

## Oracle numeric summary (generated 18 h real-WRF runs, exact-driver)

All four ran a **physics-pristine, WRFGPU2_ORACLE-instrumented `wrf.exe`** to `SUCCESS COMPLETE WRF` and produced a **non-trivial** savepoint with **both LW and SW fired** (max |·| over the run; `lw/sw` = both-nonzero flags):

| Scheme | RTHRATLW (K/s) | RTHRATSW (K/s) | GLW (W/m²) | OLR (W/m²) | SWDOWN (W/m²) | lw/sw fired | #nonzero |
|---|---|---|---|---|---|---|---|
| ra3 (CAM) | 0.00216 | 0.000115 | 435.9 | 324.6 | 1064 | ✓/✓ | 40 |
| ra5 (Goddard-new) | 0.00223 | 0.000706 | 398.8 | 322.8 | 1069 | ✓/✓ | 13 |
| ra7 (FLG/UCLA) | 0.00191 | 0.000673 | 396.3 | 0* | 1103 | ✓/✓ | 5 |
| ra99 (GFDL-Eta) | 0.000919 | 0.000368 | 391.7 | 0* | 1069 | ✓/✓ | 5 |

\* FLG and GFDL-Eta populate fewer of the standard wrfout radiation diagnostics than CAM (they do not write the `OLR` variable), but their core radiation IS exercised and non-trivial: `lw_nonzero_fields = [RTHRATLW, GLW]` (heating rate + surface downwelling LW) and `sw_nonzero_fields = [RTHRATSW, SWDOWN]` (heating rate + surface SW) for both. The savepoint records each module's real field set faithfully (scheme-absent fields go to `missing_fields`, never faked). Values are physical (surface SW ~1.06-1.10 kW/m² at the 12:00 UTC near-solar-noon frame; surface LW down ~390-436 W/m²).

`proofs/v018/ra_family_status.json` → **`full_ship_gate: true`** (6/6 tail schemes meet their bar).

## Unresolved risks (out of RA-tail scope — flagged for the trunk owner)

**Pre-existing operational-radiation NaN on `worker/opus/v018-trunk`.** During GPU+CPU verification I found `tests/test_{rrtm_lw,cdudhia_sw}_operational_wiring.py` execution tests fail: the operational radiation coupler `_physics_step_forcing(..., run_radiation=True).carry.rthraten` returns **all-NaN** for ra_sw/ra_lw in {1,4} (Dudhia/GSFC/RRTM/RRTMG — the *operational* path, class (a), NOT the RA tail). **Proven pre-existing**: it fails identically with my changes `git stash`-ed (clean base), on **both GPU and CPU**, so it is not introduced by this RA-tail work (which is pure fail-close metadata for reference-only/compiled-out schemes that never execute). Most-likely cause: the recent `#37` conditional-State-leaf refactor (commit `7bb30275`, "default mp=8 allocates no qh/Nh/…") left the radiation forcing path reading an unallocated leaf. **This is a v0.18 release blocker that belongs to the trunk owner, not the RA tail.** My RA-tail consistency + oracle tests are green on GPU+CPU.
