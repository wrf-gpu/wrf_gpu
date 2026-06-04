# v0.9.0 namelist compatibility — WRF-user-friendly scheme validation

**Date:** 2026-06-04
**Worker:** Opus 4.8 (1M)
**Branch:** `worker/opus/v090-namelist-compat` (from `worker/opus/trunk-0.9.0` @ 7b7c26e)
**Scope:** Low-risk IO/validation only. No core/physics/dynamics changes.

## Objective

Make the GPU port's namelist handling WRF-user-friendly: a user switching from
`real.exe`/`wrf.exe` can bring their existing `namelist.input` and get CLEAR,
SPECIFIC fail-closed errors for schemes not yet ported — distinguishing a
*recognized WRF v4 scheme that is not yet implemented* from a *value that is not
a valid WRF option at all*.

## Parser assessment (does the port consume a real WRF namelist.input?)

**Yes.** The port reads a standard Fortran WRF v4 `namelist.input`. It uses a
**custom, lightweight Fortran-namelist parser** (`_parse_wrf_namelist` in
`namelist_check.py`), *not* `f90nml`. It correctly splits all standard groups
(`&time_control`, `&domains`, `&physics`, `&fdda`, `&dynamics`, `&bdy_control`,
`&grib2`, `&namelist_quilt`), strips `!` comments, lowercases keys, and parses
comma-separated per-domain value lists. Verified against:

* `/home/enric/src/wrf_pristine/WRF/run/namelist.input` (8 groups, `max_dom=2`),
* `/home/enric/src/wrf_pristine/WRF/test/em_real/oracle_run/namelist.input`
  (Thompson/KF/MYNN/Noah-MP/RRTMG suite),
* `/mnt/data/canairy_meteo/runs/cu0_confirm/.../namelist.input`.

The custom parser is adequate for the scheme-validation layer; I did **not**
rewrite the IO path.

### Real compatibility limitations found (reported + the safe one fixed)

1. **Fortran repeat-count syntax `N*value` was not handled** (FIXED, low-risk
   parser change). `mp_physics = 3*8` (a *very* common WRF idiom meaning three
   domains of `8`) was parsed as the literal string `'3*8'` and would be
   mis-flagged as an invalid value. Now expands to `[8, 8, 8]`; bare `N*`
   (keep-defaults) is dropped. Added `_expand_repeat`.

2. **Real-data WRF diffusion defaults fail closed (reported, NOT changed —
   correct behavior).** WRF's recommended real-data settings `diff_opt=1` +
   `km_opt=4` (horizontal Smagorinsky) are genuinely unimplemented — only the
   constant-K path `diff_opt=2 / km_opt=1` is wired
   (`dynamics/explicit_diffusion.py`). These now fail closed with the specific
   "recognized WRF scheme, NOT YET IMPLEMENTED" message rather than a generic
   one. A user must switch to `diff_opt=2, km_opt=1` (or `0`) to run an existing
   real-data namelist. This is an honest physics gap, not a parser bug.

## Three-outcome validation (now in place)

`validate_supported_namelist()` classifies every rejected selection against the
full WRF v4 catalog (`wrf_scheme_catalog.py`, 131 codes across 14 keys,
transcribed from `WRF/run/README.namelist` with per-entry line refs):

* **(a) implemented/accepted -> pass.**
* **(b) recognized WRF v4 scheme, NOT YET IMPLEMENTED -> fail closed, specific:**
  `physics.mp_physics=28 (aerosol-aware Thompson (water/ice-friendly)):
  recognized WRF v4 microphysics scheme, NOT YET IMPLEMENTED in the GPU port.
  Supported mp_physics values: 0, 1, 2, 3, 4, 6, 8, 10, 16. ...`
* **(c) not a valid WRF v4 option -> fail closed:**
  `physics.mp_physics=99 is not a recognized WRF v4 microphysics option. ...`

Fail-closed-loud safety preserved — the validator never silently accepts an
unimplemented scheme. The reference fail-closed schemes (GF cu=3, New Tiedtke
cu=16, MYJ bl=2, Janjic sf=2, classic RRTM ra_lw=1, Dudhia ra_sw=1) remain
*accepted* at the namelist layer with their existing reference notes intact;
their operational-scan fail-close is a downstream runtime concern unchanged here.

## Implemented-scheme matrix

| Parameter | Selectable (impl / accepted-ref) |
|---|---|
| mp_physics | 0,1,2,3,4,6,**8(impl)**,10,16 |
| cu_physics | 0,1,2,6 ; 3 GF-ref, 16 New-Tiedtke-ref |
| bl_pbl_physics | 0,**1**,**5**,**7**,8 ; 2 MYJ-ref |
| sf_sfclay_physics | 0,1,**5**,7 ; 2 Janjic-ref |
| sf_surface_physics | 0,2,**4** |
| ra_sw_physics | 0,**4(op)** ; 1 Dudhia-ref |
| ra_lw_physics | 0,**4(op)** ; 1 RRTM-ref |
| dynamics | rk_order=3; diff_opt 0/2 + km_opt 0/1; diff_6th_opt 0/2; damp_opt 0/3; w_damping 0/1; sf_urban_physics 0 |

## Files changed

* `src/gpuwrf/io/wrf_scheme_catalog.py` (new) — full WRF v4 code->name catalog.
* `src/gpuwrf/io/namelist_check.py` — three-outcome classification + repeat-count
  parsing; `UnsupportedSelection` gains `outcome`/`wrf_scheme`.
* `tests/test_namelist_check.py` — 26 tests (was 4).
* `docs/namelist-compatibility.md` (new) — the bring-your-namelist story.

## Commands run

* `python -m pytest tests/test_namelist_check.py -q` -> **26 passed**.
* `assert_registry_consistent()` -> OK.
* End-to-end Path-based validation (CLI entry simulation) -> fail-closes
  correctly with the specific message.

## Proof objects

* 26-test suite (implemented suite; recognized-but-unimplemented mp=28/50,
  bl=4 QNSE, cu=5 Grell-3D, sf_surface=3 RUC/5 CLM4, sf_sfclay=3 GFS, radiation;
  invalid values; reference-failclosed message preservation; dynamics
  diff_opt=1/km_opt=4; repeat-count expansion; real oracle namelist parse).
* `docs/namelist-compatibility.md` (naive-agent gate doc).

## Risk

**Low.** Pure IO/validation; no physics/dynamics touched. The catalog is
pure data. The only behavior change to an *accepting* path is the repeat-count
fix (strictly more permissive, and only for the genuine Fortran `N*value`
idiom). One existing test assertion string was updated to match the improved
messages; values/supported-lists it checked are still present.

## Next decision needed

None blocking. Optional follow-ups: (i) wire `diff_opt=1`/`km_opt=4` real-data
diffusion (physics sprint, not this lane); (ii) if/when the CLI on
`worker/opus/readme-runnability` merges to trunk, this validation layer is
already its fail-closed dependency — no integration change needed.
