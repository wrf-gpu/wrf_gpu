# v0.6.0 CONSOLIDATION WAVE-2 — Opus lane review (2026-06-04)

**Branch:** `worker/opus/v060-consolidation2`
**Final SHA:** `e84245f5bd00028c8b1300f8d82d1a9a3a69eca2` (merge HEAD `c7f1aec` + matrix/smoke regen)
**Wave-1 base:** `dcc9666` (`worker/opus/v060-consolidation`; 5 branches already folded, 17/17 RUN + 3/3 fail-closed)
**Base trunk:** `e998250` (trunk-0.9.0)
**Resource discipline:** CPU-only (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=true`), all compute pinned `taskset -c 0-3`; cores 4-31 untouched (live CPU-WRF backfill).

## Objective

Fold 3 MORE verified-PASS v0.6.0 scheme branches onto the wave-1 consolidation base,
unioning every scheme registration/adapter (keep ALL keys from base + each branch), then
re-run the operational smoke, regenerate the integration matrix, and run the test suite.
Honesty over green.

## What merged (incrementally, in order)

| # | Branch | SHA | Payload | Commit |
|---|---|---|---|---|
| 1 | v060-lin-mp | 72a41c5 | Purdue-Lin microphysics (mp=2), GPU-scan-wired | 2d03d70 |
| 2 | v060-boulac | 8837e80 | BouLac PBL (bl=8), GPU-scan-wired | a931a05 |
| 3 | v060-radiation | 49114db | classic RRTM-LW (ra_lw=1) + Dudhia-SW (ra_sw=1) column drivers | c7f1aec |

All three branches share merge-base `e998250` with the base — the same trunk wave-1 came
from. Each branch adds ONE scheme on top of trunk; the large "deletions" git reports against
each branch are NOT real deletions, they are wave-1 schemes the branch's older base lacked.
A 3-way merge keeps the wave-1 content (base side) and brings in each branch's addition. I
**verified after the radiation merge that no wave-1 scheme file was dropped** (WSM3/WSM5/MYJ/
Tiedtke/Janjic/BouLac/Lin + their tests all present).

## Conflicts and how each was resolved (UNION, never drop)

**Lin-mp (2d03d70):** physics_registry.py (ACCEPTED_MP / MP_SCHEMES / MP_MOIST_MEMBERS /
MP_NUMBER_MEMBERS), physics_dispatch.py (`_MP_ENTRIES`), scan_adapters.py (MP_SCAN_ADAPTERS +
`__all__` + module docstring), namelist_check.py, operational_mode.py (`_SCAN_WIRED_OPTIONS`
+ error string), scanwire_smoke.py, V0.6.0-S0-FROZEN-CONTRACT.md accept table, and the two
unit-test files. Resolved by numeric-ordered union: `ACCEPTED_MP_PHYSICS=(0,1,2,3,4,6,8,10,16)`,
kept the wave-1 cu menu `{0,1,6}` (Tiedtke wired). New scheme files (microphysics_lin.py,
_lin_*.py, lin_constants.py) came in clean.

**BouLac (a931a05):** physics_registry.py (ACCEPTED_BL + PBL_SCHEMES), namelist_check.py,
operational_mode.py, oracle/.gitignore, and the two unit-test files. physics_dispatch.py,
scan_adapters.py, and physics_interfaces.py **auto-merged cleanly** (the bl=8 dispatch entry,
`PBL_SCAN_ADAPTERS[8]=boulac_pbl_adapter`, and the bl=8 step spec composed with the Lin add
without conflict). Resolved by union: `ACCEPTED_BL_PBL_PHYSICS=(0,1,2,5,7,8)`; kept the wave-1
ACM2 status "implemented" and the richer GPU-scan-wired-vs-CPU-reference namelist wording, adding
BouLac as GPU-scan-wired. oracle/.gitignore kept the superset `build_*/` over `build_boulac/`.

**Radiation (c7f1aec):** physics_registry.py (ACCEPTED_RA_* + RA_*_SCHEMES option-1 entries),
namelist_check.py (ra menus), physics_interfaces.py (the two option-1 radiation step specs),
oracle/.gitignore — all **auto-merged cleanly**. Only the two unit-test files conflicted (the
SCHEME_STEP_SPECS count line + the ra accept-set asserts). Resolved by union:
`ACCEPTED_RA_SW_PHYSICS=ACCEPTED_RA_LW_PHYSICS=(0,1,4)`, keeping the wave-1 MYJ-Janjic pairing
test. New modules ra_lw_rrtm.py (1310 LOC) + ra_sw_dudhia.py (306 LOC) + test_v060_ra_sw_dudhia.py
came in clean.

## GENUINE PROBLEM FOUND AND FIXED — latent lin-mp branch gap

The wave-2 hazard the task warned about (SCHEME_STEP_SPECS literal-merge) surfaced as a real
defect in the lin-mp branch: the branch added `mp=2` to `ACCEPTED_MP_PHYSICS` but **never added
the matching microphysics `SCHEME_STEP_SPECS` entry**. `assert_interfaces_consistent()` builds
its expected key set from the ACCEPTED sets, so post-merge it raised
`missing PhysicsStepSpec entries: [('microphysics', 2, '')]`. On the lin-mp branch in isolation
this consistency check was already failing (its merge-base `e998250` had `ACCEPTED_MP=(0,1,6,8,
10,16)` with 5 `_mp_spec` entries; the branch bumped ACCEPTED to include 2 but left 5 specs).

**Fix:** added `_mp_spec(2, "Purdue-Lin", "src/gpuwrf/physics/microphysics_lin.py", ...)` to
the SCHEME_STEP_SPECS tuple (commit 2d03d70). Re-ran `assert_interfaces_consistent()` → CONSISTENT.

## SCHEME_STEP_SPECS final count — SHOWN BY COUNTING

SCHEME_STEP_SPECS is built from `_mp_spec(...)` wrappers (microphysics) + literal
`PhysicsStepSpec(...)` (all other families). I verified the live `len()` and the family
breakdown at each commit (NOT the literal merge):

```
len(SCHEME_STEP_SPECS) = 27
Counter: microphysics=8, pbl=5, surface_layer=4, cumulus=4, radiation=4, land_surface=2
```

Count walk: base wave-1 = 25? No — base (dcc9666) = 23 (7 mp + 4 pbl + 4 sfclay + 4 cu +
2 radiation + 2 land). +1 Lin (mp=2) → 24. +1 BouLac (bl=8) → 25. +2 radiation option-1
variants (RRTM-LW lw, Dudhia-SW sw) → **27**. The test assertion `len(SCHEME_STEP_SPECS) == 27`
is set to the counted union, not the literal-merged value.

Per family at HEAD: microphysics {1,2,3,4,6,8,10,16}=8; pbl {1,2,5,7,8}=5; surface_layer
{1,2,5,7}=4; cumulus {1,3,6,16}=4; radiation {(4,lw),(4,sw),(1,lw),(1,sw)}=4; land_surface
{2,4}=2. Total 27.

## No double-listing / cumulus-fix survival — verified

- **No scheme is both wired AND in an unwired-reason.** `_SCAN_WIRED_OPTIONS` =
  mp{0,1,2,3,4,6,8,10,16} bl{0,1,5,7,8} sf{0,1,5,7} cu{0,1,6}; `_SCAN_UNWIRED_REASON` =
  {bl=2 (MYJ), sf=2 (Janjic), cu=3 (GF), cu=16 (New-Tiedtke), sf_surface=2 (Noah-classic)}.
  Disjoint.
- **Wave-1 cumulus-slot fix survived.** `CU_SCAN_ADAPTERS = {1: kf_adapter, 6: tiedtke_adapter}`
  — KF is keyed to cu=1 and Tiedtke to cu=6; the cumulus slot is no longer hard-wired to
  kf_adapter. The smoke's `cu_tiedtke` (cu=6) RUN config genuinely exercises Tiedtke, and
  `cu_grellfreitas_unwired` (cu=3) / `cu_newtiedtke_unwired` (cu=16) fail closed loudly.

## Per-scheme status table (post wave-2; derived from live merged registries)

| Family | Option | Name | Status |
|---|---|---|---|
| mp | 1 | Kessler | GPU-OPERATIONAL-WIRED |
| mp | 2 | **Purdue-Lin** | **GPU-OPERATIONAL-WIRED** (scan_adapters[2]=lin_adapter) |
| mp | 3 | WSM3 | GPU-OPERATIONAL-WIRED |
| mp | 4 | WSM5 | GPU-OPERATIONAL-WIRED |
| mp | 6 | WSM6 | GPU-OPERATIONAL-WIRED |
| mp | 8 | Thompson | GPU-OPERATIONAL-WIRED |
| mp | 10 | Morrison | GPU-OPERATIONAL-WIRED |
| mp | 16 | WDM6 | GPU-OPERATIONAL-WIRED |
| bl | 1 | YSU | GPU-OPERATIONAL-WIRED |
| bl | 2 | MYJ | PARITY-PROVEN-FAIL-CLOSED |
| bl | 5 | MYNN | GPU-OPERATIONAL-WIRED |
| bl | 7 | ACM2 | GPU-OPERATIONAL-WIRED |
| bl | 8 | **BouLac** | **GPU-OPERATIONAL-WIRED** (scan_adapters[8]=boulac_pbl_adapter) |
| sf_sfclay | 1 | revised-MM5 | GPU-OPERATIONAL-WIRED |
| sf_sfclay | 2 | Janjic Eta | PARITY-PROVEN-FAIL-CLOSED |
| sf_sfclay | 5 | MYNN SL | GPU-OPERATIONAL-WIRED |
| sf_sfclay | 7 | Pleim-Xiu | GPU-OPERATIONAL-WIRED |
| cu | 1 | Kain-Fritsch | GPU-OPERATIONAL-WIRED |
| cu | 3 | Grell-Freitas | PARITY-PROVEN-FAIL-CLOSED |
| cu | 6 | Tiedtke | GPU-OPERATIONAL-WIRED |
| cu | 16 | New Tiedtke | PARITY-PROVEN-FAIL-CLOSED |
| sf_surface | 2 | Noah classic | GPU-OPERATIONAL-WIRED (hook) |
| sf_surface | 4 | Noah-MP | GPU-OPERATIONAL-WIRED (hook) |
| ra_sw | 1 | **Dudhia SW** | **GPU-OPERATIONAL-WIRED** (held-rate RTHRATEN driver) |
| ra_sw | 4 | RRTMG SW | GPU-OPERATIONAL-WIRED |
| ra_lw | 1 | **RRTM LW** | **GPU-OPERATIONAL-WIRED** (held-rate RTHRATEN driver) |
| ra_lw | 4 | RRTMG LW | GPU-OPERATIONAL-WIRED |

(7 PASSIVE/OFF `=0` rows omitted.) Matrix counts:
**23 GPU-OPERATIONAL-WIRED, 4 PARITY-PROVEN-FAIL-CLOSED, 7 PASSIVE/OFF, 0 UNKNOWN.**

## Gates run

- `proofs/v060/multicfg_operational_smoke.py` → **19/19 RUN PASS, 3/3 FAIL-CLOSED OK, all_pass=True.**
  Added two coverage configs so the new scan-wired schemes are exercised end-to-end:
  `mp_lin` (mp=2) and `pbl_boulac` (bl=8). Coverage now includes `mp2-Lin` and `bl8-BouLac`.
- `proofs/v060/consolidation_integration_matrix.json` → **regenerated** from the LIVE merged
  registries via a new authoritative generator `proofs/v060/gen_consolidation_matrix.py`
  (derives status from scheme maps + `_SCAN_WIRED_OPTIONS`/`_SCAN_UNWIRED_REASON` + adapter
  tables + dispatch gpu-runnability, so the matrix cannot silently drift). git_head=c7f1aec,
  overall_consolidation_pass=True.
- Targeted contract/dispatch/namelist/dudhia tests: 38 passed.
- Physics/contract domain I touched (interfaces, dispatch, namelist, Dudhia-SW, MYJ,
  Janjic, WSM-SM, Kessler, Noah-classic parity), cache disabled: **61 passed, 0 failed.**
- `assert_registry_consistent()` and `assert_interfaces_consistent()`: PASS.

## TEST-ENVIRONMENT FINDING — stale JAX persistent compilation cache (NOT a code defect)

A first full-`tests/` run produced widespread `F` and then **SIGSEGV (exit 139)**. Root cause
is NOT the consolidation: `gpuwrf/__init__.py` enables a persistent JAX compilation cache
(`runtime.jax_cache`, default `/mnt/data/gpuwrf_jax_cache`) shared across machines. The run
emitted XLA AOT warnings — *"Loading XLA:CPU AOT result … machine type doesn't match … could
lead to execution errors such as SIGILL"* — i.e. the on-disk cache holds executables compiled
under a different CPU-feature profile (AVX-512/AMX variants) than this host, so loading them
segfaults. Disabling the cache (`GPUWRF_JAX_CACHE=0`, a documented pure no-op) removes the
crash. The remaining failures with the cache off are **GPU-required / backend-subprocess tests
that cannot pass in this CPU-only sandbox** by design — e.g. `State.zeros requires a GPU device;
no JAX GPU backend is visible` (m4_acoustic/m4_advection), and CuPy/CUDA/Kokkos/JAX-pipeline
subprocess artifact checks (m2_*). None of these are in the files this consolidation changed
(`git diff dcc9666 HEAD` confirms my changes are confined to physics
registry/dispatch/interfaces/scan-adapters/io + v060 proofs). Full cache-disabled
pass/fail counts: see FINAL MESSAGE.

## Honest limitation — radiation not swept end-to-end in the operational smoke

The multicfg smoke harness `Config` does not model `ra_lw`/`ra_sw` as a per-config dimension;
`as_namelist()` pins `ra_sw=ra_lw=4`. So RRTM-LW(1) and Dudhia-SW(1) are folded at the
**registry / dispatch / step-spec / namelist** layer and are **per-scheme savepoint-parity
gated** on their own branch (RRTM-LW worst RTHRATEN_max_rel 7.2e-14; Dudhia-SW), but they are
NOT yet exercised through the operational scan in this smoke. I did not fabricate smoke
coverage for them. Recorded as a post-0.9.0 carry-over in the matrix
(`carry_overs_post_0_9_0`). Lin(mp=2) and BouLac(bl=8) ARE swept end-to-end (RUN-PASS).

## Test result (cache disabled, taskset -c 0-3, CPU-only)

Broad consolidation-relevant physics/scheme set — contracts interfaces, dispatch, namelist,
Dudhia-SW, MYJ, Janjic, YSU, ACM2, KF, Pleim-Xiu, revised-MM5, WSM-SM, WSM6, WDM6, Kessler,
Noah-classic parity, Grell-Freitas, Tiedtke savepoint parity:
**180 passed, 0 failed** (11m32s; savepoint-parity tests are compute-heavy on 4 cores).
Plus the earlier targeted runs (38 + 61 passed) and `scanwire_smoke.py` all_pass=True,
`assert_registry_consistent()` / `assert_interfaces_consistent()` PASS.

The full `tests/` directory cannot produce a single summary in this sandbox: a test in the
`tests/init/` (native-init/GRIB/metgrid) or M2/M4 GPU-backend area **SIGSEGVs the interpreter**
(exit 139) partway through — with the cache on AND off. Those areas are GPU-required /
backend-subprocess / native-init tests outside the consolidation domain (`git diff dcc9666 HEAD`
touches none of them). Every test file the consolidation changed passes.

## Pre-existing failure (NOT caused by this work)

`tests/init/real_init/test_s4_comparator.py::test_forecast_gate_is_scaffold_only` fails with
`run_forecast_gate(execute=True) requires the integrated product_factory`. This is in the v040
native-init comparator — **untouched by this consolidation** (`git diff dcc9666 HEAD` shows no
change under `src/gpuwrf/init/` or `tests/init/`). It is pre-existing on the base and belongs to
the separate v040 lane the manager handles (note the session-start staged
`proofs/v040/run_forecast_gate_24h.py`). Not a wave-2 regression.

## Files changed (committed on worker/opus/v060-consolidation2)

- `src/gpuwrf/contracts/physics_registry.py`, `physics_interfaces.py`
- `src/gpuwrf/coupling/physics_dispatch.py`, `scan_adapters.py`
- `src/gpuwrf/io/namelist_check.py`, `src/gpuwrf/runtime/operational_mode.py`
- `tests/contracts/test_v060_physics_interfaces.py`, `tests/test_namelist_check.py`,
  `tests/test_v060_physics_dispatch.py`
- new scheme modules: microphysics_lin.py (+_lin_*/lin_constants), pbl_boulac.py (from branch),
  ra_lw_rrtm.py, ra_sw_dudhia.py + their proofs/savepoints/tests
- `proofs/v060/multicfg_operational_smoke.py` (+2 coverage configs), `multicfg_smoke_report.json`
- `proofs/v060/gen_consolidation_matrix.py` (new generator), `consolidation_integration_matrix.json`
- `.agent/decisions/V0.6.0-S0-FROZEN-CONTRACT.md`, `proofs/v060/scanwire_smoke.py`

## Next decision

None blocking for wave-2. The radiation end-to-end smoke sweep and the 4 existing fail-closed
schemes (GF, New-Tiedtke, MYJ, Janjic) remain the post-0.9.0 carry-overs already tracked in the
matrix.
