# v0.9.0 Release-Trunk Merge — 7 validated branches → single trunk + GPU green-suite classification

**Author:** Opus 4.8 (1M context) release-integration lane, 2026-06-04
**Branch:** `worker/opus/v090-release-trunk` (from `worker/opus/trunk-0.9.0` @ `7b7c26e`, the v0.6.0-closed base)
**Merged trunk SHA:** `5670a0e`
**Mode:** CPU fp64 (`taskset -c 0-3`) for merges + import checks; **GPU-visible** (RTX 5090 cuda:0, x64 on) for the full pytest suite. Cores 4-31 (live 28-rank cpu-wrf-backfill, pid 133927) untouched throughout. GPU lock claimed/released around each GPU run; `cpu_cores_4_31` claim preserved.

## Objective

Merge the 7 validated v0.9.0 feature branches into one release trunk (combine intents, never clobber), confirm `import gpuwrf` after each, then run the full suite GPU-visible and classify every failure as real-regression vs known-acceptable. Honesty over green.

## PART 1 — The 7 merges

All 7 merged in the contracted order. **All seven merged cleanly; exactly ONE conflict, in a proof-artifact JSON (not source).**

| # | Branch | Head | Merge commit | Result |
|---|---|---|---|---|
| 1 | physics-consolidation | bd1e3cf | 8187fab | clean (first merge) |
| 2 | diffopt1-smagorinsky | 6e90d03 | 637adc6 | clean — ort auto-merged `operational_mode.py` + `namelist_check.py`; both intents combined (verified) |
| 3 | gf-scanwire | c580df1 | d46b622 | **1 conflict** — `proofs/v060/multicfg_smoke_report.json` (worktree-path string). Resolved: took gf-scanwire side; `sweep_rationale` auto-resolved to the v0.9.0 text. GF cu=3 wired (verified). |
| 4 | qkefix-followup | 6215014 | 6b08146 | clean (ort) — qke cold-start seed + d02/d03 replay hardening |
| 5 | deadcode-cleanup | c2cd5c1 | a2a686d | clean (ort) — dead acoustic helpers removed AND diffopt1 intent preserved (verified); 4 dead tests deleted |
| 6 | testsweep | 6b4d898 | c74e078 | clean (ort) — no conflict with deadcode test deletions; `_m9_snapshot` allowance present |
| 7 | readme-runnability | bf280f9 | 81fbc79 | clean (ort) — `README.md` auto-merged despite divergence from `e998250`; `gpuwrf run` CLI works |

Conflict-hotspot files (`operational_mode.py`, `scan_adapters`, `namelist_check`, `physics_registry`, `physics_dispatch`, the tests) all auto-merged with the `ort` strategy into NON-overlapping regions; I verified combined intent (e.g. `operational_mode.py` carries diffopt1 `explicit_diffusion` imports AND the deadcode removal AND physics changes; `CU_SCAN_ADAPTERS = {1:kf, 2:bmj, 3:gf, 6:tiedtke}`). No conflict markers remain anywhere (`git grep` clean). `import gpuwrf` OK after every merge. Diff vs base: 117 files, +18992/−1215.

## PART 2 — Full GPU-visible suite

`PYTHONPATH=src taskset -c 0-3 python -m pytest tests/ -q` with GPU visible (jax backend=gpu, x64 on). 1256 tests, 38 min.

**Result @ 81fbc79 (pre-fix): 89 failed, 1132 passed, 33 skipped, 2 xfailed.** The suite ran to completion across all 1256 tests with **no SIGSEGV abort** — `test_dycore_100_steps.py` degraded to a benign KeyError instead of the CPU-only `libjax_common.so` native crash, satisfying the contract's GPU-env expectation.

## PART 3 — Failure classification (the honest part)

**Real merge regressions: 0. Fixed in this lane: 2. Known-acceptable environment residuals: 87.**

### Baseline diff (the decisive proof)
I re-ran the EXACT failing test files on the BASE `7b7c26e` with GPU visible (minus the 2 namelist tests I fixed and the savepoint test that SEGVs in CPU isolation). **Base = 86 failures, merged-trunk (same scope) = 86 failures, and the failure SETS are byte-identical** (`diff` empty). Every failure in that scope is pre-existing on the v0.6.0-closed base.

### The 2 genuine merge-surfaced failures — FIXED
`tests/test_namelist_check.py::{test_recognized_but_unimplemented_dynamics_option, test_real_wrf_namelist_input_is_consumable}`. The physics-consolidation merge (namelist-compat sub-branch) ADDED tests asserting `diff_opt=1/km_opt=4` must raise `UnsupportedNamelistOption` ("NOT YET IMPLEMENTED"). The diffopt1-smagorinsky merge then actually **WIRED** `diff_opt=1/km_opt=4` (the WRF real-data-default 2-D Smagorinsky path, parity-proven in `proofs/v090/diffopt1_smagorinsky_parity.json`). Two branches developed in parallel against the same base; neither saw the other; the merge surfaced the stale assertions. **The production code is correct.** I updated both tests to the now-correct combined behavior (diff_opt=1/km_opt=4 validates cleanly; the full pristine `em_real` oracle namelist is now fully consumable; `km_opt=99` still fails closed). Commit `5670a0e`. `test_namelist_check.py` now 26/26 PASS. This is the only test change — no tolerance loosened, no fix weakened.

### The 87 environment residuals (NOT regressions; base-identical)
- **8 files — scratch-venv missing optional deps.** M2 backend-bakeoff edge tests spawn subprocesses in `data/scratch/m2-*-venv` lacking jax/zarr/cupy/triton (`ModuleNotFoundError`); CuPy/Triton `ArgInfo` TypeErrors are ADR-001-superseded backends not provisioned here.
- **18 files — missing data/oracle/corpus fixtures.** FileNotFoundError/CalledProcessError/KeyError downstream of absent `data/profiler_artifacts/*`, `data/canairy_meteo/runs/wrf_l3/*`, Fortran oracle binaries, `wrf_pristine` reference trees, publication-audit citation files.
- **13 files — pre-existing dycore COMPARATOR-HELPER bug.** `m6b*/m6x*` validation-mode comparator scripts hit `theta=None` (`AttributeError`) / asserts in the synthetic-dryrun path. These are explicitly "not the production dycore path." Production dycore (`mu_t_advance.py`, `core/acoustic.py`, `core/dycore.py`) and the comparator scripts are UNCHANGED by all 7 merges.
- **1 file — `test_dycore_100_steps.py`.** Documented `FLAKY_ENV_NATIVE_JAX_CRASH` + missing WRF reference savepoints; benign KeyError in the warm GPU suite, SEGV in CPU isolation. Contract-listed as known-acceptable.

## Operational scheme matrix (intact)

Scan-wired (GPU-operational through the coupler): **MP {1,2,3,4,6,10,16}, CU {1,2,3,6} (GF cu=3 NOW wired via `gf_adapter`/`gfdrv_batched`), PBL {1,7,8}, SFCLAY {1,7}**, RRTMG ra=4, Noah-MP(4)+Noah-classic(2). 26 schemes covered by run configs. **Fail-closed set: bl2-MYJ, cu16-NewTiedtke, sf2-Janjic** (coupler rejects loudly). (One stale `excluded_documented_todo` line in `multicfg_smoke_report.json` still calls GF a TODO — pre-gf-scanwire artifact; the live code + the authoritative `covered_by_run_configs` both include cu3-GF. Left unrewritten to avoid fabricating a proof run.)

## Deliverables
- 7 merge commits + 1 conflict resolution + 1 namelist test-fix on `worker/opus/v090-release-trunk` (`5670a0e`).
- `proofs/v090/release_trunk_greensuite.json` (this classification).
- this review.

## Risks / carry-over
1. This deployment cannot run the full historical M2-M7 milestone suite green (missing optional deps + data/oracle/corpus fixtures). The validation+benchmark burst should run on the provisioned corpus host, or re-provision those fixtures.
2. Pre-existing `m6b*/m6x*` comparator-helper `theta=None` bug — validation-mode only, separate carry-over to triage (pre-dates this lane).
3. Run the savepoint dycore tier GPU-visible inside the warm full suite (not isolated/CPU) to avoid the native SEGV.
