# v0.18 Integration Report

**Branch:** `worker/gpt/v018-integration`  
**Base:** `worker/opus/v018-trunk` (`fbec3544`)  
**Date:** 2026-06-16 (perf + honesty-fix closeout 2026-06-17)  
**Verdict:** **GREEN — FIX-then-ACCEPT closed.** Set-union integration and family
tests are green; the PERF-NEUTRALITY gate is **RESOLVED PERF-NEUTRAL** (superseded
by `perf_neutrality_FINAL.md` / `perf_rootcause_opus.md` — the "+7.71 % / +41.82 %"
below were cross-session hibernate clock-drift + a now-fixed `292a4431` carry
program-shape, NOT a real regression); and the two integration-honesty-critic
must-fixes (F1 silent-RED WRF-oracle test, F2 perf evidence/prose mismatch) are
closed. See **"Honesty-fix closeout (2026-06-17)"** at the end of this report for
the authoritative final state; the perf prose immediately below is the
pre-resolution record kept for provenance.

## Merge Summary

Merged the eight accepted family branches in contract order:

| Family | Branch | Merge result |
|---|---|---|
| MP | `worker/gpt/v018-mp` @ `268d0c8e` | fast-forward |
| CU | `worker/gpt/v018-cu` @ `6e9f8bd0` | clean merge |
| PBL/schemes | `worker/gpt/v018-schemes` @ `5e171e11` | clean merge |
| RA | `worker/gpt/v018-ra` @ `beb53041` | conflict, union-resolved |
| LSM | `worker/opus/v018-lsm` @ `5ce9a80e` | conflict, union-resolved |
| RAINNC/QVAPOR | `worker/gpt/v018-rainnc-qvapor` @ `8545ef3f` | clean merge |
| K2 | `worker/gpt/v018-k2` @ `4eaa7391` | clean merge |
| radfix | `worker/gpt/v018-radfix` @ `4e5a8f96` | clean merge |

## Conflict / Union Log

- `src/gpuwrf/io/scheme_catalog.py` RA conflict: preserved MP per-code fail-closed reasons, RA reference/compiled-out reasons for `ra_*={3,5,7,99,14,24}`, and PBL scheme-level fail-closed reasons.
- `tests/contracts/test_v060_physics_interfaces.py` RA conflict: unioned expected physics specs to base + harvested + PBL + RA coverage.
- `src/gpuwrf/io/scheme_catalog.py` LSM conflict: preserved LSM architecture-boundary fail-closed table for CLM4/CTSM (`sf_surface_physics=5/6`) while retaining MP/RA/PBL per-code and scheme-level reasons.
- No scheme/leaf family additions were intentionally dropped.

## Deferred Cleanups

1. CU stale wording fixed in `src/gpuwrf/runtime/operational_mode.py`: CU 5/93 now cite real nontrivial oracles instead of "all trial columns are null".
2. MP cleanup:
   - `proofs/v018/mp_endpoint_manifest.json` now labels MP17/19/21/22 as MP18 NSSL aliases.
   - Unsupported `mp_physics` suite message now includes `24,26`.
3. radfix cleanup:
   - `_total_or_legacy_field` now reads authoritative total fields directly; no per-step max/reduction fallback.
   - Synthetic tests build real `State` objects via `State.replace`.
   - One-time fail-loud `mu_total` check added for concrete operational initialization; HLO/lowering probes skip host concretization.
4. Provenance cleanup:
   - MP full-WRF wording now says physics-pristine, `WRFGPU2_ORACLE` dump-instrumented, numerically inert; no clean uninstrumented `wrf.exe` rebuild claimed.
   - CU/LSM/PBL status manifests now carry concise oracle provenance notes.

## Scheme / Leaf No-Clobber

Proof object: `proofs/v018/scheme_count_no_clobber.json` (`checks.all_green=true`).

Final operational sets:

- `mp_physics`: `[0,1,2,3,4,6,8,10,13,14,16,24,26,28,97]`
- `cu_physics`: `[0,1,2,3,6]`
- `bl_pbl_physics`: `[0,1,2,3,5,7,8,11,12,99]`
- `sf_sfclay_physics`: `[0,1,2,3,5,7,91]`
- `sf_surface_physics`: `[0,1,2,4,7]`
- `ra_lw_physics`: `[0,1,4,31]`
- `ra_sw_physics`: `[0,1,2,4]`

Final reference-only sets:

- `cu_physics`: `[4,5,14,16,93,94,95,96,99]`
- `bl_pbl_physics`: `[4,10,16,17]`
- `sf_surface_physics`: `[3,8]`
- `ra_lw_physics`: `[3,5,7,99]`
- `ra_sw_physics`: `[3,5,7,99]`

Boundary / recognized fail-closed sets proved:

- MP tail: `[5,7,9,11,17,18,19,21,22,27,29,30,32,38,40,50,51,52,53,55,56,95,96]`
- CU tail not accepted: `[7,10,11]`
- PBL CAM-UW boundary: `[9]`
- LSM architecture boundary: `[5,6]`
- RA compiled-out: `ra_lw/ra_sw [14,24]`

State leaf proof confirms the required accumulated precip, hail, aerosol, and hail-accumulator leaves are present and ordered.

## Test Results

All non-perf functional gates run on the integrated branch are green:

- Core consolidated family/registry/namelist gate:
  `154 passed`
- RAINNC/QVAPOR + Thompson/aerosol/hail substrate:
  `27 passed, 1 skipped`
- PBL/LSM inherited v017/v018 gates:
  `22 passed`
- K2/default-off + boundary/domain tests:
  `41 passed`
- Radiation/radfix wiring:
  `18 passed`
- CPU-only #69/radfix hygiene recheck after dropping per-step total/legacy reductions:
  `23 passed`
- K2 GPU-lock wrapped default-off graph proof:
  `proofs/v018/k2_flag_off_graph.json`, `passed=true`, no collectives, selected runner is the default function.

## Perf Neutrality

> **SUPERSEDED (2026-06-17).** The PENDING-REMEASURE / "+7.71 %" / "+41.82 %"
> verdict in this section was a pre-resolution snapshot taken during cross-session
> GPU clock drift (post-hibernate). It is **superseded by
> `proofs/v018/perf_neutrality_FINAL.md` + `proofs/v018/perf_rootcause_opus.md`**,
> which establish v0.18 is **PERF-NEUTRAL** vs v0.17 (committed GPT independent
> series −1.05 % = v0.18 faster; Opus structural root-cause: cold-process warm-cost
> +0.07 % noise, the only real regressor — the 81→74 carry narrowing — reverted
> bit-identically by `292a4431`). The text below is retained for provenance only.

Proof object: `proofs/v018/perf_neutrality_comparison.json`.

Canonical harness:

```bash
scripts/with_gpu_lock.sh --label gpt-integ-v018-confirm -- \
  env PYTHONPATH=src GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 JAX_ENABLE_X64=true \
  JAX_ENABLE_COMPILATION_CACHE=true JAX_COMPILATION_CACHE_DIR=<DATA_ROOT>/gpuwrf_jax_cache \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  TF_GPU_ALLOCATOR=cuda_malloc_async taskset -c 0-3 \
  python proofs/perf/warmed_timing.py
```

Measured warmed throughput:

- v0.18 integration confirm: **22.237 s/forecast-hour** (`61.770 ms/step`)
- v0.17 same-session rerun: **20.645 s/forecast-hour** (`57.348 ms/step`)
- v0.17 committed artifact (`worker/opus/v017-release:proofs/perf/warmed_timing.json`): **15.680 s/forecast-hour** (`43.556 ms/step`)

Deltas:

- v0.18 vs same-session v0.17: **+7.71%** (`+1.592 s/forecast-hour`)
- v0.18 vs committed v0.17 artifact: **+41.82%** (`+6.557 s/forecast-hour`)

**PERF-NEUTRALITY hard gate: PENDING-REMEASURE, not green.** The committed v0.17
artifact is **15.680 s/forecast-hour**, while the same v0.17 case rerun in this
session measured **20.645 s/forecast-hour** after hibernate drift. The apparent
**+41.82%** v0.18-vs-committed delta is therefore not accepted as a stable signal;
the only same-session signal recorded here is **+7.71%**, and it must be remeasured
after hibernate on a warmed, clock-stable GPU.

Note: system hibernate occurred during the v0.17 rerun compile path, so the same-session rerun is recorded separately from the committed v0.17 artifact. No new GPU perf re-measure was run after the hibernate decision.

Other default-path overhead candidates observed during the merge:

- RAINNC/QVAPOR plus hail/aerosol substrate widened the default `State` pytree even when those schemes are default-off; this can affect donation/dealiasing and HLO shape pressure.
- The public forecast wrapper still applies `_dealias_pytree_buffers` over the wider `State` tree once per call; not per-step, but a plausible warm-call overhead contributor.
- Registry/catalog fail-closed additions are mostly Python validation and are lower-priority for timestep overhead, but the larger default suite resolution path remains worth checking.
- K2 default-off is unlikely: `proofs/v018/k2_flag_off_graph.json` proves the selected runner is the default function and no collectives are emitted.
- The prime suspect radfix per-step fallback was removed here: production now reads `state.mu_total` / other total fields directly, and synthetic fixtures sync legacy/total fields through `State.replace`.

## Commands Run

- `git worktree add <USER_HOME>/src/wrf_gpu2/.wt-v018-integration -b worker/gpt/v018-integration worker/opus/v018-trunk`
- `git merge` for all eight family branches listed above.
- `python -m py_compile ...` for touched runtime/coupler/test modules.
- `PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true pytest -q ...` family suites listed above.
- `scripts/with_gpu_lock.sh --label gpt-integ -- ... scripts/verify_multigpu_dgx_sim.py --check flag-off`
- `scripts/with_gpu_lock.sh --label gpt-integ-v017-baseline -- ... python proofs/perf/warmed_timing.py`
- `scripts/with_gpu_lock.sh --label gpt-integ-v018-confirm -- ... python proofs/perf/warmed_timing.py`
- `PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true ... pytest -q tests/test_v014_mynn_surface_layer_regressions.py tests/test_rrtm_lw_operational_wiring.py tests/test_cdudhia_sw_operational_wiring.py`
- `python -m json.tool ...` on new proof JSON files.
- `git diff --check`

## Unresolved Risks

- ~~Release blocker: perf neutrality is pending remeasurement and is not green.~~
  **RESOLVED** — perf-neutral per `perf_neutrality_FINAL.md` (committed GPT series
  −1.05 % + Opus structural root-cause); the +7.71 %/+41.82 % were cross-session
  clock drift + the `292a4431`-fixed carry shape.
- Carried v0.18 items are documented in `KNOWN_ISSUES.md` (RAINNC bounded
  accumulated-precip residual; CLM4/CTSM v1.0 boundary fail-closed; K2 experimental
  specified-BC; Shin-Hong TKE-diagnostic follow-up). The F1 warm-process oracle was
  **fixed in code** (not carried) — see closeout below.

## Next Decision

None blocking for tag. The perf comparison is resolved perf-neutral; the
honesty-critic must-fixes are closed. Proceed to README / sanitize / tag.

---

## Honesty-fix closeout (2026-06-17, Opus honesty-fix worker)

Closes the two `integration_honesty_critic_opus.md` MUST-FIX items + the noahmp
path NIT before tag.

### F1 — silent-RED WRF-oracle test → **REGRESSION-FIXED in code (best-of-both)**

`tests/test_m5_thompson_process_residuals.py::test_rain_evaporation_and_warm_graupel_melt_cell_matches_wrf_mass_oracle`
was RED on trunk (qv abs err 4.93e-8 vs 1e-9 tol, ~49×) and was absent from the
integration test run → it would have shipped silently RED.

**Root cause (proven against pristine WRF at `<USER_HOME>/src/wrf_pristine`):**
accepted commit `044bb65a` ("v018 thompson cold-process fidelity") bundled, besides
the genuine cold-process additions, a new diagnostic graupel-number distribution
(`_graupel_distribution` + `_reset_mp8_graupel_number`). The graupel-MELT rate
(`prr_gml`) was NOT updated to match: WRF (module_mp_thompson.F:2802-2806) applies an
`N0_melt = (1.E-4/rg)*ogg2*lamg` override for sparse graupel (`rg*ng < 1.E-4`), which
the port omitted — so warm cells under-melted the now-finer-resolved graupel,
leaving a spurious `qg≈4.9e-7` where WRF melts it fully to 0. Independent secondary
leak: the new rci/sci cloud-ice-collection family lacked WRF's `if(temp<T_0)` cold
gate (line 2554), so it could form graupel at warm cells.

**Fix (HONEST, code, no tolerance loosening):**
1. Transcribed WRF's `N0_melt` sparse-graupel override into the `prr_gml` melt rate.
2. Gated the rci/sci ice-collection family on `state.T < T_0` (WRF's cold block).

**Result — strictly MORE WRF-faithful than both v0.17 and trunk** (vs the WRF mass
oracle, fixture `analytic-thompson-column-v1`):

| field | v0.17 max-abs-err | broken trunk | **after fix** |
|---|---|---|---|
| qv | 2.95e-6 | 1.02e-7 | **1.02e-7** |
| qr | 3.31e-8 | 4.42e-7 | **4.94e-11** |
| qg | 2.95e-6 | 4.92e-7 | **7.72e-9** |
| T  | 8.30e-3 | 9.47e-4 | **9.47e-4** |

Cell (2,2) qv err 4.93e-8 → **1.2e-13** (bit-exact WRF). The cold-process gain is
preserved: `test_thompson_cold_collection_oracle` (both column-rain-sink + warm-
inactive cases) stays GREEN; `tier1`/`tier2`/`precip_oracle` GREEN. The m5 file is
now in the run suite and green. Also audited and folded in the previously-excluded
oracle tests (m5 process-residuals, cold-collection, precip, noahmp energy).

### F2 — Opus +0.54 % perf figure lacked a committed JSON → **DOC reconciled**

No committed JSON in the repo contains `21.1516 / 21.2661`. `perf_neutrality_FINAL.md`
and `perf_rootcause_opus.md` now state the Opus +0.54 % pair is prose-only
(JSON not captured), and that the committed dual-confirm rests on the committed GPT
series (`gpt_verify_*`, −1.05 % = v0.18 faster) + the Opus structural root-cause.
Perf-neutral verdict unchanged and no longer overclaims a missing artifact.

### Path NIT — noahmp energy savepoint gate hardcoded a non-existent path → **FIXED**

`proofs/noahmp/energy_savepoint_gate.py` defaulted `WRF_PRISTINE_ROOT` to
`ROOT.parent/"wrf_pristine"/"WRF"` = `<USER_HOME>/src/wrf_gpu2/wrf_pristine/WRF`
(does not exist). Now defaults to the canonical `<USER_HOME>/src/wrf_pristine/WRF`
(env `WRF_PRISTINE_ROOT` still overrides; repo-sibling kept as last-resort fallback).
`test_real_wrf_energy_savepoint_parity` now RUNS and passes (was effectively
skipping/failing on the path).
