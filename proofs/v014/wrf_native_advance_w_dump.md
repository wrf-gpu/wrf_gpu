# v0.14 WRF-native intra-`advance_w` oracle — Switzerland h36, call 21601→21602

Proof object for the sprint `2026-06-11-v014-wrf-native-advance-w-dump`.
Companion files: `wrf_native_advance_w_dump.py` (assembler + comparator +
captured replica), `wrf_native_advance_w_dump.json` (all numbers),
`wrf_native_advance_w_dump_wrf_patch.diff` (disposable WRF instrumentation).

## What was built

A disposable instrumented WRF v4.7.1 copy
(`/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF`, additive env-gated patches in
`module_small_step_em.F`, `solve_em.F`, `module_em.F`) re-ran the bit-exact
36h30m 24-rank d01 Switzerland truth twice (~25 min each) and dumped, at
`itimestep=7201, rk_step=1, iteration=1` (the first-bad RK1 acoustic substep,
WRF call 21601→21602):

1. `advance_w` entry inputs, all internals (`rhs` at 3 build stages, `wdwn`,
   Thomas forward RHS / fwd / back, pre-damp solved `w`), exit `w/ph/t_2ave`,
   and the in-loop `calc_p_rho` inputs/outputs (48 per-rank files, fp32
   big-endian stream + sidecars);
2. `rk_tendency` `rw_tend`/`ph_tend` snapshots after each contributor
   (`advect_w` → `pg_buoy_w` → `w_damp` → `coriolis` → `curvature`) plus
   `*_tendf` (48 files).

Dumps live outside git under
`/mnt/data/wrf_gpu_validation/v014_switzerland_awd_dump/awd_dumps/` (96 files,
130,257,560 bytes; per-file sha256 in the JSON `manifest`). Gating sanity: the
dumped small-step `mu''` equals the independent HPG (call 21602 − 21601) `mu`
increment to fp32 roundoff (interior rmse 1.86e-5 vs field rms 0.135).

The JAX side reuses the proven h36/call-21601 stage context
(`switzerland_advance_w_term_split.py`) plus a captured replica of
`advance_w_wrf` asserted bit-identical to production (max abs diff 0.0 on all
three outputs).

## Findings (earliest mismatching terms, in WRF execution order)

| # | Term | Verdict | Evidence (interior rmse vs WRF native) |
|---|------|---------|----------------------------------------|
| 1 | `pg_buoy_w` stage pressure input | **ROOT CAUSE, FIXED** | gap 511 of 1337 (38% rel, k-profile = whole near-surface peak). Operator+mu+mub+cqw exact: with WRF `p`, JAX pg_buoy matches to 4.6e-4. Cause: stage-entry `diagnose_pressure_al_alt` recompute instead of WRF's carried `grid%p`; the carried JAX leaf `state.p_perturbation` is bit-exact vs WRF `p` (diff 0.0). |
| 2 | w cosine-Coriolis term | **MISSING in JAX, FIXED** | 172 rms, mid-column 230–256; WRF `coriolis` adds it to `rw_tend` (module_big_step_utilities_em.F:3836-3843). |
| 3 | w curvature term | **MISSING in JAX, FIXED** | 7.2 rms (module_big_step_utilities_em.F:4283-4291). |
| 4 | `top_lid=True` (production) vs WRF open top | **PROVEN UNFAITHFUL** | dumped `flags_toplid=F`; with the lid the replica zeroes top-face `rhs`/`w` and the Thomas back-substitution propagates the error down the whole column (k-profile decays from top). Open-top operator isolation collapses to fp32 roundoff (`ph_out` 5.2e-7). |
| 5 | `advect_w` + filters | excluded | JAX vs WRF 4.85 rmse (4.1% rel of 118.6). |
| 6 | `w_damp` | excluded | WRF contribution identically 0 at this step (and JAX's would be too). |
| 7 | `advance_w` implicit solve + `calc_p_rho` operators | **proven WRF-faithful** | all-WRF-inputs open-top isolation: `rhs_c` 1.4e-8, `w_out` 1.2e-3, `ph_out` 5.2e-7; `calc_p_rho` `p` 5.3e-8, `al` 1.3e-11. |

Remaining named non-dominant input gaps (next targets): `ph_tend` (13.9% rel),
`t_2` coupled work theta (53.7% rel, small absolute), `ww` (51.1% rel of a tiny
0.003-rms field), `mu''` (15.5% rel) — all quantified in the JSON
`input_cascade`.

## Production fixes implemented (`src/gpuwrf/runtime/operational_mode.py`)

1. `pg_buoy_w` now consumes the carried `state.p_perturbation` (WRF
   `calc_p_rho_phi` carry cadence) instead of a per-stage
   `diagnose_pressure_al_alt` recompute — WRF-faithful and one fewer full-grid
   diagnostics pass per RK stage.
2. WRF `rk_tendency` cosine-Coriolis + curvature vertical-momentum terms added
   to the once-per-stage `rw_tend` assembly (`GPUWRF_W_CORIOLIS=0` opt-out).

## Stage-gate results (interior rmse vs WRF-native `w_out`/`ph_out` at the dumped substep)

| Config | w_out | ph_out |
|--------|------:|-------:|
| pre-fix production (`top_lid=True`) | 923.0 | 0.4353 |
| pre-fix + WRF rw_tend + open top (oracle bound) | 35.5 | 0.01924 |
| **post-fix production (`top_lid=True`)** | **109.5 (−88%)** | **0.0554 (−87%)** |
| **post-fix + open top** | **35.7 (−96%)** | **0.01925 (−96%)** |

Post-fix + open top equals the oracle bound (feeding WRF's exact `rw_tend`
adds nothing) — the `rw_tend` lane is closed; the residual 0.019 ph is carried
by the smaller named inputs above.

GPU jitted-production gates (`switzerland_acoustic_substep_blocker.py
--stage-compare`) recorded in `switzerland_acoustic_substep_blocker.json`
under tags `v014_awd_fixes_lid` / `v014_awd_fixes_open`; see the sprint review
for the numbers.
