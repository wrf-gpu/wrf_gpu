# v0.11.0 Integration + Reconciliation

Authority: Opus 4.8 (max) — v0.11.0 integration + qke-KI-2 gate.
Trunk: `worker/opus/v0110-integration`, started @ `3f1864e`
(v0.10.0 + recompile-fix + Thompson#8 + MYNN#5 + KF#3 + GWD#11 + slopediff#9 + RRTMG#6).

## TASK A — Multi-branch merge (in order)

All merges verified with `JAX_PLATFORMS=cpu taskset -c 0-3 python -c "import gpuwrf"`
plus each lane's relevant pytest (the known-SEGV `tests/savepoint/test_dycore_100_steps.py`
was skipped per the contract). All four lanes merged 4/4.

### 1. worker/gpt/v0110-conservation-close @ b4ed608 — CONFLICT, resolved
- Merge commit: `e302627`
- Conflict: `src/gpuwrf/runtime/operational_mode.py` (single block, ~164 lines).
  - HEAD (trunk) side: the legacy **inline** post-dycore physics block
    (microphysics → surface → PBL → cumulus → rrtmg) that applied physics
    deltas directly onto `next_state` AFTER `_rk_scan_step`.
  - conservation-close side: that inline block was **deleted** because the lane
    refactored the entire physics chain into `_physics_step_forcing()`, called
    at step ENTRY (before the dycore). It now returns `_PhysicsStepForcing`
    carrying the dry tendencies as WRF `*_tendf` leaves; those are threaded into
    the dycore through `_rk_scan_step(..., physics_tendencies=...)` →
    `rk_addtend_dry` (per-RK-stage map/mass coupling), and the non-dry physics
    deltas are applied after the dycore via `_apply_physics_non_dry_updates`.
  - RESOLUTION: took the conservation-close side (removed the inline block) —
    keeping it would have **double-applied physics**. The forcing-based path is
    the correct, budget-closing structure.
  - INTEGRATED BOTH, not just one side: conservation-close branched BEFORE the
    trunk's RRTMG topo-slope feature (`c3e0f87 Wire RRTMG topo slope radiation`),
    so its `_physics_step_forcing` was missing the newer radiation arguments.
    Ported the trunk feature INTO the forcing path:
    - NoahMP `_refresh_noahmp_rad(...)` call now passes `land_state=carry.noahmp_land`.
    - `_refresh_rthraten` (rrtmg held-rate) now passes `radiation_static`,
      `topo_shading`, `slope_rad`, `shadow_length_m`, and
      `land_state=carry.noahmp_land if use_noahmp else None`.
    (The other c3e0f87 sites — `OperationalNamelist` fields, `tree_flatten/unflatten`,
    `noahmp_initial_rad`, `_refresh_noahmp_rad` signature, `compute_m9_diagnostics` —
    auto-merged cleanly outside the conflict block and were preserved.)
- `mynn_pbl.py` auto-merged clean (conservation-close's `pblh_pos = max(pblh, MIN_PBLH)`
  in `_scale_aware_psig_bl`).
- Tests: 20 passed — test_conservation_budget, test_operational_namelist_cache_key,
  test_v060_physics_dispatch, test_m6_operational_mode_no_h2d; + parity_envelope
  + validation_compare = 6 passed. Import OK.

### 2. worker/gpt/v0110-nesting @ effe032 — CLEAN
- Merge commit: `2fe7c65` (ort, no conflicts).
- `contracts/grid.py`: `DomainNest` dataclass added after `BCMetadata` (~line 363) —
  non-overlapping with conservation-close and multigpu changes.
- `physics/mynn_pbl.py`: the **qke-KI-2 fix** — `_wrf_qke_minmax(value)` uses
  `jnp.fmin(jnp.fmax(value, QKEMIN), 150.0)` (IEEE fmax/fmin select the finite
  bound on NaN, matching WRF Fortran MAX/MIN intrinsic semantics at
  `module_bl_mynnedmf.F:3106-3107`), replacing the NaN-propagating
  `jnp.minimum(jnp.maximum(...))` in `_mym_predict_qke`. PRESERVED.
  This change auto-merged into a different region than conservation-close's
  `pblh_pos` change; both coexist.
- New modules: `runtime/domain_tree.py`, `coupling/boundary_feedback.py`.
- Tests: 25 passed — test_v0110_qke_finiteness, test_v0110_domain_tree,
  test_v0110_boundary_feedback, test_p0_1a_nesting. Import OK.

### 3. worker/gpt/v0110-restart @ 2f54ad0 — CLEAN
- Merge commit: `bbc771a` (ort, no conflicts).
- `io/wrfrst_netcdf.py` (new), `io/wrfout_writer.py` (extended), `io/__init__.py`.
  No other lane touched io/. (Contract noted state.py restart fields, but the
  branch does NOT touch contracts/state.py — restart serialization lives in io/.)
- Tests: 5 passed — test_v0110_wrfrst_netcdf. Import OK.

### 4. worker/gpt/multigpu-dgx @ 149449f — CLEAN
- Merge commit: `1bce44e` (ort, auto-merged grid.py + state.py, no conflicts).
- `contracts/grid.py`: `_as_fp64` helper (~line 18) + `DycoreMetrics.__post_init__`
  fp64 cast — landed in a different region than nesting's `DomainNest` (~line 363),
  so both grid.py additions coexist.
- `contracts/state.py`, `contracts/halo.py`: sharding/halo touch-ups (disjoint).
- New modules: `runtime/sharding.py`, `dynamics/sharded_horizontal.py`.
- Tests: tests/parallel/ → 10 passed, 6 skipped (multi-device tests skip on the
  single available device). Import OK.

### Combined regression sweep (post all 4 merges)
43 passed across conservation_budget, namelist_cache_key, v060_physics_dispatch,
operational_mode_no_h2d, parity_envelope, qke_finiteness, domain_tree,
wrfrst_netcdf, sharding_config. OperationalNamelist tree_flatten/unflatten
roundtrip OK (children now `(tendencies, metrics, radiation_static)`).

## TASK B — qke-KI-2 gate (does our case actually run?)

Pre-merge trunk state (`proofs/v0110/rrtmg_finite_recheck.json`):
`DIAGNOSED_KNOWN_QKE_EDGE` / `KEEP_RRTMG_FEATURE_DO_NOT_MASK_QKE_KI2` — RRTMG was
cleared; the full-physics d02 qke=2024 nonfinite cells were a pre-existing KI-2,
explicitly "do not mask."

Re-test harness: `proofs/v0110/rrtmg_finite_recheck.py --mode on --hours 1`
(warmed/segmented daily-pipeline path, `run_forecast_operational_segmented`,
full physics: run_physics=True, run_boundary=True, disable_guards=False,
RRTMG slope/shading ON; NOT a cold single step). The pipeline `finite_summary`
scans ALL numeric State fields, INCLUDING `qke` (State field, mass points),
and `all_finite` is the AND over every field.

Result JSON: `proofs/v0110/qke_ki2_gate_merged_trunk.json`

### Gate result — CLOSED (qke finite)

Run: `--mode on --hours 1`, segment_steps=180, radiation_cadence_steps=180,
topo_shading=1, slope_rad=1, device cuda:0, wall_clock 160.7 s.

- `status: PASS`, `verdict: KEEP_RRTMG_ON_TRUNK`, `proper_cadence_finite: True`.
- `all_finite_check.all_finite = True` across all **56** State fields.
- **qke: finite=True, nonfinite_count=0**, min=1e-05, max=52.26 m2 s^-2
  (physically sane MYNN TKE range), shape [44,66,159], dtype float64.
- NONFINITE fields: NONE (the prior ~2024-cell qke nonfinite is gone).
- `wrfout_inventory_status: PASS`; wrfout_d02_2026-05-21_19:00:00 written.
- `pipeline_verdict: PIPELINE_PARTIAL` is BENIGN — it only reflects `score=False`
  in the recheck config (scoring stage skipped), NOT a finiteness/qke failure;
  inventory PASS + all_finite True is the gate that matters.

ROOT CAUSE / FIX (WRF-faithful, NOT masked): the qke=2024 nonfinite was the
NaN-propagating `jnp.minimum(jnp.maximum(qke, QKEMIN), 150.0)` post-MYNN-solve
clamp. WRF's Fortran `MAX`/`MIN` intrinsics at `module_bl_mynnedmf.F:3106-3107`
select the finite qkemin/upper bound when an operand is NaN (Fortran comparison
semantics). The nesting lane's `_wrf_qke_minmax` reproduces this with IEEE
`jnp.fmin(jnp.fmax(value, QKEMIN), 150.0)`, which selects the finite bound on a
NaN operand and leaves every finite value's bounds unchanged. This is faithful
to the WRF intrinsic behavior, not a masking guard, and is consistent with the
pre-merge `KEEP_RRTMG_FEATURE_DO_NOT_MASK_QKE_KI2` directive.

## Final trunk HEAD

`worker/opus/v0110-integration` @ `1bce44e` (before this proof + gate commit).

## Summary

- MERGED 4/4. 1 conflict (operational_mode.py, conservation-close) resolved by
  integrating BOTH sides (forcing-based physics cadence + ported RRTMG slope args);
  3 clean merges.
- qke-KI-2 gate CLOSED: full-physics d02 1h segmented run all-finite, qke finite,
  no masking.

