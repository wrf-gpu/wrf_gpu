# v0.18 Operational Radiation NaN Fix

## Objective

Fix the v0.18 release blocker where operational radiation produced all-NaN
`rthraten` for RRTM/Dudhia/RRTMG wiring paths, while preserving the #37
conditional-State-leaf default allocation gate.

## Root Cause

The NaN was not caused by a missing #37 conditional microphysics leaf. The same
focused failure reproduces on the parent of suspect commit `7bb30275`
(`8faf93f6`): `5 failed, 13 passed` in
`proofs/v018/radfix_parent_8faf_cpu_operational_wiring.log`.

The actual cause was a legacy-vs-c2 total-field alias mismatch in operational
physics prep. The synthetic operational wiring/smoke states initialize legacy
`mu` and `ph`, but leave c2 total aliases `mu_total` and `ph_total` as zeros.
Surface/PBL prep then used the zero total aliases:

- hydrostatic pressure prep needed `mu_total`
- density prep needed `mu_total` and `ph_total`

That made pressure faces degenerate (`p_down == p_up`) and geopotential
differences zero, so the density reconstruction hit `0 / (p * log(1))` at
`src/gpuwrf/coupling/physics_couplers.py:1209` and emitted NaN. Surface/PBL
state then became NaN before radiation, so every active radiation scheme
returned all-NaN `rthraten`.

## Fix

Added `_total_or_legacy_field` at
`src/gpuwrf/coupling/physics_couplers.py:1145`, and used it in:

- `_wrf_hydrostatic_pressure_from_state` at line `1161`
- `_wrf_phy_prep_rho_from_state` at lines `1196-1197`

The helper keeps c2 total fields when they are initialized. It falls back to the
legacy alias only when the total alias is exactly zero and the legacy alias has
nonzero signal. This fixes the operational synthetic state path without adding
conditional #37 leaves, widening tolerances, masking NaNs, or clamping
radiation output.

Added a focused regression test:

- `tests/test_v014_mynn_surface_layer_regressions.py:111`
  `test_wrf_phy_prep_rho_falls_back_to_legacy_fields_when_totals_uninitialized`

## Scope

Before the fix:

- `proofs/v018/radfix_before_cpu_operational_wiring.log`: focused CPU wiring
  repro, `5 failed, 13 passed`; failures were all-NaN `rthraten`.
- `proofs/v018/radfix_before_scope_clean_trunk.log`: `(ra_sw=0, ra_lw=0)` was
  finite zero; every active tested SW/LW combo among `ra_sw in {0,1,2,4}` and
  `ra_lw in {0,1,4}` had `all_nan=True`, `nan_count=72/72`.

After the fix:

- `proofs/v018/radfix_after_scope_cpu.log`: every tested combo is finite with
  `nan_count=0/72`; active-combo max absolute `rthraten` is in the
  `9.399356e-05` to `4.634871e-04` range, so the fix is not merely non-NaN.

## Validation

Commands were run with CPU cores pinned to `0-3`; GPU commands used
`scripts/with_gpu_lock.sh --label gpt-radfix` and the shared JAX compilation
cache under `/mnt/data/gpuwrf_jax_cache`.

- CPU focused wiring + regression:
  `proofs/v018/radfix_after_cpu_focused.log` - `22 passed in 42.59s`.
- CPU conditional-State gate:
  `proofs/v018/radfix_after_cpu_conditional_state.log` -
  `12 passed, 1 skipped in 3.64s` (`State.zeros` GPU-only case skipped on CPU).
- CPU operational radiation coupled smoke:
  `proofs/v018/radfix_after_cpu_operational_radiation_smoke.log` -
  `7 passed in 116.83s`.
- GPU focused operational wiring:
  `proofs/v018/radfix_after_gpu_operational_wiring.log` -
  `18 passed in 42.34s`.
- GPU #37 conditional allocation check:
  `proofs/v018/radfix_after_gpu_conditional_state.log` -
  `1 passed in 3.19s`.
- GPU operational radiation coupled smoke:
  `proofs/v018/radfix_after_gpu_operational_radiation_smoke.log` -
  `7 passed in 67.71s`.
- Default-path allocation gate:
  `proofs/v018/radfix_after_default_path_allocation_gate.log` confirms mp=8
  remains `active_fields=60`, `tree_leaves=60`, and every conditional leaf
  (`qh`, `Nh`, `qvolg`, `qvolh`, `nwfa`, `nifa`, `hail_acc`) remains absent.

## Risks

The fallback intentionally handles the transitional synthetic-state alias case
only. If a real c2 state has genuinely initialized nonzero total fields, it
continues to use those totals. If both total and legacy aliases are zero, the
state remains invalid rather than silently fabricating physics inputs.
