# Sprint Contract — M10: Static-Field + LU_INDEX Parity Fix

**Sprint ID**: `2026-05-28-m10-lu-index-static-fix`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m10-lu-index-static-fix`
**Worktree**: `/tmp/wrf_gpu2_m10`
**Wall-time**: 4-12 h (target 1 day)
**GPU usage**: YES — for validation forecast
**Sandbox**: `--sandbox danger-full-access`

## Why this sprint

M9 divergence map confirmed: LU_INDEX has a max-abs delta of 14 categories every hour (constant static-field defect). RCA from iter2 already flagged this. The State pytree in `src/gpuwrf/contracts/state.py` has fields like `xland`, `lakemask`, `roughness_m` but no `lu_index` leaf. WRF uses LU_INDEX to drive surface category-dependent parametrisations (roughness, albedo, emissivity per category). Until LU_INDEX is correct, M12 (surface flux), M13 (radiation), and M16 (land surface) all run on the wrong category map.

This sprint adds LU_INDEX to the State, populates it from the same wrfinput file that produces the existing xland/lakemask/roughness_m, and verifies bitwise match against WRF.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs

1. `proofs/m9/divergence_map.json` — confirms LU_INDEX defect
2. `src/gpuwrf/contracts/state.py` — State pytree definition
3. `src/gpuwrf/runtime/operational_mode.py` — where lu_index needs to be plumbed
4. Wherever xland/roughness_m are populated — same path will populate lu_index
5. `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/wrfinput_d02` — reference LU_INDEX values
6. WRF user-guide: LU_INDEX is on the cross-grid (mass points), integer dtype, USGS or MODIS land-use category

## Acceptance

### AC1 — State extended with lu_index leaf

`src/gpuwrf/contracts/state.py` adds `lu_index: jax.Array` (int32, shape (ny, nx) — mass-grid 2D). State init path populates it from wrfinput.

### AC2 — Static-field parity test

`tests/savepoint/test_static_fields.py` (NEW) asserts bitwise match against wrfinput for LU_INDEX, HGT, LANDMASK, XLAND, ROUGHNESS_M, soil category fields. **Passes on canary 20260521**.

### AC3 — Operational run uses correct LU_INDEX

Re-run `operational_mode` for Canary 20260521 24h with new LU_INDEX wired through. Emit `proofs/m10/static_field_parity_after_fix.json`:
```json
{
  "lu_index": {"max_abs_diff": 0, "rmse": 0, "verdict": "BITWISE_MATCH"},
  "hgt": {"verdict": "BITWISE_MATCH"},
  "landmask": {"verdict": "BITWISE_MATCH"},
  ...
}
```

### AC4 — INV-8 added to invariant ladder check

The savepoint smoke now includes static-field parity. `taskset -c 0-3 pytest -q tests/savepoint/` reports the new test PASSING.

### AC5 — Skill regression check (small, not the gate)

Re-run iter2 5-day Canary case skill diff `scripts/m7_gpu_vs_cpu_skill_diff.py` with new LU_INDEX. Compare to post_iter2_skill_diff.json. Emit `proofs/m10/post_m10_skill_diff.json`. Acceptance: **non-regression** on T2/U10/V10 RMSE (does not need to improve, must not get worse).

### AC6 — `.agent/sprints/2026-05-28-m10-lu-index-static-fix/worker-report.md`

Standard format. Verdict: `M10_COMPLETE` if AC1-AC5 all pass; `M10_PARTIAL` with explicit unfinished list otherwise.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: ALLOWED for the validation forecast + skill diff. Single GPU instance.
3. **Files writable**: `src/gpuwrf/contracts/state.py` (extend, don't rewrite), `src/gpuwrf/runtime/operational_mode.py` (wire-through only), `tests/savepoint/test_static_fields.py` (NEW), `proofs/m10/**`, `.agent/sprints/2026-05-28-m10-lu-index-static-fix/**`.
4. **Files NOT writable**: anything physics-coupler, anything dycore, anything boundary, anything governance.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Single-purpose**: this sprint only adds the lu_index leaf + populates it + verifies static-field parity + runs skill non-regression. NO other physics or dycore changes.
8. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: m10 DONE exit=$?" Enter`.
9. **End with verdict**: `M10_COMPLETE` / `M10_PARTIAL` + one-line headline.
