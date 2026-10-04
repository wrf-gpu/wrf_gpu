# SPRINT A (Claude-MAX) — v0.20.0 Wave-1: S1 + S2 — HANDOFF

**Worker:** Claude-MAX frontrunner (pane 0:7). **Branch:** `worker/claude/v020-s1s2-compaction`.
**Date:** 2026-06-21. **Mode:** CPU-only (JAX x64; GPU left to the live corpus). **Manager:** 0:1.
**Status:** S1 + S2 COMPLETE, all gates green. fp64_default BIT-IDENTICAL. Manager-approved scope.

---

## Objective
Implement S1 (remove hot-carry total aliases + dtype audit tooling + dtype-stability test) and
S2 (intrinsic fp64-island locking) from `FINAL_FP32_SPRINT_PLAN.md §3`, as a precision-invariant
liveness/safety-net refactor that keeps the shipped v0.19.x line runnable and the `fp64_default`
path (incl. the nested fused cascade) **bit-identical**. No fp32 is added here — this prepares the
ground for the S3–S8 perturbation-authoritative work.

## Key finding (mapped, not invented) — scope of S1, manager-approved
The brief's literal S1 ("remove `p_total`/`ph_total`/`mu_total`, reconstruct totals only at I/O
from base+perturbation") is **NOT achievable bit-identically**, for a concrete provable reason:
- The dycore is **total-authoritative**. `small_step_prep`/`small_step_finish` and
  `operational_mode` recover the base as `(state.p_total − state.p_perturbation)` **every RK
  stage** and rebuild `p_total_new = (p_total_old − p_pert_old) + p_pert_new`; `state.p_total`
  is also consumed **directly** (EOS inverse density, `_advance_chunk`).
- Carrying the perturbation + a STATIC explicit base and reconstructing `total = base + pert`
  changes the last-ULP arithmetic (`(a−b)+b ≠ a` in IEEE fp64) and removes the direct-total reads
  → fails the bit-identical gate. That is exactly the **perturbation-authoritative rewrite = S4**
  (roadmap §8.6(3)), tolerance-gated and behind the additive fp32 mode flag.

**What the codebase ACTUALLY carries** (per family): a legacy alias `p` that is a *bitwise-identical
duplicate* of the authoritative `p_total` (kept in sync at every `replace`, equal at every
construction site), plus the independent `p_perturbation`. Same for `ph`/`mu`.

**S1 delivered (the maximal bit-identical liveness reduction, manager-approved):** remove the
legacy duplicate leaves `p`/`ph`/`mu` from the State hot carry (re-exposed as read-only properties
aliasing the totals). This removes **3 full-grid arrays per step from the scan carry**
(74→71 carry leaves; 37→35 full-grid 3-D leaves) with **zero numerical change**. The literal
total-removal is deferred to S4 (manager confirmed). Back-channel: `QUESTION.md` + 0:1 message;
manager APPROVED 2026-06-21.

---

## Files changed

### New
- `src/gpuwrf/profiling/dtype_audit.py` — S1 audit tooling: `field_dtypes`/`named_dtypes`
  (per-field dtype through the hot path), `count_converts`/`count_converts_for` (XLA
  convert-element-type counter split f32→f64 / f64→f32 / other, StableHLO + classic HLO),
  `assert_dtype_stable`/`diff_float_dtypes` (FAIL on silent f32↔f64 promotion via `eval_shape`).
- `tests/test_v020_dtype_stability.py` — the dtype-stability gate (8 tests): fp64_default hot step
  is dtype-stable; the detector FIRES on injected f64→f32 AND f32→f64; JAX's own fori_loop carry
  check is a documented second layer; convert-counter direction/StableHLO parsing.
- `proofs/v020/s1s2/harness.py` — CPU bit-identity (`compare`) / liveness (`liveness`) /
  nested-trace (`nested`) harness on a balanced warm-bubble operational `_advance_chunk`.

### S1 — State pytree surgery (bit-identical)
- `src/gpuwrf/contracts/state.py` — removed `p`/`ph`/`mu` from `__slots__` and
  `_state_field_shapes`; added read-only `p`/`ph`/`mu` properties → totals; `__init__` folds the
  legacy kwargs into the totals (back-compat; total wins); `replace` redirects legacy writes to
  the totals and preserves the EXACT total↔perturbation delta arithmetic.
- `src/gpuwrf/contracts/precision.py` — removed `p`/`ph`/`mu` from `STATE_FIELD_ORDER` (keeps it
  == `State.__slots__`); `dtype_for` falls back to `PRECISION_MATRIX` so
  `BaseState`/`BoundaryState`/`Tendencies` allocators keyed by the legacy names are unchanged.
- Tests updated for the −3 leaf count (tripwire updates, all referencing v0.20 S1):
  `test_v018_conditional_state_leaves`, `test_v017_qh_hail_state`,
  `test_v016_thompson_aero_threading`, `test_m7_restart_checkpoint_roundtrip`,
  `test_v0110_wrfrst_netcdf`, `test_m6x_c2_pgf`, `test_m6x_c2_acoustic`.

### S2 — intrinsic fp64-island locking (no-op for fp64; protects fp32 callers)
- `src/gpuwrf/contracts/precision.py` — `force_fp64_island(*arrays)`: widens
  cancellation-sensitive operator inputs to fp64; **identity (no convert op) when already fp64**,
  so fp64_default is bit-identical with zero new HLO converts.
- `src/gpuwrf/dynamics/core/calc_p_rho.py` — EOS (`_calc_al_p`): lock al/p bracket inputs.
- `src/gpuwrf/dynamics/core/acoustic.py` — PGF brackets (`advance_uv_wrf`): lock p/ph/p_base/
  al/alt before the horizontal pressure-gradient differences (covers face pairs, dpn, php).
- `src/gpuwrf/dynamics/core/advance_w.py` — implicit w·φ solve (`advance_w_wrf`) + vertical PGF
  source (`pg_buoy_w_dry`): lock the solve/buoyancy/tridiagonal inputs.
- `src/gpuwrf/dynamics/acoustic_wrf.py` — EOS helpers (`_inverse_density_from_theta_pressure`,
  `_pressure_from_theta_alt`): lock theta/pressure/alt. (`calc_coef_w_wrf_coefficients` was
  ALREADY intrinsically fp64 via `jnp.asarray(..., float64)` — noted, unchanged.)

---

## Commands run (CPU-only; `JAX_PLATFORMS=cpu PYTHONPATH=src`)
- `python proofs/v020/s1s2/harness.py save --steps 6`   (baseline captured pre-change)
- `python proofs/v020/s1s2/harness.py compare --steps 6`   → gate (a)
- `python proofs/v020/s1s2/harness.py liveness`            → gate (b)
- `python proofs/v020/s1s2/harness.py nested`              → gate (d)
- `python -m pytest tests/test_v020_dtype_stability.py`    → gate (c)
- `python -m pytest` over the State/restart/dycore/operational/idealized/dynamics suites (below)

## Proof objects (`proofs/v020/s1s2/`)
- **(a) gate_a_bit_identical.txt** — fp64_default **BIT-IDENTICAL**: 71/71 retained leaves
  bitwise-equal over 6 operational steps; the only removed leaves are exactly `state.p`/`ph`/`mu`
  (the duplicates). Covers S1 + S2 combined vs the pre-change baseline.
- **(b) gate_b_liveness.txt** — OperationalCarry scan carry 74→71 leaves; full-grid 3-D leaves
  37→35; the p/ph/mu total family now holds only `p_total`/`ph_total`/`mu_total`; `_advance_chunk`
  HLO convert ops unchanged (S2 added zero converts on the fp64 path).
- **(c) gate_c_dtype_stability.txt** — 8/8 pass: fp64_default dtype-stable; promotion detector
  fires both directions.
- **(d) gate_d_nested_trace.txt** — nested fused cascade (`GPUWRF_NESTED_FUSE` default-on) still
  LOWERS to HLO (d02 fused program) and runs on CPU (own_steps d01:1/d02:3/d03:9, finite).
- `baseline_advance.npz` — the pre-change baseline leaf dump (regenerable via `harness.py save`
  on the parent commit). `harness.py` — the reproducer. `QUESTION.md` — the back-channel.

## Test sweep (CPU)
- State/restart contract: `test_m3_state`, `test_m6_state_extension`,
  `test_v018_conditional_state_leaves`, `test_v017_qh_hail_state`,
  `test_v016_thompson_aero_threading`, `test_m7_restart_checkpoint_roundtrip`,
  `test_p0_5_restart_full_carry`, `test_v0110_wrfrst_netcdf`, `test_v020_dtype_stability`
  → **46 passed, 5 skipped**.
- Dycore/coupled/unit: `test_m6x_c2_acoustic`, `test_m6x_c2_pgf`, `test_conservation_budget`,
  `test_m6_tier2_coupled`, `test_m6b5_dycore_step_parity`, `tests/unit/`
  → **25 passed, 3 skipped, 3 xfailed**.
- Operational/idealized/dynamics: `test_v013_operational_smoke`, `tests/idealized/`,
  `tests/dynamics/`, `test_m6_horizontal_pressure_gradient_fix` → **107 passed, 6 skipped**
  (4 deselected = the 2 pre-existing radiation failures below). Combined with the two sweeps
  above: ~178 tests green, **zero new failures introduced by S1/S2**.
- **Pre-existing failures (NOT caused by this sprint — reproduced identically on a clean HEAD
  worktree):** (1) `tests/test_m7_1km_memory_audit.py::{test_static_model_tracks_state_contract_
  field_count, test_static_model_uses_precision_registry_for_known_fields}` — the audit SCRIPT's
  own `STATE_FIELD_ORDER == _state_field_shapes` contract check already mismatched on the 7
  conditional leaves before any change. (2) `tests/test_v013_operational_smoke.py::{test_ra_lw_
  operational_runs_and_cools[1], test_sw_and_lw_are_selected_independently}` — `FileNotFoundError`
  (a radiation data table absent in this environment), failing identically on clean HEAD. All left
  as-is (out of S1/S2 scope).

## Residual risks
- **Restart format**: a v0.20 wrfrst/checkpoint omits the 3 legacy p/ph/mu variables; read-back
  reconstructs them via the properties (name-based, symmetric — round-trip tests green). Reading a
  PRE-v0.20 restart still works (the legacy p/ph/mu are accepted as `__init__` kwargs and folded
  into the totals). v0.19.x `main` is untouched and stays runnable.
- **fp32 island coverage**: S2 locks the named hot operators (EOS, PGF, implicit-w·φ, +the EOS
  helpers and pg_buoy source). Any NEW cancellation bracket added in S3–S8 must call
  `force_fp64_island` at its boundary — the dtype-stability gate + convert-counter are the tripwire.
- **m7_1km_memory_audit pre-existing failure** unrelated to this sprint; flagged for a separate fix.

## Next decision needed
None blocking. S4 (perturbation-authoritative total removal) must be implemented behind the
additive fp32 mode flag, tolerance-gated, keeping fp64_default on the exact current arithmetic
(manager already confirmed). The shared-base-state contract (roadmap §8.6(3)) is the natural S4
entry point; `BaseState(pb/phb/mub)` + the `_base_*` helpers already exist as the foundation.
