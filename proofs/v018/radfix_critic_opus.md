# v0.18 Operational Radiation NaN Fix — Opus Adversarial Critic

Critic: Opus 4.8 (max). Branch `worker/gpt/v018-radfix` @ `6b70ee1e` (confirmed HEAD).
Posture: assume the fix masks rather than cures. Verdict at bottom.

## VERDICT: ACCEPT

The fix is correct, physically faithful, and a proven no-op for validated real
cases. The root-cause diagnosis is right, and — importantly for the
real-vs-artifact question — the production fallback is a *defensible* fix even
though the trigger was a malformed synthetic fixture, because it is provably
identity on every real state and provably non-fabricating on a truly-invalid
state. See "Real risk vs fixture artifact" for the nuance and the one
recommended (non-blocking) follow-up.

## 1. Real risk vs test-fixture artifact — RULING

**The trigger was a malformed SYNTHETIC fixture, not a real code path.**

- The failing synthetic state builder zeros every field, then sets only the
  *legacy* aliases `mu`/`ph` while leaving the *authoritative* totals
  `mu_total`/`ph_total` at zero, and constructs the State **directly** via
  `State(**fields)`:
  `tests/test_rrtm_lw_operational_wiring.py:78-97` (`mu=...90000`, no `mu_total`)
  and the same pattern in `tests/test_cdudhia_sw_operational_wiring.py`.
- `State.__init__` does **not** sync legacy↔total
  (`src/gpuwrf/contracts/state.py:636-648` stores both args verbatim). The
  legacy/total aliasing is enforced **only** in `State.replace`
  (`state.py:816-832`: `values[legacy] = values[total]` and vice versa). So a
  state built with `State(**...)` from a dict that omits `mu_total` is internally
  inconsistent in a way no real state is.
- Every REAL init/restart path populates `mu_total` (nonzero) and keeps the
  legacy alias equal to it:
  - idealized IC: `src/gpuwrf/ic_generators/idealized.py:549-550` sets both
    `mu` and `mu_total` to the same array.
  - d02 real replay: `src/gpuwrf/integration/d02_replay.py:2076,2086`
    builds via `state.replace(..., mu_total=mub+mu_perturbation, ...)`, which
    triggers `sync_total_legacy_perturbation("mu_total","mu",...)` →
    legacy `mu` is set IDENTICAL to `mu_total`.
  - restart: `src/gpuwrf/io/wrfrst_netcdf.py:632-636` reads `mu_total` as an
    exact stored leaf; writer persists it via
    `_base_pair(state.mu_total, state.mu_perturbation)` at line 378. Roundtrip
    is exact and nonzero.
  - daily/nested pipelines re-sync legacy from total:
    `daily_pipeline.py:384`, `nested_pipeline.py:430`
    (`mu=...mu_total`, `ph=...ph_total`).
- The dycore reads `state.mu_total`/`ph_total` everywhere as authoritative
  (`rk_addtend_dry.py`, `flux_advection.py`, `acoustic_wrf.py`,
  `small_step_prep.py`). A real state with zero `mu_total` would produce
  zero/NaN dynamics from step 1 — it cannot reach radiation looking "healthy",
  so the zero-total/valid-legacy combination is structurally a fixture-only
  artifact.

**Ruling:** No legitimate real init/restart/scheme path produces
`mu_total==0` with a valid legacy `mu`. The blocker was a fixture-shape bug.

**Why the production fallback is nonetheless acceptable (not masking):** the
helper is gated on the *global* condition `max(|total|)==0 AND max(|legacy|)>0`
and is a strict identity whenever total carries any signal (proven below). It
therefore (a) cannot perturb any real state, and (b) cannot "repair" a partially
corrupted real state — a real state with a single zero `mu_total` cell keeps its
real (zero) value and would still surface a genuine bug. The only behaviour it
changes is the all-zero-total / valid-legacy case, which is exactly the
fixture shape. So it tolerates the transitional alias inconsistency without
hiding any plausible real init/restart defect.

**Recommended (NON-BLOCKING) follow-up, not a must-fix:** the cleaner long-term
fix is to make the synthetic fixtures build through `State.replace`/an init
helper so totals stay populated, OR add an `assert max|mu_total|>0` in the real
operational state constructor so a future genuine zero-mass init fails loudly
instead of silently using the legacy alias. The current fallback is safe to ship
as-is; this is hygiene, not a correctness gap.

## 2. Physical faithfulness — IDENTICAL, not merely non-NaN

Legacy `mu` and `mu_total` are the SAME physical quantity ("Pa column dry mass
on mass points", `state.py:443`), kept byte-equal by
`sync_total_legacy_perturbation` (`state.py:824/826`). Same for `ph`/`ph_total`
("total geopotential", `state.py:440`). The fallback substitutes one alias of an
identical quantity for another — it is NOT a dry-vs-moist swap and cannot
introduce a wrong rho/pressure. Where the fallback fires (fixtures),
legacy carries the intended column mass / geopotential, so the reconstructed
`p_hyd`, `psfc`, and `rho` are the physically intended values.

## 3. NO-OP for real states — PROVEN

`_total_or_legacy_field` (`physics_couplers.py:1145-1154`):
`use_legacy = (max|total|==0) & (max|legacy|>0)`; `jnp.where(use_legacy, legacy, total)`.
`jnp.where` with a scalar False returns `total` unchanged. Direct unit proof
(CPU, fp64), with legacy deliberately set to a WRONG value to force a mismatch:

- A  total-valid → returns TOTAL byte-for-byte, NOT legacy: **True / True**
- A2 total valid but with one zero cell → kept verbatim incl. the zero cell
  (global gate, no per-cell mixing; out[0,0]=0.0): **True**
- B  total all-zero, legacy valid → falls back to legacy: **True**
- C  both zero → stays zero, no fabrication: **True**
- D  legacy field absent (None) → returns total: **True**

Both callers (`_wrf_hydrostatic_pressure_from_state:1161`,
`_wrf_phy_prep_rho_from_state:1196-1197`) route both `mu`/`ph` through the helper;
on any real state (valid total) they receive `mu_total`/`ph_total` exactly as
before the patch. Therefore the validated 72h identity greens are byte-for-byte
unchanged. The helper is the only behavioural change to these prep functions;
the rest of the line is identical to pre-patch.

These prep functions feed surface/PBL/NoahMP prep
(`noahmp_surface_hook.py:116-117`, `physics_couplers.py:1618-1619,1986-1987`) —
the exact path that NaN'd before radiation — and run in real operational runs as
a no-op.

## 4. Correct, not just non-NaN — CONFIRMED

`proofs/v018/radfix_after_scope_cpu.log`: every active SW/LW combo finite,
`nan_count=0/72`, `rthraten` absmax in `9.4e-05 … 4.6e-04` K/s (≈ 0.3–1.7 K/h
radiative heating — physically sane). Pure-dynamics (ra_sw=0, ra_lw=0) correctly
exactly zero. Before: every active combo all-NaN `72/72`
(`radfix_before_scope_clean_trunk.log`).

## 5. No #37 regression — CONFIRMED

mp=8 default allocation gate holds: `active_fields=60`, `tree_leaves=60`, every
conditional leaf (`qh/Nh/qvolg/qvolh/nwfa/nifa/hail_acc`) absent
(`radfix_after_default_path_allocation_gate.log`). The fix adds NO conditional
State leaves and does not touch `state.py` allocation logic (the large `state.py`
delta in this branch is the pre-existing set-UNION/67-leaf refactor from
`8faf93f6`/`7bb30275`, not this commit). Conditional-state suite green (below).

## 6. Re-run gates (this critic)

CPU (taskset 0-3, OMP=4, JAX_PLATFORMS=cpu):
- `test_rrtm_lw_operational_wiring` + `test_cdudhia_sw_operational_wiring` +
  `test_v014_mynn_surface_layer_regressions`: **22 passed**.
- `test_v018_conditional_state_leaves` + `test_v017_apply_physics_non_dry_conditional`
  + `test_v017_qh_hail_state`: **12 passed, 1 skipped** (GPU-only State.zeros).

GPU (under `scripts/with_gpu_lock.sh --label opus-radfix-critic`, lock acquired
and released cleanly; stale holder file ignored, no compute job was holding it):
- wiring (rrtm + cdudhia) + `test_v018_conditional_state_leaves`: **20 passed**.

Parent-bisection claim verified: `radfix_parent_8faf_cpu_operational_wiring.log`
shows `8faf93f6` (parent of suspect `7bb30275`) already fails `5 failed, 13 passed`
with all-NaN `rthraten` — so the #37 conditional-leaf refactor is NOT the cause,
as claimed.

## Findings summary

- Root cause diagnosis: CORRECT (zero `mu_total`/`ph_total` → degenerate
  `p_down==p_up` → `0/(p*log(1))` NaN at `physics_couplers.py:1209`, propagating
  to surface/PBL before radiation).
- No masking, no clamp, no tolerance widening, no NaN-swallow.
- No-op on real states: PROVEN (unit + caller analysis + alias-sync proof).
- Physically identical fallback (legacy == total by construction): PROVEN.
- Trigger was a malformed synthetic fixture, but the production fallback is safe
  (identity on real, non-fabricating on invalid). Optional hygiene follow-up:
  build fixtures via `State.replace`/assert nonzero `mu_total` at real init.

Ship it.
