# Positive-definite / monotonic scalar transport — proof object

Sprint: opt-in positive-definite (PD) and monotonic flux limiters on top of the
WRF flux-form h5/v3 scalar advection.

Branch: `worker/opus/pd-monotonic-advection` off integration HEAD `e602122`.

## What was implemented

`src/gpuwrf/dynamics/flux_advection.py` — new function `advect_scalar_flux_limited`
plus four private helpers (`_flux_upwind_face`, `_flux_upwind_face_z`,
`_high_order_flux_{x,y,z}`, `_neighbour_min_max`).  Returns the COUPLED tendency
`d(mu*phi)/dt`, the SAME return contract as the plain `advect_scalar_flux`, so a
caller selects the limiter purely by the option value.

WRF source (pristine v4 `dyn_em/module_advect_em.F`):
* `advect_scalar_pd` (`:6069-7885`) — Smolarkiewicz MWR-1989 FCT, option **1**
  (positive-definite).
* `advect_scalar_mono` (`:9495-10560`) — Zalesak-style inflow/outflow
  renormalization, option **2** (monotonic).

NOTE on option numbering: WRF canonical `moist_adv_opt`/`scalar_adv_opt` is
`0=none, 1=PD, 2=monotonic, 3=WENO, 4=WENO-PD`.  The sprint brief listed
"2=monotonic / 3=PD"; this implementation uses the WRF-canonical values
(1=PD, 2=mono) and documents the mapping in the module header.  WENO (3/4) is
OUT OF SCOPE (a separate ~1000-LOC reconstruction).

Scope restriction (same as the plain path): PERIODIC x/y, unit-map, h=5/v=3.
The default path (`moist_adv_opt=0/1`, plain `advect_scalar_flux`,
`use_flux_advection`) is BYTE-IDENTICAL: the change is purely additive (379
insertions, 0 deletions; `git diff --stat e602122 -- flux_advection.py`).

## FCT algorithm (faithful to WRF)

1. Low-order monotone (donor-cell, CFL-clamped) fluxes `fqxl/fqyl/fqzl` from
   `field_old`: `fqxl = mu_face*(dx/dt)*flux_upwind(q_im1, q_i, cr)`,
   `cr = vel*dt/dx/mu_face`.
2. Antidiffusive fluxes `A = high_order - low_order` (high-order = the plain-path
   `ru*flux5(...)` / `v3 vflux`).
3. Low-order updated coupled value `ph_low` (PD) / `ph_upwind` (mono) from
   `field_old`, `mu_old`, `dt`.
4. Per-cell scale — PD: one factor on the OUTGOING antidiffusive flux so the cell
   cannot empty below zero; mono: separate inflow/outflow factors bounding the
   cell to its `field_old` 6-neighbour min/max — combined at each FACE by the WRF
   donor / `min(scale_in, scale_out)` rule (verified sign-by-sign against the
   Fortran, including the z-flux mass-coordinate sign inversion).
5. Tendency `-div(A_limited + low_order)`.

## Validation (all CPU-jax: `JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3`)

`tests/dynamics/test_pd_monotonic_advection.py` — 14 tests, all PASS:

| test | property |
|---|---|
| `test_pd_keeps_blob_nonnegative` | (a) PD: positive blob stays >= 0 (strict, 2-D divergence-free) |
| `test_plain_path_undershoots_blob` | sanity: plain h5/v3 DOES undershoot (< -1e-6) — non-trivial |
| `test_mono_introduces_no_new_extrema` | (b) mono: no new extrema (strict, 2-D divergence-free) |
| `test_plain_path_overshoots_blob` | sanity: plain h5/v3 DOES overshoot — non-trivial |
| `test_limiter_conserves_total_mass[1]` / `[2]` | (c) mass conserved to < 1e-12 (both limiters) |
| `test_pd_3d_divergence_free_*` | 3-D PD: exact mass conservation + essentially positive |
| `test_mono_3d_divergence_free_*` | 3-D mono: exact mass conservation + essentially bounded |
| `test_smooth_field_limiter_inactive_matches_plain` | (d) no-bind: smooth field -> limited == plain (rel < 1e-10) |
| `test_jax_pd_x_matches_wrf_transcription` | parity: x-PD vs WRF Fortran transcription (< 1e-13) |
| `test_jax_pd_z_matches_wrf_transcription` | parity: z-PD vs WRF transcription (< 1e-13) |
| `test_jax_pd_3d_matches_wrf_transcription` | parity: full 3-D PD vs WRF transcription (< 1e-12) |
| `test_jax_mono_3d_matches_wrf_transcription` | parity: full 3-D mono vs WRF transcription (< 1e-12) |
| `test_rejects_unsupported_option` | opt 0/3/4 raise (limiter is PD/mono only) |

Key faithfulness evidence: an INDEPENDENT direct transcription of the WRF Fortran
(donor-cell + flux5/flux3 + ph_low + flux_out + the exact conditional scaling) is
embedded in the test file; the JAX tendency matches it to ROUND-OFF in x, z and a
full 3-D divergence-free flow.

## Honest finding on genuine 3-D positivity (WRF-faithful, not a bug)

WRF's `advect_scalar_pd` is documented in its own source as "a first cut at a
positive definite advection option".  In strictly 1-D / 2-D directionally-split
transport it is exactly positive-definite (the strict gates above pass at 1e-12).
In GENUINE multi-dimensional flow the per-cell single-scale renormalization
admits tiny O(antidiffusive) excursions: on a sharp blob in a divergence-free x-z
flow, PD bottoms at ~-2.0e-3 and mono ranges over [-2.0e-3, +1.016].  The
INDEPENDENT WRF Fortran transcription produces the SAME residual to round-off
(PD min -0.00201; mono min -0.00201 / max 1.01586 — bit-identical to the JAX
max 1.0158649654088823).  So the JAX limiter is faithful to WRF, and the small
3-D excursion is WRF's own behaviour, not a port defect.  The 3-D tests assert
EXACT mass conservation plus this WRF-faithful small-bound behaviour (tol 2e-2).

## Default-path no-regression (the absolute guardrail)

* `git diff --stat e602122 -- flux_advection.py` => 379 insertions, 0 deletions
  (the default `advect_scalar_flux` and the momentum path are byte-identical).
* Idealized dycore gates re-run on the default plain h5/v3 path (CPU):
  * Straka density current: `verdict=PASS status=RAN_TO_COMPLETION checks=6/6`.
  * Skamarock warm bubble:  `verdict=PASS status=RAN_TO_COMPLETION checks=6/6`.
* `tests/dynamics/test_flux_advection_map_factors.py`,
  `tests/dynamics/test_advect_w_topface.py`: PASS (existing flux-form tests).

## Operational RK3 wiring (follow-on sprint `worker/opus/pd-rk3-wiring`)

The limiter is now WIRED into the operational RK3 scalar-advection path and the
catalog flipped.  Files changed:

* `src/gpuwrf/runtime/operational_mode.py`
  * `OperationalNamelist.scalar_adv_opt: int = 0` (canonical: 0=plain, 1=PD,
    2=monotonic) threaded through `from_grid` + the pytree `tree_flatten`/
    `tree_unflatten` aux (so a distinct option is a distinct jit cache key).
  * `_augment_large_step_tendencies(..., step_origin=None)` — the start-of-step
    haloed state (WRF `_1` / `scalar_old` / `mu_old`).  The theta scalar path
    selects `advect_scalar_flux_limited` when `scalar_adv_opt in {1,2}` AND
    `rk_step == rk_order` (WRF applies the limiter on the FINAL RK3 stage only,
    `module_em.F:1265`), passing `field_old = step_origin.theta-300`,
    `mu_old = step_origin.mu_total`, `mut = current mu`, `dt = full step`
    (`dt_step*(rk_order-rk_step+1)` = full dt on the final stage).  Otherwise the
    plain `advect_scalar_flux` runs.  The branch is a STATIC Python `if` on
    compile-time constants, so `scalar_adv_opt=0` emits the identical XLA program.
  * `_rk_scan_step` passes `step_origin=rk1_reference` (the start-of-step `u_1`
    reference) into the augment.
* `src/gpuwrf/io/scheme_catalog.py` — `moist_adv_opt` / `scalar_adv_opt` wired
  set flipped `{0,1}` → `{0,1,2}` (1=PD, 2=monotonic now IMPLEMENTED; 3=WENO,
  4=WENO-PD stay RECOGNIZED_FAIL_CLOSED).  `_ADV_OPT_REASON` corrected to the WRF
  canonical numbering.

### Operational-path validation (CPU-jax)

`tests/dynamics/test_pd_rk3_operational.py` — 7 tests, all PASS:

| test | property |
|---|---|
| `test_operational_default_path_byte_identical_to_plain` | (GUARDRAIL) opt=0 augment theta-tend == plain `advect_scalar_flux` (`assert_array_equal`) |
| `test_operational_limiter_inactive_on_non_final_rk_stages` | opt=1/2 on RK stages 1 & 2 are byte-identical to plain (rk3-only) |
| `test_operational_pd_keeps_scalar_nonnegative` | (a) opt=1 keeps a positive scalar >= 0 over 30 steps; plain undershoots < -1e-6 |
| `test_operational_mono_introduces_no_new_extrema` | (b) opt=2 introduces no new extrema; plain over/undershoots |
| `test_operational_tendency_conserves_coupled_mass[0,1,2]` | (c) coupled-mass tendency telescopes to < 1e-12 for opt 0/1/2 |

Proof JSON (`proofs/pd_monotonic/operational_pd_rk3_proof.py` →
`operational_pd_rk3_proof.json`), `verdict=PASS`: PD min `-1.6e-14`, mono in
`[0.0, 0.999]` (plain `[-0.216, 1.211]`), conservation rel-total-tend `~1e-18`,
default byte-identical + rk1/rk2 inactive both True, full `_rk_scan_step` finite
for opt 0/1/2, catalog {0,1,2}=implemented / {3,4}=fail-closed.

### Default-path no-regression (the absolute guardrail, on the WIRING branch)

`proofs/pd_monotonic/operational_idealized_no_regression.json` — the idealized
dycore gates re-run on the DEFAULT path (`scalar_adv_opt=0`, CPU):
* Straka density current: `verdict=PASS status=RAN_TO_COMPLETION` (6/6;
  mass-drift 2.25e-9, front 14150 m, max|w| 14.6).
* Skamarock warm bubble:  `verdict=PASS status=RAN_TO_COMPLETION` (6/6;
  thermal-rise 1924 m, mass-drift 0.0).

`tests/dynamics/` full suite: 45 passed.  `tests/test_namelist_recognition_breadth.py`
(updated for the {0,1,2} flip): 10 passed.  `tests/test_cli.py`: 16 passed.

## Not done / remaining (honest)

* WENO (opt 3/4) intentionally skipped (fail-closed; ~1000-LOC reconstruction).
* GPU + real-case validation deferred to v0.13 (CPU-jax parity + idealized only).
* The limiter is wired for THETA (the operational dry-dynamics scalar).  Moisture
  species (qv/qc/...) ride the physics path, not this dry flux-form theta path;
  extending the operational limiter to moisture transport is a separate step.
