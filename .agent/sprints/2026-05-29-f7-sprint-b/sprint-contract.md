# Sprint Contract — F7 Sprint B: damping + large-step dry tendency → idealized-case truth

**Sprint ID**: `2026-05-29-f7-sprint-b`
**Frontrunner**: Opus 4.8 (in-process Agent subagent, high/max effort)
**Branch**: `worker/opus/f7-sprint-b` (work in the main tree on this branch; commit incrementally)
**GPU**: YES — every python/pytest under `taskset -c 0-3`; confirm `cuda:0` first. Keep `jax_enable_x64=True`.
**Builds on**: Sprint A (merged, `manager-2026-05-23` tip `9ae962c`) — the acoustic small-step core is now WRF-cadence-faithful, conservative, and stub-free, but **undamped**: at operational dt the acoustic/gravity modes ring up to u/v ~5e5, w ~1.5e4 m/s (mass still conserved, bounded, no critical violation in 12 steps). WRF suppresses this with explicit damping that Sprint A deliberately disabled to isolate the core.

## Project endpoint (the bar)

A real WRF v4 GPU port that runs real WRF/published test cases with near-identical results / RMSE on all values, **no shortcuts** (no masking clamps, no JAX-vs-JAX self-compares, no synthetic happy-paths), GPU-efficient, massive speedup on this RTX 5090. This sprint produces the first **physics-truth** evidence: idealized cases with published reference solutions.

## Cardinal rule

**WRF Fortran source is ground truth, not this contract.** Read the cited WRF source and verify equations/signs/coupling/orders yourself before implementing. WRF source: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/` (`module_advect_em.F`, `module_em.F`, `module_small_step_em.F`, `module_big_step_utilities_em.F`, `solve_em.F`). If contract and WRF disagree, WRF wins — note it.

## Read first
1. This contract + Sprint A worker report `.agent/sprints/2026-05-29-f7-acoustic-core/worker-report.md` + Sprint A proofs `proofs/f7a2/`.
2. The cadence spec `proofs/f5/wrf_cadence_spec.md` items **3** (rk_tendency dry dynamics / flux-form advection) and **4** (rk_addtend_dry).
3. The F7-MEGA critique `.agent/sprints/2026-05-29-f7-mega-dry-dycore-rewrite/critique.md` — it already covers Sprint B WRF facts: advection order is **config-driven** (freeze to h=5/v=3 as a *scoped restriction*, not a WRF fact); `rk_addtend_dry` is **field-specific** map/mass-coupled, not a generic `add_scaled_tendencies`; theta scalar advection is not optional for a dry dycore benchmark.
4. The idealized harness `src/gpuwrf/ic_generators/idealized.py` (`run_density_current_case`, `run_warm_bubble_case`) + `proofs/f2/straka_density_current_verdict.md`, `proofs/f2/skamarock_bubble_verdict.md` (references + the metrics the harness already checks: `front_position_900s`, `theta_prime_min_900s`, `max_abs_w_900s`, `relative_mass_drift`). These were built in F2 but never ran (no GPU in the codex sandbox) — you have a GPU.
5. Current `src/gpuwrf/dynamics/advection.py` (periodic `jnp.roll` upwind to replace), `src/gpuwrf/runtime/operational_mode.py` `_rk_scan_step`.

## Scope — two coupled blocks, committed incrementally

### Block 1 — Damping (so the core runs at WRF operational dt)
- Re-enable WRF **`w_damping`** (vertical-velocity damping) and **Rayleigh damping** at the model top (`dampcoef`/`zdamp`), per WRF (`module_em.F` / `solve_em.F` damping calls). Verify the **divergence damping** already wired in Sprint A (`smdiv` pressure-memory in `calc_p_rho`, `emdiv`/`mudf` in `advance_uv`) is active and WRF-correct.
- Add the **explicit second-order diffusion** path needed for the Straka benchmark (the Straka et al. 1993 reference solution is defined *with* constant viscosity ν = 75 m²/s on u, v, θ). This is part of the test definition, not a masking clamp.
- After Block 1: re-run the 12-step audit at the **operational** config (dt=6 s, WRF damping ON) and a dt sweep — the acoustic ringing (u/v, w) must drop to **physical** magnitudes (O(1–50) m/s), `first_critical_violation` still null, and the under-damped finiteness test (`test_step2_operational_theta_stays_finite`) should now pass.

### Block 2 — Large-step dry tendency (so perturbations propagate physically)
- Replace periodic `jnp.roll` advection with **WRF flux-form mass-coupled advection**, frozen at **h_sca_adv_order=5 / v_sca_adv_order=3** (state this as a scoped restriction; WRF selects order from config). Mass/map-factor coupling per WRF `module_advect_em.F`. Cover momentum (u,v,w) and theta scalar advection.
- Implement **`rhs_ph`** as the source of `ph_tend` before `advance_w` (`module_em.F:1224-1266`).
- Confirm **`pg_buoy_w`** large-step buoyancy source (partly from Sprint A) matches WRF.
- Implement **`rk_addtend_dry`**: per-RK-stage merge of (RK1-fixed physics tendencies — zero when physics-off) + per-stage dry-dynamics tendencies into ru/rv/rw/t/ph/mu with **field-specific** map/mass coupling (`module_em.F:1711-1782`). Wire RK stage descriptors (`dt_rk`, `dts_rk`, `number_of_small_timesteps` = 1, n/2, n) if not already correct from F7.A.

## Acceptance gates (all required for `F7B_COMPLETE`)

- **AC1 — Straka density current, RAN_TO_COMPLETION near reference.** `run_density_current_case(require_gpu=True)` → `status=RAN_TO_COMPLETION`, `verdict=PASS`, with `front_position_900s`, `theta_prime_min_900s`, `max_abs_w_900s` within a *documented* tolerance of Straka et al. (1993) (front ≈ 14–17 km at 900 s with ν=75; min θ′ ≈ −9 to −10 K order; max |w| ~ O(10) m/s) and `relative_mass_drift` ≤ 1e-3. Cite the reference in the proof.
- **AC2 — Skamarock warm bubble, RAN_TO_COMPLETION, plausible.** `run_warm_bubble_case(require_gpu=True)` → `RAN_TO_COMPLETION` with a symmetric, bounded rising thermal (no NaN, no runaway), max |w| physical.
- **AC3 — 12-step audit clean at operational dt with damping ON.** `taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 12` at dt=6 s, WRF damping ON: `first_critical_violation == null` for all a/b/c/d AND u/v / w transient deltas are physical (O(≤100) m/s, not 1e5). No clamp/limiter masking.
- **AC4 — no regression.** Sprint A's gates still hold (no-stub audit, flat-rest=0, analytic oracle, 300-step conservation). Re-run them.
- **AC5 — the 3 red tests resolved honestly.** Fix, don't just delete: (a) `test_ph_tend_matches_validation_bound_theta_delta_formula` — update to assert the *new* WRF `advance_w` ph behavior (the old 0.01·Δθ stub is gone by design; document the change as INV-6-compliant since the asserted code was a deleted stub); (b) `test_step2_operational_theta_stays_finite` — should pass once damping is on; (c) `test_mu_persistence_two_substeps` — replace the unphysical zero-geopotential fixture with a hydrostatically-balanced one. No tolerance widened, no `xfail` added.

## Proof objects (write into `proofs/f7b/`)
`straka_density_current.json` + verdict.md + plots; `skamarock_warm_bubble.json` + verdict.md + plots; `audit_operational_dt.json` + `audit_summary.md` (Block-1 12-step at dt=6 ON); `damping_dt_sweep.json`; `advection_order_proof.md` (WS5/3 verification + a 1-D linear-advection convergence/shape check vs analytic); `regression_recheck.json` (Sprint A gates re-run); `worker-report.md` (AGENTS.md handoff format) ending `F7B_COMPLETE` or `F7B_PARTIAL` + precise gaps.

## Hard rules
1. `taskset -c 0-3`; confirm `cuda:0`; fp64.
2. WRF source is ground truth; cite `file:line` in every new/changed operator docstring.
3. **No masking clamps/caps/sanitizers.** WRF's `w_damping`/Rayleigh/divergence/explicit-diffusion ARE physics (and part of the Straka definition) — those are allowed and required; an ad-hoc clamp to force a gate green is not. State which damping you enabled and its WRF coefficient source.
4. **No performance work** (no fp32, no fusion-for-speed) — correctness first; perf is F7-perf.
5. Commit incrementally on `worker/opus/f7-sprint-b`; do not push to any remote; do not switch branches.
6. Files writable: `src/gpuwrf/dynamics/**`, `src/gpuwrf/runtime/operational_mode.py`/`operational_state.py`, `src/gpuwrf/ic_generators/idealized.py` (fix harness bugs minimally + document), `scripts/**` (idealized runners / audit instrumentation, never weaken invariants), `tests/**` (fix the 3 red tests per AC5; add tests; never weaken others), `proofs/f7b/**`, this sprint folder.
7. Files NOT writable: governance, memory, skills, ADRs, plan, physics-scheme code (microphysics/PBL/radiation/land), comparator scripts under `scripts/m6b6_*`.
8. If full scope can't land cleanly, deliver the largest gated subset (at minimum Block 1 + AC3 + AC4, i.e. a damped stable core at operational dt) and mark `F7B_PARTIAL` with precise gaps. Honest partial > green self-compare.

## Forward pointer (NOT this sprint)
- **F7 dycore close + GPT-5.5 pre-close critique**: once AC1/AC2 pass, the manager runs a GPT-5.5 WRF-domain code review of the whole dycore before declaring the dycore milestone done.
- **M9**: instrumented WRF Fortran savepoints → per-operator WRF↔JAX parity (the rigorous near-identical-RMSE-vs-real-WRF gate).
- **Physics RK1-bundle cadence, real lateral BC, moisture/scalar skill**: Phase B.
