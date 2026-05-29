# Sprint Contract — F7 Sprint C: rk_addtend_dry + large-step PGF cadence → idealized cases PASS (dycore close)

**Sprint ID**: `2026-05-29-f7-sprint-c`
**Frontrunner**: Opus 4.8 (in-process Agent subagent, high/max effort)
**Branch**: `worker/opus/f7-sprint-c` (work in the main tree on this branch; commit incrementally)
**GPU**: YES — every python/pytest under `taskset -c 0-3`; confirm `cuda:0` first; fp64.
**Builds on**: Sprint A (acoustic core) + Sprint B (damping, advection, 5 bug fixes) — both merged, `manager-2026-05-23` tip `c721604`.

## Project endpoint (the bar)

A real WRF v4 GPU port that runs real/published test cases with near-identical results / RMSE on all values, **no shortcuts**, GPU-efficient, massive speedup on this RTX 5090. This sprint targets the first **physics-truth PASS**: the Straka density current and Skamarock warm bubble matching published reference solutions. That is the F7 dynamical-core close.

## The single localized gap (from Sprint B's verified diagnosis)

Sprint B got the bubble to **rise buoyantly** but θ′ is **not transported** because the flow does not **circulate**: the WRF **large-step dry-tendency merge `rk_addtend_dry` + horizontal-PGF cadence** is missing. Sprint B's verified WRF findings (treat as established, but re-verify against source):
- WRF puts the **horizontal pressure-gradient force in the large-step `ru/rv_tend`** (`module_em.F:1325`) — this **corrects Sprint A's "PGF double-count" note** (Sprint A wrongly removed a large-step PGF term that WRF keeps). The large-step PGF and the small-step `advance_uv` acoustic-PGF perturbation are **different parts of the split-explicit scheme, not a double-count**.
- A Sprint-B prototype produced the right u-tendency (~0.6 m/s²) but it did not move u in the integrated state, because the operational tendency cadence (`add_scaled_tendencies` + `advance_uv` + `small_step_finish`) does not net correctly **without `rk_addtend_dry`**.

## Cardinal rule

**WRF Fortran source is ground truth.** Verify the PGF split and the `rk_addtend_dry` coupling against `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/` (`module_em.F`, `solve_em.F`, `module_small_step_em.F`, `module_big_step_utilities_em.F`). If anything here disagrees with WRF, WRF wins — note it.

## Read first
1. This contract + Sprint B worker report `.agent/sprints/2026-05-29-f7-sprint-b/worker-report.md` + Sprint B proofs `proofs/f7b/` (esp. the idealized verdicts + `damping_dt_sweep.json` + `audit_summary.md`).
2. Cadence spec items 3+4: `proofs/f5/wrf_cadence_spec.md`.
3. WRF: `rk_tendency` (large-step dry dynamics incl. PGF, `module_em.F:855-1388`, PGF at `:1325`), `rk_addtend_dry` (`module_em.F:1711-1782`), the RK-loop add/boundary cadence in `solve_em.F:1837-2210`.
4. Current code: `src/gpuwrf/runtime/operational_mode.py` (`_rk_scan_step`, `add_scaled_tendencies`, the tendency cadence), `src/gpuwrf/dynamics/flux_advection.py`, `src/gpuwrf/dynamics/core/advance_w.py` (`pg_buoy_w`), `src/gpuwrf/dynamics/core/acoustic.py` (`advance_uv_wrf` small-step PGF).

## Scope

1. **Implement `rk_addtend_dry`** (`module_em.F:1711-1782`): per-RK-stage merge of (RK1-fixed physics tendencies — zero when physics-off) + per-stage dry-dynamics tendencies into `ru/rv/rw/t/ph/mu` tendencies, with **field-specific map-factor and mass coupling** (u: `muu`/msf; v: `muv`/msf; w/θ/ph: `mut`/msf; the exact factors per WRF). NOT a generic `add_scaled_tendencies`.
2. **Restore + reconcile the large-step horizontal PGF** in the per-stage dry tendency (`rk_tendency` path, `module_em.F:1325`), and ensure it nets correctly with the small-step `advance_uv` acoustic PGF — verify against WRF that there is no actual double-count (the two are distinct split terms). Fix the operational tendency cadence so the large-step momentum tendency actually moves the integrated `u/v` (so the flow circulates and transports θ′).
3. **Wire RK stage descriptors** (`dt_rk`, `dts_rk`, `number_of_small_timesteps` = 1, n/2, n) into the merge if not already correct.
4. If a residual growing mode still escapes the 12-step operational-dt audit after the cadence is correct, wire the WRF Smagorinsky / constant-K horizontal mixing (`km_opt`, the `explicit_diffusion.py` const-K path already exists) at WRF coefficients — but FIRST check whether correct circulation alone removes the instability (the Sprint-B step-6-7 growth may be an artifact of the broken cadence).
5. **Investigate the 1 remaining red test** `test_step2_operational_theta_stays_finite` (Sprint B: "a separate first-substep defect on the legacy non-prep path on real d02 IC"). Determine if it is the same cadence bug; fix if so, otherwise document precisely. No tolerance widened / no xfail.

## Acceptance gates (all required for `F7C_COMPLETE` = F7 dycore close)

- **AC1 — Straka density current PASS.** `run_density_current_case(require_gpu=True)` → `RAN_TO_COMPLETION`, `verdict=PASS`: finite to 900 s; `front_position_900s` within ~2 km of the ~15 km reference; `theta_prime_min_900s` ≈ −9 to −10 K order; `max_abs_w_900s` ~ O(10) m/s; rotor proxy in range; `relative_mass_drift` ≤ 1e-8. (Straka uses ν=75 m²/s, already in `explicit_diffusion.py`.)
- **AC2 — Skamarock warm bubble PASS.** `run_warm_bubble_case(require_gpu=True)` → `RAN_TO_COMPLETION`, `verdict=PASS`: θ′ **transported** (positive-θ centroid rises ≥ 500 m), θ′max in range, max |w| in range (≤30 m/s), symmetric, mass-conserving.
- **AC3 — 12-step operational-dt audit clean.** `taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 12` at dt=6 s, WRF damping ON: `first_critical_violation == null` for all a/b/c/d, physical transient magnitudes, no masking clamp.
- **AC4 — no regression.** Sprint A/B gates still hold (no-stub, flat-rest=0, analytic oracle, 300-step conservation, advection-order proof). Re-run.
- **AC5 — the last red test resolved or precisely documented** (no weakening).

## Proof objects (into `proofs/f7c/`)
`straka_density_current.json`+verdict+plots; `skamarock_warm_bubble.json`+verdict+plots; `audit_operational_dt.json`+`audit_summary.md`; `rk_addtend_dry_proof.md` (WRF cadence verification + the PGF-split reconciliation, showing u/v now move correctly); `regression_recheck.json`; `worker-report.md` (AGENTS.md format) ending `F7C_COMPLETE` or `F7C_PARTIAL` + precise gaps.

## Hard rules
1. `taskset -c 0-3`; `cuda:0`; fp64.
2. WRF source is ground truth; cite `file:line` in every new/changed operator docstring.
3. **No masking clamps/caps/sanitizers.** WRF damping/diffusion (named, with WRF coefficients) is allowed; ad-hoc clamps to force a gate green are not.
4. **No performance work** (no fp32/fusion) — correctness only; perf is F7-perf.
5. Commit incrementally on `worker/opus/f7-sprint-c`; do not push; do not switch branches.
6. Files writable: `src/gpuwrf/dynamics/**`, `src/gpuwrf/runtime/operational_mode.py`/`operational_state.py`, `src/gpuwrf/ic_generators/idealized.py` (minimal harness fixes, documented), `scripts/**` (instrumentation, never weaken invariants), `tests/**` (fix per AC5, add tests, never weaken others), `proofs/f7c/**`, this sprint folder.
7. Files NOT writable: governance, memory, skills, ADRs, plan, physics-scheme code, comparator scripts under `scripts/m6b6_*`.
8. If full scope can't land, deliver the largest gated subset and mark `F7C_PARTIAL` with precise gaps. If AC1+AC2 pass but AC3 has a residual instability, that is still a strong result — report it honestly.

## Forward pointer
- On AC1+AC2 PASS → manager runs the **GPT-5.5 WRF-domain pre-close code critique** of the whole dycore, then declares the F7 dynamical-core milestone closed.
- Then: **F7-perf** (XLA fusion + fp32 downcast + speedup recert), and **M9** (instrumented WRF savepoints → per-operator parity, the rigorous near-identical-RMSE-vs-real-WRF gate).
