# Sprint Contract — F7H: fix the geopotential (ph) restoring coupling + settle buoyancy against WRF ground truth → idealized cases PASS

**Sprint ID**: `2026-05-29-f7h-geopotential-carry-and-wrf-truth`
**Frontrunner**: Opus 4.8 (in-process Agent subagent, high/max effort)
**Branch**: `worker/opus/f7d-pressure-mass-fix` (CONTINUE — tip `4ce8f07` carries F7D mass fix, F7F IC-rebalance + calc_p_rho_phi geopotential-term fix, F7G WIP signed-metric+staging). The manager merges the chain once the idealized cases PASS.
**GPU**: YES — `taskset -c 0-3`; `PYTHONPATH=src`; `cuda:0`; fp64.

## Why this sprint (the reframe)

Six sprints + two GPT-5.5 council rounds each fixed a real bug but did NOT pass the idealized cases. The chase has fixated on "`pg_buoy_w(grid%p)/analytic ≈ 9.4–19×`" as *the* bug. **That framing is probably a red herring**: in WRF's mass coordinate `pg_buoy_w = g·(rdn·Δp' − c1f·mu')` for a warm bubble legitimately differs from the naive parcel estimate `g·θ'/θ0 = 0.0654` — so 0.62 may be roughly the *correct buoyancy forcing*, not a 9.4× error. AC1 already proved the IC is discretely consistent with `calc_p_rho_phi` (round-trips to 1e-12, **both** sign conventions — so it is NOT a sign or IC-balance bug).

**The leading hypothesis (from the prior frontrunner, captured at hand-off):** the geopotential perturbation is **frozen** — `ph_perturbation = ph_work + ph_save` stays at **131.83** while `w` grows linearly, i.e. the acoustic-scan geopotential WORK array `ph_work` (advance_w's `ph_next`) is **≈0 and not responding to `w`**. If true, the vertical w/ph coupling is broken: buoyancy forces `w`, but the geopotential never rises, so the perturbation pressure never builds the **restoring** gradient → `w` grows without bound → runaway. This is a plumbing/carry bug (work-array not accumulating through `lax.scan`), a *different class* than the numerical-formulation fixes already tried.

## Cardinal rule
WRF Fortran source is ground truth: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/`. Verify against `module_small_step_em.F` advance_w geopotential finish (`:1581-1586`, the `ph` update from solved `w`), the small-step `ph_2`/`ph_save`/`ph_1` work lifetimes (`:125-190`, `:259-277`), and `solve_em.F` acoustic loop. If WRF disagrees with this contract, WRF wins — note it.

## Scope — EMPIRICAL, in this order. Instrument and TEST; do not reason open-endedly.

### Phase 1 — confirm or refute the ph-carry hypothesis (FAST, do first)
1. Instrument a warm-bubble run: trace, per acoustic substep AND per step, `max|w|`, `max|ph_work|`, `max|ph_perturbation|`, `max|p_perturbation|`. Write `proofs/f7h/ph_carry_trace.json`.
2. Decide: does `ph_work`/`ph_perturbation` **grow as `w` grows** (restoring develops) or stay **frozen** (broken coupling)? This is a yes/no with the trace.
3. If frozen: find where the acoustic-scan carry drops the geopotential update. Check: does `acoustic_substep_core` return the updated `ph` (advance_w `ph_next`) into the scan carry? Does `_acoustic_core_state_from_prep` seed `ph`/`ph_work` correctly (WRF RK1 `ph_2 = ph_1 − ph_2 = 0`, evolves, finish adds `ph_save`)? Is `ph_save` correct? Trace the value through one full RK stage. Fix the carry so `ph` (hence `ph_perturbation`) evolves with `w`. Cite WRF `:1581-1586`.

### Phase 2 — settle the buoyancy magnitude against WRF ground truth (do if Phase 1 doesn't fully close, OR to confirm)
4. Stop arguing about 9.4× by getting WRF's actual numbers. Build the WRF idealized **em_quarter_ss** (warm bubble) case: configure + `./compile em_quarter_ss` (or reuse if a build exists — check `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/` and `wrf_gpu` builds), run `ideal.exe` + a few steps of `wrf.exe` with light instrumentation dumping `grid%p`, `grid%al`, `grid%ph`, the `pg_buoy_w` `rw_tend`, `w`, `t_2ave`, `muave` after the IC rebalance and after steps 1–3. Compare JAX (same IC) field-by-field. Write `proofs/f7h/wrf_vs_jax_warmbubble.json` with the per-field deltas. THIS is the definitive answer to "is JAX's grid%p / pg_buoy_w / ph correct?" If the WRF ideal build is infeasible in a bounded effort, say so explicitly and fall back to the analytic discrete-balance check, documenting the limitation.

## Acceptance gates (for `F7H_COMPLETE` = F7 dycore close)
- **AC1 — ph restoring develops**: in the instrumented warm-bubble run, `ph_perturbation` and `p_perturbation` **evolve with `w`** (not frozen); document the trace. The vertical w/ph coupling is live.
- **AC2 — Skamarock warm bubble PASS**: finite to 500 s, thermal rises (centroid ≥ 500 m), `max|w|` **saturates/bounded physical** (≤30 m/s, NOT linear-in-t), θ′ transported, symmetric, mass-conserving.
- **AC3 — Straka density current PASS**: finite to 900 s, front ≈ 15 km (±~2 km), min θ′ ≈ −9..−10 K, max|w| O(10), mass drift ≤ 1e-8.
- **AC4 — WRF ground-truth (if built)**: JAX vs WRF `grid%p`/`al`/`ph`/`rw_tend`/`w` agree within a documented tolerance after the IC + first steps; OR a documented reason the WRF ideal build was deferred.
- **AC5 — no regression**: A/B/C/D/F gates hold (no-stub, flat-rest=0, analytic dipole, 300-step conservation, mass identities); nothing weakened/xfailed.

## Hard rules
1. `taskset -c 0-3`; `PYTHONPATH=src`; `cuda:0`; fp64.
2. WRF source is ground truth; cite file:line.
3. **No masking clamps/caps/sanitizers, no coefficient tuning, no synthetic pressure.** Fix the coupling/carry; if the cases STILL fail after Phase 1 + Phase 2 ground truth, mark `F7H_PARTIAL`, deliver the WRF-vs-JAX deltas, and STOP for manager review — do NOT add an 8th workaround.
4. **No performance work.**
5. **Time-box the reasoning**: instrument and run; if a hypothesis isn't confirmed by a trace within a bounded effort, move to the WRF ground-truth comparison. Do not spend >30 min reasoning without running something.
6. Commit incrementally on `worker/opus/f7d-pressure-mass-fix`; do not push; do not switch branches.
7. Files writable: `src/gpuwrf/**`, `scripts/**` (instrumentation + WRF-ideal harness, never weaken invariants), `tests/**` (add/fix, never weaken), `proofs/f7h/**`, this sprint folder. WRF Fortran instrumentation: you MAY add print/dump statements to a COPY of the WRF source or a build dir, but do NOT modify the canonical Gen2 `wrf.exe` source tree in place without copying first (the Gen2 baseline sha must not change — see the project rule). Prefer a separate ideal build dir.
8. Files NOT writable: governance, memory, skills, ADRs, plan, physics-scheme code, comparator scripts under `scripts/m6b6_*`.

## Deliverables
`proofs/f7h/ph_carry_trace.json` (Phase 1), `proofs/f7h/wrf_vs_jax_warmbubble.json` (Phase 2, if built), `straka_density_current.json`+verdict+plots, `skamarock_warm_bubble.json`+verdict+plots, `ph_coupling_fix.md` (what was broken + WRF file:line + before/after trace), `regression_recheck.json`, `worker-report.md` (AGENTS.md format) ending `F7H_COMPLETE` or `F7H_PARTIAL` + precise gaps + (if PARTIAL) the WRF-vs-JAX deltas that localize the remaining discrepancy.

## Forward pointer
- On AC2+AC3 PASS → manager GPT-5.5 pre-close critique → F7 dycore CLOSE → merge f7d chain to `manager-2026-05-23`.
- The WRF-ideal savepoint harness built here seeds **M9** (per-operator WRF↔JAX parity = the rigorous near-identical-RMSE-vs-real-WRF gate).
