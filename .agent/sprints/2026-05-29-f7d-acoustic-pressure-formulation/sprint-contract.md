# Sprint Contract — F7D: acoustic-core total-mass (MUT/MUTS) semantics fix → idealized cases PASS (dycore close)

**Sprint ID**: `2026-05-29-f7d-acoustic-pressure-formulation`
**Frontrunner**: Opus 4.8 (in-process Agent subagent, high/max effort)
**Branch**: `worker/opus/f7d-pressure-mass-fix` (work in the main tree on this branch; commit incrementally)
**GPU**: YES — `taskset -c 0-3`; confirm `cuda:0`; fp64.
**Builds on**: Sprints A+B+C (merged, `manager-2026-05-23` tip `1e039ac`). Circulation works; the idealized cases still go non-finite (~80–100 s).

## Binding spec for this sprint

The authoritative spec is the GPT-5.5 WRF-verified findings: **`.agent/sprints/2026-05-29-f7d-acoustic-pressure-formulation/gpt-findings.md`** — read it in full; it supersedes the earlier frontrunner hypothesis. **IMPORTANT: do NOT make the small-step work pressure `p` "absolute p-prime"** — GPT-5.5 verified against WRF that this would double-count the large-step PGF/`pg_buoy_w` and is wrong. The real bug is total-mass semantics.

## The verified root cause

JAX's `prep.mut` currently equals `MUB` (base dry mass), but WRF's `grid%mut` is the **full stage-entry dry mass** `MUB + MU_current` (`module_em.F:184-187`, `module_big_step_utilities_em.F:3912-3916`), and `MUTS = MUT + MU_work` (`module_small_step_em.F:1102-1107`). So `calc_p_rho` is fed the wrong total-mass denominator: JAX passes `prep.mut`/`uv_state.mut` where WRF passes `grid%muts` (`solve_em.F:2628-2635`, `4164-4171`). The work-pressure *formula* (`calc_p_rho.py:73-87` ≈ WRF `module_small_step_em.F:522-528`) is structurally correct — it is being fed the wrong total mass. This breaks the acoustic restoring loop, so buoyancy + PGF pump momentum with no restoring → linear u/w runaway → NaN.

## Cardinal rule
WRF Fortran source is ground truth. Verify every change against `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/` at the line refs GPT-5.5 cited. If anything disagrees, WRF wins — note it.

## Scope — the GPT-5.5 fix spec (implement exactly; §3.2 of gpt-findings.md)

1. **`SmallStepPrepState` mass semantics** (`src/gpuwrf/dynamics/core/small_step_prep.py`):
   - Make `prep.mut` mean WRF `grid%mut` = full stage-entry dry mass `MUB + MU_current` (currently it is `MUB`). Keep an explicit base field if needed.
   - `mu_work` = WRF `MU_2` after `small_step_prep`: RK1 → `0`; later RK steps → `MU_ref − MU_current` (`module_small_step_em.F:196-215`).
   - `prep.muts = prep.mut + mu_work` (= `MUB + MU_current + MU_work`).
   - Recompute `muu/muv` from `prep.mut` and `muus/muvs` from `prep.muts` (`module_small_step_em.F:172-207`).
   - Build `theta_work/u_work/v_work/w_work` with the WRF current/stage mass pairs (`module_small_step_em.F:238-276`).
2. **`calc_p_rho` call sites**:
   - `calc_p_rho_wrf`: pass `mut=prep.muts` into `_calc_al_p` (NOT `prep.mut`).
   - `acoustic_substep_core`: pass the live `muts_new` into `calc_p_rho_step` as the WRF `Mut` denominator (NOT `uv_state.mut`).
   - Rename `_calc_al_p`'s `mut` param to `muts_total` to prevent recurrence.
   - **Keep the numerator as work variables** (`mu_work`, `ph_work`, `theta_work`) — do NOT make it absolute.
3. **Downstream `mut` consumers** (audit all, the semantic change ripples):
   - `advance_mu_t_wrf`: keep `mu_work_old = inputs.muts − inputs.mut`, but `inputs.mut` is now full stage-entry dry mass.
   - `calc_coef_w_wrf_coefficients`: must receive WRF `grid%mut` full stage-entry mass (`solve_em.F:2676-2681`), not `MUB`.
   - `advance_w_wrf`: must receive WRF `mut` full stage-entry mass + live `muts` (`module_small_step_em.F:1178-1185`).
   - `small_step_finish_wrf`: audit reconstruction denominators/numerators after the change (`module_small_step_em.F:379-430`).
4. **Keep the WRF split for vertical buoyancy** (do NOT change): `pg_buoy_w` stays the once-per-RK-stage `rw_tend` source from stage-entry absolute `p`/`mu` (`module_em.F:1361-1368`); do not feed substep work `p` into `pg_buoy_w`. The restoring loop is: `advance_uv(work p/al)` → `advance_mu_t(mu/theta/ww)` → `advance_w(w/ph, t_2ave, muave)` → `calc_p_rho_step(work p/al from live MUTS/theta/ph)` → next `advance_uv`.

## Acceptance gates (all required for `F7D_COMPLETE` = F7 dycore close)

- **AC1 — RK1 source-parity unit test**: at a stage-entry rest state, work `mu/theta/ph` are 0 and work `p/al` are 0, while the independently-computed stage absolute `p_buoy` is nonzero for a warm/cold bubble (proves WRF's split is preserved). (gpt-findings §3 check 1)
- **AC2 — one-column / small-2D acoustic probe**: after `advance_mu_t` + `advance_w`, `calc_p_rho_step` changes work `p/al` from their step-0 values and updates `pm1`; the next `advance_uv` consumes that changed work `p`. (check 2 — proves the restoring loop is live)
- **AC3 — idealized cases PASS / no runaway**: `flat_rest` stays exact 0; **Straka density current and Skamarock warm bubble stay finite past the 80–100 s failure window** and `max|w|` **stops the coherent linear growth** (saturates / oscillates), trending toward the published-reference envelope (Straka front ≈ 15 km, min θ′ ≈ −9..−10 K, max|w| O(10); warm-bubble θ′ transported, centroid rises ≥500 m, w in range). No clamp/masking — require pressure/mass substep traces as proof.
- **AC4 — 12-step operational-dt audit improves**: `first_critical_violation` moves past Sprint C's step 8 toward clean; report honestly.
- **AC5 — no regression**: Sprint A/B/C gates (no-stub, flat-rest=0, analytic dipole, 300-step conservation, advection order, circulation) re-run and hold; no test weakened/xfailed.

(Falsifiable check #3 — WRF substep-granularity savepoint comparison — is deferred to M9, which builds the instrumented WRF savepoints. It is the rigorous near-identical-RMSE-vs-WRF gate; note it but it is out of scope here.)

## Proof objects (into `proofs/f7d/`)
`mass_semantics_proof.md` (the MUT/MUTS fix, with WRF file:line + before/after `prep.mut` values); `rk1_source_parity.json` (AC1); `acoustic_restoring_probe.json` (AC2); `straka_density_current.json`+verdict+plots, `skamarock_warm_bubble.json`+verdict+plots (AC3); `audit_operational_dt.json`+`audit_summary.md` (AC4); `regression_recheck.json` (AC5); `worker-report.md` (AGENTS.md format) ending `F7D_COMPLETE` or `F7D_PARTIAL` + precise gaps.

## Hard rules
1. `taskset -c 0-3`; `cuda:0`; fp64.
2. WRF source is ground truth; cite `file:line` in every changed operator docstring.
3. **No masking clamps/caps/sanitizers.** WRF damping/diffusion (named, WRF coefficients) is fine; ad-hoc clamps to force a gate green are not. Require pressure/mass substep traces as proof, not just "it's finite now."
4. **No performance work** (no fp32/fusion) — correctness only; perf is F7-perf.
5. Commit incrementally on `worker/opus/f7d-pressure-mass-fix`; do not push; do not switch branches.
6. Files writable: `src/gpuwrf/dynamics/**`, `src/gpuwrf/runtime/operational_mode.py`/`operational_state.py`, `src/gpuwrf/ic_generators/idealized.py` (minimal harness fixes, documented), `scripts/**` (instrumentation, never weaken invariants), `tests/**` (add/fix, never weaken), `proofs/f7d/**`, this sprint folder.
7. Files NOT writable: governance, memory, skills, ADRs, plan, physics-scheme code, comparator scripts under `scripts/m6b6_*`.
8. If the fix does not fully land, deliver the largest correct gated subset and mark `F7D_PARTIAL` with precise evidence (esp. whether the mass-semantics change alone removes the runaway). Honest partial > green self-compare.

## Forward pointer
- On AC3 PASS → manager runs the **GPT-5.5 WRF-domain pre-close code critique** of the whole dycore, then declares the **F7 dynamical-core milestone closed**.
- Then **F7-perf** (XLA fusion + fp32 downcast + speedup recert ≥10×) and **M9** (instrumented WRF savepoints → per-operator parity, the rigorous near-identical-RMSE-vs-real-WRF gate).
