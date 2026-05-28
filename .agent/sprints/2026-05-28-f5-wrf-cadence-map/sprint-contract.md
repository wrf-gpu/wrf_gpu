# Sprint Contract — F5: WRF Source-Cadence Map (pre-rewrite spec)

**Sprint ID**: `2026-05-28-f5-wrf-cadence-map`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/f5-wrf-cadence-map`
**Worktree**: `/tmp/wrf_gpu2_f5`
**Wall-time**: 1-2 days
**GPU usage**: NO (pure reading)
**Sandbox**: default (workspace-write fine — read-only WRF Fortran + write a spec doc)

## Why this sprint

Per F4 critique (`.agent/sprints/2026-05-28-f4-gpt55-plan-grounding/critique.md` Q4):
"We should read WRF Fortran before more JAX changes. Not just instrument it: read it and write a cadence map."

Per F3 Opus arch review (`.agent/sprints/2026-05-28-f3-agy-architecture-followup/findings.md`):
the dycore needs ~950 LOC rewrite mirroring WRF's RK3+small-step cadence.
Without a written cadence spec, the rewrite worker will reproduce the same
structural mistakes (missing advance_uv, mu_save lost across stages, etc).

This sprint produces the spec the rewrite sprint will implement against.

## Binding goal (universal)

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72h RMSE on T2/U10/V10
**statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins
on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs (read in order)

1. `.agent/sprints/2026-05-28-f3-agy-architecture-followup/findings.md` — Opus arch review with WRF file:line refs
2. `.agent/sprints/2026-05-28-f4-gpt55-plan-grounding/critique.md` — what the cadence map must answer
3. `.agent/sprints/2026-05-28-agy-dycore-deep-review/findings.md` — original agy diagnosis
4. WRF Fortran source at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/`:
   - `solve_em.F` — top-level dycore driver
   - `module_em.F` — RK stages (`rk_tendency`, `rk_phys_tend`, `rk_scalar_tend`, `rk_addtend_dry`)
   - `module_small_step_em.F` — acoustic small steps (`advance_uv`, `advance_mu_t`, `advance_w`)
   - `module_big_step_utilities_em.F` — utility decouple/couple functions

## Deliverable

Write `proofs/f5/wrf_cadence_spec.md`. Use Markdown. This is a spec — concrete, no fluff. Structure:

### Section 1 — Top-level RK3+small-step driver (from solve_em.F)

For each RK stage (1, 2, 3):
- Number of acoustic substeps (1 / number/2 / number)
- What state is saved at stage start (theta_1, mu_1, ph_1, etc — name each)
- What tendencies are computed once on RK1 and held constant (physics tendencies)
- What tendencies are recomputed per stage (advection)
- What the RK predictor combines (formulas with file:line refs)

### Section 2 — Inside one acoustic substep (from module_small_step_em.F)

Order of operations with file:line refs:
1. `advance_uv` — what reads / what writes; what's coupled
2. `advance_mu_t` — what reads / what writes; what's coupled
3. `advance_w` — what reads / what writes; specifically how acoustic-`p` is propagated (the `c2a = cpovcv*(pb+p)/alt` term and the divergent vertical mass flux)
4. After all substeps in a stage: what is decoupled / cleaned up

For each operation, state which variables are TOTAL fields and which are PERTURBATIONS at the boundary. Note any factor coupling (map factors, c1h, c2h).

### Section 3 — Cross-stage state carry

- `mu_save` — when computed, when consumed, what RK2/RK3 expect from RK1
- `muts` vs `mut` vs `muave` vs `mudf` — which is total perturbation, which is work array
- `theta` vs `theta_1` — when current vs stage-start is used
- Boundary apply: does WRF apply lateral BC inside or outside RK3?

### Section 4 — Decouple/couple discipline

Reference `module_big_step_utilities_em.F` to spell out when coupling `*mu`, `*muts`, `*c1h/c2h` is applied or undone. Critical for `_decouple_theta_after_advance` parity.

### Section 5 — JAX dycore gap map

For each WRF function in sections 1-4, list:
- Is there a JAX equivalent in `src/gpuwrf/`? (yes/no + file:line)
- If yes, does the cadence position match? (yes/no + analysis)
- If missing or wrong, what needs to be built/fixed?

This produces the concrete punchlist for the rewrite sprint.

### Section 6 — Specific answers to F4 Q4

F4 listed unanswered cadence questions. Answer each explicitly:
- When does WRF save stage-start fields?
- When is large-step advection recomputed?
- Where do physics tendencies enter RK3?
- When is `mu/muts/muave/mudf` physical perturbation vs work array?
- When is perturbation pressure/geopotential diagnosed or advanced?
- Which arrays are total and which are perturbations at each boundary between routines?

## Acceptance

- **AC1**: `proofs/f5/wrf_cadence_spec.md` exists, sections 1-6 filled, every claim with WRF source file:line refs.
- **AC2**: Section 5 (gap map) names ≥5 concrete JAX changes needed for the rewrite sprint.
- **AC3**: `.agent/sprints/2026-05-28-f5-wrf-cadence-map/worker-report.md` written with verdict.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3` for any tooling.
2. **No GPU** (no JAX/dycore execution).
3. **Files writable**: `proofs/f5/**`, `.agent/sprints/2026-05-28-f5-.../**`.
4. **Files NOT writable**: any JAX source code, any model code, any tests, governance, plan, ADRs.
5. **DO NOT MODIFY** the canairy_meteo WRF source — read-only reference.
6. **No remote push.**
7. **Manager repo ONLY**.
8. **Auto-notify on exit**: `tmux send-keys -t 0:0 "AGENT REPORT: f5 DONE exit=$?" Enter`.
9. **End with verdict**: `F5_COMPLETE` if AC1-AC3 pass; `F5_PARTIAL` with explicit gaps.
