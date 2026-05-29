# Sprint Contract — F7I: kill the residual 2Δz vertical mode → idealized cases PASS → F7 dycore CLOSE (WRF ground-truth)

**Sprint ID**: `2026-05-29-f7i-2dz-mode-wrf-truth`
**Frontrunner**: Opus 4.8 (in-process Agent subagent, high/max effort)
**Branch**: `worker/opus/f7d-pressure-mass-fix` (CONTINUE — tip `86a334a`, carries the validated grid%p-refresh fix). Merge the chain once the idealized cases PASS.
**GPU**: YES — `PYTHONPATH=src taskset -c 0-3`; `cuda:0`; fp64.

## State (one bug from dycore close)
The big runaway is FIXED (grid%p refresh from finished ph'/θ; warm-bubble max|w|@100s 44.7→4.3, thermal rises). ONE residual: a **2Δz vertical acoustic mode at the bubble-center column** (w alternating ±9 between adjacent levels) grows → detonates ~180 s, before the 500 s/900 s gates. This is an off-centering/coefficient question in the implicit w/ph solve — NOT a workaround/tuning question. A parallel GPT-5.5 audit is running on the same residual (`.agent/sprints/2026-05-29-f7i-2dz-mode-wrf-truth/gpt-coefaudit-findings.md` when ready — read it and fold in).

## Charter — settle it with WRF GROUND TRUTH (definitive), this is the priority
After ~9 theory-iterations, stop guessing: get WRF's actual numbers for the SAME case and diff the implicit-solve.
1. **Build WRF `em_quarter_ss`** (the warm-bubble idealized case) in a SEPARATE tree — copy the WRF source (do NOT modify the pinned Gen2 em_real tree / its sha), `./configure` + `./compile em_quarter_ss`. If a usable ideal build already exists, reuse it.
2. Run `ideal.exe` + a few steps of `wrf.exe` with light instrumentation dumping, for the bubble-center column: the `calc_coef_w` tridiagonal coefficients, `w`, `ph`, `p`, `t_2ave`, `muave`, `epssm` per acoustic substep after the IC and steps 1–3. Write `proofs/f7i/wrf_em_quarter_ss_savepoints.json`.
3. Diff JAX (same IC) field-by-field, focused on the implicit w/ph solve at the center column → `proofs/f7i/wrf_vs_jax_implicit_w.json`. This pinpoints exactly where JAX's coefficients/off-centering diverge and admit the 2Δz mode.
4. **Land the fix** (the WRF-correct off-centering / coefficient form — `epssm` applied to both the w update and the ph/pressure coupling per WRF; signed metrics). Re-run the idealized cases.
5. **If the WRF ideal build is genuinely infeasible in a bounded effort** (toolchain/time): say so explicitly, and instead implement the fix from the GPT audit's WRF-source analysis + a numerical 2Δz-damping check, documenting the build blocker. Do not let the build block the fix.

## Acceptance gates (for `F7I_COMPLETE` = F7 dycore CLOSE)
- **AC1 — 2Δz mode gone**: warm-bubble center-column `w(k)` shows NO 2Δz alternating-sign growth; document the trace before/after.
- **AC2 — Skamarock warm bubble PASS**: finite to 500 s, thermal rises (centroid ≥ 500 m), max|w| bounded physical (≤30), θ′ transported, symmetric, mass-conserving.
- **AC3 — Straka density current PASS**: finite to 900 s, front ≈ 15 km (±~2 km), min θ′ ≈ −9..−10 K, max|w| O(10), mass drift ≤ 1e-8.
- **AC4 — WRF ground truth**: JAX vs WRF `calc_coef_w` coeffs + w/ph/p at the center column agree within documented tolerance after IC + first steps (or a documented build-deferral with the analytic fix justification).
- **AC5 — no regression**: A/B/C/D/F/H gates hold (no-stub, flat-rest=0, analytic dipole, 300-step conservation, grid%p-refresh, mass identities); nothing weakened/xfailed; no clamps/caps/epssm-fudge-beyond-WRF.

## Hard rules
1. `PYTHONPATH=src taskset -c 0-3`; `cuda:0`; fp64. WRF source is ground truth; cite file:line.
2. **No masking clamps/caps/sanitizers; no epssm/coefficient tuning beyond the WRF-correct value/form.** The fix must be the WRF-faithful off-centering/coefficient. If the cases still fail after the WRF-grounded fix, mark `F7I_PARTIAL`, deliver the WRF-vs-JAX implicit-solve deltas, and STOP for manager review.
3. **No performance work.** Commit incrementally on `worker/opus/f7d-pressure-mass-fix`; no push; no branch switch.
4. WRF instrumentation: copy the source to a separate build dir; do NOT modify the canonical Gen2 wrf.exe tree in place (its sha must not change).
5. Files writable: `src/gpuwrf/**`, `scripts/**` (WRF-ideal harness + comparator + instrumentation, never weaken invariants), `tests/**` (add/fix, never weaken), `proofs/f7i/**`, this sprint folder, a separate WRF ideal build dir under `/home/enric/src/` or `/mnt/data/`. NOT writable: governance, memory, skills, ADRs, plan, physics-scheme code, the canonical Gen2 WRF tree, `scripts/m6b6_*`.

## Deliverables
`proofs/f7i/`: `wrf_em_quarter_ss_savepoints.json` (or build-deferral note), `wrf_vs_jax_implicit_w.json`, `center_column_w_trace.json` (2Δz before/after), `straka_density_current.json`+verdict+plots, `skamarock_warm_bubble.json`+verdict+plots, `2dz_fix.md` (root cause + WRF file:line + fix), `regression_recheck.json`. `worker-report.md` (AGENTS.md format) ending `F7I_COMPLETE` or `F7I_PARTIAL` + precise gaps.

## Forward pointer
On AC2+AC3 PASS → manager GPT-5.5 **pre-close critique** of the whole dycore (per principal's firm rule: GPT review before any major milestone close) → **F7 dycore CLOSE** → merge f7d chain to `manager-2026-05-23` → Phase B (M11/M17 first physics on a correct core). The WRF-ideal savepoint harness built here SEEDS M9 per-operator parity (the rigorous near-identical-vs-WRF gate).
