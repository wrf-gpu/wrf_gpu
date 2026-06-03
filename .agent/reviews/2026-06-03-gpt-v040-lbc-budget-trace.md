# GPT v0.4.0 LBC Budget Trace Handoff - 2026-06-03

## Objective

Empirically localize the remaining v0.4.0 standalone forecast bias after
`a1def01` without masking, tolerance loosening, self-compare, or synthetic
fixtures. Primary discriminator: `20260429_18z_l2_72h_20260524T204451Z`;
secondary check: `20260521_18z_l3_24h_20260522T133443Z`.

## Decision

The first budget failure visible in real h-output states is **column-mass
divergence from low-level prognostic momentum residual entering `advance_mu_t`**.
This is not the prior specified/nested loop-bound bug. PSFC is low because MU is
low, and U10 follows bottom-level prognostic U almost one-to-one.

The strongest upstream momentum lead is **large-step horizontal PGF / momentum
balance**. Applied to each side's h1/h2 output state, PGF residuals are 4-6x WRF
term RMSE, much larger than Coriolis residual ratios. This is a lead, not yet a
surgical fix: these are h-output state formula evaluations, not paired in-loop
WRF/JAX savepoints inside the same RK/acoustic substep.

No production fix applied. v0.4.0 cannot close from this sprint.

## Budget Table

Primary case `20260429_18z_l2_72h_20260524T204451Z`, JAX minus WRF:

| Lead | PSFC mean Pa | MU mean Pa | U10 mean m/s | bottom U k0 mean m/s | U10-bottomU corr | formula DMDT mean Pa/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| h1 | -29.679 | -29.953 | -0.350 | -0.359 | 0.9972 | -0.0001065 |
| h2 | -64.946 | -64.988 | -0.284 | -0.293 | 0.9951 | -0.0062370 |
| h6 | -242.089 | -242.453 | +0.295 | +0.333 | 0.9947 | n/a |
| h12 | -214.952 | -235.726 | +1.591 | +1.699 | 0.9925 | n/a |
| h24 | -439.824 | -489.946 | +1.365 | +1.426 | 0.9996 | n/a |

Primary h1/h2 term ranking:

| Rank | Candidate | h1 residual | h2 residual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Column-mass divergence from low-level momentum | MU -29.95 Pa, PSFC -29.68 Pa, DMDT -1.07e-4 Pa/s | MU -64.99 Pa, PSFC -64.95 Pa, DMDT -6.24e-3 Pa/s, hourly dMU/dt -9.73e-3 Pa/s | Direct first budget failure |
| 2 | Large-step PGF / momentum balance | ru RMSE ratio 5.38, rv 4.08 | ru RMSE ratio 5.95, rv 4.06 | Strongest upstream lead, not in-loop-isolated |
| 3 | Coriolis | ru RMSE ratio 0.055, rv 0.027 | ru RMSE ratio 0.135, rv 0.056 | Amplifier after U/V drift |
| 4 | Surface drag | UST +0.019 m/s, QKE k0 -0.070 | UST +0.026 m/s, QKE k0 -0.020 | Not supported as too-weak drag |
| 5 | Specified LBC/end-step boundary | PSFC ring -23.97 Pa vs interior -31.67 Pa | PSFC ring -67.54 Pa vs interior -64.04 Pa | Not boundary-ring-local |
| 6 | PSFC/U10 diagnostic conversion | GPU PSFC-vs-formula RMSE 0.0026 Pa; U10-bottomU corr 0.997 | GPU PSFC-vs-formula RMSE 0.0026 Pa; U10-bottomU corr 0.995 | Rejected |

Secondary case `20260521_18z_l3_24h_20260522T133443Z` agrees on structure:
U10-bottomU correlation is 0.985 at h1 and 0.974 at h2, PGF residual ratios are
~5.4/~3.9 at h1 and ~5.25/~4.12 at h2, and Coriolis ratios stay much smaller.

## WRF Source Anchors

SHA256-verified source files:

- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F`
  `d0341b700c39c8baf6a4af34637901d333265494edef43debec7ed19227b0092`
- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_big_step_utilities_em.F`
  `a6eac120af17109af00c18fb6e08277d288067882e7d4865545604f8a9c1d766`
- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_em.F`
  `b378409455b5c0c97630d11c197b40947a456c0ffe4ef73d8ae71d54152d4a70`

Formula anchors:

- MU/divergence writeback: `dyn_em/module_small_step_em.F:1094-1108`
- PGF u: `dyn_em/module_big_step_utilities_em.F:2453-2488`
- PGF v: `dyn_em/module_big_step_utilities_em.F:2373-2404`
- Coriolis driver/body: `dyn_em/module_em.F:1402-1428`,
  `dyn_em/module_big_step_utilities_em.F:3924-4134`

## Files Changed

- `proofs/v040/lbc_budget_trace.py`
- `proofs/v040/lbc_budget_trace.json`
- `.agent/reviews/2026-06-03-gpt-v040-lbc-budget-trace.md`

No production model code changed.

## Commands Run

- `git worktree add -b worker/gpt/v040-lbc-budget-trace /home/enric/src/wrf_gpu2/.claude/worktrees/v040-budget-trace worker/gpt/v040-mu-continuity-fix`
- `git -C /home/enric/src/wrf_gpu2/.claude/worktrees/v040-budget-trace log -1 --oneline --decorate`
- `python -m py_compile proofs/v040/lbc_budget_trace.py`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=true JAX_DISABLE_JIT=true JAX_ENABLE_COMPILATION_CACHE=false XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 PYTHONPATH=src:proofs/v040:. python proofs/v040/lbc_budget_trace.py --out proofs/v040/lbc_budget_trace.json`
- Several read-only `python`, `rg`, `sed`, `nl`, `sha256sum`, and `git status` inspections.

No GPU forecast job was launched in this sprint; the proof uses existing real
JAX forecast artifacts and matching CPU WRF wrfout files.

## Proof Objects Produced

- `proofs/v040/lbc_budget_trace.json`

Inherited from `a1def01` because production code was not edited:

- `proofs/v040/mu_continuity_savepoint_parity.json`: PASS
- `proofs/v040/idealized_no_regression_report.json`: PASS
- `proofs/v040/replay_path_impact_check.json`: PASS

No `forecast_gate_postfix2_report.json` was produced because no fix was applied.
The known 2-date postfix bias remains uncollapsed.

## Unresolved Risks

- The JAX artifact set starts at h1, so h1 hourly tendency from h0->h1 is not
  available without a new h0/JAX dump. The report uses h1 snapshot residuals and
  h2 hourly residuals.
- PGF/Coriolis/MU-divergence terms are evaluated from h-output states, not paired
  in-loop savepoints. PGF is therefore a strong upstream lead, not a proven
  formula defect.
- Surface-drag exact tendency is not pair-scored because JAX wrfout does not emit
  WRF `DTAUX3D/DTAUY3D`; available diagnostics do not support too-weak drag.

## Next Decision Needed

Approve a focused in-loop savepoint sprint for the first 1-2 real forecast hours:

1. Dump JAX h0 plus RK/acoustic substep savepoints around `rk_tendency`
   PGF/advection/Coriolis, `advance_uv`, and `advance_mu_t`.
2. Add matching WRF instrumentation in `module_em.F` and
   `module_big_step_utilities_em.F` for the same real case and substep.
3. Compare PGF component terms (`ph`, `alt*p`, `al*pb`, nonhydrostatic term 4),
   momentum advection/diffusion, Coriolis, and boundary-save tendencies before
   applying any production patch.

