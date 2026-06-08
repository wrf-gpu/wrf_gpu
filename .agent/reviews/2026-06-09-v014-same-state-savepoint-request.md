# V0.14 Same-State Savepoint Request Review

Date: 2026-06-09
Agent: GPT-5.5 xhigh sidecar
Mode: CPU-only manifest packaging

## Objective

Package Helmholtz's accepted h10 dynamic attribution selection into a compact
CPU-WRF savepoint request manifest for the first same-state localization run.
No WRF instrumentation, JAX comparison, production `src/` edit, WRF edit, GPU
run, equivalence claim, or root-cause claim was made.

## Files Changed

- `proofs/v014/same_state_savepoint_request.py`
- `proofs/v014/same_state_savepoint_request.json`
- `proofs/v014/same_state_savepoint_request.md`
- `.agent/reviews/2026-06-09-v014-same-state-savepoint-request.md`

Unrelated dirty/staged files already present in the worktree were not modified.

## Commands Run

- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 python proofs/v014/same_state_savepoint_request.py`
- `python -m json.tool proofs/v014/same_state_savepoint_request.json >/tmp/same_state_savepoint_request.validated.json`
- `python -m py_compile proofs/v014/same_state_savepoint_request.py`
- Focused inspection: `jq` checks for selected-cell count, term groups, full-column flag, and dependency status.

All contract validation commands exited 0.

## Proof Objects Produced

- `proofs/v014/same_state_savepoint_request.json`
- `proofs/v014/same_state_savepoint_request.md`

The JSON contains exactly 24 selected mass-grid cells for h10
(`2026-05-02T04:00:00+00:00`), all sourced from
`proofs/v014/dynamic_field_attribution.json`.

## Manifest Contents

Selected run/domain/lead:

- Run id: `20260501_18z_l2_72h_20260519T173026Z`
- Domain: `d02`
- Lead: `h10`
- Selected valid time: `2026-05-02T04:00:00+00:00`

The manifest includes per-cell native U/V/W/PH stagger context, zero-based
stop-exclusive patch bounds, one-based inclusive WRF/Fortran translations, and
full-column native patch bounds. First-probe reporting levels are:

`0, 1, 2, 16, 17, 18, 24, 25, 26, 28, 29, 30, 31, 32`

Full native vertical columns are required for the WRF savepoint artifact.

Requested samples:

- RK stages: 1, 2, 3.
- Acoustic substeps: first and last substep for every requested RK stage.

Requested term groups:

`stage_input`, `mass_coupling`, `momentum_advection`,
`scalar_theta_mu_advection`, `diffusion`, `horizontal_pgf`, `coriolis`,
`source_tendency_folding`, `small_step_prep`, `acoustic_uv`, `mu_theta`,
`w_ph`, `pressure_rho_refresh`, `boundary_spec_relax`, `final_stage_state`.

The expected WRF savepoint artifact schema is included in JSON, including
global provenance attributes, native-stagger dataset metadata, selection echo,
patch bounds, RK/acoustic timing metadata, and companion build/run artifacts.

## Dependency Status

Sartre's WRF source/build feasibility proof appeared in the workspace during
this sprint and is referenced as available:

- `proofs/v014/same_state_wrf_savepoint_feasibility.json`
- `proofs/v014/same_state_wrf_savepoint_feasibility.md`

The request manifest still uses Helmholtz's
`proofs/v014/dynamic_field_attribution.json` as the authoritative selected
h10/cell/level source. Sartre's proof is used only for dependency status and
WRF source/build feasibility context.

## Unresolved Risks

- The request assumes the next instrumentation worker follows Sartre's
  disposable-copy recommendation and records exact WRF source/build/patch
  provenance.
- The exact h10 model-step mapping must still be proven by the WRF
  instrumentation worker with a marker savepoint before accepting term data.
- Cropped patch savepoints must preserve halo 8 and full native vertical
  columns; otherwise same-state term comparison can be invalid.

## Next Decision Needed

Open the WRF instrumentation sprint to generate source-derived CPU-WRF h10
savepoints from the requested 24-cell manifest, then run the same-state
JAX CPU term comparison under a separate contract.
