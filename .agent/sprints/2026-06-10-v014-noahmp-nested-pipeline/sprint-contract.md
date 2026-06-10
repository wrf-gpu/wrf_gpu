# Sprint Contract: V0.14 Noah-MP In Standalone Nested Pipeline

Date: 2026-06-10
Owner: manager
Assignee: Fable high in isolated worktree
Status: OPEN

## Objective

Fix the v0.14 release blocker found in the Canary h24 residual review:
standalone live-nested runs currently leave land skin temperature frozen because
`src/gpuwrf/integration/nested_pipeline.py` builds each domain's
`OperationalNamelist` without enabling Noah-MP / `sf_surface_physics=4` and
without seeding the `noahmp_land` carry.

Endpoint: Canary/Switzerland nested validation runs use the same prognostic
Noah-MP land path as CPU truth when the input namelist selects
`sf_surface_physics=4`, so land `TSK`, `HFX`, `T2`, and `PBLH` evolve rather
than staying prescribed/frozen.

## Current State

- Main branch/head: `worker/gpt/v013-close-manager` at `7c819067`.
- Current Canary 72h GPU run is still running and must not be stopped. It is a
  useful pre-fix baseline, not a release-green run.
- Run root:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_moistcqw_20260610T171818Z`
- Fable h24 review:
  `.agent/reviews/2026-06-10-v014-fable-canary-h24-residual.md`
- Proof summary:
  `proofs/v014/canary_h24_residual_adjudication.{json,md}`
- Root cause from that review:
  `nested_pipeline._make_namelist` omits `use_noahmp`, `sf_surface_physics`,
  `noahmp_static`, and Noah-MP parameter bundles. `OperationalNamelist.use_noahmp`
  defaults `False`, so the land tile stays on the prescribed bulk path.

## Non-Goals

- Do not change the Noah-MP coupler physics unless a hard proof shows the
  current coupler cannot be wired correctly. The coupler has separate Step-1
  closure proofs.
- Do not touch the running GPU job or launch GPU validation from this worker.
- Do not change field tolerance policy or comparator manifests.
- Do not broaden FP32/memory behavior in this sprint.
- Do not hide residuals with clamps, masks, writer-only substitutions, or
  JAX-vs-JAX self-comparisons.

## File Ownership

Allowed source files:

- `src/gpuwrf/integration/nested_pipeline.py`
- `src/gpuwrf/runtime/domain_tree.py` only if carry seeding cannot be done
  cleanly in `nested_pipeline.py`; prefer a narrow API addition over changing
  timestep logic.
- `tests/*noahmp*nested*` or a similarly focused new test file.
- `proofs/v014/noahmp_nested_pipeline_*`
- This sprint folder's worker report.

Read-only reference files:

- `src/gpuwrf/io/noahmp_land_init.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/runtime/operational_state.py`
- `src/gpuwrf/integration/daily_pipeline.py`
- `proofs/m20/tost_noahmp_runner.py`
- `proofs/noahmp/s6b_activate_validate.py`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_surface_land_flux_handoff.md`
- `proofs/v014/noahmp_step1_closure.md`

## Implementation Guidance

Mirror the existing proven wiring:

1. For each domain in `_load_domains`, read per-domain `sf_surface_physics`.
2. If the option is `4`, build:
   - `noahmp_land, noahmp_static, noahmp_init_meta =
     build_noahmp_land_state(run_dir, domain)`
   - `energy_params, rad_params, nroot = build_noahmp_params(noahmp_static)`
   - initial held radiation via `noahmp_initial_rad(state, namelist,
     land_state=noahmp_land)` if needed by the carry.
3. Build the namelist with `use_noahmp=True`, `sf_surface_physics=4`,
   `noahmp_static`, `noahmp_energy_params`, `noahmp_rad_params`,
   `noahmp_nroot`, `noahmp_julian`, and `noahmp_yearlen`.
4. Ensure the initial `OperationalCarry` for each domain receives the matching
   `noahmp_land` and `noahmp_rad`. If `domain_tree.run_operational_domain_tree`
   cannot currently accept preseeded carries from bundles, add the narrowest
   explicit mechanism. Do not change `_advance_chunk` or physics ordering.
5. Record per-domain metadata in the nested-pipeline payload:
   `sf_surface_physics`, `use_noahmp`, `noahmp_static_loaded`,
   `noahmp_land_seeded`, `noahmp_n_land_cells`, and provenance path.
6. If the namelist selects an unsupported land option, fail closed with a clear
   error rather than silently falling back to frozen prescribed land.

Be careful: existing CPU setup-only logic in `run_one_case_v0120.py` is not a
full device/carry construction proof because `State.zeros` currently requires a
visible GPU. For CPU-only proof, either write a static/carry-shape construction
probe that avoids the GPU-only constructor or clearly document the limitation and
prove every part that can be CPU-proven. The manager will run GPU gates after
review.

## Acceptance Criteria

The worker may mark this sprint ready for manager review only if all are true:

- Source diff is narrow and isolated to the nested Noah-MP wiring path.
- If `sf_surface_physics=4`, nested bundles carry `use_noahmp=True` and a
  non-null Noah-MP static/land seed for every domain that uses option 4.
- The initial carry seeding path is structurally stable across segments; no
  `None -> NoahMPLandState` promotion can occur inside a JAX scan.
- The writer-side diagnostics see the active Noah-MP carry so `TSK`, `T2`,
  `HFX`, and `LH` are sourced from the evolved land state over land.
- Unsupported land-surface configs fail closed with a diagnostic reason.
- No host/device transfer is added inside timestep loops.

## Required Validation Commands

Run from the isolated worktree. Use CPU-only where applicable:

```bash
python -m py_compile \
  src/gpuwrf/integration/nested_pipeline.py \
  src/gpuwrf/runtime/domain_tree.py
```

```bash
PYTHONPATH=src pytest -q tests/test_v013_tost_wrfbdy_fix.py
```

Add and run at least one focused test/proof that verifies, without a long GPU
forecast, that a real Canary L2 nested case with `sf_surface_physics=4` builds
per-domain namelists/carry seeds with Noah-MP active. The proof must write:

```text
proofs/v014/noahmp_nested_pipeline_activation.{json,md}
```

That proof should include at minimum:

- selected run id and run dir;
- per-domain `sf_surface_physics`;
- `use_noahmp`;
- whether `noahmp_static`, `noahmp_energy_params`, `noahmp_rad_params`, and
  `noahmp_land` are non-null;
- `n_land_cells`;
- initial land-mean `TSK`;
- whether this proof avoided GPU use or required a GPU-visible construction.

If a very short CPU-only one-step probe is possible without violating the GPU
lock, add it. If not possible because this architecture requires GPU-backed
`State`, say so in the proof and leave the short GPU h1-h4 gate to the manager.

## Manager GPU Gates After Review

The manager, not the worker, will run these after merge if CPU/source proof is
accepted:

1. Exact-branch memory preflight with Noah-MP active on nested path.
2. Canary d02 h1-h4 GPU gate:
   - d02 land-mean `TSK` bias `<= 2 K` at h2-h4,
   - d02 land `HFX` bias `<= 40 W/m2` at h2-h4,
   - no unacceptable regression in the 17-field RMSE table.
3. Full Canary d02 72h rerun.
4. Switzerland/Gotthard 72h GPU run only after Canary short gate is green or
   formally bounded.

## Proof Object

Produce:

- `.agent/sprints/2026-06-10-v014-noahmp-nested-pipeline/worker-report.md`
- `proofs/v014/noahmp_nested_pipeline_activation.{json,md}`
- a git commit on the worker branch, or an explicit `NO_SOURCE_FIX` report if
  the source cannot be safely fixed with current evidence.

## Handoff Requirements

The report must include:

- objective;
- files changed;
- commands run and return codes;
- proof objects produced;
- exact commit hash;
- unresolved risks;
- whether GPU gates can start after manager merge.

Completion marker to manager pane `0:2`:

```bash
tmux send-keys -t 0:2 'FABLE NOAHMP_NESTED_PIPELINE DONE - see .agent/sprints/2026-06-10-v014-noahmp-nested-pipeline/worker-report.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
