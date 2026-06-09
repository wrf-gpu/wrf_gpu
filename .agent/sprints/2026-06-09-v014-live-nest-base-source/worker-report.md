# Worker Report

Summary:

GPT implemented a parent-aware live-nest child initialization path for d02. The
source path passes the already-loaded parent case into child `build_replay_case`,
interpolates parent terrain to the child grid with the WRF SINT host reference,
applies WRF-style `blend_terrain`, recomputes `PB/MUB/PHB/T_INIT/ALB`, and
rebuilds metrics before timestep ownership. CPU-WRF h0 is used only as a
validation oracle.

Files Changed:

- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/integration/nested_pipeline.py`
- `proofs/v014/live_nest_base_source_fix.py`
- `proofs/v014/live_nest_base_source_fix.json`
- `proofs/v014/live_nest_base_source_fix.md`
- `.agent/reviews/2026-06-09-v014-live-nest-base-source-fix.md`

Commands Run:

- `python -m py_compile src/gpuwrf/integration/d02_replay.py src/gpuwrf/integration/nested_pipeline.py src/gpuwrf/nesting/interp.py src/gpuwrf/nesting/boundary_construction.py proofs/v014/live_nest_base_source_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/live_nest_base_source_fix.py`
- `python -m json.tool proofs/v014/live_nest_base_source_fix.json >/tmp/live_nest_base_source_fix.validated.json`

Proof Objects:

- `proofs/v014/live_nest_base_source_fix.json`
- `proofs/v014/live_nest_base_source_fix.md`
- `.agent/reviews/2026-06-09-v014-live-nest-base-source-fix.md`

Risks:

- The base-state mismatch is closed, but V10/grid-field divergence is not proven
  reduced or closed.
- Initial dynamic perturbation residuals remain visible: P `33.4765625` Pa and
  MU `12.2550048828125` Pa on the target patch.
- Total-state target-patch deltas improve substantially despite those residuals:
  P_TOTAL `1080.4921875` -> `33.43062101097894` Pa, MU_TOTAL
  `1038.0496826171875` -> `12.299452038438176` Pa, PH_TOTAL `878.0291748046875`
  -> `0.09377109122578986`.
- SINT is host-side at initialization only. It does not add timestep-loop
  transfers, but a later startup-performance sprint may port it if startup cost
  matters.

Handoff:

Manager should accept this only as a scoped base-state source fix. The next gate
is the init-override/direct grid-field proof plus same-state momentum/mass
tendency localization before any TOST resume.
