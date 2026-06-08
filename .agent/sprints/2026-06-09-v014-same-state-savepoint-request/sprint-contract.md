# Sprint Contract: V0.14 Same-State Savepoint Request Manifest

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Convert the accepted dynamic attribution output into a compact, exact
CPU-WRF savepoint request manifest for the first same-state localization run.

This sprint does not instrument WRF, does not compare JAX terms, and does not
edit production source. It packages the selected lead/cells/levels/patches and
requested term groups so a WRF instrumentation worker can write exactly the
needed savepoints without rediscovering the debug target.

## Inputs

- `proofs/v014/dynamic_field_attribution.json`
- `proofs/v014/dynamic_field_attribution.md`
- `proofs/v014/same_state_tendency_localization_plan.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

## Write Scope

- `proofs/v014/same_state_savepoint_request.py`
- `proofs/v014/same_state_savepoint_request.json`
- `proofs/v014/same_state_savepoint_request.md`
- `.agent/reviews/2026-06-09-v014-same-state-savepoint-request.md`

No `src/` edits. No WRF edits. No GPU.

## Required Content

The request manifest must include:

- run id, selected domain, selected lead, selected valid time, and source proof
  hash or file metadata;
- the 24 selected mass-grid cells from Helmholtz, with native U/V/W/PH stagger
  context and patch bounds;
- recommended vertical levels plus a clear statement that full columns are
  required for stencil/vertical-coupling terms;
- requested RK stages and acoustic substep samples for the first pass;
- requested WRF source term groups: stage input, mass coupling, momentum
  advection, scalar/theta/mu advection, diffusion, horizontal PGF, Coriolis,
  source-tendency folding, small-step prep, acoustic U/V, MU/theta, W/PH,
  pressure/rho refresh, boundary/spec-relax, final stage state;
- compact output artifact schema expected from WRF savepoints;
- a concise Markdown summary with no large tables.

## Commands

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 \
  python proofs/v014/same_state_savepoint_request.py
python -m json.tool proofs/v014/same_state_savepoint_request.json \
  >/tmp/same_state_savepoint_request.validated.json
python -m py_compile proofs/v014/same_state_savepoint_request.py
```

## Acceptance Criteria

- Script exits 0 CPU-only.
- JSON validates and contains exactly 24 selected cells.
- The manifest is sufficient for a WRF instrumentation worker to implement the
  first h10 same-state savepoint without rereading broad comparator outputs.
- No equivalence or root-cause claim is made.

## Closeout

Close with proof paths, commands run, selected lead/cell count, requested term
groups, and any missing dependency on Sartre's WRF source/build feasibility.
