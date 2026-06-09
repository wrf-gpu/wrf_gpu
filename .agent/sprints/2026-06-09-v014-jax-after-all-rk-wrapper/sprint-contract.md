# Sprint Contract: V0.14 JAX After-All-RK Same-State Wrapper

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Compare JAX CPU internals against the green CPU-WRF h10 target surface:
`post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.

This sprint must not fix production model code unless the manager opens a
separate source-changing contract. Its purpose is to run or build the smallest
same-state JAX wrapper that reaches the same named surface and reports the first
JAX-vs-WRF mismatch by field/operator/cadence.

## Inputs

- `proofs/v014/wrf_post_rk_refresh_localization.json`
- `proofs/v014/wrf_post_rk_refresh_localization.md`
- `proofs/v014/wrf_same_state_marker_savepoint.json`
- `proofs/v014/same_state_savepoint_request.json`
- `proofs/v014/dynamic_field_attribution.json`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

## Write Scope

Repository write scope:

- `proofs/v014/jax_after_all_rk_wrapper.py`
- `proofs/v014/jax_after_all_rk_wrapper.json`
- `proofs/v014/jax_after_all_rk_wrapper.md`
- `.agent/reviews/2026-06-09-v014-jax-after-all-rk-wrapper.md`

No production `src/` edits. No WRF source edits. No GPU unless the helper first
passes on CPU and the manager explicitly chooses a short GPU sanity probe. No
TOST, no Switzerland validation, no FP32 source landing.

## Required Work

1. Identify the JAX runtime functions and state fields corresponding to WRF's
   `after_all_rk_steps` pre-halo surface for the selected h10 patch.
2. Build the smallest CPU-only wrapper that initializes or reconstructs the
   needed JAX state for the selected h10 patch and emits comparable
   `T/P/PB/U/V/W/PH/MU/MUB` values at the named surface.
3. Compare against `proofs/v014/wrf_post_rk_refresh_localization.json`.
4. Produce a compact verdict:
   - `JAX_SURFACE_MATCH_<surface>` if the wrapper is green at the target;
   - `JAX_MISMATCH_<field_or_operator>` if a same-surface mismatch is found;
   - `WRAPPER_BLOCKED_<reason>` only with exact missing state/API/logs and the
     next command.

## Commands / Validation

At minimum, run:

```bash
python -m py_compile proofs/v014/jax_after_all_rk_wrapper.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/jax_after_all_rk_wrapper.py
python -m json.tool proofs/v014/jax_after_all_rk_wrapper.json \
  >/tmp/jax_after_all_rk_wrapper.validated.json
```

## Acceptance Criteria

- No production `src/` edits.
- JSON validates and records the exact JAX functions/fields inspected.
- The result compares against the WRF green target, not retained wrfout alone
  and not a JAX-vs-JAX self-compare.
- The proof names the first same-surface mismatch or the exact missing wrapper
  prerequisite.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, and the next decision: source fix sprint, narrower wrapper sprint, or
debug escalation after repeated failure.
