# Sprint Contract: V0.14 JAX Pre-Halo Capture Hook

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Add the smallest default-off CPU/proof capture hook needed to expose the JAX
state corresponding to WRF's green target:
`post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.

Then rerun a proof-local same-surface JAX comparison against
`proofs/v014/wrf_post_rk_refresh_localization.json`.

This sprint is source-changing, but it is **not** a numerical fix. It must not
change normal forecast behavior when the hook is disabled.

## Inputs

- `proofs/v014/jax_after_all_rk_wrapper.json`
- `proofs/v014/wrf_post_rk_refresh_localization.json`
- `proofs/v014/wrf_post_rk_refresh_localization.md`
- `proofs/v014/same_state_savepoint_request.json`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

## Write Scope

Allowed source write scope:

- `src/gpuwrf/runtime/operational_mode.py`

Allowed proof/test write scope:

- `proofs/v014/jax_pre_halo_capture.py`
- `proofs/v014/jax_pre_halo_capture.json`
- `proofs/v014/jax_pre_halo_capture.md`
- `.agent/reviews/2026-06-09-v014-pre-halo-capture-hook.md`
- a focused test file under `tests/` only if needed to prove the hook is
  default-off and does not alter normal returns.

No WRF source edits. No GPU. No TOST. No Switzerland validation. No FP32 source
landing. No production dycore correctness fix in this sprint.

## Required Work

1. Inspect `src/gpuwrf/runtime/operational_mode.py::_acoustic_scan` and
   `_rk_scan_step`. Confirm the target state exists immediately after
   `_carry_from_finished_stage(...)` / sharded carry halo handling and before
   `apply_halo(next_carry.state, halo_spec(...))`.
2. Add a minimal default-off capture path. Prefer a proof/debug API that keeps
   normal forecast entry points byte/behavior identical when disabled. The hook
   may return an auxiliary captured `State` from a proof-only function or expose
   a small helper, but must not add host/device transfers to normal timestep
   loops.
3. Write `proofs/v014/jax_pre_halo_capture.py` to exercise the hook on CPU and
   compare the emitted `T/P/PB/U/V/W/PH/MU/MUB` patch against Boole's WRF green
   target where feasible.
4. If the full h10 CPU replay is too expensive or missing prerequisite state,
   still prove the hook captures the correct cadence on a smaller CPU fixture
   and emit `HOOK_GREEN_COMPARE_BLOCKED_<reason>` with the exact missing input
   needed for the h10 compare. Do not fake a same-surface comparison.
5. Produce a compact verdict:
   - `JAX_MISMATCH_<field_or_operator>` if the same-surface comparison runs and
     finds a mismatch;
   - `JAX_SURFACE_MATCH_after_all_rk_pre_halo` if it is green for the selected
     patch;
   - `HOOK_GREEN_COMPARE_BLOCKED_<reason>` if the hook is proven but the h10
     compare still lacks concrete state/input.

## Commands / Validation

At minimum, run:

```bash
python -m py_compile src/gpuwrf/runtime/operational_mode.py proofs/v014/jax_pre_halo_capture.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/jax_pre_halo_capture.py
python -m json.tool proofs/v014/jax_pre_halo_capture.json \
  >/tmp/jax_pre_halo_capture.validated.json
```

Also run the most focused existing tests covering operational mode or replay
entry points that can complete on CPU without starting a long forecast.

## Acceptance Criteria

- Normal forecast APIs keep the same return type and behavior when the hook is
  disabled.
- No GPU, WRF, TOST, Switzerland, or FP32 work.
- JSON validates and records the exact source function/cadence captured.
- The proof clearly distinguishes a successful hook from a successful h10
  same-surface numerical comparison.
- If a mismatch is found, the result names the first field/operator/cadence
  without launching a source fix in the same sprint.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, and next decision: source fix sprint, narrower capture/wrapper sprint,
or debug escalation after repeated failure.
