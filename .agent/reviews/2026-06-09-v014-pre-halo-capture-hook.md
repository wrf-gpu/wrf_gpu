# Review: V0.14 JAX Pre-Halo Capture Hook

Verdict: `HOOK_GREEN_COMPARE_BLOCKED_NO_JAX_H10_PRESTEP_CARRY`.

Objective: add and prove a default-off JAX hook for the final RK3 post-refresh, pre-halo state.

Files changed:
- `src/gpuwrf/runtime/operational_mode.py`
- `proofs/v014/jax_pre_halo_capture.py`
- `proofs/v014/jax_pre_halo_capture.json`
- `proofs/v014/jax_pre_halo_capture.md`
- `.agent/reviews/2026-06-09-v014-pre-halo-capture-hook.md`
- `tests/test_v014_pre_halo_capture.py`

Result:
- The hook is private/proof-only and default-off.
- The disabled RK path returns `OperationalCarry`, not an auxiliary capture tuple.
- The capture path returns the same normal carry plus the final RK3 pre-halo `State`.
- A same-surface h10 comparison is still blocked by missing JAX h10 pre-step carry/checkpoint.

Unresolved risks:
- No first numerical JAX operator mismatch is named by this sprint.
- The accepted WRF green patch is available, but the JAX h10 input/carry is not.

Next decision: provide/build the JAX h10 pre-step carry checkpoint or open a source-fix sprint only after a same-surface mismatch is emitted.
