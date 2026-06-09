# Manager Closeout

Merge Decision: accept and land blocker proof artifacts.

Objective: compare JAX CPU internals against the green WRF post-`after_all_rk_steps`
pre-halo surface. The sprint proved this cannot be done honestly through the
current runtime API because the pre-halo state is immediately haloed before any
public return path.

Accepted verdict: `WRAPPER_BLOCKED_NO_JAX_PRE_HALO_STATE_API`.

Roadmap effect: open a source-changing but non-corrective debug-hook sprint.
The allowed source change should be limited to a default-off CPU/proof capture
around `runtime/operational_mode.py::_acoustic_scan` before `apply_halo`, plus a
proof script that reruns the same-surface JAX comparison.

Manager validation:

- `python -m json.tool proofs/v014/jax_after_all_rk_wrapper.json >/tmp/jax_after_all_rk_wrapper.manager.validated.json`
- `python -m py_compile proofs/v014/jax_after_all_rk_wrapper.py`
