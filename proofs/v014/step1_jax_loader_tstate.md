# V0.14 Step-1 JAX Loader T_STATE

Verdict: `STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`.

## Result

- CPU backend: `cpu`.
- WRF pre-call truth reused: `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`.
- First material stage/field: `raw_child_state` / `T_STATE`.
- Raw child `T_STATE = State.theta - 300 K` vs WRF pre-call: max_abs `5.490173101425171`, rmse `1.9175184863907806`.
- Live child `T_STATE` vs WRF pre-call: max_abs `5.490173101425171`, rmse `1.9175184863907806`.
- Haloed step-entry `T_STATE` vs WRF pre-call: max_abs `5.490173101425171`, rmse `1.9175184863907806`.
- Haloed step-entry interior max_abs `5.490173101425171`; boundary-band max_abs `5.284271240234375` (band width `5`).
- `T_STATE` transition max_abs raw->live `0.0`, live->boundary `0.0`, boundary->carry `0.0`, carry->halo `0.0`.
- PB raw->WRF max_abs `2627.3828125`; live->WRF max_abs `0.05357326504599769`.
- Wrong full-theta mapping has approximately 300 K bias: bias `298.9558355811426`, max_abs `300.3310546875`.

## Interpretation

- The `T_STATE` residual is already visible in raw d02 wrfinput theta and is carried unchanged through live-nest init, boundary packaging, carry construction, and the step-entry halo path.
- The live-nest stage does materially change and improve the base fields (`PB/PHB/MUB`) against WRF, so the remaining theta mismatch is the live-nest state/base semantic split, not a boundary package, carry, or halo mutation.
- The residual is not boundary-only: the interior max is material and comparable to the full-domain maximum.
- WRF `T_STATE` still maps to JAX `State.theta - 300 K`; comparing WRF perturbation theta directly to full theta produces the expected ~300 K offset.

Detailed stage and field tables are in `proofs/v014/step1_jax_loader_tstate.json`.
