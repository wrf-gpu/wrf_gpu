You are Fable/Mythos in tmux `0:1`, assigned the hard v0.14 MYNN driver source-output sprint for `/home/enric/src/wrf_gpu2`.

Read first:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/sprints/2026-06-10-v014-mynn-driver-source-output-fable/sprint-contract.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `proofs/v014/step1_source_fidelity_closure.md`

Objective:

Fix, or prove one exact blocker for, the remaining Step-1 source-fidelity divergence: JAX MYNN driver/kernel source outputs are about an order of magnitude too weak versus WRF.

Accepted proof facts:

- Verdict: `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_MYNN_DRIVER_SOURCE_OUTPUT`.
- Strict after-conv vs JAX dry `T_TENDF`: max_abs `2457.578397008898`, RMSE `21.364579991779515`.
- JAX mass-coupled MYNN `RTHBLTEN`: max_abs `260.83156991819124`.
- WRF mass-coupled `RTHBLTEN`: max_abs `2522.90576171875`.
- JAX mass-coupled qv source: max_abs `0.045505018412171354`.
- WRF `QV_TEND`: max_abs `0.4930315017700195`.
- Same-boundary scalar inputs are close: `T` max_abs `5.788684885033035e-05`, `QV` `5.969281098756885e-08`, `P` `0.0390625`.
- Radiation-held-rate and WRF `conv_t_tendf_to_moist` are secondary now.

Method:

This is the hard debug core. Use a whole endpoint plan, not micro-steps:

1. Emit one disposable WRF Step-1 MYNN driver hook around `module_bl_mynnedmf_driver`: input columns/fluxes/turbulence state before MYNN, raw `dth1/dqv1` or equivalent after `mynnedmf_post_run`, and module-em mass-scaled `RTHBLTEN/RQVBLTEN`.
2. Compare that exact boundary to JAX `_mynn_column_from_state` and `step_mynn_pbl_column` outputs. Rule in/out timestep semantics, column orientation, PBL top/masks, vertical grid, surface flux inputs, exchange coefficients/turbulence state, and source scaling constants.
3. If local, fix JAX MYNN adapter/kernel source semantics. Keep performance architecture intact: no production CPU-WRF dependency, no timestep-loop host/device transfer, no broad dycore rewrite.
4. Rerun strict Step-1 proofs. If not closed, return one exact MYNN boundary blocker and shortest next proof/fix route.

Rules:

- CPU-only unless you have a very short low-VRAM GPU sanity check that does not delay future validation; default is CPU-only.
- No Hermes/Telegram.
- Keep output compact.
- Do not edit unrelated memory/FP32/TOST/Switzerland/release docs.
- Do not commit yourself unless explicitly necessary; manager will review, gate, commit, and merge.

Deliver:

- `proofs/v014/mynn_driver_source_output_fix.py`
- `proofs/v014/mynn_driver_source_output_fix.json`
- `proofs/v014/mynn_driver_source_output_fix.md`
- `.agent/reviews/2026-06-10-v014-mynn-driver-source-output-fix.md`
- updated Step-1 proof artifacts if rerun
- focused tests if production code changes

Completion marker:

`FABLE MYNN_DRIVER_SOURCE_OUTPUT DONE - see proofs/v014/mynn_driver_source_output_fix.md`
