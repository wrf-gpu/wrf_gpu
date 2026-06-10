# Worker Report

Summary: Fable root-caused and fixed the MYNN source-output deficit; strict
Step-1 remains blocked on surface-layer flux boundary.

## Objective

Fix, or prove one exact blocker for, the remaining Step-1 MYNN driver/kernel
source-output divergence.

## Files Changed

- `src/gpuwrf/physics/mynn_pbl.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/integration/d02_replay.py`
- `proofs/v014/mynn_driver_source_output_fix.py`
- `proofs/v014/mynn_driver_source_output_fix.json`
- `proofs/v014/mynn_driver_source_output_fix.md`
- `proofs/v014/mynn_driver_source_output_fix_wrf_patch.diff`
- `.agent/reviews/2026-06-10-v014-mynn-driver-source-output-fix.md`
- `tests/test_v014_mynn_coldstart_init.py`
- refreshed Step-1 proof artifacts and `proofs/v014/same_input_contract_builder.py`.

## Outcome

Verdict:

`MYNN_SOURCE_ROOT_CAUSED_INIT_QKE_FIXED_KERNEL_PROVEN_NEXT_SFCLAY_STEP1_FLUX_BOUNDARY`

Root cause: the JAX replay/live-nest cold-start path was missing WRF
`mym_initialize` level-2 equilibrium QKE initialization. The previous taper-only
seed was orders of magnitude too small in unstable initial layers.

Implemented: `mynn_coldstart_init_columns`, exposed through
`mynn_coldstart_qke_from_state`, wired into d02 replay cold-start seeding.

## Proof Summary

- WRF driver boundary + WRF init qke: JAX kernel reproduces WRF raw `RTHBLTEN`
  with strong-cell ratio median `0.9982`, corr `1.0000`, RMSE `2.6e-06`.
- Strict Step-1 after-conv residual improved from max_abs `2457.578397008898`,
  RMSE `21.364579991779515` to max_abs `1497.6112512148795`, RMSE
  `13.468453371786723`.
- Remaining blocker is Step-1 surface-layer flux boundary (`TSK/ZNT/UST/HFX/QFX`
  and sfclayrev first-call semantics).

## Commands Run

Worker reported py_compile, focused tests, MYNN battery, proof reruns, JSON
validation, and `git diff --check` all passing.

## Next Boundary

Surface-layer driver hook around `module_sf_mynn`/sfclayrev for
`TSK/ZNT/UST/HFX/QFX` in/out; port first-call `flag_iter`/UST first-guess
semantics plus skin-temperature/roughness sourcing into the JAX surface adapter.
