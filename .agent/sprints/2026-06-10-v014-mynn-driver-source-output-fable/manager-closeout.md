# Manager Closeout

Date: 2026-06-10 02:21 WEST
Manager: Codex
Worker: Fable/Mythos in tmux `0:1`

## Outcome

Closed as a valid hard-debug fix and frontier shift.

Verdict:

`MYNN_SOURCE_ROOT_CAUSED_INIT_QKE_FIXED_KERNEL_PROVEN_NEXT_SFCLAY_STEP1_FLUX_BOUNDARY`

The order-10 weak MYNN source-output bug was root-caused and partially fixed:
the production path now mirrors WRF's first-call `mym_initialize` level-2
equilibrium QKE initialization. The MYNN kernel itself is proven faithful when
fed WRF-equivalent boundary inputs.

The strict Step-1 gate remains red. The new blocker is the Step-1 surface-layer
flux/input boundary feeding MYNN.

## Proof Objects

- `proofs/v014/mynn_driver_source_output_fix.py`
- `proofs/v014/mynn_driver_source_output_fix.json`
- `proofs/v014/mynn_driver_source_output_fix.md`
- `proofs/v014/mynn_driver_source_output_fix_wrf_patch.diff`
- `.agent/reviews/2026-06-10-v014-mynn-driver-source-output-fix.md`
- `tests/test_v014_mynn_coldstart_init.py`
- refreshed Step-1 proof artifacts.

## Manager-Rerun Evidence

- 17 targeted MYNN/source tests passed.
- New MYNN proof reproduced:
  `MYNN_SOURCE_ROOT_CAUSED_INIT_QKE_FIXED_KERNEL_PROVEN_NEXT_SFCLAY_STEP1_FLUX_BOUNDARY`.
- Step-1 source-fidelity proof reproduced:
  `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_MYNN_DRIVER_SOURCE_OUTPUT`.
- Strict after-conv residual improved to max_abs `1497.6112512148795`, RMSE
  `13.468453371786723`, but remains far above release tolerance.

## Merge Decision:

Merge. This is a real correctness fix and a better root-cause frontier, not a
validation release. Do not start TOST, Switzerland, broad FP32, or long GPU
validation until the surface-layer boundary is fixed or explicitly bounded.

## Unresolved Risks

- Step-1 surface-layer flux boundary differs: `ustar` bias `-0.077` / max
  `0.176`, `HFX` RMSE `24.6 W/m^2`, `QFX` bias `-2.1e-5`.
- Inputs differ materially: `TSK` up to `8.3 K`, `ZNT` up to `0.97 m`.
- sfclayrev first-call semantics differ: JAX starts from `ustar=0`, WRF uses a
  first guess / `flag_iter` behavior.
- WRF's MYNN first-call path uses an uninitialized local `rmol`; deterministic
  rmol-pinned truth exists and should be the next strict target.

## Next Sprint

Open a GPT-5.5 xhigh surface-layer boundary sprint first. Endpoint:
emit a WRF step-1 surface-driver hook around `module_sf_mynn`/sfclayrev for
`TSK/ZNT/UST/HFX/QFX` in/out, port first-call `flag_iter`/UST first-guess
semantics plus skin-temperature/roughness sourcing into the JAX surface adapter,
then rerun the strict Step-1 proofs against deterministic rmol-pinned truth.

Do not spend Fable again on the surface-layer sprint unless the GPT sprint fails
to localize/fix it or leaves the method uncertain.
