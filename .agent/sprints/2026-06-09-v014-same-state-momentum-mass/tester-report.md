# Tester Report

## Tests Added Or Run

The worker added `proofs/v014/same_state_momentum_mass.py`, then ran the
contract-required CPU-only validation sequence. The manager also validated the
generated JSON independently.

Commands:

- `python -m py_compile proofs/v014/same_state_momentum_mass.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_state_momentum_mass.py`
- `python -m json.tool proofs/v014/same_state_momentum_mass.json >/tmp/same_state_momentum_mass.validated.json`
- `python -m json.tool proofs/v014/same_state_momentum_mass.json >/tmp/same_state_momentum_mass.manager.validated.json`

## Results

The proof ran successfully on CPU and emitted valid JSON/Markdown. Verdict:
`JAX_MISMATCH_U_post_after_all_rk_steps_pre_halo`.

First failing field:

- Field: `U`
- Surface: `post_after_all_rk_steps_pre_halo`
- Max abs: `6.292358893898424`
- RMSE: `2.032497018496295`
- Worst native key: `[4, 13]`
- JAX vs WRF: `-4.735481996086533` vs `1.55687689781189`

The table also reports large mismatches for `V`, `W`, `T`, `P`, `MU`, and base
fields. Base-field interpretation is restricted because the carry predates the
live-nest base-source partial fix.

## Fixtures Used

- WRF target proof: `proofs/v014/wrf_post_rk_refresh_localization.json`
- Existing h10 checkpoint/carry discovered by the proof helper
- Live-nest base-source proof used only for priority notes:
  `proofs/v014/live_nest_base_source_fix.json`

## Gaps

No GPU was used. No TOST or Switzerland validation was run. This proof does not
close the grid symptom; it localizes it to before RK halo/output and points the
next sprint one layer earlier inside final RK momentum/mass/theta-pressure
assembly.

Decision:

Accept the proof as a useful localization artifact. Do not accept it as a
correctness fix or full-domain parity proof.
