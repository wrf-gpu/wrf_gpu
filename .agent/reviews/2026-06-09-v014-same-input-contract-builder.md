# V0.14 Same-Input Contract Builder Review

## Objective

Build the missing same-input comparison contract/tooling, then rerun a strict early-step candidate comparison if technically possible.

## Files Changed

- `proofs/v014/same_input_contract_builder.py`
- `proofs/v014/same_input_contract_builder.json`
- `proofs/v014/same_input_contract_builder.md`
- `.agent/reviews/2026-06-09-v014-same-input-contract-builder.md`

No `src/gpuwrf/**` files changed. `early_step_discriminator.py/json/md` were not updated.

## Commands Run

- `git rev-parse --show-toplevel`
- `git branch --show-current`
- `git log -1 --oneline --decorate`
- `git status --short`
- `python -m py_compile proofs/v014/same_input_contract_builder.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_contract_builder.py`
- `python -m json.tool proofs/v014/same_input_contract_builder.json >/tmp/same_input_contract_builder.validated.json`
- `git diff -- src/gpuwrf`

## Proof Objects Produced

- `proofs/v014/same_input_contract_builder.json`
- `proofs/v014/same_input_contract_builder.md`

Verdict: `SAME_INPUT_CONTRACT_BLOCKED_NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH_STEP_1`.

The CPU proof-local loader constructed d02 `State`, `Tendencies`, `BaseState`/metrics, `OperationalNamelist`, and initial `OperationalCarry` without `State.zeros`. It also froze a 16-field WRF/JAX schema for `T/P/PB/PH/PHB/MU/MUB/U/V/W` plus active moisture.

## Unresolved Risks

- No strict numerical comparison ran because no full-domain WRF d02 step-1 `post_after_all_rk_steps_pre_halo` truth surface exists.
- Existing step-6000 surfaces are non-candidate and patch/tile scoped.
- The proof namelist does not load radiation/GWDO static attachments because no timestep execution was attempted.

## Next Decision

Run a disposable CPU-WRF step-1 full-domain post-RK/pre-halo hook into `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/`, convert it to the accepted npz schema, then rerun `same_input_contract_builder.py`.
