# Worker Report

## Summary:

Built the proof-local same-input contract builder and closed it fail-closed with
verdict `SAME_INPUT_CONTRACT_BLOCKED_NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH_STEP_1`.

The CPU proof path now constructs the d02 `State`, `Tendencies`,
`BaseState`/metrics, `OperationalNamelist`, and initial `OperationalCarry`
without calling `State.zeros` or requiring a visible GPU. It also freezes the
WRF/JAX field schema for `T`, `P`, `PB`, `PH`, `PHB`, `MU`, `MUB`, `U`, `V`,
`W`, `QVAPOR`, `QCLOUD`, `QRAIN`, `QICE`, `QSNOW`, and `QGRAUP`.

## Files Changed

- `proofs/v014/same_input_contract_builder.py`
- `proofs/v014/same_input_contract_builder.json`
- `proofs/v014/same_input_contract_builder.md`
- `.agent/reviews/2026-06-09-v014-same-input-contract-builder.md`

No production `src/gpuwrf/**` files were changed.

## Commands Run

- `python -m py_compile proofs/v014/same_input_contract_builder.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_contract_builder.py`
- `python -m json.tool proofs/v014/same_input_contract_builder.json >/tmp/same_input_contract_builder.validated.json`
- `git diff -- src/gpuwrf`

## Proof Objects Produced

- `proofs/v014/same_input_contract_builder.json`
- `proofs/v014/same_input_contract_builder.md`

## Unresolved Risks

No strict numerical comparison ran because no full-domain CPU-WRF d02 step-1
truth surface exists at `post_after_all_rk_steps_pre_halo`. Existing step-6000
surfaces are patch/tile scoped and are intentionally rejected as weak inputs for
the same-input contract.

## Next Decision

Run a disposable CPU-WRF step-1 full-domain hook that writes the accepted npz
truth contract, then rerun `proofs/v014/same_input_contract_builder.py`.
