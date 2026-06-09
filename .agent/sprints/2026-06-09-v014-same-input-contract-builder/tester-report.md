# Tester Report

## Decision:

Pass as a validated fail-closed tooling sprint. The proof builder runs under
CPU-only JAX, emits valid JSON/Markdown, and leaves production `src/gpuwrf/**`
untouched. It does not pass as a numerical grid-parity comparison because the
required WRF step-1 truth surface is absent.

## Commands Re-Run By Manager

- `python -m py_compile proofs/v014/same_input_contract_builder.py`
- `python -m json.tool proofs/v014/same_input_contract_builder.json >/tmp/same_input_contract_builder.manager.pre_run.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_contract_builder.py`
- `python -m json.tool proofs/v014/same_input_contract_builder.json >/tmp/same_input_contract_builder.manager.validated.json`
- `git diff -- src/gpuwrf`

## Results

- Python compilation passed.
- CPU-only proof run completed in about 8 seconds.
- Reproduced verdict:
  `SAME_INPUT_CONTRACT_BLOCKED_NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH_STEP_1`.
- JSON validation passed.
- `git diff -- src/gpuwrf` was empty.

## Coverage

The test coverage confirms the new proof-local loader can build the initial d02
same-input object graph without GPU-only zero constructors and that the schema is
materialized consistently. It does not exercise a WRF-vs-JAX residual
calculation because no accepted truth npz exists yet.

## Residual Risk

The next sprint must test the accepted npz truth path after a disposable WRF
hook emits full-domain step-1 arrays. Until that runs, no field/operator root
cause has been proven.
