# v0.11.0 Recompile Fix Verification

**Verdict: FAIL.** Gate 1 fails before any hot chunk can be measured.

## Gate Results

| Gate | Result | Evidence |
|---|---:|---|
| 1. Speedup / no-recompile | FAIL | Fixed-code 3x180-step repro began chunk 1, then raised `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'Leaf'` before `END_CHUNK 1`; no hot chunk timings exist. Log: `proofs/v0110/recompile_fix_fixed_3chunks.log`. |
| 2. Bit-identity vs baseline | NOT RUN / blocked | No fixed forecast output exists because Gate 1 cannot complete the first chunk; bit-identity cannot be established. |
| 3. Default / flag-off unchanged | NOT RUN / blocked | Verification stopped after hard Gate 1 failure; no unchanged/default pass claim is made. |

## Key Evidence

- HEAD verified: `77700d9 (HEAD -> worker/gpt/recompile-diag) Fix operational chunk JIT cache reuse`.
- Fixed command used `/tmp/wrf_gpu_run.sh`, `PYTHONPATH=src`, and `taskset -c 0-27` with `JAX_LOG_COMPILES=1`.
- Fixed run reached `BEGIN_CHUNK 1 start_step=1`, traced `_advance_chunk`, then failed in `src/gpuwrf/contracts/state.py:567` while rebuilding `State` with `lu_index`.
- Control on the same HEAD and same L2 d02 fixture using `_initial_carry_for_run` completed one cold one-step `_advance_chunk` in `114.803` s. This isolates the failure to the new committed-carry path rather than the fixture or `_advance_chunk` generally.
- Static-holder unit test passed: `2 passed in 2.55s`.

## Decision

Do not merge this fix as-is. The operational committed-carry path is not runnable, so the claimed hot-chunk speedup and value preservation are unproven.
