# V0.14 Same-Input Contract Builder

Verdict: `SAME_INPUT_CONTRACT_BLOCKED_NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH_STEP_1`.

## Result

- CPU proof-local loader: `READY_CPU_INITIAL_D02_CONTRACT_WITH_JAX_PARENT_BOUNDARY_PACKAGE`.
- `State`, `Tendencies`, `BaseState`/metrics, `OperationalNamelist`, and initial `OperationalCarry` were constructed without `State.zeros`.
- Frozen field schema covers `16` WRF/JAX fields, including active moisture.
- WRF step-1 post-RK/pre-halo truth: `MISSING`.
- Strict comparison run: `False`.

## Blocker

No strict comparison ran because no full-domain WRF truth surface exists for d02 step 1 at `post_after_all_rk_steps_pre_halo`.
Existing step-6000 patch surfaces are non-candidate and tile/patch scoped.

Next decision: run a disposable CPU-WRF step-1 full-domain hook into the accepted npz truth contract, then rerun this builder.
