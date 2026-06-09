# V0.14 Same-Input Contract Builder

Date: 2026-06-09

Verdict:
`SAME_INPUT_CONTRACT_BLOCKED_NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH_STEP_1`.

What changed:

- `proofs/v014/same_input_contract_builder.py` now builds the initial d02
  same-input CPU/JAX object contract without visible GPU and without
  `State.zeros`.
- The proof constructs `State`, `Tendencies`, `BaseState`/metrics,
  `OperationalNamelist`, parent-boundary package, and initial
  `OperationalCarry`.
- The WRF/JAX schema is frozen for `T`, `P`, `PB`, `PH`, `PHB`, `MU`, `MUB`,
  `U`, `V`, `W`, `QVAPOR`, `QCLOUD`, `QRAIN`, `QICE`, `QSNOW`, and `QGRAUP`.
- No production `src/gpuwrf/**` source changed.

What remains:

- No strict WRF-vs-JAX comparison has run.
- Missing exact truth artifact:
  `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`.
- Next sprint should patch/run a disposable CPU-WRF tree to emit full-domain d02
  step-1 `post_after_all_rk_steps_pre_halo` arrays, then rerun the contract
  builder for the first strict per-field residual table.

Manager note:

Do not resume TOST or Switzerland validation from this proof. It only removes
the CPU-loader/schema blocker. It does not prove or fix the dynamic divergence.
