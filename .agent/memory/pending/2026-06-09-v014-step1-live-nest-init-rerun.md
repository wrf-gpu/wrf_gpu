# V0.14 Step-1 Live-Nest Init Rerun

Date: 2026-06-09

Verdict: `STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_CLOSED_NEXT_T`.

`proofs/v014/step1_live_nest_init_rerun.*` reran the strict d02 Step-1
same-input WRF-vs-JAX comparison with native live-nest child base initialization
semantics mirrored in the CPU-only proof loader. The accepted WRF truth npz was
reused:
`/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`.

Base residuals are now closed:

- `MUB` max_abs `0.05002361937658861`, RMSE `0.008025019829604947`
- `PB` max_abs `0.05357326504599769`, RMSE `0.004296943965085442`
- `PHB` max_abs `0.10811684231157415`, RMSE `0.02459295000326211`

Remaining strict-comparison residuals are dynamic/operator work:

- first divergent schema field: `T`
- largest max_abs field: `P`, max_abs `1561.2503728885986`, RMSE
  `305.9413510899027`
- also material: `PH`, `MU`, `W`, `U`, `V`

Do not reopen base-init debugging unless a later proof contradicts this. The
next v0.14 debug sprint should localize Step-1 operator/source boundaries for
`T/P/PH/MU`. TOST, Switzerland, FP32, and memory follow-ups remain paused behind
the grid-parity gate.
