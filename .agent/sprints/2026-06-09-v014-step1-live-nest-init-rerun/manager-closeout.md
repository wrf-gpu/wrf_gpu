# Manager Closeout

## Outcome

The sprint is closed as a validated proof/falsifier.

Final verdict:
`STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_CLOSED_NEXT_T`.

The live-nest child base initialization path closes the previously dominant
raw-init `MUB/PB/PHB` residuals. The d02 Step-1 WRF-vs-JAX comparison still
diverges, so the v0.14 grid-parity gate remains blocked.

## Proof Objects

- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `proofs/v014/step1_live_nest_init_rerun.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-init-rerun.md`
- `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`

## Merge Decision:

Merge proof, review, sprint-closeout, roadmap, and pending-memory artifacts only.
No production model source changed in this sprint.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_live_nest_init_rerun.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_init_rerun.py`
- `python -m json.tool proofs/v014/step1_live_nest_init_rerun.json >/tmp/step1_live_nest_init_rerun.manager.validated.json`
- `git diff -- src/gpuwrf`

The rerun reproduced the same verdict. `git diff -- src/gpuwrf` remained empty.

## Key Numbers

Base residuals are closed:

- `MUB`: max_abs `0.05002361937658861`, RMSE `0.008025019829604947`
- `PB`: max_abs `0.05357326504599769`, RMSE `0.004296943965085442`
- `PHB`: max_abs `0.10811684231157415`, RMSE `0.02459295000326211`

Remaining leading residuals:

- first divergent field: `T`
- largest max_abs field: `P`, max_abs `1561.2503728885986`, RMSE
  `305.9413510899027`
- `PH`: max_abs `77.6192303625287`, RMSE `19.320745387744648`
- `MU`: max_abs `36.543234083976586`, RMSE `1.6018856311784238`

## Scope Changes

None. The sprint stayed proof-only, CPU-only, and did not touch production
source, TOST, Switzerland validation, FP32, or memory source work.

## Lessons

The live-nest base-state work was a real necessary correction, but it is not the
complete grid-parity root cause. The fastest rigorous path is now to instrument
substage/operator boundaries inside the single Step-1 proof loop, not to resume
long validation runs or keep iterating on initialization.

## Next Sprint

Open `v014-step1-t-p-operator-localization`: build or reuse strict savepoints
around Step-1 dynamics/physics/source boundaries and localize the first `T`
divergence plus the dominant `P/PH/MU` residuals. The proof gate should be a
specific operator/source boundary or a narrowly justified production fix.
