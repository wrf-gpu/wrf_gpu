# Manager Closeout

## Outcome

The sprint is closed as a validated strict comparison proof.

Final verdict:
`STEP1_SAME_INPUT_COMPARISON_EXECUTED_FIRST_DIVERGENT_T`.

This is the first successful full-domain same-input WRF-vs-JAX comparison for
d02 step 1 at `post_after_all_rk_steps_pre_halo`.

## Proof Objects

- `proofs/v014/step1_same_input_truth.py`
- `proofs/v014/step1_same_input_truth.json`
- `proofs/v014/step1_same_input_truth.md`
- `proofs/v014/step1_same_input_truth_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-same-input-truth.md`
- `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`

## Merge Decision:

Merge proof, review, sprint-closeout, roadmap, and memory artifacts only. No
production model source changed in this sprint.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_same_input_truth.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_same_input_truth.py`
- `python -m json.tool proofs/v014/step1_same_input_truth.json >/tmp/step1_same_input_truth.manager.validated.json`
- `git diff -- src/gpuwrf`

The re-run reproduced the same verdict. `git diff -- src/gpuwrf` remained empty.

## Key Numbers

- First divergent schema field: `T`.
- Largest residuals:
  - `MUB`: max_abs `2635.640625`, RMSE `98.13000038547803`
  - `PB`: max_abs `2627.3828125`, RMSE `47.826296821589736`
  - `PHB`: max_abs `2237.9423828125`, RMSE `45.35253861292826`
  - `P`: max_abs `1561.1123921205437`, RMSE `305.75054216524205`
  - `PH`: max_abs `77.61962727618265`, RMSE `19.316800336111022`

## Lessons

The debug ladder has crossed from tooling blockers to a real full-domain
first-divergence proof. The dominant base/mass residuals mean the next sprint
should target live-nest child base-state initialization or falsify that path
with an init-override comparison. A late RK-only fix would be premature.

## Next Sprint

Open a source/falsifier sprint that applies or verifies the native live-nest
base-state initialization path, then reruns this exact step-1 comparison. TOST,
Switzerland, FP32, and memory cleanup remain paused behind this grid-parity gate.
