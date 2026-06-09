# Review: V0.14 Step-1 Same-Input Truth

Verdict: `STEP1_SAME_INPUT_COMPARISON_EXECUTED_FIRST_DIVERGENT_T`.

objective: produce full-domain CPU-WRF d02 step-1 post-RK/pre-halo truth and run the strict same-input JAX pre-halo comparison, or fail closed with the exact blocker.

files changed:
- `proofs/v014/step1_same_input_truth.py`
- `proofs/v014/step1_same_input_truth.json`
- `proofs/v014/step1_same_input_truth.md`
- `proofs/v014/step1_same_input_truth_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-same-input-truth.md`

commands run:
- `python -m py_compile proofs/v014/step1_same_input_truth.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_same_input_truth.py`
- `python -m json.tool proofs/v014/step1_same_input_truth.json >/tmp/step1_same_input_truth.validated.json`
- `git diff -- src/gpuwrf`
- `build_wrf: tcsh ./compile em_real`
- `run_wrf: /home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin/mpirun --oversubscribe -np 28 /mnt/data/wrf_gpu2/v014_step1_same_input_truth/run/wrf.exe`

proof objects produced:
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_same_input_truth.json`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_same_input_truth.md`
- `/home/enric/src/wrf_gpu2/proofs/v014/step1_same_input_truth_wrf_patch.diff`
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-same-input-truth.md`
- `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`

unresolved risks:
- This is the first strict full-domain step-1 comparison; residuals name the first divergent field but do not localize the responsible operator.
- The disposable WRF tree inherits prior v014 scratch WRF hook scaffolding; the proof patch diff records only the added step-1 hook block.

next decision: If comparison executed, localize the first divergent field one operator earlier; if blocked, apply the exact blocker patch/tool named in this JSON.
