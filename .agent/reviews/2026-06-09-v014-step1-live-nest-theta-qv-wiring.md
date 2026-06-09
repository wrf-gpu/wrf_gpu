# Review: V0.14 Step-1 Live-Nest Theta/QV Production Wiring

Verdict: `STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD`.

Findings:
- HIGH: `build_replay_case` now calls `_wrf_live_nest_transient_adjust_mub` and `_wrf_live_nest_adjust_tempqv` inside the `live_nest_parent` branch (production_wired=`True`); the corrected QVAPOR replaces the raw load.
- HIGH: Production helper theta vs same-boundary WRF pre-call truth max_abs `5.788684885033035e-05` K (gate 0.001 K; closes=`True`); QVAPOR max_abs `5.970267497393267e-08`.
- HIGH: Transient adjust-base MUB matches the WRF adjust hook within `4.521e-04` Pa; final BaseState MUB unchanged (target delta `4.648e-03` Pa, domain max_abs `0.05002361937658861` Pa).
- MEDIUM: Harness mirror reproduces the production helper output exactly (theta/qv max_abs `0.0` / `0.0`).
- MEDIUM: Step-1 16-field comparison first divergent field = `T`, largest residual = `P`.

Evidence:
- WRF adjust hook: `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`
- Same-boundary WRF pre-call truth: `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`
- Step-1 truth NPZ: `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`

objective: wire WRF theta_m + adjust_tempqv into production live-nest init and run the next Step-1 comparison.

files changed:
- `src/gpuwrf/integration/d02_replay.py`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `proofs/v014/step1_live_nest_theta_qv_wiring.json`
- `proofs/v014/step1_live_nest_theta_qv_wiring.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md`

commands run:
- `python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_qv_wiring.py`
- `python -m json.tool proofs/v014/step1_live_nest_theta_qv_wiring.json >/tmp/step1_live_nest_theta_qv_wiring.validated.json`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_m7_l2_d02_replay.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py`
- `git diff --stat`

unresolved risks:
- Production live-nest theta_m + adjust_tempqv init closes vs WRF pre-call truth (theta max_abs 5.788684885033035e-05 K, qv max_abs 5.970267497393267e-08).
- Step-1 16-field comparison still divergent; first divergent (schema order) = T; largest residual = P max_abs 974.9820434775493.
- build_replay_case calls State.zeros (GPU-only), so the CPU proof exercises the exact production helpers it consumes plus a static wiring check, not the full GPU build_replay_case object.
- The Step-1 16-field comparison is post-RK/pre-halo; residuals after init closure name a field-level symptom (next operator), not yet the exact dycore/physics operator.

next decision needed: Run the next operator-localization sprint at Step-1 field P (worst cell {'i': 1, 'j': 30, 'k': 1}, boundary band True).
