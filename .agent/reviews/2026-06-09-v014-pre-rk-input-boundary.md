# Review: V0.14 Pre-RK Input Boundary

Verdict: `PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED`.

objective: produce explicit WRF and JAX step-6000 pre-RK input-boundary truth for h10 d02 over T/P/PB/MU/MUB and decide whether the produced JAX step-5999 carry is already wrong before current-step physics/RK.

files changed:
- `proofs/v014/pre_rk_input_boundary.py`
- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/pre_rk_input_boundary.md`
- `proofs/v014/pre_rk_input_boundary_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-pre-rk-input-boundary.md`

commands run:
- `python -m py_compile proofs/v014/pre_rk_input_boundary.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/pre_rk_input_boundary.py`
- `python -m json.tool proofs/v014/pre_rk_input_boundary.json >/tmp/pre_rk_input_boundary.validated.json`
- `mkdir -p /tmp/wrf_gpu2_v014_pre_rk_input_boundary`
- `rsync -a /mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF/ /tmp/wrf_gpu2_v014_pre_rk_input_boundary/WRF/`
- `rsync -a /mnt/data/wrf_gpu2/v014_post_rk_refresh/run_case3/ /tmp/wrf_gpu2_v014_pre_rk_input_boundary/run_case3/`
- `cd /tmp/wrf_gpu2_v014_pre_rk_input_boundary/WRF && patch -p1 < /home/enric/src/wrf_gpu2/proofs/v014/pre_rk_input_boundary_wrf_patch.diff || true  # hunk 2 applies; hunk 1 is declarations-only against pristine`
- `insert wrfgpu2_prerk_* declarations next to existing wrfgpu2_marker_* declarations in scratch dyn_em/solve_em.F`
- `cd /tmp/wrf_gpu2_v014_pre_rk_input_boundary/WRF && timeout 3600 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH NETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build PNETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build WRFIO_NCD_LARGE_FILE_SUPPORT=1 tcsh ./compile em_real > /tmp/wrf_gpu2_v014_pre_rk_input_boundary/compile_pre_rk_input_boundary_dmpar.log 2>&1`
- `cd /tmp/wrf_gpu2_v014_pre_rk_input_boundary/run_case3 && find . -maxdepth 1 \( -name 'rsl.error.*' -o -name 'rsl.out.*' -o -name 'wrfout_d0*' -o -name 'wrfrst_d0*' -o -name '*stdout.log' \) -delete`
- `ln -sf /tmp/wrf_gpu2_v014_pre_rk_input_boundary/WRF/main/wrf.exe /tmp/wrf_gpu2_v014_pre_rk_input_boundary/run_case3/wrf.exe`
- `cd /tmp/wrf_gpu2_v014_pre_rk_input_boundary/run_case3 && timeout 3600 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 WRFGPU2_PRE_RK_INPUT=1 WRFGPU2_PRE_RK_INPUT_ROOT=/tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output WRFGPU2_PRE_RK_INPUT_GRID=2 WRFGPU2_PRE_RK_INPUT_START_STEP=6000 WRFGPU2_PRE_RK_INPUT_END_STEP=6000 mpirun --oversubscribe -np 28 ./wrf.exe > /tmp/wrf_gpu2_v014_pre_rk_input_boundary/run_case3/pre_rk_input_boundary_28rank_stdout.log 2>&1`
- `cd /tmp/wrf_gpu2_v014_pre_rk_input_boundary/run_case3 && timeout 20 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 WRFGPU2_PRE_RK_INPUT=1 WRFGPU2_PRE_RK_INPUT_ROOT=/tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output WRFGPU2_PRE_RK_INPUT_GRID=2 WRFGPU2_PRE_RK_INPUT_START_STEP=6000 WRFGPU2_PRE_RK_INPUT_END_STEP=6000 ./wrf.exe 2>&1 | tee /tmp/wrf_gpu2_v014_pre_rk_input_boundary/run_case3/pre_rk_input_boundary_singleton_stdout.log`

proof objects produced:
- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/pre_rk_input_boundary.md`
- `proofs/v014/pre_rk_input_boundary_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-pre-rk-input-boundary.md`

unresolved risks:
- Only the selected h10 d02 mass patch was compared; broader field coverage remains a follow-up.
- WRF truth is source-hook output, not retained wrfout inspection.

next decision needed: Trace the JAX checkpoint/prestep carry producer and previous-step WRF/JAX update path.
