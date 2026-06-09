# V0.14 WRF Same-State Marker Savepoint

Verdict: `MARKER_GREEN`.

This sprint used CPU-only WRF (`CUDA_VISIBLE_DEVICES=`, `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=1`, 28 MPI ranks). No Hermes was used. The original WRF tree at `/home/enric/src/wrf_pristine/WRF` was not patched; all WRF source edits were made in `/mnt/data/wrf_gpu2/v014_same_state_wrf/WRF`. Repo `src/` was not edited.

## Provenance

- Source WRF: `/home/enric/src/wrf_pristine/WRF`, head `f52c197ed39d12e087d02c50f412d90d418f6186`, describe `v4.7.1-dirty`.
- Disposable WRF copy: `/mnt/data/wrf_gpu2/v014_same_state_wrf/WRF`.
- Disposable run dir: `/mnt/data/wrf_gpu2/v014_same_state_wrf/run_case3`.
- Target case: `20260501_18z_l2_72h_20260519T173026Z`, domain `d02`, h10 valid time `2026-05-02_04:00:00`.
- Final disposable `wrf.exe` sha256: `96fc731fd818936e0f59c2ae745c8b4e9c24091c04b0fb7f4f4e9ee2321678dd`.
- Patch diff: `proofs/v014/wrf_same_state_marker_patch.diff`.

## Step And Index Proof

The final post marker reports `domain_id 2`, `current_timestr_before_step 2026-05-02_03:59:54`, `grid_itimestep_after_increment 6000`, `dt_seconds 6`, and `lead_seconds_after_step 36000`, mapping step 6000 to h10 `2026-05-02_04:00:00`.

Zero-based to Fortran native conversion echoed by the marker:

- selected mass cell `(y=9,x=13)` -> `(j=10,i=14)`.
- mass patch zero `y [1,18), x [5,22)` -> Fortran `j 2..18, i 6..22`.
- U patch zero `y [1,18), x_stag [5,23)` -> Fortran `j 2..18, i 6..23`.
- V patch zero `y_stag [1,19), x [5,22)` -> Fortran `j 2..19, i 6..22`.
- W/PH patch zero `kstag [0,2), y [1,18), x [5,22)`.

## Marker Verdicts

- Early h10 marker: mapping green, not same-state. Compared to its run h10 wrfout, max_abs was `T=5.702972412109375`, `P=1981.2389628887177`, `V=57.9480562210083`, `W=0.00010097026824995581`; `PB/U/PH` were roundoff or exact.
- Failed post noemit: no post files emitted because the post hook still gated on `rk_step == rk_order`; after the RK loop `rk_step=4`, `rk_order=3`.
- Post_t2 marker: green for `P/PB/U/V/W/PH`, but history `T` mismatched by max_abs `5.702972412109375` because the hook sampled `grid%t_2` (`THM`).
- Post_t1 marker: green for `P/PB/U/V/W/PH`, but history `T` mismatched by max_abs `5.702606201171875` because the hook sampled `grid%t_1`.
- Current thphy marker: green. The hook samples history-backed `T` via `grid%th_phy_m_t0` and post-RK `P/PB/U/V/W/PH`.

## Final Comparison

Post marker files: `marker_post_d2_step_6000_is_1_ie_23_js_18_je_33.txt, marker_post_d2_step_6000_is_1_ie_23_js_1_je_17.txt`.

Unique scalar counts: `T/P/PB=289`, `U=306`, `V=306`, `W/PH=578`. Duplicate overlap records: `17`, max duplicate delta `0.0`.

Against scratch h10 wrfout:

- `T/P/PB`: max_abs `0.0`.
- `U/V/W`: max_abs `8.881784197001252e-16`, `8.881784197001252e-16`, `4.440892098500626e-16`.
- `PH`: max_abs `5.329070518200751e-15`.

Against provided CPU h10 wrfout:

- `T/P/PB`: max_abs `0.0`.
- `U/V/W/PH`: max_abs `4.76837158203125e-07`, `9.5367431640625e-07`, `1.1920928977282585e-07`, `1.9073486328125e-06`.

## Logs And Proofs

- Final run stdout: `/mnt/data/wrf_gpu2/v014_same_state_wrf/run_case3/marker_run_post_thphy_28rank_stdout.log`.
- Final RSL logs: `/mnt/data/wrf_gpu2/v014_same_state_wrf/run_case3/rsl.error.*` and `rsl.out.*`.
- Final comparison JSON: `/mnt/data/wrf_gpu2/v014_same_state_wrf/compare_post_thphy_marker.json`.
- Archived comparison JSON: `/mnt/data/wrf_gpu2/v014_same_state_wrf/compare_archived_marker_runs.json`.
- Build logs: `/mnt/data/wrf_gpu2/v014_same_state_wrf/compile_dmpar*.log` and `/mnt/data/wrf_gpu2/v014_same_state_wrf/configure_dmpar.log`.
- Archived attempts: `/mnt/data/wrf_gpu2/v014_same_state_wrf/run_case3/first_28rank_early_marker_run`, `post_marker_noemit_run`, `post_t2_marker_run`, `post_t1_marker_run`, `failed_serial_rsl_before_dmpar_marker`, `aborted_single_rank_marker_run`.

## Commands

```bash
rsync -a /home/enric/src/wrf_pristine/WRF/ /mnt/data/wrf_gpu2/v014_same_state_wrf/WRF/
rsync -a /mnt/data/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z/ /mnt/data/wrf_gpu2/v014_same_state_wrf/run_case3/
env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH NETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build PNETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build WRFIO_NCD_LARGE_FILE_SUPPORT=1 sh -c "printf '34\n1\n' | ./configure > /mnt/data/wrf_gpu2/v014_same_state_wrf/configure_dmpar.log 2>&1"
timeout 5400 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH NETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build PNETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build WRFIO_NCD_LARGE_FILE_SUPPORT=1 tcsh ./compile em_real > /mnt/data/wrf_gpu2/v014_same_state_wrf/compile_dmpar.log 2>&1
timeout 1800 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH NETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build PNETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build WRFIO_NCD_LARGE_FILE_SUPPORT=1 tcsh ./compile em_real > /mnt/data/wrf_gpu2/v014_same_state_wrf/compile_dmpar_post_thphy_marker.log 2>&1
timeout 3600 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 WRFGPU2_SAMESTATE=1 WRFGPU2_SAMESTATE_ROOT=/mnt/data/wrf_gpu2/v014_same_state_wrf/marker_output WRFGPU2_SAMESTATE_GRID=2 WRFGPU2_SAMESTATE_START_STEP=6000 WRFGPU2_SAMESTATE_END_STEP=6000 mpirun --oversubscribe -np 28 ./wrf.exe > /mnt/data/wrf_gpu2/v014_same_state_wrf/run_case3/marker_run_post_thphy_28rank_stdout.log 2>&1
python - <<'PY'  # parsed marker_post_d2_step_6000_*.txt and compared to scratch/provided CPU h10 wrfout; wrote /mnt/data/wrf_gpu2/v014_same_state_wrf/compare_post_thphy_marker.json
python - <<'PY'  # parsed archived marker attempts; wrote /mnt/data/wrf_gpu2/v014_same_state_wrf/compare_archived_marker_runs.json
diff -u --label a/dyn_em/solve_em.F --label b/dyn_em/solve_em.F /mnt/data/wrf_gpu2/v014_same_state_wrf/WRF/dyn_em/solve_em.F.before_v014_marker /mnt/data/wrf_gpu2/v014_same_state_wrf/WRF/dyn_em/solve_em.F > proofs/v014/wrf_same_state_marker_patch.diff
```

## Next Sprint Recommendation

Run V0.15 dynamic localization from this green same-state point: keep the `grid%th_phy_m_t0` history-backed `T` source, add routine-boundary term-group emitters around the post-RK same-state location, and localize dynamic-field terms for `T/P/PB/U/V/W/PH` against the green marker before any GPU work.
