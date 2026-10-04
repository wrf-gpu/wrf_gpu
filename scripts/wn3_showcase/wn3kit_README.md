# WN3 3-nest arm kit (LW9 = the tree of all delivered twins) — built by wn3/W6/make_wn3_kit.sh

Tree: `source` -> wn3 W6/source = 9029eb3bf (main a905c640e + flags_lw9) + 691b6a51f; src tree 9e11b06b; flags
`source/scripts/bench/flags_lw9.env` (sha 84f310de). Read-only snapshot; for code fixes make your own detached snapshot
(`git worktree add --detach`, + `ln -s <USER_HOME>/src/wrf_pristine <snap>/data/wrf_pristine`) and point a copy of arm.sh at it.

Cases (symlink dirs to the server inputs, identical to the CPU twins): <USER_HOME>/wrf_gpu2_lanes/wn3/cases/{20260227,20260502,
20260614,20260220,20260608,20260120}_18z_a1. CPU-WRF reference (read-only, hourly): <DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_<case>/run/run/.
GPU frames start at the wrfinput time (tau 0 = CPU f006); the scorer pairs by valid time.

## One GPU arm (3 cases in parallel, or 1 case; 3 h ≈ 4-6 min incl. init on the warm cache)
    cat > @K@/my.env <<'X'          # your overrides, sourced AFTER flags_lw9.env (e.g. GPUWRF_SFCLAY_NATIVE_REAL=0)
    X
    env GPUWRF_GPU_ARM_CPUS=10,11,26,27 <USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh --label @LANE@ -- \
      timeout --signal=TERM --kill-after=60 1800 env ROOT=@K@ GPUWRF_CENSUS=0 ARM_ENV_OVERRIDE=@K@/my.env \
      bash @K@/parallel.sh @K@ @K@/arms/A1 3 20260220_18z_a1
- KEEP GPUWRF_GPU_ARM_CPUS=10,11,26,27 in the arm env (it enters the AOT cheap key, SI38): other values re-lower ~78 s/case.
- Any GPUWRF_* override changes the cheap key -> one cold compile (~15 min, into YOUR cache_c0), then warm.
- GPUWRF_CENSUS=1 uses @K@/cache_c1 (created cold on first use).
- Output: arms/A1/<case>/wrfout/wrfout_d0{1,2,3}_* (t0 + hourly, full 375-field schema), receipt.json (segments, outputs),
  arm_env.txt (exact env), cli_stdout.json; arms/A1/{start,end}.json, dmon.log, hostmem.log.
## Score vs CPU-WRF (CPU, lane cores, JAX_PLATFORMS=cpu)
    JAX_PLATFORMS=cpu taskset -c 8,9,12,13,24,25,28,29 nice -n 19 python3 <USER_HOME>/src/wrf_gpu2_wt/wn3/scripts/wn3_score.py \
      --cpu-dir <DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_<case>/run/run --gpu-dir <arm>/<case>/wrfout --out <arm>/<case>/d6_3h --hours 3
  -> d6_3h/dNN.json (per field/frame RMSE etc. vs the frozen D6 manifest). alisios' Q2/RH2/T2 sigma_T screen is NOT in it.
## Existing LW9 outputs you can diagnose WITHOUT a GPU (read-only, do not modify: delivered to alisios)
- <USER_HOME>/wrf_gpu2_lanes/wn3/W6/bench24_p3/20260227_18z_a1/wrfout (24 h, nccopy-deflated, values identical; 0502/0614
  outputs deleted for disk 2026-10-03 ~14:30Z after the alisios verdict; their sealed screens stay under alisios R32-31/runs)
- <USER_HOME>/wrf_gpu2_lanes/wn3/W6/twins24_p3/{20260220,20260608,20260120}_18z_a1/wrfout (24 h, nccopy-deflated, values identical)
## Known output conventions / gaps (do not chase these as physics)
- SMOIS, SH2O, TSLB, ALBEDO, EMISS, LAI, ISLTYP are written as 0 in all frames (writer gap; cadence-out fixes it) -> read land
  state from the run (receipt/proofs) or wait for the writer fix; global attrs are incomplete (PARENT_ID etc.; same fix).
- LH, GLW, PBLH, QKE are non-zero at t0 (in-step history diag); CPU has 0 there.
Questions: one line to MAILBOX 'fid-xx -> wn3'.
## Score against alisios' FROZEN twin thresholds (exact replica of their R32-31 run; CPU ~1 min/case)
    python3 <USER_HOME>/src/wrf_gpu2_wt/wn3/scripts/wn3_twin_manifest.py <issue> <arm>/<case> <hours> @K@/source <arm>/<case>/manifest.json --version-tag <tag>
    bash <USER_HOME>/wrf_gpu2_lanes/wn3/W6/r32_prerun.sh <issue> <arm>/<case>/manifest.json @LANE@_<tag>.json attr_shim   # formal after the writer fix
    python3 <USER_HOME>/wrf_gpu2_lanes/wn3/W6/r32_summary.py <USER_HOME>/wrf_gpu2_lanes/wn3/r32_copy/R32-31/@LANE@_<tag>.json
  LW9 baselines (read-only): <DATA_ROOT>/alisios/state/manager/runs/v32_sprints_20261002/R32-31/runs/GPU_SCREEN_<issue>_LW9_24h_attr_shim_v1.json
