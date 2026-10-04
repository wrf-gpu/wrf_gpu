# BENCH — one-command PROD benchmark (gate S1)
Usage: `scripts/bench_prod.sh [--hours 3] [--cold] [--arms N] [--nsys] [--extra-hours H] [--census-arm] [--xla-flags F] [--tag T] [--src REPO] [--rev REF] [--print-src]`
- Harness: bench-owned `scripts/bench/s0_nested_harness.py` (real `gpuwrf.cli run`, d01+d02, PROD case
  `<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725`, read-only). `--src REPO --rev REF` (default `HEAD`) measure an
  IMMUTABLE commit: a detached worktree snapshot is created/reused at `$BENCH_SNAP_ROOT/<rev12>` and imports come
  ONLY from it (`REPO=$BENCH_SRC_REPO`). rev + `src/gpuwrf` hash key the cache and are recorded in `bench.json`.
  Refuses a dirty repo `src/gpuwrf` or a missing launch path; `--print-src` prints paths + `DIRTY`, no GPU.
  `--xla-flags F` exports `XLA_FLAGS=F` for every arm AFTER s0_env.sh (which unsets it); recorded in bench.json + receipts.
- EACH ARM takes the GPU lock itself (`with_gpu_lock --label bench` + `taskset -c $BENCH_CPUS`, default 12,13,28,29 — the
  only wrf_gpu2 cores; needs lane lease), so short probes slip in between arms. Do NOT pre-wrap this script in
  with_gpu_lock (a nested flock would deadlock; the script refuses). GPU arms nice 10, CPU work nice 19, OMP_NUM_THREADS=2.
  `S0_MIN_FREE_MIB` guards free VRAM (default 26000; the desktop holds ~3.6 GB, so 28672 refuses).
- Output `$BENCH_ROOT/<tag>/{cold,warm1..,nsys}/` + `bench.json`; cache `…/cache/<src_tree12>` (path printed).
  `$BENCH_ROOT` default `<USER_HOME>/wrf_gpu2_lanes/bench` (disk). Disk only: never `/tmp` (RAM tmpfs) or `<DATA_ROOT>`;
  `TMPDIR=$S/tmp`. `--cold` needs that cache empty; warm needs it filled.
- GPU-arm wall cap (AGENTS.md): warm/nsys 1800 s; cold 3000 s (cold compile exempt; raise `ARM_TIMEOUT_COLD` for a
  first cold-setup arm). `--extra-hours H` adds one extra WARM arm of H hours (`extra${H}h`, wrfout kept for scoring).
  Timing arms always run `GPUWRF_CENSUS=0` (production default; census is NOT bit-transparent on the real d01 step).
  `--census-arm` adds one WARM arm with `GPUWRF_CENSUS=1` (named `census`) whose manifest/counts are attached to
  bench.json (F2). Neither the extra nor the census arm enters the A/A headline. `--cold` = cold + two warm arms.
How to read `bench.json`: `s1_warm_s_per_fc_h` = mean over warm arms of the mean s/fc-h of root segments 2–3 (segment 1 carries trace cost).
  Each segment is normalized by its OWN `root_steps`: seg1/seg2 are 67 steps = 3618 s model time; the FINAL segment is
  66 steps = 3564 s (a 3 h run ends one root step short) — so per-fc-h is not 3×3600.
- `headline_valid=false` + `reject_reasons` + `s1_warm_s_per_fc_h=null` when arms are incomplete or span different src
  trees: no headline on bad/incompatible evidence. Per arm/segment: `stepping`, `output`, `output_compile` ([M] timers;
  compile attributed only on a real interval overlap with an output call), `host_idle_est` ([I]), `vram_peak_mib_nvidia_smi`,
  `jax_memory_stats`, `src` (`path`,`rev`,`tree_hash`).
- `aa.aa_floor_pct` = |warm1−warm2|/warm1 vs historical 6.77 % (FINDINGS A5) / main's contention floor 13 % (FINDINGS A1b).
- `nsys` (with `--nsys`): `per_step_class_kernels_htod_dtoh` = kernels + HtoD/DtoH/DtoD per d01/d02 step class; `gpu_idle_frac_in_segment` uses one window (`device_span_s` = same op set as `device_union_s`).
Noise: single GPU shared with the desktop (~3.6 GB VRAM); ALISIOS owns cores 0-11 + SMT 16-27, so wrf_gpu2 is limited to 12,13,28,29.  Test: `pytest --basetemp <USER_HOME>/wrf_gpu2_lanes/bench/pytest tests/test_bench_summary.py`.
