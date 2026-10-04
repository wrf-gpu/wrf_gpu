# Cross-domain parallel-compile — GPU A/B recipe (manager runs on the 5090)

Branch: `worker/opus/vnext-parallel-compile`. Shared warm cache dir =
version-keyed `<DATA_ROOT>/gpuwrf_jax_cache` (do NOT blow away between A/B legs;
only clear the per-version subdir to force a cold leg). Canary 9-nest 20240901,
de-fuse path (`GPUWRF_NESTED_FUSE=0`).

## Knobs added
- `GPUWRF_NESTED_PARALLEL_COMPILE` — `0`=opt-out (sequential, today), `N`=worker
  count override, unset=auto `min(n_domains, cpu, RAM//28)` capped 4.
- `GPUWRF_JAX_CACHE_LOCK` — default ON when cache enabled; engages JAX's per-dir
  cross-process FileLock + the atomic temp+rename writer. `=0` = legacy unlocked.
- `GPUWRF_JAX_CACHE_LOCK_TIMEOUT` — FileLock acquire timeout (default 10s).

## Stderr markers bracketing the cold-compile wall
`[parallel-compile] PREWARM_START de-fuse nest`
`[parallel-compile] PREWARM_DONE active=.. source=.. workers=.. wall_s=.. warm_all=.. error=..`
Plus `meta['parallel_compile']['nested_precompile']` in the run meta JSON
(per-domain `compile_seconds`, `wrote_entry`, fallbacks).

## STEP 0 — baseline cold A (sequential)
Clear ONLY the version subdir. Run de-fuse to first output:
```
GPUWRF_NESTED_FUSE=0 GPUWRF_NESTED_PARALLEL_COMPILE=0 \
  <nested_pipeline run, canary 9-nest 20240901, 1 output interval>
```
Record wall-to-first-output (~50 min expected), peak host-RAM, peak VRAM. Save wrfout_A.

## STEP 1 — parallel sweep B
For `P in 1 2 3 4 9`: clear ONLY the version subdir, then:
```
GPUWRF_NESTED_FUSE=0 GPUWRF_NESTED_PARALLEL_COMPILE=$P GPUWRF_JAX_CACHE_LOCK=1 \
  <same forecast>
```
Record wall-to-first-output, peak host-RAM (RSS sampler / `/usr/bin/time -v`),
peak VRAM (`nvidia-smi --query-gpu=memory.used -lms 500`), and the PREWARM_DONE
line + `nested_precompile_report()`. Expect wall ~50 min -> ~6-12 min, plateauing
as autotune GPU-contention / host-RAM saturate. The plateau P is the recommended
default (pre-pick 4).

## STEP 2 — BIT-IDENTITY GATE (load-bearing)
Byte-compare the chosen-P parallel wrfout vs wrfout_A:
```
python proofs/v018/oom_fix/compare_wrfout_exact.py wrfout_A wrfout_P
```
ALL fields byte-identical. Parallel compile changes only WHEN/WHERE XLA
partitions the compile, never the executable bytes — MUST be 0-diff. Any diff is
a stop-ship bug (likely a prewarm spec/HLO mismatch).

## STEP 3 — warm-cache regression
Re-run the chosen-P leg WITHOUT clearing the cache -> seconds-scale warm start
(confirms the lock/atomic-writer did not break the B1 warm hit) + byte-identical.

## PASS
(a) chosen-P cold wall <= ~12 min (>=3-4x vs ~50 min sequential);
(b) peak host-RAM and VRAM bounded (no OOM; RAM ~ de-fuse per-domain * P);
(c) wrfout byte-identical to sequential de-fuse;
(d) warm re-run = seconds + byte-identical.
Report the wall-vs-P curve + peak RAM/VRAM-vs-P so the default P is data-chosen.

## CPU test results (this branch, JAX_PLATFORMS=cpu)
- `tests/test_parallel_compile.py` — 9 passed (spawn workers write distinct
  entries + main warm-hits; crashed-worker fail-open; gate no-op-when-fused /
  active-when-defused / =0-disables; default workers capped at 4; env help).
- `tests/test_b1_cache_version_keying.py` — 31 passed incl. 5 new cache-safety:
  lock-sentinel enables FileLock; concurrent spawn writers no corruption;
  locked write is a byte-identical warm hit for the default reader; atomic writer
  leaves no truncated entry on crash.
- `test_b2_compile_efficiency.py`, `test_v0130_compile_speed.py`,
  `test_v013_compile_perf2.py`, `test_v0110_domain_tree.py`,
  `test_operational_namelist_cache_key.py` — green in isolation (no regression).
- KNOWN pre-existing flake: `test_aot_warm_compile_is_cache_hit_and_identical`
  fails ONLY when run in the same process after `test_b2_compile_efficiency.py`
  (a JAX `_cache`-singleton + tmp-dir cross-test isolation issue). Reproduces
  with `GPUWRF_JAX_CACHE_LOCK=0` (all new behavior off) => NOT introduced here.
