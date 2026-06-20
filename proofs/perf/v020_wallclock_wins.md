# v0.2.0 wall-clock easy-wins — proof record

Sprint: get the big EASY wall-clock wins for v0.2.0 (low/med risk, no kernel
rewrites, no fidelity loss). Source of truth: `.agent/reviews/2026-06-01-gpt-
wallclock-optimization.md`. Gate: each win must keep forecast outputs
NUMERICS-IDENTICAL (bit-identical, or STOP).

Base: branch `perf-wins` off `worker/opus/final-verdict` tip `135b6dc`.

## Win #1 — persistent JAX compilation cache  ✅ SHIPPED (LOW risk)

What: central import-time hook (`gpuwrf/runtime/jax_cache.py`, wired in
`gpuwrf/__init__.py`) enables JAX's on-disk persistent compilation cache at
`<DATA_ROOT>/gpuwrf_jax_cache` for every entry path. Env override
(`GPUWRF_JAX_CACHE_DIR` / standard `JAX_COMPILATION_CACHE_DIR`), kill switch
(`GPUWRF_JAX_CACHE=0`), `min_compile_time_secs=0`, graceful no-op if the dir is
unavailable.

Numerics: NEUTRAL by construction — the cache returns the IDENTICAL XLA
executable keyed by HLO+backend+flags; no float op changes. Verified the cache
activates on import and writes cache entries.

Wall-clock A/B (d02 1h segmented, fresh cache dir, cold then hot) — MEASURED on
RTX 5090, `proofs/perf/win1_cache_ab_timing.jsonl`:
- COLD (cache miss, full XLA compile): forecast compile+run = **257.7 s**
- HOT  (cache hit, fresh process reading the on-disk cache): **33.5 s**
- => the persistent cache removed **224.2 s (87%)** of cold compile on this job,
  with ZERO numerics change. The 33.5 s residual is the cache read + the actual
  GPU forecast compute (the irreducible part).

This is the per-repeat-run saving on EVERY job that reuses an already-compiled
executable: our own re-validation, and any daily run whose program signature is
unchanged. It matches the GPT estimate (80-90% on short/validation jobs; cold
compile is ~40% of the d02 24h wall, amortized less on long runs but still ~5
min). NOTE: a daily run with a DIFFERENT init clock currently still misses the
cache because `time_utc` is in the static key — that is exactly what Win #2 would
have fixed (see below).

## Win #2 — remove `time_utc` from the static JIT cache key  ⛔ STOPPED (gate fail)

What was implemented (then REVERTED from the model files): carry the two
radiation-clock scalars (`start_julian_day`, `start_utc_minute`) as DYNAMIC fp64
JAX leaves on `OperationalNamelist` instead of baking `time_utc` into static aux,
so daily runs with a different init clock reuse the same compiled executable.

Numerics evidence:
- coszen + `rrtmg_theta_tendency` + SWDOWN/GLW, called STANDALONE on the real
  d02 state, are BIT-IDENTICAL between the dynamic-scalar and legacy-static
  paths: `proofs/perf/win2_rad_kernel_diff.json` -> all max|diff| = 0.0.
- BUT the full 1h segmented forecast (radiation compiled INSIDE the 360-step
  `_advance_chunk` scan) diverges: `proofs/perf/win2_determinism_and_hlo.json`
  -> dynamic-vs-legacy max|diff| = 4.46 Pa on `p_perturbation`, while the SAME
  dynamic path run 3x is bit-identical (determinism floor = 0.0).

Diagnosis: the radiation MATH is identical; the divergence is an XLA
COMPILATION artifact — threading the clock as two extra scalar PARAMETERS (vs
baked compile-time constants) changes how XLA fuses/schedules the whole scan
body (FMA contraction / op ordering), producing fp-rounding differences that
amplify over 360 acoustic steps. Magnitude ~4e-5 relative (sub-physical), but
NOT bit-identical.

Decision: the sprint gate is explicitly BIT-IDENTICAL. Win #2 cannot meet it as
designed, so it is STOPPED and the model-file changes reverted. The forecast
remains exactly as on the base branch. (The proof scripts/JSONs are kept.)

Recommendation for a separate scoped decision (principal): the divergence is a
sub-physical fp-rounding artifact, not a physics change, and would PASS the
project's normal RMSE/operational-equivalence gate (`feedback_validation_
philosophy`: operational RMSE > bitwise parity). If the principal accepts
RMSE-equivalence (not bit-identity) for this win, Win #2 can ship and delivers
the biggest daily-cadence cache reuse (each day's different init clock would hit
the warm cache instead of recompiling). That is a gate-relaxation call, not an
easy-win, so it is deferred.

## Win #3 — double-buffered wrfout output  ✅ SHIPPED (MED risk)

What: split `write_wrfout_netcdf` into `prepare_wrfout_payload` (the device->host
pull, main thread, live state resident) + `write_prepared_wrfout` (pure host
NetCDF write). New `AsyncWrfoutWriter` (single worker thread, bounded queue=2)
writes hour N while the GPU advances hour N+1. `daily_pipeline` uses it by default
(`config.async_output`, kill switch), join()ing all writes before any wrfout is
read and on every failure/exit path.

Numerics: NEUTRAL — only the device->host pull is unchanged and the NetCDF bytes
are byte-for-byte identical to the synchronous path. Proven by
`tests/test_async_wrfout_equiv.py`: async-vs-sync wrfout variables bit-identical;
multi-hour ordering preserved; fail-closed on write error. Single writer thread
=> deterministic file ordering, no NetCDF4 concurrency hazard. `_advance_chunk`
does not donate its carry, so the async snapshot cannot race a reused buffer.

Estimated gain: overlaps the synchronous output/CPU work the GPT analysis put at
~110 s (d02 24h) / ~176 s (d03 24h) — the realistic recovery is 60-85% of that
after hour 1, growing as a share once the compile cache (Win #1) lands.

## Win #4 — continuous resident carry  ⛔ NOT DONE (large restructure — flagged)

Evaluated per directive. The daily pipeline calls
`run_forecast_operational(state, namelist, 1.0)` per hour; each call RE-INITs the
OperationalCarry (held `rthraten` + small-step save-family scratch reset) and
restarts the global step index at 1, which re-aligns the radiation cadence to
hour boundaries. A continuous carry would therefore CHANGE the radiation timing
(numerics NOT identical to the current product) AND require re-plumbing the
hourly land-state refresh (`t_skin`/`soil_moisture`/`xland`/`lakemask` injected
between hours) and the M9 snapshot into a resident carry. That is the
"correctness-sensitive large restructure" the GPT analysis (idea #2) and the
principal both flagged. STOPPED — recommend a separate scoped sprint with an
explicit product-equivalence proof.
