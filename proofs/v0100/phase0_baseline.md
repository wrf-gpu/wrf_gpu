# v0.10.0 Phase 0 Baseline

Status: **partial, GPU-blocked**. I did not modify `src/` kernel logic. Fresh nsys and production daily-wrapper timings could not be produced in this environment because CUDA is not visible and `nsys profile` is denied by the host.

## Executive Verdict

| Question | Result |
|---|---|
| Acoustic unroll hook | **Absent.** Opus assumed a `GPUWRF_ACOUSTIC_UNROLL` hook exists; the source does not contain it. Phase 2 must add it before any A/B. |
| Fresh >=200-step nsys | **Blocked.** No CUDA device and nsys permission failure. Stale artifacts match ~6,890 loop kernels/step and ~3,600 D2D/step, but they are not a clean Phase 0 baseline. |
| Thompson `nstep_col` | Rain/ice/snow sampled wet columns are zero-clip-safe at cap 16, but graupel has **zero wet-column evidence**. Global `NSED_MAX` safe cap: **none**. |
| Daily wrapper host timing | **Blocked.** Production one-hour breakdown needs live CUDA. Daily host share remains unknown from fresh data. |
| HLO/StableHLO | CPU StableHLO audit complete: `_enforce_operational_precision(force_fp64=True)` emits 26 `convert`s; column `moveaxis` paths lower to real transposes; public entries donate state, `_advance_chunk` itself does not. |

Ceiling read: because the clean profile is blocked and the acoustic hook is absent while Thompson lower cap is unproven globally, current evidence does **not** justify committing to Opus's 1.8-2.6x warmed estimate. Use GPT's **1.35-1.75x warmed** as the realistic planning range until fresh Phase 0/2/4 profiles say otherwise.

## Blocker Evidence

Commands were run with `PYTHONPATH=src` and CPU-side probes pinned with `taskset -c 0-3`.

- `nvidia-smi`: failed, "couldn't communicate with the NVIDIA driver".
- `ls -l /dev/nvidia*`: no NVIDIA device nodes.
- `JAX_PLATFORMS=cuda,cpu ... jax.devices()`: CUDA init failed with `CUDA_ERROR_NO_DEVICE`; only `[CpuDevice(id=0)]` is visible.
- `/usr/local/cuda/bin/nsys profile ... python -c 'print(...)'`: failed before workload execution with `open: Operation not permitted [system:1]`.

That blocks coupled/dycore `nsys`, guards on/off, and production daily-wrapper timing. I did not substitute stale or CPU numbers as fresh GPU results.

## 1. Acoustic Unroll Hook

Source search:

```text
rg -n "GPUWRF_ACOUSTIC_UNROLL|ACOUSTIC_UNROLL" src/gpuwrf/runtime/operational_mode.py src/gpuwrf
# no matches
```

The active acoustic scan has no `unroll=` parameter:

```text
src/gpuwrf/runtime/operational_mode.py:1318:        def body(scan_acoustic: AcousticCoreState, _):
src/gpuwrf/runtime/operational_mode.py:1319:            return acoustic_substep_core(
src/gpuwrf/runtime/operational_mode.py:1326:            ), None
src/gpuwrf/runtime/operational_mode.py:1328:        acoustic, _ = jax.lax.scan(body, acoustic, xs=None, length=int(stage.number_of_small_timesteps))
```

By contrast, Thompson sedimentation does have an env-backed unroll hook:

```text
src/gpuwrf/physics/thompson_column.py:971:        return max(1, int(os.environ.get("GPUWRF_THOMPSON_SED_UNROLL", "2")))
src/gpuwrf/physics/thompson_column.py:1209:    (q_out, num_out, ppt), _ = jax.lax.scan(
src/gpuwrf/physics/thompson_column.py:1211:        unroll=_sed_unroll(),
```

Phase 2 scope: add a static `_acoustic_unroll()` helper in `operational_mode.py`, read `GPUWRF_ACOUSTIC_UNROLL` with default `1`, and pass `unroll=_acoustic_unroll()` at line 1328. Then A/B `1,2,4` with fresh nsys, compile time, memory, and WRF fixture gates before changing any default.

## 2. Fresh nsys Baseline

Requested matrix was full coupled d02 and dycore-only, guards on/off, after at least 200 warm steps, with autotuning artifacts settled to about zero. This is **not available** because CUDA and nsys are blocked.

Stale branch artifacts, not fresh Phase 0 results:

| Metric | Stale value |
|---|---:|
| Coupled warmed step | 43.556 ms/step, consistent with the 43-46 ms prior range |
| Loop elementwise kernels | 6,890.694/step |
| D2D memcpy instances | 3,601.139/step |
| Autotune/debug artifact kernels | 226.694/step, not settled |
| Dycore warmed step | 16.898 ms/step |
| Dycore HBM floor | 3.159 ms/step |
| Dycore wall/HBM floor | 5.349x |

Stale isolated phase minima from `proofs/perf/phase_breakdown.json`: Thompson 19.959 ms, surface 3.994 ms, MYNN 2.897 ms, boundary 1.000 ms, advection tendencies 0.991 ms, small-step prep 0.998 ms, pressure/EOS 1.000 ms, `calc_coef_w` 0.337 ms.

Interpretation: the old data supports the launch-count diagnosis but is not a clean baseline because artifacts are nonzero and the run was not reproduced under the requested >=200 warm-step protocol.

## 3. Thompson `nstep_col` Histogram

Proof: `proofs/v0100/thompson_nstep_histogram.json`.

Sample: seven wrfout snapshots across d02/d03, including wetter `surface_geo_v2_1` d02 hours and production d02/d03 snapshots, using dt=10 s and current Thompson fall-speed plus `_nstep_per_column` formulas. This is CPU-safe and does not run a forecast.

Merged wet-column histogram:

| Species | Wet columns | Max | P99 | P99.9 | Clips @16 | Clips @32 | Clips @64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rain | 4,395 | 2 | 2 | 2 | 0 | 0 | 0 |
| ice | 1,338 | 3 | 1 | 2 | 0 | 0 | 0 |
| snow | 1,932 | 1 | 1 | 1 | 0 | 0 | 0 |
| graupel | 0 | n/a | n/a | n/a | 0 | 0 | 0 |

Verdict: active sampled rain/ice/snow are zero-clip-safe at `NSED_MAX=16`, but there is no graupel wet-column evidence. Global Phase 4 cap reduction is therefore **not proven**. `recommended_nsed_safe_cap = null`; sentinel `nsed_safe_cap=none`.

Do not silently lower or clip. Phase 4 needs a graupel-rich WRF fixture or a species/bucket strategy with explicit proof.

## 4. Daily Wrapper Timing

Proof status: blocked. The added script `proofs/v0100/daily_wrapper_timing.py` refuses to produce a fake timing when JAX has no CUDA backend.

Requested fresh breakdown was forecast, finite-summary full-State D2H, M9 diagnostics, output-pack D2H, NetCDF write, and land refresh for one representative production forecast hour. No fresh host share is available.

Only stale context: `proofs/v090/speedup_d02/pipeline_run_20260521.json` reports d02 24 h total wall about 1166.5 s and forecast-only about 1113.2 s, an aggregate non-forecast share of about 4.6%. That is not the requested per-hour timing breakdown and should not size Phase 1 by itself.

## 5. HLO / StableHLO / Donation Audit

Proof: `proofs/v0100/hlo_audit.json`.

Precision enforcement source:

```text
src/gpuwrf/runtime/operational_mode.py:503:        updates = {
src/gpuwrf/runtime/operational_mode.py:504:            field: getattr(state, field).astype(jnp.float64) for field in STATE_FIELD_ORDER
src/gpuwrf/runtime/operational_mode.py:510:        return state.replace(_cast=False, **updates)
```

CPU StableHLO for `_enforce_operational_precision(force_fp64=True)` emits:

- `stablehlo.convert`: 26
- non-fp64 default input fields: 23
- `stablehlo.reshape`: 0
- `stablehlo.transpose`: 0

Column layout findings:

| Function | Lowering | Classification |
|---|---:|---|
| `_to_columns` | 1 `stablehlo.transpose` | real transpose |
| `_from_columns` | 1 `stablehlo.transpose` | real transpose |
| `_flatten_columns_to_batch` | 1 `stablehlo.reshape` | reshape/bitcast only |
| `_unflatten_batch_to_columns` | 1 `stablehlo.reshape` | reshape/bitcast only |
| `thompson._fill_down` | 3 transposes, 3 reshapes, 2 converts | real transpose |
| `_thompson_column_from_state` | 15 transposes, 2 converts | real transpose |

Donation/static alias audit:

```text
src/gpuwrf/runtime/operational_mode.py:2383:def _advance_chunk(
src/gpuwrf/runtime/operational_mode.py:2415:    carry, _ = jax.lax.scan(body, carry, indices)
src/gpuwrf/runtime/operational_mode.py:2513:@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
src/gpuwrf/runtime/operational_mode.py:2641:@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
src/gpuwrf/runtime/operational_mode.py:2691:@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
src/gpuwrf/runtime/operational_mode.py:2754:@partial(jax.jit, static_argnames=("hours", "debug"), donate_argnums=(0,))
```

`_advance_chunk` is a plain helper with no jit decorator and no local `donate_argnums`. Public whole-forecast entries donate state arg 0. `run_forecast_operational_segmented` is a host loop over `_advance_chunk`, so donation/aliasing there needs a GPU buffer-assignment audit before claiming in-place reuse.

## Optimization Scope Finalization

Proceed only with scopes supported by this Phase 0 evidence:

- Phase 1 host/materialization work remains plausible but unquantified. Require the daily-wrapper timing script on a live GPU before ranking finite-summary, M9, output-pack, NetCDF, and land-refresh wins.
- Phase 2 must first add the missing acoustic unroll hook, then reprofile. There is no active `GPUWRF_ACOUSTIC_UNROLL` wire today.
- Phase 3/HLO cleanup can target real column transposes and precision-convert boundaries, but GPU codegen and buffer assignment still need CUDA.
- Phase 4 must not lower global `NSED_MAX` from this sample. Active rain/ice/snow support a cap-16 hypothesis, but graupel evidence is absent.
- Planning ceiling stays at 1.35-1.75x warmed until a clean GPU baseline resolves launch-count artifacts, acoustic unroll A/B, and graupel-safe Thompson cap/bucketing.

## Handoff

- objective: resolve v0.10.0 Phase 0 open questions with proof objects and no kernel-code changes.
- files changed: only `proofs/v0100/*` scripts plus `phase0_baseline.md/json`.
- commands run: source `rg/nl` audit, CUDA visibility checks, nsys smoke, CPU StableHLO audit, Thompson histogram, Python/shell/json validation.
- proof objects produced: `phase0_baseline.md`, `phase0_baseline.json`, `hlo_audit.json`, `thompson_nstep_histogram.json`, and profiling/audit scripts under `proofs/v0100/`.
- unresolved risks: no fresh nsys, no guards on/off matrix, no production daily-wrapper timing, no GPU buffer-assignment alias report, no graupel wet-column Thompson proof.
- commit status: attempted, but blocked because Git metadata is read-only: `fatal: Unable to create 'REPO/.git/worktrees/perf-v0100/index.lock': Read-only file system`.
- next decision needed: restore GPU/nsys access and rerun Phase 0 GPU lanes before implementation begins.
