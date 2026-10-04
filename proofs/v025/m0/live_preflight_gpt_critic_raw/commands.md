# CPU/device-denied critic command record

All executable checks were run with `DevicePolicy=closed`, `MemoryMax=24G`,
`AllowedCPUs=0-3`, `JAX_PLATFORMS=cpu`, and `CUDA_VISIBLE_DEVICES=""` unless
the command was a pure Git/text query. No GPU query, import, context, compile,
run, profiler capture, receipt request, lock request, or coordination action
was performed.

## Candidate/proof authority

```text
git merge-base --is-ancestor a5e8f5d8 57e8dac7
git diff --name-status a5e8f5d8 57e8dac7
git rev-parse a5e8f5d8:src/gpuwrf 57e8dac7:src/gpuwrf
sha256(candidate proof bytes)
sha256(canonical JSON after removing evidence_sha256)
```

The proof generator was run twice in a fresh local shared clone at exact commit
`a5e8f5d8`, on a local branch with the author's exact branch name. Both clean
regenerations produced file SHA
`3536d250f6220df0e0d8aa58db97dbe2aef0b062aac46e6ec39941ac4c1c14fb`;
the frozen candidate is
`957ccc563601b3ad2e89c0450fb1c1fe2cc1974286eaabe8fa2dfd0b211ff6bc`.
The temporary clone was removed after its diff was recorded.

## Source/readiness

```text
env -u GPUWRF_WRF_ROOT -u GPUWRF_WRF_SRC python:
  gpuwrf.physics.rrtmg_lw._native_lw_tables()

GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF
GPUWRF_WRF_SRC=<USER_HOME>/src/wrf_pristine/WRF python:
  gpuwrf.physics.rrtmg_lw._native_lw_tables()

python:
  load_fast_call_arguments(..., cpu_device_adapter=True)
  prepare_exact_call(...)
  _run_forecast_operational_jit.lower(*prepared_arguments)
```

The missing-root call reproduced `FileNotFoundError`. The bound call returned a
29-field `_LWNTableBundle`; the exact production lowering returned
`jax._src.stages.Lowered`, invoked `_native_lw_tables()` twice, and took
26.583 seconds on CPU.

A Python call profiler around `build_session_preflight_identity` observed no
`_native_lw_tables` or `load_fast_call_arguments` call even though the outer
preflight returned `PASS`.

## Authorization

```text
run_gpu_arm.authorise(
  "baseline-census",
  fresh affirmative temp receipt,
  canonical holder/token carrying label "invented-lock-label",
  consume=True,
  temporary spend ledger,
)
```

The call returned authorized and spent the temporary receipt. The committed
ledger stayed byte-identical at
`b4544ff8e3f2ef8b087a16a9e2f8743cbf8a22d59d456ff7845b9af9daed4ce0`.

## Profiler/static audit

The retained v0.15 CSV/JSON files were parsed directly. No profiler executable
was run against a device. `analyze_trace.py` was read to verify that `steps`
defaults to 200 and that `span` is only the first-to-last device-event window.
All retained files were searched for the required
`GPUWRF_M0_FORECAST_INTEGRATION` range; there were zero occurrences.

## Tests

```text
pytest -q -rs tests/v025/test_m0_live_preflight_repair.py
pytest -q -rs tests/v025
pytest -q -rs tests/v025/test_m0_live_preflight_gpt_critic.py
```

Before adding the critic test:

- candidate focus: `34 passed`;
- full frozen baseline: `909 passed, 2 skipped`;
- critic attacks against the candidate: `1 passed, 5 failed`.

JUnit XML files in this directory retain those three runs. A final full run is
recorded separately after all critic deliverables are complete.

Final full suite with the six critic tests collected:

- `910 passed, 5 failed, 2 skipped` in 67.04 seconds;
- the five failures are exactly the five pre-registered rejection relations;
- the one additional pass is the retained-NVTX metadata observation.
