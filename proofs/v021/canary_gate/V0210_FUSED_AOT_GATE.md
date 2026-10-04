# V0210 Fused+AOT Gate

Date: 2026-06-25
Branch: `worker/opus/vnext-parallel-compile`

## Verdict

PASS.

Fresh warm FUSED+AOT loaded the fused executable from AOT, did not re-lower/recompile
`jit_fused_jit`, remained bit-identical to the capture process, and ran in the
fused-speed regime rather than the de-fuse regime.

Steady warm metric used for the verdict:

- Fresh warm `s_per_step_b`: `1.2027` s/root-step
- Historical fused reference from sprint brief: `~1.376` s/root-step
- Historical de-fuse reference from sprint brief: `~1.635` s/root-step

The first subtraction in each process is not used for the verdict because the
single-step call pays one-time AOT load/cache-shape recovery cost. The second
pair (`WARM_1b/WARM_Kb`) is the stable measurement after variants are resident.

## Commits Under Gate

- `1f35c483` - `Implement fused cascade AOT warm start`
- `26bf142f` - `Add v021 per-step timing gate driver`
- `210dd65c` - `Add fused AOT gate runner`

## Commands

CPU cheap-key injectivity:

```bash
env PYTHONPATH=src:. JAX_PLATFORMS=cpu \
  pytest -q tests/test_aot_executable.py::test_fused_cheap_key_folds_edge_geometry
```

Result: `1 passed in 5.32s`.

GPU gate:

```bash
proofs/v021/canary_gate/run_v0210_fused_aot_gate.sh
```

The runner wraps `proofs/v021/canary_gate/perstep_timing_driver.py` with
`scripts/with_gpu_lock.sh`, `CANARY_INPUT_DIR=<DATA_ROOT>/wrf_downscale/runs/20240901/cpu`,
`MAXDOM=3`, `GPUWRF_NESTED_AOT=1`, `GPUWRF_NESTED_PARALLEL_COMPILE=0`, and no de-fuse flag.

## Evidence

Logs:

- Capture/cold serialize: `proofs/v021/canary_gate/logs/fused_aot_capture.log`
- Fresh warm load: `proofs/v021/canary_gate/logs/fused_aot_warm_load.log`

Capture process:

- Fused default active: `MARKER:PREWARM_DONE ... source=skip:fused-default`
- `d01` loaded from AOT.
- `fused/d02` cold miss captured two runtime shape variants:
  - `cheap_key=4c14e73fbb57...`
  - `cheap_key=6db965ea2ff5...`
- `CALL COLD root_steps=2 wall_s=742.8327 finite=True`
- `WARM_1b=31.2921`, `WARM_Kb=43.4229`, `s_per_step_b=1.1028`
- `vram_flat=True`, `rss_flat=True`, `all_finite=True`
- `BIT_ID ... repeat_exact=True repeat_max_abs_diff=0`

Fresh warm process:

- Fused default active: `MARKER:PREWARM_DONE ... source=skip:fused-default`
- `fused/d02 loaded=true source=aot_blob ... cheap_key=4c14e73fbb57`
- Negative checks in warm log:
  - no `fallback:missing`
  - no `fallback:fused-jit-compiled`
  - no `Compiling module jit_fused_jit`
  - no `DRIVER RAISED`
- `CALL COLD root_steps=2 wall_s=80.4293 finite=True`
- `WARM_1b=28.5643`, `WARM_Kb=41.7945`, `s_per_step_b=1.2027`
- `vram_flat=True`, `rss_flat=True`, `all_finite=True`
- `BIT_ID ... repeat_exact=True repeat_max_abs_diff=0`
- `REF_COMPARE ... ref_equal=True`

Digest proof objects:

- `proofs/v021/canary_gate/ts_proof_fused_aot_capture/warmK_state_digest.json`
- `proofs/v021/canary_gate/ts_proof_fused_aot_warm/warmK_state_digest.json`
- Matching SHA-256: `7596e0628ab0ea74f45d968248511b8a265618c0dcbb497b4ec0125648e08f62`
- Leaves: `171`
- Bytes hashed: `344432672`

## Notes

The first measured subtraction is distorted (`s_per_step=-0.0761` in the fresh
warm process) because `WARM_1` absorbs one-time load/cache recovery while
`WARM_K` benefits from already-resident executables. The gate therefore uses
the second-pass driver metric `s_per_step_b`, which is the intended steady-state
difference after the warm process has loaded the executable.
