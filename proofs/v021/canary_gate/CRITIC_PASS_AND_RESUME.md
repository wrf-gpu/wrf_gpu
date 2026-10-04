# v0.21.0 FUSED+AOT — CRITIC PASS + RESUME-AFTER-HIBERNATE

**State when interrupted (2026-06-25 ~20:1x):** the user hibernated the workstation. Manager
TERM-stopped the running GPU s/step gate (clean, exact-PID) so the GPU was free for hibernate.
Nothing lost: the gate was only at `MARKER:PROCESS_START` (fused cold-compile had not finished).
Implementation is committed: **`1f35c483`** "Implement fused cascade AOT warm start" (worktree clean).

## Gate ladder status

| gate | status |
|---|---|
| (a) implementation committed | ✅ `1f35c483` (9 files +778/-173, clean tree) |
| (b) **Opus critic** (static correctness) | ✅ **PASS** — see below |
| (c) **GPU s/step gate** (runtime/warm-load/bit-id) | ⏸ **INTERRUPTED for hibernate — RE-RUN ON RESUME** |
| (d) present validated result to the user | pending (after c) |
| (e) tag/push (the user DIRECT auth) + curated wrfgpu | pending (after d) |

## (b) Opus critic verdict — PASS (commit 1f35c483)

- **Edge-geometry cheap_key injectivity = CLEAN** (the #1 silent-wrong-load risk). Independent CPU
  test proved DIFFERENT keys for: child weight value, parent_grid_ratio, child cadence, parent
  cadence, bdy_width, child count, **child-order swap**, parent namelist aux, child namelist aux;
  negative control (identical rebuild) = SAME key (process-stable, no false churn). Mechanism:
  `_FusedAuxNamelist` (`domain_tree.py:1212-1308`) passed as `namelist` to `cheap_key`; arrays
  content-hashed by `.tobytes()` so `InterpWeights` flip the key by value.
- **Runtime untouched** — inner `fused_jit` physics (`domain_tree.py:1396-1425`) byte-identical;
  only a serialize/load wrapper added.
- **Fail-open** — every AOT error path falls back to recompile / `fused_jit(*args)`, never a stale
  blob (verify-fail → recompile `:1624-1654`; load never raises; key-builder catches BaseException).
- **De-fuse-default revert COMPLETE** — fused is runtime default again; de-fuse opt-in; docs reverted.
- **No leak** — per-program closure evicted with `_FUSED_PROGRAM_CACHE` weakref (`:1804`); no
  v0.19.1-class self-ref.
- 72 related tests pass + critic's own injectivity test.

**Zero BLOCKING static defects.** Nice-to-haves (non-blocking): (1) add the child-order case to the
in-repo `test_fused_cheap_key_folds_edge_geometry` (`tests/test_aot_executable.py:97`) — critic proved
it externally, this is just regression-net hygiene; (2) comment/assert the `_namelist_static_aux`
non-pytree fallback is dead for production; (3) ensure the GPU gate runs ≥1 **verify-ON** pass to
exercise the HLO-match assertion end-to-end.

## RESUME — re-run the GPU s/step gate (GPU must be free; lock-wrapped)

From worktree `<USER_HOME>/src/wrf_gpu2_wt/parallel-compile` (branch worker/opus/vnext-parallel-compile):

```
scripts/with_gpu_lock.sh --timeout 7200 --label fused-aot-gate -- bash -lc '
  set -euo pipefail
  mkdir -p proofs/v021/canary_gate/logs
  L1=proofs/v021/canary_gate/logs/fused_aot_capture.log
  L2=proofs/v021/canary_gate/logs/fused_aot_capture2.log
  export PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false
  python proofs/v021/canary_gate/perstep_timing_driver.py 2>&1 | tee "$L1"
  python proofs/v021/canary_gate/perstep_timing_driver.py 2>&1 | tee "$L2"
'
```
Driver config when interrupted: maxdom=3, kwarm=12, defuse=0, fuse=unset (→ fused default). Cold
compile is ~50 min host-bound (GPU idle during lowering — expected). The codex GPT implementer
(its window holds the exact invocation it built) can also re-launch it; either is fine.

### PASS criteria (the release condition — honest pass/fail)
1. Warm s/step **== FUSED ~1.376** (within ~5%), **NOT** de-fuse 1.635 (same fused exec → must match).
2. Warm-load = SECONDS, `loaded=true`/`source=aot_blob`, ZERO "Compiling module"/re-lower.
3. Bit-identical: warm output == cold output, max_abs_diff 0.
4. Memory flat (no leak); report peak VRAM + host RSS. Include ≥1 verify-ON pass.

Write the result to `proofs/v021/canary_gate/V0210_FUSED_AOT_GATE.md`. On PASS → present to the user →
(the user direct auth) tag v0.21.0 + push origin, then build+PII-scan+push curated wrfgpu. Then rewrite
the v0.21 docs (README/CHANGELOG/RELEASE_NOTES/USER_README currently still describe the reverted
de-fuse-default story) to the FUSED+AOT story, and update CANARY-BENCHMARK-LEDGER.md with the new row.
