# c1-A5-H3 Discriminator Worker Report

## Objective

Test whether removing the `rk3_step` `_flat_dz`-based acoustic auto-promotion materially reduces the 1h sanitize/nonfinite signature.

## Patch

Picked Option A because Option B requires an import/call-site wiring change and the assignment was constrained to one line.

```diff
-    n_acoustic = max(int(n_acoustic), required_n_acoustic(grid, dt))
+    n_acoustic = int(n_acoustic)  # honor requested value; user must pick correct CFL
```

## 1h Probe

Command run once:

```bash
python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6x-fallback-c1/c1_a5_h3_1h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a5_h3_1h --skip-nsys --skip-legacy-baseline-sanitize-audit
```

Result: failed before final verdict JSON was written. The forecast output manifest was written, then the run failed in `_trace_transfers(...)` with:

```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: [0] Failed to load in-memory CUBIN (compiled for a different GPU?).: CUDA_ERROR_OUT_OF_MEMORY: out of memory
```

Startup also emitted repeated CUDA allocation backoff/OOM messages, and forecast output serialization emitted:

```text
RuntimeWarning: overflow encountered in cast
```

## Metrics

- sanitize_firing_rate: unavailable; run failed before `_run_sanitize_audit`
- nonfinite_count: unavailable; run failed before `_run_sanitize_audit`
- fired_steps: unavailable; run failed before `_run_sanitize_audit`
- n_acoustic_actual: unavailable from this failed run; final payload containing `n_acoustic` was not emitted

## Proof Objects

- Forecast output manifest: `artifacts/m6x-fallback-c1/c1_a5_h3_1h.outputs.json`
- Forecast NPZ: `/home/enric/.cache/gpuwrf_outputs/m6/c1_a5_h3_1h/wrfout_gpu_d02_p001h.npz`
- Missing final verdict: `artifacts/m6x-fallback-c1/c1_a5_h3_1h.json` was not produced

## Discriminator Conclusion

Inconclusive. The one-line `rk3.py` patch was applied, and the single requested 1h probe was attempted, but the probe failed during transfer audit before sanitize/nonfinite metrics were produced.

## Unresolved Risks

- The H3 discriminator still lacks the required 1h sanitize/nonfinite evidence.
- The entrypoint computes `n_acoustic = max(requested_n_acoustic, required_n_acoustic_for_state(...))` before calling `rk3_step`; this report does not change that behavior because the task prohibited bundling.

## Next Decision Needed

Free GPU memory or authorize a probe configuration that skips the transfer audit, then rerun exactly one 1h discriminator probe. If the intended discriminator is to force the requested acoustic count end-to-end, authorize the separate script/driver-level change explicitly.
