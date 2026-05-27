# Prioritized Optimization Roadmap

Verdict: optimize the warm path by reducing tiny kernel launches and D2D materialization first, while separately removing cold-JIT cost for operations. Do not start with precision or multi-GPU.

| Rank | Sprint | Scope | Estimated wall-time saving | Correctness risk | Effort |
|---:|---|---|---:|---|---|
| 1 | Elementwise launch/D2D fusion sprint | Collapse RK/acoustic scan-carry housekeeping, save-family updates, dtype enforcement, finite guards, and repeated add/multiply/subtract chains. Target `loop_add_fusion_4`, `loop_multiply_fusion`, `loop_subtract_fusion`, and D2D copies. | 1.5-3.0 s per warm 1h; ~35-70 s per 24h steady run. | Medium. Guards and save-family fields are correctness-sensitive. | Medium-high |
| 2 | Cold-JIT cache/precompile sprint | Persistent JAX compile cache plus precompiled 1h operational executable for fixed d02 shape/static args. Add compile-log proof. | 100+ s per fresh daily run; possibly more in the current 24h pipeline where first two hours total 558.6 s. | Low. | Low-medium |
| 3 | XLA memory-profile and aliasing sprint | Capture HLO/XLA memory profile, then reduce live ranges and buffer copies in pressure/geopotential, save-family, and physics adapter paths. | 0.3-1.0 s per warm 1h plus lower 1 km peak memory; main value is headroom and removing 6.97 GB transient estimate. | Medium. | Medium |
| 4 | Vertical solver specialization sprint | Evaluate validated 44-level batched Thomas/PCR or coefficient+solve fusion for FP64 acoustic solve. Keep PCR as baseline. | 0.1-0.4 s per warm 1h direct; ~5-12 s per 24h if adjacent temp/reverse kernels also collapse. | High. | High |
| 5 | Physics column-layout sprint | Reduce `moveaxis`, concatenate, mass/face interpolation, and full-field tendency materialization in Thompson/MYNN/RRTMG adapters. | 0.1-0.3 s per warm 1h direct; larger if memory pressure drops. | Medium-high because M7 skill is already blocked. | Medium |

Deferred work:
- Precision downcast: do not prioritize until launch/memory profiling lands. The only large FP64 candidate is `w` (~16.2 MiB saving at 1 km if FP32), and acoustic stability risk is high. BF16 hydrometeor/scalar savings are larger (~95.4 MiB at 1 km) but explicitly require new validation proof.
- Multi-GPU: important for future 1 km throughput, but premature before the single-GPU fused warm path and halo cadence are stable.
- NCU counter enablement: useful for confirming bandwidth vs compute, but not required to begin rank 1 and rank 2 because the launch and cold-JIT evidence is already decisive.

Top 3 highest-value opportunities:
1. Reduce launch count and D2D materialization in the operational timestep scan.
2. Enable persistent compile cache / precompile to remove 100 s-class cold starts.
3. Capture XLA memory profile and use it to shorten transient live ranges.
