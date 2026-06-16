---
name: profiling-nvidia-gpu
description: Guides NVIDIA GPU profiling, profiler artifact capture, transfer audits, and benchmark JSON reporting.
---

## When to use

Use for GPU performance claims, M2 bakeoff metrics, transfer audits, and kernel launch or register-pressure analysis.

## Inputs required

Benchmark command, hardware, backend, fixture id, expected metrics, and artifact destination.

## Workflow

1. Confirm correctness passed.
2. Capture environment and hardware.
3. Run profiler or dry-run if unavailable.
4. Parse report into JSON.
5. Store artifacts outside large-file paths and summarize key metrics.

## Hard rules

- Do not profile broken physics as success.
- Do not claim speedup from noisy timing alone.
- Missing profiler tools must be reported, not hidden.

## Deliverables

Profiler command, raw artifact path, parsed JSON, transfer audit summary.

## Validation

Parsed JSON contains benchmark name, backend, hardware, wall time or profiler metrics, and artifact paths.

## Common failure modes

Including compile time accidentally, profiling first-run only, ignoring hidden transfers, and comparing different fixtures.
