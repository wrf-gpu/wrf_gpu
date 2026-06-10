# Worker Report

## Summary

Summary:

GPT-5.5 xhigh diagnosed the remaining fixed-Canary h1/h4 `PSFC` residual after
the LBC cadence fix. The analysis found a real vapor-light pressure-state
residual: GPU moisture is present in `QVAPOR`, but the GPU pressure/`PSFC`
state does not carry the surface vapor-column load that CPU WRF carries.

## Files Changed

- `.agent/reviews/2026-06-10-v014-gpt-psfc-vapor-light-analysis.md`

## Commands Run

- CPU-only direct NetCDF pressure-budget probes over the fixed Canary h1 and
  h1-h4 outputs.
- CPU-only reads of the h1/h4 grid-comparator JSON/markdown artifacts.
- Source audit of writer/runtime/dycore pressure-state paths.

## Proof Objects

- `.agent/reviews/2026-06-10-v014-gpt-psfc-vapor-light-analysis.md`
- fixed run h1/h4 comparators under
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_lbcfix_20260610T151455Z/`

## Risks

- The exact CPU WRF `p8w` diagnostic formula still has a smaller unresolved
  `~14 Pa` gap relative to the simple extrapolation used in the analysis.
- The report is diagnostic only; it does not implement or validate a production
  fix.

## Handoff

Open a focused `PSFC` moist pressure-state closure sprint before Switzerland GPU
or v0.14 field-parity promotion. Do not accept a comparator tolerance or
output-only PSFC clamp as the fix.
