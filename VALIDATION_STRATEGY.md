# Validation Strategy

No bitwise equality is required unless a specific debug mode declares it. The normal target is physically valid, statistically defensible behavior against WRF fixtures, analytic oracles, and operational verification.

## Tier 1 - Micro-Fixture Parity

Compare isolated kernels or functions against WRF-derived fixtures or analytic oracles. Each fixture needs variable names, units, shape, precision, tolerance, source commit, and scenario.

## Tier 2 - Invariants

Run conservation and bounds checks: mass, energy where applicable, tracer positivity, monotonicity or bounds, NaN/Inf checks, and known edge cases.

## Tier 3 - Short-Run Convergence

Run short forecast windows and timestep convergence checks. Drift must stay inside the documented envelope for the scheme and variable.

## Tier 4 - Probabilistic Consistency

For chaotic full-model output, compare against an ensemble-derived distribution using a probtest/PyCECT-style method. Passing means the GPU result is statistically consistent, not bit-identical.
