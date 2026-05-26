# Localization Memo

## Verdict: NAMED-FIX:boundary_application

## Evidence summary
- `proof_step46_violation.json`: first failed step 46 at lead 460.0 s with |V|max 11.480102 m/s.
- `proof_step46_violation.json`: bad V-stagger cell is [2, 1, 0] with snapshots for theta, mu, u, v, w, and ww at steps 45 and 46.
- `proof_wrf_reference_compare.json`: Gen2 WRF interpolated V at that cell is -11.397122 m/s.
- `proof_wrf_reference_compare.json`: Gen2 WRF same-level nearby max |V| is 11.398314 m/s; domain vertical max |V| is 11.398314 m/s.
- `proof_operator_decomposition.json`: dominant available WRF-ordered stage term is `boundary_application`.
- `proof_first_divergent_step.json`: earliest detectable divergence in the step-40..46 window is step None.

## Recommended next sprint
- `2026-05-25-m6b-v3-boundary-application-fix`: Fix and validate the `boundary_application` V tendency that dominates the 20260521 step-46 acceleration.

## Risks / caveats
- WRF history is hourly; the proof uses linear interpolation for the 10 s step-46 valid time.
- The V3 operational wrapper does not expose separate pressure-gradient, Coriolis, and vertical-advection term arrays, so Stage 3 localizes to the available WRF-ordered operational stages.
- GPU-vs-CPU parity risk remains linked to the sister gpu-cpu-step2 sprint because this localizer runs the JAX operational path only.
