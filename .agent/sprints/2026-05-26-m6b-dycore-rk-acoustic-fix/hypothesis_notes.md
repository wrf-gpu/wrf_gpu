# Hypothesis Notes

## 1. Sign error or missing density coupling in V PGF

Checked WRF `module_small_step_em.F` around `advance_uv`: the V pressure-gradient tendency is `v = v - dts*cqv*dpxy + c1h*mudf_xy`, with `dpxy` using `msfvy/msfvx`, `muv`, and perturbation `mu`. The validation-only `acoustic_wrf.horizontal_pressure_gradient` matches the same sign and coupling shape. This did not explain why the operational V3 run diverges while B6 parity must remain bitwise.

## 2. Missing Rayleigh / vertical Coriolis on V

Rayleigh damping in this path is configured for vertical velocity only, and the localizer shows W remains bounded near 1 m/s while V alone grows after step 40. This hypothesis did not match the defect shape.

## 3. dt_sub vs dt misuse

`operational_mode._acoustic_scan` uses `dt_sub = dt_s / acoustic_substeps`, and existing contract tests pin that cadence. The runaway appears in the broad `_rk_scan_step`/`dycore_rk_acoustic` boundary rather than in the shared `acoustic_substep_core` validation recurrence. No direct full-dt V increment was found in the acoustic core.

## 4. Stagger mismatch / reduced V advection in operational wrapper

Matched. The operational RK/acoustic wrapper computes reduced M4 advection tendencies on the real d02 V field before each acoustic scan. That V self-advection is periodic/reduced-dycore logic, not a validated WRF `advance_uv` V pressure-gradient tendency, and it can amplify the bad-cell V while the shared savepoint core remains bitwise. The fix keeps the resident/base V tendency but suppresses the unvalidated reduced V self-advection inside `operational_mode._rk_scan_step`.

## 5. Moisture coupling error

`cqv` only participates in the validation-only WRF PGF helper. The operational V3 failure path uses the shared core plus RK composition and did not expose a separate `cqv` PGF array. This hypothesis was ruled out for this operational wrapper fix.

## 6. Boundary forcing not applied

The term budget shows boundary application contributed 0 at the bad-cell decomposition point, and the wrapper still applies Gen2 lateral boundaries after physics. This did not match the dominant +21.98 m/s step-46 growth.
