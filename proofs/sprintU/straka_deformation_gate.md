# Sprint U / P0-2 — Straka gate A/B with the WRF deformation momentum diffusion

Date: 2026-05-29

The WRF deformation-tensor momentum diffusion (`use_deformation_momentum_diffusion=True`)
was run through the full 900 s Straka density-current close gate (same IC, nu=75,
flux advection, rigid lid + damping as the F7N close).

## Result: Straka PASSES 6/6 with the deformation operator

| check | value | passed |
|---|---|---|
| all_snapshots_finite | 1.0 | ✅ |
| theta_prime_min_900s | -9.95 K | ✅ |
| max_abs_w_900s | 14.55 m/s | ✅ |
| front_position_900s | 14150 m | ✅ |
| rotor_count_proxy_900s | 4 | ✅ |
| relative_mass_drift | 1.4e-16 | ✅ |

(proof: `proofs/sprintU/straka_deformation_gate.json`)

## Interpretation

The WRF-faithful deformation momentum operator (factor-2 diagonal + du/dz<->dw/dx
cross terms) does NOT regress the Straka close gate — it PASSES 6/6 with metrics
essentially equal to the scalar-flux-divergence close (front 14.15 km, θ′min
-9.95 K, max|w| 14.5 m/s), and the relative mass drift is even smaller
(1.4e-16 vs 2.25e-9). This confirms F7M's finding that the touchdown residual was
NOT diffusion-controlled (it was the F7N vertical-momentum sign bug), and that the
deformation operator is a safe, WRF-faithful momentum-diffusion choice.

## Default decision

The close-gate default keeps `use_deformation_momentum_diffusion=False` (the
conservative scalar flux-divergence) because that is the exact operator the F7N
close validated and the CI close gate runs. The deformation operator is wired,
analytically validated (`test_deformation_momentum_diffusion.py`), and gate-proven
here; flipping it to default is a one-line change once a Phase-B sprint adopts it
as the production momentum diffusion. The real-case operational path uses the
6th-order numerical filter + WRF damping (not const-nu), so const-nu momentum
diffusion (and hence this operator) is active only for the Straka-style const-nu
configuration.
