# M6.x Numerical Stability Spike Worker Report

## Objective

Run Gemini's two cheap c1 diagnostics before c2-A1 freezes architecture:

1. Flat vs compact Schar-style mountain warm-bubble run to separate terrain metric/base-state failure from general formulation failure.
2. Brute-force `smdiv=0.1` plus top-level Rayleigh `w` sponge to separate missing damping infrastructure from fundamental formulation error.

The role prompt referenced upstream c2/Gemini context files that are not present at the named paths in this checkout. I followed the local constitution/AGENTS read order, used the role prompt as the operative diagnostic contract, and produced proof JSONs rather than claiming physics correctness from inspection.

## Test 1: Flat vs Schar Mountain

Script: `scripts/m6_spike_test1_flat_vs_mountain.py`

Proof object: `artifacts/m6/spike/test1_flat_vs_mountain_result.json`

Setup:

- Grid: `64 x 64 x 40`, `dx=dy=400 m`, `dz=100 m`, `dt=2 s`, `n_acoustic=8`, duration `3000 s`.
- Physics off. Path is `gpuwrf.dynamics.step.step`, so this exercises c1 dycore with c1 acoustic buoyancy.
- Mountain case uses a compact Schar-style `cos^2` ridge, `peak=250 m`, `half_width=5000 m`.
- To make the terrain diagnostic non-vacuous, `ph` uses terrain-following interface heights, `state.p` follows the terrain-column sigma pressure, and `state.pb` remains the flat-column pressure profile. That exposes the missing terrain metric/base-state cancellation class instead of merely carrying unused `grid.terrain_height`.

Results:

| case | first_nonfinite_step | surviving_seconds | last finite `w_max` |
| --- | ---: | ---: | ---: |
| flat | 76 | 150.0 | `2.4759242870891734e13 m s^-1` |
| Schar mountain | 36 | 70.0 | `6.369308079636649e15 m s^-1` |

Interpretation:

- Flat terrain fails before 300 s, so terrain metric terms are not the sole dominant residual.
- Mountain terrain fails much faster than flat terrain, so terrain/base-state handling still worsens the instability materially.
- Test 1 verdict in the JSON is `FLAT_ALSO_UNSTABLE_FORMULATION_DOMINANT`.

WRF reference note: WRF's small-step horizontal pressure-gradient path carries coupled geopotential/pressure/base-pressure terms and map-scale factors in both x and y. See `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:828-862` and `:902-936`. c1's reduced pressure-gradient path does not carry an equivalent terrain metric/base-state cancellation.

## Test 2: Brute-Force Damping

Script: `scripts/m6_spike_test2_brute_force_damping.py`

Proof object: `artifacts/m6/spike/test2_brute_damping_result.json`

Additional proof object from the existing discriminator:

- `artifacts/m6/spike/test2_existing_warm_bubble_result.json`

Temporary branch-only source change:

- `src/gpuwrf/dynamics/acoustic.py` now keeps the existing WRF-style `SMDIV_DIVERGENCE_DAMPING = 0.1` and adds `SPIKE_RAYLEIGH_SPONGE_TOP_LEVELS = 10`, `SPIKE_RAYLEIGH_SPONGE_MAX_FRACTION = 0.5`.
- `_spike_rayleigh_sponge_w()` damps top `w` faces with the requested ramp: `1 - 0.5 * (k - nz + 10) / 10` for `k > nz - 10`.
- This is explicitly named spike-only and should not be interpreted as production damping infrastructure.

Results:

- New brute-force damping script: first nonfinite step `76`, surviving seconds `150.0`, `survived_600s=false`.
- Existing `scripts/m6_warm_bubble_test.py` run to `artifacts/m6/spike/test2_existing_warm_bubble_result.json`: `FAIL_DYCORE_INADEQUATE`, `finite_state_all_times=false`; 300 s and 600 s diagnostics are null because the run becomes nonfinite first.

Interpretation:

- Brute `smdiv` plus the top Rayleigh sponge does not stabilize the flat warm-bubble case beyond 600 s.
- The Test 2 JSON verdict is `STILL_UNSTABLE_BEFORE_600S_FORMULATION_ERROR_IMPLICATED`.
- This argues against "c1 math is sound but missing damping" as the primary explanation.

WRF reference note: the `smdiv` pressure carry shape in WRF is at `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:557-565`; c1 already had the corresponding `p += smdiv * (p - p_prev)` shape before this spike. Adding Rayleigh damping did not change the failure step in the flat diagnostic.

## Net Conclusion

Both diagnostics point to a fundamental formulation problem, not merely missing damping. The mountain result adds evidence that terrain metric/base-state terms are also missing or underrepresented, but because flat terrain fails at 150 s, c2-A1 should not scope the ADR as "just add metric terms."

Most likely c2 requirements:

- Native prognostic/base-state decomposition from day 1: at minimum `p_total`, `p'`, `pb`, `ph_total`, `ph'`, `phb`, `mu`, and `mub`/dry-column base mass equivalents need explicit roles rather than being reconstructed ad hoc.
- A well-balanced horizontal pressure-gradient formulation that cancels hydrostatic terrain-following slopes, including the analogues of WRF's `ph/php`, `p/dpn`, `pb`, `al/alt`, and metric/map-factor terms.
- Damping hooks (`smdiv`, Rayleigh upper sponge) should exist, but as stabilizers around a correct operator, not as the architectural fix.
- The c2-A1 ADR should reject any state layout where terrain metrics are only passive `GridSpec.terrain_height` metadata and where pressure perturbation is inferred late from `state.p - state.pb` without a clear base-state contract.

## Commands Run

- `python -m py_compile scripts/m6_spike_test1_flat_vs_mountain.py`
- `python -m py_compile scripts/m6_spike_test2_brute_force_damping.py`
- `PYTHONPATH=src python scripts/m6_spike_test1_flat_vs_mountain.py`
- `python -m py_compile src/gpuwrf/dynamics/acoustic.py`
- `PYTHONPATH=src python scripts/m6_spike_test2_brute_force_damping.py`
- `PYTHONPATH=src pytest -q tests/test_m6x_fallback_c1_acoustic.py` -> `13 passed in 42.21s`
- `PYTHONPATH=src python scripts/m6_warm_bubble_test.py --output artifacts/m6/spike/test2_existing_warm_bubble_result.json` -> expected nonzero diagnostic failure, artifact written
- `PYTHONPATH=src pytest -q tests/test_m4_acoustic.py tests/test_m4_dycore_step.py tests/test_m4_rk3.py` -> `10 passed in 99.28s`

## Files Changed

- Created `scripts/m6_spike_test1_flat_vs_mountain.py`
- Created `scripts/m6_spike_test2_brute_force_damping.py`
- Modified `src/gpuwrf/dynamics/acoustic.py` with the temporary spike-only Rayleigh sponge
- Created `artifacts/m6/spike/test1_flat_vs_mountain_result.json`
- Created `artifacts/m6/spike/test2_brute_damping_result.json`
- Created `artifacts/m6/spike/test2_existing_warm_bubble_result.json`
- Created this worker report

## Unresolved Risks

- Test 1 uses a compact Schar-style ridge proxy, not a full canonical Schar 2002 mountain-wave benchmark.
- The mountain diagnostic intentionally exposes missing terrain/base-state cancellation by using flat `pb` against terrain-following `p/ph`. That is appropriate for a cheap architecture spike, but it is not a WRF parity fixture.
- The Rayleigh sponge is hardcoded and branch-only per prompt. It should be removed or redesigned behind an explicit c2 damping interface before any production merge.

## Next Decision Needed

c2-A1 should proceed on the assumption that c1's remaining issue is a formulation/state-decomposition problem, with terrain metric support required but not sufficient. The ADR should prioritize a well-balanced pressure/geopotential/mass decomposition and treat damping as secondary infrastructure.
