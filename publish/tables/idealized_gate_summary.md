# Idealized Dynamical-Core Gate Summary

Two published idealized benchmarks gate the dry dynamical core on the **unified
operational dycore path** (the same RK3 split-explicit acoustic solver used in the
coupled real-case forecast, with Coriolis f=0 ⇒ bit-identical to the f-free core).
Both ran to completion on cuda:0, fp64, CPU affinity 0–3.

Latest gate proofs: `proofs/f7n/` (regression recheck after the
vertical-momentum-advection sign fix + conservative const-K diffusion, branch
`worker/opus/f7d-pressure-mass-fix`, 2026-05-29). Coriolis-era re-confirmation:
`proofs/wind/idealized_postfix/` (warm bubble + Straka, f=0, 2 passed).

## Skamarock & Wicker (1998) rising warm bubble — PASS

dt = 0.1 s, 5000 steps, snapshots at 100/250/500 s.
Proof: `proofs/f7n/skamarock_bubble_diagnostics.json`,
`proofs/f7n/skamarock_bubble_verdict.md`.

| Check | reference target | GPU metric | pass |
|---|---|---:|:--:|
| theta′ max @500s | 0.5 ≤ max(θ′) ≤ 2.5 K | 1.920 K | ✅ |
| max \|w\| @500s | 1 ≤ max(\|w\|) ≤ 30 m/s | 11.680 m/s | ✅ |
| thermal rise @500s | centroid rises ≥ 500 m | 1924.3 m | ✅ |
| horizontal drift @500s | drift ≤ 250 m | 1.8e−12 m (symmetric) | ✅ |
| dry-column mass drift | rel ≤ 1e−8 | 0.0 | ✅ |
| all snapshots finite | finite | finite | ✅ |

Reference: https://www2.mmm.ucar.edu/people/skamarock/Papers/cv_20.pdf

## Straka et al. (1993) density current — PASS

dt = 0.1 s, 9000 steps, snapshot at 900 s.
Proof: `proofs/f7n/straka_density_current_diagnostics.json`,
`proofs/f7n/straka_density_current_verdict.md`.

| Check | reference target | GPU metric | pass |
|---|---|---:|:--:|
| front position @900s | \|x − 15000\| ≤ 2000 m | 14150 m | ✅ |
| theta′ min @900s | −25 ≤ min(θ′) ≤ −5 K | −9.971 K | ✅ |
| max \|w\| @900s | 1 ≤ max(\|w\|) ≤ 50 m/s | 14.575 m/s | ✅ |
| rotor-count proxy @900s | 2 ≤ count ≤ 4 | 4 | ✅ |
| dry-column mass drift | rel ≤ 1e−8 | 2.25e−9 | ✅ |
| all snapshots finite | finite | finite | ✅ |

References: https://www2.mmm.ucar.edu/projects/srnwp_tests/density/density.html ·
https://journals.ametsoc.org/view/journals/mwre/141/4/mwr-d-12-00144.1.xml

## Regression / M4 dycore unit gates — PASS

`proofs/f7n/regression_recheck.json`:
`pytest tests/test_m4_acoustic.py tests/test_m4_dycore_step.py tests/test_m4_tier2_invariants.py`
→ **10 passed**; both idealized close-gate verdicts PASS.

## Figures

θ′ panels (PPM, P6 binary) copied to `publish/figures/idealized/`:

- `warm_bubble_theta_prime_100s.ppm`, `..._250s.ppm`, `..._500s.ppm`
  (rising thermal evolution; source `proofs/f7n/plots/`)
- `density_current_theta_prime_900s.ppm`
  (cold-front rotor structure at 900 s; source `proofs/f7n/plots/`)

Maximum-w-vs-time traces are also available at
`proofs/f7d/plots/warm_bubble_maxw_vs_t.txt` and
`proofs/f7d/plots/straka_maxw_vs_t.txt`.

## Provenance note

These idealized cases used a rigid-lid configuration; the open-top real-case path
is validated separately (see `publish/tables/v010_d02_validation.md`). The
idealized gates certify the dry dynamical core (advection, acoustic substep,
buoyancy, mass conservation), not the coupled physics or real-terrain boundary
treatment.
