# M6.x Worker Report - WRF-Canonical Dycore Completion

## Objective

Replace the M4 proxy acoustic dycore and static `mu` behavior with WRF-cited physical sound speed, per-cell acoustic CFL diagnostics, and canonical dry-column mass continuity while preserving the M4 JAX-XLA resident state architecture. The sprint goal was AC1-AC10, including a 24 h finite d02 forecast and speedup >=4x.

## Result

Status: **FAIL / not merge-ready**.

The branch improves the immediate M6-S5 failure mode but does not satisfy AC4 or AC5. With the best tested configuration, the coupled forecast remains clean for the first hour, then drifts into sanitize clipping and physical-range failure by the 2-6 h window. Because AC4 and AC5 are not green, ADR-007 was **not** amended to PASS and no PASS ADR-015 was created.

## Files Changed

- `src/gpuwrf/dynamics/acoustic.py`
- `src/gpuwrf/dynamics/tendencies.py`
- `src/gpuwrf/dynamics/advection.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/contracts/precision.py`
- `src/gpuwrf/coupling/driver.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `scripts/m6_full_domain_batching.py`
- `tests/test_m6x_dycore_completion.py`
- `tests/test_m6x_cfl_diagnostic.py`
- `tests/test_m6x_mu_continuity.py`

No files under `src/gpuwrf/physics/` were modified.

## WRF Source Citations Used

- Acoustic compressibility source: `dyn_em/module_small_step_em.F:233`, where WRF computes `c2a = cpovcv*(pb+p)/alt`.
- Small-step pressure relation: `dyn_em/module_small_step_em.F:527-528`.
- Vertical implicit acoustic coefficients: `dyn_em/module_small_step_em.F:626-648`.
- Column dry mass continuity comments and flux-divergence form: `dyn_em/module_small_step_em.F:1076-1088`.
- WRF small-step `DMDT` accumulation and `MU` update: `dyn_em/module_small_step_em.F:1094-1105`.
- WRF `advance_all` `mu_tend` application: `dyn_em/module_small_step_em.F:1915-1918`.
- Big-step continuity formulation and `divv` layer terms: `dyn_em/module_big_step_utilities_em.F:718-753`.
- RK dry tendency treatment of `MU_TEND`: `dyn_em/module_em.F:1779-1783`.
- WRF RRTMG top-layer heating zeroing: `phys/module_ra_rrtmg_sw.F:9559-9560` and `phys/module_ra_rrtmg_lw.F:3518-3520`.

## Implementation Notes

AC1/AC2 were partially implemented. `acoustic.py` now computes dry-air `c^2 = gamma R T` from the current layer temperature and provides `acoustic_cfl_diagnostic`, including `sqrt(1.4 * 287 * 300) ~= 347.19 m/s`. The code also uses fixed base pressure `pb` so pressure-gradient forcing is based on perturbation pressure (`p - pb`) rather than `total pressure - horizontal mean`. That change was decisive: without `pb`, the dycore blew up around 40-50 ten-second steps; with `pb`, a one-hour coupled run stayed finite.

However, the final tested branch still contains a bounded reduced acoustic coupling: `MAX_INVERSE_DENSITY = 0.02` and `PRESSURE_IMPLICIT_RELAXATION = 0.05`. Removing these stabilizers and using full physical inverse density made the reduced dycore fail by step 2. This is not a clean WRF-canonical acoustic completion.

AC3 was implemented in `tendencies.py` as `dmu/dt = -div_h(integral mu * wind d_eta)`, using non-periodic C-grid face interpolation for specified/nested boundaries. `advection.py` includes this tendency in the RK tendency path. The unit tests cover zero-divergence and manufactured divergent-wind cases.

The base-pressure state extension added a static `pb` leaf to `State`. Physics adapters still consume total `state.p`, while acoustic pressure gradients use `state.p - state.pb`. This preserved the existing SoA pytree model but increases persistent state size.

The RRTMG adapter now zeroes the top-layer heating before applying radiation tendency, matching WRF SW/LW behavior. This is not a physics-kernel edit; `src/gpuwrf/physics/` remains unchanged. It fixed a step-360 top-layer outlier where direct SW+LW heating reached about `-817 K/s` in the top layer and would otherwise drive theta to the 150 K sanitize bound.

## Measurements

Unit tests:

```text
pytest -q tests/test_m4_dycore_step.py tests/test_m6x_*.py
12 passed in 14.19s
```

Physics diff check:

```text
git diff main...HEAD -- src/gpuwrf/physics/
<empty>
```

One-hour coupled cadence probe with the stable reduced coupling and top-layer radiation fix:

```text
step 360: changed=121, fired=65/360, theta=[288.48, 492.81] K,
qv_max=0.01275, |v|max=10.52 m/s, |w|max=7.22 m/s,
mu=[62818.0, 120000.0]
summary: fired_rate=0.1806, changed=2080, nonfinite=0, clip=2080
```

The one-hour physical fields are acceptable for theta/qv/w, but sanitize firing is already above the AC4 threshold because `mu` reaches the legacy 120000 Pa clamp.

Six-hour direct proof with stronger acoustic damping (`PRESSURE_IMPLICIT_RELAXATION=0.05`, `MAX_INVERSE_DENSITY=0.02`) still fails:

```text
step 360:  changed=0,    theta=[290.60, 492.81], qv_max=0.01253, |w|max=1.36,  mu=[68924, 106150]
step 720:  changed=580,  theta=[288.06, 492.66], qv_max=0.01382, |w|max=2.45,  mu=[52763, 120000]
step 1080: changed=1809, theta=[286.89, 500.75], qv_max=0.01671, |w|max=5.09,  mu=[14536, 120000]
step 1440: changed=4730, theta=[284.90, 550.00], qv_max=0.01993, |w|max=7.73,  mu=[1000, 120000]
step 1800: changed=8003, theta=[280.03, 550.00], qv_max=0.02426, |w|max=8.68,  mu=[1000, 120000]
step 2160: changed=54923, theta=[150.00, 492.13], qv_max=0.03405, |w|max=9.31,  mu=[1000, 120000]
summary fired_rate=0.7602, changed=14880761, nonfinite=0, clip=14880761
```

This fails AC4 (`sanitize_firing_rate < 5%`) and AC5 (physical 24 h forecast). It also shows the reduced dycore is still being kept finite by sanitize after the 2-6 h window.

Unclipped-`mu` diagnostic over 6 h confirms the clamp is not merely too narrow:

```text
step 360:  mu=[62818, 132665], theta=[288.48, 492.81], qv_max=0.01275, nonfinite=0
step 720:  mu=[24717, 650879], theta=[-619595, 993443], qv_max=0.04316, nonfinite=0
step 1080: mu=[-2.59e7, 9.23e7], theta=nan, qv=nan, nonfinite=2568
```

So widening/removing the old `mu` clamp would not produce a valid forecast; it exposes a real reduced-dycore failure.

## AC Status

- AC1 physical sound speed: **partial**. `c^2 = gamma R T` is implemented and tested, but stable runs require reduced pressure/inverse-density coupling.
- AC2 per-cell CFL: **implemented** in diagnostics and tested for `n_acoustic=4`.
- AC3 canonical mu continuity: **implemented in code and unit-tested**, but coupled forecast behavior is not stable.
- AC4 Tier-2 lifted-cap PASS: **fail**. Six-hour direct proof has sanitize firing 76%.
- AC5 24 h finite/valid: **fail**. The model violates physical bounds well before 24 h.
- AC6 speedup preserved: **not rerun to PASS** because AC4/AC5 fail first.
- AC7 post-init transfer regression: **not closed**. Earlier smoke still showed the existing warmed audit H2D around 167904 bytes.
- AC8 no physics-kernel changes: **pass** by empty `git diff main...HEAD -- src/gpuwrf/physics/`.
- AC9 ADR-007 PASS: **not done** because evidence is failing.
- AC10 ADR-015: **not created as PASS ADR** because the acoustic formulation still contains non-canonical stabilizers.

## Commands Run

```bash
pytest -q tests/test_m4_dycore_step.py tests/test_m6x_*.py
git diff main...HEAD -- src/gpuwrf/physics/
python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6/performance/full_domain_batching_1h_probe.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/full_domain_batching_1h_probe --skip-nsys --skip-legacy-baseline-sanitize-audit
python scripts/m6_full_domain_batching.py --hours 6 --tier2-hours 6 --output artifacts/m6/performance/full_domain_batching_m6x_failed_6h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/full_domain_batching_m6x_failed_6h --skip-nsys --skip-legacy-baseline-sanitize-audit
```

The official 6 h harness was stopped after several minutes without completing; direct JAX probes above are the proof object for the failure decision.

## Unresolved Risks

The reduced M4 dycore still lacks enough WRF EM coupling to keep mass, pressure, and scalar transport mutually consistent over multi-hour windows. The `pb` split fixed the most obvious pressure-gradient bug, but the remaining stable configuration depends on non-canonical acoustic damping and still fails by 6 h. This supports the M6-S5 concern that option (a)-narrowed may still be insufficient without a larger WRF pressure/mass/vertical-implicit port.

## Next Decision Needed

Do not dispatch M7 from this branch. The manager should decide whether to:

1. Continue option (a) with a narrower follow-up focused on canonical WRF pressure/mu coupling (`pb`, `mub`, `mut`, `muu/muv`, `calc_p_rho`, `advance_uv/advance_all`) rather than only sound speed and mu tendency.
2. Escalate to option (c) re-architecture as allowed by the sprint contract, likely a true Klemp-Skamarock vertical-implicit pressure/mass path or a different dycore core.

