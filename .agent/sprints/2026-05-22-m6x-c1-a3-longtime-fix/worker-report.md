# c1-A3 Worker Report — Long-Time Fix-Hint Sequence

## Objective

Apply the ordered c1-A3 long-time instability sequence from the role prompt:
pre-flight sanitize-disable diagnostic, FIX #A non-periodic dycore stencil
edges, FIX #B acoustic `dz` floor, FIX #C1 Klemp/WRF `smdiv`, and FIX #C2
horizontal `ph` advection. Re-run the 1 h coupled probe after each fix and
stop short of 24 h/speedup if the final 1 h probe remains red.

The requested in-checkout path
`.agent/sprints/2026-05-22-m6x-bughunt3-longtime/bughunt3-report.md` was absent
from `/tmp/wrf_gpu2_c1`. I located and read the report in the sibling bughunt
worktree:
`/tmp/wrf_gpu2_m6x_bughunt3/.agent/sprints/2026-05-22-m6x-bughunt3-longtime/bughunt3-report.md`.
The controlling §3/§5 lines used here were bughunt3 lines 91-180 and 203-213.

## Files Changed

- `scripts/m6_full_domain_batching.py`
- `src/gpuwrf/dynamics/acoustic.py`
- `src/gpuwrf/dynamics/advection.py`
- `tests/test_m4_advection.py`
- `tests/test_m6x_fallback_c1_acoustic.py`
- `.agent/sprints/2026-05-22-m6x-c1-a3-longtime-fix/worker-report.md`

Frozen-file audit: `git diff -- src/gpuwrf/contracts/state.py src/gpuwrf/dynamics/tridiag.py src/gpuwrf/coupling/physics_couplers.py` produced no diff. I did not edit `state.py`, `tridiag.py`, `physics_couplers.py`, or any `src/gpuwrf/physics/**` file.

## Fixes Applied

### STEP 0 — sanitize-disabled diagnostic

Added `--disable-sanitize` to `scripts/m6_full_domain_batching.py`. This path runs the same coupled candidate timestep without `sanitize_state`, counts non-finite State leaves, and writes a dedicated diagnostic JSON. The final committed diagnostic implementation avoids host/device scalar reads inside the timestep loop; scalar counts are read after the loop.

Result: `artifacts/m6x-fallback-c1/c1_a3_sanitize_disabled_1h_diagnostic.json` went non-finite at step 45, with `nonfinite_steps=316`, `max_nonfinite_count=6,932,090`, and `all_state_leaves_finite=false`. This matches bughunt3 §5 lines 211-212: sanitize is catching a real instability symptom, not initiating it.

### STEP 1 — FIX #A periodic to edge-mirror in the seven named dycore stencils

Implemented the c1-A2 edge-mirror pattern from `physics_couplers.py:98-109` in the seven named call sites from bughunt3 §3 lines 96-99 and §5 line 203:

- `acoustic._grad_x_to_u`, `_grad_y_to_v`: boundary face gradients are zero instead of wrapping opposite-domain pressure.
- `acoustic._mass_to_u_face_2d`, `_mass_to_v_face_2d`: μ edge faces mirror the adjacent edge mass point.
- `advection._periodic_flux5_faces`: historical name retained, but boundary flux faces now use local edge values and interior high-order stencils use clipped edge extrapolation.
- `advection._mass_to_u_face`, `_mass_to_v_face`: edge faces mirror local mass values.

Added boundary regressions for x/y pressure gradients, acoustic μ face interpolation, advection u/v face interpolation, and x/y scalar flux boundary faces. The old acoustic manufactured-wave oracle was updated from a periodic `jnp.roll` Laplacian to the non-wrapping interior Laplacian.

1 h result: `c1_a3_fixA_1h.json` failed. `fired_steps=319`, `nonfinite_count=791,254,094`, `clip_count=274,104,113`, `step_firing_rate=0.8861111111111111`. The expected ≥10x/≥5x drop did not occur.

### STEP 2 — FIX #B acoustic `dz` floor

Applied the one-line floor from bughunt3 §3 lines 121-150:
`jnp.where(dz > 0.0, jnp.maximum(dz, 1.0), _flat_dz(grid))` in
`acoustic._layer_thickness_m`, matching `advection._dz_from_state`.

Added `test_layer_thickness_m_floors_at_1m_for_small_positive_dz`.

1 h result: `c1_a3_fixAB_1h.json` failed. `fired_steps=319`, `nonfinite_count=791,637,687`, `clip_count=274,864,417`, `step_firing_rate=0.8861111111111111`.

### STEP 3 — FIX #C1 WRF/Klemp `smdiv`

Implemented Klemp 2007 §3d pressure divergence damping as requested in bughunt3 §3 lines 156-178 and §5 line 207. The acoustic scan carry now includes previous undamped perturbation pressure. After the main pressure update, the code applies:

```python
p_pert_next = p_pert_undamped + smdiv * (p_pert_undamped - p_prev)
```

with `SMDIV_DIVERGENCE_DAMPING = 0.1`, citing WRF
`dyn_em/module_small_step_em.F:557-565`. Added a regression proving the previous-pressure carry changes pressure by the expected `smdiv` amount. The manufactured acoustic response test now expects the documented `sqrt(1 + smdiv)` pressure-response factor.

1 h result: `c1_a3_fixABC1_1h.json` failed. `fired_steps=322`, `nonfinite_count=787,529,577`, `clip_count=260,700,699`, `step_firing_rate=0.8944444444444445`.

### STEP 4 — FIX #C2 horizontal `ph` advection

Added `advect_w_face_scalar_horizontal` and included `ph` in
`compute_advection_tendencies`. The helper collocates mass-point horizontal
velocities to w-face levels via `_mass_to_w_face`, computes non-periodic
horizontal face fluxes, and returns horizontal flux divergence on `(nz+1, ny,
nx)` `ph`. This follows bughunt3 §3 lines 158-178 and WRF citations
`module_em.F:1292 advect_ph_implicit`, `module_em.F:436`, and
`module_big_step_utilities_em.F:1365,1435` (`rhs_ph` geopotential tendency).

Added `test_ph_horizontal_advection_updates_w_face_tendency`.

1 h result: `c1_a3_fixABCfull_1h.json` failed. `fired_steps=326`,
`nonfinite_count=821,114,794`, `clip_count=315,768,791`,
`step_firing_rate=0.9055555555555556`. The final probe also emitted a
`PH` overflow warning during float32 output serialization before the formal JSON
was written.

## Commands Run

```bash
sed -n '1,260p' .agent/sprints/2026-05-22-m6x-c1-a3-longtime-fix/role-prompts/worker.md
sed -n '1,260p' PROJECT_CONSTITUTION.md
sed -n '1,260p' AGENTS.md
sed -n '1,280p' .agent/sprints/2026-05-22-m6x-c1-a3-longtime-fix/sprint-contract.md
sed -n '/^## 3\./,/^## 4\./p' /tmp/wrf_gpu2_m6x_bughunt3/.agent/sprints/2026-05-22-m6x-bughunt3-longtime/bughunt3-report.md
sed -n '/^## 5\./,/^## 6\./p' /tmp/wrf_gpu2_m6x_bughunt3/.agent/sprints/2026-05-22-m6x-bughunt3-longtime/bughunt3-report.md
sed -n '90,120p' src/gpuwrf/coupling/physics_couplers.py
sed -n '557,565p' /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F
sed -n '1288,1295p' /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_em.F
rg -n "rhs_ph|SUBROUTINE rhs_ph|advect_ph" /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_big_step_utilities_em.F /mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_em.F
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6x-fallback-c1/c1_a3_sanitize_disabled_1h_diagnostic.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a3_sanitize_disabled_1h_diagnostic --skip-nsys --skip-legacy-baseline-sanitize-audit --disable-sanitize
pytest -q tests/test_m4_advection.py tests/test_m4_acoustic.py tests/test_m6x_fallback_c1_*.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6x-fallback-c1/c1_a3_fixA_1h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a3_fixA_1h --skip-nsys --skip-legacy-baseline-sanitize-audit
pytest -q tests/test_m4_advection.py tests/test_m4_acoustic.py tests/test_m6x_fallback_c1_*.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6x-fallback-c1/c1_a3_fixAB_1h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a3_fixAB_1h --skip-nsys --skip-legacy-baseline-sanitize-audit
pytest -q tests/test_m4_advection.py tests/test_m4_acoustic.py tests/test_m6x_fallback_c1_*.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6x-fallback-c1/c1_a3_fixABC1_1h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a3_fixABC1_1h --skip-nsys --skip-legacy-baseline-sanitize-audit
pytest -q tests/test_m4_advection.py tests/test_m4_acoustic.py tests/test_m6x_fallback_c1_*.py
PYTHONPATH=src python scripts/m6_full_domain_batching.py --hours 1 --tier2-hours 1 --output artifacts/m6x-fallback-c1/c1_a3_fixABCfull_1h.json --output-dir /home/enric/.cache/gpuwrf_outputs/m6/c1_a3_fixABCfull_1h --skip-nsys --skip-legacy-baseline-sanitize-audit
pytest -q tests/test_m4_advection.py tests/test_m4_acoustic.py tests/test_m6x_fallback_c1_*.py
git diff -- src/gpuwrf/contracts/state.py src/gpuwrf/dynamics/tridiag.py src/gpuwrf/coupling/physics_couplers.py
```

Final test result: `32 passed in 34.45s`.

## Proof Objects Produced

- `artifacts/m6x-fallback-c1/c1_a3_sanitize_disabled_1h_diagnostic.json`
- `artifacts/m6x-fallback-c1/c1_a3_fixA_1h.json`
- `artifacts/m6x-fallback-c1/c1_a3_fixAB_1h.json`
- `artifacts/m6x-fallback-c1/c1_a3_fixABC1_1h.json`
- `artifacts/m6x-fallback-c1/c1_a3_fixABCfull_1h.json`
- Matching `.outputs.json` manifests for each sanitized 1 h probe.
- WRF output NPZs under `/home/enric/.cache/gpuwrf_outputs/m6/c1_a3_*`.
- JAX trace artifacts listed inside each probe JSON.

## Result

Status: **all requested fixes implemented, final 1 h gate still FAIL**.

I did not run the 24 h or speedup probes and did not amend ADR-007 because the
contract gates those on a passing 1 h probe. The ordered sequence did not close
the long-time instability. FIX #A and #B were essentially neutral, C1 slightly
reduced nonfinite count but increased fired steps, and C2 made the final 1 h
sanitize metrics worse.

## Unresolved Risks

- Per bughunt3 §5 lines 213-214, the next candidates are re-opening bug-hunt #2
  Hypothesis A around buoyancy in `_vertical_implicit_w`, or moving to c2
  semi-implicit/reference-state work.
- `rg -n "jnp\.roll" src/gpuwrf/dynamics/acoustic.py src/gpuwrf/dynamics/advection.py`
  still shows `jnp.roll` in general derivative utilities such as
  `derivative5_upwind`; those were outside the seven named FIX #A call sites but
  are still used by velocity advection. This may be another periodic-wrap source
  for the manager/reviewer to consider.
- `host_device_transfer_bytes` rose from `167,904` in FIX #A/#B to `7,387,776`
  after C1/C2 trace runs. I am not making a performance claim, but this should be
  reviewed before any future speedup gate.
- The final C2 probe emitted an overflow warning while writing `PH` to float32,
  consistent with the red sanitize metrics.

## Next Decision Needed

Escalate per sprint contract and bughunt3 §5: decide whether to re-open
bug-hunt #2 Hypothesis A for missing buoyancy in `_vertical_implicit_w`, audit
the remaining periodic velocity-advection derivative helpers, or move to the c2
semi-implicit path.
