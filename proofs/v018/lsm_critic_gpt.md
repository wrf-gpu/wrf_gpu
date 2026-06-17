# v0.18 LSM Family GPT Critic Final Reverify

**Final verdict: ACCEPT.**

Branch checked: `worker/opus/v018-lsm` at `8fff69f6`.
Scope: reverify only the two prior GPT must-fixes from the `a4ee9153` critic pass:

1. `sf_surface_physics=4` Noah-MP generic coupled gate crash / missing carry seed.
2. FLQC ceiling relabel from "same WRF PSIT floor" to honest conservative guard.

## Must-Fix 1: Noah-MP Generic Coupled Gate

Accepted.

Code review:

- `OperationalNamelist` now has append-only `noahmp_land` with default `None`
  (`src/gpuwrf/runtime/operational_mode.py`).
- The field is included in `OperationalNamelist.tree_flatten` / `tree_unflatten` as
  static aux via `_StaticHolder`, not silently dropped. This is appropriate for the
  generic gate path because `_initial_carry_for_run` consumes it at trace setup and
  promotes it into the real `OperationalCarry` pytree.
- `_initial_carry_for_run` now seeds both `noahmp_land` and a concrete finite
  3-tuple `noahmp_rad = noahmp_initial_rad(...)` when `use_noahmp=True` and a
  `noahmp_land` bundle is supplied.
- The production daily/nested path remains non-breaking: leaving `noahmp_land=None`
  preserves the existing post-`carry.replace` seeding path.
- `proofs/v016/coupled_coverage_gate.py` now builds the full Noah-MP real-case
  bundle for option 4: `build_noahmp_land_state`, `build_noahmp_params`,
  `noahmp_static`, `noahmp_energy_params`, `noahmp_rad_params`, `noahmp_nroot`, and
  `noahmp_land`.

Verification:

- CPU-only pytree assertion passed:
  `OperationalNamelist` flatten/unflatten preserves `noahmp_land`, and committed
  JSON assertions for `lsm1`, `lsm4`, and `lsm7` all pass.
- Before the hibernate warning, I also got a direct GPU probe:
  `PYTREE_NOAHMP_OK children=47 land_type=NoahMPLandState rad_shapes=[(128, 128), (128, 128), (128, 128)] nroot=4`.
- Committed/fresh `proofs/v016/coverage/lsm4_gate.json` is a genuine coupled PASS:
  `all_finite=true`, `bounds_violations=[]`, `hard_gate_fails=[]`,
  `review_flags=[]`, worst dynamics RMSE ratio `0.2325706656435294`.

Ruling: the fix is faithful infrastructure, not a physics hack. It supplies the same
WRF-derived Noah-MP land/radiation state that the proven production harnesses already
seeded manually; it does not mask or damp model state.

## Must-Fix 2: FLQC Relabel

Accepted.

Code/report review:

- FLHC remains correctly labeled as WRF's exact heat-coefficient ceiling from
  `module_sf_sfclay.F:706` (`PSIT>=2`), giving
  `FLHC <= CPM*RHOX*UST*KARMAN/(2*PRT)`.
- FLQC is no longer claimed to share that exact land+water PSIT floor. The source now
  labels it as a conservative guard from WRF's water-branch `PSIQ>=2` ceiling at
  `module_sf_sfclay.F:731`, explicitly noting that WRF does not floor land `PSIQ`.
- The arithmetic remains numerically identical to the accepted guard:
  `flqc_max = rho * mavail * ust * KARMAN_SFCLAY / PSIQ_WATER_FLOOR`.
- I found and fixed one stale docstring sentence in
  `src/gpuwrf/coupling/slab_surface_hook.py` that still grouped FLQC with the exact
  FLHC ceiling. No executable arithmetic changed.
- `proofs/v018/lsm_family_report.md` and
  `proofs/v018/lsm_family_status.json` now match the code and no longer overclaim the
  FLQC guard.

Verification:

- CPU text check passed: source distinguishes exact FLHC ceiling from conservative
  FLQC guard, uses `PSIQ_WATER_FLOOR`, and no longer contains the stale overclaim.
- CPU pytest subset passed:
  `11 passed in 2.75s`
  (`tests/test_v018_lsm_architecture_boundary.py`,
  `test_soilprop_matches_pxlsm_oracle_loam_and_sand`,
  `test_landuse_season_matches_wrf`).
- `proofs/v016/coverage/lsm1_gate.json`: PASS, `all_finite=true`,
  `bounds_violations=[]`, worst dynamics RMSE ratio `0.16465064046982456`.
- `proofs/v016/coverage/lsm7_gate.json`: PASS, `all_finite=true`,
  `bounds_violations=[]`, worst dynamics RMSE ratio `0.18049805268483624`.

Ruling: the FLHC PSIT-floor proof remains WRF-faithful, and the FLQC wording is now
honest. No ad-hoc clamp is being misrepresented as a WRF exact land formula.

## Regression Scope

No new long GPU run was started after the hibernate warning. I relied on the committed
L2 gate JSON plus code reading and CPU-only checks as requested.

No regression evidence found in the touched paths:

- Noah-MP addition is append-only/default-`None` and preserves the existing pipeline
  post-replace seeding path.
- `sf=1`, `sf=4`, and `sf=7` committed/fresh L2 gate JSONs all pass with finite
  states, zero bounds violations, no hard fails, and no review flags.
- The FLQC relabel is documentation/constant naming only; the numeric guard remains
  unchanged.

Residual note: I did not rerun 72h sibling-family greens during this final lean pass.
The code review shows no broad-path change beyond the append-only namelist field and
the already-green LSM gate artifacts.

## Commands Run

```bash
# Stopped the unnecessary already-started sf=7 rerun after the hibernate warning.
kill -TERM <sf7_gate_pid> <with_gpu_lock_pid>

env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true pytest -q \
  tests/test_v018_lsm_architecture_boundary.py \
  tests/test_v018_lsm_static_extract.py::test_soilprop_matches_pxlsm_oracle_loam_and_sand \
  tests/test_v018_lsm_static_extract.py::test_landuse_season_matches_wrf
# 11 passed in 2.75s

env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python - <<'PY'
# asserted lsm1/lsm4/lsm7 committed JSON PASS fields and OperationalNamelist
# noahmp_land flatten/unflatten preservation
PY
# CPU_ASSERTIONS_OK lsm_json=1,4,7 noahmp_land_roundtrip=True

python - <<'PY'
# asserted slab_surface_hook/report FLQC relabel text and absence of stale overclaim
PY
# CPU_TEXT_CHECK_OK flqc_relabel_no_overclaim=True
```

## Proof Objects

- `proofs/v018/lsm_critic_gpt.md` (this final report).
- `proofs/v016/coverage/lsm4_gate.json` (Noah-MP generic coupled PASS).
- `proofs/v016/coverage/lsm1_gate.json` and `proofs/v016/coverage/lsm7_gate.json`
  (slab/PX still PASS after FLQC relabel).
