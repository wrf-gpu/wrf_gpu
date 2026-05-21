# M5-S3.zzz Worker Report - RRTMG LW Closeout

## Objective

Close the M5-S3.z LW intermediate-oracle debt by replacing nearest-pressure LW gas optical depths and uniform Planck fractions with WRF-bound `setcoef` + `taumol` branches in `src/gpuwrf/physics/rrtmg_lw.py`, validate all 16 LW bands against `data/fixtures/rrtmg-intermediate-oracle-v1.npz`, and then check whether this is sufficient for strict Tier-1 LW broadband parity.

One repository metadata repair was required before model-code work: `.agent/sprints/2026-05-21-m5-s3zzz-rrtmg-lw-closeout/sprint-contract.md` was absent. I created it from the manager-provided sprint prompt/AC list so the local "no model code without a sprint contract" rule was satisfied.

## Files Changed

- `src/gpuwrf/physics/rrtmg_lw.py`
  - Added a WRF-native LW table loader from the pinned raw `RRTMG_LW_DATA` payload inside `rrtmg-tables-v1.npz`.
  - Added WRF `setcoef` gas columns, pressure/temperature interpolation factors, minor-gas scaling, species ratios, CFC/CCl4 columns, and H2O continuum factors.
  - Added all 16 reduced-g `taumol` branches plus Planck `fracs` interpolation.
  - Kept an explicit branch-acceptance mask and nearest-pressure fallback path; all 16 current branches are accepted.
  - Split gas-only `intermediate.tau` from cloud-added transfer optical depth.
  - Added a `lax.scan` barrier over LW band output selection.
- `src/gpuwrf/validation/rrtmg_intermediate_oracles.py`
  - Kept/used per-band LW `validate_lw_taug_per_band` and `validate_lw_fracs_per_band`.
  - Added LW `setcoef`/`taumol` WRF citations to the artifact.
  - Made LW band status depend on per-band `taug`/`fracs` gates, with `FULL_BRANCH_ACCEPTED` vs `FALLBACK_NEAREST_PRESSURE`.
  - Pinned intermediate validation to CPU because it is a correctness proof and avoids GPU allocator pressure/backend float32 drift.
- `tests/test_m5_rrtmg_intermediate_oracles.py`
  - Added coverage for LW `fracs` validation and the now-passing intermediate artifact.
- `.agent/decisions/ADR-009-rrtmg-jax-implementation.md`
  - Held ADR-009 at NOT-PARITY because LW Tier-1 still fails after `taumol/fracs` closure.
  - Documented the remaining root cause as downstream `cldprmc`/`rtrnmc` transfer/source parity debt.
- `artifacts/m5/*`
  - Regenerated intermediate, per-band, Tier-1, Tier-2, HLO, profile, and gate artifacts.
- `.agent/sprints/2026-05-21-m5-s3zzz-rrtmg-lw-closeout/sprint-contract.md`
  - Added missing contract derived from the sprint prompt.

No changes were made to `src/gpuwrf/physics/rrtmg_sw.py`; `git diff main...HEAD -- src/gpuwrf/physics/rrtmg_sw.py` is empty.

## WRF Binding

Common LW setup follows WRF `setcoef` at `module_ra_rrtmg_lw.F:3556-3921`. The Planck/non-isothermal source fields already wired by M5-S3.z are kept from the same setcoef/rtrnmc path, with `rtrnmc` source machinery cited at `module_ra_rrtmg_lw.F:3270-3340`.

Per-band branch citations and acceptance:

| Band | WRF branch | Major/minor path implemented | taug | fracs | Status |
|---:|---|---|---|---|---|
| 1 | `module_ra_rrtmg_lw.F:5073-5166` | H2O + N2 continuum, lower/upper corradj | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 2 | `module_ra_rrtmg_lw.F:5169-5238` | H2O + self/foreign continuum | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 3 | `module_ra_rrtmg_lw.F:5241-5553` | H2O/CO2 binary + N2O minor | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 4 | `module_ra_rrtmg_lw.F:5556-5812` | lower H2O/CO2, upper O3/CO2, stratospheric factors | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 5 | `module_ra_rrtmg_lw.F:5815-6087` | H2O/CO2 + O3/CCl4 minor, upper O3/CO2 | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 6 | `module_ra_rrtmg_lw.F:6090-6175` | H2O + CO2 minor + CFC11/CFC12 | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 7 | `module_ra_rrtmg_lw.F:6178-6449` | H2O/O3 + CO2 minor, upper O3 + CO2 minor | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 8 | `module_ra_rrtmg_lw.F:6452-6572` | H2O/O3 + CO2/O3/N2O minor + CFCs | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 9 | `module_ra_rrtmg_lw.F:6575-6835` | H2O/CH4 + N2O minor | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 10 | `module_ra_rrtmg_lw.F:6838-6902` | H2O + self/foreign continuum | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 11 | `module_ra_rrtmg_lw.F:6905-6982` | H2O + O2 minor | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 12 | `module_ra_rrtmg_lw.F:6985-7186` | H2O/CO2 lower, upper zero | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 13 | `module_ra_rrtmg_lw.F:7189-7445` | H2O/N2O + CO2/CO minor, upper O3 minor | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 14 | `module_ra_rrtmg_lw.F:7448-7506` | CO2 lower/upper + lower continuum | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 15 | `module_ra_rrtmg_lw.F:7509-7731` | N2O/CO2 + N2 minor, upper zero | PASS | PASS | FULL_BRANCH_ACCEPTED |
| 16 | `module_ra_rrtmg_lw.F:7734-7940` | H2O/CH4 lower, CH4 upper | PASS | PASS | FULL_BRANCH_ACCEPTED |

## Before/After Evidence

Before this sprint, `artifacts/m5/rrtmg_per_band_status.json` listed all 16 LW bands as failing/debt from the M5-S3.z nearest-pressure approximation. After this sprint, `artifacts/m5/rrtmg_per_band_status.json` lists all 16 LW bands as `FULL_BRANCH_ACCEPTED`, with separate `taug_gate=PASS` and `fracs_gate=PASS`.

Current intermediate proof:

- `artifacts/m5/rrtmg_intermediate_validation.json`: `pass=true`.
- LW per-band `taug/fracs`: all `PASS` at `abs<=1e-8 + rel<=1e-4`.
- WRF single-precision `secdiff`: accepted at `abs<=1e-6 + rel<=1e-6`; max abs `1.283116610739654e-7`.
- Clip-pinning check: the oracle NPZ has zero `*_clip_count` fields.

Representative LW per-band max errors from the final status artifact:

- Band 1: `max_abs_taug=2.458167402888648`, `max_rel_taug=4.387697350516683e-6`, `max_abs_fracs=0`.
- Band 7: `max_abs_taug=4.291534423828125e-5`, `max_rel_taug=6.11781068026994e-6`, `max_abs_fracs=2.9802322387695312e-8`.
- Band 16: `max_abs_taug=7.149072659196065e-6`, `max_rel_taug=2.741183112998325e-6`, `max_abs_fracs=2.834048817845769e-8`.

Strict Tier-1 LW did not close:

- `artifacts/m5/tier1_rrtmg_lw_parity.json`: `pass=false`.
- Max abs residuals: `flux_down=59.568065480560136 W m-2`, `flux_up=46.99548470324402 W m-2`, `toa_up=23.93536747974062 W m-2`, `column_net_heating=19.463177478278368 W m-2`.
- Heating max abs is `9.681600304132737e-05 K s-1`, inside the `1e-4 K s-1` absolute threshold, but flux fields fail.

Launch and transfer evidence:

- `artifacts/m5/rrtmg_profile.json`: `raw_hlo_launch_marker_count=97`, `kernel_launches_per_step=97`, so raw equals reported and no `min(raw, cap)` launch fudge is used.
- LW raw launches are `43`; AC4 target `<=4` is not met despite the `lax.scan` barrier.
- Transfer audit fields remain zero: `host_to_device_bytes_post_init=0`, `device_to_host_bytes_post_init=0`, `host_device_transfer_bytes=0`.

Harness symbols preserved:

- `nm data/scratch/wrf_rrtmg_harness` still exposes `__module_ra_rrtmg_sw_MOD_rrtmg_swrad`, `__module_ra_rrtmg_lw_MOD_rrtmg_lwrad`, `__rrtmg_lw_setcoef_MOD_setcoef`, `__rrtmg_lw_taumol_MOD_taumol`, `__rrtmg_lw_rtrnmc_MOD_rtrnmc`, `__rrtmg_sw_setcoef_MOD_setcoef_sw`, `__rrtmg_sw_taumol_MOD_taumol_sw`, and `__rrtmg_sw_spcvmc_MOD_spcvmc_sw`.

## Commands Run

```bash
PYTHONPATH=src JAX_PLATFORM_NAME=cpu python -m gpuwrf.validation.rrtmg_intermediate_oracles
python scripts/m5_run_rrtmg.py
python scripts/m5_gate_rrtmg.py
cat artifacts/m5/rrtmg_intermediate_validation.json | jq '.lw'
cat artifacts/m5/rrtmg_per_band_status.json | jq '.lw_bands'
cat artifacts/m5/tier1_rrtmg_lw_parity.json | jq '.pass, .per_field_max_abs_err'
pytest -q tests/test_m5_rrtmg_*.py
git diff main...HEAD -- src/gpuwrf/physics/rrtmg_sw.py
nm data/scratch/wrf_rrtmg_harness | rg -i "rrtmg_(sw|lw)(rad|init)|taumol|rtrnmc|setcoef|spcvmc"
python - <<'PY'
import numpy as np
with np.load('data/fixtures/rrtmg-intermediate-oracle-v1.npz', allow_pickle=False) as f:
    print(len([name for name in f.files if name.endswith('_clip_count')]))
PY
```

## Proof Objects Produced

- `artifacts/m5/rrtmg_intermediate_validation.json`
- `artifacts/m5/rrtmg_per_band_status.json`
- `artifacts/m5/tier1_rrtmg_lw_parity.json`
- `artifacts/m5/tier1_rrtmg_sw_parity.json`
- `artifacts/m5/tier2_rrtmg_invariants.json`
- `artifacts/m5/rrtmg_profile.json`
- `artifacts/m5/rrtmg_gate_result.json`
- `artifacts/m5/hlo_dump/rrtmg_lw_production.txt`
- `artifacts/m5/hlo_dump/rrtmg_lw_debug_stripped.txt`
- `artifacts/m5/hlo_dump/rrtmg_sw_production.txt`
- `artifacts/m5/hlo_dump/rrtmg_sw_debug_stripped.txt`

## Unresolved Risks

- LW Tier-1 still fails after complete `taumol/fracs` closure, so the remaining dominant LW error is downstream transfer/source behavior, not gas optical depth.
- The next LW oracle should bind `cldprmc` and `rtrnmc` internals, especially source recurrence, surface reflection, and band/g-point flux accumulation.
- The `lax.scan` barrier is present but did not achieve launch fusion; raw LW launches remain `43`, and combined raw launches are `97`.
- Native LW tables are reconstructed at runtime from the pinned raw payload. The implementation now caches host NumPy arrays to avoid JAX tracer leakage, but a future cleanup should move these native tables into first-class `RRTMGTableBundle` leaves.
- `python scripts/m5_run_rrtmg.py` and `python scripts/m5_gate_rrtmg.py` exit nonzero because Tier-1 SW/LW strict gates still fail; this is expected for this handoff and documented in ADR-009.

## Next Decision Needed

Start M5-S3.zzzzz for LW `cldprmc` + `rtrnmc` intermediate oracles before any LW-PARITY claim. M5-S3.zzzz should continue SW broadband closeout independently; this worker did not touch `rrtmg_sw.py`.
