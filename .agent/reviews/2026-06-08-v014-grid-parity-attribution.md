# V0.14 Grid-Parity Attribution

Date: 2026-06-08
Worker: GPT xhigh
Scope: CPU-only retained-wrfout attribution. No GPU run. No TOST resume. No `src/` edits.

## Objective

Implement the V0.14 all-comparable-field CPU-WRF-vs-GPU-WRF grid-cell envelope and rank the next root-cause fix target using the retained Case 3 wrfouts plus Case 1/2 JSON aggregates.

## Files Changed

- `proofs/v014/grid_cell_envelope.py`
- `proofs/v014/grid_cell_envelope.json`
- `proofs/v014/grid_cell_envelope.md`
- `.agent/reviews/2026-06-08-v014-grid-parity-attribution.md`

## Commands Run

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m py_compile proofs/v014/grid_cell_envelope.py
JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 24-31 python proofs/v014/grid_cell_envelope.py
JAX_PLATFORMS=cpu PYTHONPATH=src python -m json.tool proofs/v014/grid_cell_envelope.json >/dev/null
```

I also ran CPU-only read/probe commands to inspect the writer inventory, retained wrfout variables, static-field mismatch ordering, and small shift probes for `XLAT`, `XLONG`, `HGT`, and `MAPFAC_M`.

## Proof Objects

- `proofs/v014/grid_cell_envelope.json`
- `proofs/v014/grid_cell_envelope.md`

The JSON validates. The final proof run produced `3` cases: `1` spatial retained-wrfout case and `2` aggregate-only JSON cases.

## Coverage

Case 3 retained wrfouts:

- Compared `50` dynamic fields with full metrics by lead and lead block.
- Audited `48` static/grid/time-invariant fields separately from prognostic RMSE.
- Audited `2` time metadata fields.
- Enumerated `26` writer fields absent from retained GPU wrfouts.
- Enumerated `4` GPU-emitted writer fields absent from CPU truth: `QNSNOW`, `QNGRAUPEL`, `QNCLOUD`, `QNCCN`.
- Found `0` dimension-incompatible emitted fields.

Cases 1/2:

- Retained GPU wrfouts were not available.
- Used stored JSON aggregates for `T2`, `U10`, and `V10`.
- Marked all other writer fields spatial-unavailable for those cases.

## Key Findings

- The highest-confidence root signal is static metric mismatch, not station skill. Case 3 has `31` non-exact static/grid fields.
- Largest static mismatches are vertical-coordinate coefficients: `C2H`/`C2F` max `95,000 Pa`, `C4F` max `26,782.75 Pa`, `C4H` max `26,740.12 Pa`, `RDN` max `161.67`. `ZNU`, `ZNW`, `DNW`, `RDNW`, `FNM`, `FNP`, `P_TOP`, `RDX`, `RDY`, `LANDMASK`, and `LU_INDEX` are exact.
- Horizontal static mismatch is also present but smaller: `HGT` max `228.13 m`, lat/lon max about `0.027 deg`, map factors max about `9.52e-4`.
- Dynamic errors are broad and consistent with bad metric/mass coupling: `PSFC` RMSE `525 Pa`, `PH` `336 m2 s-2`, `MU` `274 Pa`, `P` `228 Pa`, `U` `4.61 m/s`, `V` `5.83 m/s`, `U10` `2.07 m/s`, `V10` `2.52 m/s`.
- Surface/radiation are secondary but not clean: `SWDOWN` RMSE `113 W/m2`, `GLW` `25.6 W/m2`, `HFX` `64.3 W/m2`, `LH` `36.9 W/m2`, `TSK` `3.05 K`.
- Spatial splits show the problem is not only land or boundary-frame: PSFC is similar over land/ocean, V10 is worse over ocean, and 3D U/V remain large over all quadrants.

## Ranked Root-Cause Hypotheses

1. **WRF vertical-coordinate / grid-metric payload mismatch is the first fix target.**
   Evidence: the biggest static failures are `C2H/C2F/C4H/C4F/RDN`; these feed pressure, mass weighting, hydrostatic reconstruction, PGF, and vertical operators. Writer code copies these from `GridSpec.metrics`, so the likely owner is metric construction or propagation, not writer formatting.

2. **Pressure-gradient / mass-wind coupling is the next operator suspect after metrics are exact.**
   Evidence: PSFC/MU/P/PH and 3D U/V are already divergent, and U10/V10 are coupled to low-level winds. This should not be fixed first while the metric payload is non-identical.

3. **Radiation/surface-energy coupling is a secondary amplifier.**
   Evidence: SW/LW and flux errors are large enough to affect T2/TSK/PBLH, but they are downstream of the metric/mass/wind failure until proven otherwise.

4. **Pure 10 m diagnostic bug is disfavored.**
   Evidence: 3D U/V RMSE exceeds U10/V10 RMSE, so the surface wind diagnostics are not the only failing layer.

## Next Fix Sprint Recommendation

Run a narrow CPU-first metric-parity fix sprint before any dycore or radiation changes.

Writable ownership for that sprint:

- `src/gpuwrf/init/real_init/vertical_coord.py`
- `src/gpuwrf/dynamics/metrics.py`
- `src/gpuwrf/contracts/grid.py` only if the metric contract is wrong and an ADR/manager approval exists
- proof files under `proofs/v014/`
- one review report under `.agent/reviews/`

Read-only unless evidence proves a mapping bug:

- `src/gpuwrf/io/wrfout_writer.py`
- `src/gpuwrf/runtime/operational_mode.py`
- pressure-gradient/acoustic/diffusion code
- radiation and surface-layer code

Proof gates:

1. Static metric gate: compare `ZNU/ZNW/DN/DNW/RDN/RDNW/FNM/FNP/CF*/C*H/C*F/P_TOP/RDX/RDY/MAPFAC*/F/E/SINALPHA/COSALPHA/HGT/XLAT/XLONG` against CPU truth at h1. Required result: exact for fields that should be payload copies, documented predeclared tolerance only for known fp32 C1 noise.
2. Writer-payload gate: write one wrfout frame without advancing and prove the static payload matches the in-memory `DycoreMetrics`.
3. Re-run `proofs/v014/grid_cell_envelope.py`. Required movement: static mismatch count to zero or documented fp32-only exceptions, then reassess PSFC/U/V/U10/V10 before touching dynamics.
4. Only if static metrics are exact and grid RMSE remains high, start the same-state first-timestep PGF/mass-wind localization sprint.

## Unresolved Risks

- Case 1/2 have no retained GPU wrfouts, so all non-`T2/U10/V10` attribution is Case 3 only.
- The current proof compares emitted wrfout payloads. It strongly implicates `GridSpec.metrics`, but it does not by itself prove whether runtime dynamics consumed the same bad metrics or only emitted them.
- No WRF savepoint tendency localization was run in this sprint.

## Memory-Patch Recommendation

Make the grid-first process rule stable memory: before any TOST, FP32, or speed sprint can close, all emitted static grid/vertical metrics must be exact or predeclared-tolerance-matching against CPU-WRF truth, and the all-comparable-field grid envelope must be rerun.
