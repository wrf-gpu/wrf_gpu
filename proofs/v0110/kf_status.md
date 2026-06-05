# v0.11.0 KF Status

## Results

1. **Operational wiring**: PASS. `cu_physics=1` is accepted, registry status is `implemented`, dispatch routes to `gpuwrf.physics.cumulus_kf.step_kf_column`, and `CU_SCAN_ADAPTERS[1]` is `kf_adapter`.
2. **Column/savepoint parity**: PASS. Max tendency abs error `7.989e-08`, max relative error `7.111e-05`, max `RAINCV` abs error `1.864e-06` mm, all cases pass the predeclared v060 KF tolerances.
3. **d01 cu0-vs-cu1 sanity**: PASS_REUSED_GPU_ARTIFACTS. Reused the v040k two-date 6h d01 GPU artifacts: cu1 is executed/complete, stable finite, physical-range OK, and `RAINNC` changes versus cu0. Direct KF heating/`RAINCV` are proven by the column WRF savepoints because the d01 artifacts do not expose `RTHCUTEN`/`RAINCV` diagnostics.

## Carry-Over

- No fresh v0110 GPU d01 forecast was run; GPU use was deferred by the sprint constraint.
- The reused d01 artifacts keep the known `STABLE_BUT_CORE_FIELD_MISMATCH` verdict versus CPU-WRF core fields.
- `RAINC` is zero in the reused d01 artifacts; precipitation response appears in `RAINNC`, while direct KF `RAINCV` parity is covered by `proofs/v060/kf_savepoint_parity_report.json`.
