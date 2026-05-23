# Worker Report - M6.x ADR-023 MPAS Column-Slice Oracle

Summary: Implemented the F1 closure artifact as a NumPy-only MPAS-derived single-column slice oracle, a pytest comparison against the current c2 `vertical_acoustic_update`, and a warm-bubble fixture on disk. The oracle does not claim MPAS equivalence; it ports the named MPAS recurrence blocks and reports the measured prototype deviation.

## Files Changed

- `src/gpuwrf/validation/mpas_oracles/__init__.py`
- `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py`
- `tests/test_m6x_mpas_column_slice_oracle.py`
- `data/fixtures/mpas_column_slice/warm_bubble_2km.npz` - produced on disk, 24 KB, left ignored by git per repository binary-fixture rule.
- `.agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/proof_slice.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/proof_deviation.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/worker-report.md`

## MPAS Lines Ported

- `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/MPAS-Model-5.3/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1589-1651`: `dtseps`, `cofrz`, `cofwr`, `cofwz`, `coftz`, `cofwt`, `a_tri`, `alpha_tri`, and `gamma_tri`.
- Same file `:2038-2041`: `resm = (1 - epssm) / (1 + epssm)`.
- Same file `:2146-2169`: off-centered `rs`, `ts`, and `rw_p` RHS assembly.
- Same file `:2175-2182`: upward sweep and downward back-substitution.
- Same file `:2184-2193`: Rayleigh damping block represented with `dss == 0`, making it an exact no-op for this validation slice.
- Same file `:2195-2208`: `wwAvg` accumulation plus `rho_pp` and `rtheta_pp` reconstruction.
- Scenario setup cites `/mnt/data/.../src/core_init_atmosphere/mpas_init_atm_cases.F:1657-1690` for the warm-bubble theta perturbation form.

## Discretization Choices

The implementation uses one-based NumPy work arrays so loop bounds match the Fortran blocks directly. The slice is a flat, dry, single-column reduction with uniform `dz`, `zz = 1`, `fzm = fzp = 0.5`, zero horizontal fluxes, zero Rayleigh damping, and `epssm = 0.1` by default. `rw_p` is MPAS native `(rho*omega)'`; the public `w` output converts it to a WRF-compatible vertical velocity through the reduced column metric used by this fixture. `rho_pp` maps to `rho_perturbation`, `rtheta_pp / rho_base` maps to `theta_perturbation`, and `mu_perturbation` is reconstructed as `g * integral(rho_pp dz)` for comparison shape compatibility.

## Commands Run

- `pytest tests/test_m6x_mpas_column_slice_oracle.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/proof_slice.txt`
  Output: `4 passed in 3.76s`.
- `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/proof_no_regression.txt`
  Output: `15 passed in 15.63s`.
- `PYTHONPATH=src python ... | tee .agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/proof_deviation.txt`
  Output: `slice_peak_m_s=1.327232772280`, `c2_peak_m_s=1.301713641057`, `peak_amplitude_error_fraction=0.019227321504`, `trajectory_rmse_fraction=0.387407212716`.
- An initial deviation command without `PYTHONPATH=src` failed with `ModuleNotFoundError: No module named 'gpuwrf'`; it was rerun successfully with `PYTHONPATH=src`.

## Proof Objects

- `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py`
- `tests/test_m6x_mpas_column_slice_oracle.py`
- `data/fixtures/mpas_column_slice/warm_bubble_2km.npz`
- `.agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/proof_slice.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/proof_deviation.txt`

## Risks

- This is a symbolic MPAS-source slice, not a built MPAS executable savepoint. MPAS build/run extraction remains open.
- The `rw_p` to WRF-style `w` conversion is a reduced-column metric choice and should be revisited in the production-grade sprint before using this as a percent-level MPAS equivalence claim.
- Full trajectory RMSE is 38.7% even though peak amplitude error is 1.92%; the prototype operator is close on peak rise but not yet line-by-line MPAS behavior.

## Handoff

Objective: close F1 by adding a non-tautological MPAS-derived column-slice reference and a c2 comparison test.

Files changed: listed above.

Commands run: listed above with outputs.

Proof objects produced: listed above.

Unresolved risks: MPAS executable extraction, metric conversion, and trajectory-shape mismatch require production-grade follow-up.

Next decision needed: production-grade ADR-023 should decide whether the 1.92% peak error and 38.7% trajectory RMSE are acceptable for the next F6 ladder rung, or whether the operator must be brought closer to the MPAS recurrence before warm-bubble/d02 work continues.
