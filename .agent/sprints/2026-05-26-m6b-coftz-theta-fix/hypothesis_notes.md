# Hypothesis Notes

## H1: coftz sign or factor error

Investigated in `src/gpuwrf/dynamics/vertical_implicit_solver.py`. The direct coefficient value was not sign-flipped. The implemented scoped patch lets callers provide a stable theta reference for `coftz`, avoiding the instantaneous-theta feedback path while preserving the legacy default.

## H2: theta_face source mismatch

Best nearest-fit hypothesis for the allowed files. WRF `advance_w` uses `t_2ave`/running-average theta in the vertical implicit coupling, while the allowed MPAS-style path built `coftz` from instantaneous `state.theta`. Patched `_mpas_recurrence_vertical_update` to pass `theta_base` as `theta_coefficient`. Focused regression passes.

## H3: rdzw stagger error

Not changed. The contract-owned update still uses the existing `coftz[1:] * rw_p[1:] - coftz[:-1] * rw_p[:-1]` mass-level divergence. The validation evidence did not isolate a stagger-only fix inside the allowed files.

## H4: epssm time weighting wrong

Not changed. The coefficient builder still uses `dtseps = 0.5 * dt * (1 + epssm)` and `resm = (1 - epssm) / (1 + epssm)`.

## H5: t_2ave / theta double-update

Still a risk. The fixed 20260509 replay is unchanged, and the supported real-IC split comparison fails from step 2 in `theta`, `t_2ave`, and `ph_tend`. That points at the current operational shared-core path rather than the patched `vertical_implicit_solver.py` / `acoustic_wrf.py` MPAS recurrence path.

## H6: GPU vs CPU theta arithmetic

Not supported by this sprint. The focused regression is deterministic and passes, while the real-IC parity failure appears as a path/recurrence issue, not a GPU-only arithmetic discrepancy.
