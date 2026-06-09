# Pending Memory: V0.14 Same-State Momentum/Mass Localization

Status: pending promotion after fresh h10 carry regeneration on current code.

Lesson:

- The v0.14 grid-divergence search now has a named same-state failure surface:
  selected h10 `U` mismatches WRF at `post_after_all_rk_steps_pre_halo`.
- First mismatch: `U` max_abs `6.292358893898424`, RMSE
  `2.032497018496295`, worst native key `[4, 13]`, JAX
  `-4.735481996086533` vs WRF `1.55687689781189`.
- This moves the debug target earlier than output writing, station/TOST
  interpolation, or RK halo exchange. Next source localization should instrument
  final RK U/V tendency/acoustic update, mass coupling, and theta-pressure
  source assembly.
- The h10 carry used by this proof predates the live-nest base-source partial
  fix, so base-field residuals in this artifact are not current-code
  attribution evidence.

Evidence:

- `proofs/v014/same_state_momentum_mass.json`
- `proofs/v014/same_state_momentum_mass.md`
- `.agent/reviews/2026-06-09-v014-same-state-momentum-mass.md`
- `.agent/sprints/2026-06-09-v014-same-state-momentum-mass/manager-closeout.md`
