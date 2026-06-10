# Memory Patch

Reviewer Status: accepted by manager as the current v0.14 compact handoff.

## Durable Facts

- Step-1 source-fidelity closure is still not achieved.
- `rad_rk_tendf=1` source-leaf mode now carries MYNN `RTHBLTEN/RQVBLTEN` and
  applies WRF `conv_t_tendf_to_moist` before `DryPhysicsTendencies.t_tendf`.
- The strict Step-1 residual remains large: max_abs `2457.578397008898`, RMSE
  `21.364579991779515`.
- Radiation-held-rate and moist conversion are now secondary.
- The single remaining blocker is MYNN driver/kernel source output:
  JAX mass-coupled `RTHBLTEN` max_abs `260.83156991819124` versus WRF
  `2522.90576171875`; JAX qv source max_abs `0.045505018412171354` versus WRF
  `0.4930315017700195`.

## Next Action

Use Fable/Mythos for the next hard sprint, because two GPT source-fidelity
sprints have now localized but not fixed the same hard kernel-level blocker.
Before assigning, send `/compact` to `tmux 0:1`, wait about two minutes, then
send one endpoint-defined prompt with delayed repeated Enter presses.
