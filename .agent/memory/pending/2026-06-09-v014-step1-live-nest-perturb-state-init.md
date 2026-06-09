# V0.14 Step-1 Live-Nest Perturb-State Init

Date: 2026-06-09

Reviewer Status: pending. Do not apply to stable memory until reviewer approval.

Scope:

- v0.14 grid-parity debug.
- Live-nest `raw_child_state -> live_child_state` perturbation-state
  initialization for `P_STATE/MU_STATE/W_STATE`.

Evidence:

- `proofs/v014/step1_live_nest_perturb_state_init.json`
- `proofs/v014/step1_live_nest_perturb_state_init.md`
- `.agent/sprints/2026-06-09-v014-step1-live-nest-perturb-state-init/manager-closeout.md`

Finding:

WRF live-nest `P/MU/W` perturbation-state initialization is the active Step-1
boundary. Current JAX keeps raw `wrfinput_d02` `P/MU/W` through raw child, live
child, boundary package, initial carry, halo entry, and
`_physics_step_forcing`; WRF recomputes or adjusts these before
`first_rk_step_part1_call`.

Proof-local WRF transcriptions reduce residuals:

- `P_STATE`: `69.96875` -> `3.9458582235092763` Pa via start-domain pressure
  recompute.
- `MU_STATE`: `13.256103515625` -> `0.047773029698646496` Pa via `press_adj`.
- `W_STATE`: `0.7605466246604919` -> `1.2992081932505783e-07` m/s via
  `set_w_surface`.

Next action:

Emit WRF `start_domain(nest,.TRUE.)` internal surfaces after the hypsometric
`P/al/alt` recompute and immediately before/after `press_adj`, including
`P_STATE`, `MU_STATE`, `al`, `alt`, `alb`, `PH_STATE`, `PB`, `MUB`, `PHB`,
`theta`, `qv`, `HT`, and `HT_FINE`. Patch `d02_replay` only if that surface
closes `P_STATE` and preserves GPU-native execution.
