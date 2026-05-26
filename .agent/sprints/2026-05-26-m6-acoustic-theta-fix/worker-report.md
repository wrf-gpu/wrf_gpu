# Worker Report - M6 Acoustic Theta Operator Fix

Summary: BLOCKED_ACCEPTANCE_NOT_MET. Implemented the WRF theta mass-coupling
algebra in the shared acoustic core and restored `advance_mu_t_wrf` theta fluxes
to WRF `t_1`. The named blocker improved: the step-17 bad-cell one-acoustic
substep delta at `[12,30,62]` is now 9.46 K, below the contract's 50 K
regression bound, and the original step-18 16207 K failure no longer appears.
Full acceptance is still blocked because the guard-disabled 360-step replay now
first fails later at theta step 47, cell `[11,31,67]`, still in `acoustic`, with
a near-singular `c1h*muts+c2h` denominator diagnosed at step 46.

## Files changed

- `src/gpuwrf/dynamics/mu_t_advance.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `tests/test_m6_acoustic_theta_fix.py`
- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/worker-report.md`
- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/hypothesis_notes.md`
- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/proof_*.json`
- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/validation_*.txt`
- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/{baseline,fixed}/proof_*.json`

External ignored fixtures:
- `data/fixtures/m6-acoustic-theta-fix/step17_input_state.npz`
- `data/fixtures/m6-acoustic-theta-fix/step46_input_state.npz`

## Algebraic fix

WRF `module_small_step_em.F` mass-couples theta before `advance_mu_t`
(`small_step_prep` lines 259-264), advances that mass-coupled quantity, then
decouples with saved pre-coupled theta (`small_step_finish` lines 408-413). The
JAX core was advancing perturbation theta directly and only decoupling after the
update. I added the pre-advance mass-coupling projection in
`acoustic_substep_core` and changed the post projection to use the saved
pre-coupled theta. I also changed `advance_mu_t_wrf` theta fluxes from
`theta_ave` back to WRF `t_1`.

## Commands run and output

- `taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 25 --output .agent/sprints/2026-05-26-m6-acoustic-theta-fix/baseline/`
  - Output: `status OK`; first explosive step reproduced as `theta`, step `18`, cell `[12,30,62]`, value `16207.9404296875`; first operator `acoustic`.
- Step-17 extraction command
  - Output: wrote `data/fixtures/m6-acoustic-theta-fix/step17_input_state.npz`; theta at bad cell before step 18 was `2656.549560546875 K`.
- WRF cross-check probes
  - Output: no-prep one-substep delta `441.839 K`; WRF mass-coupled delta `9.461 K`; implemented one-substep delta `9.461 K`.
- `taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 360 --output .agent/sprints/2026-05-26-m6-acoustic-theta-fix/fixed/`
  - Output: command exit 0, acceptance failed later; first explosive step `theta`, step `47`, cell `[11,31,67]`, value `-226584368.0`; first operator `acoustic`.
- `taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all`
  - Output: `passed true`; `outcome: SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`; diverging field count `0`.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10`
  - Output: `status PASS`; `final_max_abs_delta 0.0`; all tracked fields finite.
- `taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v`
  - Output: `12 passed in 0.95s`.
- `taskset -c 0-3 pytest tests/test_m6_acoustic_theta_fix.py -v`
  - Output: `1 passed in 16.25s`.

## Proof objects

- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/proof_baseline_reproduces.json`
- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/proof_wrf_fortran_crosscheck.json`
- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/proof_fix_validation.json`
- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/proof_one_substep_probe.json`
- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/proof_step46_probe.json`
- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/hypothesis_notes.md`

## Risks

- The contract's full Stage 3 guard-disabled 1h acceptance is not met.
- The remaining step-47 failure is still acoustic theta, but it is a new layer:
  step-46 diagnostics show `c1h*muts+c2h = 2.7745` at `[11,31,67]` after theta
  has already left physical bounds. This likely requires correcting RK/acoustic
  save-family semantics around `t_1` / `t_save`, which is outside this sprint's
  allowed write files.
- The `.npz` fixtures are intentionally ignored under `data/` and are not in git.

## Handoff

Objective: fix the named acoustic theta tendency bug exposed after the HPG fix.

Files changed: listed above.

Commands run: all contract validation commands were run; stdout/stderr captures
are in the sprint folder.

Proof objects produced: listed above.

Unresolved risks: full guard-disabled 360-step acceptance still fails at a later
acoustic theta projection.

Next decision needed: dispatch the next focused sprint on RK/acoustic save-family
semantics and the near-singular `c1h*muts+c2h` theta projection at step 46/47.
