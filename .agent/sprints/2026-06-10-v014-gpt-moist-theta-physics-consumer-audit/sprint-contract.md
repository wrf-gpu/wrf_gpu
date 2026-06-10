# Sprint Contract: V0.14 GPT Moist-Theta Physics Consumer Audit

Date: 2026-06-10 WEST
Owner: GPT-5.5 xhigh in tmux
Manager: `worker/gpt/v013-close-manager`
Base commit: `281b439f`

## Objective

Audit all production physics/coupling consumers of `state.theta` and local
theta-to-temperature helpers after Fable/Mythos localized the active NoahMP
forcing bug to a moist-potential-temperature convention mismatch:

- Runtime `state.theta` appears to be WRF coupled/moist potential temperature
  `theta_m`.
- WRF physics receives dry-temperature/dry-theta-derived values, with the
  inverse `theta_dry = theta_m / (1 + R_v/R_d * qv_mixing)`.
- The active Fable lane is fixing/proving the NoahMP/surface path. This GPT
  sprint must not edit source or race that lane.

The goal is to produce a compact compatibility map that says which consumers
are safe, which likely need moist-to-dry decoupling, which are already fed WRF
dry oracle inputs in proofs only, and what follow-up tests/gates are required.

## Required Work

1. Read the relevant contracts and initialization convention:
   - `src/gpuwrf/contracts/state.py`
   - `src/gpuwrf/init/real_init/**`
   - any `_wrf_use_theta_m`, theta coupling, or gas-constant code.
2. Search all `src/gpuwrf/physics`, `src/gpuwrf/coupling`,
   `src/gpuwrf/runtime`, and `src/gpuwrf/dynamics` consumers of
   `state.theta`, `.theta`, `_potential_to_temperature`,
   `_temperature_from_theta`, `theta_m`, `THM`, and related conversions.
3. Classify each consumer:
   - `MUST_USE_MOIST_THETA`
   - `MUST_DECOUPLE_TO_DRY`
   - `ALREADY_DECOUPLED`
   - `PROOF_ONLY_OR_ORACLE_INPUT`
   - `UNCLEAR_NEEDS_TEST`
4. Identify the minimal safe helper/API shape if a shared conversion helper is
   needed, including expected qv units and constant source.
5. List the exact focused regression/proof gates needed after Fable's fix.

## Allowed Files

May write only:

- `proofs/v014/moist_theta_physics_consumer_audit.json`
- `proofs/v014/moist_theta_physics_consumer_audit.md`
- `.agent/reviews/2026-06-10-v014-gpt-moist-theta-physics-consumer-audit.md`

Do not edit `src/gpuwrf/**`, tests, sprint contracts other than this folder,
TOST, Switzerland, Grid-Delta, memory, FP32, or Fable proof/source files.

## Gates

Required:

```bash
python -m json.tool proofs/v014/moist_theta_physics_consumer_audit.json >/tmp/moist_theta_physics_consumer_audit.validated.json
git diff --check
```

CPU only. Use `JAX_PLATFORMS=cpu` and `CUDA_VISIBLE_DEVICES=` for any tiny
optional probes. Do not use the GPU.

## Handoff Requirements

Write a compact report with:

- objective;
- files changed;
- commands run;
- proof objects produced;
- top compatibility table;
- likely fix points and why;
- risks/performance implications;
- exact next gates after Fable.

Completion marker:

```bash
tmux send-keys -t 0:2 'GPT MOIST_THETA_CONSUMER_AUDIT DONE - see proofs/v014/moist_theta_physics_consumer_audit.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```

