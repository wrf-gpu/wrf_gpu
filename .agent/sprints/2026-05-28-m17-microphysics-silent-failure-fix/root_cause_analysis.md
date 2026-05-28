# M17 Root-Cause Analysis: Thompson Smoke NOISY_ZERO

## Verdict

The 3-step smoke finding is a false positive for Thompson coupling, not a Thompson adapter silent failure.

The Thompson adapter in `src/gpuwrf/coupling/physics_couplers.py` is wired through:

- lines 486-503: `_thompson_column_from_state` reads `qv/qc/qr/qi/qs/qg/Ni/Nr`, temperature, pressure, and density from `State`.
- lines 506-520: `_state_from_thompson_output` writes `theta/qv/qc/qr/qi/qs/qg/Ni/Nr` back to `State`.
- lines 553-565: `thompson_adapter` calls `step_thompson_column` and returns the reassembled state.

The smoke flatline originates from the physical inputs, not from an early return, missing state writeback, stale JIT path, or tendency-vs-absolute confusion. On the pinned 20260521 initial condition, all hydrometeor fields are zero and qv is subsaturated everywhere:

- `qc/qr/qi/qs/qg/Ni/Nr count_gt_0 = 0`
- maximum liquid supersaturation `qv / qvsw - 1 = -0.0723501209000571`
- maximum ice supersaturation `qv / qvsi - 1 = -0.17643158960008598`

With no cloud, rain, ice, snow, graupel, or supersaturation, one Thompson source/sink step is allowed to be identity. The direct representative-slab proof in `proofs/m17/thompson_initial_condition_probe.json` confirms the kernel and adapter both produce exact zero deltas for that initial cloud-free/subsaturated slab.

## Smoke Evidence

The original smoke report `proofs/diagnostic_harness/diagnostic_report_smoke_3step.json` reported:

- `microphysics_thompson.verdict = NOISY_ZERO`
- comments: `6/7 expected fields have delta = 0 across the run: qr, qc, qg, qs, qi, qv`
- `qv/qc/qr/qi/qs/qg max_abs_delta_per_step = 0.0`
- `theta max_abs_delta_per_step = 5.684341886080802e-14`

That 3-step horizon is too short for this specific initial condition. The 1h harness rerun with radiation disabled to avoid RRTMG autotune OOM shows Thompson is active without any Thompson code change:

- `microphysics_thompson.verdict = ACTIVE`
- mean deltas per step:
  - `qv = 3.000456771602819e-8`
  - `qc = 2.6217087686801035e-8`
  - `qr = 1.2576105205073461e-8`
  - `qi = 5.310053296389558e-10`
  - `qs = 2.3032194522735172e-10`
  - `qg = 2.95497965288851e-12`
  - `theta = 7.452387785882553e-5`

Proof: `proofs/m17/diagnostic_report_after_fix.json`.

## Cause Classification

Category: **(f) other**.

The apparent failure is a diagnostic-horizon/initial-condition false positive. The adapter inputs are connected, outputs are written back, and the 1h diagnostic proof shows all Thompson water species become nonzero once upstream model evolution creates active microphysics conditions.

## Fix Decision

No behavioral Thompson-section fix was applied. Introducing artificial qv or hydrometeor deltas into cloud-free, subsaturated columns would create an unphysical source term and would violate the project rule that physics claims require evidence.

The next actionable defect exposed by the 1h harness is outside this sprint scope: `theta_in_bounds` first violates at step 141 with `first_violation_operator = dycore_rk3`, while Thompson remains active.
