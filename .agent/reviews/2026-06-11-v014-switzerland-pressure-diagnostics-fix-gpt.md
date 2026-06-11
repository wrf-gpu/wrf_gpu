# V0.14 Switzerland Pressure-Diagnostics Fix (GPT)

Date: 2026-06-11
Worker: GPT-5.5 xhigh, branch `worker/gpt/v014-switzerland-pressure-diagnostics-fix`
Sprint: `.agent/sprints/2026-06-11-v014-switzerland-pressure-diagnostics-fix/sprint-contract.md`

## Verdict

`EXACT_ROOT_NO_FIX`

The manager suspicion is real algebraically but is **not** the h36 release
blocker. `State.mu_total` is already `MUB + MU`, so current
`_absolute_diagnostics` does form `muts = MUB + 2*MU`; however replacing it with
WRF-faithful `muts = State.mu_total` collapses the 30-step large-step-PGF
contribution by only `2.04%` (`-34.0508 -> -33.3570 Pa/cell/h`), far below the
70% gate.

No model-code patch was committed because every WRF-faithful local
`_absolute_diagnostics` variant tested here leaves the mass-venting signal
essentially intact. The exact remaining target is the staged native-face
large-step HPG pressure/inverse-density input set after WRF `rk_step_prep` and
`rk_phys_bc_dry_1`, especially `pb_al` on U/V faces. A source patch without that
WRF face savepoint would be speculative.

## Hypothesis Table

| hypothesis / variant | 30-step PGF contribution | result |
|---|---:|---|
| current source / explicit `MUB + 2*MU` | `-34.0508 Pa/cell/h` | reproduced current path |
| `muts = State.mu_total` | `-33.3570 Pa/cell/h` | only `2.04%` collapse |
| `muts = mub + mu_pert` | `-33.3570 Pa/cell/h` | exact equivalent |
| `alt = al + alb` | `-33.3569 Pa/cell/h` | no movement |
| `p = EOS(theta, al+alb)` | `-33.3664 Pa/cell/h` | no movement |
| `p = EOS(theta, al+alb)` and `alt = al+alb` | `-33.3664 Pa/cell/h` | no movement |
| `p_alt` only with combined diagnostic | `-48.0833 Pa/cell/h` | unchanged pressure branch |
| `pb_al` only with combined diagnostic | `-68.4420 Pa/cell/h` | still dominant bad branch |

H36 start base-state parity is clean: GPU-built `PB` vs CPU h36 is exact
(`max_abs 0.0`), `PHB max_abs 0.0078125`, `MUB max_abs 0.00390625`. The issue is
not a bad h36 base-state load.

## Source Fix Summary

No source fix. The WRF-faithful `muts` correction is valid in isolation, but it
does not solve the blocker and was not patched.

## H36 Gate Result

Short dry h36 probes ran over 30 model steps with finite state and resource CSVs.
No 1 h fixed forecast was run because no candidate source fix reached the
contract's 70% short-probe collapse threshold.

## Files Changed

- `proofs/v014/switzerland_pressure_diagnostics_fix.py`
- `proofs/v014/switzerland_pressure_diagnostics_fix.json`
- `.agent/reviews/2026-06-11-v014-switzerland-pressure-diagnostics-fix-gpt.md`

## Commands Run

- `git branch --show-current`
- `git log -1 --oneline --decorate --stat`
- `sed -n ... PROJECT_CONSTITUTION.md AGENTS.md sprint-contract.md`
- `sed -n ... .agent/skills/validating-physics/SKILL.md`
- `sed -n ... .agent/skills/building-wrf-oracles/SKILL.md`
- `sed -n ... .agent/skills/profiling-nvidia-gpu/SKILL.md`
- `scripts/run_gpu_lowprio.sh --resource-label gpt_pressure_diag_muts_probe -- python proofs/v014/switzerland_pressure_diagnostics_fix.py --step-probe --steps 30 ...`
- `scripts/run_gpu_lowprio.sh --resource-label gpt_pressure_diag_next_probe -- python proofs/v014/switzerland_pressure_diagnostics_fix.py --step-probe --steps 30 --only ...`
- `python proofs/v014/switzerland_pressure_diagnostics_fix.py`
- `python -m py_compile proofs/v014/switzerland_pressure_diagnostics_fix.py`
- `python -m json.tool proofs/v014/switzerland_pressure_diagnostics_fix.json >/tmp/switzerland_pressure_diagnostics_fix.validated.json`
- `git diff --check`

## Proof Objects And Resource CSV Roots

- Main proof: `proofs/v014/switzerland_pressure_diagnostics_fix.json`
- Proof script: `proofs/v014/switzerland_pressure_diagnostics_fix.py`
- H36 run root: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/run_h36`
- Resource CSVs:
  - `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/resources/gpt_pressure_diag_muts_probe_*`
  - `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/resources/gpt_pressure_diag_next_probe_*`
- GPU resource summary: muts probe max GPU memory `7568 MiB`, util `80%`; next probe max GPU memory `5294 MiB`, util `75%`.

## Unresolved Risks

- There is still no WRF h36 native-face savepoint for `horizontal_pressure_gradient`
  after `rk_step_prep` and `rk_phys_bc_dry_1`; without it, the exact face-array
  difference inside `pb_al` cannot be patched safely.
- The probes are diagnostic local variants, not physical forecast candidates.
- Full Switzerland 72 h was not run and should not be run until the face-level
  HPG input mismatch is closed.

## Next Manager Action

Instrument WRF at h36 after `rk_step_prep` + `rk_phys_bc_dry_1` and immediately
inside `horizontal_pressure_gradient` to emit, on native U/V faces: `p`, `al`,
`alt`, `pb`, `p_alt_term`, `pb_al_term`, and final `dpx/dpy`. Then compare those
arrays against JAX `large_step_horizontal_pgf` before dispatching the final model
patch.
