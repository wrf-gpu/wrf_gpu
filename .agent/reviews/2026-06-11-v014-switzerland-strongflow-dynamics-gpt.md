# V0.14 Switzerland Strong-Flow Dynamics Attribution (GPT)

Date: 2026-06-11
Worker: GPT-5.5 xhigh, branch `worker/gpt/v014-switzerland-strongflow-dynamics`
Sprint: `.agent/sprints/2026-06-11-v014-switzerland-strongflow-dynamics-attribution/sprint-contract.md`

## Verdict

`EXACT_ROOT_NO_FIX`

The h36 Switzerland d01 post-LBC strong-flow dry mass venting is not caused by
the rigid top lid, damping, sixth-order filtering, Coriolis, microphysics, or
PBL/radiation. It localizes to the **large-step horizontal PGF hydrostatic
first-three-term branch**: the `ph + p/alt + pb/al` terms in
`large_step_horizontal_pgf` (`src/gpuwrf/dynamics/core/rk_addtend_dry.py`) as
consumed by the operational RK/acoustic path.

This sprint did not patch model code because that root is outside the contract's
allowed edit surface. The next implementation sprint should explicitly include
`src/gpuwrf/dynamics/core/rk_addtend_dry.py` and add/consume a WRF h36
`rk_tendency -> horizontal_pressure_gradient` savepoint for the first-three PGF
subterms.

## Exact Attribution

Top-lid A/B, 1 h h36 forecast:

| variant | finite | MU bias h37 | PSFC bias h37 | net influx h36->37 | excess outflux vs CPU |
|---|---:|---:|---:|---:|---:|
| CPU truth | yes | 0.0 Pa | 0.0 Pa | -74.515 Pa/cell/h | 0.000 |
| rigid lid baseline | yes | -54.423 Pa | -56.588 Pa | -103.130 Pa/cell/h | -28.615 |
| WRF-faithful open top | yes | -64.582 Pa | -66.760 Pa | -103.047 Pa/cell/h | -28.532 |
| no microphysics | yes | -56.941 Pa | -58.911 Pa | -103.580 Pa/cell/h | -29.065 |

Open top explains only `0.29%` of the baseline excess outflux and worsens MU/PSFC
bias. The release blocker is not a rigid-lid reflection.

Dry 60-step h36 probes:

| variant | steps | status | MU mean delta |
|---|---:|---|---:|
| rigid + boundary | 60 | finite | -13.649 Pa |
| open + boundary | 60 | finite, worse | -18.410 Pa |
| no Rayleigh/w-damping | 60 | finite, neutral | -13.607 Pa |
| no diff6 | 60 | finite, neutral | -13.689 Pa |
| no Coriolis | 60 | finite, neutral/slightly worse | -14.132 Pa |
| primitive advection | 17 | unstable | +0.777 Pa before blow-up |

PGF knockout / split, first 30 model steps (300 s):

| variant | MU delta at step 30 | implied contribution |
|---|---:|---:|
| baseline | -6.706 Pa | - |
| zero large-step PGF | -3.868 Pa | large-step PGF = -34.051 Pa/cell/h |
| hydro first-three PGF only | -6.505 Pa | hydro terms = -31.646 Pa/cell/h |
| nonhydro fourth PGF only | -4.150 Pa | fourth term = -3.380 Pa/cell/h |

The measured PGF contribution matches the previously established target
(`~28-31 Pa/cell/h` excess dry outflux). The hydrostatic first-three terms carry
`~93%` of the large-step PGF contribution. Knockouts are proof-only: zeroing PGF
destabilizes by step 38, and zeroing acoustic UV PGF goes nonfinite at step 21.

## Files Changed

- `proofs/v014/switzerland_strongflow_dynamics.py`
- `proofs/v014/switzerland_strongflow_dynamics.json`
- `proofs/v014/switzerland_strongflow_dynamics_step_probe.json`
- `proofs/v014/switzerland_strongflow_dynamics_knockout_probe.json`
- `proofs/v014/switzerland_strongflow_dynamics_pgf_split_probe.json`
- `.agent/reviews/2026-06-11-v014-switzerland-strongflow-dynamics-gpt.md`

No model-code files were edited.

## Proof Objects And Run Roots

- Main proof: `proofs/v014/switzerland_strongflow_dynamics.json`
- GPU step proof: `proofs/v014/switzerland_strongflow_dynamics_step_probe.json`
- PGF knockout proof: `proofs/v014/switzerland_strongflow_dynamics_knockout_probe.json`
- PGF component split: `proofs/v014/switzerland_strongflow_dynamics_pgf_split_probe.json`
- Open-top 1 h run: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output_openlid_gpt`
- Resource logs: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/resources/gpt_*`

## Commands Run

- `git log -1 --oneline --decorate`
- `git branch --show-current`
- `python -m py_compile proofs/v014/switzerland_strongflow_dynamics.py`
- `scripts/run_gpu_lowprio.sh ... -- python proofs/v014/switzerland_strongflow_dynamics.py --step-probe --steps 60 ...`
- `scripts/run_gpu_lowprio.sh ... -- python proofs/v014/switzerland_strongflow_dynamics.py --forecast-variant open_lid --hours 1`
- `scripts/run_gpu_lowprio.sh ... -- python proofs/v014/switzerland_strongflow_dynamics.py --step-probe --only rigid_zero_large_step_pgf --only rigid_zero_coriolis --only rigid_zero_acoustic_uv_pgf ...`
- `scripts/run_gpu_lowprio.sh ... -- python proofs/v014/switzerland_strongflow_dynamics.py --step-probe --only rigid_large_step_pgf_hydro_only --only rigid_large_step_pgf_nh4_only ...`
- `python proofs/v014/switzerland_strongflow_dynamics.py`
- `python -m json.tool proofs/v014/switzerland_strongflow_dynamics.json >/tmp/switzerland_strongflow_dynamics.validated.json`

## Unresolved Risks

- The exact algebraic bug inside the hydrostatic PGF first-three branch is not yet
  split into `ph`, `p*alt`, and `pb*al` subterms against WRF source output.
- The proof-only knockouts are intentionally unphysical; they localize the term
  but are not candidate fixes.
- Full Switzerland 72 h was not rerun; the short gate is designed to dispatch the
  final implementation sprint.

## Next Manager Action

Dispatch one final implementation sprint targeting
`src/gpuwrf/dynamics/core/rk_addtend_dry.py::large_step_horizontal_pgf`.
Required first step: add a WRF h36 strong-flow savepoint at
`rk_tendency -> horizontal_pressure_gradient` that emits the hydrostatic
first-three PGF subterms on native U/V faces, then compare JAX subterm-by-subterm
before patching. Acceptance should be the same h36 1 h/3 h gate, followed by the
72 h Switzerland release gate.
