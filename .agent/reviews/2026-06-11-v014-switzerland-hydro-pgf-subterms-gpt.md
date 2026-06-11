# V0.14 Switzerland Hydro-PGF Subterms (GPT)

Date: 2026-06-11
Worker: GPT-5.5 xhigh, branch `worker/gpt/v014-switzerland-hydro-pgf-subterms`
Sprint: `.agent/sprints/2026-06-11-v014-switzerland-hydro-pgf-subterms/sprint-contract.md`

## Verdict

`EXACT_ROOT_NO_FIX`

The h36 first-30-step dry mass-venting signal is no longer just "hydro first-three
PGF". It is in the **pressure/inverse-density hydrostatic PGF pair**:

- `p_alt_term = (alt_l + alt_r) * (p_r - p_l)`
- `pb_al_term = (al_l + al_r) * (pb_r - pb_l)`

The `ph` geopotential-gradient term is not the mass-venting driver; it is
stabilizing in this probe. WRF specified/nested outer-face loop bounds were tested
and are not the blocker: WRF-style edge skipping had exactly `0.0` collapse of the
PGF contribution over the first 30 steps.

No model code was patched because the remaining direct target is the staged
`rk_step_prep` pressure/inverse-density diagnostic inputs (`p`, `al`, `alt`) as
consumed by `large_step_horizontal_pgf`; patching without a WRF h36 subterm
savepoint would be speculative.

## Subterm Attribution

H36 dry step probe, first 30 model steps / 300 s. Contribution is relative to
`zero_large_step_pgf`.

| variant | MU delta step 30 | contribution |
|---|---:|---:|
| baseline | -6.705918 Pa | -34.051 Pa/cell/h |
| hydro first-three only | -6.505500 Pa | -31.646 Pa/cell/h |
| `ph_only` | -0.024390 Pa | +46.128 Pa/cell/h |
| `p_alt_only` | -7.873808 Pa | -48.065 Pa/cell/h |
| `pb_al_only` | -9.639814 Pa | -69.258 Pa/cell/h |
| WRF specified-edge-only hydro | -3.868352 Pa | 0.000 Pa/cell/h |
| full WRF specified-edge skip | -6.705918 Pa | -34.051 Pa/cell/h |

Same-state pressure diagnostic at h36:

- `alt_eos - (al + alb)` relative mean-abs: `6.06e-4`, max-abs: `2.09e-3`.
- This is too small to explain the signal by itself.
- `p_perturbation` is large enough to matter directly: mean `-220.66 Pa`, RMSE
  `377.19 Pa`, max-abs `1291.60 Pa`.

## Files Changed

- `proofs/v014/switzerland_hydro_pgf_subterms.py`
- `proofs/v014/switzerland_hydro_pgf_subterms.json`
- `.agent/reviews/2026-06-11-v014-switzerland-hydro-pgf-subterms-gpt.md`

No model-code files were edited.

## Commands Run

- `git branch --show-current`
- `git log -1 --oneline --decorate --stat`
- `cat PROJECT_CONSTITUTION.md`
- `cat AGENTS.md`
- `cat .agent/sprints/2026-06-11-v014-switzerland-hydro-pgf-subterms/sprint-contract.md`
- `cat .agent/skills/validating-physics/SKILL.md`
- `cat .agent/skills/designing-gpu-state/SKILL.md`
- `cat .agent/skills/reporting-to-human/SKILL.md`
- `python -m py_compile proofs/v014/switzerland_hydro_pgf_subterms.py`
- `scripts/run_gpu_lowprio.sh --resource-log-dir /mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/resources --resource-label gpt_hydro_pgf_subterms --resource-interval 5 -- python proofs/v014/switzerland_hydro_pgf_subterms.py --step-probe --steps 30 --print-first 2 --print-every 10`
- `python proofs/v014/switzerland_hydro_pgf_subterms.py`
- `git log -1 --oneline`
- `python -m py_compile proofs/v014/switzerland_hydro_pgf_subterms.py`
- `python -m json.tool proofs/v014/switzerland_hydro_pgf_subterms.json >/tmp/switzerland_hydro_pgf_subterms.validated.json`
- `git diff --check`

## Proof Objects And Run Roots

- Main proof: `proofs/v014/switzerland_hydro_pgf_subterms.json`
- Proof script: `proofs/v014/switzerland_hydro_pgf_subterms.py`
- H36 run root reused: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable`
- Resource CSVs:
  - `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/resources/gpt_hydro_pgf_subterms_gpu_usage.csv`
  - `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/resources/gpt_hydro_pgf_subterms_process_usage.csv`
  - `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/resources/gpt_hydro_pgf_subterms_system_memory.csv`
- GPU resource summary: max GPU memory `7603 MiB`, max GPU util `74%`, max RSS
  `5289408 KiB`.

## Unresolved Risks

- There is no WRF h36 `horizontal_pressure_gradient` savepoint that emits
  `p_alt_term` and `pb_al_term` on native U/V faces. The proof therefore names the
  concrete pressure/inverse-density branch and implementation target, but does not
  claim the final code patch.
- The isolated subterm probes are diagnostic knockouts, not physical alternate
  configurations; they are appropriate for attribution, not validation of a fix.
- The prior moist-cqw pressure-state work is default-ON in this tree, so this
  sprint did not retest that older dry/moist pressure closure as a source fix.

## Next Manager Action

Instrument or reuse a WRF h36 savepoint at
`rk_tendency -> horizontal_pressure_gradient` that writes, on U and V faces:

- `p`, `al`, `alt`, `pb`
- `p_alt_term`
- `pb_al_term`
- final `dpx/dpy` first-three sum

Then compare those against
`src/gpuwrf/dynamics/core/rk_addtend_dry.py::_absolute_diagnostics` and
`large_step_horizontal_pgf`. The implementation sprint should target staged/live
`p/al/alt` input parity first, not ph, top lid, map factors, specified-edge loop
bounds, or Coriolis.
