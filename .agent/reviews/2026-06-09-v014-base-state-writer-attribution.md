# V0.14 Base-State Writer Attribution Review

Date: 2026-06-09
Reviewer: GPT-5.5 xhigh
Decision: ACCEPT

## Objective

Close the CPU-only attribution sprint for remaining h1 static/base-state wrfout
differences after the stale `GridSpec.metrics` writer fix, without source edits.

## Verdict

The proof satisfies the contract. It names the exact `wrfinput_d02`, CPU h0/h1
wrfout, and fresh GPU h1 wrfout files; validates CPU-vs-GPU native input parity;
and classifies every target field.

| Field | Classification |
| --- | --- |
| `PHB` | `cpu_output_convention` |
| `MUB` | `forecast_step_change` |
| `PB` | `forecast_step_change` |
| `HGT` | `cpu_output_convention` |
| `XLAT` | `writer_fallback` |
| `XLONG` | `writer_fallback` |

Same-state dynamic localization can proceed with documented exclusions. No
base-state source fix is required first. A writer-only lat/lon payload fix
remains if exact static wrfout parity is required before promotion.

## Commands Run

- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 python proofs/v014/base_state_writer_attribution.py`
- `python -m json.tool proofs/v014/base_state_writer_attribution.json >/tmp/base_state_writer_attribution.validated.json`
- `python -m py_compile proofs/v014/base_state_writer_attribution.py`

The three contract commands exited 0. The `py_compile` cache artifact was
removed after validation to keep the repository write set limited to the
contract files.

Supplemental local sprint-template check:

- `python scripts/close_sprint.py .agent/sprints/2026-06-09-v014-base-state-writer-attribution`

This exited 1 because the sprint folder contains only `sprint-contract.md` and
is missing the standard template reports/artifacts directory. Those files are
outside this contract's permitted write scope, so this does not change the
contract-gate verdict above.

## Proof Objects

- `proofs/v014/base_state_writer_attribution.py`
- `proofs/v014/base_state_writer_attribution.json`
- `proofs/v014/base_state_writer_attribution.md`

## Risks

- `PB`/`MUB` are h1 evolved state-split symptoms, not static input mismatches;
  this proof does not name the dynamic operator that first changes the split.
- `XLAT`/`XLONG` remain a writer fallback because runtime state lacks lat/lon
  arrays; this is a writer-only correctness issue for exact wrfout parity.

## Next Decision

Proceed to same-state dynamic localization with these exclusions recorded.
