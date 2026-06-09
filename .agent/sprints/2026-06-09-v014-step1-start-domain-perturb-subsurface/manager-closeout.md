# Manager Closeout: V0.14 Step-1 Start-Domain Perturbation Subsurface

Date: 2026-06-09 20:04 WEST

## Outcome

Closed with verdict
`STEP1_START_DOMAIN_PERTURB_SUBSURFACE_LOCALIZED_CURRENT_JAX_AL_ALT_BASE_INPUT_GAP`.

Merge Decision: commit and push the proof artifacts. Do not merge a production
source change from this sprint.

The WRF internal `start_domain(nest,.TRUE.)` order is no longer the unknown:
hypsometric `P/al/alt`, `press_adj`, and W-surface handling are captured and
checked. The candidate "patch now using current JAX inputs" is refuted because
the remaining residuals are still above the material gates.

## Proof Objects

- `proofs/v014/step1_start_domain_perturb_subsurface.py`
- `proofs/v014/step1_start_domain_perturb_subsurface.json`
- `proofs/v014/step1_start_domain_perturb_subsurface.md`
- `proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`

## Key Metrics

- WRF P from internal ALT versus WRF after-hypsometric P:
  max_abs `0.015625`, RMSE `0.0017691372004962024`.
- WRF `press_adj` versus WRF after-press MU:
  max_abs `4.547473508864641e-13`, RMSE `9.500192094660529e-14`.
- WRF after-W branch versus accepted pre-call W:
  max_abs `5.960464477539063e-08`.
- Current JAX pressure formula versus WRF after-hypsometric P:
  max_abs `3.9458582235092763`, RMSE `0.3832298992869327`.
- Current JAX `press_adj` formula versus WRF after-press MU:
  max_abs `0.047773029698646496`, RMSE `0.0010454860097534014`.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_start_domain_perturb_subsurface.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_start_domain_perturb_subsurface.py`
- `python -m json.tool proofs/v014/step1_start_domain_perturb_subsurface.json >/tmp/step1_start_domain_perturb_subsurface.manager_final.json`
- `git diff -- src/gpuwrf` with no output
- `git diff --check` on the sprint artifacts

## Scope

No production source, GPU validation, TOST, Switzerland, FP32 source work,
memory source work, or Hermes was used.

## Next Sprint

Open a narrow current-JAX-input split for live-nest `start_domain`: compare
final blended `HT`, `PB/MUB/PHB`, `PH_STATE`, pre-`press_adj` `MU`, and
diagnosed `AL/ALT` against WRF internal `after_hypsometric` truth. Patch
`d02_replay.py` only if this input gap is proven and closes below the material
gate.
