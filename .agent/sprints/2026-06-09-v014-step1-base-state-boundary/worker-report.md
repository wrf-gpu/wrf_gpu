# Worker Report

## Summary:

Built and ran a CPU-only proof for the live-nest d02 Step-1 `start_domain_em`
base-state boundary before the `AL/ALT` pass. The proof reuses the accepted WRF
truth root from the predecessor sprint and compares current JAX/live-nest base
inputs against fp64/fp32 and cp=1004.0/cp=1004.5 source-order families.

Verdict:
`STEP1_BASE_STATE_BOUNDARY_LOCALIZED_P_SURF_MUB_FP32_SOURCE_ARITHMETIC`.

## Files Changed

- `proofs/v014/step1_base_state_boundary.py`
- `proofs/v014/step1_base_state_boundary.json`
- `proofs/v014/step1_base_state_boundary.md`
- `.agent/reviews/2026-06-09-v014-step1-base-state-boundary.md`
- `.agent/sprints/2026-06-09-v014-step1-base-state-boundary/*-report.md`
- `.agent/sprints/2026-06-09-v014-step1-base-state-boundary/manager-closeout.md`
- `.agent/sprints/2026-06-09-v014-step1-base-state-boundary/artifacts/.gitkeep`

## Commands Run

- `python -m py_compile proofs/v014/step1_base_state_boundary.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_base_state_boundary.py`
- `python -m json.tool proofs/v014/step1_base_state_boundary.json >/tmp/step1_base_state_boundary.validated.json`
- `git diff -- src/gpuwrf`
- `python scripts/close_sprint.py .agent/sprints/2026-06-09-v014-step1-base-state-boundary`

## Proof Objects

- `proofs/v014/step1_base_state_boundary.json`
- `proofs/v014/step1_base_state_boundary.md`
- `.agent/reviews/2026-06-09-v014-step1-base-state-boundary.md`
- Reused WRF truth root:
  `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715/wrf_truth`

## Risks

- No fresh WRF `p_surf_before_mub` scalar was emitted in this sprint; the proof
  recovers `p_surf` from WRF `MUB + P_TOP`.
- Current proof-local fp32/cp=1004.5 p_surf formula remains above P/MU gates;
  no production patch is safe from this sprint alone.

## Handoff

Next worker should instrument or exactly emulate the WRF `p_surf` expression and
`MUB` assignment before patching `d02_replay.py`.
