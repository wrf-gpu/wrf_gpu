# Tester Report

## Tests Added Or Run

- `python -m py_compile proofs/v014/step1_base_state_boundary.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_base_state_boundary.py`
- `python -m json.tool proofs/v014/step1_base_state_boundary.json >/tmp/step1_base_state_boundary.validated.json`
- `git diff -- src/gpuwrf`

## Results

All required validation commands passed. The proof executed on CPU only and
reported no production `src/gpuwrf/**` diff.

## Fixtures Used

- Existing predecessor WRF truth root:
  `/mnt/data/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715/wrf_truth`
- Current live-nest Step-1 JAX loader via `PYTHONPATH=src`.

## Gaps

The test did not run WRF or GPU. It did not emit a fresh WRF scalar for
`p_surf_before_mub`; the JSON records that as the remaining truth/source gap.

## Decision:

PASS for this sprint's CPU-only proof gate. Not ready for production patch.
