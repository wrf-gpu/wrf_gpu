# Tester Report

## Tests Added Or Run

- `python -m py_compile proofs/v014/previous_step_handoff_bisect.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/previous_step_handoff_bisect.py`
- `python -m json.tool proofs/v014/previous_step_handoff_bisect.json >/tmp/previous_step_handoff_bisect.manager.validated.json`

## Results

Decision: pass for an evidence sprint.

The script compiles, the required CPU invocation regenerates the repo proof
objects from the compact replay artifact, and the JSON validates. The accepted
classification is `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`.

Key checks:

- `final_reproducer_identity.all_target_fields_exact` is `true`.
- `midscan_capture_identity.instrumented_final_matches_producer_shape` is
  `true`.
- The first captured final-cycle surface,
  `after_segment_replay_d02_step5997_before_final_partial_parent`, already has
  `all_target_fields_match_wrf_truth=false` and
  `static_base_fields_match_wrf_truth=false`.
- Worst target field at that surface is `MUB`, max_abs `1050.3046875`.

## Fixtures Used

- CPU-WRF h10 pre-RK truth from `proofs/v014/pre_rk_input_boundary.json`.
- Producer checkpoint
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`.
- Compact replay artifact
  `/mnt/data/wrf_gpu2/v014_previous_step_handoff_bisect/previous_step_handoff_bisect.live_replay_compact.json`.

## Gaps

The test does not prove the first wrong write before d02 step 5997. It proves
that the final partial subcycle is downstream of the error and should not be
debugged first.

## Decision

Accept. The artifact is strong enough to open the earlier-source bisection
sprint and to keep TOST paused.
