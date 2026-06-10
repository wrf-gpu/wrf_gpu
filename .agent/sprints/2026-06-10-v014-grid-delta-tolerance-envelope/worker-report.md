# Worker Report: V0.14 Grid-Delta Tolerance Envelope

Summary: GPT-5.5 xhigh produced the requested pre-result tolerance manifest
candidate for the v0.14 Grid-Delta Atlas, plus a short rationale and review
handoff. No model source, validation runner, TOST, Switzerland, release, memory,
or Fable Step-1 files were edited.

Files produced:

- `proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json`
- `proofs/v014/grid_delta_atlas/TOLERANCE_MANIFEST_CANDIDATE.md`
- `.agent/reviews/2026-06-10-v014-grid-delta-tolerance-envelope.md`

Key result:

- Ten hard release-gate fields use pre-existing documented RMSE thresholds:
  `T2`, `U10`, `V10`, `PSFC`, `RAINNC`, `T`, `U`, `V`, `W`, `QVAPOR`.
- Static geometry/vertical/base fields have exact or tight formula checks.
- `P`, `PH`, `MU`, and `RAINC` remain critical report-only fields because no
  independently frozen v0.14 all-cell threshold was found.
- Historical red-run metadata states the old v0.12 proof remains failing under
  this manifest, so the envelope does not widen tolerances to pass known drift.

Worker caveat:

- The worker could not send the completion marker back to tmux from its sandbox
  and began to consider a Hermes fallback. The manager killed the completed
  worker window to honor the principal's no-Hermes directive.
