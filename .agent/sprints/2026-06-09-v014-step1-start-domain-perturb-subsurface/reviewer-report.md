# Reviewer Report: V0.14 Step-1 Start-Domain Perturbation Subsurface

Decision: `ACCEPT_LOCALIZATION_OPEN_NEXT_INPUT_SPLIT`.

The sprint satisfies its contract as a truth-surface/localization sprint. It
captures exact WRF internal surfaces around hypsometric `P/al/alt`,
`press_adj`, and W-surface handling, validates the generated proof artifacts,
and does not overclaim a source fix.

Accepted evidence:

- WRF emitted 28 d02 patch files for each required internal surface.
- WRF internal ordering is now supported: hypsometric `P/al/alt` before
  `press_adj`, then W-surface handling.
- WRF internal formula checks are tight enough for localization:
  `P` from internal `ALT` max_abs `0.015625`, `press_adj` max_abs
  `4.547473508864641e-13`, and W branch max_abs
  `5.960464477539063e-08`.
- The worker correctly refused a production patch because current JAX inputs
  still leave `P` max_abs `3.9458582235092763` Pa and `MU` max_abs
  `0.047773029698646496` Pa.

Reviewer caveats:

- The WRF text truth is external to git; the committed JSON records metadata
  and checksums, but future replay depends on the scratch artifact root.
- The next sprint must target current JAX input construction for `AL/ALT` and
  base-state surfaces rather than re-opening source-ordering, acoustic,
  boundary, or physics-tendency hypotheses.

Recommended next action:

Open a narrow CPU-only proof sprint comparing current JAX final blended terrain
and base/diagnostic surfaces against WRF `after_hypsometric` truth. Permit a
`d02_replay.py` patch only if it closes the material `P/MU` input gap without a
GPU-host dependency.
