# V0.14 Short Field H1 Residual — Fable Verdict

Date: 2026-06-10 WEST. Base verified: `41468af4`. CPU-only (no GPU used).

## Verdict: `FIXED` (root cause) — one GPU rerun required before the 72h gates

**Bug class: DYCORE EOS MOISTURE FACTOR.** `src/gpuwrf/dynamics/acoustic_wrf.py`
used `qvf = 1 + 0.608*qv` (WRF's `p608`, virtual-temperature convention) in
both EOS helpers where WRF uses `qvf = 1 + rvovrd*qv`, `rvovrd = Rv/Rd ≈ 1.6084`
(dry-α_d convention; `module_big_step_utilities_em.F:1064,1118,1140`,
`module_model_constants.F:41,97`). The GPU pressure column was vapor-light by
~the precipitable-water weight (−300 Pa at the surface, decaying aloft) — the
dominant driver of the PSFC/P/PH/MU residual family, and very likely the broad-P
component of the long-open one-RK-step dynamics frontier.

**Proof (bit-level, WRF-anchored):** EOS inverted on each wrfout with its own
discrete α_d: CPU-WRF truth satisfies 1.608 (rmse 22 Pa = fp32 rounding) and
fails 0.608 (rmse 550); pre-fix GPU is the exact mirror (5.8 vs 556 Pa).
Hydrostatic cross-check: CPU PSFC = moist integral to 0.94 Pa; GPU PSFC =
dry integral −30 Pa. **Fix applied** (+11/−2 lines, `RVOVRD = R_V/R_D`): the
production helper now reproduces CPU-WRF truth P to rmse 27 Pa (20× closure).
Tests: 28 passed / 2+3 skipped across acoustic+replay subsets; the single
failure also fails at HEAD pre-fix (qv=0 synthetic fixture — provably
unaffected). `git diff --check` clean. Full proof:
`proofs/v014/short_field_h1_residual_classification.{py,json,md}`.

## Evidence table (h1, GPU−CPU)

| Field | RMSE | Bias | Class |
|---|---:|---:|---|
| PSFC | 323.1 Pa | −313.8 | EOS (−228 diag) + MU (−85) — FIXED root cause |
| P | 129.8 Pa | −85.7 | EOS, vapor-column profile (−296 @k0 → −28 aloft) — FIXED |
| PH | 85.2 | −53.3 | EOS-coupled hydrostatic response — FIXED |
| MU | 122.0 Pa | −85.1 | EOS-coupled + init-mode transient |
| T | 1.46 K | +0.67 | interior-dominant transient (int 1.61 / bdy 0.59) |
| U / V | 0.96 / 2.10 | −0.07 / −1.66 | dynamics response at h1 |
| QVAPOR | 3.3e-4 | +3.4e-6 | small |
| MUB / PB | 9.3 / 4.5 Pa | ~0 | 154/10494 cells >1 Pa, ALL spec_bdy band, ≤90 Pa |
| HFX / LH / PBLH | 38 / 54 / 79 | −5.9 / +12 / +24 | downstream (EOS + rad timing) |
| SWDOWN | 55.6 | −55.6 | radiation timing: COSZEN −0.0551 ≈ 15–20 min (xtime+radt/2 seed) |

## Provenance finding

- **Stale-input hypothesis DISMISSED**: the backfill manifest proves CPU truth
  ran `wrf-only` in the *same* `wrf_l2` run_dir the GPU symlinks resolve to —
  identical wrfinput/wrfbdy/namelist. Same case, same inputs.
- **Real caveat — init-mode mismatch by construction**: CPU d02 init =
  real.exe `wrfinput_d02` (`input_from_file=.true.`); GPU =
  `standalone_native_init_nested` (live-nest from d01; `wrfinput_d02` never
  read). CPU-WRF itself spins up +47 Pa domain-mean MU in hour 1 from its
  real-init. A perfect GPU live-nest cannot match this truth tightly at h1;
  the 72h tolerance manifest must carry an init-transient envelope term (or
  the gate must pair like-for-like init).

## Next command

GPU rerun of the same 1h falsifier on the fixed branch (exact command in
`proofs/v014/short_field_h1_residual_classification.md`), then
`scripts/compare_wrfout_grid.py`. Decision rule: if the post-fix P/PSFC biases
collapse to init-transient scale (~±50 Pa class), start the 72h gates; the
remaining radiation-timing (COSZEN) and spec_bdy MUB/PB classes are separate,
bounded items for the tolerance manifest.

## Handoff

- objective: classify/close the v0.14 h1 field-parity blocker — DONE (root
  cause fixed, classification complete).
- files changed: `src/gpuwrf/dynamics/acoustic_wrf.py` (+11/−2, EOS constants);
  new `proofs/v014/short_field_h1_residual_classification.{py,json,md}`; this
  review. NOTE: dycore edit is outside the default file-ownership list but
  inside the contract's "explicitly required by a proof and small" clause —
  manager review requested.
- commands run: see proof md (py_compile, proof run, json.tool, pytest
  subsets, git diff --check).
- proof objects: `proofs/v014/short_field_h1_residual_classification.{json,md}`.
- unresolved risks: (1) GPU rerun not yet done (no GPU lock) — fix proven
  against truth files, not yet end-to-end; (2) init-mode mismatch needs a
  tolerance-manifest or gate-pairing decision; (3) radiation-timing COSZEN
  offset (~15–20 min) unfixed, separate sprint; (4) GPU-only golden fixtures
  (e.g. `tests/savepoint/fixtures/wrf_b6_100step/golden`) may need
  regeneration after the EOS fix — could not be exercised CPU-only.
- next decision needed: manager approves GPU rerun + decides init-mode gate
  pairing policy.
