# B1 Thompson — REAL WRF-oracle per-process parity

**Verdict: `B1_PARITY_PASS` (substantive).** Every prognostic water field (6 mass
species + 2 number concentrations) matches the WRF oracle inside the frozen
Phase-B transcription band. Theta matches WRF to within float32 storage
precision (≤ ~1 ULP everywhere); its nominal band "fail" is a
tolerance-vs-storage artifact, not a physics or transcription gap. This is a
**JAX-vs-WRF** comparison — no self-compare.

## What was run

- Oracle: `/mnt/data/wrf_gpu2/physics_oracle/microphysics/` — real WRF **v4.7.1**
  `mp_gt_driver` pre→post savepoints, `mp_physics=8`,
  `is_aerosol_aware=FALSE`, `is_hail_aware=FALSE`, grid_id 1 (parent domain).
- Format: raw **big-endian float64** (`dtype='>f8'`), C-order reshape to
  `(nj=57, nk=44, ni=91)` per `manifest.json` (authoritative). Vertical (k) is
  the **middle** axis; the harness transposes `(nj,nk,ni)→(nk,nj,ni)` before
  flattening to `(n_columns=5187, n_levels=44)`. `refl_10cm` (253 NaNs, a
  diagnostic artifact) is masked out per the contract.
- dt = **18 s** (domain-1 `time_step` from the source-run `namelist.input`).
- Harness: `gpuwrf.validation.tier1_thompson.run_oracle_parity_f64`
  (adapted from B1's `compare_against_oracle`, which was built for the frozen
  HDF5 schema; this adds the raw-`.f64` reader the factory actually emitted).
  fp64 throughout (`jax_enable_x64` at `gpuwrf` import).
- Runner: `proofs/b1/run_oracle_parity.py`. Proof JSON: `proofs/b1/oracle_parity.json`.
- Mask: inactive-physical moist mask — a column cell is enforced when
  pre-condensate ≥ 1e-8 kg/kg OR pre-qv ≥ 1e-6 kg/kg. 228 228 / 228 228 cells
  enforced (the whole atmosphere here has qv above the vapour floor).

## Per-field WRF parity (moist mask, n=228 228 cells)

| field | max abs err | max rel err | band (abs / rel) | PASS |
|-------|-------------|-------------|------------------|------|
| qv    | 3.45e-10 kg/kg | 4.39e-08 | 1e-9 / 1e-6 | **PASS** |
| qc    | 0.0 | 0.0 | 1e-9 / 1e-6 | **PASS** |
| qr    | 0.0 | 0.0 | 1e-9 / 1e-6 | **PASS** |
| qi    | 0.0 | 0.0 | 1e-9 / 1e-6 | **PASS** |
| qs    | 0.0 | 0.0 | 1e-9 / 1e-6 | **PASS** |
| qg    | 0.0 | 0.0 | 1e-9 / 1e-6 | **PASS** |
| ni (Ni) | 0.0 | 0.0 | 1e-3 / 1e-4 | **PASS** |
| nr (Nr) | 0.0 | 0.0 | 1e-3 / 1e-4 | **PASS** |
| th (theta) | 3.37e-05 K | 1.13e-07 | 1e-6 / 1e-7 | band-FAIL → **storage-precision PASS** |

Water-mass + surface-precip closure: max rel residual **0.0** (exact). PASS.

## Honest scope of this oracle savepoint

This particular WRF savepoint is a **near-inactive microphysics step**: the only
nonzero condensate in the pre-state is a trace `qc ≤ 3.6e-8 kg/kg`; rain, ice,
snow, graupel and both number concentrations are identically zero in **both**
the WRF input and output (md5-confirmed all-zeros). WRF here only evaporates that
trace cloud water (Δqv ~ 3.6e-8 kg/kg, Δtheta ~ 1e-4 K). Consequences:

- The kernel's source/sink + sedimentation path is confirmed to make the
  **same** decision WRF does on this state (evaporate the trace qc, leave the
  rest), to float32 precision.
- This savepoint does **not** exercise the unported cross-species collection
  terms (`qr_acr_qs`/`qr_acr_qg`, the racs/sacr/racg/gacr lookup tables). With
  zero qr/qs/qg in the column those processes are inactive in WRF too, so they
  cannot be the cause of any disagreement here — and they aren't: every
  hydrometeor matches WRF exactly. **The unported collection does NOT limit
  parity on this oracle** (it would only bind on a precipitating, mixed-phase
  column, which this savepoint is not). That remains an open limit for a future
  precipitating oracle, not a measured failure today.

## Theta diagnosis (why the band "fails" but the physics passes)

The entire oracle state is **float32-exact** (verified: `th`, `qv`, `pii`, `p`,
`dz8w` round-trip through float32 unchanged) — WRF runs/stores fp32 and the
factory upcast to fp64 for the dump. Therefore:

- WRF's reported theta change over the step is quantized to the float32 grid:
  the only distinct |Δtheta_WRF| values are **{1, 2, 3} × the float32 ULP**
  (3.05e-5, 6.10e-5, 9.16e-5 K at theta≈297 K). The physical theta tendency of
  this near-inactive step is *at or below WRF's own storage granularity*.
- The JAX fp64 theta change agrees with WRF to **≤ ~1 float32 ULP**:
  **99.998 %** of cells within 1 ULP, **100 %** within 2 ULP, median error
  exactly 0, worst case **1.10 ULP** (= the 3.37e-5 K max abs error).
- The frozen transcription band `rel=1e-7` corresponds to ≈ 0.3 float32 ULP at
  theta≈297 K — i.e. **tighter than the oracle's own storage precision can
  represent**. No fp64 kernel can pass a 0.3-ULP band against a 1-ULP-quantized
  reference; this is a tolerance-vs-storage mismatch, not a kernel error.

I did **not** loosen the band: `phase_b_savepoint.PHASE_B_TOLERANCES` is a frozen
SHARED-CORE file (manager-merge only), and the right fix is a manager/ladder
decision (e.g. a float32-ULP-aware theta band for fp32-sourced oracles), not a
lane edit. The proof records `th_storage_precision_pass=true` and
`all_fields_within_float32_storage_precision=true` so the substantive result is
unambiguous.

## Bottom line

JAX Thompson reproduces this real WRF mp_gt_driver step to float32 storage
precision on **all** prognostic fields, with exact water-mass closure. No
transcription bug was found; no kernel change was needed. The one nominal band
miss (theta) is below the oracle's fp32 storage granularity. The unported
cross-species collection is **not** exercised by — and does not limit — this
oracle; quantifying it needs a precipitating mixed-phase savepoint.
