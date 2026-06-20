# V0.14 Short Field H1 Residual Classification

Verdict: `FIXED_EOS_MOISTURE_FACTOR_GPU_RERUN_REQUIRED`.

Generated: 2026-06-10 WEST. CPU-only proof; GPU used: `False`.
Companion JSON: `proofs/v014/short_field_h1_residual_classification.json`.

## Root cause (proven, fixed)

The GPU dycore equation of state used the wrong moisture factor:
`qvf = 1 + 0.608*qv` (WRF's `p608 = rvovrd-1`, the virtual-temperature/moist-
density convention) where WRF uses `qvf = 1 + rvovrd*qv` with
`rvovrd = Rv/Rd = 461.6/287 ≈ 1.6084` (the dry-alpha_d/dry-theta convention).
WRF anchors: `share/module_model_constants.F:41,97`,
`dyn_em/module_big_step_utilities_em.F:1064,1118,1140`.

Bit-level proof (EOS inverted on each wrfout with that file's own discrete
`alpha_d` recovered from PH/PHB, MU/MUB, DNW, C1H/C2H):

| File | qvf = 1+1.608·qv | qvf = 1+0.608·qv |
|---|---:|---:|
| CPU-WRF truth h1 | rmse **22.4 Pa** (fp32 rounding) | rmse 549.8 Pa |
| GPU pre-fix h1 | rmse 556.1 Pa | rmse **5.8 Pa** |

Each model is self-consistent with its own constant; CPU-WRF proves 1.608 is
the WRF factor. Hydrostatic cross-check: CPU PSFC = moist-hydrostatic integral
to 0.94 Pa bias; GPU PSFC = **dry**-hydrostatic −29.8 Pa (vapor column weight
~199 Pa missing). Delta-P profile decays −296 Pa (k=0) → −28 Pa (k≥20),
the column-vapor signature. This is the dominant driver of the
PSFC/P/PH/MU broad residual family and the long-open "one-RK-step P/PH/MU
dynamics frontier" broad-P component.

Fix applied: `src/gpuwrf/dynamics/acoustic_wrf.py` — both
`_pressure_from_theta_alt` and `_inverse_density_from_theta_pressure` now use
`RVOVRD = R_V/R_D` (+11/−2 lines, constants + comment). Post-fix the
production helper reproduces CPU-WRF truth P to rmse 27.0 Pa / max 55.5 Pa
(vs ~556 Pa pre-fix; 20× closure) and `alpha_d` to rel-rmse 7.2e-4.

## Residual classes at h1 (full decomposition)

| Field | RMSE | Bias | Class |
|---|---:|---:|---|
| PSFC | 323.1 Pa | −313.8 Pa | EOS factor (−228.5 diag part) + MU (−85) |
| P | 129.8 Pa | −85.7 Pa | EOS factor (height-decaying vapor profile) |
| PH | 85.2 | −53.3 | EOS-coupled hydrostatic response |
| MU | 122.0 Pa | −85.1 Pa | EOS-coupled dynamics + init-mode mismatch |
| T | 1.46 K | +0.67 K | interior-dominant (interior 1.61 vs boundary 0.59) |
| U / V | 0.96 / 2.10 m/s | −0.07 / −1.66 | dynamics response, h1 transient |
| QVAPOR | 3.3e-4 | +3.4e-6 | small |
| MUB / PB | 9.3 / 4.5 Pa | ~0 | 154/10494 cells >1 Pa, ALL in spec_bdy band, max ~90 Pa |
| SWDOWN / SWNORM | 55.6 / 57.4 | −55.6 / −55.6 | radiation timing: COSZEN −0.0551 (~15–20 min, WRF seeds xtime+radt/2) |
| HFX / LH / PBLH | 38.2 / 53.9 / 79.0 | −5.9 / +12.0 / +24.0 | downstream of EOS + radiation timing |

## Provenance finding

- Stale-input hypothesis DISMISSED: backfill manifest
  (`.cpu_wrf_backfill/20260603T000612Z_manifest.json`) proves CPU truth was run
  `wrf-only` **in the same run_dir** the GPU symlinks point to — identical
  `wrfinput_d01/d02`, `wrfbdy_d01`, `namelist.input`.
- Real provenance caveat: **init-mode mismatch by construction.** CPU truth has
  `input_from_file=.true.,.true.` (d02 from real.exe `wrfinput_d02`); the GPU
  run used `init_mode=standalone_native_init_nested` (d02 live-nested from
  d01, `wrfinput_d02` never read). Even a perfect GPU model will not converge
  to this CPU truth at h1 to tight tolerances; CPU-WRF's own d02 spins up
  +47 Pa domain-mean MU in the first hour from its real-init. The 72h gate
  tolerance manifest must treat the init transient as an envelope term, or the
  gate must pair like-for-like init modes.

## Commands run

- `python -m py_compile src/gpuwrf/dynamics/acoustic_wrf.py proofs/v014/short_field_h1_residual_classification.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/short_field_h1_residual_classification.py`
- `python -m json.tool proofs/v014/short_field_h1_residual_classification.json` (valid)
- `pytest tests/test_m6x_c2_acoustic.py tests/test_m6x_pressure_diagnose_wiring.py tests/test_m6x_c2_pgf.py tests/test_m6x_c2_scan.py tests/test_m6x_adr023_column_solver.py tests/test_m6_horizontal_pressure_gradient_fix.py tests/test_m6x_vertical_acoustic_oracle.py` → 24 passed, 1 failed, 3 skipped; the single failure (`test_diagnostic_pressure_al_alt_matches_base_rest_state`) **also fails at HEAD pre-fix** (phb=0 synthetic fixture, qv=0 so the fix cannot affect it) — pre-existing.
- `pytest tests/test_m7_l2_d02_replay.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py` → 4 passed, 2 skipped.
- `git diff --check` clean.

## Next command (manager, GPU lock required)

Re-run the 1h falsifier on the fixed branch, then the comparator:

```bash
RUN_ROOT=<DATA_ROOT>/wrf_gpu_validation/v014_short_field_falsifier_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_ROOT"/{gpu_output,proofs,resources}
GPUWRF_RESOURCE_LOG_DIR="$RUN_ROOT/resources" GPUWRF_RESOURCE_LABEL=v014_short_field_h1_eosfix \
scripts/run_gpu_lowprio.sh -- python proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
  --run-root /tmp/v0120_merged_run_root \
  --cpu-truth-root <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output \
  --run-id 20260501_18z_l2_72h_20260519T173026Z --hours 1 \
  --output-root "$RUN_ROOT/gpu_output" --proof-dir "$RUN_ROOT/proofs"
```

Expected post-fix at h1: PSFC bias ~−85→small (init-transient scale), P low-level
bias −296→init-transient scale; SWDOWN/COSZEN offset persists (separate
radiation-timing class); MUB/PB boundary cells persist (separate class).
