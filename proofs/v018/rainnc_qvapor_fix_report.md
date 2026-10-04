# v0.18 RAINNC + QVAPOR Fix Report

## Outcome

RAINNC is **not green**.  The v0.18 Thompson cold-process oracle is green for
the targeted WRF terms, but the full 72 h Switzerland coupled gate did not
close:

- `RAINNC` pooled RMSE: **5.078705 -> 5.220097 mm** (`FAIL`, tolerance 1.0 mm).
- `QVAPOR` Switzerland pooled RMSE: **5.858611e-4 -> 5.583986e-4 kg/kg**
  (`PASS`, tolerance 1.0e-3 kg/kg).

This is not an accumulator/wrfout convention bug.  The v018 decomposition closes
to roundoff with `RAINNC = liquid_derived + SNOWNC + GRAUPELNC`, and `RAINC=0`
on both CPU and GPU for this case.  The remaining RAINNC residual is true
accumulated-precipitation production/partition, mainly derived liquid plus snow:

| Channel | v017 RMSE | v018 RMSE |
| --- | ---: | ---: |
| `RAINNC` | 5.078705 | 5.220097 |
| `LIQ_DERIVED = RAINNC-SNOWNC-GRAUPELNC` | 4.152220 | 4.297448 |
| `SNOWNC` | 2.985266 | 2.984297 |
| `GRAUPELNC` | 0.475496 | 0.462751 |

## What Was Fixed

The initial premise that `qr_acr_qs` / `qr_acr_qg` were absent was false: they
were present.  The actual process-level divergences found and fixed were:

- Missing WRF `prs_sci`, `prr_rci`, and `prg_rci` cold ice-collection terms.
- WRF mp=8 diagnostic graupel number reset: JAX was using a hardwired graupel
  number path that made `idx_g1=37` where WRF used roughly `19-20`.
- Effective WRF mp=8 `qr_acr_qg` table view: WRF reads the practical
  `idx_bg1=5` plane in its allocated table layout; the fixture now materializes
  that bit-faithful view.

## Validation

Process oracle:

- `proofs/v018/thompson_process_oracle.py`
- `proofs/v018/thompson_process_oracle.json`
- State oracle root:
  `<DATA_ROOT>/wrf_gpu2/v018_thompson_process_oracle/state/microphysics_coldmix`
- Process oracle root:
  `<DATA_ROOT>/wrf_gpu2/v018_thompson_process_oracle/process`

Key process comparisons:

| Term | WRF abs | JAX abs | L1 rel |
| --- | ---: | ---: | ---: |
| `prg_rcg` | 3.557349e-05 | 3.557349e-05 | 3.400845e-08 |
| `prr_rcg` | 3.557349e-05 | 3.557349e-05 | 3.400845e-08 |
| `prg_rci` | 4.442608e-04 | 4.442608e-04 | 6.863691e-08 |
| `prr_rci` | 1.689926e-04 | 1.689926e-04 | 9.978350e-08 |
| `prs_sci` | 2.986451e-07 | 2.986451e-07 | 6.524182e-07 |
| `prr_rcs` | 3.314994e-03 | 3.314994e-03 | 5.791046e-08 |
| `prs_rcs` | 1.367317e-04 | 1.367317e-04 | 0.0 |

The process oracle reports `rci_sci_terms_match_wrf=true` and no WRF-active /
JAX-zero terms among the compared cold collection family.

72 h Switzerland gate:

- Run root:
  `<DATA_ROOT>/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_20260616T102724Z`
- GPU run log:
  `<DATA_ROOT>/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_20260616T102724Z/switzerland_d01_72h_gpu.log`
- Full grid compare:
  `<DATA_ROOT>/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_20260616T102724Z/switzerland_d01_72h_grid_compare.json`
- Target metric summary:
  `proofs/v018/switzerland_72h_target_metrics.json`

The GPU pipeline itself was green: 72 wrfouts, dimension compare PASS,
`wrfout_inventory_status=PASS`, wrapper rc 0.  The CPU-only full grid comparator
ran over 72 paired files and 102 numeric fields; its overall verdict is FAIL
because multiple non-target dynamic fields still differ, with `RAINNC` still
outside its frozen tolerance.

## QVAPOR Verdict

No v018 QVAPOR code change was applied.  Switzerland `QVAPOR` remains green and
slightly improved in this run.  The Canary QVAPOR miss remains scoped by the
existing independent v015 verdict:

- `proofs/v015/qvapor_attribution_independent.json`
- `proofs/v015/qvapor_green_scope_verdict.md`

That evidence exonerates the MYNN surface-flux law and attributes the Canary
miss to MYNN-EDMF marine entrainment-depth / vertical redistribution fidelity,
not a bounded local surface-layer fix.

## Commands Run

```bash
env JAX_PLATFORMS=cpu PYTHONPATH=src pytest -q \
  tests/test_thompson_precip_oracle.py \
  tests/test_thompson_cold_collection_oracle.py

env JAX_PLATFORMS=cpu PYTHONPATH=src python proofs/v018/thompson_process_oracle.py

scripts/with_gpu_lock.sh --label gpt-rainnc -- bash -lc 'taskset -c 0-3 env ... \
  python -m gpuwrf run \
    --input-dir <DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu \
    --output-dir <DATA_ROOT>/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_20260616T102724Z/gpu_output \
    --scratch-dir <DATA_ROOT>/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_20260616T102724Z/scratch \
    --domain d01 --hours 72 \
    --proof-dir <DATA_ROOT>/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_20260616T102724Z/proofs \
    --compare-cpu-dir <DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu'

env PYTHONPATH=src python scripts/compare_wrfout_grid.py \
  --cpu-dir <DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu \
  --gpu-dir <DATA_ROOT>/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_20260616T102724Z/gpu_output \
  --domain d01 --min-lead 1 --max-lead 72 \
  --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
  --out-json <DATA_ROOT>/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_20260616T102724Z/switzerland_d01_72h_grid_compare.json \
  --out-md <DATA_ROOT>/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_20260616T102724Z/switzerland_d01_72h_grid_compare.md \
  --progress 25
```

## Unresolved Risk / Next Localizer

The compared cold Thompson process family is faithful, but full coupled RAINNC
is still bounded-but-off.  The next useful localizer is not another table guess:
diff WRF vs JAX precipitation production/partition along the coupled trajectory,
starting with derived liquid accumulation and snow sedimentation/placement,
then fall-speed/substep timing if the direct production terms match.
