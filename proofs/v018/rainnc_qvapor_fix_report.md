# v0.18 RAINNC + QVAPOR Report

## Verdict

RAINNC remains a bounded miss.  No tolerance was widened.

QVAPOR passes the Switzerland 72 h gate and improved versus the v017 baseline.

## Process Oracle Findings

Pristine-WRF per-process oracle:

- `prr_wau` warm-rain autoconversion: matched, L1-relative `6.49e-6`.
- `prr_rcw` warm-rain accretion: matched, L1-relative `6.73e-7`.
- `prr_rcg/prg_rcg` rain-graupel cold collection: matched, L1-relative `3.40e-8`.
- `prr_rci/prg_rci/prs_sci` rain/snow collecting cloud ice: matched, L1-relative `1.00e-7`, `6.86e-8`, `6.52e-7`.
- `pri_wfz/pni_wfz` cloud-water freezing was WRF-active/JAX-zero in the bounded pass; fixed by adding reduced default-IN qcfz planes from `freezeH2O.dat` and applying WRF source-stage cloud-water freezing.

After the qcfz fix, no material mass-production process remains WRF-active/JAX-zero in the compared production set.  The remaining WRF-active/JAX-zero terms are Hallett-Mossop ice-number/mass terms with tiny mass rates (`pri_ihm=2.19e-9`, `prs_ihm=1.81e-9`, `prg_ihm=3.71e-10` abs-sum in the coldmix oracle).  The remaining non-green production signal is diffuse/staging: WRF stages warm rain, cloud freezing, riming, deposition, and conservation from one source state; the current JAX body still applies some of those processes sequentially.  The one-step coldmix precip partition itself is close (`pptrain` L1-relative `4.91e-4`; snow/ice/graupel surface partition inactive in this fixture).

## 72 h Switzerland Gate

CPU reference: `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`

Post-qcfz GPU run: `/mnt/data/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_qcfz_20260616T115735Z`

Wrapper result: `PIPELINE_GREEN`, dimension compare PASS, 72 wrfouts, inventory PASS.

Target RMSE closure:

- RAINNC: `5.0787054202182125 -> 5.2200971263847045 -> 5.220155956735137` mm, FAIL vs `1.0`.
- QVAPOR: `5.858611198575604e-4 -> 5.583985710404751e-4 -> 5.58447999338664e-4`, PASS vs `1.0e-3`.
- QRAIN: `3.226197032078842e-5 -> 5.955657960599811e-6 -> 5.955062566604952e-6`.
- QGRAUP: `3.8374543503046615e-6 -> 1.9462071224597008e-6 -> 1.9469118661019955e-6`.
- QNRAIN: `2127767.2624371015 -> 313.11025291019195 -> 313.1023411137776`.

Accumulator decomposition again rules out writer/convention error:

- `RAINC` RMSE is `0.0`.
- `RAINNC = derived liquid + SNOWNC + GRAUPELNC` closes as true accumulated precipitation.
- Derived-liquid RMSE: `4.150177713162069 -> 4.296899246212194 -> 4.296910789520173`.
- `SNOWNC` RMSE: `2.908838444559882 -> 2.908122512935265 -> 2.9081612239109256`.
- `GRAUPELNC` RMSE: `0.4710224637351543 -> 0.4584771976481336 -> 0.4584574568873638`.

## Proof Objects

- `proofs/v018/thompson_process_oracle.py`
- `proofs/v018/thompson_process_oracle.json`
- `proofs/v018/switzerland_72h_target_metrics.json`
- `/mnt/data/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_qcfz_20260616T115735Z/switzerland_d01_72h_grid_compare.json`
- `/mnt/data/wrf_gpu_validation/v018_rainnc_qvapor_switzerland_d01_72h_qcfz_20260616T115735Z/switzerland_d01_72h_grid_compare.md`

## Carried Risk

RAINNC is not green after faithful cold-collision, ice-collection, WRF mp8 graupel-number, and qcfz source-stage fixes.  The remaining residual is attributed to broader Thompson production/partition staging and coupled accumulated-precip propagation, not to a single bounded missing production term found in this pass.
