# v0.11.0 MYNN EDMF Status

Objective: enable the operational MYNN EDMF and cloud/moisture terms that affect U10/V10/T2/Q2/PBLH, matching the Canary WRF MYNN options where proven.

## What Changed

- The operational MYNN coupler is wired with `_MYNN_EDMF = True`.
- DMP mass flux now carries scalar and momentum plume fluxes, matching WRF defaults `bl_mynn_edmf=1` and `bl_mynn_edmf_mom=1`.
- The DMP land/water branch now follows WRF source behavior for `hux`, width factors, `acfac`, and entrainment/exchange factor.
- MYNN column thermodynamics now read `qc` and `qi`, compute liquid-water potential temperature and moist virtual potential temperature, and feed cloud-aware `dtl`, `dqw`, `dtv` into level-2.5 closure.
- Turbulence now includes scalar variance production terms `pdt`, `pdq`, and `pdc` rather than the previous dry-zero path.
- U/V mean tendencies now consume EDMF momentum arrays when EDMF is enabled.

## Operational Options Checked

Corpus namelists select MYNN via `bl_pbl_physics = 5, 5` and MYNN surface layer via `sf_sfclay_physics = 5, 5`, for example:

- `<DATA_ROOT>/canairy_meteo/gen2_archive/provenance/namelists/namelist_20240315_00z_A.input`

WRF Registry defaults checked from the available WRF sources:

- `bl_mynn_cloudpdf=2`
- `bl_mynn_edmf=1`
- `bl_mynn_edmf_mom=1`
- `bl_mynn_edmf_tke=0`
- `bl_mynn_cloudmix=1`
- `bl_mynn_mixqt=0`
- `icloud_bl=1`
- `bl_mynn_closure=2.6`

## Proofs

- `proofs/mynn_edmf/mf_oracle_compare.json`: WRF DMP scalar mass-flux savepoint parity passes.
- `proofs/mynn_edmf/integration_mf_vs_ed.json`: 120-minute frozen-forcing column proxy is finite, but does not support a skill win because near-surface qv is lower with MF than ED-only in this proxy.
- `artifacts/m5/tier1_mynn_parity.json`: old MYNN tier-1 analytic fixture still passes tolerances after scalar variance terms, with larger el/km/kh drift recorded.
- `tests/test_mynn_edmf_oracle.py`: focused EDMF oracle/unit coverage passes, including operational enablement, land/water branch, momentum flux arrays, and cloud-condensate sensitivity.

Commands run:

- `git log -1 --oneline`
- `git branch --show-current`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m py_compile src/gpuwrf/physics/mynn_pbl.py src/gpuwrf/physics/mynn_edmf.py src/gpuwrf/coupling/physics_couplers.py`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_mynn_edmf_oracle.py -q`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python proofs/mynn_edmf/jax_oracle.py`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python proofs/mynn_edmf/integration_oracle.py`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_m5_mynn_tier1.py tests/test_m5_mynn_tier2.py tests/test_m5_mynn_radicand.py -q`

## Status Against Endpoint

Endpoint 1: partial pass. EDMF scalar and momentum mass flux are wired and operationally on, and qc/qi-aware thermodynamics are active. Full WRF `icloud_bl=1` cloud PDF/cloudmix is not complete.

Endpoint 2: pass for the available WRF DMP savepoint. `s_aw` rel max error is 0.0048347386 and `s_awqv` rel max error is 0.0048342746 against tolerance 0.05. No WRF momentum savepoint was available for `s_awu/s_awv`.

Endpoint 3: not run. A short d02 finite/stable and not-worse skill A/B against the dry-2.5 baseline still needs a paired GPU run through `/tmp/wrf_gpu_run.sh`; historical artifacts were not substituted.

## Carry

- Complete WRF `mym_condensation`/cloud PDF/cloudmix behavior for `icloud_bl=1`.
- Add qc/qi/cloud-fraction writeback if the manager approves expanding the frozen B2 write contract.
- Add WRF savepoint parity for `s_awu/s_awv`.
- Run paired short d02 dry-2.5 vs EDMF A/B and report U10/V10/T2/Q2/PBLH skill deltas.
- Re-baseline or tighten the tier-1 analytic MYNN fixture after deciding whether scalar variance production belongs in that fixture's oracle.
