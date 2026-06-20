# v0.13 Validation Campaign - CPU Lanes A4-A7

Base: worker/opus/v0120-integration @ 0b917ea6. Branch: validation-cpu-a4a7.
CPU only (JAX_PLATFORMS=cpu), pinned `taskset -c 12-23`. OUT=<DATA_ROOT>/wrf_gpu_validation/v0130_campaign_cpu.

| Lane | Verdict | Headline |
| --- | --- | --- |
| A4 | PASS | Straka front 14150 m, warm bubble theta'=1.92 K, conservation rel-residual 0.0/-2.45e-16, restart bit-identical |
| A5 | FAIL | `AttributeError: '_NLStub' has no attribute 'ra_sw_physics'` - smoke never runs |
| A6 | PASS | 89 passed / 0 failed in 8m44s |
| A7 | PASS | all 6 oracles PASS vs gfortran; fakemesh partition-invariant bit-identity (P=2,4,8) |

## A5 FAILURE (immediate debug lane)

`proofs/v060/multicfg_operational_smoke.py:554` calls `_resolve_operational_suite(_NLStub(cfg))`.
`operational_mode.py:2485` does `int(getattr(namelist, key))` over `_SCAN_WIRED_OPTIONS`, which v0.13
extended with `ra_sw_physics`/`ra_lw_physics` (lines 2431, 2435) - WITHOUT a getattr default. The proof
stub `_NLStub.__init__` (lines 298-314) never sets those two attrs -> deterministic crash on the FIRST config.

- Scope: PROOF-HARNESS GAP, not a production regression. The real `OperationalNamelist`
  (`operational_mode.py:393`) defines `ra_sw_physics:int=4` + `ra_lw_physics` with defaults, so the public
  `run_forecast_operational` path is fine.
- Honest fail-closed assessment: NOT possible this run - the smoke crashed before any accept/reject logic,
  so the full scheme matrix (MP/PBL/SFC/LSM/CU/RA families) through the operational coupler is UNVERIFIED.
- Fix hint: add `self.ra_sw_physics`/`self.ra_lw_physics` to `_NLStub.__init__`, then re-run A5.
- Repro: `taskset -c 12-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src python proofs/v060/multicfg_operational_smoke.py --steps 8 --out /tmp/a5.json`

## Note on log noise
A5/A7 logs carry thousands of benign XLA:CPU AOT "machine feature +prefer-no-gather not supported" warnings
from a stale JIT cache (different CPU feature set). Non-fatal; A7 PASS confirms they don't affect correctness.
They are NOT the A5 cause.
