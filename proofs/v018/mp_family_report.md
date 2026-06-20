# v0.18 Microphysics Family Batch Report

Date: 2026-06-16
Branch: `worker/gpt/v018-mp`
Base: `worker/opus/v018-trunk` at `fbec35443b10b665415d271b15245f8a06f7cd08`

## Outcome

Full ship gate: `true`.

No new microphysics scheme was promoted to operational GPU support in this
closer pass. The existing operational MP set was preserved exactly:

`{0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 16, 24, 26, 28, 97}`

The previous still-open closer set is now closed to the v0.18 bar as
reference-only / fail-closed with exact WRF oracle evidence:

`{5, 9, 18, 27, 29, 40, 50, 51, 52, 53, 56}`

Exact-module rule applied: an oracle counts only for the exact WRF
module/routine it drives. MP95 remains validated only by
`module_mp_etanew.F:ETAMP_NEW`; MP5 now has its own `module_mp_fer_hires.F:FER_HIRES`
oracle and does not reuse the MP95 artifact.

## Closer Endpoints

| MP | WRF scheme | Endpoint class | Oracle evidence |
| --- | --- | --- | --- |
| 5 | Ferrier-HRW new Eta | ref-with-oracle / fail-closed | Standalone pristine-WRF driver calls `FER_HIRES`; savepoints in `proofs/v018/mp_oracles/ferrier_hires/savepoints` |
| 9 | Milbrandt-Yau 2-moment | ref-with-oracle / fail-closed | Active pristine-WRF full-model oracle at `proofs/v018/mp_oracles/wrf_full_model/mp9/oracle_summary.json` |
| 18 | NSSL 2-moment 4-ice + CCN | ref-with-oracle / fail-closed | Active pristine-WRF full-model oracle at `proofs/v018/mp_oracles/wrf_full_model/mp18/oracle_summary.json` |
| 27 | UDM 7-class / UFS double-moment | ref-with-oracle / fail-closed | Active pristine-WRF full-model oracle at `proofs/v018/mp_oracles/wrf_full_model/mp27/oracle_summary.json` |
| 29 | RCON | ref-with-oracle / fail-closed | Active pristine-WRF full-model oracle at `proofs/v018/mp_oracles/wrf_full_model/mp29/oracle_summary.json` |
| 40 | Morrison aerosol | ref-with-oracle / fail-closed | Active pristine-WRF full-model oracle at `proofs/v018/mp_oracles/wrf_full_model/mp40/oracle_summary.json` |
| 50 | P3 1-category | ref-with-oracle / fail-closed | Active pristine-WRF full-model oracle at `proofs/v018/mp_oracles/wrf_full_model/mp50/oracle_summary.json` |
| 51 | P3 1-category + cloud number | ref-with-oracle / fail-closed | Active pristine-WRF full-model oracle at `proofs/v018/mp_oracles/wrf_full_model/mp51/oracle_summary.json` |
| 52 | P3 2-category | ref-with-oracle / fail-closed | Active pristine-WRF full-model oracle at `proofs/v018/mp_oracles/wrf_full_model/mp52/oracle_summary.json` |
| 53 | P3 1-category 3-moment | ref-with-oracle / fail-closed | Active pristine-WRF full-model oracle at `proofs/v018/mp_oracles/wrf_full_model/mp53/oracle_summary.json` |
| 56 | NTU multi-moment | ref-with-oracle / fail-closed | Active pristine-WRF full-model oracle at `proofs/v018/mp_oracles/wrf_full_model/mp56/oracle_summary.json` |

The full-WRF oracles use a physics-pristine, WRFGPU2_ORACLE-instrumented
`wrf.exe` with `em_quarter_ss` for one simulated minute. The `wrfinput_d01`
files were seeded in a small central patch with only fields exposed by the
selected `mp_physics`; each committed JSON summary records the `wrf.exe` hash,
exact physics-module checksums, driver checksums, raw `wrfout`/log hashes, WRF
success line, two history times, seeded variables, and nonzero seeded
microphysics-field deltas. The instrumentation is dump-only/numerically inert;
no clean uninstrumented full-`wrf.exe` rebuild is claimed.

## Other Endpoint Classes

Already reference-with-oracle / fail-closed:

- MP7 Goddard 4-ice / NUWRF: `proofs/v018/mp_oracles/goddard4ice`
- MP38 Thompson graupel-hail: `proofs/v018/mp_oracles/thompgh`
- MP95 Ferrier old Eta / etampnew: `proofs/v018/mp_oracles/ferrier_etanew`

Proven irrelevant/no-op / fail-closed:

- MP11 CAM 5.1: CAM-specific macrophysics/cloud-fraction/radiation dependency.
- MP17/19/21/22 NSSL legacy options: superseded by MP18 modifier flags per
  `doc/README.NSSLmp`.
- MP30/32 HUJI SBM: research spectral-bin/table architecture outside the lean
  operational bulk-MP target.
- MP55 Jensen-ISHMAEL: initial-release habit research model with external
  tables.
- MP96 MadWRF: `CASE (MADWRF_MP)` only emits `wrf_debug` in the MP driver.

## Proof Objects

- `proofs/v018/mp_family_status.json`: requested closer status; `full_ship_gate=true`, `still_open=[]`.
- `proofs/v018/mp_endpoint_manifest.json`: machine-readable endpoint class for
  every operational and requested-open MP code.
- `proofs/v018/mp_oracles/ferrier_hires`: exact MP5 `FER_HIRES` oracle driver,
  source checksums, and four savepoints.
- `proofs/v018/mp_oracles/wrf_full_model`: active pristine-WRF full-model oracle
  JSON summaries for MP9/18/27/29/40/50/51/52/53/56.
- `proofs/v018/mp_oracles/ferrier_etanew`: exact MP95 `ETAMP_NEW` oracle source
  and fp32/fp64 savepoints.
- `proofs/v018/mp_oracles/goddard4ice`: exact MP7 Goddard 4-ice oracle source
  and fp32/fp64 savepoints.
- `proofs/v018/mp_oracles/thompgh`: exact MP38 Thompson-GH oracle source and
  fp32 savepoint.

## Validation Commands

Endpoint/status/manifest/catalog gate:

```bash
PYTHONPATH=src JAX_PLATFORM_NAME=cpu pytest -q tests/test_v018_mp_family_fail_closed.py
```

Result: `34 passed in 3.64s`.

Full CPU gate:

```bash
PYTHONPATH=src JAX_PLATFORM_NAME=cpu pytest -q \
  tests/test_v018_mp_family_fail_closed.py \
  tests/test_scheme_catalog_fail_closed.py \
  tests/test_namelist_check.py \
  tests/test_v060_physics_dispatch.py \
  tests/test_v017_qh_hail_state.py
```

Result: `121 passed, 1 skipped in 3.35s`.

GPU smoke through the required lock:

```bash
scripts/with_gpu_lock.sh --label gpt-mp -- taskset -c 0-3 env \
  OMP_NUM_THREADS=4 \
  PYTHONPATH=src \
  JAX_PLATFORMS=cuda \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_ENABLE_COMPILATION_CACHE=true \
  JAX_COMPILATION_CACHE_DIR=<DATA_ROOT>/gpuwrf_jax_cache \
  pytest -q \
    tests/test_v017_qh_hail_state.py::test_state_zeros_hail_leaves_zero_on_gpu \
    tests/test_v013_operational_smoke.py::test_microphysics_operational_runs_and_mutates
```

Result: `15 passed in 14.60s`.

## Decision

The MP family closer is an honesty/fail-closed reference-oracle endpoint pass,
not an operational GPU port of the large MP schemes. The v0.18 bar is met
because every requested scheme is now operational, reference-with-real-oracle, or
proven irrelevant/no-op; no still-open MP scheme remains.
