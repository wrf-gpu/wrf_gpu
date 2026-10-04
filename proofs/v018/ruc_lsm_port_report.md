# RUC LSM JAX Port Report

## Objective

Complete a faithful CPU/fp64 JAX port of the WRF RUC land-surface model
no-snow land-column path (`sf_surface_physics=3`) and validate it against the
staged pristine-WRF oracle savepoint.

## Status

Oracle result: **GREEN** in the extracted integration-base work tree
`.wt-gpt-v018-ruc-lsm`, based on `worker/opus/v017-integration`
(`2de17c71aa1007bae02ff7e7db79d59f52bb7b75`).

Branch/commit status: **blocked by sandbox**. Creating
`worker/gpt/v018-ruc-lsm` failed because `.git/refs/...` is read-only:
`Unable to create ... worker/gpt/v018-ruc-lsm.lock: Read-only file system`.
No commit hash was produced. Exact changes are preserved in
`proofs/v018/ruc_lsm_port.patch`.

## Files Changed

- `.wt-gpt-v018-ruc-lsm/src/gpuwrf/physics/lsm_ruc.py`
- `.wt-gpt-v018-ruc-lsm/tests/test_v017_lsm_adv.py`
- `.wt-gpt-v018-ruc-lsm/tests/test_v018_ruc_lsm_parity.py`
- `proofs/v018/ruc_lsm_port.patch`
- `proofs/v018/ruc_lsm_parity_metrics.json`

## Validation

Command:

```bash
GPUWRF_JAX_CACHE=0 JAX_ENABLE_X64=1 JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu PYTHONPATH=src pytest -q tests/test_v017_lsm_adv.py tests/test_v018_ruc_lsm_parity.py
```

Result: `12 passed in 12.09s`.

Metrics proof: `proofs/v018/ruc_lsm_parity_metrics.json`

- all fields GREEN: `true`
- worst absolute residual: `LH max_abs=2.897803869483795e-04`, tolerance `5.0e-04`
- worst relative residual: `SH2O max_rel=5.960929348270716e-05`, tolerance `1.0e-04`
- `SH2O max_abs=2.2930633854056914e-05`, tolerance `3.0e-05`

## Proof Objects

- `proofs/v018/ruc_lsm_port.patch`
  - sha256: `36b9d7c90b8bc6bb2108b7dfd66304e302053e2d5e90b33acf90b64ec3ce0b14`
- `proofs/v018/ruc_lsm_parity_metrics.json`
  - sha256: `b38bff2882c11e21f923db0b12e9c70d8a26077610e61ab763d4e20f0df90227`

## Unresolved Risks

- No GPU smoke was run, per instruction.
- Registry/catalog/scan wiring was intentionally not edited. RUC remains
  fail-closed in the shared registry until the integration owner wires the land
  carry path.
- The validated path is the oracle-covered warm land/no-snow RUC path. Snow,
  sea-ice, water, and broader table classes still require staged oracle regimes
  before operational enablement.
- The sandbox prevented branch creation and commit. Apply the patch artifact to
  a writable checkout before merge review.

## Next Decision

Registry wiring follow-up: integrate `sf_surface_physics=3` only after the
manager-owned land carry interface is ready and this patch is applied on the
stable v0.18 trunk branch.
