# Tests

`pytest tests/` exercises the GPU port: dynamical-core idealized gates, per-scheme
savepoint-parity oracles (vs an unmodified WRF build), namelist + scheme-catalog validation,
the standalone native-init and nested pipelines, IO round-trips, restart bit-identity, and
conservation budgets.

Running:

- **Most tests require a GPU** (the operational `State` allocates on the device). On a
  CPU-only host these skip with a clear reason (see `tests/conftest.py`); run on a CUDA GPU
  for the full suite.
- Some tests need WRF reference / corpus fixtures that are not vendored (purged or external);
  these skip with a reason when the data is absent.
- For the heavy savepoint suite, **per-file isolation** is the reliable method
  (`JAX_PLATFORMS=cpu` for the CPU-runnable subset); a single-process full-suite run can
  exhaust the CPU XLA backend on the largest coupled-step parity tests.

Proof objects produced or checked by the suite live under `proofs/`.
