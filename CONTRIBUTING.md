# Contributing to wrf_gpu

Thanks for your interest. wrf_gpu is a GPU-native, WRF-compatible regional NWP
system (a clean JAX rewrite validated against WRF as an oracle, not a Fortran
port). This guide is for external contributors. The project was developed by a
team of AI agents under human direction; the agent-side process is documented
separately in [`CONTRIBUTING_AGENT.md`](CONTRIBUTING_AGENT.md) and the
[`.agent/`](.agent/) development log.

## Ground rules

- **Validation, not bitwise parity.** Correctness is judged against WRF as an
  oracle through a tiered pyramid: micro fixture / savepoint parity → physical
  invariants (conservation) → short-run / convergence → station-RMSE
  equivalence. See [`VALIDATION_STRATEGY.md`](VALIDATION_STRATEGY.md).
- **Every claim needs evidence.** A physics/numerics change must come with a
  proof object (fixture, analytic oracle, conservation budget, or ensemble).
  Performance changes need a profiler artifact and a transfer audit — no perf
  claim without measurement.
- **Honesty over spin.** Reports, PR descriptions, and docs must be honest about
  missing evidence and bounded acceptances. No over-claiming.
- **GPU memory discipline.** No host/device transfers inside the timestep loop,
  and no precision downcast on the operational path, without an ADR.

## Development setup

```bash
git clone https://github.com/wrf-gpu/wrf_gpu.git && cd wrf_gpu
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # CPU jax + pytest; import-checks without a GPU
pip install --upgrade "jax[cuda13]"   # add the CUDA build to actually run on GPU
```

See [`docs/quickstart.md`](docs/quickstart.md) for the full prerequisites and a
first forecast.

## Tests

```bash
pytest -q                         # the pytest suite (CPU for most; some need a GPU)
```

GPU-gated tests are marked (`close_gate`) and require a JAX CUDA backend. The
reproducibility and community-validation suites are documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and
[`docs/VALIDATION.md`](docs/VALIDATION.md).

## Pull requests

1. Branch from `main` with a purpose name (e.g. `fix/mynn-cloudpdf`).
2. Keep changes focused; keep doc/hygiene changes separate from code changes.
3. Include the proof object(s) for any physics/numerics/perf change.
4. Make sure `pytest -q` passes on a clean checkout (CPU rows at minimum).
5. Describe what you changed, what you ran, the proof produced, and any
   unresolved risk.

## Scope

The supported WRF namelist matrix and the deliberate boundaries are in
[`docs/namelist-compatibility.md`](docs/namelist-compatibility.md) and the
[`README.md`](README.md) scope table. Unsupported options fail closed with a
named reason — new schemes must be oracle-proven before they are wired into the
operational loop. The remaining gap to a complete WRF v4 port is tracked in
[`docs/GPU_PORT_GAPS_TODO.md`](docs/GPU_PORT_GAPS_TODO.md) and
[`PROJECT_PLAN.md`](PROJECT_PLAN.md).

## Reporting issues

Open an issue with: the namelist (or the failing scheme/option), the command,
the observed vs expected behaviour, and your environment (GPU, VRAM, CUDA
driver, JAX version). For numerical discrepancies, attach the relevant proof
JSON or `compare_wrfout_grid.py` output if you have it.

## License

By contributing you agree your contributions are licensed under the project
[`LICENSE`](LICENSE). See [`LICENSE_NOTES.md`](LICENSE_NOTES.md) for the
WRF-derived-material notes.
