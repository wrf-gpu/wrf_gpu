# v0.9.0 README runnability infrastructure — Opus lane delivery

Date: 2026-06-03
Author: Opus implementer lane (worker/opus/readme-runnability)
Base: e9982500e4a299ba0531893a080cb248283f6588 (trunk-0.9.0)
Plan followed: .agent/reviews/2026-06-03-gpt-readme-runnability-plan.md (P0 lanes A, B, D + dry smoke)

## Objective

Make the v0.9.0 binding gate reconstructable by a clean-room naive agent: build
a public `gpuwrf run` entrypoint, remove hardcoded `/mnt/data` / `/home/enric`
paths from the runnable path, and rewrite the README install + quickstart + run
sections so README.md alone is sufficient. Infrastructure only — the release
asset and the full GPU clean-clone gate are the manager's release-time work.

## What was built

### Path indirection — `gpuwrf.config.paths` (GPT plan lane B)
- New `src/gpuwrf/config/{__init__,paths}.py`: env-overridable helpers
  `repo_root / data_root / canairy_root / wrf_l3_root / wrf_l2_root / aemet_root /
  jax_cache_dir / mpirun_path / wrf_exe_path`. All `GPUWRF_*`-backed with
  checkout-relative defaults; precedence is CLI arg > env > default.
- Wired the **active runtime defaults on the `gpuwrf run` path** to it:
  - `io/gen2_accessor.py`: `GEN2_READ_ONLY_ROOT`, `DEFAULT_M6_GEN2_RUN_DIR`
  - `integration/daily_pipeline.py`: `RUN_ROOT`
  - `validation/forecast_vs_obs.py`: `DEFAULT_AEMET_ROOT`
- Back-compat: `export GPUWRF_CANAIRY_ROOT=/mnt/data/canairy_meteo` restores the
  workstation layout (and the Gen2 write-protection) unchanged.

### Public CLI — `gpuwrf run` (GPT plan lane A)
- New `src/gpuwrf/cli.py` (argparse, no new deps) + `src/gpuwrf/__main__.py`
  (`python -m gpuwrf`) + `[project.scripts] gpuwrf = "gpuwrf.cli:main"` in
  pyproject.toml.
- Thin wrapper over `daily_pipeline.execute_daily_pipeline` — no physics/dynamics
  reimplementation, and `runtime/operational_mode.py` untouched.
- `run` validates fail-closed BEFORE any heavy import: input-dir exists, namelist
  is `<input-dir>/namelist.input`, hours>0, then `validate_supported_namelist`
  (v0.7.0 registry). Heavy imports (JAX/pipeline/netCDF4) are deferred so
  `--help` and arg/namelist validation need no GPU.
- `DailyPipelineConfig(run_id=abs(input_dir), run_root=input_dir.parent, …,
  score=False, restart_at_hour=None, repeat=False)` — README gate runs ONE
  forecast + ONE dimension compare, not the full release matrix.
- `--compare-cpu-dir` runs `compare_wrfout_dimensions` (P0 dimension-only) and
  writes `<proof-dir>/dimension_compare.json`; exits 0 only on
  `PIPELINE_GREEN` + dim `PASS`.
- New `tests/test_cli.py`: 10 no-GPU tests (parser, fail-closed paths, dimension
  comparator PASS/FAIL/missing). `tests/test_cli.py + test_namelist_check.py` =
  13 passed.

### README quickstart (GPT plan lane D)
- New top "Quickstart: install and run one forecast" section (clone → venv →
  `jax[cuda13]` with nightly fallback → `pip install -e .` → `GPUWRF_*`/JAX env →
  exact `gpuwrf run --compare-cpu-dir` → expected outputs + PASS criteria).
- Replaced the stale `v0.1.0 (release candidate, tag PENDING)` banner with a
  concise `v0.9.0` heading pointing at `gpuwrf run`.
- **Additive only (116 insertions / 1 deletion).** The proof-table summary,
  Honest limitations, scope, and post-0.9.0 TODO sections are byte-for-byte
  unchanged (no merge collision with the README-owning lane).
- `.gitignore`: ignore `.gpuwrf-cache/` and `/runs/` (Quickstart artifacts).
  `data/` was already ignored, so the README's `data/...` defaults stay out of git.

### Dry smoke proof
- `proofs/v090/readme_runnability_dry_smoke.py` builds a fresh venv, does a real
  `pip install -e .`, and asserts import + console script + `--help`/`run --help`
  + every clean-failure path. `proofs/v090/readme_runnability_dry_smoke.json`:
  **status PASS, 10/10 checks** (Python 3.13.11).

## Commands run
- `python -m gpuwrf --help`, `run --help`, and all fail paths (rc=2, no traceback).
- `pytest tests/test_cli.py tests/test_namelist_check.py -q` → 13 passed.
- `python proofs/v090/readme_runnability_dry_smoke.py` → PASS (fresh venv install).

## Carry-over for the manager (release-time)
1. **Publish the public CPU-WRF d02 1h sample** as a release asset
   (`gpuwrf-canary-d02-1h.tgz`) and pin its URL + sha256 in the README placeholder.
   Build a `proofs/v090/readme_case_manifest.json` per GPT plan lane C.
2. **Full GPU clean-clone gate** (GPT plan lane E): clone the tag, fresh venv,
   real `jax[cuda13]`, run `gpuwrf run` on the sample, assert exit 0 + dim PASS.
3. **Finalize the v0.9.0 status prose + scope matrix** wording (owned by the
   README/v0.6.0 lane). I only changed the banner heading + added one bridging
   sentence; the detailed historical status prose is unchanged.
4. **Optional lane-B widening (P1):** `tier4_rmse_harness.py`,
   `comparator_common.py`, `cpu_wrf_baseline.py`, `d02_replay.py`, and the
   `land_state.py` metadata literal still carry `/mnt/data` defaults. They are
   NOT on the `gpuwrf run` runtime path (validation harnesses / CPU baseline /
   metadata), so I left them to avoid conflict surface; `gpuwrf.config.paths`
   already exposes the helpers (`wrf_l2_root`, `mpirun_path`, `wrf_exe_path`) to
   wire them when desired.

## Unresolved risks
- The dry smoke inherits CPU deps via `--system-site-packages`; it proves the
  gpuwrf package install + entrypoint, not a from-scratch `jax[cuda13]` GPU wheel
  resolution (that is the release-time GPU gate, intentionally out of scope).
- README clone URL uses `github.com/wrf-gpu/wrf_gpu` per the release-protocol
  memory; confirm at tag time.
