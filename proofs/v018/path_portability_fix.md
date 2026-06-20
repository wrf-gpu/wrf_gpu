# v0.18 Path-Portability Fix — clean public checkout runs the nested Canary case via env-overrides

**Branch:** `worker/gpt/v018-integration`
**Goal (release gate):** a clean PUBLIC checkout must RUN the nested Canary case
end-to-end via environment overrides with ZERO PII. The v0.18 sanitize (PR #62)
and the first path-fix were both incomplete; this completes both and PROVES it by
running the clean-env Canary case to GREEN.

## Required environment variables (the only ones a user must set)

A naive user on their own machine sets exactly two paths and the case runs:

| Variable | Meaning | Value used in the proof run |
| --- | --- | --- |
| `GPUWRF_WRF_ROOT` | Root of a pristine WRF v4 source/run tree (RRTMG `.F` sources, `run/` tables, `run/CCN_ACTIVATE.BIN`) | `<USER_HOME>/src/wrf_pristine/WRF` |
| `GPUWRF_CANAIRY_ROOT` | Canary Gen2 / CPU-WRF corpus root (run inputs + backfill reference) | `<DATA_ROOT>/canairy_meteo` |

Optional (performance / scratch, all defaulted): `JAX_ENABLE_COMPILATION_CACHE`,
`JAX_COMPILATION_CACHE_DIR`, `GPUWRF_TMPDIR`. `GPUWRF_WRF_ROOT` pointing at the
dev's real WRF install is *correct* — that simulates a user's WRF install via the
env; the SOURCE TREE contains no hardcoded home path.

## Single source of truth

Every functional runtime path now resolves through `src/gpuwrf/config/paths.py`
env-overrides. There is no hardcoded `<USER_HOME>/<name>` on any runtime read path.

## Functional path constants routed through `config.paths` (this fix)

| File | Was | Now |
| --- | --- | --- |
| `scripts/extract_rrtmg_tables.py` (`LW_SOURCE`/`SW_SOURCE`/DATA via `WRF_ROOT`) | `_resolve_wrf_root()` honored only `$GPUWRF_WRF_SRC`, then fell back to a non-env `<USER_HOME>/src/wrf_pristine/WRF` → `FileNotFoundError` on a clean-env run that sets `GPUWRF_WRF_ROOT` | First candidate is now `config.paths.wrf_root()` (`GPUWRF_WRF_ROOT`); legacy `$GPUWRF_WRF_SRC` + historical artifacts path kept as later fallbacks; final fallback names `wrf_root()`. This is the runtime blocker (A): `gpuwrf.physics.rrtmg_lw` loads this module dynamically and reads `extractor.LW_SOURCE`. |
| `scripts/build_thompson_aero_tables.py` (`CCN_BIN`) | `Path("<USER_HOME>/.../run/CCN_ACTIVATE.BIN")` (hardcoded + non-env) | `_ccn_bin_path()` → `config.paths.wrf_run_dir() / "CCN_ACTIVATE.BIN"` (`GPUWRF_WRF_ROOT`). (B) |
| `scripts/m6_full_domain_batching.py` (`TMP_ROOT`, `DEFAULT_FORECAST_OUTPUT_DIR`) | `Path(os.environ.get("GPUWRF_TMPDIR", "<USER_HOME>/.cache/gpuwrf_tmp"))` + `.mkdir()` AT IMPORT → crashes on a clean checkout (cannot `mkdir <USER_HOME>`) | `config.paths.tmp_root()` (`GPUWRF_TMPDIR`, default `~/.cache/gpuwrf`); output dir derived from `TMP_ROOT`. Surfaced by pytest collection. |
| `scripts/m6_run_coupled_forecast.py` (`TMP_ROOT`, `--output-dir` default) | same import-time `<USER_HOME>/.cache` mkdir + `--output-dir` default `<USER_HOME>/.cache/...` | `config.paths.tmp_root()`; `--output-dir` default derived from `TMP_ROOT`. |

## PII scrubbed — comment / docstring / citation strings (dev home path → `<USER_HOME>`)

These are file:line provenance citations (not runtime reads), so per the rule they
are PII-scrubbed in place to `<USER_HOME>`, not env-routed:

- `src/` (6): `physics/lsm_pleim_xiu.py`, `physics/lsm_ruc.py`, `physics/lsm_ssib.py`,
  `coupling/slab_surface_hook.py`, `coupling/pleim_xiu_surface_hook.py`,
  `io/lsm_static_extract.py`.
- `scripts/extract_rrtmg_tables.py` docstring (removed the `<USER_HOME>` literal candidate as part of the env-routing).
- `proofs/v018/` (49 files): oracle build scripts (`build_and_run.sh`, `build_*_oracle.sh`,
  `run_parallel.sh`), JSON oracle/source manifests (`*_reference_oracle.json`,
  `cu_family_status.json`, `mp_endpoint_manifest.json`, `mp_oracles/**/oracle_summary.json`,
  `camuw_pbl9_endpoint_classification.json`, …), checksum/manifest `.txt`, and critic
  `.md` reports — all provenance citing where the pristine WRF source lived.
  (the dev `miniconda3` home path → `<USER_HOME>/miniconda3` covered by the same scrub.)

## Re-audit (final)

```
grep -rn "<dev-home-path>" src/ docs/ scripts/ README.md KNOWN_ISSUES.md proofs/v018  ->  0 matches
```

- 0 dev-home-path PII anywhere in the public scope.
- 0 FUNCTIONAL `<USER_HOME>` Path constants that mkdir/read at import (the m6
  import-time breakers are fixed; the only remaining `<USER_HOME>` strings in `src/`
  are docstrings and source-citation strings — e.g. `forcing_decode.py` `source_refs`,
  `wrf_scheme_catalog.WRF_README_SOURCE`, `gen2_accessor` `source_citations` — which
  the rule explicitly permits as citations).
- `import gpuwrf` OK; `pytest --collect-only` collects 2183 tests with no
  import-time path failure; fast CPU subset (`-k "paths or config or scheme_catalog
  or namelist"`) = 123 passed, 7 skipped.

### Residual (non-blocking, PII-clean, not on the run/import path)

Legacy alternate-backend / Fortran-harness dev tooling still carries
`<USER_HOME>/...wrf_gpu_src/...` constants that reference the now-defunct Gen2
`wrf_gpu_src` artifacts tree (does not exist even on the dev box). These are
**already PII-clean (`<USER_HOME>`)**, are only touched inside `main()`/functions when
the script is invoked directly (they do NOT break `import gpuwrf` or pytest
collection), and are NOT on the Canary run path:
`scripts/{precision_bench,pubtest_common,extract_thompson_tables,m6b0r_relinked_extract}.py`,
`scripts/diag/d03_pressure_knockout.py`, and the `if [[ -f <USER_HOME>/... ]]; then source`
guarded build harnesses (`src/gpuwrf/backends/{cuda_tile,kokkos}/build.sh`,
`scripts/m2_run_*.sh`, `scripts/wrf_*_harness_build.sh`). No config.paths helper
maps to that abandoned tree; converting dead tooling is out of scope for the run gate.

## Clean-env Canary run — GREEN (the proof)

Command (exactly the gate command, only `GPUWRF_WRF_ROOT` + `GPUWRF_CANAIRY_ROOT`
as the runtime path env):

```
GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF \
GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo \
JAX_ENABLE_COMPILATION_CACHE=true \
JAX_COMPILATION_CACHE_DIR=<DATA_ROOT>/gpuwrf_jax_cache \
PYTHONPATH=src scripts/with_gpu_lock.sh --label opus-pathfix2 -- \
  taskset -c 0-3 python3 proofs/v0120/powered_tost_n15/run_one_case_v0120.py \
  --run-id 20260501_18z_l2_72h_20260519T173026Z --hours 1 \
  --output-root /tmp/pathfix2_verify
```

Result: **`"verdict": "L2_D02_GREEN"`** on the first attempt (no FileNotFound, no
`<USER_HOME>` path error), `blocked_reason: null`.

```
statuses: { bounds: PASS, pipeline: PIPELINE_GREEN, rmse: PASS, wall_clock: PASS }
bounds:   all_numeric_fields_finite = true, failures = []
          theta_min/max = 290.0 / 494.7 K ; |u|max=33.7 |v|max=9.2 |w|max=1.18 m/s
tier4 RMSE vs Gen2 backfill (d02, valid 2026-05-01T19:00Z):
  T2  rmse=0.2023 K   max_abs=2.2483  (thr 3.0)  PASS
  U10 rmse=0.1329 m/s max_abs=1.1630  (thr 7.5)  PASS
  V10 rmse=0.1310 m/s max_abs=1.3516  (thr 7.5)  PASS
  failures: []
```

Finite-GREEN + tolerance-GREEN. Reference WRF source resolved via
`GPUWRF_WRF_ROOT` (RRTMG `module_ra_rrtmg_lw.F` + `run/` tables); run inputs +
backfill reference via `GPUWRF_CANAIRY_ROOT`. No tracked source carries a
hardcoded home path.
