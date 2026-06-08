# v0.13 Tier-1 #2 — Powered n=15 TOST rc=2 fix (KI-5)

**Branch:** `worker/opus/v013-tost-rc2-fix` (base `3240966`)
**Constraint honoured:** all diagnosis + fixes on CPU (`JAX_PLATFORMS=cpu`, no GPU context). The GPU owns the GWD-nested gate; the n=15 campaign is a later manager-run GPU step.

---

## rc=2 root cause (TWO conflated sources)

On v0.12.0 the powered-TOST harness lived on a separate branch (`b12c817 v0120-tostprep`) and was **never merged to trunk nor root-caused** — it was deferred. The campaign reported `rc=2` (`L2_D02_BLOCKED`, 0/15 → ABORT). Tracing the two scripts shows the `rc=2` had two distinct sources that were conflated:

### Source A — per-case `run_one_case_v0120.py`
`main()` returns `2` whenever the verdict is `L2_D02_BLOCKED`. That verdict is set when
`execute_daily_pipeline` returns `PIPELINE_BLOCKED` (it catches **every** exception from the
GPU forecast and converts it to `PIPELINE_BLOCKED`), or when post-forecast scoring throws.

* **On v0.12.0 GPU:** the forecast itself raised inside `_run_forecast_sequence` → `PIPELINE_BLOCKED` → `L2_D02_BLOCKED` → `rc=2` per case. The reason was written to the nested pipeline JSON but **not surfaced** in the per-case summary, so the campaign log only showed bare `rc=2`.
* **On CPU (this repro):** the forecast cannot even start — `gpuwrf.contracts.state._gpu_device()` raises `State.zeros requires a GPU device; no JAX GPU backend is visible` → `PIPELINE_BLOCKED` → `rc=2`. This is the deliberate GPU-only guard, **not a bug**; it is exactly why the forecast half is intrinsically a GPU step and only the scoring half is CPU-reproducible.

### Source B — orchestrator `run_powered_tost_n15_v0120.py`
`main()` returned `rc=2` whenever `len(all_tost_scores) < 2` (*"fewer than 2 included cases — cannot compute TOST"*). This conflated **three** different states:
1. **0 cases scored** = genuine total failure (correctly an abort);
2. **1 case scored in `--case` single-case DEBUG mode** = actually a **success** — but it still returned `rc=2`, making it **impossible to drive a single case through for diagnosis**;
3. **1 case scored in a full 15-case campaign** = under-powered.

---

## Fixes (all in owned files: `proofs/v0120/powered_tost_n15/` + `proofs/` + `tests/`)

1. **rc semantics split** (orchestrator): `0 scored` → `rc=2` + `powered_tost_abort.json` (lists every exclusion reason); `1 scored` → `powered_tost_single_case.json`, `rc=0` under `--case`/`--allow-single` (scoring proven) else `rc=2` (under-powered); `≥2` → full TOST aggregate, `rc=0`.
2. **`--allow-single` flag** added.
3. **GPU lock wrapper is now only-if-present** (`GPUWRF_GPU_LOCK_WRAPPER`, default `/tmp/wrf_gpu_run_lowprio.sh`): absent → launch the per-case runner **directly**. Prevents the flock double-wrap deadlock (the 4.6h hang noted in the v0.12.0 release report). The orchestrator must **never** itself be wrapped in a lock wrapper.
4. **Env-overridable data roots** (`GPUWRF_L2_INIT_ROOT`, `GPUWRF_L2_CPU_ROOT`, `GPUWRF_TOST_MERGED_ROOT`, `GPUWRF_TOST_GPU_RUNS_ROOT`).
5. **`blocked_reason` threaded through**: per-case summary carries a top-level `blocked_reason`; `run_gpu_forecast` reads it on `rc≠0` and puts it in the exclusion record + campaign log → a GPU-forecast failure during the campaign is **self-diagnosing**.

---

## Proof (CPU, mandatory) — rc=0

### Scoring path on REAL GPU data
`proofs/v013/tost_scoring_path_cpu_proof.py` → `proofs/v013/tost_scoring_path_cpu_proof.json`, verdict **`SCORING_PATH_RC0_PROVEN`**. It runs the campaign's own `score_one_case` → `paired_score` + `score_cell_level` + `aggregate_tost` on a **retained real GPU wrfout** (v0.9.0-era `case1_L2`, 72 d02 files, all finite) vs CPU-WRF truth `20260530_18z`, on CPU. **Different IC → genuine non-zero deltas** (T2 rmse=1.67 K r=0.60, U10 1.77 m/s r=0.86, V10 2.07 m/s r=0.86; 251 856 cells/field; 5 088 complete AEMET pairs; `aggregate_tost` ran). **Not an equivalence claim** — solely a code-path proof.

### Orchestrator rc semantics (live invocations)
* `--skip-gpu --case <RID>` (stand-in GPU dir = retained CPU truth): **before fix rc=2** (n<2 ABORT despite scoring); **after fix rc=0** + `powered_tost_single_case.json`.
* `--skip-gpu --case BOGUS_NONEXISTENT_CASE`: **rc=2** + `powered_tost_abort.json` (genuine-failure path preserved).

### Regression guard
`tests/test_v013_tost_rc2_fix.py` — **4 passed**, CPU-only (monkeypatched, no GPU/JAX/corpus): single-case `rc=0`, zero-case `rc=2`+abort-json, `--allow-single` `rc=0`, lock-wrapper only-if-present.

---

## GPU-campaign runbook (manager-run, later GPU step)

Prerequisites already verified present: `/mnt/data/canairy_meteo/runs/wrf_l2/<RID>/` (init), `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/<RID>/` (73 d01 + 73 d02 CPU truth each), `/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations/` (AEMET parquet). Merged run root is rebuilt automatically by `prepare_merged_run_root()`.

**Step 0 — single-case GPU smoke (root-cause the remaining forecast blocker FIRST):**
```bash
# Run the orchestrator DIRECTLY (NO outer /tmp/wrf_gpu_run*.sh wrap — double-wrap deadlock).
PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  taskset -c 0-3 \
  python proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py \
    --case 20260429_18z_l2_72h_20260524T204451Z
# rc=0 => the GPU forecast + scoring both work; proceed to the full campaign.
# rc=2 => read the per-case summary's NEW "blocked_reason" field
#         (proofs/v0120/powered_tost_n15/pipeline_proofs/<RID>/l2_d02_validation_summary.json)
#         — it now names the exact GPU forecast failure. Likely candidates:
#         (a) merged-run-root layout vs build_l2_d02_replay_case expectations
#             (1 d02 t=0 snapshot vs >=2 history files),
#         (b) fp64 VRAM ceiling on a 24h d02 run (RRTMG g-point temp; this base
#             already merged the g-point chunking T1#3, which should mitigate it).
```

**Step 1 — full powered n=15 (one GPU job at a time; the lock wrapper serialises per-case):**
```bash
PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  taskset -c 0-3 \
  python proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py --resume
```
Outputs: per-case `case_<RID>.json` (auto-committed), `powered_tost_result.json`, `cell_level_stats.json`, `POWERED_TOST_AND_CELL_STATS.md`, `/tmp/v0120_powered_tost.done`. rc=0 on `n≥2` scored; rc=2 only on `0` scored (abort JSON) or a full campaign collapsed to 1 case.

**Notes / guardrails**
* Stale **empty** GPU dirs from the v0.12.0 attempt remain at `/tmp/v0120_powered_tost_runs/l2_d02_*` (0 files each). They are benign — the skip-run check counts wrfout files (`>=25`), not dir existence — but `rm -rf /tmp/v0120_powered_tost_runs` before a fresh campaign keeps the dry-run plan honest.
* Cores 0–3 only (`taskset -c 0-3`); never touch cores 4–31 / the nightly job.
* fp64 (`JAX_ENABLE_X64=true`) per ADR-029.
* The campaign is **n=15, under-powered** for a 10 % RMSE effect (ADR-029 planning n≈27); report power honestly — the report already emits the power-caveat table.
* This `rc=2` is the **same** `L2_D02_BLOCKED` family seen in the nesting GPU smoke; the `blocked_reason` surfacing helps both lanes.
