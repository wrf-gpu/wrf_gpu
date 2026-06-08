# v0.13 Tier1 #2 (KI-5) — Powered-TOST n=15 `wrfbdy_d02` fix

**Status:** UNBLOCKED (CPU setup-verify PASS). The full n=15 GPU campaign is the
manager's next GPU step (command below).

**Branch:** `worker/opus/v013-tost-wrfbdy` (base = `worker/opus/v0120-integration`
tip `4347698d`).

**Owned files (only these changed):**
- `proofs/v0120/powered_tost_n15/run_one_case_v0120.py`
- `proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py`
- `proofs/v013/tost_wrfbdy_fix.py` / `.md` (this proof)

No `src/gpuwrf` physics/dynamics/registry/init code was touched — the fix only
*routes* the per-case setup through existing, validated native-init/live-nest code.

---

## Root cause (confirmed)

The powered-TOST per-case GPU forecast failed `rc=2`:

```
FileNotFoundError: standalone native-init requires wrfbdy_d02 for lateral forcing
(no CPU-WRF wrfout history present in /tmp/v0120_merged_run_root/<RID>):
.../wrfbdy_d02
```

The L2 corpus cases are a **max_dom=2 one-way nest** (`namelist.input`):

```
&domains  max_dom = 2,  parent_grid_ratio = 1, 3
&bdy_control  specified = .true., .false.   nested = .false., .true.
&domains  feedback = 0
```

`real.exe` writes lateral boundary forcing **only for the outermost SPECIFIED
domain (d01)**. So every case dir retains `wrfinput_d01` + `wrfinput_d02` +
`wrfbdy_d01`, but there is **no `wrfbdy_d02`** — a nest never has one; its lateral
boundary comes from the parent d01 at runtime.

The OLD per-case path forced a **d02-only single-domain standalone forecast**
(`build_l2_daily_case` → `build_replay_case(domain="d02")`). With <2
`wrfout_d02` history files in the merged run root, `build_replay_case`
auto-detected the standalone native-init branch and called
`load_wrfbdy_boundary_leaves(run, domain="d02")` →
`wrfbdy_path_for_run(run, "d02")` → `FileNotFoundError(wrfbdy_d02)`. The wrfout
history was also purged from the corpus, so the alternative wrfout-replay path was
unavailable too. Either way the d02-only framing was wrong: a nest's d02 forecast
must be driven by its parent d01, not by a non-existent `wrfbdy_d02`.

This is exactly why the production L3/nested gate runs cleanly with no wrfbdy
error — it uses the **live-nested** driver, which never asks for a nest wrfbdy.

---

## The routing change

Route the per-case GPU forecast through the SAME standalone live-nested driver
the production nested CLI uses (`python -m gpuwrf.cli run --max-dom 2`):

`gpuwrf.integration.nested_pipeline.execute_nested_pipeline(max_dom=2)`

- **d01** runs standalone: IC from `wrfinput_d01`, LBC decoded from `wrfbdy_d01`
  (`build_replay_case(domain="d01", standalone=True)`).
- **d02** is IC-only: `build_replay_case(domain="d02",
  load_lateral_boundaries=False)` takes the *no-disk-LBC* branch, leaving the
  `*_bdy` leaves at their `State.zeros` shapes; the **live parent d01 supplies
  d02's boundary package every parent step**
  (`build_child_boundary_package`). `load_wrfbdy_boundary_leaves` is **never
  called for d02**, so `wrfbdy_d02` is never read.
- The driver writes one `wrfout_<domain>_<valid_time>` per domain per forecast
  hour into the output dir. The `wrfout_d02_*` files are scored exactly as before
  (d02 T2/U10/V10 vs CPU-WRF d02 truth + AEMET). **The d02 scoring config is
  unchanged.**

This reuses the validated runtime (`run_operational_domain_tree`, the same one
behind the v0.11.0 24h d01→d02→d03 nesting proof and the v0.13 GWD/2-way gates).
It needs ONLY real.exe's own outputs — **no CPU-WRF wrfout history**, which
sidesteps the purge entirely.

### Concrete edits

`run_one_case_v0120.py`
- Replaced `execute_daily_pipeline(case_builder=build_l2_daily_case)` (d02-only
  standalone) with `execute_nested_pipeline(NestedPipelineConfig(max_dom=2,
  feedback=False))`. Output dir keeps the historical `l2_d02_<RID>` name so the
  scorer's `wrfout_d02_*` glob is unchanged.
- Scoring/RMSE/bounds/wall now read the `wrfout_d02_*` history the nested driver
  writes (the scored domain), via the existing `write_tier4_rmse` /
  `write_bounds_check` / `write_wall_clock` helpers.
- Added `--setup-only`: a CPU-safe verification (no GPU, no forecast) that
  exercises the LBC-source routing logic that previously failed.

`run_powered_tost_n15_v0120.py`
- `prepare_merged_run_root`: now stages only the live-nested inputs
  (`wrfinput_d01/d02`, `wrfbdy_d01`, `namelist.input`) — drops the purge-fragile
  d01/d02 wrfout-history symlinks (the nested path does not need them; CPU-WRF
  d02 truth is read directly from `L2_CPU_ROOT` by the scorer).
- `validate_gpu_output` + the skip-run gate: expect `FORECAST_HOURS` (24) hourly
  d02 wrfouts, matching what the nested/daily writer emits (leads 1..24, no t=0
  frame). The scorer only scores leads in `(0, 24]`, so this is the complete set.
  (The old `FORECAST_HOURS + 1` = 25 assumed a non-existent t=0 frame.)

---

## CPU setup-verify (mandatory, done)

Note: the device-resident `State.zeros` constructor mandates a JAX GPU backend,
so the full device build cannot run CPU-only. The setup-verify therefore
exercises everything that *decides which LBC source each domain reads from disk*
— the exact logic that raised the `wrfbdy_d02` error — with no device op.

```
JAX_PLATFORMS=cpu PYTHONPATH=src python proofs/v013/tost_wrfbdy_fix.py
# also: ... run_one_case_v0120.py --run-id <RID> --setup-only
```

Result (`proofs/v013/tost_wrfbdy_fix.json`, case
`20260429_18z_l2_72h_20260524T204451Z`): **`verdict: PASS`**, all 8 checks green:

| check | value |
| --- | --- |
| `verdict_setup_ok` | SETUP_OK |
| `is_nest_max_dom_2` | true |
| `one_way_nest` (feedback=0) | true |
| `d01_lbc_wrfbdy_d01_resolves` | true |
| `wrfbdy_d02_absent` | true (so the OLD path's FileNotFoundError is real) |
| `wrfbdy_d02_not_required` | true |
| `both_wrfinput_present` | true |
| `d02_parent_is_d01_ratio_gt_1` | d01, ratio=3 |

Grids loaded: d01 93×59×44, d02 159×66×44. `init_mode =
standalone_native_init_nested`. **No `FileNotFoundError`.**

The existing rc-semantics regression suite still passes:
`tests/test_v013_tost_rc2_fix.py` → 4 passed.

---

## What is NOT proven here (honest gaps)

- The full 72h/24h **GPU forecast** (numerical stability over the run, finite
  output, fp64) was NOT run — a 2-way GPU job is occupying the GPU and the
  campaign is the manager's GPU step. The setup-verify proves the case BUILDS
  cleanly (no wrfbdy error); it does not prove the forecast converges. The
  live-nested d01→d02 path is the SAME one already GPU-validated in the v0.11.0
  nesting proof and the v0.13 GWD/2-way gates, so the build→forecast transition is
  low-risk, but it is unproven for THIS scorer wiring until the GPU run lands.
- TOST equivalence is NOT scored here (needs the GPU wrfouts × n cases).

---

## Manager: run the n=15 GPU campaign (next GPU step)

After the GPU frees (one job at a time; cores 0-3 only):

```bash
cd <this worktree root>
/tmp/wrf_gpu_run_lowprio.sh taskset -c 0-3 \
    env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py --resume
```

- Single-case smoke first (recommended): append `--case
  20260429_18z_l2_72h_20260524T204451Z --allow-single` to drive one case end to
  end (forecast → d02 wrfout → score) and confirm rc=0 before the full sweep.
- `--resume` skips cases that already have a `case_<RID>.json` proof.
- Per-case proofs land in `proofs/v0120/powered_tost_n15/`; the aggregate TOST is
  `powered_tost_result.json` + `cell_level_stats.json`.

**Merge recommendation:** merge `worker/opus/v013-tost-wrfbdy` into the v0.13
integration branch once the GPU smoke (single case rc=0) confirms the forecast +
score path; the CPU setup-verify alone unblocks the campaign but does not close
the equivalence gate.
