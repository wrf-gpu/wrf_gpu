# v0.6.0 Integration Handoff — 12 physics schemes + State materialization + dispatcher

Date: 2026-06-03
Author: Opus 4.8 (1M) integrator
Branch: `worker/opus/v060-integration` (base S0 `0ab2c7b`)
Environment: JAX CPU only, cores 0-3 (`taskset -c 0-3`). NO GPU run (the integrated
forecast gate's GPU run is DEFERRED to the manager).

## Objective

Assemble the 12 common-menu physics schemes (each already savepoint-parity-passed
in its own file-disjoint lane) into the operational model: merge the lanes,
materialize the deferred S0 State leaves, wire the scheme dispatcher into the
operational step (fail-closed), confirm no regression, and wire-but-not-run the
integrated multi-config forecast gate.

## What is wired (done)

### STEP 1 — Merge (clean)
All 12 lanes merged onto `worker/opus/v060-integration`:
kessler, wsm6, morrison, wdm6, ysu, acm2, sfclayrev1, pxsfclay, kf, grell-freitas (gf2),
tiedtke, noah-classic. The lanes were genuinely file-disjoint:
- **No lane modified any frozen S0 interface** (`physics_registry.py`,
  `physics_interfaces.py`, `namelist_check.py`, `wrfout_writer.py`) — verified by
  `git diff S0..HEAD` on each (empty). The freeze held.
- The only merge conflicts were shared per-scheme oracle BUILD-SCRATCH:
  `proofs/v060/oracle/{.gitignore,build_and_run.sh,dump_to_json.py}` and
  `proofs/v060/savepoints*/wrf_source_checksums.txt`. Resolved WRF-faithfully:
  `.gitignore` unioned to a superset; each lane's generic `build_and_run.sh` /
  `dump_to_json.py` renamed to `<scheme>_*` so every oracle reproduction script is
  preserved; checksum files unioned (sorted-unique). **No scheme code, no test, no
  committed savepoint JSON conflicted.**

Module-name note (informational, not a defect): merged module filenames differ from
the S0 spec text in three cases — Noah-classic is `lsm_noah_classic.py` (not
`noah_classic.py`), revised-MM5 sfclay is `sfclay_revised_mm5.py` (not
`surface_layer_sfclayrev.py`), Pleim-Xiu is `sfclay_pleim_xiu.py` (not
`surface_layer_px.py`). The dispatcher records the actual names; all 22 routed
entrypoints were verified to exist on their modules.

### STEP 2 — State-leaf materialization
Appended the 3 deferred S0 additive leaves AT THE END of `State.__slots__` /
`STATE_FIELD_ORDER` / `PRECISION_MATRIX` / `_state_field_shapes` (53 -> 56 leaves):
- `Nc` (QNCLOUD, FP32-gated, mass_3d) — WDM6 cloud-droplet number
- `Nn` (QNCCN, FP32-gated, mass_3d) — WDM6 CCN number
- `rainc_acc` (RAINC, FP64, surface_2d) — cumulus precip accumulator

Byte/pytree-order compatibility preserved: existing leaves keep their positions;
the 3 new `__init__` params default to `None` -> zeros, so a pre-v0.6.0 `cls(*children)`
flatten with the old leaf count still reconstructs (additive leaves zero-backfill).

- **Restart**: bumped checkpoint `FORMAT_VERSION` 2 -> 3 (`SUPPORTED = (1,2,3)`).
  The reader now accepts a recorded order that is a PREFIX of the current schema whose
  only missing tail leaves are the additive ones, and zero-backfills them (cold-start).
  Any non-prefix / non-additive divergence still fails closed. Verified: a hand-built
  v2 checkpoint (old leaf order, no Nc/Nn/rainc_acc) reads back with the additive leaves
  zeroed and existing leaves preserved.
- **wrfout**: NO change needed. The S0 writer already self-gates QNCLOUD/QNCCN from
  `Nc`/`Nn` and RAINC from `rainc_acc` (`_optional_field_array` source lists); now that
  the leaves exist they are emitted automatically.
- **cugd_\* correction (S0)**: applied. No inert `cugd_*` State carry is threaded for
  cu=3; GF + Tiedtke route via the combined `R*CUTEN` + `RAINCV`/`PRATEC` family + shallow
  diagnostics (registry `CUMULUS_TENDENCY_MEMBERS`). Noah-classic's 4-layer land carry
  (`flx4/fvb/fbur/fgsn/smcrel/xlaidyn`, num_soil_layers=4) stays in the `PhysicsCarry`
  sibling tree — it does NOT reinterpret the 2-D `State.soil_moisture`.

### STEP 3 — Dispatcher + namelist matrix
New module `src/gpuwrf/coupling/physics_dispatch.py` is the single fail-closed
`(family, option) -> scheme` router. It records per-scheme: owner module, entrypoint,
calling convention, GPU-runnability, and carry/tendency/accumulator members from the
frozen registry. `resolve_physics_suite(namelist)` returns a validated `PhysicsSuite`
with `gpu_gate_ready` (True only when every non-disabled scheme is GPU-runnable).

`OperationalNamelist` gained 5 static-aux scheme-selection fields
(`mp_physics`/`bl_pbl_physics`/`sf_sfclay_physics`/`cu_physics`/`sf_surface_physics`),
threaded through `tree_flatten`/`tree_unflatten`, defaulting to the v0.2.0 validated
baseline (Thompson/MYNN/MYNN-sfclay/Noah-MP, no cumulus). All 6 public operational
forecast entry points now call `_resolve_operational_suite(namelist)` (next to the
existing `rk_order==3` guard) which fail-closes on out-of-matrix options AND on
parity-passed schemes whose scan adapter is not yet threaded (see below).
`namelist_check.py` already enforced the full S0 accept-matrix and fails closed; the
dispatcher is defense-in-depth.

## What the manager runs (deferred / honest gaps)

### The integrated forecast gate is WIRED-READY, NOT RUN
`proofs/v060/forecast_gate_harness.py` defines the 3 canonical GPU-runnable combos
and validates them on CPU (`--validate`, all 3 namelist-accepted + dispatch-resolved +
GPU-gate-ready). The GPU forecast + CPU-WRF scoring is gated behind `--run`
(MANAGER-only; refuses without the per-combo scan adapters + a GPU backend):
- **combo_1**: Thompson(8)/MYNN(5)/MYNN-sfclay(5)/Noah-MP(4)/KF(1) + RRTMG (v0.2.0 + KF)
- **combo_2**: WSM6(6)/YSU(1)/revised-MM5(1)/Noah-classic(2)/KF(1) + RRTMG
- **combo_3**: Morrison(10)/ACM2(7)/Pleim-Xiu(7)/Noah-MP(4)/no-cumulus + RRTMG

GF (cu=3) and Tiedtke (cu=6/16) are intentionally NOT in any canonical combo.

### IMPORTANT honest scope boundary
The operational SCAN currently threads only the v0.2.0 wired suite
(Thompson/MYNN/MYNN-sfclay/Noah-MP-or-bulk, no cumulus). The other 11 schemes passed
per-scheme WRF-savepoint parity at the KERNEL level, but their State<->scheme SCAN
adapters (column-view in / tendency out, threaded into `_physics_boundary_step`) are
the manager-scheduled forecast-gate work. To stay honest, selecting a non-wired scheme
in the production scan **fails closed loudly** (`UnsupportedSchemeSelection`) rather
than silently running the default scheme. The dispatcher already maps each option to
the correct proven entrypoint + calling convention, so threading the per-scheme
adapter is the remaining mechanical step the manager performs before `--run`.

### GF/Tiedtke GPU-batching status
Both are faithful CPU-NumPy reference ports (not jit/vmap'd). Selectable through the
dispatcher and savepoint-parity-gated, but flagged `gpu_runnable=False`. They are
excluded from the GPU forecast gate. GPU-batching (jit/vmap) is a tracked TODO.

## STEP 4 — Regression (all PASS on CPU)
- 12 per-scheme savepoint-parity tests: **135 assertions PASS** post-merge AND
  post-materialization (no regression from the State-leaf change).
- Conservation budget: PASS. Restart full-carry: PASS. Checkpoint roundtrip: PASS
  (the 53->56 leaf-count guard assertion was updated — it still bitwise-verifies all
  leaves incl. the 3 new ones; this is the correct schema-count update, not masking).
- Restart v2->v3 backward-compat read: verified (additive leaves zero-backfilled).
- New `tests/test_v060_physics_dispatch.py`: 11 PASS. `test_namelist_check.py`: 3 PASS.
  `tests/contracts/`: 6 PASS.
- All 12 per-scheme parity reports verdict PASS / overall_pass=True.

Pre-existing non-regressions: GPU-requiring source-introspection tests
(`test_m6*_no_h2d`, `test_m6b_operational_theta_fix`) fail identically on the S0 base
on CPU (`State.zeros requires a GPU device`); confirmed by checking out `0ab2c7b`.
Not introduced by this integration.

## Files changed
- Merge (12 lanes): all per-scheme `src/gpuwrf/physics/*.py`, tests, proofs (file-disjoint).
- `src/gpuwrf/contracts/state.py` — append Nc/Nn/rainc_acc (slots, __init__, shapes).
- `src/gpuwrf/contracts/precision.py` — STATE_FIELD_ORDER + PRECISION_MATRIX additive leaves.
- `src/gpuwrf/runtime/checkpoint.py` — format v3 + prefix-compatible backward-compat read.
- `src/gpuwrf/runtime/operational_mode.py` — 5 scheme-selection namelist fields + fail-closed `_resolve_operational_suite`.
- `src/gpuwrf/coupling/physics_dispatch.py` — NEW dispatcher (selection authority).
- `tests/test_v060_physics_dispatch.py` — NEW dispatch tests.
- `tests/test_m7_restart_checkpoint_roundtrip.py` — leaf-count guard 53->56.
- `proofs/v060/forecast_gate_harness.py`, `gen_integration_report.py`, `integration_report.json`,
  `forecast_gate_readiness.json` — NEW.

## Commands run
- `git merge` x12 (conflict resolution: gitignore union, per-scheme oracle script rename, checksum union).
- `pytest` (12 parity + conservation + restart + dispatch + namelist + contracts) = 164 PASS.
- `python proofs/v060/forecast_gate_harness.py --validate` = all 3 combos GPU-gate-ready.
- `python proofs/v060/gen_integration_report.py` = ok=True.

## Proof objects
- `proofs/v060/integration_report.json` — merge/materialization/dispatcher/regression/gate.
- `proofs/v060/forecast_gate_readiness.json` — 3 canonical combos, READY_NOT_RUN.

## Finding: import-time proof regeneration (3 lanes)
Importing `cumulus_grell_freitas`, `sfclay_pleim_xiu`, or `cumulus_tiedtke` (which the
dispatcher imports) RE-EMITS that scheme's `*_savepoint_parity_report.json` at import
time — only the worktree path / timestamp / elapsed change; the checksums and verdicts
are identical. The committed (lane-original) reports are left in place; the regenerated
copies are NOT committed. A scheme module should not write a proof file on import — flag
for a small lane-owned fix (move the report emit behind `if __name__ == "__main__"` or a
test fixture). Verdicts are unaffected.

## Unresolved risks / next decision
1. **Per-scheme SCAN adapters** (column-view <-> State tendency, threaded into the
   operational scan) for the 11 non-baseline schemes are the manager's forecast-gate
   work. The dispatcher gives the option->entrypoint->convention map; the scan currently
   fail-closes on them rather than running them.
2. **GF/Tiedtke GPU batching** (jit/vmap) is a TODO; excluded from the GPU gate for now.
3. **Forecast-gate GPU run** is manager-scheduled (single GPU job): build each combo's
   namelist, thread its scan adapters, run vs corpus CPU-WRF d02, score per-lead
   gridpoint-paired bias/RMSE (continuous_gate pattern), one proof JSON per combo.
4. Recommend a GPT-5.5 cross-check of the dispatcher fail-closed boundary + the restart
   v3 backward-compat read before the manager threads the scan adapters.
