# v0.6.0 scan-wire handoff — 11 new schemes into the operational forecast scan

Date: 2026-06-03
Author: Opus 4.8 (1M) integrator
Branch: `worker/opus/v060-scanwire` (base `0d985cc` = v0.6.0 integration)
Environment: JAX CPU only, cores 0-3 (`taskset -c 0-3`). NO GPU run (the GPU
multi-config forecast gate is DEFERRED to the manager).

## Objective

The v0.6.0 integration (0d985cc) merged + State-materialized + dispatcher-routed
the 12 physics schemes, but the operational forecast SCAN threaded only the v0.2.0
suite. This sprint wires the genuinely scan-runnable NEW schemes' State<->scheme
SCAN ADAPTERS into the operational loop (dispatcher-routed, WRF call order,
fail-closed), fixes the parity-report re-emit, runs per-scheme integration smokes,
and prepares the integrated multi-config forecast gate `--run`-ready.

## What is wired (done) — HONEST tracability audit

A close read of each kernel established the REAL tracability (the integration's
`gpu_runnable` flags were aspirational for the PBL schemes). Counts: **6 of the 11
new schemes scan-wired into the GPU scan + KF** (the 12th, v0.2.0 baseline-extension)
= **7 State<->scheme adapters** in the NEW module `coupling/scan_adapters.py`.

### Scan-wired into the GPU scan (jit/vmap-traceable, dispatcher-routed)
- **Microphysics** mp=1 Kessler, mp=6 WSM6, mp=10 Morrison, mp=16 WDM6 — pure `jnp`
  column kernels batched over `(ncol=ny*nx, nlev=nz)`; adapters slice State ->
  kernel -> apply `PhysicsTendency.state_replacements` + mm `accumulator_increments`
  (in-place + `+=`). WDM6 materializes the additive `Nc`/`Nn` leaves (Nn was
  returned in diagnostics; the adapter threads it into State so CCN evolves).
- **Surface layer** sf_sfclay=1 revised-MM5, sf_sfclay=7 Pleim-Xiu — vectorized `jnp`
  `*_run` paths writing the SAME B2 flux handles (`ustar`/`theta_flux`/`qv_flux`/
  `tau_u`/`tau_v`/`rhosfc`/`fltv`) that the v0.2.0 `surface_adapter` writes; drop
  into the surface-layer slot.
- **Cumulus** cu=1 KF — jit-able (`jax.lax.cond`) per-column kernel, `jax.vmap`'d over
  the grid; its `w0avg`/`nca` persistent carry rides a NEW additive
  `OperationalCarry.cumulus_carry` leaf (seeded by `_initial_carry_for_run` BEFORE
  the scan so the `jax.lax.scan` carry pytree is stable). Tendencies applied
  `state += dt*tend`; RAINCV accumulates into `rainc_acc`.

Dispatch: `operational_mode._physics_boundary_step_with_limiter_diagnostics` now
selects per-slot by the namelist's STATIC physics options (compile-time Python
branch, NO per-step `lax.cond`, NO dispatch overhead). Selecting the v0.2.0 defaults
(mp=8 / sf_sfclay=5 / cu=0) is byte-for-byte the validated path.

### Kept FAIL-CLOSED in the scan (loud reject + scheme-specific reason)
`_resolve_operational_suite` rejects these with an actionable message
(`_SCAN_UNWIRED_REASON`):
- **bl_pbl=1 YSU / bl_pbl=7 ACM2** — single-column HOST-NumPy kernels (`_scalar` +
  Python `range` loops over levels, ~30-42 of them); NOT `jax.lax.scan`-traceable on
  a device State. They passed per-column savepoint parity but need a jit/vmap
  rewrite. **CROSS-MODEL FIX** (GPT, since these are GPT-authored lanes).
- **cu=3 Grell-Freitas / cu=6,16 Tiedtke** — faithful CPU-NumPy reference ports
  (`gpu_runnable=False`); selectable CPU-only, excluded from the GPU scan. GPU-batch
  TODO.
- **sf_surface=2 Noah-classic** — `sflx_step` IS jnp-traceable, but it needs a
  WRF-faithful surface-forcing assembler + 4-layer soil prognostic coupler (the
  analogue of the dedicated `coupling/noahmp_surface_hook.py`). Building that inside
  this sprint would ship an unvalidated land coupling, so it is fail-closed and
  flagged as the manager/cross-model land-coupler task. NOTE the dispatcher maps the
  legacy `use_noahmp=False` toggle to land option 2 = the BULK surface path (not the
  Noah-classic LSM); only an EXPLICIT `sf_surface_physics=2` fails closed.

## Import-time / test report re-emit fix (task 2)

The integration flagged "import-time JSON re-emit" for GF / PX-sfclay / Tiedtke. The
actual mechanism (verified empirically — importing the modules does NOT re-emit): the
parity **tests** (`test_grell_freitas_cumulus.py`, `test_v060_sfclay_pleim_xiu.py`,
`test_tiedtke_cumulus_oracle.py`) unconditionally OVERWROTE the committed authoritative
report on every pytest run, then asserted. Fix: gate the write behind an explicit
`GPUWRF_WRITE_PARITY_REPORT=1` env opt-in; by default the test ASSERTS the verdict
WITHOUT clobbering the committed report. Verified: running the 3 tests leaves the 3
committed JSONs byte-identical (md5 unchanged); `GPUWRF_WRITE_PARITY_REPORT=1`
regenerates deterministically. The committed lane reports stay authoritative.

## Per-scheme integration smoke (task 3) — `proofs/v060/scanwire_smoke.py` — ALL PASS

For each scan-wired scheme, drove its State<->scheme adapter (the EXACT function the
scan body calls) for a few steps on a small physically-reasonable C-grid State:
- 4 microphysics: executed, finite/no-NaN, conservation OK (column-water rel change
  0.01-0.18%).
- 2 surface-layer: executed, finite, ustar in physical band (>0).
- KF: finite, `(w0avg, nca)` carry threaded + finite, conservation OK (KF may not
  trigger convection on a near-neutral smoke column — the wiring is validated by a
  finite consistent run + a stable evolving carry).
- Fail-closed boundary: `_resolve_operational_suite` ACCEPTS all 5 wired combos and
  REJECTS all 5 unwired schemes.

## Integrated multi-config forecast gate (task 4) — `proofs/v060/forecast_gate_harness.py`

`--validate` now also reports the OPERATIONAL-SCAN-WIRE status per combo (not just
dispatch/GPU-gate). HONEST finding: of the 3 canonical combos, only **combo_1**
(v0.2.0 + KF) is fully scan-wired. **Canonical combo_2** contains YSU + Noah-classic
and **combo_3** contains ACM2 — host-NumPy / un-coupled schemes NOT yet in the GPU
scan, so combos 2/3 are NOT GPU-scan-runnable as defined. Added 3 SCAN_WIRED_COMBOS
that swap the unwired PBL/land for MYNN/Noah-MP so the newly-wired schemes run
end-to-end now (WSM6/revised-MM5/KF; Morrison/Pleim-Xiu; WDM6 Nc/Nn + KF carry).
`gpu_runnable_now` = 4 combos. `--run` refuses cleanly (needs a GPU backend + corpus
CPU-WRF d02 + Noah-MP init bundle) and points at the wiring + `_build_combo_namelist`.

## What the manager runs (deferred)

1. **GPU multi-config forecast gate** (single GPU job): for each `gpu_runnable_now`
   combo, build the namelist (`_build_combo_namelist`), run `run_forecast_operational`
   over a corpus CPU-WRF d02 case, emit wrfout (QNCLOUD/QNCCN/RAINC), score per-lead
   gridpoint-paired bias/RMSE vs CPU-WRF (continuous_gate pattern), one proof JSON per
   combo under `proofs/v060/forecast_gate/`.
2. **CROSS-MODEL carry-over** (record in the v0.6.0 roadmap, do not silently drop):
   jit/vmap rewrite of YSU(1)/ACM2(7) PBL (GPT-authored) -> Opus or GPT;
   Noah-classic(2) surface coupler; GPU-batch GF(3)/Tiedtke(6,16).

## Files changed
- `src/gpuwrf/coupling/scan_adapters.py` — NEW: 7 State<->scheme scan adapters
  (4 MP + 2 sfclay + KF) + `initial_kf_carry` + the MP/SFCLAY/CU dispatch tables.
- `src/gpuwrf/runtime/operational_mode.py` — dispatcher-routed physics tile
  (microphysics/surface-layer/cumulus slots); `_SCAN_WIRED_OPTIONS` expanded +
  `_SCAN_UNWIRED_REASON`; `_resolve_operational_suite` scheme-specific fail-closed;
  `_initial_carry_for_run` (seeds the KF carry); 6 entry-point carry constructions.
- `src/gpuwrf/runtime/operational_state.py` — additive `OperationalCarry.cumulus_carry`
  leaf (+ `initial_operational_carry` param), same pattern as `noahmp_land`.
- `tests/test_grell_freitas_cumulus.py`, `tests/test_v060_sfclay_pleim_xiu.py`,
  `tests/test_tiedtke_cumulus_oracle.py` — gated report write behind
  `GPUWRF_WRITE_PARITY_REPORT=1`.
- `proofs/v060/forecast_gate_harness.py` — scan-wire status per combo + SCAN_WIRED
  combos + `_build_combo_namelist` + honest `--run` refusal.
- NEW: `proofs/v060/scanwire_smoke.py`, `proofs/v060/gen_scanwire_report.py`,
  `proofs/v060/scanwire_report.json`, `proofs/v060/scanwire_smoke.json`;
  regenerated `proofs/v060/forecast_gate_readiness.json`.

## Commands run
- `python proofs/v060/scanwire_smoke.py` -> all 7 per-scheme smokes + fail-closed PASS.
- `python proofs/v060/forecast_gate_harness.py --validate` -> 4 gpu_runnable_now combos.
- `python proofs/v060/gen_scanwire_report.py` -> overall_pass=True.
- `pytest` dispatch + namelist + contracts + restart + full-carry + the 6 wired-scheme
  parity tests + WSM6/WDM6 parity + ACM2/YSU + the 3 gated report tests -> PASS.

## Proof objects
- `proofs/v060/scanwire_report.json` — which schemes scan-wired + GPU-runnable,
  per-scheme integration-smoke results, conservation, dispatch matrix, forecast-gate
  readiness. overall_pass=True.
- `proofs/v060/scanwire_smoke.json` — per-scheme smoke detail.
- `proofs/v060/forecast_gate_readiness.json` — combos + scan-wire status, gpu_runnable_now.

## Unresolved risks / next decision
1. **Pre-existing failure (NOT introduced here):** `test_noahmp_checkpoint_v2.py::
   test_format_version_is_2` asserts FORMAT_VERSION==2 but the INTEGRATION commit
   (0d985cc) bumped `runtime/checkpoint.py` FORMAT_VERSION 2->3. Stale test from the
   integration lane; out of this sprint's scope. (Also the pre-existing GPU-requiring
   `test_m6*_no_h2d` / `test_m6b_operational_theta_fix` fail identically on CPU at the
   base, per the integration handoff.)
2. **Surface-layer bottom-level inputs:** the new sfclay adapters feed the kernels the
   LOWEST model level fields + use lowest-level pressure as the PSFC proxy. This is a
   reasonable smoke-level coupling; the manager's GPU gate vs CPU-WRF will validate the
   surface-layer fluxes against the real WRF surface_driver inputs (and may want the
   true PSFC from the dycore base-state rather than the lowest-level proxy).
3. **YSU/ACM2/Noah-classic/GF/Tiedtke** are fail-closed, not silently defaulted — a
   run selecting them rejects loudly with the reason. The jit/vmap rewrites + the
   Noah-classic coupler are the recorded cross-model carry-over.
4. Recommend a GPT-5.5 cross-check of (a) the sfclay bottom-level coupling and (b) the
   KF `w0avg`/`nca` carry seeding/threading before the manager schedules the GPU gate.
