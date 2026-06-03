# GPT v0.4.0 MU Continuity Fix Handoff

## Objective

Implement a WRF-faithful, BC-conditional dry-column mass update for specified/nested
real domains, prove it against WRF small-step savepoints, protect periodic idealized
cases, rerun the two-date v0.4.0 standalone forecast gate, and answer replay impact.

## Files Changed

- `src/gpuwrf/dynamics/mu_t_advance.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/dynamics/core/dycore.py`
- `src/gpuwrf/dynamics/core/coupled.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/unit/test_mu_t_lbc_bounds.py`
- `proofs/v040/wrf_advance_mu_t_driver.F90`
- `proofs/v040/mu_continuity_savepoint_parity.py`
- `proofs/v040/mu_continuity_savepoint_parity.json`
- `proofs/v040/idealized_no_regression_report.py`
- `proofs/v040/idealized_no_regression_report.json`
- `proofs/v040/replay_path_impact_check.py`
- `proofs/v040/replay_path_impact_check.json`
- `proofs/v040/run_forecast_gate_24h.py`
- `proofs/v040/forecast_gate_postfix_raw.json`
- `proofs/v040/forecast_gate_postfix_report.json`
- refreshed idealized proof JSON/MD under `proofs/verify_run/row1` and `row2`

## BC-Conditional Design

- `advance_mu_t_wrf` now dispatches on `periodic_x/specified/nested`.
- Default periodic path is preserved for idealized cases.
- Specified/nested path uses WRF small-step bounds from
  `module_small_step_em.F:1048-1063`: y excludes the outer lateral rows; x excludes
  the outer columns unless the domain is periodic in x.
- MU/MUDF/MUTS/MUAVE/WW/theta and theta flux tendency updates are written only on
  the active WRF loop bounds; lateral rows/columns are left for LBC machinery.
- Flags are threaded through `AcousticCoreConfig`, operational namelist routing,
  and the shared `DycoreCoreConfig`/`CoupledCoreConfig` config path.

## Commands Run

- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=true PYTHONPATH=src:. pytest -q tests/unit/test_mu_t_lbc_bounds.py`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=true PYTHONPATH=src:proofs/v040:. python proofs/v040/mu_continuity_savepoint_parity.py --out proofs/v040/mu_continuity_savepoint_parity.json`
- `nvidia-smi`
- `taskset -c 0-3 env VERIFY_RUN_GPU=1 JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src:. bash scripts/verify/idealized_warmbubble.sh`
- `nvidia-smi`
- `taskset -c 0-3 env VERIFY_RUN_GPU=1 JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src:. bash scripts/verify/idealized_straka.sh`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=true PYTHONPATH=src:proofs/v040:. python proofs/v040/idealized_no_regression_report.py`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=true PYTHONPATH=src:proofs/v040:. python proofs/v040/replay_path_impact_check.py`
- `nvidia-smi`
- `taskset -c 0-3 env JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src:proofs/v040:. python proofs/v040/run_forecast_gate_24h.py --hours 24 --case-id 20260429_18z_l2_72h_20260524T204451Z --case-id 20260521_18z_l3_24h_20260522T133443Z --out proofs/v040/forecast_gate_postfix_raw.json --output-root /tmp/v040_forecast_gate_postfix_mu_fix`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=true PYTHONPATH=src:proofs/v040:. python -m py_compile ...`

## Proof Objects

- Savepoint parity: `proofs/v040/mu_continuity_savepoint_parity.json` = PASS.
  - WRF source file verified clean for `dyn_em/module_small_step_em.F`.
  - WRF small-step source SHA256:
    `cabf1a177d50fb0096db79644af20cfe6d75217dbe63ab406a7e29bb54c17634`.
  - fp64 source-formula worst residual: theta interior `1.1368683772161603e-13`
    versus allowed `1.0197689329187736e-08`.
  - linked unmodified WRF object worst residual: MUTS interior `0.0078125`
    versus allowed `0.26541783203125` (WRF object is RWORDSIZE=4).
- Idealized no-regression: `proofs/v040/idealized_no_regression_report.json` = PASS.
  - Periodic `advance_mu_t` path is bit-identical versus diagnosis branch.
  - Warm bubble PASS / ran to completion.
  - Straka density current PASS / ran to completion.
- Replay impact: `proofs/v040/replay_path_impact_check.json` = PASS, impact `unchanged`.
  - Real-field old-vs-new locality changes are confined to the lateral strip.
  - Replay applies CPU-WRF lateral boundary forcing after the dycore step, so the
    v0.1.0/v0.2.0 replay proof path is neutral, not regressed.
- Forecast gate: `proofs/v040/forecast_gate_postfix_report.json` = FAIL for closure.
  - Raw runner verdict: `STABLE_BUT_CORE_FIELD_MISMATCH`.
  - `two_date_bias_collapsed`: `no`.
  - 20260429 uses the scored L2/backfill d01 case because the same-date L3 oracle
    lacks the t0 CPU reference required by the 24h runner.

## Two-Date Before/After

20260429 (`20260429_18z_l2_72h_20260524T204451Z`):

- PSFC mean h2+: `-301.0805 -> -310.7574`; worst abs: `470.6757 -> 480.5485`.
- U10 mean h2+: `1.1943 -> 1.2074`; worst abs: `2.0922 -> 2.1048`.
- T2 mean h2+: `0.0400 -> 0.0339`; worst abs: `0.9257 -> 0.9195`.
- V10 mean h2+: `-0.1100 -> -0.0880`; worst abs: `1.3081 -> 1.2851`.

20260521 (`20260521_18z_l3_24h_20260522T133443Z`):

- PSFC mean h2+: `-116.2162 -> -135.5845`; worst abs: `191.2985 -> 210.9383`.
- U10 mean h2+: `1.2420 -> 1.2422`; worst abs: `1.5916 -> 1.5926`.
- T2 mean h2+: `0.2534 -> 0.2467`; worst abs: `1.2356 -> 1.2289`.
- V10 mean h2+: `0.6865 -> 0.6827`; worst abs: `1.8729 -> 1.8696`.

## Close Decision

v0.4.0 cannot close from this fix. The WRF small-step mass path is now
savepoint-correct for specified/nested loop bounds, but the production 24h
standalone forecast bias did not collapse toward the ADR-029 margins.

## Unresolved Risks / Next Lead

- The prior localization is incomplete for the production standalone forecast:
  correcting `advance_mu_t` specified/nested bounds changes the gate only at noise
  level and slightly worsens PSFC worst bias on both dates.
- Next lead: capture real forecast savepoints across `advance_mu_t`,
  `small_step_finish`, and end-step boundary application at h1/h2, compare against
  WRF, and trace MU/PSFC/U10 budgets through PGF and surface-drag coupling.
