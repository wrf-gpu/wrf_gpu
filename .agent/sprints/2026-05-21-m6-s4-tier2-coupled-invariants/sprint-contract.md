# Sprint Contract — M6-S4 Tier-2 Coupled Invariants + F-S4-1/2/3 Binding Prereqs

**Sprint ID**: `2026-05-21-m6-s4-tier2-coupled-invariants`
**Created**: 2026-05-21 16:00
**Status**: ACTIVE — dispatching now
**Trigger**: M6-S3 Opus reviewer §M6-S4 Binding mandates 3 F-S4-* prereqs; ADR-014 authorizes State extension + Gen2 re-pin

## Objective

Implement Tier-2 coupled conservation diagnostics for the M6-S2 + M6-S3 forecast driver, with three binding prereqs from M6-S3 reviewer:
1. F-S4-1: extend State pytree with prescribed land leaves (per ADR-014)
2. F-S4-2: re-pin Gen2 reference run to one with hourly d02 history
3. F-S4-3: measure Tier-2 conservation PRE-sanitize_state

## Acceptance

- **AC1 F-S4-1 State extension**: `src/gpuwrf/contracts/state.py` + `precision.py` extended with `xland, lakemask, mavail, roughness_m` (and optionally `pblh`) per ADR-014. `tests/test_m6_state_extension.py` extended for new leaves. `build_initial_state` loads from prescribed land state.
- **AC2 F-S4-2 Gen2 re-pin**: M6-S2a accessor + driver + manifest re-pinned to `20260520_18z_l3_24h_20260521T045821Z` (25 hourly d02 wrfout). New `artifacts/m6/gen2_manifest_v2.json` with SHA-pinned new files.
- **AC3 F-S4-3 PRE-sanitize_state Tier-2**: instrument `coupling/driver.py` with pre-sanitize tap OR sanitize-OFF parallel scan. Tier-2 conservation diagnostic measures budget closure on pre-sanitize state.
- **AC4 Tier-2 invariant kernels**: `src/gpuwrf/validation/tier2_coupled.py` with:
  - Mass conservation: `d/dt(mu) = -∇·(mu·u)` budget per cell
  - Dry mass conservation: `d/dt(rho_d) = 0` (no source/sink)
  - Water budget: `d/dt(rho·qv + rho·qc + ...) = source - sink - precip_outflow`
  - TKE positivity + bounds: `qke ≥ 0`, no NaN
  - Hydrometeor positivity: `qv, qc, qr, qi, qs, qg ≥ 0`
  - Boundary-flux closure (relaxation zone tendency = WRF-style `lbc_fcx_gcx` contribution)
- **AC5 Per-quantity per-cell residual artifact**: `artifacts/m6/tier2_coupled_invariants.json` per-leaf per-step max-abs residual + budget closure ratio.
- **AC6 Strict closure thresholds**:
  - Dry mass: `max_abs_residual < 1e-10 kg/m²` 
  - Total water: `< 1e-8 kg/kg` over 1h forecast
  - Hydrometeor positivity: 0 violations
  - TKE: 0 negative values
  - NaN/Inf: 0 in any prognostic
- **AC7 Sanitize-OFF parallel measurement**: forecast finite for ≥1h with sanitize OFF, OR pre-sanitize tap captures conservation cleanly before sanitize fires.
- **AC8 Schema + ADR**: `Tier2CoupledInvariants` schema in `proof_schemas.py`. ADR-014 ratified to ACCEPTED status (manager closeout step).

## Files Worker May Modify

- `src/gpuwrf/contracts/state.py` (per ADR-014; F-S4-1)
- `src/gpuwrf/contracts/precision.py` (per ADR-014)
- `src/gpuwrf/coupling/driver.py` (pre-sanitize tap or sanitize-OFF scan for F-S4-3)
- `src/gpuwrf/coupling/physics_couplers.py` (only for State extension consumption)
- `src/gpuwrf/validation/tier2_coupled.py` (NEW)
- `src/gpuwrf/io/{gen2_accessor.py, land_state.py}` (F-S4-2 re-pin; populate new State leaves)
- `src/gpuwrf/io/proof_schemas.py` (add `Tier2CoupledInvariants`)
- `scripts/m6_run_tier2_coupled.py` (NEW), `m6_gate_tier2_coupled.py` (NEW)
- `tests/test_m6_state_extension.py` (extend for new leaves), `tests/test_m6_tier2_*` (NEW)
- `artifacts/m6/gen2_manifest_v2.json` (NEW), `tier2_coupled_invariants.json` (NEW)
- `.agent/decisions/ADR-014-m6-state-extension-prescribed-land.md` (ratify status; non-bottleneck)
- Worker report

## Files Worker Must NOT Modify

- `src/gpuwrf/physics/**` (all CLOSED including M5-S3.zzzz SW PARITY)
- `src/gpuwrf/dynamics/**` (M4 frozen)
- Sister sprint M5-S3.zzzzz LW broadband (in flight)
- Other ADRs (modulo ADR-010 cross-reference update if needed)
- `/mnt/data/canairy_meteo/**` (READ-ONLY)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (mandatory)
- Wall-time: **20-32h** (was 16-24h, +6-8h for F-S4-1 ADR + re-pin)
- Worktree: `/tmp/wrf_gpu2_m6s4` (NEW)
- Branch: `worker/codex/m6-s4-tier2-coupled-invariants`

## HARD RULES

1. ADR-014 binding: State extension must follow exact leaf names + shapes + dtypes
2. Gen2 re-pin to `20260520_18z_l3_24h_20260521T045821Z` (confirm path on disk; if not exact, use closest hourly-history L3 run)
3. PRE-sanitize_state Tier-2 measurement (F-S4-3) is the binding test, not post-sanitize
4. NO `min(raw, cap)` fudge
5. Cite WRF source for any conservation formula (`module_em.F` for mass; `module_diagnostics_driver.F` for water budget)
6. `/exit` slash-command; watchdog + multi-Enter handle auto-notify

## End-goal context

M6-S4 closure unblocks M6-S5 (ADR-007 4× verdict can use re-pinned Gen2 + measure end-to-end wall) + M6-S6 (Tier-3 TSC1.0) + M6-S7 (Tier-4 probtest) + M6-S8 (operational comparison). Critical-path for M6 GREEN → M7 dispatch.
