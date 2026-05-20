# ADR-006 - Thompson JAX Implementation Mapping

Date: 2026-05-20
Author: M5-S1 worker draft (Codex gpt-5.5)
Status: ACCEPTED worker draft, pending manager closeout integration
Scope: post-hoc implementation record for the M5-S1 Thompson microphysics column source/sink subset.

## Decision

Decision: Implement the M5-S1 Thompson column as a single JAX source/sink kernel with sedimentation removed, driven by a Path-B-strict analytic fixture whose NumPy oracle is a WRF-style tendency ledger independent of the JAX helper sequence. The public API is `step_thompson_column(state, dt, *, debug=False) -> state`, where `state` is the `ThompsonColumnState` pytree carrying `qv`, `qc`, `qr`, `qi`, `qs`, `qg`, `Ni`, `Nr`, `T`, `p`, and `rho`.

This ADR is not a new forward architecture decision. ADR-001 remains the backend decision and ADR-005 remains the first-physics-suite decision. ADR-006 records what was actually mapped in M5-S1 so the next M5 workers can reuse or tighten the pattern.

## WRF source mapping

WRF source mapping: the source of truth is `../wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre`.

- Driver boundary: `mp_gt_driver` is the WRF call boundary at lines 1070-1564. M5-S1 mirrors the per-column state handoff: `T`, `p`, water species, ice/rain number concentrations, and density. The density formula used in the JAX and fixture code is from line 1270.
- Local Thompson source/sink body: `mp_thompson` begins at line 1573. Source/sink term naming and tendency meaning are documented in lines 1717-1727.
- Saturation setup: water and ice saturation ratios, supersaturation flags, diffusion/viscosity, `ocp`, and `lvap` setup are mapped from lines 2040-2064. The exact RSLF/RSIF polynomial helper formulas are mapped from lines 5444-5495.
- Warm-rain and cloud adjustment: cloud-water condensation/evaporation with a fixed three-iteration Newton update is mapped from lines 3456-3556. Berry-Reinhardt autoconversion is mapped from lines 2242-2258. Rain collecting cloud water uses the WRF collection-rate shape from lines 2260-2268; the generated `t_Efrw` lookup table is not a fixture input, so this sprint uses a bounded collection-efficiency proxy and documents that as a remaining WRF-parity gap. Rain evaporation uses the Srivastava-Coen formula from lines 3561-3636.
- Deposition/sublimation: cloud-ice particle diameter and moment terms are mapped from lines 2709-2727. Snow and graupel deposition/sublimation follow the capacitance/ventilation forms from lines 2745-2770, with analytic moment proxies for snow and mp=8 graupel because the WRF lookup tables are generated in `module_mp_thompson` init and are not carried as M5-S1 inputs.
- Freezing/melting: rain freezing uses the non-table cold-branch in lines 2658-2669; instant cloud-ice melting and cloud-water freezing use lines 4007-4028; snow/graupel melting uses the WRF heat/diffusion structure from lines 2845-2889.
- Tendency/update bookkeeping: source/sink tendencies, latent heating structure, and final mass/number constraints are mapped from lines 2967-3260 and 4033-4142. The fixture generator computes through WRF-style tendency variables (`qvten`, `qcten`, `tten`) and process-rate names before final application; the JAX kernel computes the same updates directly in fused helpers.

## Sedimentation status

Sedimentation status: OUT for M5-S1 per ADR-005 and the sprint contract. The skipped WRF sedimentation path starts at the terminal-velocity/substep block around lines 3655-3972, including `vtrk`, `vtik`, `vtsk`, `vtgk`, sediment fluxes, and precipitation accumulation. The fixture is generated with no boundary fluxes and no precipitation fallout fields, so Tier-2 total water conservation is expected to close to fp64 roundoff.

Other skipped-because-out-of-scope sections:

- Aerosol wet scavenging and activation table paths, including optional `nwfa`, `nifa`, `nbca`, and `nc` coupling. The M5-S1 frozen target requires `Ni` and `Nr`, not a full aerosol-aware Thompson-2014 path.
- WRF-generated lookup tables that are not M5-S1 fixture variables: cloud-water collection efficiency `t_Efrw`, rain/cloud freezing tables `tpg_qrfz`/`tpi_qrfz`/`tni_qrfz`/`tnr_qrfz`, and several snow/graupel moment tables. Their formula-shaped call sites are included where practical; exact table parity remains a follow-up WRF-wrapper or table-export task.
- Radar reflectivity and effective-radius diagnostics after `mp_gt_driver` returns. They are not prognostic outputs for the frozen M5-S1 fixture.
- Variable-density hail/graupel volume prognostic `qb` and graupel number `ng`; the frozen M5-S1 prognostics are the ADR-005 set only.

## Kernel fusion

The worker implemented one public jitted step and one hand-stripped sibling:

- `src/gpuwrf/physics/thompson_column.py`: production implementation with `debug` static and source/sink helpers.
- `src/gpuwrf/physics/thompson_column_debug_stripped.py`: physically stripped sibling with debug calls removed.

The JAX implementation deliberately keeps helper functions small and scientifically named: saturation adjustment, ice source/sink, warm rain, finishing floors, and debug checks. XLA fuses the source/sink path into one HLO-derived launch on the analytic fixture. There are no `jnp.array`, `jnp.zeros`, or `jnp.empty` calls in the traced source/sink body.

## HLO Auditability

An auditor can re-derive the HLO identity proof by rerunning `python scripts/m5_run_thompson.py`, which compiles `step_thompson_column(..., debug=False)` and the hand-stripped sibling in `src/gpuwrf/physics/thompson_column_debug_stripped.py`, normalizes volatile HLO spelling, and rewrites `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`. The committed truncated HLO text is for audit readability only; the authoritative diff is re-derived from the committed Python source by that script and must be 0 bytes.

## Tolerances

Hydrometeor output tolerances follow ADR-005: `abs=1e-10`, `rel=1e-8` for `qv`, `qc`, `qr`, `qi`, `qs`, `qg`. Number concentrations use `abs=1e-3`, `rel=1e-6` for `Ni` and `Nr`. Temperature uses `abs=1e-8`, `rel=1e-10` because latent heating is fp64 but the fixture and candidate use separate NumPy/JAX transcendental implementations.

## Gate dry-run

Gate dry-run: `artifacts/m5/thompson_gate_result.json` is GO on the worker run: Tier-1 passed, Tier-2 passed, and HLO-derived launches were within the ADR-001/ADR-005 threshold. Register and local-memory counters remain `null` because Nsight perf counters are blocked on this workstation by the known `ERR_NVGPUCTRPERM` policy; this is recorded in `artifacts/m5/thompson_profile.json`.

## Known limits

This sprint proves the JAX/fixture/tier-validation pipeline and a real branchy Thompson-shaped source/sink column, but it is not full WRF Thompson. The largest scientific gaps are exact WRF lookup-table parity, aerosol-aware activation/scavenging, graupel volume/hail details, and sedimentation. Sedimentation must be a dedicated follow-up sprint before any precipitation fallout claim is made. Exact table parity should be closed either by exporting the WRF tables into the fixture or by a dedicated Fortran-wrapper sprint.
