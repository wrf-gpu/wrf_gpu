# ADR-006 - Thompson JAX Implementation Mapping

Date: 2026-05-20
Author: M5-S1 worker draft (Codex gpt-5.5)
Status: ACCEPTED worker draft, pending manager closeout integration
Scope: post-hoc implementation record for the M5-S1 Thompson microphysics column source/sink subset.

## Decision

Decision: keep the JAX Thompson source/sink candidate implementation, with attempt-4 WRF-order sequencing and cloud-ice-number fixes, and use a compiled WRF Fortran harness as the Tier-1 oracle. The public JAX API remains `step_thompson_column(state, dt, *, debug=False) -> state`, where `state` is the `ThompsonColumnState` pytree carrying `qv`, `qc`, `qr`, `qi`, `qs`, `qg`, `Ni`, `Nr`, `T`, `p`, and `rho`.

This ADR is not a forward architecture decision. ADR-001 remains the backend decision and ADR-005 remains the first-physics-suite decision. ADR-006 records the implementation and oracle mapping actually used by this sprint.

## WRF source mapping

WRF source mapping: the source of truth is `../wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre`.

The Fortran harness in `scripts/wrf_thompson_harness.f90` uses a locally compiled WRF v4.7.1 `module_mp_thompson` object. It calls `thompson_init` first, because lookup-table allocation and initialization live in lines 604-1064 and must happen before the driver path. It then calls `mp_gt_driver`, whose call boundary is lines 1070-1564. The harness passes the frozen M5-S1 prognostic fields and writes outputs in `ES24.16E3` text format before the Python fixture packager creates `fixtures/samples/analytic-thompson-column-v1.npz`.

Build dependency tree:

- `data/scratch/module_mp_thompson_nosed.o` compiled from `../wrf_gpu/.../module_mp_thompson.F.pre` with the attempt-4 no-sedimentation patch
- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_mp_radar.o`
- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/share/module_model_constants.o`
- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/frame/module_wrf_error.o`
- harness-local stubs for namelist and single-rank DM helpers used by Thompson table initialization

`gfortran` is not installed on this workstation and the WRF `.mod` files are NVHPC-built, so `scripts/wrf_thompson_harness_build.sh` uses `nvfortran` for ABI compatibility. The compiled harness lives only under `data/scratch/` and is referenced by SHA-256 in the fixture manifest's `files` list. The manifest keeps `source: wrf-derived` because the M1 manifest schema only permits `analytic` or `wrf-derived`; the more specific `wrf-thompson-via-fortran-harness` marker is recorded in `source_commit`.

The JAX candidate maps these WRF sections:

- driver prep and density formula: lines 1070-1274
- saturation helpers: lines 5444-5495
- cloud condensation adjustment: lines 3456-3556
- Berry-Reinhardt warm-rain autoconversion: lines 2242-2258
- rain-cloud-water collection shape: lines 2260-2268
- Srivastava-Coen rain evaporation: lines 3561-3636
- cloud-ice/snow/graupel deposition and sublimation: lines 2709-2770
- rain freezing and snow/graupel melting: lines 2658-2669 and 2845-2889
- tendency bookkeeping and final mass/number constraints: lines 2967-3260 and 4033-4142

Attempt 4 changed the JAX process order to follow WRF checkpoints: source/sink staging and conservation (2917-3247), working-state update before condensation (3250-3273), cloud condensation/evaporation (3456-3558), rain evaporation (3561-3638), instant cloud-ice melt/cloud-water freeze (4005-4031), and final write/balance (4033-4142). It also changed cloud-ice number handling to match lines 2719-2727: `pni_ide` is active only in the sublimation branch, while positive deposition partitions mass without creating new `Ni`.

## Sedimentation status

Sedimentation status: OUT for M5-S1 per ADR-005. The WRF sedimentation and precipitation accumulation path starts around lines 3655-4003. Attempt 4 replaced the attempt-3 `dz=1.0e30` workaround with a local source patch: `scripts/wrf_thompson_harness_build.sh` copies `module_mp_thompson.F.pre` to `data/scratch/module_mp_thompson_nosed.F90` and inserts a no-sedimentation patch immediately before the sedimentation flux loops. The patch zeroes `vtrk`, `vtnrk`, `vtik`, `vtnik`, `vtsk`, `vtgk`, `vtngk`, `vtck`, and `vtnck`, so the sedimentation loops at lines 3854-4003 execute with zero terminal velocities and no fallout flux. The harness now passes physical `dz=1000 m`.

The JAX kernel itself contains no sedimentation, terminal-velocity, substepping, or precipitation-accumulation code. Aerosol activation/scavenging, exact generated lookup-table parity, radar/effective-radius diagnostics, and hail/graupel volume state are also outside the M5-S1 candidate.

## Kernel fusion

The JAX implementation uses one public `@jax.jit` with `dt` and `debug` static. The process-order refactor is still a single fused JAX call; it only reorders existing source/sink helpers and splits warm-rain collection from rain evaporation so the checkpoints match WRF source order. `src/gpuwrf/physics/thompson_column_debug_stripped.py` is a hand-stripped sibling with debug calls physically omitted. `python scripts/m5_run_thompson.py` recompiles both paths and rewrites `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff`; the diff is 0 bytes on the worker run when HLO identity holds.

There are no `jnp.array`, `jnp.zeros`, or `jnp.empty` calls in the traced Thompson body. Container construction is via `state.replace(...)` around fused expressions over existing leaves.

## HLO Auditability

An auditor can re-derive the HLO identity proof by rerunning `python scripts/m5_run_thompson.py`. The committed truncated HLO files are readability artifacts only; the proof is the regenerated zero-byte diff from the committed source and stripped sibling.

## M5-S1.x table export amendment

M5-S1.x adds a reproducible Thompson table-export path in `scripts/extract_thompson_tables.py`. The extractor compiles a scratch copy of the WRF `module_mp_thompson.F.pre`, injects a read-only `m5_dump_thompson_tables` subroutine before `END MODULE module_mp_thompson`, calls `thompson_init`, and writes the initialized private table state to a stream dump that Python repacks as `data/fixtures/thompson-tables-v1.npz`.

Exported and pinned tables include:

- `t_Efrw`, initialized by `table_Efrw` at `module_mp_thompson.F.pre:4921-4977` and consumed by rain-collecting-cloud-water at `module_mp_thompson.F.pre:2260-2268`.
- `tps_iaus`, `tni_iaus`, and `tpi_ide`, initialized by `qi_aut_qs` at `module_mp_thompson.F.pre:4870-4913` and consumed by cloud-ice deposition/autoconversion at `module_mp_thompson.F.pre:2719-2742`.
- Rain-freezing tables `tpi_qrfz`, `tpg_qrfz`, `tni_qrfz`, and `tnr_qrfz`, initialized by `freezeH2O` at `module_mp_thompson.F.pre:4664-4855` and consumed at `module_mp_thompson.F.pre:2658-2669`.
- Snow moment coefficients `sa`, `sb`, `cse`, and `csg`, initialized or declared at `module_mp_thompson.F.pre:337-356,730-750` and consumed by the Field snow-moment path at `module_mp_thompson.F.pre:2093-2191`.
- Graupel moment/coefficient arrays `cge`, `cgg`, `am_g`, `av_g`, `bv_g`, and `rho_g`, declared/initialized at `module_mp_thompson.F.pre:73-156,760-770`.

The current JAX hot path wires the small active rain/cloud collection table, ice autoconversion/deposition tables, and snow moment coefficients through `src/gpuwrf/physics/thompson_tables.py` and `src/gpuwrf/physics/thompson_column.py`. The large rain-freezing tables are extracted and pinned in the asset, but are not yet passed through the jitted timestep body because dynamic gathers from the 4-D rain-freezing tables produced an HLO/launch regression during M5-S1.x (5 counted fusions even after removing the large tables from the hot path; 9 counted fusions with packed rain-freezing tables; 23 with direct dynamic 4-D table reads). This is recorded as the M5-S1.x blocker rather than hidden by gate relabeling.

## M5-S1.y HLO and Residual Amendment

M5-S1.y wires the rain-freezing path through a default-IN packed gather rather than a direct 4-D dynamic gather. WRF's non-aerosol-aware branch fixes `xni = 1.0 * 1000.` and derives `idx_IN` from that value (`module_mp_thompson.F.pre:2637-2656`), so `src/gpuwrf/physics/thompson_tables.py:104-113` slices the already-pinned `tpi_qrfz`, `tpg_qrfz`, `tni_qrfz`, and `tnr_qrfz` arrays at zero-based `idx_IN=27` and packs `(idx_r, idx_r1, idx_tc) -> (pri, prg, pni, pnr)` as a 2-D JAX array. The JAX index math in `src/gpuwrf/physics/thompson_column.py:263-269` and `src/gpuwrf/physics/thompson_column.py:556-584` maps WRF rain table indexes from `module_mp_thompson.F.pre:2374-2400` and consumes the rain-freezing tables at `module_mp_thompson.F.pre:2658-2669`.

The resulting HLO evidence is mixed and must stay visible for M6 planning: `artifacts/m5/thompson_profile.json` records `kernel_launches_per_step=10` and `raw_hlo_launch_marker_count=10`, which is +5 over the post-M5-S1.x baseline of 5 and satisfies the launch target. It also records `hlo_full_bytes=421083`, which exceeds the M5-S1.y 350 KB target; `scripts/m5_gate_thompson.py:78-80` therefore emits `GRAY-ZONE` rather than hiding the miss.

M5-S1.y also tightens three process mappings:

- Rain evaporation: `src/gpuwrf/physics/thompson_column.py:463-513` now carries the WRF rain-evaporation guard from `module_mp_thompson.F.pre:3564-3638`, including the `prw_vcd > 0` skip at `module_mp_thompson.F.pre:3565-3566` and graupel-melt evaporation factor at `module_mp_thompson.F.pre:3617-3620`.
- Cloud-water/ice nucleation: `src/gpuwrf/physics/thompson_column.py:595-607` adds the non-aerosol deposition-nucleation branch from `module_mp_thompson.F.pre:2684-2695`, with `TNO=5.0` and `ATO=0.304` defined from `module_mp_thompson.F.pre:188-189` in `src/gpuwrf/physics/thompson_constants.py:20-21`. The nucleated number is staged so it does not incorrectly alter the cloud-ice distribution used by `module_mp_thompson.F.pre:2710-2742`.
- Graupel sublimation/melting: `src/gpuwrf/physics/thompson_column.py:617-635` and `src/gpuwrf/physics/thompson_column.py:672-684` continue to map the WRF melt/sublimation formulas at `module_mp_thompson.F.pre:2845-2889` and `module_mp_thompson.F.pre:2760-2770`, with the cross-deposition vapor limiter from `module_mp_thompson.F.pre:2922-2939`.

Strict parity remains incomplete. The dominant remaining `qg/qv/T` gap is a mixed-phase graupel residual at the precipitating-column cell where WRF's rain/snow/graupel collision-table paths (`module_mp_thompson.F.pre:2547-2609`) can offset graupel sublimation; those collision tables (`tmr_racg`, `tcr_gacr`, `tnr_*`, `t*c_racs*`) are not part of the M5-S1.x table asset. `src/gpuwrf/validation/tier2_thompson.py:53-76` adds a WRF-linked one-step aggregate budget check so Tier-2 is no longer only JAX self-conservation; it explicitly records the remaining tracked-number carry-forward caveat because the current harness exposes only `Ni/Nr`, not WRF's internal `Ns/Ng/Qb` finalization state.

## Tolerances

The Fortran harness gives a structurally independent oracle and attempt 4 restored the ADR-005 strict Tier-1 tolerances: `abs=1e-10, rel=1e-8` for water species and `abs=1e-3, rel=1e-6` for `Ni/Nr`; `output_T` is recorded with strict `abs=1e-8, rel=1e-8`. These strict tolerances currently fail. The attempt-4 max absolute errors are `qv=1.4304079020558032e-05`, `qc=1.517228938283358e-04`, `qr=4.760876436193939e-06`, `qi=1.3708094759935232e-04`, `qs=1.447943623500527e-04`, `qg=1.5218435328806104e-05`, `Ni=126975.12500000044`, `Nr=67300.453125`, and `T=0.040290844661740266 K`. The order fix confirmed the diagnosis by reducing the main temperature error below 0.1 K, but exact table/moment parity remains unresolved.

After the M5-S1.x table export, current max absolute residuals are `qv=4.7608781466789915e-06`, `qc=1.2657008724556353e-07`, `qr=4.760876436193939e-06`, `qi=1.269506099959173e-07`, `qs=9.2685395199886e-11`, `qg=2.9509294563467847e-06`, `Ni=126975.12500000079`, `Nr=67300.453125`, and `T=0.011792500929288963 K`. This confirms the table export removed the largest snow, cloud-water, and cloud-ice proxy residuals, but strict ADR-005 remains blocked by rain evaporation/melting, graupel depletion, number concentration, and the rain-freezing-table HLO regression.

## Gate dry-run

Gate dry-run: `artifacts/m5/thompson_gate_result.json` is expected to report `FALLBACK`/correctness failure after attempt 4 because Tier-1 does not meet the restored strict tolerances. This is not a backend performance fallback claim; it is a physics-parity blocker recorded in `.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/BLOCKER-m5-s1-attempt4-tolerance.md`. Register and local-memory counters remain `null` because Nsight perf counters are blocked by the known workstation perfmon policy.
